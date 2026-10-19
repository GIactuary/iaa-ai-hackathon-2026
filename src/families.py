"""Model families for phase 5 step 7 (P7, C5, J-model). Design: decisions/phase5_step7_design.md.

Every family fits the rate y / ec with weight ec^(2 - p) (times an optional extra weight), which under Tweedie
deviance equals fitting y with a log(ec) offset. For a Poisson frequency p = 1; for a gamma severity on rows with n
claims, pass ec = n, p = 2 and weight = n. `rate(X)` returns the predicted rate; a model started from another model's
log rate (`init`) needs the same `init` when predicting.

- `LGB`: LightGBM, plain or started from a GLM (init); early stopping on a validation set when one is given.
- `Cat`: CatBoost.
- `EBM`: interpret's ExplainableBoostingRegressor with pairwise terms.
- `GlumGLM`: elastic-net GLM (glum).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OBJECTIVE_POWER = {"poisson": 1.0, "gamma": 2.0}


def _p(objective: str, power: float) -> float:
    return OBJECTIVE_POWER.get(objective, power)


def rate_weights(y, ec, p, weight=None):
    y, ec = np.asarray(y, float), np.asarray(ec, float)
    w = np.power(ec, 2 - p) * (1.0 if weight is None else np.asarray(weight, float))
    return y / ec, w


class LGB:
    def __init__(self, objective="tweedie", power=1.5, leaves=31, min_leaf=400, lr=0.05, n_trees=3000, threads=4, seed=0):
        self.objective, self.power, self.leaves, self.min_leaf, self.lr = objective, power, leaves, min_leaf, lr
        self.n_trees, self.threads, self.seed = int(n_trees), threads, seed

    def _params(self):
        p = dict(objective=self.objective, num_leaves=self.leaves, min_child_samples=self.min_leaf, learning_rate=self.lr,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=self.threads, random_state=self.seed, verbose=-1)
        if self.objective == "tweedie":
            p["tweedie_variance_power"] = self.power
        return p

    def fit(self, X, y, ec, weight=None, init=None, val=None, es_rounds=100):
        """val: (X, y, ec, weight, init) for early stopping; sets best_iter_."""
        import lightgbm as lgb
        p = _p(self.objective, self.power)
        t, w = rate_weights(y, ec, p, weight)
        self.model_ = lgb.LGBMRegressor(n_estimators=self.n_trees, **self._params())
        kw = dict(sample_weight=w, init_score=init)
        if val is not None:
            Xv, yv, ecv, wv, iv = val
            tv, wv2 = rate_weights(yv, ecv, p, wv)
            kw |= dict(eval_set=[(Xv, tv)], eval_sample_weight=[wv2], eval_init_score=[iv] if iv is not None else None,
                       callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        self.model_.fit(X, t, **kw)
        self.best_iter_ = int(self.model_.best_iteration_ or self.n_trees)
        return self

    def rate(self, X, init=None):
        raw = self.model_.predict(X, raw_score=True, num_iteration=self.best_iter_)
        return np.exp(raw + (0.0 if init is None else np.asarray(init, float)))

    def contrib(self, X, init=None):
        """Per-feature contributions to the log rate (SHAP, from LightGBM)."""
        return self.model_.predict(X, pred_contrib=True, num_iteration=self.best_iter_)


class Cat:
    def __init__(self, objective="tweedie", power=1.5, depth=6, lr=0.1, n_trees=3000, threads=4, seed=0, cat_features=()):
        self.objective, self.power, self.depth, self.lr, self.n_trees = objective, power, depth, lr, int(n_trees)
        self.threads, self.seed, self.cat_features = threads, seed, list(cat_features)

    def fit(self, X, y, ec, weight=None, init=None, val=None, es_rounds=100):
        from catboost import CatBoostRegressor, Pool
        p = _p(self.objective, self.power)
        loss = {"tweedie": f"Tweedie:variance_power={self.power}", "poisson": "Poisson"}.get(self.objective, self.objective)
        t, w = rate_weights(y, ec, p, weight)
        self.model_ = CatBoostRegressor(loss_function=loss, iterations=self.n_trees, learning_rate=self.lr, depth=self.depth,
                                        thread_count=self.threads, random_seed=self.seed, verbose=0, allow_writing_files=False)
        train = Pool(X, t, weight=w, cat_features=self.cat_features, baseline=init)
        if val is not None:
            Xv, yv, ecv, wv, iv = val
            tv, wv2 = rate_weights(yv, ecv, p, wv)
            self.model_.fit(train, eval_set=Pool(Xv, tv, weight=wv2, cat_features=self.cat_features, baseline=iv),
                            early_stopping_rounds=es_rounds, use_best_model=True)
            self.best_iter_ = int(self.model_.get_best_iteration()) + 1
        else:
            self.model_.fit(train)
            self.best_iter_ = self.n_trees
        return self

    def rate(self, X, init=None):
        from catboost import Pool
        raw = self.model_.predict(Pool(X, cat_features=self.cat_features), prediction_type="RawFormulaVal")
        return np.exp(raw + (0.0 if init is None else np.asarray(init, float)))


class EBM:
    def __init__(self, objective="tweedie", power=1.5, interactions=10, outer_bags=4, threads=4, seed=0):
        self.objective, self.power, self.interactions, self.outer_bags = objective, power, interactions, int(outer_bags)
        self.threads, self.seed = threads, seed

    def fit(self, X, y, ec, weight=None, init=None, val=None, es_rounds=None):
        from interpret.glassbox import ExplainableBoostingRegressor
        p = _p(self.objective, self.power)
        obj = {"tweedie": f"tweedie_deviance:variance_power={self.power}", "poisson": "poisson_deviance", "gamma": "gamma_deviance"}[self.objective]
        t, w = rate_weights(y, ec, p, weight)
        self.model_ = ExplainableBoostingRegressor(objective=obj, interactions=self.interactions, outer_bags=self.outer_bags,
                                                   n_jobs=self.threads, random_state=self.seed)
        self.model_.fit(X, t, sample_weight=w)
        self.best_iter_ = None
        return self

    def rate(self, X, init=None):
        # with a log-link objective, predict() returns the mean; the log is taken so `init` composes the same way
        mu = self.model_.predict(X)
        return mu * (1.0 if init is None else np.exp(np.asarray(init, float)))

    def importances(self) -> dict:
        return dict(zip(self.model_.term_names_, map(float, self.model_.term_importances())))


class GlumGLM:
    def __init__(self, objective="tweedie", power=1.5, alpha=1e-4, l1_ratio=0.5):
        self.objective, self.power, self.alpha, self.l1_ratio = objective, power, alpha, l1_ratio

    def fit(self, X, y, ec, weight=None, init=None, val=None, es_rounds=None):
        from glum import GeneralizedLinearRegressor, TweedieDistribution
        p = _p(self.objective, self.power)
        fam = {"poisson": "poisson", "gamma": "gamma"}.get(self.objective) or TweedieDistribution(self.power)
        t, w = rate_weights(y, ec, p, weight)
        self.model_ = GeneralizedLinearRegressor(family=fam, link="log", alpha=self.alpha, l1_ratio=self.l1_ratio,
                                                 drop_first=True, max_iter=300)
        self.model_.fit(X, t, sample_weight=w, offset=init)
        self.best_iter_ = None
        return self

    def rate(self, X, init=None):
        return self.model_.predict(X, offset=init)


def as_category(df: pd.DataFrame, cols) -> pd.DataFrame:
    """String columns to pandas categories with the levels of the whole frame, so every split shares them."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype("string").fillna("blank").astype("category")
    return out


def as_string(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype("string").fillna("blank").astype(str)
    return out
