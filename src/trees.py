"""Boosted trees (LightGBM) with the objective that matches the GLM family: the bounded-adjustment half of the pair.

The tree starts from the same base as the GLM: an intercept-only fit plus the offset, passed as LightGBM's initial
score. Without this, a Poisson model with an exposure offset would start at a rate of 1 and spend hundreds of trees
walking down to the base frequency. Predictions are on the link scale, so they blend directly with the GLM.

Shapley contributions (`contrib`) come from LightGBM's exact TreeSHAP (`pred_contrib`). On the link scale they add up
exactly: base + sum of contributions + offset = prediction.

DEFAULT_PARAMS are the 2025 hand-set starting point (learning rate 0.01, a few thousand shallow trees, row and
column sampling). They are a starting point for phase 4, not a tuned result for 2026.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .families import Family
from .glm import fit_glm, is_categorical

DEFAULT_PARAMS = dict(num_leaves=6, learning_rate=0.01, num_boost_round=4000, max_bin=200, bagging_fraction=0.7,
                      bagging_freq=1, feature_fraction=0.25, min_data_in_leaf=8, lambda_l1=0.05, lambda_l2=0.5)


class TreeModel:
    def __init__(self, family: Family, params: dict | None = None, seed: int = 42, n_jobs: int = 2):
        self.family, self.seed, self.n_jobs = family, seed, n_jobs
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    def _frame(self, d: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.cols_ if c not in d.columns]
        if missing:
            raise KeyError(f"columns missing at prediction time: {missing}")
        t = d[self.cols_].copy()
        for c in self.cats_:
            s = t[c].astype("string")
            s = s.where(s.isin(self.levels_[c]))                                        # unseen levels become blank
            t[c] = pd.Categorical(s, categories=self.levels_[c])
        for c in t.columns:
            if pd.api.types.is_bool_dtype(t[c].dtype):
                t[c] = t[c].astype(int)
        return t

    def fit(self, d: pd.DataFrame, y, offset=None, weight=None) -> "TreeModel":
        fam = self.family
        y = fam.check_target(y)
        n = len(y)
        off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
        self.cols_ = list(d.columns)
        self.cats_ = [c for c in self.cols_ if is_categorical(d[c]) and not pd.api.types.is_bool_dtype(d[c].dtype)]
        self.levels_ = {c: sorted(d[c].dropna().astype(str).unique()) for c in self.cats_}
        # common starting point: the intercept-only GLM, so the trees fit departures from the base rate
        b0, _ = fit_glm(np.ones((n, 1)), y, fam, off, weight)
        self.base_ = float(b0[0])
        p = dict(self.params)
        rounds = int(p.pop("num_boost_round"))
        p.update(fam.lgbm_params(), seed=self.seed, deterministic=True, force_row_wise=True, num_threads=self.n_jobs,
                 verbosity=-1, boost_from_average=False)
        ds = lgb.Dataset(self._frame(d), label=fam.lgbm_label(y), weight=weight, init_score=off + self.base_,
                         categorical_feature=self.cats_ or "auto", free_raw_data=False)
        self.booster_ = lgb.train(p, ds, num_boost_round=rounds)
        return self

    def predict_link(self, d: pd.DataFrame, offset=None) -> np.ndarray:
        raw = self.booster_.predict(self._frame(d), raw_score=True)
        return raw + self.base_ + (0 if offset is None else np.asarray(offset, dtype=float))

    def predict(self, d: pd.DataFrame, offset=None) -> np.ndarray:
        return self.family.inverse(self.predict_link(d, offset))

    def contrib(self, d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Shapley contributions by input on the link scale, and the base each record starts from (offset excluded)."""
        c = self.booster_.predict(self._frame(d), pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=self.cols_, index=d.index), c[:, -1] + self.base_

    def importance(self) -> pd.DataFrame:
        """Total split gain by input, as a share of the total. Descriptive only; use permutation importance to rank."""
        g = self.booster_.feature_importance(importance_type="gain")
        t = pd.DataFrame(dict(var=self.cols_, gain=g)).sort_values("gain", ascending=False)
        t["share"] = t.gain / t.gain.sum()
        return t.reset_index(drop=True)
