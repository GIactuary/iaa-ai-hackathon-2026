"""The filed algorithm, without its data-specific parts (playbook principles 8 to 10).

- `ModelPair`: one forward GLM and one boosted tree, fitted on the same rows, averaged on the link scale with a weight
  fixed in advance (0.5). On a log link the blend is GLM price x (tree price / GLM price) ^ 0.5: the GLM is the rating
  plan and the tree a bounded adjustment. It reads as credibility weighting with Z = 0.5 set before seeing results.
- `ReferralRule`: refers a record when it is out of evidence (a size beyond `size_ratio_limit` times the largest
  training value in its segment, or a category level never seen in training) or when the two models disagree by more
  than `disagreement_limit`. A disagreement flag alone keeps the model price and goes to review; the treatment of
  out-of-evidence records is a judgment for the actuary and is left open here.
- `PredictionRanges`: a range for each prediction from the out-of-fold misses within its predicted-value fifth, with
  cross-fitted coverage so the coverage figure is itself out of fold.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .families import Family
from .glm import ForwardGLM, is_categorical
from .trees import TreeModel

GLM_WEIGHT = 0.5


# ------------------------------------------------------------------------------------------------ the pair
class ModelPair:
    def __init__(self, glm: ForwardGLM, trees: TreeModel, glm_cols=None, tree_cols=None, glm_weight: float = GLM_WEIGHT):
        """glm and trees are unfitted templates (copied on every fit, so one pair can be refitted fold by fold).
        glm_cols / tree_cols: the inputs each model sees (default: every column)."""
        if glm.family.name != trees.family.name:
            raise ValueError("the GLM and the tree must use the same family")
        self.glm_t, self.trees_t, self.glm_cols, self.tree_cols = glm, trees, glm_cols, tree_cols
        self.family: Family = glm.family
        self.glm_weight = float(glm_weight)

    def _g(self, d):
        return d if self.glm_cols is None else d[self.glm_cols]

    def _t(self, d):
        return d if self.tree_cols is None else d[self.tree_cols]

    def clone(self) -> "ModelPair":
        """An unfitted copy. Use one per fold so parallel folds never share a fitted model."""
        return ModelPair(self.glm_t, self.trees_t, self.glm_cols, self.tree_cols, self.glm_weight)

    def fit(self, d: pd.DataFrame, y, offset=None, weight=None) -> "ModelPair":
        self.glm = copy.deepcopy(self.glm_t).fit(self._g(d), y, offset, weight)
        self.trees = copy.deepcopy(self.trees_t).fit(self._t(d), y, offset, weight)
        return self

    def predict(self, d: pd.DataFrame, offset=None) -> pd.DataFrame:
        """Link-scale predictions: glm, gbm and the fixed blend."""
        g = self.glm.predict_link(self._g(d), offset)
        t = self.trees.predict_link(self._t(d), offset)
        return pd.DataFrame(dict(glm=g, gbm=t, blend=self.glm_weight * g + (1 - self.glm_weight) * t), index=d.index)

    def adjustment(self, pred: pd.DataFrame) -> np.ndarray:
        """Tree adjustment to the GLM on the link scale, as a factor for log links: blend = GLM x factor."""
        a = (1 - self.glm_weight) * (pred.gbm.values - pred.glm.values)
        return np.exp(a)


# ------------------------------------------------------------------------------------------------ referral
class ReferralRule:
    def __init__(self, size_ratio_limit: float, disagreement_limit: float, size_cols=(), segment_col: str | None = None,
                 category_cols=None):
        """size_cols: positive size measures checked against the largest training value in the record's segment.
        segment_col: the segment (for example a location); None compares with the whole training set.
        category_cols: columns checked for levels never seen in training (default: every categorical column)."""
        self.size_ratio_limit, self.disagreement_limit = float(size_ratio_limit), float(disagreement_limit)
        self.size_cols, self.segment_col, self.category_cols = list(size_cols), segment_col, category_cols

    @classmethod
    def from_thresholds(cls, thresholds, **kw) -> "ReferralRule":
        return cls(thresholds["refer_size_ratio"], thresholds["refer_disagreement"], **kw)

    def fit(self, d: pd.DataFrame) -> "ReferralRule":
        """Record the training evidence: largest size by segment, and every category level seen."""
        self.overall_max_ = {c: float(d[c].max()) for c in self.size_cols}
        self.seg_max_ = ({c: d.groupby(d[self.segment_col].astype(str))[c].max() for c in self.size_cols}
                         if self.segment_col else {})
        cats = self.category_cols if self.category_cols is not None else [c for c in d.columns if is_categorical(d[c])]
        self.levels_ = {c: set(d[c].dropna().astype(str)) for c in cats}
        return self

    def evidence(self, d: pd.DataFrame) -> pd.DataFrame:
        """Out-of-evidence check alone; it needs no model, so it can run on the test inputs on day one."""
        n = len(d)
        ratio, driver = np.zeros(n), np.array([""] * n, dtype=object)
        for c in self.size_cols:
            if self.segment_col:
                mx = d[self.segment_col].astype(str).map(self.seg_max_[c]).astype(float).fillna(self.overall_max_[c]).values
            else:
                mx = np.full(n, self.overall_max_[c])
            v = d[c].astype(float).values
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(mx > 0, v / mx, np.where(v > 0, np.inf, 0.0))
            r = np.nan_to_num(r, nan=0.0)
            bigger = r > ratio
            ratio[bigger], driver[bigger] = r[bigger], c
        unseen = [[] for _ in range(n)]
        for c, seen in self.levels_.items():
            if c not in d.columns:
                continue
            s = d[c].astype("string")
            bad = s.notna().values & ~s.isin(seen).fillna(False).values
            for i in np.where(bad)[0]:
                unseen[i].append(f"{c}={s.iloc[i]}")
        out = pd.DataFrame(dict(size_ratio=ratio, size_driver=driver, unseen=["; ".join(u) for u in unseen]), index=d.index)
        out["size_flag"] = out.size_ratio > self.size_ratio_limit
        out["unseen_flag"] = out.unseen != ""
        out["out_of_evidence"] = out.size_flag | out.unseen_flag
        return out

    def apply(self, d: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        """Evidence check plus model disagreement. pred: link-scale glm and gbm columns from ModelPair.predict."""
        out = self.evidence(d)
        out["gap"] = np.expm1(np.abs(pred.glm.values - pred.gbm.values))
        out["gap_flag"] = out.gap > self.disagreement_limit
        out["referred"] = out.out_of_evidence | out.gap_flag
        reasons = []
        for r in out.itertuples():
            why = []
            if r.size_flag:
                why.append(f"{r.size_driver} {r.size_ratio:.2f}x the largest in its segment")
            if r.unseen_flag:
                why.append(f"unseen level {r.unseen}")
            if r.gap_flag:
                why.append(f"models {r.gap:.0%} apart")
            reasons.append("; ".join(why))
        out["reason"] = reasons
        out["treatment"] = np.where(out.out_of_evidence, "referred: out of evidence, treatment is a judgment",
                                    np.where(out.gap_flag, "referred: model price, for review", "model"))
        return out


# ------------------------------------------------------------------------------------------------ ranges
class PredictionRanges:
    def __init__(self, family: Family, lower_q: float, upper_q: float, n_groups: int = 5, scale: str | None = None):
        """Range for a prediction = the lower_q and upper_q percentiles of out-of-fold misses in its predicted-value group.

        scale: "link" (miss = link(y) - eta; the default for log_gaussian) or "ratio" (miss = y / prediction; the
        default for Poisson, gamma and Tweedie, where y can be 0). Binomial targets get calibration bands from
        evidence.lift_table instead.
        """
        if family.name == "binomial":
            raise ValueError("for a yes/no target use probability bands (evidence.lift_table with ci=True)")
        if not 0 <= lower_q < upper_q <= 1:
            raise ValueError("need 0 <= lower_q < upper_q <= 1")
        self.family, self.lower_q, self.upper_q, self.n_groups = family, float(lower_q), float(upper_q), int(n_groups)
        self.scale = scale or ("link" if family.name == "log_gaussian" else "ratio")

    @classmethod
    def from_thresholds(cls, family: Family, thresholds, **kw) -> "PredictionRanges":
        return cls(family, thresholds["range_lower_q"], thresholds["range_upper_q"], **kw)

    def _miss(self, y, eta):
        if self.scale == "link":
            return self.family.link(y) - eta
        return np.asarray(y, dtype=float) / self.family.inverse(eta)

    def _group(self, eta):
        return np.clip(np.searchsorted(self.edges_[1:-1], eta, side="right"), 0, self.n_groups - 1)

    def _quantiles(self, m, g):
        return np.array([[np.quantile(m[g == k], self.lower_q), np.quantile(m[g == k], self.upper_q)] if (g == k).any()
                         else [np.nan, np.nan] for k in range(self.n_groups)])

    def fit(self, y, eta) -> "PredictionRanges":
        """y and out-of-fold eta (never fitted-row predictions)."""
        eta = np.asarray(eta, dtype=float)
        self.edges_ = np.quantile(eta, np.linspace(0, 1, self.n_groups + 1))
        self.q_ = self._quantiles(self._miss(y, eta), self._group(eta))
        return self

    def interval(self, eta) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds on the response scale."""
        eta = np.asarray(eta, dtype=float)
        q = self.q_[self._group(eta)]
        if self.scale == "link":
            return self.family.inverse(eta + q[:, 0]), self.family.inverse(eta + q[:, 1])
        mu = self.family.inverse(eta)
        return mu * q[:, 0], mu * q[:, 1]

    def coverage(self, y, eta, folds) -> dict:
        """Cross-fitted coverage: each fold's records are checked against percentiles from the other folds."""
        y, eta = np.asarray(y, dtype=float), np.asarray(eta, dtype=float)
        m, g = self._miss(y, eta), self._group(eta)
        lo, hi = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for a, b in folds:
            q = self._quantiles(m[a], g[a])
            lo[b], hi[b] = q[g[b], 0], q[g[b], 1]
        inside, above, below = (m >= lo) & (m <= hi), m > hi, m < lo
        mu = self.family.inverse(eta)
        rows = []
        for k in range(self.n_groups):
            s = g == k
            if self.scale == "link":
                med = float(np.median(eta[s]))
                f_lo = float(self.family.inverse(med + self.q_[k, 0]) / self.family.inverse(med))
                f_hi = float(self.family.inverse(med + self.q_[k, 1]) / self.family.inverse(med))
            else:
                f_lo, f_hi = float(self.q_[k, 0]), float(self.q_[k, 1])
            rows.append(dict(group=k + 1, n=int(s.sum()), median_pred=float(np.median(mu[s])), lo_factor=f_lo, hi_factor=f_hi,
                             coverage=float(inside[s].mean()), above=float(above[s].mean()), below=float(below[s].mean())))
        return dict(by_group=rows, total=dict(n=len(y), coverage=float(inside.mean()), above=float(above.mean()),
                                              below=float(below.mean()), nominal=self.upper_q - self.lower_q),
                    edges_link=[float(e) for e in self.edges_], scale=self.scale)
