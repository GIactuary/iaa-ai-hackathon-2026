"""Forward-built GLM (playbook principle 9): the rating plan half of the pair.

Start from the base rate. At each step, add the variable that lowers cross-validated error the most; stop when the
best remaining variable gains less than `min_gain` (relative, on the family's score scale). Category levels with
fewer than `min_level_count` records are pooled into "other"; if the pooled group is itself too small it joins the
base level. Both thresholds come from decisions/thresholds.md (`ForwardGLM.from_thresholds`).

Families: log-Gaussian (least squares on the log target), Poisson, gamma and Tweedie with a log link, binomial with a
logit link, all with an optional offset (for example log exposure) and record weights. Fitting is iteratively
reweighted least squares with step-halving, so offsets and weights are exact for every family.

Inputs must already be clean: a blank in a GLM input raises an error, because principle 3 says each blank is
classified by meaning in clean.py before it reaches a model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import lstsq
from sklearn.model_selection import KFold

from .families import ETA_CLIP, Family

OTHER = "(other)"


def is_categorical(s: pd.Series) -> bool:
    return (s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s.dtype)
            or pd.api.types.is_bool_dtype(s.dtype))


# ------------------------------------------------------------------------------------------------ fitting
def _wls(X, z, w):
    sw = np.sqrt(w)
    return lstsq(X * sw[:, None], z * sw, lapack_driver="gelsy", check_finite=False)[0]


def fit_glm(X, y, family: Family, offset=None, weight=None, beta0=None, max_iter: int = 100, tol: float = 1e-8):
    """Maximum-likelihood GLM coefficients. Returns (beta, info) with info = converged, iterations, deviance."""
    n = len(y)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
    if family.closed_form:
        beta = _wls(X, family.link(y) - off, w)
        return beta, dict(converged=True, iterations=1, deviance=float(np.sum(w * family.row_loss(y, off + X @ beta))))

    def dev(eta):
        return float(np.sum(w * family.row_loss(y, eta)))

    if beta0 is None:
        eta = np.clip(family.link(family.start_mu(y, w)), -ETA_CLIP, ETA_CLIP)
        beta, d_old = None, np.inf
    else:
        beta = np.asarray(beta0, dtype=float)
        eta = off + X @ beta
        d_old = dev(eta)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        mu, g = family.inverse(eta), family.dmu_deta(eta)
        g = np.maximum(g, 1e-12)
        W = w * g ** 2 / np.maximum(family.variance(mu), 1e-12)
        z = (eta - off) + (y - mu) / g
        b_new = _wls(X, z, W)
        eta_new = np.clip(off + X @ b_new, -ETA_CLIP, ETA_CLIP)
        d_new = dev(eta_new)
        if beta is not None:
            step = 0
            while (not np.isfinite(d_new) or d_new > d_old * (1 + 1e-12)) and step < 30:
                b_new = (beta + b_new) / 2
                eta_new = np.clip(off + X @ b_new, -ETA_CLIP, ETA_CLIP)
                d_new = dev(eta_new)
                step += 1
        done = np.isfinite(d_old) and abs(d_old - d_new) <= tol * (abs(d_new) + 0.1)
        beta, eta, d_old = b_new, eta_new, d_new
        if done:
            converged = True
            break
    return beta, dict(converged=converged, iterations=it, deviance=d_old)


def dispersion(X, y, beta, family: Family, offset=None, weight=None) -> float:
    """1 for Poisson and binomial; the Pearson estimate otherwise (log-Gaussian: residual variance on the log scale)."""
    if family.name in ("poisson", "binomial"):
        return 1.0
    n, p = X.shape
    off = np.zeros(n) if offset is None else offset
    w = np.ones(n) if weight is None else weight
    eta = off + X @ beta
    if family.closed_form:
        r2 = (family.link(y) - eta) ** 2
    else:
        mu = family.inverse(eta)
        r2 = (y - mu) ** 2 / np.maximum(family.variance(mu), 1e-12)
    return float(np.sum(w * r2) / max(n - p, 1))


# ------------------------------------------------------------------------------------------------ the model
class ForwardGLM:
    def __init__(self, family: Family, min_gain: float, min_level_count: int, max_vars: int = 40, n_folds: int = 5,
                 selection_seed: int = 0, transforms: dict | None = None, as_categorical=(), candidates=None,
                 forced=(), n_jobs: int = 1):
        """
        min_gain, min_level_count: from decisions/thresholds.md (glm_entry_gain, glm_min_level_count).
        transforms: {column: "log1p" | "log"} for numeric inputs that enter on a log scale.
        as_categorical: numeric columns to treat as categories (for example a coded class).
        candidates: columns the forward search may consider (default: every column given to fit).
        forced: columns entered first, before the search (for example a variable required by the use).
        """
        self.family, self.min_gain, self.min_level_count = family, float(min_gain), int(min_level_count)
        self.max_vars, self.n_folds, self.selection_seed = max_vars, n_folds, selection_seed
        self.transforms = dict(transforms or {})
        self.as_categorical, self.candidates, self.forced, self.n_jobs = set(as_categorical), candidates, list(forced), n_jobs

    @classmethod
    def from_thresholds(cls, family: Family, thresholds, **kw) -> "ForwardGLM":
        return cls(family, min_gain=thresholds["glm_entry_gain"], min_level_count=int(thresholds["glm_min_level_count"]), **kw)

    # ---- design --------------------------------------------------------------------------------------
    def _setup(self, d: pd.DataFrame, weight):
        cols = list(self.candidates) if self.candidates is not None else list(d.columns)
        for c in self.forced:
            if c not in cols:
                cols.append(c)
        self.cat = [c for c in cols if is_categorical(d[c]) or c in self.as_categorical]
        self.num = [c for c in cols if c not in self.cat]
        for c in cols:
            if d[c].isna().any():
                raise ValueError(f"GLM input {c!r} has blanks; classify them in clean.py first (principle 3)")
        for c, t in self.transforms.items():
            if c in self.num and t not in ("log1p", "log"):
                raise ValueError(f"transform for {c!r} must be 'log1p' or 'log'")
            if c in self.num and t == "log" and (d[c] <= 0).any():
                raise ValueError(f"{c!r} has values <= 0; use 'log1p'")
        self.num = [c for c in self.num if d[c].nunique() > 1]
        self.levels = {}
        for c in self.cat:
            s = d[c].astype(str)
            vc = s.value_counts()
            if len(vc) < 2:
                continue
            base = vc.index[0]
            keep = [l for l in vc.index[1:] if vc[l] >= self.min_level_count]
            small = [l for l in vc.index[1:] if vc[l] < self.min_level_count]
            pooled = small if vc[small].sum() >= self.min_level_count else []
            self.levels[c] = dict(base=base, keep=keep, other=pooled, to_base=[l for l in small if l not in pooled],
                                  counts={str(k): int(v) for k, v in vc.items()})
        self.cat = [c for c in self.cat if c in self.levels and (self.levels[c]["keep"] or self.levels[c]["other"])]

    def _mat(self, d: pd.DataFrame, g: str) -> np.ndarray:
        if g in self.num:
            v = d[g].astype(float).values
            if np.isnan(v).any():
                raise ValueError(f"GLM input {g!r} has blanks at prediction time")
            t = self.transforms.get(g)
            v = np.log1p(np.clip(v, 0, None)) if t == "log1p" else (np.log(v) if t == "log" else v)
            return v[:, None]
        L, s = self.levels[g], d[g].astype(str)
        cols = [(s == l).values.astype(float) for l in L["keep"]]
        if L["other"]:
            cols.append(s.isin(L["other"]).values.astype(float))
        return np.column_stack(cols)

    def colnames(self, g: str) -> list[str]:
        if g in self.num:
            t = self.transforms.get(g)
            return [f"{t}({g})" if t else g]
        L = self.levels[g]
        return [f"{g}={l}" for l in L["keep"]] + ([f"{g}={OTHER}"] if L["other"] else [])

    # ---- fit -----------------------------------------------------------------------------------------
    def fit(self, d: pd.DataFrame, y, offset=None, weight=None) -> "ForwardGLM":
        fam = self.family
        y = fam.check_target(y)
        n = len(y)
        off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
        w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
        self._setup(d, w)
        G = {g: self._mat(d, g) for g in self.num + self.cat}
        folds = list(KFold(self.n_folds, shuffle=True, random_state=self.selection_seed).split(np.arange(n)))
        ones = np.ones((n, 1))

        def cv(groups):
            X = np.hstack([ones] + [G[g] for g in groups])
            eta = np.zeros(n)
            for a, b in folds:
                beta, _ = fit_glm(X[a], y[a], fam, off[a], w[a])
                eta[b] = off[b] + X[b] @ beta
            return fam.score(y, eta, w)

        sel = [g for g in self.forced if g in G]
        cur = cv(sel)
        self.path = [dict(step=0, var="(base rate only)" if not sel else "+".join(sel), cv=cur, rel_gain=None)]
        pool = [g for g in G if g not in sel]
        while len(sel) < self.max_vars and pool:
            scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(cv)(sel + [g]) for g in pool)
            k = int(np.argmin(scores))
            best, g = scores[k], pool[k]
            rel = (cur - best) / cur if cur else 0.0
            self.candidates_last_ = sorted(zip(pool, scores), key=lambda t: t[1])
            if rel < self.min_gain:
                self.stop_reason_ = f"best remaining variable {g!r} gains {rel:.4%}, below {self.min_gain:.4%}"
                break
            sel.append(g)
            pool.remove(g)
            cur = best
            self.path.append(dict(step=len(self.path), var=g, cv=cur, rel_gain=rel))
        else:
            self.stop_reason_ = "maximum number of variables reached" if pool else "no variables left"
        self.sel = sel
        X = np.hstack([ones] + [G[g] for g in sel])
        self.beta, self.fit_info_ = fit_glm(X, y, fam, off, w)
        self.phi_ = dispersion(X, y, self.beta, fam, off, w)
        eta = off + X @ self.beta
        if fam.closed_form:
            W = w
        else:
            mu, gd = fam.inverse(eta), fam.dmu_deta(eta)
            W = w * gd ** 2 / np.maximum(fam.variance(mu), 1e-12)
        self.cov_ = self.phi_ * np.linalg.pinv((X * W[:, None]).T @ X)
        self.means_ = {g: G[g].mean(0) for g in sel}
        self.intercept_ = float(self.beta[0])
        self.n_fit_ = n
        return self

    # ---- predict and explain -------------------------------------------------------------------------
    def _slices(self):
        k = 1
        for g in self.sel:
            m = len(self.colnames(g))
            yield g, slice(k, k + m)
            k += m

    def terms(self, d: pd.DataFrame) -> pd.DataFrame:
        """Each variable's contribution on the link scale, centred on the average fitted record."""
        out = {g: (self._mat(d, g) - self.means_[g]) @ self.beta[sl] for g, sl in self._slices()}
        return pd.DataFrame(out, index=d.index)

    def base_link(self) -> float:
        """Link-scale prediction for the average fitted record, before any offset."""
        return self.intercept_ + sum(float(self.means_[g] @ self.beta[sl]) for g, sl in self._slices())

    def predict_link(self, d: pd.DataFrame, offset=None) -> np.ndarray:
        eta = self.base_link() + self.terms(d).sum(axis=1).values if self.sel else np.full(len(d), self.intercept_)
        return eta + (0 if offset is None else np.asarray(offset, dtype=float))

    def predict(self, d: pd.DataFrame, offset=None) -> np.ndarray:
        return self.family.inverse(self.predict_link(d, offset))

    def unseen_levels(self, d: pd.DataFrame) -> dict:
        """Levels at prediction time never seen in fitting. They are rated at the base level, and referred."""
        out = {}
        for g in self.sel:
            if g in self.cat:
                L = self.levels[g]
                known = {L["base"], *L["keep"], *L["other"], *L["to_base"]}
                bad = ~d[g].astype(str).isin(known)
                if bad.any():
                    out[g] = int(bad.sum())
        return out

    def coef_table(self) -> pd.DataFrame:
        """Coefficients with standard errors; relativity = exp(coef) for log and logit links (odds ratio for logit)."""
        se = np.sqrt(np.clip(np.diag(self.cov_), 0, None))
        rows = [dict(var="(intercept)", term="(intercept)", coef=self.beta[0], se=se[0])]
        for g, sl in self._slices():
            for name, b, s in zip(self.colnames(g), self.beta[sl], se[sl]):
                rows.append(dict(var=g, term=name, coef=b, se=s))
        t = pd.DataFrame(rows)
        t["z"] = t.coef / t.se.replace(0, np.nan)
        if self.family.link_name in ("log", "log1p", "logit"):
            t["relativity"] = np.exp(t.coef)
            t["rel_lo95"] = np.exp(t.coef - 1.96 * t.se)
            t["rel_hi95"] = np.exp(t.coef + 1.96 * t.se)
        return t

    def levels_table(self) -> pd.DataFrame:
        """How each categorical variable's levels were treated: base, own relativity, pooled, or merged into base."""
        rows = []
        for g, L in self.levels.items():
            for lev, n in L["counts"].items():
                role = ("base" if lev == L["base"] else "own level" if lev in L["keep"]
                        else "pooled into (other)" if lev in L["other"] else "merged into base (pool too small)")
                rows.append(dict(var=g, level=lev, n=n, treatment=role, selected=g in self.sel))
        return pd.DataFrame(rows)
