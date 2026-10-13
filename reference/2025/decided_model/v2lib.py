"""Decided model (v2): forward-built GLM + LightGBM, 50/50 on the log scale, fixed in advance.

Decisions applied: J2 sale-time variables dropped (and the ages computed from the sale year);
J3 referral rule; J9 two models, equal weights. Cleaning rules are the same as mine_models.py.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
os.environ.setdefault("SEED", "42")
import mine_models as M

W = M.W; OUT = W / "out"
SALE = ["MoSold", "YrSold", "SaleType", "SaleCondition"]
SALE_DERIVED = []      # J2: house and remodel age stay, measured at the valuation date (the sale date in this data)
ABOVE = ["GrLivArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF"]
BSMT = ["TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]
ORD_BASE = set(M.ORD)                                         # ordinal codes: the GLM uses their numeric _o form
EXCL = lambda d: (d.Neighborhood == "Edwards") & (d.GrLivArea > 4000)
SIZE_K, GAP_D = 1.25, 0.30                                    # referral thresholds, fixed before looking at flags
NONMARKET_IDS = [1227]       # screen: partial sale of an unfinished new build at a fraction of its value (424 is a normal sale)

raw_tr_all = M.tr.copy()
raw_tr = raw_tr_all[~EXCL(raw_tr_all)].reset_index(drop=True)
raw_te = M.te.copy()
y = np.log1p(raw_tr.SalePrice.values)
NONMARKET = raw_tr_all[raw_tr_all.Id.isin(NONMARKET_IDS)].assign(ppsf=lambda d: d.SalePrice / d.GrLivArea)[["Id", "Neighborhood", "ppsf"]]
rm = lambda a, b=0: float(np.sqrt(np.mean(np.square(np.asarray(a) - np.asarray(b)))))


def prep(raw, drop=()):
    d = M.clean(raw)
    return d.drop(columns=[c for c in SALE + SALE_DERIVED + ["Id", "SalePrice"] + list(drop) if c in d.columns])


def is_cat(s):
    return s.dtype == object or str(s.dtype) in ("str", "string")


# ---------------------------------------------------------------- forward GLM
class FwdGLM:
    """Gaussian GLM on log price (log link on price), built forward: a variable enters only if it
    lowers 5-fold CV error by at least 0.2% (relative). Categorical levels with fewer than 5
    houses are pooled. Numeric inputs skewed above 0.75 enter as log(1 + x)."""
    MIN_GAIN, RARE, MAX_VARS = 0.002, 5, 40
    NOT_GLM = {"QualSF"}          # an interaction; the GLM keeps quality and area as separate rating variables

    def _setup(self, d):
        self.num = [c for c in d.columns if not is_cat(d[c]) and d[c].nunique() > 1 and c not in self.NOT_GLM]
        self.cat = [c for c in d.columns if is_cat(d[c]) and c not in ORD_BASE and d[c].nunique() > 1]
        self.logc = {c for c in self.num if d[c].min() >= 0 and d[c].skew() > 0.75}
        self.levels = {}
        for c in self.cat:
            vc = d[c].value_counts()
            keep = [l for l in vc.index[1:] if vc[l] >= self.RARE]       # vc.index[0] is the base level
            other = [l for l in vc.index[1:] if vc[l] < self.RARE]
            self.levels[c] = dict(base=vc.index[0], keep=keep, other=other if vc[other].sum() >= self.RARE else [])

    def _mat(self, d, g):
        if g in self.num:
            v = d[g].astype(float).values
            return (np.log1p(np.clip(v, 0, None)) if g in self.logc else v)[:, None]
        L = self.levels[g]; cols = [(d[g] == l).values.astype(float) for l in L["keep"]]
        if L["other"]:
            cols.append(d[g].isin(L["other"]).values.astype(float))
        return np.column_stack(cols) if cols else np.zeros((len(d), 0))

    def colnames(self, g):
        if g in self.num:
            return [g]
        L = self.levels[g]; return [f"{g}={l}" for l in L["keep"]] + ([f"{g}=other"] if L["other"] else [])

    def fit(self, d, yy, seed=0):
        self._setup(d)
        G = {g: self._mat(d, g) for g in self.num + self.cat}
        G = {g: m for g, m in G.items() if m.shape[1]}
        folds = list(KFold(5, shuffle=True, random_state=seed).split(d))
        n = len(yy)

        def cv(groups):
            X = np.column_stack([np.ones(n)] + [G[g] for g in groups]); e = np.zeros(n)
            for a, b in folds:
                beta = np.linalg.lstsq(X[a], yy[a], rcond=None)[0]; e[b] = X[b] @ beta - yy[b]
            return rm(e)

        sel, cur = [], cv([]); self.path = [dict(var="(base rate only)", cv=cur)]
        while len(sel) < self.MAX_VARS:
            best = min(((cv(sel + [g]), g) for g in G if g not in sel), default=None)
            if best is None or best[0] > cur * (1 - self.MIN_GAIN):
                break
            cur = best[0]; sel.append(best[1]); self.path.append(dict(var=best[1], cv=cur))
        self.sel = sel
        X = np.column_stack([np.ones(n)] + [G[g] for g in sel])
        self.beta = np.linalg.lstsq(X, yy, rcond=None)[0]
        self.means = {g: G[g].mean(0) for g in sel}
        self.mean_pred = float((X @ self.beta).mean())
        return self

    def terms(self, d):
        """Per-variable contribution in log terms, centred on the average training house."""
        out, k = {}, 1
        for g in self.sel:
            m = self._mat(d, g); b = self.beta[k:k + m.shape[1]]; k += m.shape[1]
            out[g] = (m - self.means[g]) @ b
        return pd.DataFrame(out, index=d.index)

    def predict(self, d):
        return self.mean_pred + self.terms(d).sum(axis=1).values

    def coef_table(self):
        rows, k = [], 1
        for g in self.sel:
            names = self.colnames(g)
            for nme in names:
                rows.append(dict(var=g, term=nme, coef=float(self.beta[k]), log=g in self.logc)); k += 1
        return rows


# ---------------------------------------------------------------- trees and Lasso
class Trees:
    def __init__(self, seed=42):
        self.seed = seed

    def frame(self, d):
        t = d.copy()
        for c in self.cats:
            t[c] = pd.Categorical(t[c], categories=self.levels[c])
        return t[self.cols]

    def fit(self, d, yy):
        self.cols = list(d.columns); self.cats = [c for c in self.cols if is_cat(d[c])]
        self.levels = {c: sorted(d[c].unique()) for c in self.cats}
        self.m = LGBMRegressor(objective="regression", num_leaves=6, learning_rate=0.01, n_estimators=4000, max_bin=200,
                               subsample=0.7, subsample_freq=1, colsample_bytree=0.25, min_child_samples=8,
                               reg_alpha=0.05, reg_lambda=0.5, random_state=self.seed, verbosity=-1, n_jobs=1)
        self.m.fit(self.frame(d), yy); return self

    def predict(self, d):
        return self.m.predict(self.frame(d))

    def contrib(self, d):
        c = self.m.predict(self.frame(d), pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=self.cols, index=d.index), c[:, -1]


class LassoLin:
    def fit(self, d, yy):
        self.num = [c for c in d.columns if not is_cat(d[c])]; self.cat = [c for c in d.columns if is_cat(d[c])]
        self.logc = [c for c in self.num if d[c].min() >= 0 and d[c].skew() > 0.75]
        X = self._x(d); self.cols = X.columns[X.nunique() > 1]; X = X[self.cols]
        self.s = StandardScaler().fit(X)
        self.m = LassoCV(alphas=np.logspace(-4.5, -1.5, 30), max_iter=20000, cv=5, random_state=0).fit(self.s.transform(X), yy)
        return self

    def _x(self, d):
        x = d[self.num].astype(float).copy(); x[self.logc] = np.log1p(x[self.logc].clip(lower=0))
        return pd.concat([x, pd.get_dummies(d[self.cat], dtype=float)], axis=1)

    def predict(self, d):
        return self.m.predict(self.s.transform(self._x(d).reindex(columns=self.cols, fill_value=0.0)))


# ---------------------------------------------------------------- the pipeline
ENG_DROP = ("TotalPorchSF", "HasGarage", "HasBsmt")   # failed the drop-one test (not worse on all three split seeds)


class Pipeline:
    def __init__(self, seed=42, drop=None, lasso=False):
        self.seed, self.drop, self.lasso = seed, tuple(ENG_DROP if drop is None else drop), lasso

    def fit(self, raw, yy):
        d = prep(raw, self.drop)
        self.glm = FwdGLM().fit(d, yy); self.trees = Trees(self.seed).fit(d, yy)
        self.las = LassoLin().fit(d, yy) if self.lasso else None
        self.nbmax = raw.groupby("Neighborhood").GrLivArea.max()
        return self

    def predict(self, raw):
        d = prep(raw, self.drop)
        out = dict(glm=self.glm.predict(d), gbm=self.trees.predict(d))
        if self.las is not None:
            out["lasso"] = self.las.predict(d)
        out["blend"] = (out["glm"] + out["gbm"]) / 2
        return pd.DataFrame(out, index=raw.index)

    def size_ratio(self, raw):
        return (raw.GrLivArea / raw.Neighborhood.map(self.nbmax)).fillna(0).values

    def capped(self, raw):
        """Shrink every floor area so the house equals the largest evidenced house in its neighbourhood."""
        r = raw.copy(); f = np.minimum(1.0, (r.Neighborhood.map(self.nbmax) / r.GrLivArea).fillna(1.0).values)
        for c in ABOVE + BSMT + ["GarageArea"]:
            r[c] = r[c] * f
        return r

    def price(self, raw):
        """Filed algorithm: blend, with the size-check referral treatment (J3 option b)."""
        p = self.predict(raw); ratio = self.size_ratio(raw); size_flag = ratio > SIZE_K
        gap = np.expm1(np.abs(p.glm - p.gbm).values); gap_flag = gap > GAP_D
        final = p.blend.values.copy()
        if size_flag.any():
            final[size_flag] = self.predict(self.capped(raw[size_flag])).blend.values
        # individual rating from the excluded non-market comparable: same neighbourhood, a new build sold before
        # completion, more than twice the largest priced house there (J3)
        comp = NONMARKET.set_index("Neighborhood").ppsf
        prof = (size_flag & (ratio > 2) & (raw.SaleType.str.strip() == "New").values & (raw.SaleCondition == "Partial").values
                & raw.Neighborhood.isin(comp.index).values)
        if prof.any():
            final[prof] = np.log1p(raw.GrLivArea.values[prof] * raw.Neighborhood.map(comp).values[prof])
        p["ratio"], p["gap"], p["size_flag"], p["gap_flag"], p["comparable"], p["final"] = ratio, gap, size_flag, gap_flag, prof, final
        p["treatment"] = np.where(prof, "comparable sale", np.where(size_flag, "size capped", np.where(gap_flag, "referred, model price", "model")))
        return p
