"""Independent solution: base models and out-of-fold predictions.

Same 5-fold split (KFold shuffle, seed 42) and same 2-row outlier exclusion as the
winner, so the OOF scores are directly comparable. Writes OOF and test predictions
for every base model to out/mine_oof.parquet and out/mine_test.parquet.
Usage: python3 mine_models.py <workdir> [seed]
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import os
W = Path(os.environ.get("WORK", sys.argv[1] if len(sys.argv) > 1 else ".")); SEED = int(os.environ.get("SEED", sys.argv[2] if len(sys.argv) > 2 else 42))
OUT = W / "out"; t0 = time.time()
tr = pd.read_csv(W / "data/train.csv"); te = pd.read_csv(W / "data/test.csv")

# ---------- cleaning: NA semantics from data_description.txt ----------
NONE = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
ZERO = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
        'GarageCars', 'GarageArea']
QUAL = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ORD = {**{c: QUAL for c in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
                            'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']},
       'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
       'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
       'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
       'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
       'Functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},
       'CentralAir': {'N': 0, 'Y': 1}, 'PavedDrive': {'N': 0, 'P': 1, 'Y': 2}}
lf_med = tr.groupby("Neighborhood")["LotFrontage"].median(); lf_all = tr["LotFrontage"].median()


def clean(d):
    d = d.copy()
    d[NONE] = d[NONE].fillna("None"); d[ZERO] = d[ZERO].fillna(0)
    d["LotFrontage"] = d["LotFrontage"].fillna(d["Neighborhood"].map(lf_med)).fillna(lf_all)
    d["GarageYrBlt"] = d["GarageYrBlt"].fillna(d["YearBuilt"])      # no garage: neutral year, flag below
    for c in d.columns:
        if d[c].dtype == object or str(d[c].dtype) in ("str", "string"):
            d[c] = d[c].fillna(tr[c].mode().iloc[0]).astype(str)
    for c, m in ORD.items():
        d[c + "_o"] = d[c].map(m).fillna(0)
    d["TotalSF"] = d["TotalBsmtSF"] + d["1stFlrSF"] + d["2ndFlrSF"]
    d["TotalBath"] = d["FullBath"] + .5 * d["HalfBath"] + d["BsmtFullBath"] + .5 * d["BsmtHalfBath"]
    d["TotalPorchSF"] = d[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']].sum(axis=1)
    d["HouseAge"] = (d["YrSold"] - d["YearBuilt"]).clip(lower=0)
    d["RemodelAge"] = (d["YrSold"] - d["YearRemodAdd"]).clip(lower=0)
    d["QualSF"] = d["OverallQual"] * d["GrLivArea"]
    d["HasGarage"] = (d["GarageArea"] > 0).astype(int); d["HasBsmt"] = (d["TotalBsmtSF"] > 0).astype(int)
    d["MSSubClass"] = d["MSSubClass"].astype(str)
    return d


trc, tec = clean(tr), clean(te)
out_tr = (trc.Neighborhood == "Edwards") & (trc.GrLivArea > 4000)
trc = trc[~out_tr].reset_index(drop=True)
y = np.log1p(trc.pop("SalePrice")).values
ids_te = tec["Id"].values
trc = trc.drop(columns="Id"); tec = tec.drop(columns="Id")
cat_cols = [c for c in trc.columns if trc[c].dtype == object or str(trc[c].dtype) in ("str", "string")]
num_cols = [c for c in trc.columns if c not in cat_cols]

# ---------- design matrix for linear / kernel models ----------
full = pd.concat([trc, tec], keys=["tr", "te"])
lin = full[num_cols].astype(float).copy()
skewed = [c for c in num_cols if lin[c].min() >= 0 and lin[c].skew() > 0.75]
lin[skewed] = np.log1p(lin[skewed])
lin = pd.concat([lin, pd.get_dummies(full[cat_cols], dtype=float)], axis=1)
lin = lin.loc[:, lin.loc["tr"].nunique() > 1]
X_lin_tr, X_lin_te = lin.loc["tr"].values, lin.loc["te"].values

# ---------- tree-model matrices ----------
tree = full.copy()
for c in cat_cols:
    tree[c] = pd.Categorical(tree[c])
X_tr_tree, X_te_tree = tree.loc["tr"].reset_index(drop=True), tree.loc["te"].reset_index(drop=True)
X_tr_cb = full.loc["tr"].reset_index(drop=True); X_te_cb = full.loc["te"].reset_index(drop=True)


def scaled(fit_rows, pred_rows):
    s = StandardScaler().fit(fit_rows)
    return s.transform(fit_rows), s.transform(pred_rows)


MODELS = {
    "ridge": lambda: RidgeCV(alphas=np.logspace(-1, 3, 30)),
    "lasso": lambda: LassoCV(alphas=np.logspace(-4.5, -1.5, 30), max_iter=20000, cv=5, random_state=0),
    "enet": lambda: ElasticNetCV(l1_ratio=[.3, .6, .9], alphas=np.logspace(-4.5, -1.5, 20), max_iter=20000, cv=5,
                                 random_state=0),
    "svr": lambda: SVR(C=20, epsilon=0.008, gamma=0.0003),
    "kridge": lambda: KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5),
    "lgb": lambda: LGBMRegressor(objective="regression", num_leaves=6, learning_rate=0.01, n_estimators=4000,
                                 max_bin=200, subsample=0.7, subsample_freq=1, colsample_bytree=0.25,
                                 min_child_samples=8, reg_alpha=0.05, reg_lambda=0.5, random_state=SEED,
                                 verbosity=-1, n_jobs=2),
    "xgb": lambda: XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, min_child_weight=2,
                                subsample=0.7, colsample_bytree=0.3, reg_alpha=0.005, enable_categorical=True,
                                tree_method="hist", random_state=SEED, n_jobs=2),
    "cat": lambda: CatBoostRegressor(iterations=2500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                                     random_seed=SEED, verbose=0, thread_count=2),
}
LINEAR = {"ridge", "lasso", "enet", "svr", "kridge"}


def fit_predict(name, tri, X_pred_lin, X_pred_tree, X_pred_cb):
    """Fit one base model on training rows tri; predict the given rows."""
    m = MODELS[name]()
    if name in LINEAR:
        s_ = StandardScaler().fit(X_lin_tr[tri]); m.fit(s_.transform(X_lin_tr[tri]), y[tri]); return m.predict(s_.transform(X_pred_lin))
    if name == "cat":
        m.fit(X_tr_cb.iloc[tri], y[tri], cat_features=cat_cols); return m.predict(X_pred_cb)
    m.fit(X_tr_tree.iloc[tri], y[tri]); return m.predict(X_pred_tree)


if __name__ == "__main__":
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(kf.split(X_lin_tr))
    oof = pd.DataFrame(index=range(len(y))); pte = pd.DataFrame(index=range(len(tec)))
    scores = {}
    for name, make in MODELS.items():
        o = np.zeros(len(y)); p = np.zeros(len(tec))
        for tri, vai in folds:
            m = make()
            if name in LINEAR:
                a, b = scaled(X_lin_tr[tri], X_lin_tr[vai]); m.fit(a, y[tri]); o[vai] = m.predict(b)
            elif name == "cat":
                m.fit(X_tr_cb.iloc[tri], y[tri], cat_features=cat_cols); o[vai] = m.predict(X_tr_cb.iloc[vai])
            else:
                m.fit(X_tr_tree.iloc[tri], y[tri]); o[vai] = m.predict(X_tr_tree.iloc[vai])
        # refit on all training rows for test predictions
        m = make()
        if name in LINEAR:
            a, b = scaled(X_lin_tr, X_lin_te); m.fit(a, y); p = m.predict(b)
        elif name == "cat":
            m.fit(X_tr_cb, y, cat_features=cat_cols); p = m.predict(X_te_cb)
        else:
            m.fit(X_tr_tree, y); p = m.predict(X_te_tree)
        oof[name], pte[name] = o, p
        scores[name] = float(np.sqrt(np.mean((o - y) ** 2)))
        print(f"{name:7s} OOF RMSLE {scores[name]:.5f}   ({time.time()-t0:.0f}s)", flush=True)

    oof["y"] = y
    oof.to_parquet(OUT / f"mine_oof_s{SEED}.parquet"); pte.assign(Id=ids_te).to_parquet(OUT / f"mine_test_s{SEED}.parquet")
    json.dump(scores, open(OUT / f"mine_scores_s{SEED}.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s")
