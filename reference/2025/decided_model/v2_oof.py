"""Stage A: out-of-fold predictions for the decided model on three split seeds (same folds as before)."""
import sys, json, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import v2lib as V

def one(seed, a, b, drop=None):
    P = V.Pipeline(seed=seed, drop=drop, lasso=drop is None).fit(V.raw_tr.iloc[a], V.y[a])
    p = P.predict(V.raw_tr.iloc[b]); p["row"] = b; p["sel"] = [",".join(P.glm.sel)] * len(b)
    return p

if __name__ == "__main__":
    drop = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None
    tag = "_drop_" + "_".join(drop) if drop else ""
    for seed in [42, 1, 2]:
        folds = list(KFold(5, shuffle=True, random_state=seed).split(V.raw_tr))
        parts = Parallel(n_jobs=2)(delayed(one)(seed, a, b, drop) for a, b in folds)
        o = pd.concat(parts).sort_values("row").reset_index(drop=True); o["y"] = V.y
        o.to_parquet(V.OUT / f"v2_oof{tag}_s{seed}.parquet")
        print(seed, tag, {c: round(V.rm(o[c], o.y), 5) for c in ["glm", "gbm", "lasso", "blend"] if c in o}, flush=True)
