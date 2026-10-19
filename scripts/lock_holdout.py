import sys; sys.path.insert(0, ".")
import hashlib, json
import numpy as np, pandas as pd
from src import data_rules as dr, validation as v
from src.thresholds import load_thresholds
th = load_thresholds()
rows = []
for line in dr.LINES:
    p = dr.read_raw(f"policies_{line}")
    cut = dr.sum_insured_cutoff(p.sum_insured, th["si_unit_multiple"]) if line != "motor" else None
    p = dr.apply_policy_rules(p, line, cut)
    rows.append(p[["row_id", "customer_id", "period_start"]].assign(line=line))
r = pd.concat(rows, ignore_index=True)
cust = np.array(sorted(r.customer_id.unique()))
lock = v.lock_holdout(cust, share=th["holdout_share"], path="data/work/holdout_lock.json", seed=int(th["holdout_seed"]))
hold = set(lock["holdout_ids"])
r["hold"] = r.customer_id.isin(hold); r["yr"] = r.period_start.dt.year
print("customers:", len(cust), "| held out:", len(hold), f"({len(hold)/len(cust):.4f})")
print("rows held out:", int(r.hold.sum()), "of", len(r), f"({r.hold.mean():.4f})")
print("held-out rows by line and year:"); print(pd.crosstab(r.line, [r.yr], values=r.hold, aggfunc="sum").astype(int))
print("lock spec:", {k: lock[k] for k in ("share", "seed", "n_total", "ids_sha256", "method", "created")})
print("holdout_ids sha256:", hashlib.sha256("\n".join(lock["holdout_ids"]).encode()).hexdigest())
