"""Motor-only Y1 premium against the improved shape (K9+K7b): P2-V gains and its public-board spread on random thirds (rule M2-R).
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_motor_premium_check.py
"""
import sys, pickle
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.argv = ["x"]
import numpy as np, pandas as pd
import scripts.part2_candidates as pc
from scripts.part2_model import developed
from src import build, data_rules as dr, validation as v

data = build.build_all()
w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", pc.ROOT / "data" / "raw_part2" / "developed")
lines = {}
for line in dr.LINES:
    tr = data[line][0]
    lines[line] = pc.Line(line, tr[~tr["holdout"]].reset_index(drop=True), developed(line), w_em, w_dv)
P1 = {l: pickle.loads((pc.WORK / f"p1_{l}.pkl").read_bytes()) for l in dr.LINES}
for l in dr.LINES:
    P1[l]["r3"] = pickle.loads((pc.WORK / f"p1r3_{l}.pkl").read_bytes())
allr = pd.concat([L.tr[["customer_id", "yr", "n_claims"]] for L in lines.values()])
cy = allr.groupby(["customer_id", "yr"])["n_claims"].sum()
for L in lines.values():
    n = pd.Series(cy.reindex(pd.MultiIndex.from_arrays([L.tr["customer_id"], L.tr["yr"] - 1])).to_numpy())
    L.h = np.where(n.isna(), "none", np.where(n >= 2, "2+", n.fillna(0).astype(int).astype(str)))
S = pc.score(lines, P1, {"incumbent": "K0"})
out = {}
for name in ("V1", "V2"):
    s = S[name]
    motor = s["_line"] == "motor"
    inc = s["K9+K7b"]
    cand = np.where(motor, s["K11_premium"], inc)          # motor-only premium, other lines unchanged
    y = s["_y"]
    null = v.tweedie_unit_deviance(y, np.full(len(y), y.mean()), 1.5)
    full = (inc.sum() - cand.sum()) / null.sum()
    bt = v.paired_bootstrap_groups(inc, cand, s["_g"])
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(1000):
        m = rng.random(len(y)) < 1 / 3
        diffs.append((inc[m].sum() - cand[m].sum()) / null[m].sum())
    diffs = np.array(diffs)
    out[name] = (full, bt, diffs)
    print(f"{name}: motor-only premium against K9+K7b: deviance gain {bt['gain']:,.0f} [{bt['lo']:,.0f}, {bt['hi']:,.0f}]; "
          f"D2 gain on all rows {full:+.5f}; on a random third: sd {diffs.std():.5f}, 90% {np.quantile(diffs, .05):+.5f} to {np.quantile(diffs, .95):+.5f}, "
          f"looks worse {np.mean(diffs < 0):.2f}")
