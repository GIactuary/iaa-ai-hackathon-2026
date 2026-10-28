"""Part 2 level study, on 2024 emerging rows (no board use).

The final shape fitted on 2023 rows predicts every 2024 row; each line is put at its deviance-optimal level, then:
  1. sensitivity: pooled D2 when one line's level is off by a given factor;
  2. precision: on 1,000 random thirds of rows (the public board's share), the optimal level found on the third
     against the optimal level on the other two-thirds, per line.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_level_study.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import scripts.part2_candidates as pc
from scripts.part2_model import developed
from src import build, data_rules as dr, market, validation as v
from src.evidence import write_evidence

OUT = market.OUT / "part2_level_study.json"
P = 1.5


def opt_scale(y, mu):
    """The multiplier c minimising Tweedie deviance of c * mu: sum(y mu^(1-p)) / sum(mu^(2-p))."""
    return float(np.sum(y * mu ** (1 - P)) / np.sum(mu ** (2 - P)))


def main():
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", pc.ROOT / "data" / "raw_part2" / "developed")
    lines, P1 = {}, {}
    for line in dr.LINES:
        lines[line] = pc.Line(line, data[line][0].reset_index(drop=True), developed(line), w_em, w_dv)
        P1[line] = pickle.loads((pc.WORK / f"h_p1_{line}.pkl").read_bytes())
    allr = pd.concat([L.tr[["customer_id", "yr", "n_claims"]] for L in lines.values()])
    cy = allr.groupby(["customer_id", "yr"])["n_claims"].sum()
    for L in lines.values():
        n = pd.Series(cy.reindex(pd.MultiIndex.from_arrays([L.tr["customer_id"], L.tr["yr"] - 1])).to_numpy())
        L.h = np.where(n.isna(), "none", np.where(n >= 2, "2+", n.fillna(0).astype(int).astype(str)))
    ys, mus, ln = [], [], []
    for line, L in lines.items():
        yr = L.tr["yr"].to_numpy()
        fit, sc = yr == 2023, yr == 2024
        c = pc.candidates(L, P1[line], fit, {"incumbent": "K0"})
        mu = (c["K11_premium"] if line == "motor" else c["K0"] if line == "buildings" else c["K9+K7b"])[sc]
        y = L.y[sc]
        ys.append(y); mus.append(mu * opt_scale(y, mu)); ln.append(np.full(len(y), line))
    y, mu, ln = np.concatenate(ys), np.concatenate(mus), np.concatenate(ln)
    null = v.tweedie_unit_deviance(y, np.full(len(y), y.mean()), P).sum()
    base = v.tweedie_unit_deviance(y, mu, P)
    d2 = float(1 - base.sum() / null)
    res = {"d2_at_optimal_levels": d2, "rows": {l: int((ln == l).sum()) for l in dr.LINES}, "sensitivity": {}, "thirds": {}}
    for line in dr.LINES:
        m = ln == line
        res["sensitivity"][line] = {}
        for f in (0.8, 0.9, 0.95, 1.05, 1.1, 1.25):
            dd = v.tweedie_unit_deviance(y[m], mu[m] * f, P).sum() - base[m].sum()
            res["sensitivity"][line][str(f)] = float(-dd / null)
    rng = np.random.default_rng(0)
    draws = {l: [] for l in dr.LINES}
    for _ in range(1000):
        pub = rng.random(len(y)) < 1 / 3
        for line in dr.LINES:
            m = ln == line
            a = opt_scale(y[m & pub], mu[m & pub])
            b = opt_scale(y[m & ~pub], mu[m & ~pub])
            draws[line].append(np.log(a / b))
    for line, d in draws.items():
        d = np.array(d)
        res["thirds"][line] = dict(sd_log=float(d.std()), q05=float(np.exp(np.quantile(d, .05))), q95=float(np.exp(np.quantile(d, .95))))
    pc.log("D2 at optimal levels", round(d2, 4))
    for line in dr.LINES:
        pc.log(line, "D2 change if level off by", {k: round(x, 4) for k, x in res["sensitivity"][line].items()})
        t = res["thirds"][line]
        pc.log(line, "third's optimal level over the rest's: sd %.3f, 90%% %.3f to %.3f" % (t["sd_log"], t["q05"], t["q95"]))
    write_evidence({"part2_level_study": res}, OUT)


if __name__ == "__main__":
    main()
