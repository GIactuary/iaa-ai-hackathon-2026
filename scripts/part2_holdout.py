"""Part 2 hold-out opening (P2-H; decisions/part2_m2_design.md, hold-out section, written before it was opened).

The frozen model I against K0 (the check file's shape) and K9+K7b (the improved file before the motor premium), on
hold-out customers only. H1: fit non-hold-out 2023, score hold-out 2024. H2: fit all non-hold-out rows, score every
hold-out row. Deviance at a common level per line, as P2-V.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_holdout.py
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

OUT = market.OUT / "part2_holdout.json"
KEEP = ("K0", "K9+K7b", "I")


def main():
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", pc.ROOT / "data" / "raw_part2" / "developed")
    lines, P1 = {}, {}
    for line in dr.LINES:
        L = pc.Line(line, data[line][0].reset_index(drop=True), developed(line), w_em, w_dv)
        lines[line] = L
        cache = pc.WORK / f"h_p1_{line}.pkl"
        if cache.exists():
            P1[line] = pickle.loads(cache.read_bytes())
        else:
            P1[line] = pc.part1_fits(L)
            P1[line]["r3"] = pc.part1_fits_r3(L)
            cache.write_bytes(pickle.dumps(P1[line]))
        pc.log("Part 1 fits on the full frame", line, "hold-out rows", int(L.tr["holdout"].sum()))
    allr = pd.concat([L.tr[["customer_id", "yr", "n_claims"]] for L in lines.values()])
    cy = allr.groupby(["customer_id", "yr"])["n_claims"].sum()
    for L in lines.values():
        n = pd.Series(cy.reindex(pd.MultiIndex.from_arrays([L.tr["customer_id"], L.tr["yr"] - 1])).to_numpy())
        L.h = np.where(n.isna(), "none", np.where(n >= 2, "2+", n.fillna(0).astype(int).astype(str)))

    res = {}
    for name in ("H1", "H2"):
        acc, level, claims = {}, {}, {}
        for line, L in lines.items():
            hold, yr = L.tr["holdout"].to_numpy(bool), L.tr["yr"].to_numpy()
            fit, sc = (~hold & (yr == 2023), hold & (yr == 2024)) if name == "H1" else (~hold, hold)
            c = pc.candidates(L, P1[line], fit, {"incumbent": "K0"})
            c["I"] = c["K11_premium"] if line == "motor" else c["K9+K7b"]
            y = L.y[sc]
            for k in KEEP:
                mu = c[k][sc]
                level[f"{line}|{k}"] = float(mu.sum() / y.sum())
                acc.setdefault(k, []).append(v.tweedie_unit_deviance(y, mu * y.sum() / mu.sum(), 1.5))
            acc.setdefault("_y", []).append(y)
            acc.setdefault("_g", []).append(L.g[sc])
            acc.setdefault("_line", []).append(np.full(int(sc.sum()), line))
            claims[line] = dict(rows=int(sc.sum()), claims=int(L.tr.loc[sc, "n_claims"].sum()), cost=float(y.sum()),
                                largest_claim_row=float(y.max()))
        S = {k: np.concatenate(x) for k, x in acc.items()}
        y, ln = S["_y"], S["_line"]
        d2 = {}
        for k in KEEP:
            null = v.tweedie_unit_deviance(y, np.full(len(y), y.mean()), 1.5).sum()
            d2[k] = {"pooled": float(1 - S[k].sum() / null)}
            for line in dr.LINES:
                m = ln == line
                nl = v.tweedie_unit_deviance(y[m], np.full(int(m.sum()), y[m].mean()), 1.5).sum()
                d2[k][line] = float(1 - S[k][m].sum() / nl)
        cmp = {}
        for a in ("K0", "K9+K7b"):
            bt = v.paired_bootstrap_groups(S[a], S["I"], S["_g"])
            cmp[f"I_vs_{a}"] = dict(gain=bt["gain"], lo=bt["lo"], hi=bt["hi"],
                                    per_line={l: float((S[a] - S["I"])[ln == l].sum()) for l in dr.LINES})
        res[name] = dict(d2=d2, compare=cmp, level_ratio=level, scored=claims)
        pc.log(name, "D2", {k: {l: round(x, 4) for l, x in d.items()} for k, d in d2.items()})
        pc.log(name, {k: (round(x["gain"]), round(x["lo"]), round(x["hi"]), {l: round(g) for l, g in x["per_line"].items()}) for k, x in cmp.items()})
        pc.log(name, "scored", claims)
    res["reading"] = ("final file is I" if any(res[h]["compare"]["I_vs_K0"]["gain"] > 0 for h in ("H1", "H2"))
                      else "I worse than K0 in both: the actuary decides")
    pc.log("reading:", res["reading"])
    write_evidence({"part2_holdout": res}, OUT)


if __name__ == "__main__":
    main()
