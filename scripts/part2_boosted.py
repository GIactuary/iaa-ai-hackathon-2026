"""Boosted correction (K15; decisions/part2_m2_design.md, written before this ran).

Per line, a LightGBM on emerging rows starts from the final model's prediction F (as the offset) and learns only what
F misses. Tested against F under P2-V on every emerging customer, line by line; the exhibit reports the fields the
correction uses most, from a fit on all rows.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_boosted.py
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
from src import build, data_rules as dr, fields, market, validation as v
from src.evidence import write_evidence
from src.families import LGB, as_category
from src.glm import WEATHER_VARS

OUT = market.OUT / "part2_boosted.json"


def inputs(L) -> pd.DataFrame:
    fe_ = fields.tree_fields(L.tr, 11)
    drop = {"yr", "holdout"}
    num = [c for c in fe_["num"] if c not in drop]
    cat_ = [c for c in fe_["cat"] if c not in drop]
    X = L.tr[num + cat_].copy()
    for wv in WEATHER_VARS:
        X[f"norm_{wv}"] = L.trx[f"norm_{wv}"].to_numpy(float)
    return as_category(X.reset_index(drop=True), cat_)


def correction(L, X: pd.DataFrame, mu: np.ndarray, fit: np.ndarray, contrib: bool = False):
    """F x the boosted correction; the correction is fitted on `fit` rows with F (at the fitting rows' level) as offset."""
    y = L.tr["costc"].to_numpy(float)
    off = mu * y[fit].sum() / mu[fit].sum()
    fi = np.flatnonzero(fit)
    a, b = v.make_folds(len(fi), [11], n_folds=5, groups=L.tr["customer_id"].astype(str).to_numpy()[fi])[11][0]
    ia, ib = fi[a], fi[b]
    m = LGB("tweedie", 1.5, 15, 400, 0.02, 2000, 4, seed=11).fit(X.iloc[ia], y[ia], off[ia], val=(X.iloc[ib], y[ib], off[ib], None, None))
    n = m.best_iter_
    m = LGB("tweedie", 1.5, 15, 400, 0.02, n, 4, seed=11).fit(X.iloc[fi], y[fi], off[fi])
    out = mu * m.rate(X)
    if not contrib:
        return out, n
    c = m.contrib(X.iloc[fi])[:, :-1]
    imp = pd.Series(np.abs(c).mean(axis=0), index=X.columns).sort_values(ascending=False)
    spread = np.log(m.rate(X.iloc[fi]))
    return out, n, imp, spread


def main():
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", pc.ROOT / "data" / "raw_part2" / "developed")
    lines, P1, XS = {}, {}, {}
    for line in dr.LINES:
        L = pc.Line(line, data[line][0].reset_index(drop=True), developed(line), w_em, w_dv)
        lines[line] = L
        P1[line] = pickle.loads((pc.WORK / f"h_p1_{line}.pkl").read_bytes())
        XS[line] = inputs(L)
    allr = pd.concat([L.tr[["customer_id", "yr", "n_claims"]] for L in lines.values()])
    cy = allr.groupby(["customer_id", "yr"])["n_claims"].sum()
    for L in lines.values():
        n = pd.Series(cy.reindex(pd.MultiIndex.from_arrays([L.tr["customer_id"], L.tr["yr"] - 1])).to_numpy())
        L.h = np.where(n.isna(), "none", np.where(n >= 2, "2+", n.fillna(0).astype(int).astype(str)))

    base_candidates = pc.candidates
    TREES = {}

    def final_shape(c, line):
        return c["K11_premium"] if line == "motor" else c["K0"] if line == "buildings" else c["K9+K7b"]

    def with_boost(L, P1_, fit, stage):
        c = base_candidates(L, P1_, fit, stage)
        c["F"] = final_shape(c, L.line)
        c["K15"], n = correction(L, XS[L.line], c["F"], fit)
        TREES.setdefault(L.line, []).append(n)
        return c

    pc.candidates = with_boost
    S = pc.score(lines, P1, {"incumbent": "K0"})
    res = {"pooled": pc.compare(S, "F", "K15"), "per_line": {}, "trees_per_split": TREES}
    for line in dr.LINES:
        r = {}
        for name in ("V1", "V2"):
            m = S[name]["_line"] == line
            bt = v.paired_bootstrap_groups(S[name]["F"][m], S[name]["K15"][m], S[name]["_g"][m])
            r[name] = dict(gain=bt["gain"], lo=bt["lo"], hi=bt["hi"])
        r["passes"] = bool(r["V1"]["gain"] > 0 and r["V2"]["gain"] > 0 and r["V2"]["lo"] > 0)
        res["per_line"][line] = r
        pc.log(line, {k: (round(x["gain"]), round(x["lo"]), round(x["hi"])) for k, x in r.items() if k != "passes"}, "passes" if r["passes"] else "fails",
               "trees per split", TREES[line])
    p = res["pooled"]
    pc.log("pooled", {k: (round(x["gain"]), round(x["lo"]), round(x["hi"])) for k, x in p.items() if k != "passes"})
    res["d2"] = {k: {n: x for n, x in d.items() if n in ("K0", "F", "K15")} for k, d in pc.d2(S).items()}
    pc.log("D2", {k: {n: round(x, 4) for n, x in d.items()} for k, d in res["d2"].items()})

    # exhibit: what the correction uses, fitted on every row
    pc.candidates = base_candidates
    res["exhibit"] = {}
    for line, L in lines.items():
        allfit = np.ones(len(L.tr), bool)
        c = base_candidates(L, P1[line], allfit, {"incumbent": "K0"})
        _, n, imp, spread = correction(L, XS[line], final_shape(c, line), allfit, contrib=True)
        res["exhibit"][line] = dict(trees=n, top_fields={k: float(x) for k, x in imp.head(12).items()},
                                    correction_p05_p95=[float(np.exp(np.quantile(spread, .05))), float(np.exp(np.quantile(spread, .95)))])
        pc.log("exhibit", line, "trees", n, "correction 5%-95%", [round(x, 3) for x in res["exhibit"][line]["correction_p05_p95"]],
               {k: round(x, 4) for k, x in imp.head(8).items()})
    write_evidence({"part2_boosted": res}, OUT)


if __name__ == "__main__":
    main()
