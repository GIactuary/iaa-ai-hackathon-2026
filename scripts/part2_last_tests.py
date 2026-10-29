"""Two last structural tests (K17 pooled tree, K18 premium remainder; decisions/part2_m2_design.md, written before this ran).

Incumbent C: motor by peril (K16), buildings K0, contents K9+K7b; motor also against K11 (the final file's motor).
  K18  the part of the emerging log written premium our fields do not predict, as one GLM term on top of C
  K17  one LightGBM per line on Part 1 and emerging fitting rows together, with a market flag
Scored per line under P2-V on every emerging customer, with the 2025-mix reading and hold-out H1 and H2 beside it.
Stage k18 runs in minutes; stage k17 refits a tree on about a million rows per split.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_last_tests.py k18 | k17
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import scripts.part2_candidates as pc
from scripts.part2_model import developed
from scripts.part2_motor_peril import by_line, final_shape, mix_weights
from src import build, data_rules as dr, fields, market, validation as v
from src.evidence import write_evidence
from src.families import LGB, as_category
from src.glm import cat

OUT = market.OUT / "part2_last_tests.json"


def current_shape(c, line):
    return c["K16_motor_peril"] if line == "motor" else final_shape(c, line)


class Pooled:
    """K17 inputs for one line: Part 1 rows then emerging rows, the fields both markets carry plus a market flag."""

    def __init__(self, L):
        cols = pc.common_tree_cols(L)
        cats = [c for c in cols if c in fields.tree_fields(L.tr, 11)["cat"]]
        both = pd.concat([L.dv[cols], L.tr[cols]], ignore_index=True)
        both["market"] = ["developed"] * len(L.dv) + ["emerging"] * len(L.tr)
        X = as_category(both, cats + ["market"])
        self.Xd, self.Xe = X.iloc[:len(L.dv)].reset_index(drop=True), X.iloc[len(L.dv):].reset_index(drop=True)
        self.yd = L.dv["costc"].to_numpy(float)
        self.ecd = (L.dv["exposure"] * L.dv["completion"]).to_numpy(float)
        self.ye = L.tr["costc"].to_numpy(float)
        self.ece = L.e * L.tr["completion"].to_numpy(float)
        self.ge = L.tr["customer_id"].astype(str).to_numpy()

    def rate(self, fit: np.ndarray) -> tuple[np.ndarray, int]:
        fi = np.flatnonzero(fit)
        a, b = v.make_folds(len(fi), [11], n_folds=5, groups=self.ge[fi])[11][0]
        ia, ib = fi[a], fi[b]
        Xa = pd.concat([self.Xd, self.Xe.iloc[ia]], ignore_index=True)
        ya, ea = np.concatenate([self.yd, self.ye[ia]]), np.concatenate([self.ecd, self.ece[ia]])
        m = LGB("tweedie", 1.5, 31, 400, 0.05, 3000, 4, seed=11).fit(Xa, ya, ea, val=(self.Xe.iloc[ib], self.ye[ib], self.ece[ib], None, None))
        n = m.best_iter_
        Xf = pd.concat([self.Xd, self.Xe.iloc[fi]], ignore_index=True)
        m = LGB("tweedie", 1.5, 31, 400, 0.05, n, 4, seed=11).fit(Xf, np.concatenate([self.yd, self.ye[fi]]), np.concatenate([self.ecd, self.ece[fi]]))
        return m.rate(self.Xe), n


def main():
    stage = sys.argv[1]
    assert stage in ("k18", "k17")
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", pc.ROOT / "data" / "raw_part2" / "developed")
    lines, P1, W, RES, POOL = {}, {}, {}, {}, {}
    for line in dr.LINES:
        L = pc.Line(line, data[line][0].reset_index(drop=True), developed(line), w_em, w_dv)
        lines[line] = L
        P1[line] = pickle.loads((pc.WORK / f"h_p1_{line}.pkl").read_bytes())
        W[line] = mix_weights(L.tr, data[line][1], line)
        RES[line] = np.log(L.tr["written_premium"].to_numpy(float)) - P1[line]["r3"]["lwp_tr"]
        if stage == "k17":
            POOL[line] = Pooled(L)
    assert "peril" in P1["motor"]["r3"], "run scripts/part2_motor_peril.py first"
    allr = pd.concat([L.tr[["customer_id", "yr", "n_claims"]] for L in lines.values()])
    cy = allr.groupby(["customer_id", "yr"])["n_claims"].sum()
    for L in lines.values():
        n = pd.Series(cy.reindex(pd.MultiIndex.from_arrays([L.tr["customer_id"], L.tr["yr"] - 1])).to_numpy())
        L.h = np.where(n.isna(), "none", np.where(n >= 2, "2+", n.fillna(0).astype(int).astype(str)))

    base_candidates = pc.candidates
    TREES = {}
    name = "K18" if stage == "k18" else "K17"

    def with_new(L, P1_, fit, stage_):
        c = base_candidates(L, P1_, fit, stage_)
        out = {"C": current_shape(c, L.line), "F": final_shape(c, L.line)}
        if name == "K18":
            out["K18"] = pc.layer(L, out["C"], pd.DataFrame({"lwp_res": RES[L.line]}), fit)
        else:
            r, n = POOL[L.line].rate(fit)
            TREES.setdefault(L.line, []).append(n)
            mu = pc.line_level(L, r, fit) * L.year_factor(pc.elasticities(L, fit, pc.WIDE_WEATHER[L.line]), fit)
            if L.line == "motor":
                mu = pc.layer(L, mu, pd.DataFrame({"gender": cat(L.tr["gender"])}), fit)
            out["K17"] = mu
            pc.log("K17", L.line, "trees", n)
        return out

    pc.candidates = with_new
    S = pc.score(lines, P1, {"incumbent": "K0"})
    w = {nm: np.concatenate([W[l][(lines[l].tr["yr"].to_numpy() == 2024) if nm == "V1" else slice(None)] for l in dr.LINES])
         for nm in ("V1", "V2")}
    res = {"p2v": by_line(S, "C", name), "p2v_2025_mix": by_line(S, "C", name, w),
           "motor_vs_final_file": by_line(S, "F", name)["motor"]}
    d2 = pc.d2(S)
    res["d2"] = {k: {n: x for n, x in d.items() if n in ("C", "F", name)} for k, d in d2.items()}
    for key in ("p2v", "p2v_2025_mix"):
        for scope, r in res[key].items():
            pc.log(name, key, scope, {k: (round(x["gain"]), round(x["lo"]), round(x["hi"])) for k, x in r.items() if k != "passes"}, "passes" if r["passes"] else "fails")
    pc.log("D2", {k: {n: round(x, 4) for n, x in d.items()} for k, d in res["d2"].items()})

    # hold-out readings, as scripts/part2_holdout.py: fitted without the hold-out customers
    res["holdout"] = {}
    for hn in ("H1", "H2"):
        acc = {}
        for line, L in lines.items():
            hold, yr = L.tr["holdout"].to_numpy(bool), L.tr["yr"].to_numpy()
            fit, sc = (~hold & (yr == 2023), hold & (yr == 2024)) if hn == "H1" else (~hold, hold)
            c = with_new(L, P1[line], fit, {"incumbent": "K0"})
            y = L.y[sc]
            for k in ("C", name):
                acc.setdefault(k, []).append(v.tweedie_unit_deviance(y, c[k][sc] * y.sum() / c[k][sc].sum(), 1.5))
            acc.setdefault("_g", []).append(L.g[sc])
            acc.setdefault("_line", []).append(np.full(int(sc.sum()), line))
        H = {k: np.concatenate(x) for k, x in acc.items()}
        r = {}
        for scope in ("pooled",) + dr.LINES:
            m = np.ones(len(H["_g"]), bool) if scope == "pooled" else H["_line"] == scope
            bt = v.paired_bootstrap_groups(H["C"][m], H[name][m], H["_g"][m])
            r[scope] = dict(gain=bt["gain"], lo=bt["lo"], hi=bt["hi"])
        res["holdout"][hn] = r
        pc.log(name, hn, {s: (round(x["gain"]), round(x["lo"]), round(x["hi"])) for s, x in r.items()})
    pc.candidates = base_candidates
    if TREES:
        res["trees_per_fit"] = TREES
    prev = json.loads(OUT.read_text(encoding="utf-8")).get("part2_last_tests", {}) if OUT.exists() else {}
    prev.pop("meta", None)
    prev[name] = res
    write_evidence({"part2_last_tests": prev}, OUT)


if __name__ == "__main__":
    main()
