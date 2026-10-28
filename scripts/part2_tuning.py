"""Part 2 tuning round (P2-F (a); decisions/part2_m2_design.md, tuning round, written before this ran).

Settings are chosen on Part 1 rows (fold 1 of five customer-grouped inner folds, seed 11; Tweedie 1.5 deviance on
capped cost at common level): the motor base GLM's and buildings attritional peril GLMs' training power, and the
contents tree's leaves and minimum leaf size. The tuned shape T is then tested against the incumbent I under P2-V.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_tuning.py
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
from src.glm import TweedieGLM, cat
from src.thresholds import load_thresholds

OUT = market.OUT / "part2_tuning.json"


def inner_fold(dv):
    gd = dv["customer_id"].astype(str).to_numpy()
    return v.make_folds(len(dv), [11], n_folds=5, groups=gd)[11][0]


def dev_common(y, mu):
    mu = mu * y.sum() / mu.sum()
    return float(v.tweedie_unit_deviance(y, mu, 1.5).sum())


def tune_power(F, y, ec, a, b, grid):
    res = {}
    for p in grid:
        m = TweedieGLM(power=p).fit(F.iloc[a], y[a], ec[a])
        res[round(p, 2)] = dev_common(y[b], m.rate(F.iloc[b]) * ec[b])
    return min(res, key=res.get), res


def main():
    th = load_thresholds()
    grid = list(np.round(np.arange(th["power_min"], th["power_max"] + 1e-9, th["power_step"]), 2))
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", pc.ROOT / "data" / "raw_part2" / "developed")
    lines, P1, tuned, report = {}, {}, {}, {}
    for line in dr.LINES:
        tr = data[line][0]
        L = pc.Line(line, tr[~tr["holdout"]].reset_index(drop=True), developed(line), w_em, w_dv)
        lines[line] = L
        P1[line] = pickle.loads((pc.WORK / f"p1_{line}.pkl").read_bytes())
        P1[line]["r3"] = pickle.loads((pc.WORK / f"p1r3_{line}.pkl").read_bytes())
        dv = L.dv
        a, b = inner_fold(dv)
        if line == "motor":
            hd, ht = pc.premium_hat(L, [L.tr])
            Fd, Ft = L.Fd.assign(lwp_hat=hd), L.Ft.assign(lwp_hat=ht)
            y, ec = dv["costc"].to_numpy(float), (dv["exposure"] * dv["completion"]).to_numpy(float)
            best, res = tune_power(Fd, y, ec, a, b, grid)
            tuned[line] = {"base": pc.rate_of(TweedieGLM(power=best).fit(Fd, y, ec), L, Ft)}
            report[line] = {"base_power": best, "grid_deviance": res}
        elif line == "buildings":
            per, rep = {}, {}
            for p in L.perils:
                if p in P1[line]["fs_hazard"]:
                    per[p] = P1[line]["fs_hazard"][p]
                    continue
                y = dv[f"costc_{p}"].to_numpy(float)
                ec = (dv["exposure"] * dv[f"completion_{p}"]).to_numpy(float)
                best, res = tune_power(L.Fd, y, ec, a, b, grid)
                per[p] = pc.rate_of(TweedieGLM(power=best).fit(L.Fd, y, ec), L, L.Ft)
                rep[p] = {"power": best, "grid_deviance": res}
            tuned[line] = {"peril": per}
            report[line] = rep
        else:
            cols = pc.common_tree_cols(L)
            fe_ = fields.tree_fields(L.tr, 11)
            cats = [c for c in cols if c in fe_["cat"]]
            both = as_category(pd.concat([dv[cols], L.tr[cols]], ignore_index=True), cats)
            Xd, Xe = both.iloc[:len(dv)].reset_index(drop=True), both.iloc[len(dv):].reset_index(drop=True)
            y, ec = dv["costc"].to_numpy(float), (dv["exposure"] * dv["completion"]).to_numpy(float)
            res = {}
            for leaves in (15, 31, 63):
                for ml in (100, 400, 1600):
                    m = LGB("tweedie", 1.5, leaves, ml, 0.05, 3000, 4, seed=11).fit(Xd.iloc[a], y[a], ec[a], val=(Xd.iloc[b], y[b], ec[b], None, None))
                    res[f"{leaves}|{ml}"] = (dev_common(y[b], m.rate(Xd.iloc[b]) * ec[b]), m.best_iter_)
            k = min(res, key=lambda t: res[t][0])
            leaves, ml = map(int, k.split("|"))
            m = LGB("tweedie", 1.5, leaves, ml, 0.05, res[k][1], 4, seed=11).fit(Xd, y, ec)
            tuned[line] = {"tree": m.rate(Xe)}
            report[line] = {"tree_setting": k, "trees": res[k][1], "grid_deviance": {t: r[0] for t, r in res.items()}}
        pc.log("tuned", line, {k: v_ for k, v_ in report[line].items() if k != "grid_deviance"} if line != "buildings" else {p: r["power"] for p, r in report[line].items()})

    allr = pd.concat([L.tr[["customer_id", "yr", "n_claims"]] for L in lines.values()])
    cy = allr.groupby(["customer_id", "yr"])["n_claims"].sum()
    for L in lines.values():
        n = pd.Series(cy.reindex(pd.MultiIndex.from_arrays([L.tr["customer_id"], L.tr["yr"] - 1])).to_numpy())
        L.h = np.where(n.isna(), "none", np.where(n >= 2, "2+", n.fillna(0).astype(int).astype(str)))

    base_candidates = pc.candidates

    def with_tuned(L, P, fit, stage):
        c = base_candidates(L, P, fit, stage)
        yf_b = c["K9+K7b"] / c[{"motor": "K0", "buildings": "K3", "contents": "tree"}[L.line]]
        if L.line == "motor":
            yf_b = yf_b / pc.layer(L, c["K0"], pd.DataFrame({"gender": cat(L.tr["gender"])}), fit) * c["K0"]
            c["I"] = c["K11_premium"]
            t = pc.layer(L, pc.line_level(L, tuned["motor"]["base"], fit), pd.DataFrame({"gender": cat(L.tr["gender"])}), fit)
        elif L.line == "buildings":
            c["I"] = c["K9+K7b"]
            t = pc.by_peril(L, tuned["buildings"]["peril"], fit, set())
        else:
            c["I"] = c["K9+K7b"]
            t = pc.line_level(L, tuned["contents"]["tree"], fit)
        c["T"] = t * yf_b
        return c

    pc.candidates = with_tuned
    S = pc.score(lines, P1, {"incumbent": "K0"})
    cmp = pc.compare(S, "I", "T")
    pc.log("T against I", {k: (round(x["gain"]), round(x["lo"]), round(x["hi"]), {l: round(g) for l, g in x["per_line"].items()}) for k, x in cmp.items() if k != "passes"}, "passes" if cmp["passes"] else "fails")
    d2 = pc.d2(S)
    pc.log("D2", {k: {n: round(x, 4) for n, x in d.items() if n in ("K0", "I", "T")} for k, d in d2.items()})
    write_evidence({"part2_tuning": dict(settings=report, T_against_I=cmp, d2={k: {n: x for n, x in d.items() if n in ("K0", "I", "T")} for k, d in d2.items()})}, OUT)


if __name__ == "__main__":
    main()
