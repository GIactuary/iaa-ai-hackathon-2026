"""Motor by peril (K16; decisions/part2_m2_design.md, written before this official run, after the gap reviews).

K16 prices motor with one Part 1 GLM per peril (W2 factors plus lwp_hat), each peril at its emerging level, then F's
gender layer and own-year weather. Buildings and contents keep F. Scored against F, line by line:
  P2-V on every emerging customer (V1 fit 2023, score 2024; V2 customer-grouped five-fold, seeds 11, 22, 33), as K15;
  H1 and H2 on the original hold-out customers, as scripts/part2_holdout.py;
  a supplementary reading with each row weighted to the 2025 mix of new business and age band.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_motor_peril.py
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

OUT = market.OUT / "part2_motor_peril.json"
# the gap reviews' motor gains for K16 against F (A batch3 / batch5, B e2_motor_peril); the check is 5% (design note)
REVIEWED = {"V1": 9965, "V2": 13443, "H1": 3454, "H2": 5952}


def final_shape(c, line):
    return c["K11_premium"] if line == "motor" else c["K0"] if line == "buildings" else c["K9+K7b"]


def mix_weights(tr: pd.DataFrame, te: pd.DataFrame, line: str) -> np.ndarray:
    """2025 exposure share over training exposure share, by new business and age band (supplementary reading only)."""
    age = "driver_age_band" if line == "motor" else "policyholder_age_band"
    def cell(df):
        return df["new_business"].astype(bool).astype(str) + "|" + df[age].astype(str)
    ct, ce = cell(tr), cell(te)
    st = tr["exposure"].groupby(ct).sum() / tr["exposure"].sum()
    se = te["exposure"].groupby(ce).sum() / te["exposure"].sum()
    return ct.map(se / st).fillna(1.0).to_numpy(float)


def by_line(S: dict, a: str, b: str, w: dict | None = None) -> dict:
    """b against a per line and pooled, V1 and V2; gain > 0 means b is better. w: row weights per test."""
    res = {}
    for scope in ("pooled",) + dr.LINES:
        r = {}
        for name in ("V1", "V2"):
            m = np.ones(len(S[name]["_g"]), bool) if scope == "pooled" else (S[name]["_line"] == scope)
            ww = w[name][m] if w else 1.0
            bt = v.paired_bootstrap_groups(S[name][a][m] * ww, S[name][b][m] * ww, S[name]["_g"][m])
            r[name] = dict(gain=bt["gain"], lo=bt["lo"], hi=bt["hi"])
        r["passes"] = bool(r["V1"]["gain"] > 0 and r["V2"]["gain"] > 0 and r["V2"]["lo"] > 0)
        res[scope] = r
    return res


def holdout(lines: dict, P1: dict) -> dict:
    """H1 and H2 as scripts/part2_holdout.py, F against K16."""
    res = {}
    for name in ("H1", "H2"):
        acc = {}
        for line, L in lines.items():
            hold, yr = L.tr["holdout"].to_numpy(bool), L.tr["yr"].to_numpy()
            fit, sc = (~hold & (yr == 2023), hold & (yr == 2024)) if name == "H1" else (~hold, hold)
            c = pc.candidates(L, P1[line], fit, {"incumbent": "K0"})
            F = final_shape(c, line)
            y = L.y[sc]
            for k, mu in (("F", F), ("K16", c.get("K16_motor_peril", F))):
                acc.setdefault(k, []).append(v.tweedie_unit_deviance(y, mu[sc] * y.sum() / mu[sc].sum(), 1.5))
            acc.setdefault("_y", []).append(y)
            acc.setdefault("_g", []).append(L.g[sc])
            acc.setdefault("_line", []).append(np.full(int(sc.sum()), line))
        S = {k: np.concatenate(x) for k, x in acc.items()}
        r = {}
        for scope in ("pooled",) + dr.LINES:
            m = np.ones(len(S["_g"]), bool) if scope == "pooled" else S["_line"] == scope
            bt = v.paired_bootstrap_groups(S["F"][m], S["K16"][m], S["_g"][m])
            r[scope] = dict(gain=bt["gain"], lo=bt["lo"], hi=bt["hi"])
        y = S["_y"]
        null = v.tweedie_unit_deviance(y, np.full(len(y), y.mean()), 1.5).sum()
        r["d2"] = {k: float(1 - S[k].sum() / null) for k in ("F", "K16")}
        res[name] = r
        pc.log(name, {s: (round(x["gain"]), round(x["lo"]), round(x["hi"])) for s, x in r.items() if s != "d2"}, "D2", r["d2"])
    return res


def main():
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", pc.ROOT / "data" / "raw_part2" / "developed")
    lines, P1, W = {}, {}, {}
    for line in dr.LINES:
        L = pc.Line(line, data[line][0].reset_index(drop=True), developed(line), w_em, w_dv)
        lines[line] = L
        cache = pc.WORK / f"h_p1_{line}.pkl"
        P1[line] = pickle.loads(cache.read_bytes())
        if line == "motor" and "peril" not in P1[line]["r3"]:
            hd, ht = pc.premium_hat(L, [L.tr])
            gap = float(np.max(np.abs(ht - P1[line]["r3"]["lwp_tr"])))
            pc.log("premium term refitted; largest difference from the cached one", gap)
            assert gap < 1e-6, "the premium term does not reproduce; refit every r3 part instead"
            P1[line]["r3"]["peril"] = pc.motor_perils_r3(L, L.Fd.assign(lwp_hat=hd), L.Ft.assign(lwp_hat=ht))
            cache.write_bytes(pickle.dumps(P1[line]))
        W[line] = mix_weights(L.tr, data[line][1], line)
    allr = pd.concat([L.tr[["customer_id", "yr", "n_claims"]] for L in lines.values()])
    cy = allr.groupby(["customer_id", "yr"])["n_claims"].sum()
    for L in lines.values():
        n = pd.Series(cy.reindex(pd.MultiIndex.from_arrays([L.tr["customer_id"], L.tr["yr"] - 1])).to_numpy())
        L.h = np.where(n.isna(), "none", np.where(n >= 2, "2+", n.fillna(0).astype(int).astype(str)))

    base_candidates = pc.candidates

    def with_k16(L, P1_, fit, stage):
        c = base_candidates(L, P1_, fit, stage)
        c["F"] = final_shape(c, L.line)
        c["K16"] = c.get("K16_motor_peril", c["F"])
        return c

    pc.candidates = with_k16
    S = pc.score(lines, P1, {"incumbent": "K0"})
    pc.candidates = base_candidates
    w = {name: np.concatenate([W[l][(lines[l].tr["yr"].to_numpy() == 2024) if name == "V1" else slice(None)] for l in dr.LINES])
         for name in ("V1", "V2")}
    res = {"p2v": by_line(S, "F", "K16"), "p2v_2025_mix": by_line(S, "F", "K16", w)}
    res["d2"] = {k: {n: x for n, x in d.items() if n in ("K0", "F", "K16", "K1")} for k, d in pc.d2(S).items()}
    for key in ("p2v", "p2v_2025_mix"):
        for scope, r in res[key].items():
            pc.log(key, scope, {k: (round(x["gain"]), round(x["lo"]), round(x["hi"])) for k, x in r.items() if k != "passes"}, "passes" if r["passes"] else "fails")
    pc.log("D2", {k: {n: round(x, 4) for n, x in d.items()} for k, d in res["d2"].items()})
    res["holdout"] = holdout(lines, P1)

    got = {"V1": res["p2v"]["motor"]["V1"]["gain"], "V2": res["p2v"]["motor"]["V2"]["gain"],
           "H1": res["holdout"]["H1"]["motor"]["gain"], "H2": res["holdout"]["H2"]["motor"]["gain"]}
    res["reproduction"] = {k: dict(reviewed=REVIEWED[k], official=got[k], ratio=got[k] / REVIEWED[k],
                                   within_5pct=bool(abs(got[k] / REVIEWED[k] - 1) <= 0.05)) for k in REVIEWED}
    pc.log("reproduction", {k: (round(x["official"]), x["reviewed"], round(x["ratio"], 3)) for k, x in res["reproduction"].items()})

    # exhibit: each motor peril's emerging level against Part 1's, on every row (cost shares, capped)
    L = lines["motor"]
    em = {p: float(L.tr[f"costc_{p}"].sum()) for p in L.perils}
    dv = {p: float(L.dv[f"costc_{p}"].sum()) for p in L.perils}
    res["peril_shares"] = {p: dict(emerging=em[p] / sum(em.values()), part1=dv[p] / sum(dv.values()),
                                   ratio=(em[p] / sum(em.values())) / (dv[p] / sum(dv.values()))) for p in L.perils}
    pc.log("motor capped cost shares, emerging against Part 1", {p: (round(x["emerging"], 3), round(x["part1"], 3)) for p, x in res["peril_shares"].items()})
    write_evidence({"part2_motor_peril": res}, OUT)


if __name__ == "__main__":
    main()
