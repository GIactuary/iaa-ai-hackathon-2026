"""Motor credibility re-rate (K14; decisions/part2_m2_design.md, written before this ran).

On the fitting rows of each split, one Tweedie GLM layer on the incumbent I gives each motor factor level's deviation
from Part 1's relativity; a customer-clustered sandwich gives its standard error; each factor's deviations are then
credibility-weighted toward zero (empirical Bayes: Z = tau2 / (tau2 + se2), tau2 from the factor's own levels).
K14 is tested against I under P2-V; the full-weight layer (Z = 1) is scored beside it as the other end of the bracket.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_motor_rerate.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.sparse as sp

import scripts.part2_candidates as pc
from scripts.part2_model import developed
from src import build, data_rules as dr, market, validation as v
from src.evidence import write_evidence
from src.glm import TweedieGLM, cat

OUT = market.OUT / "part2_motor_rerate.json"
P = 1.5


def credibility_layer(L, mu: np.ndarray, F: pd.DataFrame, fit: np.ndarray) -> tuple:
    """Returns (credibility-weighted prediction, full-weight prediction, per-factor table)."""
    y = L.tr["costc"].to_numpy(float)
    m = TweedieGLM(power=P).fit(F[fit], y[fit], mu[fit])
    X = sp.hstack([sp.csr_matrix(np.ones((int(fit.sum()), 1))), m._x(F[fit])]).tocsr()
    b = np.r_[m.model_.intercept_, m.model_.coef_]
    r, w = y[fit] / mu[fit], np.power(mu[fit], 2 - P)
    mh = np.exp(X @ b)
    A = (X.T @ X.multiply((w * mh ** (2 - P))[:, None])).toarray()
    s = w * (r - mh) * mh ** (1 - P)
    g = pd.factorize(L.tr.loc[fit, "customer_id"].astype(str))[0]
    G = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(g.max() + 1, len(g)))
    Sc = (G @ X.multiply(s[:, None])).toarray()
    Ai = np.linalg.pinv(A)
    se = np.sqrt(np.clip(np.diag(Ai @ (Sc.T @ Sc) @ Ai), 0, None))[1:]
    beta = b[1:]
    shrunk, table, j = np.zeros_like(beta), [], 0
    for c in m.columns_:
        k = len(m.levels_[c]) - 1
        bf, sf = beta[j:j + k], se[j:j + k]
        tau2 = max(0.0, float(np.mean(bf ** 2 - sf ** 2))) if k else 0.0
        z = tau2 / (tau2 + sf ** 2) if tau2 > 0 else np.zeros(k)
        shrunk[j:j + k] = z * bf
        table.append(dict(factor=c, levels=k, tau=float(np.sqrt(tau2)), mean_z=float(z.mean()) if k else 0.0,
                          detail=[dict(level=str(l), dev=float(np.exp(x)), se=float(e), z=float(zz), cred_dev=float(np.exp(zz * x)))
                                  for l, x, e, zz in zip(m.levels_[c][1:], bf, sf, z)]))
        j += k
    f = F[m.columns_].copy()
    for c in m.columns_:
        f[c] = f[c].where(f[c].isin(m.levels_[c]), m.levels_[c][0])
    cred = mu * np.exp(m._x(f) @ shrunk)
    full = mu * m.rate(F)
    cred *= mu[fit].sum() / cred[fit].sum()
    full *= mu[fit].sum() / full[fit].sum()
    return cred, full, table


def factors(L) -> pd.DataFrame:
    return L.Ft.drop(columns="yr").assign(region=cat(L.tr["region"]))


def main():
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", pc.ROOT / "data" / "raw_part2" / "developed")
    lines, P1 = {}, {}
    for line in dr.LINES:
        tr = data[line][0]
        lines[line] = pc.Line(line, tr[~tr["holdout"]].reset_index(drop=True), developed(line), w_em, w_dv)
        P1[line] = pickle.loads((pc.WORK / f"p1_{line}.pkl").read_bytes())
        P1[line]["r3"] = pickle.loads((pc.WORK / f"p1r3_{line}.pkl").read_bytes())
    allr = pd.concat([L.tr[["customer_id", "yr", "n_claims"]] for L in lines.values()])
    cy = allr.groupby(["customer_id", "yr"])["n_claims"].sum()
    for L in lines.values():
        n = pd.Series(cy.reindex(pd.MultiIndex.from_arrays([L.tr["customer_id"], L.tr["yr"] - 1])).to_numpy())
        L.h = np.where(n.isna(), "none", np.where(n >= 2, "2+", n.fillna(0).astype(int).astype(str)))

    base_candidates = pc.candidates

    def with_rerate(L, P1_, fit, stage):
        c = base_candidates(L, P1_, fit, stage)
        if L.line == "motor":
            c["I"] = c["K11_premium"]
            c["K14_cred"], c["K14_full"], _ = credibility_layer(L, c["I"], factors(L), fit)
        else:
            c["I"] = c["K9+K7b"]
            c["K14_cred"] = c["K14_full"] = c["I"]
        return c

    pc.candidates = with_rerate
    S = pc.score(lines, P1, {"incumbent": "K0"})
    res = {}
    for cand in ("K14_cred", "K14_full"):
        cmp = pc.compare(S, "I", cand)
        res[cand] = cmp
        pc.log(cand, "against I", {k: (round(x["gain"]), round(x["lo"]), round(x["hi"])) for k, x in cmp.items() if k != "passes"},
               "passes" if cmp["passes"] else "fails")
    d2 = pc.d2(S)
    res["d2"] = {k: {n: x for n, x in d.items() if n in ("K0", "I", "K14_cred", "K14_full")} for k, d in d2.items()}
    pc.log("D2", {k: {n: round(x, 4) for n, x in d.items()} for k, d in res["d2"].items()})

    # M2-R: expected public gain against its paired spread on random thirds
    board = {}
    for name in ("V1", "V2"):
        s = S[name]
        inc, cand, y = s["I"], s["K14_cred"], s["_y"]
        null = v.tweedie_unit_deviance(y, np.full(len(y), y.mean()), 1.5)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(1000):
            mk = rng.random(len(y)) < 1 / 3
            diffs.append((inc[mk].sum() - cand[mk].sum()) / null[mk].sum())
        diffs = np.array(diffs)
        board[name] = dict(d2_gain=float((inc.sum() - cand.sum()) / null.sum()), sd=float(diffs.std()),
                           q05=float(np.quantile(diffs, .05)), q95=float(np.quantile(diffs, .95)), looks_worse=float(np.mean(diffs < 0)))
    res["board"] = board
    pc.log("board", {k: {n: round(x, 5) for n, x in b.items()} for k, b in board.items()})

    # exhibit: each factor's spread and credibility, fitted on all fitting rows
    pc.candidates = base_candidates
    L = lines["motor"]
    allfit = np.ones(len(L.tr), bool)
    c = base_candidates(L, P1["motor"], allfit, {"incumbent": "K0"})
    _, _, table = credibility_layer(L, c["K11_premium"], factors(L), allfit)
    res["factors_all_rows"] = table
    for t in table:
        pc.log("factor", t["factor"], "levels", t["levels"], "tau %.3f" % t["tau"], "mean Z %.2f" % t["mean_z"])
    write_evidence({"part2_motor_rerate": res}, OUT)


if __name__ == "__main__":
    main()
