"""Phase 5 step 8: family weights (J-model). Design written first: decisions/phase5_step8_design.md.

Reads the step 7 out-of-time predictions (data/work/phase5_step7/<line>_<A|B>.parquet). Per line:
  - weights on the simplex minimising Tweedie deviance (power 1.5) at a common level, arithmetic mix;
  - cross-fitted: weights fitted on A score B and weights fitted on B score A;
  - challengers: every candidate (within-family blend), a geometric mix;
  - final weights on A and B pooled.
Run: .venv/Scripts/python.exe -I scripts/phase5_step8.py
FAMILY_COLS below names the column that stands for each family; set it from the check-in 4 verdicts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.phase5_step6 import test
from src import market
from src import data_rules as dr, validation as v
from src.evidence import write_evidence
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT, WORK7, WORK = market.OUT, market.WORK / "phase5_step7", market.WORK / "phase5_step8"
P = 1.5
# Check-in 4 verdicts (D24): N1 (a) F4 with the held-over factors in motor and buildings; N2 (a) per-peril EBM and
# LightGBM in motor; N3 (a) the boosted correction may enter the final price
FAMILY_COLS = {"motor": {"F1": "F1", "F2": "FS:lgb", "F3": "FS:ebm", "F4": "F4+all"},
               "buildings": {"F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4+all"},
               "contents": {"F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4"}}
CANDIDATES = ["ref", "cand:glum", "cand:glum_std", "cand:glum_spline", "cand:lgb", "cand:cat", "cand:ebm", "cand:f4"]


def common(Pm: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each column rescaled to the actual total, so the weights decide ranking only."""
    return Pm * (y.sum() / Pm.sum(axis=0))


def deviance(y, mu):
    return v.tweedie_unit_deviance(y, mu, P).sum()


def fit_weights(Pm: np.ndarray, y: np.ndarray, geometric: bool = False) -> np.ndarray:
    k = Pm.shape[1]
    C = common(Pm, y)
    L = np.log(C)

    def mu_of(w):
        if geometric:
            m = np.exp(L @ w)
            return m * (y.sum() / m.sum())
        return C @ w

    def obj(w):
        return deviance(y, mu_of(w)) / len(y)

    def jac(w):
        mu = mu_of(w)
        g = 2 * np.power(mu, -P) * (mu - y)          # d deviance / d mu
        if geometric:
            return (L * (g * mu)[:, None]).sum(axis=0) / len(y)   # ignores the rescaling, which only shifts the level
        return (C * g[:, None]).sum(axis=0) / len(y)

    w0 = np.full(k, 1.0 / k)
    r = minimize(obj, w0, jac=None if geometric else jac, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                 constraints=[dict(type="eq", fun=lambda w: w.sum() - 1.0)], options=dict(maxiter=500, ftol=1e-12))
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def blend(Pm, y, w, geometric=False):
    C = common(Pm, y)
    if geometric:
        m = np.exp(np.log(C) @ w)
        return m * (y.sum() / m.sum())
    return C @ w


def run() -> None:
    th = load_thresholds()
    WORK.mkdir(parents=True, exist_ok=True)
    fr = {s: {l: pd.read_parquet(WORK7 / f"{l}_{s}.parquet") for l in dr.LINES} for s in ("A", "B")}
    weights, out_frames = {}, {s: [] for s in ("A", "B")}
    sets = {"families": lambda l: ["ref"] + list(FAMILY_COLS[l].values()), "within": lambda l: CANDIDATES}
    for l in dr.LINES:
        weights[l] = {}
        for sname, cols_of in sets.items():
            cols = cols_of(l)
            for geo in (False, True):
                key = f"{sname}|{'geo' if geo else 'arith'}"
                wA = fit_weights(fr["A"][l][cols].to_numpy(), fr["A"][l]["y"].to_numpy(), geo)
                wB = fit_weights(fr["B"][l][cols].to_numpy(), fr["B"][l]["y"].to_numpy(), geo)
                pool = pd.concat([fr["A"][l], fr["B"][l]], ignore_index=True)
                # pooled: each year rescaled to its own total first, so the two years' levels do not mix
                PmA = common(fr["A"][l][cols].to_numpy(), fr["A"][l]["y"].to_numpy())
                PmB = common(fr["B"][l][cols].to_numpy(), fr["B"][l]["y"].to_numpy())
                wP = fit_weights(np.vstack([PmA, PmB]), pool["y"].to_numpy(), geo)
                weights[l][key] = dict(cols=cols, fit_A=wA.tolist(), fit_B=wB.tolist(), pooled=wP.tolist())
                fr["A"][l][f"blend:{key}"] = blend(fr["A"][l][cols].to_numpy(), fr["A"][l]["y"].to_numpy(), wB, geo)
                fr["B"][l][f"blend:{key}"] = blend(fr["B"][l][cols].to_numpy(), fr["B"][l]["y"].to_numpy(), wA, geo)
        for s in ("A", "B"):
            out_frames[s].append(fr[s][l])
    frames = {s: pd.concat(out_frames[s], ignore_index=True) for s in ("A", "B")}
    for l in dr.LINES:
        for s in ("A", "B"):
            for fam, col in FAMILY_COLS[l].items():
                frames[s].loc[frames[s]["line"] == l, f"fam:{fam}"] = frames[s].loc[frames[s]["line"] == l, col]
    tests = {
        "blend vs W2 GLM": test(frames, "ref", "blend:families|arith", th),
        "within-family blend vs blend": test(frames, "blend:families|arith", "blend:within|arith", th),
        "geometric vs arithmetic": test(frames, "blend:families|arith", "blend:families|geo", th),
        **{f"blend vs {fam}": test(frames, f"fam:{fam}", "blend:families|arith", th) for fam in ("F1", "F2", "F3", "F4")},
    }
    d2 = {}
    for s, f in frames.items():
        for ln in list(dr.LINES) + ["all"]:
            d = f if ln == "all" else f[f["line"] == ln]
            yy = d["y"].to_numpy()
            for col in ["ref", "blend:families|arith", "blend:within|arith", "blend:families|geo"] + [f"fam:{x}" for x in ("F1", "F2", "F3", "F4")]:
                mu = v.rescale_to_total(d[col].to_numpy(), yy, by=d["line"].to_numpy())
                d2.setdefault(s, {}).setdefault(ln, {})[col] = float(v.d2_tweedie(yy, mu, d["exposure"]))
    for k, t in tests.items():
        print(k)
        for ln, r in t.items():
            a = r["A"]
            print(f"  {ln:9s} A {a['gain']:10.1f} [{a['boot']['lo']:10.1f}, {a['boot']['hi']:10.1f}]  B {r['B']['gain']:10.1f}  keep {r['verdict']['keep']}")
    for l, ws in weights.items():
        for key, w in ws.items():
            print(l, key, {c: round(x, 3) for c, x in zip(w["cols"], w["pooled"])})
    for s in ("A", "B"):
        frames[s].to_parquet(WORK / f"blend_{s}.parquet")
    write_evidence({"phase5_step8": dict(weights=weights, tests=tests, ranking_d2=d2, family_cols=FAMILY_COLS)}, OUT / "phase5_step8.json")


if __name__ == "__main__":
    run()
