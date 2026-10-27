"""Addendum checks to the Part 2 decision evidence (decisions/part2_decision_evidence_design.md, addendum):
controlled new-business and gender relativities from a Poisson GLM, and large losses by peril against the developed
market's exceedance rates.
Run: .venv/Scripts/python.exe -I scripts/part2_decision_evidence_extra.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from part2_decision_evidence import CAPS, DEV, EMG, rows
from src import data_rules as dr
from src.evidence import write_evidence
from src.glm import band, cat, quantile_edges

OUT = Path(__file__).resolve().parents[1] / "outputs" / "part2_decision_evidence_extra.json"


def factors(df: pd.DataFrame, line: str, market: str) -> pd.DataFrame:
    f = pd.DataFrame({"pay": cat(df["payment_method"]), "missed": band(df["missed_payments_12m"], [-1, 0, 1, 99], ["0", "1", "2+"]),
                      "channel": cat(df["channel"]), "urban": cat(df["urban_density"]), "region": cat(df["region"]),
                      "new_bus": cat(df["new_business"].astype(int))}, index=df.index)
    f["age"] = cat(df["driver_age_band" if line == "motor" else "policyholder_age_band"])
    if line == "motor":
        f["ncb"] = cat(df["ncb_years"].clip(upper=10))
        f["vvg"], f["cover"], f["use"] = cat(df["vehicle_value_group"]), cat(df["cover_type"]), cat(df["vehicle_use"])
        f["fuel"], f["body"], f["young"] = cat(df["fuel_type"]), cat(df["body_type"]), cat(df["named_young_driver"])
        f["points"] = band(df["licence_points"], [-1, 0, 3, 6, 99])
        f["vage"] = band(df["vehicle_age"], [-1, 1, 3, 6, 10, 15, 999])
    else:
        f["si"] = band(df["sum_insured"], quantile_edges(df["sum_insured"], 10))
        f["ptype"], f["flood"], f["ad"] = cat(df["property_type"]), cat(df["flood_zone"]), cat(df["accidental_damage_cover"])
    if line == "buildings":
        f["constr"], f["soil"], f["occ"] = cat(df["construction_type"]), cat(df["soil_type"]), cat(df["occupancy"])
        f["built"] = band(df["year_built"], [0, 1919, 1944, 1969, 1989, 2009, 3000])
        f["beds"] = cat(df["bedrooms"].clip(upper=5))
    if line == "contents":
        f["tenure"], f["security"] = cat(df["tenure"]), cat(df["security_level"])
        f["hvi"], f["bike"] = cat(df["high_value_items"].clip(upper=4)), cat(df["bicycle_cover"])
    if market == "emerging":
        f["gender"] = cat(df["gender"])
    return f


def poisson_glm(f: pd.DataFrame, y: np.ndarray, e: np.ndarray, base: dict) -> dict:
    """Poisson GLM with log-exposure offset (fitted as the rate y/e with weight e); Wald intervals from the Fisher
    information. Each factor's base level is its most common level, or the level given in `base`."""
    cols, names = [np.ones(len(f))], ["intercept"]
    for c in f.columns:
        lv = f[c].value_counts()
        b = base.get(c, lv.index[0])
        for l in sorted(set(lv.index) - {b}):
            cols.append((f[c].to_numpy() == l).astype(float)); names.append(f"{c}={l}")
    X = sp.csr_matrix(np.column_stack(cols))
    m = PoissonRegressor(alpha=1e-8, fit_intercept=False, max_iter=3000, tol=1e-9).fit(X, y / e, sample_weight=e)
    mu = np.exp(X @ m.coef_) * e
    cov = np.linalg.inv((X.T @ sp.diags(mu) @ X).toarray())
    se = np.sqrt(np.diag(cov))
    out = {}
    for j, n in enumerate(names):
        if n.startswith(("new_bus=", "gender=")):
            out[n] = dict(relativity=float(np.exp(m.coef_[j])), lo=float(np.exp(m.coef_[j] - 1.96 * se[j])), hi=float(np.exp(m.coef_[j] + 1.96 * se[j])))
    return out


def main():
    res = {"controlled": {}, "large_by_peril": {}}
    cl = {"emerging": dr.read_raw("claims", EMG), "developed": dr.read_raw("claims", DEV)}
    for line in dr.LINES:
        res["controlled"][line] = {}
        for market, root, years in (("emerging", EMG, (2023, 2024)), ("developed", DEV, (2021, 2022, 2023, 2024))):
            df, _ = rows(line, root, years)
            f = factors(df, line, market)
            f["yr"] = cat(df["yr"])
            res["controlled"][line][market] = poisson_glm(f, df["y_n"].to_numpy(float), df["e"].to_numpy(float), {"new_bus": "0", "gender": "F"})
            print(line, market, res["controlled"][line][market], flush=True)
        per = {}
        net = {}
        for mk, c in cl.items():
            c = c[(c["line"] == line) & (c["status"] != "repudiated")]
            net[mk] = ((c["incurred_to_date"] - c["recoveries"]).clip(lower=0), c["peril"])
        for p in sorted(set(net["emerging"][1]) | set(net["developed"][1])):
            ne, nd = net["emerging"][0][net["emerging"][1] == p], net["developed"][0][net["developed"][1] == p]
            cap = CAPS[line]
            above_e, above_d = int((ne > cap).sum()), int((nd > cap).sum())
            if above_e + above_d == 0:
                continue
            expected = len(ne) * above_d / max(len(nd), 1)
            prob = float(poisson.cdf(above_e, expected) if above_e < expected else poisson.sf(above_e - 1, expected))
            xs = lambda s: float((s - cap).clip(lower=0).sum() / s.clip(upper=cap).sum()) if len(s) else None
            per[p] = dict(emerging_claims=int(len(ne)), emerging_above=above_e, developed_claims=int(len(nd)), developed_above=above_d,
                          expected_above_at_developed_rate=float(expected), prob_as_far=prob,
                          excess_share_emerging=xs(ne), excess_share_developed=xs(nd),
                          capped_cost_emerging=float(ne.clip(upper=cap).sum()))
        res["large_by_peril"][line] = per
        print(line, per, flush=True)
    write_evidence({"part2_decision_evidence_extra": res}, OUT)


if __name__ == "__main__":
    main()
