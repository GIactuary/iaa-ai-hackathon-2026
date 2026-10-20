"""Model-ready rows for each line, built once and cached in data/work/lines/ (regenerable, kept out of git).

Each training row carries: the repaired and recovered policy fields (data_rules R1 to R9), address and customer
fields, the hold-out flag, reporting completion (cost-weighted, count-weighted and per peril; P10), and targets by
peril, uncapped and capped at the line's agreed cap (C1). Test rows carry the same inputs with completion 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import data_rules as dr, targets as tg, validation as v
from src import market
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
CACHE = market.WORK / "lines"
LOCK = market.WORK / "holdout_lock.json"
CUSTOMER_COLS = ["customer_id", "credit_band", "licence_points", "prior_claims_5y", "date_of_birth", "household_size",
                 "years_at_address", "customer_since_year", "occupation", "marital_status"]
LARGE_LOSS_PERILS = market.LARGE_LOSS_PERILS


def _cap(th, line):
    return th[f"cap_{line}"]


def build_all(rebuild: bool = False) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return {line: (train, test)}; reads the cache unless `rebuild` or the cache is missing."""
    paths = {l: (CACHE / f"{l}_train.parquet", CACHE / f"{l}_test.parquet") for l in dr.LINES}
    if not rebuild and all(a.exists() and b.exists() for a, b in paths.values()):
        return {l: (pd.read_parquet(a), pd.read_parquet(b)) for l, (a, b) in paths.items()}
    th = load_thresholds()
    claims = dr.read_raw("claims")
    curves, weights = tg.delay_curves(claims), tg.incidence_weights(claims)
    cshares, nshares = tg.peril_cost_shares(claims), tg.peril_count_shares(claims)
    ad, cu = dr.read_raw("addresses"), dr.read_raw("customers")
    out, custs = {}, []
    for line in dr.LINES:
        tr, te = dr.read_raw(f"policies_{line}"), dr.read_raw(f"test_{line}")
        cut = dr.sum_insured_cutoff(tr["sum_insured"], th["si_unit_multiple"]) if line != "motor" else None
        tr, te = dr.apply_policy_rules(tr, line, cut), dr.apply_policy_rules(te, line, cut)
        tr, te = dr.recover_from_history(tr, te, line)
        frames = []
        for df in (tr, te):
            df = df.merge(ad, on="address_id", how="left").merge(cu[CUSTOMER_COLS], on="customer_id", how="left")
            comp = tg.row_completion(df, line, curves, weights, cshares)
            df = df.join(comp)
            df["completion_count"] = tg.row_completion(df, line, curves, weights, nshares)["completion"]
            df["yr"] = df["period_start"].dt.year
            frames.append(df)
        tr, te = frames
        tr = tr.merge(tg.row_targets(tr, claims, line, cap=_cap(th, line)), on="row_id", how="left")
        tr["customer_id"] = tr["customer_id"].astype("string")
        ll = list(LARGE_LOSS_PERILS[line])
        perils = sorted(c[5:] for c in tr.columns if c.startswith("cost_"))
        tr["costc_large"] = tr[[f"costc_{p}" for p in ll]].sum(axis=1)
        tr["costc_other"] = tr[[f"costc_{p}" for p in perils if p not in ll]].sum(axis=1)
        tr["cost_large"] = tr[[f"cost_{p}" for p in ll]].sum(axis=1)
        tr["cost_other"] = tr[[f"cost_{p}" for p in perils if p not in ll]].sum(axis=1)
        out[line] = (tr, te)
        custs.append(tr["customer_id"])
    cust = np.array(sorted(pd.concat(custs).unique()))
    keep = pd.Series(v.fitting_mask(LOCK, cust), index=cust)
    CACHE.mkdir(parents=True, exist_ok=True)
    for line, (tr, te) in out.items():
        tr["holdout"] = ~tr["customer_id"].map(keep).to_numpy()
        tr["completion"] = line_completion_by_year(tr)
        tr.to_parquet(paths[line][0]); te.to_parquet(paths[line][1])
    return out


def line_completion_by_year(tr: pd.DataFrame) -> pd.Series:
    """Line-level completion weighted by each year's own peril mix (P10 fix, exchange D16).

    Weighting every year by the 2021-2022 peril mix overstated 2024 motor by about 5%: bodily injury, the slowest
    peril, fell from 42% to 33% of cost after its 2023 severity step. Each year's weights are its perils' shares of
    completion-adjusted capped cost, estimated on non-hold-out rows only.
    """
    perils = sorted(c[6:] for c in tr.columns if c.startswith("costc_") and c not in ("costc_large", "costc_other"))
    fit = ~tr["holdout"]
    comp = pd.Series(0.0, index=tr.index)
    for y in sorted(tr["yr"].unique()):
        m = (tr["yr"] == y)
        mf = m & fit
        ult = {p: tr.loc[mf, f"costc_{p}"].sum() / (tr.loc[mf, "exposure"] * tr.loc[mf, f"completion_{p}"]).sum() for p in perils}
        tot = sum(ult.values())
        comp[m] = sum(ult[p] / tot * tr.loc[m, f"completion_{p}"] for p in perils)
    return comp
