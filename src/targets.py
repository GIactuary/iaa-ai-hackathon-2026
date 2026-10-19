"""Row targets and reporting completion (spec section 4; register P1, P10).

- `row_targets`: each row's net cost and claim count, in total and by peril (repudiated claims excluded).
- `delay_curves`, `incidence_weights`, `peril_cost_shares`: estimated from the claims file.
- `row_completion`: for each row and peril, the expected share of its claims reported by the valuation date, from the
  row's own dates, the peril's reporting-delay curve and the peril's seasonal pattern; plus a line-level figure that
  weights perils by their share of mature cost (late claims are not larger here, so counts stand in for amounts).

The delay curve of each peril is the empirical distribution of report lags of accident year 2021 losses, which are
observed for three to four years. It was backtested at a pseudo-valuation of 31 Dec 2023: 13 of 15 perils within
1.5% (docs/research/2026-10-18_round2_research.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_rules as dr

VALUATION = pd.Timestamp("2024-12-31")
CURVE_YEAR = 2021
SHARE_YEARS = (2021, 2022)
GRID_START = pd.Timestamp("2021-01-01")


def _non_repudiated(claims: pd.DataFrame) -> pd.DataFrame:
    c = claims[claims["status"] != "repudiated"].copy()
    c["lag"] = (c["report_date"] - c["loss_date"]).dt.days
    return c


def delay_curves(claims: pd.DataFrame, max_age: int = 2000) -> dict[tuple[str, str], np.ndarray]:
    """F[age] = share of the peril's accident-year-2021 claims reported within `age` days of loss."""
    c = _non_repudiated(claims)
    c = c[c["loss_date"].dt.year == CURVE_YEAR]
    ages = np.arange(max_age + 1)
    out = {}
    for key, g in c.groupby(["line", "peril"]):
        lags = np.sort(g["lag"].to_numpy())
        out[key] = np.searchsorted(lags, ages, side="right") / len(lags)
    return out


def incidence_weights(claims: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Relative daily claim incidence by calendar month (shares from 2021-2022 losses, spread over the month's days)."""
    c = _non_repudiated(claims)
    c = c[c["loss_date"].dt.year.isin(SHARE_YEARS)]
    out = {}
    for key, g in c.groupby(["line", "peril"]):
        share = g["loss_date"].dt.month.value_counts(normalize=True).reindex(range(1, 13), fill_value=0.0)
        days_in_month = pd.Series([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], index=range(1, 13))
        out[key] = (share / days_in_month).to_numpy()
    return out


def peril_cost_shares(claims: pd.DataFrame) -> dict[str, pd.Series]:
    """Each peril's share of the line's net cost in 2021-2022 (mature years)."""
    c = _non_repudiated(claims)
    c = c[c["loss_date"].dt.year.isin(SHARE_YEARS)]
    net = c["incurred_to_date"] - c["recoveries"]
    s = net.groupby([c["line"], c["peril"]]).sum()
    return {line: s[line] / s[line].sum() for line in s.index.get_level_values(0).unique()}


def row_completion(rows: pd.DataFrame, line: str, curves: dict, weights: dict, shares: dict,
                   valuation: pd.Timestamp = VALUATION) -> pd.DataFrame:
    """Expected reported share by peril for each row, and the cost-weighted line figure in column `completion`.

    Rows that end after the valuation date (the 2025 test rows) get completion 1: they are priced at ultimate.
    """
    days = pd.date_range(GRID_START, valuation)
    age = (valuation - days).days.to_numpy()
    month_idx = days.month.to_numpy() - 1
    start = ((rows["period_start"] - GRID_START).dt.days).to_numpy()
    end = ((rows["period_end"] - GRID_START).dt.days).to_numpy()
    after = rows["period_start"] > valuation
    if (end[~after] >= len(days)).any():
        raise ValueError("a training row ends after the valuation date")
    s_idx = np.clip(start, 0, len(days) - 1)
    e_idx = np.clip(end, 0, len(days) - 1)
    out = pd.DataFrame(index=rows.index)
    total = np.zeros(len(rows))
    for peril, share in shares[line].items():
        F = curves[(line, peril)]
        w = weights[(line, peril)][month_idx]
        g = w * F[np.minimum(age, len(F) - 1)]
        cg = np.concatenate([[0.0], np.cumsum(g)])
        cw = np.concatenate([[0.0], np.cumsum(w)])
        wsum = cw[e_idx + 1] - cw[s_idx]
        # A row whose months carry no historical incidence falls back to equal weight on each day
        cu = np.concatenate([[0.0], np.cumsum(F[np.minimum(age, len(F) - 1)])])
        uniform = (cu[e_idx + 1] - cu[s_idx]) / (e_idx - s_idx + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            comp = np.where(wsum > 0, (cg[e_idx + 1] - cg[s_idx]) / np.where(wsum > 0, wsum, 1.0), uniform)
        comp = np.where(after, 1.0, comp)
        out[f"completion_{peril}"] = comp
        total += share * comp
    out["completion"] = total
    return out


def row_targets(rows: pd.DataFrame, claims: pd.DataFrame, line: str) -> pd.DataFrame:
    """Net cost and claim count per row, in total and by peril; zero for rows without a claim."""
    m = dr.match_claims_to_rows(rows, claims, line)
    m = m[~m["repudiated"]]
    cost = m.pivot_table(index="row_id", columns="peril", values="net_cost", aggfunc="sum", fill_value=0.0)
    cnt = m.pivot_table(index="row_id", columns="peril", values="claim_id", aggfunc="count", fill_value=0)
    out = pd.DataFrame(index=rows["row_id"])
    out = out.join(cost.add_prefix("cost_")).join(cnt.add_prefix("n_")).fillna(0.0)
    out["cost"] = out.filter(regex="^cost_").sum(axis=1)
    out["n_claims"] = out.filter(regex="^n_").sum(axis=1)
    return out.reset_index()
