"""The 2025 level (spec sections 4 and 5; P1, P9, P11; C1 to C4).

- `excess_factors`: total cost over capped cost, per peril and per load group (C2).
- `peril_trends`: annual frequency and capped-severity trends per peril, fitted on the experience years and held
  inside the agreed bounds (C3); bodily injury severity is a step from 2023 where 2023 is in the data.
- `indication`: capped loss cost per unit of exposure per peril, each experience year trended to the target year,
  averaged by exposure: the developed, trended indication.
- `off_balance`: the factor that moves a model's predictions on the latest experience rows to the indication,
  and its distance from the model's own projection (the C4 level check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import market
STEP_PERILS = market.STEP_PERILS


def perils_of(df: pd.DataFrame) -> list[str]:
    return sorted(c[6:] for c in df.columns if c.startswith("costc_") and c not in ("costc_large", "costc_other"))


def _ec(df: pd.DataFrame, p: str) -> pd.Series:
    """Completed exposure for a peril; times the row's weather multiplier for that peril where one is given
    (column wm_<peril>), so frequency reads at normal weather (weather-normalised experience, K4)."""
    ec = df["exposure"] * df[f"completion_{p}"]
    return ec * df[f"wm_{p}"] if f"wm_{p}" in df.columns else ec


def excess_factors(df: pd.DataFrame, groups: dict[str, tuple]) -> dict:
    """Per peril and per group: uncapped cost over capped cost on the given rows."""
    per = {p: float(df[f"cost_{p}"].sum() / max(df[f"costc_{p}"].sum(), 1e-9)) for p in perils_of(df)}
    grp = {g: float(df[[f"cost_{p}" for p in ps]].sum().sum() / max(df[[f"costc_{p}" for p in ps]].sum().sum(), 1e-9))
           for g, ps in groups.items()}
    return dict(peril=per, group=grp)


def _loglinear(years, values, weights):
    x = np.asarray(years, dtype=float); yv = np.log(np.asarray(values, dtype=float)); w = np.asarray(weights, dtype=float)
    xm = np.average(x, weights=w)
    slope = np.sum(w * (x - xm) * (yv - np.average(yv, weights=w))) / np.sum(w * (x - xm) ** 2)
    return float(np.exp(slope) - 1)


def peril_trends(df: pd.DataFrame, line: str, lo: float, hi: float) -> pd.DataFrame:
    """Annual frequency and severity trend per peril, bounded to [lo, hi]. Frequency uses completed exposure."""
    rows = []
    years = sorted(df["yr"].unique())
    for p in perils_of(df):
        ec = _ec(df, p).groupby(df["yr"]).sum()
        n = df[f"n_{p}"].groupby(df["yr"]).sum()
        c = df[f"costc_{p}"].groupby(df["yr"]).sum()
        freq, sev = n / ec, c / n.clip(lower=1)
        step_year = STEP_PERILS.get((line, p))
        step = step_year is not None and max(years) >= step_year and sum(y >= step_year for y in years) >= 1
        f_raw = _loglinear(years, freq[years], n[years]) if len(years) > 1 else 0.0
        if step:
            s_raw, sev_level = 0.0, float(c[[y for y in years if y >= step_year]].sum() / n[[y for y in years if y >= step_year]].sum())
        else:
            s_raw, sev_level = (_loglinear(years, sev[years], n[years]) if len(years) > 1 else 0.0), None
        f_sel, s_sel = float(np.clip(f_raw, lo, hi)), float(np.clip(s_raw, lo, hi))
        rows.append(dict(peril=p, freq_trend_fitted=f_raw, freq_trend=f_sel, sev_trend_fitted=s_raw, sev_trend=s_sel,
                         bounded=bool(f_sel != f_raw or s_sel != s_raw), sev_step_level=sev_level,
                         **{f"freq_{y}": float(freq[y]) for y in years}, **{f"sev_{y}": float(sev[y]) for y in years}))
    return pd.DataFrame(rows).set_index("peril")


def indication(df: pd.DataFrame, trends: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Capped loss cost per unit of exposure per peril, trended from each experience year to the target year and
    averaged with exposure weights."""
    years = sorted(df["yr"].unique())
    expo = df["exposure"].groupby(df["yr"]).sum()
    out = {}
    for p, t in trends.iterrows():
        ec = _ec(df, p).groupby(df["yr"]).sum()
        n = df[f"n_{p}"].groupby(df["yr"]).sum()
        c = df[f"costc_{p}"].groupby(df["yr"]).sum()
        lc = []
        for y in years:
            k = target_year - y
            freq = n[y] / ec[y] * (1 + t.freq_trend) ** k
            sev = t.sev_step_level if pd.notna(t.sev_step_level) else c[y] / max(n[y], 1) * (1 + t.sev_trend) ** k
            lc.append(freq * sev)
        out[p] = dict(indicated_capped_lc=float(np.average(lc, weights=expo[years])), **{f"lc_{y}": float(v) for y, v in zip(years, lc)})
    return pd.DataFrame(out).T


def group_trend(trends: pd.DataFrame, peril_lc: pd.Series, perils) -> float:
    """Annual loss-cost trend of a group of perils, weighted by each peril's indicated capped loss cost."""
    w = peril_lc[list(perils)]
    r = [(1 + trends.loc[p, "freq_trend"]) * (1 + trends.loc[p, "sev_trend"]) - 1 for p in perils]
    return float(np.average(r, weights=w))


def off_balance(model_rate_last_year, exposure_last_year, indicated_lc: float, trend: float, years_ahead: int) -> dict:
    """Factor moving the model's latest-year rows to the indication, and the C4 check against its own projection."""
    model_lc = float(np.sum(np.asarray(model_rate_last_year) * np.asarray(exposure_last_year)) / np.sum(exposure_last_year))
    factor = indicated_lc / model_lc
    projection = (1 + trend) ** years_ahead
    return dict(factor=factor, model_projection=projection, gap=factor / projection - 1, model_lc_last_year=model_lc,
                indicated_lc=indicated_lc)
