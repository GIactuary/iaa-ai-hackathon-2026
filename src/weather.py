"""Weather (J-weather (c), C6): address climate normals for the rating plan, and a separate, disclosed
adjustment for the year's weather.

- `normals`: each address's average over the fitting years only, so a scored year never leaks into its own normal.
- `attach`: joins normals and the row-year deviation from them to policy rows.
- `elasticities`: per peril, the Poisson log-rate response of claim frequency to its weather measure's deviation
  (the same address in different years), fitted on the fitting rows with completed exposure.
- `year_factor`: the row's multiplier for the scored year: 1 + sum over the adjusted perils of the peril's share
  of the line's cost times (exp(b x deviation) - 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from src.glm import WEATHER_VARS

ADJUSTED = {"buildings": {"storm": "wind_exposure_index", "subsidence": "dry_summer_index", "escape_of_water": "freezing_days"},
            "contents": {"escape_of_water": "freezing_days"},
            "motor": {"glass_hail": "hail_days"}}


def normals(weather: pd.DataFrame, years) -> pd.DataFrame:
    w = weather[weather["year"].isin(list(years))]
    return w.groupby("address_id")[list(WEATHER_VARS)].mean().add_prefix("norm_")


def attach(rows: pd.DataFrame, weather: pd.DataFrame, norms: pd.DataFrame) -> pd.DataFrame:
    """Add normals and the deviation of the row's own calendar year from them."""
    out = rows.drop(columns=[c for c in rows.columns if c.startswith(("norm_", "dev_"))], errors="ignore").join(norms, on="address_id")
    yw = weather.rename(columns={"year": "yr"})[["address_id", "yr", *WEATHER_VARS]]
    idx = out.index
    out = out.merge(yw, on=["address_id", "yr"], how="left", suffixes=("", "_w"))
    out.index = idx
    for wv in WEATHER_VARS:
        out[f"dev_{wv}"] = out[wv] - out[f"norm_{wv}"]
        out = out.drop(columns=wv)
    return out


def elasticities(rows: pd.DataFrame, line: str) -> dict:
    """{peril: (variable, b)}: change in log frequency per unit of deviation."""
    out = {}
    for peril, wv in ADJUSTED[line].items():
        d = rows[["exposure", f"completion_{peril}", f"n_{peril}", f"dev_{wv}"]].dropna()
        ec = (d["exposure"] * d[f"completion_{peril}"]).to_numpy()
        x = d[[f"dev_{wv}"]].to_numpy()
        sd = float(x.std()) or 1.0          # standardise: deviations on a small scale (dry summer, sd 0.13) stalled the solver at 0
        m = PoissonRegressor(alpha=0.0, max_iter=1000, tol=1e-8).fit(x / sd, d[f"n_{peril}"].to_numpy() / ec, sample_weight=ec)
        out[peril] = (wv, float(m.coef_[0] / sd))
    return out


def normals_leave_own_year_out(rows: pd.DataFrame, weather: pd.DataFrame, fit_years, score_mask) -> pd.DataFrame:
    """Normals that never contain the row's own weather: a fitting row gets the average of the other fitting years,
    a scored row the average of all fitting years. Returns `rows` with norm_ and dev_ columns."""
    out = []
    fy = list(fit_years)
    for y in fy:
        r = rows[(rows["yr"] == y) & ~score_mask]
        others = [o for o in fy if o != y] or fy
        out.append(attach(r, weather, normals(weather, others)))
    out.append(attach(rows[score_mask], weather, normals(weather, fy)))
    rest = rows[~score_mask & ~rows["yr"].isin(fy)]
    if len(rest):
        out.append(attach(rest, weather, normals(weather, fy)))
    return pd.concat(out).loc[rows.index]


def year_factor(rows: pd.DataFrame, line: str, b: dict, shares: dict) -> np.ndarray:
    """Multiplier per row for the year's weather; deviations that are missing contribute nothing."""
    f = np.ones(len(rows))
    for peril, (wv, coef) in b.items():
        dev = rows[f"dev_{wv}"].fillna(0.0).to_numpy()
        f += shares.get(peril, 0.0) * (np.exp(coef * dev) - 1)
    return f
