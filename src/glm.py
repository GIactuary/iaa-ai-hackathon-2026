"""The GLM family: a Tweedie GLM on one-hot rating factors, with exposure x completion as an exact offset.

Exposure enters as the metric uses it. Tweedie deviance is homogeneous of degree 2 - p, so fitting the rate
y / (e c) with weight (e c)^(2 - p) gives the same fit as row cost with a log(e c) offset (validation design;
research note section 6).

`features` builds the main-effect factors of the plain baseline. Optional groups add the address climate normals
(J-weather). Band edges are taken from the fitting rows only and passed on, so scored rows are banded the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import TweedieRegressor

WEATHER_VARS = ("wind_exposure_index", "dry_summer_index", "freezing_days", "hail_days", "rainfall_mm",
                "wildfire_index", "mean_temperature_c")


def band(x, edges, labels=None):
    return pd.cut(x, edges, labels=labels, include_lowest=True).astype("string").fillna("blank")


def cat(x):
    return x.astype("string").fillna("blank")


def quantile_edges(x: pd.Series, q: int) -> np.ndarray:
    e = np.unique(x.dropna().quantile(np.linspace(0, 1, q + 1)).to_numpy())
    e[0], e[-1] = -np.inf, np.inf
    return e


def fit_edges(df: pd.DataFrame, line: str, normals: bool = False) -> dict:
    """Band edges learned from the fitting rows."""
    edges = {}
    if line != "motor":
        edges["sum_insured"] = quantile_edges(df["sum_insured"], 10)
    if normals:
        for wv in WEATHER_VARS:
            edges[f"norm_{wv}"] = quantile_edges(df[f"norm_{wv}"], 5)
    return edges


def features(df: pd.DataFrame, line: str, edges: dict, normals: bool = False) -> pd.DataFrame:
    age = df["driver_age"] if line == "motor" else df["policyholder_age"]
    f = pd.DataFrame({
        "region": cat(df["region"]), "urban": cat(df["urban_density"]), "pay": cat(df["payment_method"]),
        "missed": band(df["missed_payments_12m"], [-1, 0, 1, 99], ["0", "1", "2+"]), "credit": cat(df["credit_band"]),
        "channel": cat(df["channel"]), "new_bus": cat(df["new_business"]),
        "age": band(age, [0, 20, 24, 29, 39, 49, 59, 69, 79, 200]),
        "yr": cat(df["yr"]),
    }, index=df.index)
    if line == "motor":
        f["ncb"] = cat(df["ncb_years"].clip(upper=10))
        f["vvg"] = cat(df["vehicle_value_group"])
        f["cover"], f["use"], f["fuel"], f["body"] = cat(df["cover_type"]), cat(df["vehicle_use"]), cat(df["fuel_type"]), cat(df["body_type"])
        f["mileage"] = band(df["annual_mileage"], [-1, 0, 5000, 7500, 10000, 15000, 20000, 1e7])
        f["points"] = band(df["licence_points"], [-1, 0, 3, 6, 99])
        f["young"], f["telem"] = cat(df["named_young_driver"]), cat(df["telematics_opt_in"])
    else:
        f["si"] = band(df["sum_insured"], edges["sum_insured"])
        f["ptype"], f["flood"] = cat(df["property_type"]), cat(df["flood_zone"])
        f["ad_cover"] = cat(df["accidental_damage_cover"])
    if line == "buildings":
        f["constr"], f["soil"], f["occ"] = cat(df["construction_type"]), cat(df["soil_type"]), cat(df["occupancy"])
        f["built"] = band(df["year_built"], [0, 1919, 1944, 1969, 1989, 2009, 3000])
        f["beds"] = cat(df["bedrooms"].clip(upper=5))
    if line == "contents":
        f["tenure"], f["security"] = cat(df["tenure"]), cat(df["security_level"])
        f["hvi"] = cat(df["high_value_items"].clip(upper=4))
        f["bike"] = cat(df["bicycle_cover"])
    if normals:
        for wv in WEATHER_VARS:
            f[f"n_{wv}"] = band(df[f"norm_{wv}"], edges[f"norm_{wv}"])
    return f


class TweedieGLM:
    """Tweedie GLM, log link, on one-hot main effects; the first (most common) level of each factor is the base."""

    def __init__(self, power: float = 1.5, alpha: float = 1e-5):
        self.power, self.alpha = power, alpha

    def _x(self, f):
        rows, cols, j = [], [], 0
        for c in self.columns_:
            vals = f[c].to_numpy()
            for lvl in self.levels_[c][1:]:
                r = np.flatnonzero(vals == lvl)
                rows.append(r); cols.append(np.full(len(r), j)); j += 1
        r, k = np.concatenate(rows), np.concatenate(cols)
        return sp.csr_matrix((np.ones(len(r)), (r, k)), shape=(len(f), j))

    def fit(self, f: pd.DataFrame, y, ec):
        self.columns_ = list(f.columns)
        self.levels_ = {}
        for c in self.columns_:
            vc = f[c].value_counts()
            self.levels_[c] = [vc.index[0]] + sorted(set(vc.index) - {vc.index[0]})
        y, ec = np.asarray(y, dtype=float), np.asarray(ec, dtype=float)
        self.model_ = TweedieRegressor(power=self.power, link="log", alpha=self.alpha, solver="lbfgs", max_iter=3000, tol=1e-7)
        self.model_.fit(self._x(f), y / ec, sample_weight=np.power(ec, 2 - self.power))
        return self

    def rate(self, f: pd.DataFrame) -> np.ndarray:
        """Predicted cost per unit of exposure; unseen levels fall back to the base level."""
        f = f[self.columns_].copy()
        for c in self.columns_:
            f[c] = f[c].where(f[c].isin(self.levels_[c]), self.levels_[c][0])
        return self.model_.predict(self._x(f))

    def relativities(self) -> pd.DataFrame:
        coef, out, j = self.model_.coef_, [], 0
        for c in self.columns_:
            out.append((c, self.levels_[c][0], 1.0))
            for lvl in self.levels_[c][1:]:
                out.append((c, lvl, float(np.exp(coef[j]))))
                j += 1
        return pd.DataFrame(out, columns=["factor", "level", "relativity"])
