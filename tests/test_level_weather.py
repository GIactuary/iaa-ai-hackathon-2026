"""Tests for src/level.py and src/weather.py on synthetic data with known answers."""
import numpy as np
import pandas as pd
import pytest

from src import level as lv, weather as wx


def _rows(freq_growth=0.10, sev_growth=0.05, years=(2021, 2022, 2023, 2024), n_rows=4000, seed=0, peril="storm"):
    rng = np.random.default_rng(seed)
    parts = []
    for i, y in enumerate(years):
        lam = 0.2 * (1 + freq_growth) ** i
        n = rng.poisson(lam, n_rows).astype(float)
        sev = 1000.0 * (1 + sev_growth) ** i
        parts.append(pd.DataFrame({"yr": y, "exposure": 1.0, f"completion_{peril}": 1.0, f"n_{peril}": n,
                                   f"costc_{peril}": n * sev, f"cost_{peril}": n * sev * 1.1}))
    return pd.concat(parts, ignore_index=True)


def test_trend_recovers_growth_and_bounds():
    df = _rows()
    t = lv.peril_trends(df, "buildings", -0.05, 0.12)
    assert t.loc["storm", "freq_trend"] == pytest.approx(0.10, abs=0.02)
    assert t.loc["storm", "sev_trend"] == pytest.approx(0.05, abs=1e-9)
    t2 = lv.peril_trends(_rows(freq_growth=0.30), "buildings", -0.05, 0.12)
    assert t2.loc["storm", "freq_trend"] == 0.12 and bool(t2.loc["storm", "bounded"])


def test_step_peril_uses_level_from_step_year():
    df = _rows(peril="bodily_injury", sev_growth=0.0)
    df.loc[df.yr >= 2023, "costc_bodily_injury"] *= 0.8
    t = lv.peril_trends(df, "motor", -0.05, 0.12)
    assert t.loc["bodily_injury", "sev_trend"] == 0.0
    assert t.loc["bodily_injury", "sev_step_level"] == pytest.approx(800.0)


def test_indication_flat_data_returns_the_loss_cost():
    df = _rows(freq_growth=0.0, sev_growth=0.0)
    t = lv.peril_trends(df, "buildings", -0.05, 0.12)
    t[["freq_trend", "sev_trend"]] = 0.0                      # the arithmetic, without sampling noise in the trend
    ind = lv.indication(df, t, 2025)
    assert ind.loc["storm", "indicated_capped_lc"] == pytest.approx(df.costc_storm.sum() / df.exposure.sum(), rel=1e-9)


def test_excess_factors_and_off_balance():
    df = _rows()
    ef = lv.excess_factors(df, {"large": ("storm",)})
    assert ef["peril"]["storm"] == pytest.approx(1.1) and ef["group"]["large"] == pytest.approx(1.1)
    ob = lv.off_balance([100.0, 100.0], [1.0, 1.0], 110.0, 0.05, 1)
    assert ob["factor"] == pytest.approx(1.1) and ob["gap"] == pytest.approx(1.1 / 1.05 - 1)


def test_normals_use_only_given_years_and_deviation():
    w = pd.DataFrame({"address_id": ["A"] * 3, "year": [2021, 2022, 2023], **{v: [1.0, 3.0, 10.0] for v in wx.WEATHER_VARS}})
    n = wx.normals(w, [2021, 2022])
    assert n.loc["A", "norm_wind_exposure_index"] == 2.0
    rows = pd.DataFrame({"address_id": ["A"], "yr": [2023]})
    a = wx.attach(rows, w, n)
    assert a["dev_wind_exposure_index"].iloc[0] == 8.0


def test_elasticity_recovers_known_effect_and_year_factor():
    rng = np.random.default_rng(1)
    dev = rng.normal(0, 1, 200000)
    n = rng.poisson(0.05 * np.exp(0.25 * dev))
    rows = pd.DataFrame({"exposure": 1.0, "completion_storm": 1.0, "n_storm": n, "dev_wind_exposure_index": dev,
                         "completion_subsidence": 1.0, "n_subsidence": rng.poisson(0.01, 200000), "dev_dry_summer_index": rng.normal(0, 1, 200000),
                         "completion_escape_of_water": 1.0, "n_escape_of_water": rng.poisson(0.02, 200000), "dev_freezing_days": rng.normal(0, 1, 200000)})
    b = wx.elasticities(rows, "buildings")
    assert b["storm"][1] == pytest.approx(0.25, abs=0.03)
    f = wx.year_factor(pd.DataFrame({"dev_wind_exposure_index": [1.0], "dev_dry_summer_index": [0.0], "dev_freezing_days": [0.0]}),
                       "buildings", {"storm": ("wind_exposure_index", 0.25)}, {"storm": 0.2})
    assert f[0] == pytest.approx(1 + 0.2 * (np.exp(0.25) - 1))


def test_elasticity_on_small_scale_deviation_is_not_stuck_at_zero():
    rng = np.random.default_rng(2)
    dev = rng.normal(0, 0.13, 300000)
    n = rng.poisson(0.01 * np.exp(1.5 * dev))
    rows = pd.DataFrame({"exposure": 1.0, "completion_subsidence": 1.0, "n_subsidence": n, "dev_dry_summer_index": dev,
                         "completion_storm": 1.0, "n_storm": rng.poisson(0.05, 300000), "dev_wind_exposure_index": rng.normal(0, 1, 300000),
                         "completion_escape_of_water": 1.0, "n_escape_of_water": rng.poisson(0.02, 300000), "dev_freezing_days": rng.normal(0, 6, 300000)})
    b = wx.elasticities(rows, "buildings")
    assert b["subsidence"][1] == pytest.approx(1.5, abs=0.3)


def test_leave_own_year_out_normals():
    w = pd.DataFrame({"address_id": ["A"] * 3, "year": [2021, 2022, 2023], **{v: [1.0, 3.0, 10.0] for v in wx.WEATHER_VARS}})
    rows = pd.DataFrame({"address_id": ["A", "A", "A"], "yr": [2021, 2022, 2023]})
    out = wx.normals_leave_own_year_out(rows, w, [2021, 2022], (rows["yr"] == 2023).to_numpy())
    assert out["norm_wind_exposure_index"].tolist() == [3.0, 1.0, 2.0]
    assert out["dev_wind_exposure_index"].tolist() == [-2.0, 2.0, 8.0]


def test_weather_multiplier_normalises_frequency():
    df = _rows(freq_growth=0.0, sev_growth=0.0)
    df["wm_storm"] = np.where(df.yr == 2024, 1.5, 1.0)          # 2024 expected 50% more storm from its weather
    df.loc[df.yr == 2024, "n_storm"] *= 1.5
    df.loc[df.yr == 2024, "costc_storm"] *= 1.5
    t = lv.peril_trends(df, "buildings", -0.05, 0.12)
    assert abs(t.loc["storm", "freq_2024"] - t.loc["storm", "freq_2021"]) < 0.02
