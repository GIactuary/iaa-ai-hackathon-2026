"""Tests for src/drift.py (Part 2 kit): each check finds what was planted and stays quiet when nothing changed."""
import numpy as np
import pandas as pd

from src import drift


def frame(n=5000, seed=0, si_scale=1.0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    return pd.DataFrame({"row_id": [f"R{i:07d}" for i in range(n)], "customer_id": [f"{i:08d}" for i in range(n)],
                         "period_start": start, "period_end": start + pd.Timedelta(days=364), "exposure": 1.0,
                         "sum_insured": rng.lognormal(12, 0.5, n) * si_scale, "region": rng.choice(["R01", "R02", "R03"], n)})


def test_identical_frames_show_no_change():
    a = frame()
    s = drift.column_shifts(a, a.copy())
    assert s["sum_insured"]["psi"] < 1e-9 and s["region"]["psi"] < 1e-9
    assert drift.schema_diff(a, a) == dict(added=[], removed=[], retyped={})


def test_planted_changes_are_found():
    a, b = frame(seed=1), frame(seed=2, si_scale=1.5)
    b.loc[:99, "exposure"] = 365.0                         # entered in days
    b.loc[100:149, "sum_insured"] *= 1000                  # unit error
    b.loc[150:199, "customer_id"] = b.loc[150:199, "customer_id"].str.lstrip("0")
    b.loc[200:399, "region"] = "R99"
    b = b.drop(columns=["period_end"]).assign(period_end=b["period_end"], new_col=1)
    s = drift.column_shifts(a, b)
    assert s["sum_insured"]["psi"] > 0.25
    assert "R99" in s["region"]["new_levels"]
    d = drift.detectors(b, a)
    assert d["exposure_equals_day_count"] == 100 and d["sum_insured_above_10x_ref_p99"] >= 50
    assert len(d["customer_id_lengths"]) > 1
    assert drift.schema_diff(a, b)["added"] == ["new_col"]


def test_psi_reference_values():
    assert drift.psi([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert abs(drift.psi([0.5, 0.5], [0.7, 0.3]) - ((0.2) * np.log(0.7 / 0.5) + (-0.2) * np.log(0.3 / 0.5))) < 1e-12
