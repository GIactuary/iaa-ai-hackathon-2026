"""Tests for src/targets.py: completion arithmetic on a hand-made peril, and row targets."""
import numpy as np
import pandas as pd

from src import targets as tg


def _claims(lags, line="motor", peril="own_damage", year=2021):
    n = len(lags)
    loss = pd.to_datetime([f"{year}-06-15"] * n)
    return pd.DataFrame({"claim_id": [f"C{i}" for i in range(n)], "policy_id": ["P"] * n, "line": line, "peril": peril,
                         "loss_date": loss, "report_date": loss + pd.to_timedelta(lags, unit="D"), "status": "closed",
                         "incurred_to_date": 100.0, "recoveries": 0.0, "paid_to_date": 100.0})


def test_delay_curve_is_empirical_cdf():
    F = tg.delay_curves(_claims([0, 10, 10, 100]), max_age=200)[("motor", "own_damage")]
    assert F[0] == 0.25 and F[9] == 0.25 and F[10] == 0.75 and F[99] == 0.75 and F[100] == 1.0


def test_row_completion_full_for_old_rows_partial_for_recent():
    c = _claims([0, 30, 60, 90])
    curves, weights, shares = tg.delay_curves(c), tg.incidence_weights(c), tg.peril_cost_shares(c)
    rows = pd.DataFrame({"period_start": pd.to_datetime(["2023-01-01", "2024-12-01", "2025-01-01"]),
                         "period_end": pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"])})
    out = tg.row_completion(rows, "motor", curves, weights, shares)
    assert out["completion"].iloc[0] == 1.0
    # December 2024 losses are 0 to 30 days old at valuation: only lag-0 claims (and lag 30 on 1 Dec) reported
    assert 0.25 <= out["completion"].iloc[1] < 0.27
    assert out["completion"].iloc[2] == 1.0


def test_row_targets_excludes_repudiated_and_sums_by_peril():
    rows = pd.DataFrame({"row_id": ["R1", "R2"], "policy_id": ["P1", "P2"],
                         "period_start": pd.to_datetime(["2023-01-01"] * 2), "period_end": pd.to_datetime(["2023-12-31"] * 2)})
    c = pd.DataFrame({"claim_id": ["a", "b", "c"], "policy_id": ["P1", "P1", "P1"], "line": "motor",
                      "peril": ["theft", "theft", "own_damage"], "loss_date": pd.to_datetime(["2023-02-01"] * 3),
                      "status": ["closed", "repudiated", "open"], "incurred_to_date": [500.0, 0.0, 300.0],
                      "recoveries": [100.0, 0.0, 0.0]})
    t = tg.row_targets(rows, c, "motor").set_index("row_id")
    assert t.loc["R1", "cost"] == 700.0 and t.loc["R1", "n_claims"] == 2 and t.loc["R1", "cost_theft"] == 400.0
    assert t.loc["R2", "cost"] == 0.0
