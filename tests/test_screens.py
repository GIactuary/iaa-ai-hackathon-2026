"""Tests for src/screens.py on small synthetic frames."""
import numpy as np
import pandas as pd

from src import screens


def test_claim_screen_flags_only_upper_tail_by_peril():
    rng = np.random.default_rng(1)
    base = np.exp(rng.normal(7, 1, 500))
    costs = np.concatenate([base, [np.exp(7 + 6)]])
    c = pd.DataFrame({"claim_id": [f"C{i}" for i in range(len(costs))], "line": "motor", "peril": "own_damage",
                      "net_cost": costs, "repudiated": False})
    other = pd.DataFrame({"claim_id": ["X1", "X2"], "line": "motor", "peril": "theft",
                          "net_cost": [1000.0, 0.0], "repudiated": [False, True]})
    out = screens.claim_screen(pd.concat([c, other], ignore_index=True), z=4)
    assert out["claim_id"].tolist() == ["C500"]
    assert out["z"].iloc[0] > 4


def test_size_referrals_within_segment_and_unseen_segment():
    train = pd.DataFrame({"tenure": ["owner", "owner", "renter"], "bedrooms": [2, 2, 2], "sum_insured": [100.0, 80.0, 50.0]})
    test = pd.DataFrame({"row_id": ["a", "b", "c", "d"], "tenure": ["owner", "owner", "renter", "renter"],
                         "bedrooms": [2, 2, 2, 5], "sum_insured": [124.0, 126.0, 70.0, 10.0]})
    out = screens.size_referrals(train, test, "sum_insured", ("tenure", "bedrooms"), 1.25)
    assert out["row_id"].tolist() == ["b", "c", "d"]
    assert out.set_index("row_id").loc["d", "reason"] == "segment unseen in training"


def test_size_referrals_whole_book():
    train = pd.DataFrame({"engine_size_cc": [1000.0, 3000.0]})
    test = pd.DataFrame({"row_id": ["a", "b"], "engine_size_cc": [3700.0, 3800.0]})
    out = screens.size_referrals(train, test, "engine_size_cc", (), 1.25)
    assert out["row_id"].tolist() == ["b"]


def test_unseen_levels_and_impossible_motor():
    assert screens.unseen_levels(pd.DataFrame({"x": ["a", "b"]}), pd.DataFrame({"x": ["a", "c", None]}), ["x"]) == {"x": ["c"]}
    df = pd.DataFrame({"ncb_years": [3, 1], "years_licensed": [2, 5], "driver_age": [30, 21]})
    f = screens.impossible_motor(df)
    assert f["flag_ncb_above_licence"].tolist() == [True, False]
    assert f["flag_licence_before_16"].tolist() == [False, False]
    f2 = screens.impossible_motor(pd.DataFrame({"ncb_years": [0], "years_licensed": [10], "driver_age": [24]}))
    assert bool(f2["flag_licence_before_16"].iloc[0])
