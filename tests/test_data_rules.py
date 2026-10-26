"""Tests for src/data_rules.py: synthetic cases for each rule, plus checks on the real files when they are present."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data_rules as dr

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HAVE_DATA = (RAW / "policies_motor.parquet").exists()


def _rows(starts, ends, exposure, **extra):
    df = pd.DataFrame({"row_id": [f"R{i}" for i in range(len(starts))], "period_start": pd.to_datetime(starts),
                       "period_end": pd.to_datetime(ends), "exposure": exposure})
    for k, v in extra.items():
        df[k] = v
    return df


def test_exposure_three_bases():
    df = _rows(["2023-01-01", "2023-01-01", "2023-01-01", "2023-07-01", "2023-01-01"],
               ["2023-12-31", "2023-12-31", "2023-12-31", "2023-12-31", "2023-01-01"],
               [1.0, 365.0, 1.0139, 0.5111, 1.0])
    out = dr.repair_exposure(df)
    assert out["dq_exposure_days"].tolist() == [False, True, False, False, True]
    assert out["dq_exposure_360"].tolist() == [False, False, True, True, False]
    np.testing.assert_allclose(out["exposure"], [1.0, 1.0, 1.0, 184 / 365, 1 / 365])


def test_exposure_unexplained_raises():
    df = _rows(["2023-01-01"], ["2023-12-31"], [0.7])
    with pytest.raises(ValueError, match="no exposure basis"):
        dr.repair_exposure(df)


def test_sum_insured_divides_only_above_cutoff():
    train = pd.Series(np.arange(1, 101) * 1000.0)
    cutoff = dr.sum_insured_cutoff(train, 10)
    df = pd.DataFrame({"sum_insured": [50_000.0, 200_000_000.0, cutoff]})
    out = dr.repair_sum_insured(df, cutoff)
    assert out["dq_sum_insured_x1000"].tolist() == [False, True, False]
    assert out["sum_insured"].tolist() == [50_000.0, 200_000.0, cutoff]


def test_construction_codes_map_to_documented_levels():
    df = pd.DataFrame({"construction_type": ["Brick", "BRICK", "stone", "T", "w", "NON_STANDARD", None]})
    out = dr.normalise_construction(df)
    assert out["construction_type"].tolist()[:6] == ["brick", "brick", "stone", "timber", "timber", "non_standard"]
    assert pd.isna(out["construction_type"].iloc[6])
    assert out["dq_construction_timber_code"].tolist() == [False, False, False, True, True, False, False]


def test_construction_unknown_code_raises():
    with pytest.raises(ValueError, match="not in the map"):
        dr.normalise_construction(pd.DataFrame({"construction_type": ["glass"]}))


def test_customer_id_padding():
    df = pd.DataFrame({"customer_id": pd.array(["00640486", "640486", "5180324"], dtype="string")})
    out = dr.pad_customer_id(df)
    assert out["customer_id"].tolist() == ["00640486", "00640486", "05180324"]
    assert out["dq_customer_id_padded"].tolist() == [False, True, True]


def test_mileage_zero_is_kept_as_a_value():
    df = _rows(["2023-01-01"], ["2023-12-31"], [1.0], customer_id=pd.array(["00000001"], dtype="string"),
               annual_mileage=[0.0])
    out = dr.apply_policy_rules(df, "motor")
    assert out["annual_mileage"].tolist() == [0.0]


def _motor(rows):
    cols = ["row_id", "policy_id", "make_model", "period_start", "vehicle_age", "engine_size_cc",
            "overnight_parking", "annual_mileage"]
    df = pd.DataFrame(rows, columns=cols)
    df["period_start"] = pd.to_datetime(df["period_start"])
    return df


def test_recovery_uses_same_vehicle_and_respects_time():
    train = _motor([
        ["A1", "P1", "MM1", "2022-01-01", np.nan, np.nan, None, np.nan],       # no earlier row: fixed fields from later
        ["A2", "P1", "MM1", "2023-01-01", 5.0, 1600.0, "garage", 8000.0],
        ["A3", "P1", "MM2", "2024-06-01", np.nan, np.nan, None, np.nan],       # new vehicle: no donor for vehicle fields
        ["B1", "P2", "MM1", "2024-01-01", 3.0, 2000.0, "street", 12000.0],
    ])
    test = _motor([
        ["T1", "P1", "MM1", "2025-01-01", np.nan, np.nan, None, np.nan],       # donor A2 is earlier
        ["T2", "P2", "MM1", "2025-01-01", np.nan, np.nan, None, np.nan],
    ])
    tr, te = dr.recover_from_history(train, test, "motor")
    tr = tr.set_index("row_id"); te = te.set_index("row_id")
    assert tr.loc["A1", "vehicle_age"] == 4.0 and tr.loc["A1", "engine_size_cc"] == 1600.0
    assert pd.isna(tr.loc["A1", "annual_mileage"])                             # mileage only from earlier rows
    assert pd.isna(tr.loc["A3", "vehicle_age"]) and pd.isna(tr.loc["A3", "engine_size_cc"])
    assert tr.loc["A3", "overnight_parking"] == "garage"                       # parking follows the policy
    assert te.loc["T1", "vehicle_age"] == 7.0 and te.loc["T1", "annual_mileage"] == 8000.0
    assert te.loc["T2", "vehicle_age"] == 4.0 and te.loc["T2", "overnight_parking"] == "street"
    assert bool(te.loc["T1", "dq_vehicle_age_recovered"]) and not bool(tr.loc["A2", "dq_vehicle_age_recovered"])


def test_recovery_never_takes_a_later_row_for_test():
    train = _motor([["A1", "P1", "MM1", "2024-01-01", 2.0, 1400.0, "garage", 5000.0]])
    test = _motor([["T1", "P1", "MM1", "2025-01-01", np.nan, np.nan, None, np.nan],
                   ["T2", "P1", "MM9", "2025-03-01", 1.0, 3000.0, "drive", 9000.0],
                   ["T3", "P1", "MM9", "2025-02-01", np.nan, np.nan, None, np.nan]])
    _, te = dr.recover_from_history(train, test, "motor")
    te = te.set_index("row_id")
    assert te.loc["T1", "engine_size_cc"] == 1400.0
    assert pd.isna(te.loc["T3", "engine_size_cc"])                             # only donor (T2) starts later


def test_claims_match_by_loss_date_and_exclude_repudiated_cost():
    rows = _rows(["2023-01-01", "2023-06-01"], ["2023-05-31", "2023-12-31"], [151 / 365, 214 / 365],
                 policy_id=["P1", "P1"])
    claims = pd.DataFrame({"claim_id": ["C1", "C2", "C3", "C4"], "policy_id": ["P1"] * 4, "line": ["motor"] * 4,
                           "loss_date": pd.to_datetime(["2023-02-01", "2023-07-01", "2022-12-15", "2023-08-01"]),
                           "status": ["closed", "open", "closed", "repudiated"],
                           "incurred_to_date": [1000.0, 500.0, 300.0, 0.0], "recoveries": [100.0, 0.0, 0.0, 0.0]})
    m = dr.match_claims_to_rows(rows, claims, "motor").set_index("claim_id")
    assert m.loc["C1", "row_id"] == "R0" and m.loc["C2", "row_id"] == "R1"
    assert m.loc["C1", "net_cost"] == 900.0 and m.loc["C4", "net_cost"] == 0.0 and bool(m.loc["C4", "repudiated"])
    assert dr.unmatched_claims(rows, claims, "motor")["claim_id"].tolist() == ["C3"]


def test_apply_policy_rules_needs_cutoff_for_property():
    with pytest.raises(ValueError, match="si_cutoff"):
        dr.apply_policy_rules(pd.DataFrame(), "buildings")


@pytest.mark.skipif(not HAVE_DATA, reason="competition data not present")
@pytest.mark.parametrize("line", dr.LINES)
def test_real_files_repair_cleanly(line):
    train = dr.read_raw(f"policies_{line}")
    test = dr.read_raw(f"test_{line}")
    cutoff = dr.sum_insured_cutoff(train["sum_insured"], 10) if line != "motor" else None
    tr = dr.apply_policy_rules(train, line, cutoff)
    te = dr.apply_policy_rules(test, line, cutoff)
    days = dr.row_days(tr)
    np.testing.assert_allclose(tr["exposure"], days / 365, atol=6e-4)
    assert not te["dq_exposure_days"].any() and not te["dq_exposure_360"].any()
    if line == "motor":
        cust = dr.read_raw("customers")
        assert tr["customer_id"].isin(cust["customer_id"]).all() and te["customer_id"].isin(cust["customer_id"]).all()
    else:
        assert tr["sum_insured"].max() < cutoff and te["sum_insured"].max() < cutoff
    if line == "buildings":
        assert set(tr["construction_type"].dropna()) == {"brick", "stone", "timber", "non_standard"}
    tr2, te2 = dr.recover_from_history(tr, te, line)
    assert (tr2["row_id"].values == tr["row_id"].values).all() and (te2["row_id"].values == te["row_id"].values).all()
    for col in dr.RECOVERABLE[line]:
        assert tr2[col].isna().sum() < 0.5 * tr[col].isna().sum()
        assert tr2.loc[tr2[f"dq_{col}_recovered"], col].notna().all()


def _claims(status, peril, incurred, recoveries, year=2023):
    n = len(status)
    return pd.DataFrame({"claim_id": [f"C{i}" for i in range(n)], "line": "motor", "peril": peril, "status": status,
                         "loss_date": pd.Timestamp(f"{year}-06-01"), "incurred_to_date": incurred, "recoveries": recoveries})


def test_open_recoveries_applied_at_peril_rate():
    # own damage: closed claims recover 25% of incurred; theft has too few closed claims for its rate to be used
    st = ["closed"] * 4 + ["open", "open", "closed", "open", "repudiated"]
    pe = ["own_damage"] * 6 + ["theft", "theft", "own_damage"]
    c = _claims(st, pe, [100.0, 300.0, 200.0, 200.0, 400.0, 0.0, 50.0, 80.0, 90.0], [0, 100, 50, 50, 0, 0, 10, 0, 0])
    out, rep = dr.add_open_recoveries(c, (2023,), max_open_share=0.01, min_closed=2)
    assert rep["applied"] and rep["not_applied_thin"] == ["motor|theft"]
    np.testing.assert_allclose(out["recoveries"], [0, 100, 50, 50, 100.0, 0, 10, 0, 0])   # 400 x 0.25 on the open claim
    assert out["dq_recovery_expected"].tolist() == [False] * 4 + [True] + [False] * 4
    assert out["recoveries_recorded"].tolist() == c["recoveries"].tolist()


def test_open_recoveries_off_when_open_claims_already_recover():
    c = _claims(["closed", "open", "open"], ["own_damage"] * 3, [100.0, 100.0, 100.0], [20, 5, 0])
    out, rep = dr.add_open_recoveries(c, (2023,), max_open_share=0.01, min_closed=1)
    assert not rep["applied"] and "50.00%" in rep["reason"]
    assert out["recoveries"].tolist() == [20, 5, 0] and not out["dq_recovery_expected"].any()


def test_open_recoveries_rate_uses_only_the_given_years():
    c = pd.concat([_claims(["closed"], ["own_damage"], [100.0], [50.0], year=2020),
                   _claims(["closed", "open"], ["own_damage"] * 2, [100.0, 100.0], [10.0, 0.0])], ignore_index=True)
    out, _ = dr.add_open_recoveries(c, (2023,), max_open_share=0.01, min_closed=1)
    assert out["recoveries"].iloc[-1] == pytest.approx(10.0)
