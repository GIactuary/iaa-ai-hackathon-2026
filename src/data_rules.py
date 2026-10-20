"""Data rules (phase 1, spec section 1): read the raw files, repair the planted errors by stated rules, flag every
repaired row, and match claims to policy rows. Every rule and its evidence is written in `decisions/data_rules.md`.

The same rules run on training and test files. Nothing here drops a row: repairs change a value and set a flag,
and the columns that may never be inputs are listed so model code can refuse them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import market
RAW = market.RAW                      # data/raw for Part 1
LINES = market.LINES
ID_COLS = ("row_id", "policy_id", "customer_id", "address_id", "claim_id", "repairer_id", "third_party_id")

# Outcomes or measures taken during the row: never rating inputs (brief section 7; data_rules.md R7)
NEVER_INPUTS = ("renewal_decision", "customer_service_contacts")
# Marketing and identity fields kept out of the rating plan, used only in the proxy test (P6)
EXCLUDED_FROM_PLAN = ("surname", "email_domain", "marketing_segment", "campaign_code", "branch_id",
                      "marketing_consent", "document_preference")
# Documented levels of construction_type (data page) and the codes that map to them (data_rules.md R3)
CONSTRUCTION_MAP = {"brick": "brick", "stone": "stone", "non_standard": "non_standard", "t": "timber", "w": "timber"}

DAY_COUNT_TOL = 0.5      # an exposure within half a day of the row's day count was entered in days
YEAR_BASIS_TOL = 0.0006  # exposures are given to 4 decimals; this separates days/365 from days/360 for every row


def read_raw(name: str, root: Path = RAW) -> pd.DataFrame:
    """Read one competition file with identifiers as text and dates as timestamps."""
    df = pd.read_parquet(Path(root) / f"{name}.parquet")
    for c in ID_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    for c in ("period_start", "period_end", "loss_date", "report_date", "date_of_birth"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def row_days(df: pd.DataFrame) -> pd.Series:
    return (df["period_end"] - df["period_start"]).dt.days + 1


def repair_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """R1. Exposure is the row's length in years (days / 365). Rows entered as a day count or on a 360-day year
    are restated from the dates and flagged. A row that fits none of the three bases raises."""
    out = df.copy()
    days = row_days(out)
    on_365 = (out["exposure"] - days / 365).abs() < YEAR_BASIS_TOL
    as_days = ~on_365 & ((out["exposure"] - days).abs() < DAY_COUNT_TOL)
    on_360 = ~on_365 & ~as_days & ((out["exposure"] - days / 360).abs() < YEAR_BASIS_TOL)
    bad = ~(on_365 | as_days | on_360)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows fit no exposure basis, e.g. row {out.loc[bad, 'row_id'].iloc[0]}")
    out["dq_exposure_days"] = as_days
    out["dq_exposure_360"] = on_360
    out.loc[as_days | on_360, "exposure"] = days[as_days | on_360] / 365
    return out


def sum_insured_cutoff(train_sum_insured: pd.Series, multiple: float) -> float:
    """R2 cut-off: `multiple` times the line's training 99th percentile (threshold si_unit_multiple)."""
    return float(multiple * train_sum_insured.quantile(0.99))


def repair_sum_insured(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """R2. A sum insured above the cut-off was entered 1,000 times too large: divide by 1,000 and flag."""
    out = df.copy()
    flag = out["sum_insured"] > cutoff
    out["dq_sum_insured_x1000"] = flag
    out.loc[flag, "sum_insured"] = out.loc[flag, "sum_insured"] / 1000
    return out


def normalise_construction(df: pd.DataFrame) -> pd.DataFrame:
    """R3. One spelling per documented level; codes T (2021-2022) and W (2023 on) are both timber."""
    out = df.copy()
    raw = out["construction_type"].astype("string").str.strip().str.lower()
    mapped = raw.map(CONSTRUCTION_MAP)
    unknown = raw.notna() & mapped.isna()
    if unknown.any():
        raise ValueError(f"construction codes not in the map: {sorted(raw[unknown].unique())}")
    out["dq_construction_timber_code"] = raw.isin(["t", "w"]).fillna(False)
    out["construction_type"] = mapped
    return out


def pad_customer_id(df: pd.DataFrame, width: int = 8) -> pd.DataFrame:
    """R4. Customer ids lost leading zeros on some motor rows; pad back to the customers table's width."""
    out = df.copy()
    short = out["customer_id"].str.len() < width
    out["dq_customer_id_padded"] = short.fillna(False)
    out["customer_id"] = out["customer_id"].str.zfill(width)
    return out


# R5 was withdrawn on 2026-10-18: annual_mileage of 0 is a real low-mileage value, not a blank code
# (decisions/data_rules.md R5; exchange D8).

# R8. Blanks recovered from the same policy's other rows. Fixed attributes may come from any row of the same
# group on training data; declared mileage, and every recovery on test rows, only from an earlier row.
RECOVERABLE = {
    "motor": {"engine_size_cc": (("policy_id", "make_model"), True),
              "overnight_parking": (("policy_id",), True),
              "annual_mileage": (("policy_id", "make_model"), False)},
    "buildings": {"smoke_alarms": (("policy_id", "address_id"), True)},
    "contents": {"security_level": (("policy_id",), True),
                 "high_value_items": (("policy_id",), True)},
}


def _fill_group(df: pd.DataFrame, col: str, keys: tuple, later_allowed: pd.Series) -> pd.Series:
    """Nearest earlier non-blank value in the group; then, where allowed, the nearest later one."""
    g = df.groupby(list(keys), sort=False, dropna=False)[col]
    filled = g.ffill()
    later = g.bfill()
    return filled.where(filled.notna(), later.where(later_allowed))


def recover_from_history(train: pd.DataFrame, test: pd.DataFrame, line: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R8. Fill recoverable blanks from other rows of the same policy (and vehicle or address), using training
    rows as donors for test rows. Every filled value is flagged `dq_<col>_recovered`. Rows that stay blank keep
    the blank for the model's own missing handling."""
    both = pd.concat([train.assign(_test=False), test.assign(_test=True)], ignore_index=True)
    both = both.sort_values(["policy_id", "period_start", "row_id"], kind="stable")
    for col, (keys, fixed) in RECOVERABLE[line].items():
        later_allowed = ~both["_test"] if fixed else pd.Series(False, index=both.index)
        new = _fill_group(both, col, keys, later_allowed)
        both[f"dq_{col}_recovered"] = both[col].isna() & new.notna()
        both[col] = new
    if line == "motor":
        # Vehicle age recovers through the model year implied by another row of the same vehicle
        model_year = both["period_start"].dt.year - both["vehicle_age"]
        both["_my"] = model_year
        later_allowed = ~both["_test"]
        my = _fill_group(both, "_my", ("policy_id", "make_model"), later_allowed)
        rec = both["vehicle_age"].isna() & my.notna()
        both["dq_vehicle_age_recovered"] = rec
        both.loc[rec, "vehicle_age"] = both.loc[rec, "period_start"].dt.year - my[rec]
        both = both.drop(columns="_my")
    both = both.sort_index()
    tr = both[~both["_test"]].drop(columns="_test").reset_index(drop=True)
    te = both[both["_test"]].drop(columns="_test").reset_index(drop=True)
    return tr, te


def apply_policy_rules(df: pd.DataFrame, line: str, si_cutoff: float | None = None) -> pd.DataFrame:
    """All row repairs for one policy or test file. `si_cutoff` is required for buildings and contents."""
    if line not in LINES:
        raise ValueError(f"unknown line {line!r}")
    if line != "motor" and si_cutoff is None:
        raise ValueError("si_cutoff is required for property lines")
    out = repair_exposure(df)
    if line == "motor":
        out = pad_customer_id(out)
    else:
        out = repair_sum_insured(out, si_cutoff)
    if line == "buildings":
        out = normalise_construction(out)
    return out


def match_claims_to_rows(rows: pd.DataFrame, claims: pd.DataFrame, line: str) -> pd.DataFrame:
    """R6. Each claim belongs to the row of its policy whose period contains the loss date. Returns the matched
    claims with row_id and net cost (incurred less recoveries); repudiated claims are kept with a flag so counts
    can be reported, and claims matching no row are returned separately via `unmatched_claims`."""
    cl = claims[claims["line"] == line]
    m = cl.merge(rows[["row_id", "policy_id", "period_start", "period_end"]], on="policy_id", how="inner")
    m = m[(m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])]
    if m["claim_id"].duplicated().any():
        raise ValueError("a claim matches more than one row")
    m = m.drop(columns=["period_start", "period_end"]).copy()
    m["repudiated"] = m["status"].eq("repudiated")
    m["net_cost"] = np.where(m["repudiated"], 0.0, m["incurred_to_date"] - m["recoveries"])
    return m


def unmatched_claims(rows: pd.DataFrame, claims: pd.DataFrame, line: str) -> pd.DataFrame:
    matched = match_claims_to_rows(rows, claims, line)["claim_id"]
    cl = claims[claims["line"] == line]
    return cl[~cl["claim_id"].isin(matched)]
