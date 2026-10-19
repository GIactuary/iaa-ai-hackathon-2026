"""Phase 2 screens (spec section 2): the claim outlier screen and the train-versus-test evidence check.

Both read their thresholds from decisions/thresholds.md (`screen_z`, `refer_size_ratio`), so neither runs on a value
that was not written down first. Screens only list records; exclusion and referral are decided by the actuary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Size measures and the segments they are compared within (agreed with the actuary, thresholds.md refer_size_ratio)
SIZE_CHECKS = {
    "buildings": [("sum_insured", ("property_type", "bedrooms"))],
    "contents": [("sum_insured", ("tenure", "bedrooms"))],
    "motor": [("vehicle_value_group", ()), ("engine_size_cc", ()), ("annual_mileage", ())],
}


def claim_screen(matched_claims: pd.DataFrame, z: float) -> pd.DataFrame:
    """Claims whose log net cost is more than `z` standard deviations above their peril's mean.

    `matched_claims` is the output of data_rules.match_claims_to_rows for every line, concatenated. Repudiated and
    zero-cost claims are left out of the fit. Returns the flagged claims with their peril mean, sd and z-score.
    """
    c = matched_claims[~matched_claims["repudiated"] & (matched_claims["net_cost"] > 0)].copy()
    c["log_cost"] = np.log(c["net_cost"])
    stats = c.groupby(["line", "peril"])["log_cost"].agg(peril_mean="mean", peril_sd="std")
    c = c.join(stats, on=["line", "peril"])
    c["z"] = (c["log_cost"] - c["peril_mean"]) / c["peril_sd"]
    c["fitted_cost"] = np.exp(c["peril_mean"])
    return c[c["z"] > z].sort_values("z", ascending=False)


def size_referrals(train: pd.DataFrame, test: pd.DataFrame, measure: str, segment: tuple, ratio: float) -> pd.DataFrame:
    """Test rows whose `measure` exceeds `ratio` times the largest training value in the same segment, plus test rows
    in a segment with no training rows at all."""
    seg = list(segment)
    if seg:
        mx = train.groupby(seg, dropna=False)[measure].max().rename("train_max")
        t = test.join(mx, on=seg)
    else:
        t = test.assign(train_max=train[measure].max())
    t = t.assign(measure=measure, ratio_to_max=t[measure] / t["train_max"])
    out_of_range = t[measure] > ratio * t["train_max"]
    unseen_segment = t["train_max"].isna() & t[measure].notna()
    t["reason"] = np.where(unseen_segment, "segment unseen in training", "beyond the segment's training maximum")
    return t[out_of_range | unseen_segment]


def unseen_levels(train: pd.DataFrame, test: pd.DataFrame, columns) -> dict:
    """Category levels present in the test file and absent from training, by column."""
    out = {}
    for col in columns:
        new = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
        if new:
            out[col] = sorted(map(str, new))
    return out


def impossible_motor(df: pd.DataFrame) -> pd.DataFrame:
    """Flags for motor combinations that cannot happen as recorded (data_rules.md, phase 2 items)."""
    return pd.DataFrame({
        "flag_ncb_above_licence": df["ncb_years"] > df["years_licensed"],
        "flag_licence_before_16": df["years_licensed"] > df["driver_age"] - 16,
    }, index=df.index)
