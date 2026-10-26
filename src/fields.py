"""Inputs the tree families may use (Y2, decided 2026-10-24, exchange D32; docs/specs/SPEC_part2_settings.md section 2).

The trees get every legitimate field of the built rows, not only the fields behind the W2 GLM. Left out, each with
its reason: identifiers and dates, exposure and the hold-out flag, the target and its pieces, data-quality flags,
outcomes and post-event fields (data_rules.NEVER_INPUTS), the P6 marketing and identity fields, previous insurer
(a marketing field, never merged in Part 1), the incumbent's premium (it enters through Y1 only), and gender (held
out until the actuary rules on it, exchange D33). A code with more levels than `code_levels_max` (thresholds file)
enters only credibility-encoded (`credibility_encode`), never as a raw category: on Part 1, make_model's 300 levels
as a raw category lost 414,287 of deviance in motor (post-mortem PM2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_rules as dr, history as hs, validation as v

REASONS = {
    **dict.fromkeys(("row_id", "policy_id", "customer_id", "address_id", "claim_id", "period_start", "period_end",
                     "date_of_birth", "line"), "identifier or date"),
    **dict.fromkeys(("exposure", "holdout", "yr"), "exposure, hold-out or year (the year enters separately)"),
    **dict.fromkeys(dr.NEVER_INPUTS, "outcome or post-event field (R7)"),
    **dict.fromkeys(dr.EXCLUDED_FROM_PLAN, "P6 marketing or identity field"),
    "previous_insurer": "marketing field, close to P6; never merged in Part 1 (phase 5 step 5 design)",
    "written_premium": "the incumbent's premium enters through Y1 only",
    "discount_applied": "part of the incumbent's premium; enters through Y1 only",
    "gender": "held out until the actuary rules on it (exchange D33)",
}
PREFIXES = {"cost": "target", "n_": "target (claim counts)", "completion": "reporting completion of the target",
            "dq_": "data-quality flag", "wx_": "weather (Y4: one factor per line)", "norm_": "weather (Y4: one factor per line)",
            "h_": "claims history (enters as the carried history columns)"}


def is_category(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def tree_fields(df: pd.DataFrame, max_levels: float) -> dict:
    """Split the columns of `df` (the fitting rows) into numeric inputs, raw categories, codes to encode, and the
    excluded columns with their reasons. A category's level count ignores blanks."""
    num, cat, enc, excluded = [], [], [], {}
    for c in df.columns:
        reason = REASONS.get(c) or next((r for p, r in PREFIXES.items() if c.startswith(p)), None)
        if reason:
            excluded[c] = reason
        elif not is_category(df[c]):
            num.append(c)
        elif df[c].nunique(dropna=True) > max_levels:
            enc.append(c)
        else:
            cat.append(c)
    return dict(num=num, cat=cat, encode=enc, excluded=excluded)


def credibility_encode(code: pd.Series, yr: np.ndarray, A: np.ndarray, E: np.ndarray, fit: np.ndarray,
                       groups: np.ndarray, seed: int, folds: int = 5) -> tuple[np.ndarray, float]:
    """Log Bühlmann-Straub mod of actual (A) against expected (E) cost by level of `code`, as M7 did for make_model
    (scripts/phase5_step5.py). k is estimated on the fitting rows. Fitting rows get the mod from the other
    customer-grouped inner folds, so no row's own claims enter its own value; other rows get it from every fitting
    row. Blanks are a level of their own; a level never seen in fitting gets 0 (no adjustment)."""
    c = code.astype("string").fillna("blank").to_numpy()
    fi = np.flatnonzero(fit)
    bs = hs.buhlmann_straub_k(c[fi], np.asarray(yr)[fi], A[fi], E[fi])
    k = float(bs["k"])
    if bs["units_multi"] == 0 or np.isnan(k):
        raise ValueError(f"credibility constant for {code.name!r} cannot be estimated: the within-level variance needs two or more years per level")
    if np.isinf(k):                     # no between-level variance: the code carries no signal, every level gets 0
        return np.zeros(len(c)), k

    def mods(idx):
        s = pd.DataFrame({"c": c[idx], "A": A[idx], "E": E[idx]}).groupby("c")[["A", "E"]].sum()
        return np.log((s["A"] + k) / (s["E"] + k))

    out = np.array(pd.Series(c).map(mods(fi)).fillna(0.0), dtype=float)
    fold = v.make_folds(len(fi), [seed], n_folds=folds, groups=np.asarray(groups)[fi])[seed]
    for a, b in fold:
        out[fi[b]] = pd.Series(c[fi[b]]).map(mods(fi[a])).fillna(0.0).to_numpy()
    return out, k
