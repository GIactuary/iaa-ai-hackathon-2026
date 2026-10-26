"""What changed: compare a new market's files with a reference market, and a market's training rows with its test rows
(Part 2 kit, decisions/part2_kit_design.md section 2). Everything here reports; nothing repairs or drops a row.

- `schema_diff`: columns added, removed or retyped.
- `numeric_shift`, `categorical_shift`: quantiles and the population stability index (PSI) against the reference's
  deciles or levels; new and vanished levels.
- `detectors`: the Part 1 traps as tests (exposure entered in days, sum insured unit errors, identifier length,
  duplicated row ids, outcome or post-event columns present, undocumented codes).
- `market_files`: the sample submission against the test rows, reference policies against the test files, the expense table.
- `claims_summary`: peril mix, frequency and severity by accident year, report delays, repudiation and open shares,
  and the largest claims on the log-cost screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that were outcomes or post-event measures in Part 1 (data_rules.NEVER_INPUTS, X2): flagged if present
POST_EVENT = ("renewal_decision", "customer_service_contacts", "repairer_id", "third_party_id")
ID_COLS = ("row_id", "policy_id", "customer_id", "address_id", "claim_id")
PSI_BINS = 10


def schema_diff(ref: pd.DataFrame, new: pd.DataFrame) -> dict:
    rc, nc = set(ref.columns), set(new.columns)
    retyped = {c: (str(ref[c].dtype), str(new[c].dtype)) for c in rc & nc if _kind(ref[c]) != _kind(new[c])}
    return dict(added=sorted(nc - rc), removed=sorted(rc - nc), retyped=retyped)


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "text"


def psi(ref_share: np.ndarray, new_share: np.ndarray, floor: float = 1e-4) -> float:
    """Population stability index: sum of (new - ref) x ln(new / ref) over bins. Under 0.1 is usually read as stable,
    0.1 to 0.25 as a moderate shift, above 0.25 as a large one."""
    r = np.clip(np.asarray(ref_share, float), floor, None)
    n = np.clip(np.asarray(new_share, float), floor, None)
    return float(np.sum((n - r) * np.log(n / r)))


def numeric_shift(ref: pd.Series, new: pd.Series) -> dict:
    r, n = ref.dropna().astype(float), new.dropna().astype(float)
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    out = dict(blank_ref=float(ref.isna().mean()), blank_new=float(new.isna().mean()),
               q_ref=r.quantile(q).round(4).tolist() if len(r) else [], q_new=n.quantile(q).round(4).tolist() if len(n) else [])
    if len(r) and len(n) and r.nunique() > 1:
        edges = np.unique(r.quantile(np.linspace(0, 1, PSI_BINS + 1)).to_numpy())
        edges[0], edges[-1] = -np.inf, np.inf
        rs = np.histogram(r, edges)[0] / len(r)
        ns = np.histogram(n, edges)[0] / len(n)
        out["psi"] = psi(rs, ns)
        out["beyond_ref_max"] = float((n > r.max()).mean())
        out["below_ref_min"] = float((n < r.min()).mean())
    return out


def categorical_shift(ref: pd.Series, new: pd.Series, top: int = 5) -> dict:
    r = ref.astype("string").fillna("(blank)").value_counts(normalize=True)
    n = new.astype("string").fillna("(blank)").value_counts(normalize=True)
    levels = r.index.union(n.index)
    rs, ns = r.reindex(levels, fill_value=0.0), n.reindex(levels, fill_value=0.0)
    move = (ns - rs).abs().sort_values(ascending=False)
    return dict(blank_ref=float(ref.isna().mean()), blank_new=float(new.isna().mean()), psi=psi(rs.to_numpy(), ns.to_numpy()),
                new_levels=sorted(map(str, set(n.index) - set(r.index)))[:20], vanished_levels=sorted(map(str, set(r.index) - set(n.index)))[:20],
                n_levels_ref=int(len(r)), n_levels_new=int(len(n)),
                largest_moves={str(k): (round(float(rs[k]), 4), round(float(ns[k]), 4)) for k in move.index[:top]})


def column_shifts(ref: pd.DataFrame, new: pd.DataFrame, max_levels: int = 200) -> dict:
    """Shift of every shared column (identifiers and free-text columns with many levels are skipped)."""
    out = {}
    for c in sorted(set(ref.columns) & set(new.columns)):
        if c in ID_COLS or c.endswith("_id"):
            continue
        k = _kind(ref[c])
        if k == "number" and _kind(new[c]) == "number":
            out[c] = dict(kind="number", **numeric_shift(ref[c], new[c]))
        elif k == "date":
            out[c] = dict(kind="date", min_ref=str(ref[c].min()), max_ref=str(ref[c].max()), min_new=str(new[c].min()), max_new=str(new[c].max()))
        elif ref[c].nunique() <= max_levels:
            out[c] = dict(kind="category", **categorical_shift(ref[c], new[c]))
    return out


def detectors(df: pd.DataFrame, ref: pd.DataFrame | None = None, documented: dict | None = None) -> dict:
    """The Part 1 traps, as counts of rows that look affected."""
    out = {}
    if {"exposure", "period_start", "period_end"} <= set(df.columns):
        days = (pd.to_datetime(df["period_end"]) - pd.to_datetime(df["period_start"])).dt.days + 1
        out["exposure_above_1"] = int((df["exposure"] > 1.0001).sum())
        out["exposure_equals_day_count"] = int(((df["exposure"] - days).abs() <= 0.5).sum())
        out["exposure_zero_or_less"] = int((df["exposure"] <= 0).sum())
    if "sum_insured" in df.columns and ref is not None and "sum_insured" in ref.columns:
        p99 = ref["sum_insured"].quantile(0.99)
        out["sum_insured_above_10x_ref_p99"] = int((df["sum_insured"] > 10 * p99).sum())
    for c in ID_COLS:
        if c in df.columns:
            ln = df[c].astype("string").str.len().value_counts()
            out[f"{c}_lengths"] = {int(k): int(v) for k, v in ln.items()} if len(ln) <= 6 else f"{len(ln)} different lengths"
    if "row_id" in df.columns:
        out["duplicated_row_id"] = int(df["row_id"].duplicated().sum())
    out["post_event_columns_present"] = [c for c in POST_EVENT if c in df.columns]
    for c, levels in (documented or {}).items():
        if c in df.columns:
            bad = df[c].dropna().astype("string").str.lower()
            out[f"{c}_undocumented"] = {str(k): int(v) for k, v in bad[~bad.isin(levels)].value_counts().head(10).items()}
    return out


def market_files(files: dict[str, pd.DataFrame]) -> dict:
    """Checks on the files a market adds beyond the core tables: the sample submission covers the test rows exactly,
    reference policies are test rows of the line they claim, and the expense table as given."""
    tests = {s[len("test_"):]: df for s, df in files.items() if s.startswith("test_") and "row_id" in df.columns}
    test_ids = {l: set(df["row_id"].astype("string")) for l, df in tests.items()}
    all_ids = set().union(*test_ids.values()) if test_ids else set()
    out = {}
    if "sample_submission" in files and tests:
        ids = files["sample_submission"]["row_id"].astype("string")
        out["sample_submission"] = dict(rows=int(len(ids)), test_rows=int(sum(len(df) for df in tests.values())),
                                        duplicated=int(ids.duplicated().sum()), not_in_test=int((~ids.isin(all_ids)).sum()),
                                        test_rows_missing=int(len(all_ids - set(ids))))
    if "reference_policies" in files and tests:
        r = files["reference_policies"]
        found = r["row_id"].astype("string").isin(all_ids)
        own_line = [str(i) in test_ids.get(str(l), set()) for i, l in zip(r["row_id"], r.get("line", pd.Series([None] * len(r))))]
        out["reference_policies"] = dict(rows=int(len(r)), by_line={str(k): int(v) for k, v in r["line"].value_counts().items()} if "line" in r else {},
                                         in_test=int(found.sum()), in_own_line_test=int(sum(own_line)))
    if "expenses" in files:
        out["expenses"] = files["expenses"].astype(object).where(files["expenses"].notna(), None).to_dict("records")
    return out


def claims_summary(claims: pd.DataFrame, z: float) -> dict:
    c = claims.copy()
    c["net_cost"] = c["incurred_to_date"] - c.get("recoveries", 0.0)
    c["lag"] = (pd.to_datetime(c["report_date"]) - pd.to_datetime(c["loss_date"])).dt.days
    c["ay"] = pd.to_datetime(c["loss_date"]).dt.year
    rep = c["status"].eq("repudiated") if "status" in c.columns else pd.Series(False, index=c.index)
    ok = c[~rep & (c["net_cost"] > 0)].copy()
    ok["lc"] = np.log(ok["net_cost"])
    st = ok.groupby(["line", "peril"])["lc"].agg(["mean", "std"])
    ok = ok.join(st, on=["line", "peril"])
    ok["z"] = (ok["lc"] - ok["mean"]) / ok["std"]
    by_ay = ok.groupby(["line", "ay"]).agg(claims=("net_cost", "size"), mean_cost=("net_cost", "mean")).round(1)
    return dict(
        claims=int(len(c)), repudiated_share=float(rep.mean()),
        open_share=float(c["status"].eq("open").mean()) if "status" in c.columns else None,
        nil_or_negative=int((c.loc[~rep, "net_cost"] <= 0).sum()),
        peril_mix={f"{l}|{p}": round(float(v), 4) for (l, p), v in c[~rep].groupby(["line", "peril"]).size().div(c[~rep].groupby("line").size(), level="line").items()},
        by_accident_year={f"{l}|{y}": dict(r) for (l, y), r in by_ay.iterrows()},
        report_lag_days={l: g["lag"].quantile([0.5, 0.9, 0.99]).round(0).tolist() for l, g in c.groupby("line")},
        negative_lag=int((c["lag"] < 0).sum()),
        screen_hits=ok.loc[ok["z"] > z, ["claim_id", "line", "peril", "net_cost", "z"]].sort_values("z", ascending=False).head(10).round(2).to_dict("records"))
