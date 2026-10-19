"""Point-in-time claims history with Bühlmann-Straub credibility (phase 5 step 4; C7; P5).

Design: `decisions/phase5_step4_design.md`. For each target row with cutoff c = min(row start, data end), and for a
scope (its own policy, the customer's other policies, other customers at the address):
  A = claims on the scope's earlier rows reported before c (all statuses),
  E = sum over those rows of class frequency x exposure before c x share of claims expected reported by c.
The experience mod is (A + k) / (E + k), the Bühlmann-Straub estimate Z A/E + (1 - Z) with Z = E / (E + k). A row with
no history has E = 0 and mod 1. k is estimated from the fitting years by `buhlmann_straub_k`.

Days are counted from targets.GRID_START (1 January 2021). Hold-out customers must be removed from the source rows
and claims by the caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import targets as tg

GROUPS = {"motor": {"fault": ("own_damage", "third_party_prop", "bodily_injury"), "nonfault": ("glass_hail", "theft")},
          "buildings": {"weather": ("storm", "flood", "subsidence"), "other": ("escape_of_water", "fire", "accidental_damage")},
          "contents": {"weather": ("flood",), "other": ("theft", "escape_of_water", "accidental_damage")}}


def day(ts) -> np.ndarray:
    return (pd.to_datetime(ts) - tg.GRID_START).dt.days.to_numpy() if isinstance(ts, pd.Series) else (pd.Timestamp(ts) - tg.GRID_START).days


def reported_share_cum(claims: pd.DataFrame, line: str) -> np.ndarray:
    """H[x] = sum over u = 0..x of F(u), F the line's count-weighted reporting-delay curve (P10 curves)."""
    curves, shares = tg.delay_curves(claims), tg.peril_count_shares(claims)[line]
    F = sum(w * curves[(line, p)] for p, w in shares.items())
    return np.cumsum(F)


def completed_exposure(s, e, c, H) -> np.ndarray:
    """Exposure in years before cutoff c of rows covering days s..e, each day weighted by the share of its claims
    expected to be reported by c: sum over d in [s, min(e, c - 1)] of F(c - 1 - d), divided by 365."""
    s, e, c = (np.asarray(x, dtype=np.int64) for x in (s, e, c))
    m = np.minimum(e, c - 1)
    ok = s <= m
    hi = np.clip(c - 1 - s, 0, len(H) - 1)
    lo_idx = c - 2 - m
    lo = np.where(lo_idx >= 0, H[np.clip(lo_idx, 0, len(H) - 1)], 0.0)
    # ages beyond the curve are fully reported: F = 1 there
    over_hi = np.maximum(c - 1 - s - (len(H) - 1), 0)
    over_lo = np.maximum(lo_idx - (len(H) - 1), 0)
    out = (H[hi] + over_hi) - (lo + over_lo)
    return np.where(ok, out / 365.0, 0.0)


def plain_exposure(s, e, c) -> np.ndarray:
    s, e, c = (np.asarray(x, dtype=np.int64) for x in (s, e, c))
    return np.clip(np.minimum(e, c - 1) - s + 1, 0, None) / 365.0


def scope_sums(targets: pd.DataFrame, src_rows: pd.DataFrame, src_claims: pd.DataFrame, key: list[str],
               H: dict[str, np.ndarray], rate_cols=("rate", "rate_avg"), claim_flags=()) -> pd.DataFrame:
    """For each target row, sums over source rows sharing `key` that start before the row's cutoff `c`.

    targets: key columns and `c` (cutoff day). src_rows: key columns, `line`, `s`, `e` (first and last day) and the
    rate columns (claims per exposure year). src_claims: key columns and `report` (day), plus boolean `claim_flags`.
    Returns, aligned to targets: E_<rate> (completed expected claims), X (exposure years), A (claims reported before
    c) and A_<flag> for each flag.
    """
    u = targets[key + ["c"]].drop_duplicates().reset_index(drop=True)
    r = u.merge(src_rows[key + ["line", "s", "e", *rate_cols]], on=key, how="inner")
    r = r[r["s"] < r["c"]]
    out = u.copy()
    if len(r):
        ce = np.zeros(len(r))
        for ln, h in H.items():
            m = (r["line"] == ln).to_numpy()
            ce[m] = completed_exposure(r["s"].to_numpy()[m], r["e"].to_numpy()[m], r["c"].to_numpy()[m], h)
        agg = {f"E_{rc}": r[rc].to_numpy() * ce for rc in rate_cols}
        agg["X"] = plain_exposure(r["s"], r["e"], r["c"])
        g = pd.DataFrame(agg).assign(**{k: r[k].to_numpy() for k in key + ["c"]}).groupby(key + ["c"]).sum()
        out = out.merge(g.reset_index(), on=key + ["c"], how="left")
    q = u.merge(src_claims[key + ["report", *claim_flags]], on=key, how="inner")
    q = q[q["report"] < q["c"]]
    qa = {"A": np.ones(len(q))} | {f"A_{f}": q[f].to_numpy().astype(float) for f in claim_flags}
    ga = pd.DataFrame(qa).assign(**{k: q[k].to_numpy() for k in key + ["c"]}).groupby(key + ["c"]).sum()
    out = out.merge(ga.reset_index(), on=key + ["c"], how="left")
    cols = [f"E_{rc}" for rc in rate_cols] + ["X", "A"] + [f"A_{f}" for f in claim_flags]
    for col in cols:
        out[col] = out[col].fillna(0.0) if col in out.columns else 0.0
    res = targets[key + ["c"]].merge(out, on=key + ["c"], how="left")
    return res[cols].fillna(0.0).set_axis(targets.index)


def buhlmann_straub_k(unit, period, A, E) -> dict:
    """Non-parametric Bühlmann-Straub estimate of k = s2 / a for claim ratios X = A / E with weights E.

    unit, period: identifiers; A, E: claims and expected claims per (unit, period) record (aggregated here).
    Returns k, s2, a, the Poisson-gamma check (s2 = mean ratio, moment estimate of a), and the counts used.
    If a <= 0, k is infinite: the data shows no between-risk variance.
    """
    df = pd.DataFrame({"u": np.asarray(unit), "p": np.asarray(period), "A": np.asarray(A, float), "E": np.asarray(E, float)})
    df = df.groupby(["u", "p"], sort=False)[["A", "E"]].sum().reset_index()
    df = df[df["E"] > 0]
    df["X"] = df["A"] / df["E"]
    gu = df.groupby("u", sort=False)
    wi, Ai, ni = gu["E"].sum(), gu["A"].sum(), gu.size()
    Xi = Ai / wi
    df["Xi"] = df["u"].map(Xi)
    multi = ni[ni > 1].index
    dm = df[df["u"].isin(multi)]
    s2 = float((dm["E"] * (dm["X"] - dm["Xi"]) ** 2).sum() / (ni[ni > 1] - 1).sum())
    w = float(wi.sum()); r = len(wi); Xbar = float(Ai.sum() / w)
    a = float(((wi * (Xi - Xbar) ** 2).sum() - (r - 1) * s2) / (w - (wi ** 2).sum() / w))
    # Poisson-gamma check: A_i ~ Poisson(theta_i E_i), mean theta = Xbar, Var theta = a, so
    # Var(A_i) = Xbar E_i + a E_i^2: a = [sum (A_i - Xbar E_i)^2 - sum A_i] / sum E_i^2, and s2 = Xbar
    a_pg = float((((Ai - Xbar * wi) ** 2).sum() - Ai.sum()) / (wi ** 2).sum())
    k = s2 / a if a > 0 else float("inf")
    k_pg = Xbar / a_pg if a_pg > 0 else float("inf")
    return dict(k=float(k), s2=s2, a=a, xbar=Xbar, k_poisson_gamma=float(k_pg), a_poisson_gamma=a_pg,
                units=int(r), units_multi=int(len(multi)), records=int(len(df)), expected=w, claims=float(Ai.sum()))


def mod(A, E, k: float) -> np.ndarray:
    """Credibility-weighted experience mod (A + k) / (E + k); 1 where k is infinite or E is 0."""
    A, E = np.asarray(A, float), np.asarray(E, float)
    if not np.isfinite(k):
        return np.ones(len(A))
    return (A + k) / (E + k)


def credibility(E, k: float) -> np.ndarray:
    E = np.asarray(E, float)
    return np.zeros(len(E)) if not np.isfinite(k) else E / (E + k)
