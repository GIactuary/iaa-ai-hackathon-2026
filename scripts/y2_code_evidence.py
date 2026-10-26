"""Evidence for the Y2 code-level threshold (register Y2-T): for each candidate code, how much data each level has and
how credible its claim experience is, in the emerging market and, for comparison, the developed one.

Claim frequency, not cost: frequency needs no large-loss cap (the Part 2 caps are not yet re-confirmed) and is the
steadier signal. Per line, non-repudiated claims matched to their rows (R6); expected claims = exposure x the line's
frequency in that year. Bühlmann-Straub across levels, with each year as one period (src/history.py), gives k; a
level with expected claims E is given credibility Z = E / (E + k). sd_relativity is the spread of the levels' true
frequency relativities the data supports (square root of the between-level variance over the mean).
Run: .venv/Scripts/python.exe -I scripts/y2_code_evidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import data_rules as dr, history as hs
from src.build import CUSTOMER_COLS
from src.evidence import write_evidence

ROOT = Path(__file__).resolve().parents[1]
MARKETS = {"emerging": (ROOT / "data/raw_part2/emerging", (2023, 2024)), "developed": (ROOT / "data/raw_part2/developed", (2021, 2022, 2023, 2024))}
CODES = {"motor": ["make_model", "occupation", "occupation_clean", "region", "vehicle_colour", "energy_rating"],
         "buildings": ["occupation", "occupation_clean", "region", "energy_rating"],
         "contents": ["occupation", "occupation_clean", "region", "energy_rating"]}


def line_frame(raw: Path, years: tuple, line: str, ad, cu, claims) -> pd.DataFrame:
    tr = dr.read_raw(f"policies_{line}", raw)
    tr = dr.apply_policy_rules(tr, line, float("inf") if line != "motor" else None)
    tr = tr.merge(ad, on="address_id", how="left").merge(cu[[c for c in CUSTOMER_COLS if c in cu.columns]], on="customer_id", how="left")
    tr["yr"] = tr["period_start"].dt.year
    tr = tr[tr["yr"].isin(years)].reset_index(drop=True)
    m = dr.match_claims_to_rows(tr, claims, line)
    tr["n"] = tr["row_id"].map(m[~m["repudiated"]].groupby("row_id").size()).fillna(0.0)
    freq = tr.groupby("yr")["n"].sum() / tr.groupby("yr")["exposure"].sum()
    tr["E"] = tr["exposure"] * tr["yr"].map(freq)
    tr["occupation_clean"] = tr["occupation"].astype("string").str.strip().str.lower()
    return tr


def code_stats(tr: pd.DataFrame, code: str) -> dict:
    c = tr[code].astype("string").fillna("blank")
    bs = hs.buhlmann_straub_k(c, tr["yr"], tr["n"], tr["E"])
    lv = tr.groupby(c).agg(E=("E", "sum"), n=("n", "sum"), rows=("n", "size"))
    k = bs["k"]
    z = lambda e: float(e / (e + k)) if np.isfinite(k) else 0.0
    return dict(levels=int(len(lv)), rows_median=float(lv["rows"].median()), claims_median=float(lv["n"].median()),
                claims_min=float(lv["n"].min()), levels_under_30_claims=int((lv["n"] < 30).sum()),
                k=float(k), z_median_level=z(lv["E"].median()), z_smallest_level=z(lv["E"].min()), z_largest_level=z(lv["E"].max()),
                sd_relativity=float(np.sqrt(max(bs["a"], 0.0)) / bs["xbar"]))


def run() -> dict:
    out = {}
    for mkt, (raw, years) in MARKETS.items():
        ad, cu, claims = dr.read_raw("addresses", raw), dr.read_raw("customers", raw), dr.read_raw("claims", raw)
        for line, codes in CODES.items():
            tr = line_frame(raw, years, line, ad, cu, claims)
            for code in codes:
                if code in tr.columns:
                    out[f"{mkt}|{line}|{code}"] = s = code_stats(tr, code)
                    print(f"{mkt:9} {line:9} {code:17} levels {s['levels']:3}  claims/level median {s['claims_median']:7.0f} min {s['claims_min']:5.0f}  "
                          f"<30 claims {s['levels_under_30_claims']:3}  k {s['k']:9.1f}  Z median {s['z_median_level']:.2f} (min {s['z_smallest_level']:.2f})  sd rel {s['sd_relativity']:.3f}", flush=True)
    write_evidence({"y2_code_evidence": out}, ROOT / "outputs" / "y2_code_evidence.json")
    return out


if __name__ == "__main__":
    run()
