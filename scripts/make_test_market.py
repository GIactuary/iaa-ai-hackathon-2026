"""A synthetic test market for the Part 2 kit: Part 1's raw files with known changes planted, so the what-changed
report can be shown to find each one (decisions/part2_kit_design.md section 5). Written to data/work/test_market/.

Planted, with a fixed seed:
  P1  motor: 500 training rows with exposure entered as a day count
  P2  buildings: 300 training rows with sum insured 1,000 times too large; every sum insured +20% (a real shift)
  P3  motor test: column telematics_score removed
  P4  motor test: 2% of rows with a fuel type never seen in training (hydrogen)
  P5  contents: customer_id with the leading zeros stripped on 1,000 training rows
  P6  claims: 50 claims reported before their loss date
Run: .venv/Scripts/python.exe -I scripts/make_test_market.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "data" / "raw", ROOT / "data" / "work" / "test_market"
PLANTED = dict(P1=500, P2=300, P4_share=0.02, P5=1000, P6=50)


def run() -> None:
    rng = np.random.default_rng(2026)
    DST.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("*.parquet"):
        shutil.copy(f, DST / f.name)
    m = pd.read_parquet(DST / "policies_motor.parquet")
    i = rng.choice(len(m), PLANTED["P1"], replace=False)
    days = (pd.to_datetime(m["period_end"]) - pd.to_datetime(m["period_start"])).dt.days + 1
    m.loc[m.index[i], "exposure"] = days.iloc[i].to_numpy().astype(float)
    m.to_parquet(DST / "policies_motor.parquet")
    b = pd.read_parquet(DST / "policies_buildings.parquet")
    b["sum_insured"] = b["sum_insured"] * 1.2
    j = rng.choice(len(b), PLANTED["P2"], replace=False)
    b.loc[b.index[j], "sum_insured"] = b["sum_insured"].iloc[j].to_numpy() * 1000
    b.to_parquet(DST / "policies_buildings.parquet")
    tm = pd.read_parquet(DST / "test_motor.parquet").drop(columns=["telematics_score"])
    k = rng.random(len(tm)) < PLANTED["P4_share"]
    tm["fuel_type"] = tm["fuel_type"].astype("string")
    tm.loc[k, "fuel_type"] = "hydrogen"
    tm.to_parquet(DST / "test_motor.parquet")
    c = pd.read_parquet(DST / "policies_contents.parquet")
    c["customer_id"] = c["customer_id"].astype("string")
    s = rng.choice(len(c), PLANTED["P5"], replace=False)
    c.loc[c.index[s], "customer_id"] = c["customer_id"].iloc[s].str.lstrip("0")
    c.to_parquet(DST / "policies_contents.parquet")
    cl = pd.read_parquet(DST / "claims.parquet")
    kind = cl["report_date"].dtype
    cl["report_date"], cl["loss_date"] = pd.to_datetime(cl["report_date"]), pd.to_datetime(cl["loss_date"])
    q = rng.choice(len(cl), PLANTED["P6"], replace=False)
    cl.loc[cl.index[q], "report_date"] = cl["loss_date"].iloc[q].to_numpy() - np.timedelta64(30, "D")
    if not pd.api.types.is_datetime64_any_dtype(kind):          # keep the raw file's own date format
        cl["report_date"], cl["loss_date"] = cl["report_date"].dt.date, cl["loss_date"].dt.date
    cl.to_parquet(DST / "claims.parquet")
    print("written", DST)


if __name__ == "__main__":
    run()
