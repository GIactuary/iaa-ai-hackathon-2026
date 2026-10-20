"""The market being priced (Part 2 kit, decisions/part2_kit_design.md section 3).

Every value the Part 1 code fixed is read from config/<market>.json, chosen by the environment variable MARKET
(default part1). config/part1.json reproduces Part 1 exactly, and Part 1 keeps its own folders (data/work, outputs);
any other market works in data/work/markets/<name> and outputs/markets/<name>, so nothing of Part 1 is overwritten.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAME = os.environ.get("MARKET", "part1")
CFG = json.loads((ROOT / "config" / f"{NAME}.json").read_text(encoding="utf-8"))
assert CFG["name"] == NAME, f"config/{NAME}.json names itself {CFG['name']}"

RAW = ROOT / CFG["raw_dir"]
WORK = ROOT / "data" / "work" if NAME == "part1" else ROOT / "data" / "work" / "markets" / NAME
OUT = ROOT / "outputs" if NAME == "part1" else ROOT / "outputs" / "markets" / NAME
THRESHOLDS = ROOT / CFG["thresholds_file"]
LINES = tuple(CFG["lines"])
VALUATION = pd.Timestamp(CFG["valuation"])
GRID_START = pd.Timestamp(CFG["grid_start"])
CURVE_YEAR = int(CFG["curve_year"])
SHARE_YEARS = tuple(CFG["share_years"])
FIT_YEARS = tuple(CFG["fit_years"])
TARGET_YEAR = int(CFG["target_year"])
SCORINGS = {k: dict(fit_years=tuple(v["fit_years"]), score_year=int(v["score_year"]), completion=bool(v["completion"]))
            for k, v in CFG["scorings"].items()}
LARGE_LOSS_PERILS = {l: tuple(p) for l, p in CFG["large_loss_perils"].items()}
STEP_PERILS = {(l, p): int(y) for l, p, y in CFG["step_perils"]}
WEATHER_ADJUSTED = CFG["weather_adjusted"]
SIZE_CHECKS = {l: [(m, tuple(s)) for m, s in v] for l, v in CFG["size_checks"].items()}
if NAME != "part1":
    WORK.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(parents=True, exist_ok=True)
