"""Fast mode for a new market (Part 2 kit, decisions/part2_kit_design.md section 4): the step 7 and step 8 work in
about an hour instead of six.

Each family of the frozen structure (freeze_build.COLMAP) keeps Part 1's tuned setting and tree count; it is fitted on
each scoring's fitting years and prices the scored year (non-hold-out rows), and the family weights are refitted by the
step 8 method (pooled over the scorings, each year's prices first rescaled to its actual total). The weights are
written where scripts/freeze_build.py reads them. Part 1's tuned settings are copied into the market's folder only if
the market has none of its own (a full tuning run writes its own).
Run: MARKET=<name> .venv/Scripts/python.exe -I scripts/fast_weights.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.freeze_build import COLMAP, MIX, family_predictions
from scripts.phase5_step8 import common, fit_weights
from src import build, market, validation as v
from src.evidence import write_evidence
from src.thresholds import load_thresholds

PART1_STEP7 = market.ROOT / "data" / "work" / "phase5_step7"


def run() -> None:
    th = load_thresholds()
    w7 = market.WORK / "phase5_step7"
    w7.mkdir(parents=True, exist_ok=True)
    for line in market.LINES:
        if not (w7 / f"result_{line}.json").exists():
            shutil.copy(PART1_STEP7 / f"result_{line}.json", w7 / f"result_{line}.json")
    ks_all = json.loads((market.WORK / "phase5_step4" / "k.json").read_text())
    data = build.build_all()
    weights, scores = {}, {}
    for line in market.LINES:
        tuned = json.loads((w7 / f"result_{line}.json").read_text())["tuned"]
        cols = list(COLMAP[line])
        blocks = []
        for s, plan in market.SCORINGS.items():
            ks = ks_all[s]
            for r in ks.values():
                r["k"] = r["k_poisson_gamma"]
            h = pd.read_parquet(market.WORK / "phase5_step4" / f"history_{s}.parquet")
            h = h[h["line"] == line].drop(columns="line").drop_duplicates("row_id").set_index("row_id")
            full = data[line][0]
            tr = full[~full["holdout"] & full["yr"].isin(plan["fit_years"])].reset_index(drop=True)
            te0 = full[~full["holdout"] & (full["yr"] == plan["score_year"])].reset_index(drop=True)
            preds = family_predictions(line, tr, te0, h, ks, tuned, th, plan["score_year"], max(plan["fit_years"]))
            cl = te0["completion"].to_numpy() if plan["completion"] else 1.0
            y = te0["cost"].to_numpy()
            P = np.column_stack([preds[c] * cl for c in cols])
            blocks.append((common(P, y), y, common((preds["ref"] * cl)[:, None], y)[:, 0], te0["exposure"].to_numpy()))
        Pm = np.vstack([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        w = fit_weights(Pm, y, geometric=MIX[line] == "geo")
        weights[line] = {f"families|{MIX[line]}": dict(cols=cols, pooled=w.tolist())}
        # in-sample check of the pooled weights against the W2 GLM, per scoring (apparent, not out of sample)
        for (Pc, yy, ref, e), s in zip(blocks, market.SCORINGS):
            bl = np.exp(np.log(Pc) @ w) if MIX[line] == "geo" else Pc @ w
            bl = bl * yy.sum() / bl.sum()
            scores.setdefault(line, {})[s] = dict(d2_blend_apparent=float(v.d2_tweedie(yy, bl, e)), d2_w2=float(v.d2_tweedie(yy, ref, e)))
        print(line, {c: round(x, 3) for c, x in zip(cols, w)}, scores[line], flush=True)
    write_evidence({"phase5_step8": dict(weights=weights, scores_apparent=scores, mode="fast")}, market.OUT / "phase5_step8.json")


if __name__ == "__main__":
    run()
