"""Check-in 2, L8: the joint drop of the five V1 factors that each improved their line when removed one at a time.
Criterion written first in decisions/phase5_checkin2.md. Run: .venv/Scripts/python.exe -I scripts/phase5_l8_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joblib import Parallel, delayed

from scripts.phase5_step5 import fit_score, load_plan, paired, spec_columns
from src import build, data_rules as dr, validation as v
from src.evidence import write_evidence
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
KEPT = {"motor": ["G1", "M9"], "buildings": ["B9"], "contents": []}
DROP = {"motor": ["body"], "buildings": ["ptype", "beds"], "contents": ["new_bus", "ptype"]}


def run(line):
    th = load_thresholds(); weather = dr.read_raw("weather")
    full = {l: build.build_all()[l][0] for l in dr.LINES}; full = {l: d[~d["holdout"]] for l, d in full.items()}
    out = {}
    for s in v.SCORINGS:
        P = load_plan(line, s, th, full, weather)
        act = dict.fromkeys(KEPT[line])
        with_ = fit_score(P, spec_columns(P, act))[1]
        without = fit_score(P, spec_columns(P, act, DROP[line]))[1]
        r = paired(P, with_, without, s, s == "A")
        out[s] = {k: x for k, x in r.items() if k not in ("dev_a", "dev_b")}
    out["confirmed"] = bool(out["A"]["gain"] > 0 and out["B"]["gain"] > 0)
    a = out["A"]
    print(f"{line}: dropping {DROP[line]} saves A {a['gain']:,.1f} [{a['boot']['lo']:,.1f}, {a['boot']['hi']:,.1f}], B {out['B']['gain']:,.1f}; "
          f"D2 A {a['d2_ref']:.4%} -> {a['d2_cand']:.4%}; confirmed {out['confirmed']}", flush=True)
    return out


if __name__ == "__main__":
    res = dict(zip(dr.LINES, Parallel(n_jobs=3)(delayed(run)(l) for l in dr.LINES)))
    write_evidence({"phase5_l8_check": dict(drop=DROP, kept=KEPT, by_line=res)}, ROOT / "outputs" / "phase5_l8_check.json")
