"""Step 7 amendment (decisions/phase5_step7_design.md section 8): the EBM per peril in buildings and contents.

Adds a column FS:ebm to the step 7 out-of-time frames of each property line (scorings A and B), built exactly as the
motor per-peril EBM (scripts/phase5_step7.freq_sev), and tests it against the per-line EBM and the W2 GLM.
Run: .venv/Scripts/python.exe -I scripts/phase5_step7b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import scripts.phase5_step7 as s7
from scripts.phase5_step6 import test
from src.evidence import write_evidence
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "work" / "phase5_step7"
LINES = ("contents", "buildings")


def run() -> None:
    th = load_thresholds()
    res = {}
    for line in LINES:
        tuned = json.loads((WORK / f"result_{line}.json").read_text())["tuned"]
        fr = {}
        for name in ("A", "B"):
            path = WORK / f"{line}_{name}.parquet"
            f = pd.read_parquet(path)
            if "FS:ebm" not in f.columns:
                D = s7.load(line, name); D["line"] = line
                y = D["tr"].loc[D["sc"], "cost"].to_numpy()
                assert np.allclose(y, f["y"].to_numpy()), f"{line} {name}: scored rows are not in the frame's order"
                f["FS:ebm"] = s7.freq_sev("ebm", th, D, tuned, name)
                f.to_parquet(path)
                s7.log(line, name, "per-peril EBM added")
            fr[name] = f
        res[line] = {"FS:ebm vs cand:ebm": test(fr, "cand:ebm", "FS:ebm", th, lines=[line]),
                     "FS:ebm vs W2 GLM": test(fr, "ref", "FS:ebm", th, lines=[line])}
        for k, t in res[line].items():
            x = t[line]; a = x["A"]
            print(f"{line} {k}: A {a['gain']:,.0f} [{a['boot']['lo']:,.0f}, {a['boot']['hi']:,.0f}], B {x['B']['gain']:,.0f}, keep {x['verdict']['keep']}", flush=True)
    write_evidence({"phase5_step7b": res}, ROOT / "outputs" / "phase5_step7b.json")


if __name__ == "__main__":
    run()
