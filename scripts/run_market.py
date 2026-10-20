"""One command for a market, from raw files to a submission (Part 2 kit, decisions/part2_kit_design.md section 4).

  MARKET=<name> .venv/Scripts/python.exe -I scripts/run_market.py [--from STAGE] [--to STAGE]

The market's values come from config/<name>.json and its thresholds file; nothing runs on a threshold that is not
written and dated. Stages, in order (each runs as its own process with MARKET set, and its log is kept):
  lock          lock the market's hold-out (share and seed from its thresholds file), unless one is locked already
  build         data rules, targets and reporting completion (src/build.py)
  what_changed  the new market against Part 1, and its training rows against its test rows
  history       point-in-time claims history and the credibility constant (scripts/phase5_step4.py, prepare only)
  weights       fast mode: each family with Part 1's tuned settings, fitted out of time, and the family weights
  level         the W2 GLM and the decided level chain (scripts/submit_w2.py)
  blend         the blend of families priced for the target year (scripts/freeze_build.py)
  referral      the referral rule (scripts/phase6_referral.py, scripts/apply_referral.py)
The full step 7 tuning is a separate run (scripts/phase5_step7.py) when the actuary chooses to spend the time.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-I", "-u"]
STAGES = ["lock", "build", "what_changed", "history", "weights", "level", "blend", "referral"]


def commands(market: str) -> dict[str, list[list[str]]]:
    code = lambda s: PY + ["-c", "import sys; sys.path.insert(0, '.'); " + s]
    return {
        "lock": [code("from scripts.run_market import lock; lock()")],
        "build": [code("from src import build; build.build_all(rebuild=True); print('built')")],
        "what_changed": [PY + ["scripts/what_changed.py", "--new", "{raw}", "--name", market]],
        "history": [code("import json; import scripts.phase5_step4 as s4; ks = {n: s4.prepare(n) for n in s4.PLANS}; "
                         "(s4.WORK / 'k.json').write_text(json.dumps(ks, indent=1))")],
        "weights": [PY + ["scripts/fast_weights.py"]],
        "level": [PY + ["scripts/submit_w2.py"]],
        "blend": [PY + ["scripts/freeze_build.py"]],
        "referral": [PY + ["scripts/phase6_referral.py"], PY + ["scripts/apply_referral.py"]],
    }


def lock() -> None:
    """Lock the market's hold-out once, as Part 1 did (scripts/lock_holdout.py), into the market's own folder."""
    import numpy as np
    import pandas as pd
    from src import build, data_rules as dr, validation as v
    from src.thresholds import load_thresholds
    if build.LOCK.exists():
        print("hold-out already locked:", build.LOCK); return
    th = load_thresholds()
    ids = []
    for line in dr.LINES:
        p = dr.read_raw(f"policies_{line}")
        cut = dr.sum_insured_cutoff(p.sum_insured, th["si_unit_multiple"]) if line != "motor" else None
        ids.append(dr.apply_policy_rules(p, line, cut)["customer_id"])
    cust = np.array(sorted(pd.concat(ids).unique()))
    build.LOCK.parent.mkdir(parents=True, exist_ok=True)
    lk = v.lock_holdout(cust, share=th["holdout_share"], path=str(build.LOCK), seed=int(th["holdout_seed"]))
    print("locked", len(lk["holdout_ids"]), "of", len(cust), "customers into", build.LOCK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STAGES[0], choices=STAGES)
    ap.add_argument("--to", dest="stop", default=STAGES[-1], choices=STAGES)
    a = ap.parse_args()
    market = os.environ.get("MARKET", "part1")
    sys.path.insert(0, str(ROOT))
    from src import market as mk
    logs = mk.WORK / "run_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MARKET=market)
    todo = STAGES[STAGES.index(a.start): STAGES.index(a.stop) + 1]
    if market == "part1" and "what_changed" in todo:
        todo.remove("what_changed")
    for stage in todo:
        for i, cmd in enumerate(commands(market)[stage]):
            cmd = [c.replace("{raw}", str(mk.RAW)) for c in cmd]
            t = time.time()
            log = logs / f"{stage}{i or ''}.log"
            with open(log, "w", encoding="utf-8") as fh:
                r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            print(f"{stage:13s} {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t:5.0f} s  (log {log})", flush=True)
            if r.returncode != 0:
                sys.exit(f"stopped at {stage}: see {log}")
    print("done:", mk.OUT / "submissions")


if __name__ == "__main__":
    main()
