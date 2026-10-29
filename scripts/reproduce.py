"""One command per part: rebuild a final submission file from the raw competition files and check it byte for byte.

  .venv/Scripts/python.exe -I scripts/reproduce.py --part 1
  .venv/Scripts/python.exe -I scripts/reproduce.py --part 2

Each stage runs as its own process with its log in data/work/reproduce/. At the end the file's SHA-256 is compared
with the hash of the file the actuary uploaded, recorded in outputs/submission_hashes.json.

Part 1, the final file `blend_final_all_v2.csv` (decisions/submissions.md, U2): lock the hold-out, apply the data
rules, build claims history on the non-hold-out customers and on every customer (phase 7), then the W2 GLM with the decided level chain, the blend of
model families and the referral rule, each refitted on every customer (P7_ALL=1). The families use the tuned
settings recorded in outputs/phase5_step7.json; the tuning search itself (scripts/phase5_step7.py) takes days and is
not rerun.
Part 2, the final file `part2_v2_nb-cred_wx-row_mprem_mperil_bchk_all.csv`: Part 1's built rows (the risk shape is
fitted on them), then the emerging hold-out lock, data rules and the final model (scripts/part2_model_v2.py): motor
priced by peril with the premium term (H-MP (a)), buildings at the check file's shape (H-F (b)), the new-business
factor at the claim-count credibility standard (H-NS (a)), every emerging row.

The raw files go in data/raw/ (Part 1) and data/raw_part2/ (Part 2), as downloaded (README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-I", "-u"]
LOGS = ROOT / "data" / "work" / "reproduce"
HASHES = ROOT / "outputs" / "submission_hashes.json"
P2_FILE = "part2_v2_nb-cred_wx-row_mprem_mperil_bchk_all"
P2_FLAGS = ["--nb", "cred", "--weather", "row", "--motor-premium", "--motor-peril", "--buildings-check", "--all"]


def code(s: str) -> list[str]:
    return PY + ["-c", "import sys; sys.path.insert(0, '.'); " + s]


def restore_tuned() -> list[str]:
    """The step 7 tuned settings, from the committed evidence file into the cache freeze_build reads."""
    return code("import json; from src import market; "
                "d = json.loads((market.OUT / 'phase5_step7.json').read_text())['phase5_step7']['by_line']; "
                "w = market.WORK / 'phase5_step7'; w.mkdir(parents=True, exist_ok=True); "
                "[(w / f'result_{l}.json').write_text(json.dumps({'tuned': d[l]['tuned']})) for l in d]; print('restored', list(d))")


def stages(part: int) -> list[tuple[str, dict, list[str]]]:
    history = ("import json; import scripts.phase5_step4 as s4; ks = {n: s4.prepare(n) for n in s4.PLANS}; "
               "(s4.WORK / 'k.json').write_text(json.dumps(ks, indent=1))")
    p1_base = [("part1_lock", {"MARKET": "part1"}, code("from scripts.run_market import lock; lock()")),
               ("part1_build", {"MARKET": "part1"}, code("from src import build; build.build_all(rebuild=True); print('built')"))]
    if part == 1:
        all_ = {"MARKET": "part1", "P7_ALL": "1", "P5_HOLDOUT": "include"}
        return p1_base + [("part1_history", {"MARKET": "part1"}, code(history)),      # the W2 loader reads both histories
                          ("part1_history_all", all_, code(history)),
                          ("part1_tuned_settings", {"MARKET": "part1"}, restore_tuned()),
                          ("part1_w2", all_, PY + ["scripts/submit_w2.py"]),
                          ("part1_blend", all_, PY + ["scripts/freeze_build.py"]),
                          ("part1_referral", all_, PY + ["scripts/apply_referral.py"])]
    return p1_base + [("part2_lock", {"MARKET": "part2"}, code("from scripts.run_market import lock; lock()")),
                      ("part2_build", {"MARKET": "part2"}, code("from src import build; build.build_all(rebuild=True); print('built')")),
                      ("part2_model", {"MARKET": "part2"}, PY + ["scripts/part2_model_v2.py", *P2_FLAGS])]


def target(part: int) -> Path:
    if part == 1:
        return ROOT / "outputs" / "submissions" / "blend_final_all_v2.csv"
    return ROOT / "outputs" / "markets" / "part2" / "submissions" / f"{P2_FILE}.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2], required=True)
    ap.add_argument("--from", dest="start", default=None, help="skip the stages before this one")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    for d in (ROOT / "outputs" / "submissions", ROOT / "outputs" / "markets" / "part2" / "submissions"):
        d.mkdir(parents=True, exist_ok=True)           # not in a fresh checkout: submission files are not committed
    todo = stages(a.part)
    if a.start:
        todo = todo[[s[0] for s in todo].index(a.start):]
    for name, env, cmd in todo:
        t = time.time()
        log = LOGS / f"{name}.log"
        with open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, stdout=fh, stderr=subprocess.STDOUT)
        print(f"{name:22s} {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t:6.0f} s  (log {log.relative_to(ROOT)})", flush=True)
        if r.returncode != 0:
            sys.exit(f"stopped at {name}")
    out = target(a.part)
    want = json.loads(HASHES.read_text(encoding="utf-8"))[out.name]["sha256"]
    got = sha256(out)
    print(f"{out.relative_to(ROOT)}: sha256 {got}")
    print("matches the uploaded file byte for byte" if got == want else f"DIFFERS from the uploaded file ({want})")
    sys.exit(0 if got == want else 1)


if __name__ == "__main__":
    main()
