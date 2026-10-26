"""Data rule R10 (Y3) measured on one market's raw files: the pattern check, the recovery rate by peril, and the change
in net cost by line and accident year on claims matched to training rows (decisions/data_rules.md R10).

The change in each line's target-year indication needs the market's dated caps and trend bounds, so it runs in the
market's own build (MARKET=<name>, src/build.py), not here. On Part 1 this script reproduces the post-mortem's PM5
(outputs/postmortem.json), which is the check that the rule does what PM5 measured.
Run: .venv/Scripts/python.exe -I scripts/r10_recoveries.py --raw data/raw_part2/emerging --years 2023 2024
     --thresholds decisions/thresholds_part2.md --name part2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import data_rules as dr
from src.evidence import write_evidence
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]


def measure(raw: Path, years: tuple, th_file: Path) -> dict:
    th = load_thresholds(th_file)
    claims = dr.read_raw("claims", raw)
    adj, rep = dr.add_open_recoveries(claims, years, th["rec_open_max_share"], th["rec_min_closed"])
    by_line = {}
    for line in dr.LINES:
        rows = dr.read_raw(f"policies_{line}", raw)
        before = dr.match_claims_to_rows(rows, claims, line)
        after = dr.match_claims_to_rows(rows, adj, line)
        before, after = before[~before["repudiated"]], after[~after["repudiated"]]
        ay = lambda m: m.groupby(m["loss_date"].dt.year)["net_cost"].sum()
        t = pd.DataFrame({"net_recorded": ay(before), "net_adjusted": ay(after)})
        t["expected_recoveries"] = t["net_recorded"] - t["net_adjusted"]
        t["overstatement"] = t["expected_recoveries"] / t["net_adjusted"]
        op = before[before["status"] == "open"]
        t["open_share_of_incurred"] = op.groupby(op["loss_date"].dt.year)["incurred_to_date"].sum() / before.groupby(before["loss_date"].dt.year)["incurred_to_date"].sum()
        by_line[line] = t.fillna(0.0).reset_index(names="accident_year").to_dict("records")
    return dict(report=rep, by_line=by_line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    res = measure(Path(a.raw), tuple(a.years), Path(a.thresholds))
    write_evidence({"r10_recoveries": res}, ROOT / "outputs" / f"r10_recoveries_{a.name}.json")
    rep = res["report"]
    print(a.name, "applied" if rep.get("applied") else f"not applied: {rep.get('reason')}",
          f"open with recovery {rep['open_with_recovery']} of {rep['open_claims']}; thin perils {rep.get('not_applied_thin')}")
    for k, r in rep["rates"].items():
        if r["recoveries"] > 0:
            print(f"  {k}: closed {int(r['closed'])}, with recovery {int(r['with_recovery'])}, rate {r['rate']:.4f}, applied {bool(r['applied'])}")
    for line, recs in res["by_line"].items():
        print(f"  {line}: " + "; ".join(f"{int(r['accident_year'])} {r['overstatement']:+.4%} (open {r['open_share_of_incurred']:.1%})" for r in recs))
