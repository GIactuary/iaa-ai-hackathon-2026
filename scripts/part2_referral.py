"""Part 2 referral rule on the 2025 rows (decisions/part2_m2_design.md, referral section, written before this ran).

R1 size beyond refer_size_ratio x the segment's training maximum, or an unseen segment; R2 a segment with fewer than
refer_min_segment training rows. Training rows: Part 1's developed rows and every emerging row. R3 does not apply
(one family per line). Treatment (Part 1's Q2 (a)): the model price stands; referred rows are listed for review.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_referral.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.part2_model import developed
from src import build, data_rules as dr, market, screens
from src.evidence import write_evidence
from src.thresholds import load_thresholds

OUT = market.OUT / "part2_referrals.json"
LIST = market.WORK / "referrals_2025.csv"


def main():
    th = load_thresholds()
    data = build.build_all()
    rows, res = [], {}
    for line in dr.LINES:
        em, te = data[line]
        for measure, seg in screens.SIZE_CHECKS[line]:
            cols = [measure, *seg]
            tr = pd.concat([developed(line)[cols], em[cols]], ignore_index=True)
            r1 = screens.size_referrals(tr, te, measure, seg, th["refer_size_ratio"])
            for _, r in r1.iterrows():
                rows.append(dict(row_id=str(r["row_id"]), line=line, reason="R1",
                                 detail=f"{measure} {r[measure]:,.0f}: " + (r["reason"] if pd.isna(r["ratio_to_max"]) else f"{r['ratio_to_max']:.2f} x the segment maximum")))
            n_thin = 0
            if seg:
                n = tr.groupby(list(seg), dropna=False).size().rename("n_train")
                t = te.join(n, on=list(seg))
                thin = t[t["n_train"].fillna(0) < th["refer_min_segment"]]
                n_thin = len(thin)
                for _, r in thin.iterrows():
                    rows.append(dict(row_id=str(r["row_id"]), line=line, reason="R2",
                                     detail=f"{'/'.join(seg)} segment with {int(r['n_train']) if pd.notna(r['n_train']) else 0} training rows"))
            res[f"{line}:{measure}"] = dict(r1=int(len(r1)), r2=int(n_thin), test_rows=int(len(te)), training_rows=int(len(tr)))
            print(line, measure, res[f"{line}:{measure}"], flush=True)
    ref = pd.DataFrame(rows, columns=["row_id", "line", "reason", "detail"])
    ref.to_csv(LIST, index=False)
    res["referred_rows"] = int(ref["row_id"].nunique())
    res["r3"] = "not applicable: one family per line, no blend to disagree with"
    res["treatment"] = "model price stands; rows listed for underwriting review (Part 1 Q2 (a), for the actuary to confirm)"
    print("referred rows", res["referred_rows"])
    write_evidence({"part2_referrals": res}, OUT)


if __name__ == "__main__":
    main()
