"""Phase 6: referral, treatment tests and prediction ranges. Design and thresholds written first: decisions/phase6_design.md.

  R1  size beyond 1.25 x the segment's training maximum (phase 2 check, on the frozen model's training rows)
  R2  the size segment holds fewer than refer_min_segment training rows (X3)
  R3  a family with positive weight prices the row more than refer_disagreement away from the blend
Out of time (step 8 cross-fitted frames): is R3 a warning sign, and do treatments T1 (highest family) or T2 (W2 GLM)
beat T0 (keep the blend) on the keep rule? Ranges: lowest to highest family price per 2025 row.
Run: .venv/Scripts/python.exe -I scripts/phase6_referral.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.phase5_step6 import test
from scripts.phase5_step8 import FAMILY_COLS, common
from scripts.submit_w2 import FIT_YEARS
from src import build, data_rules as dr, screens, validation as v
from src.evidence import write_evidence
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT, WORK = ROOT / "outputs", ROOT / "data" / "work" / "phase6"
MIX = {"motor": "arith", "buildings": "geo", "contents": "arith"}


def disagreement(fam: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Largest ratio, either way, between a family's price and the blend's (1 = all agree)."""
    r = fam / blend[:, None]
    return np.maximum(r.max(axis=1), 1 / r.min(axis=1))


def run() -> None:
    th = load_thresholds()
    WORK.mkdir(parents=True, exist_ok=True)
    lim = 1 + th["refer_disagreement"]
    data = build.build_all()
    S8 = json.loads((OUT / "phase5_step8.json").read_text())["phase5_step8"]
    res = {"size": {}, "oot": {}, "ranges": {}, "referrals_2025": {}}
    ref_rows = []
    # R1 and R2 on the 2025 rows, against the frozen model's training rows
    for line in dr.LINES:
        tr, te = data[line]
        tr = tr[~tr["holdout"] & tr["yr"].isin(FIT_YEARS)]
        for measure, seg in screens.SIZE_CHECKS[line]:
            r1 = screens.size_referrals(tr, te, measure, seg, th["refer_size_ratio"])
            for _, r in r1.iterrows():
                ref_rows.append(dict(row_id=str(r["row_id"]), line=line, reason="R1", detail=f"{measure} {r[measure]:,.0f}: {r['ratio_to_max']:.2f} x the segment maximum"))
            if seg:
                n = tr.groupby(list(seg), dropna=False).size().rename("n_train")
                t = te.join(n, on=list(seg))
                thin = t[t["n_train"].fillna(0) < th["refer_min_segment"]]
                for _, r in thin.iterrows():
                    ref_rows.append(dict(row_id=str(r["row_id"]), line=line, reason="R2",
                                         detail=f"{'/'.join(seg)} segment with {int(r['n_train']) if pd.notna(r['n_train']) else 0} training rows"))
            res["size"][f"{line}:{measure}"] = dict(r1=int(len(r1)), r2=int(len(thin)) if seg else 0)
    # R3 on the 2025 rows, and the ranges
    for line in dr.LINES:
        fp = pd.read_parquet(ROOT / "data" / "work" / "freeze" / f"family_preds_{line}.parquet")
        fam = fp[[c for c in fp.columns if c.startswith("fam:")]].to_numpy()
        dis = disagreement(fam, fp["blend"].to_numpy())
        flag = dis > lim
        for rid, d in zip(fp.loc[flag, "row_id"], dis[flag]):
            ref_rows.append(dict(row_id=rid, line=line, reason="R3", detail=f"families up to {d:.2f} x apart from the blend"))
        width = fam.max(axis=1) / fam.min(axis=1)
        res["ranges"][line] = dict(width_quantiles=dict(zip(("q50", "q90", "q99"), np.quantile(width, [0.5, 0.9, 0.99]).tolist())),
                                   blend_inside=float(np.mean((fp["blend"] >= fam.min(axis=1) * (1 - 1e-9)) & (fp["blend"] <= fam.max(axis=1) * (1 + 1e-9)))))
        res["referrals_2025"][line] = dict(rows=int(len(fp)), r3=int(flag.sum()), r3_share=float(flag.mean()))
    # Out of time: R3 as a warning sign, and treatments T1 and T2 against T0
    frames = {}
    for s, wkey in (("A", "fit_B"), ("B", "fit_A")):        # each year scored by weights fitted on the other
        f = pd.read_parquet(ROOT / "data" / "work" / "phase5_step8" / f"blend_{s}.parquet")
        parts = []
        for line in dr.LINES:
            d = f[f["line"] == line].copy()
            y = d["y"].to_numpy()
            w = S8["weights"][line][f"families|{MIX[line]}"]
            cols = [c for c, x in zip(w["cols"], w[wkey]) if x > 1e-9]
            blend = d[f"blend:families|{MIX[line]}"].to_numpy()
            fam = common(d[cols].to_numpy(), y)
            dis = disagreement(fam, blend)
            flag = dis > lim
            d["T0"] = blend
            d["T1"] = np.where(flag, fam.max(axis=1), blend)
            d["T2"] = np.where(flag, common(d[["ref"]].to_numpy(), y)[:, 0], blend)
            d["flag"] = flag
            mu = v.rescale_to_total(blend, y)
            dev = v.tweedie_unit_deviance(y, mu)
            res["oot"].setdefault(line, {})[s] = dict(
                flagged=int(flag.sum()), share=float(flag.mean()),
                dev_per_cost_flagged=float(dev[flag].sum() / mu[flag].sum()) if flag.any() else None,
                dev_per_cost_rest=float(dev[~flag].sum() / mu[~flag].sum()),
                actual_over_expected_flagged=float(y[flag].sum() / mu[flag].sum()) if flag.any() else None)
            parts.append(d)
        frames[s] = pd.concat(parts, ignore_index=True)
    res["tests"] = {"T1 vs T0": test(frames, "T0", "T1", th), "T2 vs T0": test(frames, "T0", "T2", th)}
    pd.DataFrame(ref_rows).to_csv(WORK / "referrals_2025.csv", index=False)
    rr = pd.DataFrame(ref_rows)
    res["referral_counts"] = rr.groupby(["line", "reason"]).size().rename("rows").reset_index().to_dict("records") if len(rr) else []
    res["referred_rows_distinct"] = int(rr["row_id"].nunique()) if len(rr) else 0
    for k, t in res["tests"].items():
        print(k)
        for ln, x in t.items():
            a = x["A"]; print(f"   {ln:9s} A {a['gain']:9.0f} [{a['boot']['lo']:9.0f}, {a['boot']['hi']:9.0f}] B {x['B']['gain']:9.0f} keep {x['verdict']['keep']}")
    print(json.dumps({k: res[k] for k in ("size", "referrals_2025", "oot", "ranges", "referral_counts")}, indent=1, default=float))
    write_evidence({"phase6": res}, OUT / "phase6.json")


if __name__ == "__main__":
    run()
