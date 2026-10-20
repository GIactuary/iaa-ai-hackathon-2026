"""Apply the phase 6 referral rule to the frozen file (check-in 6, P1 (a) and P2 (a)).

- Motor rows where a family with positive weight prices more than refer_disagreement away from the blend (R3) take the
  W2 GLM's price (w2_glm_v1u, already at the decided 2025 level); the motor line is then rescaled to its decided total.
- Buildings and contents keep the blend on R3 rows; R1 and R2 rows keep the blend in every line.
- Every referred row (R1, R2, R3) is listed for underwriting review in data/work/phase6/referrals_2025.csv
  (written by scripts/phase6_referral.py, which must run first).
Run: .venv/Scripts/python.exe -I scripts/apply_referral.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.phase6_referral import disagreement
from src import data_rules as dr
from src.evidence import write_evidence
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
import os
ALL = os.environ.get("P7_ALL") == "1"
SRC, NAME = ("blend_final_all_v1", "blend_final_all_v2") if ALL else ("blend_frozen_v1", "blend_frozen_v2")
W2FILE = "w2_glm_v1u_all.csv" if ALL else "w2_glm_v1u.csv"
GLM_FALLBACK = ("motor",)


def run() -> None:
    th = load_thresholds()
    lim = 1 + th["refer_disagreement"]
    sub = pd.read_csv(OUT / "submissions" / f"{SRC}.csv", dtype={"row_id": str})
    w2 = pd.read_csv(OUT / "submissions" / W2FILE, dtype={"row_id": str}).set_index("row_id")["prediction"]
    sub = sub.set_index("row_id")
    info = {}
    for line in GLM_FALLBACK:
        fp = pd.read_parquet(ROOT / "data" / "work" / "freeze" / f"family_preds_{line}{'_all' if ALL else ''}.parquet").set_index("row_id")
        fam = fp[[c for c in fp.columns if c.startswith("fam:")]].to_numpy()
        assert np.allclose(fp["blend"], sub.loc[fp.index, "prediction"], rtol=1e-6, atol=1e-3), "family predictions do not match the frozen file"
        flag = disagreement(fam, fp["blend"].to_numpy()) > lim
        new = fp["blend"].to_numpy().copy()
        new[flag] = w2.loc[fp.index[flag]].to_numpy()
        total = fp["blend"].sum()
        new *= total / new.sum()
        sub.loc[fp.index, "prediction"] = new
        info[line] = dict(rows=int(len(fp)), replaced=int(flag.sum()), share=float(flag.mean()),
                          ratio_on_replaced=np.quantile(new[flag] / fp["blend"].to_numpy()[flag], [0.1, 0.5, 0.9]).tolist(),
                          total=float(total))
        print(f"{line}: {flag.sum():,} rows at the W2 GLM price; median ratio to the blend {np.median(new[flag] / fp['blend'].to_numpy()[flag]):.3f}", flush=True)
    sub = sub.reset_index()
    ss = dr.read_raw("sample_submission")
    sub = ss[["row_id"]].astype(str).merge(sub, on="row_id", how="left", validate="one_to_one")
    assert sub["prediction"].notna().all() and (sub["prediction"] > 0).all() and len(sub) == len(ss)
    path = OUT / "submissions" / f"{NAME}.csv"
    sub[["row_id", "prediction"]].to_csv(path, index=False, float_format="%.4f")
    print("written", path, len(sub))
    write_evidence({"apply_referral": dict(source=SRC, file=str(path.relative_to(ROOT)), by_line=info)}, OUT / f"submission_{NAME}.json")


if __name__ == "__main__":
    run()
