"""New-business credibility standard (N2; decisions/part2_m2_design.md, M2-N re-read, written before this ran).

Within each P2-V split the new-business factor is read off the fitting rows as part2_model_v2.py does, under four
standards (none, pure premium, claim count, full), and applied to the scored rows. Shapes: F and F with motor by peril
(K16). Each standard is compared with the claim-count standard (M2-N (b)), unweighted and at the 2025 mix.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_nb_standard.py   (after scripts/part2_motor_peril.py)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import scripts.part2_candidates as pc
from scripts.part2_model import developed
from scripts.part2_model_v2 import FULL_CRED
from scripts.part2_motor_peril import by_line, final_shape, mix_weights
from src import build, data_rules as dr, market
from src.evidence import write_evidence

OUT = market.OUT / "part2_nb_standard.json"
STANDARDS = ("none", "pp", "count", "full")
FACTORS = {}


def nb_factors(L, mu: np.ndarray, fit: np.ndarray) -> dict:
    """The new-business factor under each standard, from the fitting rows."""
    tr = L.tr
    nb = tr["new_business"].astype(bool).to_numpy()
    y = tr["costc"].to_numpy(float)
    a, b = fit & nb, fit & ~nb
    ae = (y[a].sum() / mu[a].sum()) / (y[b].sum() / mu[b].sum())
    n_nb = float(tr.loc[a, "n_claims"].sum())
    cl = fit & (tr["n_claims"].to_numpy() > 0)
    sev = y[cl] / tr.loc[cl, "n_claims"].to_numpy(float)
    cv2 = float(sev.var() / sev.mean() ** 2)
    z = {"none": 0.0, "pp": min(1.0, np.sqrt(n_nb / (FULL_CRED * (1 + cv2)))), "count": min(1.0, np.sqrt(n_nb / FULL_CRED)), "full": 1.0}
    return dict(ae=float(ae), claims=n_nb, cv2=cv2, z={k: float(x) for k, x in z.items()},
                factor={k: float(1 + x * (ae - 1)) for k, x in z.items()})


def main():
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", pc.ROOT / "data" / "raw_part2" / "developed")
    lines, P1, W = {}, {}, {}
    for line in dr.LINES:
        L = pc.Line(line, data[line][0].reset_index(drop=True), developed(line), w_em, w_dv)
        lines[line] = L
        P1[line] = pickle.loads((pc.WORK / f"h_p1_{line}.pkl").read_bytes())
        W[line] = mix_weights(L.tr, data[line][1], line)
    assert "peril" in P1["motor"]["r3"], "run scripts/part2_motor_peril.py first"
    allr = pd.concat([L.tr[["customer_id", "yr", "n_claims"]] for L in lines.values()])
    cy = allr.groupby(["customer_id", "yr"])["n_claims"].sum()
    for L in lines.values():
        n = pd.Series(cy.reindex(pd.MultiIndex.from_arrays([L.tr["customer_id"], L.tr["yr"] - 1])).to_numpy())
        L.h = np.where(n.isna(), "none", np.where(n >= 2, "2+", n.fillna(0).astype(int).astype(str)))

    base_candidates = pc.candidates

    def with_nb(L, P1_, fit, stage):
        c = base_candidates(L, P1_, fit, stage)
        out = {}
        nb = L.tr["new_business"].astype(bool).to_numpy()
        for shape, mu in (("F", final_shape(c, L.line)), ("F16", c.get("K16_motor_peril", final_shape(c, L.line)))):
            f = nb_factors(L, mu, fit)
            FACTORS.setdefault(f"{L.line}|{shape}", []).append(f)
            for s in STANDARDS:
                out[f"{shape}|{s}"] = mu * np.where(nb, f["factor"][s], 1.0)
        return out

    pc.candidates = with_nb
    S = pc.score(lines, P1, {"incumbent": "K0"})
    pc.candidates = base_candidates
    w = {name: np.concatenate([W[l][(lines[l].tr["yr"].to_numpy() == 2024) if name == "V1" else slice(None)] for l in dr.LINES])
         for name in ("V1", "V2")}
    res = {"vs_count": {}, "vs_count_2025_mix": {}, "d2": pc.d2(S)}
    for shape in ("F", "F16"):
        for s in ("none", "pp", "full"):
            a, b = f"{shape}|count", f"{shape}|{s}"
            res["vs_count"][b] = by_line(S, a, b)
            res["vs_count_2025_mix"][b] = by_line(S, a, b, w)
            for key in ("vs_count", "vs_count_2025_mix"):
                pc.log(key, b, {sc: {n: (round(x[n]["gain"]), round(x[n]["lo"]), round(x[n]["hi"])) for n in ("V1", "V2")} for sc, x in res[key][b].items()})
    # the factors each split read, summarised; and the factors on every row (what a final file would use)
    res["factors_by_split"] = {k: dict(z_pp=[x["z"]["pp"] for x in v], z_count=[x["z"]["count"] for x in v],
                                       ae=[x["ae"] for x in v], cv2=[x["cv2"] for x in v]) for k, v in FACTORS.items()}
    res["factors_all_rows"] = {}
    for line, L in lines.items():
        c = base_candidates(L, P1[line], np.ones(len(L.tr), bool), {"incumbent": "K0"})
        allfit = np.ones(len(L.tr), bool)
        for shape, mu in (("F", final_shape(c, line)), ("F16", c.get("K16_motor_peril", final_shape(c, line)))):
            res["factors_all_rows"][f"{line}|{shape}"] = nb_factors(L, mu, allfit)
            pc.log("all rows", line, shape, res["factors_all_rows"][f"{line}|{shape}"])
    pc.log("D2", {k: {n: round(x, 4) for n, x in d.items()} for k, d in res["d2"].items()})
    write_evidence({"part2_nb_standard": res}, OUT)


if __name__ == "__main__":
    main()
