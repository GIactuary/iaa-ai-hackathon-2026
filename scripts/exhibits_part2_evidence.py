"""Evidence for the Part 2 write-up's fairness and weather appendices: the two sets of numbers the model's own
evidence files do not record (written for the documentation, 29 October; no model is changed or chosen here).

Gender (Appendix B). The motor price is the Part 1 shape (no gender: Part 1 carries none) times a one-factor gender
layer fitted on emerging rows (P2-S (a)). This reruns the motor shape of the final file exactly as
`scripts/part2_model_v2.py` builds it (every emerging row, `--all`; motor priced by peril with the premium term, K16,
register H-MP (a)), and records:
  - the layer's relativities, and a 95% interval for men against women from 2,000 customer resamples (layer refitted
    on each resample, the Part 1 shape held fixed);
  - actual over expected by gender on the emerging rows, before and after the layer (what the layer corrects);
  - for the 2025 book, the price as built, for the other gender with everything else the same, and without gender
    (the layer removed and the motor total held, so the price is revenue-neutral);
  - a check that the shape rerun here is the one in the submission file: within new and renewal business, the file's
    prediction over (shape x own-year weather x exposure) must be one constant.
Weather (Appendix C). For each weather response in the final file: the spread of the row-year deviations it was
fitted on, the peril's share of the line's capped cost, and the 2025 book's average deviation.

Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/exhibits_part2_evidence.py
Writes outputs/markets/part2/part2_doc_evidence.json. Reads the submission files (not committed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import scripts.part2_model_v2 as m2
from scripts.part2_candidates import WIDE_WEATHER, Line, elasticities
from scripts.part2_model import developed
from src import build, data_rules as dr, market, weather as wx
from src.evidence import write_evidence
from src.glm import TweedieGLM, cat

ROOT = Path(__file__).resolve().parents[1]
OUT = market.OUT
FILES = {"final": "part2_v2_nb-pp_wx-row_mprem_mperil_bchk_all"}     # H-MP (a), H-NS (b)
N_BOOT, BOOT_SEED = 2000, 0


class Recorded(TweedieGLM):
    """TweedieGLM that keeps the last fitted instance: in the motor branch of `shapes` the last fit is the gender layer."""
    last = None

    def fit(self, *a, **k):
        out = super().fit(*a, **k)
        Recorded.last = self
        return out


def other(g: pd.Series) -> pd.Series:
    return g.map({"M": "F", "F": "M"}).fillna(g)


def gender_evidence(variant: str, L: Line, tr: pd.DataFrame, te: pd.DataFrame, w_em: pd.DataFrame, ref_ids: list) -> dict:
    m2.MOTOR_PREMIUM, m2.MOTOR_PERIL, m2.BUILDINGS_CHECK = True, True, True
    fit = np.ones(len(tr), bool)
    r_tr, r_te = m2.shapes(L, te, w_em, fit)
    layer = Recorded.last
    rel = layer.relativities()
    g_tr, g_te = cat(tr["gender"]), cat(te["gender"])
    lay_tr = layer.rate(pd.DataFrame({"gender": g_tr}))
    lay_te = layer.rate(pd.DataFrame({"gender": g_te}))
    lay_te_other = layer.rate(pd.DataFrame({"gender": other(g_te)}))
    base_tr = r_tr / lay_tr

    # the shape rerun here is the one in the file: prediction / (shape x weather x exposure) constant within NB status
    b = elasticities(L, fit, WIDE_WEATHER["motor"])
    norms = wx.normals(w_em, market.FIT_YEARS)
    cc = {p: float(tr[f"costc_{p}"].sum()) for p in L.perils}
    shares = {p: cc[p] / sum(cc.values()) for p in L.perils}
    yf_te = wx.year_factor(wx.attach(te, w_em, norms), "motor", b, shares)
    sub = pd.read_csv(OUT / "submissions" / f"{FILES[variant]}.csv", dtype={"row_id": str}).set_index("row_id")["prediction"]
    pred = sub.reindex(te["row_id"].astype(str)).to_numpy(float)
    k = pred / (r_te * yf_te * te["exposure"].to_numpy(float))
    nb = te["new_business"].astype(bool).to_numpy()
    check = {s: dict(mean=float(k[m].mean()), max_rel_dev=float(np.max(np.abs(k[m] / k[m].mean() - 1))))
             for s, m in (("new_business", nb), ("renewal", ~nb))}

    # actual over expected by gender on the emerging rows (capped cost, each line at its own total), before and after
    ec = L.e * tr["completion"].to_numpy(float)
    yf_tr = L.year_factor(b, fit)
    y = tr["costc"].to_numpy(float)
    ae = {}
    for name, rate in (("without_layer", base_tr), ("with_layer", r_tr)):
        mu = rate * yf_tr * ec
        mu = mu * y.sum() / mu.sum()
        ae[name] = {g: float(y[(g_tr == g).to_numpy()].sum() / mu[(g_tr == g).to_numpy()].sum()) for g in sorted(g_tr.unique())}

    # 95% interval for M against F: the layer refitted on customer resamples, shape held fixed
    off = base_tr * ec
    cust = tr["customer_id"].astype(str).to_numpy()
    uniq, inv = np.unique(cust, return_inverse=True)
    rows_of = pd.Series(np.arange(len(tr))).groupby(inv).apply(np.array).to_list()
    rng = np.random.default_rng(BOOT_SEED)
    fM = pd.DataFrame({"gender": pd.array(["M"], dtype="string")})
    fF = pd.DataFrame({"gender": pd.array(["F"], dtype="string")})
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(uniq), len(uniq))])
        m = TweedieGLM().fit(pd.DataFrame({"gender": g_tr.to_numpy()[idx]}), y[idx], off[idx])
        draws.append(float(m.rate(fM)[0] / m.rate(fF)[0]))
    mf = float(layer.rate(fM)[0] / layer.rate(fF)[0])

    # the 2025 book: as built, the other gender, and without gender at the same motor total
    e25 = te["exposure"].to_numpy(float)
    neutral = pred / lay_te
    neutral *= pred.sum() / neutral.sum()
    flipped = pred / lay_te * lay_te_other
    book = {}
    for g in sorted(g_te.unique()):
        m = (g_te == g).to_numpy()
        book[g] = dict(rows=int(m.sum()), exposure=float(e25[m].sum()),
                       annual_price=float(pred[m].sum() / e25[m].sum()),
                       annual_price_without_gender=float(neutral[m].sum() / e25[m].sum()))
    refs = []
    ids = te["row_id"].astype(str).to_numpy()
    for rid in ref_ids:
        i = int(np.flatnonzero(ids == rid)[0])
        refs.append(dict(row_id=rid, gender=str(g_te.iloc[i]), exposure=float(e25[i]),
                         annual_price=float(pred[i] / e25[i]), annual_price_other_gender=float(flipped[i] / e25[i]),
                         annual_price_without_gender=float(neutral[i] / e25[i])))
    return dict(file=FILES[variant], layer=rel.to_dict("records"), men_over_women=mf,
                men_over_women_ci=[float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                n_boot=N_BOOT, boot_seed=BOOT_SEED, actual_over_expected=ae,
                fitting_rows=dict(rows=len(tr), by_gender={g: int((g_tr == g).sum()) for g in sorted(g_tr.unique())}),
                book_2025=book, neutral_scale=float(pred.sum() / (pred / lay_te).sum()), reference_policies=refs,
                file_check=check)


def weather_evidence(data: dict, w_em: pd.DataFrame) -> dict:
    final = json.loads((OUT / f"{FILES['final']}.json").read_text(encoding="utf-8"))[FILES["final"]]["by_line"]
    norms = wx.normals(w_em, market.FIT_YEARS)
    out = {}
    for line in dr.LINES:
        tr, te = data[line]
        trx = wx.normals_leave_own_year_out(tr, w_em, market.FIT_YEARS, np.zeros(len(tr), bool))
        tex = wx.attach(te, w_em, norms)
        perils = sorted(c[6:] for c in tr.columns if c.startswith("costc_") and c not in ("costc_large", "costc_other"))
        cc = {p: float(tr[f"costc_{p}"].sum()) for p in perils}
        rows = []
        for peril, (wv, coef) in final[line]["weather"]["elasticities"].items():
            d = trx[f"dev_{wv}"].dropna()
            dt = tex[f"dev_{wv}"]
            rows.append(dict(peril=peril, measure=wv, coefficient=coef, share_of_line_cost=cc[peril] / sum(cc.values()),
                             dev_sd_fitting=float(d.std()), dev_p05_p95_fitting=[float(d.quantile(0.05)), float(d.quantile(0.95))],
                             dev_mean_2025=float(np.average(dt.fillna(0.0), weights=te["exposure"])),
                             normal_mean_2025=float(np.average(tex[f"norm_{wv}"].fillna(tex[f"norm_{wv}"].mean()), weights=te["exposure"]))))
        out[line] = dict(responses=rows, mode="one factor per line" if line == "buildings" else "row",
                         factor_2025_mean=final[line]["weather"]["factor_2025_mean"],
                         factor_2025_p05_p95=final[line]["weather"]["factor_2025_p05_p95"])
    return out


def main() -> None:
    m2.TweedieGLM = Recorded
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", ROOT / "data" / "raw_part2" / "developed")
    ref = dr.read_raw("reference_policies")
    tr, te = data["motor"]
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    L = Line("motor", tr, developed("motor"), w_em, w_dv)
    ref_ids = [r for r in ref["row_id"].astype(str) if r.startswith("M")]
    res = {"gender": {}, "weather": weather_evidence(data, w_em)}
    for variant in FILES:
        res["gender"][variant] = gender_evidence(variant, L, tr, te, w_em, ref_ids)
        g = res["gender"][variant]
        print(variant, f"M/F {g['men_over_women']:.4f} [{g['men_over_women_ci'][0]:.4f}, {g['men_over_women_ci'][1]:.4f}]",
              "check", g["file_check"], flush=True)
    write_evidence({"part2_doc_evidence": res}, OUT / "part2_doc_evidence.json", seeds=[BOOT_SEED])
    print("written", OUT / "part2_doc_evidence.json")


if __name__ == "__main__":
    main()
