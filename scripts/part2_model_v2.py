"""Part 2 improved file: the combination that passed P2-V (K9 plus row-level weather, decisions/part2_m2_design.md),
with the level chain of the check file, and switches for the milestone 2 decisions (register M2-W, M2-N).

Shape by line (each fitted on Part 1 rows unless stated, P2-C (b)):
  motor      Part 1 W2 GLM, times a gender layer fitted on emerging fitting rows (P2-S (a))
  buildings  by peril: Part 1 GLM per peril; fire and flood as frequency x severity with address hazards (wildfire
             normal; river distance and elevation); each peril calibrated to the emerging fitting rows' capped cost
             times its excess factor
  contents   Part 1 LightGBM on the fields both markets carry
  all        each row's own-year weather (responses fitted on emerging fitting rows), --weather line keeps Y4's one
             factor per line instead
Level: 2025 indication per line at normal weather, Part 1's trends (P2-T), excess factor mixed (P2-L), applied to the
shape's 2024 level at normal weather. New business (--nb none | cred | full): the factor of M2-N, with renewals
rebalanced so each line's total on the fitting rows is unchanged.
--buildings-check: buildings at the check file's shape (hold-out option, decisions/part2_m2_design.md).
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_model_v2.py --nb cred --weather row [--motor-premium] [--buildings-check] [--all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.part2_candidates import WIDE_WEATHER, Line, elasticities, hazard, layer, premium_hat
from scripts.part2_model import BASE_YEAR, average_over_unknown, developed, part1_level, w2_factors
from src import build, data_rules as dr, fields, level as lv, market, validation as v, weather as wx
from src.evidence import write_evidence
from src.families import LGB, as_category
from src.glm import TweedieGLM, cat
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT = market.OUT
FULL_CRED = 1082.0          # classical limited-fluctuation standard (M2-N (b))
MOTOR_PREMIUM = False       # set by --motor-premium
BUILDINGS_CHECK = False     # set by --buildings-check: the check file's buildings shape (K0), weather one factor per line


def glm_rate(m, L, F):
    return average_over_unknown(m, F, L.Ft, L.e, [c for c in L.unknown if c in F.columns])


def shapes(L: Line, te: pd.DataFrame, w_em: pd.DataFrame, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Annual uncapped-equivalent rate at normal weather for fitting rows and test rows (arbitrary level)."""
    dv, line = L.dv, L.line
    Fte = w2_factors(te, line, L.edges)
    Fte["yr"] = BASE_YEAR
    ecd = (dv["exposure"] * dv["completion"]).to_numpy(float)
    if line == "motor":
        Fd, Ft = L.Fd, L.Ft
        if MOTOR_PREMIUM:                  # Y1 in motor: passed P2-V on its own (decisions/part2_m2_design.md)
            hd, ht, he = premium_hat(L, [L.tr, te])
            Fd, Ft, Fte = Fd.assign(lwp_hat=hd), Ft.assign(lwp_hat=ht), Fte.assign(lwp_hat=he)
        base = TweedieGLM().fit(Fd, dv["costc"], ecd)
        r_tr, r_te = glm_rate(base, L, Ft), glm_rate(base, L, Fte)
        ec = L.e * L.tr["completion"].to_numpy(float)
        g = cat(L.tr["gender"])
        m = TweedieGLM().fit(pd.DataFrame({"gender": g})[fit], L.tr.loc[fit, "costc"].to_numpy(float), (r_tr * ec)[fit])
        return r_tr * m.rate(pd.DataFrame({"gender": g})), r_te * m.rate(pd.DataFrame({"gender": cat(te["gender"])}))
    if line == "buildings" and BUILDINGS_CHECK:
        m = TweedieGLM().fit(L.Fd, dv["costc"], ecd)
        return glm_rate(m, L, L.Ft), glm_rate(m, L, Fte)
    if line == "buildings":
        Ht = hazard(te, w_em, market.FIT_YEARS)
        r_tr, r_te = np.zeros(len(L.tr)), np.zeros(len(te))
        for p in L.perils:
            ecp_d = (dv["exposure"] * dv[f"completion_{p}"]).to_numpy(float)
            keep = {"fire": ["wf"], "flood": ["river", "elev"]}.get(p)
            if keep:
                Fd_, Ft_, Fe_ = (pd.concat([L.Fd, L.Hd[keep]], axis=1), pd.concat([L.Ft, L.Ht[keep]], axis=1), pd.concat([Fte, Ht[keep]], axis=1))
                freq = TweedieGLM(power=1.0).fit(Fd_, dv[f"n_{p}"], ecp_d)
                has = ((dv[f"n_{p}"] > 0) & (dv[f"cost_{p}"] > 0)).to_numpy()
                n = dv.loc[has, f"n_{p}"].to_numpy(float)
                sev = TweedieGLM(power=2.0).fit(Fd_[has], dv.loc[has, f"cost_{p}"].to_numpy(float), n, weight=n)
                a_tr, a_te = glm_rate(freq, L, Ft_) * glm_rate(sev, L, Ft_), glm_rate(freq, L, Fe_) * glm_rate(sev, L, Fe_)
            else:
                m = TweedieGLM().fit(L.Fd, dv[f"costc_{p}"], ecp_d)
                a_tr, a_te = glm_rate(m, L, L.Ft), glm_rate(m, L, Fte)
            ecp = L.ecp(p)
            target = float(L.tr.loc[fit, f"costc_{p}"].sum()) * L.elf(p, fit)
            k = target / float(np.sum(a_tr[fit] * ecp[fit]))
            r_tr += a_tr * k
            r_te += a_te * k
        return r_tr, r_te
    # contents: Part 1 LightGBM
    drop = ("region", "make_model", "occupation", "yr", "holdout", "gender")
    fe_, fd_ = fields.tree_fields(L.tr, 11), fields.tree_fields(dv, 11)
    cols = [c for c in fe_["num"] + fe_["cat"] if c in fd_["num"] + fd_["cat"] and c not in drop]
    cats = [c for c in cols if c in fe_["cat"]]
    both = as_category(pd.concat([dv[cols], L.tr[cols], te[cols]], ignore_index=True), cats)
    Xd = both.iloc[:len(dv)].reset_index(drop=True)
    Xt = both.iloc[len(dv):len(dv) + len(L.tr)].reset_index(drop=True)
    Xe = both.iloc[len(dv) + len(L.tr):].reset_index(drop=True)
    gd = dv["customer_id"].astype(str).to_numpy()
    yd = dv["costc"].to_numpy(float)
    a, b = v.make_folds(len(dv), [11], n_folds=5, groups=gd)[11][0]
    m = LGB("tweedie", 1.5, 31, 400, 0.05, 3000, 4, seed=11).fit(Xd.iloc[a], yd[a], ecd[a], val=(Xd.iloc[b], yd[b], ecd[b], None, None))
    m = LGB("tweedie", 1.5, 31, 400, 0.05, m.best_iter_, 4, seed=11).fit(Xd, yd, ecd)
    return m.rate(Xt), m.rate(Xe)


def run(nb_mode: str, weather_mode: str, all_rows: bool, motor_premium: bool = False, buildings_check: bool = False) -> None:
    global MOTOR_PREMIUM, BUILDINGS_CHECK
    MOTOR_PREMIUM, BUILDINGS_CHECK = motor_premium, buildings_check
    th = load_thresholds()
    p1 = part1_level()
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", ROOT / "data" / "raw_part2" / "developed")
    norms = wx.normals(w_em, market.FIT_YEARS)
    tag = f"part2_v2_nb-{nb_mode}_wx-{weather_mode}" + ("_mprem" if motor_premium else "") + ("_bchk" if buildings_check else "") + ("_all" if all_rows else "")
    sub, info = [], {}
    for line in dr.LINES:
        tr, te = data[line]
        tr = (tr if all_rows else tr[~tr["holdout"]]).reset_index(drop=True)
        te = te.reset_index(drop=True)
        L = Line(line, tr, developed(line), w_em, w_dv)
        fit = np.ones(len(tr), bool)
        r_tr, r_te = shapes(L, te, w_em, fit)

        # weather: each row's own year (responses fitted on the fitting rows) or Y4's one factor per line
        b = elasticities(L, fit, WIDE_WEATHER[line])
        yf_tr = L.year_factor(b, fit)
        cc = {p: float(tr[f"costc_{p}"].sum()) for p in L.perils}
        shares = {p: cc[p] / sum(cc.values()) for p in L.perils}
        tex = wx.attach(te, w_em, norms)
        yf_te = wx.year_factor(tex, line, b, shares)
        if weather_mode == "line" or (line == "buildings" and buildings_check):
            yf_te = np.full(len(te), float(np.average(yf_te, weights=te["exposure"])))

        # new business (M2-N): actual over model, new against renewal, on capped cost, then credibility
        nb_tr, nb_te = tr["new_business"].astype(bool).to_numpy(), te["new_business"].astype(bool).to_numpy()
        ec = L.e * tr["completion"].to_numpy(float)
        mu = r_tr * yf_tr * ec
        y = tr["costc"].to_numpy(float)
        ae = (y[nb_tr].sum() / mu[nb_tr].sum()) / (y[~nb_tr].sum() / mu[~nb_tr].sum())
        n_nb = float(tr.loc[nb_tr, "n_claims"].sum())
        z = {"none": 0.0, "cred": min(1.0, float(np.sqrt(n_nb / FULL_CRED))), "full": 1.0}[nb_mode]
        rel = 1.0 + z * (ae - 1.0)
        rebal = float(mu.sum() / (mu * np.where(nb_tr, rel, 1.0)).sum())
        nbf_tr, nbf_te = np.where(nb_tr, rel, 1.0) * rebal, np.where(nb_te, rel, 1.0) * rebal

        # level chain (as the check file, on the new shape at normal weather)
        trx = wx.attach(tr, w_em, norms)
        trn = trx.copy()
        for p, (wv, bb) in b.items():
            trn[f"wm_{p}"] = np.exp(bb * trn[f"dev_{wv}"].fillna(0.0))
        trends = lv.peril_trends(trn, line, th["trend_min"], th["trend_max"])
        p1t = {t["peril"]: t for t in p1[line]["final"]["trends_normalised"]}
        for p in trends.index:
            trends.loc[p, "freq_trend"] = float(np.clip(p1t[p]["freq_trend"], th["trend_min"], th["trend_max"]))
            if pd.isna(trends.loc[p, "sev_step_level"]):
                trends.loc[p, "sev_trend"] = float(np.clip(p1t[p]["sev_trend"], th["trend_min"], th["trend_max"]))
        ind = lv.indication(trn, trends, market.TARGET_YEAR)["indicated_capped_lc"]
        if line == "motor":
            elf = float(tr["cost"].sum() / tr["costc"].sum())
        else:
            elf = float(sum(tr[f"costc_{p}"].sum() * L.dev_ratio[p] for p in L.perils) / tr["costc"].sum())
        last = (tr["yr"] == 2024).to_numpy()
        model_lc = float(np.sum(r_tr[last] * nbf_tr[last] * L.e[last]) / L.e[last].sum())
        ob = float(ind.sum() * elf / model_lc)
        pred = r_te * nbf_te * ob * yf_te * te["exposure"].to_numpy(float)
        written = (te["written_premium"] * te["exposure"]).to_numpy(float)
        info[line] = dict(rows=len(te), new_business=dict(actual_over_model=float(ae), claims=n_nb, z=z, factor=float(rel), renewal_rebalance=rebal),
                          weather=dict(mode=weather_mode, elasticities={p: [w, c] for p, (w, c) in b.items()},
                                       factor_2025_mean=float(np.average(yf_te, weights=te["exposure"])),
                                       factor_2025_p05_p95=[float(np.quantile(yf_te, 0.05)), float(np.quantile(yf_te, 0.95))]),
                          excess_factor=elf, indication_capped_lc=float(ind.sum()), off_balance=ob,
                          predicted=float(pred.sum()), written=float(written.sum()), predicted_over_written=float(pred.sum() / written.sum()),
                          mean_per_exposure=float(pred.sum() / te["exposure"].sum()))
        print(f"{line}: NB a/e {ae:.3f} Z {z:.2f} factor {rel:.3f}; weather 2025 mean {info[line]['weather']['factor_2025_mean']:.4f} "
              f"[{info[line]['weather']['factor_2025_p05_p95'][0]:.3f}, {info[line]['weather']['factor_2025_p05_p95'][1]:.3f}]; ELF {elf:.4f}; "
              f"mean per year {info[line]['mean_per_exposure']:.1f}; predicted/written {pred.sum() / written.sum():.3f}", flush=True)
        sub.append(pd.DataFrame({"row_id": te["row_id"].astype(str), "prediction": pred}))
    sub = pd.concat(sub, ignore_index=True)
    ss = dr.read_raw("sample_submission")
    sub = ss[["row_id"]].astype(str).merge(sub, on="row_id", how="left", validate="one_to_one")
    assert sub["prediction"].notna().all() and (sub["prediction"] > 0).all() and len(sub) == len(ss)
    path = OUT / "submissions" / f"{tag}.csv"
    sub[["row_id", "prediction"]].to_csv(path, index=False, float_format="%.4f")
    print("written", path, len(sub))
    write_evidence({tag: dict(file=str(path.relative_to(ROOT)), rows=len(sub), by_line=info)}, OUT / f"{tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", choices=["none", "cred", "full"], default="cred")
    ap.add_argument("--weather", choices=["row", "line"], default="row")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--motor-premium", action="store_true")
    ap.add_argument("--buildings-check", action="store_true")
    a = ap.parse_args()
    run(a.nb, a.weather, a.all, a.motor_premium, a.buildings_check)
