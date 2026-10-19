"""Phase 5 follow-up on K3 and K4 (actuary's notes on check-in 1).

K3: where does the motor C4 gap (indication 5.3% below the model's own projection) come from? Compares, per line,
    the GLM's fitted level against actual capped cost by year (a log-link Tweedie GLM does not reproduce totals), and
    three 2025 levels: the indication, a plain projection of the latest year's own experience, and the model's.
K4: weather-normalised experience (the actuary's proposal). Each row-year's expected claims are scaled by its
    weather: per peril exp(b x deviation from the address normal), line-level 1 + sum(share x (multiplier - 1)).
    Trends, the indication and the GLM are fitted on that normal level; the scored year is then multiplied by its
    actual weather. Tested on A and B against the current method, on level and on ranking.
Run: .venv/Scripts/python.exe -I scripts/phase5_k3k4.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import build, data_rules as dr, level as lv, validation as v, weather as wx
from src.evidence import write_evidence
from src.glm import TweedieGLM, features, fit_edges
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT, WORK = ROOT / "outputs", ROOT / "data" / "work" / "phase5_k3k4"


def run_line(line):
    th = load_thresholds()
    tr0 = build.build_all()[line][0]
    tr0 = tr0[~tr0["holdout"]].reset_index(drop=True)
    if os.environ.get("P5_SAMPLE"):
        tr0 = tr0.sample(frac=float(os.environ["P5_SAMPLE"]), random_state=0).reset_index(drop=True)
    weather = dr.read_raw("weather")
    perils = lv.perils_of(tr0)
    plans = {**{k: dict(fit_years=s["fit_years"], score_year=s["score_year"]) for k, s in v.SCORINGS.items()},
             "final": dict(fit_years=(2021, 2022, 2023, 2024), score_year=None)}
    res = {}
    for name, plan in plans.items():
        fy, sy = plan["fit_years"], plan["score_year"]
        tr = wx.attach(tr0, weather, wx.normals(weather, fy))
        fit = tr["yr"].isin(fy).to_numpy(); last = max(fy); lastm = fit & (tr["yr"] == last).to_numpy()
        target = sy if sy is not None else 2025
        e, c = tr["exposure"].to_numpy(), tr["completion"].to_numpy()
        b = wx.elasticities(tr[fit], line)
        share = {p: float(tr.loc[fit, f"cost_{p}"].sum() / tr.loc[fit, "cost"].sum()) for p in perils}
        yf = wx.year_factor(tr, line, b, share)
        trn = tr.copy()
        for p, (wvar, bb) in b.items():
            trn[f"wm_{p}"] = np.exp(bb * trn[f"dev_{wvar}"].fillna(0.0))
        elf = float(tr.loc[fit, "cost"].sum() / tr.loc[fit, "costc"].sum())
        trends = lv.peril_trends(tr[fit], line, th["trend_min"], th["trend_max"])
        trends_n = lv.peril_trends(trn[fit], line, th["trend_min"], th["trend_max"])
        ind, ind_n = lv.indication(tr[fit], trends, target), lv.indication(trn[fit], trends_n, target)
        edges = fit_edges(tr[fit], line)
        F = features(tr, line, edges)
        m1 = TweedieGLM().fit(F[fit], tr.loc[fit, "costc"], e[fit] * c[fit])
        m1n = TweedieGLM().fit(F[fit], tr.loc[fit, "costc"], e[fit] * c[fit] * yf[fit])

        def at(mask, year):
            G = F[mask].copy(); G["yr"] = str(year); return G

        # balance: fitted capped cost against actual, by year (V1 with its own year factor)
        bal = {}
        for y in fy:
            m = fit & (tr["yr"] == y).to_numpy()
            bal[int(y)] = float((m1.rate(F[m]) * e[m] * c[m]).sum() / tr.loc[m, "costc"].sum())
        gt, gt_n = lv.group_trend(trends, ind["indicated_capped_lc"], perils), lv.group_trend(trends_n, ind_n["indicated_capped_lc"], perils)
        k = target - last
        ind_lc, ind_lc_n = float(ind["indicated_capped_lc"].sum()), float(ind_n["indicated_capped_lc"].sum())
        # plain projection of the latest year's own experience, peril by peril
        emp = 0.0
        for p in perils:
            ecp = (tr.loc[lastm, "exposure"] * tr.loc[lastm, f"completion_{p}"]).sum()
            lc_p = tr.loc[lastm, f"costc_{p}"].sum() / ecp
            if pd.notna(trends.loc[p, "sev_step_level"]):
                lc_p = tr.loc[lastm, f"n_{p}"].sum() / ecp * trends.loc[p, "sev_step_level"]
            emp += lc_p * ((1 + trends.loc[p, "freq_trend"]) * (1 + trends.loc[p, "sev_trend"])) ** k
        r_last, r_last_n = m1.rate(at(lastm, last)), m1n.rate(at(lastm, last))
        model_lc = float(np.sum(r_last * e[lastm]) / e[lastm].sum())
        model_lc_n = float(np.sum(r_last_n * e[lastm]) / e[lastm].sum())
        actual_last = float(tr.loc[lastm, "costc"].sum() / (e[lastm] * c[lastm]).sum())
        res[name] = dict(balance_by_year=bal, elf=elf, indication=ind_lc, indication_normalised=ind_lc_n,
                         empirical_projection=float(emp), model_projection=model_lc * (1 + gt) ** k,
                         model_lc_last=model_lc, actual_capped_lc_last=actual_last, line_trend=gt, line_trend_normalised=gt_n,
                         gap_model=ind_lc / (model_lc * (1 + gt) ** k) - 1, gap_empirical=ind_lc / emp - 1,
                         gap_normalised=ind_lc_n / (model_lc_n * (1 + gt_n) ** k) - 1,
                         trends=trends[["freq_trend_fitted", "freq_trend", "sev_trend_fitted", "sev_trend"]].round(5).reset_index().to_dict("records"),
                         trends_normalised=trends_n[["freq_trend_fitted", "freq_trend", "sev_trend_fitted", "sev_trend"]].round(5).reset_index().to_dict("records"),
                         elasticities={p: dict(variable=w_, b=bb) for p, (w_, bb) in b.items()},
                         year_factor_by_year={int(y): float(np.average(yf[(tr["yr"] == y).to_numpy()], weights=e[(tr["yr"] == y).to_numpy()])) for y in sorted(tr["yr"].unique())})
        if sy is None:
            te = wx.attach(build.build_all()[line][1], weather, wx.normals(weather, fy))
            res[name]["year_factor_2025"] = float(np.average(wx.year_factor(te, line, b, share), weights=te["exposure"]))
            continue
        sc = (tr["yr"] == sy).to_numpy(); S, es = at(sc, last), e[sc]
        r, rn = m1.rate(S) * elf * es, m1n.rate(S) * elf * es
        p = {"L_model": r * (1 + gt) ** k, "L_ind": r * ind_lc / model_lc, "L_emp": r * emp / model_lc,
             "L_ind_n": rn * ind_lc_n / model_lc_n * yf[sc], "V1": r, "V1n": rn * yf[sc]}
        f = pd.DataFrame({"line": line, "customer_id": tr.loc[sc, "customer_id"].to_numpy(), "y": tr.loc[sc, "cost"].to_numpy(),
                          "exposure": es, "completion": c[sc], **p})
        WORK.mkdir(parents=True, exist_ok=True); f.to_parquet(WORK / f"{line}_{name}.parquet")
    print(line, "done", flush=True)
    return res


def main():
    th = load_thresholds()
    build.build_all()
    res = dict(zip(dr.LINES, Parallel(n_jobs=3)(delayed(run_line)(l) for l in dr.LINES)))
    frames = {s: pd.concat([pd.read_parquet(WORK / f"{l}_{s}.parquet") for l in dr.LINES], ignore_index=True) for s in v.SCORINGS}
    levels = {s: {ln: {k: float((g[k] * g["completion"]).sum() / g["y"].sum()) for k in ("L_model", "L_ind", "L_emp", "L_ind_n")}
                  for ln, g in list(f.groupby("line")) + [("all", f)]} for s, f in frames.items()}
    for s in levels:
        print(f"\nLevel ratios, scoring {s} (prediction x completion over actual):")
        print(pd.DataFrame(levels[s]).T.round(4).to_string())
    # ranking: normalised GLM (scored with its year's weather) against V1, keep rule
    tests = {}
    for s in v.SCORINGS:
        use_c = v.SCORINGS[s]["completion"]; tests[s] = {}
        for ln in list(dr.LINES) + ["all"]:
            d = frames[s] if ln == "all" else frames[s][frames[s]["line"] == ln]
            y = d["y"].to_numpy(); cc = d["completion"].to_numpy() if use_c else 1.0
            pa = v.rescale_to_total(d["V1"].to_numpy() * cc, y, by=d["line"].to_numpy())
            pb = v.rescale_to_total(d["V1n"].to_numpy() * cc, y, by=d["line"].to_numpy())
            da, db = v.tweedie_unit_deviance(y, pa), v.tweedie_unit_deviance(y, pb)
            rec = dict(gain=float((da - db).sum()))
            if s == "A":
                rec["boot"] = v.paired_bootstrap_groups(da, db, d["customer_id"].to_numpy(), n_boot=2000)
            tests[s][ln] = rec
    verdict = {ln: v.keep_verdict({s: tests[s][ln]["gain"] for s in tests}, tests["A"][ln]["boot"], int(th["keep_years_better"])) for ln in tests["A"]}
    print("\nRanking, weather-normalised GLM against V1:")
    for ln in tests["A"]:
        a = tests["A"][ln]
        print(f"  {ln:9s} A saved {a['gain']:9.1f} [{a['boot']['lo']:9.1f}, {a['boot']['hi']:9.1f}] | B saved {tests['B'][ln]['gain']:9.1f} | keep {verdict[ln]['keep']}")
    for l in dr.LINES:
        for name in ("A", "B", "final"):
            r = res[l][name]
            print(f"\n{l} {name}: balance by year {({y: round(x, 4) for y, x in r['balance_by_year'].items()})}")
            print(f"   indication {r['indication']:.1f} | empirical projection {r['empirical_projection']:.1f} | model projection {r['model_projection']:.1f} "
                  f"| gap to model {r['gap_model']:+.4f} | gap to empirical {r['gap_empirical']:+.4f} | normalised indication {r['indication_normalised']:.1f}, gap {r['gap_normalised']:+.4f}")
            print(f"   year factor by year {({y: round(x, 4) for y, x in r['year_factor_by_year'].items()})}" + (f" | 2025 {r['year_factor_2025']:.4f}" if 'year_factor_2025' in r else ""))
            if name == "final":
                t, tn = pd.DataFrame(r["trends"]).set_index("peril"), pd.DataFrame(r["trends_normalised"]).set_index("peril")
                print(pd.concat({"raw": t[["freq_trend_fitted", "sev_trend_fitted"]], "normalised": tn[["freq_trend_fitted", "sev_trend_fitted"]]}, axis=1).round(4).to_string())
    write_evidence({"phase5_k3k4": dict(by_line=res, levels=levels, ranking=tests, verdict=verdict)}, OUT / "phase5_k3k4.json")


if __name__ == "__main__":
    main()
