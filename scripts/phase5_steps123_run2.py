"""Phase 5, steps 1 to 3, second run, after the first run (scripts/phase5_steps123.py) found:
  - two-group capped GLMs (the first form of C2) rank worse than one capped GLM with a line excess factor (V1);
  - climate normals that include a fitting row's own year leak that year's weather into the "normal";
  - the subsidence elasticity stalled at zero because its deviation was not standardised.
Variants, all on the shared feature code, scored on A and B, keep rule against the previous one:
  V0  uncapped GLM
  V1  capped GLM times the line excess factor (passed in run 1)
  V5  V1 with climate normals that leave out each fitting row's own year
  V6  V5 times the year-weather multiplier (elasticities on standardised deviations)
  V7  V1 times the year-weather multiplier, without normals
Level (step 2) on V1 at line level: per-peril trends (C3), indication, off-balance (P11), C4 gap.
Run: .venv/Scripts/python.exe -I scripts/phase5_steps123_run2.py
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
OUT = ROOT / "outputs"
WORK = ROOT / "data" / "work" / "phase5_run2"


def run_line(line: str) -> dict:
    th = load_thresholds()
    tr0 = build.build_all()[line][0]
    tr0 = tr0[~tr0["holdout"]].reset_index(drop=True)
    if os.environ.get("P5_SAMPLE"):
        tr0 = tr0.sample(frac=float(os.environ["P5_SAMPLE"]), random_state=0).reset_index(drop=True)
    weather = dr.read_raw("weather")
    perils = lv.perils_of(tr0)
    plans = {**{k: dict(fit_years=s["fit_years"], score_year=s["score_year"]) for k, s in v.SCORINGS.items()},
             "final": dict(fit_years=(2021, 2022, 2023, 2024), score_year=None)}
    results = {}
    for name, plan in plans.items():
        fy, sy = plan["fit_years"], plan["score_year"]
        score_mask = (tr0["yr"] == sy).to_numpy() if sy is not None else np.zeros(len(tr0), bool)
        tr = wx.normals_leave_own_year_out(tr0, weather, fy, score_mask)
        fit = tr["yr"].isin(fy).to_numpy()
        last = max(fy); lastm = fit & (tr["yr"] == last).to_numpy()
        target = sy if sy is not None else 2025
        edges = fit_edges(tr[fit], line, normals=True)
        F0, F1 = features(tr, line, edges), features(tr, line, edges, normals=True)
        e, c = tr["exposure"].to_numpy(), tr["completion"].to_numpy()
        elf = float(tr.loc[fit, "cost"].sum() / tr.loc[fit, "costc"].sum())
        trends = lv.peril_trends(tr[fit], line, th["trend_min"], th["trend_max"])
        ind = lv.indication(tr[fit], trends, target)
        m0 = TweedieGLM().fit(F0[fit], tr.loc[fit, "cost"], e[fit] * c[fit])
        m1 = TweedieGLM().fit(F0[fit], tr.loc[fit, "costc"], e[fit] * c[fit])
        m5 = TweedieGLM().fit(F1[fit], tr.loc[fit, "costc"], e[fit] * c[fit])

        def at(F, mask, year):
            G = F[mask].copy(); G["yr"] = str(year); return G

        ind_line = float(ind["indicated_capped_lc"].sum())
        gt = lv.group_trend(trends, ind["indicated_capped_lc"], perils)
        lvl = {vn: lv.off_balance(m.rate(at(F, lastm, last)), e[lastm], ind_line, gt, target - last) | dict(trend=gt)
               for vn, m, F in (("V1", m1, F0), ("V5", m5, F1))}
        # per-peril capped loss cost by year, to explain any C4 gap
        by_year = {p: {int(y): float(tr.loc[fit & (tr["yr"] == y).to_numpy(), f"costc_{p}"].sum()
                                    / (tr.loc[fit & (tr["yr"] == y).to_numpy(), "exposure"] * tr.loc[fit & (tr["yr"] == y).to_numpy(), f"completion_{p}"]).sum())
                       for y in fy} for p in perils}
        b = wx.elasticities(tr[fit], line)
        share = {p: float(tr.loc[fit, f"cost_{p}"].sum() / tr.loc[fit, "cost"].sum()) for p in perils}
        res = dict(excess_factor_line=elf, trends=trends.reset_index().to_dict("records"),
                   indication=ind.reset_index().rename(columns={"index": "peril"}).to_dict("records"),
                   indicated_capped_lc=ind_line, line_trend=gt, level=lvl, capped_lc_by_year=by_year,
                   weather_elasticities={p: dict(variable=wv, b=bb) for p, (wv, bb) in b.items()}, cost_share=share)
        if name == "final":
            res["relativities_V1"] = m1.relativities().to_dict("records")
            res["relativities_V5"] = m5.relativities().to_dict("records")
        results[name] = res
        if sy is None:
            continue
        sc = score_mask
        S0, S1, es = at(F0, sc, last), at(F1, sc, last), e[sc]
        yf = wx.year_factor(tr[sc], line, b, share)
        p = {"V0": m0.rate(S0) * es, "V1": m1.rate(S0) * elf * es, "V5": m5.rate(S1) * elf * es}
        p["V6"], p["V7"] = p["V5"] * yf, p["V1"] * yf
        k = sy - last
        p["L_none"] = p["V1"]
        p["L_trend"] = p["V1"] * (1 + gt) ** k
        p["L_ind"] = p["V1"] * lvl["V1"]["factor"]
        f = pd.DataFrame({"line": line, "customer_id": tr.loc[sc, "customer_id"].to_numpy(), "y": tr.loc[sc, "cost"].to_numpy(),
                          "exposure": es, "completion": tr.loc[sc, "completion"].to_numpy(), "year_factor": yf, **p})
        WORK.mkdir(parents=True, exist_ok=True)
        f.to_parquet(WORK / f"{line}_{name}.parquet")
    print(f"{line} done", flush=True)
    return results


def compare(frames, scoring, ref, cand):
    use_c = v.SCORINGS[scoring]["completion"]
    out = {}
    for ln in list(dr.LINES) + ["all"]:
        d = frames if ln == "all" else frames[frames["line"] == ln]
        y = d["y"].to_numpy(); cc = d["completion"].to_numpy() if use_c else 1.0
        pa = v.rescale_to_total(d[ref].to_numpy() * cc, y, by=d["line"].to_numpy())
        pb = v.rescale_to_total(d[cand].to_numpy() * cc, y, by=d["line"].to_numpy())
        da, db = v.tweedie_unit_deviance(y, pa), v.tweedie_unit_deviance(y, pb)
        rec = dict(gain=float((da - db).sum()), d2_ref=v.d2_tweedie(y, pa, d["exposure"]), d2_cand=v.d2_tweedie(y, pb, d["exposure"]))
        if scoring == "A":
            rec["boot"] = v.paired_bootstrap_groups(da, db, d["customer_id"].to_numpy(), n_boot=2000)
        out[ln] = rec
    return out


def main():
    th = load_thresholds()
    build.build_all()
    res = dict(zip(dr.LINES, Parallel(n_jobs=3)(delayed(run_line)(l) for l in dr.LINES)))
    frames = {s: pd.concat([pd.read_parquet(WORK / f"{l}_{s}.parquet") for l in dr.LINES], ignore_index=True) for s in v.SCORINGS}
    tests = {}
    for ref, cand in (("V0", "V1"), ("V1", "V5"), ("V5", "V6"), ("V1", "V7")):
        a, b = compare(frames["A"], "A", ref, cand), compare(frames["B"], "B", ref, cand)
        verdict = {ln: v.keep_verdict({"A": a[ln]["gain"], "B": b[ln]["gain"]}, a[ln]["boot"], int(th["keep_years_better"])) for ln in a}
        tests[f"{cand}_vs_{ref}"] = dict(A=a, B=b, verdict=verdict)
        print(f"\n{cand} vs {ref}:")
        for ln in a:
            print(f"  {ln:9s} A: D2 {a[ln]['d2_ref']:.4%} -> {a[ln]['d2_cand']:.4%}, saved {a[ln]['gain']:9.1f} "
                  f"[{a[ln]['boot']['lo']:9.1f}, {a[ln]['boot']['hi']:9.1f}] | B: {b[ln]['d2_ref']:.4%} -> {b[ln]['d2_cand']:.4%}, "
                  f"saved {b[ln]['gain']:9.1f} | keep {verdict[ln]['keep']}")
    levels = {}
    for s in v.SCORINGS:
        f = frames[s]
        levels[s] = {ln: {lvar: float((g[lvar] * g["completion"]).sum() / g["y"].sum()) for lvar in ("L_none", "L_trend", "L_ind")}
                     for ln, g in list(f.groupby("line")) + [("all", f)]}
        print(f"\nLevel ratios (prediction x completion over actual), scoring {s}:")
        print(pd.DataFrame(levels[s]).T.round(3).to_string())
    for l in dr.LINES:
        fin = res[l]["final"]
        print(f"\n{l} final: excess factor {fin['excess_factor_line']:.3f}; indicated capped loss cost {fin['indicated_capped_lc']:.1f}; "
              f"line trend {fin['line_trend']:.4f}; C4 gap V1 {fin['level']['V1']['gap']:+.4f}, V5 {fin['level']['V5']['gap']:+.4f}")
        print("  capped loss cost by year:", {p: {y: round(x, 1) for y, x in d.items()} for p, d in fin["capped_lc_by_year"].items()})
        print("  weather elasticities (per unit of deviation):", {p: round(x["b"], 4) for p, x in fin["weather_elasticities"].items()})
    write_evidence({"phase5_steps123_run2": dict(by_line=res, tests=tests, levels=levels)}, OUT / "phase5_steps123_run2.json")


if __name__ == "__main__":
    main()
