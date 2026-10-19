"""Phase 5, steps 1 to 3: cap and load (C1, C2), trend and level (C3, C4, P11), weather (J-weather, C6).

Nested variants, each scored out of time on A and B and tested against the previous one with the keep rule:
  V0  uncapped Tweedie GLM (the phase 4 baseline, refitted here on the shared feature code)
  V1  GLM on capped cost, times the line's excess factor
  V2  two GLMs on capped cost (large-loss perils, other perils), each times its own excess factor (C2)
  V3  V2 with address climate normals as rating factors (J-weather plan part)
  V4  V3 times the year-weather multiplier for storm, subsidence, escape of water and glass (C6)
Step 2 adds the 2025-style level to V2: per-peril trends (bounded, C3), the indication, the off-balance (P11) and
the gap to the model's own projection (C4), reported as level ratios. A final run on all non-hold-out years gives
the real 2025 trends, indication and level check. Hold-out customers are never used.
Run: .venv/Scripts/python.exe -I scripts/phase5_steps123.py
"""
from __future__ import annotations

import json
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
WORK = ROOT / "data" / "work" / "phase5"
VARIANTS = ["V0", "V1", "V2", "V3", "V4"]


def run_line(line: str) -> dict:
    th = load_thresholds()
    data = build.build_all()
    tr0 = data[line][0]
    tr0 = tr0[~tr0["holdout"]].reset_index(drop=True)
    if os.environ.get("P5_SAMPLE"):                      # smoke run on a sample of rows
        tr0 = tr0.sample(frac=float(os.environ["P5_SAMPLE"]), random_state=0).reset_index(drop=True)
    weather = dr.read_raw("weather")
    perils = lv.perils_of(tr0)
    ll = list(build.LARGE_LOSS_PERILS[line]); other = [p for p in perils if p not in ll]
    groups = {"large": ll, "other": other}
    results, frames = {}, {}
    plans = {**{k: dict(fit_years=s["fit_years"], score_year=s["score_year"]) for k, s in v.SCORINGS.items()},
             "final": dict(fit_years=(2021, 2022, 2023, 2024), score_year=None)}
    for name, plan in plans.items():
        fy, sy = plan["fit_years"], plan["score_year"]
        tr = wx.attach(tr0, weather, wx.normals(weather, fy))
        fit = tr["yr"].isin(fy).to_numpy()
        last = max(fy)
        lastm = (tr["yr"] == last).to_numpy() & fit
        target_year = sy if sy is not None else 2025
        edges = fit_edges(tr[fit], line, normals=True)
        F0, F1 = features(tr, line, edges), features(tr, line, edges, normals=True)
        e = tr["exposure"].to_numpy()
        comp = {"line": tr["completion"].to_numpy()}
        # group completions weighted by each group's capped cost in the fitting rows
        for g, ps in groups.items():
            w = np.array([tr.loc[fit, f"costc_{p}"].sum() for p in ps]); w = w / w.sum()
            comp[g] = sum(wi * tr[f"completion_{p}"].to_numpy() for wi, p in zip(w, ps))
        ef = lv.excess_factors(tr[fit], groups)
        elf_line = float(tr.loc[fit, "cost"].sum() / tr.loc[fit, "costc"].sum())
        trends = lv.peril_trends(tr[fit], line, th["trend_min"], th["trend_max"])
        ind = lv.indication(tr[fit], trends, target_year)

        def at_year(F, mask, year):
            G = F[mask].copy(); G["yr"] = str(year); return G

        m0 = TweedieGLM().fit(F0[fit], tr.loc[fit, "cost"], e[fit] * comp["line"][fit])
        m1 = TweedieGLM().fit(F0[fit], tr.loc[fit, "costc"], e[fit] * comp["line"][fit])
        m2 = {g: TweedieGLM().fit(F0[fit], tr.loc[fit, f"costc_{g}"], e[fit] * comp[g][fit]) for g in groups}
        m3 = {g: TweedieGLM().fit(F1[fit], tr.loc[fit, f"costc_{g}"], e[fit] * comp[g][fit]) for g in groups}

        # level (step 2) on V2 and V3: off-balance each group to the indication using the latest fitted year's rows
        level = {}
        for vn, models, F in (("V2", m2, F0), ("V3", m3, F1)):
            for g, ps in groups.items():
                rate_last = models[g].rate(at_year(F, lastm, last))
                ind_g = float(ind.loc[ps, "indicated_capped_lc"].sum())
                gt = lv.group_trend(trends, ind["indicated_capped_lc"], ps)
                level[(vn, g)] = lv.off_balance(rate_last, e[lastm], ind_g, gt, target_year - last) | dict(group_trend=gt)
        b = wx.elasticities(tr[fit], line)
        cost_share = {p: float(tr.loc[fit, f"cost_{p}"].sum() / tr.loc[fit, "cost"].sum()) for p in perils}
        res = dict(excess_factor_line=elf_line, excess_factors=ef, trends=trends.reset_index().to_dict("records"),
                   indication=ind.reset_index().rename(columns={"index": "peril"}).to_dict("records"),
                   level={f"{k[0]}_{k[1]}": val for k, val in level.items()},
                   weather_elasticities={p: dict(variable=wv, b=bb) for p, (wv, bb) in b.items()}, cost_share=cost_share,
                   relativities_V2={g: m.relativities().to_dict("records") for g, m in m2.items()} if name == "final" else None)
        results[name] = res
        if sy is None:
            continue
        sc = (tr["yr"] == sy).to_numpy()
        S0, S1 = at_year(F0, sc, last), at_year(F1, sc, last)
        es = e[sc]
        p = {"V0": m0.rate(S0) * es, "V1": m1.rate(S0) * elf_line * es}
        p["V2"] = sum(m2[g].rate(S0) * ef["group"][g] for g in groups) * es
        p["V3"] = sum(m3[g].rate(S1) * ef["group"][g] for g in groups) * es
        p["V4"] = p["V3"] * wx.year_factor(tr[sc], line, b, cost_share)
        # level variants on V2: no trend (last-year level), the model's own trend projection, the indication
        k = sy - last
        p["L_none"] = p["V2"]
        p["L_trend"] = sum(m2[g].rate(S0) * ef["group"][g] * (1 + level[("V2", g)]["group_trend"]) ** k for g in groups) * es
        p["L_ind"] = sum(m2[g].rate(S0) * ef["group"][g] * level[("V2", g)]["factor"] for g in groups) * es
        p["L_ind_V4"] = sum(m3[g].rate(S1) * ef["group"][g] * level[("V3", g)]["factor"] for g in groups) * es * wx.year_factor(tr[sc], line, b, cost_share)
        f = pd.DataFrame({"line": line, "customer_id": tr.loc[sc, "customer_id"].to_numpy(), "y": tr.loc[sc, "cost"].to_numpy(),
                          "exposure": es, "completion": tr.loc[sc, "completion"].to_numpy(), **p})
        frames[name] = f
    WORK.mkdir(parents=True, exist_ok=True)
    for name, f in frames.items():
        f.to_parquet(WORK / f"{line}_{name}.parquet")
    print(f"{line} done", flush=True)
    return results


def compare(frames: pd.DataFrame, scoring: str, ref: str, cand: str) -> dict:
    """Keep-rule pieces for one scoring: deviance saved by cand over ref at a common level, per line and combined."""
    use_c = v.SCORINGS[scoring]["completion"]
    out = {}
    for ln in list(dr.LINES) + ["all"]:
        d = frames if ln == "all" else frames[frames["line"] == ln]
        y = d["y"].to_numpy(); c = d["completion"].to_numpy() if use_c else 1.0
        pa = v.rescale_to_total(d[ref].to_numpy() * c, y, by=d["line"].to_numpy())
        pb = v.rescale_to_total(d[cand].to_numpy() * c, y, by=d["line"].to_numpy())
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
    for ref, cand in (("V0", "V1"), ("V1", "V2"), ("V0", "V2"), ("V2", "V3"), ("V3", "V4"), ("V2", "V4")):
        a, b = compare(frames["A"], "A", ref, cand), compare(frames["B"], "B", ref, cand)
        verdict = {ln: v.keep_verdict({"A": a[ln]["gain"], "B": b[ln]["gain"]}, a[ln]["boot"], int(th["keep_years_better"])) for ln in a}
        tests[f"{cand}_vs_{ref}"] = dict(A=a, B=b, verdict=verdict)
        print(f"\n{cand} vs {ref}:")
        for ln in a:
            print(f"  {ln:9s} A: D2 {a[ln]['d2_ref']:.4%} -> {a[ln]['d2_cand']:.4%}, saved {a[ln]['gain']:8.1f} "
                  f"[{a[ln]['boot']['lo']:8.1f}, {a[ln]['boot']['hi']:8.1f}] | B: {b[ln]['d2_ref']:.4%} -> {b[ln]['d2_cand']:.4%}, "
                  f"saved {b[ln]['gain']:8.1f} | keep {verdict[ln]['keep']}")
    levels = {}
    for s in v.SCORINGS:
        f = frames[s]
        c = f["completion"].to_numpy()
        levels[s] = {ln: {lvar: float((g[lvar] * g["completion"]).sum() / g["y"].sum()) for lvar in ("L_none", "L_trend", "L_ind", "L_ind_V4")}
                     for ln, g in list(f.groupby("line")) + [("all", f)]}
        print(f"\nLevel ratios (prediction x completion over actual), scoring {s}:")
        print(pd.DataFrame(levels[s]).T.round(3).to_string())
    for l in dr.LINES:
        fin = res[l]["final"]
        print(f"\n{l} final (fit 2021-2024, priced for 2025): excess factors {json.dumps({k: round(x, 3) for k, x in fin['excess_factors']['group'].items()})}")
        t = pd.DataFrame(fin["trends"]).set_index("peril")[["freq_trend_fitted", "freq_trend", "sev_trend_fitted", "sev_trend", "bounded", "sev_step_level"]]
        print(t.round(4).to_string())
        print("  level check (gap of indication to model projection):", {k: round(x["gap"], 4) for k, x in fin["level"].items()})
        print("  weather elasticities:", {p: round(x["b"], 4) for p, x in fin["weather_elasticities"].items()})
    write_evidence({"phase5_steps123": dict(by_line=res, tests=tests, levels=levels)}, OUT / "phase5_steps123.json")


if __name__ == "__main__":
    main()
