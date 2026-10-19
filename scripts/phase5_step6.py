"""Phase 5 step 6: model structure (P3) and the per-peril excess factor (K2, C2). Design written first:
decisions/phase5_step6_design.md.

On the working base W2 columns of each line (scripts/phase5_l8_check.py KEPT and DROP), non-hold-out customers:
  S1  one capped Tweedie GLM per line (the reference)
  S2  one capped Tweedie GLM per peril, summed
  S3  Poisson frequency x gamma severity per peril, both on the W2 columns
  S4  as S3 with severity by year only
each times the line excess factor (L) or, for S2 to S4, each peril's own factor (P). Scored on A and B at a common
level per line in two forms: bare (rate at the latest fitting year) and priced (off-balance to the weather-normalised
indication, per line for S1 and per peril for S2 to S4, times the scored year's weather). Keep rule throughout.
Run: .venv/Scripts/python.exe -I scripts/phase5_step6.py   (P6_STAGE=tests reruns the tests from saved frames)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.phase5_l8_check import DROP, KEPT
from scripts.phase5_step5 import load_plan, spec_columns
from src import build, data_rules as dr, level as lv, validation as v, weather as wx
from src.evidence import write_evidence
from src.glm import TweedieGLM
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT, WORK = ROOT / "outputs", ROOT / "data" / "work" / "phase5_step6"
MODELS = ("S1-L", "S2-L", "S2-P", "S3-L", "S3-P", "S4-L", "S4-P")
FORMS = ("bare", "priced")
PERIL_STRUCTS = ("S2", "S3", "S4")


def fit_plan(line: str, name: str) -> dict:
    """Fit every structure on one plan; return score-row predictions (A, B) or final-fit relativities (final)."""
    th = load_thresholds()
    weather = dr.read_raw("weather")
    full = {l: build.build_all()[l][0] for l in dr.LINES}
    full = {l: d[~d["holdout"]] for l, d in full.items()}
    P = load_plan(line, name, th, full, weather)
    tr, fit, sc, last = P["tr"], P["fit"], P["sc"], P["last"]
    cols = spec_columns(P, dict.fromkeys(KEPT[line]), DROP[line])
    G = P["F"][cols].rename(columns=lambda x: x.split(":")[-1])
    perils = lv.perils_of(tr)
    e = tr["exposure"].to_numpy()
    lastm = fit & (tr["yr"] == last).to_numpy()

    def at(mask):
        S = G[mask].copy(); S["yr"] = str(last); return S

    masks = {"sc": sc, "last": lastm}
    rates, models = {}, {}
    m = TweedieGLM().fit(G[fit], tr.loc[fit, "costc"], e[fit] * tr.loc[fit, "completion"].to_numpy())
    models[("S1", "line")] = m
    for k, mk in masks.items():
        rates[("S1", "line", k)] = m.rate(at(mk)) if mk.any() else np.zeros(0)
    for p in perils:
        ecp = e * tr[f"completion_{p}"].to_numpy()
        n = tr[f"n_{p}"].to_numpy().astype(float)
        cm = fit & (n > 0)
        m2 = TweedieGLM().fit(G[fit], tr.loc[fit, f"costc_{p}"], ecp[fit])
        mf = TweedieGLM(power=1.0).fit(G[fit], n[fit], ecp[fit])
        ms = TweedieGLM(power=2.0).fit(G[cm], tr.loc[cm, f"costc_{p}"], n[cm], weight=n[cm])
        ms0 = TweedieGLM(power=2.0).fit(G.loc[cm, ["yr"]], tr.loc[cm, f"costc_{p}"], n[cm], weight=n[cm])
        models |= {("S2", p): m2, ("S3f", p): mf, ("S3s", p): ms, ("S4s", p): ms0}
        for k, mk in masks.items():
            if not mk.any():
                continue
            S = at(mk)
            fr = mf.rate(S)
            rates |= {("S2", p, k): m2.rate(S), ("S3", p, k): fr * ms.rate(S), ("S4", p, k): fr * ms0.rate(S)}
    exc = lv.excess_factors(tr[fit], {})["peril"]
    elf = P["elf"]
    if name == "final":
        rel = {f"{s}|{p}": mm.relativities().to_dict("records") for (s, p), mm in models.items() if s in ("S1", "S3f", "S3s")}
        n_claims = {p: int(tr.loc[fit, f"n_{p}"].sum()) for p in perils}
        print(f"{line} final done", flush=True)
        return dict(excess_line=elf, excess_peril=exc, relativities=rel, claims=n_claims, columns=cols)

    # level chain (K3, K4): weather-normalised indication for the scored year from the fitting years only
    sy = int(tr.loc[sc, "yr"].iloc[0])
    fy = tuple(sorted(tr.loc[fit, "yr"].unique()))
    trx = wx.attach(tr, weather, wx.normals(weather, fy))
    b = wx.elasticities(trx[fit], line)
    share = {p: float(tr.loc[fit, f"cost_{p}"].sum() / tr.loc[fit, "cost"].sum()) for p in perils}
    yf = wx.year_factor(trx, line, b, share)
    wm = {p: (np.exp(b[p][1] * trx[f"dev_{b[p][0]}"].fillna(0.0).to_numpy()) if p in b else np.ones(len(tr))) for p in perils}
    trn = trx.copy()
    for p in b:
        trn[f"wm_{p}"] = wm[p]
    trends = lv.peril_trends(trn[fit], line, th["trend_min"], th["trend_max"])
    ind = lv.indication(trn[fit], trends, sy)["indicated_capped_lc"]
    el = e[lastm]
    ob = {"S1": float(ind.sum() / (np.sum(rates[("S1", "line", "last")] * el / yf[lastm]) / el.sum()))}
    for s in PERIL_STRUCTS:
        for p in perils:
            ob[f"{s}|{p}"] = float(ind[p] / (np.sum(rates[(s, p, "last")] * el / wm[p][lastm]) / el.sum()))

    use_c = v.SCORINGS[name]["completion"]
    es = e[sc]
    cl = tr.loc[sc, "completion"].to_numpy() if use_c else 1.0
    cp = {p: (tr.loc[sc, f"completion_{p}"].to_numpy() if use_c else 1.0) for p in perils}
    f = pd.DataFrame({"line": line, "customer_id": tr.loc[sc, "customer_id"].to_numpy(), "y": tr.loc[sc, "cost"].to_numpy(),
                      "exposure": es})
    r1 = rates[("S1", "line", "sc")]
    f["bare:S1-L"] = r1 * elf * es * cl
    f["priced:S1-L"] = r1 * elf * ob["S1"] * yf[sc] * es * cl
    for s in PERIL_STRUCTS:
        for form in FORMS:
            comp = {}
            for p in perils:
                x = rates[(s, p, "sc")] * es * cp[p]
                if form == "priced":
                    x = x * ob[f"{s}|{p}"] * wm[p][sc]
                comp[p] = x
                f[f"{form}:{s}:{p}"] = x
            f[f"{form}:{s}-L"] = sum(comp.values()) * elf
            f[f"{form}:{s}-P"] = sum(comp[p] * exc[p] for p in perils)
    for p in perils:
        f[f"y:{p}"] = tr.loc[sc, f"cost_{p}"].to_numpy()
    WORK.mkdir(parents=True, exist_ok=True)
    f.to_parquet(WORK / f"{line}_{name}.parquet")
    print(f"{line} {name} done", flush=True)
    return dict(excess_line=elf, excess_peril=exc, off_balance=ob, indication=ind.to_dict(), elasticities={p: list(x) for p, x in b.items()},
                columns=cols, perils=perils)


def _dev(d: pd.DataFrame, col: str) -> np.ndarray:
    y = d["y"].to_numpy()
    return v.tweedie_unit_deviance(y, v.rescale_to_total(d[col].to_numpy(), y, by=d["line"].to_numpy()))


def test(frames: dict, ref: str, cand: str, th, lines=None) -> dict:
    """Keep-rule test of cand against ref (columns 'form:model'), per line and combined, at a common level per line."""
    out = {}
    for ln in (lines or list(dr.LINES)) + ["all"]:
        r = {}
        for s in v.SCORINGS:
            d = frames[s] if ln == "all" else frames[s][frames[s]["line"] == ln]
            if ln == "all" and lines:
                d = d[d["line"].isin(lines)]
            da, db = _dev(d, ref), _dev(d, cand)
            rec = dict(gain=float((da - db).sum()))
            if s == "A":
                rec["boot"] = v.paired_bootstrap_groups(da, db, d["customer_id"].to_numpy(), n_boot=2000)
            r[s] = rec
        r["verdict"] = v.keep_verdict({s: r[s]["gain"] for s in v.SCORINGS}, r["A"]["boot"], int(th["keep_years_better"]))
        out[ln] = r
    return out


def ranking_d2(frames: dict) -> dict:
    out = {}
    for s, f in frames.items():
        for ln in list(dr.LINES) + ["all"]:
            d = f if ln == "all" else f[f["line"] == ln]
            y = d["y"].to_numpy()
            for col in [c for c in f.columns if c.count(":") == 1 and c.split(":")[0] in FORMS]:
                mu = v.rescale_to_total(d[col].to_numpy(), y, by=d["line"].to_numpy())
                out.setdefault(s, {}).setdefault(ln, {})[col] = dict(d2=v.d2_tweedie(y, mu, d["exposure"]),
                                                                     level_ratio=float(d[col].sum() / y.sum()))
    return out


def peril_diagnostic(frames: dict, perils: dict) -> dict:
    """Each structure's peril component against the peril's actual cost, at a common level per peril."""
    out = {}
    for s, f in frames.items():
        for ln, ps in perils.items():
            d = f[f["line"] == ln]
            for p in ps:
                y = d[f"y:{p}"].to_numpy()
                for form in FORMS:
                    dev = {st: float(v.tweedie_unit_deviance(y, v.rescale_to_total(d[f"{form}:{st}:{p}"].to_numpy(), y)).sum())
                           for st in PERIL_STRUCTS}
                    out.setdefault(ln, {}).setdefault(p, {}).setdefault(form, {})[s] = dev
    return out


def run_tests(fits: dict) -> dict:
    th = load_thresholds()
    frames = {s: pd.concat([pd.read_parquet(WORK / f"{l}_{s}.parquet") for l in dr.LINES], ignore_index=True) for s in v.SCORINGS}
    tests = {}
    for form in FORMS:
        for s in PERIL_STRUCTS:
            tests[f"{form}|{s}-L vs S1-L"] = test(frames, f"{form}:S1-L", f"{form}:{s}-L", th)
            tests[f"{form}|{s}-P vs {s}-L"] = test(frames, f"{form}:{s}-L", f"{form}:{s}-P", th)
            tests[f"{form}|{s}-P vs S1-L"] = test(frames, f"{form}:S1-L", f"{form}:{s}-P", th)
        tests[f"{form}|S3-L vs S2-L"] = test(frames, f"{form}:S2-L", f"{form}:S3-L", th)
        tests[f"{form}|S4-L vs S3-L"] = test(frames, f"{form}:S3-L", f"{form}:S4-L", th)
        tests[f"{form}|S2-L vs S4-L"] = test(frames, f"{form}:S4-L", f"{form}:S2-L", th)
        tests[f"{form}|S4-P vs S3-P"] = test(frames, f"{form}:S3-P", f"{form}:S4-P", th)
        tests[f"{form}|S3-P vs S2-P"] = test(frames, f"{form}:S2-P", f"{form}:S3-P", th)
        tests[f"{form}|S2-P vs S4-P"] = test(frames, f"{form}:S4-P", f"{form}:S2-P", th)
    perils = {l: fits[l]["A"]["perils"] for l in dr.LINES}
    return dict(tests=tests, ranking_d2=ranking_d2(frames), peril_diagnostic=peril_diagnostic(frames, perils))


def show(res: dict) -> None:
    for key, t in res["tests"].items():
        print(f"\n{key}")
        for ln, r in t.items():
            a = r["A"]
            print(f"  {ln:9s} A {a['gain']:10.1f} [{a['boot']['lo']:10.1f}, {a['boot']['hi']:10.1f}]  B {r['B']['gain']:10.1f}  keep {r['verdict']['keep']}")
    for s, by in res["ranking_d2"].items():
        print(f"\nranking D2 and level ratio, scoring {s}")
        for ln, cols in by.items():
            print(f"  {ln:9s} " + "  ".join(f"{c}={x['d2']:.4%}/{x['level_ratio']:.3f}" for c, x in cols.items() if c.startswith("priced")))


def main():
    if os.environ.get("P6_STAGE") == "tests":
        import json
        fits = json.loads((OUT / "phase5_step6.json").read_text())["phase5_step6"]["fits"]
    else:
        tasks = [(l, n) for l in ("motor", "buildings", "contents") for n in ("A", "B", "final")]
        out = Parallel(n_jobs=4)(delayed(fit_plan)(l, n) for l, n in tasks)
        fits = {}
        for (l, n), r in zip(tasks, out):
            fits.setdefault(l, {})[n] = r
    res = run_tests(fits)
    show(res)
    write_evidence({"phase5_step6": dict(fits=fits, **res, kept=KEPT, drop=DROP)}, OUT / "phase5_step6.json")


if __name__ == "__main__":
    main()
