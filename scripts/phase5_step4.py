"""Phase 5 step 4: claims history (C7). Design and thresholds written first: decisions/phase5_step4_design.md.

For each plan (scoring A, scoring B, final fit), on non-hold-out customers only:
  1. a Poisson GLM per line on the V1 rating factors gives each row's class frequency (claims of all statuses);
  2. point-in-time history per row for its policy (H-P), the customer's other policies (H-C) and other customers at
     the address (H-A), as claims A and expected claims E before the cutoff min(row start, data end);
  3. k per scope from the fitting years by the Bühlmann-Straub estimator; mod = (A + k) / (E + k).
Then per line, each piece is added to V1 (capped Tweedie GLM x line excess factor) as log(mod), and scored on A and B
with the keep rule. Variants of H-P are scored against H-P. Pieces that pass are combined and each is dropped in turn.
Run: .venv/Scripts/python.exe -I scripts/phase5_step4.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import build, data_rules as dr, history as hs, validation as v
from src.evidence import write_evidence
from src.glm import TweedieGLM, band, cat, features, fit_edges, quantile_edges
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
# Phase 7: P5_HOLDOUT=include builds the history on every customer, hold-out included, into its own folder
INCLUDE_HOLDOUT = os.environ.get("P5_HOLDOUT") == "include"
OUT, WORK = ROOT / "outputs", ROOT / "data" / "work" / ("phase5_step4_all" if INCLUDE_HOLDOUT else "phase5_step4")
PLANS = {"A": dict(fit_years=(2021, 2022), score_year=2023), "B": dict(fit_years=(2021, 2022, 2023), score_year=2024),
         "final": dict(fit_years=(2021, 2022, 2023, 2024), score_year=None)}
PROPERTY = ("buildings", "contents")
PIECES = ("H-P", "H-C", "H-A", "H-D", "H-R")
K_EST = os.environ.get("P5_K", "np")          # "np": non-parametric Bühlmann-Straub (run 1); "pg": Poisson-gamma (run 2)
SUFFIX = "" if K_EST == "np" else f"_{K_EST}"
P_VARIANTS = ("P-raw", "P-band", "P-avg", "P-mult", "P-group")


def _train(line):
    tr = build.build_all()[line][0]
    return tr.reset_index(drop=True) if INCLUDE_HOLDOUT else tr[~tr["holdout"]].reset_index(drop=True)


def prepare(plan_name: str) -> dict:
    """History columns for every row of the plan's fitting and scoring years (and the 2025 test rows for 'final')."""
    plan = PLANS[plan_name]
    fy, sy = plan["fit_years"], plan["score_year"]
    data_end = hs.day(pd.Timestamp(f"{max(fy) + 1}-01-01"))
    claims = dr.read_raw("claims")
    H = {ln: hs.reported_share_cum(claims, ln) for ln in dr.LINES}
    src_rows, src_claims, tgt, kdata = [], [], {}, {}
    for line in dr.LINES:
        tr = _train(line)
        m = dr.match_claims_to_rows(tr, claims, line)
        m = m.merge(tr[["row_id", "customer_id", "address_id", "yr"]], on="row_id", how="inner")
        n_all = m.groupby("row_id").size().reindex(tr["row_id"]).fillna(0).to_numpy()
        fit = tr["yr"].isin(fy).to_numpy()
        F = features(tr, line, fit_edges(tr[fit], line))
        ecn = (tr["exposure"] * tr["completion_count"]).to_numpy()
        pois = TweedieGLM(power=1.0).fit(F[fit], n_all[fit], ecn[fit])
        rate = np.where(fit, pois.rate(F), np.nan)
        rate_avg = float(n_all[fit].sum() / ecn[fit].sum())
        g1, g2 = hs.GROUPS[line].values()
        m["g1"], m["g2"] = m["peril"].isin(g1), m["peril"].isin(g2)
        mf = m[m["yr"].isin(fy)]
        sh1 = float(mf["g1"].mean())
        s = hs.day(tr["period_start"]); e = hs.day(tr["period_end"])
        src_rows.append(pd.DataFrame({"row_id": tr["row_id"], "line": line, "policy_id": tr["policy_id"], "customer_id": tr["customer_id"],
                                      "address_id": tr["address_id"], "s": s, "e": e, "rate": rate, "rate_avg": rate_avg,
                                      "rate_g1": rate * sh1, "rate_g2": rate * (1 - sh1)})[fit])
        src_claims.append(pd.DataFrame({"line": line, "policy_id": m["policy_id"], "customer_id": m["customer_id"],
                                        "address_id": m["address_id"], "report": hs.day(m["report_date"]),
                                        "rep": m["repudiated"].to_numpy(), "g1": m["g1"].to_numpy(), "g2": m["g2"].to_numpy()}))
        # k data: one record per row of the fitting years, claims of the row against completed expected claims
        kdata[line] = pd.DataFrame({"policy_id": tr["policy_id"], "customer_id": tr["customer_id"], "address_id": tr["address_id"],
                                    "yr": tr["yr"], "A": n_all, "E": rate * ecn, "Eavg": rate_avg * ecn,
                                    "A1": m[m["g1"]].groupby("row_id").size().reindex(tr["row_id"]).fillna(0).to_numpy(),
                                    "A2": m[m["g2"]].groupby("row_id").size().reindex(tr["row_id"]).fillna(0).to_numpy(),
                                    "E1": rate * sh1 * ecn, "E2": rate * (1 - sh1) * ecn})[fit]
        rows = tr if sy is None else tr[tr["yr"].isin(fy + (sy,))]
        if sy is None:
            te = build.build_all()[line][1]
            rows = pd.concat([tr, te.assign(yr=2025)], ignore_index=True)
        t = rows[["row_id", "policy_id", "customer_id", "address_id"]].copy()
        t["c"] = np.minimum(hs.day(rows["period_start"]), data_end)
        tgt[line] = t
    src_rows, src_claims = pd.concat(src_rows, ignore_index=True), pd.concat(src_claims, ignore_index=True)
    # k per scope
    kd = pd.concat(kdata.values(), keys=kdata.keys(), names=["line", None]).reset_index(level=0)
    k = {}
    for line in dr.LINES:
        d = kd[kd["line"] == line]
        k[f"P_{line}"] = hs.buhlmann_straub_k(d["policy_id"], d["yr"], d["A"], d["E"])
        k[f"Pavg_{line}"] = hs.buhlmann_straub_k(d["policy_id"], d["yr"], d["A"], d["Eavg"])
        k[f"Pg1_{line}"] = hs.buhlmann_straub_k(d["policy_id"], d["yr"], d["A1"], d["E1"])
        k[f"Pg2_{line}"] = hs.buhlmann_straub_k(d["policy_id"], d["yr"], d["A2"], d["E2"])
    k["C"] = hs.buhlmann_straub_k(kd["customer_id"], kd["yr"], kd["A"], kd["E"])
    dp = kd[kd["line"].isin(PROPERTY)]
    k["Ad_property"] = hs.buhlmann_straub_k(dp["address_id"], dp["yr"], dp["A"], dp["E"])
    dm = kd[kd["line"] == "motor"]
    k["Ad_motor"] = hs.buhlmann_straub_k(dm["address_id"], dm["yr"], dm["A"], dm["E"])
    # scope sums
    rc, fl = ("rate", "rate_avg", "rate_g1", "rate_g2"), ("rep", "g1", "g2")
    allt = pd.concat(tgt.values(), keys=tgt.keys(), names=["line", None]).reset_index(level=0).reset_index(drop=True)
    P = hs.scope_sums(allt, src_rows, src_claims, ["policy_id"], H, rc, fl)
    Call = hs.scope_sums(allt, src_rows, src_claims, ["customer_id"], H, rc, fl)
    C = Call - P
    ad = {}
    for grp, lines in (("property", PROPERTY), ("motor", ("motor",))):
        sr, scl = src_rows[src_rows["line"].isin(lines)], src_claims[src_claims["line"].isin(lines)]
        a_all = hs.scope_sums(allt, sr, scl, ["address_id"], H, ("rate",), ())
        a_own = hs.scope_sums(allt, sr, scl, ["customer_id", "address_id"], H, ("rate",), ())
        ad[grp] = a_all - a_own
    is_prop = allt["line"].isin(PROPERTY).to_numpy()
    Ad = pd.DataFrame({col: np.where(is_prop, ad["property"][col], ad["motor"][col]) for col in ad["property"].columns})
    out = pd.DataFrame({"line": allt["line"], "row_id": allt["row_id"],
                        **{f"P_{c}": P[c] for c in P.columns}, **{f"C_{c}": C[c] for c in ("A", "E_rate", "A_rep", "X")},
                        "Ad_A": Ad["A"], "Ad_E": Ad["E_rate"], "Ad_X": Ad["X"]})
    for c in out.columns:
        if c not in ("line", "row_id"):
            out[c] = out[c].clip(lower=0.0)    # differences of sums: guard against -0.0 rounding
    WORK.mkdir(parents=True, exist_ok=True)
    out.to_parquet(WORK / f"history_{plan_name}.parquet")
    print(f"history {plan_name} done: {len(out)} rows", flush=True)
    return k


def hist_columns(h: pd.DataFrame, k: dict, line: str, edges_band=None) -> pd.DataFrame:
    """Candidate columns from the history sums of one line's rows."""
    kad = k["Ad_property" if line in PROPERTY else "Ad_motor"]["k"]
    mods = {"h_p": hs.mod(h["P_A"], h["P_E_rate"], k[f"P_{line}"]["k"]),
            "h_c": hs.mod(h["C_A"], h["C_E_rate"], k["C"]["k"]),
            "h_a": hs.mod(h["Ad_A"], h["Ad_E"], kad),
            "h_p_avg": hs.mod(h["P_A"], h["P_E_rate_avg"], k[f"Pavg_{line}"]["k"]),
            "h_p_g1": hs.mod(h["P_A_g1"], h["P_E_rate_g1"], k[f"Pg1_{line}"]["k"]),
            "h_p_g2": hs.mod(h["P_A_g2"], h["P_E_rate_g2"], k[f"Pg2_{line}"]["k"])}
    out = pd.DataFrame({c: np.log(x) for c, x in mods.items()}, index=h.index)
    out["mod_p"] = mods["h_p"]
    out["h_r"] = cat(((h["P_A_rep"] + h["C_A_rep"]) > 0).map({True: "yes", False: "no"}))
    out["h_raw_n"] = band(h["P_A"], [-1, 0, 1, 1e9], ["0", "1", "2+"])
    out["h_raw_x"] = band(h["P_X"], [-1, 0, 1.01, 2.01, 3.01, 99], ["0", "to1", "to2", "to3", "3+"])
    return out


def run_line(line: str, ks: dict) -> dict:
    th = load_thresholds()
    nq = int(th["hist_mod_quantiles"])
    tr0 = _train(line)
    res = {}
    for name in ("A", "B", "final"):
        plan = PLANS[name]; fy, sy = plan["fit_years"], plan["score_year"]
        h = pd.read_parquet(WORK / f"history_{name}.parquet")
        h = h[h["line"] == line].drop(columns="line").drop_duplicates("row_id").set_index("row_id")
        keep_years = fy + ((sy,) if sy is not None else ())
        tr = tr0[tr0["yr"].isin(keep_years)].reset_index(drop=True)
        hc = hist_columns(h.reindex(tr["row_id"]).reset_index(drop=True), ks[name], line)
        fit = tr["yr"].isin(fy).to_numpy()
        last = max(fy)
        edges = fit_edges(tr[fit], line)
        F = features(tr, line, edges)
        F["h_d"] = band(tr["prior_claims_5y"], [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"])
        pe = h.reindex(tr["row_id"])["P_E_rate"].to_numpy() > 0
        qe = quantile_edges(hc.loc[fit & pe, "mod_p"], nq)
        F["h_p_band"] = band(hc["mod_p"], qe).where(pe, "none")
        for c in ("h_p", "h_c", "h_a", "h_p_avg", "h_p_g1", "h_p_g2"):
            F[c] = hc[c].to_numpy()
        F["h_r"], F["h_raw_n"], F["h_raw_x"] = hc["h_r"].to_numpy(), hc["h_raw_n"].to_numpy(), hc["h_raw_x"].to_numpy()
        base_cols = list(features(tr.iloc[:5], line, edges).columns)
        e, c = tr["exposure"].to_numpy(), tr["completion"].to_numpy()
        elf = float(tr.loc[fit, "cost"].sum() / tr.loc[fit, "costc"].sum())
        y, ec = tr.loc[fit, "costc"].to_numpy(), e[fit] * c[fit]
        specs = {"V1": [], "H-P": ["h_p"], "H-C": ["h_c"], "H-A": ["h_a"], "H-D": ["h_d"], "H-R": ["h_r"],
                 "P-raw": ["h_raw_n", "h_raw_x"], "P-band": ["h_p_band"], "P-avg": ["h_p_avg"], "P-group": ["h_p_g1", "h_p_g2"],
                 "ALL": ["h_p", "h_c", "h_a", "h_d", "h_r"], "PD": ["h_p", "h_d"]}
        models = {s: TweedieGLM().fit(F.loc[fit, base_cols + cols], y, ec) for s, cols in specs.items()}
        rec = dict(k={kk: vv for kk, vv in ks[name].items()}, elf=elf,
                   coef={s: {r.factor + ("" if r.level == "coefficient" else f"={r.level}"): r.relativity
                             for r in m.relativities().itertuples() if r.factor.startswith("h_")} for s, m in models.items()},
                   share_with_history={sc: float((h.reindex(tr["row_id"])[col].to_numpy()[fit] > 0).mean())
                                       for sc, col in (("P", "P_E_rate"), ("C", "C_E_rate"), ("Ad", "Ad_E"))})
        if sy is None:
            # history on the 2025 test rows (hold-out customers' history is empty until phase 7)
            te = build.build_all()[line][1]
            ht = h.reindex(te["row_id"]).reset_index(drop=True)
            hct = hist_columns(ht, ks[name], line)
            rec["test_history"] = {sc: dict(share_with_history=float((ht[col] > 0).mean()), mean_E=float(ht[col].mean()),
                                            share_with_claim=float((ht[a] > 0).mean()))
                                   for sc, col, a in (("P", "P_E_rate", "P_A"), ("C", "C_E_rate", "C_A"), ("Ad", "Ad_E", "Ad_A"))}
            rec["test_mod_p_quantiles"] = {str(q): float(np.quantile(hct["mod_p"], q)) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
            rec["test_P_E_quantiles"] = {str(q): float(np.quantile(ht["P_E_rate"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
            res[name] = rec
            continue
        sc = (tr["yr"] == sy).to_numpy()
        S = F[sc].copy(); S["yr"] = str(last)
        p = {s: m.rate(S[base_cols + specs[s]]) * elf * e[sc] for s, m in models.items()}
        p["P-mult"] = p["V1"] * hc.loc[sc, "mod_p"].to_numpy()
        f = pd.DataFrame({"line": line, "customer_id": tr.loc[sc, "customer_id"].to_numpy(), "row_id": tr.loc[sc, "row_id"].to_numpy(),
                          "y": tr.loc[sc, "cost"].to_numpy(), "exposure": e[sc], "completion": c[sc], **p})
        f.to_parquet(WORK / f"pred_{line}_{name}{SUFFIX}.parquet")
        res[name] = rec
    print(f"{line} done", flush=True)
    return res


def compare(frames, scoring, ref, cand, lines=None):
    use_c = v.SCORINGS[scoring]["completion"]
    out = {}
    for ln in (lines or list(dr.LINES) + ["all"]):
        d = frames if ln == "all" else frames[frames["line"] == ln]
        yv = d["y"].to_numpy(); cc = d["completion"].to_numpy() if use_c else 1.0
        pa = v.rescale_to_total(d[ref].to_numpy() * cc, yv, by=d["line"].to_numpy())
        pb = v.rescale_to_total(d[cand].to_numpy() * cc, yv, by=d["line"].to_numpy())
        da, db = v.tweedie_unit_deviance(yv, pa), v.tweedie_unit_deviance(yv, pb)
        rec = dict(gain=float((da - db).sum()), d2_ref=v.d2_tweedie(yv, pa, d["exposure"]), d2_cand=v.d2_tweedie(yv, pb, d["exposure"]))
        if scoring == "A":
            rec["boot"] = v.paired_bootstrap_groups(da, db, d["customer_id"].to_numpy(), n_boot=2000)
        out[ln] = rec
    return out


def test(frames, ref, cand, th):
    a, b = compare(frames["A"], "A", ref, cand), compare(frames["B"], "B", ref, cand)
    verdict = {ln: v.keep_verdict({"A": a[ln]["gain"], "B": b[ln]["gain"]}, a[ln]["boot"], int(th["keep_years_better"])) for ln in a}
    return dict(A=a, B=b, verdict=verdict)


def show(label, t):
    print(f"\n{label}:")
    for ln, a in t["A"].items():
        b = t["B"][ln]
        print(f"  {ln:9s} A: D2 {a['d2_ref']:.4%} -> {a['d2_cand']:.4%}, saved {a['gain']:9.1f} [{a['boot']['lo']:9.1f}, {a['boot']['hi']:9.1f}]"
              f" | B: {b['d2_ref']:.4%} -> {b['d2_cand']:.4%}, saved {b['gain']:9.1f} | keep {t['verdict'][ln]['keep']}")


def main():
    th = load_thresholds()
    build.build_all()
    if os.environ.get("P5_SKIP_PREP"):
        import json
        ks = json.loads((WORK / "k.json").read_text())
    else:
        ks = {name: prepare(name) for name in PLANS}
        import json
        (WORK / "k.json").write_text(json.dumps(ks, indent=1))
    if K_EST == "pg":
        # Run 2: the Poisson-gamma moment estimator (design amendment, decisions/phase5_step4_design.md section 7)
        for kk in ks.values():
            for r in kk.values():
                r["k_nonparametric"], r["k"] = r["k"], r["k_poisson_gamma"]
    for name, kk in ks.items():
        print(f"\nk, plan {name}: " + ", ".join(f"{s} {r['k']:.2f} (Poisson-gamma {r['k_poisson_gamma']:.2f})" for s, r in kk.items()))
    res = dict(zip(dr.LINES, Parallel(n_jobs=3)(delayed(run_line)(l, ks) for l in dr.LINES)))
    frames = {s: pd.concat([pd.read_parquet(WORK / f"pred_{l}_{s}{SUFFIX}.parquet") for l in dr.LINES], ignore_index=True) for s in v.SCORINGS}
    tests = {}
    for piece in PIECES + ("ALL",):
        tests[f"{piece}_vs_V1"] = t = test(frames, "V1", piece, th); show(f"{piece} vs V1", t)
    # combination of the pieces that passed: policy history and declared history, each dropped in turn
    for ref in ("H-P", "H-D"):
        tests[f"PD_vs_{ref}"] = t = test(frames, ref, "PD", th); show(f"PD (H-P + H-D) vs {ref}", t)
    for var in P_VARIANTS:
        tests[f"{var}_vs_H-P"] = t = test(frames, "H-P", var, th); show(f"{var} vs H-P", t)
    for name in ("A", "B", "final"):
        for l in dr.LINES:
            print(f"\n{l} {name}: coefficients", {s: {kk: round(vv, 4) for kk, vv in cf.items()} for s, cf in res[l][name]["coef"].items() if cf})
            print(f"   share of fitting rows with history {res[l][name]['share_with_history']}")
            if name == "final":
                print("   test history", res[l][name]["test_history"], "\n   test mod_p quantiles", res[l][name]["test_mod_p_quantiles"])
    write_evidence({f"phase5_step4{SUFFIX}": dict(k=ks, by_line=res, tests=tests, k_estimator=K_EST)}, OUT / f"phase5_step4{SUFFIX}.json")


if __name__ == "__main__":
    main()
