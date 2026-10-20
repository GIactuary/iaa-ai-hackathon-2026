"""Phase 5 step 5: engineered and further features (C8, widened). Design and thresholds written first:
decisions/phase5_step5_design.md.

Per line, on non-hold-out customers, scorings A and B, keep rule throughout:
  1. add-one: each candidate added to B4 (V1 plus the claims-history pieces that passed step 4), or swapped for the
     V1 factor it refines;
  2. full model: B4 plus every add-one passer; drop-one rounds until every remaining candidate passes;
  3. drop-one of each V1 factor from the final model, for information.
Needs the step 4 history files (scripts/phase5_step4.py). Run: .venv/Scripts/python.exe -I scripts/phase5_step5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.phase5_step4 import PLANS, hist_columns
from src import market
from src import build, data_rules as dr, history as hs, validation as v, weather as wx
from src.evidence import write_evidence
from src.glm import TweedieGLM, band, cat, features, fit_edges, quantile_edges
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT, WORK4, WORK = market.OUT, market.WORK / "phase5_step4", market.WORK / "phase5_step5"
# Claims-history pieces in the base, per line: those that passed step 4 (outputs/phase5_step4.json), provisional
# until the actuary's check-in 2 verdict
BASE_PIECES = json.loads((WORK4 / "base_pieces.json").read_text()) if (WORK4 / "base_pieces.json").exists() else {}
HIST_COLS = {"H-P": ["h_p"], "H-C": ["h_c"], "H-A": ["h_a"], "H-D": ["h_d"], "H-R": ["h_r"], "P-group": ["h_p_g1", "h_p_g2"]}


def qband(x, fit_mask, q):
    return band(x, quantile_edges(x[fit_mask], q))


def other_lines_in_force(rows: pd.DataFrame, line: str, all_rows: dict) -> np.ndarray:
    """G9: number of the customer's other lines with a row in force on this row's start date."""
    t = rows[["customer_id", "period_start"]].reset_index()
    n = np.zeros(len(rows))
    for ol, o in all_rows.items():
        if ol == line:
            continue
        m = t.merge(o[["customer_id", "period_start", "period_end"]], on="customer_id", suffixes=("", "_o"))
        m = m[(m["period_start_o"] <= m["period_start"]) & (m["period_end"] >= m["period_start"])]
        hit = np.zeros(len(rows), bool); hit[m["index"].unique()] = True
        n += hit
    return n


def proximity(df):
    d, el = df["distance_to_river_km"], df["elevation_m"]
    low = (d < 1) & (el < 5)
    near = ((d < 1) | (el < 5)) & ~low
    return pd.Series(np.where(low, "low", np.where(near, "near", "far")), index=df.index).where(d.notna(), "blank")


def candidates(tr: pd.DataFrame, line: str, fit: np.ndarray, th, all_rows: dict, weather, fy, score_mask) -> dict:
    """{id: (DataFrame of candidate columns, list of V1 columns it replaces)}."""
    nq, tq = int(th["feat_quantiles"]), int(th["tele_quantiles"])
    age = tr["driver_age"] if line == "motor" else tr["policyholder_age"]
    c = {"G1": ({"g_volx": cat(tr["voluntary_excess"])}, []),
         "G2": ({"g_ren": band(tr["renewal_count"], [-1, 0, 1, 3, 6, 10, 999])}, []),
         "G3": ({"g_quote": band(tr["days_quote_to_inception"], [-1, 0, 3, 7, 14, 21, 999])}, []),
         "G4": ({"missed": band(tr["missed_payments_12m"], [-1, 0, 99], ["0", "1+"])}, ["missed"]),
         "G5": ({"g_hh": cat(tr["household_size"].clip(upper=7))}, []),
         "G6": ({"g_mar": cat(tr["marital_status"])}, []),
         "G7": ({"g_yaa": band(tr["years_at_address"], [-1, 0, 2, 5, 10, 20, 999])}, []),
         "G8": ({"g_occ": cat(tr["occupation"])}, []),
         "G9": ({"g_lines": cat(pd.Series(other_lines_in_force(tr, line, all_rows), index=tr.index).astype(int))}, [])}
    bage = tr["yr"] - tr["year_built"]
    bage_band = band(bage, [-99, 10, 25, 50, 75, 100, 9999])
    if line == "motor":
        lic_age = tr["driver_age"] - tr["years_licensed"]
        tele = qband(tr["telematics_score"], fit & tr["telematics_score"].notna().to_numpy(), tq)
        mil = qband(tr["annual_mileage"].where(tr["annual_mileage"] > 0), fit, nq).where(tr["annual_mileage"] != 0, "zero")
        c |= {"M1": ({"m_ncbgt": cat(tr["ncb_years"] > tr["years_licensed"])}, []),
              "M2": ({"m_u16": cat(lic_age < 16)}, []),
              "M3": ({"m_vpc": qband(tr["vehicle_value_group"] / (tr["engine_size_cc"] / 1000), fit, nq)}, []),
              "M4": ({"telem": tele.where(tr["telematics_score"].notna(), "not_opted")}, ["telem"]),
              "M5": ({"m_agencb": band(tr["driver_age"], [0, 24, 29, 49, 69, 200]) + "|" + band(tr["ncb_years"], [-1, 0, 3, 8, 99])}, []),
              "M6": ({"mileage": mil}, ["mileage"]),
              "M7": ({"m_mm": None}, []),
              "M8": ({"m_ylic": band(tr["years_licensed"], [-1, 0, 1, 2, 4, 9, 19, 999])}, []),
              "M9": ({"m_vage": band(tr["vehicle_age"], [-1, 1, 3, 6, 10, 15, 999])}, []),
              "M10": ({"m_cc": qband(tr["engine_size_cc"], fit, nq)}, []),
              "M11": ({"m_trans": cat(tr["transmission"])}, []),
              "M12": ({"m_seats": cat(tr["seats"])}, []),
              "M13": ({"m_assist": cat(tr["driver_assist"])}, []),
              "M14": ({"m_park": cat(tr["overnight_parking"])}, []),
              "M15": ({"m_colour": cat(tr["vehicle_colour"])}, [])}
    if line != "motor":
        dist = band(tr["distance_to_river_km"], [-1, 0.25, 0.5, 1, 2, 5, 999])
        elev = band(tr["elevation_m"], [-1, 2, 5, 10, 20, 9999])
        score = cat(tr["flood_zone"]) + "|" + proximity(tr)
        spb = qband(tr["sum_insured"] / tr["bedrooms"].clip(lower=1), fit, nq)
    if line == "buildings":
        clay = tr["soil_type"] == "clay"
        nd = wx.normals_leave_own_year_out(tr[["address_id", "yr"]], weather, fy, score_mask)["norm_dry_summer_index"]
        terc = "dry" + band(nd, quantile_edges(nd[fit & clay.fillna(False).to_numpy()], 3))
        c |= {"B1": ({"built": bage_band}, ["built"]),
              "B2": ({"b_spb": spb}, []),
              "B3a": ({"b_dist": dist, "b_elev": elev}, []),
              "B3b": ({"b_fscore": score}, []),
              "B4": ({"b_claydry": terc.where(clay.fillna(False), "not_clay")}, []),
              "B5": ({"b_unocc": band(tr["days_unoccupied"], [-1, 14, 29, 9999], ["0-14", "15-29", "30+"])}, []),
              "B6": ({"b_roof": cat(tr["roof_type"])}, []),
              "B7": ({"b_base": cat(tr["basement"])}, []),
              "B8": ({"b_trees": cat(tr["trees_within_5m"])}, []),
              "B9": ({"b_treeclay": cat(tr["trees_within_5m"].astype("string") + "|" + clay.astype("string"))}, []),
              "B10": ({"b_listed": cat(tr["listed_building"])}, []),
              "B11": ({"b_epc": cat(tr["energy_rating"])}, []),
              "B12": ({"b_solar": cat(tr["solar_panels"])}, []),
              "B13": ({"b_smoke": cat(tr["smoke_alarms"])}, []),
              "B14": ({"b_mort": cat(tr["mortgage"])}, [])}
    if line == "contents":
        c |= {"C1": ({"c_spb": spb}, []),
              "C2": ({"c_beds": cat(tr["bedrooms"].clip(upper=6))}, []),
              "C3": ({"c_bage": bage_band}, []),
              "C4": ({"c_wfh": cat(tr["works_from_home"])}, []),
              "C5": ({"c_lodg": cat(tr["lodgers"])}, []),
              "C6": ({"c_base": cat(tr["basement"])}, []),
              "C7": ({"c_fscore": score}, []),
              "C8": ({"c_listed": cat(tr["listed_building"])}, []),
              "C9": ({"c_roof": cat(tr["roof_type"])}, []),
              "C10": ({"c_epc": cat(tr["energy_rating"])}, [])}
    return {k: (cols, rep) for k, (cols, rep) in c.items()}


def make_model_encoding(tr, fit, base_pred_fit, seed: int, folds: int = 5) -> np.ndarray:
    """M7: log Bühlmann-Straub mod of capped cost against the base model, by make_model. Fitting rows get the mod
    from the other inner folds (grouped by customer); other rows get it from all fitting rows."""
    fi = np.flatnonzero(fit)
    A, E = tr["costc"].to_numpy()[fi], base_pred_fit
    mm, yr = tr["make_model"].to_numpy()[fi], tr["yr"].to_numpy()[fi]
    k = hs.buhlmann_straub_k(mm, yr, A, E)["k"]
    out = np.zeros(len(tr))
    sums = pd.DataFrame({"mm": mm, "A": A, "E": E}).groupby("mm")[["A", "E"]].sum()
    allm = tr["make_model"].map(np.log((sums["A"] + k) / (sums["E"] + k))).fillna(0.0).to_numpy()
    out[:] = allm
    fold = v.make_folds(len(fi), [seed], n_folds=folds, groups=tr["customer_id"].to_numpy()[fi])[seed]
    for a, b in fold:
        s = pd.DataFrame({"mm": mm[a], "A": A[a], "E": E[a]}).groupby("mm")[["A", "E"]].sum()
        enc = pd.Series(np.log((s["A"] + k) / (s["E"] + k)))
        out[fi[b]] = pd.Series(mm[b]).map(enc).fillna(0.0).to_numpy()
    return out, k


def load_plan(line, name, th, all_rows_by_plan, weather):
    plan = PLANS[name]; fy, sy = plan["fit_years"], plan["score_year"]
    tr0 = build.build_all()[line][0]
    tr0 = tr0[~tr0["holdout"]].reset_index(drop=True)
    ks = json.loads((WORK4 / "k.json").read_text())[name]
    for r in ks.values():                     # the Poisson-gamma k (step 4 design amendment, section 7)
        r["k"] = r["k_poisson_gamma"]
    h = pd.read_parquet(WORK4 / f"history_{name}.parquet")
    h = h[h["line"] == line].drop(columns="line").drop_duplicates("row_id").set_index("row_id")
    years = fy + ((sy,) if sy is not None else ())
    tr = tr0[tr0["yr"].isin(years)].reset_index(drop=True)
    fit = tr["yr"].isin(fy).to_numpy()
    score_mask = (tr["yr"] == sy).to_numpy() if sy is not None else np.zeros(len(tr), bool)
    edges = fit_edges(tr[fit], line)
    F = features(tr, line, edges)
    base_cols = list(F.columns)
    hc = hist_columns(h.reindex(tr["row_id"]).reset_index(drop=True), ks, line)
    hc["h_d"] = band(tr["prior_claims_5y"], [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"]).to_numpy()
    hcols = [c for p in BASE_PIECES.get(line, []) for c in HIST_COLS[p]]
    for col in hcols:
        F[col] = hc[col].to_numpy()
    cands = candidates(tr, line, fit, th, all_rows_by_plan, weather, fy, score_mask)
    for cid, (cols, rep) in cands.items():
        for col, x in cols.items():
            if x is not None:
                F[f"{cid}:{col}"] = pd.Series(x, index=tr.index).to_numpy()
    return dict(tr=tr, F=F, fit=fit, sc=score_mask, base_cols=base_cols + hcols, cands=cands, last=max(fy),
                elf=float(tr.loc[fit, "cost"].sum() / tr.loc[fit, "costc"].sum()))


def spec_columns(P, active: dict[str, list[str]], drop_base=()):
    """Columns of a model: the base (less replaced or dropped V1 factors) plus active candidates."""
    replaced = {r for cid in active for r in P["cands"][cid][1]} | set(drop_base)
    cols = [c for c in P["base_cols"] if c not in replaced]
    for cid in active:
        cols += [f"{cid}:{col}" for col in P["cands"][cid][0]]
    return cols


def fit_score(P, cols):
    tr, F, fit, sc = P["tr"], P["F"], P["fit"], P["sc"]
    e, c = tr["exposure"].to_numpy(), tr["completion"].to_numpy()
    G = F[cols].rename(columns=lambda x: x.split(":")[-1])
    m = TweedieGLM().fit(G[fit], tr.loc[fit, "costc"], e[fit] * c[fit])
    S = G[sc].copy(); S["yr"] = str(P["last"])
    scored = m.rate(S) * P["elf"] * e[sc] if sc.any() else np.zeros(0)
    return m, scored, m.rate(G[fit]) * e[fit] * c[fit]


def paired(P, pa, pb, scoring, boot):
    tr, sc = P["tr"], P["sc"]
    y = tr.loc[sc, "cost"].to_numpy()
    cc = tr.loc[sc, "completion"].to_numpy() if v.SCORINGS[scoring]["completion"] else 1.0
    a, b = v.rescale_to_total(pa * cc, y), v.rescale_to_total(pb * cc, y)
    da, db = v.tweedie_unit_deviance(y, a), v.tweedie_unit_deviance(y, b)
    rec = dict(gain=float((da - db).sum()), d2_ref=v.d2_tweedie(y, a, tr.loc[sc, "exposure"]), d2_cand=v.d2_tweedie(y, b, tr.loc[sc, "exposure"]),
               dev_a=da, dev_b=db)
    if boot:
        rec["boot"] = v.paired_bootstrap_groups(da, db, tr.loc[sc, "customer_id"].to_numpy(), n_boot=2000)
    return rec


def run_line(line):
    th = load_thresholds()
    weather = dr.read_raw("weather")
    full_rows = {l: build.build_all()[l][0] for l in dr.LINES}
    full_rows = {l: d[~d["holdout"]] for l, d in full_rows.items()}
    Ps = {s: load_plan(line, s, th, full_rows, weather) for s in v.SCORINGS}
    # M7 needs the base model's fitted cost on the fitting rows
    if line == "motor":
        for s, P in Ps.items():
            _, _, fitted = fit_score(P, P["base_cols"])
            enc, kmm = make_model_encoding(P["tr"], P["fit"], fitted, int(th["split_seed_1"]))
            P["F"]["M7:m_mm"] = enc
            P["cands"]["M7"] = ({"m_mm": None}, [])
            P["k_make_model"] = kmm
    cache = {}

    def preds(key, active, drop_base=()):
        if key not in cache:
            cache[key] = {s: fit_score(P, spec_columns(P, active, drop_base))[1] for s, P in Ps.items()}
        return cache[key]

    def keep_test(ref, cand):
        A = paired(Ps["A"], ref["A"], cand["A"], "A", True)
        B = paired(Ps["B"], ref["B"], cand["B"], "B", False)
        verdict = v.keep_verdict({"A": A["gain"], "B": B["gain"]}, A["boot"], int(th["keep_years_better"]))
        strip = lambda r: {k: x for k, x in r.items() if k not in ("dev_a", "dev_b")}
        return dict(A=strip(A), B=strip(B), verdict=verdict, dev={"A": (A["dev_a"], A["dev_b"]), "B": (B["dev_a"], B["dev_b"])})

    base = preds("base", {})
    ids = list(Ps["A"]["cands"])
    add_one = {cid: keep_test(base, preds(("add", cid), {cid: None})) for cid in ids}
    passed = [cid for cid in ids if add_one[cid]["verdict"]["keep"]]
    # B3a and B3b are alternatives: keep the one that saves more on A if both pass
    for a_, b_ in (("B3a", "B3b"),):
        if a_ in passed and b_ in passed:
            passed.remove(a_ if add_one[a_]["A"]["gain"] < add_one[b_]["A"]["gain"] else b_)
    rounds, active = [], list(passed)
    for r in range(int(th["drop_one_max_rounds"])):
        full = preds(("full", tuple(active)), dict.fromkeys(active))
        res = {cid: keep_test(preds(("full", tuple(x for x in active if x != cid)), dict.fromkeys(x for x in active if x != cid)), full)
               for cid in active}
        fails = [cid for cid in active if not res[cid]["verdict"]["keep"]]
        rounds.append(dict(active=list(active), results=res, fails=fails))
        if not fails:
            break
        active = [cid for cid in active if cid not in fails]
    final = preds(("full", tuple(active)), dict.fromkeys(active))
    base_drop = {}
    for col in Ps["A"]["base_cols"]:
        if col == "yr":
            continue
        base_drop[col] = keep_test(preds(("dropbase", tuple(active), col), dict.fromkeys(active), (col,)), final)
    final_vs_base = keep_test(base, final)
    # final-fit relativities for the kept candidates
    Pf = load_plan(line, "final", th, full_rows, weather)
    if line == "motor" and "M7" in active:
        _, _, fitted = fit_score(Pf, Pf["base_cols"])
        Pf["F"]["M7:m_mm"], Pf["k_make_model"] = make_model_encoding(Pf["tr"], Pf["fit"], fitted, int(th["split_seed_1"]))
    mf = fit_score(Pf, spec_columns(Pf, dict.fromkeys(active)))[0]
    rel = mf.relativities()
    np.savez_compressed(WORK / f"dev_{line}.npz", **{f"{s}_{w}": final_vs_base["dev"][s][i] for s in ("A", "B") for i, w in enumerate(("base", "final"))})
    strip = lambda t: {k: x for k, x in t.items() if k != "dev"}
    out = dict(add_one={k: strip(t) for k, t in add_one.items()}, passed_add_one=passed,
               rounds=[dict(active=r["active"], fails=r["fails"], results={k: strip(t) for k, t in r["results"].items()}) for r in rounds],
               kept=active, base_factor_drop={k: strip(t) for k, t in base_drop.items()}, final_vs_base=strip(final_vs_base),
               relativities_final=rel.to_dict("records"), base_pieces=BASE_PIECES.get(line, []),
               k_make_model={s: P.get("k_make_model") for s, P in Ps.items()})
    print(f"{line} done: kept {active}", flush=True)
    return out


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    print("base pieces:", BASE_PIECES)
    res = dict(zip(dr.LINES, Parallel(n_jobs=3)(delayed(run_line)(l) for l in dr.LINES)))
    for l, r in res.items():
        print(f"\n== {l}: base pieces {r['base_pieces']}")
        print("add-one (A saved [interval] | B saved | keep):")
        for cid, t in r["add_one"].items():
            a = t["A"]
            print(f"  {cid:5s} {a['gain']:9.1f} [{a['boot']['lo']:9.1f}, {a['boot']['hi']:9.1f}] | {t['B']['gain']:9.1f} | {t['verdict']['keep']}")
        for i, rd in enumerate(r["rounds"]):
            print(f"drop-one round {i + 1}: fails {rd['fails']}")
            for cid, t in rd["results"].items():
                a = t["A"]
                print(f"  {cid:5s} {a['gain']:9.1f} [{a['boot']['lo']:9.1f}, {a['boot']['hi']:9.1f}] | {t['B']['gain']:9.1f} | {t['verdict']['keep']}")
        fb = r["final_vs_base"]
        print(f"kept {r['kept']}; final vs base: A {fb['A']['d2_ref']:.4%} -> {fb['A']['d2_cand']:.4%}, saved {fb['A']['gain']:.1f} "
              f"[{fb['A']['boot']['lo']:.1f}, {fb['A']['boot']['hi']:.1f}]; B {fb['B']['d2_ref']:.4%} -> {fb['B']['d2_cand']:.4%}, saved {fb['B']['gain']:.1f}")
        print("V1 factor drop-one from the final model:")
        for col, t in r["base_factor_drop"].items():
            a = t["A"]
            print(f"  {col:8s} {a['gain']:9.1f} [{a['boot']['lo']:9.1f}, {a['boot']['hi']:9.1f}] | {t['B']['gain']:9.1f} | {t['verdict']['keep']}")
    # combined (all lines) final against base
    comb = {}
    for s in ("A", "B"):
        d = [np.load(WORK / f"dev_{l}.npz") for l in dr.LINES]
        comb[s] = float(sum((x[f"{s}_base"] - x[f"{s}_final"]).sum() for x in d))
    print("\ncombined deviance saved, final against base:", comb)
    write_evidence({"phase5_step5": dict(by_line=res, combined_gain=comb)}, OUT / "phase5_step5.json")


if __name__ == "__main__":
    main()
