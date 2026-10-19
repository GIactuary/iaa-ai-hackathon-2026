"""Phase 5 step 7: model families (P7, C5, J-model) on the working base W2. Design and thresholds written first:
decisions/phase5_step7_design.md.

Per line, non-hold-out customers:
  1. tune each candidate on the five customer-grouped inner folds (seed 11) of scoring A's fitting years:
     power over the grid, then the other settings; pick the strongest candidate per family by inner-fold deviance;
  2. report the chosen settings' spread on seeds 22 and 33 against the W2 GLM;
  3. refit each family's model out of time (A and B), with and without the carried candidates, and in motor as
     frequency x severity per peril; save the scored rows for the tests and for step 8.
Run one line:  .venv/Scripts/python.exe -I scripts/phase5_step7.py motor      (P7_THREADS sets the threads, default 4)
Run the tests: .venv/Scripts/python.exe -I scripts/phase5_step7.py tests
P7_SMOKE=1 shrinks every grid to one setting and two folds, for a check that the code runs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.phase5_l8_check import DROP, KEPT
from scripts.phase5_step4 import hist_columns
from scripts.phase5_step5 import load_plan, spec_columns
from scripts.phase5_step6 import test
from src import build, data_rules as dr, level as lv, validation as v
from src.evidence import write_evidence
from src.families import EBM, LGB, Cat, GlumGLM, as_category, as_string
from src.glm import TweedieGLM
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT, WORK, WORK4 = ROOT / "outputs", ROOT / "data" / "work" / "phase5_step7", ROOT / "data" / "work" / "phase5_step4"
THREADS = int(os.environ.get("P7_THREADS", "4"))
SMOKE = bool(os.environ.get("P7_SMOKE"))

# Raw fields behind the W2 GLM columns, for the tree families and the EBM (design section 1)
RAW_NUM = {"all": ["missed_payments_12m", "prior_claims_5y"],
           "motor": ["driver_age", "ncb_years", "vehicle_value_group", "annual_mileage", "licence_points", "voluntary_excess", "vehicle_age"],
           "buildings": ["policyholder_age", "sum_insured", "year_built"],
           "contents": ["policyholder_age", "sum_insured", "high_value_items"]}
RAW_CAT = {"all": ["region", "urban_density", "payment_method", "credit_band", "channel"],
           "motor": ["new_business", "cover_type", "vehicle_use", "fuel_type", "named_young_driver", "telematics_opt_in"],
           "buildings": ["new_business", "flood_zone", "accidental_damage_cover", "construction_type", "soil_type", "occupancy", "trees_within_5m"],
           "contents": ["flood_zone", "accidental_damage_cover", "tenure", "security_level", "bicycle_cover"]}
# Carried candidates (L4, L7): tree form and GLM form
CARRIED = {"motor": {"hist_c": (["h_c"], ["h_c"]), "hist_a": (["h_a"], ["h_a"]), "hist_r": (["h_r"], ["h_r"]),
                     "ylic": (["years_licensed"], ["M8:m_ylic"])},
           "buildings": {"hist_c": (["h_c"], ["h_c"]), "hist_a": (["h_a"], ["h_a"]), "hist_r": (["h_r"], ["h_r"]),
                         "volx": (["voluntary_excess"], ["G1:g_volx"]), "flood": (["distance_to_river_km", "elevation_m"], ["B3a:b_dist", "B3a:b_elev"])},
           "contents": {"hist_c": (["h_c"], ["h_c"]), "hist_a": (["h_a"], ["h_a"]), "hist_r": (["h_r"], ["h_r"]),
                        "volx": (["voluntary_excess"], ["G1:g_volx"])}}
# Spline variant of the elastic-net GLM: W2 band column -> raw field (log for sum insured)
SPLINES = {"motor": {"age": "driver_age", "ncb": "ncb_years", "vvg": "vehicle_value_group", "mileage": "annual_mileage"},
           "buildings": {"age": "policyholder_age", "si": "sum_insured", "built": "year_built"},
           "contents": {"age": "policyholder_age", "si": "sum_insured"}}
POWERS = [round(x, 1) for x in np.arange(1.1, 1.95, 0.1)]
GRIDS = {"glum": [dict(alpha=a, l1_ratio=l) for a in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2) for l in (0.0, 0.5, 1.0)],
         "lgb": [dict(leaves=a, min_leaf=b) for a in (15, 31, 63) for b in (100, 400, 1600)],
         "cat": [dict(depth=d) for d in (4, 6, 8)],
         "ebm": [dict(interactions=k) for k in (0, 5, 15)],
         "f4": [dict(leaves=a, min_leaf=b) for a in (7, 15, 31) for b in (400, 1600)]}
MIDDLE = {"glum": dict(alpha=1e-4, l1_ratio=0.5), "lgb": dict(leaves=31, min_leaf=400), "cat": dict(depth=6),
          "ebm": dict(interactions=5), "f4": dict(leaves=15, min_leaf=400)}
FAMILY_OF = {"glm_w2": "F1", "glum": "F1", "glum_spline": "F1", "lgb": "F2", "cat": "F2", "ebm": "F3", "f4": "F4"}
if SMOKE:
    WORK = WORK.with_name("phase5_step7_smoke")
    POWERS = [1.5]
    GRIDS = {k: [MIDDLE[k]] for k in GRIDS}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ data
def load(line: str, name: str) -> dict:
    th = load_thresholds()
    weather = dr.read_raw("weather")
    full = {l: build.build_all()[l][0] for l in dr.LINES}
    full = {l: d[~d["holdout"]] for l, d in full.items()}
    P = load_plan(line, name, th, full, weather)
    tr, fit, sc, last = P["tr"], P["fit"], P["sc"], P["last"]
    w2 = spec_columns(P, dict.fromkeys(KEPT[line]), DROP[line])
    ks = json.loads((WORK4 / "k.json").read_text())[name]
    for r in ks.values():
        r["k"] = r["k_poisson_gamma"]
    h = pd.read_parquet(WORK4 / f"history_{name}.parquet")
    h = h[h["line"] == line].drop(columns="line").drop_duplicates("row_id").set_index("row_id")
    hc = hist_columns(h.reindex(tr["row_id"]).reset_index(drop=True), ks, line)
    # GLM frame: W2 columns plus the carried GLM columns
    gl_extra = sorted({c for t, g in CARRIED[line].values() for c in g if ":" in c})
    G = P["F"][w2 + gl_extra].copy()
    for c in ("h_c", "h_a"):
        G[c] = hc[c].to_numpy()
    G["h_r"] = hc["h_r"].to_numpy()
    G = G.rename(columns=lambda x: x.split(":")[-1])
    G.loc[sc, "yr"] = str(last)
    gcat = [c for c in G.columns if not pd.api.types.is_float_dtype(G[c])]
    G = as_category(G, gcat)
    # tree frame: raw fields, policy history log(mod), year, plus carried raw fields
    num = RAW_NUM["all"] + RAW_NUM[line]
    cat = RAW_CAT["all"] + RAW_CAT[line]
    T = tr[num].astype(float).copy()
    T["h_p"] = P["F"]["h_p"].to_numpy()
    T["yr"] = tr["yr"].astype(float).to_numpy()
    T.loc[sc, "yr"] = float(last)
    for c in cat:
        T[c] = tr[c]
    for _, (tcols, _g) in CARRIED[line].items():
        for c in tcols:
            if c in ("h_c", "h_a"):
                T[c] = hc[c].to_numpy()
            elif c == "h_r":
                T[c] = hc["h_r"].to_numpy()
            else:
                T[c] = tr[c].astype(float).to_numpy()
    tcat = cat + ["h_r"]
    return dict(P=P, tr=tr, fit=fit, sc=sc, last=last, w2=[c.split(":")[-1] for c in w2], G=G,
                T=as_category(T, tcat), Ts=as_string(T, tcat), tcat=tcat,
                tbase=num + ["h_p", "yr"] + cat, elf=P["elf"], perils=lv.perils_of(tr))


def spline_frame(D: dict, fit_mask: np.ndarray, knots: int) -> pd.DataFrame:
    """W2 GLM columns with the banded fields replaced by cubic B-spline bases (linear beyond the end knots)."""
    from sklearn.preprocessing import SplineTransformer
    G = D["G"][D["w2"]].copy()
    for band, raw in SPLINES[D["line"]].items():
        x = D["tr"][raw].astype(float)
        x = np.log(x.clip(lower=1)) if raw == "sum_insured" else x
        x = x.fillna(x[fit_mask].median()).to_numpy().reshape(-1, 1)
        st = SplineTransformer(n_knots=knots, degree=3, knots="quantile", extrapolation="linear", include_bias=False).fit(x[fit_mask])
        B = st.transform(x)
        G = G.drop(columns=band)
        for j in range(B.shape[1]):
            G[f"{band}_s{j}"] = B[:, j]
    return G


# ------------------------------------------------------------------------------------------------ fitting
def make(kind: str, power: float, setting: dict, th, D, objective="tweedie", n_trees=None):
    nt = int(n_trees or (200 if SMOKE else th["max_trees"]))
    if kind in ("glum", "glum_spline"):
        return GlumGLM(objective, power, **setting)
    if kind in ("lgb", "f4"):
        return LGB(objective, power, lr=th["gbm_learning_rate"], n_trees=nt, threads=THREADS, **setting)
    if kind == "cat":
        return Cat(objective, power, lr=th["cat_learning_rate"], n_trees=nt, threads=THREADS, cat_features=D["tcat_used"], **setting)
    if kind == "ebm":
        return EBM(objective, power, outer_bags=th["ebm_outer_bags"], threads=THREADS, **setting)
    raise ValueError(kind)


def frame_for(kind: str, D: dict, cols=None, spline=None):
    if kind == "glum":
        X = D["G"][cols or D["w2"]]
    elif kind == "glum_spline":
        X = spline
    elif kind == "ebm" or kind == "cat":
        X = D["Ts"][cols or D["tbase"]]
    else:
        X = D["T"][cols or D["tbase"]]
    D["tcat_used"] = [c for c in X.columns if c in D["tcat"]]
    return X


def fold_dev(y, pred) -> float:
    return float(v.tweedie_unit_deviance(y, v.rescale_to_total(pred, y)).sum())


def glm_folds(D, folds, fi):
    """The W2 GLM inside each inner fold: its deviance on the held-out fold and its log rate (the F4 start)."""
    tr, G = D["tr"], D["G"]
    y, ec = tr["costc"].to_numpy()[fi], (tr["exposure"] * tr["completion"]).to_numpy()[fi]
    out = []
    for a, b in folds:
        m = TweedieGLM().fit(G[D["w2"]].iloc[fi[a]], y[a], ec[a])
        ra, rb = m.rate(G[D["w2"]].iloc[fi[a]]), m.rate(G[D["w2"]].iloc[fi[b]])
        out.append(dict(init_a=np.log(ra), init_b=np.log(rb), dev=fold_dev(y[b], rb * ec[b])))
    return out


def cv(kind, power, setting, th, D, X, folds, fi, gf=None, es=True):
    tr = D["tr"]
    y, ec = tr["costc"].to_numpy()[fi], (tr["exposure"] * tr["completion"]).to_numpy()[fi]
    devs, iters = [], []
    for k, (a, b) in enumerate(folds):
        ia = gf[k]["init_a"] if kind == "f4" else None
        ib = gf[k]["init_b"] if kind == "f4" else None
        m = make(kind, power, setting, th, D)
        val = (X.iloc[fi[b]], y[b], ec[b], None, ib) if es and kind in ("lgb", "f4", "cat") else None
        m.fit(X.iloc[fi[a]], y[a], ec[a], init=ia, val=val, es_rounds=int(th["early_stop_rounds"]))
        devs.append(fold_dev(y[b], m.rate(X.iloc[fi[b]], init=ib) * ec[b]))
        iters.append(m.best_iter_)
    return dict(dev=float(np.mean(devs)), devs=devs, iters=iters)


def tune(kind, th, D, X, folds, fi, gf):
    """Stage 1: power at the middle setting; stage 2: the grid at the best power."""
    res = []
    for p in POWERS:
        r = cv(kind, p, MIDDLE[kind], th, D, X, folds, fi, gf)
        res.append(dict(power=p, setting=MIDDLE[kind], **r)); log(D["line"], kind, "power", p, round(r["dev"], 1))
    bp = min(res, key=lambda r: r["dev"])["power"]
    for s in GRIDS[kind]:
        if s == MIDDLE[kind]:
            continue
        r = cv(kind, bp, s, th, D, X, folds, fi, gf)
        res.append(dict(power=bp, setting=s, **r)); log(D["line"], kind, "setting", s, round(r["dev"], 1))
    best = min(res, key=lambda r: r["dev"])
    return dict(trials=res, best=best)


# ------------------------------------------------------------------------------------------------ line run
def run_line(line: str) -> None:
    th = load_thresholds()
    WORK.mkdir(parents=True, exist_ok=True)
    D = load(line, "A"); D["line"] = line
    fi = np.flatnonzero(D["fit"])
    n_folds = 2 if SMOKE else int(th["inner_folds"])
    seeds = [int(th["split_seed_1"]), int(th["split_seed_2"]), int(th["split_seed_3"])]
    cust = pd.factorize(D["tr"]["customer_id"].to_numpy()[fi])[0]   # integer codes: string ids made GroupKFold take minutes
    folds = v.make_folds(len(fi), seeds, n_folds=n_folds, groups=cust)
    s1 = seeds[0]
    gf = {s: glm_folds(D, folds[s], fi) for s in ([s1] if SMOKE else seeds)}
    log(line, "W2 GLM inner deviance", round(float(np.mean([g["dev"] for g in gf[s1]])), 1))
    spline = spline_frame(D, D["fit"], int(th["spline_knots"]))
    tuned = {}
    for kind in ("lgb", "f4", "glum", "ebm", "cat"):
        path = WORK / f"tune_{line}_{kind}.json"
        if path.exists():
            tuned[kind] = json.loads(path.read_text()); continue
        X = frame_for(kind, D)
        tuned[kind] = tune(kind, th, D, X, folds[s1], fi, gf[s1])
        path.write_text(json.dumps(tuned[kind], default=float))
    # spline variant of the elastic-net GLM at the best glum setting
    path = WORK / f"tune_{line}_glum_spline.json"
    if path.exists():
        tuned["glum_spline"] = json.loads(path.read_text())
    else:
        b = tuned["glum"]["best"]
        r = cv("glum_spline", b["power"], b["setting"], th, D, spline, folds[s1], fi)
        tuned["glum_spline"] = dict(trials=[dict(power=b["power"], setting=b["setting"], **r)], best=dict(power=b["power"], setting=b["setting"], **r))
        path.write_text(json.dumps(tuned["glum_spline"], default=float))
    tuned["glm_w2"] = dict(best=dict(power=1.5, setting={}, dev=float(np.mean([g["dev"] for g in gf[s1]])), devs=[g["dev"] for g in gf[s1]]))
    # strongest candidate per family, by inner-fold deviance (J-model (a))
    chosen = {}
    for fam in ("F1", "F2", "F3", "F4"):
        kinds = [k for k, f in FAMILY_OF.items() if f == fam]
        chosen[fam] = min(kinds, key=lambda k: tuned[k]["best"]["dev"])
    log(line, "chosen", chosen)
    # spread on seeds 22 and 33 for each family's choice
    spread = {}
    path = WORK / f"spread_{line}.json"
    if path.exists():
        spread = json.loads(path.read_text())
    elif not SMOKE:
        for fam, kind in chosen.items():
            b = tuned[kind]["best"]
            spread[fam] = {}
            for s in seeds:
                if kind == "glm_w2":
                    spread[fam][str(s)] = dict(dev=float(np.mean([g["dev"] for g in gf[s]])))
                    continue
                X = spline if kind == "glum_spline" else frame_for(kind, D)
                spread[fam][str(s)] = cv(kind, b["power"], b["setting"], th, D, X, folds[s], fi, gf[s])
            spread[fam]["glm"] = {str(s): float(np.mean([g["dev"] for g in gf[s]])) for s in seeds}
            log(line, fam, "spread", {s: round(x["dev"] - spread[fam]["glm"][s], 1) for s, x in spread[fam].items() if s != "glm"})
        path.write_text(json.dumps(spread, default=float))
    oot = out_of_time(line, th, tuned, chosen)
    (WORK / f"result_{line}.json").write_text(json.dumps(dict(tuned=tuned, chosen=chosen, spread=spread, oot=oot), default=float))
    log(line, "done")


def n_trees_of(tuned, kind):
    it = [i for i in tuned[kind]["best"].get("iters", []) if i]
    return int(round(np.mean(it))) if it else None


def out_of_time(line, th, tuned, chosen) -> dict:
    """Refit each family's model (and variants) on each scoring's fitting years; save the scored rows."""
    info = {}
    for name in ("A", "B"):
        D = load(line, name); D["line"] = line
        tr, fit, sc, last = D["tr"], D["fit"], D["sc"], D["last"]
        e, c = tr["exposure"].to_numpy(), tr["completion"].to_numpy()
        ec = e * c
        y = tr["costc"].to_numpy()
        use_c = v.SCORINGS[name]["completion"]
        cs = c[sc] if use_c else 1.0
        f = pd.DataFrame({"line": line, "customer_id": tr.loc[sc, "customer_id"].to_numpy(), "y": tr.loc[sc, "cost"].to_numpy(),
                          "exposure": e[sc]})
        g = TweedieGLM().fit(D["G"][D["w2"]][fit], y[fit], ec[fit])
        init_fit, init_sc = np.log(g.rate(D["G"][D["w2"]][fit])), np.log(g.rate(D["G"][D["w2"]][sc]))
        f["ref"] = g.rate(D["G"][D["w2"]][sc]) * D["elf"] * e[sc] * cs
        spline = spline_frame(D, fit, int(th["spline_knots"]))
        models = {}

        def fit_pred(kind, cols=None):
            b = tuned[kind]["best"]
            X = spline if kind == "glum_spline" else frame_for(kind, D, cols)
            m = make(kind, b["power"], b["setting"], th, D, n_trees=n_trees_of(tuned, kind))
            ia, ib = (init_fit, init_sc) if kind == "f4" else (None, None)
            m.fit(X[fit], y[fit], ec[fit], init=ia)
            return m, m.rate(X[sc], init=ib) * D["elf"] * e[sc] * cs, X

        for fam, kind in chosen.items():
            if kind == "glm_w2":
                f[fam] = f["ref"]; continue
            m, p, X = fit_pred(kind)
            f[fam] = p; models[fam] = (kind, m, X)
            log(line, name, fam, kind, "fitted")
        # every candidate's own out-of-time prediction, for information
        for kind in ("glum", "glum_spline", "lgb", "cat", "ebm", "f4"):
            if kind in chosen.values():
                f[f"cand:{kind}"] = f[[fm for fm, k in chosen.items() if k == kind][0]]
                continue
            f[f"cand:{kind}"] = fit_pred(kind)[1]
        # carried candidates: the group, then one at a time for LightGBM and the elastic-net GLM
        for fam, kind in chosen.items():
            if kind in ("glm_w2", "glum_spline"):
                base_kind = "glum"
            else:
                base_kind = kind
            gl = base_kind == "glum"
            base_cols = D["w2"] if gl else D["tbase"]
            groups = {"all": [c for t, gg in CARRIED[line].values() for c in (gg if gl else t)]}
            if base_kind in ("lgb", "glum"):
                groups |= {k: (gg if gl else t) for k, (t, gg) in CARRIED[line].items()}
            f[f"{fam}|base"] = f[f"cand:{base_kind}"]
            for gname, add in groups.items():
                add = [a.split(":")[-1] for a in add]
                f[f"{fam}+{gname}"] = fit_pred(base_kind, base_cols + add)[1]
                log(line, name, fam, "carried", gname, "fitted")
        # motor: frequency x severity per peril (M1 (a)) for the elastic-net GLM, LightGBM and the EBM
        if line == "motor":
            for kind in ("glum", "lgb", "ebm"):
                f[f"FS:{kind}"] = freq_sev(kind, th, D, tuned, name)
                log(line, name, "FS", kind, "fitted")
        if name == "A":
            info["explain"] = explain(line, D, models)
        f.to_parquet(WORK / f"{line}_{name}.parquet")
    return info


def freq_sev(kind, th, D, tuned, name):
    tr, fit, sc, last = D["tr"], D["fit"], D["sc"], D["last"]
    e = tr["exposure"].to_numpy()
    use_c = v.SCORINGS[name]["completion"]
    b = tuned["glum" if kind == "glum" else kind]["best"]
    X = frame_for(kind, D)
    total = np.zeros(int(sc.sum()))
    es_path = WORK / f"fs_trees_{kind}.json"
    trees = json.loads(es_path.read_text()) if es_path.exists() else {}
    for p in D["perils"]:
        ecp = e * tr[f"completion_{p}"].to_numpy()
        n = tr[f"n_{p}"].to_numpy().astype(float)
        cm = fit & (n > 0)
        parts = []
        for obj, rows, yy, ee, ww in (("poisson", fit, n, ecp, None), ("gamma", cm, tr[f"costc_{p}"].to_numpy(), n, n)):
            key = f"{p}|{obj}"
            setting = dict(b["setting"])
            if kind == "lgb" and key not in trees:
                # trees by early stopping on fold 1 of seed 11 of scoring A's fitting rows
                ri = np.flatnonzero(rows)
                fold = v.make_folds(len(ri), [int(th["split_seed_1"])], groups=pd.factorize(tr["customer_id"].to_numpy()[ri])[0])[int(th["split_seed_1"])][0]
                a_, b_ = ri[fold[0]], ri[fold[1]]
                m = make(kind, b["power"], setting, th, D, objective=obj)
                m.fit(X.iloc[a_], yy[a_], ee[a_], weight=None if ww is None else ww[a_],
                      val=(X.iloc[b_], yy[b_], ee[b_], None if ww is None else ww[b_], None), es_rounds=int(th["early_stop_rounds"]))
                trees[key] = m.best_iter_
                es_path.write_text(json.dumps(trees))
            m = make(kind, b["power"], setting, th, D, objective=obj, n_trees=trees.get(key))
            m.fit(X[rows], yy[rows], ee[rows], weight=None if ww is None else ww[rows])
            parts.append(m.rate(X[sc]))
        cp = tr.loc[sc, f"completion_{p}"].to_numpy() if use_c else 1.0
        total += parts[0] * parts[1] * e[sc] * cp
    return total * D["elf"]


def explain(line, D, models) -> dict:
    """Largest inputs per family model and the shape of the main size effect, on 20,000 scored rows of scoring A."""
    rng = np.random.default_rng(0)
    sc_idx = np.flatnonzero(D["sc"])
    take = rng.choice(sc_idx, size=min(20000, len(sc_idx)), replace=False)
    out = {}
    shape_var = {"motor": ["driver_age", "vehicle_value_group"], "buildings": ["sum_insured", "policyholder_age"],
                 "contents": ["sum_insured", "policyholder_age"]}[line]
    for fam, (kind, m, X) in models.items():
        Xs = X.iloc[take]
        r = {}
        if kind in ("lgb", "f4"):
            ctb = m.contrib(Xs)[:, :-1]
            imp = dict(zip(X.columns, np.abs(ctb).mean(axis=0)))
        elif kind == "ebm":
            imp = m.importances()
        elif kind == "cat":
            from catboost import Pool
            sv = m.model_.get_feature_importance(Pool(Xs.iloc[:5000], cat_features=m.cat_features), type="ShapValues")[:, :-1]
            imp = dict(zip(X.columns, np.abs(sv).mean(axis=0)))
        else:
            imp = {}
        r["top"] = sorted(((k, float(x)) for k, x in imp.items()), key=lambda t: -t[1])[:8]
        if kind in ("lgb", "f4", "ebm", "cat"):
            r["shape"] = {}
            sub = Xs.iloc[:3000].copy()
            for var in shape_var:
                if var not in X.columns:
                    continue
                grid = np.unique(np.nanquantile(X[var].astype(float), np.linspace(0.02, 0.98, 9)))
                vals = []
                for gv in grid:
                    s2 = sub.copy(); s2[var] = gv
                    init = np.zeros(len(s2)) if kind == "f4" else None   # F4: the boosted part on top of the GLM
                    vals.append(float(np.mean(np.log(m.rate(s2, init=init)))))
                vals = np.array(vals)
                r["shape"][var] = dict(grid=grid.tolist(), relativity=np.exp(vals - vals[len(vals) // 2]).tolist())
        out[fam] = r
    return out


# ------------------------------------------------------------------------------------------------ tests
def run_tests() -> None:
    th = load_thresholds()
    frames = {s: pd.concat([pd.read_parquet(WORK / f"{l}_{s}.parquet") for l in dr.LINES], ignore_index=True) for s in v.SCORINGS}
    res = {l: json.loads((WORK / f"result_{l}.json").read_text()) for l in dr.LINES}
    tests = {}
    for fam in ("F1", "F2", "F3", "F4"):
        tests[f"{fam} vs W2 GLM"] = test(frames, "ref", fam, th)
    for kind in ("glum", "glum_spline", "lgb", "cat", "ebm", "f4"):
        tests[f"cand:{kind} vs W2 GLM"] = test(frames, "ref", f"cand:{kind}", th)
    for fam in ("F1", "F2", "F3", "F4"):
        cols = [c for c in frames["A"].columns if c.startswith(f"{fam}+")]
        for c in cols:
            lines = [l for l in dr.LINES if frames["A"].loc[frames["A"]["line"] == l, c].notna().all()]
            sub = {s: f.dropna(subset=[c]) for s, f in frames.items()}
            tests[f"{c} vs {fam}|base"] = test(sub, f"{fam}|base", c, th, lines=lines)
    fm = {s: f[f["line"] == "motor"] for s, f in frames.items()}
    for kind, base in (("glum", "cand:glum"), ("lgb", "cand:lgb"), ("ebm", "cand:ebm")):
        tests[f"FS:{kind} vs {base}"] = test(fm, base, f"FS:{kind}", th, lines=["motor"])
    d2 = {}
    for s, f in frames.items():
        for ln in list(dr.LINES) + ["all"]:
            d = f if ln == "all" else f[f["line"] == ln]
            yy = d["y"].to_numpy()
            for col in ["ref", "F1", "F2", "F3", "F4"]:
                mu = v.rescale_to_total(d[col].to_numpy(), yy, by=d["line"].to_numpy())
                d2.setdefault(s, {}).setdefault(ln, {})[col] = float(v.d2_tweedie(yy, mu, d["exposure"]))
    for key, t in tests.items():
        print(key)
        for ln, r in t.items():
            a = r["A"]
            print(f"  {ln:9s} A {a['gain']:10.1f} [{a['boot']['lo']:10.1f}, {a['boot']['hi']:10.1f}]  B {r['B']['gain']:10.1f}  keep {r['verdict']['keep']}")
    write_evidence({"phase5_step7": dict(by_line=res, tests=tests, ranking_d2=d2)}, OUT / "phase5_step7.json")


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "tests":
        run_tests()
    else:
        run_line(arg)
