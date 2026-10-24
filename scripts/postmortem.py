"""Part 1 post-mortem (decisions/postmortem_design.md, written before this ran): which fenced inputs carry
out-of-time signal.

Per line, non-hold-out customers, LightGBM on scorings A and B:
  R      the step 7 tree inputs (raw fields behind the W2 GLM plus the carried candidates)
  PM1a   R + the row's realised weather for its own year
  PM1b   PM1a + the address's climate normal over the fitting years
  PM2    R + every legitimate field the trees did not see
  PM3    PM2 + PM1b
  PM4a   R + log written premium and discount applied
  PM4b   R started from log written premium (offset)
and the W2 GLM with and without log written premium as one numeric term (PM4c).
Settings (leaves, minimum leaf rows, tree count) are chosen the same way for R and every candidate, on fold 1 of
scoring A's customer-grouped inner folds, and reused on B.

Run one line:  .venv/Scripts/python.exe -I scripts/postmortem.py motor     (PM_THREADS, default 4)
Run the tests: .venv/Scripts/python.exe -I scripts/postmortem.py tests
Recoveries on open claims (PM5, a measurement): .venv/Scripts/python.exe -I scripts/postmortem.py recoveries
The amendment (PM2m, PM3m, PM6), then the tests again: .venv/Scripts/python.exe -I scripts/postmortem.py amend
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

import scripts.phase5_step7 as s7
from scripts.phase5_step6 import test
from src import market
from src import data_rules as dr, validation as v
from src.evidence import write_evidence
from src.families import LGB, as_category
from src.glm import WEATHER_VARS, TweedieGLM
from src.thresholds import load_thresholds

OUT, WORK = market.OUT, market.WORK / "postmortem"
THREADS = int(os.environ.get("PM_THREADS", "4"))
GRID = [dict(leaves=a, min_leaf=b) for a in (15, 31) for b in (400, 1600)]
VARIANTS = ("R", "PM1a", "PM1b", "PM2", "PM3", "PM4a", "PM4b")
# Never inputs in PM2 (design, "PM2 fields")
EXCLUDE = {"row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "date_of_birth",
           "exposure", "holdout", "renewal_decision", "customer_service_contacts", "written_premium", "discount_applied"}
EXCLUDE_PREFIX = ("cost", "costc", "n_", "completion", "dq_")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wide_fields(tr: pd.DataFrame, used: list[str]) -> list[str]:
    return [c for c in tr.columns if c not in EXCLUDE and c not in used and not c.startswith(EXCLUDE_PREFIX)]


def weather_frame(tr: pd.DataFrame, fit: np.ndarray, weather: pd.DataFrame) -> pd.DataFrame:
    """Realised weather of the row's own year, and the address's mean over the fitting years only."""
    w = weather.rename(columns={"year": "yr"})
    real = tr[["address_id", "yr"]].merge(w[["address_id", "yr", *WEATHER_VARS]], on=["address_id", "yr"], how="left")
    fy = sorted(tr.loc[fit, "yr"].unique())
    norm = w[w["yr"].isin(fy)].groupby("address_id")[list(WEATHER_VARS)].mean()
    out = pd.DataFrame(index=tr.index)
    for c in WEATHER_VARS:
        out[f"wx_{c}"] = real[c].to_numpy(float)
        out[f"norm_{c}"] = tr["address_id"].map(norm[c]).to_numpy(float)
    return out


def frames(line: str, name: str, weather: pd.DataFrame) -> dict:
    D = s7.load(line, name); D["line"] = line
    tr, fit = D["tr"], D["fit"]
    carried = [c for t, _ in s7.CARRIED[line].values() for c in t]
    ref = D["tbase"] + carried
    T = D["T"][ref].copy()
    W = weather_frame(tr, fit, weather)
    wide = wide_fields(tr, ref)
    wcat = [c for c in wide if not pd.api.types.is_numeric_dtype(tr[c]) or pd.api.types.is_bool_dtype(tr[c])]
    X = pd.concat([T, W, tr[wide].reset_index(drop=True)], axis=1)
    for c in wide:
        if c not in wcat:
            X[c] = X[c].astype(float)
    X = as_category(X, wcat)
    X["lwp"] = np.log(tr["written_premium"].clip(lower=1e-6)).to_numpy()
    X["discount_applied"] = tr["discount_applied"].to_numpy(float)
    wx = [f"wx_{c}" for c in WEATHER_VARS]
    nm = [f"norm_{c}" for c in WEATHER_VARS]
    widem = [c for c in wide if c != "make_model"]
    cols = {"R": ref, "PM1a": ref + wx, "PM1b": ref + wx + nm, "PM2": ref + wide, "PM3": ref + wide + wx + nm,
            "PM4a": ref + ["lwp", "discount_applied"], "PM4b": ref, "PM2m": ref + widem, "PM3m": ref + widem + wx + nm}
    return dict(D=D, X=X, cols=cols, wide=wide)


def choose(line: str, F: dict, th, power: float, variants=VARIANTS) -> dict:
    """Setting and tree count for every variant, on fold 1 of scoring A's fitting rows (seed 11)."""
    path = WORK / f"settings_{line}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    D, X = F["D"], F["X"]
    tr = D["tr"]
    fi = np.flatnonzero(D["fit"])
    seed = int(th["split_seed_1"])
    cust = pd.factorize(tr["customer_id"].to_numpy()[fi])[0]
    a, b = v.make_folds(len(fi), [seed], n_folds=int(th["inner_folds"]), groups=cust)[seed][0]
    a, b = fi[a], fi[b]
    y, ec = tr["costc"].to_numpy(), (tr["exposure"] * tr["completion"]).to_numpy()
    for var in variants:
        if var in done:
            continue
        Xv = X[F["cols"][var]]
        init = X["lwp"].to_numpy() if var == "PM4b" else None
        trials = []
        for s in GRID:
            m = LGB("tweedie", power, lr=th["gbm_learning_rate"], n_trees=int(th["max_trees"]), threads=THREADS, **s)
            m.fit(Xv.iloc[a], y[a], ec[a], init=None if init is None else init[a],
                  val=(Xv.iloc[b], y[b], ec[b], None, None if init is None else init[b]), es_rounds=int(th["early_stop_rounds"]))
            pb = m.rate(Xv.iloc[b], init=None if init is None else init[b]) * ec[b]
            trials.append(dict(setting=s, trees=m.best_iter_, dev=s7.fold_dev(y[b], pb)))
        best = min(trials, key=lambda t: t["dev"])
        done[var] = dict(best=best, trials=trials)
        path.write_text(json.dumps(done, default=float, indent=1))
        log(line, var, "chosen", best["setting"], best["trees"], round(best["dev"], 1))
    return done


def out_of_time(line: str, name: str, F: dict, th, power: float, settings: dict, variants=VARIANTS) -> pd.DataFrame:
    D, X = F["D"], F["X"]
    tr, fit, sc = D["tr"], D["fit"], D["sc"]
    e, c = tr["exposure"].to_numpy(), tr["completion"].to_numpy()
    ec, y = e * c, tr["costc"].to_numpy()
    cs = c[sc] if v.SCORINGS[name]["completion"] else 1.0
    f = pd.DataFrame({"line": line, "customer_id": tr.loc[sc, "customer_id"].to_numpy(), "y": tr.loc[sc, "cost"].to_numpy(),
                      "exposure": e[sc]})
    for var in variants:
        b = settings[var]["best"]
        Xv = X[F["cols"][var]]
        init = X["lwp"].to_numpy() if var == "PM4b" else None
        m = LGB("tweedie", power, lr=th["gbm_learning_rate"], n_trees=b["trees"], threads=THREADS, **b["setting"])
        m.fit(Xv[fit], y[fit], ec[fit], init=None if init is None else init[fit])
        f[var] = m.rate(Xv[sc], init=None if init is None else init[sc]) * D["elf"] * e[sc] * cs
        log(line, name, var, "fitted")
        if var in ("PM3", "PM3m") and name == "A":
            explain(line, m, Xv, sc, F["wide"], suffix="" if var == "PM3" else "_m")
    if variants != VARIANTS:
        return f
    G = D["G"][D["w2"]]
    g = TweedieGLM().fit(G[fit], y[fit], ec[fit])
    f["glm"] = g.rate(G[sc]) * D["elf"] * e[sc] * cs
    Gp = G.copy(); Gp["lwp"] = X["lwp"].to_numpy()
    gp = TweedieGLM().fit(Gp[fit], y[fit], ec[fit])
    f["glm_prem"] = gp.rate(Gp[sc]) * D["elf"] * e[sc] * cs
    log(line, name, "GLMs fitted")
    return f


def explain(line, m, Xv, sc, wide, suffix=""):
    rng = np.random.default_rng(0)
    take = rng.choice(np.flatnonzero(sc), size=min(20000, int(sc.sum())), replace=False)
    ctb = np.abs(m.contrib(Xv.iloc[take])[:, :-1]).mean(axis=0)
    top = sorted(zip(Xv.columns, map(float, ctb)), key=lambda t: -t[1])
    (WORK / f"explain_{line}{suffix}.json").write_text(json.dumps(dict(top=top[:25], wide=wide), indent=1))


def run_line(line: str) -> None:
    th = load_thresholds()
    WORK.mkdir(parents=True, exist_ok=True)
    power = json.loads((market.WORK / "phase5_step7" / f"tune_{line}_lgb.json").read_text())["best"]["power"]
    weather = dr.read_raw("weather")
    settings = None
    for name in ("A", "B"):
        path = WORK / f"{line}_{name}.parquet"
        if path.exists():
            continue
        F = frames(line, name, weather)
        log(line, name, "frames ready", {k: len(c) for k, c in F["cols"].items()})
        if settings is None:
            if name != "A":
                settings = choose(line, frames(line, "A", weather), th, power)
            else:
                settings = choose(line, F, th, power)
        out_of_time(line, name, F, th, power, settings).to_parquet(path)
    log(line, "done")


AMEND = ("PM2m", "PM3m")


def premium_parts(line: str, name: str, F: dict) -> dict:
    """PM6: log written premium split into the part the PM3m inputs predict and the remainder, the premium model
    fitted on the fitting rows only; the W2 GLM plus each part as one numeric term, priced on the scored rows."""
    import lightgbm as lgb
    th = load_thresholds()
    D, X = F["D"], F["X"]
    tr, fit, sc = D["tr"], D["fit"], D["sc"]
    Xp, lwp = X[F["cols"]["PM3m"]], X["lwp"].to_numpy()
    fi = np.flatnonzero(fit)
    seed = int(th["split_seed_1"])
    a, b = v.make_folds(len(fi), [seed], n_folds=int(th["inner_folds"]),
                        groups=pd.factorize(tr["customer_id"].to_numpy()[fi])[0])[seed][0]
    pm = lgb.LGBMRegressor(n_estimators=int(th["max_trees"]), learning_rate=th["gbm_learning_rate"], num_leaves=63,
                           min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           n_jobs=THREADS, random_state=0, verbose=-1)
    pm.fit(Xp.iloc[fi[a]], lwp[fi[a]], eval_set=[(Xp.iloc[fi[b]], lwp[fi[b]])],
           callbacks=[lgb.early_stopping(int(th["early_stop_rounds"]), verbose=False)])
    hat = pm.predict(Xp, num_iteration=pm.best_iteration_)
    res = lwp - hat
    r2 = float(1 - np.var(res[sc]) / np.var(lwp[sc]))
    e, c = tr["exposure"].to_numpy(), tr["completion"].to_numpy()
    ec, y = e * c, tr["costc"].to_numpy()
    cs = c[sc] if v.SCORINGS[name]["completion"] else 1.0
    G = D["G"][D["w2"]]
    out = {}
    for col, x in (("glm_phat", hat), ("glm_pres", res)):
        Gx = G.copy(); Gx["lwp_part"] = x
        g = TweedieGLM().fit(Gx[fit], y[fit], ec[fit])
        out[col] = g.rate(Gx[sc]) * D["elf"] * e[sc] * cs
    info = dict(r2_scored=r2, trees=int(pm.best_iteration_))
    if name == "A":
        rng = np.random.default_rng(0)
        take = rng.choice(np.flatnonzero(sc), size=min(20000, int(sc.sum())), replace=False)
        ctb = np.abs(pm.predict(Xp.iloc[take], pred_contrib=True, num_iteration=pm.best_iteration_)[:, :-1]).mean(axis=0)
        info["top"] = sorted(zip(Xp.columns, map(float, ctb)), key=lambda t: -t[1])[:20]
    return dict(cols=out, info=info)


def amend_line(line: str) -> None:
    """The amendment (design, last section): PM2m, PM3m and PM6, added to the saved scored rows."""
    th = load_thresholds()
    power = json.loads((market.WORK / "phase5_step7" / f"tune_{line}_lgb.json").read_text())["best"]["power"]
    weather = dr.read_raw("weather")
    info, settings = {}, None
    for name in ("A", "B"):
        path = WORK / f"{line}_{name}.parquet"
        f = pd.read_parquet(path)
        if all(c in f.columns for c in (*AMEND, "glm_phat", "glm_pres")):
            continue
        F = frames(line, name, weather)
        if settings is None:
            settings = choose(line, F if name == "A" else frames(line, "A", weather), th, power, AMEND)
        g = out_of_time(line, name, F, th, power, settings, AMEND)
        for var in AMEND:
            f[var] = g[var].to_numpy()
        pp = premium_parts(line, name, F)
        for col, x in pp["cols"].items():
            f[col] = x
        info[name] = pp["info"]
        log(line, name, "premium model R2 on scored rows", round(pp["info"]["r2_scored"], 4))
        f.to_parquet(path)
    if info:
        (WORK / f"premium_{line}.json").write_text(json.dumps(info, default=float, indent=1))
    log(line, "amendment done")


def run_tests() -> None:
    th = load_thresholds()
    fr = {s: pd.concat([pd.read_parquet(WORK / f"{l}_{s}.parquet") for l in dr.LINES], ignore_index=True) for s in v.SCORINGS}
    amended = all(c in fr["A"].columns for c in (*AMEND, "glm_phat", "glm_pres"))
    cols = (*VARIANTS, *(AMEND if amended else ()), "glm", "glm_prem", *(("glm_phat", "glm_pres") if amended else ()))
    tests = {f"{var} vs R": test(fr, "R", var, th) for var in cols if var.startswith("PM")}
    tests["R vs W2 GLM"] = test(fr, "glm", "R", th)
    tests["PM4c: W2 GLM + premium vs W2 GLM"] = test(fr, "glm", "glm_prem", th)
    if amended:
        tests["PM6: W2 GLM + predicted premium vs W2 GLM"] = test(fr, "glm", "glm_phat", th)
        tests["PM6: W2 GLM + premium remainder vs W2 GLM"] = test(fr, "glm", "glm_pres", th)
    d2 = {}
    for s, f in fr.items():
        for ln in list(dr.LINES) + ["all"]:
            d = f if ln == "all" else f[f["line"] == ln]
            yy = d["y"].to_numpy()
            for col in cols:
                mu = v.rescale_to_total(d[col].to_numpy(), yy, by=d["line"].to_numpy())
                d2.setdefault(s, {}).setdefault(ln, {})[col] = float(v.d2_tweedie(yy, mu, d["exposure"]))
    for key, t in tests.items():
        print(key)
        for ln, r in t.items():
            a = r["A"]
            print(f"  {ln:9s} A {a['gain']:10.1f} [{a['boot']['lo']:10.1f}, {a['boot']['hi']:10.1f}]  B {r['B']['gain']:10.1f}  keep {r['verdict']['keep']}")
    settings = {l: json.loads((WORK / f"settings_{l}.json").read_text()) for l in dr.LINES}
    explain_ = {l: json.loads((WORK / f"explain_{l}.json").read_text()) for l in dr.LINES}
    rec = json.loads((WORK / "recoveries.json").read_text()) if (WORK / "recoveries.json").exists() else None
    extra = {}
    if amended:
        extra = dict(explain_m={l: json.loads((WORK / f"explain_{l}_m.json").read_text()) for l in dr.LINES},
                     premium={l: json.loads((WORK / f"premium_{l}.json").read_text()) for l in dr.LINES})
    write_evidence({"postmortem": dict(tests=tests, ranking_d2=d2, settings=settings, explain=explain_, recoveries=rec, **extra)},
                   OUT / "postmortem.json")


def recoveries() -> dict:
    """PM5, a measurement: expected recoveries on claims open at the valuation date, at the recovery rate of the
    claim's peril on closed claims of accident years 2021-2024; the change in net cost by line and accident year,
    and in each line's indicated 2025 capped loss cost (the peril indication of the level chain, every customer,
    without the weather normalisation, which moves frequency only)."""
    from src import build, level as lv
    th = load_thresholds()
    claims = dr.read_raw("claims")
    out = {}
    for line in dr.LINES:
        tr = build.build_all()[line][0]
        m = dr.match_claims_to_rows(tr, claims, line)
        m = m[~m["repudiated"]].copy()
        m["ay"] = m["loss_date"].dt.year
        closed = m[m["status"] == "closed"]
        rate = closed.groupby("peril")["recoveries"].sum() / closed.groupby("peril")["incurred_to_date"].sum()
        m["exp_rec"] = np.where(m["status"] == "open", m["incurred_to_date"] * m["peril"].map(rate).fillna(0.0), 0.0)
        by_ay = m.groupby("ay").agg(net=("net_cost", "sum"), exp_rec=("exp_rec", "sum"))
        by_ay["overstatement"] = by_ay["exp_rec"] / (by_ay["net"] - by_ay["exp_rec"])
        cap = float(th[f"cap_{line}"])
        m["rec_capped"] = m["net_cost"].clip(upper=cap)
        m["adj_capped"] = (m["net_cost"] - m["exp_rec"]).clip(upper=cap)
        frames_ = {}
        for label, col in (("as_recorded", "rec_capped"), ("recoveries_adjusted", "adj_capped")):
            df = tr.copy()
            for p in lv.perils_of(tr):
                s = m[m["peril"] == p].groupby("row_id")[col].sum()
                df[f"costc_{p}"] = df["row_id"].map(s).fillna(0.0).to_numpy()
            frames_[label] = df
        check = float(frames_["as_recorded"][[f"costc_{p}" for p in lv.perils_of(tr)]].sum().sum() / tr["costc"].sum())
        if abs(check - 1) > 1e-9:
            raise ValueError(f"{line}: rebuilt capped cost differs from the build ({check})")
        ind = {}
        for label, df in frames_.items():
            t = lv.peril_trends(df, line, float(th["trend_min"]), float(th["trend_max"]))
            ind[label] = float(lv.indication(df, t, market.TARGET_YEAR)["indicated_capped_lc"].sum())
        out[line] = dict(recovery_rate_closed=rate.to_dict(), by_accident_year=by_ay.reset_index().to_dict("records"),
                         indicated_capped_lc_2025=ind, level_change=ind["recoveries_adjusted"] / ind["as_recorded"] - 1)
        log(line, "level change", round(out[line]["level_change"], 5), by_ay["overstatement"].round(4).to_dict())
    (WORK / "recoveries.json").write_text(json.dumps(out, default=float, indent=1))
    return out


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "tests":
        run_tests()
    elif arg == "recoveries":
        recoveries()
    elif arg == "amend":
        for ln in sys.argv[2:] or dr.LINES:
            amend_line(ln)
    else:
        run_line(arg)
