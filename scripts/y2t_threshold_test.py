"""Y2-T threshold test (decisions/y2t_threshold_test_design.md, committed before this ran): which code-level threshold
predicts best out of sample, and past which threshold the gains stop.

Each candidate threshold N is a regime: the codes with more than N levels enter LightGBM as one credibility-weighted
number (src/fields.py), the rest as raw categories. R0 leaves the codes above 20 levels out altogether.
  emerging V1  out of time, fit 2023, score 2024, claim count (Poisson)          - the selection
  emerging V2  customer-grouped five-fold cross-validation over 2023-2024, seeds 11, 22, 33, claim count
  part1        scorings A and B, claim count and capped cost (Tweedie 1.5)
Run: .venv/Scripts/python.exe -I scripts/y2t_threshold_test.py [emerging_v1|emerging_v2|part1|select]  (Y2T_THREADS, default 4)
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
from scipy.special import xlogy

from src import data_rules as dr, fields, validation as v
from src.build import CUSTOMER_COLS
from src.evidence import write_evidence
from src.families import LGB, as_category
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "work" / "y2t"
OUT = ROOT / "outputs" / "y2t_threshold_test.json"
EMERGING = ROOT / "data" / "raw_part2" / "emerging"
THREADS = int(os.environ.get("Y2T_THREADS", "4"))
GRID = [dict(leaves=a, min_leaf=b) for a in (15, 31, 63) for b in (100, 400, 1600)]
LR, ES_ROUNDS, MAX_TREES, SEED = 0.05, 100, 3000, 11
CANDIDATES = {"emerging": (4, 5, 6, 7, 20, 100, 300), "part1": (4, 5, 7, 11, 20, 300)}
R0_CUT = 20                                          # R0: codes above this many levels are left out altogether
HELPERS = ("y_n", "y_c", "ec_n", "ec_c", "period", "occupation_clean")


def log(*a):
    WORK.mkdir(parents=True, exist_ok=True)
    msg = time.strftime("%H:%M:%S") + " " + " ".join(map(str, a))
    print(msg, flush=True)
    with open(WORK / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def deviance(y, mu, target: str) -> np.ndarray:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    if target == "count":
        return 2 * (xlogy(y, y / mu) - (y - mu))
    return v.tweedie_unit_deviance(y, mu, 1.5)


# ------------------------------------------------------------------------------------------------ data
def emerging_rows(line: str) -> pd.DataFrame:
    ad, cu, claims = dr.read_raw("addresses", EMERGING), dr.read_raw("customers", EMERGING), dr.read_raw("claims", EMERGING)
    tr = dr.read_raw(f"policies_{line}", EMERGING)
    tr = dr.apply_policy_rules(tr, line, float("inf") if line != "motor" else None)     # R2 left out (design)
    tr = tr.merge(ad, on="address_id", how="left").merge(cu[[c for c in CUSTOMER_COLS if c in cu.columns]], on="customer_id", how="left")
    tr["yr"] = tr["period_start"].dt.year
    tr = tr[tr["yr"].isin((2023, 2024))].reset_index(drop=True)
    m = dr.match_claims_to_rows(tr, claims, line)
    tr["y_n"] = tr["row_id"].map(m[~m["repudiated"]].groupby("row_id").size()).fillna(0.0).to_numpy()
    tr["ec_n"] = tr["exposure"].to_numpy(float)
    tr["occupation_clean"] = tr["occupation"].astype("string").str.strip().str.lower()
    return tr


def part1_rows(line: str) -> pd.DataFrame:
    from src import build
    tr = build.build_all()[line][0]
    tr = tr[~tr["holdout"]].reset_index(drop=True)
    tr["y_n"], tr["y_c"] = tr["n_claims"].to_numpy(float), tr["costc"].to_numpy(float)
    tr["ec_n"] = (tr["exposure"] * tr["completion_count"]).to_numpy(float)
    tr["ec_c"] = (tr["exposure"] * tr["completion"]).to_numpy(float)
    tr["occupation_clean"] = tr["occupation"]
    return tr


def levels(df: pd.DataFrame, fit: np.ndarray) -> dict:
    base = df.drop(columns=[c for c in HELPERS if c in df.columns])
    f = fields.tree_fields(base[fit], 10 ** 9)
    return {c: int(base.loc[fit, c].nunique(dropna=True)) for c in f["cat"]}


def regime(df: pd.DataFrame, fit: np.ndarray, n_max: float | None, clean: bool) -> dict:
    """Inputs of one candidate on these fitting rows. n_max None is R0."""
    base = df.drop(columns=[c for c in HELPERS if c in df.columns])
    if clean:
        base = base.assign(occupation=df["occupation_clean"])
    f = fields.tree_fields(base[fit], R0_CUT if n_max is None else n_max)
    if n_max is None:
        f = dict(f, dropped=f["encode"], encode=[])
    return f


# ------------------------------------------------------------------------------------------------ one fit
def frame(df: pd.DataFrame, f: dict, clean: bool, fit: np.ndarray, score: np.ndarray, y, E, period) -> pd.DataFrame:
    src = df.assign(occupation=df["occupation_clean"]) if clean else df
    X = src[f["num"]].astype(float).copy()
    X["yr"] = df["yr"].astype(float).to_numpy()
    X.loc[score, "yr"] = float(df.loc[fit, "yr"].max())
    for c in f["cat"]:
        X[c] = src[c].to_numpy()
    X = as_category(X, f["cat"])
    for c in f["encode"]:
        enc, _k = fields.credibility_encode(src[c], period, y, E, fit, df["customer_id"].astype(str).to_numpy(), SEED)
        X[f"enc_{c}"] = enc
    return X


def fit_predict(X, y, ec, fit, score, groups, objective: str) -> tuple[np.ndarray, dict]:
    """Setting on fold 1 of the fitting rows' customer-grouped inner folds (seed 11), then refit on all fitting rows."""
    fi = np.flatnonzero(fit)
    a, b = v.make_folds(len(fi), [SEED], n_folds=5, groups=groups[fi])[SEED][0]
    ia, ib = fi[a], fi[b]
    best = None
    for s in GRID:
        m = LGB(objective, 1.5, s["leaves"], s["min_leaf"], LR, MAX_TREES, THREADS, seed=SEED).fit(
            X.iloc[ia], y[ia], ec[ia], val=(X.iloc[ib], y[ib], ec[ib], None, None), es_rounds=ES_ROUNDS)
        mu = m.rate(X.iloc[ib]) * ec[ib]
        d = float(deviance(y[ib], mu * y[ib].sum() / mu.sum(), "count" if objective == "poisson" else "cost").sum())
        if best is None or d < best[0]:
            best = (d, s, m.best_iter_)
    _, s, nt = best
    m = LGB(objective, 1.5, s["leaves"], s["min_leaf"], LR, nt, THREADS, seed=SEED).fit(X.iloc[fi], y[fi], ec[fi])
    return m.rate(X.iloc[np.flatnonzero(score)]), dict(setting=s, trees=int(nt))


def candidate_list(market: str, df: pd.DataFrame, fit: np.ndarray, with_clean: bool) -> list[tuple[str, float | None, bool, dict]]:
    """(name, n_max, clean, inputs) for R0 and every candidate threshold. Every line carries every name, so lines can
    be summed; candidates that give the same inputs in a line are fitted once (run_split)."""
    out = [("R0", None, False, regime(df, fit, None, False))]
    for clean in ((False, True) if with_clean else (False,)):
        for n in CANDIDATES[market]:
            out.append((("clean " if clean else "") + f"N{n}", n, clean, regime(df, fit, n, clean)))
    return out


def run_split(market: str, line: str, df: pd.DataFrame, fit: np.ndarray, score: np.ndarray, target: str, period,
              with_clean: bool) -> dict:
    y = df["y_n" if target == "count" else "y_c"].to_numpy(float)
    ec = df["ec_n" if target == "count" else "ec_c"].to_numpy(float)
    groups = df["customer_id"].astype(str).to_numpy()
    yr = df["yr"].to_numpy()
    rate_by_year = pd.Series(y[fit]).groupby(yr[fit]).sum() / pd.Series(ec[fit]).groupby(yr[fit]).sum()
    E = ec * pd.Series(yr).map(rate_by_year).fillna(float(y[fit].sum() / ec[fit].sum())).to_numpy()
    objective = "poisson" if target == "count" else "tweedie"
    res, done = {}, {}
    for name, n_max, clean, f in candidate_list(market, df, fit, with_clean):
        key = (tuple(f["num"]), tuple(f["cat"]), tuple(f["encode"]), clean and "occupation" in f["cat"] + f["encode"])
        if key in done:                                   # the same inputs as an earlier candidate in this line
            first = done[key]
            res[name] = dict(res[first], n_max=n_max, clean=clean, same_as=first)
            continue
        X = frame(df, f, clean, fit, score, y, E, period)
        rate, info = fit_predict(X, y, ec, fit, score, groups, objective)
        res[name] = dict(n_max=n_max, clean=clean, encode=f["encode"], raw_codes=f["cat"], dropped=f.get("dropped", []),
                         rate=rate, same_as=None, **info)
        done[key] = name
        log(market, line, target, name, "encoded", f["encode"], info)
    return res


def with_mu(cands: dict, e: np.ndarray) -> dict:
    """Candidate records with expected claims (rate x exposure) in place of the rate, ready to save."""
    return {k: {**{kk: vv for kk, vv in c.items() if kk != "rate"}, "mu": (c["rate"] * e).tolist()} for k, c in cands.items()}


# ------------------------------------------------------------------------------------------------ designs
def emerging_v1() -> dict:
    out = {}
    for line in dr.LINES:
        df = emerging_rows(line)
        fit, score = (df["yr"] == 2023).to_numpy(), (df["yr"] == 2024).to_numpy()
        period = (df["yr"] * 10 + df["period_start"].dt.quarter).to_numpy()      # k needs two or more periods (design)
        r = run_split("emerging", line, df, fit, score, "count", period, with_clean=True)
        sc = df.loc[score]
        out[line] = dict(levels=levels(df, fit), rows_fit=int(fit.sum()), rows_score=int(score.sum()), y=sc["y_n"].tolist(),
                         ec=sc["ec_n"].tolist(), customer=sc["customer_id"].astype(str).tolist(),
                         cands=with_mu(r, sc["ec_n"].to_numpy()))
    (WORK / "emerging_v1.json").write_text(json.dumps(out, default=float))
    return out


def emerging_v2() -> dict:
    out = {}
    for line in dr.LINES:
        df = emerging_rows(line)
        period = df["yr"].to_numpy()
        groups = df["customer_id"].astype(str).to_numpy()
        per_seed = {}
        for seed in (11, 22, 33):
            mu = {}
            meta = {}
            for k, (a, b) in enumerate(v.make_folds(len(df), [seed], n_folds=5, groups=groups)[seed]):
                fit, score = np.zeros(len(df), bool), np.zeros(len(df), bool)
                fit[a], score[b] = True, True
                r = run_split("emerging", f"{line} seed {seed} fold {k}", df, fit, score, "count", period, with_clean=True)
                for name, c in r.items():
                    mu.setdefault(name, np.zeros(len(df)))[b] = c["rate"] * df["ec_n"].to_numpy()[b]
                    meta.setdefault(name, []).append({kk: vv for kk, vv in c.items() if kk != "rate"})
            per_seed[seed] = {name: dict(mu=m.tolist(), folds=meta[name]) for name, m in mu.items()}
        out[line] = dict(levels=levels(df, np.ones(len(df), bool)), y=df["y_n"].tolist(), ec=df["ec_n"].tolist(), customer=groups.tolist(), seeds=per_seed)
    (WORK / "emerging_v2.json").write_text(json.dumps(out, default=float))
    return out


def part1() -> dict:
    out = {}
    for line in dr.LINES:
        df = part1_rows(line)
        out[line] = {}
        for s, (fit_years, score_year, completion) in {"A": ((2021, 2022), 2023, False), "B": ((2021, 2022, 2023), 2024, True)}.items():
            fit, score = df["yr"].isin(fit_years).to_numpy(), (df["yr"] == score_year).to_numpy()
            sc = df.loc[score]
            for target in ("count", "cost"):
                r = run_split("part1", f"{line} {s}", df, fit, score, target, df["yr"].to_numpy(), with_clean=False)
                e = (sc["exposure"] * (sc["completion_count" if target == "count" else "completion"] if completion else 1.0)).to_numpy(float)
                out[line][f"{s}|{target}"] = dict(levels=levels(df, fit), y=sc["y_n" if target == "count" else "y_c"].tolist(), ec=e.tolist(),
                                                  customer=sc["customer_id"].astype(str).tolist(),
                                                  cands=with_mu(r, e))
    (WORK / "part1.json").write_text(json.dumps(out, default=float))
    return out


# ------------------------------------------------------------------------------------------------ selection
def score_block(blocks: list[dict], target: str, ci: float) -> dict:
    """blocks: one per line, each {y, customer, cands: {name: mu}}; deviance at a common level per line, summed over lines."""
    names = [n for n in blocks[0]["cands"] if all(n in b["cands"] for b in blocks)]
    dev, cust = {n: [] for n in names}, []
    per_line = {}
    for b in blocks:
        y = np.asarray(b["y"], float)
        cust += list(b["customer"])
        pl = {}
        for n in names:
            mu = np.asarray(b["cands"][n], float)
            d = deviance(y, mu * y.sum() / mu.sum(), target)
            dev[n].append(d)
            pl[n] = float(d.sum())
        per_line[b["line"]] = pl
    dev = {n: np.concatenate(d) for n, d in dev.items()}
    cust = np.asarray(cust)
    tot = {n: float(d.sum()) for n, d in dev.items()}
    best = min(tot, key=tot.get)
    vs_best = {n: v.paired_bootstrap_groups(dev[n], dev[best], cust, ci=ci) for n in names}       # gain > 0: best better
    vs_r0 = {n: v.paired_bootstrap_groups(dev["R0"], dev[n], cust, ci=ci) for n in names if n != "R0"}  # gain > 0: n better
    return dict(total=tot, best=best, vs_best=vs_best, vs_r0=vs_r0, per_line=per_line)


def pick(sc: dict, nmax: dict) -> str | None:
    """Lowest N within reach of the best (its interval over the best includes zero); R0 and cleaned variants excluded."""
    reach = [n for n, b in sc["vs_best"].items() if n in nmax and b["lo"] <= 0]
    return min(reach, key=lambda n: nmax[n]) if reach else None


def regime_range(enc: list[str], lv: dict) -> tuple[int, int]:
    """Range of thresholds that give this regime: from the largest raw code's levels to one below the smallest encoded."""
    raw = [c for c in lv if c not in enc]
    return (max(lv[c] for c in raw) if raw else 0, (min(lv[c] for c in enc) - 1) if enc else 10 ** 9)


def select() -> dict:
    ci = float(load_thresholds(ROOT / "decisions" / "thresholds_part2.md")["y2t_select_ci"])
    res = {"selection_ci": ci}
    v1 = json.loads((WORK / "emerging_v1.json").read_text())
    blocks = [dict(line=l, y=d["y"], customer=d["customer"], cands={n: c["mu"] for n, c in d["cands"].items()}) for l, d in v1.items()]
    main = [n for n in v1["motor"]["cands"] if not n.startswith(("R0", "clean"))]
    nmax = {n: v1["motor"]["cands"][n]["n_max"] for n in main}
    s1 = score_block(blocks, "count", ci)
    s1_main = score_block([dict(b, cands={n: m for n, m in b["cands"].items() if not n.startswith("clean")}) for b in blocks], "count", ci)
    lv = {}
    for d in v1.values():
        for c, k in d["levels"].items():
            lv[c] = max(lv.get(c, 0), k)
    chosen = pick(s1_main, nmax)
    res["emerging_v1"] = dict(all_candidates=s1, main=s1_main, chosen=chosen,
                              chosen_range=regime_range(sorted({c for d in v1.values() for c in d["cands"][chosen]["encode"]}), lv) if chosen else None,
                              per_line_chosen={l: pick(score_block([b], "count", ci), nmax) for l, b in zip(v1, [dict(b, cands={n: m for n, m in b["cands"].items() if not n.startswith("clean")}) for b in blocks])},
                              candidates={l: {n: {k: c[k] for k in ("n_max", "clean", "encode", "raw_codes", "dropped", "setting", "trees")} for n, c in d["cands"].items()} for l, d in v1.items()},
                              levels=lv)
    if (WORK / "emerging_v2.json").exists():
        v2 = json.loads((WORK / "emerging_v2.json").read_text())
        # each row's mean deviance over the three seeds' out-of-fold predictions
        avg = []
        for l, d in v2.items():
            names = list(d["seeds"]["11"].keys())
            y = np.asarray(d["y"], float)
            mu = {}
            for n in names:
                devs = [deviance(y, (m := np.asarray(d["seeds"][s][n]["mu"], float)) * y.sum() / m.sum(), "count") for s in ("11", "22", "33")]
                mu[n] = np.mean(devs, axis=0)
            avg.append((l, y, d["customer"], mu))
        names = [n for n in avg[0][3] if not n.startswith("clean")]
        cust = np.concatenate([np.asarray(c) for _, _, c, _ in avg])
        dev = {n: np.concatenate([m[n] for *_, m in avg]) for n in avg[0][3]}
        tot = {n: float(dv.sum()) for n, dv in dev.items()}
        best = min((n for n in names), key=tot.get)
        vs_best = {n: v.paired_bootstrap_groups(dev[n], dev[best], cust, ci=ci) for n in names}
        sc2 = dict(total=tot, best=best, vs_best=vs_best, vs_r0={n: v.paired_bootstrap_groups(dev["R0"], dev[n], cust, ci=ci) for n in dev if n != "R0"},
                   per_line={l: {n: float(m[n].sum()) for n in m} for l, _, _, m in avg})
        nm2 = {n: nmax.get(n) for n in names if n in nmax}
        res["emerging_v2"] = dict(scores=sc2, chosen=pick(sc2, nm2))
        res["consistent"] = res["emerging_v2"]["chosen"] == chosen
    if (WORK / "part1.json").exists():
        p1 = json.loads((WORK / "part1.json").read_text())
        res["part1"] = {}
        for key in ("A|count", "A|cost", "B|count", "B|cost"):
            blocks = [dict(line=l, y=d[key]["y"], customer=d[key]["customer"], cands={n: c["mu"] for n, c in d[key]["cands"].items()}) for l, d in p1.items()]
            sc = score_block(blocks, key.split("|")[1], ci)
            nm = {n: p1["motor"][key]["cands"][n]["n_max"] for n in p1["motor"][key]["cands"] if not n.startswith("R0")}
            res["part1"][key] = dict(scores=sc, chosen=pick(sc, nm),
                                     candidates={l: {n: {k: c[k] for k in ("n_max", "encode", "raw_codes", "dropped", "setting", "trees")} for n, c in d[key]["cands"].items()} for l, d in p1.items()})
    write_evidence({"y2t_threshold_test": res}, OUT)
    log("selection", "V1 chosen", chosen, res["emerging_v1"]["chosen_range"], "V2 chosen", res.get("emerging_v2", {}).get("chosen"))
    return res


if __name__ == "__main__":
    {"emerging_v1": emerging_v1, "emerging_v2": emerging_v2, "part1": part1, "select": select}[sys.argv[1]]()
