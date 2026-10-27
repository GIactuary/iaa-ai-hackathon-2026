"""Evidence for the open Part 2 decisions (decisions/part2_decision_evidence_design.md, written before this ran).

Descriptive figures (new business, blanks, gender, large losses, reporting, one year of change) and the exploratory
carry-over test: emerging-only (E), developed-only (D) and developed start with emerging boosting (DE), scored out of
time (fit 2023, score 2024) and by customer-grouped cross-validation over 2023-2024 (seeds 11, 22, 33).
Run: .venv/Scripts/python.exe -I scripts/part2_decision_evidence.py   (P2E_THREADS, default 4)
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
EMG, DEV = ROOT / "data" / "raw_part2" / "emerging", ROOT / "data" / "raw_part2" / "developed"
OUT = ROOT / "outputs" / "part2_decision_evidence.json"
THREADS = int(os.environ.get("P2E_THREADS", "4"))
CAPS = {"motor": 100_000.0, "buildings": 200_000.0, "contents": 30_000.0}     # Part 1 caps, provisional here
AGE_BANDS = ([0, 24, 34, 49, 64, 200], ["17-24", "25-34", "35-49", "50-64", "65+"])
SEED, LR, ES = 11, 0.05, 100
NOT_CARRIED = ("region", "make_model", "occupation")
LOG_T0 = time.time()


def log(*a):
    print(f"{time.time() - LOG_T0:7.0f}s", *a, flush=True)


# ------------------------------------------------------------------------------------------------ data
def rows(line: str, root: Path, years: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    ad, cu, claims = dr.read_raw("addresses", root), dr.read_raw("customers", root), dr.read_raw("claims", root)
    tr = dr.apply_policy_rules(dr.read_raw(f"policies_{line}", root), line, float("inf") if line != "motor" else None)
    keep = [c for c in CUSTOMER_COLS + ["gender"] if c in cu.columns and c not in ("credit_band", "date_of_birth")]
    tr = tr.merge(ad, on="address_id", how="left").merge(cu[keep], on="customer_id", how="left")
    tr["yr"] = tr["period_start"].dt.year
    tr = tr[tr["yr"].isin(years)].reset_index(drop=True)
    m = dr.match_claims_to_rows(tr, claims, line)
    m = m[~m["repudiated"]].copy()
    m["capped"] = m["net_cost"].clip(upper=CAPS[line])
    tr["y_n"] = tr["row_id"].map(m.groupby("row_id").size()).fillna(0.0).to_numpy(float)
    tr["y_c"] = tr["row_id"].map(m.groupby("row_id")["capped"].sum()).fillna(0.0).to_numpy(float)
    tr["e"] = tr["exposure"].to_numpy(float)
    age = "driver_age" if line == "motor" else "policyholder_age"
    if age in tr.columns:                                   # developed: exact age to the emerging bands
        tr[f"{age}_band"] = pd.cut(tr[age], AGE_BANDS[0], labels=AGE_BANDS[1]).astype("string")
        tr = tr.drop(columns=[age])
    return tr, m


def rating_fields(df: pd.DataFrame) -> dict:
    base = df.drop(columns=[c for c in ("y_n", "y_c", "e", "yr") if c in df.columns])
    return fields.tree_fields(base, 11)


def ratio_ci(n1, e1, n0, e0) -> dict:
    r = (n1 / e1) / (n0 / e0)
    se = np.sqrt(1 / max(n1, 1) + 1 / max(n0, 1))
    return dict(ratio=float(r), lo=float(r * np.exp(-1.96 * se)), hi=float(r * np.exp(1.96 * se)))


def group_table(df: pd.DataFrame, by: pd.Series) -> dict:
    g = pd.DataFrame({"k": by.astype("string").fillna("blank"), "e": df["e"], "n": df["y_n"], "c": df["y_c"]}).groupby("k").sum()
    return {k: dict(rows=int((by.astype("string").fillna("blank") == k).sum()), exposure=float(r.e), claims=int(r.n),
                    freq=float(r.n / r.e), sev=float(r.c / r.n) if r.n else None, cost_per_year=float(r.c / r.e))
            for k, r in g.iterrows()}


# ------------------------------------------------------------------------------------------------ descriptive
def descriptive(em: dict, dv: dict, cl_e: pd.DataFrame, cl_d: pd.DataFrame) -> dict:
    out = {}
    for line in dr.LINES:
        e, d = em[line], dv[line]
        test = dr.read_raw(f"test_{line}", EMG)
        f = rating_fields(e)
        size = {int(y): dict(rows=int((e.yr == y).sum()), exposure=float(e.loc[e.yr == y, "e"].sum()), claims=int(e.loc[e.yr == y, "y_n"].sum()),
                             new_business=float(e.loc[e.yr == y, "new_business"].mean())) for y in (2023, 2024)}
        size["test"] = dict(rows=int(len(test)), new_business=float(test["new_business"].mean()))
        e23 = e[e.yr == 2023]
        nb = e23["new_business"].astype(bool)
        nb_tab = group_table(e23, nb.map({True: "new", False: "renewal"}))
        nb_ci = ratio_ci(e23.loc[nb, "y_n"].sum(), e23.loc[nb, "e"].sum(), e23.loc[~nb, "y_n"].sum(), e23.loc[~nb, "e"].sum())
        nb_by_channel = {ch: ratio_ci(g.loc[g.new_business.astype(bool), "y_n"].sum(), g.loc[g.new_business.astype(bool), "e"].sum(),
                                      g.loc[~g.new_business.astype(bool), "y_n"].sum(), g.loc[~g.new_business.astype(bool), "e"].sum())
                         for ch, g in e23.groupby("channel")}
        rate_cols = [c for c in f["num"] + f["cat"] + f["encode"]]
        blank = e[rate_cols].isna()
        anyb = blank.any(axis=1)
        per_field = {}
        for c in rate_cols:
            s = blank[c]
            if s.mean() > 0.05:
                per_field[c] = dict(share=float(s.mean()), **ratio_ci(e.loc[s, "y_n"].sum(), e.loc[s, "e"].sum(), e.loc[~s, "y_n"].sum(), e.loc[~s, "e"].sum()))
        blanks = dict(share_rows_any=float(anyb.mean()), any=ratio_ci(e.loc[anyb, "y_n"].sum(), e.loc[anyb, "e"].sum(), e.loc[~anyb, "y_n"].sum(), e.loc[~anyb, "e"].sum()),
                      per_field=per_field)
        g = e["gender"]
        gender = dict(table=group_table(e, g), freq_ratio_m_over_f=ratio_ci(e.loc[g == "M", "y_n"].sum(), e.loc[g == "M", "e"].sum(),
                                                                           e.loc[g == "F", "y_n"].sum(), e.loc[g == "F", "e"].sum()))
        large = {}
        for mk, cl in (("emerging", cl_e), ("developed", cl_d)):
            c = cl[(cl["line"] == line) & (cl["status"] != "repudiated")]
            net = (c["incurred_to_date"] - c["recoveries"]).clip(lower=0)
            large[mk] = dict(claims=int(len(c)), above_cap=int((net > CAPS[line]).sum()), share_cost_above_cap=float((net - CAPS[line]).clip(lower=0).sum() / net.sum()),
                             p99=float(net.quantile(0.99)), p999=float(net.quantile(0.999)), max=float(net.max()),
                             by_peril={p: dict(claims=int(len(x)), above=int((x > CAPS[line]).sum()), max=float(x.max())) for p, x in net.groupby(c["peril"])})
        rep = {}
        for mk, cl, y0 in (("emerging", cl_e, 2023), ("developed", cl_d, 2021)):
            c = cl[(cl["line"] == line) & (cl["status"] != "repudiated") & (cl["loss_date"].dt.year == y0)]
            lag = (c["report_date"] - c["loss_date"]).dt.days
            win = c["report_date"] <= pd.Timestamp(f"{y0 + 1}-12-31")
            rep[mk] = {f"within_{d}": float((lag[win] <= d).mean()) for d in (30, 90, 180, 365)}
            if mk == "developed":
                rep[mk]["reported_after_window_share"] = float((~win).mean())
        yoy = {}
        for mk, df, cl, (y0, y1) in (("emerging", e, cl_e, (2023, 2024)), ("developed", d, cl_d, (2021, 2022))):
            r = {}
            for y in (y0, y1):
                rr = df[df.yr == y]
                c = cl[(cl["line"] == line) & (cl["status"] != "repudiated") & cl["policy_id"].isin(rr["policy_id"]) &
                       (cl["loss_date"].dt.year == y) & (cl["report_date"] <= pd.Timestamp(f"{y}-12-31"))]
                net = (c["incurred_to_date"] - c["recoveries"]).clip(upper=CAPS[line])
                r[y] = dict(freq=float(len(c) / rr["e"].sum()), sev=float(net.mean()))
            yoy[mk] = dict(freq_change=r[y1]["freq"] / r[y0]["freq"] - 1, sev_change=r[y1]["sev"] / r[y0]["sev"] - 1, years=[y0, y1])
        out[line] = dict(size=size, new_business=dict(table=nb_tab, freq_ratio=nb_ci, freq_ratio_by_channel=nb_by_channel),
                         blanks=blanks, gender=gender, large_losses=large, reporting=rep, one_year_change=yoy,
                         developed_rows=int(len(d)), developed_claims=int(d["y_n"].sum()))
        log("descriptive", line)
    return out


# ------------------------------------------------------------------------------------------------ carry-over test
def deviance(y, mu, target):
    if target == "count":
        return 2 * (xlogy(y, y / mu) - (y - mu))
    return v.tweedie_unit_deviance(y, mu, 1.5)


def fit(X, y, e, fi, groups, objective, min_leaf, init=None):
    """Early stopping on fold 1 of five customer-grouped inner folds of the fitting rows, then refit on all of them."""
    a, b = v.make_folds(len(fi), [SEED], n_folds=5, groups=groups[fi])[SEED][0]
    ia, ib = fi[a], fi[b]
    ii = (lambda idx: None if init is None else init[idx])
    m = LGB(objective, 1.5, 31, min_leaf, LR, 3000, THREADS, seed=SEED).fit(X.iloc[ia], y[ia], e[ia], init=ii(ia),
                                                                           val=(X.iloc[ib], y[ib], e[ib], None, ii(ib)), es_rounds=ES)
    return LGB(objective, 1.5, 31, min_leaf, LR, m.best_iter_, THREADS, seed=SEED).fit(X.iloc[fi], y[fi], e[fi], init=ii(fi))


def carry_over(em: dict, dv: dict) -> dict:
    out = {}
    for target in ("count", "cost"):
        objective, ycol = ("poisson", "y_n") if target == "count" else ("tweedie", "y_c")
        res = {}
        for line in dr.LINES:
            e, d = em[line], dv[line]
            fe = rating_fields(e)
            fd = rating_fields(d)
            cols = [c for c in fe["num"] + fe["cat"] if c in fd["num"] + fd["cat"]]
            cats = [c for c in cols if c in fe["cat"]]
            d_cols = [c for c in cols if c not in NOT_CARRIED]
            e_cols = d_cols + [c for c in ("region",) if c in fe["cat"]]      # region: emerging codes, a raw input of E and DE only
            cats = cats + [c for c in e_cols if c not in cols]
            both = as_category(pd.concat([d[d_cols], e[e_cols]], ignore_index=True), cats)
            Xd, Xe = both.iloc[:len(d)].reset_index(drop=True), both.iloc[len(d):].reset_index(drop=True)
            gd, ge = d["customer_id"].astype(str).to_numpy(), e["customer_id"].astype(str).to_numpy()
            yd, ye, ed, ee = d[ycol].to_numpy(float), e[ycol].to_numpy(float), d["e"].to_numpy(float), e["e"].to_numpy(float)
            mD = fit(Xd[d_cols], yd, ed, np.arange(len(d)), gd, objective, 400)
            rD = mD.rate(Xe[d_cols])
            init = np.log(rD)
            log(target, line, "D fitted", mD.best_iter_, "trees on", len(d), "developed rows")
            yr = e["yr"].to_numpy()
            fi, sc = np.flatnonzero(yr == 2023), np.flatnonzero(yr == 2024)
            mE = fit(Xe[e_cols], ye, ee, fi, ge, objective, 100)
            mDE = fit(Xe[e_cols], ye, ee, fi, ge, objective, 100, init=init)
            v1 = dict(idx=sc, E=mE.rate(Xe[e_cols].iloc[sc]), D=rD[sc], DE=mDE.rate(Xe[e_cols].iloc[sc], init=init[sc]),
                      trees=dict(E=mE.best_iter_, DE=mDE.best_iter_, D=mD.best_iter_))
            v2 = {k: np.zeros(len(e)) for k in ("E", "DE")}
            dv2 = {k: np.zeros(len(e)) for k in ("E", "D", "DE")}
            for s in (11, 22, 33):
                pr = {k: np.zeros(len(e)) for k in ("E", "DE")}
                for a, b in v.make_folds(len(e), [s], n_folds=5, groups=ge)[s]:
                    pr["E"][b] = fit(Xe[e_cols], ye, ee, a, ge, objective, 100).rate(Xe[e_cols].iloc[b])
                    pr["DE"][b] = fit(Xe[e_cols], ye, ee, a, ge, objective, 100, init=init).rate(Xe[e_cols].iloc[b], init=init[b])
                for k, r in (("E", pr["E"]), ("D", rD), ("DE", pr["DE"])):
                    mu = r * ee
                    dv2[k] += deviance(ye, mu * ye.sum() / mu.sum(), target) / 3
            res[line] = dict(v1=v1, v2=dv2, y=ye, e=ee, g=ge, cols=dict(D=d_cols, E=e_cols))
            log(target, line, "E and DE done")
        out[target] = summarise(res, target)
    return out


def summarise(res: dict, target: str) -> dict:
    def block(devs: dict, y_all, mu0_all, g_all):
        tot = {k: float(x.sum()) for k, x in devs.items()}
        null = float(deviance(y_all, mu0_all, target).sum())
        pairs = {f"{a}_vs_{b}": v.paired_bootstrap_groups(devs[a], devs[b], g_all) for a, b in (("D", "E"), ("E", "DE"), ("D", "DE"))}
        return dict(total=tot, null=null, d2={k: 1 - t / null for k, t in tot.items()}, pairs=pairs)

    out = {}
    for name in ("v1", "v2"):
        devs, ys, m0, gs, per_line = {k: [] for k in ("E", "D", "DE")}, [], [], [], {}
        for line, r in res.items():
            if name == "v1":
                idx = r["v1"]["idx"]
                y, e, g = r["y"][idx], r["e"][idx], r["g"][idx]
                dl = {}
                for k in ("E", "D", "DE"):
                    mu = r["v1"][k] * e
                    dl[k] = deviance(y, mu * y.sum() / mu.sum(), target)
            else:
                y, e, g, dl = r["y"], r["e"], r["g"], r["v2"]
            mu0 = e * y.sum() / e.sum()
            per_line[line] = block(dl, y, mu0, g)
            for k in dl:
                devs[k].append(dl[k])
            ys.append(y); m0.append(mu0); gs.append(np.char.add(line + ":", g.astype(str)))
        out[name] = dict(all_lines=block({k: np.concatenate(x) for k, x in devs.items()}, np.concatenate(ys), np.concatenate(m0), np.concatenate(gs)),
                         per_line=per_line)
    out["trees_v1"] = {line: r["v1"]["trees"] for line, r in res.items()}
    out["inputs"] = {line: r["cols"] for line, r in res.items()}
    return out


def main():
    em, dv = {}, {}
    cl_e, cl_d = None, None
    for line in dr.LINES:
        em[line], _ = rows(line, EMG, (2023, 2024))
        dv[line], _ = rows(line, DEV, (2021, 2022, 2023, 2024))
    cl_e, cl_d = dr.read_raw("claims", EMG), dr.read_raw("claims", DEV)
    log("rows read", {l: (len(em[l]), len(dv[l])) for l in dr.LINES})
    ev = dict(descriptive=descriptive(em, dv, cl_e, cl_d))
    ev["carry_over"] = carry_over(em, dv)
    write_evidence({"part2_decision_evidence": ev}, OUT, seeds=[11, 22, 33])
    log("written", OUT)


if __name__ == "__main__":
    main()
