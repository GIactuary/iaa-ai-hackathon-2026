"""Stages for the decided model: H hold-out, T full fit + test + explanation, P permutation importance.
Usage: WORK=. python3 v2_full.py H T P   -> out/v2_full.json (merged across runs)
"""
import sys, json, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import v2lib as V

OUT = V.OUT; stages = sys.argv[1:] or ["H", "T", "P"]
path = OUT / "v2_full.json"
R = json.load(open(path)) if path.exists() else {}
y, raw = V.y, V.raw_tr
key = lambda c: c[:-2] if c.endswith("_o") else c


def save():
    json.dump(R, open(path, "w"), indent=1, default=float)


# ------------------------------------------------------------------ H: locked hold-out
if "H" in stages:
    n = len(y); rng = np.random.default_rng(2026)
    hold = np.sort(rng.choice(n, size=int(round(0.2 * n)), replace=False)); fit = np.setdiff1d(np.arange(n), hold)
    inner = list(KFold(5, shuffle=True, random_state=7).split(fit))

    def f_inner(a, b):
        P = V.Pipeline().fit(raw.iloc[fit[a]], y[fit[a]]); return b, P.predict(raw.iloc[fit[b]]).blend.values
    cvp = np.zeros(len(fit))
    for b, p in Parallel(n_jobs=2)(delayed(f_inner)(a, b) for a, b in inner):
        cvp[b] = p
    P = V.Pipeline().fit(raw.iloc[fit], y[fit])
    app = P.predict(raw.iloc[fit]); ho = P.predict(raw.iloc[hold])
    w = pd.read_parquet(OUT / "winner_holdout_preds.parquet"); wa = pd.read_parquet(OUT / "winner_apparent_preds.parquet")
    assert (w.row.values == hold).all()
    old = json.load(open(OUT / "walkthrough.json"))["filing"]["holdout"]["current"]
    e_m, e_c = ho.blend.values - y[hold], w.pred.values - y[hold]
    bs = []; r2 = np.random.default_rng(0)
    for _ in range(2000):
        i = r2.integers(0, len(hold), len(hold)); bs.append(V.rm(e_c[i]) - V.rm(e_m[i]))
    R["holdout"] = dict(n_fit=len(fit), n_hold=len(hold),
                        proposed=dict(apparent=V.rm(app.blend, y[fit]), cv=V.rm(cvp, y[fit]), holdout=V.rm(ho.blend, y[hold])),
                        current=dict(apparent=V.rm(wa.pred, wa.y), cv=old["cv"], holdout=V.rm(w.pred, w.y)),
                        components={k: dict(apparent=V.rm(app[k], y[fit]), holdout=V.rm(ho[k], y[hold])) for k in ["glm", "gbm"]},
                        gain_ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
    print("H", json.dumps(R["holdout"], default=float), flush=True); save()


# ------------------------------------------------------------------ T: full fit, test prices, explanation
if "T" in stages:
    P = V.Pipeline(lasso=True).fit(raw, y)
    te = V.raw_te.reset_index(drop=True)
    pt = P.price(te); pt["Id"] = te.Id.values
    pt.to_parquet(OUT / "v2_test.parquet")
    pd.DataFrame(dict(Id=te.Id, SalePrice=np.expm1(pt.final))).to_csv(OUT / "submission_v2.csv", index=False)
    # GLM: ladder, coefficients, selection
    R["glm_path"] = P.glm.path; R["glm_coef"] = P.glm.coef_table(); R["glm_sel"] = P.glm.sel
    R["glm_logc"] = sorted(c for c in P.glm.sel if c in P.glm.logc)
    # referral list
    win = pd.read_csv(OUT / "winner_submission_post.csv").set_index("Id").SalePrice
    ref = pt[pt.size_flag | pt.gap_flag].copy()
    ref = ref.join(te[["Neighborhood", "GrLivArea", "OverallQual", "YearBuilt", "SaleCondition"]])
    ref["nbmax"] = ref.Neighborhood.map(P.nbmax); ref["model"] = np.expm1(ref.blend); ref["final_p"] = np.expm1(ref.final)
    ref["current"] = ref.Id.map(win)
    R["referral"] = ref[["Id", "Neighborhood", "GrLivArea", "nbmax", "ratio", "gap", "OverallQual", "YearBuilt", "SaleCondition", "size_flag", "gap_flag",
                         "treatment", "model", "final_p", "current"]].sort_values(["size_flag", "ratio"], ascending=False).to_dict("records")
    R["referral_counts"] = dict(size=int(pt.size_flag.sum()), gap=int(pt.gap_flag.sum()), either=int((pt.size_flag | pt.gap_flag).sum()),
                               comparable=int(pt.comparable.sum()), nonmarket_ppsf=float(V.NONMARKET.ppsf.iloc[0]))
    # unseen category levels in test
    trc, tec = V.prep(raw), V.prep(te)
    R["unseen_levels"] = {c: int((~tec[c].isin(set(trc[c]))).sum()) for c in trc.columns if V.is_cat(trc[c]) and (~tec[c].isin(set(trc[c]))).any()}
    # adjustment factor
    adj = np.exp(0.5 * (pt.gbm - pt.glm))
    R["adjust_test"] = dict(pcts={str(q): float(np.percentile(adj, q)) for q in [5, 10, 25, 50, 75, 90, 95]},
                            within5=float((np.abs(adj - 1) <= .05).mean()), within10=float((np.abs(adj - 1) <= .10).mean()),
                            max_up=float(adj.max()), max_down=float(adj.min()),
                            hist=np.histogram(np.clip(adj, 0.8, 1.2), bins=np.linspace(0.8, 1.2, 21))[0].tolist())
    # adjustment drivers: half the gap between tree contribution and GLM term, by variable
    gt = P.glm.terms(tec); tc, tbase = P.trees.contrib(tec)
    gk = gt.T.groupby(gt.columns.map(key)).sum().T; tk = tc.T.groupby(tc.columns.map(key)).sum().T
    allk = sorted(set(gk.columns) | set(tk.columns))
    gk, tk = gk.reindex(columns=allk, fill_value=0.0), tk.reindex(columns=allk, fill_value=0.0)
    ad = 0.5 * (tk - gk)
    imp = ad.abs().mean().sort_values(ascending=False)
    R["adjust_drivers"] = [dict(var=k, mean_abs=float(v), in_glm=k in set(map(key, P.glm.sel))) for k, v in imp.head(12).items()]
    R["adjust_const"] = float(0.5 * (tbase[0] - P.glm.mean_pred))
    # rating example house and the odd house
    for name, hid in [("house", 2640), ("house_odd", 1850)]:
        i = int(np.where(te.Id.values == hid)[0][0]); r = te.iloc[i]
        g_terms = gk.iloc[i]; g_terms = g_terms[[k for k in allk if k in set(map(key, P.glm.sel))]]
        top = g_terms.abs().sort_values(ascending=False).index[:8]
        a_terms = ad.iloc[i]; atop = a_terms.abs().sort_values(ascending=False).index[:5]
        R[name] = dict(Id=hid, features={k: str(r[k]) for k in ["Neighborhood", "GrLivArea", "OverallQual", "OverallCond", "YearBuilt",
                                                                 "TotalBsmtSF", "GarageCars", "KitchenQual"]},
                       glm=float(pt.glm[i]), gbm=float(pt.gbm[i]), blend=float(pt.blend[i]), final=float(pt.final[i]),
                       lasso=float(pt.lasso[i]), current=float(win[hid]), glm_base=float(P.glm.mean_pred),
                       glm_terms=[dict(var=k, v=float(g_terms[k])) for k in top], glm_rest=float(g_terms.drop(top).sum()),
                       adj_terms=[dict(var=k, v=float(a_terms[k])) for k in atop],
                       adj_rest=float(a_terms.drop(atop).sum() + R["adjust_const"]),
                       size_flag=bool(pt.size_flag[i]), gap=float(pt.gap[i]))
    # exclusion investigation: how the model prices the two excluded sales
    ex = V.raw_tr_all[V.EXCL(V.raw_tr_all)]
    pe = P.predict(ex)
    R["excluded"] = [dict(Id=int(r.Id), GrLivArea=int(r.GrLivArea), OverallQual=int(r.OverallQual), YearBuilt=int(r.YearBuilt),
                          SaleCondition=r.SaleCondition, SaleType=r.SaleType, TotalBsmtSF=float(r.TotalBsmtSF), price=float(r.SalePrice),
                          model=float(np.expm1(pe.blend.iloc[j])), ratio=float(r.GrLivArea / P.nbmax.get("Edwards")))
                     for j, (_, r) in enumerate(ex.iterrows())]
    # explanation curves on the 1,463 training houses: ALE for numerics, average relativity for neighbourhood
    def preds(rr):
        p = P.predict(rr); return p.glm.values, p.blend.values, p.gbm.values

    def ale(var, edges, setter, centre):
        acc = {"glm": [0.0], "blend": [0.0], "gbm": [0.0]}; counts = []
        v = raw[var].values
        for k in range(1, len(edges)):
            lo, hi = edges[k - 1], edges[k]
            rows = raw[(v > lo) & (v <= hi)] if k > 1 else raw[(v >= lo) & (v <= hi)]
            counts.append(len(rows))
            if len(rows) == 0:
                for m in acc: acc[m].append(acc[m][-1]); continue
            a, b = preds(setter(rows, lo)), preds(setter(rows, hi))
            for j, m in enumerate(["glm", "blend", "gbm"]):
                acc[m].append(acc[m][-1] + float(np.mean(b[j] - a[j])))
        c = int(np.argmin(np.abs(np.asarray(edges) - centre)))
        return dict(edges=[float(e) for e in edges], n=counts, **{m: [float(np.exp(x - acc[m][c])) for x in acc[m]] for m in acc})

    def set_qual(rows, v):
        return rows.assign(OverallQual=v)

    def set_area(rows, v):
        r = rows.copy(); f = v / r.GrLivArea
        for c in V.ABOVE:
            r[c] = r[c] * f
        return r

    def set_year(rows, v):
        return rows.assign(YearBuilt=v, YearRemodAdd=np.maximum(rows.YearRemodAdd, v))

    R["ale"] = dict(OverallQual=ale("OverallQual", list(range(1, 11)), set_qual, 5),
                    GrLivArea=ale("GrLivArea", list(np.unique(np.quantile(raw.GrLivArea, np.linspace(0, 1, 13)).round(0))), set_area, 1500),
                    YearBuilt=ale("YearBuilt", list(np.unique(np.quantile(raw.YearBuilt, np.linspace(0, 1, 11)).round(0))), set_year, 1975))
    nb = raw.Neighborhood.value_counts(); rel = {}
    for lev in nb.index:
        g_, b_, t_ = preds(raw.assign(Neighborhood=lev)); rel[lev] = (g_.mean(), b_.mean())
    mg = np.average([rel[l][0] for l in nb.index], weights=nb.values); mb = np.average([rel[l][1] for l in nb.index], weights=nb.values)
    R["nbhd_rel"] = [dict(nb=l, n=int(nb[l]), glm=float(np.exp(rel[l][0] - mg)), blend=float(np.exp(rel[l][1] - mb))) for l in nb.index]
    print("T done", R["referral_counts"], R["adjust_test"]["pcts"], flush=True); save()


# ------------------------------------------------------------------ P: permutation importance, seed-42 folds
GROUPS = {"Overall quality": ["OverallQual"], "Living area (above ground)": V.ABOVE, "Neighbourhood": ["Neighborhood"],
          "Basement area": V.BSMT, "Year built": ["YearBuilt"], "Overall condition": ["OverallCond"], "Year remodelled": ["YearRemodAdd"],
          "Garage": ["GarageCars", "GarageArea"], "Lot area": ["LotArea"], "Kitchen quality": ["KitchenQual"],
          "Exterior quality": ["ExterQual"], "Bathrooms": ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"],
          "Fireplaces": ["Fireplaces", "FireplaceQu"], "Zoning": ["MSZoning"], "Building class": ["MSSubClass"],
          "Functional": ["Functional"], "Central air": ["CentralAir"]}
if "P" in stages:
    folds = list(KFold(5, shuffle=True, random_state=42).split(raw))

    def f_perm(a, b):
        P = V.Pipeline().fit(raw.iloc[a], y[a]); va = raw.iloc[b].reset_index(drop=True); yb = y[b]
        base = P.predict(va); out = {}
        rng = np.random.default_rng(0)
        for gname, cols in GROUPS.items():
            inc = {"glm": [], "gbm": [], "blend": []}
            for rep in range(3):
                perm = rng.permutation(len(va)); vp = va.copy()
                for c in cols:
                    vp[c] = va[c].values[perm]
                pp = P.predict(vp)
                for m in inc:
                    inc[m].append(V.rm(pp[m], yb) - V.rm(base[m], yb))
            out[gname] = {m: float(np.mean(v)) for m, v in inc.items()}
        return out
    res = Parallel(n_jobs=2)(delayed(f_perm)(a, b) for a, b in folds)
    R["perm"] = sorted([dict(var=g, **{m: float(np.mean([r[g][m] for r in res])) for m in ["glm", "gbm", "blend"]}) for g in GROUPS],
                       key=lambda d: -d["blend"])
    print("P", R["perm"][:6], flush=True); save()
