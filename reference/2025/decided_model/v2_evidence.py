"""Assemble every number the filing page shows for the decided model -> out/v2_page.json."""
import json, itertools, numpy as np, pandas as pd
from pathlib import Path
import v2lib as V

OUT = V.OUT; J = json.load(open(OUT / "walkthrough.json")); R = json.load(open(OUT / "v2_full.json"))
rm = V.rm; SEEDS = [42, 1, 2]; P = {}
for k in ["data", "hist_raw", "hist_log", "missing", "missing_total_cells", "missing_cols", "scatter", "corr", "qual", "lgb_curve", "optuna", "hist_test"]:
    P[k] = J[k]
P["point_of_use"], P["sale_vars"] = J["filing"]["point_of_use"], J["filing"]["sale_vars"]
P.update({k: R[k] for k in R})

oof = {s: pd.read_parquet(OUT / f"v2_oof_s{s}.parquet") for s in SEEDS}
win = {s: pd.read_parquet(OUT / f"winner_oof_s{s}.parquet") for s in SEEDS}

# 1. paired comparison with the current model, and the alternatives considered
rows, rng = [], np.random.default_rng(0)
for s in SEEDS:
    o, w = oof[s], win[s]; assert np.allclose(o.y, w.y)
    ea, eb = (w.pred - w.y).values, (o.blend - o.y).values
    bs = [rm(ea[i]) - rm(eb[i]) for i in (rng.integers(0, len(ea), len(ea)) for _ in range(2000))]
    rows.append(dict(seed=s, current=rm(ea), proposed=rm(eb), gain=rm(ea) - rm(eb), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)),
                     glm=rm(o.glm, o.y), gbm=rm(o.gbm, o.y), lasso=rm(o.lasso, o.y), lasso_gbm=rm((o.lasso + o.gbm) / 2, o.y),
                     three=rm((o.lasso + o.glm + o.gbm) / 3, o.y)))
P["compare"] = rows
print(pd.DataFrame(rows).round(5).to_string())

# 2. lift, double lift, A/E, actual vs predicted (seed 42)
o, w = oof[42], win[42]
act, pm, pw = np.expm1(o.y.values), np.expm1(o.blend.values), np.expm1(w.pred.values)
dec = pd.qcut(pm, 10, labels=False)
P["calibration"] = pd.DataFrame({"d": dec, "a": act, "p": pm}).groupby("d").agg(actual=("a", "mean"), pred=("p", "mean"), n=("a", "size")).reset_index().round(0).to_dict("records")
ratio = pm / pw; dd = pd.qcut(ratio, 10, labels=False)
dl = pd.DataFrame({"d": dd, "a": act, "m": pm, "w": pw, "r": ratio}).groupby("d").agg(actual=("a", "sum"), mine=("m", "sum"), winner=("w", "sum"), ratio_lo=("r", "min"), ratio_hi=("r", "max"), n=("a", "size"))
dl["ae_mine"] = dl.actual / dl.mine; dl["ae_winner"] = dl.actual / dl.winner
P["double_lift"] = dl.reset_index().round(4).to_dict("records")
seg = V.raw_tr[["Neighborhood", "OverallQual"]].copy(); seg["a"], seg["p"], seg["w"] = act, pm, pw
def ae(col):
    g = seg.groupby(col).agg(n=("a", "size"), a=("a", "sum"), p=("p", "sum"), w=("w", "sum")); g["ae_mine"] = g.a / g.p; g["ae_winner"] = g.a / g.w
    return g.reset_index()[[col, "n", "ae_mine", "ae_winner"]].round(4)
P["ae_nbhd"] = ae("Neighborhood").sort_values("n", ascending=False).to_dict("records")
P["ae_qual"] = ae("OverallQual").to_dict("records")
P["avp"] = dict(y=o.y.round(4).tolist(), p=o.blend.round(4).tolist())

# 3. 90% range, decided shape (J5): by predicted-price fifth, 8th to 98th percentile of residuals, cross-fitted
from sklearn.model_selection import KFold
folds = list(KFold(5, shuffle=True, random_state=42).split(o))
def ranges(pred, yy):
    r = yy - pred; q = pd.qcut(pred, 5, labels=False); lo_f, hi_f = np.zeros(len(r)), np.zeros(len(r))
    for a, b in folds:
        for k in range(5):
            ia, ib = a[q[a] == k], b[q[b] == k]
            lo_f[ib], hi_f[ib] = np.quantile(r[ia], .08), np.quantile(r[ia], .98)
    inside = (r >= lo_f) & (r <= hi_f); above = r > hi_f; below = r < lo_f
    full = [(float(np.quantile(r[q == k], .08)), float(np.quantile(r[q == k], .98))) for k in range(5)]
    med = [float(np.expm1(np.median(pred[q == k]))) for k in range(5)]
    out = [dict(fifth=k + 1, lo=float(np.exp(full[k][0])), hi=float(np.exp(full[k][1])), median_price=med[k],
                width=med[k] * (np.exp(full[k][1]) - np.exp(full[k][0])), coverage=float(inside[q == k].mean()),
                above=float(above[q == k].mean()), below=float(below[q == k].mean())) for k in range(5)]
    edges = [float(x) for x in np.quantile(pred, [0, .2, .4, .6, .8, 1])]
    tot = dict(coverage=float(inside.mean()), above=float(above.mean()), below=float(below.mean()),
               width_median=float(np.median(np.expm1(pred) * (np.exp(hi_f) - np.exp(lo_f)))))
    return dict(by_fifth=out, total=tot, edges=edges)
P["ranges"] = dict(proposed=ranges(o.blend.values, o.y.values), current=ranges(w.pred.values, w.y.values))
E = P["ranges"]["proposed"]
def band(p):
    k = int(np.clip(np.searchsorted(E["edges"][1:-1], p, side="right"), 0, 4)); b = E["by_fifth"][k]
    return [float(np.expm1(p) * b["lo"]), float(np.expm1(p) * b["hi"])], k + 1
for h in ["house", "house_odd"]:
    P[h]["interval"], P[h]["fifth"] = band(P[h]["final"])

# 4. impact on the test houses
t = pd.read_parquet(OUT / "v2_test.parquet"); ws = pd.read_csv(OUT / "winner_submission_post.csv")
te = V.raw_te.reset_index(drop=True)
m = t.merge(ws, on="Id").merge(te[["Id", "Neighborhood", "OverallQual", "GrLivArea"]], on="Id")
m["mine"] = np.expm1(m.final); m["chg"] = m.mine / m.SalePrice - 1
bins = [-1, -.15, -.10, -.05, -.02, .02, .05, .10, .15, 10]; labels = ["< −15%", "−15 to −10%", "−10 to −5%", "−5 to −2%", "±2%", "+2 to +5%", "+5 to +10%", "+10 to +15%", "> +15%"]
m["band"] = pd.cut(m.chg, bins, labels=labels)
P["disloc_bands"] = [dict(band=str(b), n=int(n)) for b, n in m.band.value_counts().reindex(labels).fillna(0).items()]
P["disloc_stats"] = dict(n=len(m), within5=float((m.chg.abs() <= .05).mean()), within10=float((m.chg.abs() <= .10).mean()),
                         over15=int((m.chg.abs() > .15).sum()), median_abs=float(m.chg.abs().median()), mean_chg=float(m.mine.sum() / m.SalePrice.sum() - 1))
top = m.reindex(m.chg.abs().sort_values(ascending=False).index).head(10)
top["referred"] = top.size_flag | top.gap_flag
P["disloc_top"] = top[["Id", "Neighborhood", "OverallQual", "GrLivArea", "SalePrice", "mine", "chg", "referred"]].round(4).to_dict("records")

# 5. model choice: development run with eight models (before J2), and the decided pair
old = {s: pd.read_parquet(OUT / f"mine_oof_s{s}.parquet") for s in SEEDS}
names = ["ridge", "lasso", "enet", "svr", "kridge", "lgb", "xgb", "cat"]
sc = lambda c: float(np.mean([rm(old[s][list(c)].mean(axis=1), old[s].y) for s in SEEDS]))
best = {}
for k in [1, 2, 3, 4]:
    combos = sorted((sc(c), c) for c in itertools.combinations(names, k)); best[k] = dict(n=len(combos), top=[dict(models=list(c), err=e) for e, c in combos[:3]])
best["trees3"] = sc(("lgb", "xgb", "cat")); best["best_tree"] = min(sc((t_,)) for t_ in ["lgb", "xgb", "cat"])
res = old[42][names].sub(old[42].y, axis=0)
P["model_choice"] = dict(best=best, err_corr=res.corr().round(3).to_dict(), eight_fitted=float(pd.read_csv(OUT / "comparison.csv").mine_rmsle.mean()))

# 6. GLM selection stability over the 15 cross-validation fits
cnt = {}
for s in SEEDS:
    for sel in oof[s].sel.unique():
        for v_ in sel.split(","):
            cnt[v_] = cnt.get(v_, 0) + 1
P["glm_stability"] = dict(sorted(cnt.items(), key=lambda kv: -kv[1]))

# 7. drop-one test of engineered variables
drops = {}
for f in sorted(OUT.glob("v2_oof_drop_*_s42.parquet")):
    nm = f.name[len("v2_oof_drop_"):-len("_s42.parquet")]
    fs = [OUT / f"v2_oof_drop_{nm}_s{s}.parquet" for s in SEEDS]
    if all(x.exists() for x in fs):
        d_ = [pd.read_parquet(x) for x in fs]
        pre = {s: pd.read_parquet(OUT / f"v2_oof_pre_s{s}.parquet") for s in SEEDS}
        drops[nm] = dict(with_=float(np.mean([rm(pre[s].blend, pre[s].y) for s in SEEDS])), without=float(np.mean([rm(x.blend, x.y) for x in d_])),
                         per_seed=[rm(x.blend, x.y) - rm(pre[s].blend, pre[s].y) for x, s in zip(d_, SEEDS)])
P["drop_one"] = drops

# 7b. does model disagreement flag error on houses with known prices? (all three splits pooled)
g = pd.concat([oof[s_].assign(gap=np.abs(oof[s_].glm - oof[s_].gbm), e=oof[s_].blend - oof[s_].y) for s_ in SEEDS])
top = g.gap >= g.gap.quantile(.98)
P["gap_check"] = dict(top2=rm(g.e[top]), rest=rm(g.e[~top]), gap_cut=float(np.expm1(g.gap.quantile(.98))))

# 8. scatter-stage screen: price on size, quality and neighbourhood, all 1,465 sales
tr = V.raw_tr_all
X = pd.get_dummies(tr.Neighborhood, dtype=float); X["lgla"] = np.log(tr.GrLivArea); X["q"] = tr.OverallQual
yy = np.log(tr.SalePrice.values); b = np.linalg.lstsq(X.values, yy, rcond=None)[0]; r = yy - X.values @ b; z = r / r.std()
scr = tr.assign(z=z, fit=np.exp(X.values @ b)).reindex(np.argsort(-np.abs(z)))
P["screen"] = scr[["Id", "Neighborhood", "GrLivArea", "OverallQual", "YearBuilt", "SaleCondition", "SaleType", "SalePrice", "fit", "z"]].head(6).round(2).to_dict("records")
P["screen_rank_424"] = int(np.where(scr.Id.values == 424)[0][0]) + 1

json.dump(P, open(OUT / "v2_page.json", "w"), default=float)
print("ranges", json.dumps(P["ranges"], indent=0, default=float)[:1500])
print("disloc", P["disloc_stats"]); print("drop_one", drops); print("stability", P["glm_stability"])
