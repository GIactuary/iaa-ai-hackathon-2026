"""Exploratory diagnostic after the check file scored 0.14266 on the public board (2026-10-28): where does the gap
to the leaders come from, the level or the shape? Not a model choice; the choices go through P2-V.

Emulates the Kaggle metric on emerging fitting rows (hold-out untouched): fit on 2023, score 2024, the three lines
pooled, Tweedie 1.5 D2 against the pooled mean. The scored target is 2024 net cost reported to date, so every
prediction is multiplied by the row's reporting completion.
  shape variants: Part 1 base GLM; plus an emerging layer (a GLM on 2023 rows with the base as offset) with region;
  with region and gender (motor); the emerging-only GLM
  premium variants (measurement only; the decided Y1 uses the predicted part alone): the base with log written
  premium as one numeric term in the layer; with its predicted part (lwp_hat, LightGBM on the W2 factors and
  region, fitted on 2023 rows); written premium alone as the rate
  tree variants: LightGBM fitted on Part 1 rows (Tweedie 1.5, capped cost, completed exposure) on the fields both
  markets carry less region, make_model and occupation (as the P2-C test's D), alone and averaged with the base GLM
  in the log
  level variants: each line at its actual 2024 total (perfect level); each line at 2023's completed capped loss cost
  trended one year with Part 1's line trend, times the excess factor (the check file's way)
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_gap_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.part2_model import BASE_YEAR, average_over_unknown, developed, part1_level, unknown_blanks, w2_factors
from src import build, data_rules as dr, validation as v
from src.evidence import write_evidence
from src.glm import TweedieGLM, cat, quantile_edges

OUT = Path(__file__).resolve().parents[1] / "outputs" / "markets" / "part2" / "part2_gap_diagnostic.json"


def layer(F: pd.DataFrame, y, ec, base_rate, fit) -> np.ndarray:
    """Emerging layer: a GLM on the fitting rows with the base rate as an offset; returns base x layer for all rows."""
    m = TweedieGLM().fit(F[fit], y[fit], ec[fit] * base_rate[fit])
    return base_rate * m.rate(F)


def main():
    data = build.build_all()
    p1 = part1_level()
    rows, r2, trees = {}, {}, {}
    for line in dr.LINES:
        tr = data[line][0]
        tr = tr[~tr["holdout"]].reset_index(drop=True)
        dv = developed(line)
        edges = {"sum_insured": quantile_edges(dv["sum_insured"], 10)} if line != "motor" else {}
        Fd = w2_factors(dv, line, edges)
        base = TweedieGLM().fit(Fd, dv["costc"], dv["exposure"] * dv["completion"])
        Ft = w2_factors(tr, line, edges)
        Ft["yr"] = BASE_YEAR
        e = tr["exposure"].to_numpy(float)
        r = average_over_unknown(base, Ft, Ft, e, unknown_blanks(Fd, Ft))
        y = tr["costc"].to_numpy(float)
        ec = e * tr["completion"].to_numpy(float)
        fit = (tr["yr"] == 2023).to_numpy()
        reg = pd.DataFrame({"region": cat(tr["region"])})
        shapes = {"base": r, "base+region": layer(reg, y, ec, r, fit)}
        if line == "motor":
            shapes["base+region+gender"] = layer(reg.assign(gender=cat(tr["gender"])), y, ec, r, fit)
        else:
            shapes["base+region+gender"] = shapes["base+region"]
        lwp = np.log(tr["written_premium"].to_numpy(float))
        import lightgbm as lgb
        Xp = Ft.drop(columns="yr").assign(region=reg["region"]).astype("category")
        fi = np.flatnonzero(fit)
        a, b = v.make_folds(len(fi), [11], n_folds=5, groups=tr["customer_id"].astype(str).to_numpy()[fi])[11][0]
        pm = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, n_jobs=4, random_state=0, verbose=-1)
        pm.fit(Xp.iloc[fi[a]], lwp[fi[a]], eval_set=[(Xp.iloc[fi[b]], lwp[fi[b]])], callbacks=[lgb.early_stopping(100, verbose=False)])
        hat = pm.predict(Xp, num_iteration=pm.best_iteration_)
        r2[line] = float(1 - np.var((lwp - hat)[~fit]) / np.var(lwp[~fit]))
        shapes["base+lwp_hat"] = layer(pd.DataFrame({"lwp_hat": hat}), y, ec, r, fit)
        shapes["base+log_premium"] = layer(pd.DataFrame({"lwp": lwp}), y, ec, r, fit)
        shapes["premium_only"] = tr["written_premium"].to_numpy(float)
        from src import fields
        from src.families import LGB, as_category
        drop = ("region", "make_model", "occupation", "yr", "holdout")
        fe_, fd_ = fields.tree_fields(tr, 11), fields.tree_fields(dv, 11)
        cols = [c for c in fe_["num"] + fe_["cat"] if c in fd_["num"] + fd_["cat"] and c not in drop]
        cats = [c for c in cols if c in fe_["cat"]]
        both = as_category(pd.concat([dv[cols], tr[cols]], ignore_index=True), cats)
        Xd, Xe = both.iloc[:len(dv)].reset_index(drop=True), both.iloc[len(dv):].reset_index(drop=True)
        gd = dv["customer_id"].astype(str).to_numpy()
        yd, ed = dv["costc"].to_numpy(float), (dv["exposure"] * dv["completion"]).to_numpy(float)
        a2, b2 = v.make_folds(len(dv), [11], n_folds=5, groups=gd)[11][0]
        mt = LGB("tweedie", 1.5, 31, 400, 0.05, 3000, 4, seed=11).fit(Xd.iloc[a2], yd[a2], ed[a2], val=(Xd.iloc[b2], yd[b2], ed[b2], None, None))
        mt = LGB("tweedie", 1.5, 31, 400, 0.05, mt.best_iter_, 4, seed=11).fit(Xd, yd, ed)
        rt = mt.rate(Xe)
        trees[line] = dict(trees=int(mt.best_iter_), inputs=cols)
        shapes["tree_part1"] = rt
        shapes["glm_tree_blend"] = np.exp(0.5 * np.log(r) + 0.5 * np.log(rt))
        own = TweedieGLM().fit(Ft[fit], y[fit], ec[fit])
        shapes["emerging-only"] = own.rate(Ft)
        sc = ~fit
        lt = float(p1[line]["final"]["line_trend_normalised"])
        lc23 = float(y[fit].sum() / ec[fit].sum())
        elf = float(tr.loc[fit, "cost"].sum() / tr.loc[fit, "costc"].sum())
        rows[line] = dict(y=tr.loc[sc, "cost"].to_numpy(float), e=e[sc], comp=tr.loc[sc, "completion"].to_numpy(float),
                          g=(line + ":" + tr.loc[sc, "customer_id"].astype(str)).to_numpy(),
                          shapes={k: s[sc] for k, s in shapes.items()}, fit_mean={k: float(np.sum(s[fit] * ec[fit]) / ec[fit].sum()) for k, s in shapes.items()},
                          lc_trended=lc23 * (1 + lt) * elf)
    res = {}
    names = list(rows["motor"]["shapes"])
    for lvl in ("perfect", "trended_2023"):
        res[lvl] = {}
        devs = {}
        for n in names:
            ys, mus, gs = [], [], []
            for line, d in rows.items():
                rate = d["shapes"][n]
                if lvl == "perfect":
                    mu = rate * d["e"] * d["comp"]
                    mu = mu * d["y"].sum() / mu.sum()
                else:
                    mu = rate / d["fit_mean"][n] * d["lc_trended"] * d["e"] * d["comp"]
                ys.append(d["y"]); mus.append(mu); gs.append(d["g"])
            y, mu, g = np.concatenate(ys), np.concatenate(mus), np.concatenate(gs)
            dv_ = v.tweedie_unit_deviance(y, mu, 1.5)
            null = v.tweedie_unit_deviance(y, np.full(len(y), y.mean()), 1.5).sum()
            devs[n] = dv_
            per_line = {}
            for l, m, yy in zip(rows, mus, ys):
                per_line[l] = float(1 - v.tweedie_unit_deviance(yy, m, 1.5).sum() / v.tweedie_unit_deviance(yy, np.full(len(yy), yy.mean()), 1.5).sum())
            res[lvl][n] = dict(d2=float(1 - dv_.sum() / null), d2_by_line=per_line, predicted_over_actual={l: float(np.sum(m) / np.sum(yy)) for l, m, yy in zip(rows, mus, ys)})
        res[lvl]["base+region+gender_over_base"] = v.paired_bootstrap_groups(devs["base"], devs["base+region+gender"], g)
        print(lvl, {n: round(res[lvl][n]["d2"], 4) for n in names}, flush=True)
        print("   D2 by line", {n: {l: round(x, 4) for l, x in res[lvl][n]["d2_by_line"].items()} for n in names}, flush=True)
        print("   predicted/actual by line", {n: {l: round(x, 3) for l, x in res[lvl][n]["predicted_over_actual"].items()} for n in names}, flush=True)
    res["premium_model_r2_2024"] = r2
    res["tree_part1"] = trees
    print("premium model R2 on 2024", r2)
    write_evidence({"part2_gap_diagnostic": res}, OUT)


if __name__ == "__main__":
    main()
