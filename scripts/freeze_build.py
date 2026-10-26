"""The frozen Part 1 model (check-in 5, O1 (a), O2 (b); D25): the blend of families, refitted on every non-hold-out
row 2021-2024 and priced for 2025.

Per line:
  1. each family with a pooled step 8 weight above zero is refitted on 2021-2024 with its step 7 tuned setting and
     tree count, and predicts every 2025 row at the 2024 year level;
       motor      0.666 EBM per peril (frequency x severity), 0.334 LightGBM started from the W2 GLM with the
                  held-over factors                                                   plain weighted average
       buildings  0.788 LightGBM started from the W2 GLM with the held-over factors, 0.212 CatBoost   log average
       contents   0.910 standardised spline GLM, 0.058 LightGBM, 0.032 EBM            plain weighted average
  2. contents rows whose security level or high-value items stay blank are priced, in every family, at the
     exposure-weighted average over the known values (P8-R2 (a));
  3. each family's 2025 predictions are scaled to one common total, mixed with the pooled weights, and the blend is
     scaled to the line's 2025 level from the decided chain (line excess factor, off-balance to the
     weather-normalised indication, 2025 weather as one factor per line), taken from the W2 file w2_glm_v1u.
     As in step 8, the weights decide ranking; the level comes from the indication.
Run: .venv/Scripts/python.exe -I scripts/freeze_build.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import scripts.phase5_step7 as s7
from scripts.phase5_step4 import hist_columns
from scripts.submit_w2 import ALL, FIT_YEARS, WORK4, w2_frame
from src import market
from src import build, data_rules as dr, fields, level as lv
from src.evidence import write_evidence
from src.families import as_category, as_string
from src.glm import TweedieGLM, fit_edges
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT, WORK7 = market.OUT, market.WORK / "phase5_step7"
NAME = "blend_final_all_v1" if ALL else "blend_frozen_v1"     # P7_ALL=1: the phase 7 refit on every customer
MIX = {"motor": "arith", "buildings": "geo", "contents": "arith"}
# step 8 family column -> how to build it here
COLMAP = {"motor": {"FS:ebm": ("fs", "ebm"), "F4+all": ("f4", True)},
          "buildings": {"F4+all": ("f4", True), "F2": ("cat", False)},
          "contents": {"F1": ("glum_spline", False), "F2": ("lgb", False), "F3": ("ebm", False)}}
BLANK = {"contents": {"security_level": "security", "high_value_items": "hvi"}}   # raw field -> W2 factor


def log(*a):
    import time
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def frames(df: pd.DataFrame, line: str, edges, h, ks, fit: np.ndarray, test: np.ndarray, last: int = max(FIT_YEARS)) -> dict:
    """W2 GLM frame, tree frame (with the held-over raw fields) and spline frame for training and 2025 rows together."""
    G = w2_frame(df, line, edges, h, ks)
    G.loc[test, "yr"] = str(last)
    gcat = [c for c in G.columns if not pd.api.types.is_float_dtype(G[c])]
    G = as_category(G, gcat)
    hc = hist_columns(h.reindex(df["row_id"]).reset_index(drop=True), ks, line)
    carried = [c for t, _ in s7.CARRIED[line].values() for c in t]
    fl = None
    if market.TREE_INPUTS == "wide":
        # Y2: every legitimate field (src/fields.py); codes with many levels are encoded in family_predictions,
        # once the W2 GLM's expected cost exists. The carried raw fields are already among them; history is kept.
        fl = fields.tree_fields(df[fit], load_thresholds()["code_levels_max"])
        num, cat = fl["num"], fl["cat"]
        carried = [c for c in carried if c in ("h_c", "h_a", "h_r")]
    else:
        num, cat = s7.RAW_NUM["all"] + s7.RAW_NUM[line], s7.RAW_CAT["all"] + s7.RAW_CAT[line]
    T = df[num].astype(float).copy()
    T["h_p"] = G["h_p"].to_numpy()
    T["yr"] = df["yr"].astype(float).to_numpy()
    T.loc[test, "yr"] = float(last)
    for c in cat:
        T[c] = df[c].to_numpy()
    for c in carried:
        T[c] = hc[c].to_numpy() if c in ("h_c", "h_a", "h_r") else df[c].astype(float).to_numpy()
    tcat = cat + ["h_r"]
    out = dict(G=G, T=as_category(T, tcat), Ts=as_string(T, tcat), tcat=tcat, tbase=num + ["h_p", "yr"] + cat,
               carried=carried, fields=fl)
    if line in s7.SPLINES:
        from sklearn.preprocessing import SplineTransformer
        th = load_thresholds()
        S = G[[c for c in G.columns]].copy()
        for band, raw in s7.SPLINES[line].items():
            x = df[raw].astype(float)
            x = np.log(x.clip(lower=1)) if raw == "sum_insured" else x
            x = x.fillna(x[fit].median()).to_numpy().reshape(-1, 1)
            st = SplineTransformer(n_knots=int(th["spline_knots"]), degree=3, knots="quantile", extrapolation="linear", include_bias=False).fit(x[fit])
            B = st.transform(x)
            S = S.drop(columns=band)
            for j in range(B.shape[1]):
                S[f"{band}_s{j}"] = B[:, j]
        out["S"] = S
    return out


def expand_blanks(te: pd.DataFrame, tr: pd.DataFrame, line: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """2025 rows with every remaining blank (P8-R2) replaced by each known value in turn: returns the expanded rows,
    the index of the original row for each, and its weight (training exposure share; product over blank fields)."""
    rows = te.assign(_orig=np.arange(len(te)), _w=1.0)
    for raw in BLANK.get(line, {}):
        share = tr.loc[tr[raw].notna()].groupby(raw)["exposure"].sum()
        share = share / share.sum()
        blank = rows[raw].isna()
        keep = rows[~blank]
        rep = rows[blank].merge(pd.DataFrame({raw + "_v": share.index, "_s": share.to_numpy()}), how="cross")
        rep[raw] = rep[raw + "_v"]; rep["_w"] = rep["_w"] * rep["_s"]
        rows = pd.concat([keep, rep.drop(columns=[raw + "_v", "_s"])], ignore_index=True)
    return rows.drop(columns=["_orig", "_w"]), rows["_orig"].to_numpy(), rows["_w"].to_numpy()


def collapse(pred: np.ndarray, orig: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(orig, weights=pred * w, minlength=n) / np.bincount(orig, weights=w, minlength=n)


def family_predictions(line: str, tr: pd.DataFrame, te0: pd.DataFrame, h, ks, tuned: dict, th, score_year: int, last: int) -> dict:
    """Fit every family of the frozen blend on `tr` and price the rows of `te0` at the year `last` (cost per row, before
    any level). Remaining contents blanks are averaged over the known values (P8-R2). Key "ref" is the W2 GLM."""
    te, orig, wgt = expand_blanks(te0, tr, line)
    df = pd.concat([tr, te.assign(yr=score_year)], ignore_index=True)
    fit = np.r_[np.ones(len(tr), bool), np.zeros(len(te), bool)]
    test = ~fit
    Fr = frames(df, line, fit_edges(tr, line), h, ks, fit, test, last)
    e = df["exposure"].to_numpy()
    ec = e * df["completion"].to_numpy()
    y = df["costc"].to_numpy()
    D = dict(tcat_used=[], line=line)
    Gw = Fr["G"].drop(columns=[c for c in ("h_c", "h_a", "h_r") if c in Fr["G"]])
    g = TweedieGLM().fit(Gw[fit], y[fit], ec[fit])
    init = np.log(g.rate(Gw))
    preds = {"ref": collapse(np.exp(init[test]) * e[test], orig, wgt, len(te0))}
    if Fr["fields"]:
        # Y2: each code with many levels enters as its credibility-weighted log mod of capped cost against the W2 GLM
        for c in Fr["fields"]["encode"]:
            enc, _k = fields.credibility_encode(df[c], df["yr"].to_numpy(), y, np.exp(init) * ec, fit,
                                                df["customer_id"].astype(str).to_numpy(), int(th["split_seed_1"]))
            for f in ("T", "Ts"):
                Fr[f][f"enc_{c}"] = enc
            Fr["tbase"].append(f"enc_{c}")
    for col, (kind, carried) in COLMAP[line].items():
        if kind == "fs":
            X = Fr["Ts"][Fr["tbase"]]
            b = tuned["ebm"]["best"]
            tot = np.zeros(int(test.sum()))
            for p in lv.perils_of(tr):
                # claims that cost something: a nil claim (net cost 0) has no severity; one hold-out motor claim
                # (CL45729785) is the only case, so the frozen non-hold-out fits are unchanged
                n = np.where(df[f"costc_{p}"].fillna(0).to_numpy() > 0, df[f"n_{p}"].fillna(0).to_numpy(), 0.0).astype(float)
                ecp = e * df[f"completion_{p}"].to_numpy()
                cm = fit & (n > 0)
                mf = s7.make("ebm", b["power"], b["setting"], th, D, objective="poisson").fit(X[fit], n[fit], ecp[fit])
                ms = s7.make("ebm", b["power"], b["setting"], th, D, objective="gamma").fit(X[cm], df.loc[cm, f"costc_{p}"].to_numpy(), n[cm], weight=n[cm])
                tot += mf.rate(X[test]) * ms.rate(X[test])
                log(line, "EBM per peril", p)
            rate = tot
        else:
            b = tuned[kind]["best"]
            if kind == "glum_spline":
                X = Fr["S"][[c for c in Fr["S"].columns if c not in ("h_c", "h_a", "h_r") and not c.startswith(("m_ylic", "g_volx", "b_dist", "b_elev"))]]
            else:
                cols = Fr["tbase"] + (Fr["carried"] if carried else [])
                X = (Fr["Ts"] if kind in ("cat", "ebm") else Fr["T"])[cols]
                D["tcat_used"] = [c for c in X.columns if c in Fr["tcat"]]
            m = s7.make(kind, b["power"], b["setting"], th, D, n_trees=s7.n_trees_of(tuned, kind))
            ia, ib = (init[fit], init[test]) if kind == "f4" else (None, None)
            m.fit(X[fit], y[fit], ec[fit], init=ia)
            rate = m.rate(X[test], init=ib)
            log(line, col, kind, "fitted", "trees", getattr(m, "best_iter_", None))
        preds[col] = collapse(rate * e[test], orig, wgt, len(te0))
    return preds


def run() -> None:
    th = load_thresholds()
    data = build.build_all()
    ks = json.loads((WORK4 / "k.json").read_text())["final"]
    for r in ks.values():
        r["k"] = r["k_poisson_gamma"]
    hist = pd.read_parquet(WORK4 / "history_final.parquet")
    S8 = json.loads((OUT / "phase5_step8.json").read_text())["phase5_step8"]
    w2 = pd.read_csv(OUT / "submissions" / ("w2_glm_v1u_all.csv" if ALL else "w2_glm_v1u.csv"))
    w2["row_id"] = w2["row_id"].astype(str)
    sub, info = [], {}
    for line in ("contents", "buildings", "motor"):          # quickest first, so an error shows early
        tuned = json.loads((WORK7 / f"result_{line}.json").read_text())["tuned"]
        wk = S8["weights"][line][f"families|{MIX[line]}"]
        weights = {c: w for c, w in zip(wk["cols"], wk["pooled"]) if w > 1e-9}     # solver zeros come back as 1e-17
        weights = {c: w / sum(weights.values()) for c, w in weights.items()}
        assert set(weights) <= set(COLMAP[line]), (line, weights)     # a family the weights drop is simply not mixed in
        tr, te0 = data[line]
        tr = tr[(ALL | ~tr["holdout"]) & tr["yr"].isin(FIT_YEARS)].reset_index(drop=True)
        te0 = te0.reset_index(drop=True)
        h = hist[hist["line"] == line].drop(columns="line").drop_duplicates("row_id").set_index("row_id")
        preds = family_predictions(line, tr, te0, h, ks, tuned, th, market.TARGET_YEAR, max(FIT_YEARS))
        preds.pop("ref")
        target = w2.merge(te0[["row_id"]].astype(str), on="row_id")
        assert len(target) == len(te0)
        T_line = float(target["prediction"].sum())
        shaped = {c: p * (T_line / p.sum()) for c, p in preds.items()}
        if MIX[line] == "geo":
            bl = np.exp(sum(w * np.log(shaped[c]) for c, w in weights.items()))
        else:
            bl = sum(w * shaped[c] for c, w in weights.items())
        bl = bl * (T_line / bl.sum())
        # each family's 2025 prices at the blend's total, kept for the phase 6 disagreement referral and the ranges
        (market.WORK / "freeze").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row_id": te0["row_id"].astype(str), "blend": bl, **{f"fam:{c}": shaped[c] for c in weights}}).to_parquet(
            market.WORK / "freeze" / f"family_preds_{line}{'_all' if ALL else ''}.parquet")
        w2p = te0[["row_id"]].astype(str).merge(w2, on="row_id", how="left")["prediction"].to_numpy()
        info[line] = dict(weights=weights, mix=MIX[line], total=T_line, rows=len(te0),
                          corr_log_with_w2=float(np.corrcoef(np.log(bl), np.log(w2p))[0, 1]),
                          ratio_to_w2_q=np.quantile(bl / w2p, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
                          min_rate=float((bl / te0["exposure"]).min()),
                          family_corr={c: float(np.corrcoef(np.log(p), np.log(w2p))[0, 1]) for c, p in preds.items()})
        log(line, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in info[line].items() if k != "ratio_to_w2_q"}))
        sub.append(pd.DataFrame({"row_id": te0["row_id"].astype(str), "prediction": bl, "line": line}))
    sub = pd.concat(sub, ignore_index=True)
    ss = dr.read_raw("sample_submission")
    sub = ss[["row_id"]].astype(str).merge(sub, on="row_id", how="left", validate="one_to_one")
    assert sub["prediction"].notna().all() and (sub["prediction"] > 0).all() and len(sub) == len(ss)
    path = OUT / "submissions" / f"{NAME}.csv"
    sub[["row_id", "prediction"]].to_csv(path, index=False, float_format="%.4f")
    log("written", path, len(sub))
    write_evidence({"freeze_build": dict(file=str(path.relative_to(ROOT)), rows=len(sub), by_line=info)}, OUT / f"submission_{NAME}.json")


if __name__ == "__main__":
    run()
