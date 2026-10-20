"""Interim submission: the working base W2 (one capped Tweedie GLM per line), priced for 2025 with the decided level.

Per line, on every non-hold-out row 2021-2024:
  - the W2 GLM (V1 factors less the L8 drops, policy history log(mod), declared history, and the step 5 additions);
  - rate of each 2025 row at the 2024 year level;
  - times the line excess factor (K2, M2), the off-balance to the weather-normalised indication for 2025 (K3, K4;
    the model's 2024 level read at normal weather, as in step 6), and the 2025 weather as one factor per line (M3 (b)).
The fitted relativities are checked against the step 5 loader's final fit, so the 2025 rows get exactly the W2 model.
Run: .venv/Scripts/python.exe -I scripts/submit_w2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.phase5_l8_check import DROP, KEPT
from scripts.phase5_step4 import hist_columns
from src import market
from src import build, data_rules as dr, level as lv, weather as wx
from src.evidence import write_evidence
from src.glm import TweedieGLM, band, cat, features, fit_edges
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
import os
# Phase 7 final refit: P7_ALL=1 fits on every customer (hold-out opened 2026-10-20), with the all-customer history
ALL = os.environ.get("P7_ALL") == "1"
OUT, WORK4 = market.OUT, market.WORK / ("phase5_step4_all" if ALL else "phase5_step4")
FIT_YEARS = market.FIT_YEARS
# Blanks that stay blank after recovery from the same policy are UNKNOWN (decisions/missing_values.md: they claim like
# everyone else). Where their own level is too thin to estimate, a 2025 row with the blank is priced at the average
# of the known levels, weighted by training exposure. Pass --own-level to keep the P8-R rule (own level) instead.
UNKNOWN_AVERAGE = {"contents": ["security", "hvi"]}
OWN_LEVEL = "--own-level" in sys.argv
NAME = ("w2_glm_v1" if OWN_LEVEL else "w2_glm_v1u") + ("_all" if ALL else "")


def average_over_unknown(m, F: pd.DataFrame, Ft: pd.DataFrame, expo: np.ndarray, factors: list[str]) -> np.ndarray:
    """Rate of each row; where a listed factor is 'blank', the exposure-weighted average over its known levels
    (several blank factors: over every combination, weights multiplied)."""
    rate = m.rate(F)
    shares = {}
    for f in factors:
        ex = pd.Series(expo).groupby(Ft[f].to_numpy()).sum().drop("blank", errors="ignore")
        shares[f] = ex / ex.sum()
    blank = {f: (F[f] == "blank").to_numpy() for f in factors}
    any_blank = np.logical_or.reduce(list(blank.values()))
    sub = F[any_blank].copy()
    combos = [({}, 1.0)]
    for f in factors:
        combos = [({**c, f: lvl}, w * sh) for c, w in combos for lvl, sh in shares[f].items()]
    acc = np.zeros(len(sub))
    for c, w in combos:
        S = sub.copy()
        wt = np.ones(len(sub))
        for f, lvl in c.items():
            bf = blank[f][any_blank]
            S.loc[bf, f] = lvl
            wt = wt * np.where(bf, shares[f][lvl], 1.0)
        acc += m.rate(S) * wt
    # combinations repeat for rows where only some factors are blank; normalise by the summed weights
    norm = np.zeros(len(sub))
    for c, w in combos:
        wt = np.ones(len(sub))
        for f, lvl in c.items():
            wt = wt * np.where(blank[f][any_blank], shares[f][lvl], 1.0)
        norm += wt
    rate[any_blank] = acc / norm
    return rate


def w2_frame(df: pd.DataFrame, line: str, edges: dict, h: pd.DataFrame, ks: dict) -> pd.DataFrame:
    """The W2 columns for any rows (training or 2025), built as steps 4 and 5 built them."""
    F = features(df, line, edges)
    hc = hist_columns(h.reindex(df["row_id"]).reset_index(drop=True), ks, line)
    F["h_p"] = hc["h_p"].to_numpy()
    F["h_d"] = band(df["prior_claims_5y"], [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"]).to_numpy()
    if line == "motor":
        F["g_volx"] = cat(df["voluntary_excess"]).to_numpy()
        F["m_vage"] = band(df["vehicle_age"], [-1, 1, 3, 6, 10, 15, 999]).to_numpy()
    if line == "buildings":
        clay = df["soil_type"] == "clay"
        F["b_treeclay"] = cat(df["trees_within_5m"].astype("string") + "|" + clay.astype("string")).to_numpy()
    return F.drop(columns=DROP[line])


def run() -> None:
    th = load_thresholds()
    weather = dr.read_raw("weather")
    data = build.build_all()
    ks = json.loads((WORK4 / "k.json").read_text())["final"]
    for r in ks.values():
        r["k"] = r["k_poisson_gamma"]
    hist = pd.read_parquet(WORK4 / "history_final.parquet")
    sub, info = [], {}
    for line in dr.LINES:
        tr, te = data[line]
        tr = tr[(ALL | ~tr["holdout"]) & tr["yr"].isin(FIT_YEARS)].reset_index(drop=True)
        te = te.reset_index(drop=True)
        h = hist[hist["line"] == line].drop(columns="line").drop_duplicates("row_id").set_index("row_id")
        edges = fit_edges(tr, line)
        Ft, Fe = w2_frame(tr, line, edges, h, ks), w2_frame(te, line, edges, h, ks)
        Fe["yr"] = str(max(FIT_YEARS))
        e, c = tr["exposure"].to_numpy(), tr["completion"].to_numpy()
        m = TweedieGLM().fit(Ft, tr["costc"], e * c)
        # check (Part 1 only): the same relativities as the step 5 loader's final fit of W2
        if market.NAME == "part1":
            from scripts.phase5_step5 import fit_score, load_plan, spec_columns   # check on the frozen (non-hold-out) fit only
            full = {l: d[0][~d[0]["holdout"]] for l, d in data.items()}
            P = load_plan(line, "final", th, full, weather)
            ref = fit_score(P, spec_columns(P, dict.fromkeys(KEPT[line]), DROP[line]))[0].relativities()
            mine = m.relativities()
            a = ref.assign(factor=ref["factor"].str.split(":").str[-1]).set_index(["factor", "level"])["relativity"]
            b = mine.set_index(["factor", "level"])["relativity"]
            assert ALL or (set(a.index) == set(b.index) and np.allclose(a.loc[b.index], b, rtol=1e-6)), f"{line}: W2 relativities differ"
        rate_te = m.rate(Fe) if OWN_LEVEL or line not in UNKNOWN_AVERAGE else average_over_unknown(m, Fe, Ft, e, UNKNOWN_AVERAGE[line])
        elf = float(tr["cost"].sum() / tr["costc"].sum())
        # level: weather-normalised indication for 2025 against the model's 2024 level at normal weather
        trx = wx.attach(tr, weather, wx.normals(weather, FIT_YEARS))
        b_el = wx.elasticities(trx, line)
        perils = lv.perils_of(tr)
        share = {p: float(tr[f"cost_{p}"].sum() / tr["cost"].sum()) for p in perils}
        yf = wx.year_factor(trx, line, b_el, share)
        trn = trx.copy()
        for p, (wv, bb) in b_el.items():
            trn[f"wm_{p}"] = np.exp(bb * trn[f"dev_{wv}"].fillna(0.0))
        trends = lv.peril_trends(trn, line, th["trend_min"], th["trend_max"])
        ind = lv.indication(trn, trends, market.TARGET_YEAR)["indicated_capped_lc"]
        last = (tr["yr"] == max(FIT_YEARS)).to_numpy()
        model_lc = float(np.sum(m.rate(Ft[last]) * e[last] / yf[last]) / e[last].sum())
        ob = float(ind.sum() / model_lc)
        gt = lv.group_trend(trends, ind, perils)
        gap = float(ind.sum() / (model_lc * (1 + gt)) - 1)
        tex = wx.attach(te, weather, wx.normals(weather, FIT_YEARS))
        w25 = float(np.average(wx.year_factor(tex, line, b_el, share), weights=te["exposure"]))
        pred = rate_te * elf * ob * w25 * te["exposure"].to_numpy()
        written = te["written_premium"].to_numpy() * te["exposure"].to_numpy()
        info[line] = dict(rows=len(te), excess_factor=elf, off_balance=ob, weather_2025=w25, indication_capped_lc=float(ind.sum()),
                          model_lc_2024_normal=model_lc, line_trend=gt, c4_gap=gap, c4_band=th["level_band"],
                          predicted=float(pred.sum()), written=float(written.sum()), predicted_over_written=float(pred.sum() / written.sum()),
                          mean_per_exposure=float(pred.sum() / te["exposure"].sum()))
        print(f"{line}: excess {elf:.4f}, off-balance {ob:.4f}, 2025 weather {w25:.4f}, C4 gap {gap:+.2%}, "
              f"predicted {pred.sum():,.0f} against written {written.sum():,.0f} ({pred.sum() / written.sum():.3f})", flush=True)
        sub.append(pd.DataFrame({"row_id": te["row_id"].astype(str), "prediction": pred, "line": line}))
    sub = pd.concat(sub, ignore_index=True)
    ss = dr.read_raw("sample_submission")
    sub = ss[["row_id"]].astype(str).merge(sub, on="row_id", how="left", validate="one_to_one")
    assert sub["prediction"].notna().all() and (sub["prediction"] > 0).all() and len(sub) == len(ss)
    path = OUT / "submissions" / f"{NAME}.csv"
    sub[["row_id", "prediction"]].to_csv(path, index=False, float_format="%.4f")
    print("written", path, len(sub))
    write_evidence({"submission_w2": dict(file=str(path.relative_to(ROOT)), rows=len(sub), by_line=info, own_level=OWN_LEVEL)},
                   OUT / f"submission_{NAME}.json")


if __name__ == "__main__":
    run()
