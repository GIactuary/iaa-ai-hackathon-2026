"""Part 2 (emerging market) model, built to the actuary's decisions (docs/specs/SPEC_part2_build.md).

Milestone 1, the check file (P2-P (a)):
  - base GLM (P2-G (a), P2-C (b)): Part 1's W2 factors adapted to the fields both markets carry (age as the five
    emerging bands; no credit band, mileage or telematics; region left out, the markets' regions differ), fitted on
    every Part 1 row 2021-2024 on capped cost with completed exposure; the emerging rows take its relativities
  - level (P2-T, P2-L, P2-R, Y4): emerging capped loss cost per peril at normal weather, trended to 2025 with Part 1's
    selected peril trends; Part 1's weather elasticities; excess-loss factor mixed by line (motor own, property
    Part 1's excess shares by peril); 2025 weather as one factor per line
  - every emerging figure uses the fitting rows only (hold-out locked, P2-H)
  - honest score of the shape on emerging rows: the base against an emerging-only GLM fitted on 2023 (out of time)

Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_model.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

assert os.environ.get("MARKET") == "part2", "run with MARKET=part2"

from scripts.phase5_l8_check import DROP           # Part 1's W2 drops (L8)
from src import build, data_rules as dr, level as lv, market, validation as v, weather as wx
from src.evidence import write_evidence
from src.glm import TweedieGLM, band, cat, quantile_edges
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
DEV_CACHE = ROOT / market.CFG["developed_cache"]
P1_LEVEL = ROOT / "outputs" / "phase5_k3k4.json"
OUT = market.OUT
AGE_BANDS = ([0, 24, 34, 49, 64, 200], ["17-24", "25-34", "35-49", "50-64", "65+"])
BASE_YEAR = "2024"


def developed(line: str) -> pd.DataFrame:
    """Part 1's built rows (all customers; Part 1's hold-out was opened in its phase 7), ages to emerging bands."""
    df = pd.read_parquet(DEV_CACHE / f"{line}_train.parquet")
    age = "driver_age" if line == "motor" else "policyholder_age"
    df[f"{age}_band"] = pd.cut(df[age], AGE_BANDS[0], labels=AGE_BANDS[1]).astype("string")
    return df


def w2_factors(df: pd.DataFrame, line: str, edges: dict) -> pd.DataFrame:
    """Part 1's W2 (src/glm.features plus declared history and the step 5 additions, less the L8 drops), adapted:
    age as the emerging bands; credit band, mileage and telematics removed; region removed (P2-C (b))."""
    f = pd.DataFrame({
        "urban": cat(df["urban_density"]), "pay": cat(df["payment_method"]),
        "missed": band(df["missed_payments_12m"], [-1, 0, 1, 99], ["0", "1", "2+"]),
        "channel": cat(df["channel"]), "new_bus": cat(df["new_business"]),
        "age": cat(df["driver_age_band" if line == "motor" else "policyholder_age_band"]),
        "yr": cat(df["yr"]),
        "h_d": band(df["prior_claims_5y"], [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"]),
    }, index=df.index)
    if line == "motor":
        f["ncb"] = cat(df["ncb_years"].clip(upper=10))
        f["vvg"] = cat(df["vehicle_value_group"])
        f["cover"], f["use"], f["fuel"], f["body"] = cat(df["cover_type"]), cat(df["vehicle_use"]), cat(df["fuel_type"]), cat(df["body_type"])
        f["points"] = band(df["licence_points"], [-1, 0, 3, 6, 99])
        f["young"] = cat(df["named_young_driver"])
        f["g_volx"] = cat(df["voluntary_excess"])
        f["m_vage"] = band(df["vehicle_age"], [-1, 1, 3, 6, 10, 15, 999])
    else:
        f["si"] = band(df["sum_insured"], edges["sum_insured"])
        f["ptype"], f["flood"] = cat(df["property_type"]), cat(df["flood_zone"])
        f["ad_cover"] = cat(df["accidental_damage_cover"])
    if line == "buildings":
        f["constr"], f["soil"], f["occ"] = cat(df["construction_type"]), cat(df["soil_type"]), cat(df["occupancy"])
        f["built"] = band(df["year_built"], [0, 1919, 1944, 1969, 1989, 2009, 3000])
        f["beds"] = cat(df["bedrooms"].clip(upper=5))
        clay = df["soil_type"] == "clay"
        f["b_treeclay"] = cat(df["trees_within_5m"].astype("string") + "|" + clay.astype("string"))
    if line == "contents":
        f["tenure"], f["security"] = cat(df["tenure"]), cat(df["security_level"])
        f["hvi"] = cat(df["high_value_items"].clip(upper=4))
        f["bike"] = cat(df["bicycle_cover"])
    return f.drop(columns=[c for c in DROP[line] if c in f.columns])


def unknown_blanks(Fd: pd.DataFrame, Fe: pd.DataFrame) -> list[str]:
    """Factors whose blank level in Part 1 cannot stand for the emerging market's blanks: blank on more than twice
    the share of emerging test rows as of Part 1 rows. Part 1 recovered such blanks from history, so its blank level
    is nearly empty (contents security blank 0.005). Flood zone blanks (same share in both markets, informative per
    decisions/missing_values.md) keep their own level."""
    return [c for c in Fd.columns if (Fe[c] == "blank").mean() > 0 and (Fe[c] == "blank").mean() > 2 * (Fd[c] == "blank").mean()]


def average_over_unknown(m: TweedieGLM, F: pd.DataFrame, Ffit: pd.DataFrame, expo: np.ndarray, factors: list[str]) -> np.ndarray:
    """P2-B (a), Part 1's treatment of thin blank levels (scripts/submit_w2.py): a blank in these factors is priced at
    the average of the factor's known levels, weighted by the emerging fitting rows' exposure."""
    rel = m.relativities().set_index(["factor", "level"])["relativity"]
    rate = m.rate(F)
    for c in factors:
        known = Ffit[c] != "blank"
        w = pd.Series(expo[known.to_numpy()]).groupby(Ffit.loc[known, c].to_numpy()).sum()
        avg = float(np.average([rel.get((c, l), 1.0) for l in w.index], weights=w.to_numpy()))
        blank_rel = float(rel.get((c, "blank"), 1.0))
        rate = np.where(F[c].to_numpy() == "blank", rate * avg / blank_rel, rate)
    return rate


def tweedie_dev(y, mu):
    return v.tweedie_unit_deviance(y, mu, 1.5)


def common_level_d2(y, ec, rate) -> dict:
    """D2 of a rate's shape on scored rows: prediction rescaled to the actual total, against a flat rate."""
    mu = rate * ec
    mu = mu * y.sum() / mu.sum()
    flat = ec * y.sum() / ec.sum()
    d, d0 = tweedie_dev(y, mu), tweedie_dev(y, flat)
    return dict(deviance=float(d.sum()), null=float(d0.sum()), d2=float(1 - d.sum() / d0.sum())), d


def part1_level() -> dict:
    return json.loads(P1_LEVEL.read_text(encoding="utf-8"))["phase5_k3k4"]["by_line"]


def run() -> None:
    th = load_thresholds()
    weather = dr.read_raw("weather")
    norms = wx.normals(weather, market.FIT_YEARS)
    p1 = part1_level()
    data = build.build_all()
    sub, info = [], {}
    for line in dr.LINES:
        tr, te = data[line]
        tr = tr[~tr["holdout"]].reset_index(drop=True)            # P2-H: the hold-out stays locked
        te = te.reset_index(drop=True)
        dv = developed(line)
        edges = {"sum_insured": quantile_edges(dv["sum_insured"], 10)} if line != "motor" else {}

        # base GLM on Part 1 rows (P2-C (b), P2-G (a))
        Fd = w2_factors(dv, line, edges)
        base = TweedieGLM().fit(Fd, dv["costc"], dv["exposure"] * dv["completion"])
        Ft, Fe = w2_factors(tr, line, edges), w2_factors(te, line, edges)
        Ft["yr"], Fe["yr"] = BASE_YEAR, BASE_YEAR
        unknown = unknown_blanks(Fd, Fe)
        e_fit = tr["exposure"].to_numpy(float)
        r_tr, r_te = average_over_unknown(base, Ft, Ft, e_fit, unknown), average_over_unknown(base, Fe, Ft, e_fit, unknown)

        # honest shape score, out of time on emerging 2024: base against an emerging-only GLM fitted on 2023
        s24, s23 = (tr["yr"] == 2024).to_numpy(), (tr["yr"] == 2023).to_numpy()
        y, ec = tr["costc"].to_numpy(float), (tr["exposure"] * tr["completion"]).to_numpy(float)
        Fe23 = w2_factors(tr, line, edges)
        Fe23["yr"] = BASE_YEAR
        own = TweedieGLM().fit(Fe23[s23], y[s23], ec[s23])
        sc_base, d_base = common_level_d2(y[s24], ec[s24], r_tr[s24])
        sc_own, d_own = common_level_d2(y[s24], ec[s24], own.rate(Fe23[s24]))
        g = tr.loc[s24, "customer_id"].astype(str).to_numpy()
        boot = v.paired_bootstrap_groups(d_own, d_base, g)            # gain > 0: the Part 1 base is better

        # level chain on emerging fitting rows (P2-T, P2-L, P2-R; weather as Y4 with Part 1's elasticities)
        el = {p: (e["variable"], float(e["b"])) for p, e in p1[line]["final"]["elasticities"].items()}
        perils = lv.perils_of(tr)
        share = {p: float(tr[f"cost_{p}"].sum() / tr["cost"].sum()) for p in perils}
        trx = wx.attach(tr, weather, norms)
        yf = wx.year_factor(trx, line, el, share)
        trn = trx.copy()
        for p, (wv, b) in el.items():
            trn[f"wm_{p}"] = np.exp(b * trn[f"dev_{wv}"].fillna(0.0))
        trends = lv.peril_trends(trn, line, th["trend_min"], th["trend_max"])          # emerging fit, for the record
        own_trends = trends[["freq_trend", "sev_trend"]].copy()
        p1t = {t["peril"]: t for t in p1[line]["final"]["trends_normalised"]}
        for p in trends.index:
            trends.loc[p, "freq_trend"] = float(np.clip(p1t[p]["freq_trend"], th["trend_min"], th["trend_max"]))
            if pd.isna(trends.loc[p, "sev_step_level"]):
                trends.loc[p, "sev_trend"] = float(np.clip(p1t[p]["sev_trend"], th["trend_min"], th["trend_max"]))
        ind = lv.indication(trn, trends, market.TARGET_YEAR)["indicated_capped_lc"]
        last = s24
        e = tr["exposure"].to_numpy(float)
        model_lc = float(np.sum(r_tr[last] * e[last] / yf[last]) / e[last].sum())
        ob = float(ind.sum() / model_lc)
        gt = lv.group_trend(trends, ind, perils)

        # excess-loss factor (P2-L (a)): motor its own large claims; property Part 1's excess shares by peril
        if line == "motor":
            elf = float(tr["cost"].sum() / tr["costc"].sum())
            elf_basis = "emerging own"
        else:
            dev_ratio = {p: float(dv[f"cost_{p}"].sum() / dv[f"costc_{p}"].sum()) for p in perils}
            elf = float(sum(tr[f"costc_{p}"].sum() * dev_ratio[p] for p in perils) / tr["costc"].sum())
            elf_basis = "Part 1 excess shares by peril on emerging capped cost"
        elf_own = float(tr["cost"].sum() / tr["costc"].sum())

        tex = wx.attach(te, weather, norms)
        w25 = float(np.average(wx.year_factor(tex, line, el, share), weights=te["exposure"]))
        pred = r_te * ob * elf * w25 * te["exposure"].to_numpy(float)
        written = (te["written_premium"] * te["exposure"]).to_numpy(float)
        info[line] = dict(rows=len(te), base_rows=len(dv), blanks_at_known_average=unknown, shape_v1=dict(part1_base=sc_base, emerging_own=sc_own, base_over_own=boot),
                          excess_factor=elf, excess_basis=elf_basis, excess_factor_own=elf_own, off_balance=ob, weather_2025=w25,
                          indication_capped_lc=float(ind.sum()), model_lc_2024_normal=model_lc, line_trend=gt,
                          trends_used=trends[["freq_trend", "sev_trend"]].to_dict(), trends_emerging_fit=own_trends.to_dict(),
                          predicted=float(pred.sum()), written=float(written.sum()), predicted_over_written=float(pred.sum() / written.sum()),
                          mean_per_exposure=float(pred.sum() / te["exposure"].sum()),
                          relativities=base.relativities().to_dict(orient="records"))
        print(f"{line}: shape D2 base {sc_base['d2']:.4f} vs emerging-only {sc_own['d2']:.4f} (gain {boot['gain']:,.0f} [{boot['lo']:,.0f}, {boot['hi']:,.0f}]); "
              f"ELF {elf:.4f} ({elf_basis}; own {elf_own:.4f}), off-balance {ob:.4f}, trend {gt:+.2%}, 2025 weather {w25:.4f}; "
              f"predicted {pred.sum():,.0f} against written {written.sum():,.0f} ({pred.sum() / written.sum():.3f})", flush=True)
        sub.append(pd.DataFrame({"row_id": te["row_id"].astype(str), "prediction": pred}))
    sub = pd.concat(sub, ignore_index=True)
    ss = dr.read_raw("sample_submission")
    sub = ss[["row_id"]].astype(str).merge(sub, on="row_id", how="left", validate="one_to_one")
    assert sub["prediction"].notna().all() and (sub["prediction"] > 0).all() and len(sub) == len(ss)
    path = OUT / "submissions" / "part2_check_v1.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub[["row_id", "prediction"]].to_csv(path, index=False, float_format="%.4f")
    print("written", path, len(sub))
    write_evidence({"part2_check_v1": dict(file=str(path.relative_to(ROOT)), rows=len(sub), by_line=info)}, OUT / "part2_check_v1.json")


if __name__ == "__main__":
    run()
