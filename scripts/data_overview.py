"""Data overview for the actuary: portfolio by line and year, peril mix, claim sizes, and one-way relationships.

Aggregates only, written to data/work/data_overview.json (the page embeds them). Hold-out customers are left out so
their outcomes stay unread until phase 7. Loss cost and frequency use completed exposure (exposure x reporting
completion, P10), so 2024 is comparable with earlier years. Claims are uncapped here.
Run: .venv/Scripts/python.exe -I scripts/data_overview.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import data_rules as dr, targets as tg, validation as v
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]


def oneway(df, col, order=None, label=None):
    g = df.groupby(col, observed=True, dropna=False)
    t = pd.DataFrame({"exposure": g["exposure"].sum(), "ec": g["ec"].sum(), "en": g["en"].sum(), "n": g["n_claims"].sum(), "cost": g["cost"].sum(),
                      "wp": g["wp_exp"].sum()})
    t = t[t["exposure"] > 0]
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    base_lc = df["cost"].sum() / df["ec"].sum()
    out = []
    for k, r in t.iterrows():
        lc = r.cost / r.ec
        out.append(dict(level=str(k) if not (isinstance(k, float) and np.isnan(k)) else "blank",
                        exposure_share=r.exposure / t["exposure"].sum(), frequency=r.n / r.en,
                        severity=(r.cost / r.n) if r.n else None, loss_cost=lc, relativity=lc / base_lc,
                        premium_relativity=(r.wp / r.exposure) / (df["wp_exp"].sum() / df["exposure"].sum()),
                        claims=int(r.n)))
    return dict(variable=label or col, levels=out)


def main():
    th = load_thresholds()
    claims = dr.read_raw("claims")
    curves, weights, shares = tg.delay_curves(claims), tg.incidence_weights(claims), tg.peril_cost_shares(claims)
    cshares = tg.peril_count_shares(claims)
    ad, cu = dr.read_raw("addresses"), dr.read_raw("customers")
    w = dr.read_raw("weather")
    norms = w[w["year"] <= 2024].groupby("address_id")[["wind_exposure_index", "dry_summer_index", "rainfall_mm"]].mean()
    lines, cust_all = {}, []
    for line in dr.LINES:
        tr, te = dr.read_raw(f"policies_{line}"), dr.read_raw(f"test_{line}")
        cut = dr.sum_insured_cutoff(tr["sum_insured"], th["si_unit_multiple"]) if line != "motor" else None
        tr, te = dr.apply_policy_rules(tr, line, cut), dr.apply_policy_rules(te, line, cut)
        tr, te = dr.recover_from_history(tr, te, line)
        tr = tr.merge(ad, on="address_id", how="left").merge(cu, on="customer_id", how="left").join(norms, on="address_id")
        te = te.merge(ad, on="address_id", how="left")
        tr = tr.join(tg.row_completion(tr, line, curves, weights, shares))
        tr = tr.join(tg.row_completion(tr, line, curves, weights, cshares)[["completion"]].rename(columns={"completion": "completion_count"}))
        tr = tr.merge(tg.row_targets(tr, claims, line), on="row_id", how="left")
        lines[line] = (tr, te)
        cust_all.append(tr["customer_id"])
    cust = np.array(sorted(pd.concat(cust_all).unique()))
    keep = pd.Series(v.fitting_mask(ROOT / "data" / "work" / "holdout_lock.json", cust), index=cust)

    out = {"lines": {}, "notes": {}}
    for line, (tr, te) in lines.items():
        tr = tr[tr["customer_id"].map(keep).to_numpy()].copy()
        tr["yr"] = tr["period_start"].dt.year
        tr["ec"] = tr["exposure"] * tr["completion"]
        tr["en"] = tr["exposure"] * tr["completion_count"]
        tr["wp_exp"] = tr["written_premium"] * tr["exposure"]
        g = tr.groupby("yr")
        years = pd.DataFrame({"policies": g["policy_id"].nunique(), "rows": g.size(), "exposure": g["exposure"].sum(),
                              "claims": g["n_claims"].sum(), "cost": g["cost"].sum(), "ec": g["ec"].sum(), "en": g["en"].sum(),
                              "premium": g["wp_exp"].sum(), "completion": g["completion"].mean()})
        years["frequency"] = years.claims / years.en
        years["severity"] = years.cost / years.claims
        years["loss_cost"] = years.cost / years.ec
        years["premium_per_exposure"] = years.premium / years.exposure
        years["loss_ratio_developed"] = years.loss_cost / years.premium_per_exposure
        tey = dict(rows=len(te), exposure=float(te["exposure"].sum()), mean_exposure=float(te["exposure"].mean()),
                   new_business=float(te["new_business"].mean()),
                   premium_per_exposure=float((te["written_premium"] * te["exposure"]).sum() / te["exposure"].sum()))
        perils = [c[5:] for c in tr.columns if c.startswith("cost_")]
        pm = [dict(peril=p, claims=int(tr[f"n_{p}"].sum()), cost=float(tr[f"cost_{p}"].sum()),
                   frequency_per_1000=float(tr[f"n_{p}"].sum() / (tr["exposure"] * tr["completion_" + p]).sum() * 1000) if "completion_" + p in tr else float(tr[f"n_{p}"].sum() / tr["en"].sum() * 1000),
                   severity=float(tr[f"cost_{p}"].sum() / max(tr[f"n_{p}"].sum(), 1))) for p in perils]
        m = dr.match_claims_to_rows(tr, claims, line)
        m = m[~m["repudiated"] & (m["net_cost"] > 0)]
        sizes = m.groupby("peril")["net_cost"].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).unstack()
        sizes["max"] = m.groupby("peril")["net_cost"].max()
        ow = []
        age = "driver_age" if line == "motor" else "policyholder_age"
        tr["age_band"] = pd.cut(tr[age], [0, 20, 24, 29, 39, 49, 59, 69, 79, 200], labels=["17-20", "21-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"])
        tr["missed"] = pd.cut(tr["missed_payments_12m"], [-1, 0, 1, 99], labels=["0", "1", "2+"])
        if line == "motor":
            tr["ncb"] = tr["ncb_years"].clip(upper=10).map(lambda x: "10+" if x == 10 else str(int(x)))
            tr["mileage"] = pd.cut(tr["annual_mileage"], [-1, 0, 5000, 7500, 10000, 15000, 20000, 1e7], labels=["0", "1-5k", "5-7.5k", "7.5-10k", "10-15k", "15-20k", "20k+"]).astype("string").fillna("blank")
            tr["points"] = pd.cut(tr["licence_points"], [-1, 0, 3, 6, 99], labels=["0", "1-3", "4-6", "7+"])
            tr["vvg"] = pd.cut(tr["vehicle_value_group"], [0, 4, 8, 12, 16, 20], labels=["1-4", "5-8", "9-12", "13-16", "17-20"])
            tr["telem"] = np.where(~tr["telematics_opt_in"], "not opted in", pd.cut(tr["telematics_score"], [0, 40, 60, 80, 100], labels=["score 1-40", "41-60", "61-80", "81-100"]).astype(str))
            specs = [("age_band", None, "Driver age"), ("ncb", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"], "NCB years"),
                     ("vvg", None, "Vehicle value group"), ("cover_type", ["comprehensive", "third_party_fire_theft", "third_party_only"], "Cover"),
                     ("mileage", ["0", "1-5k", "5-7.5k", "7.5-10k", "10-15k", "15-20k", "20k+", "blank"], "Annual mileage"),
                     ("points", None, "Licence points"), ("credit_band", None, "Credit band"), ("missed", None, "Missed payments (12m)"),
                     ("vehicle_use", None, "Vehicle use"), ("telem", ["not opted in", "score 1-40", "41-60", "61-80", "81-100"], "Telematics"),
                     ("named_young_driver", None, "Named young driver"), ("region", None, "Region")]
        else:
            q = tr["sum_insured"].quantile(np.linspace(0, 1, 6)).to_numpy()
            tr["si_band"] = pd.cut(tr["sum_insured"], q, include_lowest=True, labels=[f"Q{i}" for i in range(1, 6)])
            tr["flood"] = tr["flood_zone"].map(lambda x: "blank" if pd.isna(x) else str(int(x)))
            tr["dry_clay"] = np.where(tr["soil_type"].eq("clay"), "clay, ", "other soil, ") + pd.cut(tr["dry_summer_index"], [0, 0.3, 0.6, 1.01], labels=["wet summers", "medium", "dry summers"]).astype(str)
            specs = [("si_band", None, "Sum insured fifth"), ("flood", ["1", "2", "3", "4", "5", "blank"], "Flood zone"),
                     ("property_type", None, "Property type"), ("age_band", None, "Policyholder age"),
                     ("credit_band", None, "Credit band"), ("missed", None, "Missed payments (12m)"), ("region", None, "Region")]
            if line == "buildings":
                tr["built"] = pd.cut(tr["year_built"], [0, 1919, 1944, 1969, 1989, 2009, 3000], labels=["pre-1920", "1920-44", "1945-69", "1970-89", "1990-2009", "2010+"])
                specs += [("built", None, "Year built"), ("construction_type", None, "Construction"),
                          ("dry_clay", ["other soil, wet summers", "other soil, medium", "other soil, dry summers", "clay, wet summers", "clay, medium", "clay, dry summers"], "Soil and summer dryness (address normal)")]
            else:
                tr["hvi"] = tr["high_value_items"].clip(upper=4).map(lambda x: "blank" if pd.isna(x) else ("4+" if x == 4 else str(int(x))))
                specs += [("tenure", None, "Tenure"), ("security_level", None, "Security"), ("hvi", ["0", "1", "2", "3", "4+", "blank"], "High-value items")]
        for col, order, label in specs:
            ow.append(oneway(tr, col, order, label))
        out["lines"][line] = dict(years=years.reset_index().to_dict("records"), test=tey, perils=pm,
                                  claim_sizes=sizes.reset_index().round(0).to_dict("records"), oneways=ow,
                                  rows=len(tr), customers=int(tr["customer_id"].nunique()))
    allc = pd.concat([lines[l][0][["customer_id"]].assign(line=l) for l in lines])
    allc = allc[allc["customer_id"].map(keep).to_numpy()]
    nl = allc.groupby("customer_id")["line"].nunique().value_counts().sort_index()
    out["notes"]["customers_by_lines_held"] = {int(k): int(v) for k, v in nl.items()}
    (ROOT / "data" / "work" / "data_overview.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    for line in out["lines"]:
        y = pd.DataFrame(out["lines"][line]["years"]).set_index("yr")
        print(line, y[["rows", "exposure", "claims", "frequency", "severity", "loss_cost", "premium_per_exposure", "loss_ratio_developed"]].round(3).to_string())


if __name__ == "__main__":
    main()
