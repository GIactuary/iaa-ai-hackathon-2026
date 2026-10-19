"""Phase 4: three baselines scored on the agreed out-of-time scorings A and B, and an early submission file.

Baselines (spec E1):
  constant  one rate per line per unit of completed exposure
  incumbent written premium times exposure (the insurer's current price), rescaled to the common level when scored
  glm       a plain Tweedie GLM per line (power 1.5, log link), main effects only, accident-year factor and the row
            completion offset (P1, P10); no capping, no engineered features, no weather
Hold-out customers are never used. Run from the repo root: .venv/Scripts/python.exe -I scripts/phase4_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import TweedieRegressor

from src import data_rules as dr, targets as tg, validation as v
from src.evidence import write_evidence
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "work" / "holdout_lock.json"
OUT = ROOT / "outputs"
P = v.TWEEDIE_POWER


def band(x, edges, labels=None):
    b = pd.cut(x, edges, labels=labels).astype("string")
    return b.fillna("blank")


def cat(x):
    return x.astype("string").fillna("blank")


def features(df: pd.DataFrame, line: str, si_edges=None) -> pd.DataFrame:
    age = df["driver_age"] if line == "motor" else df["policyholder_age"]
    f = pd.DataFrame({
        "region": cat(df["region"]), "urban": cat(df["urban_density"]), "pay": cat(df["payment_method"]),
        "missed": band(df["missed_payments_12m"], [-1, 0, 1, 99], ["0", "1", "2+"]), "credit": cat(df["credit_band"]),
        "channel": cat(df["channel"]), "new_bus": cat(df["new_business"]),
        "age": band(age, [0, 20, 24, 29, 39, 49, 59, 69, 79, 200]),
        "yr": cat(df["period_start"].dt.year),
    }, index=df.index)
    if line == "motor":
        f["ncb"] = cat(df["ncb_years"].clip(upper=10))
        f["vvg"] = cat(df["vehicle_value_group"])
        f["cover"], f["use"], f["fuel"], f["body"] = cat(df["cover_type"]), cat(df["vehicle_use"]), cat(df["fuel_type"]), cat(df["body_type"])
        f["mileage"] = band(df["annual_mileage"], [-1, 0, 5000, 7500, 10000, 15000, 20000, 1e7])
        f["points"] = band(df["licence_points"], [-1, 0, 3, 6, 99])
        f["young"], f["telem"] = cat(df["named_young_driver"]), cat(df["telematics_opt_in"])
    else:
        f["si"] = band(df["sum_insured"], si_edges)
        f["ptype"], f["flood"] = cat(df["property_type"]), cat(df["flood_zone"])
        f["ad_cover"] = cat(df["accidental_damage_cover"])
    if line == "buildings":
        f["constr"], f["soil"], f["occ"] = cat(df["construction_type"]), cat(df["soil_type"]), cat(df["occupancy"])
        f["built"] = band(df["year_built"], [0, 1919, 1944, 1969, 1989, 2009, 3000])
        f["beds"] = cat(df["bedrooms"].clip(upper=5))
    if line == "contents":
        f["tenure"], f["security"] = cat(df["tenure"]), cat(df["security_level"])
        f["hvi"] = cat(df["high_value_items"].clip(upper=4))
        f["bike"] = cat(df["bicycle_cover"])
    return f


class PlainGLM:
    """Tweedie GLM on one-hot main effects. Exposure x completion enters as an exact offset: the rate target with
    weight (exposure x completion)^(2-p) has the same deviance as row cost with a log offset (homogeneity of degree
    2-p)."""

    def __init__(self, alpha=1e-5):
        self.alpha = alpha

    def _x(self, f):
        rows, cols, j = [], [], 0
        for c in f.columns:
            vals = f[c].to_numpy()
            for lvl in self.levels_[c][1:]:
                r = np.flatnonzero(vals == lvl)
                rows.append(r); cols.append(np.full(len(r), j)); j += 1
        r, k = np.concatenate(rows), np.concatenate(cols)
        return sp.csr_matrix((np.ones(len(r)), (r, k)), shape=(len(f), j))

    def fit(self, f, y, ec):
        self.levels_ = {c: [f[c].value_counts().index[0]] + sorted(set(f[c].unique()) - {f[c].value_counts().index[0]}) for c in f.columns}
        self.model_ = TweedieRegressor(power=P, link="log", alpha=self.alpha, solver="lbfgs", max_iter=2000, tol=1e-6)
        self.model_.fit(self._x(f), y / ec, sample_weight=np.power(ec, 2 - P))
        return self

    def rate(self, f):
        f = f.copy()
        for c in f.columns:
            f[c] = f[c].where(f[c].isin(self.levels_[c]), self.levels_[c][0])
        return self.model_.predict(self._x(f))


def load_line(line, th, claims, curves, weights, shares, ad, cu):
    tr, te = dr.read_raw(f"policies_{line}"), dr.read_raw(f"test_{line}")
    cut = dr.sum_insured_cutoff(tr["sum_insured"], th["si_unit_multiple"]) if line != "motor" else None
    tr, te = dr.apply_policy_rules(tr, line, cut), dr.apply_policy_rules(te, line, cut)
    tr, te = dr.recover_from_history(tr, te, line)
    out = []
    for df in (tr, te):
        df = df.merge(ad, on="address_id", how="left").merge(cu[["customer_id", "credit_band", "licence_points"]], on="customer_id", how="left")
        df = df.join(tg.row_completion(df, line, curves, weights, shares)[["completion"]])
        out.append(df)
    tr, te = out
    tr = tr.merge(tg.row_targets(tr, claims, line)[["row_id", "cost"]], on="row_id", how="left")
    return tr, te


def main():
    th = load_thresholds()
    claims = dr.read_raw("claims")
    curves, weights, shares = tg.delay_curves(claims), tg.incidence_weights(claims), tg.peril_cost_shares(claims)
    ad, cu = dr.read_raw("addresses"), dr.read_raw("customers")
    data = {}
    all_cust = []
    for line in dr.LINES:
        data[line] = load_line(line, th, claims, curves, weights, shares, ad, cu)
        all_cust.append(data[line][0]["customer_id"])
    cust = np.array(sorted(pd.concat(all_cust).unique()))
    keep = pd.Series(v.fitting_mask(LOCK, cust), index=cust)

    results, preds = {}, {k: [] for k in v.SCORINGS}
    for line in dr.LINES:
        tr, te = data[line]
        tr = tr[tr["customer_id"].map(keep).to_numpy()].reset_index(drop=True)
        tr["yr"] = tr["period_start"].dt.year
        si_edges = None if line == "motor" else np.unique(np.r_[0, tr["sum_insured"].quantile(np.linspace(0.1, 0.9, 9)).to_numpy(), 1e12])
        f = features(tr, line, si_edges)
        ec = (tr["exposure"] * tr["completion"]).to_numpy()
        for name, s in v.SCORINGS.items():
            fit = tr["yr"].isin(s["fit_years"]).to_numpy()
            sc = (tr["yr"] == s["score_year"]).to_numpy()
            e, c, y = tr.loc[sc, "exposure"].to_numpy(), tr.loc[sc, "completion"].to_numpy(), tr.loc[sc, "cost"].to_numpy()
            scale = c if s["completion"] else np.ones_like(c)
            rate_const = tr.loc[fit, "cost"].sum() / ec[fit].sum()
            glm = PlainGLM().fit(f[fit], tr.loc[fit, "cost"].to_numpy(), ec[fit])
            fs = f[sc].copy(); fs["yr"] = str(max(s["fit_years"]))
            preds[name].append(pd.DataFrame({
                "line": line, "customer_id": tr.loc[sc, "customer_id"].to_numpy(), "y": y, "exposure": e,
                "constant": rate_const * e * scale,
                "incumbent": tr.loc[sc, "written_premium"].to_numpy() * e * scale,
                "glm": glm.rate(fs) * e * scale}))
            print(f"{line} {name}: fitted on {fit.sum():,} rows, scored on {sc.sum():,}", flush=True)

    for name in v.SCORINGS:
        d = pd.concat(preds[name], ignore_index=True)
        rep = {}
        for model in ("constant", "incumbent", "glm"):
            r = v.score_report(d["y"], d[model], d["exposure"], d["line"])
            common = v.rescale_to_total(d[model], d["y"], by=d["line"])
            for ln in list(dr.LINES) + ["all"]:
                m = (d["line"] == ln).to_numpy() if ln != "all" else np.ones(len(d), bool)
                lo, hi = v.d2_interval_groups(d["y"][m], common[m], d["exposure"][m], d["customer_id"][m], n_boot=1000)
                r.loc[ln, "d2_ranking_lo"], r.loc[ln, "d2_ranking_hi"] = lo, hi
            rep[model] = r
        results[name] = {m: r.reset_index().to_dict("records") for m, r in rep.items()}
        print(f"\n=== Scoring {name}: fit {v.SCORINGS[name]['fit_years']}, score {v.SCORINGS[name]['score_year']}")
        print(pd.concat(rep, names=["model"]).round(4).to_string())

    # Early submission: the plain GLM refitted on every non-hold-out row 2021-2024, priced at the 2024 year level
    sub = []
    for line in dr.LINES:
        tr, te = data[line]
        tr = tr[tr["customer_id"].map(keep).to_numpy()].reset_index(drop=True)
        si_edges = None if line == "motor" else np.unique(np.r_[0, tr["sum_insured"].quantile(np.linspace(0.1, 0.9, 9)).to_numpy(), 1e12])
        f = features(tr, line, si_edges)
        glm = PlainGLM().fit(f, tr["cost"].to_numpy(), (tr["exposure"] * tr["completion"]).to_numpy())
        ft = features(te, line, si_edges); ft["yr"] = "2024"
        pred = glm.rate(ft) * te["exposure"].to_numpy()
        sub.append(pd.DataFrame({"row_id": te["row_id"].astype(str), "prediction": pred, "line": line,
                                 "written": te["written_premium"].to_numpy() * te["exposure"].to_numpy()}))
    sub = pd.concat(sub, ignore_index=True)
    ss = dr.read_raw("sample_submission")
    sub = ss[["row_id"]].astype(str).merge(sub, on="row_id", how="left", validate="one_to_one")
    assert sub["prediction"].notna().all() and (sub["prediction"] > 0).all() and len(sub) == len(ss)
    (OUT / "submissions").mkdir(parents=True, exist_ok=True)
    path = OUT / "submissions" / "baseline_glm_v0.csv"
    sub[["row_id", "prediction"]].to_csv(path, index=False, float_format="%.4f")
    summ = sub.groupby("line").agg(rows=("row_id", "size"), predicted=("prediction", "sum"), written=("written", "sum"))
    summ["predicted_over_written"] = summ["predicted"] / summ["written"]
    print("\nSubmission written:", path, "\n", summ.round(3).to_string())
    results["submission"] = dict(file=str(path.relative_to(ROOT)), rows=len(sub), by_line=summ.reset_index().to_dict("records"))
    write_evidence({"phase4_baselines": results}, OUT / "phase4_baselines.json")


if __name__ == "__main__":
    main()
