"""Phase 7: score the frozen pipeline once on the hold-out customers. Design written first: decisions/phase7_design.md.

For scoring A (fit 2021-2022, score 2023) and B (fit 2021-2023, score 2024): every family of the frozen blend is fitted
on the non-hold-out rows of the fitting years with their frozen claims history and k, exactly as in step 7, and prices
the hold-out customers' rows of the scored year. Their claims history comes from the all-customer build
(scripts/phase5_step4.py with P5_HOLDOUT=include), point in time. The blend uses the pooled step 8 weights and the
referral rule. Compared with the W2 GLM and the phase 4 plain GLM; the W2 GLM is also priced by the decided level chain.
Run: .venv/Scripts/python.exe -I scripts/phase7_holdout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.freeze_build import COLMAP, MIX, family_predictions, log
from scripts.phase4_baselines import PlainGLM, features as v0_features
from scripts.phase5_step6 import test
from scripts.phase6_referral import disagreement
from scripts.submit_w2 import w2_frame
from src import build, data_rules as dr, level as lv, validation as v, weather as wx
from src.evidence import write_evidence
from src.glm import TweedieGLM, fit_edges
from src.thresholds import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT, W4, W4A, W7 = ROOT / "outputs", ROOT / "data" / "work" / "phase5_step4", ROOT / "data" / "work" / "phase5_step4_all", ROOT / "data" / "work" / "phase5_step7"
PLANS = {"A": ((2021, 2022), 2023), "B": ((2021, 2022, 2023), 2024)}


def level_factor(tr, line, th, weather, fy, sy, Ft, m) -> float:
    """The decided chain for the W2 GLM: excess factor x off-balance to the weather-normalised indication for sy x
    the scored year's weather as one factor per line (applied by the caller)."""
    trx = wx.attach(tr, weather, wx.normals(weather, fy))
    b_el = wx.elasticities(trx, line)
    perils = lv.perils_of(tr)
    share = {p: float(tr[f"cost_{p}"].sum() / tr["cost"].sum()) for p in perils}
    yf = wx.year_factor(trx, line, b_el, share)
    trn = trx.copy()
    for p, (wv, bb) in b_el.items():
        trn[f"wm_{p}"] = np.exp(bb * trn[f"dev_{wv}"].fillna(0.0))
    trends = lv.peril_trends(trn, line, th["trend_min"], th["trend_max"])
    ind = lv.indication(trn, trends, sy)["indicated_capped_lc"]
    last = (tr["yr"] == max(fy)).to_numpy()
    e = tr["exposure"].to_numpy()
    model_lc = float(np.sum(m.rate(Ft[last]) * e[last] / yf[last]) / e[last].sum())
    elf = float(tr["cost"].sum() / tr["costc"].sum())
    return elf * float(ind.sum()) / model_lc, b_el, share


def run() -> None:
    th = load_thresholds()
    lim = 1 + th["refer_disagreement"]
    data = build.build_all()
    weather = dr.read_raw("weather")
    S8 = json.loads((OUT / "phase5_step8.json").read_text())["phase5_step8"]
    k_frozen = json.loads((W4 / "k.json").read_text())
    frames, info = {}, {}
    for s, (fy, sy) in PLANS.items():
        ks = k_frozen[s]
        for r in ks.values():
            r["k"] = r["k_poisson_gamma"]
        h_f = pd.read_parquet(W4 / f"history_{s}.parquet")
        h_a = pd.read_parquet(W4A / f"history_{s}.parquet")
        parts = []
        for line in dr.LINES:
            tuned = json.loads((W7 / f"result_{line}.json").read_text())["tuned"]
            wk = S8["weights"][line][f"families|{MIX[line]}"]
            weights = {c: w for c, w in zip(wk["cols"], wk["pooled"]) if w > 1e-9}
            weights = {c: w / sum(weights.values()) for c, w in weights.items()}
            full = data[line][0]
            tr = full[~full["holdout"] & full["yr"].isin(fy)].reset_index(drop=True)
            te0 = full[full["holdout"] & (full["yr"] == sy)].reset_index(drop=True)
            hf = h_f[h_f["line"] == line].drop(columns="line").drop_duplicates("row_id").set_index("row_id")
            ha = h_a[(h_a["line"] == line) & h_a["row_id"].isin(te0["row_id"])].drop(columns="line").drop_duplicates("row_id").set_index("row_id")
            assert len(ha) == len(te0), f"{line} {s}: hold-out rows missing from the all-customer history"
            h = pd.concat([hf, ha[~ha.index.isin(hf.index)]])
            preds = family_predictions(line, tr, te0, h, ks, tuned, th, sy, max(fy))
            use_c = v.SCORINGS[s]["completion"]
            cl = te0["completion"].to_numpy() if use_c else 1.0
            y = te0["cost"].to_numpy()
            fam = {c: preds[c] * cl for c in weights}
            ref = preds["ref"] * cl
            common = lambda x: x * (y.sum() / x.sum())
            sh = {c: common(p) for c, p in fam.items()}
            if MIX[line] == "geo":
                bl = np.exp(sum(w * np.log(sh[c]) for c, w in weights.items()))
            else:
                bl = sum(w * sh[c] for c, w in weights.items())
            bl = common(bl)
            if line == "motor":                                   # referral rule (Q1 (a))
                flag = disagreement(np.column_stack([sh[c] for c in weights]), bl) > lim
                bl = common(np.where(flag, common(ref), bl))
            # the plain baseline GLM of phase 4 (uncapped cost, main effects, year factor)
            si = None if line == "motor" else np.unique(np.r_[0, tr["sum_insured"].quantile(np.linspace(0.1, 0.9, 9)).to_numpy(), 1e12])
            g0 = PlainGLM().fit(v0_features(tr, line, si), tr["cost"].to_numpy(), (tr["exposure"] * tr["completion"]).to_numpy())
            f0 = v0_features(te0, line, si); f0["yr"] = str(max(fy))
            v0p = g0.rate(f0) * te0["exposure"].to_numpy() * cl
            # the W2 GLM priced by the decided level chain, for the level check
            Ft = w2_frame(tr, line, fit_edges(tr, line), hf, ks)
            m = TweedieGLM().fit(Ft, tr["costc"], tr["exposure"] * tr["completion"])
            lf, b_el, share = level_factor(tr, line, th, weather, fy, sy, Ft, m)
            tex = wx.attach(te0, weather, wx.normals(weather, fy))
            wsc = float(np.average(wx.year_factor(tex, line, b_el, share), weights=te0["exposure"]))
            priced = ref * lf * wsc
            info.setdefault(line, {})[s] = dict(rows=len(te0), claims=int((y > 0).sum()), level_ratio_priced=float(priced.sum() / y.sum()),
                                                level_factor=lf, weather=wsc)
            parts.append(pd.DataFrame({"line": line, "customer_id": te0["customer_id"].to_numpy(), "y": y, "exposure": te0["exposure"].to_numpy(),
                                       "ref": ref, "blend": bl, "v0": v0p, **{f"fam:{c}": fam[c] for c in weights}}))
            log(s, line, "scored", len(te0), "rows")
        frames[s] = pd.concat(parts, ignore_index=True)
        frames[s].to_parquet(ROOT / "data" / "work" / f"phase7_holdout_{s}.parquet")
    tests = {"blend vs W2 GLM": test(frames, "ref", "blend", th), "W2 GLM vs plain GLM": test(frames, "v0", "ref", th),
             "blend vs plain GLM": test(frames, "v0", "blend", th)}
    d2 = {}
    for s, f in frames.items():
        for ln in list(dr.LINES) + ["all"]:
            d = f if ln == "all" else f[f["line"] == ln]
            yy = d["y"].to_numpy()
            for col in ("v0", "ref", "blend"):
                mu = v.rescale_to_total(d[col].to_numpy(), yy, by=d["line"].to_numpy())
                d2.setdefault(s, {}).setdefault(ln, {})[col] = float(v.d2_tweedie(yy, mu, d["exposure"]))
    for k, t in tests.items():
        print(k)
        for ln, x in t.items():
            a = x["A"]; print(f"   {ln:9s} A {a['gain']:9.0f} [{a['boot']['lo']:9.0f}, {a['boot']['hi']:9.0f}] B {x['B']['gain']:9.0f} keep {x['verdict']['keep']}")
    print(json.dumps(d2, indent=1)); print(json.dumps(info, indent=1, default=float))
    write_evidence({"phase7_holdout": dict(tests=tests, ranking_d2=d2, by_line=info)}, OUT / "phase7_holdout.json")


if __name__ == "__main__":
    run()
