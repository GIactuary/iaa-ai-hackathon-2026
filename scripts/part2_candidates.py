"""Part 2 milestone 2: candidate shapes tested under P2-V (decisions/part2_m2_design.md, written before this ran).

Every candidate gives, for each emerging fitting row, the expected net cost reported to date (uncapped). Scoring is
per line at common level, pooled over lines, on V1 (fit 2023, score 2024) and V2 (customer-grouped five-fold
cross-validation, seeds 11, 22, 33). Parts fitted on Part 1 rows are fitted once.
Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/part2_candidates.py [stage ...]
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.part2_model import BASE_YEAR, average_over_unknown, developed, unknown_blanks, w2_factors
from src import build, data_rules as dr, fields, market, validation as v, weather as wx
from src.evidence import write_evidence
from src.families import LGB, as_category
from src.glm import TweedieGLM, band, cat, quantile_edges

ROOT = Path(__file__).resolve().parents[1]
OUT = market.OUT / "part2_candidates.json"
WORK = market.WORK / "m2"
LARGE = market.LARGE_LOSS_PERILS
SEEDS = (11, 22, 33)
T0 = time.time()


def log(*a):
    print(f"{time.time() - T0:7.0f}s", *a, flush=True)


# ------------------------------------------------------------------------------------------------ data per line
def hazard(df: pd.DataFrame, weather: pd.DataFrame, years) -> pd.DataFrame:
    """Address hazard terms: wildfire normal (numeric), distance-to-river and elevation bands."""
    n = wx.normals(weather, years)
    wf = df["address_id"].map(n["norm_wildfire_index"]).astype(float)
    return pd.DataFrame({"wf": wf.fillna(wf.mean()).to_numpy(),
                         "river": band(df["distance_to_river_km"], [-1, 0.25, 0.5, 1, 2, 5, 1e9]).to_numpy(),
                         "elev": band(df["elevation_m"], [-1e9, 2, 5, 10, 20, 50, 1e9]).to_numpy()}, index=df.index)


class Line:
    def __init__(self, line: str, tr: pd.DataFrame, dv: pd.DataFrame, w_em: pd.DataFrame, w_dv: pd.DataFrame):
        self.line, self.tr, self.dv = line, tr, dv
        self.edges = {"sum_insured": quantile_edges(dv["sum_insured"], 10)} if line != "motor" else {}
        self.Fd = w2_factors(dv, line, self.edges)
        self.Ft = w2_factors(tr, line, self.edges)
        self.Ft["yr"] = BASE_YEAR
        self.unknown = unknown_blanks(self.Fd, self.Ft)
        self.e = tr["exposure"].to_numpy(float)
        self.perils = sorted(c[6:] for c in tr.columns if c.startswith("costc_") and c not in ("costc_large", "costc_other"))
        self.y = tr["cost"].to_numpy(float)
        self.g = (line + ":" + tr["customer_id"].astype(str)).to_numpy()
        self.Hd, self.Ht = hazard(dv, w_dv, (2021, 2022, 2023, 2024)), hazard(tr, w_em, market.FIT_YEARS)
        # excess factor per peril (P2-L (a)): motor its own (on the fitting rows of each split), property Part 1's
        self.dev_ratio = {p: float(dv[f"cost_{p}"].sum() / max(dv[f"costc_{p}"].sum(), 1e-9)) for p in self.perils}

    def ecp(self, p):
        return self.e * self.tr[f"completion_{p}"].to_numpy(float)

    def elf(self, p, fit):
        if self.line == "motor":
            return float(self.tr.loc[fit, f"cost_{p}"].sum() / max(self.tr.loc[fit, f"costc_{p}"].sum(), 1e-9))
        return self.dev_ratio[p]


def rate_of(m: TweedieGLM, L: Line, F: pd.DataFrame) -> np.ndarray:
    return average_over_unknown(m, F, L.Ft, L.e, [c for c in L.unknown if c in F.columns])


# ------------------------------------------------------------------------------------------------ Part 1 fits (once)
def part1_fits(L: Line) -> dict:
    dv, line = L.dv, L.line
    ecd = (dv["exposure"] * dv["completion"]).to_numpy(float)
    out = {"K0": rate_of(TweedieGLM().fit(L.Fd, dv["costc"], ecd), L, L.Ft)}
    per, fs, fs_h = {}, {}, {}
    for p in L.perils:
        ecp = (dv["exposure"] * dv[f"completion_{p}"]).to_numpy(float)
        per[p] = rate_of(TweedieGLM().fit(L.Fd, dv[f"costc_{p}"], ecp), L, L.Ft)
        if p in LARGE[line]:
            for key, Fd_, Ft_ in (("plain", L.Fd, L.Ft), ("hazard", pd.concat([L.Fd, L.Hd], axis=1), pd.concat([L.Ft, L.Ht], axis=1))):
                if key == "hazard":
                    keep = ["wf"] if p == "fire" else (["river", "elev"] if p == "flood" else [])
                    if not keep:
                        continue
                    Fd_ = pd.concat([L.Fd, L.Hd[keep]], axis=1)
                    Ft_ = pd.concat([L.Ft, L.Ht[keep]], axis=1)
                freq = TweedieGLM(power=1.0).fit(Fd_, dv[f"n_{p}"], ecp)
                has = ((dv[f"n_{p}"] > 0) & (dv[f"cost_{p}"] > 0)).to_numpy()      # gamma needs a positive cost
                n = dv.loc[has, f"n_{p}"].to_numpy(float)
                sev = TweedieGLM(power=2.0).fit(Fd_[has], dv.loc[has, f"cost_{p}"].to_numpy(float), n, weight=n)
                r = rate_of(freq, L, Ft_) * rate_of(sev, L, Ft_)
                (fs if key == "plain" else fs_h)[p] = r
    out["peril"], out["fs"], out["fs_hazard"] = per, fs, fs_h
    # Part 1 LightGBM on the fields both markets carry (as the diagnostic)
    drop = ("region", "make_model", "occupation", "yr", "holdout", "gender")
    fe_, fd_ = fields.tree_fields(L.tr, 11), fields.tree_fields(dv, 11)
    cols = [c for c in fe_["num"] + fe_["cat"] if c in fd_["num"] + fd_["cat"] and c not in drop]
    cats = [c for c in cols if c in fe_["cat"]]
    both = as_category(pd.concat([dv[cols], L.tr[cols]], ignore_index=True), cats)
    Xd, Xe = both.iloc[:len(dv)].reset_index(drop=True), both.iloc[len(dv):].reset_index(drop=True)
    gd = dv["customer_id"].astype(str).to_numpy()
    yd = dv["costc"].to_numpy(float)
    a, b = v.make_folds(len(dv), [11], n_folds=5, groups=gd)[11][0]
    m = LGB("tweedie", 1.5, 31, 400, 0.05, 3000, 4, seed=11).fit(Xd.iloc[a], yd[a], ecd[a], val=(Xd.iloc[b], yd[b], ecd[b], None, None))
    m = LGB("tweedie", 1.5, 31, 400, 0.05, m.best_iter_, 4, seed=11).fit(Xd, yd, ecd)
    out["tree"] = m.rate(Xe)
    return out


# ------------------------------------------------------------------------------------------------ candidates per split
def line_level(L: Line, rate: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """A one-number-per-line shape (K0, tree): rate x line excess factor x line completion."""
    comp = L.tr["completion"].to_numpy(float)
    elf = float(sum(L.tr.loc[fit, f"costc_{p}"].sum() * L.elf(p, fit) for p in L.perils) / L.tr.loc[fit, "costc"].sum())
    return rate * elf * L.e * comp


def by_peril(L: Line, rates: dict, fit: np.ndarray, uncapped: set) -> np.ndarray:
    """Sum over perils: each peril's rate calibrated to the fitting rows' completed capped cost of that peril, times
    its excess factor; for perils priced uncapped (frequency x severity) the same total, so only the shape differs."""
    mu = np.zeros(len(L.tr))
    for p in L.perils:
        ecp = L.ecp(p)
        target = float(L.tr.loc[fit, f"costc_{p}"].sum()) * L.elf(p, fit)
        r = rates[p]
        a = target / max(float(np.sum(r[fit] * ecp[fit])), 1e-12)
        mu += r * a * ecp
    return mu


def layer(L: Line, mu: np.ndarray, F: pd.DataFrame, fit: np.ndarray) -> np.ndarray:
    y = L.tr["costc"].to_numpy(float)
    m = TweedieGLM().fit(F[fit], y[fit], mu[fit])
    return mu * m.rate(F)


def candidates(L: Line, P1: dict, fit: np.ndarray, stage: dict) -> dict:
    c = {}
    c["K0"] = line_level(L, P1["K0"], fit)
    c["K1"] = by_peril(L, P1["peril"], fit, set())
    k2 = dict(P1["peril"]); k2.update(P1["fs"])
    c["K2"] = by_peril(L, k2, fit, set(P1["fs"]))
    k3 = dict(k2); k3.update(P1["fs_hazard"])
    c["K3"] = by_peril(L, k3, fit, set(P1["fs_hazard"]))
    c["tree"] = line_level(L, P1["tree"], fit)
    inc = c[stage["incumbent"]]
    def norm(mu):
        return mu * L.tr.loc[fit, "cost"].sum() / mu[fit].sum()
    blend = np.exp(0.5 * np.log(norm(inc)) + 0.5 * np.log(norm(c["tree"])))
    c["K4_blend"] = blend
    c["K4_tree_contents"] = c["tree"] if L.line == "contents" else inc
    c["K5_gender"] = layer(L, inc, pd.DataFrame({"gender": cat(L.tr["gender"])}), fit) if L.line == "motor" else inc
    return c


# ------------------------------------------------------------------------------------------------ scoring
def score(lines: dict, P1: dict, stage: dict) -> dict:
    """Row deviances per candidate for V1 and V2 (V2 averaged over the seeds), pooled over lines."""
    out = {"V1": {}, "V2": {}}
    for name in ("V1", "V2"):
        acc = {}
        for line, L in lines.items():
            yr = L.tr["yr"].to_numpy()
            if name == "V1":
                splits = [((yr == 2023), (yr == 2024))]
                reps = 1
            else:
                splits = []
                for s in SEEDS:
                    for a, b in v.make_folds(len(L.tr), [s], n_folds=5, groups=L.tr["customer_id"].astype(str).to_numpy())[s]:
                        fm = np.zeros(len(L.tr), bool); fm[a] = True
                        splits.append((fm, ~fm))
                reps = len(SEEDS)
            dev = {}
            for fit, sc in splits:
                cs = candidates(L, P1[line], fit, stage)
                for k, mu in cs.items():
                    m = mu[sc] * L.y[sc].sum() / mu[sc].sum()
                    d = dev.setdefault(k, np.zeros(len(L.tr)))
                    d[sc] += v.tweedie_unit_deviance(L.y[sc], m, 1.5) / reps
            mask = (yr == 2024) if name == "V1" else np.ones(len(L.tr), bool)
            for k, d in dev.items():
                acc.setdefault(k, []).append(d[mask])
            acc.setdefault("_g", []).append(L.g[mask])
            acc.setdefault("_y", []).append(L.y[mask])
            acc.setdefault("_line", []).append(np.full(int(mask.sum()), line))
        out[name] = {k: np.concatenate(x) for k, x in acc.items()}
    return out


def compare(S: dict, a: str, b: str) -> dict:
    """b against a: gain > 0 means b has lower deviance."""
    r = {}
    for name in ("V1", "V2"):
        da, db, g = S[name][a], S[name][b], S[name]["_g"]
        bt = v.paired_bootstrap_groups(da, db, g)
        per_line = {l: float((da - db)[S[name]["_line"] == l].sum()) for l in dr.LINES}
        r[name] = dict(gain=bt["gain"], lo=bt["lo"], hi=bt["hi"], per_line=per_line)
    r["passes"] = bool(r["V1"]["gain"] > 0 and r["V2"]["gain"] > 0 and r["V2"]["lo"] > 0)
    return r


def d2(S: dict) -> dict:
    out = {}
    for name in ("V1", "V2"):
        y = S[name]["_y"]
        null = v.tweedie_unit_deviance(y, np.full(len(y), y.mean()), 1.5).sum()
        out[name] = {k: float(1 - d.sum() / null) for k, d in S[name].items() if not k.startswith("_")}
    return out


def main():
    data = build.build_all()
    w_em, w_dv = dr.read_raw("weather"), dr.read_raw("weather", ROOT / "data" / "raw_part2" / "developed")
    WORK.mkdir(parents=True, exist_ok=True)
    lines, P1 = {}, {}
    for line in dr.LINES:
        tr = data[line][0]
        lines[line] = Line(line, tr[~tr["holdout"]].reset_index(drop=True), developed(line), w_em, w_dv)
        cache = WORK / f"p1_{line}.pkl"
        if cache.exists():
            P1[line] = pickle.loads(cache.read_bytes())
        else:
            P1[line] = part1_fits(lines[line])
            cache.write_bytes(pickle.dumps(P1[line]))
        log("Part 1 fits", line)
    res, inc = {}, "K0"
    for cand in ("K1", "K2", "K3"):
        S = score(lines, P1, {"incumbent": inc})
        cmp = compare(S, inc, cand)
        res[cand] = dict(against=inc, **cmp, against_K0=compare(S, "K0", cand) if inc != "K0" else None, d2=d2(S))
        log(cand, "against", inc, {k: (round(x["gain"]), round(x["lo"]), round(x["hi"])) for k, x in cmp.items() if k != "passes"}, "passes" if cmp["passes"] else "fails")
        if cmp["passes"]:
            inc = cand
    S = score(lines, P1, {"incumbent": inc})
    for cand in ("tree", "K4_blend", "K4_tree_contents", "K5_gender"):
        cmp = compare(S, inc, cand)
        res[cand] = dict(against=inc, **cmp, against_K0=compare(S, "K0", cand), d2=d2(S))
        log(cand, "against", inc, {k: (round(x["gain"]), round(x["lo"]), round(x["hi"])) for k, x in cmp.items() if k != "passes"}, "passes" if cmp["passes"] else "fails")
    res["d2_final_stage"] = d2(S)
    res["incumbent_after_K1_K3"] = inc
    log("D2 (common level per line, pooled)", {k: {n: round(x, 4) for n, x in d.items()} for k, d in res["d2_final_stage"].items()})
    write_evidence({"part2_candidates": res}, OUT)


if __name__ == "__main__":
    main()
