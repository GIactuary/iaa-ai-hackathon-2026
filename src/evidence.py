"""Exhibit calculations for the filing structure (docs/FILING_TEMPLATE.md), written to one evidence JSON file.

Every function takes actuals and out-of-fold predictions; none fits a model. Predictions passed here are on the
response scale (price, count, amount, probability) unless the argument is named eta (link scale).

| Filing section          | Function                                  |
|-------------------------|-------------------------------------------|
| 4 Development           | seed_scores                               |
| 5 Validation            | seed_scores, score_table (apparent / cross-validated / hold-out), lift_table, double_lift, disagreement_check |
| 6 Rating plan           | adjustment_summary                        |
| 8 Impact                | dislocation                               |
| 9 Segment testing       | ae_by_segment                             |
| 5 or 10 Ranking power   | gini                                      |

The notebook builder reads the file written by `write_evidence` and fills the notebook's numbers from it.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .families import Family


def _arr(x):
    return np.asarray(x, dtype=float)


def _bins(key, weight, n_bins):
    """Bins of equal total weight (equal exposure for counts, equal record count otherwise), lowest key first."""
    order = np.argsort(key, kind="stable")
    w = weight[order]
    cw = np.cumsum(w) - w / 2
    b = np.empty(len(key), dtype=int)
    b[order] = np.minimum((cw / w.sum() * n_bins).astype(int), n_bins - 1)
    return b


def _wilson(k, n, z=1.96):
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ------------------------------------------------------------------------------------------------ scores
def seed_scores(family: Family, y, oof_by_seed: dict, cols, weight=None) -> dict:
    """Out-of-fold score of each model on each split seed, with the mean and the spread (max - min) across seeds."""
    rows = []
    for c in cols:
        per = {int(s): family.score(y, o[c], weight) for s, o in oof_by_seed.items()}
        v = list(per.values())
        rows.append(dict(model=c, by_seed=per, mean=float(np.mean(v)), spread=float(max(v) - min(v))))
    return dict(metric=family.score_name, rows=rows)


def score_table(family: Family, models: dict) -> list[dict]:
    """Apparent against cross-validated against hold-out (principle 11).

    models: {name: {"apparent": (y, eta[, w]), "cv": (...), "holdout": (...)}}; any part may be missing or None
    (the hold-out stays None until phase 7). Apparent is scored on the rows the model was fitted on and is labelled
    as such everywhere it appears.
    """
    out = []
    for name, parts in models.items():
        row = dict(model=name)
        for k in ("apparent", "cv", "holdout"):
            v = parts.get(k)
            row[k] = None if v is None else family.score(v[0], v[1], v[2] if len(v) > 2 else None)
        if row["apparent"] is not None and row["cv"]:
            row["apparent_over_cv"] = row["apparent"] / row["cv"]
        out.append(row)
    return out


def gini(y, pred, exposure=None) -> float:
    """Normalised Gini: how well the predictions rank the actuals (1 = perfect order, 0 = random)."""
    y, p = _arr(y), _arr(pred)
    w = np.ones(len(y)) if exposure is None else _arr(exposure)

    def g(key):
        o = np.lexsort((np.arange(len(key)), -key))
        cw, cy = np.cumsum(w[o]) / w.sum(), np.cumsum(y[o]) / y.sum()
        return float(np.sum(np.diff(np.concatenate([[0], cw])) * (cy - cw)))

    key = p / w if exposure is not None else p
    perfect = y / w if exposure is not None else y
    return g(key) / g(perfect)


# ------------------------------------------------------------------------------------------------ lift
def lift_table(y, pred, n_bins: int = 10, exposure=None, ci: bool = False) -> list[dict]:
    """Actual against predicted by predicted-value bin (deciles by default). With exposure, bins hold equal exposure
    and the means are per unit of exposure. ci=True adds a 95% Wilson interval on the actual rate (yes/no targets:
    the calibration-by-decile exhibit)."""
    y, p = _arr(y), _arr(pred)
    w = np.ones(len(y)) if exposure is None else _arr(exposure)
    key = p / w
    b = _bins(key, w, n_bins)
    rows = []
    for k in range(n_bins):
        s = b == k
        a, pr, ex = y[s].sum(), p[s].sum(), w[s].sum()
        r = dict(bin=k + 1, n=int(s.sum()), exposure=float(ex), actual=float(a), predicted=float(pr),
                 actual_mean=float(a / ex), predicted_mean=float(pr / ex), ae=float(a / pr) if pr else None,
                 key_lo=float(key[s].min()), key_hi=float(key[s].max()))
        if ci:
            r["ci_lo"], r["ci_hi"] = _wilson(a, s.sum())
        rows.append(r)
    return rows


def double_lift(y, pred_current, pred_proposed, n_bins: int = 10, exposure=None) -> list[dict]:
    """Sort by proposed / current; in each bin, which model's total is closer to the actual total?"""
    y, pc, pp = _arr(y), _arr(pred_current), _arr(pred_proposed)
    w = np.ones(len(y)) if exposure is None else _arr(exposure)
    ratio = pp / pc
    b = _bins(ratio, w, n_bins)
    rows = []
    for k in range(n_bins):
        s = b == k
        a, c, p = y[s].sum(), pc[s].sum(), pp[s].sum()
        rows.append(dict(bin=k + 1, n=int(s.sum()), exposure=float(w[s].sum()), actual=float(a), current=float(c),
                         proposed=float(p), ae_current=float(a / c), ae_proposed=float(a / p),
                         ratio_lo=float(ratio[s].min()), ratio_hi=float(ratio[s].max())))
    return rows


def disagreement_check(family: Family, y, oof_by_seed: dict, top_share: float) -> dict:
    """Do the records where the GLM and tree disagree most carry more error? Pools all seeds. top_share must be
    written down before running (decisions/thresholds.md)."""
    g = pd.concat([pd.DataFrame(dict(y=_arr(y), gap=np.abs(o.glm.values - o.gbm.values), blend=o.blend.values))
                   for o in oof_by_seed.values()], ignore_index=True)
    cut = g.gap.quantile(1 - top_share)
    top = g.gap >= cut
    return dict(top_share=top_share, gap_cut=float(np.expm1(cut)), n_top=int(top.sum()),
                score_top=family.score(g.y[top], g.blend[top]), score_rest=family.score(g.y[~top], g.blend[~top]))


# ------------------------------------------------------------------------------------------------ segments, impact
def ae_by_segment(y, pred, segment, exposure=None) -> list[dict]:
    """Actual over expected by segment, largest segments first. Read every A/E beside its count."""
    df = pd.DataFrame(dict(seg=np.asarray(segment).astype(str), a=_arr(y), p=_arr(pred),
                           w=np.ones(len(y)) if exposure is None else _arr(exposure)))
    g = df.groupby("seg").agg(n=("a", "size"), exposure=("w", "sum"), actual=("a", "sum"), predicted=("p", "sum"))
    g["ae"] = g.actual / g.predicted
    g = g.sort_values("n", ascending=False).reset_index().rename(columns={"seg": "segment"})
    return g.to_dict("records")


DISLOCATION_EDGES = [-np.inf, -0.15, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.15, np.inf]


def dislocation(new, current, edges=DISLOCATION_EDGES) -> dict:
    """How far each record's prediction moves from the current model to the proposed one."""
    new, cur = _arr(new), _arr(current)
    chg = new / cur - 1
    labels = [f"{a:+.0%} to {b:+.0%}" if np.isfinite(a) and np.isfinite(b) else (f"below {b:+.0%}" if np.isfinite(b) else f"above {a:+.0%}")
              for a, b in zip(edges[:-1], edges[1:])]
    counts = pd.cut(chg, edges, labels=labels).value_counts().reindex(labels).fillna(0).astype(int)
    return dict(bands=[dict(band=k, n=int(v)) for k, v in counts.items()],
                n=int(len(chg)), within5=float(np.mean(np.abs(chg) <= 0.05)), within10=float(np.mean(np.abs(chg) <= 0.10)),
                over15=int(np.sum(np.abs(chg) > 0.15)), median_abs=float(np.median(np.abs(chg))),
                total_change=float(new.sum() / cur.sum() - 1))


def adjustment_summary(factor) -> dict:
    """Distribution of the tree adjustment factor (blend / GLM) across records."""
    f = _arr(factor)
    return dict(percentiles={str(q): float(np.percentile(f, q)) for q in (5, 10, 25, 50, 75, 90, 95)},
                within5=float(np.mean(np.abs(f - 1) <= 0.05)), within10=float(np.mean(np.abs(f - 1) <= 0.10)),
                max_up=float(f.max()), max_down=float(f.min()))


# ------------------------------------------------------------------------------------------------ output
def to_jsonable(x):
    """Plain JSON types; NaN and infinity become null."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return to_jsonable(x.to_dict("records"))
    if isinstance(x, pd.Series):
        return to_jsonable(x.to_dict())
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (dt.date, dt.datetime)):
        return x.isoformat()
    return x


def run_metadata(seeds=None) -> dict:
    import lightgbm, scipy, sklearn
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip() or None
    except OSError:
        commit = None
    return dict(created=dt.datetime.now().isoformat(timespec="seconds"), git_commit=commit, seeds=seeds,
                versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                              scipy=scipy.__version__, scikit_learn=sklearn.__version__, lightgbm=lightgbm.__version__))


def write_evidence(evidence: dict, path: str | Path, seeds=None) -> Path:
    """Write the evidence dictionary (plus run metadata under "meta") as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = to_jsonable({**evidence, "meta": run_metadata(seeds)})
    path.write_text(json.dumps(body, indent=1, allow_nan=False), encoding="utf-8", newline="\n")
    return path
