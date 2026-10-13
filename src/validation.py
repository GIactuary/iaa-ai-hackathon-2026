"""Validation design (playbook principle 6): K-fold on three split seeds, a locked hold-out, a paired bootstrap.

- `make_folds` gives the same folds to every model, so every comparison is paired on the same rows.
- `lock_holdout` draws the hold-out once and writes it to disk. Later calls re-read the same draw and refuse a
  different one. `fitting_mask` returns the rows models may use; `open_holdout` returns the hold-out rows and records
  each opening in the lock file, refusing a second opening unless it is explicitly allowed and given a reason.
- `oof_predictions` runs any fit-and-predict function over the folds and checks that every row is predicted exactly
  once per seed.
- `paired_bootstrap` compares two sets of out-of-fold predictions on the same rows, for any metric.

The split seeds and the hold-out seed are not defaulted: they are fixed in the validation design (phase 3) and
passed in, so they are written down before anything is fitted.

If records are dated, random folds leak the future: hold out the latest periods instead (`lock_holdout(order=...)`)
and discuss the fold design with the actuary before fitting.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


N_FOLDS = 5
BOOTSTRAP_N, BOOTSTRAP_SEED = 2000, 0

Folds = list[tuple[np.ndarray, np.ndarray]]


# ------------------------------------------------------------------------------------------------ folds
def make_folds(n: int, seeds, n_folds: int = N_FOLDS, strata=None, groups=None) -> dict[int, Folds]:
    """Shuffled K-fold splits of rows 0..n-1, one set per seed (three seeds, per principle 6).

    strata: labels to balance across folds (for example a claim indicator or a binned target).
    groups: records that must stay together (for example several rows of one policy).
    """
    idx = np.arange(n)
    out = {}
    for s in seeds:
        if groups is not None and strata is not None:
            sp = StratifiedGroupKFold(n_folds, shuffle=True, random_state=s).split(idx, strata, groups)
        elif groups is not None:
            sp = GroupKFold(n_folds, shuffle=True, random_state=s).split(idx, groups=groups)
        elif strata is not None:
            sp = StratifiedKFold(n_folds, shuffle=True, random_state=s).split(idx, strata)
        else:
            sp = KFold(n_folds, shuffle=True, random_state=s).split(idx)
        out[s] = [(np.asarray(a), np.asarray(b)) for a, b in sp]
        check_folds(out[s], n)
    return out


def check_folds(folds: Folds, n: int) -> None:
    """Every row is in exactly one validation fold and never in its own training fold."""
    seen = np.zeros(n, dtype=int)
    for a, b in folds:
        if np.intersect1d(a, b).size:
            raise AssertionError("a row is in both the training and validation part of a fold")
        if len(a) + len(b) != n:
            raise AssertionError("a fold does not cover every row")
        seen[b] += 1
    if not np.all(seen == 1):
        raise AssertionError("validation folds do not partition the rows")


# ------------------------------------------------------------------------------------------------ hold-out
class HoldoutError(RuntimeError):
    pass


def _ids(ids) -> list[str]:
    ids = [str(i) for i in np.asarray(ids)]
    if len(set(ids)) != len(ids):
        raise HoldoutError("record ids must be unique")
    return ids


def _fingerprint(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def lock_holdout(ids, share: float, path: str | Path, seed: int, strata=None, order=None) -> dict:
    """Draw the hold-out once and write it to `path`. Returns the lock record (without reading it for scoring).

    order: if given (for example a date), the hold-out is the latest `share` of records by that order instead of a
    random draw. strata: labels to keep in proportion between the hold-out and the fitting rows.
    """
    path = Path(path)
    ids_s = _ids(ids)
    spec = dict(share=float(share), seed=int(seed), n_total=len(ids_s), ids_sha256=_fingerprint(ids_s),
                method="latest" if order is not None else ("stratified" if strata is not None else "random"))
    if path.exists():
        lock = json.loads(path.read_text(encoding="utf-8"))
        diff = {k: (lock.get(k), v) for k, v in spec.items() if lock.get(k) != v}
        if diff:
            raise HoldoutError(f"a different hold-out is already locked at {path}: {diff}. It is never redrawn.")
        return lock
    if not 0 < share < 1:
        raise HoldoutError("share must be between 0 and 1")
    n = len(ids_s)
    k = int(round(share * n))
    if order is not None:
        pos = np.argsort(np.asarray(order), kind="stable")[n - k:]
    elif strata is not None:
        from sklearn.model_selection import train_test_split
        _, pos = train_test_split(np.arange(n), test_size=k, random_state=seed, stratify=np.asarray(strata))
    else:
        pos = np.random.default_rng(seed).choice(n, size=k, replace=False)
    lock = dict(spec, created=dt.datetime.now().isoformat(timespec="seconds"),
                holdout_ids=sorted(ids_s[i] for i in pos), opened=[])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    return lock


def _load_lock(path, ids) -> tuple[dict, list[str]]:
    path = Path(path)
    if not path.exists():
        raise HoldoutError(f"no hold-out locked at {path}; call lock_holdout first")
    lock = json.loads(path.read_text(encoding="utf-8"))
    ids_s = _ids(ids)
    if _fingerprint(ids_s) != lock["ids_sha256"]:
        raise HoldoutError("these records are not the records the hold-out was drawn from")
    return lock, ids_s


def fitting_mask(path, ids) -> np.ndarray:
    """True for rows models may be fitted, tuned and selected on (everything outside the hold-out)."""
    lock, ids_s = _load_lock(path, ids)
    hold = set(lock["holdout_ids"])
    return np.array([i not in hold for i in ids_s])


def open_holdout(path, ids, reason: str, allow_reopen: bool = False) -> np.ndarray:
    """True for hold-out rows. Records the opening; a second opening needs allow_reopen=True and is recorded too."""
    if not reason.strip():
        raise HoldoutError("give a reason for opening the hold-out")
    lock, ids_s = _load_lock(path, ids)
    if lock["opened"] and not allow_reopen:
        raise HoldoutError(f"the hold-out was already opened on {lock['opened'][0]['when']} "
                           f"({lock['opened'][0]['reason']}); it is opened once")
    lock["opened"].append(dict(when=dt.datetime.now().isoformat(timespec="seconds"), reason=reason, reopen=bool(lock["opened"])))
    Path(path).write_text(json.dumps(lock, indent=1), encoding="utf-8")
    hold = set(lock["holdout_ids"])
    return np.array([i in hold for i in ids_s])


# ------------------------------------------------------------------------------------------------ out of fold
def oof_predictions(fit_predict: Callable[[np.ndarray, np.ndarray, int], pd.DataFrame], n: int,
                    folds_by_seed: dict[int, Folds], n_jobs: int = 1) -> dict[int, pd.DataFrame]:
    """Out-of-fold predictions for every seed.

    fit_predict(train_rows, valid_rows, seed) must fit on train_rows only and return a DataFrame with one row per
    valid_rows entry, in the same order (one column per model).
    """
    out = {}
    for s, folds in folds_by_seed.items():
        check_folds(folds, n)
        parts = Parallel(n_jobs=n_jobs)(delayed(fit_predict)(a, b, s) for a, b in folds)
        frames = []
        for (a, b), p in zip(folds, parts):
            if len(p) != len(b):
                raise AssertionError("fit_predict returned the wrong number of rows")
            frames.append(p.reset_index(drop=True).assign(row=b, fold=len(frames)))
        o = pd.concat(frames).sort_values("row").set_index("row")
        if not (o.index.values == np.arange(n)).all():
            raise AssertionError("out-of-fold predictions do not cover every row exactly once")
        out[s] = o
    return out


# ------------------------------------------------------------------------------------------------ paired bootstrap
def paired_bootstrap(y, pred_a, pred_b, metric: Callable, weight=None, n_boot: int = BOOTSTRAP_N,
                     seed: int = BOOTSTRAP_SEED, greater_is_better: bool = False) -> dict:
    """Paired bootstrap of metric(a) - metric(b) on the same rows: resample the records, rescore both, repeat.

    metric(y, pred, weight) -> float. gain > 0 means b is better, whichever direction the metric runs.
    """
    y = np.asarray(y, dtype=float)
    pa, pb = np.asarray(pred_a, dtype=float), np.asarray(pred_b, dtype=float)
    n = len(y)
    if not (len(pa) == len(pb) == n):
        raise ValueError("y and both predictions must be the same rows")
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
    sign = -1.0 if greater_is_better else 1.0
    sa, sb = metric(y, pa, w), metric(y, pb, w)
    rng = np.random.default_rng(seed)
    gains = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        gains[k] = sign * (metric(y[i], pa[i], w[i]) - metric(y[i], pb[i], w[i]))
    gain = sign * (sa - sb)
    return dict(score_a=float(sa), score_b=float(sb), gain=float(gain), rel_gain=float(gain / sa) if sa else float("nan"),
                lo=float(np.quantile(gains, 0.025)), hi=float(np.quantile(gains, 0.975)),
                share_b_better=float(np.mean(gains > 0)), n=int(n), n_boot=int(n_boot), boot_seed=int(seed))


def paired_by_seed(y, oof_a: dict[int, np.ndarray], oof_b: dict[int, np.ndarray], **kw) -> list[dict]:
    """One paired bootstrap per split seed. Both models must have been run on the same folds."""
    if set(oof_a) != set(oof_b):
        raise ValueError("both models need out-of-fold predictions for the same seeds")
    return [dict(split_seed=s, **paired_bootstrap(y, oof_a[s], oof_b[s], **kw)) for s in oof_a]


def drop_one_verdict(score_with: dict[int, float], score_without: dict[int, float], min_seeds_worse: int,
                     greater_is_better: bool = False) -> dict:
    """Drop-one test (principle 7): keep a feature only if removing it worsens the score on enough seeds."""
    diffs = {s: (score_with[s] - score_without[s]) if greater_is_better else (score_without[s] - score_with[s]) for s in score_with}
    worse = sum(d > 0 for d in diffs.values())
    return dict(keep=worse >= min_seeds_worse, seeds_worse=int(worse), worsening_by_seed={int(s): float(d) for s, d in diffs.items()})
