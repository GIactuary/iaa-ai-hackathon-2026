import numpy as np
import pandas as pd
import pytest

from src.validation import (HoldoutError, check_folds, drop_one_verdict, fitting_mask, lock_holdout, make_folds,
                            oof_predictions, open_holdout, paired_bootstrap, paired_by_seed)


SEEDS = (11, 22, 33)


def rmse(y, p, w):
    return float(np.sqrt(np.sum(w * (y - p) ** 2) / np.sum(w)))


def test_folds_partition_rows_on_three_seeds():
    f = make_folds(103, SEEDS)
    assert sorted(f) == [11, 22, 33]
    for folds in f.values():
        check_folds(folds, 103)
        assert len(folds) == 5
    assert not np.array_equal(f[11][0][1], f[22][0][1])
    assert np.array_equal(make_folds(103, SEEDS)[11][0][1], f[11][0][1])          # reproducible


def test_grouped_and_stratified_folds():
    groups = np.repeat(np.arange(50), 3)
    strata = np.tile([0, 1, 1], 50)
    for kw in (dict(groups=groups), dict(strata=strata), dict(groups=groups, strata=strata)):
        f = make_folds(150, SEEDS, **kw)
        for folds in f.values():
            for a, b in folds:
                if "groups" in kw:
                    assert not set(groups[a]) & set(groups[b])


def test_string_group_ids_give_the_folds_scikit_learn_gives_them():
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    rng = np.random.default_rng(3)
    groups = np.array([f"C{i:05d}" for i in rng.integers(0, 400, 1200)], dtype=object)
    strata = rng.integers(0, 2, 1200)
    for s in SEEDS:
        want = [b for _, b in GroupKFold(5, shuffle=True, random_state=s).split(np.arange(1200), groups=groups)]
        assert all(np.array_equal(b, w) for (_, b), w in zip(make_folds(1200, [s], groups=groups)[s], want))
        want = [b for _, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(np.arange(1200), strata, groups)]
        assert all(np.array_equal(b, w) for (_, b), w in zip(make_folds(1200, [s], strata=strata, groups=groups)[s], want))


def test_holdout_locked_once_and_opened_once(tmp_path):
    ids = np.arange(1000, 1500)
    p = tmp_path / "holdout.json"
    lock = lock_holdout(ids, 0.2, p, seed=7)
    assert len(lock["holdout_ids"]) == 100
    assert lock_holdout(ids, 0.2, p, seed=7)["holdout_ids"] == lock["holdout_ids"]     # re-read, not redrawn
    with pytest.raises(HoldoutError, match="never redrawn"):
        lock_holdout(ids, 0.2, p, seed=8)
    fit = fitting_mask(p, ids)
    assert fit.sum() == 400
    with pytest.raises(HoldoutError, match="not the records"):
        fitting_mask(p, ids[:-1])
    hold = open_holdout(p, ids, reason="phase 7 validation report")
    assert hold.sum() == 100 and not (hold & fit).any()
    with pytest.raises(HoldoutError, match="opened once"):
        open_holdout(p, ids, reason="again")
    open_holdout(p, ids, reason="independent review asked for a rerun", allow_reopen=True)


def test_holdout_by_date_takes_latest(tmp_path):
    ids = np.arange(200)
    order = np.arange(200)[::-1]                       # id 0 is the latest
    lock = lock_holdout(ids, 0.1, tmp_path / "h.json", seed=5, order=order)
    assert sorted(map(int, lock["holdout_ids"])) == list(range(20))


def test_oof_runner_and_paired_bootstrap():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = X @ np.array([1.0, -0.5, 0.2]) + rng.normal(0, 0.3, 600)
    folds = make_folds(len(y), SEEDS)

    def fit_predict(a, b, seed):
        A = np.column_stack([np.ones(len(a)), X[a]])
        beta = np.linalg.lstsq(A, y[a], rcond=None)[0]
        return pd.DataFrame(dict(mean_only=np.full(len(b), y[a].mean()), linear=np.column_stack([np.ones(len(b)), X[b]]) @ beta))

    oof = oof_predictions(fit_predict, len(y), folds)
    w = np.ones(len(y))
    for o in oof.values():
        assert len(o) == len(y) and rmse(y, o.linear, w) < rmse(y, o.mean_only, w)
    rows = paired_by_seed(y, {s: o.mean_only.values for s, o in oof.items()}, {s: o.linear.values for s, o in oof.items()},
                          metric=rmse, n_boot=300)
    for r in rows:
        assert r["gain"] > 0 and r["lo"] > 0 and r["share_b_better"] == 1.0
    same = paired_bootstrap(y, oof[11].linear, oof[11].linear, metric=rmse, n_boot=100)
    assert same["gain"] == 0 and same["lo"] == 0 and same["hi"] == 0
    # a metric where higher is better: gain still positive when b is better
    r2 = lambda yy, p, ww: 1 - rmse(yy, p, ww) ** 2 / np.var(yy)
    g = paired_bootstrap(y, oof[11].mean_only, oof[11].linear, metric=r2, greater_is_better=True, n_boot=100)
    assert g["gain"] > 0 and g["lo"] > 0


def test_drop_one_verdict():
    assert drop_one_verdict({42: .10, 1: .10, 2: .10}, {42: .11, 1: .12, 2: .101}, 3)["keep"]
    v = drop_one_verdict({42: .10, 1: .10, 2: .10}, {42: .11, 1: .12, 2: .099}, 3)
    assert not v["keep"] and v["seeds_worse"] == 2
