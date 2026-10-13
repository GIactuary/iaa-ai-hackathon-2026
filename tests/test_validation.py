import numpy as np
import pandas as pd
import pytest

from src.families import LogGaussian, Poisson
from src.validation import (HoldoutError, check_folds, drop_one_verdict, fitting_mask, lock_holdout, make_folds,
                            oof_predictions, open_holdout, paired_bootstrap, paired_by_seed)


def test_folds_partition_rows_on_three_seeds():
    f = make_folds(103)
    assert sorted(f) == [1, 2, 42]
    for folds in f.values():
        check_folds(folds, 103)
        assert len(folds) == 5
    assert not np.array_equal(f[42][0][1], f[1][0][1])
    assert np.array_equal(make_folds(103)[42][0][1], f[42][0][1])          # reproducible


def test_grouped_and_stratified_folds():
    groups = np.repeat(np.arange(50), 3)
    strata = np.tile([0, 1, 1], 50)
    for kw in (dict(groups=groups), dict(strata=strata), dict(groups=groups, strata=strata)):
        f = make_folds(150, **kw)
        for folds in f.values():
            for a, b in folds:
                if "groups" in kw:
                    assert not set(groups[a]) & set(groups[b])


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
    lock = lock_holdout(ids, 0.1, tmp_path / "h.json", order=order)
    assert sorted(map(int, lock["holdout_ids"])) == list(range(20))


def test_oof_runner_and_paired_bootstrap(reg_data):
    d, y = reg_data
    fam = LogGaussian()
    z = fam.link(y)
    X = d[["x0", "x1", "x2", "x3"]].values
    folds = make_folds(len(y))

    def fit_predict(a, b, seed):
        A = np.column_stack([np.ones(len(a)), X[a]])
        beta = np.linalg.lstsq(A, z[a], rcond=None)[0]
        good = np.column_stack([np.ones(len(b)), X[b]]) @ beta
        return pd.DataFrame(dict(mean_only=np.full(len(b), z[a].mean()), linear=good))

    oof = oof_predictions(fit_predict, len(y), folds)
    for s, o in oof.items():
        assert len(o) == len(y)
        assert fam.score(y, o.linear) < fam.score(y, o.mean_only)
    rows = paired_by_seed(y, {s: o.mean_only.values for s, o in oof.items()}, {s: o.linear.values for s, o in oof.items()}, family=fam)
    for r in rows:
        assert r["gain"] > 0 and r["lo"] > 0 and r["share_b_better"] == 1.0
    # identical predictions: zero gain, interval straddles zero trivially
    same = paired_bootstrap(y, oof[42].linear, oof[42].linear, family=fam, n_boot=200)
    assert same["gain"] == 0 and same["lo"] == 0 and same["hi"] == 0


def test_bootstrap_with_generic_metric(count_data):
    d, y, e, _ = count_data
    fam = Poisson()
    good = np.log(e) - 2 + 0.4 * d.x1 - 0.3 * d.x2
    flat = np.log(e) + np.log(y.sum() / e.sum())
    a = paired_bootstrap(y, flat, good, family=fam, n_boot=300)
    b = paired_bootstrap(y, flat, good, metric=lambda yy, p, w: fam.score(yy, p, w), n_boot=300)
    assert a["gain"] == pytest.approx(b["gain"]) and a["lo"] > 0 and b["lo"] > 0


def test_drop_one_verdict():
    assert drop_one_verdict({42: .10, 1: .10, 2: .10}, {42: .11, 1: .12, 2: .101}, 3)["keep"]
    v = drop_one_verdict({42: .10, 1: .10, 2: .10}, {42: .11, 1: .12, 2: .099}, 3)
    assert not v["keep"] and v["seeds_worse"] == 2
