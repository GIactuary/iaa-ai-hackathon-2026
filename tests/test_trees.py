import numpy as np
import pytest

from src.families import Binomial, Gamma, LogGaussian, Poisson, Tweedie
from src.trees import TreeModel

FAST = dict(num_boost_round=300, learning_rate=0.05)


def test_regression_beats_base_and_shapley_adds_up(reg_data):
    d, y = reg_data
    fam = LogGaussian()
    m = TreeModel(fam, FAST, n_jobs=2).fit(d.iloc[:600], y[:600])
    eta = m.predict_link(d.iloc[600:])
    base = np.full(200, fam.link(y[:600]).mean())
    assert fam.score(y[600:], eta) < 0.8 * fam.score(y[600:], base)
    c, b = m.contrib(d.iloc[600:])
    assert np.allclose(c.sum(axis=1).values + b, eta, atol=1e-8)
    assert list(c.columns) == list(d.columns)
    assert m.importance().share.sum() == pytest.approx(1)


def test_poisson_offset_starts_at_base_rate(count_data):
    d, y, e, _ = count_data
    off = np.log(e)
    m = TreeModel(Poisson(), dict(num_boost_round=1), n_jobs=2).fit(d, y, offset=off)
    # after a single tree the fitted total is already near the actual total: the start is the base frequency
    assert m.predict(d, offset=off).sum() == pytest.approx(y.sum(), rel=0.02)
    m = TreeModel(Poisson(), FAST, n_jobs=2).fit(d.iloc[:15000], y[:15000], offset=off[:15000])
    eta = m.predict_link(d.iloc[15000:], offset=off[15000:])
    c, b = m.contrib(d.iloc[15000:])
    assert np.allclose(c.sum(axis=1).values + b + off[15000:], eta, atol=1e-8)
    flat = off[15000:] + np.log(y[:15000].sum() / e[:15000].sum())
    assert Poisson().score(y[15000:], eta) < Poisson().score(y[15000:], flat)


def test_unseen_level_is_blank_not_error(reg_data):
    d, y = reg_data
    m = TreeModel(LogGaussian(), FAST).fit(d, y)
    new = d.head(3).copy()
    new.loc[new.index[0], "region"] = "NEVER_SEEN"
    assert np.isfinite(m.predict_link(new)).all()


@pytest.mark.parametrize("fam", [Gamma(), Tweedie(1.5), Binomial()])
def test_objectives_run(fam, count_data):
    d, y, e, _ = count_data
    rng = np.random.default_rng(3)
    target = {"gamma": lambda: rng.gamma(2.0, np.exp(7 + 0.3 * d.x1) / 2.0),
              "tweedie": lambda: np.where(y > 0, y * rng.gamma(2.0, 500, len(y)), 0.0),
              "binomial": lambda: (y > 0).astype(float)}[fam.name]()
    m = TreeModel(fam, dict(num_boost_round=50)).fit(d, target)
    assert m.booster_.params["objective"] == fam.lgbm_objective
    p = m.predict(d)
    assert np.isfinite(p).all() and (p > 0).all()


def test_reproducible(reg_data):
    d, y = reg_data
    a = TreeModel(LogGaussian(), FAST, seed=1).fit(d, y).predict_link(d)
    b = TreeModel(LogGaussian(), FAST, seed=1).fit(d, y).predict_link(d)
    assert np.array_equal(a, b)
