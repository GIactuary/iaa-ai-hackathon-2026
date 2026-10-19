"""Tests for src/history.py and the GLM's numeric terms, on synthetic data with known answers."""
import numpy as np
import pandas as pd
import pytest

from src import history as hs
from src.glm import TweedieGLM


def test_completed_exposure_full_reporting_equals_plain_exposure():
    H = np.cumsum(np.ones(50))                       # F = 1 at every age: everything reported at once
    s, e, c = np.array([0, 0, 100]), np.array([364, 99, 200]), np.array([365, 50, 150])
    assert np.allclose(hs.completed_exposure(s, e, c, H), hs.plain_exposure(s, e, c))
    assert hs.plain_exposure([0], [364], [365])[0] == pytest.approx(1.0)
    assert hs.plain_exposure([10], [20], [5])[0] == 0.0  # row starts after the cutoff


def test_completed_exposure_partial_reporting():
    # F(u) = 0 for u < 10, then 1: claims take 10 days to be reported
    F = np.r_[np.zeros(10), np.ones(40)]
    H = np.cumsum(F)
    # a row covering days 0..99 with cutoff 100: only days 0..89 are old enough
    assert hs.completed_exposure([0], [99], [100], H)[0] * 365 == pytest.approx(90)
    # beyond the curve every day counts in full
    assert hs.completed_exposure([0], [99], [1000], H)[0] * 365 == pytest.approx(100)


def test_scope_sums_respects_cutoff_and_key():
    H = {"motor": np.cumsum(np.ones(3000))}
    src_rows = pd.DataFrame({"policy_id": ["p1", "p1", "p2"], "line": "motor", "s": [0, 365, 0], "e": [364, 729, 364],
                             "rate": [0.2, 0.2, 0.5], "rate_avg": 0.1})
    src_claims = pd.DataFrame({"policy_id": ["p1", "p1", "p2"], "report": [100, 500, 10], "rep": [False, True, False]})
    targets = pd.DataFrame({"policy_id": ["p1", "p1", "p2", "p3"], "c": [365, 730, 365, 365]}, index=[10, 11, 12, 13])
    out = hs.scope_sums(targets, src_rows, src_claims, ["policy_id"], H, claim_flags=("rep",))
    assert list(out.index) == [10, 11, 12, 13]
    assert out.loc[10, "A"] == 1 and out.loc[11, "A"] == 2 and out.loc[11, "A_rep"] == 1
    assert out.loc[10, "E_rate"] == pytest.approx(0.2) and out.loc[11, "E_rate"] == pytest.approx(0.4)
    assert out.loc[12, "E_rate"] == pytest.approx(0.5) and out.loc[13].sum() == 0


def test_buhlmann_straub_recovers_known_k():
    rng = np.random.default_rng(1)
    n_units, years, a_true = 40000, 4, 0.25        # theta ~ gamma, mean 1, variance 0.25, so k = 1 / 0.25 = 4
    theta = rng.gamma(1 / a_true, a_true, n_units)
    E = rng.uniform(0.05, 0.3, (n_units, years))
    A = rng.poisson(theta[:, None] * E)
    unit = np.repeat(np.arange(n_units), years); period = np.tile(np.arange(years), n_units)
    r = hs.buhlmann_straub_k(unit, period, A.ravel(), E.ravel())
    assert r["k"] == pytest.approx(4.0, rel=0.15)
    assert r["k_poisson_gamma"] == pytest.approx(4.0, rel=0.15)
    assert r["s2"] == pytest.approx(1.0, rel=0.05)


def test_buhlmann_straub_no_between_variance_gives_infinite_k():
    rng = np.random.default_rng(2)
    E = rng.uniform(0.1, 0.3, (20000, 3))
    A = rng.poisson(E)
    r = hs.buhlmann_straub_k(np.repeat(np.arange(20000), 3), np.tile(np.arange(3), 20000), A.ravel(), E.ravel())
    assert r["k"] > 50 or not np.isfinite(r["k"])
    assert np.all(hs.mod([1, 0], [0.5, 0.5], float("inf")) == 1)


def test_mod_and_credibility():
    assert hs.mod([0], [0], 4)[0] == 1.0             # no history: the prior mean
    assert hs.mod([0], [1], 4)[0] == pytest.approx(0.8)
    assert hs.mod([2], [1], 4)[0] == pytest.approx(1.2)
    assert hs.credibility([1], 4)[0] == pytest.approx(0.2)


def test_glm_numeric_term_recovers_coefficient():
    rng = np.random.default_rng(3)
    n = 60000
    x = rng.normal(0, 0.5, n)
    g = rng.choice(["a", "b"], n, p=[0.6, 0.4])
    mu = 100 * np.exp(0.7 * x + np.where(g == "b", 0.3, 0.0))
    y = rng.poisson(mu / 50) * 50.0
    f = pd.DataFrame({"g": pd.Series(g, dtype="string"), "x": x})
    m = TweedieGLM(power=1.5, alpha=0.0).fit(f, y, np.ones(n))
    rel = m.relativities().set_index(["factor", "level"])["relativity"]
    assert rel[("x", "coefficient")] == pytest.approx(0.7, abs=0.05)
    assert rel[("g", "b")] == pytest.approx(np.exp(0.3), rel=0.05)
    assert m.rate(f.iloc[:5]).shape == (5,)
