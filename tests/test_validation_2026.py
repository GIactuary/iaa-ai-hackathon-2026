"""Tests for the 2026 additions to src/validation.py."""
import numpy as np
import pytest

from src import validation as v


def test_unit_deviance_matches_closed_form_at_1_5():
    y = np.array([0.0, 5000.0, 120.0]); mu = np.array([100.0, 100.0, 120.0])
    np.testing.assert_allclose(v.tweedie_unit_deviance(y, mu), 4 * (np.sqrt(y) - np.sqrt(mu)) ** 2 / np.sqrt(mu), atol=1e-9)
    np.testing.assert_allclose(v.tweedie_unit_deviance(np.array([0.0, 5000.0]), np.array([100.0, 100.0])), [40.0, 1474.31], atol=0.01)


def test_unit_deviance_refuses_zero_prediction():
    with pytest.raises(ValueError, match="strictly positive"):
        v.tweedie_unit_deviance(np.array([1.0]), np.array([0.0]))


def test_d2_zero_for_constant_rate_and_one_for_perfect():
    y = np.array([0.0, 10.0, 0.0, 30.0]); e = np.array([1.0, 1.0, 0.5, 0.5])
    assert abs(v.d2_tweedie(y, v.constant_rate(y, e), e)) < 1e-12
    assert v.d2_tweedie(y, np.where(y > 0, y, 1e-9), e) > 0.99


def test_rescale_to_total_by_group():
    out = v.rescale_to_total([1.0, 1.0, 2.0], [3.0, 1.0, 4.0], by=["m", "m", "b"])
    np.testing.assert_allclose(out, [2.0, 2.0, 4.0])


def test_score_report_has_lines_and_all():
    y = np.array([0.0, 100.0, 0.0, 50.0]); pred = np.array([20.0, 60.0, 10.0, 30.0]); e = np.ones(4)
    r = v.score_report(y, pred, e, ["motor", "motor", "contents", "contents"])
    assert list(r.index) == ["contents", "motor", "all"]
    assert r.loc["all", "level_ratio"] == pytest.approx(120 / 150)


def test_group_bootstrap_detects_real_gain_and_not_noise():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(2000), 3)
    dev_a = rng.gamma(2, 1, len(groups)) + 0.05
    dev_b = dev_a - 0.05
    r = v.paired_bootstrap_groups(dev_a, dev_b, groups, n_boot=300)
    assert r["gain"] > 0 and r["lo"] > 0
    noise = v.paired_bootstrap_groups(dev_a, dev_a + rng.normal(0, 1, len(groups)), groups, n_boot=300)
    assert noise["lo"] < 0 < noise["hi"]


def test_keep_verdict_needs_both_scorings_and_interval():
    boot_ok, boot_bad = dict(lo=1.0, hi=5.0), dict(lo=-1.0, hi=5.0)
    assert v.keep_verdict({"A": 3.0, "B": 1.0}, boot_ok, 2)["keep"]
    assert not v.keep_verdict({"A": 3.0, "B": -1.0}, boot_ok, 2)["keep"]
    assert not v.keep_verdict({"A": 3.0, "B": 1.0}, boot_bad, 2)["keep"]


def test_d2_interval_contains_point_estimate():
    rng = np.random.default_rng(3)
    e = np.ones(3000); groups = np.repeat(np.arange(1000), 3)
    true = rng.gamma(1, 50, 3000)
    y = np.where(rng.random(3000) < 0.1, rng.gamma(2, true * 5), 0.0)
    mu = true * 1.0
    lo, hi = v.d2_interval_groups(y, mu, e, groups, n_boot=300)
    assert lo < v.d2_tweedie(y, mu, e) < hi
