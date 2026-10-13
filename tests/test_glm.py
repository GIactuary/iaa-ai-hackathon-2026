import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, PoissonRegressor

from src.families import Binomial, Gamma, LogGaussian, Poisson, Tweedie
from src.glm import ForwardGLM, fit_glm
from src.thresholds import load_thresholds


def test_irls_matches_sklearn_poisson_with_offset(count_data):
    d, y, e, _ = count_data
    X = d[["x1", "x2", "x3"]].values
    beta, info = fit_glm(np.column_stack([np.ones(len(y)), X]), y, Poisson(), offset=np.log(e))
    assert info["converged"]
    # an offset in a Poisson GLM equals modelling the rate y/e with weight e
    sk = PoissonRegressor(alpha=0, max_iter=1000, tol=1e-10).fit(X, y / e, sample_weight=e)
    assert beta[1:] == pytest.approx(sk.coef_, abs=1e-5)
    assert beta[0] == pytest.approx(sk.intercept_, abs=1e-5)


def test_log_gaussian_matches_least_squares(reg_data):
    d, y = reg_data
    X = d[["x0", "x1"]].values
    beta, _ = fit_glm(np.column_stack([np.ones(len(y)), X]), y, LogGaussian())
    lr = LinearRegression().fit(X, np.log1p(y))
    assert beta[1:] == pytest.approx(lr.coef_) and beta[0] == pytest.approx(lr.intercept_)


def test_forward_poisson_recovers_truth_and_balances(count_data, thresholds_path):
    d, y, e, truth = count_data
    t = load_thresholds(thresholds_path)
    m = ForwardGLM.from_thresholds(Poisson(), t, candidates=["x1", "x2", "x3", "x4", "region"]).fit(d, y, offset=np.log(e))
    assert m.sel[:2] == ["x1", "x2"] or set(m.sel[:3]) >= {"x1", "x2"}
    assert "region" in m.sel
    c = m.coef_table().set_index("term")
    assert c.loc["x1", "coef"] == pytest.approx(truth["x1"], abs=0.05)
    assert c.loc["x2", "coef"] == pytest.approx(truth["x2"], abs=0.05)
    # Poisson with log link and an intercept: fitted total equals actual total (the balance property)
    assert m.predict(d, offset=np.log(e)).sum() == pytest.approx(y.sum(), rel=1e-6)
    # the rare level Z (3 records) is pooled or merged, never given its own relativity
    lv = m.levels_table().set_index(["var", "level"])
    assert lv.loc[("region", "Z"), "treatment"] != "own level"
    # each entry on the ladder gained at least the threshold
    assert all(p["rel_gain"] >= t["glm_entry_gain"] for p in m.path[1:])


def test_forward_log_gaussian_skips_noise(reg_data):
    d, y = reg_data
    m = ForwardGLM(LogGaussian(), min_gain=0.002, min_level_count=5, transforms={"size": "log"}).fit(d, y)
    assert "size" in m.sel and "region" in m.sel
    # terms are centred: the average record is priced at the base
    tr = m.terms(d)
    assert np.allclose(tr.mean(), 0, atol=1e-10)
    assert np.allclose(m.predict_link(d), m.base_link() + tr.sum(axis=1))
    assert m.stop_reason_


def test_terms_and_unseen_levels(reg_data):
    d, y = reg_data
    m = ForwardGLM(LogGaussian(), min_gain=0.002, min_level_count=5).fit(d, y)
    new = d.head(4).copy()
    new.loc[new.index[0], "region"] = "NEVER_SEEN"
    assert m.unseen_levels(new) == {"region": 1}
    p = m.predict_link(new)
    assert np.isfinite(p).all()


def test_blank_inputs_refused(reg_data):
    d, y = reg_data
    d2 = d.copy()
    d2.loc[d2.index[5], "x1"] = np.nan
    with pytest.raises(ValueError, match="blanks"):
        ForwardGLM(LogGaussian(), 0.002, 5).fit(d2, y)


@pytest.mark.parametrize("fam", [Gamma(), Tweedie(1.5), Binomial()])
def test_other_families_fit(fam, count_data):
    d, y, e, _ = count_data
    rng = np.random.default_rng(3)
    if fam.name == "gamma":
        target = rng.gamma(2.0, np.exp(7 + 0.3 * d.x1) / 2.0)
        off = None
    elif fam.name == "tweedie":
        target = np.where(y > 0, y * rng.gamma(2.0, 500, len(y)), 0.0)
        off = np.log(e)
    else:
        target = (y > 0).astype(float)
        off = None
    m = ForwardGLM(fam, min_gain=0.002, min_level_count=5, candidates=["x1", "x2", "x5", "region"]).fit(d, target, offset=off)
    assert m.fit_info_["converged"]
    assert "x1" in m.sel
    p = m.predict(d, offset=off)
    assert np.isfinite(p).all() and (p > 0).all()
    assert "relativity" in m.coef_table()
