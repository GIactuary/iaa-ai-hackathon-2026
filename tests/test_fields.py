"""Tests for src/fields.py (Y2): the field split and the credibility encoding."""
import numpy as np
import pandas as pd

from src import fields


def test_tree_fields_split_and_reasons():
    df = pd.DataFrame({"row_id": ["a", "b"], "exposure": [1.0, 0.5], "costc": [0.0, 10.0], "completion_x": [1.0, 1.0],
                       "dq_flag": [False, True], "gender": ["M", "F"], "written_premium": [100.0, 200.0],
                       "campaign_code": ["c1", "c2"], "renewal_decision": ["renewed", "lapsed_by_customer"],
                       "sum_insured": [1e5, 2e5], "new_business": [True, False], "region": ["E01", "E02"],
                       "make_model": ["m1", "m2"]})
    f = fields.tree_fields(df, max_levels=1)
    assert f["num"] == ["sum_insured"]
    assert f["cat"] == []
    assert f["encode"] == ["new_business", "region", "make_model"]
    assert set(f["excluded"]) == {"row_id", "exposure", "costc", "completion_x", "dq_flag", "gender", "written_premium",
                                  "campaign_code", "renewal_decision"}
    assert fields.tree_fields(df, max_levels=2)["cat"] == ["new_business", "region", "make_model"]


def test_credibility_encode_out_of_fold_blank_and_unseen():
    rng = np.random.default_rng(0)
    n = 4000
    code = pd.Series(rng.choice(["hi", "lo", None], n))
    E = np.ones(n)
    A = np.where(code == "hi", 2.0, np.where(code == "lo", 0.5, 1.0)) * rng.gamma(2.0, 0.5, n)
    fit = np.arange(n) < 3000
    code.iloc[-5:] = "never_seen"
    enc, k = fields.credibility_encode(code, rng.choice([2023, 2024], n), A, E, fit, np.arange(n) // 2, seed=11)
    assert k > 0
    hi, lo = enc[(code == "hi").to_numpy()], enc[(code == "lo").to_numpy()]
    assert hi.mean() > 0.5 and lo.mean() < -0.5                 # log 2 and log 0.5, shrunk a little
    assert np.isfinite(enc[code.isna().to_numpy()]).all()       # blank is a level of its own
    assert (enc[-5:] == 0).all()                                # a level never seen in fitting gets no adjustment
    # fitting rows get values from other folds, so a level's fitting rows do not all share one value
    assert len(np.unique(enc[fit & (code == "hi").to_numpy()])) > 1
    assert len(np.unique(enc[~fit & (code == "hi").to_numpy()])) == 1


def test_credibility_encode_refuses_one_year():
    import pytest
    code = pd.Series(["a", "b"] * 50)
    with pytest.raises(ValueError, match="two or more years"):
        fields.credibility_encode(code, np.full(100, 2023), np.ones(100), np.ones(100), np.ones(100, bool), np.arange(100), seed=11)


def test_credibility_encode_no_signal_gives_zero():
    rng = np.random.default_rng(1)
    n = 2000
    code = pd.Series(rng.choice(list("abcd"), n))
    A = rng.poisson(1.0, n).astype(float)        # the same rate in every level
    enc, k = fields.credibility_encode(code, rng.choice([2023, 2024], n), A, np.ones(n), np.ones(n, bool), np.arange(n), seed=11)
    assert not np.isfinite(k) and (enc == 0).all() or abs(enc).max() < 0.05
