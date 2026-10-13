import numpy as np
import pandas as pd
import pytest

from src.families import Binomial, LogGaussian, Poisson
from src.glm import ForwardGLM
from src.pipeline import ModelPair, PredictionRanges, ReferralRule
from src.thresholds import load_thresholds
from src.trees import TreeModel
from src.validation import make_folds, oof_predictions

FAST = dict(num_boost_round=300, learning_rate=0.05)


def _pair(fam, t):
    return ModelPair(ForwardGLM.from_thresholds(fam, t), TreeModel(fam, FAST))


def test_blend_is_fixed_average_of_links(reg_data, thresholds_path):
    d, y = reg_data
    t = load_thresholds(thresholds_path)
    p = _pair(LogGaussian(), t).fit(d, y)
    pr = p.predict(d)
    assert np.allclose(pr.blend, (pr.glm + pr.gbm) / 2)
    # on a log scale the blend is the GLM times the square root of tree / GLM
    assert np.allclose(np.exp(pr.blend), np.exp(pr.glm) * p.adjustment(pr))
    with pytest.raises(ValueError, match="same family"):
        ModelPair(ForwardGLM(LogGaussian(), 0.002, 5), TreeModel(Poisson()))


def test_pair_out_of_fold_with_offset(count_data, thresholds_path):
    d, y, e, _ = count_data
    n = 6000
    d, y, e = d.iloc[:n].reset_index(drop=True), y[:n], e[:n]
    t = load_thresholds(thresholds_path)
    fam = Poisson()
    pair = ModelPair(ForwardGLM.from_thresholds(fam, t, candidates=["x1", "x2", "x3", "region"]),
                     TreeModel(fam, FAST), tree_cols=["x1", "x2", "x3", "region"])
    off = np.log(e)
    folds = make_folds(n, seeds=(42,))

    def fp(a, b, seed):
        return pair.clone().fit(d.iloc[a], y[a], offset=off[a]).predict(d.iloc[b], offset=off[b])

    o = oof_predictions(fp, n, folds)[42]
    flat = off + np.log(y.sum() / e.sum())
    for c in ("glm", "gbm", "blend"):
        assert fam.score(y, o[c]) < fam.score(y, flat)


def test_referral_flags_out_of_evidence_and_disagreement(reg_data, thresholds_path):
    d, y = reg_data
    t = load_thresholds(thresholds_path)
    rule = ReferralRule.from_thresholds(t, size_cols=["size"], segment_col="region").fit(d)
    new = d.head(5).copy().reset_index(drop=True)
    seg_max = d.groupby("region")["size"].max()
    new.loc[0, "size"] = 1.5 * seg_max[new.loc[0, "region"]]        # 1.5x its segment's largest: out of evidence
    new.loc[1, "size"] = 1.1 * seg_max[new.loc[1, "region"]]        # 1.1x: within the 1.25 limit
    new.loc[2, "region"] = "NEW"                                    # unseen level
    pred = pd.DataFrame(dict(glm=np.zeros(5), gbm=[0, 0, 0, np.log(1.5), np.log(1.2)]))   # row 3 is 50% apart, row 4 20%
    r = rule.apply(new, pred)
    assert r.size_flag.tolist() == [True, False, False, False, False]
    assert r.unseen_flag.tolist() == [False, False, True, False, False]
    assert r.gap_flag.tolist() == [False, False, False, True, False]
    assert r.referred.tolist() == [True, False, True, True, False]
    assert r.treatment[3] == "referred: model price, for review"
    assert "1.50x" in r.reason[0] and "50%" in r.reason[3]
    # the evidence check needs no model, so it can run on the test file on day one
    assert rule.evidence(new).out_of_evidence.sum() == 2


def test_ranges_cover_close_to_nominal(reg_data, thresholds_path):
    d, y = reg_data
    t = load_thresholds(thresholds_path)
    fam = LogGaussian()
    folds = make_folds(len(y), seeds=(42,))
    pair = _pair(fam, t)
    o = oof_predictions(lambda a, b, s: pair.clone().fit(d.iloc[a], y[a]).predict(d.iloc[b]), len(y), folds)[42]
    rng = PredictionRanges.from_thresholds(fam, t).fit(y, o.blend)
    cov = rng.coverage(y, o.blend, folds[42])
    assert cov["total"]["nominal"] == pytest.approx(0.90)
    assert 0.85 <= cov["total"]["coverage"] <= 0.95
    assert len(cov["by_group"]) == 5 and sum(g["n"] for g in cov["by_group"]) == len(y)
    lo, hi = rng.interval(o.blend)
    p = fam.inverse(o.blend)
    assert (lo < p).all() and (hi > p).all()


def test_ranges_ratio_scale_for_counts(count_data):
    d, y, e, _ = count_data
    fam = Poisson()
    eta = np.log(e) - 2 + 0.4 * d.x1 - 0.3 * d.x2
    r = PredictionRanges(fam, 0.05, 0.95).fit(y, eta)
    assert r.scale == "ratio"
    lo, hi = r.interval(eta)
    assert (lo >= 0).all() and (hi >= lo).all()
    with pytest.raises(ValueError, match="probability bands"):
        PredictionRanges(Binomial(), 0.05, 0.95)
