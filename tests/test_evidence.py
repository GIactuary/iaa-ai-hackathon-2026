import json

import numpy as np
import pandas as pd
import pytest

from src.evidence import (adjustment_summary, ae_by_segment, disagreement_check, dislocation, double_lift, gini,
                          lift_table, score_table, seed_scores, write_evidence)
from src.families import Binomial, LogGaussian, Poisson


@pytest.fixture()
def counts(count_data):
    d, y, e, _ = count_data
    good = np.exp(np.log(e) - 2 + 0.4 * d.x1 - 0.3 * d.x2 + d.region.map({"A": -0.3, "B": 0, "C": 0.2, "D": 0.4, "Z": 0}).values)
    rough = np.exp(np.log(e) - 2 + 0.4 * d.x1)
    rough *= y.sum() / rough.sum()
    return d, y, e, good, rough


def test_lift_is_monotone_and_balanced(counts):
    d, y, e, good, _ = counts
    t = lift_table(y, good, exposure=e)
    assert len(t) == 10
    assert sum(r["n"] for r in t) == len(y)
    assert sum(r["exposure"] for r in t) == pytest.approx(e.sum())
    exp_share = [r["exposure"] / e.sum() for r in t]
    assert max(exp_share) - min(exp_share) < 0.01                       # equal-exposure bins
    pm = [r["predicted_mean"] for r in t]
    assert pm == sorted(pm)
    assert t[-1]["actual_mean"] > 2 * t[0]["actual_mean"]


def test_calibration_bands_for_yes_no(counts):
    d, y, e, good, _ = counts
    yes = (y > 0).astype(float)
    p = 1 - np.exp(-good)
    t = lift_table(yes, p, ci=True)
    covered = sum(r["ci_lo"] <= r["predicted_mean"] <= r["ci_hi"] for r in t)
    assert covered >= 8


def test_double_lift_favours_true_model(counts):
    d, y, e, good, rough = counts
    t = double_lift(y, rough, good, exposure=e)
    err_good = np.mean([abs(r["ae_proposed"] - 1) for r in t])
    err_rough = np.mean([abs(r["ae_current"] - 1) for r in t])
    assert err_good < err_rough
    assert t[0]["ratio_hi"] <= t[-1]["ratio_lo"]


def test_ae_by_segment_and_gini(counts):
    d, y, e, good, rough = counts
    seg = ae_by_segment(y, rough, d.region, exposure=e)
    assert seg[0]["n"] >= seg[-1]["n"]
    ae = {r["segment"]: r["ae"] for r in seg}
    assert ae["D"] > 1 > ae["A"]                                       # the rough model leaves out region
    assert gini(y, good, e) > gini(y, rough, e) > 0


def test_score_tables_and_disagreement(reg_data):
    d, y = reg_data
    fam = LogGaussian()
    z = fam.link(y)
    rng = np.random.default_rng(0)
    oof = {s: pd.DataFrame(dict(glm=z + rng.normal(0, .1, len(z)), gbm=z + rng.normal(0, .1, len(z)))) for s in (42, 1, 2)}
    for o in oof.values():
        o["blend"] = (o.glm + o.gbm) / 2
    ss = seed_scores(fam, y, oof, ["glm", "gbm", "blend"])
    rows = {r["model"]: r for r in ss["rows"]}
    assert rows["blend"]["mean"] < rows["glm"]["mean"]                 # averaging two independent errors helps
    st = score_table(fam, {"blend": dict(apparent=(y, z + 0.01), cv=(y, oof[42].blend), holdout=None)})
    assert st[0]["holdout"] is None and st[0]["apparent"] < st[0]["cv"]
    dc = disagreement_check(fam, y, oof, top_share=0.02)
    assert dc["n_top"] >= 0.02 * 3 * len(y) - 1


def test_dislocation_and_adjustment():
    cur = np.full(100, 100.0)
    new = cur * np.r_[np.full(50, 1.01), np.full(30, 1.07), np.full(20, 0.80)]
    dl = dislocation(new, cur)
    assert dl["within5"] == 0.5 and dl["over15"] == 20 and sum(b["n"] for b in dl["bands"]) == 100
    a = adjustment_summary(new / cur)
    assert a["within5"] == 0.5 and a["max_down"] == pytest.approx(0.8)


def test_write_evidence_is_strict_json(tmp_path):
    p = write_evidence(dict(x=np.float64(0.5), bad=np.nan, df=pd.DataFrame(dict(a=[1, 2]))), tmp_path / "e.json", seeds=[42])
    e = json.loads(p.read_text())
    assert e["x"] == 0.5 and e["bad"] is None and e["df"] == [{"a": 1}, {"a": 2}]
    assert e["meta"]["seeds"] == [42] and "lightgbm" in e["meta"]["versions"]
