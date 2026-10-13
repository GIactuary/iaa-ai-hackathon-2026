import json

import numpy as np
import pytest

from src.evidence import write_evidence
from src.families import LogGaussian, Poisson
from src.thresholds import load_thresholds

from synthetic_run import run


def test_regression_end_to_end(reg_run):
    ev, path = reg_run
    rows = {r["model"]: r for r in ev["seed_scores"]["rows"]}
    assert rows["blend"]["mean"] < rows["current"]["mean"]
    assert all(c["gain"] > 0 for c in ev["compare"])
    st = {r["model"]: r for r in ev["score_table"]}
    assert st["proposed"]["apparent"] < st["proposed"]["cv"]          # fitted rows flatter the model
    assert abs(st["proposed"]["holdout"] / st["proposed"]["cv"] - 1) < 0.25
    assert 0.8 <= ev["ranges"]["total"]["coverage"] <= 0.97
    e = json.loads(path.read_text())
    assert e["summary"]["n_holdout"] > 0 and e["meta"]["seeds"] == [42, 1, 2]


def test_count_end_to_end(count_data, tmp_path):
    import conftest
    (tmp_path / "thresholds.md").write_text(conftest.THRESHOLDS_MD, encoding="utf-8")
    d, y, e, _ = count_data
    n = 8000
    ev, oof = run(d.iloc[:n].reset_index(drop=True), y[:n], Poisson(), load_thresholds(tmp_path / "thresholds.md"), tmp_path,
                  offset=np.log(e[:n]), seeds=(42, 1))
    rows = {r["model"]: r for r in ev["seed_scores"]["rows"]}
    assert rows["blend"]["mean"] < rows["current"]["mean"]
    lift = ev["lift"]["proposed"]
    assert len(lift) == 10
    # out of fold, the blend's total is close to the actual total (each fold's GLM balances on its own rows)
    assert sum(r["actual"] for r in lift) / sum(r["predicted"] for r in lift) == pytest.approx(1, abs=0.05)
    assert ev["gini"]["proposed"] > ev["gini"]["current"]
    write_evidence(ev, tmp_path / "evidence.json")
