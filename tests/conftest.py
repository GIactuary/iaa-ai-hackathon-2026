"""Synthetic data for the pipeline tests: one regression target (log-normal, like a price or a severity) and one
count target with exposure (like claim frequency). Both start from scikit-learn's make_regression features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THRESHOLDS_MD = """# Thresholds (test copy)

| Key | Threshold | Value | Meaning | Used in | Written on | Rationale |
|---|---|---|---|---|---|---|
| refer_size_ratio | Out-of-evidence referral | 1.25 | size beyond 1.25x | referral | 2026-10-12 | test |
| refer_disagreement | Model disagreement | 0.30 | 30% apart | referral | 2026-10-12 | test |
| drop_one_seeds_worse | Feature kept | 3 | all 3 seeds | drop-one | 2026-10-12 | test |
| glm_entry_gain | GLM entry | 0.002 | 0.2% | forward GLM | 2026-10-12 | test |
| glm_min_level_count | Level pooling | 5 | fewer than 5 pooled | forward GLM | 2026-10-12 | test |
| holdout_share | Hold-out | 0.20 | 20% | validation | 2026-10-12 | test |
| range_lower_q | Range lower | 0.05 | | ranges | 2026-10-12 | test |
| range_upper_q | Range upper | 0.95 | | ranges | 2026-10-12 | test |
| undated | Not yet dated | 1.5 | | nothing | | test |
| open_item | Still open | | | nothing | 2026-10-12 | test |
"""


def _features(n, seed):
    X, signal = make_regression(n_samples=n, n_features=8, n_informative=4, noise=0.0, random_state=seed)
    d = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
    # a categorical column with a real effect, plus one rare level that should be pooled
    region = pd.cut(d.x0, [-np.inf, -1, 0, 1, np.inf], labels=["A", "B", "C", "D"]).astype(str)
    region.iloc[:3] = "Z"
    d["region"] = region
    d["size"] = np.exp(7 + 0.3 * d.x1)        # a positive "size" column for the out-of-evidence check
    return d, signal


@pytest.fixture(scope="session")
def reg_data():
    """Log-normal target: log(1 + y) linear in a few columns plus a region effect, plus noise."""
    d, signal = _features(800, seed=0)
    rng = np.random.default_rng(0)
    reg_eff = d.region.map({"A": -0.2, "B": 0.0, "C": 0.1, "D": 0.25, "Z": 0.0}).values
    z = 12 + 0.25 * signal / signal.std() + reg_eff + 0.3 * np.log(d["size"] / 1000) + rng.normal(0, 0.1, len(d))
    return d, np.expm1(np.asarray(z))


@pytest.fixture(scope="session")
def count_data():
    """Poisson counts with exposure: log mu = log(exposure) - 2 + 0.4 x1 - 0.3 x2 + region effect."""
    d, _ = _features(20000, seed=1)
    rng = np.random.default_rng(1)
    exposure = rng.uniform(0.2, 1.0, len(d))
    reg_eff = d.region.map({"A": -0.3, "B": 0.0, "C": 0.2, "D": 0.4, "Z": 0.0}).values
    eta = np.log(exposure) - 2 + 0.4 * d.x1 - 0.3 * d.x2 + reg_eff
    y = rng.poisson(np.exp(eta)).astype(float)
    return d, y, exposure, dict(intercept=-2.0, x1=0.4, x2=-0.3)


@pytest.fixture()
def thresholds_path(tmp_path):
    p = tmp_path / "thresholds.md"
    p.write_text(THRESHOLDS_MD, encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def reg_run(reg_data, tmp_path_factory):
    p = tmp_path_factory.mktemp("reg")
    (p / "thresholds.md").write_text(THRESHOLDS_MD, encoding="utf-8")
    d, y = reg_data
    from synthetic_run import run
    from src.evidence import write_evidence
    from src.families import LogGaussian
    from src.thresholds import load_thresholds
    ev, oof = run(d, y, LogGaussian(), load_thresholds(p / "thresholds.md"), p)
    return ev, write_evidence(ev, p / "evidence.json", seeds=[42, 1, 2])
