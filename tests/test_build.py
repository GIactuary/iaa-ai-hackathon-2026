"""Tests for src/build.py: line completion follows each year's own peril mix (exchange D16)."""
import pandas as pd
import pytest

from src.build import line_completion_by_year


def test_completion_weights_follow_each_years_peril_mix():
    # slow peril (completion 0.5) is 80% of ultimate cost in 2023 but 20% in 2024
    rows = pd.DataFrame({
        "yr": [2023, 2024], "holdout": [False, False], "exposure": [1.0, 1.0],
        "completion_slow": [0.5, 0.5], "completion_fast": [1.0, 1.0],
        "costc_slow": [80 * 0.5, 20 * 0.5], "costc_fast": [20.0, 80.0],
    })
    c = line_completion_by_year(rows)
    assert c.iloc[0] == pytest.approx(0.8 * 0.5 + 0.2 * 1.0)
    assert c.iloc[1] == pytest.approx(0.2 * 0.5 + 0.8 * 1.0)


def test_holdout_rows_do_not_set_the_weights():
    rows = pd.DataFrame({
        "yr": [2024, 2024], "holdout": [False, True], "exposure": [1.0, 1.0],
        "completion_slow": [0.5, 0.5], "completion_fast": [1.0, 1.0],
        "costc_slow": [0.0, 1000.0], "costc_fast": [10.0, 0.0],
    })
    c = line_completion_by_year(rows)
    assert c.tolist() == [1.0, 1.0]
