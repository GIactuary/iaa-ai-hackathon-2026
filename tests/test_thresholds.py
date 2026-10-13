import pytest

from src.thresholds import DEFAULT_PATH, ThresholdError, load_thresholds


def test_reads_dated_values(thresholds_path):
    t = load_thresholds(thresholds_path)
    assert t["glm_entry_gain"] == pytest.approx(0.002)
    assert t["glm_min_level_count"] == 5
    assert "undated" in t


def test_refuses_undated_open_and_missing(thresholds_path):
    t = load_thresholds(thresholds_path)
    with pytest.raises(ThresholdError, match="Written on"):
        t["undated"]
    with pytest.raises(ThresholdError, match="still open"):
        t["open_item"]
    with pytest.raises(ThresholdError, match="not in"):
        t["never_written"]
    assert load_thresholds(thresholds_path, require_dated=False)["undated"] == 1.5


def test_repo_file_parses():
    """The real decisions file parses; its values stay unusable until the actuary dates them."""
    t = load_thresholds(DEFAULT_PATH, require_dated=False)
    for key in ["screen_z", "refer_size_ratio", "refer_disagreement", "drop_one_seeds_worse",
                "glm_entry_gain", "glm_min_level_count", "holdout_share"]:
        assert key in t
        t[key]
