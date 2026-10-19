import pytest

from src.thresholds import DEFAULT_PATH, ThresholdError, load_thresholds


def test_reads_dated_values(thresholds_path):
    t = load_thresholds(thresholds_path)
    assert t["example_share"] == pytest.approx(0.25)
    assert t["example_count"] == 7
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
    """The real decisions file parses; agreed thresholds read back, and open ones stay unusable until set and dated."""
    t = load_thresholds(DEFAULT_PATH)
    assert t["holdout_share"] == pytest.approx(0.15)
    assert t["screen_z"] == 4
    with pytest.raises(ThresholdError, match="still open"):
        t["refer_disagreement"]
