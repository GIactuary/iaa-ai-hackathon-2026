"""Shared test set-up. The tests cover the process code only; no model code exists until the data arrives."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THRESHOLDS_MD = """# Thresholds (test copy)

| Key | Threshold | Value | Meaning | Used in | Written on | Rationale |
|---|---|---|---|---|---|---|
| example_share | Example share | 0.25 | | test | 2026-10-12 | test |
| example_count | Example count | 7 | | test | 2026-10-12 | test |
| undated | Not yet dated | 1.5 | | nothing | | test |
| open_item | Still open | | | nothing | 2026-10-12 | test |
"""


@pytest.fixture()
def thresholds_path(tmp_path):
    p = tmp_path / "thresholds.md"
    p.write_text(THRESHOLDS_MD, encoding="utf-8")
    return p
