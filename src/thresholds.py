"""Read decisions/thresholds.md, the one place thresholds live (CLAUDE.md rule 2: thresholds before results).

A threshold is returned only if its row has a value and a "Written on" date, so the code cannot run a test on a
threshold that was not written down first. Pass `require_dated=False` only in tests on synthetic data.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from src import market
DEFAULT_PATH = market.THRESHOLDS            # decisions/thresholds.md for Part 1


class ThresholdError(KeyError):
    """A threshold is missing, still open, or not yet dated."""


@dataclass(frozen=True)
class Threshold:
    key: str
    value: float | None
    written_on: dt.date | None
    row: dict


def _parse_value(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as e:
        raise ThresholdError(f"value {text!r} is not a plain number; write shares as decimals (0.2% as 0.002)") from e


def _parse_date(text: str) -> dt.date | None:
    text = text.strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text)
    except ValueError as e:
        raise ThresholdError(f"'Written on' date {text!r} is not YYYY-MM-DD") from e


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


class Thresholds:
    def __init__(self, rows: dict[str, Threshold], require_dated: bool = True, source: str = ""):
        self.rows, self.require_dated, self.source = rows, require_dated, source

    def get(self, key: str) -> float:
        if key not in self.rows:
            raise ThresholdError(f"threshold {key!r} is not in {self.source}; write it there before running the test that uses it")
        t = self.rows[key]
        if t.value is None:
            raise ThresholdError(f"threshold {key!r} in {self.source} has no value yet; it is still open")
        if self.require_dated and t.written_on is None:
            raise ThresholdError(f"threshold {key!r} in {self.source} has no 'Written on' date; date it before running the test that uses it")
        return t.value

    def __getitem__(self, key: str) -> float:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def as_records(self) -> list[dict]:
        """Every row as written, for the notebook's thresholds table."""
        return [dict(t.row) for t in self.rows.values()]


def load_thresholds(path: str | Path = DEFAULT_PATH, require_dated: bool = True) -> Thresholds:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = None
    rows: dict[str, Threshold] = {}
    for line in lines:
        if not line.strip().startswith("|"):
            header = None if line.strip() else header
            continue
        cells = _cells(line)
        if header is None:
            if "Key" in cells and "Value" in cells:
                header = cells
            continue
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            continue
        row = dict(zip(header, cells + [""] * (len(header) - len(cells))))
        key = row["Key"]
        if not key:
            continue
        if key in rows:
            raise ThresholdError(f"threshold {key!r} appears twice in {path}")
        rows[key] = Threshold(key, _parse_value(row["Value"]), _parse_date(row.get("Written on", "")), row)
    if not rows:
        raise ThresholdError(f"no table with 'Key' and 'Value' columns found in {path}")
    return Thresholds(rows, require_dated=require_dated, source=str(path))
