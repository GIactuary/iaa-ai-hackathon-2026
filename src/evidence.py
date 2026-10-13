"""The evidence file: every number the notebook quotes, written by the run and read by notebooks/build_notebook.py.

The run assembles a dictionary of results (structure decided once the data, the model and the evaluations are known)
and writes it with `write_evidence`. The builder fills the {{dotted.path|format}} tokens in the notebook text from it,
and the notebook's last cell checks that a clean rerun reproduces it.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGES = ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "catboost")


def to_jsonable(x):
    """Plain JSON types; NaN and infinity become null."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return to_jsonable(x.to_dict("records"))
    if isinstance(x, pd.Series):
        return to_jsonable(x.to_dict())
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (dt.date, dt.datetime)):
        return x.isoformat()
    return x


def run_metadata(seeds=None, packages=PACKAGES) -> dict:
    """When and from which commit the run was made, its seeds, and the versions of the packages that are installed."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip() or None
    except OSError:
        commit = None
    versions = {"python": platform.python_version()}
    for name in packages:
        try:
            versions[name] = importlib.import_module(name).__version__
        except ImportError:
            pass
    return dict(created=dt.datetime.now().isoformat(timespec="seconds"), git_commit=commit, seeds=seeds, versions=versions)


def write_evidence(evidence: dict, path: str | Path, seeds=None) -> Path:
    """Write the evidence dictionary (plus run metadata under "meta") as strict JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = to_jsonable({**evidence, "meta": run_metadata(seeds)})
    path.write_text(json.dumps(body, indent=1, allow_nan=False), encoding="utf-8", newline=chr(10))
    return path
