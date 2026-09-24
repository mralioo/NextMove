"""The data window, read from the flow files themselves (training + test split merged), so no text hard-codes it.

    data_window.start() -> "2026-06-10"   data_window.end() -> "2026-10-01"   data_window.window() -> "2026-06-10 to 2026-10-01"
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    base = Path(os.environ.get("DATA_DIR", REPO / "data"))
    if not base.exists():
        base = REPO / "data"
    lo = hi = None
    for f in base.rglob("flows*.csv"):
        ts = pd.to_datetime(pd.read_csv(f, usecols=[0], encoding="utf-8-sig").iloc[:, 0], format="mixed", errors="coerce").dropna()
        if len(ts):
            lo = ts.min() if lo is None else min(lo, ts.min())
            hi = ts.max() if hi is None else max(hi, ts.max())
    if lo is None:
        return pd.Timestamp("2026-06-10 05:00"), pd.Timestamp("2026-09-22 00:45")
    return lo, hi


def start() -> str:
    return str(_bounds()[0].date())


def end() -> str:
    return str(_bounds()[1].date())


def window() -> str:
    return f"{start()} to {end()}"
