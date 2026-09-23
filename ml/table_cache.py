"""Disk cache for the ~1.4M-row feature table.

`build_feature_table()` takes ~12 s (melting flows, merging weather/events, a per-closure
string scan). It is a pure function of the dataset files and the feature code, so we cache it
as parquet keyed by a fingerprint of (every file in the dataset folder: name/size/mtime,
features.py, data_loader.py). Loading the cache takes ~1 s. Drop a new dataset into data/ and
the fingerprint changes, so the table is rebuilt automatically.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("ML_CACHE_DIR", REPO_ROOT / "ml" / "cache"))


def _fingerprint(folder: str) -> str:
    h = hashlib.sha1()
    for p in sorted(Path(folder).glob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    for code in (REPO_ROOT / "ml" / "features.py", REPO_ROOT / "dashboard" / "utils" / "data_loader.py"):
        h.update(f"{code.name}:{int(code.stat().st_mtime)}".encode())
    return h.hexdigest()[:16]


def cached_feature_table(folder: str) -> pd.DataFrame:
    """Same result as features.build_feature_table(folder), ~10x faster on repeat calls."""
    from features import build_feature_table

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"feature_table_{_fingerprint(folder)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    table = build_feature_table(folder)
    for old in CACHE_DIR.glob("feature_table_*.parquet"):   # keep only the current one
        old.unlink()
    table.to_parquet(path, index=False)
    return table
