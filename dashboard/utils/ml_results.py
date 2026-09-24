"""Loaders for the TabPFN ML-engine result files written by the training scripts
(`./.venv/bin/python scripts/tasks.py train-overcrowding`, `./.venv/bin/python scripts/tasks.py train-disruption`) into `ml/output/`.

The dashboard only ever READS these files — it never calls the TabPFN API — so the
page renders instantly and works offline / in Docker. Override the location with
the RESULTS_DIR env var (the Docker image mounts `ml/output` there).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[2] / "ml" / "output"))

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", Path(__file__).resolve().parents[2] / "ml" / "checkpoints"))

FILES = {
    "heldout": "disruption_heldout_predictions.csv",
    "showcase": "disruption_showcase_series.csv",
    "case_study": "disruption_case_study.csv",
    "report": "disruption_baseline_report.json",
    "classifier": "overcrowding_predictions.csv",
}


def result_path(key: str) -> Path:
    return RESULTS_DIR / FILES[key]


def _mtime(key: str) -> float:
    p = result_path(key)
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache after a retrain
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def load_csv_result(key: str) -> pd.DataFrame | None:
    p = result_path(key)
    return _read_csv(str(p), _mtime(key)) if p.exists() else None


def load_report() -> dict | None:
    p = result_path("report")
    return _read_json(str(p), _mtime("report")) if p.exists() else None


def load_manifest() -> list[dict] | None:
    """Model-checkpoint registry written by `./.venv/bin/python scripts/tasks.py checkpoints` (ml/checkpoints/manifest.json)."""
    p = CHECKPOINT_DIR / "manifest.json"
    return _read_json(str(p), p.stat().st_mtime) if p.exists() else None
