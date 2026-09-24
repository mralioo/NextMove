"""Read access to the stored submission runs (written by evaluation/submission_run.py, see evaluation/submission_db.py)."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("SUBMISSIONS_DB", REPO / "observability" / "submissions.db"))


def exists() -> bool:
    return DB_PATH.exists()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


@st.cache_data(ttl=10, show_spinner=False)
def runs() -> list[dict]:
    with _conn() as c:
        out = []
        for r in c.execute("SELECT * FROM sub_runs ORDER BY created_at DESC"):
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json") or "{}")
            d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
            out.append(d)
        return out


@st.cache_data(ttl=10, show_spinner=False)
def answers(run_id: str) -> pd.DataFrame:
    with _conn() as c:
        df = pd.DataFrame([dict(r) for r in c.execute("SELECT * FROM sub_answers WHERE run_id=? ORDER BY seq", (run_id,))])
    for col in ("models_json", "llm_json", "calls_json", "plan_json", "facts_json", "checks_json"):
        if col in df:
            df[col] = df[col].map(lambda v: json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v)
    return df


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[key] = json.dumps(v, default=str)
        else:
            out[key] = v if not isinstance(v, list) else ", ".join(map(str, v))
    return out
