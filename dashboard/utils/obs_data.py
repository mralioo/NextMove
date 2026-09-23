"""Read-only access to the observability database written by the agent (`agent/observability.py`):
`runs` (one row per question), `spans` (ADK/OpenTelemetry trace spans), `eval_runs` / `eval_items`
(scored evaluation runs). Override the location with OBS_DB (the Docker image mounts it)."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(os.environ.get("OBS_DB", Path(__file__).resolve().parents[2] / "observability" / "agent_obs.db"))
BUDGET_S = float(os.environ.get("EVAL_LATENCY_BUDGET_S", "20"))


def db_exists() -> bool:
    return DB_PATH.exists()


def _mtime() -> float:
    try:
        return max(DB_PATH.stat().st_mtime, Path(str(DB_PATH) + "-wal").stat().st_mtime if Path(str(DB_PATH) + "-wal").exists() else 0)
    except OSError:
        return 0.0


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        conn.execute("PRAGMA query_only=ON")
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:          # table not created yet
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data(show_spinner=False, ttl=5)
def load_runs(_key: float = 0.0) -> pd.DataFrame:
    df = _query("SELECT * FROM runs ORDER BY ts")
    if df.empty:
        return df
    df["when"] = pd.to_datetime(df["ts"], unit="s")
    df["other_s"] = (df["total_s"] - df[["tools_s", "write_s"]].fillna(0).sum(axis=1) - df["route_ms"].fillna(0) / 1000).clip(lower=0)
    df["route_s"] = df["route_ms"].fillna(0) / 1000
    df["label"] = df.apply(lambda r: f"{r['when']:%m-%d %H:%M:%S} · {r['category'] or '-'} · {r['total_s']:.1f}s · {(r['question'] or '')[:60]}", axis=1)
    return df


@st.cache_data(show_spinner=False, ttl=5)
def load_spans(trace_id: str) -> pd.DataFrame:
    df = _query("SELECT * FROM spans WHERE trace_id=? ORDER BY start_time_unix_nano", (trace_id,))
    if df.empty:
        return df
    t0 = df["start_time_unix_nano"].min()
    df["start_ms"] = (df["start_time_unix_nano"] - t0) / 1e6
    df["duration_ms"] = (df["end_time_unix_nano"] - df["start_time_unix_nano"]) / 1e6
    df["attrs"] = df["attributes_json"].apply(lambda s: json.loads(s) if s else {})
    return df


@st.cache_data(show_spinner=False, ttl=5)
def load_tool_spans(_key: float = 0.0) -> pd.DataFrame:
    df = _query("SELECT name, trace_id, start_time_unix_nano s, end_time_unix_nano e FROM spans WHERE name LIKE 'mcp.tool %'")
    if df.empty:
        return df
    df["tool"] = df["name"].str.replace("mcp.tool ", "", regex=False)
    df["duration_s"] = (df["e"] - df["s"]) / 1e9
    return df


@st.cache_data(show_spinner=False, ttl=5)
def load_eval_runs(_key: float = 0.0) -> pd.DataFrame:
    df = _query("SELECT * FROM eval_runs ORDER BY ts")
    if df.empty:
        return df
    df["when"] = pd.to_datetime(df["ts"], unit="s")
    df["summary"] = df["summary_json"].apply(json.loads)
    return df


@st.cache_data(show_spinner=False, ttl=5)
def load_eval_items(eval_id: str) -> pd.DataFrame:
    df = _query("SELECT * FROM eval_items WHERE eval_id=? ORDER BY item_id, repeat_idx", (eval_id,))
    if df.empty:
        return df
    df["metrics"] = df["metrics_json"].apply(json.loads)
    df["checks"] = df["checks_json"].apply(json.loads)
    df["judge"] = df["judge_json"].apply(lambda s: json.loads(s) if s else None) if "judge_json" in df.columns else None
    return df


def key() -> float:
    return _mtime()


@st.cache_data(show_spinner=False, ttl=5)
def load_exp_runs(_key: float = 0.0) -> pd.DataFrame:
    df = _query("SELECT * FROM exp_runs ORDER BY exp_id, arm_id")
    if df.empty:
        return df
    df["when"] = pd.to_datetime(df["ts"], unit="s")
    df["params"] = df["params_json"].apply(json.loads)
    df["setup"] = df["setup_json"].apply(lambda s: json.loads(s) if s else {})
    df["summary"] = df["summary_json"].apply(lambda s: json.loads(s) if s else {})
    return df


@st.cache_data(show_spinner=False, ttl=5)
def load_exp_turns(exp_id: str) -> pd.DataFrame:
    df = _query("SELECT * FROM exp_turns WHERE exp_id=? ORDER BY arm_id, turn", (exp_id,))
    if df.empty:
        return df
    df["metrics"] = df["metrics_json"].apply(json.loads)
    df["checks"] = df["checks_json"].apply(json.loads)
    df["judge"] = df["judge_json"].apply(lambda s: json.loads(s) if s else None)
    df["facts"] = df["facts_json"].apply(lambda s: json.loads(s) if s else {})
    return df
