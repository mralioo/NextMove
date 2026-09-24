"""Storage of submission runs: one row per run (with the full system-design configuration) and one row per answered question.

SQLite file `observability/submissions.db` (override SUBMISSIONS_DB). Every run is also exported to `evaluation/submissions/<run_id>.json` (tracked in git).
The dashboard page "Submission runs" reads these tables; comparing runs = comparing their `config_json` and `metrics_json`.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("SUBMISSIONS_DB", REPO / "observability" / "submissions.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS sub_runs (
  run_id TEXT PRIMARY KEY, created_at TEXT, label TEXT, note TEXT, git_commit TEXT, status TEXT,
  config_json TEXT,      -- the system design of this run: pipeline, models per role, switches, limits, data window, ML checkpoints, versions
  metrics_json TEXT,     -- aggregate metrics of the run
  stages TEXT, n_questions INTEGER, wall_s REAL
);
CREATE TABLE IF NOT EXISTS sub_answers (
  run_id TEXT, item_id TEXT, stage TEXT, seq INTEGER, question TEXT, answer TEXT,
  source TEXT,           -- agent | authored+measured
  category TEXT, decision TEXT, verdict TEXT, confidence REAL, facts_status TEXT, sanity_ok INTEGER, guard TEXT, answer_words INTEGER,
  latency_s REAL, supervisor_s REAL, worker_s REAL, mcp_s REAL, writer_s REAL, llm_s REAL,
  llm_calls INTEGER, tok_in INTEGER, tok_out INTEGER, mcp_calls INTEGER, rounds INTEGER, n_spans INTEGER,
  models_json TEXT, llm_json TEXT, calls_json TEXT, plan_json TEXT, facts_json TEXT, checks_json TEXT,
  obs_run_id TEXT, trace_id TEXT, error TEXT,
  PRIMARY KEY (run_id, item_id)
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = lambda cur, row: {d[0]: row[i] for i, d in enumerate(cur.description)}
    conn.executescript(SCHEMA)
    return conn


def save_run(conn, run: dict, answers: list[dict]) -> None:
    conn.execute("INSERT OR REPLACE INTO sub_runs (run_id, created_at, label, note, git_commit, status, config_json, metrics_json, stages, n_questions, wall_s) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (run["run_id"], run["created_at"], run.get("label"), run.get("note"), run.get("git_commit"), run.get("status", "done"), json.dumps(run["config"], default=str),
                  json.dumps(run["metrics"], default=str), ",".join(run.get("stages", [])), len(answers), run.get("wall_s")))
    for a in answers:
        a = {**a, "run_id": run["run_id"]}
        cols = list(a)
        conn.execute(f"INSERT OR REPLACE INTO sub_answers ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [a[c] if not isinstance(a[c], (dict, list)) else json.dumps(a[c], default=str) for c in cols])
    conn.commit()


def list_runs(conn) -> list[dict]:
    return conn.execute("SELECT * FROM sub_runs ORDER BY created_at DESC").fetchall()


def answers(conn, run_id: str) -> list[dict]:
    return conn.execute("SELECT * FROM sub_answers WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()


def import_json(conn, path: Path) -> str:
    """Re-create a run from its exported JSON (evaluation/submissions/<run_id>.json): the database can always be rebuilt from the tracked files."""
    d = json.loads(Path(path).read_text())
    answers_ = d.pop("answers")
    d["config"], d["metrics"] = d["config"], d["metrics"]
    conn.execute("DELETE FROM sub_answers WHERE run_id=? OR run_id IS NULL", (d["run_id"],))
    save_run(conn, d, answers_)
    return d["run_id"]
