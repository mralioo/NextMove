"""Episodic memory: a persistent store of past executor facts, reused ACROSS sessions.

Session memory (the previous answer's facts, for follow-ups) already lives in ADK session state. Episodic memory
adds what a real operator assistant would want between conversations: if the same situation (same closure or
station + date/time) was analysed before, the facts are served from memory and the tools are skipped.

Entries are keyed by what the question is ABOUT (category + lines + stations + dates + time + duration + what-if),
not by its wording, and carry the dataset's last covered date so a new dataset (e.g. the Sept 22-30 evaluation
set) never gets stale facts. Entries expire after `ttl_s` (default 24 h). SQLite, one file per store.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path(os.environ.get("TMT_MEMORY_DB", REPO / "observability" / "memory.db"))


def key_for(plan: dict) -> str | None:
    """Stable key for what the plan is about; None when the plan has no analysable subject."""
    if plan.get("cat") not in ("C", "D"):
        return None
    subject = {k: plan.get(k) for k in ("cat", "lines", "stations", "dates", "time", "dur_min", "what_if", "raw")}
    if not (subject["stations"] or subject["lines"] or subject["dates"] or subject["raw"]):
        return None
    return hashlib.sha1(json.dumps(subject, sort_keys=True).encode()).hexdigest()[:20]


class EpisodicMemory:
    def __init__(self, path: Path | str = DEFAULT_DB, ttl_s: float = 24 * 3600) -> None:
        self.path, self.ttl_s = Path(path), ttl_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS facts (key TEXT, coverage_end TEXT, cat TEXT, facts_json TEXT, ts REAL, hits INTEGER DEFAULT 0, "
                      "PRIMARY KEY (key, coverage_end))")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def lookup(self, plan: dict, coverage_end: str) -> dict | None:
        k = key_for(plan)
        if k is None:
            return None
        with self._conn() as c:
            row = c.execute("SELECT facts_json, ts FROM facts WHERE key=? AND coverage_end=?", (k, coverage_end)).fetchone()
            if not row or time.time() - row[1] > self.ttl_s:
                return None
            c.execute("UPDATE facts SET hits = hits + 1 WHERE key=? AND coverage_end=?", (k, coverage_end))
        return {**json.loads(row[0]), "memory": "hit"}

    def store(self, plan: dict, coverage_end: str, facts: dict) -> None:
        k = key_for(plan)
        if k is None or facts.get("status") != "ok":
            return
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO facts (key, coverage_end, cat, facts_json, ts, hits) VALUES (?,?,?,?,?,0)",
                      (k, coverage_end, plan["cat"], json.dumps(facts), time.time()))

    def stats(self) -> dict:
        with self._conn() as c:
            n, h = c.execute("SELECT COUNT(*), COALESCE(SUM(hits),0) FROM facts").fetchone()
        return {"entries": n, "hits": h}
