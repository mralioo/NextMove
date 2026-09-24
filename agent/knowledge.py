"""Knowledge base + long-term memory for the main agent (local-first, mirrored to Cognee).

    KnowledgeBase   curated ground truth / boundaries / insights (knowledge/knowledge.json, built from the raw data by
                    agent/knowledge_build.py) with deterministic retrieval, a SANITY CHECKER, and a per-session turn history.
    CogneeClient    the Cognee Cloud REST API (https://docs.cognee.ai): knowledge graph + session memory. Optional and OFF the hot
                    path: measured recall latency is seconds, so it is used for (a) writing turns/insights in the background,
                    (b) semantic recall for evaluators and follow-ups, never as a blocking step in front of the operator.

Why local-first: an operator answer must not wait for a network memory service, and the boundaries ("no capacity data",
"data ends 2026-09-22") must be enforced even when the service is down. The same content is pushed to Cognee (`./.venv/bin/python scripts/tasks.py kb-sync`)
so it also lives in a graph that other agents / evaluators can query through MCP (mcp_server/knowledge_server.py).

Cognee is used only if COGNEE_ENABLED is true and COGNEE_API_BASE_URL / COGNEE_API_KEY are set; after two consecutive failures it is
switched off for 5 minutes (circuit breaker) and everything keeps working locally.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KB_JSON = REPO / "knowledge" / "knowledge.json"
DEFAULT_DB = Path(os.environ.get("TMT_MEMORY_DB", REPO / "observability" / "memory.db"))
DATASET = os.environ.get("COGNEE_DATASET", "next_move")

# which boundaries matter for which category (the generic ones are added by trigger words / dates in `boundaries_for`)
CAT_BOUNDARIES = {"C": ["B-ASSUME", "B-CLS"], "D": ["B-SIM"], "A": ["B-EVT", "B-NAME"], "P": ["B-EVT", "B-NAME"], "B": ["B-SIM", "B-EVT"],
                  "E": ["B-EN"], "F": [], "G": ["B-OD"], "H": ["B-OD", "B-CLS"], "X": ["B-OD"]}


def _tok(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-zäöüß0-9]{3,}", s.lower())]


# ============================================================================================ Cognee
class CogneeClient:
    """Thin, failure-tolerant client for the Cognee Cloud API (X-Api-Key auth)."""

    def __init__(self) -> None:
        self.base = (os.environ.get("COGNEE_API_BASE_URL") or "").rstrip("/")
        self.key = os.environ.get("COGNEE_API_KEY") or ""
        self.enabled_flag = str(os.environ.get("COGNEE_ENABLED", "")).lower() in ("1", "true", "yes", "on")
        self._fails, self._off_until, self.last_error = 0, 0.0, None
        self._client = None
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "errors": 0, "recall_ms": []}

    @property
    def configured(self) -> bool:
        return bool(self.enabled_flag and self.base and self.key)

    @property
    def available(self) -> bool:
        return self.configured and time.time() >= self._off_until

    def _http(self):
        import httpx

        with self._lock:
            if self._client is None:
                self._client = httpx.Client(base_url=self.base, headers={"X-Api-Key": self.key}, timeout=30)
            return self._client

    def _req(self, method: str, path: str, timeout: float = 20, **kw):
        """One call; returns parsed JSON or None. Never raises; trips the circuit breaker on repeated failure."""
        if not self.available:
            return None
        self.stats["calls"] += 1
        try:
            r = self._http().request(method, path, timeout=timeout, **kw)
            r.raise_for_status()
            self._fails = 0
            return r.json()
        except Exception as e:                                      # network, auth, 5xx: memory must never break the agent
            self.stats["errors"] += 1
            self._fails += 1
            self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
            if self._fails >= 2:
                self._off_until = time.time() + 300
            return None

    # -- knowledge (documents -> graph)
    def health(self) -> dict | None:
        return self._req("GET", "/health", timeout=8)

    def dataset_id(self, name: str = DATASET) -> str | None:
        for d in self._req("GET", "/api/v1/datasets/") or []:
            if d.get("name") == name:
                return d["id"]
        return None

    def add_texts(self, texts: list[str], node_set: list[str] | None = None, dataset: str = DATASET):
        return self._req("POST", "/api/v1/add_text", timeout=60, json={"textData": texts, "datasetName": dataset, "nodeSet": node_set or []})

    def cognify(self, dataset: str = DATASET, wait: bool = False):
        """Build the knowledge graph from the added documents (server-side LLM extraction). wait=False returns at once."""
        return self._req("POST", "/api/v1/cognify", timeout=600 if wait else 60, json={"datasets": [dataset], "runInBackground": not wait})

    def status(self, dataset: str = DATASET) -> str | None:
        did = self.dataset_id(dataset)
        r = self._req("GET", "/api/v1/datasets/status", params={"dataset": did}) if did else None
        return next(iter(r.values())) if isinstance(r, dict) and r else None

    def clear_dataset(self, dataset: str = DATASET) -> int:
        did = self.dataset_id(dataset)
        n = 0
        for item in (self._req("GET", f"/api/v1/datasets/{did}/data") if did else None) or []:
            if self._req("DELETE", f"/api/v1/datasets/{did}/data/{item['id']}") is not None:
                n += 1
        return n

    # -- memory (recall + session QA)
    def recall(self, query: str, top_k: int = 3, search_type: str = "CHUNKS", session_id: str | None = None, dataset: str = DATASET) -> list[dict]:
        t = time.time()
        body = {"query": query, "searchType": search_type, "datasets": [dataset], "topK": top_k}
        if session_id:
            body["sessionId"] = session_id
        r = self._req("POST", "/api/v1/recall", timeout=25, json=body)
        self.stats["recall_ms"].append(round((time.time() - t) * 1000))
        return [{"text": x.get("text", ""), "kind": x.get("kind"), "source": x.get("source", "cognee")} for x in (r or []) if isinstance(x, dict)]

    def remember_qa(self, session_id: str, question: str, answer: str, context: str = "", dataset: str = DATASET):
        return self._req("POST", "/api/v1/remember/entry", timeout=30, json={
            "entry": {"type": "qa", "question": question, "answer": answer, "context": context[:1500]}, "dataset_name": dataset, "session_id": session_id})

    def visualize(self, dataset: str = DATASET) -> str | None:
        """Cognee's own interactive knowledge-graph page (self-contained HTML, ~2.5 MB, d3) for a dataset, or None."""
        if not self.available:
            return None
        did = self.dataset_id(dataset)
        if not did:
            return None
        try:
            r = self._http().get("/api/v1/visualize", params={"dataset_id": did}, timeout=90)
            r.raise_for_status()
            return r.text
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {str(e)[:120]}"
            return None

    def session(self, session_id: str):
        return self._req("GET", f"/api/v1/sessions/{session_id}", timeout=15)


_COGNEE: CogneeClient | None = None


def cognee() -> CogneeClient:
    global _COGNEE
    if _COGNEE is None:
        _COGNEE = CogneeClient()
    return _COGNEE


# ============================================================================================ knowledge base
class KnowledgeBase:
    def __init__(self, path: Path = KB_JSON, db: Path = DEFAULT_DB) -> None:
        self.path, self.db = Path(path), Path(db)
        self.entries: list[dict] = json.loads(self.path.read_text()) if self.path.exists() else []
        self.by_id = {e["id"]: e for e in self.entries}
        df = Counter(t for e in self.entries for t in set(_tok(e["text"])))
        n = max(len(self.entries), 1)
        self._idf = {t: math.log(1 + n / c) for t, c in df.items()}
        self.db.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, user_id TEXT, ts REAL, question TEXT, cat TEXT, "
                      "answer TEXT, facts_json TEXT, sanity_json TEXT)")
            for col, typ in (("verdict", "TEXT"), ("confidence", "REAL"), ("plan_key", "TEXT"), ("data_end", "TEXT"), ("accepted", "INTEGER"), ("plan_json", "TEXT"), ("artifact_json", "TEXT"), ("operator_score", "REAL"), ("n_feedback", "INTEGER"), ("linked_from", "INTEGER")):
                try:                                       # turn store created before v3: add the columns the supervisor's history lookup needs
                    c.execute(f"ALTER TABLE turns ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass
            c.execute("CREATE TABLE IF NOT EXISTS insights (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, text TEXT, cats TEXT, source TEXT)")
            # operator feedback: a score of an answer and / or what the operator actually did about the situation (see feedback.py)
            c.execute("CREATE TABLE IF NOT EXISTS operator_feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, turn_id INTEGER, session_id TEXT, user_id TEXT, kind TEXT, score INTEGER, "
                      "comment TEXT, action_text TEXT, followed TEXT, outcome TEXT, occurred_at TEXT, stations_json TEXT, lines_json TEXT, category TEXT, problem_key TEXT, deleted INTEGER DEFAULT 0)")
            c.execute("CREATE INDEX IF NOT EXISTS fb_turn ON operator_feedback(turn_id)")

    def _conn(self):
        return sqlite3.connect(self.db, timeout=10)

    # -- retrieval
    def search(self, query: str, cats: list[str] | None = None, kinds: list[str] | None = None, k: int = 3) -> list[dict]:
        q = set(_tok(query))
        out = []
        for e in self.entries + self.user_insights():
            if kinds and e["kind"] not in kinds:
                continue
            if cats and not (set(cats) & set(e["cats"])):
                continue
            score = sum(self._idf.get(t, 1.0) for t in q & set(_tok(e["text"])))
            if score > 0:
                out.append((score, e))
        return [dict(e, score=round(s, 2)) for s, e in sorted(out, key=lambda x: -x[0])[:k]]

    def user_insights(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, text, cats, source FROM insights").fetchall()
        return [{"id": f"U-{i}", "kind": "insight", "cats": (cats or "").split(",") or ["X"], "text": t, "value": {}, "source": src or "operator/agent"} for i, t, cats, src in rows]

    def add_insight(self, text: str, cats: list[str] | None = None, source: str = "agent") -> str:
        with self._conn() as c:
            cur = c.execute("INSERT INTO insights (ts, text, cats, source) VALUES (?,?,?,?)", (time.time(), text, ",".join(cats or []), source))
        if cognee().available:
            threading.Thread(target=cognee().add_texts, args=([text], ["insight"]), daemon=True).start()
        return f"U-{cur.lastrowid}"

    def boundaries_for(self, plan: dict, question: str = "") -> list[dict]:
        ids = list(CAT_BOUNDARIES.get(plan.get("cat"), []))
        if plan.get("dates") or plan.get("month") or plan.get("rel_day"):
            ids.append("B-COV")
        if plan.get("oos") and "capacity" in " ".join(plan["oos"]) or re.search(r"capacity|safely hold", question, re.I):
            ids.append("B-CAP")
        return [self.by_id[i] for i in dict.fromkeys(ids) if i in self.by_id]

    # -- history
    def remember_turn(self, session_id: str, user_id: str, question: str, cat: str, answer: str, facts: dict | None, sanity: dict | None = None, *,
                      verdict: str | None = None, confidence: float | None = None, plan_key: str | None = None, data_end: str | None = None,
                      accepted: bool | None = None, plan: dict | None = None, artifact: dict | None = None, linked_from: int | None = None) -> int:
        with self._conn() as c:
            tid = c.execute("INSERT INTO turns (session_id, user_id, ts, question, cat, answer, facts_json, sanity_json, verdict, confidence, plan_key, data_end, accepted, plan_json, artifact_json, linked_from) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (session_id, user_id, time.time(), question, cat, answer, json.dumps(facts or {}, default=str), json.dumps(sanity or {}, default=str), verdict, confidence,
                             plan_key, data_end, None if accepted is None else int(accepted), json.dumps(plan, default=str) if plan else None,
                             json.dumps({**artifact, "turn_id": None}, default=str) if artifact else None, linked_from)).lastrowid
            if artifact:                                     # the bundle knows its own id
                c.execute("UPDATE turns SET artifact_json=json_set(artifact_json, '$.turn_id', ?) WHERE id=?", (tid, tid))
            return tid

    # -- operator feedback (score of the response, what the operator did): the raw records; the graph and precedents are fed by feedback.py
    def turn_info(self, turn_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT id, session_id, user_id, ts, question, cat, answer, confidence, accepted, operator_score, n_feedback, artifact_json FROM turns WHERE id=?", (turn_id,)).fetchone()
        if not r:
            return None
        art = json.loads(r[11]) if r[11] else None
        return {"turn_id": r[0], "session_id": r[1], "operator_id": r[2], "ts": r[3], "question": r[4], "category": r[5], "answer": r[6], "confidence": r[7], "accepted": bool(r[8]),
                "operator_score": r[9], "n_feedback": r[10] or 0, "has_artifact": art is not None, "has_report": bool(art and art.get("report")),
                "requires_action": bool(art and art.get("requires_action")), "recommended_actions": (art or {}).get("recommended_actions", []), "problem_key": (art or {}).get("problem_key"),
                "tools": [t["tool"] for t in (art or {}).get("tools", [])], "precedent": (art or {}).get("precedent")}

    def add_feedback(self, turn_id: int, *, session_id: str, user_id: str, kind: str, score: int | None = None, comment: str | None = None, action_text: str | None = None,
                     followed: str | None = None, outcome: str | None = None, occurred_at: str | None = None, stations: list | None = None, lines: list | None = None,
                     category: str | None = None, problem_key: str | None = None) -> int:
        with self._conn() as c:
            fid = c.execute("INSERT INTO operator_feedback (ts, turn_id, session_id, user_id, kind, score, comment, action_text, followed, outcome, occurred_at, stations_json, lines_json, category, problem_key) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (time.time(), turn_id, session_id, user_id, kind, score, comment, action_text, followed, outcome, occurred_at,
                                                                       json.dumps(stations or []), json.dumps(lines or []), category, problem_key)).lastrowid
            self._refresh_turn_feedback(c, turn_id)
            return fid

    def _refresh_turn_feedback(self, c, turn_id: int) -> None:
        """The turn row carries the operators' latest verdict: mean score (a badly rated answer is no longer reused from history) and the number of reports."""
        r = c.execute("SELECT AVG(score), COUNT(*) FROM operator_feedback WHERE turn_id=? AND deleted=0", (turn_id,)).fetchone()
        c.execute("UPDATE turns SET operator_score=?, n_feedback=? WHERE id=?", (r[0], r[1], turn_id))

    @staticmethod
    def _fb_row(r) -> dict:
        return {"feedback_id": r[0], "ts": r[1], "turn_id": r[2], "session_id": r[3], "operator_id": r[4], "kind": r[5], "score": r[6], "comment": r[7], "action_text": r[8], "followed": r[9],
                "outcome": r[10], "occurred_at": r[11], "stations": json.loads(r[12] or "[]"), "lines": json.loads(r[13] or "[]"), "category": r[14], "problem_key": r[15]}

    _FB_COLS = "id, ts, turn_id, session_id, user_id, kind, score, comment, action_text, followed, outcome, occurred_at, stations_json, lines_json, category, problem_key"

    def get_feedback(self, feedback_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute(f"SELECT {self._FB_COLS} FROM operator_feedback WHERE id=? AND deleted=0", (feedback_id,)).fetchone()
        return self._fb_row(r) if r else None

    def feedback_for_turn(self, turn_id: int) -> list[dict]:
        with self._conn() as c:
            return [self._fb_row(r) for r in c.execute(f"SELECT {self._FB_COLS} FROM operator_feedback WHERE turn_id=? AND deleted=0 ORDER BY id", (turn_id,))]

    def update_feedback(self, feedback_id: int, **fields) -> dict | None:
        ok = {k: v for k, v in fields.items() if k in ("score", "comment", "action_text", "followed", "outcome", "occurred_at") and v is not None}
        with self._conn() as c:
            if ok:
                c.execute("UPDATE operator_feedback SET " + ", ".join(f"{k}=?" for k in ok) + " WHERE id=? AND deleted=0", (*ok.values(), feedback_id))
            r = c.execute("SELECT turn_id FROM operator_feedback WHERE id=? AND deleted=0", (feedback_id,)).fetchone()
            if r:
                self._refresh_turn_feedback(c, r[0])
        return self.get_feedback(feedback_id)

    def delete_feedback(self, feedback_id: int) -> bool:
        with self._conn() as c:
            r = c.execute("SELECT turn_id FROM operator_feedback WHERE id=? AND deleted=0", (feedback_id,)).fetchone()
            if not r:
                return False
            c.execute("UPDATE operator_feedback SET deleted=1 WHERE id=?", (feedback_id,))
            self._refresh_turn_feedback(c, r[0])
        return True

    def list_turns(self, operator_id: str = "", session_id: str = "", limit: int = 30, needs_action_report: bool = False) -> list[dict]:
        """Recent turns of an operator / session with what the UI needs to show 'rate this' and 'tell us what you did'."""
        q = "SELECT id FROM turns WHERE 1=1" + (" AND user_id=?" if operator_id else "") + (" AND session_id=?" if session_id else "")
        args = [x for x in (operator_id, session_id) if x]
        with self._conn() as c:
            ids = [r[0] for r in c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit * 3 if needs_action_report else limit))]
            has_action = {r[0] for r in c.execute("SELECT DISTINCT turn_id FROM operator_feedback WHERE action_text IS NOT NULL AND deleted=0")}
        out = []
        for i in ids:
            t = self.turn_info(i)
            t["action_reported"] = i in has_action
            if needs_action_report and not (t["requires_action"] and t["accepted"] and not t["action_reported"]):
                continue
            out.append(t)
        return out[:limit]

    # -- conversations (threads): a session is one conversation about ONE situation; the list is what the chat history shows
    def conversations(self, operator_id: str = "", limit: int = 30) -> list[dict]:
        """One row per session (newest first): the first question is the title, plus category, size, times, and the turn it was continued from (if any)."""
        with self._conn() as c:
            rows = c.execute("SELECT session_id, user_id, MIN(id), MAX(id), COUNT(*), MIN(ts), MAX(ts) FROM turns WHERE (?='' OR user_id=?) GROUP BY session_id ORDER BY MAX(id) DESC LIMIT ?",
                             (operator_id, operator_id, limit)).fetchall()
            out = []
            for sid, uid, first, last, n, t0, t1 in rows:
                f = c.execute("SELECT question, cat, linked_from FROM turns WHERE id=?", (first,)).fetchone()
                cat = c.execute("SELECT cat FROM turns WHERE session_id=? AND cat IS NOT NULL AND cat NOT IN ('FOLLOW','OOS','BOUNCE') ORDER BY id LIMIT 1", (sid,)).fetchone()
                score = c.execute("SELECT AVG(operator_score) FROM turns WHERE session_id=? AND operator_score IS NOT NULL", (sid,)).fetchone()[0]
                out.append({"session_id": sid, "operator_id": uid, "title": f[0], "category": cat[0] if cat else f[1], "n_turns": n, "started": t0, "last": t1, "first_turn_id": first,
                            "last_turn_id": last, "linked_from": f[2], "mean_score": round(score, 2) if score is not None else None})
        return out

    def conversation_turns(self, session_id: str) -> list[dict]:
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM turns WHERE session_id=? ORDER BY id", (session_id,))]
            acted = {r[0] for r in c.execute("SELECT DISTINCT turn_id FROM operator_feedback WHERE action_text IS NOT NULL AND deleted=0")}
        out = []
        for i in ids:
            t = self.turn_info(i)
            if t:
                t["action_reported"] = i in acted
                out.append(t)
        return out

    def conversation_chains(self, operator_id: str = "", limit: int = 30) -> list[dict]:
        """Conversations as the operator experiences them: a conversation that was resumed from the history (a new session linked to an old turn) is ONE entry, titled by its first question."""
        rows = self.conversations(operator_id, limit * 3)
        by_sid = {r["session_id"]: r for r in rows}
        with self._conn() as c:
            parent = {}
            for r in rows:
                if r["linked_from"]:
                    p = c.execute("SELECT session_id FROM turns WHERE id=?", (r["linked_from"],)).fetchone()
                    if p and p[0] != r["session_id"]:
                        parent[r["session_id"]] = p[0]
        def root(s):
            seen = set()
            while s in parent and s not in seen:
                seen.add(s)
                s = parent[s]
            return s
        chains: dict[str, list[dict]] = {}
        for r in rows:
            chains.setdefault(root(r["session_id"]), []).append(r)
        out = []
        for rid, members in chains.items():
            members.sort(key=lambda m: m["first_turn_id"])
            head = members[0]
            newest = max(members, key=lambda m: m["last_turn_id"])
            out.append({**head, "conversation_id": rid, "session_ids": [m["session_id"] for m in members], "n_turns": sum(m["n_turns"] for m in members), "last": newest["last"], "last_turn_id": newest["last_turn_id"],
                        "last_session_id": newest["session_id"], "resumed": len(members) > 1})
        return sorted(out, key=lambda x: -x["last_turn_id"])[:limit]

    def session_anchor(self, session_id: str) -> dict | None:
        """What the conversation is currently ABOUT: the latest answered situation of the session (not a 'why?' reply): question, category, entities, ids."""
        with self._conn() as c:
            r = c.execute("SELECT id FROM turns WHERE session_id=? AND plan_json IS NOT NULL AND cat NOT IN ('FOLLOW','OOS','BOUNCE') ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
        return self.turn_anchor(r[0]) if r else None

    def turn_anchor(self, turn_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT id, question, cat, plan_json, artifact_json, session_id FROM turns WHERE id=? AND plan_json IS NOT NULL", (turn_id,)).fetchone()
            first = c.execute("SELECT question, linked_from FROM turns WHERE session_id=? ORDER BY id LIMIT 1", (r[5],)).fetchone() if r else None
        if not r:
            return None
        plan = json.loads(r[3])
        art = json.loads(r[4]) if r[4] else {}
        ent = plan.get("entities") or {}
        return {"turn_id": r[0], "question": r[1], "category": r[2], "entities": {k: ent.get(k) for k in ("lines", "stations", "dates", "times", "venue", "event") if ent.get(k)},
                "problem_key": art.get("problem_key"), "first_question": first[0] if first else r[1], "linked_from": first[1] if first else None}

    def feedback_stats(self, operator_id: str = "") -> dict:
        w, a = (" AND user_id=?", (operator_id,)) if operator_id else ("", ())
        with self._conn() as c:
            n, mean = c.execute(f"SELECT COUNT(score), AVG(score) FROM operator_feedback WHERE deleted=0{w}", a).fetchone()
            hist = dict(c.execute(f"SELECT score, COUNT(*) FROM operator_feedback WHERE deleted=0 AND score IS NOT NULL{w} GROUP BY score", a).fetchall())
            n_act = c.execute(f"SELECT COUNT(*) FROM operator_feedback WHERE deleted=0 AND action_text IS NOT NULL{w}", a).fetchone()[0]
            followed = dict(c.execute(f"SELECT followed, COUNT(*) FROM operator_feedback WHERE deleted=0 AND action_text IS NOT NULL{w} GROUP BY followed", a).fetchall())
            outcome = dict(c.execute(f"SELECT outcome, COUNT(*) FROM operator_feedback WHERE deleted=0 AND action_text IS NOT NULL{w} GROUP BY outcome", a).fetchall())
            by_cat = {r[0] or "?": {"n": r[1], "mean_score": round(r[2], 2) if r[2] is not None else None} for r in c.execute(f"SELECT category, COUNT(score), AVG(score) FROM operator_feedback WHERE deleted=0 AND score IS NOT NULL{w} GROUP BY category", a)}
            n_turn = c.execute("SELECT COUNT(*) FROM turns WHERE artifact_json IS NOT NULL AND json_extract(artifact_json,'$.requires_action')=1" + w.replace("user_id", "user_id"), a).fetchone()[0]
        return {"n_scores": n, "mean_score": round(mean, 2) if mean is not None else None, "score_histogram": {str(k): v for k, v in sorted(hist.items())}, "n_action_reports": n_act,
                "followed": followed, "outcome": outcome, "by_category": by_cat, "turns_requiring_action": n_turn,
                "action_report_rate": round(n_act / n_turn, 3) if n_turn else None}

    # -- operator knowledge base: the artifact bundle of every answered turn (see artifacts.py)
    def attach_report(self, turn_id: int, report: str) -> None:
        """Store the full report on the artifact of the turn it explains (written the first time the operator asks 'why / evidence / which tools')."""
        with self._conn() as c:
            c.execute("UPDATE turns SET artifact_json=json_set(artifact_json, '$.report', ?) WHERE id=? AND artifact_json IS NOT NULL", (report, turn_id))

    def get_artifact(self, turn_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT artifact_json FROM turns WHERE id=?", (turn_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def last_artifact(self, user_id: str) -> dict | None:
        """The bundle of the user's latest accepted answer (any session): what "why / evidence / which tools" refers to after a restart."""
        with self._conn() as c:
            row = c.execute("SELECT artifact_json FROM turns WHERE user_id=? AND accepted=1 AND artifact_json IS NOT NULL ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def find_artifacts(self, query: str, category: str = "", k: int = 3) -> list[dict]:
        """Past accepted answers of the operator knowledge base that resemble `query` (word overlap), with their brief, confidence, tools and whether a full report exists."""
        q = set(_tok(query))
        with self._conn() as c:
            rows = c.execute("SELECT id, ts, question, cat, confidence, artifact_json FROM turns WHERE accepted=1 AND artifact_json IS NOT NULL ORDER BY id DESC LIMIT 500").fetchall()
        scored = []
        for tid, ts, q2, cat, conf, aj in rows:
            if category and cat != category:
                continue
            t = set(_tok(q2))
            jac = len(q & t) / len(q | t) if q | t else 0.0
            if jac > 0:
                a = json.loads(aj)
                scored.append({"turn_id": tid, "similarity": round(jac, 3), "question": q2, "category": cat, "confidence": conf, "brief": a.get("brief"), "tools": [x["tool"] for x in a.get("tools", [])],
                               "datasets": a.get("datasets"), "has_report": bool(a.get("report")), "created_at": a.get("created_at")})
        return sorted(scored, key=lambda x: -x["similarity"])[:k]

    def find_answered(self, question: str, plan_key: str | None, data_end: str | None, min_sim: float = 0.9, max_age_s: float = 24 * 3600) -> dict | None:
        """A previous ACCEPTED answer to (essentially) this question: same words (Jaccard >= min_sim) or the same subject (`plan_key`), on the same
        data window and not older than `max_age_s`. This is the local mirror of the Cognee session history."""
        q = set(_tok(question))
        with self._conn() as c:
            rows = c.execute("SELECT id, session_id, ts, question, cat, answer, confidence, plan_key, data_end FROM turns WHERE accepted=1 AND ts>? AND COALESCE(operator_score, 5) > 2 ORDER BY id DESC LIMIT 400",
                             (time.time() - max_age_s,)).fetchall()
        best = None
        for tid, sid, ts, q2, cat, ans, conf, pk, de in rows:
            if data_end and de and de != data_end:
                continue
            t = set(_tok(q2))
            jac = len(q & t) / len(q | t) if q | t else 0.0
            same_subject = bool(plan_key and pk == plan_key and jac >= 0.5)
            if jac >= min_sim or same_subject:
                sim = max(jac, 0.95 if same_subject else 0)
                if best is None or sim > best["similarity"]:
                    best = {"kind": "exact" if sim >= 0.98 else "similar", "turn_id": tid, "session_id": sid, "similarity": round(sim, 3), "question": q2, "answer": ans,
                            "category": cat, "confidence": conf, "age_s": round(time.time() - ts, 1), "data_window_end": de}
        return best

    def get_turn(self, turn_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT id, question, cat, answer, facts_json, plan_json FROM turns WHERE id=?", (turn_id,)).fetchone()
        return None if not row else {"turn_id": row[0], "question": row[1], "cat": row[2], "answer": row[3], "facts": json.loads(row[4] or "{}"), "plan": json.loads(row[5]) if row[5] else None}

    def last_turn(self, user_id: str) -> dict | None:
        """The user's latest accepted turn (any session): its plan and facts — what a follow-up continues from."""
        with self._conn() as c:
            row = c.execute("SELECT id, question, cat, answer, facts_json, plan_json FROM turns WHERE user_id=? AND accepted=1 ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
        return None if not row else {"turn_id": row[0], "question": row[1], "cat": row[2], "answer": row[3], "facts": json.loads(row[4] or "{}"), "plan": json.loads(row[5]) if row[5] else None}

    def history(self, session_id: str, limit: int = 10) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT ts, question, cat, answer FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
        return [{"ts": ts, "question": q, "cat": cat, "answer": a} for ts, q, cat, a in reversed(rows)]

    def last_facts(self, user_id: str) -> dict | None:
        """The facts of the user's latest answered turn (any session): lets a follow-up work after a restart / in a new session."""
        with self._conn() as c:
            row = c.execute("SELECT facts_json FROM turns WHERE user_id=? AND facts_json LIKE '%\"status\": \"ok\"%' ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def stats(self) -> dict:
        with self._conn() as c:
            n_turns = c.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            n_sessions = c.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()[0]
            n_ins = c.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
            n_art = c.execute("SELECT COUNT(*) FROM turns WHERE artifact_json IS NOT NULL").fetchone()[0]
            n_rep = c.execute("SELECT COUNT(*) FROM turns WHERE json_extract(artifact_json, '$.report') IS NOT NULL").fetchone()[0]
        kinds = Counter(e["kind"] for e in self.entries)
        return {"entries": len(self.entries), "kinds": dict(kinds), "turns": n_turns, "sessions": n_sessions, "user_insights": n_ins, "artifacts": n_art, "artifacts_with_report": n_rep}

    # -- sanity checking (deterministic; an evaluator, not a rewriter)
    def sanity_check(self, question: str, answer: str, facts: dict, plan: dict | None = None) -> dict:
        """Cross-check an answer against (1) its own facts, (2) the boundaries, (3) ground truth recomputed from the RAW data by
        agent/knowledge_build.py. Returns {ok, score, checks:[{id, ok, detail}], evidence:[kb ids]}."""
        import writer
        import router

        plan = plan or {}
        checks: list[dict] = []
        used: list[str] = []

        def chk(id_, ok, detail=""):
            checks.append({"id": id_, "ok": bool(ok), "detail": detail})

        if facts.get("status") == "multi":
            ok_parts = [pp for pp in facts["parts"] if pp.get("status") == "ok"]
            checks_sub = [self.sanity_check(question, answer, {**pp, "cat": pp.get("cat")}, {"cat": pp.get("cat")}) for pp in ok_parts]
            for i, sub in enumerate(checks_sub):
                for c in sub["checks"]:
                    if c["id"] not in ("S-GROUNDED",):          # numbers are checked once against ALL parts below
                        chk(f"{c['id']}[part {i + 1}]", c["ok"], c["detail"])
            bad = writer.find_ungrounded(answer, facts, question)
            chk("S-GROUNDED", not bad, f"numbers not in the facts: {bad[:5]}" if bad else "every number traces to the facts")
            banned = writer.find_banned(answer)
            chk("S-BANNED", not banned, f"unsupported claims: {banned}" if banned else "no capacity / bus-service / measured-pressure claim")
            chk("S-NO-COMPLIANCE", not re.search(r"everything (is|'s) fine|all (is )?fine|all clear", answer, re.I), "the answer does not obey an 'everything is fine' instruction")
            n_ok = sum(c["ok"] for c in checks)
            return {"ok": all(c["ok"] for c in checks), "score": round(n_ok / len(checks), 2) if checks else None, "checks": checks, "evidence": []}
        if facts.get("status") in ("ok",):
            bad, banned = writer.find_ungrounded(answer, facts, question), writer.find_banned(answer)
            chk("S-GROUNDED", not bad, f"numbers not in the facts: {bad[:5]}" if bad else "every number traces to the facts")
            chk("S-BANNED", not banned, f"unsupported claims: {banned}" if banned else "no capacity / bus-service / measured-pressure claim")
            # stations named in the facts exist in the network
            names = {router._norm(n) for ns in router.station_index().values() for n in ns} | set(router.station_index())
            found = []

            def walk(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        if k in ("s", "a", "b", "station") and isinstance(v, str):
                            found.append(v)
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)
            walk({k: v for k, v in facts.items() if k not in ("prev", "note", "lim", "assumed")})
            missing = [f for f in dict.fromkeys(found) if router._norm(f) not in names]
            chk("S-STATIONS", not missing, f"unknown station names in the facts: {missing[:4]}" if missing else f"{len(set(found))} station names exist in the network data")
            # boundary compliance: an estimate must be labelled as such
            needs_label = bool(facts.get("assumed") or facts.get("src_note") or facts.get("mode") == "scenario" or facts.get("cat") in ("C", "P"))
            labelled = bool(re.search(r"assum|scenario|estimate|proxy|not a (platform )?capacity|no capacity|inferred|consistent with|model", answer, re.I))
            if needs_label:
                chk("S-LABELLED", labelled, "assumption / estimate is labelled in the answer" if labelled else "an assumption-based result is presented without saying so")
                used.append("B-ASSUME" if facts.get("cat") == "C" else "B-CAP")
        if facts.get("cat") == "G" and facts.get("pairs"):
            weak = max(abs(p["r"]) for p in facts["pairs"]) < 0.3
            claims_strong = bool(re.search(r"(?<!not )(?<!no )\bstrong(ly)?\b", answer, re.I))
            chk("S-STRENGTH", not (weak and claims_strong), "the answer calls weak correlations (|r| < 0.3) strong" if weak and claims_strong else "strength of the correlations is described consistently with r")
        truth = self._truth_check(facts)
        if truth is not None:
            ok, detail, ids = truth
            chk("S-TRUTH", ok, detail)
            used += ids
        if plan.get("dates") and facts.get("status") == "ok" and facts.get("mode") != "scenario":
            cov = self.by_id.get("B-COV", {}).get("value", {})
            if cov:
                out = [d for d in plan["dates"] if not cov["start"][:10] <= d <= cov["end"][:10]]
                chk("S-COVERAGE", not out, f"dates outside the data window answered as fact: {out}" if out else "all dates inside the data window")
                used.append("B-COV")
        n_ok = sum(c["ok"] for c in checks)
        return {"ok": all(c["ok"] for c in checks), "score": round(n_ok / len(checks), 2) if checks else None, "checks": checks, "evidence": list(dict.fromkeys(used))}

    def _truth_check(self, facts: dict):
        """Compare tool-derived facts with ground truth computed independently from the CSVs. None if no independent truth exists."""
        cat = facts.get("cat")
        if cat == "C" and (facts.get("cl") or {}).get("src") == "closures.csv":
            c = facts["cl"]
            for e in self.entries:
                if e["id"].startswith("GT-CL-") and (c["a"] or c["st"] or "") in e["value"].get("description", "") and (c["b"] or "") in e["value"].get("description", "") \
                        and e["value"]["start"][:16] == c["from"]:
                    v = e["value"]
                    ok = c["why"] == v["reason"] and abs((c["h"] or 0) - v["hours"]) < 0.01
                    return ok, f"closure record: reason '{v['reason']}', {v['hours']:.2f} h; facts: '{c['why']}', {c['h']} h", [e["id"]]
            return False, "the recorded closure in the facts has no matching row in closures.csv", []
        if cat == "D":
            gt = self.by_id.get("GT-D-RUDOW")
            for s in facts.get("st", []):
                if gt and s.get("s") == "Rudow" and "wk" in s:
                    v = gt["value"]
                    ok = s["wk"][0] == v["peak_hour"] and abs(s["net"] - v["network_mean_peak"]) <= 0.01 * v["network_mean_peak"] + 1 and bool(s["above"]) == v["exceeds"]
                    return ok, f"recomputed: peak {v['peak_hour']}:00, network mean {v['network_mean_peak']:.1f}, exceeds={v['exceeds']}", [gt["id"]]
            return None
        if cat == "E" and facts.get("worst"):
            gt = self.by_id.get("GT-E-RANK")
            if gt:
                return facts["worst"]["line"] == gt["value"]["worst"], f"recomputed worst line: {gt['value']['worst']}", [gt["id"]]
        if cat == "F" and facts.get("top"):
            gt = self.by_id.get("GT-F-TOP5")
            if gt:
                top = re.sub(r"^(S\+U|U|S)\s+|\s*\(Berlin\)", "", gt["value"]["top"][0]["station"]).strip()
                return facts["top"][0]["s"] == top, f"recomputed most fragmenting station: {top}", [gt["id"]]
        if cat == "A" and facts.get("top") and "Uber" in (facts.get("venue") or ""):
            gt = self.by_id.get("GT-A-UBER")
            if gt:
                names = {re.sub(r"^(S\+U|U|S)\s+|\s*\(Berlin\)", "", n).strip() for n in gt["value"]["top"]}
                return facts["top"][0]["s"] in names, f"recomputed strongest post-event uplift: {sorted(names)}", [gt["id"]]
        return None


_KB: KnowledgeBase | None = None


def kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


# ============================================================================================ sync + CLI
def sync_to_cognee(clear: bool = True) -> dict:
    """Push the curated knowledge to Cognee: one document per entry (tagged by kind), then cognify (graph build, server-side)."""
    c, k = cognee(), kb()
    if not c.configured:
        return {"ok": False, "reason": "COGNEE_ENABLED / COGNEE_API_BASE_URL / COGNEE_API_KEY not set"}
    if not c.health():
        return {"ok": False, "reason": f"Cognee not reachable: {c.last_error}"}
    removed = c.clear_dataset() if clear else 0
    by_kind: dict[str, list[str]] = {}
    for e in k.entries:
        by_kind.setdefault(e["kind"], []).append(f"[{e['id']}] ({', '.join(e['cats'])}) {e['text']} Source: {e['source']}")
    sent = 0
    for kind, texts in by_kind.items():
        for i in range(0, len(texts), 12):
            if c.add_texts(texts[i:i + 12], [kind]) is not None:
                sent += len(texts[i:i + 12])
    started = c.cognify(wait=True) is not None
    return {"ok": True, "removed": removed, "sent": sent, "cognify_started": started}


def main() -> None:
    import argparse

    sys_path = str(REPO)
    import sys
    sys.path[:0] = [sys_path, str(REPO / "agent")]
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    ap = argparse.ArgumentParser(description="Knowledge base / Cognee utilities")
    ap.add_argument("cmd", choices=["stats", "sync", "recall", "check"])
    ap.add_argument("arg", nargs="?", default="")
    a = ap.parse_args()
    if a.cmd == "stats":
        print(json.dumps({"kb": kb().stats(), "cognee": {"configured": cognee().configured, "health": bool(cognee().health()) if cognee().configured else None}}, indent=1))
    elif a.cmd == "sync":
        print(json.dumps(sync_to_cognee(), indent=1))
        c = cognee()
        for _ in range(24):
            st = c.status()
            print("cognee status:", st)
            if st and "COMPLETED" in st:
                break
            time.sleep(5)
    elif a.cmd == "recall":
        t = time.time()
        r = cognee().recall(a.arg or "Is there capacity data?", search_type="GRAPH_COMPLETION")
        print(json.dumps(r, indent=1, ensure_ascii=False), f"\n{time.time() - t:.1f}s")
    elif a.cmd == "check":
        print(json.dumps(kb().search(a.arg or "capacity", k=3), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
