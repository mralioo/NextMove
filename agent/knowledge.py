"""Knowledge base + long-term memory for the main agent (local-first, mirrored to Cognee).

    KnowledgeBase   curated ground truth / boundaries / insights (knowledge/knowledge.json, built from the raw data by
                    agent/knowledge_build.py) with deterministic retrieval, a SANITY CHECKER, and a per-session turn history.
    CogneeClient    the Cognee Cloud REST API (https://docs.cognee.ai): knowledge graph + session memory. Optional and OFF the hot
                    path: measured recall latency is seconds, so it is used for (a) writing turns/insights in the background,
                    (b) semantic recall for evaluators and follow-ups, never as a blocking step in front of the operator.

Why local-first: an operator answer must not wait for a network memory service, and the boundaries ("no capacity data",
"data ends 2026-09-22") must be enforced even when the service is down. The same content is pushed to Cognee (`make kb-sync`)
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
            for col, typ in (("verdict", "TEXT"), ("confidence", "REAL"), ("plan_key", "TEXT"), ("data_end", "TEXT"), ("accepted", "INTEGER"), ("plan_json", "TEXT")):
                try:                                       # turn store created before v3: add the columns the supervisor's history lookup needs
                    c.execute(f"ALTER TABLE turns ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass
            c.execute("CREATE TABLE IF NOT EXISTS insights (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, text TEXT, cats TEXT, source TEXT)")

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
                      accepted: bool | None = None, plan: dict | None = None) -> int:
        with self._conn() as c:
            return c.execute("INSERT INTO turns (session_id, user_id, ts, question, cat, answer, facts_json, sanity_json, verdict, confidence, plan_key, data_end, accepted, plan_json) "
                             "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (session_id, user_id, time.time(), question, cat, answer, json.dumps(facts or {}, default=str), json.dumps(sanity or {}, default=str), verdict, confidence,
                              plan_key, data_end, None if accepted is None else int(accepted), json.dumps(plan, default=str) if plan else None)).lastrowid

    def find_answered(self, question: str, plan_key: str | None, data_end: str | None, min_sim: float = 0.9, max_age_s: float = 24 * 3600) -> dict | None:
        """A previous ACCEPTED answer to (essentially) this question: same words (Jaccard >= min_sim) or the same subject (`plan_key`), on the same
        data window and not older than `max_age_s`. This is the local mirror of the Cognee session history."""
        q = set(_tok(question))
        with self._conn() as c:
            rows = c.execute("SELECT id, session_id, ts, question, cat, answer, confidence, plan_key, data_end FROM turns WHERE accepted=1 AND ts>? ORDER BY id DESC LIMIT 400",
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
        kinds = Counter(e["kind"] for e in self.entries)
        return {"entries": len(self.entries), "kinds": dict(kinds), "turns": n_turns, "sessions": n_sessions, "user_insights": n_ins}

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
