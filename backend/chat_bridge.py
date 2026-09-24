"""Chat bridge: the UI talks to ONE endpoint (`POST /api/v1/chat`); this module runs the question through the ADK agent server and turns its event stream into what a
desktop needs — the answer, the ids for feedback, and the *operations log* (steps, tool calls with arguments and time, LLM calls, tokens, timings).

Token counts are the providers' usage numbers when the call reported them, else a character-based estimate (≈ 4 characters per token) flagged as `estimated`.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def adk_url() -> str:
    if os.environ.get("ADK_URL"):
        return os.environ["ADK_URL"].rstrip("/")
    try:
        st = json.loads((REPO / ".run" / "services.json").read_text())
        return f"http://127.0.0.1:{st['adk-web']['port']}"
    except Exception:
        return "http://127.0.0.1:8000"


def _post(path: str, body: dict, timeout: float = 240):
    req = urllib.request.Request(adk_url() + path, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read() or b"null")


class AgentUnavailable(RuntimeError):
    pass


def ask(operator_id: str, session_id: str, message: str, app_name: str = "agent", link_turn_id: int | None = None) -> dict:
    """Run one user message. Creates the ADK session on first use; `link_turn_id` starts it CONNECTED to an earlier answer (the situation of a conversation the operator resumes)."""
    t0 = time.time()
    try:
        try:
            _post(f"/apps/{app_name}/users/{operator_id}/sessions/{session_id}", {"state": {"link_turn_id": int(link_turn_id)}} if link_turn_id else {}, 20)
        except urllib.error.HTTPError as e:            # 400/409 = the session exists already
            if e.code not in (400, 409, 422):
                raise
        events = _post("/run", {"app_name": app_name, "user_id": operator_id, "session_id": session_id, "new_message": {"role": "user", "parts": [{"text": message}]}})
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise AgentUnavailable(f"the agent server at {adk_url()} is not reachable ({type(e).__name__}); start it with `make up`")
    out = parse_events(events)
    out["session_id"] = session_id
    out["operator_id"] = operator_id
    out["wall_s"] = round(time.time() - t0, 2)
    return out


def _text(e: dict) -> str:
    parts = (e.get("content") or {}).get("parts") or []
    return "".join(p.get("text") or "" for p in parts if isinstance(p, dict))


class EventParser:
    """Incremental ADK event → operations-log parser. `feed(event)` returns the steps (dicts) that event completed; `result()` gives the full response once the stream ended."""

    def __init__(self) -> None:
        self.steps, self.tools, self.llm, self.timing, self.meta, self.answer, self.n_events = [], [], [], {}, {}, "", 0
        self.pending: dict[str, dict] = {}
        self.texts: list[str] = []

    def feed(self, e: dict) -> list[dict]:
        self.n_events += 1
        new: list[dict] = []
        cm = e.get("customMetadata") or {}
        kind = cm.get("kind")
        parts = (e.get("content") or {}).get("parts") or []
        fc = next((p["functionCall"] for p in parts if isinstance(p, dict) and p.get("functionCall")), None)
        fr = next((p["functionResponse"] for p in parts if isinstance(p, dict) and p.get("functionResponse")), None)
        if _text(e):
            self.texts.append(_text(e))
        if fc:
            self.pending[fc.get("id") or fc["name"]] = {"tool": fc["name"], "args": fc.get("args") or {}, "server": cm.get("server"), "round": cm.get("round"), "started_at_ms": cm.get("started_at_ms")}
        elif fr:
            r = fr.get("response") or {}
            base = self.pending.pop(fr.get("id") or fr["name"], {"tool": fr["name"], "args": {}})
            t = {**base, "server": r.get("server") or base.get("server"), "seconds": r.get("seconds"), "bytes": r.get("result_bytes"), "ok": r.get("ok", True), "wait_ready_ms": r.get("wait_ready_ms"),
                 "result_preview": (r.get("result_preview") or "")[:400]}
            self.tools.append(t)
            new.append({"kind": "tool", "stage": "analyst", "label": f"{t['tool']}", "seconds": t["seconds"], "detail": json.dumps(t["args"], ensure_ascii=False)[:160], "round": t.get("round")})
        elif kind == "llm":
            r = {"role": cm.get("role"), "model": cm.get("model"), "seconds": cm.get("inference_s"), "tok_in": cm.get("tok_in"), "tok_out": cm.get("tok_out"), "error": cm.get("error"),
                 "prompt_chars": len(cm.get("prompt") or ""), "response_chars": len(cm.get("response") or "")}
            self.llm.append(r)
            new.append({"kind": "llm", "stage": "writer" if r["role"] == "writer" else ("inspector" if r["role"] == "evaluator" else "dispatcher"), "label": f"{r['role']} · {r['model']}", "seconds": r["seconds"],
                        "detail": f"{r['tok_in']}→{r['tok_out']} tokens"})
        elif kind == "supervisor":
            route = cm.get("route") or {}
            new.append({"kind": "route", "stage": "dispatcher", "label": f"Dispatcher: {cm.get('decision')} · category {cm.get('category')}", "seconds": cm.get("seconds"),
                        "detail": f"{route.get('specialist')} · engine {route.get('ml_engine')} · tools {', '.join(route.get('tools') or [])}"})
        elif kind == "worker_round":
            new.append({"kind": "worker", "stage": "analyst", "label": f"Analyst round {cm.get('round')}", "seconds": cm.get("seconds"), "detail": f"tools {', '.join(cm.get('tools') or [])} · confidence {cm.get('confidence')}"})
        elif kind == "evaluator":
            new.append({"kind": "evaluator", "stage": "inspector", "label": f"Inspector round {cm.get('round')}: {cm.get('verdict')}", "seconds": cm.get("seconds"),
                        "detail": f"score {cm.get('score')} · {cm.get('model')}" + (f" · issues: {'; '.join(cm['issues'])}" if cm.get("issues") else "")})
        elif kind == "timing":
            self.timing = {k: cm.get(k) for k in ("supervisor_s", "worker_evaluator_s", "mcp_s", "writer_s", "writer_llm_s", "total_s")}
        elif kind == "answer":
            self.answer, self.meta = _text(e), cm
            new.append({"kind": "writer", "stage": "writer", "label": "Writer", "seconds": self.timing.get("writer_s"), "detail": f"{len(self.answer.split())} words · {cm.get('answer_mode') or cm.get('source')}"})
        self.steps += new
        return new

    def result(self) -> dict:
        answer = self.answer or (self.texts[-1] if self.texts else "")
        tin, tout, estimated = sum(r["tok_in"] or 0 for r in self.llm), sum(r["tok_out"] or 0 for r in self.llm), False
        if not (tin or tout) and self.llm:                                  # calls without usage: ≈ 4 characters per token
            tin, tout, estimated = sum(r["prompt_chars"] for r in self.llm) // 4, sum(r["response_chars"] for r in self.llm) // 4, True
        m = self.meta
        return {"answer": answer, "turn_id": m.get("turn_id"), "artifact_turn_id": m.get("artifact_turn_id"), "requires_action": bool(m.get("requires_action")), "precedent": m.get("precedent"),
                "source": m.get("source"), "answer_mode": m.get("answer_mode"), "guard": m.get("guard"), "steps": self.steps, "tools": self.tools, "llm": self.llm,
                "tokens": {"in": tin, "out": tout, "total": tin + tout, "estimated": estimated}, "timing": self.timing,
                "counts": {"tool_calls": len(self.tools), "distinct_tools": len({t["tool"] for t in self.tools}), "llm_calls": len(self.llm), "events": self.n_events}}


def parse_events(events: list[dict]) -> dict:
    """ADK events -> {answer, meta, steps[], tools[], llm[], tokens, timing}."""
    p = EventParser()
    for e in events:
        p.feed(e)
    return p.result()


def ask_stream(operator_id: str, session_id: str, message: str, app_name: str = "agent", link_turn_id: int | None = None):
    """Generator of (event_name, data) tuples for Server-Sent Events: `step` for every operation as the agent emits it, then `answer` (the full response) — or `error`.
    Uses the ADK server's /run_sse, so steps arrive while the agent is still working."""
    t0 = time.time()
    try:
        try:
            _post(f"/apps/{app_name}/users/{operator_id}/sessions/{session_id}", {"state": {"link_turn_id": int(link_turn_id)}} if link_turn_id else {}, 20)
        except urllib.error.HTTPError as e:
            if e.code not in (400, 409, 422):
                raise
        body = json.dumps({"app_name": app_name, "user_id": operator_id, "session_id": session_id, "new_message": {"role": "user", "parts": [{"text": message}]}, "streaming": False}).encode()
        req = urllib.request.Request(adk_url() + "/run_sse", body, {"content-type": "application/json", "accept": "text/event-stream"})
        parser = EventParser()
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                for step in parser.feed(ev):
                    yield "step", step
        out = parser.result()
        out.update(session_id=session_id, operator_id=operator_id, wall_s=round(time.time() - t0, 2))
        yield "answer", out
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        yield "error", {"message": f"the agent server at {adk_url()} is not reachable ({type(e).__name__}); start it with `make up`"}
    except Exception as e:
        yield "error", {"message": f"{type(e).__name__}: {str(e)[:200]}"}
