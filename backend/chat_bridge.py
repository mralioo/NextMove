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


def parse_events(events: list[dict]) -> dict:
    """ADK events -> {answer, meta, steps[], tools[], llm[], tokens, timing}."""
    steps, tools, llm, answer, meta, timing = [], [], [], "", {}, {}
    pending: dict[str, dict] = {}
    for e in events:
        cm = e.get("customMetadata") or {}
        kind = cm.get("kind")
        parts = (e.get("content") or {}).get("parts") or []
        fc = next((p["functionCall"] for p in parts if isinstance(p, dict) and p.get("functionCall")), None)
        fr = next((p["functionResponse"] for p in parts if isinstance(p, dict) and p.get("functionResponse")), None)
        if fc:
            pending[fc.get("id") or fc["name"]] = {"tool": fc["name"], "args": fc.get("args") or {}, "server": cm.get("server"), "round": cm.get("round"), "started_at_ms": cm.get("started_at_ms")}
        elif fr:
            r = fr.get("response") or {}
            base = pending.pop(fr.get("id") or fr["name"], {"tool": fr["name"], "args": {}})
            t = {**base, "server": r.get("server") or base.get("server"), "seconds": r.get("seconds"), "bytes": r.get("result_bytes"), "ok": r.get("ok", True), "wait_ready_ms": r.get("wait_ready_ms"),
                 "result_preview": (r.get("result_preview") or "")[:400]}
            tools.append(t)
            steps.append({"kind": "tool", "label": f"{t['tool']}", "seconds": t["seconds"], "detail": json.dumps(t["args"], ensure_ascii=False)[:160], "round": t.get("round")})
        elif kind == "llm":
            r = {"role": cm.get("role"), "model": cm.get("model"), "seconds": cm.get("inference_s"), "tok_in": cm.get("tok_in"), "tok_out": cm.get("tok_out"), "error": cm.get("error"),
                 "prompt_chars": len(cm.get("prompt") or ""), "response_chars": len(cm.get("response") or "")}
            llm.append(r)
            steps.append({"kind": "llm", "label": f"{r['role']} · {r['model']}", "seconds": r["seconds"], "detail": f"{r['tok_in']}→{r['tok_out']} tokens"})
        elif kind == "supervisor":
            route = cm.get("route") or {}
            steps.append({"kind": "route", "label": f"Dispatcher: {cm.get('decision')} · category {cm.get('category')}", "seconds": cm.get("seconds"),
                          "detail": f"{route.get('specialist')} · engine {route.get('ml_engine')} · tools {', '.join(route.get('tools') or [])}"})
        elif kind == "worker_round":
            steps.append({"kind": "worker", "label": f"Analyst round {cm.get('round')}", "seconds": cm.get("seconds"), "detail": f"tools {', '.join(cm.get('tools') or [])} · confidence {cm.get('confidence')}"})
        elif kind == "evaluator":
            steps.append({"kind": "evaluator", "label": f"Inspector round {cm.get('round')}: {cm.get('verdict')}", "seconds": cm.get("seconds"),
                          "detail": f"score {cm.get('score')} · {cm.get('model')}" + (f" · issues: {'; '.join(cm['issues'])}" if cm.get("issues") else "")})
        elif kind == "timing":
            timing = {k: cm.get(k) for k in ("supervisor_s", "worker_evaluator_s", "mcp_s", "writer_s", "writer_llm_s", "total_s")}
        elif kind == "answer":
            answer, meta = _text(e), cm
    if not answer:                                                    # a stream without a writer answer (error): last text event
        texts = [_text(e) for e in events if _text(e)]
        answer = texts[-1] if texts else ""
    tin = sum(r["tok_in"] or 0 for r in llm)
    tout = sum(r["tok_out"] or 0 for r in llm)
    estimated = False
    if not (tin or tout) and llm:                                      # calls without usage: ≈ 4 characters per token
        tin, tout, estimated = sum(r["prompt_chars"] for r in llm) // 4, sum(r["response_chars"] for r in llm) // 4, True
    steps.append({"kind": "writer", "label": "Writer", "seconds": timing.get("writer_s"), "detail": f"{len(answer.split())} words · {meta.get('answer_mode') or meta.get('source')}"}) if answer else None
    return {
        "answer": answer,
        "turn_id": meta.get("turn_id"), "artifact_turn_id": meta.get("artifact_turn_id"), "requires_action": bool(meta.get("requires_action")), "precedent": meta.get("precedent"),
        "source": meta.get("source"), "answer_mode": meta.get("answer_mode"), "guard": meta.get("guard"),
        "steps": steps, "tools": tools, "llm": llm, "tokens": {"in": tin, "out": tout, "total": tin + tout, "estimated": estimated}, "timing": timing,
        "counts": {"tool_calls": len(tools), "distinct_tools": len({t["tool"] for t in tools}), "llm_calls": len(llm), "events": len(events)},
    }
