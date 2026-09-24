"""Observability + traceability for the agent workflow, built on ADK's own telemetry.

Three layers, all landing in ONE SQLite file (`observability/agent_obs.db`, override with OBS_DB) that the
dashboard reads:

1. **Traces (OpenTelemetry spans)** — ADK already emits `invocation` and `invoke_agent <stage>` spans (and
   `call_llm` / `execute_tool` in the LLM-loop mode). `init_tracing()` attaches ADK's own
   `SqliteSpanExporter` to the global tracer provider (creating one if `adk web` hasn't), so every span of
   every run — CLI, `adk web`, bench, eval — is stored in the `spans` table. We add manual spans for what ADK
   cannot see: the direct LiteLLM calls (`llm.route`, `llm.write`, with model + token counts), every MCP tool
   call (`mcp.tool <name>`) and the number guard (`guard.check`).
2. **Runs (speed + outcome per question)** — `ObservabilityPlugin` (an ADK plugin, so it works with any Runner
   and with `adk web`) writes one row per question into `runs`: question, answer, category/confidence/router
   tier, per-stage timing (route / tools / write), model + tokens, LLM & tool call counts, guard result,
   status/error, an event timeline, and the `trace_id` that links to the spans.
3. **Evaluation** — `eval_runs` / `eval_items` hold scored evaluation runs (see evaluation/run_eval.py); each
   item links to the `runs` row and trace that produced it.
4. **Experiments** — `exp_runs` / `exp_turns` hold the component-comparison study (see experiments/).

Everything is best-effort: an observability failure never breaks or slows an answer (span export is batched
on a background thread; DB writes are a few milliseconds).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("OBS_DB", REPO_ROOT / "observability" / "agent_obs.db"))
TRACER_NAME = "talk_to_my_train"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, session_id TEXT, trace_id TEXT, ts REAL, mode TEXT, source TEXT,
  question TEXT, answer TEXT, status TEXT, error TEXT,
  category TEXT, conf REAL, tier INTEGER, facts_status TEXT, guard TEXT,
  total_s REAL, route_ms REAL, tools_s REAL, write_s REAL,
  model TEXT, tok_in INTEGER, tok_out INTEGER, n_llm_calls INTEGER, n_tool_calls INTEGER,
  events_json TEXT, plan_json TEXT, facts_json TEXT, timing_json TEXT
);
CREATE INDEX IF NOT EXISTS runs_ts ON runs(ts);
CREATE INDEX IF NOT EXISTS runs_session ON runs(session_id);
CREATE TABLE IF NOT EXISTS eval_runs (
  eval_id TEXT PRIMARY KEY, ts REAL, suite TEXT, mode TEXT, n_items INTEGER, budget_s REAL,
  git_commit TEXT, summary_json TEXT
);
CREATE TABLE IF NOT EXISTS exp_runs (
  exp_id TEXT, arm_id TEXT, arm_name TEXT, label TEXT, factor TEXT, params_json TEXT, setup_json TEXT,
  ts REAL, status TEXT, note TEXT, summary_json TEXT, PRIMARY KEY (exp_id, arm_id)
);
CREATE TABLE IF NOT EXISTS exp_turns (
  exp_id TEXT, arm_id TEXT, turn TEXT, question TEXT, answer TEXT, run_id TEXT, trace_id TEXT,
  metrics_json TEXT, checks_json TEXT, facts_json TEXT, judge_json TEXT, PRIMARY KEY (exp_id, arm_id, turn)
);
CREATE TABLE IF NOT EXISTS eval_items (
  eval_id TEXT, item_id TEXT, repeat_idx INTEGER, stage TEXT, category TEXT, question TEXT,
  run_id TEXT, trace_id TEXT, answer TEXT, latency_s REAL, metrics_json TEXT, checks_json TEXT, judge_json TEXT,
  PRIMARY KEY (eval_id, item_id, repeat_idx)
);
CREATE TABLE IF NOT EXISTS ls_runs (
  ls_id TEXT PRIMARY KEY, ts REAL, eval_id TEXT, suite TEXT, judge_model TEXT, n_items INTEGER, uploaded INTEGER, summary_json TEXT
);
CREATE TABLE IF NOT EXISTS ls_feedback (
  ls_id TEXT, item_id TEXT, key TEXT, score REAL, comment TEXT, PRIMARY KEY (ls_id, item_id, key)
);
"""

_db_lock = threading.Lock()
_tracing_ready = False


# ------------------------------------------------------------------------------------------ database
def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")       # the dashboard can read while the agent writes
    conn.executescript(SCHEMA)
    try:                                           # databases created before the LLM judge existed lack this column
        conn.execute("ALTER TABLE eval_items ADD COLUMN judge_json TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    return conn


def _write(sql: str, params: tuple) -> None:
    try:
        with _db_lock:
            conn = connect()
            conn.execute(sql, params)
            conn.commit()
            conn.close()
    except Exception:
        pass   # observability must never break an answer


# ------------------------------------------------------------------------------------------ tracing
def init_tracing() -> None:
    """Attach ADK's SQLite span exporter to the global tracer provider (idempotent)."""
    global _tracing_ready
    if _tracing_ready:
        return
    _tracing_ready = True
    try:
        from google.adk.telemetry.sqlite_span_exporter import SqliteSpanExporter
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        connect().close()                                    # create runs/eval tables up front
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):          # nobody (e.g. `adk web`) set one yet
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
        provider.add_span_processor(BatchSpanProcessor(
            SqliteSpanExporter(db_path=str(DB_PATH)), schedule_delay_millis=500, max_export_batch_size=64))
    except Exception:
        pass


def flush_traces(timeout_ms: int = 5000) -> None:
    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_ms)
    except Exception:
        pass


@contextmanager
def span(name: str, **attrs):
    """A manual span (no-op-safe). Attribute values are coerced to OTel-legal types."""
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer(TRACER_NAME)
    except Exception:
        yield None
        return
    with tracer.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            set_attr(sp, k, v)
        yield sp


def payload(obj, limit: int = 6000) -> str:
    """Compact JSON of a hand-over payload for a span attribute, cut at `limit` characters (the size is kept: `… [+N chars]`)."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + f"… [+{len(text) - limit} chars]"


import contextvars

CALL_LOG: contextvars.ContextVar = contextvars.ContextVar("tmt_call_log", default=None)
"""When set to a list, every MCP call made in this context (and in tasks created from it) appends one record: tool, server, args, result preview, timings."""


LLM_LOG: contextvars.ContextVar = contextvars.ContextVar("tmt_llm_log", default=None)
"""When set to a list, every LLM call (router, evaluator, writer) made in this context appends one record: role, model, inference seconds, tokens, prompt / response."""


def log_llm(role: str, model: str, seconds: float, usage=None, prompt: str = "", response: str = "", error: str | None = None, started: float | None = None) -> dict:
    """Record one LLM inference (the time is the whole call: network + generation). Returned so callers can also put it on their span."""
    rec = {"role": role, "model": model, "seconds": round(seconds, 3), "tok_in": getattr(usage, "prompt_tokens", None), "tok_out": getattr(usage, "completion_tokens", None),
           "prompt": payload(prompt, 1500), "response": payload(response, 1500), "error": error, "start": started}
    log = LLM_LOG.get()
    if log is not None:
        log.append(rec)
    return rec


def set_attr(sp, key: str, value) -> None:
    if sp is None or value is None:
        return
    if not isinstance(value, (str, bool, int, float)):
        value = payload(value, 6000)
    elif isinstance(value, str) and len(value) > 8000:
        value = payload(value, 8000)
    try:
        sp.set_attribute(key, value)
    except Exception:
        pass


def current_trace_id() -> str | None:
    try:
        from opentelemetry import trace
        ctx = trace.get_current_span().get_span_context()
        return format(ctx.trace_id, "032x") if ctx and ctx.is_valid else None
    except Exception:
        return None


# ------------------------------------------------------------------------------------------ run recording
def _plugin_class():
    from google.adk.plugins import BasePlugin

    class ObservabilityPlugin(BasePlugin):
        """ADK plugin: one `runs` row per question (works for every Runner and for `adk web`)."""

        def __init__(self, source: str | None = None, name: str = "observability") -> None:
            super().__init__(name=name)
            self.source = source
            self._live: dict[str, dict] = {}

        async def before_run_callback(self, *, invocation_context):
            self._live[invocation_context.invocation_id] = {
                "t0": time.time(), "events": [], "answer": "", "trace_id": current_trace_id(),
                "question": _question(invocation_context)}
            return None

        async def on_event_callback(self, *, invocation_context, event):
            live = self._live.get(invocation_context.invocation_id)
            if live is None:
                return None
            live["trace_id"] = live["trace_id"] or current_trace_id()
            parts = event.content.parts if event.content and event.content.parts else []
            calls = [p.function_call.name for p in parts if getattr(p, "function_call", None)]
            resps = [p.function_response.name for p in parts if getattr(p, "function_response", None)]
            text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
            u = getattr(event, "usage_metadata", None)
            live["events"].append({
                "t_ms": round((time.time() - live["t0"]) * 1000), "author": event.author,
                "calls": calls, "responses": resps, "text_chars": len(text),
                "tok_in": getattr(u, "prompt_token_count", None) if u else None,
                "tok_out": getattr(u, "candidates_token_count", None) if u else None})
            if text and event.is_final_response():
                live["answer"] = text
            return None

        async def after_run_callback(self, *, invocation_context):
            self._finish(invocation_context, error=None)

        async def on_run_error_callback(self, *, invocation_context, error):
            self._finish(invocation_context, error=f"{type(error).__name__}: {error}")

        def _finish(self, ctx, error: str | None) -> None:
            live = self._live.pop(ctx.invocation_id, None)
            if live is None:
                return
            try:
                record_run(ctx, live, error, self.source)
            except Exception:
                pass

    return ObservabilityPlugin


def _question(ctx) -> str:
    uc = ctx.user_content
    return " ".join(p.text for p in (uc.parts if uc and uc.parts else []) if getattr(p, "text", None)).strip()


def record_run(ctx, live: dict, error: str | None, source: str | None) -> None:
    state = dict(ctx.session.state)
    plan, timing, facts = state.get("plan") or {}, state.get("timing") or {}, state.get("facts") or {}
    events = live["events"]
    n_llm = (sum(1 for e in events if e["tok_in"] is not None) + (1 if timing.get("model") not in (None, "template") else 0)
             + (1 if plan.get("tier") == 1 else 0) + int(timing.get("evaluator_llm_calls", 0)))     # writer call + LLM router (tier 1) + LLM evaluator rounds
    n_tool = len(timing.get("calls", []))
    tok_in = (timing.get("tok_in") or 0) + sum(e["tok_in"] or 0 for e in events)
    tok_out = (timing.get("tok_out") or 0) + sum(e["tok_out"] or 0 for e in events)
    total = time.time() - live["t0"]
    _write(
        "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ctx.invocation_id, ctx.session.id, live["trace_id"], live["t0"], "fast",
         source or os.environ.get("OBS_SOURCE", "web"), live["question"], live["answer"],
         "error" if error else "ok", error, plan.get("cat"), plan.get("conf"), plan.get("tier"),
         facts.get("status"), timing.get("guard"), round(total, 3), timing.get("route_ms"),
         timing.get("tools_s"), timing.get("write_s"), timing.get("model"), tok_in or None, tok_out or None,
         n_llm, n_tool, json.dumps(events, default=str), json.dumps(plan, default=str),
         json.dumps(facts, default=str)[:20000], json.dumps(timing, default=str)))


def build_plugin(source: str | None = None):
    """The ADK plugin instance to pass to `App(plugins=[...])` / `Runner(plugins=[...])`."""
    init_tracing()
    return _plugin_class()(source=source)
