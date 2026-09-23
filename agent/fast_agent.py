"""The FAST pipeline as ADK agents:  router -> executor -> writer   (SequentialAgent).

  router    deterministic plan JSON in ~1 ms (LLM JSON fallback only if unsure)
  executor  playbook: parallel MCP tool calls (pre-warmed persistent server) -> compact facts JSON
  writer    ONE short LLM call + a deterministic number guard -> the operator-facing brief

Stages hand over *symbolic JSON in session state* ("plan", "facts"), never prose. Each stage emits an
event (visible in `adk web`) carrying its own timing; the writer's event is the final response. The
old multi-LLM supervisor/specialist/verifier design stays available with AGENT_MODE=llm.

Session state keys: plan, facts, last_facts (previous turn, for follow-ups), timing.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import executor  # noqa: E402
import router  # noqa: E402
import writer  # noqa: E402
from config import CONFIG  # noqa: E402
from mcp_runtime import get_runtime  # noqa: E402
from observability import init_tracing, set_attr, span  # noqa: E402

ROUTER_CONF_MIN = float(os.environ.get("ROUTER_CONF_MIN", "0.55"))
_MEMORY = {}


def episodic_memory():
    """The persistent fact store, only when the config asks for episodic memory."""
    if CONFIG.memory != "episodic":
        return None
    if "m" not in _MEMORY:
        from memory import EpisodicMemory
        _MEMORY["m"] = EpisodicMemory()
    return _MEMORY["m"]


def _text_event(author: str, text: str, delta: dict) -> Event:
    return Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]),
                 actions=EventActions(state_delta=delta))


def _question(ctx: InvocationContext) -> str:
    uc = ctx.user_content
    return " ".join(p.text for p in (uc.parts if uc and uc.parts else []) if p.text).strip()


class RouterAgent(BaseAgent):
    """Question -> plan JSON. Also kicks the MCP server warm-up (no-op if already running)."""
    name: str = "router"
    description: str = "Classifies the question and extracts entities (deterministic, LLM fallback)."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        t0 = time.time()
        get_runtime()                                              # start/warm the MCP server in the background
        q = _question(ctx)
        has_history = CONFIG.memory != "none" and bool(ctx.session.state.get("last_facts"))
        with span("route.classify", **{"tmt.router": CONFIG.router}) as sp:
            plan = router.route(q, has_history=has_history)
            if CONFIG.router == "tfidf" and plan["cat"] not in ("OOS", "FOLLOW"):
                import router_tfidf
                cat, prob = router_tfidf.classify(q, exclude=tuple(os.environ.get("TMT_EXCLUDE", "").split("||")) if os.environ.get("TMT_EXCLUDE") else ())
                plan.update(cat=cat, conf=round(prob, 2), tier=2, **router.CAT_DATA.get(cat, {}))
            elif CONFIG.router == "jev":
                import router_jev
                cat, prob = await asyncio.to_thread(router_jev.classify, q, has_history)
                plan.update(cat=cat, conf=round(prob, 2), tier=3, follow=(cat == "FOLLOW"), **router.CAT_DATA.get(cat, {}))
            set_attr(sp, "tmt.category", plan["cat"])
            set_attr(sp, "tmt.confidence", plan["conf"])
        low_conf = plan["conf"] < ROUTER_CONF_MIN and plan["cat"] not in ("OOS", "FOLLOW") and CONFIG.router == "rules"
        if CONFIG.router == "llm" or low_conf:
            with span("llm.route", **{"tmt.reason": "router=llm" if CONFIG.router == "llm" else "low confidence", "tmt.det_category": plan["cat"]}) as sp:
                try:
                    plan = await router.llm_route(q, plan, has_history)
                    set_attr(sp, "tmt.category", plan["cat"])
                except Exception as e:                             # network/LLM trouble: keep the deterministic plan
                    plan["llm_router_error"] = str(e)[:80]
                    set_attr(sp, "tmt.error", str(e)[:120])
        timing = {"route_ms": round((time.time() - t0) * 1000), "tier": plan["tier"], "cfg": asdict(CONFIG)}
        ctx.session.state["plan"], ctx.session.state["timing"] = plan, timing
        yield _text_event(self.name, "[plan] " + json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
                          {"plan": plan, "timing": timing})


class ExecutorAgent(BaseAgent):
    """Plan -> facts JSON via parallel MCP calls; no LLM."""
    name: str = "executor"
    description: str = "Runs the category playbook against the MCP tools (incl. the TabPFN engine)."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        t0 = time.time()
        st = ctx.session.state
        with span("executor.playbook", **{"tmt.category": st["plan"]["cat"], "tmt.tools": st["plan"].get("tools")}) as sp:
            facts, trace = await executor.execute(st["plan"], _question(ctx), st.get("last_facts"), get_runtime(), episodic_memory())
            set_attr(sp, "tmt.facts_status", facts.get("status"))
        timing = {**st.get("timing", {}), "tools_s": round(time.time() - t0, 2), "calls": trace}
        st["facts"], st["timing"] = facts, timing
        yield _text_event(self.name, "[facts] " + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))[:2000],
                          {"facts": facts, "timing": timing})


class WriterAgent(BaseAgent):
    """Facts -> operator brief (one LLM call) + numeric guard. Its event is the final answer."""
    name: str = "writer"
    description: str = "Writes the short, plain, fact-driven answer and verifies every number."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        t0 = time.time()
        st = ctx.session.state
        facts = st["facts"]
        answer, info = await writer.write(_question(ctx), facts)
        timing = {**st.get("timing", {}), "write_s": round(time.time() - t0, 2), **info}
        keep = (facts if facts.get("status") == "ok" else st.get("last_facts")) if CONFIG.memory != "none" else None   # follow-ups refer to the last real answer
        yield _text_event(self.name, answer, {"timing": timing, "last_facts": keep})


def build_fast_agent() -> SequentialAgent:
    init_tracing()
    if os.environ.get("WARM_ON_START", "1") != "0":
        get_runtime()          # start the MCP server now: its warm-up overlaps with agent start-up / the user typing
        if os.environ.get("WARM_LLM", "1") != "0":     # one tiny call; skipped by the evaluation harness
            import threading
            threading.Thread(target=writer.warm_connection, daemon=True, name="llm-warm").start()
    return SequentialAgent(name="talk_to_my_train", sub_agents=[RouterAgent(), ExecutorAgent(), WriterAgent()],
                           description="Fast operator-assistant pipeline: route -> execute -> write.")
