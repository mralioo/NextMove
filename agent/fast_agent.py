"""The pipeline as ADK agents:  supervisor -> worker -> writer   (SequentialAgent).

  supervisor  input guardrails, category + parameters, follow-up / history, objective and route (supervisor.py)
  worker      the worker <-> evaluator loop (loop.py): the specialist's MCP playbook, checked against ground truth, revised if needed
  writer      verdict -> evidence -> sources: one small-LLM call + deterministic guard (writer.py); fixed text for bounce / history

Stages hand over typed messages (schemas.py) in session state ("sp", "result", "verdict"; plus the legacy "plan" / "facts" dicts the dashboard and metrics
read). Each stage emits an event (visible in `adk web`) carrying its own timing; the writer's event is the final response. The earlier multi-LLM
supervisor / specialist / verifier loop was removed. See docs/agent_architecture_v3.md.

Session state keys: plan, sp, facts, result, verdict, last_facts + last_plan (previous accepted turn, for follow-ups), timing.
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
import artifacts  # noqa: E402
import writer  # noqa: E402
from config import CONFIG  # noqa: E402
from mcp_runtime import get_runtime  # noqa: E402
from observability import LLM_LOG, init_tracing, payload, set_attr, span  # noqa: E402
import specialists  # noqa: E402
import supervisor  # noqa: E402
from guardrails import BOUNCE_MESSAGE, SAFE_FALLBACK, check_output  # noqa: E402
from loop import LoopOutcome, worker_evaluator_loop  # noqa: E402
from schemas import EvaluatorVerdict, GraphCase, SupervisorPlan, WorkerResult, WriterInput  # noqa: E402

ROUTER_CONF_MIN = float(os.environ.get("ROUTER_CONF_MIN", "0.55"))


def _text_event(author: str, text: str, delta: dict) -> Event:
    return Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]),
                 actions=EventActions(state_delta=delta))


def _ms(t: float | None, t_start: float | None) -> int | None:
    return None if t is None or t_start is None else round((t - t_start) * 1000)


def _llm_event(author: str, r: dict, t_start: float | None) -> Event:
    """One LLM inference as an ADK event: model, inference time, tokens (usage_metadata), prompt and response previews."""
    tin, tout = r.get("tok_in"), r.get("tok_out")
    text = (f"[llm {r['role']}] {r['model']} · inference {r['seconds']} s · tokens {tin}→{tout}" + (f" · FAILED {r['error']}" if r.get("error") else ""))
    usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=tin or 0, candidates_token_count=tout or 0, total_token_count=(tin or 0) + (tout or 0)) if (tin or tout) else None
    return Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]), usage_metadata=usage,
                 custom_metadata={"kind": "llm", "role": r["role"], "model": r["model"], "inference_s": r["seconds"], "tok_in": tin, "tok_out": tout,
                                  "started_at_ms": _ms(r.get("start"), t_start), "prompt": r.get("prompt"), "response": r.get("response"), "error": r.get("error")})


def _question(ctx: InvocationContext) -> str:
    uc = ctx.user_content
    return " ".join(p.text for p in (uc.parts if uc and uc.parts else []) if p.text).strip()


def _kb():
    """The knowledge base, only in memory=cognee mode (the session / none arms of the component study stay exactly as they were)."""
    if CONFIG.memory != "cognee":
        return None
    from knowledge import kb
    return kb()


SOURCES = "\n**Sources:**"


def _strip_sources(text: str) -> str:
    """The Sources line and the confidence footer carry ids / percentages that are not in the facts: checks and stored answers use the text without them."""
    return text.split(SOURCES)[0].split("\n_Confidence")[0].split("\n**How this was worked out**")[0]


async def _report_from_artifact(q: str, plan: SupervisorPlan, art: dict) -> tuple[str, str, dict]:
    """The full report of an EARLIER answer, from its stored artifact bundle — nothing is recomputed. Returns (report, narrative, info). A report written before is reused as is."""
    if art.get("report"):
        return art["report"], art["report"], {"model": None, "guard": "stored report (nothing recomputed)"}
    f = art["facts"]
    v = art.get("verdict") or {}
    res = WorkerResult(task_id="prev", iteration=1, status="ok", facts=f, confidence=art.get("confidence") or 0.5, confidence_reasons=art.get("confidence_reasons") or [],
                       assumptions=[str(a) for a in art.get("assumptions") or []])
    ver = EvaluatorVerdict(task_id="prev", iteration=1, verdict=v.get("verdict") or "accept", objective_met=True, score=v.get("score") or 0.9, issues=v.get("issues") or [])
    wi = WriterInput(question=q, objective=plan.objective, verdict=ver, result=res, wants_argument=True)
    narrative, info = await writer.write(f"{art['question']}\n(The operator now asks: {q})", f, wi, mode="detail")
    return artifacts.full_report(art, narrative), narrative, info


class SupervisorAgent(BaseAgent):
    """SUPERVISOR: question -> `SupervisorPlan` (schemas.py). Input guardrails (unrelated → bounce), category + parameters, follow-up continuation,
    history hit, objective, route (specialist · MCP servers · datasets · ML engine), knowledge-base boundaries. Deterministic, ~1 ms; a small-LLM JSON
    call only if unsure. Also kicks the MCP server warm-up (no-op if already running)."""
    name: str = "supervisor"
    description: str = "Guards scope, classifies, extracts parameters, resolves follow-ups and history, assigns MCP servers / datasets / ML engine and the objective."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        t0 = time.time()
        llm_log: list = []
        LLM_LOG.set(llm_log)                                       # every LLM inference of this question is recorded (router, evaluator, writer) and shown as an event
        get_runtime()                                              # start/warm the MCP server in the background
        q = _question(ctx)
        st = ctx.session.state
        K = _kb()
        last, last_facts = (st.get("last_plan"), st.get("last_facts")) if CONFIG.memory != "none" else (None, None)
        if K is not None and not last_facts and supervisor.HISTORY_ON():      # history survives restarts / new sessions
            lt = K.last_turn(ctx.session.user_id)
            if lt and lt.get("plan") and lt.get("facts"):
                last, last_facts = lt["plan"], lt["facts"]
                st["last_facts"], st["last_plan"] = last_facts, last
        data_end = (K.by_id["B-COV"]["value"]["end"][:16] if K is not None and "B-COV" in K.by_id else None)
        with span("supervisor.plan", **{"tmt.router": CONFIG.router}) as sp:
            plan = await supervisor.supervise(q, last=last, last_facts=last_facts, kb=K, data_end=data_end)
            set_attr(sp, "tmt.category", plan.category)
            set_attr(sp, "tmt.confidence", plan.confidence)
            set_attr(sp, "tmt.decision", plan.decision)
            set_attr(sp, "tmt.guardrails_failed", [g.check for g in plan.guardrails if not g.passed])
            set_attr(sp, "tmt.question", q)
            set_attr(sp, "tmt.objective", plan.objective.model_dump())
            set_attr(sp, "tmt.route", plan.route.model_dump())
            set_attr(sp, "tmt.entities", plan.entities.model_dump(exclude_none=True, exclude_defaults=True))
            set_attr(sp, "tmt.follow_up", plan.follow_up.model_dump() if plan.follow_up else None)
            set_attr(sp, "tmt.history", plan.history.model_dump(exclude={"answer"}) if plan.history else None)
            set_attr(sp, "tmt.plan", payload(plan.model_dump(mode="json", exclude={"parts"}), 6000))
        legacy = plan.to_legacy()
        legacy.update(plan_id=plan.plan_id, decision=plan.decision, objective=plan.objective.model_dump(), route=plan.route.model_dump(),
                      guardrails=[g.model_dump() for g in plan.guardrails], history=plan.history.model_dump() if plan.history else None,
                      follow_up=plan.follow_up.model_dump() if plan.follow_up else None, llm_router_error=plan.llm_router_error)
        timing = {"llm": list(llm_log), "route_ms": round((time.time() - t0) * 1000), "tier": plan.tier, "cfg": asdict(CONFIG), "decision": plan.decision,
                  "guardrails": [g.model_dump() for g in plan.guardrails if not g.passed or g.action != "allow"],
                  "handover": {"plan": {**plan.model_dump(mode="json", exclude={"parts", "guardrails"}), "guardrail_results": [g.model_dump() for g in plan.guardrails]}}}
        sp_dump = plan.model_dump(mode="json")
        st["plan"], st["sp"], st["timing"], st["t_start"] = legacy, sp_dump, timing, t0
        for r in llm_log:
            yield _llm_event(self.name, r, t0)
        ev = _text_event(self.name, f"[plan · {timing['route_ms']} ms] {plan.decision} · {plan.category} · {plan.route.specialist} · engine {plan.route.ml_engine} · "
                                    f"tools {plan.route.tools} · objective: {plan.objective.statement}", {"plan": legacy, "sp": sp_dump, "timing": timing, "t_start": t0})
        ev.custom_metadata = {"kind": "supervisor", "seconds": round(time.time() - t0, 3), "route_ms": timing["route_ms"], "llm_route_s": round(sum(r["seconds"] for r in llm_log), 3),
                              "decision": plan.decision, "category": plan.category, "tier": plan.tier, "route": plan.route.model_dump(mode="json")}
        yield ev


class WorkerAgent(BaseAgent):
    """WORKER ⇄ EVALUATOR loop (loop.py): the worker runs the assigned specialist playbook over the MCP tools (datasets, analytics, TabPFN engine) and returns facts
    + confidence; the evaluator checks them against the objective, the knowledge base (ground truth, boundaries) and the knowledge graph (similar past cases) and
    either accepts, asks for a revision (validated adjustments) or rejects. Bounded by the failsafes in guardrails.LIMITS."""
    name: str = "worker"
    description: str = "Runs the assigned specialist over MCP tools; the evaluator loop verifies and, if needed, revises."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        t0 = time.time()
        st = ctx.session.state
        t_start = st.get("t_start")
        LLM_LOG.set([])
        plan = SupervisorPlan.model_validate(st["sp"])            # hard failsafe: a malformed hand-over raises, it is never guessed around
        timing = {**st.get("timing", {})}
        if plan.decision in ("bounce", "answer_from_history"):
            timing.update(tools_s=0.0, calls=[], loop=[])
            st["timing"] = timing
            yield _text_event(self.name, f"[skipped: {plan.decision} · nothing to compute]", {"timing": timing, "facts": {"status": plan.decision}})
            return
        K = _kb()
        if plan.decision == "follow_up":
            facts = {"status": "follow", "prev": st.get("last_facts") or {}, "cat": "FOLLOW"}
            res = WorkerResult(task_id="follow", iteration=1, status="follow", facts=facts, confidence=0.95, confidence_reasons=["answered from the earlier facts"])
            ver = EvaluatorVerdict(task_id="follow", iteration=1, verdict="accept", objective_met=True, score=0.95, rationale="follow-up answered from the earlier facts")
            out = LoopOutcome(result=res, verdict=ver, iterations=[], guardrails=[])
        else:
            from kgraph import kg
            with span("worker.loop", **{"tmt.category": plan.category, "tmt.tools": plan.route.tools, "tmt.ml_engine": plan.route.ml_engine}) as sp:
                out = await worker_evaluator_loop(plan, st.get("last_facts"), get_runtime(), K, kg() if K is not None else None, allow_llm=(CONFIG.memory == "cognee"))
                set_attr(sp, "tmt.iterations", len(out.iterations))
                set_attr(sp, "tmt.verdict", out.verdict.verdict)
                set_attr(sp, "tmt.confidence", out.result.confidence)
            facts = out.result.facts
            if K is not None and facts.get("status") == "ok":            # verified boundaries / insights the writer must respect
                extra = [K.by_id[i] for i in plan.kb_boundaries if i in K.by_id][:2]
                extra += [e for e in K.search(plan.question, cats=[plan.category], kinds=["insight"], k=1) if e["id"] not in {x["id"] for x in extra}]
                facts["kb"] = [{"id": e["id"], "t": e["text"][:230]} for e in extra][:3]
        calls = [{"tool": c.tool, "s": c.seconds, "round": r, "server": c.server, "args": c.args, "result_preview": c.result_preview, "bytes": c.result_bytes, "ok": c.ok,
                  "wait_ready_ms": c.wait_ready_ms, "error": c.error} for r, c in out.calls] or [{"tool": c.tool, "s": c.seconds} for c in out.result.tools_called]
        timing["llm"] = timing.get("llm", []) + [r for it in out.iterations for r in it.get("llm", [])]
        timing.update(tools_s=round(time.time() - t0, 2), calls=calls, loop=out.iterations, confidence=out.result.confidence, verdict=out.verdict.verdict,
                      evaluator_llm_calls=sum(1 for i in out.iterations if not str(i.get("model", "")).startswith("deterministic")),
                      guardrails=timing.get("guardrails", []) + [g.model_dump() for g in out.guardrails])
        timing["handover"] = {**timing.get("handover", {}),
                              "rounds": [{"round": i["i"], "overrides": i.get("overrides"), "tools": i.get("tools"), "confidence": i.get("confidence"), "status": i.get("status"), "engine": i.get("engine"),
                                          "worker_s": i.get("worker_s"), "verdict": i.get("verdict"), "score": i.get("score"), "issues": i.get("issues"), "adjustments": i.get("adjustments"),
                                          "evaluator_model": i.get("model"), "evaluator_s": i.get("evaluator_s")} for i in out.iterations],
                              "result": {**out.result.model_dump(exclude={"facts", "tools_called"}, mode="json")},
                              "verdict": out.verdict.model_dump(mode="json", exclude={"checks", "similar_cases"}), "checks": [c.model_dump() for c in out.verdict.checks],
                              "similar_cases": [c.model_dump() for c in out.verdict.similar_cases]}
        st["facts"], st["timing"] = facts, timing
        name = specialists.SPECIALISTS[plan.category].name if plan.category in specialists.SPECIALISTS else plan.category.lower()
        author = f"worker:{name}"
        # ---- the steps, as ADK events (visible in the ADK UI chat and trace): every MCP call as a function_call / function_response pair with its arguments,
        #      result preview and time, then the evaluator's verdict for that round. Custom agents make these calls themselves, so we report them here.
        by_round: dict[int, list] = {}
        for r, c in out.calls:
            by_round.setdefault(r, []).append(c)
        for it in out.iterations:
            for n, c in enumerate(by_round.get(it["i"], [])):
                cid = f"mcp-{it['i']}-{n}"
                yield Event(author=author, content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(id=cid, name=c.tool, args=c.args or {}))]),
                            custom_metadata={"round": it["i"], "server": c.server, "kind": "mcp_call", "started_at_ms": _ms(c.started_at, t_start)})
                yield Event(author=author, content=types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                    id=cid, name=c.tool, response={"ok": c.ok, "seconds": c.seconds, "server": c.server, "wait_ready_ms": c.wait_ready_ms, "result_bytes": c.result_bytes,
                                                   "result_preview": c.result_preview, **({"error": c.error} if c.error else {})}))]),
                            custom_metadata={"round": it["i"], "server": c.server, "kind": "mcp_result", "seconds": c.seconds, "started_at_ms": _ms(c.started_at, t_start),
                                             "ended_at_ms": _ms(c.started_at + c.seconds, t_start) if c.started_at else None, "wait_ready_ms": c.wait_ready_ms})
            for r in it.get("llm", []):
                yield _llm_event("evaluator", r, t_start)
            yield Event(author=author, content=types.Content(role="model", parts=[types.Part(text=(
                f"[worker round {it['i']} · {it['worker_s']} s] {name} · engine {it['engine']} · tools {it['tools']} · status {it['status']} · confidence {it['confidence']}"))]),
                custom_metadata={"round": it["i"], "kind": "worker_round", "seconds": it["worker_s"], "started_at_ms": _ms(it.get("started"), t_start), "tools": it["tools"],
                                 "engine": it["engine"], "confidence": it["confidence"], "overrides": it.get("overrides")})
            yield Event(author="evaluator", content=types.Content(role="model", parts=[types.Part(text=(
                f"[evaluator round {it['i']} · {it['evaluator_s']} s] {it['verdict']} · score {it['score']} · confidence {it['confidence']} · model {it['model']}"
                + (f" · issues: {'; '.join(it['issues'])}" if it.get("issues") else "") + (f" · adjustments: {it['adjustments']}" if it.get("adjustments") else "")))]),
                custom_metadata={"round": it["i"], "kind": "evaluator", "seconds": it["evaluator_s"], "model": it["model"], "verdict": it["verdict"], "score": it["score"],
                                 "issues": it.get("issues"), "adjustments": it.get("adjustments"), "llm_calls": len(it.get("llm", []))})
        yield Event(author=author, content=types.Content(role="model", parts=[types.Part(
            text=f"[facts · worker+evaluator {timing['tools_s']} s] confidence {out.result.confidence} · verdict {out.verdict.verdict} · {json.dumps(facts, ensure_ascii=False, separators=(',', ':'))[:1800]}")]),
            custom_metadata={"kind": "facts", "seconds": timing["tools_s"], "mcp_s": round(sum(c.seconds for _, c in out.calls), 3), "rounds": len(out.iterations)},
            actions=EventActions(state_delta={"facts": facts, "timing": timing, "result": out.result.model_dump(exclude={"facts"}, mode="json"), "verdict": out.verdict.model_dump(mode="json")}))


class WriterAgent(BaseAgent):
    """WRITER: verdict first, then evidence and references; an argument only when the operator asks why. One small-LLM call + the deterministic guard, a
    Sources line built from the datasets / tools / model / knowledge-base entries actually used, and the SAFE FALLBACK when the evaluator rejected the result.
    Bounce and history answers are fixed texts (no LLM). Its event is the final answer."""
    name: str = "writer"
    description: str = "Writes the verdict + evidence + sources brief; enforces the output guardrails."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        t0 = time.time()
        st = ctx.session.state
        t_start = st.get("t_start")
        w_log: list = []
        LLM_LOG.set(w_log)
        q = _question(ctx)
        plan = SupervisorPlan.model_validate(st["sp"])
        facts = st.get("facts") or {}
        timing = {**st.get("timing", {})}
        legacy = st.get("plan") or {}
        info: dict = {}
        source, refs = "worker", []
        narr: str | None = None                            # the writer's own text, before the deterministic method section / sources / footer are added
        art: dict | None = None                            # the artifact bundle of this turn (see artifacts.py)
        line = ""
        K = _kb()
        mode = "detail" if (writer.ANSWER_MODE() == "detail" or writer.wants_detail(q)) else "brief"
        prev_art = None
        if plan.decision == "follow_up" and plan.follow_up and plan.follow_up.mode == "explain" and mode == "detail":
            last_f = st.get("last_facts") or {}
            same = lambda a: bool(a) and json.dumps(a.get("facts"), sort_keys=True, default=str) == json.dumps(last_f, sort_keys=True, default=str)   # the bundle must be of THE answer being asked about
            prev_art = st.get("last_artifact") if same(st.get("last_artifact")) else None
            if prev_art is None and K is not None:
                cand = K.last_artifact(ctx.session.user_id)
                prev_art = cand if same(cand) else None
            if prev_art is None and last_f.get("status") == "ok":          # an answer given before artifacts were stored: the narrative can still be written from its facts
                prev_art = {"legacy": True, "question": (st.get("last_plan") or {}).get("question") or "the earlier question", "facts": last_f, "confidence": None, "created_at": "",
                            "category": last_f.get("cat"), "verdict": {}, "tools": []}
            if prev_art and (prev_art.get("facts") or {}).get("status") != "ok":
                prev_art = None
        if plan.decision == "bounce":
            answer, info, source = plan.message or BOUNCE_MESSAGE, {"model": None, "guard": "bounce (unrelated question)"}, "bounce"
        elif prev_art is not None:                         # "why / evidence / which tools / full report": answered from the stored artifact, nothing recomputed
            answer, narr, info = await _report_from_artifact(q, plan, prev_art)
            source, art = "follow_up", {**prev_art, "report": answer}
            if K is not None and prev_art.get("turn_id") and not prev_art.get("report"):
                K.attach_report(int(prev_art["turn_id"]), answer)                 # the operator knowledge base keeps the report next to the bundle
                _index_report(ctx, q, art)                                        # ... and so do the knowledge graph and the memory agent
        elif plan.decision == "answer_from_history" and plan.history:
            h = plan.history
            K1 = _kb()
            art_h = K1.get_artifact(h.turn_id) if K1 is not None else None
            # a stored answer from before the brief format is shortened; a stored bundle gives its brief and the confidence line
            answer = ((art_h["brief"] + writer.confidence_footer(art_h.get("confidence"))) if art_h and art_h.get("brief") else writer.shorten(h.answer)) \
                + f"\n_(From the accepted analysis {h.age_s / 60:.0f} min ago on the same data window; nothing was recomputed.)_"
            info, source = {"model": None, "guard": f"history ({h.kind}, similarity {h.similarity})"}, "history"
            K0 = _kb()
            old = K0.get_turn(h.turn_id) if K0 is not None else None
            if old and old.get("facts"):                    # the conversation continues from the reused answer: follow-ups need its facts and plan
                st["last_facts"], st["last_plan"] = old["facts"], old.get("plan")
                st["last_artifact"] = K0.get_artifact(h.turn_id)   # ... and "why / evidence / which tools" about it: the stored artifact bundle (None for an answer stored before artifacts existed)
        else:
            res = WorkerResult(**st["result"], facts=facts) if st.get("result") else WorkerResult(task_id="x", iteration=1, status="error", facts=facts, confidence=0.05)
            ver = EvaluatorVerdict.model_validate(st["verdict"]) if st.get("verdict") else EvaluatorVerdict(task_id="x", iteration=1, verdict="reject", objective_met=False, score=0.0)
            if ver.verdict == "reject" and facts.get("status") in ("ok", "multi"):
                answer = SAFE_FALLBACK + (" (" + "; ".join(ver.issues[:2]) + ")" if ver.issues else "")
                info, source = {"model": None, "guard": "failsafe: evaluator rejected the result"}, "safe_fallback"
            else:
                wi = WriterInput(question=q, objective=plan.objective, verdict=ver, result=res, wants_argument=writer.wants_argument(q))
                answer, info = await writer.write(q, facts, wi, mode=mode)
                narr = answer
                if facts.get("status") in ("ok", "multi"):
                    with span("writer.references") as rsp:
                        refs = writer.build_references(plan.route, res, ver)
                        line = writer.sources_line(refs, res.confidence, ver.verdict)
                        set_attr(rsp, "tmt.references", [{"kind": r.kind, "id": r.id} for r in refs])
                        set_attr(rsp, "tmt.sources_line", line)
                    if mode == "detail":
                        answer += SOURCES + line[len("**Sources:**"):]
                    else:                                    # the operator's answer: brief, no sources; how sure + how to get the details in one line
                        answer += writer.confidence_footer(res.confidence)
                source = "decline" if facts.get("status") in ("oos", "unsupported", "need") else ("follow_up" if facts.get("status") == "follow" else "worker")
        body = _strip_sources(narr if narr is not None else answer)
        outg = check_output(body, facts if prev_art is None else prev_art["facts"], q, brief=(mode == "brief" and source == "worker" and facts.get("status") in ("ok", "multi"))) if source in ("worker", "follow_up", "decline") else []
        timing = {**timing, "write_s": round(time.time() - t0, 2), **info,
                  "guardrails": timing.get("guardrails", []) + [g.model_dump() for g in outg if not g.passed]}
        timing["handover"] = {**timing.get("handover", {}), "writer": {"source": source, "objective": plan.objective.statement, "wants_argument": writer.wants_argument(q), "model": info.get("model"),
                                                                   "guard": info.get("guard"), "answer_words": len(body.split()), "references": [r.model_dump() for r in refs],
                                                                   "prompt": info.get("prompt"), "output_guardrails": [g.model_dump() for g in outg]}}
        timing.pop("prompt", None)
        now = time.time()
        prev = st.get("timing", {})
        total = round(now - t_start, 2) if t_start else None
        parts = {"supervisor_s": round(prev.get("route_ms", 0) / 1000, 3), "worker_evaluator_s": prev.get("tools_s"), "mcp_s": round(sum(c.get("s", 0) for c in prev.get("calls", [])), 2),
                 "writer_s": timing["write_s"], "writer_llm_s": round(sum(r["seconds"] for r in w_log), 2), "total_s": total}
        timing["stages"] = parts
        timing["llm"] = timing.get("llm", []) + w_log
        if source == "worker" and facts.get("status") in ("ok", "multi") and prev_art is None:
            # ---- the artifact bundle of this answer: brief now, full report if it was asked for
            try:
                from data_window import window as _window
                brief_text = writer.shorten(narr) if mode == "detail" else answer.split("\n_Confidence")[0]
                art = artifacts.build(question=q, session_id=ctx.session.id, plan=st.get("sp") or {}, result=st.get("result"), verdict=st.get("verdict"), facts=facts, timing=timing, refs=refs,
                                      brief=brief_text, sources_line=line, sanity=None, source=source, answer_mode=mode, data_window=_window())
                import feedback as fb
                art["requires_action"] = bool(writer.wants_action(q) or plan.category in ("C", "A", "P"))         # the UI then asks "what did you do about it?"
                art["recommended_actions"] = fb.recommended_actions(brief_text, facts)
                if supervisor.HISTORY_ON():                                    # operator precedents: what operators did in similar situations and how it went (feedback loop)
                    from kgraph import _h, _norm
                    art["precedents"] = fb.precedents(q, plan.category, plan.entities, exclude_key=_h(_norm(q), 12))
                    art["precedent"] = fb.precedent_line(art["precedents"]) or None
                    if art["precedent"] and mode == "brief" and "\n_Confidence" in answer:
                        answer = answer.replace("\n_Confidence", "\n" + art["precedent"] + "\n_Confidence", 1)
                if mode == "detail":
                    art["report"] = answer = artifacts.full_report(art, narr)
            except Exception as e:                                         # an artifact bug must never cost the operator the answer
                art = None
                timing["artifact_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        keep = (facts if facts.get("status") == "ok" else st.get("last_facts")) if CONFIG.memory != "none" else None   # follow-ups refer to the last real answer
        delta = {"timing": timing, "last_facts": keep}
        if source == "history" and st.get("last_plan"):
            delta["last_plan"] = st["last_plan"]
        if source == "history":
            delta["last_artifact"] = st.get("last_artifact")                   # "why / evidence" after a reused answer refers to ITS bundle
        if K is not None and source in ("worker", "follow_up", "decline", "safe_fallback"):
            timing["sanity"] = _sanity_and_remember(K, ctx, q, body, facts, legacy, plan, st.get("verdict"), st.get("result"), source, art if source == "worker" else None)
            if timing["sanity"].get("accepted") and facts.get("status") in ("ok", "multi"):
                delta["last_plan"] = st["sp"]
            if art is not None and timing["sanity"].get("turn_id"):
                art["turn_id"] = timing["sanity"]["turn_id"]
        if art is not None:
            delta["last_artifact"] = art                                   # what "why / evidence / which tools" refers to in this session
        for r in w_log:
            yield _llm_event(self.name, r, t_start)
        delta["timing"] = timing
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=(
            f"[timing] total {total} s = supervisor {parts['supervisor_s']} s + worker/evaluator {parts['worker_evaluator_s']} s (of which MCP calls {parts['mcp_s']} s, summed over parallel calls) "
            f"+ writer {parts['writer_s']} s (LLM inference {parts['writer_llm_s']} s)"))]), custom_metadata={"kind": "timing", **parts})
        final = _text_event(self.name, answer, delta)
        final.custom_metadata = {"kind": "answer", "source": source, "answer_mode": mode if source in ("worker", "follow_up") else None, "answer_words": len(answer.split()),
                                 "artifact_turn_id": (art or {}).get("turn_id"), "turn_id": timing.get("sanity", {}).get("turn_id"), "requires_action": (art or {}).get("requires_action"),
                                 "precedent": (art or {}).get("precedent"), **parts, "model": info.get("model"), "tok_in": info.get("tok_in"), "tok_out": info.get("tok_out"), "guard": info.get("guard")}
        if info.get("tok_in") or info.get("tok_out"):
            final.usage_metadata = types.GenerateContentResponseUsageMetadata(prompt_token_count=info.get("tok_in") or 0, candidates_token_count=info.get("tok_out") or 0,
                                                                              total_token_count=(info.get("tok_in") or 0) + (info.get("tok_out") or 0))
        yield final


def _index_report(ctx, question: str, art: dict) -> None:
    """A full report was generated for the first time: put it on the knowledge-graph Artifact node and in the memory agent (background, never blocks the answer)."""
    try:
        if art.get("problem_key") and supervisor.HISTORY_ON():
            from kgraph import kg
            kg().record_artifact(art["problem_key"], art)
    except Exception:
        pass
    from knowledge import cognee
    if cognee().available:
        import threading
        threading.Thread(target=cognee().remember_qa, args=(ctx.session.id, question, art["report"][:6000], artifacts.memory_summary(art)), daemon=True, name="cognee-report").start()


def _sanity_and_remember(K, ctx, question: str, answer: str, facts: dict, legacy: dict, plan: SupervisorPlan, verdict: dict | None, result: dict | None, source: str, art: dict | None = None) -> dict:
    """Cross-check the delivered answer against the knowledge base (ms, deterministic), store the turn locally (with the verdict, so the supervisor's history lookup only
    reuses ACCEPTED answers), add accepted cases to the knowledge graph, and mirror the turn to Cognee in a background thread — none of it delays or changes the answer."""
    with span("kb.sanity") as sp:
        try:
            res = K.sanity_check(question, answer, facts, legacy)
        except Exception as e:                                       # a checker bug must never break an answer
            res = {"ok": None, "score": None, "checks": [], "error": f"{type(e).__name__}: {str(e)[:100]}"}
        set_attr(sp, "tmt.sanity_ok", res.get("ok"))
        set_attr(sp, "tmt.sanity_checks", [{"id": c["id"], "ok": c["ok"], "detail": c["detail"][:110]} for c in res.get("checks", [])])
        set_attr(sp, "tmt.evidence", res.get("evidence"))
        set_attr(sp, "tmt.sanity_fails", [c["id"] for c in res.get("checks", []) if not c["ok"]])
    accepted = bool(facts.get("status") in ("ok", "multi") and (verdict or {}).get("verdict") == "accept" and res.get("ok") is not False and source == "worker")
    conf = (result or {}).get("confidence")
    tid = None
    if art is not None:
        from kgraph import _h, _norm
        art["problem_key"] = _h(_norm(question), 12)
        art["sanity"] = {"ok": res.get("ok"), "fails": [c["id"] for c in res.get("checks", []) if not c["ok"]]}
    try:
        from supervisor import plan_key
        tid = K.remember_turn(ctx.session.id, ctx.session.user_id, question, plan.category, answer, facts, res, verdict=(verdict or {}).get("verdict"), confidence=conf,
                        plan_key=plan_key(plan.category, plan.entities), data_end=(K.by_id["B-COV"]["value"]["end"][:16] if "B-COV" in K.by_id else None), accepted=accepted, plan=plan.model_dump(mode="json"), artifact=art)
        if art is not None:
            art["turn_id"] = tid
    except Exception:
        pass
    if accepted and supervisor.HISTORY_ON():          # evaluation runs (TMT_HISTORY=off) must not train the graph
        try:
            from kgraph import actions_from_facts, kg
            acts, opts = actions_from_facts(facts) if facts.get("status") == "ok" else ([], [])
            with span("kg.record_case", **{"tmt.category": plan.category, "tmt.actions": acts, "tmt.options": opts, "tmt.neo4j_mirror": kg().neo4j_configured()}):
                kg().record_case(GraphCase(question=question, category=plan.category, entities=plan.entities, answer=answer, confidence=conf, verdict="accept", actions=acts, options=opts))
                if art is not None:
                    kg().record_artifact(art["problem_key"], art)              # operator knowledge base in the graph: Problem -> Artifact -> tools / datasets / model / KB entries
        except Exception:
            pass
    from knowledge import cognee
    if cognee().available and facts.get("status") in ("ok", "need"):
        import threading
        ctxt = (artifacts.memory_summary(art) + "\n" if art is not None else "") + json.dumps({k: facts.get(k) for k in ("cat", "cl", "top", "found", "worst", "date", "assumed") if facts.get(k)}, ensure_ascii=False, default=str)
        threading.Thread(target=cognee().remember_qa, args=(ctx.session.id, question, answer, ctxt), daemon=True, name="cognee-remember").start()
    return {"turn_id": tid, "ok": res.get("ok"), "score": res.get("score"), "fails": [c["id"] for c in res.get("checks", []) if not c["ok"]], "n": len(res.get("checks", [])), "accepted": accepted}


def build_fast_agent() -> SequentialAgent:
    init_tracing()
    if os.environ.get("WARM_ON_START", "1") != "0":
        get_runtime()          # start the MCP server now: its warm-up overlaps with agent start-up / the user typing
        if os.environ.get("WARM_LLM", "1") != "0":     # one tiny call; skipped by the evaluation harness
            import threading
            threading.Thread(target=writer.warm_connection, daemon=True, name="llm-warm").start()
    return SequentialAgent(name="talk_to_my_train", sub_agents=[SupervisorAgent(), WorkerAgent(), WriterAgent()],
                           description="Operator assistant: supervisor (guardrails, route, follow-up, history) -> worker <-> evaluator loop (MCP tools, knowledge base + graph) -> writer.")
