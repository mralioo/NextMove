"""The worker ⇄ evaluator FEEDBACK LOOP (hard-bounded).

    task₁ → worker → result₁ → evaluator → verdict₁ ─ accept ─────────────────────────────► done
                                                      └ revise(adjustments) → task₂ → worker → result₂ → evaluator → verdict₂ → …
                                                      └ reject ───────────────────────────► done (the answer becomes the safe fallback)

Hard failsafes (guardrails.LIMITS): at most `max_iterations` rounds, a wall-clock `loop_deadline_s`, every hand-over validated by pydantic, a worker
that raises becomes a `WorkerResult(status="error")`. The loop returns every iteration so the trace shows what was tried and why.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import evaluator as ev
import worker as wk
from guardrails import LIMITS
from observability import payload, set_attr, span
from schemas import EvaluatorVerdict, GuardrailResult, SupervisorPlan, WorkerResult


@dataclass
class LoopOutcome:
    result: WorkerResult
    verdict: EvaluatorVerdict
    iterations: list[dict] = field(default_factory=list)
    guardrails: list[GuardrailResult] = field(default_factory=list)
    calls: list = field(default_factory=list)          # (round, ToolCall) for every MCP call of every round, with arguments and result previews


async def worker_evaluator_loop(plan: SupervisorPlan, last_facts: dict | None, mcp, kb=None, graph=None, allow_llm: bool = True) -> LoopOutcome:
    t0 = time.time()
    overrides: dict = {}
    its: list[dict] = []
    guards: list[GuardrailResult] = []
    result = verdict = None
    all_calls: list = []
    for i in range(1, LIMITS.max_iterations + 1):
        task = wk.task_from_plan(plan, i, overrides)
        with span("worker_evaluator.round", **{"tmt.round": i, "tmt.of": LIMITS.max_iterations, "tmt.overrides": overrides}) as rsp:
            try:
                result = await wk.run_worker(task, last_facts, mcp, kb)
            except Exception as e:                                            # schema / programming error: fail closed, never guess
                guards.append(GuardrailResult(stage="worker", check="worker_ran", passed=False, action="fallback", detail=f"{type(e).__name__}: {str(e)[:120]}"))
                raise
            verdict = await ev.evaluate(plan.question, plan, task, result, kb, graph, allow_llm)
            set_attr(rsp, "tmt.result", payload({"status": result.status, "confidence": result.confidence, "reasons": result.confidence_reasons, "assumptions": result.assumptions,
                                                  "tools": [{"tool": c.tool, "s": c.seconds, "ok": c.ok} for c in result.tools_called]}, 3000))
            set_attr(rsp, "tmt.verdict", verdict.verdict)
        all_calls += [(i, c) for c in result.tools_called]
        its.append({"i": i, "worker_s": result.seconds, "confidence": result.confidence, "status": result.status, "tools": [c.tool for c in result.tools_called],
                    "engine": result.ml_engine_used, "verdict": verdict.verdict, "score": verdict.score, "issues": verdict.issues[:4], "model": verdict.model,
                    "evaluator_s": verdict.seconds, "adjustments": verdict.adjustments.model_dump(exclude_none=True) if verdict.adjustments else None, "overrides": dict(overrides)})
        if verdict.verdict == "accept" or verdict.verdict == "reject":
            break
        if i == LIMITS.max_iterations:
            guards.append(GuardrailResult(stage="failsafe", check="iteration_cap", passed=False, action="escalate", detail=f"still 'revise' after {i} rounds: best result kept, flagged"))
            break
        if time.time() - t0 > LIMITS.loop_deadline_s:
            guards.append(GuardrailResult(stage="failsafe", check="loop_deadline", passed=False, action="cut", detail=f"{LIMITS.loop_deadline_s:.0f} s budget used"))
            break
        overrides = {**overrides, **(verdict.adjustments.model_dump(exclude_none=True) if verdict.adjustments else {})}     # closed list, validated
    if result.confidence < LIMITS.min_confidence and result.status == "ok":
        guards.append(GuardrailResult(stage="failsafe", check="confidence_floor", passed=False, action="escalate", detail=f"confidence {result.confidence} < {LIMITS.min_confidence}: flagged low-confidence"))
    return LoopOutcome(result=result, verdict=verdict, iterations=its, guardrails=guards, calls=all_calls)
