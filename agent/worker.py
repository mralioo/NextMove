"""WORKER: a function, not an LLM. `WorkerTask` in → `WorkerResult` out.

The supervisor decides WHAT to use (specialist, MCP tools, datasets, ML engine); the worker only executes: it runs the specialist's playbook
against the MCP server(s), and returns the compact facts together with a CONFIDENCE score and the reasons for it, the assumptions it had to make,
the datasets it touched, the tools it called and the ML engine that produced the numbers. The evaluator then decides whether that is good enough.

Confidence is not a model probability. It is a documented, deterministic estimate of how far the result can be trusted:

    start   0.90 for a result the worker produced (0.95 for a clear decline / a request for missing input: the decision itself is reliable)
    minus   0.04 per assumption made on the operator's behalf (max 0.20)
            0.10 closure simulated with an assumed diversion share           0.10 load ranking is a proxy (validated Spearman 0.44, not a capacity)
            0.15 scenario outside the data window                            0.15 anomaly causes cannot be proven; 0.20 none found
            0.20 event pattern from fewer than 5 past events                 0.35 correlations weaker than |r| 0.3
            0.05 approximate passengers per line (E) / no O-D data (H)       0.03 empirical baseline instead of TabPFN
    plus    0.05 the tool result agrees with ground truth recomputed from the raw CSVs (knowledge-base truth check)
"""
from __future__ import annotations

import time

import executor
from schemas import Entities, ToolCall, WorkerResult, WorkerTask


def score_confidence(cat: str, facts: dict, task: WorkerTask, truth_ok: bool | None = None) -> tuple[float, list[str]]:
    st = facts.get("status")
    if st in ("need", "unsupported", "oos", "follow"):
        return 0.95, [f"status '{st}': a clear decision, no numbers to trust"]
    if st == "error":
        return 0.05, ["a data tool failed"]
    if st == "multi":
        subs = [score_confidence(p.get("cat") or cat, p, task)[0] for p in facts.get("parts", []) if p.get("status") == "ok"]
        return (min(subs) if subs else 0.9), ["multi-part message: the weakest answered part sets the confidence"]
    conf, why = 0.90, ["produced by the specialist's playbook"]

    def minus(x: float, reason: str) -> None:
        nonlocal conf
        conf -= x
        why.append(f"-{x:.2f} {reason}")

    a = facts.get("assumed")
    n_ass = len(a) if isinstance(a, (dict, list)) else 0
    n_ass += 1 if facts.get("src_note") else 0
    if n_ass:
        minus(min(0.04 * n_ass, 0.20), f"{n_ass} assumption(s) made for the operator")
    if cat == "C" and ((facts.get("cl") or {}).get("src") == "hypothetical" or facts.get("src_note")):
        minus(0.10, "closure simulated; diversion share assumed")
    if cat == "P":
        minus(0.10, "load ranking is a proxy, not a capacity")
        if facts.get("mode") == "scenario":
            minus(0.15, "scenario outside the data window")
    if cat == "B":
        minus(0.15 if facts.get("found") else 0.20, "causes cannot be proven" if facts.get("found") else "no anomaly found")
    if cat == "A" and facts.get("conf") and "low" in str(facts["conf"]):
        minus(0.20, "event pattern from fewer than 5 past events")
    if cat == "G" and facts.get("pairs") and max(abs(p["r"]) for p in facts["pairs"]) < 0.3:
        minus(0.35, "correlations are weak (|r| < 0.3)")
    if cat in ("E", "H"):
        minus(0.05, "approximate passengers per line" if cat == "E" else "no origin-destination data")
    if task.route.ml_engine == "empirical" and cat in ("C", "P"):
        minus(0.03, "naive empirical baseline instead of TabPFN")
    if truth_ok:
        conf += 0.05
        why.append("+0.05 equals ground truth recomputed from the raw data")
    return round(max(0.05, min(0.99, conf)), 2), why


async def run_worker(task: WorkerTask, last_facts: dict | None, mcp, kb=None) -> WorkerResult:
    """Execute one task. Never raises: a failure becomes status 'error' (the evaluator / failsafe deal with it)."""
    from observability import CALL_LOG, payload, set_attr, span

    t0 = time.time()
    plan = task.to_legacy_plan()
    plan["engine"] = task.route.ml_engine if task.route.ml_engine == "empirical" else None       # tabpfn is the server default
    calls: list = []
    token = CALL_LOG.set(calls)                                    # every MCP call in this task (also the parallel ones) is logged with args and result preview
    try:
        with span("analyst.execute", **{"tmt.iteration": task.iteration, "tmt.category": task.category, "tmt.specialist": task.route.specialist, "tmt.mcp_servers": task.route.mcp_servers,
                                        "tmt.tools_planned": task.route.tools, "tmt.datasets": task.route.datasets, "tmt.ml_engine": task.route.ml_engine, "tmt.overrides": task.overrides,
                                        "tmt.task": payload(task.model_dump(exclude={"parts"}, mode="json"), 4000)}) as sp:
            facts, trace = await executor.execute(plan, task.question, last_facts, mcp)
            set_attr(sp, "tmt.facts_status", facts.get("status"))
            set_attr(sp, "tmt.facts", payload(facts, 4000))
            set_attr(sp, "tmt.n_mcp_calls", len(calls))
    finally:
        CALL_LOG.reset(token)
    truth_ok = None
    if kb is not None and facts.get("status") == "ok":
        try:
            tr = kb._truth_check(facts)
            truth_ok = None if tr is None else bool(tr[0])
        except Exception:
            truth_ok = None
    conf, why = score_confidence(task.category, facts, task, truth_ok)
    a = facts.get("assumed")
    assumptions = (list(a.values()) if isinstance(a, dict) else list(a or [])) + ([facts["src_note"]] if facts.get("src_note") else [])
    status = facts.get("status") if facts.get("status") in ("ok", "need", "unsupported", "oos", "follow", "multi", "error") else "error"
    return WorkerResult(task_id=task.task_id, iteration=task.iteration, status=status, facts=facts, confidence=conf, confidence_reasons=why, assumptions=[str(x) for x in assumptions],
                        datasets_used=task.route.datasets, ml_engine_used=task.route.ml_engine if task.route.ml_engine != "none" else "none",
                        tools_called=_tool_calls(calls, trace, t0), seconds=round(time.time() - t0, 2))


def _tool_calls(calls: list, trace: list, t0: float) -> list[ToolCall]:
    """MCP call records (with arguments and result previews) when the runtime logged them, else the executor's plain (tool, seconds) trace."""
    if calls:
        return [ToolCall(tool=c["tool"], server=c.get("server", "ubahn-flow-data"), seconds=float(c.get("seconds", 0.0)), args=c.get("args", {}), result_preview=c.get("result_preview", ""),
                         result_bytes=int(c.get("result_bytes", 0)), ok=bool(c.get("ok", True)), wait_ready_ms=int(c.get("wait_ready_ms", 0)), error=c.get("error"), started_at=c.get("start"))
                for c in sorted(calls, key=lambda c: c.get("start", 0))]
    return [ToolCall(tool=str(c.get("tool")), seconds=float(c.get("s", 0.0))) for c in trace]


def task_from_plan(plan, iteration: int = 1, overrides: dict | None = None) -> WorkerTask:
    """SupervisorPlan → WorkerTask, applying the evaluator's `overrides` (a closed list, see schemas.Adjustments)."""
    ent = plan.entities.model_copy(deep=True)
    route = plan.route.model_copy(deep=True)
    ov = {k: v for k, v in (overrides or {}).items() if v is not None}
    for k in ("dates", "times", "stations", "top_n", "rain", "duration_min"):
        if k in ov:
            setattr(ent, k, ov[k])
    if "ml_engine" in ov and route.ml_engine != "none":
        route.ml_engine = ov["ml_engine"]
    return WorkerTask(plan_id=plan.plan_id, iteration=iteration, question=plan.question, category=plan.category, objective=plan.objective, entities=ent, route=route, parts=plan.parts, overrides=ov)
