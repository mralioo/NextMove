"""EVALUATOR: decides whether the worker's result fulfils the question's objective — and, if not, what the worker should change.

Inputs (all typed, see schemas.py): the question, the supervisor's OBJECTIVE and category, the worker's RESULT (facts + confidence), and evidence it looks up itself
through the knowledge base (ground truth, boundaries, insights) and the knowledge graph (similar past problems with their accepted answers, actions and options).

Two stages:
  1. DETERMINISTIC checks (always, ~ms): the objective's required facts are present, the requested number of items came back, station names exist, the
     facts equal ground truth recomputed from the raw CSVs, dates are inside the data window, confidence is above the failsafe floor.
  2. LLM judgement (the SAME model role as the supervisor's "powerful" LLM — env EVALUATOR_LITELLM_MODEL overrides, else SUPERVISOR_*): does the result
     meet the objective's success criteria, and is it consistent with ground truth, boundaries and what was answered before? It may ask for a revision
     with `adjustments` from a CLOSED list (dates, times, stations, top_n, rain, duration_min, ml_engine).
     EVALUATOR_MODE: auto (default: only when stage 1 found an issue or confidence < 0.55) | always | off (deterministic only).

Verdict: accept · revise (the loop runs the worker again with the adjustments) · reject (the answer becomes the safe fallback).
"""
from __future__ import annotations

import json
import os
import re
import time

import router
from guardrails import LIMITS
from observability import payload, set_attr, span
from schemas import Adjustments, CheckResult, EvaluatorVerdict, KGCase, SupervisorPlan, WorkerResult, WorkerTask

MODE = os.environ.get("EVALUATOR_MODE", "auto")
LLM_CONF_BELOW = float(os.environ.get("EVALUATOR_CONF_BELOW", "0.55"))

SYSTEM = """You are the INSPECTOR (evaluator) in a Berlin U-Bahn operator assistant. The Analyst produced a result for a question; decide if it fulfils the objective.
You get: QUESTION, OBJECTIVE (statement + success criteria), RESULT (compact facts JSON + confidence + reasons), CHECKS (deterministic checks already run),
KNOWLEDGE (verified ground truth, boundaries, insights) and SIMILAR CASES from the knowledge graph (past problem, accepted answer, actions taken, options).
Reply with ONE JSON object and nothing else:
{"verdict":"accept|revise|reject","objective_met":true|false,"score":0-1,"issues":["short"],"adjustments":{"dates":["YYYY-MM-DD"],"times":["HH:MM"],"stations":["exact station name"],"top_n":3,"rain":true,"duration_min":60,"ml_engine":"tabpfn|empirical"} or null,"rationale":"<=40 words"}
Rules:
- accept only if every success criterion is met by the RESULT (a decline / a request for missing input is acceptable when the RESULT status says so and the question really lacks that input or the data).
- The RESULT must not contradict KNOWLEDGE (ground truth, boundaries) or, without a stated reason, the SIMILAR CASES. A missing capacity figure is never a defect: no capacity data exists.
- revise only when changing parameters would fix a concrete defect (wrong count of items, a horizon not answered, a cross-check with ml_engine=empirical for a low-confidence ML result). Put ONLY the changed keys in adjustments.
- reject when the objective cannot be met or the result contradicts ground truth.
- Do not invent numbers or stations."""


# ----------------------------------------------------------------------------------------------- stage 1
def _requirements(cat: str, facts: dict, plan: SupervisorPlan) -> list[CheckResult]:
    e = plan.entities
    out: list[CheckResult] = []

    def need(id_: str, ok: bool, detail: str) -> None:
        out.append(CheckResult(id=id_, ok=bool(ok), detail=detail))

    if facts.get("status") != "ok":
        return out
    if cat == "C":
        need("R-CLOSURE", bool(facts.get("cl")), "the closure (recorded or simulated) is stated")
        need("R-REROUTE", bool(facts.get("reroute") or (facts.get("alt") or {}).get("rail") or (facts.get("alt") or {}).get("bus")), "a reroute is given")
        need("R-PRESSURE", bool(facts.get("press") or facts.get("no_pressure_reason") or facts.get("scenario_error")), "stations under pressure, or the reason there are none")
        if e.horizon_min and (facts.get("cl") or {}).get("src") == "hypothetical":
            need("R-HORIZON", bool(facts.get("h")) or not facts.get("press"), f"the {e.horizon_min}-minute look-ahead is answered")
    elif cat == "P":
        want = e.top_n or 3
        need("R-COUNT", len(facts.get("top") or []) >= min(want, 3), f"{want} station(s) requested, {len(facts.get('top') or [])} returned")
        need("R-ASSUMPTIONS", bool(facts.get("assumed")) or facts.get("mode") == "replay", "scenario assumptions are stated")
    elif cat == "A":
        need("R-EVENT", bool(facts.get("top")) or bool(facts.get("note")), "affected stations, or a note why none")
        if e.stations:
            need("R-STATION", bool(facts.get("st")), "the station the operator asked about is answered")
    elif cat == "B":
        need("R-FOUND", bool(facts.get("found")) or bool(facts.get("note")), "anomalies, or a note why none")
    elif cat == "E":
        need("R-ENERGY", bool(facts.get("worst")) and bool(facts.get("rank")), "worst line and ranking")
    elif cat == "F":
        need("R-COUNT", len(facts.get("top") or []) >= min(e.top_n or 5, 5), "requested stations returned")
    elif cat == "G":
        need("R-PAIRS", bool(facts.get("pairs")), "station pairs returned")
    elif cat == "H":
        need("R-EVIDENCE", bool(facts.get("obs_over_exp")), "observed/expected ratios returned")
    elif cat == "D":
        need("R-STATION", bool(facts.get("st")) and all("error" not in s for s in facts["st"]), "profile for every asked station")
    return out


def deterministic(question: str, task: WorkerTask, result: WorkerResult, plan: SupervisorPlan, kb) -> tuple[list[CheckResult], list[str], list[str]]:
    """(checks, ground-truth ids, boundary ids)."""
    facts, cat = result.facts, plan.category
    checks = _requirements(cat, facts, plan)
    gt, bnd = [], []
    if facts.get("status") == "ok":
        names = {router._norm(n) for ns in router.station_index().values() for n in ns} | set(router.station_index())
        found: list[str] = []

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in ("s", "a", "b", "station") and isinstance(v, str):
                        found.append(v)
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk({k: v for k, v in facts.items() if k not in ("prev", "note", "lim", "assumed", "kb")})
        missing = [f for f in dict.fromkeys(found) if router._norm(f) not in names]
        checks.append(CheckResult(id="S-STATIONS", ok=not missing, detail=f"unknown stations: {missing[:4]}" if missing else f"{len(set(found))} station names exist"))
        if kb is not None:
            tr = kb._truth_check(facts)
            if tr is not None:
                checks.append(CheckResult(id="S-TRUTH", ok=bool(tr[0]), detail=tr[1]))
                gt += tr[2]
    checks.append(CheckResult(id="F-CONFIDENCE", ok=result.confidence >= LIMITS.min_confidence, detail=f"confidence {result.confidence} (floor {LIMITS.min_confidence})"))
    if result.status == "error":
        checks.append(CheckResult(id="F-TOOLS", ok=False, detail=str(facts.get("note", "a data tool failed"))[:160]))
    if kb is not None:
        bnd = [e["id"] for e in kb.boundaries_for(task.to_legacy_plan(), question)]
    return checks, gt, bnd


async def _quality_checks(category: str, result: WorkerResult, checks: list[CheckResult], bnd: list[str]) -> list[str]:
    """The Inspector asks the QUALITY MCP server (agent/quality_mcp.py → mcp_server/quality_server.py) to verify the facts against the normalized data and its boundaries.
    Hard findings (an impossible or contradicting figure) become failed `Q-H*` checks; soft findings (unusual, not impossible) are returned as issue texts; the boundaries used are added
    to the verdict's boundary ids. No database / no answer: nothing is added and nothing is blocked."""
    if result.facts.get("status") != "ok" or category not in ("A", "B", "C", "D", "F", "H", "P"):
        return []
    import quality_mcp
    with span("inspector.quality_mcp", **{"tmt.category": category}) as sp:
        res = await quality_mcp.call("quality_check_facts", category=category, facts_json=json.dumps(result.facts, default=str))
        if not res or not res.get("available"):
            set_attr(sp, "tmt.available", False)
            return []
        flags = []
        for c in res.get("checks", []):
            if c["hard"]:
                checks.append(CheckResult(id=c["id"], ok=bool(c["ok"]), detail=c["detail"][:200]))
            elif not c["ok"]:
                flags.append("quality flag: " + c["detail"][:170])
        for b in res.get("boundaries", []):
            if b["id"] not in bnd and len(bnd) < 12:
                bnd.append(b["id"])
        set_attr(sp, "tmt.summary", res.get("summary"))
        set_attr(sp, "tmt.checks", [{"id": c["id"], "ok": c["ok"], "hard": c["hard"], "detail": c["detail"][:120]} for c in res.get("checks", [])][:14])
        return flags[:3]


# ----------------------------------------------------------------------------------------------- stage 2
def _valid_adjustments(raw, plan: SupervisorPlan) -> Adjustments | None:
    if not isinstance(raw, dict):
        return None
    clean = {}
    if isinstance(raw.get("dates"), list):
        d = [x for x in raw["dates"] if isinstance(x, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", x)]
        if d:
            clean["dates"] = d
    if isinstance(raw.get("times"), list):
        t = [x for x in raw["times"] if isinstance(x, str) and re.fullmatch(r"\d{2}:\d{2}", x)]
        if t:
            clean["times"] = t
    if isinstance(raw.get("stations"), list):
        exact = {n for ns in router.station_index().values() for n in ns}
        s = [x for x in raw["stations"] if x in exact]
        if s:
            clean["stations"] = s
    if isinstance(raw.get("top_n"), int) and 1 <= raw["top_n"] <= 10:
        clean["top_n"] = raw["top_n"]
    if isinstance(raw.get("rain"), bool):
        clean["rain"] = raw["rain"]
    if isinstance(raw.get("duration_min"), int) and 5 <= raw["duration_min"] <= 24 * 60:
        clean["duration_min"] = raw["duration_min"]
    if raw.get("ml_engine") in ("tabpfn", "empirical") and plan.route.ml_engine != "none":
        clean["ml_engine"] = raw["ml_engine"]
    adj = Adjustments(**clean)
    return None if adj.is_empty() else adj


async def _llm(question: str, plan: SupervisorPlan, result: WorkerResult, checks: list[CheckResult], kb, cases: list[KGCase]) -> tuple[dict | None, str]:
    import litellm

    from llm_config import litellm_params, sampling_params

    cfg = litellm_params("EVALUATOR")
    if cfg is None:
        return None, "no evaluator LLM configured"
    model, kw = cfg
    know = [{"id": e["id"], "kind": e["kind"], "text": e["text"][:260]} for e in (kb.search(question, cats=[plan.category], k=3) if kb else [])]
    know += [{"id": e["id"], "kind": "boundary", "text": e["text"][:200]} for e in (kb.boundaries_for({"cat": plan.category, "oos": plan.entities.out_of_scope_terms, "dates": plan.entities.dates}, question) if kb else [])][:3]
    user = json.dumps({"QUESTION": question, "OBJECTIVE": plan.objective.model_dump(), "RESULT": {"status": result.status, "facts": result.facts, "confidence": result.confidence,
                                                                                                  "confidence_reasons": result.confidence_reasons, "assumptions": result.assumptions},
                       "CHECKS": [c.model_dump() for c in checks], "KNOWLEDGE": know, "SIMILAR_CASES": [c.model_dump(exclude={"problem_id"}) for c in cases]}, ensure_ascii=False, default=str)[:9000]
    with span("inspector.llm", **{"gen_ai.request.model": model, "tmt.prompt_chars": len(user), "tmt.prompt": payload(user, 5000)}) as sp:
        from observability import log_llm

        t0 = time.time()
        try:
            resp = await litellm.acompletion(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], max_tokens=1200, timeout=30,
                                             response_format={"type": "json_object"}, **sampling_params(model, 0), **kw)
            text = resp.choices[0].message.content or ""
            u = getattr(resp, "usage", None)
            log_llm("evaluator", model, time.time() - t0, u, user, text, started=t0)
            set_attr(sp, "tmt.seconds", round(time.time() - t0, 3))
            set_attr(sp, "gen_ai.usage.input_tokens", getattr(u, "prompt_tokens", None))
            set_attr(sp, "gen_ai.usage.output_tokens", getattr(u, "completion_tokens", None))
            set_attr(sp, "tmt.response", payload(text, 3000))
            return json.loads(text[text.index("{"): text.rindex("}") + 1]), model
        except Exception as e:                                                # an unreachable evaluator must not block the answer: deterministic verdict
            set_attr(sp, "tmt.error", f"{type(e).__name__}")
            log_llm("evaluator", model, time.time() - t0, prompt=user, error=type(e).__name__, started=t0)
            return None, f"{type(e).__name__}"


async def evaluate(question: str, plan: SupervisorPlan, task: WorkerTask, result: WorkerResult, kb=None, graph=None, allow_llm: bool = True) -> EvaluatorVerdict:
    """The verdict, recorded as an `inspector.check` span: objective, checks, evidence used, verdict, model, issues and adjustments."""
    with span("inspector.check", **{"tmt.iteration": task.iteration, "tmt.objective": plan.objective.statement, "tmt.success_criteria": plan.objective.success_criteria,
                                     "tmt.worker_confidence": result.confidence, "tmt.worker_status": result.status}) as sp:
        v = await _evaluate(question, plan, task, result, kb, graph, allow_llm)
        set_attr(sp, "tmt.verdict", v.verdict)
        set_attr(sp, "tmt.objective_met", v.objective_met)
        set_attr(sp, "tmt.score", v.score)
        set_attr(sp, "tmt.model", v.model)
        set_attr(sp, "tmt.llm_used", not v.model.startswith("deterministic"))
        set_attr(sp, "tmt.checks", [{"id": c.id, "ok": c.ok, "detail": c.detail[:120]} for c in v.checks])
        set_attr(sp, "tmt.issues", v.issues)
        set_attr(sp, "tmt.adjustments", v.adjustments.model_dump(exclude_none=True) if v.adjustments else None)
        set_attr(sp, "tmt.ground_truth_ids", v.ground_truth_ids)
        set_attr(sp, "tmt.boundary_ids", v.boundary_ids)
        set_attr(sp, "tmt.similar_cases", [{"problem": c.problem[:80], "similarity": c.similarity, "actions": c.actions[:2]} for c in v.similar_cases])
        set_attr(sp, "tmt.rationale", v.rationale)
        return v


async def _evaluate(question: str, plan: SupervisorPlan, task: WorkerTask, result: WorkerResult, kb=None, graph=None, allow_llm: bool = True) -> EvaluatorVerdict:
    t0 = time.time()
    checks, gt, bnd = deterministic(question, task, result, plan, kb)
    q_flags = await _quality_checks(plan.category, result, checks, bnd)
    cases: list[KGCase] = []
    if graph is not None and plan.category in ("A", "B", "C", "D", "E", "F", "G", "H", "P"):
        try:
            cases = graph.similar(question, plan.category, plan.entities, k=2)
        except Exception:
            cases = []
    failed = [c for c in checks if not c.ok]
    hard = [c for c in failed if c.id.startswith(("S-", "F-TOOLS", "Q-H"))]           # Q-H* = a quality-database boundary that is impossible / contradicts the raw data
    st = result.status
    if st in ("need", "unsupported", "oos", "follow"):                        # a decline / request for input: the decision is what is checked
        return EvaluatorVerdict(task_id=task.task_id, iteration=task.iteration, verdict="accept", objective_met=True, score=0.95, checks=checks, ground_truth_ids=gt, boundary_ids=bnd,
                                similar_cases=cases, rationale=f"status '{st}' is a valid outcome for this question", seconds=round(time.time() - t0, 3))
    score = round(sum(c.ok for c in checks) / len(checks), 2) if checks else 0.9
    verdict, met, issues, adj, rationale, model = "accept", True, [c.detail for c in failed] + q_flags, None, "all deterministic checks passed", "deterministic"
    if hard:
        verdict, met, rationale = ("revise" if st == "error" else "reject"), False, "a hard check failed: " + "; ".join(c.id for c in hard)
    elif failed:
        verdict, met, rationale = "revise", False, "objective requirements not fully met"
    use_llm = MODE == "always" or (MODE == "auto" and (bool(failed) or result.confidence < LLM_CONF_BELOW))
    if use_llm and allow_llm and st in ("ok", "multi"):
        j, m = await _llm(question, plan, result, checks, kb, cases)
        if j is not None:
            model = m
            v = j.get("verdict") if j.get("verdict") in ("accept", "revise", "reject") else verdict
            if hard and v == "accept":
                v = verdict                                                  # the model may not override a failed ground-truth / station check
            met = (v == "accept") and bool(j.get("objective_met", True)) and not hard
            issues = list(dict.fromkeys(issues + [str(x)[:160] for x in (j.get("issues") or [])]))[:6]
            adj = _valid_adjustments(j.get("adjustments"), plan) if v in ("revise", "reject") else None
            if v == "reject" and not hard:
                # an LLM opinion alone may not destroy a result that passed every deterministic / ground-truth check: it becomes a revision when it names a
                # concrete change, else an acceptance with the concern recorded (and the low confidence already shows in the answer)
                v = "revise" if adj is not None else "accept"
                met = v == "accept"
                issues.append("evaluator LLM voted to reject; overridden because every deterministic and ground-truth check passed")
            verdict = v
            rationale = str(j.get("rationale", ""))[:300]
            try:
                score = round(min(1.0, max(0.0, float(j.get("score", score)))), 2)
            except (TypeError, ValueError):
                pass
        else:
            model = f"deterministic (evaluator LLM unavailable: {m})"
    if verdict == "revise" and adj is None and st != "error":
        verdict, met = ("accept", False) if not hard and score >= 0.5 else ("reject", False)    # nothing concrete to change: do not loop for nothing (accepted = shown with the issues recorded)
        rationale += " · no concrete adjustment: " + ("accepted with the issues recorded" if verdict == "accept" else "rejected")
    return EvaluatorVerdict(task_id=task.task_id, iteration=task.iteration, verdict=verdict, objective_met=met, score=score, issues=issues, adjustments=adj, checks=checks,
                            ground_truth_ids=gt, boundary_ids=bnd, similar_cases=cases, rationale=rationale, model=model, seconds=round(time.time() - t0, 2))
