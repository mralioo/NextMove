"""SUPERVISOR: question → `SupervisorPlan` (category, parameters, objective, route, guardrails, history, follow-up).

What it does, in this order (all deterministic and ~1 ms unless the small-LLM router fallback is needed):

  1. classify + extract  (agent/router.py: rules → category and entities; multi-question messages are split into parts)
  2. INPUT GUARDRAILS    (agent/guardrails.py) — an unrelated question is BOUNCED: fixed reply, conversation cut, no worker and no LLM is started
  3. FOLLOW-UP           a continuation is not treated as a new question: mode `explain` answers from the earlier facts; mode `rerun` starts from
                         the earlier plan — same category, inherited parameters, the operator's change applied ("what about 18:00?")
  4. HISTORY             if an accepted answer to this question (same words or same subject, same data window, recent) exists in the turn store
                         (the local mirror of the Cognee session history) the plan says `answer_from_history` — no recomputation
  5. ASSIGN              objective (what a good answer must achieve), route (specialist, MCP servers, datasets, tools, ML engine), knowledge-base boundaries
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import router
import specialists
from config import CONFIG
from guardrails import BOUNCE_MESSAGE, LIMITS, check_input, check_route
from schemas import (Entities, FollowUp, GuardrailResult, HistoryHit, Objective, PartPlan, Route, SupervisorPlan)

ROUTER_CONF_MIN = float(os.environ.get("ROUTER_CONF_MIN", "0.55"))


def HISTORY_ON() -> bool:
    """Answer-from-history and cross-session restore are off for evaluations (TMT_HISTORY=off): a repeated question must be recomputed, not served from cache."""
    return os.environ.get("TMT_HISTORY", "on").lower() != "off"

# category -> (objective kind, statement template, success criteria the evaluator checks the result against)
OBJECTIVES = {
    "C": ("closure_impact", "Explain the {subject} closure{when}: the reason and duration when recorded (else say it is simulated as described), where passengers reroute, which stations come under pressure and where to deploy staff.",
          ["states the closure: recorded reason and duration, or that it was simulated as described", "gives a reroute (rail detour or replacement bus)", "names the stations under pressure or says why none",
           "labels the pressure figures as assumption-based estimates", "answers the look-ahead window if one was asked"]),
    "D": ("station_profile", "Give the station's weekday/weekend peak hour, compare it with the network mean, and predict the flow if a time was given.",
          ["states the peak hour and value", "compares with the network mean peak", "predicts the flow only for a time inside the data"]),
    "A": ("event_effect", "Describe the effect of {subject} on station flows: which stations feel it, by how much, when the surge peaks and where to put staff.",
          ["identifies the event or the venue pattern", "names the stations that feel it with numbers against a normal hour", "gives the surge timing / staff window", "says the venue-to-station link is inferred"]),
    "P": ("load_ranking", "Rank the {n} stations under the highest predicted load{when}, under the stated weather / event assumptions.",
          ["returns the requested number of stations, ranked", "states the scenario assumptions (date, rain, event)", "says this is a load proxy, not a platform capacity"]),
    "B": ("anomaly_cause", "Find the flow anomalies{when} and say which explanation (closure, event, weather, none) fits each.",
          ["gives station, time and observed vs usual passengers", "checks closures and events, otherwise says unexplained", "hedges causality"]),
    "E": ("energy_efficiency", "Rank the lines by energy per passenger and explain the worst one with data.",
          ["names the worst line with Wh per passenger", "compares with the best or other lines", "explains with data-based factors", "notes that passengers per line are approximated"]),
    "F": ("resilience_ranking", "Rank the stations whose closure fragments the network most, with the passengers affected per day.",
          ["ranks stations with passengers affected per day", "states the method or its assumption"]),
    "G": ("demand_coupling", "Find station pairs whose demand moves together without a direct connection.",
          ["gives pairs with correlation and distance", "says how weak or strong the correlation is against the noise level", "makes no causal claim"]),
    "H": ("reroute_evidence", "Say what the recorded closures show about passenger rerouting.",
          ["states whether rerouting is measurable, with observed/expected ratios", "says the data has no origin-destination paths"]),
}
DECLINE = Objective(kind="decline", statement="Explain plainly that the question cannot be answered from the data and offer what is possible.", success_criteria=["says it cannot be answered and why", "offers what is possible"])
BOUNCE = Objective(kind="bounce", statement="The question is unrelated to Berlin U-Bahn operations; reply with the fixed scope message.", success_criteria=[])
FOLLOW = Objective(kind="follow_up", statement="Answer the follow-up from the earlier analysis.", success_criteria=["uses only the earlier facts", "says which numbers are measured, modelled or assumed"])


def plan_key(cat: str, e: Entities) -> str:
    """Stable key for what a question is ABOUT (not how it is worded) — same subject, same key."""
    subject = {"c": cat, "l": sorted(e.lines), "s": sorted(e.stations), "d": e.dates, "t": e.times, "u": e.duration_min, "h": e.horizon_min, "v": e.venue, "e": e.event, "n": e.top_n,
               "r": e.rain, "w": e.what_if}
    return hashlib.sha1(json.dumps(subject, sort_keys=True).encode()).hexdigest()[:16]


def build_objective(cat: str, e: Entities) -> Objective:
    if cat not in OBJECTIVES:
        return DECLINE
    kind, tmpl, crit = OBJECTIVES[cat]
    short = lambda n: re.sub(r"^(S\+U|U|S)\s+|\s*\(Berlin\)", "", n).strip()
    when = f" on {e.dates[0]}" if e.dates else ""
    subject = (f"{e.lines[0] + ' ' if e.lines else ''}{' and '.join(short(s) for s in e.stations[:2])}".strip() or "closure") if cat == "C" else (e.event or e.venue or "the event")
    return Objective(kind=kind, statement=tmpl.format(subject=subject or "the", when=when, n=e.top_n or 3), success_criteria=list(crit))


def build_route(cat: str) -> Route:
    sp = specialists.SPECIALISTS.get(cat)
    if sp is None or sp.status == "planned":
        return Route(specialist="none", mcp_servers=[], datasets=[], tools=[], ml_engine="none")
    engine = ("tabpfn" if CONFIG.engine == "tabpfn" else "empirical") if sp.ml else "none"
    return Route(specialist=sp.name, mcp_servers=["ubahn-flow-data", "nextmove-knowledge"], datasets=list(sp.datasets), tools=list(sp.tools), ml_engine=engine)


def _delta(legacy: dict) -> dict:
    """The parameters a follow-up message states itself (what the operator changed)."""
    d = {}
    for k in ("dates", "times", "stations", "lines"):
        if legacy.get(k):
            d[k] = legacy[k]
    for k, name in (("n", "top_n"), ("rain", "rain"), ("dur_min", "duration_min"), ("horizon_min", "horizon_min")):
        if legacy.get(k) is not None:
            d[name] = legacy[k]
    return d


def _merge(prev: Entities, delta: dict, q: str = "") -> tuple[Entities, list[str], dict]:
    delta = dict(delta)
    if "times" in delta and len(prev.times) == 2 and len(delta["times"]) == 1:      # one new time next to a start/end pair: which of the two changed?
        if re.search(r"\b(end|ends|ending|ended|over|finish\w*)\b", q, re.I):
            delta["times"] = [prev.times[0], delta["times"][0]]
        elif re.search(r"\b(start\w*|begin\w*|kick\w*)\b", q, re.I):
            delta["times"] = [delta["times"][0], prev.times[1]]
    data = prev.model_dump()
    inherited = [k for k, v in data.items() if v not in (None, [], False, "") and k not in delta]
    for k, v in delta.items():
        data[k] = v
    return Entities(**data), inherited, delta


async def supervise(question: str, *, last: dict | None, last_facts: dict | None, kb=None, data_end: str | None = None, llm_fallback: bool = True) -> SupervisorPlan:
    """`last` is the previous accepted turn's SupervisorPlan dump (or None); `last_facts` its facts."""
    q = question.strip()
    has_history = bool(last_facts)
    # A message whose own rules already pick a category with confidence is a NEW question, not a follow-up (found by the v3 two-question run:
    # "... during the first day of the event" matched the follow-up pattern and the InnoTrans question was answered with the previous turn's closure).
    standalone = False
    if last and last_facts:
        r0 = router.route(q, has_history=False)
        standalone = r0["conf"] >= ROUTER_CONF_MIN and r0["cat"] not in ("OOS", "FOLLOW") and not router.WHY_FOLLOW.search(q)
    legacy = router.route(q, has_history=has_history and not standalone)
    tier = 0
    # ---- 2 · input guardrails FIRST: an unrelated question is bounced before any LLM call (the router fallback would cost a call and ~1.5 s)
    in_scope, guards, message = check_input(q, legacy, has_history)
    if not in_scope:
        return SupervisorPlan(question=q, decision="bounce", in_scope=False, category="BOUNCE", confidence=1.0, tier=tier, objective=BOUNCE, route=build_route("BOUNCE"),
                              guardrails=guards, message=message or BOUNCE_MESSAGE)

    router_error = None
    if llm_fallback and (CONFIG.router == "llm" or (legacy["conf"] < ROUTER_CONF_MIN and legacy["cat"] not in ("OOS", "FOLLOW") and CONFIG.router == "rules")):
        try:
            legacy = await router.llm_route(q, legacy, has_history)
            tier = 1
        except Exception as e:                                         # network / LLM trouble: keep the deterministic plan
            router_error = str(e)[:80]

    # ---- multi-question message
    if CONFIG.router == "rules" and legacy["cat"] != "FOLLOW":
        legacy = router.with_parts(legacy, q, has_history)
    ent = Entities.from_legacy(legacy)
    cat = legacy["cat"]
    follow = None

    # ---- 3 · follow-up: continue from the earlier plan (only for a message that is not a complete question of its own)
    if last and last_facts and not standalone:
        prev = SupervisorPlan.model_validate(last)
        delta = _delta(legacy)
        pure_ref = cat == "FOLLOW"
        short_change = len(q.split()) <= 16 and legacy["conf"] < ROUTER_CONF_MIN and bool(delta) and cat not in ("OOS", "BOUNCE")
        if pure_ref and (not delta or router.WHY_FOLLOW.search(q)):
            follow = FollowUp(mode="explain", parent_category=prev.category)
            cat = "FOLLOW"
        elif (pure_ref or short_change) and delta and prev.category in OBJECTIVES:
            ent, inherited, over = _merge(prev.entities, delta, q)
            follow = FollowUp(mode="rerun", parent_category=prev.category, inherited=inherited, overridden=over)
            cat = prev.category
            legacy = {**legacy, "cat": cat, "conf": 0.9, **router.CAT_DATA.get(cat, {}), **ent.to_legacy(), "parts": None}
            legacy.pop("parts", None)
            guards.append(GuardrailResult(stage="route", check="follow_up", passed=True, detail=f"continues the earlier {cat} question; changed: {sorted(over)}"))

    if follow is None and legacy["conf"] < 0.3 and cat not in ("OOS", "FOLLOW") and not legacy.get("parts"):
        cat = "OOS"                                              # no rule fired and the LLM router (if any) did not help: do not guess a category
        legacy = {**legacy, "cat": "OOS", "conf": 0.3, **router.CAT_DATA["OOS"]}
    key = plan_key(cat, ent)
    # ---- 4 · history: an accepted answer already exists
    hit = None
    if kb is not None and HISTORY_ON() and follow is None and cat not in ("OOS", "FOLLOW"):
        h = kb.find_answered(q, key, data_end)
        if h:
            hit = HistoryHit(**h)

    sp = specialists.SPECIALISTS.get(cat)
    guards.append(check_route(cat, sp.status if sp else None))
    parts = [PartPlan(text=p["text"], category=p["cat"], confidence=p["conf"], entities=Entities.from_legacy(p), route=build_route(p["cat"])) for p in (legacy.get("parts") or [])]
    if follow and follow.mode == "explain":
        decision, objective = "follow_up", FOLLOW
    elif hit:
        decision, objective = "answer_from_history", build_objective(cat, ent)
    elif cat in ("OOS",) or (sp is not None and sp.status == "planned") or sp is None and cat not in ("FOLLOW",):
        decision, objective = "decline", DECLINE
    else:
        decision, objective = "proceed", (FOLLOW if cat == "FOLLOW" else build_objective(cat, ent))
    boundaries = [e["id"] for e in kb.boundaries_for(legacy, q)] if kb is not None else []
    return SupervisorPlan(question=q, decision=decision, in_scope=True, category=cat, confidence=float(legacy["conf"]), tier=tier, entities=ent, objective=objective,
                          route=build_route(cat), guardrails=guards, history=hit, follow_up=follow, parts=parts, kb_boundaries=boundaries, llm_router_error=router_error)
