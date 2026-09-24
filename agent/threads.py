"""CONVERSATION THREADS: one conversation = one situation.

Why: the agent continues from the previous answer (follow-ups: "why?", "what about 22:30?", "and Neukölln?"). If the operator types an UNRELATED question into the same conversation, that
context can bend the answer (a short question inherits the wrong line, a rerun merges the wrong stations) and the graph gets a wrong branch. So the operator API asks — once, with two
buttons — whether the new message continues the situation or starts a new conversation, whenever `relation()` says the message is not clearly about the current situation.

`relation(question, anchor)` is deterministic (the router's own rules, no LLM, ~2 ms):
    related    the message refers to the last answer ("why?", "evidence"), or changes only a time / date / duration, or names the same line / station / event
    unrelated  a complete question of its own about another category, line, station or event
    unsure     short or vague and not tied to the situation ("and Rudow?", "what about tomorrow?")
Only `unrelated` and `unsure` produce a question to the operator.
"""
from __future__ import annotations

import re


def _short(n: str) -> str:
    return re.sub(r"\s*\(Berlin\)|^S\+U |^U |^S ", "", n).strip().lower()


def entity_set(ent: dict | None) -> set[str]:
    e = ent or {}
    out = {_short(s) for s in e.get("stations") or []} | {str(x).upper() for x in e.get("lines") or []}
    for k in ("venue", "event"):
        if e.get(k):
            out.add(str(e[k]).lower())
    return out


def relation(question: str, anchor: dict | None) -> dict:
    """Is `question` about the situation of `anchor` (see KnowledgeBase.session_anchor)?  -> {relation, reason, category, signals}"""
    import router

    q = question.strip()
    if not anchor:
        return {"relation": "related", "reason": "no earlier situation in this conversation", "signals": {}}
    r0 = router.route(q, has_history=False)
    rc = router.route(q, has_history=True)
    qents = {_short(s) for s in r0.get("stations") or []} | {str(x).upper() for x in r0.get("lines") or []}
    if r0.get("venue"):
        qents.add(str(r0["venue"]).lower())
    if r0.get("event"):
        qents.add(str(r0["event"]).lower())
    aents = entity_set(anchor.get("entities"))
    shared = qents & aents
    only_change = bool(r0.get("times") or r0.get("dates") or r0.get("dur_min") or r0.get("time")) and not qents
    sig = {"category": r0["cat"], "confidence": r0["conf"], "shared_entities": sorted(shared), "new_entities": sorted(qents - aents), "anchor_category": anchor.get("category")}
    if router.WHY_FOLLOW.search(q) or rc["cat"] == "FOLLOW":
        return {"relation": "related", "reason": "asks about the last answer", "signals": sig}
    try:
        from guardrails import check_input
        in_scope = check_input(q, r0, True)[0]
    except Exception:
        in_scope = True
    if r0["cat"] == "OOS" or not in_scope:
        return {"relation": "related", "reason": "not an operations question: answered on its own, the situation is not touched", "signals": sig}
    standalone = r0["conf"] >= 0.55
    if shared:
        return {"relation": "related", "reason": "names the same " + ("line" if any(s.startswith("U") and len(s) == 2 for s in shared) else "place") + f" ({', '.join(sorted(shared))})", "signals": sig}
    if only_change and not (standalone and r0["cat"] != anchor.get("category")):
        return {"relation": "related", "reason": "changes only the time, date or duration", "signals": sig}
    if standalone:
        why = []
        if r0["cat"] != anchor.get("category"):
            why.append(f"a different kind of question ({r0['cat']} instead of {anchor.get('category')})")
        if qents - aents:
            why.append("other " + ("stations / lines" if qents else "places") + f": {', '.join(sorted(qents - aents))}")
        if not qents and r0["cat"] == anchor.get("category"):
            return {"relation": "related", "reason": "same kind of question, no other place named", "signals": sig}
        return {"relation": "unrelated", "reason": "; ".join(why) or "a complete question of its own", "signals": sig}
    return {"relation": "unsure", "reason": "short message that does not name the current situation" + (f" but mentions {', '.join(sorted(qents))}" if qents else ""), "signals": sig}


def short_title(text: str, n: int = 80) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"
