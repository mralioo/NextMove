"""Guardrails and hard failsafes — deterministic, no LLM, ~1 ms.

Three layers, each returning `GuardrailResult`s that are stored with the run:

  INPUT     (supervisor)  length limit · prompt-injection markers · SCOPE filter: an unrelated question is BOUNCED (fixed reply, the conversation is
                          cut, no worker / LLM is started). A related question the system cannot answer (capacity, delays, costs, categories
                          with no specialist) is DECLINED with what is possible — it is not the same thing as unrelated.
  PROCESS   (worker loop) iteration cap · loop deadline · schema validation of every hand-over · tool error → escalate, never guess.
  OUTPUT    (writer)      no capacity / bus-service / measured-pressure claim · every number traceable to the facts · length cap ·
                          if the evaluator rejects the result, the answer is the SAFE FALLBACK, not the unverified numbers.

`Limits` are the hard failsafes; they can be tightened with environment variables, never bypassed by the model.
"""
from __future__ import annotations

import data_window

import os
import re
from dataclasses import dataclass

from schemas import GuardrailResult


@dataclass(frozen=True)
class Limits:
    max_question_chars: int = int(os.environ.get("GUARD_MAX_QUESTION_CHARS", "1500"))
    max_iterations: int = int(os.environ.get("LOOP_MAX_ITERS", "2"))
    loop_deadline_s: float = float(os.environ.get("LOOP_DEADLINE_S", "40"))
    max_answer_words: int = int(os.environ.get("GUARD_MAX_ANSWER_WORDS", "230"))
    min_confidence: float = float(os.environ.get("GUARD_MIN_CONFIDENCE", "0.35"))      # below this the answer is flagged as low-confidence


LIMITS = Limits()

INJECTION = re.compile(r"ignore (all |your |the |any )?(previous |prior |above )?(rules|instructions|guidelines|prompt)|disregard (the |your |all )?(rules|instructions)|"
                       r"(reveal|show|print|repeat) (me )?(your |the )?(system |hidden )?(prompt|instructions)|you are now|developer mode|jailbreak|"
                       r"pretend (to be|you are)|act as (if|an? )", re.I)
UNRELATED = re.compile(r"\b(recipes?|cook(ing)?|poems?|jokes?|song lyrics|horoscope|stock (price|market)|bitcoin|crypto\w*|capital of|who won|world cup|football score|"
                       r"translate|homework|essay|write (me )?(an? )?(code|program|script|function)|python|javascript|medical|diagnos\w+|lawsuit|dating|movies?|"
                       r"celebrit\w+|president|election|vacation|hotel|flight tickets?|weather (in|for) (paris|london|new york|rome|madrid|tokyo))\b", re.I)
STRONG_DOMAIN = re.compile(r"u-?bahn|\bmetro\b|subway|underground|\bu[1-9]\b|\bbvg\b|alstom|innotrans|\bmesse\b|passenger|commut\w*|ridership|platform|overcrowd\w*|"
                           r"station|bahnhof|fahrg\w+|closure|suspend\w*|disrupt\w*|re-?rout\w*|staff", re.I)
WEAK_DOMAIN = re.compile(r"\b(flow|line|train|event|concert|venue|arena|stadium|festival|energy|weather|rain|network|operator|anomal\w*|peak|dataset|crowd\w*|busiest|quiet\w*)\b", re.I)
GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|evening|afternoon)|help|what can you do)\W*$", re.I)

BOUNCE_MESSAGE = ("I only answer questions about Berlin U-Bahn passenger flows and operations: disruptions and reroutes, event impact, station profiles, "
                  "the busiest stations on a day, anomalies, energy per passenger, network resilience and correlations. "
                  "Your question is outside that, so I can't help with it here.")
SAFE_FALLBACK = ("I can't give a verified answer to this one: the result did not pass the automatic checks against the data. "
                 "Please rephrase it with a specific station, line, date and time inside " + data_window.window() + ", or ask about another topic I support.")


def check_input(question: str, route_plan: dict, has_history: bool = False) -> tuple[bool, list[GuardrailResult], str | None]:
    """(in_scope, results, bounce message). `route_plan` is the router's legacy plan dict (category, stations, lines, conf)."""
    res: list[GuardrailResult] = []
    q = question.strip()
    if not q:
        return False, [GuardrailResult(stage="input", check="non_empty", passed=False, action="bounce", detail="empty question")], BOUNCE_MESSAGE
    res.append(GuardrailResult(stage="input", check="length", passed=len(q) <= LIMITS.max_question_chars, action="allow" if len(q) <= LIMITS.max_question_chars else "bounce",
                               detail=f"{len(q)} chars (max {LIMITS.max_question_chars})"))
    if len(q) > LIMITS.max_question_chars:
        return False, res, BOUNCE_MESSAGE
    inj = INJECTION.search(q)
    res.append(GuardrailResult(stage="input", check="prompt_injection", passed=not inj, action="escalate" if inj else "allow",
                               detail=f"instruction to ignore the rules found ('{inj.group(0)}'): that part is refused, the rest is answered" if inj else ""))
    if GREETING.match(q):
        res.append(GuardrailResult(stage="input", check="scope", passed=False, action="bounce", detail="greeting / no question"))
        return False, res, BOUNCE_MESSAGE
    strong = bool(route_plan.get("stations") or route_plan.get("lines")) or bool(STRONG_DOMAIN.search(q))
    weak = len(WEAK_DOMAIN.findall(q))
    rules_fired = route_plan.get("conf", 0) >= 0.3 and route_plan.get("cat") not in ("OOS", "FOLLOW")
    unrelated = UNRELATED.search(q)
    short_follow = bool(has_history) and len(q.split()) <= 16 and not UNRELATED.search(q)        # 'What if it starts at 20:00?' continues the conversation
    in_scope = bool(has_history and route_plan.get("cat") == "FOLLOW") or short_follow or strong or (weak >= 2 and not unrelated) or (rules_fired and not unrelated)
    if unrelated and not strong:
        in_scope = False
    detail = ("unrelated topic ('%s') and no U-Bahn cue" % unrelated.group(0)) if (unrelated and not strong) else (
        "no U-Bahn / passenger-flow cue in the question" if not in_scope else "U-Bahn cue found" if strong else "domain words found")
    res.append(GuardrailResult(stage="input", check="scope", passed=in_scope, action="allow" if in_scope else "bounce", detail=detail))
    # an injection-only message is not a question: bounce; an injection next to a real question is refused per part
    if inj and not in_scope:
        return False, res, BOUNCE_MESSAGE
    return in_scope, res, (None if in_scope else BOUNCE_MESSAGE)


def check_route(category: str, specialist_status: str | None) -> GuardrailResult:
    if category in ("OOS", "BOUNCE"):
        return GuardrailResult(stage="route", check="supported_category", passed=False, action="allow", detail="related to the U-Bahn but not answerable from the data: declined with what is possible")
    if specialist_status in (None, "planned"):
        return GuardrailResult(stage="route", check="supported_category", passed=False, action="allow", detail=f"category {category} has no specialist yet: declined with what is possible")
    return GuardrailResult(stage="route", check="supported_category", passed=True, detail=f"specialist status: {specialist_status}")


def check_output(text: str, facts: dict, question: str) -> list[GuardrailResult]:
    import writer

    out = []
    banned, bad = writer.find_banned(text), writer.find_ungrounded(text, facts, question)
    words = len(text.split())
    out.append(GuardrailResult(stage="output", check="no_unsupported_claims", passed=not banned, action="allow" if not banned else "fallback", detail=", ".join(banned)))
    out.append(GuardrailResult(stage="output", check="numbers_grounded", passed=not bad, action="allow" if not bad else "fallback", detail=f"ungrounded: {bad[:4]}" if bad else ""))
    out.append(GuardrailResult(stage="output", check="length", passed=words <= LIMITS.max_answer_words, action="allow" if words <= LIMITS.max_answer_words else "escalate",
                               detail=f"{words} words (cap {LIMITS.max_answer_words})"))
    return out
