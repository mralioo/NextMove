"""Stage 3 of the fast pipeline: WRITER + GUARD.

The writer is the ONLY generative step on the hot path: one LLM call that turns the executor's compact
facts JSON into a short, plain-language brief for the operator. Its prompt is static and tiny (so it can
be prompt-cached) and the facts are ~1k tokens instead of the ~9k of raw tool output the old
specialist read.

The guard replaces the old LLM verifier (which cost two extra LLM round trips and re-typed the whole
draft): it is deterministic and takes milliseconds. Every number in the answer must occur in the facts
(after rounding/percent conversion); if the writer invents one, the answer is replaced by a
template rendering of the same facts. Small integers (<= 12: counts, hours-of-day) are exempt.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import litellm   # imported at start-up: the lazy import costs ~2-3 s on the first question

WRITER_SYSTEM = """You write the final answer for a Berlin U-Bahn control-room operator. Use ONLY the FACTS json.
Plain words, short sentences, no jargon, no filler. Max 110 words. Lead with the answer.
Format:
**Answer:** 1-2 sentences.
**Key facts:** up to 4 bullets: the number and what it means.
**Do now:** up to 3 bullets, only if the question asks what to do.
**Caveat:** 1 sentence with the assumption or limit that matters.
Rules:
- Every number must appear in FACTS. Never invent, add, subtract or recompute numbers.
- status "need": ask for exactly the missing input, in one sentence (list options if given).
- status "unsupported"/"oos": max 60 words, no Key facts. One sentence: it cannot be answered from this dataset/tool yet (say why in plain words). Add "finding" if present. One sentence offering what IS possible (see "have"). Never list things that are unavailable as bullets.
- status "follow": answer the follow-up from facts.prev only. For "how confident / measured vs assumed" questions: measured = closure record, station graph, past flows; modelled = normal demand per station (TabPFN forecast); assumed = share of passengers who divert and where they go (codes A1-A5). Say which numbers are which and that pressure is a scenario, not a prediction of what will happen.
- Write pressure as: "X% chance of exceeding its own busiest-5% level (Y% normally)". It is NOT a capacity limit. Never use the word "capacity" except to say none is known, and never claim measured overload.
- alt.bus are only the closest station pairs across the cut: say "a replacement bus between A and B (about N km) would be needed". Never say a bus service exists.
- Never mention JSON, keys, tools or codes.
Keys, category C: cl{line,a,b=section ends,st=station,from,to,h=hours,why=reason,src}; cut{unserved=no service,partial=other line still runs,isolated_groups=sizes of station groups cut off,isolated_only_if_no_through_trains=same but only if trains skip the closed station (say so, or omit)}; note=extra assumption to mention; alt{rail=detours{stops=number of stops,extra_stops=stops more than the closed route (not minutes),via=transfer stations},bus=[from,to,km] closest cross-cut pairs, not a real service}; press[{s,at=peak time,base=normal passengers/15min,add=extra under closure,tot,p95,p=% chance above p95 with closure,p0=same without closure,lo/hi=% if fewer/more passengers divert}]; disp=demand normally at closed stations (avg/peak per 15min, tot over window); obs=what was actually recorded vs normal; assumed=defaults used.
Keys, category D: st[{s,avg_day,wk=[peak hour,avg passengers],we=[peak hour,passengers],net=network mean weekday peak,vs_net_pct,above,pred{at,pred,real,diff=real-pred,in_sample},risk{p%,real}}].
Assumption codes (mention as ONE plain caveat, not codes): A1 share of passengers who divert is assumed (25/50/75%); A2 displaced riders go to the nearest open stations; A3 pressure is relative to each station's own history; A4 the 26 past closures show no measurable redistribution, so this is a scenario not a replay; A5 no train-load or capacity data exists."""


# ------------------------------------------------------------------------------------ guard
_NUM = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)")


def _val(tok: str) -> float:
    """'1,363' -> 1363 (thousands separator) but '3,5' -> 3.5 (decimal comma)."""
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", tok):
        return float(tok.replace(",", ""))
    return float(tok.replace(",", "."))


# Fixed numbers the writer may quote because the caveat text defines them: the assumed diversion shares
# (25/50/75%), the 26 recorded closures, the busiest-5% (p95) definition, the dataset window, the
# 15-minute data grain.
STATIC_OK = {25, 50, 75, 26, 5, 10, 90, 95, 2026, 15, 60}   # + 15-minute bins, minutes per hour


def _allowed_numbers(facts: dict, question: str = "") -> set[float]:
    """Every number in the facts and in the operator's own question (+ derived: rounding, x100, minutes)."""
    txt = json.dumps(facts, ensure_ascii=False) + " " + question
    nums = {_val(m) for m in _NUM.findall(txt)}
    extra = set()
    for n in nums:
        extra.update({round(n), round(n, 1), n * 60, n / 60, round(n * 100)})
    return nums | extra | set(map(float, STATIC_OK))


def find_ungrounded(answer: str, facts: dict, question: str = "") -> list[str]:
    allowed = _allowed_numbers(facts, question)
    bad = []
    for tok in _NUM.findall(answer):
        v = _val(tok)
        if v <= 12 and float(v).is_integer():
            continue                                   # counts / hours-of-day
        if any(abs(v - a) <= max(0.51, 0.01 * abs(a)) for a in allowed):
            continue
        bad.append(tok)
    return bad


_BANNED = [
    (re.compile(r"(exceed\w*|above|over|beyond|reach\w*|surpass\w*)\s+(its |their |the |a )?(peak |platform |station |train |safe )?capacity|"
                r"capacity\s+(is|was|will be|would be|has been)\s+(exceeded|reached|breached)", re.I), "capacity claim"),
    (re.compile(r"bus(es)? (service|line)s? (is|are) (available|running|operating)", re.I), "bus service claim"),
]


def find_banned(answer: str) -> list[str]:
    """Phrases that make claims the data cannot support (capacity, existing bus service)."""
    return [label for pat, label in _BANNED if pat.search(answer)]


# ------------------------------------------------------------------------------------ template fallback
def render_fallback(facts: dict) -> str:
    st, cat = facts.get("status"), facts.get("cat")
    if st == "need":
        opts = facts.get("options")
        extra = f" Options: " + "; ".join(f"#{o['id']} {o['line'] or ''} {o['a']}→{o['b'] or ''} {o['from']}" for o in opts) if opts else ""
        return f"**Answer:** I need more detail: {', '.join(facts['missing'])}.{extra}"
    if st in ("unsupported", "oos"):
        return (f"**Answer:** This can't be answered from the current tools/dataset"
                f"{' (' + facts['topic'] + ')' if facts.get('topic') else ''}. "
                + (facts.get("finding", "") + " " if facts.get("finding") else "")
                + "**What I can do:** " + "; ".join(facts.get("have", [])))
    if cat == "C" and st == "ok":
        c = facts["cl"]
        sect = f"{c['a']} ↔ {c['b']}" if c.get("a") else c.get("st")
        lines = [f"**Answer:** {c.get('line') or 'Station'} closure {sect}, {c['from']} to {c['to']} ({c['h']} h), reason: {c['why']}."]
        alt = facts["alt"]
        if alt["rail"]:
            lines.append("**Rail detours:** " + "; ".join(f"{p['stops']} stops via {p['via']}" for p in alt["rail"]))
        else:
            lines.append("**No rail detour:** replacement buses needed. Closest cross-cut links: "
                         + "; ".join(f"{a}–{b} ({km} km)" for a, b, km in alt["bus"]))
        if facts.get("press"):
            lines.append("**Most pressured stations** (chance of exceeding their own p95, with vs without closure): "
                         + "; ".join(f"{r['s']} {r['p']}% vs {r['p0']}%" for r in facts["press"]))
        lines.append("**Caveat:** estimates rest on assumed passenger diversion; not a capacity measurement.")
        return "\n".join(lines)
    if cat == "D" and st == "ok":
        out = []
        for s in facts["st"]:
            if "error" in s:
                out.append(f"**{s['s']}:** {s['error']}")
                continue
            out.append(f"**{s['s']}:** weekday peak at {s['wk'][0]}:00 ({s['wk'][1]} passengers/15 min), "
                       f"{'above' if s['above'] else 'below'} the network mean of {s['net']} ({s['vs_net_pct']:+}%). "
                       f"Weekend peak {s['we'][0]}:00.")
        return "\n".join(out)
    return "**Answer:** I could not produce a grounded answer for this question."


# ------------------------------------------------------------------------------------ LLM call
def warm_connection() -> None:
    """Open the HTTPS connection to the writer's LLM endpoint ahead of the first question (best effort)."""
    from llm_config import litellm_params, sampling_params

    cfg = litellm_params("WRITER")
    if cfg:
        model, kw = cfg
        try:
            litellm.completion(model=model, messages=[{"role": "user", "content": "ok"}], max_tokens=64, timeout=20,
                               **sampling_params(model), **kw)
        except Exception:
            pass


async def write(question: str, facts: dict) -> tuple[str, dict]:
    """Returns (answer, info). info: model, tokens, guard result."""
    from llm_config import litellm_params, sampling_params

    cfg = litellm_params("WRITER")
    if cfg is None:
        return render_fallback(facts), {"guard": "no-llm"}
    model, kw = cfg
    user = f"QUESTION: {question}\nFACTS: {json.dumps(facts, ensure_ascii=False, separators=(',', ':'))}"
    budget = float(os.environ.get("WRITER_TIMEOUT_S", "10"))
    try:
        resp = await asyncio.wait_for(litellm.acompletion(
            model=model, messages=[{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=1500, timeout=budget + 5, **sampling_params(model), **kw), timeout=budget)
    except (asyncio.TimeoutError, Exception) as e:      # slow/failed LLM: ship the deterministic answer, never block
        return render_fallback(facts), {"model": model, "guard": f"template ({type(e).__name__}: LLM over {budget:.0f}s budget or failed)"}
    text = (resp.choices[0].message.content or "").strip()
    u = getattr(resp, "usage", None)
    info = {"model": model, "tok_in": getattr(u, "prompt_tokens", None), "tok_out": getattr(u, "completion_tokens", None)}
    bad, banned = find_ungrounded(text, facts, question), find_banned(text)
    if bad or banned or not text:
        return render_fallback(facts), {**info, "guard": f"fallback (ungrounded: {bad[:4]}, banned: {banned})"}
    return text, {**info, "guard": "pass"}
