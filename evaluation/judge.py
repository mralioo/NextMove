"""LLM-as-judge for the evaluation harness (and the experiment suite).

The judge reads, for ONE answer:
    QUESTION      what the operator asked
    CRITERIA      the question-specific rubric, in plain language (what a good answer must contain)
    GROUND TRUTH  facts recomputed from the raw csv files (when the question has checkable ones)
    EVIDENCE      the facts the tools returned (the only material the answer is allowed to rely on)
    ANSWER        the text delivered to the operator
and returns, as JSON: per criterion `met` + a short quoted justification, 1-5 scores for relevance / faithfulness /
clarity / usefulness, and a list of claims it found unsupported by the evidence.

Why a judge and not regexes: a keyword rule cannot tell "the closure is due to a safety inspection" from "no safety
inspection was found", or a decline that offers help from one that merely apologises, or a German answer from a wrong one.
The deterministic checks that machines do better (numbers traceable to tool output, facts equal to ground truth,
latency, routing) stay deterministic; the semantic judgement moves here. The old regex rubric is kept only as a
calibration signal: `metrics.score_item` reports how often judge and regex agree.

Model: the small worker model by default (the shared main model is NOT used unless `model="main"`), temperature 0.
The judge is itself fallible (self-preference toward its own style, leniency): treat it as one instrument, read the
justifications, and use `judge_agreement` / spot checks with the main model to calibrate it.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Awaitable, Callable

CAPABILITIES = (
    "The assistant currently has tools for two question types: (C) a closure/suspension — reason, duration, reroute, stations "
    "under pressure, staff — and (D) one station's flow profile / peak vs the network mean / prediction at a time. It has NO tool yet "
    "for events, anomalies, energy, network resilience, correlations, reroute behaviour or investment questions; an honest decline "
    "that says so is the CORRECT answer to those. It must never state a platform/train capacity, invent numbers, or give figures "
    "for dates outside the data (2026-06-10 to 2026-09-22)."
)
ALLOWED_BACKGROUND = (
    "Statements the assistant is allowed to make even if not in EVIDENCE: the dataset covers 2026-06-10 to 2026-09-22 (15-minute grain); "
    "the 26 recorded closures show no measurable redistribution of passengers; diversion shares of 25/50/75 % are ASSUMPTIONS; "
    "'pressure' means the chance of exceeding a station's own busiest-5 % (p95) level for that hour; there is no capacity, headway or "
    "train-load data."
)

SYSTEM = """You are a strict, fair evaluator of answers written for a metro control-room operator. Judge ONLY from the material given.
Reply with ONE JSON object and nothing else:
{"criteria": {"<name>": {"met": true|false, "evidence": "<=20 words: a short quote from the ANSWER, or why it is missing"}},
 "relevance": 1-5, "faithfulness": 1-5, "clarity": 1-5, "usefulness": 1-5,
 "unsupported_claims": ["short verbatim claim from the ANSWER that neither EVIDENCE, GROUND TRUTH, the QUESTION nor the allowed background supports"],
 "comment": "<=25 words: the single biggest weakness, or 'none'"}
Rules:
- criterion "met" is true only if the ANSWER clearly satisfies it. A hedge, a vague mention or a wrong value is NOT met.
- relevance: 5 = addresses everything asked that the system can answer (or declines correctly what it cannot); 1 = off-topic.
- faithfulness: 5 = every claim is supported; 3 = minor unsupported wording; 1 = invents facts, states capacity, calls a model-based estimate 'measured', or contradicts EVIDENCE / GROUND TRUTH.
- clarity: 5 = short, plain, an operator under pressure could act on it; 1 = confusing or padded.
- usefulness: 5 = the operator gets what they need next; a correct decline that names a concrete alternative can score 3-4; a bare refusal 2.
- Numbers in the ANSWER that appear in EVIDENCE (or are their obvious rounding/percent form) are supported.
- Do not reward length. Do not penalise a correct decline for being short."""


def build_prompt(question: str, answer: str, facts: dict, criteria: dict[str, str], ground_truth: str | None) -> str:
    crit = "\n".join(f'- "{k}": {v}' for k, v in criteria.items()) or "- (no specific criteria: judge relevance, faithfulness, clarity, usefulness only)"
    return (f"CAPABILITIES OF THE SYSTEM:\n{CAPABILITIES}\n\nALLOWED BACKGROUND:\n{ALLOWED_BACKGROUND}\n\n"
            f"QUESTION:\n{question}\n\nCRITERIA (judge each by its name):\n{crit}\n\n"
            f"GROUND TRUTH (from the raw data files):\n{ground_truth or '(none for this question)'}\n\n"
            f"EVIDENCE (tool output the answer may rely on):\n{json.dumps(facts, ensure_ascii=False)[:3800]}\n\nANSWER:\n{answer}")


def _parse(raw: str) -> dict | None:
    try:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def normalise(j: dict, criteria: dict[str, str]) -> dict:
    """Validate/clean the judge's JSON: every criterion present as bool + evidence, scores clamped to 1..5."""
    got = j.get("criteria") if isinstance(j.get("criteria"), dict) else {}
    out = {"criteria": {}}
    for name in criteria:
        c = got.get(name)
        if isinstance(c, dict) and isinstance(c.get("met"), bool):
            out["criteria"][name] = {"met": c["met"], "evidence": str(c.get("evidence", ""))[:200]}
    for k in ("relevance", "faithfulness", "clarity", "usefulness"):
        v = j.get(k)
        out[k] = min(5, max(1, int(v))) if isinstance(v, (int, float)) else None
    out["unsupported_claims"] = [str(x)[:160] for x in (j.get("unsupported_claims") or []) if x][:6]
    out["comment"] = str(j.get("comment", ""))[:200]
    return out


async def _default_llm(prompt: str, model: str) -> tuple[str, str, int]:
    """(raw text, model name, tokens) from the small worker model (or the main one when model == 'main')."""
    import litellm

    from llm_config import litellm_params, sampling_params

    cfg = litellm_params("WRITER" if model == "main" else "ROUTER")        # ROUTER chain -> the worker model
    if cfg is None:
        raise RuntimeError("no LLM configured for the judge")
    name, kw = cfg
    r = await litellm.acompletion(model=name, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                  max_tokens=1200, timeout=45, response_format={"type": "json_object"}, **sampling_params(name, 0), **kw)
    u = getattr(r, "usage", None)
    return r.choices[0].message.content or "", name, (getattr(u, "total_tokens", 0) or 0)


async def judge_item(question: str, answer: str, facts: dict, criteria: dict[str, str], ground_truth: str | None = None, *,
                     model: str = "small", llm: Callable[[str], Awaitable[str]] | None = None) -> dict | None:
    """Judge one answer. Returns the normalised verdict (plus `model`, `tokens`) or None if the judge could not be reached /
    returned unusable JSON twice. `llm` (prompt -> raw JSON text) lets tests inject a fake judge."""
    prompt = build_prompt(question, answer, facts, criteria, ground_truth)
    for _attempt in range(2):
        try:
            if llm is not None:
                raw, name, tokens = await llm(prompt), "injected", 0
            else:
                raw, name, tokens = await _default_llm(prompt, model)
        except Exception:
            continue
        parsed = _parse(raw)
        if parsed is not None:
            v = normalise(parsed, criteria)
            v.update(model=name, tokens=tokens)
            return v
    return None


async def judge_many(jobs: list[dict], concurrency: int = 4, **kw) -> list[dict | None]:
    """Run several judge_item(**job) calls with bounded concurrency (small-model calls are cheap; be polite anyway)."""
    sem = asyncio.Semaphore(concurrency)

    async def one(job):
        async with sem:
            return await judge_item(**job, **kw)

    return await asyncio.gather(*(one(j) for j in jobs))
