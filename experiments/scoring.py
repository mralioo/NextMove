"""Scoring for the experiment suite — quantitative metrics per turn and per arm, plus the cross-arm comparison of the
ML engine's output. Reuses the evaluation harness metrics (grounding, readability, rubric completeness, fact accuracy vs
ground truth recomputed from the raw csv files) so experiments and evaluation speak the same language."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "evaluation"), str(REPO / "agent"), str(REPO)]
import dataset  # noqa: E402
import metrics as em  # noqa: E402

QUALITY_KEYS = ["route_correct", "completeness", "fact_accuracy", "hallucination_free", "readability_ok", "assumption_disclosed",
                "judge_relevance", "judge_faithfulness", "judge_clarity"]


def q1_item(truth: dict):
    item = dataset.Item("T03", "EXPERIMENT", "Q1", expected_cat="C")
    return item, em.expectations(item, truth)


def q2_item(q1_facts: dict):
    item = dataset.Item("X02", "EXPERIMENT", "Q2", expected_cat="FOLLOW")
    top = [p["s"].lower() for p in (q1_facts.get("press") or [])[:3]]
    exp = {"cat": "FOLLOW", "supported": True, "facts": [], "truth_text": None, "criteria": {
        "refers_to_the_earlier_stations": "Refers to at least one of the stations named in the EARLIER answer" + (f" ({', '.join(s.title() for s in top)})" if top else "") + " — the follow-up must build on the earlier analysis, not ask what it is about.",
        "gives_a_priority_order": "Gives an order or priority among those stations (which first, which next).",
        "states_how_sure_and_why": "Says how sure it is and why: the ranking is a model-based scenario built on assumed passenger diversion, not a measurement.",
    }, "rubric": {
        "refers_to_the_earlier_stations": lambda a, f: any(s in a.lower() for s in top) if top else False,
        "gives_a_priority_order": lambda a, f: bool(re.search(r"first|priorit|most|highest|top|then|followed|order|rank", a, re.I)),
        "states_how_sure_and_why": lambda a, f: bool(em.CAVEAT.search(a)),
    }}
    return item, exp


def expectation_for(turn: str, truth: dict, q1_facts: dict):
    """(item, expectation) for a protocol turn — the expectation carries the judge's criteria and ground-truth text."""
    return q2_item(q1_facts) if turn == "Q2" else q1_item(truth)


def score_turn(turn: str, run: dict, span_counts: dict, truth: dict, q1_facts: dict, judge: dict | None = None) -> tuple[dict, list]:
    item, exp = expectation_for(turn, truth, q1_facts)
    m, checks = em.score_item(item, run, exp, span_counts, judge=judge)
    m["memory_hit"] = float('"memory": "hit"' in (run.get("facts_json") or "") or '"memory":"hit"' in (run.get("facts_json") or ""))
    return m, checks


def quality(m: dict) -> float | None:
    vals = [m[k] for k in QUALITY_KEYS if m.get(k) is not None]
    return mean(vals) if vals else None


def arm_summary(turns: dict, runs: dict, startup_s: float | None) -> dict:
    """turns: {turn: (metrics, checks)}; runs: {turn: runs-row dict}."""
    q = {t: quality(m) for t, (m, _) in turns.items()}
    lat = {t: runs[t]["total_s"] for t in runs}
    facts = {t: json.loads(runs[t]["facts_json"] or "{}") for t in runs}
    jm = [mean(m[k] for k in ("judge_relevance", "judge_faithfulness", "judge_clarity") if m.get(k) is not None)
          for m, _ in turns.values() if any(m.get(k) is not None for k in ("judge_relevance", "judge_faithfulness", "judge_clarity"))]
    out = {
        "judge": mean(jm) if jm else None,
        "judge_agreement": (lambda v: mean(v) if v else None)([m["judge_agreement"] for m, _ in turns.values() if m.get("judge_agreement") is not None]),
        "scoring": "llm-judge + deterministic gates" if jm else "regex rubric (no judge)",
        "quality": mean(v for v in q.values() if v is not None) if any(v is not None for v in q.values()) else None,
        "quality_by_turn": q,
        "latency_s": lat, "latency_mean_s": mean(lat.values()),
        "route_ms": {t: runs[t]["route_ms"] for t in runs}, "tools_s": {t: runs[t]["tools_s"] for t in runs},
        "write_s": {t: runs[t]["write_s"] for t in runs},
        "tok_in": sum(runs[t]["tok_in"] or 0 for t in runs), "tok_out": sum(runs[t]["tok_out"] or 0 for t in runs),
        "llm_calls": sum(runs[t]["n_llm_calls"] or 0 for t in runs), "tool_calls": sum(runs[t]["n_tool_calls"] or 0 for t in runs),
        "guard": {t: runs[t]["guard"] for t in runs}, "mcp_startup_s": startup_s,
        "follow_up_answered": turns["Q2"][0].get("answered") if "Q2" in turns else None,
        "follow_up_completeness": turns["Q2"][0].get("completeness") if "Q2" in turns else None,
        "q1_completeness": turns["Q1"][0].get("completeness"), "q1_fact_accuracy": turns["Q1"][0].get("fact_accuracy"),
    }
    if "Q1" in runs and "Q1r" in runs:
        out["consistency_q1_q1r"] = em.consistency([runs["Q1"], runs["Q1r"]])
        out["speedup_q1r"] = 1 - lat["Q1r"] / lat["Q1"] if lat["Q1"] else None
        out["memory_hit_q1r"] = bool(turns["Q1r"][0].get("memory_hit"))
    out["top3_q1"] = [p["s"] for p in (facts.get("Q1", {}).get("press") or [])[:3]]
    return out


def facts_similarity(base: dict, other: dict) -> dict | None:
    """How much the pressure ranking of `other` (an arm's Q1 facts) differs from the baseline arm's."""
    a, b = base.get("press") or [], other.get("press") or []
    if not a or not b:
        return None
    sa, sb = [p["s"] for p in a], [p["s"] for p in b]
    shared = [s for s in sa if s in sb]
    pa, pb = {p["s"]: p["p"] for p in a}, {p["s"]: p["p"] for p in b}
    pairs = [(x, y) for i, x in enumerate(shared) for y in shared[i + 1:]]
    agree = [(sb.index(x) < sb.index(y)) for x, y in pairs]
    return {"top_station_same": sa[0] == sb[0], "top3_overlap": len(set(sa[:3]) & set(sb[:3])) / 3,
            "mean_abs_prob_diff_pts": mean(abs(pa[s] - pb[s]) for s in shared) if shared else None,
            "pair_order_agreement": (sum(agree) / len(agree)) if agree else None}
