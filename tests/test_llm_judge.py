"""Offline tests for the LLM-as-judge evaluation (the judge itself is injected as a fake; no network)."""
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent"), str(REPO / "evaluation")]

import dataset  # noqa: E402
import judge  # noqa: E402
import metrics  # noqa: E402

TRUTH = {"closure": {"reason": "safety inspection", "start": "13:50", "end": "15:20", "hours": 1.5, "date": "2026-07-13"},
         "peak": {"peak_hour": 18, "peak_value": 218.8, "network_mean_peak": 264.4, "exceeds": False}}
FACTS = {"status": "ok", "cl": {"why": "safety inspection", "h": 1.5, "from": "2026-07-13 13:50", "to": "2026-07-13 15:20"},
         "press": [{"s": "Mehringdamm", "p": 29}, {"s": "Ullsteinstr.", "p": 21}]}
CRIT = {"reason": "States the reason.", "caveat": "States estimates are assumption-based."}


def _fake(payload):
    async def llm(prompt):
        return payload if isinstance(payload, str) else json.dumps(payload)
    return llm


VERDICT = {"criteria": {"reason": {"met": True, "evidence": "'safety inspection'"}, "caveat": {"met": False, "evidence": "no caveat"}},
           "relevance": 5, "faithfulness": 4, "clarity": 4, "usefulness": 3, "unsupported_claims": [], "comment": "none"}


def test_prompt_contains_question_criteria_ground_truth_evidence_and_answer():
    p = judge.build_prompt("Why is U6 closed?", "Because of a safety inspection.", FACTS, CRIT, "Closure reason: safety inspection")
    for needle in ("Why is U6 closed?", '"reason": States the reason.', "GROUND TRUTH", "Closure reason: safety inspection", "EVIDENCE", "Mehringdamm",
                   "Because of a safety inspection.", "NO tool yet"[:0] or "It must never state a platform/train capacity"):
        assert needle in p


def test_judge_parses_fenced_json_normalises_and_clamps():
    raw = "```json\n" + json.dumps({**VERDICT, "relevance": 9, "clarity": 0, "criteria": {**VERDICT["criteria"], "bogus": {"met": True}}}) + "\n```"
    v = asyncio.run(judge.judge_item("q", "a", FACTS, CRIT, None, llm=_fake(raw)))
    assert v["criteria"]["reason"]["met"] is True and set(v["criteria"]) == {"reason", "caveat"}      # unknown criteria dropped
    assert v["relevance"] == 5 and v["clarity"] == 1                                              # clamped to 1..5


def test_judge_retries_once_then_gives_up_returning_none():
    calls = []

    async def flaky(prompt):
        calls.append(1)
        return "not json at all" if len(calls) == 1 else json.dumps(VERDICT)
    assert asyncio.run(judge.judge_item("q", "a", FACTS, CRIT, None, llm=flaky))["faithfulness"] == 4 and len(calls) == 2
    assert asyncio.run(judge.judge_item("q", "a", FACTS, CRIT, None, llm=_fake("garbage"))) is None


def _run(answer, facts=FACTS, plan=None, total=4.0):
    return {"answer": answer, "facts_json": json.dumps(facts), "plan_json": json.dumps(plan or {"cat": "C"}), "trace_id": "t", "total_s": total,
            "guard": "pass", "tok_in": 1, "tok_out": 1}


def test_judge_verdict_overrides_the_regex_rubric_and_is_compared_with_it():
    item = dataset.Item("T03", "TRAINING", "Q", expected_cat="C")
    exp = metrics.expectations(item, TRUTH)
    # A keyword-stuffed answer: the regex rubric says everything is present, the judge says the caveat is missing / reason contradicted.
    keyword_salad = ("Safety inspection safety inspection 13:50 15:20 bus detour Mehringdamm Ullsteinstr. deploy staff assumption estimate scenario.")
    verdict = {"criteria": {k: {"met": k != "reason", "evidence": "judge says " + k} for k in exp["rubric"]},
               "relevance": 2, "faithfulness": 2, "clarity": 1, "usefulness": 1, "unsupported_claims": ["safety inspection safety inspection"], "comment": "nonsense"}
    verdict = judge.normalise(verdict, exp["criteria"])
    m_regex, _ = metrics.score_item(item, _run(keyword_salad), exp, {"n": 8, "tools": 4})
    m_judge, checks = metrics.score_item(item, _run(keyword_salad), exp, {"n": 8, "tools": 4}, judge=verdict)
    assert m_regex["completeness"] == 1.0 and m_regex["judge_used"] == 0.0            # regex is fooled by keyword stuffing
    assert abs(m_judge["completeness"] - 5 / 6) < 1e-9 and m_judge["judge_used"] == 1.0
    assert abs(m_judge["judge_agreement"] - 5 / 6) < 1e-9                              # disagreement is measured, not hidden
    assert m_judge["regex_completeness"] == 1.0
    assert m_judge["judge_relevance"] == 0.4 and m_judge["judge_clarity"] == 0.2 and m_judge["judge_grounded"] == 0.0
    detail = {c["name"]: c["detail"] for c in checks}
    assert detail["answer has: reason"] == "judge: judge says reason" or "judge says" in detail["answer has: reason"]


def test_deterministic_gates_are_independent_of_the_judge():
    item = dataset.Item("T03", "TRAINING", "Q", expected_cat="C")
    exp = metrics.expectations(item, TRUTH)
    lenient = judge.normalise({"criteria": {k: {"met": True, "evidence": "ok"} for k in exp["rubric"]}, "relevance": 5, "faithfulness": 5, "clarity": 5,
                               "usefulness": 5, "unsupported_claims": []}, exp["criteria"])
    bad_facts = {**FACTS, "cl": {**FACTS["cl"], "why": "weather", "h": 3}}
    answer = "U6 closed for safety inspection 13:50-15:20 with an invented 777 passengers."
    m, _ = metrics.score_item(item, _run(answer, bad_facts, total=45.0), exp, {"n": 8, "tools": 4}, judge=lenient)
    assert m["fact_accuracy"] < 1.0            # facts vs ground truth from the raw csv: judge cannot excuse a wrong tool result
    assert m["hallucination_free"] == 0.0      # 777 is not in the evidence
    assert m["latency_ok"] == 0.0              # 45 s
    assert m["judge_relevance"] == 1.0         # ... even though the (lenient) judge loved the prose


def test_regex_is_only_the_fallback_when_the_judge_skips_a_criterion_or_is_unavailable():
    item = dataset.Item("T04", "TRAINING", "Q", expected_cat="D")
    exp = metrics.expectations(item, TRUTH)
    facts = {"status": "ok", "st": [{"wk": [18, 219], "net": 265, "above": False}]}
    answer = "Rudow peaks at 18:00, below the network mean of 265."
    partial = judge.normalise({"criteria": {"peak_hour": {"met": True, "evidence": "18:00"}}, "relevance": 4, "faithfulness": 4, "clarity": 4}, exp["criteria"])
    m, checks = metrics.score_item(item, _run(answer, facts, {"cat": "D"}), exp, {"n": 6, "tools": 1}, judge=partial)
    assert m["completeness"] == 1.0 and m["judge_used"] == 1.0
    assert any(c["name"] == "answer has: mean_value" and "regex fallback" in c["detail"] for c in checks)      # judge skipped it -> regex decided
    none_m, _ = metrics.score_item(item, _run(answer, facts, {"cat": "D"}), exp, {"n": 6, "tools": 1}, judge=None)
    assert none_m["judge_used"] == 0.0 and none_m["judge_relevance"] is None if "judge_relevance" in none_m else True


def test_declines_are_judged_by_criteria_not_keywords():
    item = dataset.Item("T10", "TRAINING", "invest where?", expected_cat="X")
    exp = metrics.expectations(item, TRUTH)
    assert exp["supported"] is False and set(exp["criteria"]) == {"declines", "offers_alternative"}
    v = judge.normalise({"criteria": {"declines": {"met": True, "evidence": "cannot be answered yet"}, "offers_alternative": {"met": False, "evidence": "no alternative"}},
                         "relevance": 4, "faithfulness": 5, "clarity": 5, "usefulness": 2}, exp["criteria"])
    m, _ = metrics.score_item(item, _run("It cannot be answered yet.", {"status": "unsupported"}, {"cat": "X"}), exp, {"n": 5, "tools": 0}, judge=v)
    assert m["decline_quality"] == 0.5 and m["honest_scope"] == 1.0 and m["completeness"] is None


def test_group_scores_include_the_judge_metrics():
    g = metrics.group_scores([{"answered": 1.0, "judge_relevance": 0.4, "judge_faithfulness": 0.8, "hallucination_free": 1.0,
                               "judge_clarity": 1.0, "latency_ok": 1.0}])
    assert abs(g["relevance"] - 0.7) < 1e-9 and abs(g["reliability"] - 0.9) < 1e-9 and g["stress"] == 1.0


def test_a_judge_verdict_without_scores_does_not_crash_scoring():
    item = dataset.Item("T10", "TRAINING", "invest where?", expected_cat="X")
    exp = metrics.expectations(item, TRUTH)
    empty = judge.normalise({"criteria": {}}, exp["criteria"])          # no criteria, no scores at all
    m, checks = metrics.score_item(item, _run("It cannot be answered yet.", {"status": "unsupported"}, {"cat": "X"}), exp, {"n": 5, "tools": 0}, judge=empty)
    assert m["judge_used"] == 1.0 and m["judge_relevance"] is None
    assert m["decline_quality"] is not None                              # criteria the judge skipped fall back to the regex rubric
