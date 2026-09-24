"""The operator's default answer is a brief; 'why / evidence / which tools / full report' gets the long report built from a stored artifact bundle.
Offline: no LLM, no network."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent")]

import artifacts  # noqa: E402
import guardrails  # noqa: E402
import kgraph  # noqa: E402
import knowledge  # noqa: E402
import router  # noqa: E402
import writer  # noqa: E402

LONG = """**Verdict:** The U7 closure is for track maintenance. It lasts from 20:45 to 22:45 on 2026-09-25 for 2 hours and pressure is highest at Neukölln and Hermannplatz right after the start.

**Evidence:**
- Rathaus Neukölln is unserved. [closures.csv]
- Neukölln: 117 passengers. [TabPFN forecast]

**Do now:**
- Deploy staff at Neukölln and Hermannplatz.
- Also cover Schönleinstr. and Südstern and Boddinstr.
- Provide the listed replacement buses between the two closed stations.
- A fourth bullet that must be dropped.

**Caveat:** Diversion shares are assumed; pressure is a scenario, not a measurement, and no capacity data exists at all in this dataset.
**Sources:** data: closures"""


def test_shorten_keeps_verdict_three_actions_and_one_watch_out():
    b = writer.shorten(LONG)
    assert b.startswith("**Verdict:**") and "**Evidence" not in b and "**Sources" not in b and "fourth bullet" not in b
    assert 1 <= b.count("\n- ") <= 3 and "**Watch out:**" in b and len(b.split()) <= 80


def test_detail_and_action_requests_are_recognised():
    for q in ("why?", "Why do you say that", "which tools did you call?", "show me the evidence", "what function did you use", "give me the full report", "how did you calculate that", "sources?"):
        assert writer.wants_detail(q), q
    for q in ("Which stations get the most pressure?", "Identify three anomalies that cannot be explained by closures", "At what time does the peak at Rudow take place?"):
        assert not writer.wants_detail(q), q
    assert writer.wants_action("where should additional staff be deployed?") and not writer.wants_action("At what time does the commute flow peak at Rudow?")
    assert router.WHY_FOLLOW.search("which tools did you call?") and router.WHY_FOLLOW.search("show me the evidence") and not router.WHY_FOLLOW.search("Which line is the worst?")


def test_confidence_footer_and_brief_length_cap():
    assert "high (95%)" in writer.confidence_footer(0.95) and "medium (60%)" in writer.confidence_footer(0.6) and "low (30%)" in writer.confidence_footer(0.3)
    assert writer.confidence_footer(None) == ""
    long_answer = "word " * 150
    assert not guardrails.check_output(long_answer, {}, "q", brief=True)[2].passed
    assert guardrails.check_output(long_answer, {}, "q", brief=False)[2].passed


def _art():
    timing = {"calls": [{"tool": "scenario_flow", "server": "ubahn-flow-data", "s": 3.9, "bytes": 5600, "args": {"closure_id": 1}, "round": 1}], "loop": [{}], "stages": {"total_s": 8.5, "supervisor_s": 0.06,
              "worker_evaluator_s": 6.2, "writer_s": 2.2}, "llm": [{"role": "writer", "model": "m", "seconds": 2.1, "tok_in": 100, "tok_out": 20}], "guard": "pass",
              "handover": {"checks": [{"id": "S-TRUTH", "ok": True, "detail": "matches the closure record"}], "similar_cases": []}}
    return artifacts.build(question="U7 closure?", session_id="s", plan={"category": "C", "route": {"specialist": "disruption", "datasets": ["closures"]}, "objective": {"statement": "Explain it"},
                                                                  "entities": {"lines": ["U7"]}, "decision": "proceed"},
                           result={"confidence": 0.95, "confidence_reasons": ["+0.05 equals ground truth"], "assumptions": ["25/50/75 % diversion"], "ml_engine_used": "tabpfn", "datasets_used": ["closures"]},
                           verdict={"verdict": "accept", "score": 1.0, "ground_truth_ids": ["GT-CL-1"], "boundary_ids": ["B-ASSUME"]}, facts={"status": "ok", "cat": "C"}, timing=timing, refs=[],
                           brief="**Verdict:** x", sources_line="**Sources:** data: closures", sanity={"ok": True}, source="worker", answer_mode="brief", data_window="2026-06-10 to 2026-10-01")


def test_report_method_section_comes_only_from_the_stored_bundle():
    art = _art()
    rep = artifacts.full_report(art, "**Verdict:** long text")
    assert "`scenario_flow(closure_id=1)`" in rep and "ubahn-flow-data" in rep and "✓ S-TRUTH" in rep and "GT-CL-1" in rep and "TabPFN" in rep and "Confidence 0.95 because" in rep
    assert rep.rstrip().endswith("**Sources:** data: closures")
    assert "not recorded" in artifacts.appendix({"legacy": True})
    assert len(artifacts.memory_summary(art)) <= 1400 and "scenario_flow" in artifacts.memory_summary(art)


def test_operator_knowledge_base_stores_finds_and_completes_artifacts(tmp_path):
    kb = knowledge.KnowledgeBase(db=tmp_path / "m.db")
    art = _art()
    tid = kb.remember_turn("s1", "op", "U7 closure staff?", "C", "brief", {"status": "ok"}, verdict="accept", confidence=0.95, accepted=True, artifact=art)
    got = kb.get_artifact(tid)
    assert got["turn_id"] == tid and got["tools"][0]["tool"] == "scenario_flow" and got["report"] is None
    assert kb.last_artifact("op")["turn_id"] == tid
    hit = kb.find_artifacts("U7 closure staff deploy", k=1)[0]
    assert hit["turn_id"] == tid and hit["has_report"] is False and hit["tools"] == ["scenario_flow"]
    kb.attach_report(tid, "THE REPORT")
    assert kb.get_artifact(tid)["report"] == "THE REPORT" and kb.find_artifacts("U7 closure staff", k=1)[0]["has_report"] and kb.stats()["artifacts_with_report"] == 1


def test_knowledge_graph_links_the_artifact_to_tools_datasets_model_and_kb_entries(tmp_path):
    g = kgraph.KnowledgeGraph(tmp_path / "kg.db")
    art = _art()
    art["turn_id"] = 7
    art["tools"].append({"tool": "alternate_paths", "server": "ubahn-flow-data", "seconds": 0.1})
    g.node("Problem", "pk1", text="q")
    g.record_artifact("pk1", art)
    s = g.stats()
    assert s["by_label"]["Artifact"] == 1 and s["by_label"]["Tool"] == 2 and s["by_label"]["Model"] == 1 and s["by_label"]["KBEntry"] == 2
    assert s["by_rel"]["HAS_ARTIFACT"] == 1 and s["by_rel"]["USED_TOOL"] == 2 and g.artifact_for(7)["confidence"] == 0.95
