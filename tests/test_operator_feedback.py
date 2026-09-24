"""The operator feedback loop: scores and action reports -> knowledge base, knowledge graph, precedents, history reuse; the HTTP API around it. Offline (temp databases)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent"), str(REPO)]

import feedback  # noqa: E402
import kgraph  # noqa: E402
import knowledge  # noqa: E402
import schemas as S  # noqa: E402

Q = "Line U7 is suspended between Hermannplatz and Karl-Marx-Strasse on 2026-09-25 from 20:45 for 2 hours. Where should we deploy staff?"


@pytest.fixture()
def env(tmp_path):
    kb = knowledge.KnowledgeBase(db=tmp_path / "m.db")
    g = kgraph.KnowledgeGraph(tmp_path / "kg.db")
    pk = kgraph._h(kgraph._norm(Q), 12)
    g.record_case(S.GraphCase(question=Q, category="C", entities=S.Entities(lines=["U7"], stations=["U Hermannplatz (Berlin)"]), answer="brief", verdict="accept", actions=["Deploy staff at Neukölln"]))
    art = {"question": Q, "brief": "brief", "problem_key": pk, "recommended_actions": ["Deploy staff at Neukölln", "Replacement bus between Hermannstr. and Neukölln"], "requires_action": True,
           "tools": [{"tool": "scenario_flow"}], "facts": {"status": "ok"}}
    tid = kb.remember_turn("s1", "op1", Q, "C", "brief", {"status": "ok"}, verdict="accept", confidence=0.95, accepted=True, artifact=art)
    return kb, g, tid


def test_score_and_action_are_validated_and_stored(env):
    kb, g, tid = env
    with pytest.raises(feedback.FeedbackError):
        feedback.submit(tid, score=9, kb=kb, graph=g, mirror=False)
    with pytest.raises(feedback.FeedbackError):
        feedback.submit(tid, kb=kb, graph=g, mirror=False)                        # nothing to store
    with pytest.raises(feedback.FeedbackError):
        feedback.submit(999, score=3, kb=kb, graph=g, mirror=False)               # unknown turn
    with pytest.raises(feedback.FeedbackError):
        feedback.submit(tid, action_text="x y z", followed="maybe", kb=kb, graph=g, mirror=False)
    r = feedback.submit(tid, operator_id="op1", score="up", kb=kb, graph=g, mirror=False)
    assert r["kind"] == "score" and r["score"] == 5 and r["fed"]["knowledge_graph"]
    a = feedback.submit(tid, operator_id="op1", action_text="Sent two staff to Neukölln and put a bus on U7", followed="modified", outcome="worked", kb=kb, graph=g, mirror=False)
    assert a["kind"] == "action" and any("Neukölln" in s for s in a["stations"]) and a["lines"] == ["U7"]
    info = kb.turn_info(tid)
    assert info["operator_score"] == 5 and info["n_feedback"] == 2 and info["requires_action"] and len(kb.feedback_for_turn(tid)) == 2


def test_feedback_reaches_the_graph_with_stations_lines_and_matched_recommendation(env):
    kb, g, tid = env
    feedback.submit(tid, score=4, action_text="Sent staff to Neukölln", followed="as_recommended", kb=kb, graph=g, mirror=False)
    s = g.stats()
    assert s["by_label"]["Feedback"] == 1 and s["by_label"]["OperatorAction"] == 1
    assert s["by_rel"]["RECEIVED_FEEDBACK"] >= 1 and s["by_rel"]["OPERATOR_TOOK"] == 1 and s["by_rel"]["DESCRIBES"] == 1 and s["by_rel"]["AT"] >= 1
    assert s["by_rel"].get("MATCHES", 0) == 1                                    # "Sent staff to Neukölln" carries out "Deploy staff at Neukölln"


def test_precedents_are_read_back_for_a_similar_situation_and_bad_outcomes_warn(env):
    kb, g, tid = env
    feedback.submit(tid, score=5, action_text="Sent two staff to Neukölln", followed="as_recommended", outcome="worked", kb=kb, graph=g, mirror=False)
    prec = feedback.precedents("Line U7 is suspended between Hermannplatz and Karl-Marx-Strasse tomorrow. Where do we send staff?", "C", None, graph=g)
    assert prec and prec[0]["actions"][0]["action"].startswith("Sent two staff") and prec[0]["mean_score"] == 5
    line = feedback.precedent_line(prec)
    assert line.startswith("_Precedent:") and "worked" in line and "5/5" in line
    kb.update_feedback(1, outcome="did_not_work", score=1)
    feedback.update(1, kb=kb, graph=g, outcome="did_not_work", score=1)
    assert feedback.precedent_line(feedback.precedents(Q + " again", "C", None, graph=g)).startswith("_Caution:")
    assert feedback.precedents("What is the energy per passenger of U5?", "E", None, graph=g) == []


def test_badly_rated_answers_are_not_reused_from_history(env):
    kb, g, tid = env
    assert kb.find_answered(Q, None, None)
    feedback.submit(tid, score=2, kb=kb, graph=g, mirror=False)
    assert kb.find_answered(Q, None, None) is None


def test_pending_list_and_stats(env):
    kb, g, tid = env
    assert [t["turn_id"] for t in kb.list_turns("op1", needs_action_report=True)] == [tid]
    feedback.submit(tid, action_text="Deployed staff at Hermannplatz", kb=kb, graph=g, mirror=False)
    assert kb.list_turns("op1", needs_action_report=True) == []
    st = kb.feedback_stats()
    assert st["n_action_reports"] == 1 and st["turns_requiring_action"] == 1 and st["action_report_rate"] == 1.0
    assert feedback.delete(1, kb=kb, graph=g) and kb.feedback_stats()["n_action_reports"] == 0


def test_recommended_actions_come_from_the_do_now_bullets():
    brief = "**Verdict:** x\n\n**Do now:**\n- Deploy staff at Neukölln.\n- Use the bus.\n\n**Watch out:** y\n_Confidence high_"
    assert feedback.recommended_actions(brief, {}) == ["Deploy staff at Neukölln.", "Use the bus."]


def test_http_api_round_trip(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    kb = knowledge.KnowledgeBase(db=tmp_path / "m.db")
    g = kgraph.KnowledgeGraph(tmp_path / "kg.db")
    monkeypatch.setattr(knowledge, "kb", lambda: kb)
    monkeypatch.setattr(kgraph, "kg", lambda: g)
    from backend import operator_api as api
    c = TestClient(api.app)
    tid = kb.remember_turn("s1", "op1", Q, "C", "brief", {"status": "ok"}, accepted=True, artifact={"question": Q, "brief": "b", "requires_action": True, "recommended_actions": ["Deploy staff at Neukölln"], "tools": []})
    assert c.get("/api/v1/health").json()["status"] == "ok"
    assert c.get("/api/v1/enums").json()["followed"]
    assert c.get("/api/v1/feedback/pending", params={"operator_id": "op1"}).json()[0]["turn_id"] == tid
    assert c.post(f"/api/v1/turns/{tid}/score", json={"score": 7}).status_code == 422
    assert c.post(f"/api/v1/turns/999/score", json={"score": 3}).status_code == 404
    r = c.post(f"/api/v1/turns/{tid}/score", json={"score": 4, "comment": "clear"})
    assert r.status_code == 201 and r.json()["score"] == 4
    a = c.post(f"/api/v1/turns/{tid}/action", json={"action_text": "Sent staff to Neukölln", "followed": "as_recommended"})
    assert a.status_code == 201
    fid = a.json()["feedback_id"]
    assert c.patch(f"/api/v1/feedback/{fid}", json={"outcome": "worked"}).json()["outcome"] == "worked"
    turn = c.get(f"/api/v1/turns/{tid}").json()
    assert turn["operator_score"] == 4 and len(turn["feedback"]) == 2
    assert c.get("/api/v1/feedback/pending", params={"operator_id": "op1"}).json() == []
    assert c.get("/api/v1/feedback/stats").json()["n_action_reports"] == 1
    assert c.get(f"/api/v1/turns/{tid}/artifact").json()["question"] == Q
    assert c.get(f"/api/v1/turns/{tid}/report").json()["has_report"] is False
    assert c.get("/api/v1/operator-actions").json()[0]["action"].startswith("Sent staff")
    assert c.get("/api/v1/precedents", params={"question": Q, "category": "C"}).json()["precedents"] is not None
    assert c.delete(f"/api/v1/feedback/{fid}").json() == {"deleted": fid} and c.delete(f"/api/v1/feedback/{fid}").status_code == 404
