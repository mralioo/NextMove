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


# ------------------------------------------------------------------------------------------------ operator desktop
def test_chat_bridge_turns_adk_events_into_answer_steps_tools_and_tokens():
    from backend import chat_bridge as cb
    ev = [
        {"author": "supervisor", "content": {"parts": [{"text": "[plan]"}]}, "customMetadata": {"kind": "supervisor", "seconds": 0.02, "decision": "proceed", "category": "C",
                                                                                                   "route": {"specialist": "disruption", "ml_engine": "tabpfn", "tools": ["apply_closure"]}}},
        {"author": "worker:disruption", "content": {"parts": [{"functionCall": {"id": "mcp-1-0", "name": "apply_closure", "args": {"closure_id": 1}}}]}, "customMetadata": {"kind": "mcp_call", "round": 1, "server": "s"}},
        {"author": "worker:disruption", "content": {"parts": [{"functionResponse": {"id": "mcp-1-0", "name": "apply_closure", "response": {"ok": True, "seconds": 0.3, "result_bytes": 1800, "server": "s"}}}]},
         "customMetadata": {"kind": "mcp_result"}},
        {"author": "writer", "content": {"parts": [{"text": "[llm writer]"}]}, "customMetadata": {"kind": "llm", "role": "writer", "model": "m", "inference_s": 2.5, "tok_in": 100, "tok_out": 20, "prompt": "p" * 40, "response": "r" * 8}},
        {"author": "writer", "content": {"parts": [{"text": "[timing]"}]}, "customMetadata": {"kind": "timing", "supervisor_s": 0.02, "worker_evaluator_s": 0.5, "mcp_s": 0.3, "writer_s": 2.6, "total_s": 3.2}},
        {"author": "writer", "content": {"parts": [{"text": "**Verdict:** x"}]}, "customMetadata": {"kind": "answer", "turn_id": 5, "artifact_turn_id": 5, "requires_action": True, "answer_mode": "brief", "source": "worker"}},
    ]
    r = cb.parse_events(ev)
    assert r["answer"] == "**Verdict:** x" and r["artifact_turn_id"] == 5 and r["requires_action"] is True
    assert r["counts"] == {"tool_calls": 1, "distinct_tools": 1, "llm_calls": 1, "events": 6} and r["tools"][0]["args"] == {"closure_id": 1} and r["tools"][0]["seconds"] == 0.3
    assert r["tokens"] == {"in": 100, "out": 20, "total": 120, "estimated": False} and r["timing"]["total_s"] == 3.2
    assert [s["kind"] for s in r["steps"]] == ["route", "tool", "llm", "writer"]
    est = cb.parse_events([{"author": "writer", "content": {"parts": [{"text": "x"}]}, "customMetadata": {"kind": "llm", "role": "writer", "model": "m", "inference_s": 1, "prompt": "p" * 400, "response": "r" * 80}}])
    assert est["tokens"]["estimated"] is True and est["tokens"]["in"] == 100


def test_desktop_endpoints_topology_snapshot_series_and_chat(monkeypatch):
    from fastapi.testclient import TestClient
    from backend import operator_api as api
    import chat_bridge                      # the module the API itself uses (backend/ is on sys.path once operator_api is imported)
    c = TestClient(api.app)
    topo = c.get("/api/v1/ops/topology").json()
    assert len(topo["stations"]) > 100 and topo["edges"] and {l["line"] for l in topo["lines"]} >= {"U1", "U9"}
    tl = c.get("/api/v1/ops/timeline").json()
    assert tl["default_at"] and tl["closures"]
    snap = c.get("/api/v1/ops/snapshot", params={"at": tl["default_at"]}).json()
    assert snap["network"]["total"] > 0 and snap["lines"] and snap["closures"] and snap["closures"][0]["blocked_edges"] is not None
    assert len(c.get("/api/v1/ops/series", params={"date": tl["default_at"][:10]}).json()) > 50
    monkeypatch.setattr(chat_bridge, "ask", lambda op, sid, msg, link_turn_id=None: {"answer": "ok", "session_id": sid, "operator_id": op})
    assert c.post("/api/v1/chat", json={"message": "hi", "operator_id": "o", "session_id": "s"}).json()["session_id"] == "s"
    def down(*a, **k):
        raise chat_bridge.AgentUnavailable("not reachable")
    monkeypatch.setattr(chat_bridge, "ask", down)
    assert c.post("/api/v1/chat", json={"message": "hi"}).status_code == 503


def test_a_precedent_needs_the_same_line_or_station(env):
    kb, g, tid = env
    feedback.submit(tid, score=5, action_text="Sent two staff to Neukölln", followed="as_recommended", outcome="worked", kb=kb, graph=g, mirror=False)
    u2 = "Line U2 is suspended between Bismarckstr. and Neu-Westend on 2026-09-27 from 08:30 for 4 hours. Where should we deploy staff?"
    assert feedback.precedents(u2, "C", None, graph=g) == []                                  # same words, other line and stations: no precedent
    assert feedback.precedents(Q.replace("2026-09-25", "2026-09-26"), "C", None, graph=g)      # same line and stations: precedent


def test_tidy_puts_a_blank_line_before_each_section():
    assert writer_tidy("**Verdict:** a\n**Do now:**\n- x\n**Watch out:** z") == "**Verdict:** a\n\n**Do now:**\n- x\n\n**Watch out:** z"


def writer_tidy(t):
    import writer
    return writer.tidy(t)


# ------------------------------------------------------------------------------------------------ conversation threads
def test_relation_detects_when_a_message_is_not_about_the_situation():
    import threads
    A = {"category": "C", "entities": {"lines": ["U7"], "stations": ["U Hermannplatz (Berlin)"], "dates": ["2026-09-25"]}}
    assert threads.relation("why?", A)["relation"] == "related"
    assert threads.relation("show me the evidence", A)["relation"] == "related"
    assert threads.relation("for 3 hours instead", A)["relation"] == "related"
    assert threads.relation("Which stations get pressure at Hermannplatz?", A)["relation"] == "related"
    assert threads.relation("At what time does the commute flow peak at Rudow station usually take place?", A)["relation"] == "unrelated"
    assert threads.relation("Identify three anomalies on July 19th.", A)["relation"] == "unrelated"
    assert threads.relation("And Rudow?", A)["relation"] == "unsure"
    assert threads.relation("What is the capital of France?", A)["relation"] == "related"          # off-topic: bounced on its own, the situation is untouched
    assert threads.relation("anything", None)["relation"] == "related"


def _turn(kb, sid, q, cat="C", link=None, art=None, uid="op1"):
    plan = {"category": cat, "entities": {"lines": ["U7"], "stations": ["U Hermannplatz (Berlin)"]}}
    return kb.remember_turn(sid, uid, q, cat, "answer", {"status": "ok"}, accepted=True, plan=plan, artifact=art or {"question": q, "brief": "b", "problem_key": "pk-" + sid}, linked_from=link)


def test_conversation_history_groups_resumed_conversations_into_one_entry(tmp_path):
    kb = knowledge.KnowledgeBase(db=tmp_path / "m.db")
    t1 = _turn(kb, "s1", "U7 closure question")
    _turn(kb, "s1", "why?", cat="FOLLOW")
    _turn(kb, "s2", "Rudow peak?", cat="D")
    t4 = _turn(kb, "s3", "What if it ends at 22:30?", link=t1)                                 # s3 resumes s1
    ch = kb.conversation_chains("op1")
    assert len(ch) == 2
    c1 = next(c for c in ch if c["title"] == "U7 closure question")
    assert c1["session_ids"] == ["s1", "s3"] and c1["n_turns"] == 3 and c1["resumed"] and c1["last_session_id"] == "s3" and c1["category"] == "C"
    assert kb.session_anchor("s1")["question"] == "U7 closure question"                         # 'why?' is not a new situation
    assert kb.turn_anchor(t4)["linked_from"] == t1 and kb.turn_anchor(t1)["entities"]["lines"] == ["U7"]


def test_chat_asks_before_mixing_topics_and_new_starts_a_fresh_session(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    kb = knowledge.KnowledgeBase(db=tmp_path / "m.db")
    monkeypatch.setattr(knowledge, "kb", lambda: kb)
    from backend import operator_api as api
    import chat_bridge
    calls = []
    monkeypatch.setattr(chat_bridge, "ask", lambda op, sid, msg, link_turn_id=None: calls.append((sid, msg, link_turn_id)) or {"answer": "ok", "session_id": sid})
    t1 = _turn(kb, "s1", "Line U7 is suspended between Hermannplatz and Karl-Marx-Strasse. Where to deploy staff?")
    c = TestClient(api.app)
    rudow = "At what time does the commute flow peak at Rudow station usually take place?"
    r = c.post("/api/v1/chat", json={"message": rudow, "session_id": "s1"}).json()
    assert r["needs_choice"] and r["relation"] == "unrelated" and r["recommended"] == "new" and {x["id"] for x in r["choices"]} == {"new", "continue"} and calls == []
    r = c.post("/api/v1/chat", json={"message": rudow, "session_id": "s1", "context_mode": "new"}).json()
    assert r["session_id"] != "s1" and r["context"]["mode"] == "new" and calls[-1][0] == r["session_id"]
    r = c.post("/api/v1/chat", json={"message": rudow, "session_id": "s1", "context_mode": "continue"}).json()
    assert r["session_id"] == "s1" and r["context"]["mode"] == "continue"
    assert c.post("/api/v1/chat", json={"message": "why?", "session_id": "s1"}).json()["session_id"] == "s1"      # related: no question, no interruption
    r = c.post("/api/v1/chat", json={"message": "for 3 hours instead", "link_turn_id": t1}).json()                  # resume a conversation from the history
    assert calls[-1][2] == t1 and r["context"]["mode"] == "resumed"
    r = c.post("/api/v1/chat", json={"message": rudow, "link_turn_id": t1}).json()                                  # ... and the first message is checked against it too
    assert r["needs_choice"]
    conv = c.get("/api/v1/conversations", params={"operator_id": "op1"}).json()
    assert conv[0]["title"].startswith("Line U7") and c.get("/api/v1/conversations/s1").json()["resume_turn_id"] == t1


# ------------------------------------------------------------------------------------------------ graph categorisation and no wrong branches
def _case(q, cat="C", lines=("U7",), stations=("U Hermannplatz (Berlin)",), actions=()):
    return S.GraphCase(question=q, category=cat, entities=S.Entities(lines=list(lines), stations=list(stations)), answer="brief " + q, verdict="accept", actions=list(actions))


def test_variants_share_the_situation_and_categories_have_domains_and_actions_have_types(tmp_path):
    g = kgraph.KnowledgeGraph(tmp_path / "kg.db")
    p1 = g.record_case(_case("U7 closure Hermannplatz", actions=["Deploy additional staff at Neukölln", "Replacement bus between A and B"]), conversation="s1", operator="op1")
    p2 = g.record_case(_case("what about 22:30?"), kind="variant", parent_key=p1, situation_key=p1, conversation="s1", operator="op1")
    p3 = g.record_case(_case("Rudow peak", cat="D", lines=(), stations=("U Rudow (Berlin)",)), conversation="s2", operator="op1")
    s = g.stats()
    assert s["by_label"]["Situation"] == 2 and s["by_label"]["Conversation"] == 2                     # the variant did NOT open a third situation
    assert s["by_rel"]["VARIANT_OF"] == 1 and s["by_rel"]["PART_OF"] == 3
    tax = g.taxonomy()
    assert {c["code"]: (c["label"], c["domain"]) for c in tax["categories"]} == {"C": ("Closure response", "Disruptions"), "D": ("Station profile", "Stations")}
    assert kgraph.action_types("Sent two staff to Neukölln and ordered the replacement bus") == ["staff_deployment", "replacement_bus"]
    assert kgraph.action_types("did something") == ["other"]
    assert s["by_label"]["ActionType"] >= 2 and p2 != p1 and p3


def test_feedback_is_categorised_typed_and_never_opens_a_wrong_branch(env):
    kb, g, tid = env
    feedback.submit(tid, score=4, action_text="Sent staff to Neukölln and informed passengers", followed="modified", kb=kb, graph=g, mirror=False)
    s = g.stats()
    assert s["by_rel"]["IN_CATEGORY"] >= 2 and s["by_rel"]["OF_TYPE"] >= 2 and s["by_rel"].get("ABOUT_SITUATION", 0) == 1
    assert g.taxonomy()["operator_action_types"] == {"staff_deployment": 1, "passenger_information": 1}
    # feedback for an answer whose situation was never recorded (e.g. a decline): kept, but marked and kept out of similarity / precedents
    tid2 = kb.remember_turn("s9", "op1", "Which is the best single investment?", "X", "declined", {"status": "unsupported"}, accepted=False)
    feedback.submit(tid2, score=1, kb=kb, graph=g, mirror=False)
    orphans = [p for p in g._problems() if p[2].get("kind") == "feedback_only"]
    assert len(orphans) == 1 and g.similar("Which is the best single investment?", "X") == []


def test_audit_finds_and_repair_fixes_uncategorised_branches(tmp_path):
    g = kgraph.KnowledgeGraph(tmp_path / "kg.db")
    g.node("Problem", "old1", text="Line U7 closed between A and B. What now please?", category="C", source="runtime", params={"lines": ["U7"]})       # legacy: no kind, no situation, no category edge
    g.node("Problem", "fu1", text="what about 22:30?", category="C", source="runtime", kind="situation", params={})                                    # a follow-up that became its own branch
    a = g.audit()
    assert "old1" in a["no_situation"] and "old1" in a["no_kind"] and "fu1" in a["followup_shaped"] and a["issues"] > 0
    r = g.repair()
    assert r["fixed"]["situations"] == 2 and r["fixed"]["followups"] == 1 and r["after"]["no_situation"] == 0 and r["after"]["no_kind"] == 0
    assert g.similar("what about 22:30?", "C") == []                                                     # the follow-up-shaped problem no longer matches anything
