"""The API behind the operator UI (frontend/): analytics endpoints, the cascade simulator, the streaming chat (SSE) and the incremental event parser. Offline: the agent server is faked."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent"), str(REPO)]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend import operator_api as api
    return TestClient(api.app)


def test_status_reports_agent_and_stores(client):
    s = client.get("/api/v1/status").json()
    assert s["status"] == "ok" and s["stations_count"] > 100 and {"agent_up", "models", "quality_db", "knowledge_base"} <= set(s)


def test_stations_and_edges_have_coordinates(client):
    st = client.get("/api/v1/stations").json()
    assert len(st) == 167 and all(s["lat"] and s["lon"] and s["lines"] for s in st)
    ed = client.get("/api/v1/network/edges").json()
    assert ed and {"line", "from", "to", "from_lat", "to_lon"} <= set(ed[0])


def test_flow_endpoints_are_real_and_complete_days_only(client):
    d = client.get("/api/v1/flows/daily").json()
    assert len(d["dates"]) == len(d["values"]) == len(d["rolling_mean"]) > 90
    assert min(d["values"]) > 0.4 * max(d["values"])                               # no partial day showing as a false drop to zero
    h = client.get("/api/v1/flows/hourly-profile").json()
    assert len(h) == 24 and max(r["weekday"] for r in h) > max(r["weekend"] for r in h)   # commute peaks are higher on weekdays
    one = client.get("/api/v1/flows/heatmap", params={"lines": "U8"}).json()
    assert one["count"] > 10 and len(one["values"][0]) == 24 and set(one["lines"]) == {"U8"}
    top = client.get("/api/v1/flows/heatmap", params={"lines": "U1,U2"}).json()
    assert top["count"] == 20


def test_energy_matches_the_ground_truth_and_invents_nothing(client):
    e = client.get("/api/v1/energy").json()
    r = e["ranking"]
    assert r[0]["line"] == "U5" and abs(r[0]["wh_per_pax"] - 545.5) < 1 and r[-1]["line"] == "U9" and abs(r[-1]["wh_per_pax"] - 252.6) < 1
    assert [x["efficiency_rank"] for x in r] == list(range(1, len(r) + 1)) and e["limits"]


def test_closures_and_centrality(client):
    cl = client.get("/api/v1/closures").json()
    assert cl and {c["closure_type"] for c in cl} <= {"line_suspension", "station_closure"}
    assert len(client.get("/api/v1/centrality", params={"n": 5}).json()) == 5


def test_cascade_moves_a_share_of_the_typical_flow_and_rejects_unknown_stations(client):
    body = {"closed_stations": ["U Hermannplatz (Berlin)"], "timestamp": "2026-09-25T20:45:00", "share": 0.5, "hops": 2}
    a = client.post("/api/v1/cascade", json=body).json()
    assert a["all_stations"] and a["method"] and a["assumption"] and all(s["overflow_ratio"] >= 1 for s in a["all_stations"] if not s["is_closed"])
    b = client.post("/api/v1/cascade", json={**body, "share": 0.75}).json()
    assert b["max_overflow_ratio"] >= a["max_overflow_ratio"]                       # a larger share cannot lower the peak
    assert client.post("/api/v1/cascade", json={**body, "closed_stations": ["U Nowhere"]}).status_code == 422


def test_event_parser_is_incremental_and_matches_parse_events():
    import chat_bridge as cb
    events = [
        {"customMetadata": {"kind": "supervisor", "decision": "proceed", "category": "E", "seconds": 0.01, "route": {"specialist": "energy", "ml_engine": "none", "tools": ["energy_efficiency"]}}},
        {"content": {"parts": [{"functionCall": {"id": "1", "name": "energy_efficiency", "args": {}}}]}, "customMetadata": {"server": "data", "round": 1}},
        {"content": {"parts": [{"functionResponse": {"id": "1", "name": "energy_efficiency", "response": {"seconds": 0.3, "ok": True, "result_preview": "{}"}}}]}},
        {"customMetadata": {"kind": "evaluator", "round": 1, "verdict": "accept", "score": 1.0, "seconds": 0.0, "model": "deterministic"}},
        {"customMetadata": {"kind": "llm", "role": "writer", "model": "m", "inference_s": 2.0, "tok_in": 100, "tok_out": 20}},
        {"customMetadata": {"kind": "timing", "supervisor_s": 0.01, "worker_evaluator_s": 0.3, "writer_s": 2.0, "total_s": 2.4}},
        {"content": {"parts": [{"text": "Verdict: U5"}]}, "customMetadata": {"kind": "answer", "answer_mode": "brief", "turn_id": 7}},
    ]
    p = cb.EventParser()
    stages = [[s["stage"] for s in p.feed(e)] for e in events]
    assert stages == [["dispatcher"], [], ["analyst"], ["inspector"], ["writer"], [], ["writer"]]
    r = p.result()
    assert r == cb.parse_events(events) and r["answer"] == "Verdict: U5" and r["counts"]["tool_calls"] == 1 and r["tokens"]["total"] == 120


def _sse(text):
    out = []
    for frame in text.strip().split("\n\n"):
        ev = frame.split("\n")[0].removeprefix("event: ")
        out.append((ev, json.loads(frame.split("\n")[1].removeprefix("data: "))))
    return out


def test_chat_stream_sends_steps_then_the_answer(client, monkeypatch):
    import chat_bridge

    def fake(op, sid, msg, link_turn_id=None):
        yield "step", {"kind": "route", "stage": "dispatcher", "label": "Dispatcher: proceed"}
        yield "answer", {"answer": "ok", "session_id": sid, "operator_id": op}
    monkeypatch.setattr(chat_bridge, "ask_stream", fake)
    r = client.post("/api/v1/chat/stream", json={"message": "hi there", "operator_id": "o"})
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = _sse(r.text)
    assert [e for e, _ in ev] == ["step", "answer"] and ev[1][1]["answer"] == "ok" and ev[1][1]["context"]["mode"] == "new" and ev[1][1]["session_id"].startswith("ui-")


def test_chat_stream_reports_an_unreachable_agent_as_an_error_event(client, monkeypatch):
    import chat_bridge
    monkeypatch.setattr(chat_bridge, "adk_url", lambda: "http://127.0.0.1:9")          # nothing listens there
    r = client.post("/api/v1/chat/stream", json={"message": "hi there", "operator_id": "o"})
    ev = _sse(r.text)
    assert ev[-1][0] == "error" and "not reachable" in ev[-1][1]["message"]


def test_built_ui_is_served_when_present(client):
    if (REPO / "frontend" / "dist" / "index.html").exists():
        assert client.get("/app/").status_code == 200
