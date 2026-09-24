import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "evaluation"), str(ROOT / "agent")]

import submission_db as db  # noqa: E402
import submission_run as sr  # noqa: E402
import team_evidence  # noqa: E402


def _row(item="T01", stage="TRAINING", lat=3.0, dec="proceed", status="ok", tin=100, tout=20):
    return {"item_id": item, "stage": stage, "seq": 1, "question": "q", "answer": "a b c", "source": "agent", "category": "D", "decision": dec, "verdict": "accept", "confidence": 0.9,
            "facts_status": status, "sanity_ok": 1, "guard": "pass", "answer_words": 3, "latency_s": lat, "supervisor_s": 0.01, "worker_s": 1.0, "mcp_s": 0.9, "writer_s": 2.0, "llm_s": 2.0,
            "llm_calls": 1, "tok_in": tin, "tok_out": tout, "mcp_calls": 2, "rounds": 1, "n_spans": 10, "models_json": ["m"], "llm_json": [{"role": "writer"}], "calls_json": [],
            "plan_json": {}, "facts_json": {}, "checks_json": [], "obs_run_id": "x", "trace_id": "y", "error": None}


def test_answer_record_reads_stages_tokens_and_llm_calls_from_the_stored_run():
    run = {"category": "D", "facts_status": "ok", "total_s": 3.2, "run_id": "r", "trace_id": "t", "n_spans": 5, "plan_json": json.dumps({"decision": "proceed"}), "facts_json": "{}",
           "timing_json": json.dumps({"decision": "proceed", "verdict": "accept", "confidence": 0.9, "stages": {"supervisor_s": 0.01, "worker_evaluator_s": 1.0, "mcp_s": 0.8, "writer_s": 2.0},
                                      "llm": [{"role": "writer", "model": "m", "seconds": 1.9, "tok_in": 100, "tok_out": 20}, {"role": "evaluator", "model": "m", "seconds": 0.5, "tok_in": 50, "tok_out": 5}],
                                      "calls": [{"tool": "t", "s": 0.8, "args": {"a": 1}}], "loop": [{}]})}
    rec = sr.answer_record(run, "one two three", 3.5, None, {"item_id": "T01", "stage": "TRAINING", "question": "q", "seq": 1})
    assert rec["tok_in"] == 150 and rec["tok_out"] == 25 and rec["llm_calls"] == 2 and rec["llm_s"] == 2.4 and rec["mcp_calls"] == 1 and rec["latency_s"] == 3.2 and rec["answer_words"] == 3


def test_aggregate_and_storage_round_trip(tmp_path, monkeypatch):
    rows = [_row("T01"), _row("T02", lat=5.0, dec="decline", status="unsupported"), {**_row("E1", "TEAM_EVIDENCE"), "source": "authored+measured", "latency_s": None}]
    m = sr.aggregate(rows, 10.0)
    assert m["n_items"] == 3 and m["n_agent_answers"] == 2 and m["latency_s"]["max"] == 5.0 and m["tokens"]["in"] == 200 and m["declined"] == 1 and m["answered_ok"] == 1
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "s.db")
    conn = db.connect()
    db.save_run(conn, {"run_id": "sub-x", "created_at": "now", "label": "l", "config": {"models": {"writer": "m"}}, "metrics": m, "stages": ["TRAINING"]}, rows)
    assert db.list_runs(conn)[0]["run_id"] == "sub-x"
    got = db.answers(conn, "sub-x")
    assert len(got) == 3 and got[0]["run_id"] == "sub-x" and json.loads(got[0]["models_json"]) == ["m"]


def test_stress_evidence_is_composed_from_measured_results():
    held = [{**_row("S1", "HELDOUT"), "question": "Line U7 ...", "answer": "**Verdict:** ok"}, {**_row("S6", "HELDOUT", lat=2.0, dec="decline", status="oos"), "question": "Oct 15?", "answer": "cannot"}]
    text = team_evidence.stress(held, "sub-x", "2026-06-10 to 2026-10-01")
    assert "sub-x" in text and "2× " not in text and "1× decline" in text and "median" in text
