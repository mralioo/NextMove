"""Offline tests: evaluation metrics, executor error handling, observability database. No LLM/MCP/network."""
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent"), str(REPO / "evaluation")]

import dataset  # noqa: E402
import executor  # noqa: E402
import metrics  # noqa: E402
import observability as obs  # noqa: E402

TRUTH = {"closure": {"reason": "safety inspection", "start": "13:50", "end": "15:20", "hours": 1.5, "date": "2026-07-13"},
         "peak": {"peak_hour": 18, "peak_value": 218.8, "network_mean_peak": 264.4, "exceeds": False}}


def _run(answer, facts, plan, total=4.0, guard="pass"):
    return {"answer": answer, "facts_json": json.dumps(facts), "plan_json": json.dumps(plan), "trace_id": "t", "total_s": total, "guard": guard,
            "tok_in": 10, "tok_out": 5}


def test_workbook_loads_training_questions_in_order():
    items = dataset.load_workbook_items()
    train = [i for i in items if i.stage == "TRAINING"]
    assert len(train) == 11 and train[0].id == "T01" and "Guns N' Roses" in train[0].question
    assert train[2].id == "T03" and "U6" in train[2].question


def test_good_closure_answer_scores_full_and_bad_facts_are_caught():
    item = dataset.Item("T03", "TRAINING", "U6 suspended ...", expected_cat="C")
    exp = metrics.expectations(item, TRUTH)
    facts = {"status": "ok", "cl": {"why": "safety inspection", "h": 1.5, "from": "2026-07-13 13:50", "to": "2026-07-13 15:20"},
             "press": [{"s": "Mehringdamm", "p": 29}, {"s": "Ullsteinstr.", "p": 21}]}
    good = ("**Verdict:** U6 is closed for a safety inspection from 13:50 to 15:20. A replacement bus is needed. Mehringdamm (29%) and "
            "Ullsteinstr. (21%) come under most pressure; deploy staff there. This is an assumption-based estimate.")
    m, checks = metrics.score_item(item, _run(good, facts, {"cat": "C"}), exp, {"n": 8, "tools": 4})
    assert m["completeness"] == 1.0 and m["fact_accuracy"] == 1.0 and m["hallucination_free"] == 1.0 and m["route_correct"] == 1.0
    bad_facts = {**facts, "cl": {**facts["cl"], "why": "weather", "h": 3}}
    m2, _ = metrics.score_item(item, _run(good, bad_facts, {"cat": "C"}), exp, {"n": 8, "tools": 4})
    assert m2["fact_accuracy"] < 1.0


def test_invented_number_and_slow_answer_are_penalised():
    item = dataset.Item("T04", "TRAINING", "Rudow?", expected_cat="D")
    exp = metrics.expectations(item, TRUTH)
    facts = {"status": "ok", "st": [{"wk": [18, 219], "net": 265, "above": False}]}
    m, _ = metrics.score_item(item, _run("Rudow peaks at 18:00 with 777 passengers, below the mean of 265.", facts, {"cat": "D"}, total=45.0),
                              exp, {"n": 6, "tools": 1})
    assert m["hallucination_free"] == 0.0 and m["latency_ok"] == 0.0


def test_decline_scoring_and_trap_routing():
    item = dataset.Item("C14", "STRESS", "How many passengers can the platform hold?", kind="Trap", expected_cat="C")
    exp = metrics.expectations(item, None)
    assert exp["supported"] is False
    facts = {"status": "oos"}
    ans = "**Verdict:** That can't be answered from this dataset: no capacity data exists. I can show a station's peak profile instead."
    m, _ = metrics.score_item(item, _run(ans, facts, {"cat": "OOS"}), exp, {"n": 5, "tools": 0})
    assert m["route_correct"] == 1.0 and m["honest_scope"] == 1.0 and m["decline_quality"] == 1.0 and m["answered"] == 0.0


def test_consistency_and_group_scores():
    a = _run("peak 18 with 219, mean 265", {}, {"cat": "D"})
    b = _run("peak 18 with 219, mean 265", {}, {"cat": "D"})
    c = _run("peak 7 with 500", {}, {"cat": "D"})
    assert metrics.consistency([a, b]) == 1.0 and metrics.consistency([a, c]) < 0.5 and metrics.consistency([a]) is None
    g = metrics.group_scores([{"answered": 1.0, "hallucination_free": 1.0, "latency_ok": 1.0},
                              {"answered": 0.0, "hallucination_free": 1.0, "latency_ok": 1.0}])
    assert g["relevance"] == 0.5 and g["reliability"] == 1.0 and g["stress"] == 1.0
    assert abs(g["overall"] - (0.5 * 0.3 + 1.0 * 0.3 + 1.0 * 0.2) / 0.8) < 1e-9


class _BrokenMcp:
    async def call(self, *a, **k):
        raise RuntimeError("Unknown datetime string format")


def test_tool_failure_becomes_error_status_not_a_crash():
    plan = {"cat": "D", "stations": ["U Rudow (Berlin)"], "raw": [], "dates": ["2026-09-22"], "time": "08:00", "lines": [],
            "what_if": False, "month": None}
    facts, _ = asyncio.run(executor.execute(plan, "flow at Rudow", None, _BrokenMcp()))
    assert facts["status"] == "error" and "RuntimeError" in facts["note"]


def test_observability_db_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "DB_PATH", tmp_path / "obs.db")
    conn = obs.connect()
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"runs", "eval_runs", "eval_items"} <= tables
    conn.close()
    obs._write("INSERT INTO eval_runs VALUES (?,?,?,?,?,?,?,?)", ("e1", 1.0, "training", "fast", 11, 20.0, "abc", "{}"))
    row = obs.connect().execute("select suite, n_items from eval_runs where eval_id='e1'").fetchone()
    assert tuple(row) == ("training", 11)


class _FakeMcp:
    """Answers describe_dataset / resolve_station like the real server (incl. its lenient fuzzy match)."""

    def __init__(self):
        self.calls = []

    async def call(self, tool, **kw):
        self.calls.append(tool)
        if tool == "describe_dataset":
            return {"coverage_start": "2026-06-10 05:00:00", "coverage_end": "2026-09-22 00:45:00"}
        if tool == "resolve_station":
            return [{"station_name": "U Wittenau (Berlin)"}]     # what the real tool returned for "September"
        raise AssertionError(f"unexpected tool call {tool}")


def _plan(**over):
    base = {"cat": "D", "stations": [], "raw": [], "dates": [], "time": None, "lines": [], "what_if": False, "month": None, "dur_min": None}
    return {**base, **over}


def test_dates_outside_dataset_coverage_are_refused_without_calling_tools():
    executor._COVERAGE.clear()
    mcp = _FakeMcp()
    facts, _ = asyncio.run(executor.execute(_plan(dates=["2026-09-30"], stations=["U Rudow (Berlin)"]), "flow on September 30th?", None, mcp))
    assert facts["status"] == "oos" and "2026-09-22" in facts["note"]
    assert mcp.calls == ["describe_dataset"]                    # no station_profile / prediction was attempted


def test_phantom_fuzzy_station_match_is_rejected():
    executor._COVERAGE.clear()
    facts, _ = asyncio.run(executor.execute(_plan(raw=["September"]), "flow?", None, _FakeMcp()))
    assert facts["status"] == "need" and "station" in facts["missing"]


def test_unsupported_category_states_the_true_reason():
    facts, _ = asyncio.run(executor.execute(_plan(cat="X"), "invest where?", None, _FakeMcp()))
    assert facts["status"] == "unsupported" and "not connected yet" in facts["reason"]
    assert "not connected yet" in __import__("writer").render_fallback(facts)


def test_r_questions_have_correct_expectations():
    r5 = dataset.Item("R5", "STRESS", "Was ist der Grund ...")
    assert metrics.expectations(r5, None)["supported"] is True
    r7 = dataset.Item("R7", "STRESS", "flow on September 30th?")
    assert metrics.expectations(r7, None)["supported"] is False


def test_eval_refuses_many_calls_to_the_shared_llm_unless_allowed():
    import pytest
    import run_eval

    assert run_eval.enforce_budget(1, 1, False, False) == 1                 # the default single question
    assert run_eval.enforce_budget(11, 1, False, True) == 11                # explicit opt-in
    with pytest.raises(SystemExit):
        run_eval.enforce_budget(11, 1, False, False)                        # workbook suite without --allow-many
    with pytest.raises(SystemExit):
        run_eval.enforce_budget(3, 1, True, False)                          # a MAIN-model judge doubles the calls (6 > 5)


def test_limit_question_has_six_parts_and_expectations():
    items = dataset.load_limit_items()
    assert len(items) == 1 and len(items[0].meta["parts"]) == 6
    exp = metrics.expectations(items[0], TRUTH)
    assert list(exp["rubric"]) == items[0].meta["parts"] and exp["supported"] is True


def test_oos_note_names_only_what_was_asked():
    import router
    plan = router.route(dataset.LIMIT_QUESTION)
    assert plan["cat"] == "OOS" and any("safely hold" in h for h in plan["oos"])
    facts, _ = asyncio.run(executor.execute(plan, dataset.LIMIT_QUESTION, None, _FakeMcp()))
    assert "safely hold" in facts["note"] and "delays" not in facts["note"] and "U4" not in facts["note"]
