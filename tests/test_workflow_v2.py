"""Offline tests for the v2 workflow: new categories (A, P, B, E, F, G, H), closure-matching fix, horizon parsing, data-drop merge,
knowledge base + sanity checker + Cognee circuit breaker. No LLM, no TabPFN, no network."""
import asyncio
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent"), str(REPO / "evaluation"), str(REPO / "dashboard"), str(REPO / "mcp_server")]

import executor  # noqa: E402
import knowledge  # noqa: E402
import router  # noqa: E402
import specialists  # noqa: E402

Q1 = "U8 is suspended between Hermannplatz and Neukölln. Where will passengers reroute, and which stations are at risk of overcrowding in the next 20 minutes?"
Q2 = "There is a sold-out concert at Mercedes-Benz Arena tonight at 21:00. What does the flow look like at Hermannplatz, and what should we do at 23:15 when it ends?"
Q3 = "During InnoTrans 2026, we expect major passenger flow and bad weather. Show me the 3 stations most likely to exceed safe platform capacity during the first day of the event."


# ------------------------------------------------------------------------------------------------ router
def test_challenge_questions_route_to_the_right_specialists():
    p1, p2, p3 = router.route(Q1), router.route(Q2), router.route(Q3)
    assert (p1["cat"], p2["cat"], p3["cat"]) == ("C", "A", "P")
    assert p3["n"] == 3 and p3["rain"] is True and p3["dates"] == ["2026-09-22"] and p3["assumed"]      # InnoTrans day 1 is an assumption, disclosed
    assert p2["venue"] == "Uber Arena" and p2["alias"] is True and p2["times"] == ["21:00", "23:15"] and p2["rel_day"] == "tonight"
    assert "tonight" not in p2["oos"] and p3["oos"] == ["capacity"] and p3["cat"] != "OOS"           # trap words annotate, they no longer veto the question


def test_look_ahead_horizon_is_not_a_closure_duration():
    p = router.route(Q1)
    assert p["horizon_min"] == 20 and p["dur_min"] is None
    assert router.route("U6 suspended between A and B for one hour, what happens in the next 20 minutes?")["dur_min"] == 60


def test_pure_capacity_question_is_still_declined():
    assert router.route("How many passengers can the U6 platform at Mehringdamm safely hold?")["cat"] == "OOS"


def test_other_training_categories_are_routed():
    for q, cat in (("Which metro line has the worst energy-per-passenger efficiency ratio?", "E"),
                   ("Rank the five stations whose closure would fragment the network the most.", "F"),
                   ("Are there stations whose demand depends strongly on another station despite no direct connection?", "G"),
                   ("Identify three passenger-flow anomalies that cannot be explained by station closures on June 24th.", "B")):
        assert router.route(q)["cat"] == cat


def test_specialist_registry_matches_the_router_tool_table():
    for cat, sp in specialists.SPECIALISTS.items():
        if sp.status != "planned":
            assert tuple(router.CAT_DATA[cat]["tools"]) == sp.tools, cat
            assert sp.run is not None


# ------------------------------------------------------------------------------------------------ closure matching
class _Mcp:
    def __init__(self):
        self.calls = []

    async def call(self, tool, **kw):
        self.calls.append((tool, kw))
        if tool == "describe_dataset":
            return {"coverage_start": "2026-06-10 05:00:00", "coverage_end": "2026-09-22 00:45:00"}
        if tool == "resolve_closure":
            return [{"closure_id": 7, "line": "U8", "start": "2026-07-11 08:20:00", "from_station": "U Leinestr. (Berlin)", "to_station": "U Hermannplatz (Berlin)", "station": None}]
        if tool == "apply_closure":
            return {"closure": {"line": kw.get("line"), "kind": "line_section", "from_station": kw.get("from_station"), "to_station": kw.get("to_station"),
                                "start": kw.get("start"), "end": "x", "duration_hours": 1, "reason": "hypothetical", "source": "hypothetical"}, "unserved_stations": [], "still_served_by_other_line": []}
        if tool == "alternate_paths":
            return {"alternates": [], "surface_link_candidates": [{"from": "U A (Berlin)", "to": "U B (Berlin)", "straight_line_km": 0.5}]}
        if tool == "scenario_flow":
            return {"ranked_pressure_stations": []}
        raise AssertionError(tool)


def test_a_closure_that_only_shares_a_line_and_one_station_is_not_silently_substituted():
    """The challenge review's critical defect: 'U8 Hermannplatz-Boddinstr. on Sept 15' was answered with the recorded 11 July closure."""
    executor._COVERAGE.clear()
    plan = router.route("U8 is suspended between Hermannplatz and Boddinstr. on September 15th from 17:00 for one hour. Where will passengers reroute?")
    mcp = _Mcp()
    facts, _ = asyncio.run(executor.execute(plan, "q", None, mcp))
    apply_call = next(kw for t, kw in mcp.calls if t == "apply_closure")
    assert "closure_id" not in apply_call and apply_call["start"] == "2026-09-15 17:00" and apply_call["duration_minutes"] == 60
    assert "hypothetical" in facts["src_note"] and "2026-07-11" in facts["src_note"]              # the near miss is disclosed, never used


def test_line_that_does_not_serve_the_stations_is_corrected_and_disclosed():
    executor._COVERAGE.clear()
    facts, _ = asyncio.run(executor.execute(router.route(Q1), Q1, None, _Mcp()))
    assert facts["cl"]["line"] == "U7" and "U7" in facts["assumed"]["line"] and "date" in facts["assumed"]
    assert facts["cl"]["from"].startswith("2026-09-21 17:00")                                         # latest full weekday, disclosed
    assert facts["no_pressure_reason"] and facts["reroute"][0].startswith("no rail detour exists; a replacement bus")


# ------------------------------------------------------------------------------------------------ analytics tools (real data, no model)
class _Reg:
    def __init__(self):
        self.tools = {}

    def tool(self, fn):
        self.tools[fn.__name__] = fn
        return fn


def _tools():
    import analytics_tools
    reg = _Reg()
    analytics_tools.register(reg, str(REPO / "data" / "training dataset"), lambda: None, lambda: None)
    return reg.tools


def test_event_impact_learns_the_venue_to_station_mapping_and_says_hermannplatz_is_unaffected():
    r = _tools()["event_impact"](venue="Mercedes-Benz Arena", station="Hermannplatz", end_time="23:15")
    assert r["alias_assumed"] and r["venue"] == "Uber Arena"
    assert [x["s"] for x in r["top_stations"][:2]] == ["Warschauer Str.", "Schlesisches Tor"]
    assert r["station"]["affected"] is False


def test_event_name_matches_despite_curly_apostrophes():
    r = _tools()["event_impact"](event_name="Guns N' Roses", date="2026-06-23")
    assert r["status"] == "ok" and r["event"]["date"] == "2026-06-23" and r["event"]["attendance"] == 1996


def test_energy_and_resilience_and_reroute_agree_with_raw_data_facts():
    t = _tools()
    assert t["energy_efficiency"]()["worst"] == "U5"
    top = t["network_resilience_ranking"](top_n=5)["top"]
    assert top[0]["s"] == "Alexanderplatz Bhf" and top[0]["cut_off"] == 25
    rb = t["reroute_behaviour"]()
    assert rb["status"] in ("ok", "unavailable")


# ------------------------------------------------------------------------------------------------ data drop
def test_a_second_data_file_or_folder_is_merged_not_ignored(tmp_path):
    from utils import data_loader as dl
    src = REPO / "data" / "training dataset"
    for name in ("stations_with_ubahn.csv", "flows_pre_innotrans.csv"):
        (tmp_path / "train").mkdir(exist_ok=True)
        shutil.copy(src / name, tmp_path / "train" / name)
    extra = pd.read_csv(src / "flows_pre_innotrans.csv", nrows=3)
    extra["timestamp"] = ["9/22/2026 1:00", "9/22/2026 1:15", "9/22/2026 1:30"]
    (tmp_path / "eval").mkdir()
    extra.to_csv(tmp_path / "eval" / "flows_eval.csv", index=False)
    base = len(dl.load_flows.__wrapped__(str(tmp_path / "train")))
    assert len(dl.load_flows.__wrapped__(str(tmp_path))) == base + 3                                  # parent folder: old + new rows
    extra.to_csv(tmp_path / "train" / "flows_extra.csv", index=False)
    assert len(dl.load_flows.__wrapped__(str(tmp_path / "train"))) == base + 3                        # same folder, second file


# ------------------------------------------------------------------------------------------------ knowledge base
def _kb(tmp_path):
    return knowledge.KnowledgeBase(db=tmp_path / "m.db")


def test_kb_is_built_from_raw_data_and_has_boundaries_and_ground_truth(tmp_path):
    kb = _kb(tmp_path)
    kinds = {e["kind"] for e in kb.entries}
    assert kinds == {"boundary", "ground_truth", "insight"} and kb.by_id["B-CAP"]["value"]["has_capacity_data"] is False
    assert kb.by_id["GT-E-RANK"]["value"]["worst"] == "U5" and "Alexanderplatz" in kb.by_id["GT-F-TOP5"]["value"]["top"][0]["station"]
    assert [b["id"] for b in kb.boundaries_for({"cat": "P", "oos": ["capacity"], "dates": []}, "")][:2] == ["B-EVT", "B-NAME"]
    assert kb.search("platform capacity data", k=1)[0]["id"] == "B-CAP"


def test_sanity_check_catches_a_wrong_claim_and_accepts_the_right_one(tmp_path):
    kb = _kb(tmp_path)
    wrong = kb.sanity_check("worst line?", "U9 is worst at 255 Wh per passenger.", {"status": "ok", "cat": "E", "worst": {"line": "U9", "wh": 255}, "rank": []}, {"cat": "E"})
    right = kb.sanity_check("worst line?", "U5 is worst at 550 Wh per passenger.", {"status": "ok", "cat": "E", "worst": {"line": "U5", "wh": 550}, "rank": []}, {"cat": "E"})
    assert wrong["ok"] is False and [c["id"] for c in wrong["checks"] if not c["ok"]] == ["S-TRUTH"] and right["ok"] is True


def test_sanity_check_flags_unlabelled_estimates_capacity_claims_and_strong_weak_correlations(tmp_path):
    kb = _kb(tmp_path)
    f = {"status": "ok", "cat": "P", "mode": "scenario", "top": [{"s": "Kurfürstendamm", "load90": 2241}], "assumed": ["rain=yes"]}
    res = kb.sanity_check("q", "Kurfürstendamm will exceed its capacity with 2241 passengers.", f, {"cat": "P"})
    assert {c["id"] for c in res["checks"] if not c["ok"]} == {"S-BANNED", "S-LABELLED"}
    g = kb.sanity_check("q", "Strong dependency: A and B (r=0.14).", {"status": "ok", "cat": "G", "pairs": [{"a": "Dahlem-Dorf", "b": "Tierpark", "r": 0.14}]}, {"cat": "G"})
    assert any(c["id"] == "S-STRENGTH" and not c["ok"] for c in g["checks"])


def test_history_survives_and_restores_the_last_answered_facts(tmp_path):
    kb = _kb(tmp_path)
    kb.remember_turn("s1", "operator", "q1", "C", "a1", {"status": "ok", "cat": "C", "x": 1})
    kb.remember_turn("s1", "operator", "q2", "FOLLOW", "a2", {"status": "follow"})
    assert kb.last_facts("operator")["x"] == 1 and [h["question"] for h in kb.history("s1")] == ["q1", "q2"]
    assert _kb(tmp_path).stats()["turns"] == 2                                                        # a new process sees the same history


def test_user_insights_are_searchable(tmp_path):
    kb = _kb(tmp_path)
    i = kb.add_insight("Hermannplatz shows no uplift for Uber Arena events", ["A"], "test")
    assert i.startswith("U-") and kb.search("Hermannplatz Uber Arena uplift", cats=["A"], k=1)[0]["id"] == i


def test_cognee_circuit_breaker_trips_and_never_raises(monkeypatch):
    monkeypatch.setenv("COGNEE_ENABLED", "true"); monkeypatch.setenv("COGNEE_API_BASE_URL", "http://127.0.0.1:9"); monkeypatch.setenv("COGNEE_API_KEY", "k")
    c = knowledge.CogneeClient()
    assert c.configured and c.available
    assert c.recall("x") == [] and c.recall("x") == []                                               # two failures (connection refused), no exception
    assert c.available is False and c.last_error and c.stats["errors"] == 2
    monkeypatch.setenv("COGNEE_ENABLED", "false")
    assert knowledge.CogneeClient().configured is False
