"""Offline tests for the fast pipeline's deterministic parts (router entities/categories, number guard,
template fallback). No LLM, no MCP server, no network."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent")]

import os  # noqa: E402

os.environ.setdefault("DATA_DIR", str(REPO / "data"))
import router  # noqa: E402
import writer  # noqa: E402


def test_training_questions_route_to_their_category():
    q3 = ("Line U6 is suspended on a section between Hallesches Tor and Kaiserin-Augusta-Strasse stations. "
          "What is the reason behind this closure and how long will it last?")
    p = router.route(q3)
    assert p["cat"] == "C" and p["lines"] == ["U6"]
    assert p["stations"] == ["U Hallesches Tor (Berlin)", "U Kaiserin-Augusta-Str. (Berlin)"]
    assert "scenario_flow" in p["tools"] and "closures" in p["data"]
    assert router.route("At what time does the commute flow peak at Rudow station usually take place?")["cat"] == "D"
    assert router.route("Rank the five stations whose closure would fragment the network the most.")["cat"] == "F"
    assert router.route("Which line has the worst energy-per-passenger ratio?")["cat"] == "E"


def test_entity_extraction():
    assert router.find_dates("closed on August 17th and 2026-09-21 and 30 June") == ["2026-09-21", "2026-08-17", "2026-06-30"]
    assert router.find_time("starting at 17:00") == "17:00"
    assert router.find_duration_min("for two hours") == 120 and router.find_duration_min("for 90 minutes") == 90
    assert router.find_month("in December") == "2026-12"
    assert router.route("What if the U2 were suspended tomorrow")["what_if"] is True


def test_station_resolution_and_typos():
    assert router.find_stations("Neukölln station is closed") == ["S+U Neukölln (Berlin)"]   # not Rathaus Neukölln
    assert router.find_stations("Alex and Kotti") == ["S+U Alexanderplatz Bhf (Berlin)", "U Kottbusser Tor (Berlin)"]
    assert router.route("When does Rudov peak?")["stations"] == ["U Rudow (Berlin)"]
    assert router.route("How busy is Kreuzberg usually?")["stations"] == []   # not a station: must not guess


def test_capacity_and_oos_and_followup():
    assert router.route("How many passengers can the platform safely hold?")["cat"] == "OOS"
    assert router.route("How confident are you about those stations?", has_history=True)["cat"] == "FOLLOW"


FACTS = {"status": "ok", "cat": "C", "cl": {"from": "2026-07-13 13:50", "to": "2026-07-13 15:20", "h": 1.5},
         "press": [{"s": "X", "p": 78, "p0": 10, "add": 153}], "disp": {"Y": {"tot": 1363}}}


def test_guard_accepts_grounded_numbers_and_rejects_invented_ones():
    ok = "Closed 13:50 to 15:20 (1.5 h). X: 78% chance vs 10% normally, +153 passengers; 1,363 displaced. Assumes 25%/50%/75% diversion."
    assert writer.find_ungrounded(ok, FACTS) == []
    assert writer.find_ungrounded("Expect 412 extra passengers at X", FACTS) == ["412"]


def test_guard_bans_capacity_and_bus_service_claims():
    assert writer.find_banned("78% chance of exceeding its capacity") == ["capacity claim"]
    assert writer.find_banned("No capacity data exists.") == []
    assert writer.find_banned("Bus services are available between A and B") == ["bus service claim"]
    assert writer.find_banned("A replacement bus would be needed") == []


def test_fallback_templates_are_grounded():
    facts = {"status": "ok", "cat": "C", "cl": {"line": "U6", "a": "A", "b": "B", "st": None, "from": "2026-07-13 13:50",
             "to": "2026-07-13 15:20", "h": 1.5, "why": "safety inspection"},
             "alt": {"rail": [], "bus": [["A", "B", 1.3]]}, "press": [{"s": "X", "p": 78, "p0": 10}]}
    text = writer.render_fallback(facts)
    assert writer.find_ungrounded(text, facts) == [] and writer.find_banned(text) == []
    assert "safety inspection" in text and "78%" in text
