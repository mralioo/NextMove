"""The normalization pipeline (ml/nextmove_pipeline) and the quality database / MCP server. Synthetic data for the pipeline; the database tests run on the real build and are skipped without it."""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "ml"), str(REPO / "agent"), str(REPO / "mcp_server")]

from nextmove_pipeline import baseline, config, episodes, loading, names  # noqa: E402
import quality_db  # noqa: E402

HAVE_DB = (REPO / "data" / "quality" / "quality.db").exists()
needs_db = pytest.mark.skipif(not HAVE_DB, reason="quality database not built (./.venv/bin/python ml/quality_db.py build)")


# ------------------------------------------------------------------------------------------------ pipeline pieces
def test_station_names_match_across_spellings():
    idx = names.StationNameIndex(["S+U Warschauer Str. (Berlin)", "U Spichernstr. (Berlin)", "S+U Friedrichstr. Bhf (Berlin)"])
    assert names.station_key("S+U Warschauer Str. (Berlin)") == "warschauer strasse"
    assert idx.resolve("Warschauer Straße") == "S+U Warschauer Str. (Berlin)" and idx.resolve("Spichernstr.") == "U Spichernstr. (Berlin)"
    assert idx.find_in_text("Concert near Friedrichstraße and Spichernstr.") == ["S+U Friedrichstr. Bhf (Berlin)", "U Spichernstr. (Berlin)"]


def test_loader_repairs_mojibake_and_reads_a_headerless_events_file(tmp_path):
    assert loading._fix_mojibake("U KurfÃ¼rstenstr.") == "U Kurfürstenstr." and loading._fix_mojibake("ALIZÃ‰") == "ALIZÉ" and loading._fix_mojibake("plain") == "plain"
    (tmp_path / "flows_rest.csv").write_bytes("﻿timestamp,U KurfÃ¼rstenstr. (Berlin)\n9/22/2026 5:00,10\n".encode("utf-8"))
    df = loading.load_flows(tmp_path)
    assert list(df.columns) == ["U Kurfürstenstr. (Berlin)"] and df.index[0] == pd.Timestamp("2026-09-22 05:00")
    hdr = ",".join(pd.read_csv(config.DEFAULT_DATA_DIR / "berlin_events_summer_2026_pre_innotrans.csv", nrows=0).columns)
    row = "Test Concert,2026-09-22T20:00:00+02:00,2026-09-22T22:00:00+02:00,Arena,Street 1,Berlin,Germany,Music,Rock,x,2500,http://x"
    (tmp_path / "berlin_events_summer_2026_rest.csv").write_text(row + "\n", encoding="utf-8-sig")
    ev = loading.load_events(tmp_path)
    assert ev.event_name.iloc[0] == "Test Concert" and int(ev.estimated_attendance.iloc[0]) == 2500 and hdr.startswith("event_name")


def test_closure_of_a_line_section_closes_every_station_between_the_ends():
    st = pd.DataFrame({"station_name": list("ABCDE"), "u_bahn_lines": ["U1", "U1", "U1", "U1", "U2"]})
    adj = {"A": {"B"}, "B": {"A", "C"}, "C": {"B", "D"}, "D": {"C", "E"}, "E": {"D"}}
    ix = names.StationNameIndex(list(st.station_name))
    got, how = episodes.closure_stations("Line U1 suspended on a section between A and D due to track work.", ix, adj, st)
    assert got == ["A", "B", "C", "D"] and "section" in how
    assert episodes.parse_duration("2h30min") == pd.Timedelta(hours=2, minutes=30) and episodes.parse_duration("junk") is None


def test_decomposition_is_exact_and_an_event_shows_up_in_the_rest():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-06-01", periods=24 * 4 * 42, freq="15min")
    hour = idx.hour + idx.minute / 60
    base = 100 + 80 * np.exp(-((hour - 8) ** 2) / 4) + 90 * np.exp(-((hour - 18) ** 2) / 4)
    flows = pd.DataFrame({"S1": rng.poisson(base * 1.0).astype(float), "S2": rng.poisson(base * 0.5).astype(float)}, index=idx)
    ev_slot = (idx >= "2026-06-20 19:00") & (idx < "2026-06-20 21:00")
    flows.loc[ev_slot, "S1"] *= 4
    weather = pd.DataFrame({"temp": 20 + 5 * np.sin(np.arange(len(idx)) / 96), "prcp": 0.0, "wspd": 10.0}, index=idx)
    adj = {"S1": {"S2"}, "S2": {"S1"}}
    eps = [dict(anchors=["S1"], start=pd.Timestamp("2026-06-20 19:00"), end=pd.Timestamp("2026-06-20 21:00"))]
    model = baseline.fit_normal_model(flows, weather, adj, eps, "15min")
    t = baseline.decompose(model, flows, weather)
    assert np.allclose(t["total"].to_numpy(), (t["weather"] + t["rest"]).to_numpy(), atol=1e-3)                 # total = weather + rest, exactly
    assert t["rest"].loc[ev_slot, "S1"].mean() > 0.8 and abs(t["rest"].loc[~ev_slot, "S1"].median()) < 0.15    # the event is in the rest, the normal days are ~0


# ------------------------------------------------------------------------------------------------ the quality database
@needs_db
def test_quality_database_answers_normal_boundaries_and_verdicts():
    db = quality_db.QualityDB()
    st = db.status()
    assert st["available"] and st["n_stations"] == 167 and st["train_range"][0].startswith("2026-06-10") and st["test_range"][1].startswith("2026-10-01")
    n = db.normal_at("Rudow", "2026-07-14 18:00")
    assert n["station"].startswith("U Rudow") and n["normal"] > 0 and n["boundary"]["hard_ceiling"] > n["boundary"]["station_max_ever"]
    assert db.check_value("Rudow", "2026-07-14 18:00", 100)["verdict"] in ("normal", "low")
    assert db.check_value("Rudow", "2026-07-14 18:00", 99999)["verdict"] == "impossible"
    assert any(e["kind"] == "closure" for e in db.episodes(date="2026-09-25")) and db.event_effects() and db.data_issues()


@needs_db
def test_check_facts_passes_true_facts_and_fails_impossible_ones():
    db = quality_db.QualityDB()
    ok = db.check_facts("D", {"status": "ok", "cat": "D", "st": [{"s": "Rudow", "wk": [18, 223], "we": [8, 143], "net": 266}]})
    assert ok["summary"]["n_hard_failed"] == 0 and ok["boundaries"]
    bad = db.check_facts("D", {"status": "ok", "cat": "D", "st": [{"s": "Rudow", "wk": [3, 900]}]})
    assert bad["summary"]["n_hard_failed"] == 1 and bad["checks"][0]["id"] == "Q-PEAK-HOUR"
    assert db.check_facts("P", {"status": "ok", "cat": "P", "top": [{"s": "Rudow", "load90": 9000}]})["summary"]["n_hard_failed"] == 1
    assert db.check_facts("B", {"status": "ok", "cat": "B", "found": [{"s": "Kurfürstendamm", "at": "2026-07-23 09:15", "obs": 5000, "usual": 304}]})["summary"]["n_hard_failed"] == 1
    assert db.check_facts("H", {"status": "ok", "cat": "H", "obs_over_exp": {"hop1": 0.95}})["summary"]["n_hard_failed"] == 0


@needs_db
def test_quality_mcp_server_and_the_inspector_use_it():
    import evaluator
    import quality_mcp
    from schemas import WorkerResult

    async def go():
        assert (await quality_mcp.call("quality_status"))["available"]
        r = await quality_mcp.call("quality_check_facts", category="P", facts_json=json.dumps({"status": "ok", "cat": "P", "top": [{"s": "Rudow", "load90": 9000}]}))
        assert r["summary"]["n_hard_failed"] == 1
        checks, bnd = [], []
        res = WorkerResult(task_id="t", iteration=1, status="ok", facts={"status": "ok", "cat": "P", "top": [{"s": "Rudow", "load90": 9000}]}, confidence=0.9)
        await evaluator._quality_checks("P", res, checks, bnd)
        assert any(c.id == "Q-H-CEILING" and not c.ok for c in checks) and any(b.startswith("Q-BOUND:") for b in bnd)
        good = WorkerResult(task_id="t", iteration=1, status="ok", facts={"status": "ok", "cat": "D", "st": [{"s": "Rudow", "wk": [18, 223]}]}, confidence=0.9)
        c2: list = []
        await evaluator._quality_checks("D", good, c2, [])
        assert all(c.ok for c in c2)
    asyncio.run(go())


# ------------------------------------------------------------------------------------------------ golden data (the pre-processed files) through the MCP server
HAVE_NORM = (REPO / "data" / "normalized" / "normalized_rest.csv").exists()


@pytest.mark.skipif(not HAVE_NORM, reason="normalized tables not built")
def test_golden_data_reads_the_normalized_files_and_the_model():
    import golden_data
    g = golden_data.Golden()
    cat = {c["name"]: c for c in g.catalog()}
    assert {"normalized_flows", "normalized_rest_test", "episodes", "normal_flow_model", "geocode_cache_pipeline"} <= set(cat) and all(c["exists"] for c in cat.values())
    d = g.describe("normalized_rest")
    assert d["rows"] == 8320 and d["columns"] == 167 and d["nan_share"] == 0.0 and g.describe("normalized_rest_test")["rows"] == 720
    row = g.series("Rudow", "2026-07-14 17:00", "2026-07-14 17:00")["rows"][0]
    assert abs(row["total"] - (row["weather"] + row["rest"])) < 2e-3                                            # total = weather + rest
    assert abs(row["normal"] - g.normal_flow("Rudow", "2026-07-14 17:00")["rows"][0]["normal"]) < 0.2          # the model reproduces the stored normal
    assert g.normal_flow("Rudow", "2026-10-15 18:00")["outside_training_window"] is True
    assert any(e["kind"] == "closure" for e in g.episodes(date="2026-09-25")) and g.coefficients("Rudow")["rain_both_pct"] is not None
    assert g.venue_station("Uber Arena", "Uber-Platz 1")["stations"][0]["station_name"].startswith("S+U Warschauer")
    assert g.venue_station("", "Skalitzer Straße 85-86")["stations"][0]["station_name"].startswith("U Schlesisches Tor")


@pytest.mark.skipif(not HAVE_NORM, reason="normalized tables not built")
def test_golden_tools_are_served_by_the_quality_mcp_server():
    import quality_mcp

    async def go():
        cat = await quality_mcp.golden("golden_datasets")
        assert any(c["name"] == "normal_flow_model" for c in cat)
        s = await quality_mcp.golden("golden_slice", at="2026-07-23 09:15", table="normalized_rest", top_n=3)
        assert len(s["values"]) == 3
        assert (await quality_mcp.golden("golden_geocode_cache", query="berghain"))[0]["source"] == "nominatim"
    asyncio.run(go())
