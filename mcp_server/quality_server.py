"""MCP server for the QUALITY DATABASE — normalized, high-quality data and the boundaries derived from it (FastMCP, stdio / HTTP / in-memory).

    ./.venv/bin/python mcp_server/quality_server.py                # stdio
    MCP_TRANSPORT=http MCP_PORT=8768 ./.venv/bin/python mcp_server/quality_server.py

The database (data/quality/, built by `./.venv/bin/python ml/quality_db.py build`) holds, for every station and 15-minute slot of the training AND test split, the *normal flow*
(what is normal for that time of day and day type, typical weather; fitted on clean cells), the weather part and the rest (events, closures, anomalies) — plus boundaries per station /
day type / hour, station ceilings, network bounds, episodes (events, closures) with their measured effect, unexplained spikes, outages and the data-quality issues found while building it.

Two groups of tools: `quality_*` (the derived database: boundaries and checks) and `golden_*` (the pre-processed files themselves — data/normalized and data/processed; catalog with `golden_datasets`, schema in
data/data_schema_high_quality.md).

Who uses it: the **Inspector** (the evaluator agent) calls `quality_check_facts` before an answer is shown — every passenger figure of the facts is checked against the boundaries;
`quality_check_value` / `quality_boundaries` / `quality_normal_flow` answer single questions ("is 2 600 plausible at Kurfürstendamm on a Thursday 09:15?"); any other MCP client can too.
All answers are deterministic and cite the database numbers they used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastmcp import FastMCP

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "ml"), str(REPO)]
import quality_db  # noqa: E402

mcp = FastMCP("nextmove-quality", instructions=(
    "High-quality normalized data of the Berlin U-Bahn passenger flows and the boundaries derived from it. Use quality_check_facts to verify the facts of an answer, "
    "quality_check_value for one figure, quality_boundaries for the plausible range at a station / time, quality_normal_flow for normal vs actual, and quality_data_issues "
    "for what is wrong with the raw data. Passengers are per 15-minute slot unless stated otherwise."))


def _db() -> quality_db.QualityDB:
    return quality_db.load()


@mcp.tool
def quality_status() -> dict:
    """Is the quality database built, its coverage (training and test window), size of every table, build info."""
    return _db().status()


@mcp.tool
def quality_normal_flow(station: str, at: str) -> dict:
    """Normal vs actual at one station and 15-minute slot (`at` = 'YYYY-MM-DD HH:MM'): actual, normal (typical weather), expected with the actual weather, weather effect %, the rest (event / closure /
    anomaly share) in % , whether an event / closure window was active, the boundary for that slot, and the episodes involved."""
    return _db().normal_at(station, at)


@mcp.tool
def quality_boundaries(station: str, at: str) -> dict:
    """The plausible range at a station for the weekday type and hour of `at`: normal median, mean of clean slots, p05 / p50 / p95 / p99, the normal band, the highest value ever observed at the station and the
    hard / soft ceilings (1.5x / 1.15x that maximum)."""
    b = _db().resolve(station)
    return {"error": f"unknown station '{station}'"} if b is None else {"station": b, **(_db().boundary(b, at) or {"error": "no boundary for that time"})}


@mcp.tool
def quality_check_value(station: str, at: str, value: float) -> dict:
    """Is `value` passengers per 15 minutes plausible at that station and time? verdict: normal | high | low | extreme | impossible, with the reason and the boundary used."""
    return _db().check_value(station, at, value)


@mcp.tool
def quality_station_profile(station: str) -> dict:
    """A station's bounds (max ever, peak hours weekday / weekend, daily mean, clipping share), weekday normal flow by hour and its weather effects (rain, heat, holiday) in %."""
    return _db().station_profile(station)


@mcp.tool
def quality_episodes(date: str = "", station: str = "", limit: int = 20) -> list[dict]:
    """Events (>= 2 000 visitors, mapped to a station) and closures (every station on the closed section) in the database, with their measured peak uplift / mean deviation at the anchor stations.
    Filter by `date` (YYYY-MM-DD) and / or `station`."""
    return _db().episodes(date, station, limit)


@mcp.tool
def quality_event_effects() -> list[dict]:
    """How much events (< 5 000 and >= 5 000 visitors) and closures changed the flow at their anchor stations in this data: median / p90 / max peak ratio and mean log deviation."""
    return _db().event_effects()


@mcp.tool
def quality_anomalies(date: str, top_n: int = 5) -> list[dict]:
    """The largest unexplained SPIKES of a day (outside any event / closure window): station, time, actual vs normal, deviation %."""
    return _db().anomalies(date, top_n)


@mcp.tool
def quality_weather_effects(station: str = "") -> dict:
    """Rain / heat / holiday effects in % on the flow: the network median, or one station's."""
    return _db().weather_effects(station)


@mcp.tool
def quality_network_bounds(day_type: int = -1, hour: int = -1) -> list[dict]:
    """Network-wide passenger totals per 15 minutes by day type (0 Mon-Thu, 1 Fri, 2 Sat, 3 Sun) and hour: p05 / p50 / p95 / max and the normal median. -1 = all."""
    return _db().network_bounds(None if day_type < 0 else day_type, None if hour < 0 else hour)


@mcp.tool
def quality_data_issues() -> list[dict]:
    """What is wrong or limited in the raw data, found while building the database: clipping at 500, zero-readings (outages), noise level, dropped columns, coverage, test-split shift."""
    return _db().data_issues()


@mcp.tool
def quality_check_facts(category: str, facts_json: str) -> dict:
    """THE INSPECTOR'S CHECK. `facts_json` = the facts of an answer (as the Analyst produced them), `category` = A event impact · B anomalies · C closure · D station profile · F fragmentation · H reroute ·
    P pressure ranking. Every passenger figure is checked against the boundaries. Returns {checks:[{id, ok, hard, detail, evidence}], boundaries:[...], summary:{n_checks, n_hard_failed, n_soft_flags}}.
    hard = impossible or contradicts the raw data (an answer must not be shown); soft = unusual, mention it."""
    try:
        facts = json.loads(facts_json)
    except json.JSONDecodeError as e:
        return {"available": True, "checks": [], "boundaries": [], "error": f"facts_json is not valid JSON: {e}", "summary": {"n_checks": 0, "n_hard_failed": 0, "n_soft_flags": 0}}
    return _db().check_facts(category, facts)


# ==================================================================================================== golden data: the pre-processed files themselves
# data/normalized/* (normalization pipeline tables + fitted model) and data/processed/* (geocode cache). Schema: data/data_schema_high_quality.md
import golden_data  # noqa: E402


def _g() -> golden_data.Golden:
    return golden_data.Golden()


@mcp.tool
def golden_datasets() -> list[dict]:
    """START HERE. Catalog of the pre-processed high-quality files: name, path, kind, split (train / test), size, unit and what each contains — normalized_flows (total), normalized_weather, normalized_rest,
    normal_flow_passengers (each also `_test`), episodes, normal_flow_coefficients, normal_flow_model (fitted model), geocode_cache_pipeline. Full schema: data/data_schema_high_quality.md."""
    return _g().catalog()


@mcp.tool
def golden_describe(name: str) -> dict:
    """Schema and statistics of one golden dataset (rows, columns, time range, NaN share, min / p05 / median / p95 / max, unit, description; model: terms and window; cache: entries)."""
    return _g().describe(name)


@mcp.tool
def golden_series(station: str, start: str, end: str, max_points: int = 200) -> dict:
    """One station between `start` and `end` ('YYYY-MM-DD HH:MM') across all four normalized tables: actual (reconstructed from the tables: the raw reading within rounding), normal, total, weather, rest per 15-minute slot,
    with the split (train / test). Use it to see what was normal, what the weather explains and what is left (event / closure / anomaly)."""
    return _g().series(station, start, end, max_points)


@mcp.tool
def golden_slice(at: str, table: str = "normalized_rest", top_n: int = 10, order: str = "desc") -> dict:
    """All stations at one 15-minute slot from one table (normalized_flows | normalized_weather | normalized_rest | normal_flow_passengers), largest first (`order` desc) or smallest (asc):
    'which stations deviated most from normal at 09:15?'."""
    return _g().slice(at, table, top_n, order)


@mcp.tool
def golden_episodes(date: str = "", station: str = "", kind: str = "", limit: int = 30) -> list[dict]:
    """The episodes table (events >= 2 000 visitors mapped to a station, closures with EVERY station on the closed section): id, kind, type, name, start, end, anchor stations, attendance, how the
    stations were found, split. Filter by date (YYYY-MM-DD), station and kind (event | closure)."""
    return _g().episodes(date, station, kind, limit)


@mcp.tool
def golden_coefficients(station: str = "") -> dict:
    """Weather / holiday effects in % from the normal-flow model (rain now, rain in the previous hour, both, +5 degC, school holiday weekend / weekday): one station, or the network median and extremes."""
    return _g().coefficients(station)


@mcp.tool
def golden_normal_flow(station: str, at: str, periods: int = 1) -> dict:
    """The MODEL's normal flow (typical weather) in passengers per 15 minutes for `periods` consecutive slots from `at` — for ANY timestamp, also outside the data window (e.g. a date after 2026-10-01;
    the trend term is clipped there, `outside_training_window` says so). Use it as the reference when no measurement exists."""
    return _g().normal_flow(station, at, periods)


@mcp.tool
def golden_model_info() -> dict:
    """The fitted normal-flow model: grid, number of stations, design-matrix terms, training window, mean temperature."""
    return _g().model_info()


@mcp.tool
def golden_venue_station(venue: str = "", address: str = "", k: int = 3) -> dict:
    """Which U-Bahn station serves a venue / address? Manual override (domain knowledge) first, then the geocode cache + nearest stations (offline: never a network request), then a station name in the text."""
    return _g().venue_station(venue, address, k)


@mcp.tool
def golden_geocode_cache(query: str = "", limit: int = 20) -> list[dict]:
    """Entries of the geocode cache (data/processed): 'venue|address' key, lat, lon, source (nominatim | not_found), the query used. Filter by text."""
    return _g().geocode_cache(query, limit)


if __name__ == "__main__":
    import os

    if os.environ.get("MCP_TRANSPORT") == "http":
        mcp.run(transport="http", host="127.0.0.1", port=int(os.environ.get("MCP_PORT", "8768")), show_banner=False)
    else:
        mcp.run()
