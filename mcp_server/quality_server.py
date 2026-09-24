"""MCP server for the QUALITY DATABASE — normalized, high-quality data and the boundaries derived from it (FastMCP, stdio / HTTP / in-memory).

    ./.venv/bin/python mcp_server/quality_server.py                # stdio
    MCP_TRANSPORT=http MCP_PORT=8768 ./.venv/bin/python mcp_server/quality_server.py

The database (data/quality/, built by `./.venv/bin/python ml/quality_db.py build`) holds, for every station and 15-minute slot of the training AND test split, the *normal flow*
(what is normal for that time of day and day type, typical weather; fitted on clean cells), the weather part and the rest (events, closures, anomalies) — plus boundaries per station /
day type / hour, station ceilings, network bounds, episodes (events, closures) with their measured effect, unexplained spikes, outages and the data-quality issues found while building it.

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


if __name__ == "__main__":
    import os

    if os.environ.get("MCP_TRANSPORT") == "http":
        mcp.run(transport="http", host="127.0.0.1", port=int(os.environ.get("MCP_PORT", "8768")), show_banner=False)
    else:
        mcp.run()
