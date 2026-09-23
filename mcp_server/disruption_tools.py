"""Category C (disruption response) MCP tools — the solver for questions like
training question 3: "Line X is suspended between A and B — why, how long, how
to reroute, who gets overloaded, where to deploy staff?"

Call chain the agent should follow:

    resolve_closure -> apply_closure -> alternate_paths -> scenario_flow

  resolve_closure   closures.csv lookup (deterministic: reason, start, duration)
  apply_closure     what the closure physically does to the rail graph
  alternate_paths   detour paths (+ geographic bus/walk candidates if none)
  scenario_flow     TabPFN demand baseline + explicit redistribution ASSUMPTIONS
                    -> which stations come under pressure (model estimate, not
                    a measured capacity claim)

All tools after resolve_closure take the SAME arguments — either a
`closure_id` from resolve_closure, or a hypothetical closure described by
`line` + `from_station` + `to_station` (or `station`) + `start` +
`duration_minutes` — so no state is carried between calls.

Registered onto the shared FastMCP server by mcp_server/server.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT / "ml", REPO_ROOT / "dashboard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import disruption as D  # noqa: E402
from demand_baseline import DemandBaseline  # noqa: E402
from scenario import run_scenario  # noqa: E402
from utils.data_loader import (  # noqa: E402
    load_closures,
    load_connections,
    load_flows,
    load_stations,
    station_cols,
)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    return x


def register(mcp, folder: str, get_feature_table: Callable[[], pd.DataFrame]) -> None:
    state: dict = {}

    def net() -> D.Network:
        if "net" not in state:
            state["net"] = D.Network.from_frames(
                load_stations(folder), load_connections(folder), station_cols(load_flows(folder)))
        return state["net"]

    def closures() -> pd.DataFrame:
        return load_closures(folder).reset_index(drop=True)

    def baseline() -> DemandBaseline:
        """Prepared once; restored from the checkpoint (or fitted once) on first scenario_flow call."""
        if "baseline" not in state:
            state["baseline"] = DemandBaseline.prepare(get_feature_table(), closures(), dataset_folder=folder)
        b = state["baseline"]
        if not b.is_fitted:
            b.restore_or_fit()
        return b

    def build_spec(closure_id, line, from_station, to_station, station, start, duration_minutes):
        n = net()
        if closure_id is not None:
            cl = closures()
            if not 0 <= closure_id < len(cl):
                return f"closure_id {closure_id} out of range 0..{len(cl) - 1}. Call resolve_closure."
            spec = D._closure_to_spec(n, closure_id, cl.iloc[closure_id])
            if spec.get("warnings"):
                return "closure could not be resolved to exact stations: " + "; ".join(spec["warnings"])
            return spec
        if not start or not duration_minutes:
            return "Hypothetical closure needs `start` ('YYYY-MM-DD HH:MM') and `duration_minutes`."
        return D.make_hypothetical_spec(
            n, line=line, from_station=from_station, to_station=to_station, station=station,
            start=start, duration_minutes=int(duration_minutes))

    @mcp.tool
    def resolve_closure(query: str = "", max_results: int = 5) -> list[dict]:
        """Category C step 1. Look up closures in closures.csv by free text: a closure id
        ('7'), a date ('2026-09-14'), a line ('U7'), a station name, or a reason keyword
        ('safety inspection'). Empty query lists the latest closures. Returns
        closure_id, exact resolved station names, start/end, duration and reason —
        deterministic facts straight from the data (answers 'why' and 'how long').
        If nothing matches, the closure isn't in the dataset: describe it as a
        hypothetical to apply_closure instead (line + from_station + to_station)."""
        found = D.resolve_closures(net(), closures(), query, max_results)
        return _jsonable(found) or [{"note": f"No closure in closures.csv matches '{query}'. "
                                              "Use a hypothetical closure in apply_closure."}]

    @mcp.tool
    def apply_closure(
        closure_id: int | None = None, line: str | None = None,
        from_station: str | None = None, to_station: str | None = None,
        station: str | None = None, start: str | None = None,
        duration_minutes: int | None = None,
    ) -> dict:
        """Category C step 2. What does the closure do to the rail network? Give EITHER
        a closure_id (from resolve_closure) OR a hypothetical: line ('U7') +
        from_station + to_station (section suspension) or just station (station
        closure), plus start 'YYYY-MM-DD HH:MM' and duration_minutes. Returns the
        stations on the suspended section, which are fully unserved vs still served by
        another line, the rail edges removed, and any groups of stations left with NO
        rail connection to the rest of the network (Berlin's U-Bahn graph is almost a
        tree, so this is common — those need replacement buses)."""
        spec = build_spec(closure_id, line, from_station, to_station, station, start, duration_minutes)
        if isinstance(spec, str):
            return {"error": spec}
        try:
            eff = D.apply_closure(net(), spec)
        except ValueError as e:
            return {"error": str(e)}
        cut_off = D.isolated_components(eff, net())
        return _jsonable({
            "closure": {k: v for k, v in spec.items() if k != "warnings"},
            "section_path": eff.section_path,
            "unserved_stations": eff.unserved,
            "still_served_by_other_line": eff.partially_served,
            "rail_edges_removed": [list(e) for e in eff.blocked_edges],
            "groups_cut_off_from_rail_network": cut_off
                if spec["kind"] == "line_section" else
                {"only_if_trains_do_not_run_through": cut_off},
            "notes": eff.notes,
            "assumptions": D.ASSUMPTIONS[:1] + [
                "Edge->line attribution is inferred (an edge belongs to every line serving both "
                "endpoints); connections.csv has no per-edge line label."],
        })

    @mcp.tool
    def alternate_paths(
        closure_id: int | None = None, line: str | None = None,
        from_station: str | None = None, to_station: str | None = None,
        station: str | None = None, start: str | None = None,
        duration_minutes: int | None = None, max_paths: int = 3,
    ) -> dict:
        """Category C step 3. Detour paths around the closure, fewest stations first:
        the stations, hop count, extra hops vs the suspended route, lines used and
        transfer stations. Same arguments as apply_closure. When NO rail detour exists
        (frequent), returns candidate surface links: geographically closest station
        pairs across the cut (straight-line km) — a plausibility hint for bus/walk
        replacement, NOT an existing service. Rail paths are valid graph paths; nothing
        here measures train capacity or timetable feasibility."""
        spec = build_spec(closure_id, line, from_station, to_station, station, start, duration_minutes)
        if isinstance(spec, str):
            return {"error": spec}
        try:
            eff = D.apply_closure(net(), spec)
        except ValueError as e:
            return {"error": str(e)}
        alts = D.alternate_paths(net(), eff, k=max(1, min(int(max_paths), 5)))
        surface = D.surface_links(net(), eff)
        return _jsonable({
            "closure": {k: spec.get(k) for k in ("closure_id", "kind", "line", "station",
                                                  "from_station", "to_station", "start", "end")},
            "alternates": alts,
            "surface_link_candidates": surface,
            "summary": (
                "Rail detour available." if any(a["paths"] for a in alts)
                else "No rail detour — replacement bus / surface link needed."),
            "caveat": "Graph paths only: no timetable, headway or capacity data exists in the dataset.",
        })

    @mcp.tool
    def scenario_flow(
        closure_id: int | None = None, line: str | None = None,
        from_station: str | None = None, to_station: str | None = None,
        station: str | None = None, start: str | None = None,
        duration_minutes: int | None = None, top_n: int = 10,
    ) -> dict:
        """Category C step 4 — the TabPFN-backed estimate. For each 15-min slot of the
        closure window: TabPFN regression predicts each involved station's normal
        (counterfactual) demand distribution; displaced demand is redistributed by an
        EXPLICIT ASSUMPTION (reported at low/base/high diversion share, never as one
        confident number); returns the stations ranked by probability of exceeding
        their OWN historical 95th percentile for that hour/day-type, with the added
        load, the no-closure exceedance probability for comparison, and — when the
        closure is in the dataset — a historical reality check (observed vs baseline).
        Window must lie inside the dataset coverage window. Output is a model-based
        demand-pressure estimate, NOT a measured capacity claim: the dataset has no
        capacity data, and its historical closures show no measurable redistribution.
        Same arguments as apply_closure. First call fits the TabPFN model (~1 min)."""
        spec = build_spec(closure_id, line, from_station, to_station, station, start, duration_minutes)
        if isinstance(spec, str):
            return {"error": spec}
        try:
            eff = D.apply_closure(net(), spec)
            b = baseline()
        except (ValueError, RuntimeError) as e:
            return {"error": str(e)}
        result = run_scenario(net(), b, eff, top_n=max(1, min(int(top_n), 25)))
        if "error" not in result:
            result["deployment_hint"] = (
                "ranked_pressure_stations is ordered by prob_exceed_own_p95 (base assumption); "
                "stations whose probability rises well above prob_exceed_without_closure are the "
                "staff-deployment candidates. Ranking depends on the assumed diversion share — "
                "cite the low/high columns.")
        return _jsonable(result)
