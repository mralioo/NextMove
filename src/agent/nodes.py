"""
LangGraph node functions for the U-Bahn operator agent.
Implements: keyword-first classification, clarification loop, parallel tool execution,
            response_id for feedback linkage.
"""
import sys
import json
import os
import re
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

try:
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    class HumanMessage:
        def __init__(self, content): self.content = content
    class AIMessage:
        def __init__(self, content): self.content = content

try:
    from ai_connection import AzureOpenAIClient
    _llm_client = AzureOpenAIClient()
    _LLM_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load ai_connection: {e}")
    _llm_client = None
    _LLM_AVAILABLE = False

from agent.state import AgentState
from agent.prompts import (
    SYSTEM_PROMPT, INTENT_EXTRACTION_PROMPT, CLARIFICATION_PROMPT, SYNTHESIS_PROMPT
)
from data_pipeline.loader import DataStore
from hcade.fingerprint import build_fingerprint
from hcade.recommender import recommend_action


# -----------------------------------------------------------------------
# LLM helpers
# -----------------------------------------------------------------------
def _call_llm(prompt: str, system_context: str = "") -> str:
    if not _LLM_AVAILABLE or _llm_client is None:
        return "[LLM unavailable]"
    try:
        full_prompt = f"{system_context}\n\n{prompt}".strip() if system_context else prompt
        return _llm_client.ask(full_prompt)
    except Exception as e:
        return f"[LLM error: {e}]"


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, stripping markdown code fences."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return {}


# -----------------------------------------------------------------------
# Keyword-first classifier — zero LLM cost for ~80% of queries
# -----------------------------------------------------------------------
_KW_MAP = [
    ("DISRUPTION", ["suspended", "suspension", "closure", "closed", "disruption",
                    "reroute", "line down", "out of service", "halt", "blocked"]),
    ("EVENT",      ["concert", "event", "festival", "match", "crowd", "fan",
                    "stadium", "arena", "messe", "innotrans", "expo", "conference",
                    "christmas market", "market", "fair"]),
    ("ANOMALY",    ["anomaly", "anomalies", "unusual", "spike", "unexplained",
                    "strange", "outlier", "irregular"]),
    ("EFFICIENCY", ["energy", "efficiency", "consumption", "mwh", "watt",
                    "kwh", "co2", "green", "emissions"]),
    ("NETWORK",    ["fragment", "centrality", "critical station", "bottleneck",
                    "resilience", "disconnect", "most important station"]),
    ("FORECAST",   ["forecast", "predict", "expect", "upcoming", "will happen",
                    "future", "anticipate"]),
    ("BEHAVIOR",   ["prefer", "behaviour", "behavior", "alternative route",
                    "rerouting preference", "actually take", "passengers choose"]),
]

# Month name → number mapping for natural language date parsing
_MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
}

# Known venue → nearby station mapping (supplements event_mapper VENUE_COORDS)
_VENUE_TO_STATION = {
    'uber arena':           'U Warschauer Str. (Berlin)',
    'mercedes-benz arena':  'U Warschauer Str. (Berlin)',
    'mercedes benz arena':  'U Warschauer Str. (Berlin)',
    'olympiastadion':       'U Olympia-Stadion (Berlin)',
    'waldbuehne':           'U Olympia-Stadion (Berlin)',
    'waldbuhne':            'U Olympia-Stadion (Berlin)',
    'tempodrom':            'U Gleisdreieck (Berlin)',
    'messe berlin':         'U Kaiserdamm (Berlin)',
    'velodrom':             'U Landsberger Allee (Berlin)',
    'columbiahalle':        'U Platz der Luftbruecke (Berlin)',
    'admiralspalast':       'U Friedrichstr. (Berlin)',
    'arena berlin':         'U Treptower Park (Berlin)',
    'huxleys':              'U Platz der Luftbruecke (Berlin)',
    'berliner philharmonie':'U Potsdamer Platz (Berlin)',
}


def _parse_natural_date(query: str) -> str | None:
    """
    Parse natural language dates like 'June 23rd', 'July 20', '23. June'
    into 'YYYY-MM-DD' using the training year 2026.
    Returns None if no date found.
    """
    # Already in ISO format
    iso = re.search(r'\d{4}-\d{2}-\d{2}', query)
    if iso:
        return iso.group(0)

    # "June 23rd", "June 23", "23rd June", "23 June"
    month_day = re.search(
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:st|nd|rd|th)?\b',
        query, re.IGNORECASE
    )
    day_month = re.search(
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
        query, re.IGNORECASE
    )

    month_str, day_str = None, None
    if month_day:
        month_str = month_day.group(1).lower()
        day_str   = month_day.group(2)
    elif day_month:
        day_str   = day_month.group(1)
        month_str = day_month.group(2).lower()

    if month_str and day_str:
        month_num = _MONTH_MAP.get(month_str[:3]) or _MONTH_MAP.get(month_str)
        if month_num:
            return f"2026-{month_num:02d}-{int(day_str):02d}"

    # "July 20-26" range → take start
    range_m = re.search(
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})-(\d{1,2})\b',
        query, re.IGNORECASE
    )
    if range_m:
        month_num = _MONTH_MAP.get(range_m.group(1)[:3].lower())
        if month_num:
            return f"2026-{month_num:02d}-{int(range_m.group(2)):02d}"

    return None


def _resolve_venue_station(query: str) -> str | None:
    """
    If the query mentions a known venue, return the nearest U-Bahn station.
    This lets EVENT queries work without the operator specifying a station.
    """
    lq = query.lower()
    for venue, station in _VENUE_TO_STATION.items():
        if venue in lq:
            return station
    return None


def _fast_classify(query: str) -> str | None:
    """Return query_type instantly via keywords, or None to fall back to LLM."""
    lq = query.lower()
    for qt, kws in _KW_MAP:
        if any(k in lq for k in kws):
            return qt
    return None


def _enhance_prompt(query: str) -> str:
    """
    Prompt Enhancement: annotate the query with auto-detected structured hints
    (line numbers, dates/times, station name fragments) so the LLM extracts
    intent slots more reliably on the first pass.
    """
    lines_found   = re.findall(r'\bU\d{1,2}\b', query)
    times_found   = re.findall(
        r'\b\d{1,2}:\d{2}\b|\b\d{4}-\d{2}-\d{2}\b'
        r'|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}\b',
        query, re.IGNORECASE
    )
    station_hints = re.findall(
        r'(?:at|near|station|from|to)\s+([A-Z][A-Za-z\s\-]+?)(?:\.|,|\s+and|\s+is|\s+was|$)',
        query
    )
    extras = []
    if lines_found:
        extras.append(f"[LINES DETECTED: {', '.join(set(lines_found))}]")
    if times_found:
        extras.append(f"[TIMES DETECTED: {', '.join(times_found[:3])}]")
    if station_hints:
        extras.append(f"[STATIONS HINT: {', '.join(s.strip() for s in station_hints[:3])}]")
    return (query + "\n" + " ".join(extras)) if extras else query


# -----------------------------------------------------------------------
# Clarification logic
# -----------------------------------------------------------------------
# Which query types REQUIRE which fields to proceed with meaningful analysis
_REQUIRED_FIELDS = {
    "ANOMALY":    ["station"],   # anomaly detection is per-station, not network-wide
    "DISRUPTION": [],            # can run network-wide without explicit station/line
    "EVENT":      [],            # area context is nice but not blocking
    "EFFICIENCY": [],            # works network-wide
    "NETWORK":    [],            # works network-wide
    "FORECAST":   [],            # works with any event context
    "BEHAVIOR":   [],
    "GENERAL":    [],
    "COMBINED":   [],
}

_FIELD_LABELS = {
    "station":    "which station you want to analyse",
    "line":       "which U-Bahn line is affected",
    "datetime":   "the specific date and time for the analysis",
    "event_name": "the name or type of event",
}


def _check_missing_fields(query_type: str, parsed: dict) -> list:
    """Return list of field names required but absent from parsed context."""
    missing = []
    for field in _REQUIRED_FIELDS.get(query_type, []):
        val = parsed.get(field)
        if not val or (isinstance(val, list) and len(val) == 0):
            missing.append(field)
    return missing


# -----------------------------------------------------------------------
# Tool routing
# -----------------------------------------------------------------------
_TOOL_PLANS = {
    "DISRUPTION": ["get_disruption_info", "simulate_crowd_cascade",
                   "compute_rerouting", "find_historical_analogues"],
    "EVENT":      ["get_event_context", "get_station_flow", "forecast_surge"],
    "ANOMALY":    ["detect_flow_anomaly", "get_event_context",
                   "get_disruption_info", "get_weather_at_time"],
    "EFFICIENCY": ["get_energy_efficiency"],
    "NETWORK":    ["get_network_centrality"],
    "FORECAST":   ["forecast_surge", "find_historical_analogues"],
    "BEHAVIOR":   ["analyze_route_preferences"],
    "COMBINED":   ["get_disruption_info", "simulate_crowd_cascade", "compute_rerouting",
                   "find_historical_analogues", "get_event_context", "get_station_flow",
                   "forecast_surge"],
    "GENERAL":    ["get_station_flow", "get_flow_baseline"],
}


# -----------------------------------------------------------------------
# Node 1: Intent Extractor
# -----------------------------------------------------------------------
def intent_extractor_node(state: AgentState) -> dict:
    """
    1. Keyword-first classification (~0ms, no LLM) for clear query types.
    2. Falls back to LLM for ambiguous queries, using an enhanced prompt.
    3. Auto-resolves venue→station and natural language dates.
    4. Checks for missing required fields → returns CLARIFICATION_NEEDED if any.
    No hardcoded station or date defaults are applied here.
    """
    messages = state.get("messages", [])
    last_human = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or (hasattr(m, "__class__") and "Human" in m.__class__.__name__):
            last_human = m.content
            break
    if not last_human:
        last_human = "No query provided."

    # Stage 1: fast keyword classification
    query_type = _fast_classify(last_human)
    parsed_context: dict = {}

    if query_type is not None:
        lines_m  = re.findall(r'\bU\d{1,2}\b', last_human)
        # Use natural-language date parser (catches "June 23rd", "July 20-26", ISO dates)
        dt_parsed = _parse_natural_date(last_human)
        parsed_context = {
            "query_type":       query_type,
            "station":          None,
            "line":             lines_m[0] if lines_m else None,
            "lines":            list(set(lines_m)),
            "disrupted_line":   lines_m[0] if lines_m else None,
            "disruption_type":  None,
            "objective":        None,
            "datetime":         dt_parsed,   # None if not found — no hardcoded default
            "time_horizon":     None,
            "event_name":       None,
            "event_attendance": None,
            "stations":         [],
        }
    else:
        # Stage 2: LLM-based extraction for ambiguous / complex queries
        enhanced_query = _enhance_prompt(last_human)
        prompt   = INTENT_EXTRACTION_PROMPT.format(query=enhanced_query)
        response = _call_llm(prompt)
        parsed_context = _extract_json(response)
        query_type = parsed_context.get("query_type", "GENERAL") or "GENERAL"
        parsed_context["query_type"] = query_type
        # Also apply natural language date parsing on top of LLM output
        if not parsed_context.get("datetime"):
            dt_parsed = _parse_natural_date(last_human)
            if dt_parsed:
                parsed_context["datetime"] = dt_parsed

    # Stage 3: Auto-resolve venue → station for EVENT queries
    # This prevents the agent from asking for a station when a venue is clearly named
    if query_type == "EVENT" and not parsed_context.get("station"):
        venue_station = _resolve_venue_station(last_human)
        if venue_station:
            parsed_context["station"] = venue_station
            parsed_context["stations"] = [venue_station]

    # Stage 4: check for missing required fields
    missing = _check_missing_fields(query_type, parsed_context)
    if missing:
        field_descriptions = ", ".join(_FIELD_LABELS.get(f, f) for f in missing)
        clar_prompt = CLARIFICATION_PROMPT.format(
            query=last_human,
            missing_fields=field_descriptions
        )
        clarification_q = _call_llm(clar_prompt)
        if not clarification_q or clarification_q.startswith("[LLM"):
            label = _FIELD_LABELS.get(missing[0], missing[0])
            clarification_q = f"To proceed with this analysis, I need to know {label}. Could you please specify it?"

        return {
            "query_type":             "CLARIFICATION_NEEDED",
            "parsed_context":         parsed_context,
            "clarification_question": clarification_q,
            "clarification_fields":   missing,
        }

    # No datetime default applied — leave None; tools handle it gracefully
    return {
        "query_type":             query_type,
        "parsed_context":         parsed_context,
        "clarification_question": "",
        "clarification_fields":   [],
    }


# -----------------------------------------------------------------------
# Node 2: Fingerprinter
# -----------------------------------------------------------------------
def fingerprinter_node(state: AgentState) -> dict:
    if state.get("query_type") == "CLARIFICATION_NEEDED":
        return {"fingerprint": {}}

    ctx = state.get("parsed_context", {})
    try:
        dt_str = ctx.get("datetime")
        if dt_str:
            try:
                d = DataStore.get()
                wdf = d.get("weather")
                if wdf is not None and not wdf.empty:
                    ts = pd.to_datetime(dt_str, errors="coerce")
                    if not pd.isna(ts):
                        wrow = wdf.set_index("timestamp").asof(ts)
                        if wrow is not None and not (isinstance(wrow, pd.Series) and wrow.empty):
                            ctx = dict(ctx)
                            ctx.setdefault("weather_temp", float(wrow.get("temp", 20)))
                            ctx.setdefault("weather_prcp", float(wrow.get("prcp", 0)))
                            ctx.setdefault("weather_coco", float(wrow.get("coco", 1)))
            except Exception:
                pass
        fp = build_fingerprint(ctx)
    except Exception as e:
        fp = {"error": str(e)}
    return {"fingerprint": fp}


# -----------------------------------------------------------------------
# Node 3: Tool Planner
# -----------------------------------------------------------------------
def tool_planner_node(state: AgentState) -> dict:
    qt = state.get("query_type", "GENERAL")
    if qt in ["CLARIFICATION_NEEDED", "OUT_OF_SCOPE"]:
        return {"tool_results": {"_plan": [], "_skip": True}}
    plan = _TOOL_PLANS.get(qt, _TOOL_PLANS["GENERAL"])
    return {"tool_results": {"_plan": plan}}


# -----------------------------------------------------------------------
# Node 4: Tool Executor — parallel execution, no silent station defaults
# -----------------------------------------------------------------------
def tool_executor_node(state: AgentState) -> dict:
    tr = state.get("tool_results", {})
    qt = state.get("query_type", "GENERAL")
    if tr.get("_skip") or qt in ["CLARIFICATION_NEEDED", "OUT_OF_SCOPE"]:
        return {"tool_results": {"_plan": [], "_skip": True}}

    plan = tr.get("_plan", [])
    ctx  = state.get("parsed_context", {})
    results = {}

    try:
        from mcp_tools.server import (
            get_station_flow, get_flow_baseline, detect_flow_anomaly,
            get_event_context, get_disruption_info, get_weather_at_time,
            compute_rerouting, simulate_crowd_cascade, get_network_centrality,
            compute_stress_scores, find_historical_analogues, analyze_route_preferences,
            get_energy_efficiency, forecast_surge,
        )
    except Exception as e:
        return {"tool_results": {"_error": f"Could not import MCP tools: {e}"}}

    # Resolve args — NO silent defaults for station
    stations  = ctx.get("stations") or []
    station   = ctx.get("station") or (stations[0] if stations else None)
    
    # Try resolving venue from station string or event_name
    try:
        from data_pipeline.event_mapper import VENUE_COORDS, haversine_km
        venue_query = ""
        if station and station.lower() not in DataStore.flows.columns:
            venue_query = station.lower()
        elif not station and ctx.get("event_name"):
            venue_query = str(ctx.get("event_name")).lower()
            
        coords = None
        for key, c in VENUE_COORDS.items():
            if venue_query and key in venue_query:
                coords = c
                break
        
        # Fallback to checking the raw text of the query if no match yet
        if not coords:
            msgs = state.get("messages", [])
            last_human = ""
            for m in reversed(msgs):
                if hasattr(m, "content"):
                    last_human = str(m.content).lower()
                    break
            if last_human:
                for key, c in VENUE_COORDS.items():
                    if key in last_human:
                        coords = c
                        break
                        
        if coords:
                lat, lon = coords
                min_dist = float('inf')
                closest_st = None
                for _, st_row in DataStore.stations.iterrows():
                    st_lat = st_row.get('latitude')
                    st_lon = st_row.get('longitude')
                    st_name = st_row.get('station_name')
                    if pd.notna(st_lat) and pd.notna(st_lon) and pd.notna(st_name):
                        dist = haversine_km(lat, lon, st_lat, st_lon)
                        if dist < min_dist:
                            min_dist = dist
                            closest_st = st_name
                if closest_st:
                    station = closest_st
    except Exception:
        pass

    line      = ctx.get("line") or ctx.get("disrupted_line")
    from_st   = ctx.get("disrupted_from")
    to_st     = ctx.get("disrupted_to")
    dt        = ctx.get("datetime")  # may be None

    # For time-window queries, we need a start datetime
    # Use earliest training date as neutral fallback for non-time-specific analysis
    _FALLBACK_DT = "2026-07-15 08:00:00"
    dt_start  = dt if dt else _FALLBACK_DT
    dt_end    = dt_start
    date_only = "2026-07-15"
    hour      = 8
    is_wknd   = False
    try:
        from datetime import datetime as _dt, timedelta
        dt_obj   = _dt.fromisoformat(str(dt_start).replace(" ", "T"))
        dt_end   = (dt_obj + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
        date_only = dt_obj.strftime("%Y-%m-%d")
        hour     = dt_obj.hour
        is_wknd  = dt_obj.weekday() >= 5
    except Exception:
        pass

    def _run_tool(tool_name: str) -> tuple:
        try:
            if tool_name == "get_station_flow":
                if station:
                    return tool_name, get_station_flow(station, dt_start, dt_end)
                return tool_name, {"note": "Station-level flow query skipped — no station specified in query."}

            elif tool_name == "get_flow_baseline":
                if station:
                    return tool_name, get_flow_baseline(station, hour, is_wknd)
                return tool_name, {"note": "Baseline query skipped — no station specified in query."}

            elif tool_name == "detect_flow_anomaly":
                if station:
                    return tool_name, detect_flow_anomaly(station, dt_start, dt_end)
                return tool_name, {"note": "Anomaly detection requires a station name — none specified."}

            elif tool_name == "get_event_context":
                if station:
                    return tool_name, get_event_context(station, date_only)
                return tool_name, {"note": "Event context query skipped — no station specified."}

            elif tool_name == "get_disruption_info":
                return tool_name, get_disruption_info(dt_start, dt_end, line=line, station=station)

            elif tool_name == "get_weather_at_time":
                return tool_name, get_weather_at_time(dt_start)

            elif tool_name == "compute_rerouting":
                if from_st and to_st:
                    return tool_name, compute_rerouting(from_st, to_st,
                                                        closed_stations=[station] if station else [])
                return tool_name, {"note": "Rerouting skipped — no from/to station segment specified."}

            elif tool_name == "simulate_crowd_cascade":
                closed = [station] if station else []
                return tool_name, simulate_crowd_cascade(
                    closed_stations=closed, disruption_timestamp=dt_start,
                    closed_segment_from=from_st, closed_segment_to=to_st,
                )

            elif tool_name == "get_network_centrality":
                return tool_name, get_network_centrality(top_n=10)

            elif tool_name == "compute_stress_scores":
                st_list = [station] if station else stations
                if st_list:
                    return tool_name, compute_stress_scores(st_list, dt_start)
                return tool_name, {"note": "Stress score requires a station name — none specified."}

            elif tool_name == "find_historical_analogues":
                return tool_name, find_historical_analogues(ctx, top_k=5)

            elif tool_name == "analyze_route_preferences":
                if line:
                    return tool_name, analyze_route_preferences(line, dt_start, dt_end)
                return tool_name, {"note": "Behavioral analysis skipped — no disrupted line specified."}

            elif tool_name == "get_energy_efficiency":
                return tool_name, get_energy_efficiency(line=line)

            elif tool_name == "forecast_surge":
                if station:
                    att = ctx.get("event_attendance") or 1500
                    return tool_name, forecast_surge(
                        station_name=station, event_dt=dt_start,
                        event_attendance=int(att)
                    )
                return tool_name, {"note": "Surge forecast skipped — no station specified."}

            return tool_name, {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return tool_name, {"error": str(e)}

    print(f"[DEBUG] Executing tools. station='{station}', dt='{dt}'")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(max(len(plan), 1), 6)) as executor:
        futures = {executor.submit(_run_tool, t): t for t in plan}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            
    print(f"[DEBUG] Tool results keys: {results.keys()}")
    if 'get_station_flow' in results:
        print(f"[DEBUG] get_station_flow result keys: {results['get_station_flow'].keys() if isinstance(results['get_station_flow'], dict) else results['get_station_flow']}")
    if 'forecast_surge' in results:
        print(f"[DEBUG] forecast_surge result keys: {results['forecast_surge'].keys() if isinstance(results['forecast_surge'], dict) else results['forecast_surge']}")


    return {"tool_results": results}


# -----------------------------------------------------------------------
# Node 5: Synthesizer
# -----------------------------------------------------------------------
def synthesizer_node(state: AgentState) -> dict:
    if state.get("query_type") in ["CLARIFICATION_NEEDED", "OUT_OF_SCOPE"]:
        return {"synthesis": {}, "recommendation": {}}

    tr  = state.get("tool_results", {})
    qt  = state.get("query_type", "GENERAL")
    fp  = state.get("fingerprint", {})

    synthesis = {"query_type": qt, "tool_data": tr}
    rec = {}

    try:
        if qt in ["DISRUPTION", "COMBINED"]:
            cascade_raw     = tr.get("simulate_crowd_cascade", {})
            cascade_results = cascade_raw.get("all_stations", []) if isinstance(cascade_raw, dict) else []
            analogues       = tr.get("find_historical_analogues", {})
            hist_outcomes   = analogues.get("outcomes", []) if isinstance(analogues, dict) else []
            rec = recommend_action(fp, cascade_results, hist_outcomes)
            synthesis["at_risk_count"]  = len([r for r in cascade_results if r.get("at_risk")])
            synthesis["recommendation"] = rec
    except Exception as e:
        rec = {
            "recommended_action": "Monitor situation closely",
            "action_description": f"Analysis incomplete: {e}",
            "confidence": "LOW",
            "evidence_from_history": [],
            "num_historical_matches": 0,
        }

    return {"synthesis": synthesis, "recommendation": rec}


# -----------------------------------------------------------------------
# Node 6: Response Formatter
# -----------------------------------------------------------------------
def response_formatter_node(state: AgentState) -> dict:
    qt   = state.get("query_type", "GENERAL")
    msgs = state.get("messages", [])
    last_human = msgs[-1].content if msgs else ""
    resp_id = str(uuid.uuid4())

    # Clarification — return the question directly, no LLM synthesis needed
    if qt == "CLARIFICATION_NEEDED":
        q = state.get("clarification_question", "Could you please provide more details?")
        return {
            "final_response": q,
            "response_id":    resp_id,
            "messages":       [AIMessage(content=q)],
        }
        
    if qt == "OUT_OF_SCOPE":
        msg = "This query appears to be unrelated to Berlin U-Bahn operations or management. I am an AI assistant designed to help with U-Bahn operational decisions. How can I assist you with Berlin U-Bahn operations today?"
        return {
            "final_response": msg,
            "response_id":    resp_id,
            "messages":       [AIMessage(content=msg)],
        }

    synth = state.get("synthesis", {})
    rec   = state.get("recommendation", {})

    results_str = json.dumps({"synthesis": synth, "recommendation": rec}, indent=2, default=str)
    prompt      = SYNTHESIS_PROMPT.format(results=results_str, query=last_human)
    final_resp  = _call_llm(prompt, system_context=SYSTEM_PROMPT)

    if not final_resp or final_resp.startswith("[LLM"):
        final_resp = _build_fallback_response(qt, synth, rec, last_human)

    return {
        "final_response": final_resp,
        "response_id":    resp_id,
        "messages":       [AIMessage(content=final_resp)],
    }


# -----------------------------------------------------------------------
# Fallback template (used when LLM is unreachable)
# -----------------------------------------------------------------------
def _build_fallback_response(query_type: str, synthesis: dict, rec: dict, query: str) -> str:
    lines = [
        "### U-Bahn Operations Intelligence Report",
        f"**Query type:** {query_type}",
        f"**Query:** {query}",
        "",
    ]
    tr = synthesis.get("tool_data", {})

    if "get_disruption_info" in tr:
        disrupt     = tr["get_disruption_info"]
        disruptions = disrupt.get("disruptions", []) if isinstance(disrupt, dict) else []
        if disruptions:
            lines.append("### Impact Analysis [HISTORICAL]")
            for d in disruptions:
                desc = str(d.get("description", ""))[:100]
                lines.append(f"- **{d.get('when','?')}** | {desc} | Duration: {d.get('duration','?')}")
        else:
            lines.append("**No disruptions found in the specified time window.** (Source: closures.csv [simulated])")
        lines.append("")

    if "simulate_crowd_cascade" in tr:
        cascade = tr["simulate_crowd_cascade"]
        if isinstance(cascade, dict):
            at_risk = cascade.get("at_risk_stations", [])
            if at_risk:
                lines.append(f"### Cascade Overflow Risk [SIMULATED] — {len(at_risk)} stations at risk")
                for r in at_risk[:5]:
                    lines.append(
                        f"- **{r.get('station','?')}**: overflow {r.get('overflow_ratio',1):.2f}x baseline"
                        f" | Extra pax: +{r.get('extra_passengers',0):.0f}"
                    )
            lines.append("")

    if "get_energy_efficiency" in tr:
        eff = tr["get_energy_efficiency"]
        if isinstance(eff, dict):
            ranking = eff.get("efficiency_ranking", [])
            if ranking:
                lines.append("### Energy Efficiency Ranking (worst → best) [OBSERVED]")
                for r in ranking:
                    lines.append(
                        f"- Rank {r.get('efficiency_rank','?')}: **{r.get('line','?')}**"
                        f" — {r.get('mwh_per_1k_pax', 0):.1f} MWh / 1k pax"
                    )
            lines.append("")

    if "get_network_centrality" in tr:
        cent = tr["get_network_centrality"]
        if isinstance(cent, dict):
            ranking = cent.get("ranking", [])
            if ranking:
                lines.append("### Critical Stations by Network Centrality [OBSERVED]")
                for r in ranking[:5]:
                    lines.append(
                        f"- {r.get('rank','?')}. **{r.get('station','?')}**"
                        f" — Betweenness: {r.get('betweenness', 0):.4f}"
                    )
            lines.append("")

    if "detect_flow_anomaly" in tr:
        anom = tr["detect_flow_anomaly"]
        if isinstance(anom, dict):
            anomalies = anom.get("anomalies", [])
            note      = anom.get("note", "")
            if note:
                lines.append(f"**Anomaly Detection Note:** {note}")
            elif anomalies:
                lines.append(f"### Flow Anomalies Detected [OBSERVED] — {len(anomalies)} intervals")
                for a in anomalies[:5]:
                    lines.append(
                        f"- **{a.get('timestamp','?')}**: z-score {a.get('z_score',0):.2f}"
                        f" | Flow: {a.get('flow',0):.0f}"
                    )
            else:
                lines.append("**No significant flow anomalies detected in the specified window.**")
            lines.append("")

    if rec and rec.get("recommended_action"):
        lines.append("### Recommended Action")
        lines.append(f"**{rec.get('recommended_action', 'Monitor situation')}**")
        lines.append(f"{rec.get('action_description', '')}")
        conf    = rec.get("confidence", "LOW")
        matches = rec.get("num_historical_matches", 0)
        lines.append(f"\n**HCADE Confidence:** {conf} ({matches} historical analogues)")
        if conf == "LOW":
            lines.append("⚠️ LOW confidence — treat this recommendation as indicative only.")
        evidence = rec.get("evidence_from_history", [])
        if evidence:
            lines.append("\n**Historical Evidence [HISTORICAL]:**")
            for ev in evidence:
                lines.append(f"- {ev}")
        lines.append("")

    lines.append("### Data Sources")
    lines.append(
        "flows.csv [SIMULATED] | closures.csv [SIMULATED] | "
        "weather_data.csv [REAL] | berlin_events_summer_2026.csv [REAL]"
    )
    return "\n".join(lines)
