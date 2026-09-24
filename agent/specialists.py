"""SPECIALIST agents — one per question category, each a *playbook*: a fixed sequence of MCP tool calls that turns the
supervisor's plan into a compact facts JSON (no LLM inside).

    Supervisor (router)  --plan-->  Specialist (this file)  --facts-->  Writer (one LLM call + guard)
         |                                |
         +-- boundaries / memory (Cognee) +-- MCP tools (dataset, TabPFN engine, analytics)

`SPECIALISTS` is the single source of truth for what the system can do: the executor dispatches through it, the
dashboard's workflow page draws it, and the docs are generated from it. `status`: live = answers with tools and ground
truth; partial = answers but with a stated proxy/limitation; planned = declines honestly (no tool yet).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import executor as ex

Playbook = Callable[..., Awaitable[dict]]


@dataclass(frozen=True)
class Specialist:
    cat: str
    name: str
    title: str
    role: str
    tools: tuple[str, ...]
    datasets: tuple[str, ...]
    status: str
    ml: bool = False
    examples: tuple[str, ...] = ()
    run: Playbook | None = field(default=None, compare=False, repr=False)


async def _timed(mcp, trace: list, tool: str, **kw):
    t = time.time()
    r = await mcp.call(tool, **kw)
    trace.append({"tool": tool, "s": round(time.time() - t, 2)})
    return r


def _hm(hhmm: str, plus_min: int = 0) -> str:
    h, m = map(int, hhmm.split(":"))
    tot = (h * 60 + m + plus_min) % (24 * 60)
    return f"{tot // 60:02d}:{tot % 60:02d}"


# ------------------------------------------------------------------------------------------- A  events
async def pb_events(plan: dict, question: str, mcp, trace: list) -> dict:
    venue, name = plan.get("venue"), plan.get("event")
    if not venue and not name:
        return {"status": "need", "cat": "A", "missing": ["the event or venue (a venue name or an event title from the events data)"]}
    times = plan.get("times") or []
    start_t = times[0] if times else ""
    end_t = times[-1] if len(times) >= 2 else ""
    st = ex.short(plan["stations"][0]) if plan["stations"] else ""
    date = plan["dates"][0] if plan["dates"] else ""
    r = await _timed(mcp, trace, "event_impact", venue=venue or "", event_name=name or "", date=date, station=st, top_n=3, start_time=start_t, end_time=end_t)
    if r.get("status") != "ok":
        return {"status": "need", "cat": "A", "missing": [r.get("note", "no matching event")], "options": list((r.get("venues_with_most_events") or {}))[:5]}
    assumed = list(plan.get("assumed") or [])
    if plan.get("alias"):
        assumed.append(f"'{plan.get('venue')}' in the data is the arena you called by its earlier name (same venue, assumed)")
    if plan.get("rel_day") and not date:
        assumed.append(f"'{plan['rel_day']}' has no calendar date in the data: the numbers are the pattern of past events at this venue")
    ev = r.get("event")
    end = end_t or (ev or {}).get("end") or ""
    end_assumed = False
    if not end and start_t and r.get("typical_duration_min"):
        end, end_assumed = _hm(start_t, int(r["typical_duration_min"])), True
    facts = {"status": "ok", "cat": "A", "venue": r["venue"], "n_ev": r["n_events"], "ev": ev, "top": r["top_stations"], "this": r.get("this_event"),
             "peak_min": r.get("surge_peak_min_after_end"), "typ_att": r.get("attendance_median"), "dur": r.get("typical_duration_min"),
             "st": r.get("station"), "conf": r.get("confidence"), "note": r.get("note"), "assumed": assumed,
             "clock": {"start": start_t or (ev or {}).get("start"), "end": end or None, "end_assumed": end_assumed},
             "lim": ["no venue-to-station key in the data (stations inferred from flow uplift)", "no capacity data"]}
    if end and r["top_stations"]:
        facts["act"] = [{"s": x["s"], "from": end, "to": _hm(end, 60), "add": x["excess"], "x": x["ratio"]} for x in r["top_stations"][:3]]
    return facts


# ------------------------------------------------------------------------------------------- P  pressure ranking
async def pb_pressure(plan: dict, question: str, mcp, trace: list) -> dict:
    n = plan.get("n") or 3
    assumed = list(plan.get("assumed") or [])
    if plan["dates"]:
        date = plan["dates"][0]
    else:
        first, last = await ex.coverage(mcp)
        date = ex._last_full_weekday(last)
        assumed.append(f"no date given: the latest full weekday in the data ({date}) was used")
    r = await _timed(mcp, trace, "rank_pressure", date=date, top_n=n, rain=plan.get("rain"), engine=plan.get("engine"))
    if "error" in r:
        return {"status": "need", "cat": "P", "missing": [r["error"]]}
    assumed += list(r.get("assumptions") or [])
    facts = {"status": "ok", "cat": "P", "date": r["date"], "mode": r["mode"].split(" (")[0], "top": r["top"], "assumed": assumed,
             "proxy": "highest predicted load per 15 min (90th percentile of the TabPFN forecast) at the checked peak slots; p = chance of exceeding the station's own busiest-5% level; NOT a platform capacity",
             "ev": r.get("events_that_day")}
    if r.get("note"):
        facts["note"] = r["note"]
    if r.get("event_data_note") and "innotrans" in question.lower():
        facts["ev_note"] = r["event_data_note"]
    if r.get("observed_top"):
        facts["obs"] = r["observed_top"]
        facts["overlap"] = r.get("observed_overlap")
    return facts


# ------------------------------------------------------------------------------------------- B  anomalies
async def pb_anomalies(plan: dict, question: str, mcp, trace: list) -> dict:
    if not plan["dates"]:
        return {"status": "need", "cat": "B", "missing": ["a date or date range (2026-06-10 to 2026-09-22)"]}
    ds = sorted(plan["dates"])
    cause = "weather" if re.search(r"weather|rain|storm|wind|heat|hot", question, re.I) else ""
    n = plan.get("n") or (1 if re.search(r"\ban example\b|\bone example\b", question, re.I) else 3)
    r = await _timed(mcp, trace, "find_anomalies", start_date=ds[0], end_date=ds[-1], top_n=n, cause=cause)
    if r.get("status") == "oos":
        return {"status": "oos", "cat": "B", "have": ex.HAVE, "note": r["note"]}
    return {"status": "ok", "cat": "B", "range": r["range"], "cause": cause or "unexplained-by-closure", "found": r["found"], "n_over_4z": r["n_anomalies_over_4z"],
            "note": r.get("note"), "lim": ["causes are 'consistent with', not proven; no incident log exists"]}


# ------------------------------------------------------------------------------------------- E  energy
async def pb_energy(plan: dict, question: str, mcp, trace: list) -> dict:
    r = await _timed(mcp, trace, "energy_efficiency")
    rows = r["ranking"]
    worst, best = rows[0], rows[-1]
    med_pps = sorted(x["pax_per_station_day"] for x in rows)[len(rows) // 2]
    return {"status": "ok", "cat": "E", "worst": {"line": worst["line"], "wh": worst["wh_per_pax"], "mwh_day": worst["mwh_day"], "pax_day": worst["pax_day"],
                                                    "pps": worst["pax_per_station_day"], "corr": worst["corr_energy_pax"]},
            "best": {"line": best["line"], "wh": best["wh_per_pax"], "pps": best["pax_per_station_day"]}, "median_pps": med_pps,
            "x_vs_best": round(worst["wh_per_pax"] / best["wh_per_pax"], 1),
            "rank": [[x["line"], x["wh_per_pax"], x["pax_per_station_day"], x["corr_energy_pax"]] for x in rows[:4]],
            "wk": {"energy_pct": worst["weekend_vs_weekday_energy_pct"], "pax_pct": worst["weekend_vs_weekday_pax_pct"]},
            "method": r["method"], "lim": r["limits"]}


# ------------------------------------------------------------------------------------------- F  resilience
async def pb_resilience(plan: dict, question: str, mcp, trace: list) -> dict:
    r = await _timed(mcp, trace, "network_resilience_ranking", top_n=plan.get("n") or 5)
    return {"status": "ok", "cat": "F", "top": [{"s": x["s"], "lines": x["lines"], "pax": x["pax_day_affected"], "own": x["own_pax_day"], "cut": x["cut_off"],
                                                   "frag": x["fragments"], "nbr": x["neighbours"][:3]} for x in r["top"]],
            "method": r["method"], "lim": r["limits"]}


# ------------------------------------------------------------------------------------------- G  correlation
async def pb_correlation(plan: dict, question: str, mcp, trace: list) -> dict:
    r = await _timed(mcp, trace, "correlated_stations", top_n=plan.get("n") or 4, min_hops=3)
    return {"status": "ok", "cat": "G", "pairs": [{"a": x["a"], "b": x["b"], "r": x["r"], "hops": x["hops"], "lag": x["lag_h"], "line": x["shared_lines"]} for x in r["pairs"]],
            "noise_r": r["noise_level_r"], "typical_r": r["median_abs_r_all_pairs"], "hours": r["n_hours"], "lim": r["limits"]}


# ------------------------------------------------------------------------------------------- H  reroute behaviour
async def pb_reroute(plan: dict, question: str, mcp, trace: list) -> dict:
    r = await _timed(mcp, trace, "reroute_behaviour")
    if r.get("status") != "ok":
        return {"status": "unsupported", "cat": "H", "have": ex.HAVE, "reason": r.get("note", "no evidence available")}
    g = r["groups"]
    pick = lambda k: (g.get(k) or {}).get("observed_over_expected")
    return {"status": "ok", "cat": "H", "obs_over_exp": {"closed_station": pick("station/closed_unserved"), "hop1": pick("line_section/hop1"), "hop2": pick("line_section/hop2"),
                                                        "endpoints": pick("line_section/endpoint_or_interchange")},
            "noise": r["noise_share_above_q90"], "reading": r["reading"], "lim": r["limits"]}


# ------------------------------------------------------------------------------------------- registry
def _reg(*items: Specialist) -> dict[str, Specialist]:
    return {s.cat: s for s in items}


async def _run_c(plan, question, mcp, trace):
    return await ex.playbook_c(plan, mcp, trace)


async def _run_d(plan, question, mcp, trace):
    return await ex.playbook_d(plan, mcp, trace, question)


SPECIALISTS: dict[str, Specialist] = _reg(
    Specialist("C", "disruption", "Disruption response", "closure/suspension: reason, duration, rail detours, stations under pressure, staff",
               ("resolve_closure", "apply_closure", "alternate_paths", "scenario_flow"), ("closures", "connections", "stations", "flows", "weather", "events"), "live", True,
               ("Line U6 is suspended between Hallesches Tor and Kaiserin-Augusta-Str. — why, how long, reroute, who is overloaded?",
                "U8 is suspended between Hermannplatz and Neukölln. Where will passengers reroute?"), _run_c),
    Specialist("D", "station", "Station profile", "commute peak, weekday/weekend rhythm, network-mean comparison, flow prediction at a time",
               ("resolve_station", "station_profile", "predict_expected_flow", "predict_overcrowding_risk"), ("flows",), "live", True,
               ("At what time does the commute flow peak at Rudow? Does it exceed the network mean?",), _run_d),
    Specialist("A", "events", "Event impact", "which stations feel an event/venue, when, by how much, and where to staff after it ends",
               ("event_impact",), ("events", "flows", "stations"), "partial", False,
               ("There is a sold-out concert at Mercedes-Benz Arena tonight at 21:00 — flow at Hermannplatz, what to do at 23:15?",
                "Guns N' Roses at the Uber Arena on June 23rd — flow at neighbouring stations and measures?"), pb_events),
    Specialist("P", "pressure", "Pressure ranking", "the N stations under the highest load on a day (weather / event scenario), TabPFN forecast",
               ("rank_pressure",), ("flows", "weather", "events"), "partial", True,
               ("During InnoTrans we expect major flow and bad weather — the 3 stations most likely to exceed safe platform capacity on day one?",), pb_pressure),
    Specialist("B", "anomaly", "Anomalies and root cause", "flow anomalies in a date range and which explanation (closure / event / weather / none) fits",
               ("find_anomalies",), ("flows", "weather", "events", "closures"), "partial", False,
               ("A passenger flow peak caused by bad weather in the week of July 20-26: time and station?",
                "Three anomalies on June 24th not explained by closures, with likely causes?"), pb_anomalies),
    Specialist("E", "energy", "Energy efficiency", "Wh per passenger by line, what explains the ranking",
               ("energy_efficiency",), ("energy", "flows", "stations"), "live", False,
               ("Which line has the worst energy-per-passenger ratio and why?",), pb_energy),
    Specialist("F", "resilience", "Network resilience", "stations whose closure fragments the network most, passengers affected per day",
               ("network_resilience_ranking",), ("connections", "flows", "stations"), "live", False,
               ("Rank the five stations whose closure would fragment the network the most.",), pb_resilience),
    Specialist("G", "correlation", "Latent correlations", "demand-coupled station pairs with no direct connection",
               ("correlated_stations",), ("flows", "connections", "stations"), "partial", False,
               ("Which stations depend strongly on another station without a direct connection?",), pb_correlation),
    Specialist("H", "reroute", "Reroute behaviour", "what the recorded closures show about passenger rerouting",
               ("reroute_behaviour",), ("closures", "flows"), "partial", False,
               ("Which alternative routes do passengers actually prefer during disruptions?",), pb_reroute),
    Specialist("X", "strategy", "Investment / InnoTrans routing", "cross-cutting recommendation questions (needs several specialists combined)",
               (), ("flows", "connections", "energy", "closures"), "planned", False,
               ("If you could invest in one improvement anywhere in the network, what should it be?",), None),
)


async def dispatch(plan: dict, question: str, mcp, trace: list) -> dict | None:
    """Run the specialist for plan['cat']; None if the category has no playbook (the executor then declines honestly)."""
    sp = SPECIALISTS.get(plan["cat"])
    if sp is None or sp.run is None:
        return None
    return await sp.run(plan, question, mcp, trace)
