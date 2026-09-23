"""Stage 2 of the fast pipeline: EXECUTOR — no LLM.

A *playbook* per category turns the router's plan into MCP tool calls (independent calls run in
parallel) and compresses the raw tool output into a small "facts" JSON: short keys, rounded
numbers, station names without the "U "/"(Berlin)" noise, assumptions as codes. That JSON — not
prose, not the raw 9k-token tool dumps — is what the writer sees, which is what makes the last LLM
call short and fast.

facts["status"]:  "ok" | "need" (missing input, ask the operator) | "unsupported" | "oos" | "follow" | "error" (a tool failed)
"""
from __future__ import annotations

import asyncio
import difflib
import re
import time

# What the system can do today, so the writer can offer alternatives instead of a bare refusal.
HAVE = [
    "C: closure reason/duration, rail detours, estimated pressure per station (for closures in the data or a what-if)",
    "D: a station's weekday/weekend peak hour vs the network mean, and flow/overcrowding prediction at a given time",
]
NOT_YET = {
    "A": "event impact", "B": "anomaly root-cause", "E": "energy efficiency", "F": "network resilience ranking",
    "G": "latent correlations", "H": "actual reroute behaviour", "X": "investment / InnoTrans routing",
}
H_FINDING = ("In the 26 recorded closures, neighbouring stations (1-2 hops), section endpoints and interchanges "
             "stayed at normal levels (about 10% of readings above their 90th-percentile bound, the noise rate); "
             "only closed stations dropped to 0. So no rerouting behaviour is measurable in this data.")


_COVERAGE: dict = {}


async def coverage(mcp) -> tuple[str, str]:
    """(first, last) date the loaded dataset covers — read from the data, so it extends by itself when the
    Sept 22-30 evaluation set is added."""
    if not _COVERAGE:
        d = await mcp.call("describe_dataset")
        _COVERAGE.update(start=d["coverage_start"][:10], end=d["coverage_end"][:10])
    return _COVERAGE["start"], _COVERAGE["end"]


def short(name: str | None) -> str | None:
    if not name:
        return name
    n = re.sub(r"\s*\(Berlin\)\s*$", "", name)
    return re.sub(r"^(S\+U|U|S)\s+", "", n).strip()


def _pct(p: float | None) -> int | None:
    return None if p is None else int(round(p * 100))


def _hhmm(ts: str | None) -> str | None:
    return ts[11:16] if ts and len(ts) >= 16 else ts


# ------------------------------------------------------------------------------------ Category C
def _score(c: dict, plan: dict) -> float:
    sts = set(plan["stations"])
    s = 2.0 * sum(x in sts for x in (c.get("from_station"), c.get("to_station"), c.get("station")))
    s += 1.0 if c.get("line") and c["line"] in plan["lines"] else 0.0
    if plan["dates"] and (c.get("start") or "")[:10] in plan["dates"]:
        s += 1.5
    return s


async def _find_closure(plan: dict, mcp) -> tuple[dict | None, list[dict]]:
    queries = []
    if plan["dates"]:
        queries.append(plan["dates"][0])
    if plan["lines"]:
        queries.append(plan["lines"][0])
    if plan["stations"]:
        queries.append(short(plan["stations"][0]))
    seen: dict[int, dict] = {}
    for q in queries:
        for c in await mcp.call("resolve_closure", query=q, max_results=8) or []:
            if isinstance(c, dict) and c.get("closure_id") is not None:
                seen[c["closure_id"]] = c
        if seen:
            break
    cands = list(seen.values())
    if not cands:
        return None, []
    ranked = sorted(cands, key=lambda c: -_score(c, plan))
    best = ranked[0]
    need = 2.0 if plan["stations"] else (1.0 if plan["lines"] or plan["dates"] else 0.0)
    if _score(best, plan) >= need and (len(ranked) == 1 or _score(best, plan) > _score(ranked[1], plan)):
        return best, cands
    return None, cands


def _compact_c(closure: dict, ap: dict, alt: dict, sc: dict, assumed: dict | None) -> dict:
    cl = ap.get("closure", closure)
    cut = ap.get("groups_cut_off_from_rail_network", [])
    if isinstance(cut, dict):
        cut = next(iter(cut.values()), [])
    paths = []
    for a in (alt.get("alternates") or []):
        for p in (a.get("paths") or [])[:2]:
            paths.append({"stops": p["hops"], "extra_stops": p["extra_hops_vs_original"],
                          "via": [short(x) for x in p["transfer_stations"]] or "same corridor"})
    facts = {
        "status": "ok", "cat": "C",
        "cl": {"line": cl.get("line"), "kind": cl.get("kind"), "a": short(cl.get("from_station")),
               "b": short(cl.get("to_station")), "st": short(cl.get("station")),
               "from": (cl.get("start") or "")[:16], "to": (cl.get("end") or "")[:16],
               "h": cl.get("duration_hours"), "why": cl.get("reason"), "src": cl.get("source")},
        "cut": {"unserved": [short(x) for x in ap.get("unserved_stations", [])],
                "partial": [short(x) for x in ap.get("still_served_by_other_line", [])],
                # For a station closure these groups exist ONLY IF trains do not run through the closed
                # station (the tool assumes they do), so they get a different key the writer must qualify.
                ("isolated_only_if_no_through_trains" if cl.get("kind") == "station" else "isolated_groups"):
                    [len(g) for g in cut]},
        "alt": {"rail": paths[:3],
                "bus": [[short(l["from"]), short(l["to"]), l["straight_line_km"]]
                        for l in (alt.get("surface_link_candidates") or [])[:2]]},
        "as": ["A1", "A2", "A3", "A4", "A5"],
    }
    if cl.get("kind") == "station":
        facts["note"] = "trains assumed to run through without stopping; riders of the closed station use the nearest open stations"
    if assumed:
        facts["assumed"] = assumed
    if "error" in sc:
        facts["scenario_error"] = sc["error"]
        return facts
    facts["press"] = [{
        "s": short(r["station_name"]), "at": _hhmm(r.get("peak_slot")), "base": round(r["baseline_expected"]),
        "add": round(r["added_load"]), "tot": round(r["expected_total"]), "p95": round(r["own_p95_reference"]),
        "p": _pct(r["prob_exceed_own_p95"]), "p0": _pct(r["prob_exceed_without_closure"]),
        "lo": _pct(r.get("prob_exceed_own_p95_low_assumption")), "hi": _pct(r.get("prob_exceed_own_p95_high_assumption")),
    } for r in sc.get("ranked_pressure_stations", [])[:5]]
    facts["disp"] = {short(k): {"avg": round(v["avg_baseline_per_15min"]), "peak": round(v["peak_baseline_per_15min"]),
                                "tot": round(v["total_displaced_over_window"])}
                     for k, v in list(sc.get("displaced_unserved_stations", {}).items())[:4]}
    chk = sc.get("historical_reality_check", {}).get("rows", [])
    if chk:
        facts["obs"] = [{"s": short(r["station_name"]), "seen": round(r["observed_avg_per_15min"]),
                         "normal": round(r["baseline_expected_avg_per_15min"])} for r in chk[:3]]
    return facts


async def playbook_c(plan: dict, mcp, trace: list) -> dict:
    closure, cands = await _find_closure(plan, mcp) if not plan["what_if"] else (None, [])
    assumed = None
    if closure is not None and plan.get("month") and not (closure.get("start") or "").startswith(plan["month"]):
        return {"status": "need", "cat": "C", "missing": [f"a date inside the data (no closure of these stations in {plan['month']}; "
                                                          "data covers 2026-06-10 to 2026-09-22)"],
                "note": "the only recorded closure of this section is on " + (closure.get("start") or "")[:10]}
    if closure is not None:
        args = {"closure_id": closure["closure_id"]}
    elif plan["what_if"] or (not cands and plan["stations"]):
        # Hypothetical closure: needs a line + two stations (or one station) and a date.
        sts = plan["stations"]
        if not plan["dates"] or not sts or (len(sts) == 1 and plan["lines"]):
            miss = [m for m, ok in (("date", bool(plan["dates"])), ("stations", bool(sts)),
                                    ("second station", len(sts) != 1 or not plan["lines"])) if not ok]
            return {"status": "need", "cat": "C", "missing": miss or ["closure details"],
                    "note": "closure not found in the data; describe it as line + two stations + date/time"}
        time_s = plan["time"] or "08:00"
        dur = plan["dur_min"] or 60
        if not plan["time"] or not plan["dur_min"]:
            assumed = {"start": f"{plan['dates'][0]} {time_s}", "dur_min": dur}
        args = {"start": f"{plan['dates'][0]} {time_s}", "duration_minutes": dur}
        if len(sts) >= 2:
            args.update(line=plan["lines"][0] if plan["lines"] else None, from_station=sts[0], to_station=sts[1])
            if not args["line"]:
                return {"status": "need", "cat": "C", "missing": ["line"]}
        else:
            args.update(station=sts[0])
    elif cands:
        return {"status": "need", "cat": "C", "missing": ["which closure"],
                "options": [{"id": c["closure_id"], "line": c.get("line"), "a": short(c.get("from_station") or c.get("station")),
                             "b": short(c.get("to_station")), "from": (c.get("start") or "")[:16]} for c in cands[:5]]}
    else:
        return {"status": "need", "cat": "C", "missing": ["line, stations or date of the closure"]}

    async def timed(tool, **kw):
        t = time.time()
        r = await mcp.call(tool, **kw)
        trace.append({"tool": tool, "s": round(time.time() - t, 2)})
        return r

    ap, alt, sc = await asyncio.gather(
        timed("apply_closure", **args), timed("alternate_paths", **args, max_paths=2), timed("scenario_flow", **args, top_n=5))
    if isinstance(ap, dict) and "error" in ap:
        return {"status": "need", "cat": "C", "missing": [ap["error"]]}
    return _compact_c(closure or ap.get("closure", {}), ap, alt, sc, assumed)


# ------------------------------------------------------------------------------------ Category D
async def playbook_d(plan: dict, mcp, trace: list, question: str) -> dict:
    names = list(plan["stations"])[:2]
    for raw in plan["raw"][:2]:
        r = await mcp.call("resolve_station", query=raw, max_results=1)
        cand = r[0].get("station_name") if r else None
        # the tool's fuzzy cutoff is lenient (0.4): only accept a genuinely similar name, never a phantom match
        if cand and (raw.lower() in cand.lower()
                     or difflib.SequenceMatcher(None, raw.lower(), (short(cand) or "").lower()).ratio() >= 0.75):
            names.append(cand)
    if not names:
        return {"status": "need", "cat": "D", "missing": ["station"]}

    async def one(name: str) -> dict:
        t = time.time()
        prof = await mcp.call("station_profile", station_name=name)
        trace.append({"tool": "station_profile", "s": round(time.time() - t, 2)})
        if "error" in prof:
            return {"s": short(name), "error": prof["error"]}
        out = {"s": short(name), "avg_day": round(prof["avg_daily_passengers"]),
               "wk": [prof["weekday_peak_hour"], round(prof["weekday_peak_avg_passengers"])],
               "we": [prof["weekend_peak_hour"], round(prof["weekend_peak_avg_passengers"])],
               "net": round(prof["network_mean_weekday_peak_passengers"]),
               "vs_net_pct": prof["delta_vs_network_mean_pct"], "above": prof["exceeds_network_mean_weekday_peak"]}
        if plan["dates"] and plan["time"]:
            ts = f"{plan['dates'][0]} {plan['time']}:00"
            calls = [mcp.call("predict_expected_flow", station_name=name, timestamp=ts)]
            if re.search(r"overcrowd|crowded|risk", question, re.I):
                calls.append(mcp.call("predict_overcrowding_risk", station_name=name, timestamp=ts))
            res = await asyncio.gather(*calls)
            if "error" not in res[0]:
                pred_v, real_v = round(res[0]["predicted_passengers"]), res[0]["actual_passengers"]
                out["pred"] = {"at": ts[:16], "pred": pred_v, "real": real_v, "diff": real_v - pred_v,
                               "in_sample": res[0]["seen_in_training_sample"]}
            else:
                out["pred_error"] = res[0]["error"]
            if len(res) > 1 and "error" not in res[1]:
                out["risk"] = {"p": _pct(res[1]["overcrowding_probability"]), "real": res[1]["actual_label"]}
        return out

    return {"status": "ok", "cat": "D", "st": list(await asyncio.gather(*(one(n) for n in names))),
            "def": "peak = hour-of-day average across the dataset window; network mean = average of every station's own weekday peak"}


# ------------------------------------------------------------------------------------ dispatcher
async def execute(plan: dict, question: str, last_facts: dict | None, mcp, memory=None) -> tuple[dict, list]:
    """Run the playbook. A tool/server failure never crashes the run: it becomes status "error" so the
    writer can tell the operator plainly (and the trace records what failed). With an episodic `memory`, facts
    for a situation analysed before are served from memory and the tools are skipped."""
    trace: list = []
    try:
        remember = memory is not None and plan.get("cat") in ("C", "D")
        if remember:
            _, cov_end = await coverage(mcp)
            hit = memory.lookup(plan, cov_end)
            if hit:
                return hit, [{"tool": "memory (episodic hit)", "s": 0.0}]
        facts, trace = await _execute(plan, question, last_facts, mcp, trace)
        if remember:
            memory.store(plan, cov_end, facts)
        return facts, trace
    except Exception as e:
        return {"status": "error", "cat": plan.get("cat"), "note": f"{type(e).__name__}: {str(e)[:220]}"}, trace


async def _execute(plan: dict, question: str, last_facts: dict | None, mcp, trace: list) -> tuple[dict, list]:
    cat = plan["cat"]
    if cat == "FOLLOW":
        if not last_facts:
            return {"status": "need", "cat": "FOLLOW", "missing": ["the earlier question this refers to"]}, trace
        return {"status": "follow", "prev": last_facts}, trace
    if cat in ("C", "D") and plan.get("dates"):
        first, last = await coverage(mcp)
        if any(not first <= d <= last for d in plan["dates"]):
            return {"status": "oos", "cat": cat, "have": HAVE,
                    "note": f"the date is outside the data the system holds ({first} to {last}); no forecasting or invented figures"}, trace
    if cat == "C":
        return await playbook_c(plan, mcp, trace), trace
    if cat == "D":
        return await playbook_d(plan, mcp, trace, question), trace
    if cat == "H":
        return {"status": "unsupported", "cat": "H", "have": HAVE, "finding": H_FINDING,
                "reason": "no tool measures reroute behaviour directly; a finding from the closure case study applies"}, trace
    if cat == "OOS":
        hits = plan.get("oos") or []
        asked = f"the question asks for {', '.join(repr(h) for h in hits)}, " if hits else ""
        return {"status": "oos", "cat": "OOS", "have": HAVE,
                "note": asked + "which the dataset and tools cannot provide (or it is off-topic). Mention ONLY what was asked."}, trace
    return {"status": "unsupported", "cat": cat, "topic": NOT_YET.get(cat, "this topic"), "have": HAVE,
            "reason": "the dataset has the data, but the analysis tool for this question type is not connected yet"}, trace
