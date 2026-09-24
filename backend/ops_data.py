"""Operator desktop data: city topology, line colours and a time-scrubbed snapshot of the network (flows vs typical, active closures, events, weather).

There is no live feed: the desktop replays the recorded data (training + test split) with a time cursor; every number is read from the same merged files the agent uses.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "dashboard"), str(REPO / "ml")]

LINE_COLORS = {"U1": "#52822f", "U2": "#da421e", "U3": "#16683d", "U4": "#f0d722", "U5": "#7e5330", "U6": "#8c6dab", "U7": "#528dc8", "U8": "#224f86", "U9": "#f3791d"}


def _short(name: str) -> str:
    n = name.replace(" (Berlin)", "")
    for p in ("S+U ", "U ", "S "):
        if n.startswith(p):
            return n[len(p):]
    return n


@lru_cache(maxsize=1)
def world() -> dict:
    """Everything loaded once: frames, network graph, the same-weekday-same-slot baseline."""
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    from disruption import Network                      # noqa: E402  (ml/disruption.py)
    from utils import data_loader as D                    # noqa: E402

    folder = str(next(iter(D.discover_dataset_dirs(os.environ.get("DATA_DIR", str(REPO / "data"))).values())))
    stations = D.load_stations(folder)
    flows = D.load_flows(folder)
    cols = [c for c in D.station_cols(flows) if c in set(stations["station_name"])]
    wide = flows.set_index("timestamp")[cols].astype(float)
    net = Network.from_frames(stations, D.load_connections(folder), list(cols))
    slot = wide.index.hour * 4 + wide.index.minute // 15
    baseline = wide.groupby([wide.index.dayofweek, slot]).mean()
    pos = stations.groupby("station_name")[["longitude", "latitude"]].mean()
    return {"D": D, "folder": folder, "stations": stations, "wide": wide, "net": net, "baseline": baseline, "pos": pos, "closures": D.load_closures(folder), "events": D.load_events(folder),
            "weather": D.load_weather(folder).set_index("timestamp")}


def topology() -> dict:
    w = world()
    net, pos = w["net"], w["pos"]
    names = [n for n in net.g.nodes if n in pos.index]
    st = [{"id": n, "name": _short(n), "lon": round(float(pos.loc[n, "longitude"]), 5), "lat": round(float(pos.loc[n, "latitude"]), 5), "lines": net.lines_of.get(n, []), "has_flow": n in w["wide"].columns}
          for n in names]
    edges = [{"a": a, "b": b, "lines": sorted(net.edge_lines(a, b))} for a, b in net.g.edges if a in pos.index and b in pos.index]
    lines = [{"line": ln, "color": LINE_COLORS.get(ln, "#888"), "n_stations": sum(1 for n in names if ln in net.lines_of.get(n, []))} for ln in sorted({x for n in names for x in net.lines_of.get(n, [])})]
    lon, lat = [s["lon"] for s in st], [s["lat"] for s in st]
    return {"stations": st, "edges": edges, "lines": lines, "bbox": {"min_lon": min(lon), "max_lon": max(lon), "min_lat": min(lat), "max_lat": max(lat)}}


def timeline() -> dict:
    w = world()
    idx = w["wide"].index
    cl = w["closures"]
    default = (cl["when"].max() + pd.Timedelta(minutes=15)).floor("15min") if len(cl) else idx[-1]
    return {"start": idx[0].isoformat(), "end": idx[-1].isoformat(), "step_min": 15, "default_at": default.isoformat(),
            "closures": [{"id": int(i), "when": r["when"].isoformat(), "end": r["end"].isoformat(), "label": r["description"]} for i, r in cl.reset_index(drop=True).iterrows()]}


def _nearest(ts: pd.Timestamp) -> pd.Timestamp:
    idx = world()["wide"].index
    pos = idx.searchsorted(ts, side="right") - 1
    return idx[max(0, min(pos, len(idx) - 1))]


def snapshot(at: str) -> dict:
    w = world()
    wide, base, net = w["wide"], w["baseline"], w["net"]
    t = _nearest(pd.Timestamp(at).tz_localize(None) if pd.Timestamp(at).tzinfo else pd.Timestamp(at))
    row = wide.loc[t]
    key = (t.dayofweek, t.hour * 4 + t.minute // 15)
    b = base.loc[key] if key in base.index else row * 0 + 1
    st = {n: {"v": round(float(row[n]), 1), "base": round(float(b[n]), 1), "ratio": round(float(row[n] / b[n]), 2) if b[n] > 5 else None} for n in wide.columns}
    lines = []
    for ln in sorted({x for xs in net.lines_of.values() for x in xs}):
        mem = [n for n in wide.columns if ln in net.lines_of.get(n, [])]
        if mem:
            v, bv = float(row[mem].sum()), float(b[mem].sum())
            lines.append({"line": ln, "color": LINE_COLORS.get(ln, "#888"), "load": round(v), "typical": round(bv), "ratio": round(v / bv, 2) if bv else None, "n_stations": len(mem)})
    tot, btot = float(row.sum()), float(b.sum())
    top = sorted(({"station": _short(n), "id": n, "v": s["v"], "ratio": s["ratio"], "lines": net.lines_of.get(n, [])} for n, s in st.items()), key=lambda x: -x["v"])[:8]
    # closures active at t
    closures = []
    cl = w["closures"].reset_index(drop=True)
    from disruption import _closure_to_spec, apply_closure  # noqa: E402
    for i, r in cl[(cl["when"] <= t) & (cl["end"] > t)].iterrows():
        spec = _closure_to_spec(net, int(i), r)
        item = {"id": int(i), "kind": spec["kind"], "line": spec.get("line"), "from": spec.get("from_station"), "to": spec.get("to_station"), "station": spec.get("station"),
                "start": spec["start"], "end": spec["end"], "reason": spec["reason"], "description": spec["description"], "path": [], "blocked_edges": [], "unserved": []}
        try:
            eff = apply_closure(net, spec)
            item.update(path=eff.section_path, blocked_edges=[list(e) for e in eff.blocked_edges], unserved=eff.unserved)
        except Exception as e:
            item["warning"] = str(e)[:160]
        closures.append(item)
    ev = w["events"]
    day = ev[(ev["began_local"].dt.tz_localize(None) - t).abs() < pd.Timedelta(hours=12)].copy()
    events = []
    for _, r in day.iterrows():
        s0 = r["began_local"].tz_localize(None) if r["began_local"].tzinfo else r["began_local"]
        e0 = r["estimated_end_local"]
        e0 = (e0.tz_localize(None) if getattr(e0, "tzinfo", None) else e0) if pd.notna(e0) else s0 + pd.Timedelta(hours=2)
        phase = "ongoing" if s0 <= t <= e0 else ("starting_soon" if 0 < (s0 - t).total_seconds() <= 3 * 3600 else ("ended_recently" if 0 < (t - e0).total_seconds() <= 2 * 3600 else None))
        if phase:
            events.append({"name": r["event_name"], "venue": None if pd.isna(r["venue_name"]) else r["venue_name"], "start": s0.isoformat(), "end": e0.isoformat(), "phase": phase,
                           "attendance": None if pd.isna(r.get("estimated_attendance")) else int(r["estimated_attendance"])})
    events = sorted(events, key=lambda x: -(x["attendance"] or 0))[:6]
    wt = w["weather"]
    wr = wt.iloc[max(0, wt.index.searchsorted(t, side="right") - 1)] if len(wt) else None
    weather = None if wr is None else {"temp": float(wr.get("temp", np.nan)), "prcp": float(wr.get("prcp", 0) or 0), "wspd": float(wr.get("wspd", np.nan)), "rhum": float(wr.get("rhum", np.nan))}
    alerts = []
    for c in closures:
        alerts.append({"level": "critical", "text": f"{c['line'] or 'Station'} closure: {c['description']}"})
    for n, s in sorted(st.items(), key=lambda kv: -(kv[1]["ratio"] or 0))[:40]:
        if s["ratio"] and s["ratio"] >= 2.5 and s["v"] >= 300:
            alerts.append({"level": "warning", "text": f"{_short(n)}: {s['v']:.0f} passengers/15 min, {s['ratio']}× typical", "station": n})
    for e in events:
        if e["phase"] in ("starting_soon", "ended_recently") and (e["attendance"] or 0) >= 1500:
            alerts.append({"level": "info", "text": f"Event {e['phase'].replace('_', ' ')}: {e['name']}" + (f" ({e['attendance']} expected)" if e["attendance"] else "")})
    warn = [a for a in alerts if a["level"] == "warning"][:5]
    alerts = [a for a in alerts if a["level"] != "warning"] + warn
    return {"at": t.isoformat(), "network": {"total": round(tot), "typical": round(btot), "ratio": round(tot / btot, 2) if btot else None}, "stations": st, "lines": lines, "top": top,
            "closures": closures, "events": events, "weather": weather, "alerts": alerts[:12]}


def series(date: str) -> list[dict]:
    w = world()
    wide, base = w["wide"], w["baseline"]
    d = wide[wide.index.strftime("%Y-%m-%d") == date]
    out = []
    for t, r in d.iterrows():
        key = (t.dayofweek, t.hour * 4 + t.minute // 15)
        out.append({"t": t.isoformat(), "total": round(float(r.sum())), "typical": round(float(base.loc[key].sum())) if key in base.index else None})
    return out
