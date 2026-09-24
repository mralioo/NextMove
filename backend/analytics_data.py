"""Analytics for the desktop pages (network, flow analytics, cascade simulator, energy, closures) — computed from the same merged data the agent reads (training + test split).

Every number comes from the CSV files through `ops_data.world()`; nothing is typed in. The cascade simulator uses the same assumption as the agent's closure scenario (a stated share
of the closed station's typical flow is diverted to open stations within two hops, weights 1/hops) and says so in its response.
"""
from __future__ import annotations

import re
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd

import ops_data
from ops_data import LINE_COLORS, _short


def _w() -> dict:
    return ops_data.world()


@lru_cache(maxsize=1)
def _daily_mean() -> pd.Series:
    """Mean passengers per day and station over the whole window."""
    wide = _w()["wide"]
    days = max(1, wide.index.normalize().nunique())
    return wide.sum(skipna=True) / days


@lru_cache(maxsize=1)
def _betweenness() -> dict:
    return nx.betweenness_centrality(_w()["net"].g)


def status() -> dict:
    w = _w()
    idx = w["wide"].index
    return {"status": "ok", "stations_count": int(w["wide"].shape[1]), "flows_rows": int(len(idx)), "data_window": {"start": str(idx[0]), "end": str(idx[-1])},
            "lines": sorted({x for xs in w["net"].lines_of.values() for x in xs})}


def stations() -> list[dict]:
    net, pos, dm, bt = _w()["net"], _w()["pos"], _daily_mean(), _betweenness()
    out = []
    for n in net.g.nodes:
        if n not in pos.index:
            continue
        out.append({"name": n, "short_name": _short(n), "lat": round(float(pos.loc[n, "latitude"]), 6), "lon": round(float(pos.loc[n, "longitude"]), 6), "lines": net.lines_of.get(n, []),
                    "daily_flow": round(float(dm.get(n, 0.0)), 1), "betweenness": round(float(bt.get(n, 0.0)), 5), "degree": int(net.g.degree(n))})
    return out


def edges() -> list[dict]:
    net, pos = _w()["net"], _w()["pos"]
    out = []
    for a, b in net.g.edges:
        if a not in pos.index or b not in pos.index:
            continue
        for ln in sorted(net.edge_lines(a, b)) or [""]:
            out.append({"line": ln, "from": a, "to": b, "from_lat": float(pos.loc[a, "latitude"]), "from_lon": float(pos.loc[a, "longitude"]), "to_lat": float(pos.loc[b, "latitude"]), "to_lon": float(pos.loc[b, "longitude"])})
    return out


def flows_daily() -> dict:
    wide = _w()["wide"]
    day = wide.index.normalize()
    d = wide.sum(axis=1, skipna=True).groupby(day).sum()
    n = pd.Series(1, index=wide.index).groupby(day).sum()
    d = d[n >= 0.7 * n.max()]          # a partial day (the data window ends just after midnight) would show as a false drop to zero
    r = d.rolling(7, min_periods=1).mean()
    return {"dates": [str(x.date()) for x in d.index], "values": [round(float(v)) for v in d.values], "rolling_mean": [round(float(v)) for v in r.values],
            "unit": "passengers per day, all 167 stations (interchange stations counted once)"}


def hourly_profile() -> list[dict]:
    """Network passengers per hour of the day, mean over weekdays (Mon-Fri) and weekends — the commute curves."""
    wide = _w()["wide"]
    tot = wide.sum(axis=1, skipna=True)
    df = pd.DataFrame({"t": tot, "hour": wide.index.hour, "day": wide.index.normalize(), "wk": wide.index.dayofweek < 5})
    hourly = df.groupby(["wk", "day", "hour"]).t.sum().groupby(["wk", "hour"]).mean()
    return [{"hour": f"{h:02d}:00", "weekday": round(float(hourly.get((True, h), 0.0))), "weekend": round(float(hourly.get((False, h), 0.0)))} for h in range(24)]


def heatmap(lines: list[str] | None) -> dict:
    """Mean passengers per hour of the day per station: ALL stations of one selected line, else the 20 busiest stations of the selected lines."""
    w, net = _w(), _w()["net"]
    wide = w["wide"]
    sel = [l for l in (lines or []) if l in LINE_COLORS] or sorted(LINE_COLORS)
    cols = [c for c in wide.columns if set(net.lines_of.get(c, [])) & set(sel)]
    if len(sel) > 1:
        cols = sorted(cols, key=lambda c: -_daily_mean().get(c, 0))[:20]
    else:
        cols = sorted(cols, key=lambda c: -_daily_mean().get(c, 0))
    hourly = wide[cols].groupby(wide.index.hour).sum(min_count=1) / max(1, wide.index.normalize().nunique())
    vals = [[round(float(hourly.loc[h, c]) if h in hourly.index else 0.0, 1) for h in range(24)] for c in cols]
    return {"stations": [_short(c) for c in cols], "station_ids": cols, "values": vals, "max_value": max((max(r) for r in vals), default=0), "count": len(cols), "lines": sel,
            "unit": "passengers per hour (mean over all days in the window)"}


def station_flow(name: str) -> dict:
    net, wide = _w()["net"], _w()["wide"]
    s = net.resolve_one(name)[0] or (net.resolve(name, 1) or [None])[0]
    if s is None or s not in wide.columns:
        return {"error": f"unknown station '{name}'"}
    x = wide[s]
    hour = pd.DataFrame({"v": x, "h": wide.index.hour, "d": wide.index.normalize(), "wk": wide.index.dayofweek < 5}).groupby(["wk", "d", "h"]).v.sum().groupby(["wk", "h"]).mean()
    prof = [{"hour": f"{h:02d}:00", "weekday": round(float(hour.get((True, h), 0.0)), 1), "weekend": round(float(hour.get((False, h), 0.0)), 1)} for h in range(24)]
    daily = x.groupby(wide.index.normalize()).sum(min_count=1)
    return {"station": s, "short_name": _short(s), "lines": net.lines_of.get(s, []), "daily_mean": round(float(daily.mean()), 1), "profile": prof,
            "peak_weekday_hour": max(prof, key=lambda r: r["weekday"])["hour"], "peak_weekend_hour": max(prof, key=lambda r: r["weekend"])["hour"],
            "daily": {"dates": [str(d.date()) for d in daily.index], "values": [None if pd.isna(v) else round(float(v)) for v in daily.values]}}


def closures() -> list[dict]:
    cl = _w()["closures"].reset_index(drop=True)
    out = []
    for i, r in cl.iterrows():
        out.append({"id": int(i), "when": r["when"].isoformat(), "end": r["end"].isoformat(), "duration_hours": round(float(r["duration_hours"]), 2), "description": r["description"],
                    "closure_type": "line_suspension" if r["closure_type"] == "Line suspension" else ("station_closure" if r["closure_type"] == "Station closure" else "other"),
                    "line": r["affected_line"] if isinstance(r["affected_line"], str) else None, "segment": r["affected_segment"], "reason": r["reason"]})
    return out


def centrality(n: int = 15) -> list[dict]:
    dm, bt = _daily_mean(), _betweenness()
    rows = [{"station": s, "short_name": _short(s), "daily_flow": round(float(dm.get(s, 0)), 1), "betweenness": round(float(bt.get(s, 0)), 5)} for s in _w()["wide"].columns]
    return sorted(rows, key=lambda r: -r["daily_flow"])[:n]


def energy() -> dict:
    """Energy per passenger by line (same method as the agent's energy_efficiency tool): Wh per passenger = daily MWh × 1e6 / daily passengers at the stations serving the line."""
    D, folder, w = _w()["D"], _w()["folder"], _w()
    en = D.load_energy(folder).set_index("timestamp")
    wide, net = w["wide"], w["net"]
    rows = []
    for line in en.columns:
        cols = [c for c in wide.columns if line in net.lines_of.get(c, [])]
        if not cols:
            continue
        p = wide[cols].sum(axis=1, skipna=True).resample("D").sum()
        e = en[line]
        idx = e.index.intersection(p.index)
        e, p = e.loc[idx], p.loc[idx]
        corr = float(np.corrcoef(e, p)[0, 1]) if e.std() > 0 and p.std() > 0 else None
        wk = e.index.dayofweek >= 5
        rows.append({"line": line, "wh_per_pax": round(float(e.sum() * 1e6 / p.sum()), 1), "mwh_day": round(float(e.mean()), 1), "pax_day": round(float(p.mean())), "n_stations": len(cols),
                     "pax_per_station_day": round(float(p.mean() / len(cols))), "corr_energy_pax": None if corr is None else round(corr, 2),
                     "weekend_vs_weekday_energy_pct": round(float((e[wk].mean() / e[~wk].mean() - 1) * 100), 1) if wk.any() and (~wk).any() else None,
                     "weekend_vs_weekday_pax_pct": round(float((p[wk].mean() / p[~wk].mean() - 1) * 100), 1) if wk.any() and (~wk).any() else None})
    rows.sort(key=lambda r: -r["wh_per_pax"])
    med = float(np.median([r["pax_per_station_day"] for r in rows])) if rows else 0
    for i, r in enumerate(rows, 1):
        r["efficiency_rank"] = i
        r["explanation"] = (f"{r['pax_per_station_day']:,} passengers per station and day (median line: {med:,.0f}); daily energy follows ridership with correlation {r['corr_energy_pax']}"
                            f"; at weekends energy changes {r['weekend_vs_weekday_energy_pct']} % while passengers change {r['weekend_vs_weekday_pax_pct']} %.")
    return {"ranking": rows, "unit": "Wh per passenger (lower is better)", "method": "daily MWh per line × 1e6 / daily passengers at the stations serving the line (interchange stations count for every line they serve: a proxy)",
            "limits": ["no rolling-stock, timetable or route-length data: explanations are limited to ridership and load-following evidence"], "median_pax_per_station_day": med}


def cascade(closed: list[str], timestamp: str, share: float = 0.5, hops: int = 2) -> dict:
    """What the closed stations' passengers do: `share` of the typical flow of each closed station at that weekday / 15-minute slot is diverted to the open stations within `hops`
    hops of it, weights 1/hops (normalised). overflow_ratio = (typical + diverted) / typical at the receiving station. An ASSUMPTION scenario, not a measurement."""
    w = _w()
    net, base = w["net"], w["baseline"]
    t = pd.Timestamp(timestamp)
    key = (t.dayofweek, t.hour * 4 + t.minute // 15)
    if key not in base.index:
        return {"error": f"no typical value for {timestamp}"}
    typ = base.loc[key]
    names, unknown = [], []
    for c in closed:
        r = net.resolve_one(c)[0]
        (names if r else unknown).append(r or c)
    if not names:
        return {"error": f"no known station in {closed}", "unknown": unknown}
    extra: dict[str, float] = {}
    moved = {}
    for s in names:
        flow = float(typ.get(s, 0.0)) * share
        moved[s] = round(flow, 1)
        dist = nx.single_source_shortest_path_length(net.g, s, cutoff=hops)
        cands = {n: 1.0 / d for n, d in dist.items() if d > 0 and n not in names}
        tot = sum(cands.values())
        for n, wgt in cands.items():
            extra[n] = extra.get(n, 0.0) + flow * wgt / tot
    rows = []
    for s in typ.index:
        b = float(typ[s])
        e = extra.get(s, 0.0)
        ratio = (b + e) / max(b, 10.0)
        rows.append({"station": s, "short_name": _short(s), "is_closed": s in names, "typical": round(b, 1), "extra_passengers": round(e, 1), "overflow_ratio": round(ratio, 2) if s not in names else 0.0,
                     "at_risk": bool(s not in names and ratio >= 1.3)})
    rows = [r for r in rows if r["is_closed"] or r["extra_passengers"] > 0]
    rows.sort(key=lambda r: (-r["extra_passengers"]))
    return {"closed": names, "unknown": unknown, "timestamp": str(t), "share": share, "hops": hops, "moved_passengers_per_15min": moved, "all_stations": rows,
            "max_overflow_ratio": max((r["overflow_ratio"] for r in rows if not r["is_closed"]), default=1.0),
            "method": f"{int(share * 100)} % of the typical flow of each closed station (same weekday and 15-minute slot) is diverted to open stations within {hops} hops, weights 1/hops; overflow = (typical + diverted) / typical",
            "assumption": "diversion share and destinations are assumed (the agent's closure scenario uses 25 / 50 / 75 %); this is not a measurement and no capacity data exists"}
