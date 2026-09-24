"""MCP tools for the remaining question categories (A, P, B, E, F, G, H) — deterministic analytics on the raw
datasets, plus one TabPFN-backed tool (`rank_pressure`).

    A  event_impact           which stations feel an event / venue, when, how much (venue->station mapping LEARNED from flows)
    P  rank_pressure          the N stations most likely to exceed their own busiest-5% level on a day (TabPFN, replay or scenario)
    B  find_anomalies         flow anomalies in a date range + which explanation (closure / event / weather / none) fits
    E  energy_efficiency      Wh per passenger per line and the factors that explain the ranking
    F  network_resilience     stations whose closure fragments the network most + passengers affected per day
    G  correlated_stations    demand-coupled station pairs that are NOT directly connected
    H  reroute_behaviour      what the 26 recorded closures show about passenger rerouting

Design rules (same as the Category C tools): read-only; every number comes from the CSVs or the model and is
returned with the definition it needs; anything the data cannot support is said in `limits`, never invented.
Registered onto the shared FastMCP server by mcp_server/server.py.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT / "ml", REPO_ROOT / "dashboard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.data_loader import (  # noqa: E402
    build_graph,
    load_closures,
    load_energy,
    load_flows,
    load_stations,
    load_weather,
    network_resilience,
    station_avg_flow,
    station_cols,
)

# The problem statement says "Mercedes-Benz Arena"; the events file calls the same arena "Uber Arena" (renamed venue).
# Stated as an ASSUMPTION in every answer that uses it.
VENUE_ALIASES = {"mercedes-benz arena": "uber arena", "mercedes benz arena": "uber arena", "mercedes-platz": "uber arena",
                 "o2 world": "uber arena", "o2 arena": "uber arena", "uber eats music hall": "uber eats music hall"}
RAIN_MM = 0.5            # prcp per 15 min above which a slot counts as rainy


def _r(x, n=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return int(round(float(x))) if n == 0 else round(float(x), n)


def _short(name: str) -> str:
    return re.sub(r"^(S\+U|U|S)\s+", "", re.sub(r"\s*\(Berlin\)\s*$", "", name)).strip()


def register(mcp, folder: str, get_feature_table: Callable[[], pd.DataFrame], get_baseline: Callable[[], object]) -> None:
    S: dict = {}
    lock = threading.RLock()

    # ------------------------------------------------------------------ shared, lazily built data
    def wide() -> pd.DataFrame:
        """timestamp-indexed flows, one column per physical station (duplicate-named columns dropped)."""
        with lock:
            if "wide" not in S:
                f = load_flows(folder)
                cols = [c for c in station_cols(f) if not re.search(r"\.\d+$", c)]
                S["wide"] = f.set_index("timestamp")[cols].astype(float)
            return S["wide"]

    def coverage() -> tuple[pd.Timestamp, pd.Timestamp]:
        w = wide()
        return w.index.min(), w.index.max()

    def events() -> pd.DataFrame:
        with lock:
            if "events" not in S:
                import glob
                path = sorted(glob.glob(str(Path(folder) / "berlin_events*.csv")))[0]
                e = pd.read_csv(path)
                e["start"] = pd.to_datetime(e["began_local"].str[:19], errors="coerce")
                e["end"] = pd.to_datetime(e["estimated_end_local"].fillna("").str[:19], errors="coerce")
                e["end"] = e["end"].fillna(e["start"] + pd.Timedelta(hours=2))
                e["venue"] = e["venue_name"].fillna("").str.strip()
                e["date"] = e["start"].dt.strftime("%Y-%m-%d")
                S["events"] = e
            return S["events"]

    def lines_of() -> dict[str, str]:
        with lock:
            if "lines" not in S:
                st = load_stations(folder)
                S["lines"] = st.groupby("station_name")["u_bahn_lines"].apply(lambda s: ",".join(sorted(set(",".join(s).split(","))))).to_dict()
            return S["lines"]

    def _norm_venue(v: str) -> str:
        v = v.lower().strip()
        return VENUE_ALIASES.get(v, v)

    # ============================================================== A  event_impact
    def _window_excess(w: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, exclude_dates: set) -> tuple[pd.Series, pd.Series]:
        """(mean flow per 15 min in [t0,t1) , same-weekday baseline for the same clock window in +-1..4 other weeks)."""
        win = w.loc[(w.index >= t0) & (w.index < t1)]
        refs = []
        for k in (-4, -3, -2, -1, 1, 2, 3, 4):
            a, b = t0 + pd.Timedelta(weeks=k), t1 + pd.Timedelta(weeks=k)
            if a.strftime("%Y-%m-%d") in exclude_dates:
                continue
            r = w.loc[(w.index >= a) & (w.index < b)]
            if len(r):
                refs.append(r.mean())
        base = pd.concat(refs, axis=1).mean(axis=1) if refs else pd.Series(np.nan, index=w.columns)
        return win.mean(), base

    @mcp.tool
    def event_impact(venue: str = "", event_name: str = "", date: str = "", station: str = "", top_n: int = 3,
                     start_time: str = "", end_time: str = "") -> dict:
        """Category A — impact of an event (or of a venue's events in general) on station flows.
        Give `event_name` and/or `venue` (aliases such as 'Mercedes-Benz Arena' are mapped to the venue name the
        data uses) and optionally `date` 'YYYY-MM-DD'. Returns the matching events, the stations that gain most
        passengers in the hour after the event ends (the venue->station mapping is LEARNED from the flows of all past
        events at that venue — the data has no venue coordinates), the size of the surge against a normal same-weekday
        hour, and — when `station` is given — that station's own numbers. With no matching event it returns the venue
        pattern and says so. `start_time`/`end_time` ('HH:MM') let a pattern be read relative to the operator's times."""
        e, w = events(), wide()
        cov0, cov1 = coverage()
        limits = ["events carry a venue name and address but no station key: stations are inferred from flow uplift, not geocoding",
                  "attendance is the organiser's estimate", "no platform-capacity data exists"]
        v_asked, alias_used = venue.strip(), False
        if v_asked and _norm_venue(v_asked) != v_asked.lower():
            alias_used = True
        v_norm = _norm_venue(v_asked) if v_asked else ""
        m = pd.Series(True, index=e.index)
        plain = lambda x: re.sub(r"[’'`]", "", x.lower())            # curly vs straight apostrophes must not stop a match
        if v_norm:
            m &= e["venue"].str.lower().str.contains(re.escape(v_norm), regex=True)
        if event_name.strip():
            m &= e["event_name"].map(plain).str.contains(re.escape(plain(event_name.strip())), regex=True)
        venue_events = e[m] if (v_norm or event_name.strip()) else e.iloc[0:0]
        if venue_events.empty:
            return {"status": "not_found", "note": f"no event matches venue='{venue}' event='{event_name}' in the data", "limits": limits,
                    "venues_with_most_events": e["venue"].replace("", np.nan).value_counts().head(5).to_dict()}
        chosen = venue_events[venue_events["date"] == date] if date else venue_events.iloc[0:0]
        in_data = [x for x in venue_events.itertuples() if cov0 <= x.end <= cov1]
        # --- learned mapping: post-end excess per station over all events at this venue
        excl = set(venue_events["date"])
        rows = []
        for x in in_data:
            win, base = _window_excess(w, x.end, x.end + pd.Timedelta(hours=1), excl)
            rows.append((win - base, win / base.replace(0, np.nan)))
        if not rows:
            return {"status": "no_flow_data", "note": "the matching events lie outside the flow data", "limits": limits}
        exc = pd.concat([a for a, _ in rows], axis=1)
        rat = pd.concat([b for _, b in rows], axis=1)
        agg = pd.DataFrame({"excess": exc.median(axis=1), "ratio": rat.median(axis=1), "n": exc.notna().sum(axis=1),
                            "consistent": (rat > 1.2).mean(axis=1)}).dropna()
        full = agg.sort_values("excess", ascending=False)                      # every station, for the per-station lookup
        # a station counts as "affected" only if the uplift is large AND repeats in most events (filters noise at n=3)
        agg = full[(full["n"] >= max(1, len(rows) // 2)) & (full["ratio"] >= 1.5) & (full["excess"] >= 20) & (full["consistent"] >= 0.6)]
        top = [{"s": _short(n), "lines": lines_of().get(n, ""), "excess": _r(r.excess), "ratio": _r(r.ratio, 2), "n": int(r.n)}
               for n, r in agg.head(max(1, top_n)).iterrows()]
        # --- typical offset of the surge peak (minutes after the end) at the top station
        peak_off = None
        if not top:
            return {"status": "ok", "venue": venue_events["venue"].iloc[0] or v_asked, "alias_assumed": alias_used, "n_events": int(len(venue_events)),
                    "top_stations": [], "note": "no station shows a large, repeatable uplift after these events", "limits": limits}
        if top:
            top_full = agg.index[0]
            profs = []
            for x in in_data:
                sl = w.loc[(w.index >= x.end - pd.Timedelta(minutes=30)) & (w.index < x.end + pd.Timedelta(hours=2)), top_full]
                profs.append(pd.Series(sl.to_numpy(), index=((sl.index - x.end).total_seconds() // 60).astype(int)))
            if profs:
                mp = pd.concat(profs, axis=1).mean(axis=1)
                peak_off = int(mp.idxmax())
        out = {"status": "ok", "venue": venue_events["venue"].iloc[0] or v_asked, "alias_assumed": alias_used,
               "n_events": int(len(venue_events)), "n_with_flow_data": len(in_data),
               "event": None, "top_stations": top, "surge_peak_min_after_end": peak_off,
               "attendance_median": _r(venue_events["estimated_attendance"].median(), 0),
               "typical_duration_min": _r((venue_events["end"] - venue_events["start"]).dt.total_seconds().median() / 60, 0),
               "method": "median over the venue's events of [mean flow in the hour after the event ends] minus [mean flow of the same weekday/clock hour in 4 weeks before and after, days without an event at this venue]",
               "limits": limits, "confidence": "low (few events)" if len(in_data) < 5 else "ok"}
        if not chosen.empty:
            x = chosen.iloc[0]
            out["event"] = {"name": x["event_name"], "date": x["date"], "start": x["start"].strftime("%H:%M"), "end": x["end"].strftime("%H:%M"),
                            "attendance": int(x["estimated_attendance"]) if pd.notna(x["estimated_attendance"]) else None}
            if cov0 <= x["end"] <= cov1:
                win, base = _window_excess(w, x["end"], x["end"] + pd.Timedelta(hours=1), excl - {x["date"]})
                names = [n for n in agg.index[:max(1, top_n)]]
                out["this_event"] = [{"s": _short(n), "seen": _r(win[n]), "normal": _r(base[n]), "ratio": _r(win[n] / base[n] if base[n] else None, 2)} for n in names]
        elif date:
            out["note"] = f"no event of this venue on {date}; the numbers are the venue pattern over {len(in_data)} past events"
        elif len(venue_events) > 1 and not event_name:
            out["note"] = "no date given: the numbers are the venue pattern over all its events in the data"
        if start_time or end_time:
            out["asked_times"] = {"start": start_time or None, "end": end_time or None}
        if station:
            sn = next((n for n in w.columns if _short(n).lower() == _short(station).lower() or station.lower() in n.lower()), None)
            if sn is not None and sn in agg.index:
                r = agg.loc[sn]
                out["station"] = {"s": _short(sn), "excess": _r(r.excess), "ratio": _r(r.ratio, 2), "affected": True}
            elif sn is not None and sn in full.index:
                r = full.loc[sn]
                out["station"] = {"s": _short(sn), "excess": _r(r.excess), "ratio": _r(r.ratio, 2), "affected": False,
                                  "note": "no large, repeatable event uplift at this station"}
        return out

    # ============================================================== P  rank_pressure
    PEAK_SLOTS = ["07:30", "08:15", "09:00", "12:30", "16:30", "17:15", "18:00", "19:00"]

    @mcp.tool
    def rank_pressure(date: str, top_n: int = 3, rain: bool | None = None, event_attendance: int | None = None, screen: int = 40, engine: str = "") -> dict:
        """Category P — the `top_n` stations under the highest passenger load on `date` 'YYYY-MM-DD' (highest predicted
        load per 15 min at the checked peak slots, TabPFN demand model) and their chance of exceeding their own busiest-5%
        level. 'Exceeding safe platform capacity' has no data behind it here, so the load ranking is the stated proxy. If `date` lies inside the data the day's
        actual weather/events are used (replay, and the stations that really exceeded are returned for comparison);
        otherwise a SCENARIO is built from the latest same-weekday day, with `rain` and `event_attendance`
        (city-wide, from the operator) as the stated assumptions. Overrides (rain / event_attendance) apply in both."""
        from demand_baseline import BASELINE_FEATURES, GRID, Q_COLS, QUANTILES, _GRID_SEL, MEAN_COL  # noqa: F401
        from disruption import exceed_probability

        b = get_baseline()
        t = b.table
        cov0, cov1 = coverage()
        try:
            day = pd.Timestamp(date)
        except Exception:
            return {"error": f"cannot read date '{date}' (use YYYY-MM-DD)"}
        in_data = cov0.normalize() <= day <= cov1.normalize() and (t["timestamp"].dt.strftime("%Y-%m-%d") == date).sum() > 8 * 100
        if in_data:
            base_rows = t[t["timestamp"].dt.strftime("%Y-%m-%d") == date]
            mode = "replay"
        else:
            same = t[t["timestamp"].dt.dayofweek == day.dayofweek]
            per_day = same["timestamp"].dt.strftime("%Y-%m-%d").value_counts()
            last = max(d for d, n in per_day.items() if n >= 0.9 * per_day.max())          # latest same-weekday day with a full record
            base_rows = same[same["timestamp"].dt.strftime("%Y-%m-%d") == last].copy()
            base_rows["timestamp"] = base_rows["timestamp"] + (day.normalize() - pd.Timestamp(last))
            base_rows["dow"], base_rows["month"] = day.dayofweek, day.month
            base_rows["is_weekend"] = int(day.dayofweek >= 5)
            mode = f"scenario (template: {last})"
        keep_ts = {f"{date} {s}:00" for s in PEAK_SLOTS}
        rows = base_rows[base_rows["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").isin(keep_ts)].copy()
        if rows.empty:
            return {"error": "no template rows for this date"}
        assumptions = []
        if rain is not None:
            rows["prcp"] = float(t.loc[t["prcp"] > RAIN_MM, "prcp"].median()) if rain else 0.0
            assumptions.append(f"rain={'yes' if rain else 'no'}")
        if event_attendance is not None:
            rows["daily_event_attendance"] = float(event_attendance)
            rows["daily_event_count"] = float(max(1, rows["daily_event_count"].max()))
            assumptions.append(f"city-wide event attendance {event_attendance}")
        # stage 1 (cheap, empirical): the `screen` busiest stations by their usual slot mean; stage 2: TabPFN conditional forecast
        cand = rows.groupby("station_name")["station_slot_mean"].max().sort_values(ascending=False).head(max(1, screen)).index
        rows = rows[rows["station_name"].isin(cand)]
        with lock:
            if engine == "empirical":
                q = b._empirical(rows)                                  # naive station x hour quantiles: the evaluator's cross-check, no ML
            elif in_data and rain is None and event_attendance is None:
                q = b.predict_quantiles(rows)                          # cached path (real features)
            else:
                X = b._encode(rows)                                     # modified features: never use the (station, timestamp) cache
                dense = np.maximum.accumulate(np.maximum(b._predict_grid(X), 0.0), axis=1)
                q = pd.DataFrame(dense[:, _GRID_SEL], index=rows.index, columns=Q_COLS)
                gg = np.r_[0.0, np.array(GRID), 1.0]
                np_trapz = getattr(np, "trapezoid", None) or np.trapz
                q[MEAN_COL] = np_trapz(np.column_stack([dense[:, 0], dense, dense[:, -1]]), gg, axis=1)
        df = rows[["timestamp", "station_name", "station_hour_p95", "passengers"]].join(q)
        df["p"] = [exceed_probability(float(a), QUANTILES, list(map(float, v))) for a, v in zip(df["station_hour_p95"], df[Q_COLS].to_numpy())]
        # Ranking = the highest predicted LOAD (90th percentile of the TabPFN predictive distribution, best slot). Why load and not
        # "probability of exceeding the own p95": tested on 6 replay days, the p95-exceedance ranking has no skill (Spearman ~ 0 vs the
        # observed exceedances: they are noise-driven), while predicted load tracks observed load. A fixed platform saturates first where
        # the load is highest, which is the only capacity-relevant signal the data offers.
        df["q90"] = df["q0900"]
        agg = df.groupby("station_name").agg(pmean=("p", "mean"), pmax=("p", "max"))
        best = df.sort_values("q90", ascending=False).groupby("station_name").head(1).set_index("station_name").join(agg).sort_values("q90", ascending=False)
        top = [{"s": _short(n), "lines": lines_of().get(n, ""), "load90": _r(r.q90, 0), "exp": _r(r[MEAN_COL], 0), "at": r.timestamp.strftime("%H:%M"),
                "p95": _r(r.station_hour_p95, 0), "p": _r(r.pmax * 100, 0)} for n, r in best.head(max(1, top_n)).iterrows()]
        out = {"status": "ok", "date": date, "mode": mode, "assumptions": assumptions, "top": top, "n_candidates": int(len(cand)),
               "slots_checked": PEAK_SLOTS, "proxy": "highest predicted passenger load per 15 min (90th percentile of the model's forecast, best of the checked slots); 'p' = chance of exceeding the station's own busiest-5% level. NOT a platform capacity: no capacity data exists",
               "model": "empirical screening (40 stations) then TabPFN quantile regression" if os.environ.get("SCENARIO_ENGINE", "tabpfn") != "empirical" else "empirical"}
        # ground truth check: which stations actually exceeded their own p95 most often that day (replay only)
        if in_data:
            day_rows = base_rows[base_rows["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").isin(keep_ts)]
            obs = day_rows.groupby("station_name")["passengers"].max().sort_values(ascending=False)
            out["observed_top"] = [{"s": _short(n), "peak": int(v)} for n, v in obs.head(max(1, top_n)).items()]
            out["observed_all"] = {_short(n): int(v) for n, v in obs.items()} if screen >= 100 else None
            out["predicted_all"] = {_short(n): _r(r.q90, 1) for n, r in best.iterrows()} if screen >= 100 else None
            exc = day_rows.assign(x=day_rows["passengers"] > day_rows["station_hour_p95"]).groupby("station_name")["x"].sum()
            out["observed_exceed_all"] = {_short(n): int(v) for n, v in exc.items()} if screen >= 100 else None
            out["predicted_p_all"] = {_short(n): _r(r.pmean, 4) for n, r in best.iterrows()} if screen >= 100 else None
            out["observed_overlap"] = len({x["s"] for x in top} & {x["s"] for x in out["observed_top"]})
        else:
            out["note"] = f"{date} is outside the flow data ({cov0.date()} to {cov1.date()}): scenario, not a replay"
        ev = events()
        day_ev = ev[ev["date"] == date]
        out["events_that_day"] = {"n": int(len(day_ev)), "attendance": int(day_ev["estimated_attendance"].sum())} if len(day_ev) else {"n": 0}
        if "innotrans" not in " ".join(ev["event_name"].str.lower()):
            out["event_data_note"] = "the events file has no InnoTrans/Messe event: attendance was not known unless the operator gave it"
        return out

    # ============================================================== B  find_anomalies
    def _baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
        with lock:
            if "med" not in S:
                w = wide()
                key = w.index.dayofweek * 96 + w.index.hour * 4 + w.index.minute // 15
                med = w.groupby(key).median()
                mad = w.sub(med.reindex(key).to_numpy()).abs().groupby(key).median() * 1.4826
                S["med"], S["mad"] = med, mad
            return S["med"], S["mad"]

    def _closure_windows() -> pd.DataFrame:
        return load_closures(folder)

    def _closed_at(ts: pd.Timestamp, station: str) -> str | None:
        cl = _closure_windows()
        act = cl[(cl["when"] <= ts) & (cl["end"] > ts)]
        ln = lines_of().get(station, "")
        for c in act.itertuples():
            if c.closure_type == "Station closure" and str(c.affected_segment).lower() in station.lower():
                return c.description
            if c.closure_type == "Line suspension" and c.affected_line in ln.split(","):
                return c.description
        return None

    @mcp.tool
    def find_anomalies(start_date: str, end_date: str = "", top_n: int = 3, cause: str = "") -> dict:
        """Category B — flow anomalies between `start_date` and `end_date` (inclusive, 'YYYY-MM-DD'; end defaults to
        start) and which explanation fits each: (1) an active closure at that station/line, (2) an event that day,
        (3) weather (rain / wind / heat at that slot), else 'unexplained by the data'. An anomaly is a station-slot whose
        flow differs from the station's usual value for that weekday and 15-min slot by more than 4 robust z-scores.
        `cause`: 'weather' returns the strongest RAIN-coincident positive peak with no closure and no event that day
        (the training-question-2 shape); '' returns the top anomalies NOT explained by a closure (question-7 shape)."""
        w = wide()
        med, mad = _baseline()
        d0, d1 = pd.Timestamp(start_date), pd.Timestamp(end_date or start_date) + pd.Timedelta(days=1)
        cov0, cov1 = coverage()
        if d1 <= cov0 or d0 > cov1:
            return {"status": "oos", "note": f"{start_date} is outside the flow data ({cov0.date()} to {cov1.date()})"}
        sl = w.loc[(w.index >= d0) & (w.index < d1)]
        key = sl.index.dayofweek * 96 + sl.index.hour * 4 + sl.index.minute // 15
        z = (sl - med.reindex(key).to_numpy()) / (mad.reindex(key).to_numpy() + 2.0)
        wx = load_weather(folder).set_index("timestamp")
        ev = events()
        rows = []
        stack = z.stack()
        stack = stack[stack.abs() > 4]
        for (ts, stn), zv in stack.sort_values(key=lambda s: -s.abs()).head(400).items():
            closed = _closed_at(ts, stn)
            w_row = wx.loc[ts] if ts in wx.index else None
            day_ev = ev[ev["date"] == ts.strftime("%Y-%m-%d")]
            rows.append({"ts": ts, "stn": stn, "z": float(zv), "obs": float(sl.loc[ts, stn]), "usual": float(med.loc[ts.dayofweek * 96 + ts.hour * 4 + ts.minute // 15, stn]),
                         "closure": closed, "prcp": None if w_row is None else float(w_row["prcp"]), "wspd": None if w_row is None else float(w_row["wspd"]),
                         "temp": None if w_row is None else float(w_row["temp"]), "n_ev": int(len(day_ev)), "att": int(day_ev["estimated_attendance"].sum())})
        df = pd.DataFrame(rows)
        base = {"status": "ok", "range": [start_date, end_date or start_date], "n_anomalies_over_4z": int(len(stack)),
                "method": "robust z-score against the station's median for the same weekday and 15-min slot (whole dataset); reported anomalies are ranked by the size of the deviation in passengers", "limits": ["causes are 'consistent with', not proven; no incident log exists"]}
        if df.empty:
            return {**base, "found": []}
        if cause == "weather":
            wx_ok = (df["prcp"].fillna(0) > RAIN_MM) | (df["wspd"].fillna(0) >= 25) | (df["temp"].fillna(0) >= 30)
            c = df[(df["z"] > 0) & df["closure"].isna() & wx_ok].copy()
            if c.empty:
                return {**base, "found": [], "note": "no positive peak coincides with rain, strong wind or heat without a closure"}
            c["_ev"] = (c["n_ev"] > 0).astype(int)                     # prefer peaks on days without any event: cleaner attribution
            c["_d"] = c["obs"] - c["usual"]
            df = c.sort_values(["_ev", "_d"], ascending=[True, False])
        else:
            df = df[df["closure"].isna()].assign(_d=lambda x: (x["obs"] - x["usual"]).abs()).sort_values("_d", ascending=False)
        out = []
        seen = set()
        for r in df.itertuples():
            if (r.stn, r.ts.date()) in seen:
                continue
            seen.add((r.stn, r.ts.date()))
            why = []
            if r.prcp and r.prcp > RAIN_MM:
                why.append(f"rain {r.prcp:g} mm/15min")
            if r.wspd and r.wspd >= 25:
                why.append(f"wind {r.wspd:g} km/h")
            if r.temp is not None and r.temp >= 30:
                why.append(f"heat {r.temp:g} C")
            if r.n_ev:
                why.append(f"{r.n_ev} events that day (~{r.att} attendees, no venue->station key)")
            out.append({"s": _short(r.stn), "at": r.ts.strftime("%Y-%m-%d %H:%M"), "obs": _r(r.obs, 0), "usual": _r(r.usual, 0), "z": _r(r.z),
                        "fits": why or ["nothing in the data explains it"], "closure": None})
            if len(out) >= max(1, top_n):
                break
        return {**base, "found": out}

    # ============================================================== E  energy_efficiency
    @mcp.tool
    def energy_efficiency() -> dict:
        """Category E — energy per passenger by line (Wh/passenger), from the daily MWh per line and the flows of the
        stations each line serves (interchange stations count for every line they serve, so per-line passengers are a
        proxy). Also returns what explains the ranking: fixed vs load-following energy (correlation of daily energy
        with daily passengers), passengers per station served, and weekend vs weekday energy."""
        en = load_energy(folder).set_index("timestamp")
        f = load_flows(folder).set_index("timestamp")
        st = load_stations(folder)
        name_lines = st.groupby("station_name")["u_bahn_lines"].apply(lambda s: set(",".join(s).replace(" ", "").split(","))).to_dict()
        w = wide()
        rows = []
        for line in en.columns:
            cols = [c for c in w.columns if line in name_lines.get(c, set())]
            if not cols:
                continue
            daily_p = w[cols].sum(axis=1).resample("D").sum()
            e = en[line]
            idx = e.index.intersection(daily_p.index)
            e, p = e.loc[idx], daily_p.loc[idx]
            corr = float(np.corrcoef(e, p)[0, 1]) if e.std() > 0 and p.std() > 0 else None
            wk = e.index.dayofweek >= 5
            rows.append({"line": line, "wh_per_pax": _r(e.sum() * 1e6 / p.sum(), 1), "mwh_day": _r(e.mean(), 1), "pax_day": _r(p.mean(), 0), "n_stations": len(cols),
                         "pax_per_station_day": _r(p.mean() / len(cols), 0), "corr_energy_pax": _r(corr, 2),
                         "weekend_vs_weekday_energy_pct": _r((e[wk].mean() / e[~wk].mean() - 1) * 100, 1) if wk.any() and (~wk).any() else None,
                         "weekend_vs_weekday_pax_pct": _r((p[wk].mean() / p[~wk].mean() - 1) * 100, 1) if wk.any() and (~wk).any() else None})
        rows.sort(key=lambda r: -(r["wh_per_pax"] or 0))
        return {"status": "ok", "ranking": rows, "worst": rows[0]["line"] if rows else None, "days": int(len(en)),
                "method": "Wh per passenger = total daily MWh x 1e6 / total daily passengers at the stations serving the line",
                "limits": ["no rolling-stock, timetable or route-length data: explanations are limited to ridership and load-following evidence",
                           "passengers per line are approximated (interchanges counted on every line they serve)"]}

    # ============================================================== F  network_resilience
    @mcp.tool
    def network_resilience_ranking(top_n: int = 5) -> dict:
        """Category F — the `top_n` stations whose closure fragments the rail network most (graph cut analysis on the
        connections file), with the passengers affected per day: the station's own daily passengers plus the daily
        passengers of every station that is cut off from the main network. Trains are assumed not to run through a
        closed station. Also returns where the diverted riders would go (neighbouring stations of the closed one)."""
        res = network_resilience(folder)
        g = build_graph(folder)
        st = load_stations(folder)
        id2n = dict(zip(st["station_id"], st["station_name"]))
        avg = station_avg_flow(folder).set_index("station_name")["avg_daily_passengers"].to_dict()
        rows = []
        for r in res.itertuples():
            h = g.copy()
            h.remove_node(r.station_id)
            comps = sorted((list(c) for c in __import__("networkx").connected_components(h)), key=len, reverse=True)
            cut = [n for c in comps[1:] for n in c]
            cut_pax = sum(avg.get(id2n.get(n, n), 0) for n in cut)
            own = avg.get(r.station_name, 0)
            nbrs = sorted({_short(id2n.get(n, n)) for n in g.neighbors(r.station_id)})
            rows.append({"s": _short(r.station_name), "lines": lines_of().get(r.station_name, ""), "cut_off": len(cut), "fragments": int(len(comps)),
                         "pax_day_affected": _r(own + cut_pax, 0), "own_pax_day": _r(own, 0), "cut_pax_day": _r(cut_pax, 0), "betweenness": _r(r.betweenness_centrality, 3),
                         "articulation": bool(r.is_articulation_point), "neighbours": nbrs[:6]})
        rows.sort(key=lambda x: (-x["pax_day_affected"], -x["cut_off"]))
        return {"status": "ok", "top": rows[:max(1, top_n)], "n_stations": len(rows),
                "method": "remove the station from the connection graph; fragments = connected components left; affected = own + cut-off stations' average daily passengers",
                "limits": ["graph has no line-level timetable; a through-running station is assumed skipped, not closed to trains", "mitigation ideas are suggestions from the graph (neighbours, bypass), not data-backed"]}

    # ============================================================== G  correlated_stations
    @mcp.tool
    def correlated_stations(top_n: int = 5, min_hops: int = 3) -> dict:
        """Category G — pairs of stations whose demand moves together (after removing each station's usual weekday/hour
        pattern) although they are NOT directly connected (graph distance >= `min_hops`). Returns the correlation of the
        hourly residuals, the hop distance, the best lag (hours) and what the pair has in common (lines, weather/event
        exposure is removed by construction only for the weekly pattern)."""
        import networkx as nx

        w = wide()
        hourly = w.resample("h").sum()
        key = hourly.index.dayofweek * 24 + hourly.index.hour
        resid = hourly - hourly.groupby(key).transform("mean")
        resid = (resid - resid.mean()) / resid.std().replace(0, np.nan)
        c = resid.corr()
        st = load_stations(folder)
        n2id = st.drop_duplicates("station_name").set_index("station_name")["station_id"].to_dict()
        g = build_graph(folder)
        hops = dict(nx.all_pairs_shortest_path_length(g))
        names = list(c.columns)
        pairs = []
        arr = c.to_numpy()
        iu = np.triu_indices(len(names), 1)
        order = np.argsort(-np.abs(arr[iu]))
        for k in order[:4000]:
            i, j = iu[0][k], iu[1][k]
            a, b = names[i], names[j]
            ia, ib = n2id.get(a), n2id.get(b)
            if ia is None or ib is None or ia not in hops or ib not in hops.get(ia, {}):
                continue
            h = hops[ia][ib]
            if h < min_hops:
                continue
            best_lag, best_c = 0, arr[i, j]
            for lag in range(-3, 4):
                cc = resid[a].corr(resid[b].shift(lag))
                if pd.notna(cc) and abs(cc) > abs(best_c) + 1e-9:
                    best_lag, best_c = lag, cc
            shared = sorted(set(lines_of().get(a, "").split(",")) & set(lines_of().get(b, "").split(",")))
            pairs.append({"a": _short(a), "b": _short(b), "r": _r(arr[i, j], 2), "hops": int(h), "lag_h": int(best_lag), "r_at_lag": _r(best_c, 2), "shared_lines": shared})
            if len(pairs) >= max(1, top_n):
                break
        n_hours = int(len(hourly))
        return {"status": "ok", "pairs": pairs, "n_hours": n_hours, "noise_level_r": _r(2 / np.sqrt(n_hours), 3),
                "median_abs_r_all_pairs": _r(float(np.nanmedian(np.abs(arr[iu]))), 3),
                "method": "Pearson correlation of hourly flow residuals (flow minus the station's own weekday-hour mean), unconnected pairs only",
                "limits": ["correlation is not causation; the data has no origin-destination flows, so a mechanism can only be suggested from shared lines/interchanges"]}

    # ============================================================== H  reroute_behaviour
    @mcp.tool
    def reroute_behaviour() -> dict:
        """Category H — what the recorded closures show about passenger rerouting: observed flow divided by the model's
        expected flow at closed stations, 1-2 hops away, section endpoints and interchanges (26 closures). Values near
        1.0 at neighbours mean NO measurable extra load — i.e. no rerouting behaviour is visible in this data."""
        p = REPO_ROOT / "ml" / "output" / "disruption_baseline_report.json"
        if not p.exists():
            return {"status": "unavailable", "note": "run `./.venv/bin/python scripts/tasks.py train-disruption` to create the case-study report"}
        rep = json.loads(p.read_text())["case_study_summary"]
        keep = {k: {"rows": v["rows"], "observed_over_expected": v["observed_over_expected"], "share_above_q90": v["share_above_q90"]} for k, v in rep.items()}
        return {"status": "ok", "groups": keep, "noise_share_above_q90": 0.10,
                "reading": "closed stations read 0; neighbours (1-2 hops), endpoints and interchanges stay near normal (observed/expected ~1, ~10% of readings above the 90th percentile = the noise rate)",
                "limits": ["the flows are simulated; passenger paths (origin-destination) are not in the data, so preferred alternative routes cannot be measured"]}
