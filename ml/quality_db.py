"""QUALITY DATABASE — the high-quality, normalized view of the raw data, and the boundaries derived from it.

Built from `data/` by the preprocessing pipeline (`ml/nextmove_pipeline`, see docs/ml_preprocessing_pipeline.md):

  normal flow  = what a station carries in a normal 15-minute slot (time of day × day type, typical weather), fitted on CLEAN cells (no event / closure windows)
  weather part = the share explained by rain / heat / wind          rest = what is left (events, closures, anomalies)

    data/quality/cells.npz      every (timestamp × station) cell: actual, normal, expected, total, weather, rest, excluded  (training + test split, same model)
    data/quality/quality.db     SQLite: meta, stations, boundaries, station_bounds, network_bounds, episodes, event_effects, coefficients, anomalies, daily, data_issues

Use: `QualityDB()` answers questions ("what is normal here at 18:15?", "is 2 600 passengers plausible?", "did this closure / event change anything?") and
`check_facts(category, facts)` verifies the facts of an answer against the boundaries — the Inspector (evaluator) calls it through the quality MCP server.

    ./.venv/bin/python ml/quality_db.py build [--offline] [--freq 15min]     # pipeline (if needed) + database
    ./.venv/bin/python ml/quality_db.py status
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parent
REPO = ML.parent
sys.path[:0] = [str(ML)]

OUT = REPO / "data" / "quality"
NORM = REPO / "data" / "normalized"
DAY_TYPES = {0: "Mon-Thu", 1: "Fri", 2: "Sat", 3: "Sun"}
HARD_CEILING_FACTOR = 1.5          # a claimed passenger figure above 1.5 x the highest ever observed at that station is impossible for this dataset
SOFT_CEILING_FACTOR = 1.15
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


def _day_type(idx) -> np.ndarray:
    from nextmove_pipeline.baseline import day_type
    return day_type(idx)


# ==================================================================================================== build
def build(freq: str = "15min", offline: bool = True, rebuild_pipeline: bool = False) -> dict:
    """Run the pipeline (fit on training, apply to test) if its outputs are missing, then write cells.npz and quality.db."""
    from nextmove_pipeline import config, loading
    from nextmove_pipeline.baseline import exclusion_mask
    from nextmove_pipeline.episodes import parse_duration  # noqa: F401
    from nextmove_pipeline.pipeline import run_all

    t0 = time.time()
    if rebuild_pipeline or not (NORM / "normalized_rest.csv").exists() or not (NORM / "normalized_rest_test.csv").exists():
        run_all(NORM, freq, offline, config.DEFAULT_GEOCODE_CACHE)
    OUT.mkdir(parents=True, exist_ok=True)

    def wide(name):
        a = pd.read_csv(NORM / f"{name}.csv", index_col=0, parse_dates=True)
        b = pd.read_csv(NORM / f"{name}_test.csv", index_col=0, parse_dates=True)
        return pd.concat([a, b]).sort_index()

    total, weather, rest, normal = (wide(n) for n in ("normalized_flows", "normalized_weather", "normalized_rest", "normal_flow_passengers"))
    stations = list(rest.columns)
    idx = rest.index
    split = np.where(idx <= pd.read_csv(NORM / "normalized_rest.csv", index_col=0, parse_dates=True).index[-1], "train", "test")
    actual_parts = []
    for d in (config.DEFAULT_DATA_DIR, config.TEST_DATA_DIR):
        st = loading.load_stations(d)
        actual_parts.append(loading.load_flows(d, st.station_name))
    actual = pd.concat(actual_parts).sort_index().reindex(idx)[stations]
    expected = np.log1p(normal.to_numpy()) + weather.to_numpy()           # log scale: normal + weather part
    episodes = pd.concat([pd.read_csv(NORM / "episodes.csv").assign(split="train"), pd.read_csv(NORM / "episodes_test.csv").assign(split="test")], ignore_index=True)
    for c in ("start", "end"):
        episodes[c] = pd.to_datetime(episodes[c])
    st_all = loading.load_stations(config.DEFAULT_DATA_DIR)
    adj = loading.load_network(config.DEFAULT_DATA_DIR, st_all)
    ep_dicts = [dict(anchors=r.anchors.split("; "), start=r.start, end=r.end) for r in episodes.itertuples()]
    excluded = exclusion_mask(idx, stations, adj, ep_dicts)

    np.savez_compressed(OUT / "cells.npz", ts=idx.to_numpy("datetime64[ns]").astype("int64"), stations=np.array(stations), split=split, actual=actual.to_numpy(np.float32),
                        normal=normal.to_numpy(np.float32), total=total.to_numpy(np.float32), weather=weather.to_numpy(np.float32), rest=rest.to_numpy(np.float32),
                        expected=np.expm1(expected).astype(np.float32), excluded=excluded)

    dtype, hour = _day_type(idx), np.asarray(idx.hour)
    A, N, R = actual.to_numpy(float), normal.to_numpy(float), rest.to_numpy(float)
    clean = ~excluded & ~np.isnan(A)

    rows_b, rows_s = [], []
    weekday = np.asarray(idx.dayofweek) <= 4
    for j, s in enumerate(stations):
        a, n, r, c = A[:, j], N[:, j], R[:, j], clean[:, j]
        hourly_wk = pd.Series(a[weekday & ~np.isnan(a)]).groupby(hour[weekday & ~np.isnan(a)]).mean()
        hourly_we = pd.Series(a[~weekday & ~np.isnan(a)]).groupby(hour[~weekday & ~np.isnan(a)]).mean()
        valid = ~np.isnan(a)
        clipped = float(np.mean(a[valid] == 500)) if valid.any() else 0.0
        ev = excluded[:, j] & valid
        rows_s.append(dict(station=s, max_all=float(np.nanmax(a)), max_clean=float(np.nanmax(np.where(c, a, np.nan))), p999_clean=float(np.nanquantile(a[c], 0.999)),
                           max_episode=float(np.nanmax(np.where(ev, a, np.nan))) if ev.any() else None, mean_daily=float(np.nansum(a) / max(1, len(np.unique(idx.date)))),
                           weekday_peak_hour=int(hourly_wk.idxmax()), weekday_peak_avg=float(hourly_wk.max()), weekend_peak_hour=int(hourly_we.idxmax()), weekend_peak_avg=float(hourly_we.max()),
                           clipped_at_500_share=round(clipped, 4), hard_ceiling=float(np.nanmax(a) * HARD_CEILING_FACTOR), soft_ceiling=float(np.nanmax(a) * SOFT_CEILING_FACTOR), n_cells=int(valid.sum())))
        for d in range(4):
            for h in range(24):
                m = c & (dtype == d) & (hour == h)
                if m.sum() < 8:
                    continue
                q = np.quantile(a[m], QUANTILES)
                good = m & (n >= 20) & ~np.isnan(r)
                rq = np.quantile(r[good], (0.05, 0.95)) if good.sum() >= 8 else (np.nan, np.nan)
                nm = float(np.nanmedian(n[m]))
                rows_b.append((s, d, h, int(m.sum()), nm, float(a[m].mean()), *map(float, q), float(a[m].max()), float(rq[0]), float(rq[1]),
                               float(nm * np.exp(rq[0])) if not np.isnan(rq[0]) else None, float(nm * np.exp(rq[1])) if not np.isnan(rq[1]) else None))
    bounds = pd.DataFrame(rows_b, columns=["station", "day_type", "hour", "n_clean", "normal_median", "mean_clean", "p01", "p05", "p50", "p95", "p99", "max_clean", "rest_q05", "rest_q95", "band_low", "band_high"])
    sbounds = pd.DataFrame(rows_s)

    # network per day type x hour
    net = pd.DataFrame({"total": np.nansum(A, axis=1), "normal": np.nansum(N, axis=1), "day_type": dtype, "hour": hour, "ok": ~np.all(np.isnan(A), axis=1)})
    net = net[net.ok]
    nb = net.groupby(["day_type", "hour"]).agg(n=("total", "size"), actual_p05=("total", lambda x: x.quantile(.05)), actual_p50=("total", "median"), actual_p95=("total", lambda x: x.quantile(.95)),
                                               actual_max=("total", "max"), normal_median=("normal", "median")).reset_index()

    # episodes with measured effect at the anchors
    col = {s: j for j, s in enumerate(stations)}
    eff = []
    for r_ in episodes.itertuples():
        cols = [col[a] for a in r_.anchors.split("; ") if a in col]
        i0, i1 = idx.searchsorted(r_.start), idx.searchsorted(r_.end + pd.Timedelta(hours=1))
        w = R[i0:i1][:, cols] if cols and i1 > i0 else np.empty((0, 0))
        ok = w[~np.isnan(w)]
        aw = A[i0:i1][:, cols] if cols and i1 > i0 else np.empty((0, 0))
        nw = N[i0:i1][:, cols] if cols and i1 > i0 else np.empty((0, 0))
        ratio_hour = None
        if aw.size:                                                    # the uplift of the busiest HOUR (4 slots) at the anchors: robust against single noisy 15-minute slots
            a1, n1 = np.nansum(aw, axis=1), np.nansum(nw, axis=1)
            if len(a1) >= 4 and n1.sum() > 0:
                ra, rn = np.convolve(a1, np.ones(4), "valid"), np.convolve(n1, np.ones(4), "valid")
                ratio_hour = float(np.max(ra[rn > 0] / rn[rn > 0])) if (rn > 0).any() else None
        eff.append(dict(mean_rest=float(ok.mean()) if ok.size else None, peak_rest=float(ok.max()) if ok.size else None, min_rest=float(ok.min()) if ok.size else None,
                        ratio_peak=ratio_hour, actual_peak=float(np.nanmax(aw)) if aw.size and not np.all(np.isnan(aw)) else None, n_cells=int(ok.size)))
    ep = pd.concat([episodes.assign(start=episodes.start.astype(str), end=episodes.end.astype(str)), pd.DataFrame(eff)], axis=1)
    summ = []
    for label, sel in (("event <5000 visitors", (ep.kind == "event") & (ep.attendance < 5000)), ("event >=5000 visitors", (ep.kind == "event") & (ep.attendance >= 5000)), ("closure", ep.kind == "closure")):
        g = ep[sel & ep.ratio_peak.notna()]
        if len(g):
            summ.append(dict(group=label, n=len(g), median_ratio_peak=float(g.ratio_peak.median()), p90_ratio_peak=float(g.ratio_peak.quantile(.9)), max_ratio_peak=float(g.ratio_peak.max()),
                             median_mean_rest=float(g.mean_rest.median()), min_mean_rest=float(g.mean_rest.min())))
    event_effects = pd.DataFrame(summ)

    coef = pd.read_csv(NORM / "normal_flow_coefficients.csv")
    # anomalies: large unexplained deviations outside episode windows
    big = (R >= 2.0) & ~excluded & (N >= 30) & ~np.isnan(A)                 # unexplained SPIKES (upward deviations outside event / closure windows)
    ti, si = np.nonzero(big)
    zero = (A == 0) & ~excluded & (N >= 100)                                # readings of exactly 0 where a station normally carries >= 100 per slot: outages / gaps
    zt, zs = np.nonzero(zero)
    outages = pd.DataFrame({"ts": idx[zt].astype(str), "date": idx[zt].strftime("%Y-%m-%d"), "station": np.array(stations)[zs], "normal": N[zt, zs], "split": split[zt]})
    anomalies = pd.DataFrame({"ts": idx[ti].astype(str), "date": idx[ti].strftime("%Y-%m-%d"), "station": np.array(stations)[si], "actual": A[ti, si], "normal": N[ti, si], "rest": R[ti, si],
                              "weather": weather.to_numpy()[ti, si], "split": split[ti]})
    # daily totals
    day = idx.strftime("%Y-%m-%d")
    dsum = pd.DataFrame(A, index=day, columns=stations).groupby(level=0).sum(min_count=1)
    daily = dsum.stack().rename("passengers").rename_axis(["date", "station"]).reset_index()

    # data-quality issues found while building (what a user of the raw data must know)
    issues = []
    cl_share = float(np.mean(A[~np.isnan(A)] == 500))
    issues.append(("clipping", "warning", f"{cl_share:.1%} of readings equal exactly 500 (simulation clipping) although values up to {int(np.nanmax(A))} exist; stations with the highest share: "
                   + ", ".join(sbounds.sort_values('clipped_at_500_share', ascending=False).head(3).station.str.replace(r" \(Berlin\)", "", regex=True))))
    for st_ in ("train", "test"):
        m = split == st_
        issues.append(("coverage", "info", f"{st_}: {idx[m][0]} .. {idx[m][-1]}, {int(m.sum())} slots, {int(np.isnan(A[m]).all(axis=1).sum())} slots with no data (night pause)"))
    issues.append(("outages", "warning", f"{len(outages)} readings ({len(outages) / max(1, int(((N >= 100) & ~np.isnan(A)).sum())):.1%} of the slots that normally carry >= 100) are exactly 0 outside any event or closure window: sensor gaps / outages; a claimed zero flow is not proof of a closure"))
    issues.append(("noise", "info", f"at the 15-minute grain the log residual has a standard deviation of {np.nanstd(R[(N >= 100) & ~excluded]):.2f}: single slots are noisy, judge patterns over hours, not one slot"))
    issues.append(("columns", "info", "flow column 'U Stadtmitte (Berlin).1' (a duplicate interchange column) has no station metadata and is dropped"))
    unm = episodes[(episodes.kind == "event")]
    issues.append(("events", "info", f"{len(unm)} events >= 2000 visitors mapped to a station (override / geocode / name); events below that size and events without a station are treated as noise"))
    issues.append(("weather", "info", f"weather explains {1 - np.nanvar(R) / np.nanvar(total.to_numpy()):.1%} of the variance of the total: the flows are simulated with little weather signal"))
    issues.append(("test_shift", "warning", f"the test split runs {np.nanmedian(total.to_numpy()[split == 'test']):+.2f} (log) above the training-normal: the held-out days are busier than the training normal (InnoTrans week / trend), judge them against test-window boundaries too"))

    meta = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "freq": freq, "n_stations": len(stations), "n_slots": int(len(idx)), "train_range": [str(idx[split == "train"][0]), str(idx[split == "train"][-1])],
            "test_range": [str(idx[split == "test"][0]), str(idx[split == "test"][-1])], "hard_ceiling_factor": HARD_CEILING_FACTOR, "soft_ceiling_factor": SOFT_CEILING_FACTOR, "n_episodes": int(len(ep)),
            "pipeline": "ml/nextmove_pipeline (ridge normal-flow model per station on clean cells, weather decomposition)", "build_seconds": round(time.time() - t0, 1)}
    db = OUT / "quality.db"
    if db.exists():
        db.unlink()
    with sqlite3.connect(db) as c:
        pd.DataFrame([(k, json.dumps(v)) for k, v in meta.items()], columns=["key", "value"]).to_sql("meta", c, index=False)
        st_all[["station_name", "u_bahn_lines", "longitude", "latitude"]].assign(has_flow=lambda d: d.station_name.isin(stations)).to_sql("stations", c, index=False)
        bounds.to_sql("boundaries", c, index=False)
        sbounds.to_sql("station_bounds", c, index=False)
        nb.to_sql("network_bounds", c, index=False)
        ep.to_sql("episodes", c, index=False)
        event_effects.to_sql("event_effects", c, index=False)
        coef.to_sql("coefficients", c, index=False)
        anomalies.to_sql("anomalies", c, index=False)
        outages.to_sql("outages", c, index=False)
        daily.to_sql("daily", c, index=False)
        pd.DataFrame(issues, columns=["topic", "level", "detail"]).to_sql("data_issues", c, index=False)
        c.execute("CREATE INDEX b_idx ON boundaries(station, day_type, hour)")
        c.execute("CREATE INDEX a_idx ON anomalies(date, station)")
        c.execute("CREATE INDEX d_idx ON daily(station, date)")
    load.cache_clear()
    return meta


# ==================================================================================================== query
class QualityDB:
    """Read access. Lazy: nothing is loaded until the first question; `available` is False when the database was never built."""

    def __init__(self, out: Path = OUT):
        self.out = Path(out)
        self._cells = None
        self._conn = None

    @property
    def available(self) -> bool:
        return (self.out / "quality.db").exists() and (self.out / "cells.npz").exists()

    # -- plumbing
    def sql(self, q: str, args=()) -> pd.DataFrame:
        if self._conn is None:
            self._conn = sqlite3.connect(self.out / "quality.db", check_same_thread=False)
        return pd.read_sql_query(q, self._conn, params=args)

    def cells(self) -> dict:
        if self._cells is None:
            z = np.load(self.out / "cells.npz", allow_pickle=False)
            ts = pd.to_datetime(z["ts"])
            stations = [str(s) for s in z["stations"]]
            from nextmove_pipeline.names import StationNameIndex
            self._cells = {"ts": ts, "stations": stations, "col": {s: j for j, s in enumerate(stations)}, "index": StationNameIndex(stations), **{k: z[k] for k in
                           ("split", "actual", "normal", "total", "weather", "rest", "expected", "excluded")}}
        return self._cells

    def resolve(self, station: str) -> str | None:
        c = self.cells()
        if station in c["col"]:
            return station
        return c["index"].resolve(station, cutoff=0.8) or next((s for s in c["stations"] if station.lower() in s.lower()), None)

    def _slot(self, station: str, at) -> tuple[int, int] | None:
        c = self.cells()
        j = c["col"].get(station)
        i = c["ts"].searchsorted(pd.Timestamp(at))
        if j is None or i >= len(c["ts"]) or abs((c["ts"][i] - pd.Timestamp(at)).total_seconds()) > 900:
            return None
        return i, j

    # -- questions
    def status(self) -> dict:
        if not self.available:
            return {"available": False, "hint": "build it: ./.venv/bin/python ml/quality_db.py build"}
        meta = {r.key: json.loads(r.value) for r in self.sql("SELECT * FROM meta").itertuples()}
        counts = {t: int(self.sql(f"SELECT COUNT(*) n FROM {t}").n[0]) for t in ("boundaries", "station_bounds", "network_bounds", "episodes", "anomalies", "outages", "daily", "data_issues")}
        return {"available": True, **meta, "tables": counts}

    def normal_at(self, station: str, at) -> dict:
        s = self.resolve(station)
        if s is None:
            return {"error": f"unknown station '{station}'"}
        pos = self._slot(s, at)
        if pos is None:
            return {"error": f"no cell for {station} at {at} (data window {self.cells()['ts'][0]} .. {self.cells()['ts'][-1]})"}
        i, j = pos
        c = self.cells()
        a = c["actual"][i, j]
        b = self.boundary(s, c["ts"][i])
        r = c["rest"][i, j]
        return {"station": s, "at": str(c["ts"][i]), "split": str(c["split"][i]), "actual": None if np.isnan(a) else round(float(a), 1), "normal": round(float(c["normal"][i, j]), 1),
                "expected_with_actual_weather": round(float(c["expected"][i, j]), 1), "weather_effect_pct": round(float(np.expm1(c["weather"][i, j])) * 100, 1),
                "rest_log": None if np.isnan(r) else round(float(r), 2), "rest_pct": None if np.isnan(r) else round(float(np.expm1(r)) * 100, 1), "in_episode_window": bool(c["excluded"][i, j]),
                "boundary": b, "episodes": self.episodes_at(s, c["ts"][i])}

    def boundary(self, station: str, at) -> dict | None:
        t = pd.Timestamp(at)
        d = int(_day_type(pd.DatetimeIndex([t]))[0])
        df = self.sql("SELECT * FROM boundaries WHERE station=? AND day_type=? AND hour=?", (station, d, t.hour))
        if df.empty:
            return None
        r = df.iloc[0]
        sb = self.sql("SELECT hard_ceiling, soft_ceiling, max_all FROM station_bounds WHERE station=?", (station,)).iloc[0]
        return {"day_type": DAY_TYPES[d], "hour": t.hour, "normal_median": round(r.normal_median, 1), "mean_clean": round(r.mean_clean, 1), "p05": round(r.p05, 1), "p50": round(r.p50, 1), "p95": round(r.p95, 1), "p99": round(r.p99, 1),
                "max_clean": round(r.max_clean, 1), "normal_band": [None if pd.isna(r.band_low) else round(r.band_low, 1), None if pd.isna(r.band_high) else round(r.band_high, 1)],
                "station_max_ever": round(sb.max_all, 1), "hard_ceiling": round(sb.hard_ceiling, 1), "soft_ceiling": round(sb.soft_ceiling, 1), "n_clean": int(r.n_clean)}

    def check_value(self, station: str, at, value: float) -> dict:
        """Is `value` passengers per 15 minutes plausible at that station and time?  verdict: normal | high | low | extreme | impossible."""
        s = self.resolve(station)
        if s is None:
            return {"error": f"unknown station '{station}'"}
        b = self.boundary(s, at)
        if b is None:
            return {"error": "no boundary for that station / time"}
        v = float(value)
        if v < 0 or v > b["hard_ceiling"]:
            verdict, why = "impossible", f"above {b['hard_ceiling']} (1.5 x the highest value ever observed at this station) or negative"
        elif v > b["soft_ceiling"]:
            verdict, why = "extreme", f"above the highest value ever observed ({b['station_max_ever']}) by more than 15 %"
        elif v > b["p99"]:
            verdict, why = "high", f"above the 99th percentile of clean slots ({b['p99']})"
        elif v > b["p95"]:
            verdict, why = "high", f"above the 95th percentile of clean slots ({b['p95']})"
        elif v < b["p05"]:
            verdict, why = "low", f"below the 5th percentile of clean slots ({b['p05']})"
        else:
            verdict, why = "normal", f"inside the 5-95 % range of clean slots ({b['p05']}..{b['p95']})"
        ratio = v / b["normal_median"] if b["normal_median"] else None
        return {"station": s, "at": str(pd.Timestamp(at)), "value": v, "verdict": verdict, "why": why, "ratio_to_normal": None if ratio is None else round(ratio, 2), "boundary": b}

    def station_profile(self, station: str) -> dict:
        s = self.resolve(station)
        if s is None:
            return {"error": f"unknown station '{station}'"}
        sb = self.sql("SELECT * FROM station_bounds WHERE station=?", (s,)).iloc[0].to_dict()
        prof = self.sql("SELECT day_type, hour, normal_median, p05, p95 FROM boundaries WHERE station=? ORDER BY day_type, hour", (s,))
        coef = self.sql("SELECT * FROM coefficients WHERE station=?", (s,))
        lines = self.sql("SELECT u_bahn_lines FROM stations WHERE station_name=?", (s,))
        return {"station": s, "lines": lines.u_bahn_lines.iloc[0] if len(lines) else None, "bounds": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in sb.items() if k != "station"},
                "weekday_normal_by_hour": {int(h): round(v, 1) for h, v in prof[prof.day_type <= 1].groupby("hour").normal_median.mean().items()},
                "weather_effects_pct": coef.drop(columns="station").iloc[0].to_dict() if len(coef) else None}

    def episodes_at(self, station: str, at, hours: float = 3.0) -> list[dict]:
        t = pd.Timestamp(at)
        ep = self.sql("SELECT id, kind, type, name, start, end, anchors, attendance, split, ratio_peak, mean_rest FROM episodes")
        ep["s"], ep["e"] = pd.to_datetime(ep.start), pd.to_datetime(ep.end)
        hit = ep[(ep.s - pd.Timedelta(hours=hours) <= t) & (ep.e + pd.Timedelta(hours=2) >= t) & ep.anchors.str.contains(station, regex=False)]
        return [{k: (None if pd.isna(v) else v) for k, v in r.items() if k not in ("s", "e")} for r in hit.to_dict("records")]

    def episodes(self, date: str = "", station: str = "", limit: int = 20) -> list[dict]:
        ep = self.sql("SELECT id, kind, type, name, start, end, anchors, attendance, split, ratio_peak, mean_rest FROM episodes ORDER BY start")
        if date:
            ep = ep[ep.start.str.startswith(date) | ep.end.str.startswith(date)]
        if station:
            s = self.resolve(station) or station
            ep = ep[ep.anchors.str.contains(s, regex=False)]
        return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in ep.head(limit).to_dict("records")]

    def anomalies(self, date: str, top_n: int = 5) -> list[dict]:
        df = self.sql("SELECT ts, station, actual, normal, rest, weather FROM anomalies WHERE date=? ORDER BY ABS(rest) DESC LIMIT ?", (date, top_n))
        return [{**r, "actual": round(r["actual"], 1), "normal": round(r["normal"], 1), "rest_pct": round(float(np.expm1(r["rest"])) * 100)} for r in df.to_dict("records")]

    def network_bounds(self, day_type: int | None = None, hour: int | None = None) -> list[dict]:
        q, args = "SELECT * FROM network_bounds WHERE 1=1", []
        if day_type is not None:
            q += " AND day_type=?"
            args.append(day_type)
        if hour is not None:
            q += " AND hour=?"
            args.append(hour)
        return self.sql(q, args).round(1).to_dict("records")

    def event_effects(self) -> list[dict]:
        return self.sql("SELECT * FROM event_effects").round(3).to_dict("records")

    def weather_effects(self, station: str = "") -> dict:
        c = self.sql("SELECT * FROM coefficients")
        if station:
            s = self.resolve(station)
            return c[c.station == s].iloc[0].to_dict() if s and (c.station == s).any() else {"error": f"unknown station '{station}'"}
        return {"network_median_pct": c.drop(columns="station").median().round(2).to_dict(), "stations": len(c)}

    def data_issues(self) -> list[dict]:
        return self.sql("SELECT * FROM data_issues").to_dict("records")

    # -- the Inspector's use: verify the facts of an answer against the boundaries
    def check_facts(self, category: str, facts: dict) -> dict:
        checks: list[dict] = []
        used: list[dict] = []

        def add(id_, ok, hard, detail, evidence=None):
            checks.append({"id": id_, "ok": bool(ok), "hard": bool(hard), "detail": detail, **({"evidence": evidence} if evidence else {})})

        if not self.available:
            return {"available": False, "checks": [], "boundaries": [], "summary": {"n_checks": 0, "n_hard_failed": 0, "n_soft_flags": 0}}
        default_date = None
        for k in ("date",):
            default_date = facts.get(k) or default_date
        cl = facts.get("cl") or {}
        default_date = default_date or (cl.get("from") or "")[:10] or ((facts.get("range") or [""])[0])

        def when(t):
            t = str(t)
            return pd.Timestamp(t if len(t) > 5 else f"{default_date} {t}") if (len(t) > 5 or default_date) else None

        def station_of(name):
            return self.resolve(name)

        if category == "D":
            for s in facts.get("st") or []:
                st = station_of(s.get("s", ""))
                if st is None or "wk" not in s:
                    continue
                sb = self.sql("SELECT * FROM station_bounds WHERE station=?", (st,)).iloc[0]
                hour, avg = s["wk"]
                dh = abs(int(sb.weekday_peak_hour) - int(hour))
                add("Q-PEAK-HOUR", dh <= 1, False, f"{st}: weekday peak hour {hour}:00 in the answer, {int(sb.weekday_peak_hour)}:00 in the quality database", {"db_peak_hour": int(sb.weekday_peak_hour)})
                rel = abs(avg - sb.weekday_peak_avg) / max(1.0, sb.weekday_peak_avg)
                add("Q-H-PEAK-VALUE", rel <= 0.4, True, f"{st}: weekday peak {avg} per 15 min vs database {sb.weekday_peak_avg:.0f} ({rel:.0%} apart)", {"db_weekday_peak_avg": round(float(sb.weekday_peak_avg), 1)})
                used.append({"id": f"Q-BOUND:{st}", "station": st, "weekday_peak_avg": round(float(sb.weekday_peak_avg), 1), "hard_ceiling": round(float(sb.hard_ceiling), 1)})
            if facts.get("net"):
                pass
        elif category == "B":
            for f in facts.get("found") or []:
                st, t = station_of(f.get("s", "")), when(f.get("at", ""))
                if st is None or t is None:
                    continue
                n = self.normal_at(st, t)
                if "error" in n:
                    add("Q-H-CELL", False, True, f"{st} at {t}: {n['error']}")
                    continue
                add("Q-H-OBS", n["actual"] is not None and abs(n["actual"] - f.get("obs", -1)) <= 1.5, True, f"{st} {n['at']}: {f.get('obs')} passengers in the answer, {n['actual']} in the database")
                us = f.get("usual")
                mc = (n.get("boundary") or {}).get("mean_clean")
                if us and mc:
                    add("Q-USUAL", 0.5 <= us / mc <= 2.0 or 0.5 <= us / max(1.0, n["normal"]) <= 2.0, False, f"{st}: 'usual' {us} vs database mean of clean slots {mc} (normal median {n['normal']})", {"db_mean_clean": mc})
                add("Q-ANOMALY", abs(n["rest_log"] or 0) >= 1.0, False, f"{st} {n['at']}: deviation from normal {n['rest_pct']}% (rest {n['rest_log']})" + ("; an event / closure window was active" if n["in_episode_window"] else "; no event / closure window active"),
                    {"episodes": [e["name"] for e in n["episodes"]][:2]})
                used.append({"id": f"Q-BOUND:{st}", "station": st, "at": n["at"], "boundary": n["boundary"]})
        elif category == "C":
            for r in (facts.get("press") or [])[:6]:
                st, t = station_of(r.get("s", "")), when(r.get("at", ""))
                if st is None or t is None:
                    continue
                b = self.boundary(st, t)
                if not b:
                    continue
                tot = r.get("tot")
                if tot is not None:
                    add("Q-H-CEILING", 0 <= tot <= b["hard_ceiling"], True, f"{st} {t:%H:%M}: predicted total {tot} per 15 min vs the highest ever observed {b['station_max_ever']} (hard ceiling {b['hard_ceiling']})")
                base = r.get("base")
                if base and b["mean_clean"]:
                    add("Q-BASE", 0.5 <= base / max(1.0, b["mean_clean"]) <= 2.0 or 0.5 <= base / max(1.0, b["normal_median"]) <= 3.0, False, f"{st}: 'base' {base} vs database mean of clean slots {b['mean_clean']} (median {b['normal_median']}; flows are heavy-tailed, mean > median)")
                used.append({"id": f"Q-BOUND:{st}", "station": st, "at": str(t), "boundary": b})
        elif category == "P":
            t = None
            for r in (facts.get("top") or [])[:6]:
                st = station_of(r.get("s", ""))
                if st is None:
                    continue
                sb = self.sql("SELECT * FROM station_bounds WHERE station=?", (st,)).iloc[0]
                v = r.get("load90")
                if v is not None:
                    add("Q-H-CEILING", 0 <= v <= sb.hard_ceiling, True, f"{st}: predicted load {v} per 15 min vs the highest ever observed {sb.max_all:.0f} (hard ceiling {sb.hard_ceiling:.0f})")
                    add("Q-SOFT-CEILING", v <= sb.soft_ceiling, False, f"{st}: predicted load {v} vs soft ceiling {sb.soft_ceiling:.0f}")
                used.append({"id": f"Q-BOUND:{st}", "station": st, "max_ever": round(float(sb.max_all), 1), "hard_ceiling": round(float(sb.hard_ceiling), 1)})
        elif category == "A":
            for r in (facts.get("top") or [])[:5]:
                st = station_of(r.get("s", ""))
                if st is None:
                    continue
                sb = self.sql("SELECT * FROM station_bounds WHERE station=?", (st,)).iloc[0]
                add("Q-H-EXCESS", 0 <= r.get("excess", 0) <= sb.hard_ceiling, True, f"{st}: event excess {r.get('excess')} per 15 min vs hard ceiling {sb.hard_ceiling:.0f}")
                eff = self.event_effects()
                mx = max((e["max_ratio_peak"] for e in eff if str(e["group"]).startswith("event")), default=None)
                if mx and r.get("ratio"):
                    add("Q-EVENT-RATIO", r["ratio"] <= mx * 1.5, False, f"{st}: uplift {r['ratio']}x normal vs the largest event peak measured at a venue station ({mx:.1f}x)")
                used.append({"id": f"Q-BOUND:{st}", "station": st, "hard_ceiling": round(float(sb.hard_ceiling), 1)})
        elif category == "F":
            tot_daily = float(self.sql("SELECT SUM(passengers) p FROM daily").p[0] / max(1, self.sql("SELECT COUNT(DISTINCT date) n FROM daily").n[0]))
            for r in (facts.get("top") or [])[:5]:
                st = station_of(r.get("s", ""))
                if st is None:
                    continue
                add("Q-H-AFFECTED", 0 < r.get("pax", 0) <= tot_daily * 1.02, True, f"{st}: {r.get('pax')} passengers affected per day vs the whole network's {tot_daily:.0f} per day")
                own = self.sql("SELECT mean_daily FROM station_bounds WHERE station=?", (st,)).mean_daily[0]
                if r.get("own"):
                    add("Q-OWN-DAILY", abs(r["own"] - own) / max(1.0, own) <= 0.15, False, f"{st}: own daily passengers {r['own']} vs database {own:.0f}")
        elif category == "H":
            for k, v in (facts.get("obs_over_exp") or {}).items():
                add("Q-H-RATIO", 0 <= v <= 3, True, f"observed/expected at {k}: {v}")
        # every category: nothing may name a date outside the database's window
        st = self.status()
        if default_date and st.get("test_range") and not (st["train_range"][0][:10] <= default_date[:10] <= st["test_range"][1][:10]):
            add("Q-WINDOW", False, False, f"date {default_date[:10]} is outside the database window {st['train_range'][0][:10]}..{st['test_range'][1][:10]}")
        return {"available": True, "category": category, "checks": checks, "boundaries": used,
                "summary": {"n_checks": len(checks), "n_hard_failed": sum(1 for c in checks if c["hard"] and not c["ok"]), "n_soft_flags": sum(1 for c in checks if not c["hard"] and not c["ok"])}}


@lru_cache(maxsize=1)
def load() -> QualityDB:
    return QualityDB()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "status"])
    ap.add_argument("--freq", default="15min")
    ap.add_argument("--offline", action="store_true", help="never query Nominatim (use data/processed/geocode_cache_pipeline.json)")
    ap.add_argument("--rebuild-pipeline", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        print(json.dumps(build(a.freq, a.offline, a.rebuild_pipeline), indent=1))
    else:
        print(json.dumps(QualityDB().status(), indent=1))


if __name__ == "__main__":
    main()
