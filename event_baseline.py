
import argparse
import json
import re
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SLOT = pd.Timedelta("15min")
SLOTS_PER_WEEK = 7 * 24 * 4
TOTAL_COL = "__TOTAL__"

# Annahme fuer fehlendes estimated_end_local, per Stichwort in segment/genre
DEFAULT_DURATION_H = [
    ("football", 2.0), ("soccer", 2.0), ("sport", 2.5),
    ("concert", 3.0), ("music", 3.0),
    ("theatre", 2.5), ("theater", 2.5), ("arts", 2.5),
]
FALLBACK_DURATION_H = 3.0


# ==========================================================================
# Laden
# ==========================================================================
def to_naive_local(series: pd.Series) -> pd.Series:
    """Datetime ohne Zeitzone in Berliner Ortszeit. Laesst naive Zeiten unveraendert."""
    s = pd.to_datetime(series, errors="coerce")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    if s.dtype == object:  # gemischte Offsets
        s = pd.to_datetime(series, errors="coerce", utc=True)
        return s.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    return s


def load_flows(path: Path):
    df = pd.read_csv(path)
    df["timestamp"] = to_naive_local(df["timestamp"])
    df = (df.dropna(subset=["timestamp"])
            .drop_duplicates("timestamp")
            .set_index("timestamp")
            .sort_index())
    off_grid = (df.index != df.index.floor("15min")).sum()
    if off_grid:
        print(f"  ! {off_grid} Zeitstempel liegen nicht auf dem 15-min-Raster")
    full = pd.date_range(df.index.min().floor("15min"), df.index.max().ceil("15min"), freq="15min")
    df = df.reindex(full)
    station_cols = [c for c in df.columns]
    df[TOTAL_COL] = df[station_cols].sum(axis=1, min_count=1)
    return df.astype(float), station_cols


def load_weather(path: Path) -> pd.DataFrame:
    w = pd.read_csv(path)
    if "timestamp" not in w.columns:  # erste Spalte ist unbenannt (siehe Schema)
        w = w.rename(columns={w.columns[0]: "timestamp"})
    w["timestamp"] = to_naive_local(w["timestamp"])
    return w.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def load_network(stations_path: Path, connections_path: Path):
    st = pd.read_csv(stations_path, dtype={"station_id": str})
    con = pd.read_csv(connections_path, dtype=str)
    G = nx.Graph()
    G.add_nodes_from(st["station_id"])
    G.add_edges_from(zip(con["station_id_1"], con["station_id_2"]))

    name2id = {}
    for sid, name in zip(st["station_id"], st["station_name"]):
        name2id.setdefault(name, sid)
    id2name = dict(zip(st["station_id"], st["station_name"]))
    # Liniennummern als Ziffern ("U6" -> "6"), robust gegen Schreibweisen
    line_digits = {
        sid: {m for m in re.findall(r"\d+", str(lines))}
        for sid, lines in zip(st["station_id"], st["u_bahn_lines"])
    }
    return G, name2id, id2name, line_digits


def default_duration_h(segment, genre) -> float:
    text = f"{segment or ''} {genre or ''}".lower()
    for key, hours in DEFAULT_DURATION_H:
        if key in text:
            return hours
    return FALLBACK_DURATION_H


def load_events(json_path: Path, catchment_radius_m: float, max_event_hours: float,
                hours_before: float, hours_after: float) -> pd.DataFrame:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for e in raw:
        near = e.get("nearest_station")
        cands = ([near] if near else []) + (e.get("alternative_stations") or [])
        catchment = [s["station_name"] for s in cands if s["distance_m"] <= catchment_radius_m]
        rows.append({
            "event_id": e["event_id"],
            "event_name": e.get("event_name"),
            "began_local": e.get("began_local"),
            "estimated_end_local": e.get("estimated_end_local"),
            "segment": e.get("segment"),
            "genre": e.get("genre"),
            "attendance": e.get("estimated_attendance") or 0,
            "nearest_station_id": near["station_id"] if near else None,
            "nearest_station_name": near["station_name"] if near else None,
            "nearest_distance_m": near["distance_m"] if near else None,
            "catchment": catchment,
            "flags": list(e.get("flags") or []),
        })
    ev = pd.DataFrame(rows)
    ev["begin"] = to_naive_local(ev["began_local"])
    ev["end_given"] = to_naive_local(ev["estimated_end_local"])

    ends, sources = [], []
    for r in ev.itertuples():
        flags = r.flags
        if pd.isna(r.begin):
            flags.append("begin_missing")
            ends.append(pd.NaT); sources.append(None)
            continue
        end, src = r.end_given, "given"
        if pd.isna(end) or end <= r.begin:
            end = r.begin + pd.Timedelta(hours=default_duration_h(r.segment, r.genre))
            src = "default_by_segment"
            flags.append("end_assumed")
        if end - r.begin > pd.Timedelta(hours=max_event_hours):
            end = r.begin + pd.Timedelta(hours=max_event_hours)
            flags.append("long_event_clipped")
        if not r.catchment:
            flags.append("no_catchment_station")
        ends.append(end); sources.append(src)
    ev["end"] = pd.to_datetime(pd.Series(ends, index=ev.index))
    ev["end_source"] = sources

    # --- Schritt 3: Zeitfenster auf dem 15-min-Raster ---
    ev["begin_slot"] = ev["begin"].dt.floor("15min")
    ev["end_slot"] = ev["end"].dt.ceil("15min")
    ev["window_start"] = ev["begin_slot"] - pd.Timedelta(hours=hours_before)
    ev["window_end"] = ev["end_slot"] + pd.Timedelta(hours=hours_after)
    return ev


# ==========================================================================
# Closures: Stationen und Dauer aus Freitext
# ==========================================================================
def normalize_text(text: str) -> str:
    t = str(text).lower()
    for a, b in [("ß", "ss"), ("ä", "ae"), ("ö", "oe"), ("ü", "ue")]:
        t = t.replace(a, b)
    t = re.sub(r"[-_./(),;:]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_duration(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    t = str(value).lower().replace(",", ".")
    units = {"d": 24 * 60, "day": 24 * 60, "days": 24 * 60, "tag": 24 * 60, "tage": 24 * 60,
             "h": 60, "hour": 60, "hours": 60, "std": 60,
             "min": 1, "minute": 1, "minutes": 1, "m": 1}
    hits = re.findall(r"(\d+(?:\.\d+)?)\s*(days|day|tage|tag|d|hours|hour|std|h|minutes|minute|min|m)", t)
    if hits:
        return pd.Timedelta(minutes=sum(float(v) * units[u] for v, u in hits))
    try:
        return pd.Timedelta(t)
    except (ValueError, TypeError):
        return None


class ClosureParser:
    """Findet betroffene Stationen in closures.description (Freitext)."""

    def __init__(self, G, name2id, line_digits):
        self.G = G
        self.line_digits = line_digits
        # laengste Namen zuerst, damit "Alt-Mariendorf" vor "Mariendorf" greift
        self.patterns = sorted(
            ((normalize_text(n), sid) for n, sid in name2id.items()),
            key=lambda x: -len(x[0]),
        )

    def _stations_in_order(self, text):
        found = []
        for norm, sid in self.patterns:
            m = re.search(rf"\b{re.escape(norm)}\b", text)
            if m:
                found.append((m.start(), sid))
                text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end():]
        return [sid for _, sid in sorted(found)]

    def affected(self, description):
        """Rueckgabe: (Menge Station-IDs, global_flag)."""
        text = normalize_text(description)
        stations = self._stations_in_order(text)
        lines = set(re.findall(r"\bu ?(\d{1,2})\b", text))

        affected = set()
        if lines and len(stations) >= 2:
            # Abschnitt zwischen zwei genannten Stationen auf der genannten Linie
            for ln in lines:
                sub = self.G.subgraph([s for s, d in self.line_digits.items() if ln in d])
                on_line = [s for s in stations if s in sub]
                if len(on_line) >= 2:
                    try:
                        affected.update(nx.shortest_path(sub, on_line[0], on_line[1]))
                    except nx.NetworkXNoPath:
                        affected.update(on_line)
            if not affected:
                affected.update(stations)
        elif stations:
            affected.update(stations)
        elif lines:
            affected.update(s for s, d in self.line_digits.items() if d & lines)
        else:
            return set(), True  # nichts erkannt -> konservativ: ganzes Netz

        # Umsteigende Fahrgaeste belasten direkte Nachbarn mit
        neighbours = {n for s in affected if s in self.G for n in self.G.neighbors(s)}
        return affected | neighbours, False


def load_closures(path: Path, parser: ClosureParser, id2name, buffer_h: float):
    if not path.exists():
        return pd.DataFrame(columns=["start", "end", "stations", "is_global", "description", "flags"])
    cl = pd.read_csv(path)
    cl["start"] = to_naive_local(cl["when"])
    rows = []
    for r in cl.itertuples():
        dur = parse_duration(r.duration)
        flags = []
        if dur is None:
            dur = pd.Timedelta(hours=4)
            flags.append("duration_unparsed_assumed_4h")
        ids, is_global = parser.affected(r.description)
        if is_global:
            flags.append("stations_unparsed_global")
        rows.append({
            "start": r.start,
            "end": r.start + dur + pd.Timedelta(hours=buffer_h),
            "stations": sorted(id2name[s] for s in ids if s in id2name),
            "is_global": is_global,
            "description": r.description,
            "flags": flags,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# Ausschlussmaske: welche (Zeit, Station)-Zellen sind "nicht sauber"?
# ==========================================================================
def build_exclusion_mask(index, cols, events, closures, min_attendance_global):
    T, S = len(index), len(cols)
    mask = np.zeros((T, S), dtype=bool)
    col_idx = {c: i for i, c in enumerate(cols)}
    total_i = col_idx[TOTAL_COL]
    t0 = index[0]

    def span(a, b):
        i0 = max(0, int((a - t0) / SLOT))
        i1 = min(T, int((b - t0) / SLOT) + 1)
        return slice(i0, i1) if i1 > i0 else None

    for r in events.dropna(subset=["window_start"]).itertuples():
        sl = span(r.window_start, r.window_end)
        if sl is None:
            continue
        if r.attendance >= min_attendance_global:
            mask[sl, :] = True  # grosses Event: ganzes Netz, wie in der Originalidee
        else:
            idx = [col_idx[s] for s in r.catchment if s in col_idx]
            mask[sl, idx] = True

    for r in closures.dropna(subset=["start"]).itertuples():
        sl = span(r.start, r.end)
        if sl is None:
            continue
        if r.is_global:
            mask[sl, :] = True
        else:
            idx = [col_idx[s] for s in r.stations if s in col_idx]
            mask[sl, idx] = True
            mask[sl, total_i] = True  # Umleitungen veraendern auch die Netz-Summe
    return mask


# ==========================================================================
# Baseline-Statistik (vektorisiert ueber Slots x Referenzwochen x Stationen)
# ==========================================================================
def nan_trimmed_mean(V, trim_fraction, robust=True):
    """V: L x K x S. Getrimmter Mittelwert entlang K, NaN werden ignoriert.
    Wie im Original: trimmen erst ab mehr als 3 Werten."""
    n = np.sum(~np.isnan(V), axis=1)
    if not robust:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(V, axis=1), n
    srt = np.sort(V, axis=1)  # NaN landen hinten
    g = np.where(n > 3, np.floor(n * trim_fraction).astype(int), 0)
    pos = np.arange(V.shape[1])[None, :, None]
    keep = (pos >= g[:, None, :]) & (pos < (n - g)[:, None, :])
    total = np.where(keep, srt, 0.0).sum(axis=1)
    count = keep.sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan), n


def spread_stats(V):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(V, axis=1)
        mad = np.nanmedian(np.abs(V - med[:, None, :]), axis=1)
        std = np.nanstd(V, axis=1, ddof=1)
    return med, 1.4826 * mad, std


def compute_event_window(ev, F, M, t0, cols, week_offsets, cfg):
    T = F.shape[0]
    i_start = int((ev.window_start - t0) / SLOT)
    i_end = int((ev.window_end - t0) / SLOT)
    w = np.arange(i_start, i_end + 1)                       # L
    ref = w[:, None] + SLOTS_PER_WEEK * week_offsets[None]  # L x K
    ref_ok = (ref >= 0) & (ref < T)
    ref_c = np.clip(ref, 0, T - 1)

    V_all = F[ref_c].copy()                                 # L x K x S
    V_all[~ref_ok] = np.nan
    V_clean = np.where(M[ref_c], np.nan, V_all)

    base_f, n_f = nan_trimmed_mean(V_clean, cfg.trim_fraction, cfg.robust)
    base_u, n_u = nan_trimmed_mean(V_all, cfg.trim_fraction, cfg.robust)

    use_f = n_f >= cfg.min_days_per_slot
    baseline = np.where(use_f, base_f, base_u)
    baseline = np.where(np.isnan(baseline), base_u, baseline)
    V_used = np.where(use_f[:, None, :], V_clean, V_all)
    med, rstd, std = spread_stats(V_used)

    # Streuung fuer den Z-Score: robuste Std, Untergrenze sqrt(Baseline) gegen
    # Division durch ~0 bei wenigen, fast gleichen Referenzwerten
    scale = np.fmax(np.nan_to_num(rstd), np.sqrt(np.fmax(np.nan_to_num(baseline), 1.0)))

    w_ok = (w >= 0) & (w < T)
    observed = np.full((len(w), F.shape[1]), np.nan)
    observed[w_ok] = F[w[w_ok]]

    anomaly = observed - baseline
    with np.errstate(divide="ignore", invalid="ignore"):
        anomaly_pct = np.where(baseline > 0, 100 * anomaly / baseline, np.nan)
    z = anomaly / scale

    timestamps = t0 + w * SLOT
    return {
        "timestamps": timestamps,
        "observed": observed, "baseline": baseline, "baseline_median": med,
        "baseline_std": std, "baseline_robust_std": rstd,
        "n_ref_days": np.where(use_f, n_f, n_u),
        "baseline_source": np.where(use_f, "filtered", "unfiltered"),
        "anomaly": anomaly, "anomaly_pct": anomaly_pct, "z_score": z,
    }


# ==========================================================================
# Ausgabe zusammensetzen
# ==========================================================================
def to_long(ev, res, cols, hops):
    L, S = res["observed"].shape
    ts = pd.DatetimeIndex(res["timestamps"])
    rel_start = ((ts - ev.begin_slot) / SLOT).astype(int)
    rel_end = ((ts - ev.end_slot) / SLOT).astype(int)
    phase = np.where(ts < ev.begin_slot, "pre", np.where(ts < ev.end_slot, "during", "post"))

    df = pd.DataFrame({
        "event_id": ev.event_id,
        "station_name": np.tile(np.array(cols, dtype=object), L),
        "timestamp": np.repeat(ts.values, S),
        "rel_slot": np.repeat(rel_start, S),
        "rel_slot_to_end": np.repeat(rel_end, S),
        "phase": np.repeat(phase, S),
    })
    for key in ["observed", "baseline", "baseline_median", "baseline_std", "baseline_robust_std",
                "n_ref_days", "baseline_source", "anomaly", "anomaly_pct", "z_score"]:
        df[key] = res[key].reshape(-1)
    df["in_catchment"] = df["station_name"].isin(ev.catchment)
    df["hop_from_venue"] = df["station_name"].map(hops).astype("Int64")
    return df


def write_table(df, path_parquet: Path):
    try:
        df.to_parquet(path_parquet, index=False)
        return path_parquet
    except ImportError:
        path_csv = path_parquet.with_suffix(".csv.gz")
        df.to_csv(path_csv, index=False)
        return path_csv


def overlap_info(events, closures):
    """Flags fuer Schritt 7: andere Events / Closures im selben Fenster und Einzugsgebiet."""
    info = {}
    valid = events.dropna(subset=["window_start"])
    ws, we = valid["window_start"].values, valid["window_end"].values
    for r in valid.itertuples():
        time_ov = (ws <= np.datetime64(r.window_end)) & (we >= np.datetime64(r.window_start))
        others = []
        for o in valid[time_ov].itertuples():
            if o.event_id != r.event_id and set(o.catchment) & set(r.catchment):
                others.append(o.event_id)
        cls = []
        for c in closures.itertuples():
            if pd.isna(c.start):
                continue
            if c.start <= r.window_end and c.end >= r.window_start:
                if c.is_global or set(c.stations) & set(r.catchment):
                    cls.append(str(c.description))
        info[r.event_id] = {"overlapping_events": others, "overlapping_closures": cls}
    return info


# ==========================================================================
# Main
# ==========================================================================
def main():
    p = argparse.ArgumentParser(description="Schritt 3+4: Event-Fenster und Baseline")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--events-json", type=Path, default=Path("data/processed/event_nearest_station.json"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--hours-before", type=float, default=3.0)
    p.add_argument("--hours-after", type=float, default=3.0)
    p.add_argument("--weeks-window", type=int, default=6, help="Referenzwochen je Richtung")
    p.add_argument("--min-days-per-slot", type=int, default=5)
    p.add_argument("--trim-fraction", type=float, default=0.2)
    p.add_argument("--no-robust", dest="robust", action="store_false")
    p.add_argument("--catchment-radius-m", type=float, default=1000.0)
    p.add_argument("--min-attendance-global", type=int, default=20000)
    p.add_argument("--max-event-hours", type=float, default=12.0)
    p.add_argument("--closure-buffer-h", type=float, default=1.0)
    cfg = p.parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("Lade Daten ...")
    flows, station_cols = load_flows(cfg.data_dir / "flows_pre_innotrans.csv")
    cols = station_cols + [TOTAL_COL]
    weather = load_weather(cfg.data_dir / "weather_data_pre_innotrans.csv")
    G, name2id, id2name, line_digits = load_network(
        cfg.data_dir / "stations_with_ubahn.csv", cfg.data_dir / "berlin_ubahn_connections.csv")
    events = load_events(cfg.events_json, cfg.catchment_radius_m, cfg.max_event_hours,
                         cfg.hours_before, cfg.hours_after)
    closures = load_closures(cfg.data_dir / "closures_pre_inootrans.csv", ClosureParser(G, name2id, line_digits),
                             id2name, cfg.closure_buffer_h)
    print(f"  {len(flows)} Slots, {len(station_cols)} Stationen, {len(events)} Events, "
          f"{len(closures)} Closures")
    if len(closures):
        print(f"  Closures ohne erkannte Stationen (netzweit maskiert): {int(closures['is_global'].sum())}")

    missing = [s for e in events["catchment"] for s in e if s not in flows.columns]
    if missing:
        print(f"  ! {len(set(missing))} Einzugsstationen fehlen in flows.csv, z. B. {sorted(set(missing))[:3]}")

    F = flows[cols].to_numpy(dtype=float)
    M = build_exclusion_mask(flows.index, cols, events, closures, cfg.min_attendance_global)
    print(f"  Anteil ausgeschlossener Zellen: {M[:, :-1].mean():.1%} (Stationen), "
          f"{M[:, -1].mean():.1%} (Netz-Summe)")

    t0 = flows.index[0]
    week_offsets = np.array([k for k in range(-cfg.weeks_window, cfg.weeks_window + 1) if k != 0])
    hop_cache = {}
    overlaps = overlap_info(events, closures)

    station_parts, total_parts, meta = [], [], []
    for ev in events.itertuples():
        m = {
            "event_id": ev.event_id, "event_name": ev.event_name,
            "attendance": int(ev.attendance),
            "nearest_station": ev.nearest_station_name,
            "catchment": ev.catchment,
            "flags": list(ev.flags),
        }
        if pd.isna(ev.window_start):
            m["skipped"] = "no_begin"
            meta.append(m); continue
        if ev.window_end < flows.index[0] or ev.window_start > flows.index[-1]:
            m["skipped"] = "outside_flow_period"
            meta.append(m); continue

        if ev.nearest_station_id and ev.nearest_station_id not in hop_cache and ev.nearest_station_id in G:
            lengths = nx.single_source_shortest_path_length(G, ev.nearest_station_id)
            hop_cache[ev.nearest_station_id] = {id2name[k]: v for k, v in lengths.items()}
        hops = hop_cache.get(ev.nearest_station_id, {})

        res = compute_event_window(ev, F, M, t0, cols, week_offsets, cfg)
        df = to_long(ev, res, cols, hops)

        total = df[df["station_name"] == TOTAL_COL].drop(
            columns=["station_name", "in_catchment", "hop_from_venue"])
        total = total.merge(weather, left_on="timestamp", right_index=True, how="left")
        total_parts.append(total)
        station_parts.append(df[df["station_name"] != TOTAL_COL])

        stn = df[df["station_name"] != TOTAL_COL]
        m.update({
            "begin": str(ev.begin), "end": str(ev.end), "end_source": ev.end_source,
            "window_start": str(ev.window_start), "window_end": str(ev.window_end),
            "n_slots": int(len(res["timestamps"])),
            "share_filtered_baseline": round(float((stn["baseline_source"] == "filtered").mean()), 3),
            **overlaps.get(ev.event_id, {}),
        })
        meta.append(m)

    stations_out = pd.concat(station_parts, ignore_index=True) if station_parts else pd.DataFrame()
    total_out = pd.concat(total_parts, ignore_index=True) if total_parts else pd.DataFrame()
    p1 = write_table(stations_out, cfg.out_dir / "event_windows_stations.parquet")
    p2 = write_table(total_out, cfg.out_dir / "event_windows_total.parquet")
    p3 = cfg.out_dir / "event_windows_meta.json"
    p3.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    done = sum(1 for m in meta if "skipped" not in m)
    print(f"Fertig: {done}/{len(meta)} Events verarbeitet")
    print(f"  {p1}  ({len(stations_out):,} Zeilen)")
    print(f"  {p2}  ({len(total_out):,} Zeilen)")
    print(f"  {p3}")


if __name__ == "__main__":
    main()