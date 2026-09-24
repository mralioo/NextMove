"""
Events and closures -> "episodes": a time window plus the stations it affects directly.

An episode is a plain dict:
    id          'ev_12' / 'cl_3'
    kind        'event' | 'closure'
    type        event segment (e.g. 'Music') or 'closure'
    name        event name / closure description
    start, end  naive local timestamps (Europe/Berlin)
    anchors     stations directly affected (venue station, or all closed stations)
    attendance  estimated visitors (0 for closures)
    how         how the anchors were found (for transparency / debugging)

Events are mapped to a station in this order:
    1) manual override (config.VENUE_TO_STATION), 2) geocoding + nearest station within
    config.MAX_VENUE_DISTANCE_M, 3) a station name appearing in venue/address text.
Closures:
    'Station X closed ...'                     -> [X]
    'Line U3 suspended ... between A and B'    -> every station on the path A..B on line U3
                                                  (not only the two end points)
"""
import re
from collections import Counter, deque

import pandas as pd

from . import config
from .geo import Geocoder, StationIndex
from .names import StationNameIndex, normalize_text


# =========================================================================== network helpers
def hops_from(adj: dict, anchors, max_hop: int) -> dict:
    """Breadth-first search: {station: hop distance} for all stations within max_hop."""
    dist = {a: 0 for a in anchors}
    queue = deque(anchors)
    while queue:
        u = queue.popleft()
        if dist[u] >= max_hop:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def shortest_path(adj: dict, a: str, b: str, allowed=None):
    """Shortest path a -> b (list of stations incl. both ends), optionally only through
    stations in `allowed`. Returns None if there is no such path."""
    prev, queue = {a: None}, deque([a])
    while queue:
        u = queue.popleft()
        if u == b:
            path = []
            while u is not None:
                path.append(u)
                u = prev[u]
            return path[::-1]
        for v in adj.get(u, ()):
            if v not in prev and (allowed is None or v in allowed or v == b):
                prev[v] = u
                queue.append(v)
    return None


# =========================================================================== events
def _to_local(series: pd.Series) -> pd.Series:
    """ISO timestamps with offset (e.g. +02:00) -> naive Europe/Berlin local time."""
    return pd.to_datetime(series, utc=True).dt.tz_convert("Europe/Berlin").dt.tz_localize(None)


def event_episodes(events: pd.DataFrame, stations: pd.DataFrame, station_names, geocoder: Geocoder,
                   min_attendance=config.MIN_EVENT_ATTENDANCE):
    """Large events -> episodes anchored at the venue's station. Returns (episodes, unmapped)."""
    names = StationNameIndex(station_names)
    index = StationIndex(stations[stations.station_name.isin(set(station_names))])

    # Validate the manual overrides once, so a typo shows up immediately.
    overrides = {}
    for key, station in config.VENUE_TO_STATION.items():
        if station in names.key_to_name.values():
            overrides[normalize_text(key)] = station
        else:
            print(f"[warn] VENUE_TO_STATION: '{station}' has no flow data / is not in the network")

    ev = events.copy()
    ev["start"] = _to_local(ev.began_local)
    ev["end"] = _to_local(ev.estimated_end_local)
    bad_end = ev.end.isna() | (ev.end <= ev.start)
    ev.loc[bad_end, "end"] = ev.loc[bad_end, "start"] + pd.Timedelta(hours=config.DEFAULT_EVENT_HOURS)
    ev["estimated_attendance"] = ev.estimated_attendance.fillna(0)
    ev = ev[ev.estimated_attendance >= min_attendance]

    episodes, unmapped, how_count = [], [], Counter()
    for i, r in ev.iterrows():
        venue = r.venue_name if pd.notna(r.venue_name) else ""
        address = r.address if pd.notna(r.address) else ""
        text = normalize_text(f"{venue} | {address}")

        station, how = None, None
        # 1) manual override
        station = next((s for k, s in overrides.items() if k in text), None)
        if station:
            how = "override"
        # 2) geocoding -> nearest station within walking distance
        if station is None:
            loc = geocoder.locate(address=address, venue=venue)
            if loc:
                near = index.nearest(*loc, k=1, max_distance_m=config.MAX_VENUE_DISTANCE_M)
                if near:
                    station, how = near[0]["station_name"], "geocode"
        # 3) station name mentioned in venue / address (e.g. 'Platz der Luftbrücke')
        if station is None:
            hits = names.find_in_text(text)
            if hits:
                station, how = hits[0], "name_in_address"

        if station is None:
            unmapped.append(dict(attendance=int(r.estimated_attendance), venue=venue, address=address))
            continue
        how_count[how] += 1
        episodes.append(dict(id=f"ev_{i}", kind="event", type=str(r.segment), name=r.event_name,
                             start=r.start, end=r.end, anchors=[station],
                             attendance=float(r.estimated_attendance), how=how))

    geocoder.save()
    print(f"[info] events >= {min_attendance} visitors: {len(ev)}, mapped {dict(how_count)}, "
          f"without station: {len(unmapped)}")
    for u in sorted(unmapped, key=lambda x: -x["attendance"]):
        print(f"       no station: {u['attendance']:>6}  {u['venue'] or '-'} | {u['address']}")
    return episodes, unmapped


# =========================================================================== closures
_DURATION = re.compile(r"\s*(?:(\d+)\s*d)?\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*min)?\s*")
_SECTION = re.compile(r"line\s+(u\d+).*?between\s+(.+?)\s+and\s+(.+?)(?:\s+due to\b|\.|$)", re.I)
_STATION = re.compile(r"station\s+(.+?)\s+closed", re.I)


def parse_duration(text):
    """'2h30min' / '1h' / '45min' / '1d' -> Timedelta, or None if unparseable."""
    m = _DURATION.fullmatch(str(text))
    if not m or not any(m.groups()):
        return None
    d, h, mi = (int(x) if x else 0 for x in m.groups())
    return pd.Timedelta(days=d, hours=h, minutes=mi)


def closure_stations(description: str, names: StationNameIndex, adj: dict, stations: pd.DataFrame):
    """Closure description -> (list of closed stations, how)."""
    m = _SECTION.search(description)
    if m:
        line, a_txt, b_txt = m.group(1).upper(), m.group(2), m.group(3)
        a, b = names.resolve(a_txt), names.resolve(b_txt)
        if a and b:
            # Only walk through stations served by the suspended line.
            on_line = set(stations.station_name[stations.u_bahn_lines.fillna("").str.split(",")
                                                .apply(lambda ls: line in [x.strip() for x in ls])])
            path = shortest_path(adj, a, b, allowed=on_line) or shortest_path(adj, a, b)
            if path:
                return path, f"section {line} ({len(path)} stations)"
            return [a, b], "section end points only (no path found)"
    m = _STATION.search(description)
    if m:
        s = names.resolve(m.group(1))
        if s:
            return [s], "single station"
    # Fallback: any station names in the text.
    hits = names.find_in_text(description)
    return hits, "names in text" if hits else "unresolved"


def closure_episodes(closures: pd.DataFrame, stations: pd.DataFrame, station_names, adj: dict):
    names = StationNameIndex(station_names)
    episodes, skipped = [], []
    for i, r in closures.iterrows():
        duration = parse_duration(r.duration)
        affected, how = closure_stations(str(r.description), names, adj, stations)
        affected = [s for s in affected if s in names.key_to_name.values()]
        if duration is None or not affected:
            skipped.append(r.description)
            continue
        episodes.append(dict(id=f"cl_{i}", kind="closure", type="closure", name=r.description,
                             start=r["when"], end=r["when"] + duration, anchors=affected,
                             attendance=0.0, how=how))
    print(f"[info] closures: {len(episodes)} mapped, {len(skipped)} skipped")
    for s in skipped:
        print(f"       skipped: {s}")
    return episodes


# =========================================================================== all together
def build_episodes(events, closures, stations, station_names, adj, geocoder):
    ev_eps, _ = event_episodes(events, stations, station_names, geocoder)
    cl_eps = closure_episodes(closures, stations, station_names, adj)
    return ev_eps + cl_eps


def episodes_to_frame(episodes) -> pd.DataFrame:
    """Flat table for export (anchors joined with '; ')."""
    rows = [{**e, "anchors": "; ".join(e["anchors"])} for e in episodes]
    return pd.DataFrame(rows, columns=["id", "kind", "type", "name", "start", "end",
                                       "anchors", "attendance", "how"])
