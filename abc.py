"""
event_effects.py - rudimentaere Effektschaetzung fuer Events & Sperrungen

Ansatz 2: Regression mit Event-Indikatoren (Lead/Lag-Bins, Distanz in Hops)
Ansatz 5: Hierarchische Schrumpfung (Event-Effekt = Typ-Effekt + Abweichung des Einzelevents)

Pipeline
  1) Baseline pro Station:  log1p(flow) ~ Viertelstunde x Tagtyp + Ferien + Wetter + Trend
     (gefittet NUR auf Zeiten ohne Event/Sperrung -> Effekte verfaelschen die Baseline nicht)
  2) Residuen = log-Ist - log-Baseline
  3) Residuen ~ sum_{Typ,Bin,Hop} beta * Indikator  +  sum_{Event,Bin} u_e * Indikator (nur Hop 0)
     mit Ridge auf u_e; die Staerke der Schrumpfung (tau^2) wird per EM aus den Daten geschaetzt.
  Ergebnis: Effekt in % ueber Baseline pro (Typ, Zeitfenster relativ zum Event, Hops vom Venue-Bahnhof)

Aufruf
  python event_effects.py --synthetic            # Selbsttest mit erfundenen Daten (bekannte Wahrheit)
  python event_effects.py --data ./data          # echte Hackathon-CSVs
  python event_effects.py --data ./data --placebo
  python event_effects.py --data ./data --geocode-cache analysis_output/geocode_cache.json

Event -> Station wird in dieser Reihenfolge versucht:
  1) --venue-map CSV (venue_name, address, station_name)
  2) --geocode-cache: fertige Koordinaten, naechste Station per Luftlinie (<= MAX_DISTANCE_M)
  3) VENUE_TO_STATION (Namen im Skript)
  4) Stationsname kommt im Venue-Namen oder in der Adresse vor
"""
import argparse
import json
import os
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ----------------------------------------------------------------------------- Konfiguration
MIN_ATTENDANCE = 2000            # kleinere Events ignorieren
TYPE_COL = "segment"             # Spalte in den Events, die den Event-Typ definiert (Alternative: "genre")
MIN_EVENTS_FOR_ATTENDANCE = 15   # Attendance-Steigung erst ab so vielen Events des Typs
MAX_HOP = 2                      # Einflussradius im Netz (Anzahl Stationen vom Venue-Bahnhof)
DEFAULT_EVENT_HOURS = 3          # Dauer, falls estimated_end_local fehlt
MAX_DISTANCE_M = 1500            # weiter entfernte Venues gelten als "keine Station in der Naehe"
EARTH_RADIUS_M = 6_371_000
SCHOOL_HOLIDAYS = [("2026-07-09", "2026-08-22")]   # Berlin Sommerferien - BITTE PRUEFEN

# Venue -> U-Bahn-Station (Namen muessen zu stations_with_ubahn.station_name passen).
# Nur Beispiele! Nach dem ersten Lauf werden nicht zugeordnete Venues ausgegeben -> hier ergaenzen.
VENUE_TO_STATION = {
    "uber arena": "Warschauer Straße",
    "mercedes-benz arena": "Warschauer Straße",
    "olympiastadion": "Olympia-Stadion",
    "columbiahalle": "Platz der Luftbrücke",
    "uber-platz": "Warschauer Straße",
    "olympischer platz": "Olympia-Stadion",
    "admiralspalast": "Friedrichstraße",
    "tempodrom": "Möckernbrücke",
}

H = pd.Timedelta(hours=1)
BIN_NAMES = ["pre3", "pre2", "pre1", "during", "post1", "post2"]


def bin_bounds(ep):
    s, e = ep["start"], ep["end"]
    return [(s - 3 * H, s - 2 * H), (s - 2 * H, s - H), (s - H, s), (s, e), (e, e + H), (e + H, e + 2 * H)]


def ep_segments(ep, idx):
    """Liste (bin_index, i0, i1) mit Zeilenbereichen [i0, i1) im Zeitindex."""
    out = []
    for k, (lo, hi) in enumerate(bin_bounds(ep)):
        i0, i1 = idx.searchsorted(lo), idx.searchsorted(hi)
        if i1 > i0:
            out.append((k, i0, i1))
    return out


# ----------------------------------------------------------------------------- Netz
def bfs_hops(adj, anchors, max_hop):
    dist = {a: 0 for a in anchors}
    q = deque(anchors)
    while q:
        u = q.popleft()
        if dist[u] >= max_hop:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower().replace("ß", "ss")).strip()


def text_key(t):
    """Normalisierter Freitext (Adresse, Beschreibung): klein, ss statt ß, 'str.' -> 'strasse'."""
    return re.sub(r"str\.", "strasse", norm(t))


def station_key(n):
    """'S+U Warschauer Str. (Berlin)' -> 'warschauer strasse'  (Praefix, '(Berlin)' und Abkuerzungen weg)."""
    k = text_key(n)
    k = re.sub(r"\s*\(berlin\)\s*$", "", k)
    k = re.sub(r"^(s\+u|s|u)\s+", "", k)
    return re.sub(r"\bstr\b", "strasse", k).strip()


def find_stations(text, keys):
    """Alle Stations-Keys (laenger als 3 Zeichen), die als ganzes Wort in text vorkommen, laengste zuerst."""
    hits = [k for k in keys if len(k) > 3 and re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", text)]
    return sorted(hits, key=len, reverse=True)


def haversine_m(lat1, lon1, lat2, lon2):
    """Distanz in Metern von einem Punkt zu Arrays von Punkten (vektorisiert)."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(np.asarray(lat2, float)), np.radians(np.asarray(lon2, float))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def cache_key(venue, address):
    """Schluessel wie in event_station_mapping.py: 'venue | address', NaN/leer -> ''."""
    def c(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return str(x).strip()
    return f"{c(venue)} | {c(address)}"


def extract_latlon(v):
    """Koordinaten aus einem Cache-Eintrag ziehen; akzeptiert die ueblichen Schreibweisen."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return float(v[0]), float(v[1])
        except (TypeError, ValueError):
            return None
    if not isinstance(v, dict):
        return None
    for sub in ("event_location", "location", "coords", "coordinates", "geometry",
                "point", "result", "nearest_station", "station"):
        if isinstance(v.get(sub), (dict, list, tuple)):
            hit = extract_latlon(v[sub])
            if hit:
                return hit
    lat_keys = ("lat", "latitude", "y", "lat_deg")
    lon_keys = ("lon", "lng", "long", "longitude", "x", "lon_deg")
    lat = next((v[k] for k in lat_keys if v.get(k) is not None), None)
    lon = next((v[k] for k in lon_keys if v.get(k) is not None), None)
    if lat is None or lon is None:
        return None
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def key_variants(venue, address):
    """Normalisierte Schluessel-Varianten: klein, ohne Leerzeichen um '|', Adresse allein."""
    def c(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return re.sub(r"\s+", " ", str(x)).strip().lower()
    v, a = c(venue), c(address)
    out = [f"{v}|{a}"]
    if a:
        out.append(f"|{a}")       # Eintrag ohne Venue-Namen
    if v:
        out.append(f"{v}|")
    return out


def load_geocode_cache(path):
    """Geocode-Cache laden -> {'venue | address': {'lat':.., 'lon':..}}.

    Akzeptiert ein Dict (Schluessel = Ort) oder eine Liste von Eintraegen mit
    venue_name/address, und die ueblichen Koordinaten-Schreibweisen.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):      # Liste von Event-/Ort-Eintraegen
        items = [(cache_key(e.get("venue_name"), e.get("address")), e)
                 for e in raw if isinstance(e, dict)]
    else:
        items = list(raw.items())

    cache, bad, n_ok = {}, [], 0
    for k, v in items:
        hit = extract_latlon(v)
        if hit and not (np.isnan(hit[0]) or np.isnan(hit[1])):
            parts = str(k).split("|", 1)
            nk = key_variants(parts[0], parts[1] if len(parts) > 1 else "")[0]
            cache[nk] = {"lat": hit[0], "lon": hit[1]}
            n_ok += 1
            if len(parts) > 1 and parts[1].strip():   # zusaetzlich nur ueber die Adresse auffindbar
                addr_only = key_variants("", parts[1])[0]
                cache.setdefault(addr_only, {"lat": hit[0], "lon": hit[1]})
        else:
            bad.append((k, v))
    print(f"[info] Geocode-Cache: {n_ok} von {len(items)} Orten mit Koordinaten ({path})")
    if not cache and bad:
        k, v = bad[0]
        print(f"[warn] Keine Koordinaten erkannt. Erster Eintrag zur Diagnose:\n"
              f"       Schluessel: {k!r}\n       Wert:       {json.dumps(v, ensure_ascii=False)[:300]}")
    elif bad:
        print(f"[info] {len(bad)} Cache-Eintraege ohne Koordinaten (Geocoding fehlgeschlagen), Beispiel: {bad[0][0]!r}")
    return cache


def nearest_station_lookup(stations_df, station_names):
    """Gibt f(lat, lon) -> (station_name, distanz_m) zurueck; nur Stationen mit Flow-Daten."""
    keep = set(station_names)
    st = stations_df[stations_df.station_name.isin(keep)].dropna(subset=["latitude", "longitude"])
    if st.empty:
        raise SystemExit("Keine Station mit Koordinaten und Flow-Daten - stations_with_ubahn.csv pruefen.")
    names = st.station_name.to_numpy()
    lat, lon = st.latitude.to_numpy(float), st.longitude.to_numpy(float)

    def f(plat, plon):
        d = haversine_m(plat, plon, lat, lon)
        j = int(np.argmin(d))
        return names[j], float(d[j])
    return f


def parse_duration(s):
    m = re.fullmatch(r"\s*(?:(\d+)\s*d)?\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*min)?\s*", str(s))
    if not m or not any(m.groups()):
        return None
    d, h, mi = [int(x) if x else 0 for x in m.groups()]
    return pd.Timedelta(days=d, hours=h, minutes=mi)


# ----------------------------------------------------------------------------- Daten laden (echt)
def to_local(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert("Europe/Berlin").dt.tz_localize(None)


def pick(d, stem):
    """Findet die CSV, die mit stem beginnt - faengt Varianten wie flows_pre_innotrans.csv ab."""
    hits = sorted(f for f in os.listdir(d) if f.startswith(stem) and f.endswith(".csv"))
    if not hits:
        raise SystemExit(f"Keine Datei '{stem}*.csv' in {d}")
    if len(hits) > 1:
        print(f"[warn] mehrere Dateien fuer '{stem}': {hits} -> nehme {hits[0]}")
    return os.path.join(d, hits[0])


def load_real(d, venue_map_path=None, geocode_cache_path=None):
    st = pd.read_csv(pick(d, "stations_with_ubahn"))
    con = pd.read_csv(pick(d, "berlin_ubahn_connections"))
    flows = pd.read_csv(pick(d, "flows"), parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    weather = pd.read_csv(pick(d, "weather_data"), index_col=0, parse_dates=True)
    ev = pd.read_csv(pick(d, "berlin_events_summer_2026"))
    cl = pd.read_csv(pick(d, "closures"), parse_dates=["when"])

    id2name = dict(zip(st.station_id, st.station_name))
    adj = defaultdict(set)
    for a, b in zip(con.station_id_1, con.station_id_2):
        if a in id2name and b in id2name:
            adj[id2name[a]].add(id2name[b])
            adj[id2name[b]].add(id2name[a])
    flows = flows[[c for c in flows.columns if c in set(st.station_name)]]
    vm = pd.read_csv(venue_map_path) if venue_map_path else None
    geo = load_geocode_cache(geocode_cache_path) if geocode_cache_path else None
    episodes = episodes_from_real(ev, cl, list(flows.columns), vm, geo, st)
    return flows, weather, adj, episodes


def episodes_from_real(ev, cl, station_names, venue_map=None, geo_cache=None, stations_df=None):
    key2name = {station_key(n): n for n in station_names}
    keys = list(key2name)
    mapping = {}
    for k, v in VENUE_TO_STATION.items():
        if station_key(v) in key2name:
            mapping[text_key(k)] = key2name[station_key(v)]
        else:
            print(f"[warn] VENUE_TO_STATION: '{v}' nicht im Netz (Stationsnamen pruefen)")
    pair_map = {}
    if venue_map is not None:   # optionale CSV: venue_name, address, station_name
        for _, r in venue_map.iterrows():
            sk = station_key(r["station_name"])
            if sk in key2name:
                v = text_key(r["venue_name"]) if pd.notna(r.get("venue_name")) else ""
                a = text_key(r["address"]) if pd.notna(r.get("address")) else ""
                pair_map[(v, a)] = key2name[sk]
    nearest = nearest_station_lookup(stations_df, station_names) if geo_cache and stations_df is not None else None
    if geo_cache and nearest is None:
        print("[warn] Geocode-Cache ohne stations_with_ubahn.csv uebergeben - wird ignoriert.")
    episodes, unmapped, far = [], [], []

    ev = ev.copy()
    ev["start"] = to_local(ev.began_local)
    ev["end"] = to_local(ev.estimated_end_local)
    bad = ev.end.isna() | (ev.end <= ev.start)
    ev.loc[bad, "end"] = ev.loc[bad, "start"] + pd.Timedelta(hours=DEFAULT_EVENT_HOURS)
    ev["estimated_attendance"] = ev.estimated_attendance.fillna(0)
    ev = ev[ev.estimated_attendance >= MIN_ATTENDANCE]

    how = defaultdict(int)
    for i, r in ev.iterrows():
        v = text_key(r.venue_name) if pd.notna(r.venue_name) else ""
        ad = text_key(r.address) if pd.notna(r.address) else ""
        station = pair_map.get((v, ad))
        if station:
            how["venue_map"] += 1
        if station is None and nearest is not None:      # 1. Wahl: vorberechnete Koordinaten
            g = next((geo_cache[k] for k in key_variants(r.venue_name, r.address) if k in geo_cache), None)
            if g:
                cand, dist = nearest(g["lat"], g["lon"])
                if dist <= MAX_DISTANCE_M:
                    station, how["geocode"] = cand, how["geocode"] + 1
                else:
                    far.append((dist, r.venue_name, r.address, cand))
        if station is None:
            station = next((st for k, st in mapping.items() if k in v or k in ad), None)
            how["VENUE_TO_STATION"] += station is not None
        if station is None:      # Fallback: Stationsname steht im Venue-Namen oder in der Adresse
            hits = find_stations(v + " | " + ad, keys)
            station = key2name[hits[0]] if hits else None
            how["Name in Adresse"] += station is not None
        if station is None:
            unmapped.append((r.estimated_attendance, r.venue_name, r.address))
            continue
        episodes.append(dict(id=f"ev_{i}", type=str(r[TYPE_COL]), start=r.start, end=r.end,
                             anchors=[station], name=r.event_name, attendance=r.estimated_attendance))
    print(f"[info] Events {len(ev)} (>= {MIN_ATTENDANCE} Besucher), zugeordnet: {dict(how)}, ohne Station: {len(unmapped)}")
    if far:
        uniq = {(v, a): (d, c) for d, v, a, c in sorted(far)}
        print(f"[warn] {len(far)} Events mit Koordinaten, aber naechste Station > {MAX_DISTANCE_M} m "
              f"({len(uniq)} Orte) - Netz ist partiell, Beispiele:")
        for (v, a), (d, c) in list(uniq.items())[:5]:
            print(f"   {d:6.0f} m  {v} | {a}  -> {c}")
    if unmapped:
        print("[warn] Top nicht zugeordnete Venues (nach Besucherzahl) -> VENUE_TO_STATION oder --venue-map ergaenzen:")
        seen = set()
        for att, v, a in sorted(unmapped, key=lambda x: -x[0]):
            if (v, a) not in seen and len(seen) < 15:
                seen.add((v, a))
                print(f"   {int(att):>7}  {v} | {a}")

    # Sperrungen: Stationsnamen aus der Beschreibung ziehen
    skipped = []
    for i, r in cl.iterrows():
        dur = parse_duration(r.duration)
        hits = find_stations(text_key(r.description), keys)
        if dur is None or not hits:
            skipped.append((r["duration"], r.description, "Dauer?" if dur is None else "keine Station"))
            continue
        episodes.append(dict(id=f"cl_{i}", type="closure", start=r["when"], end=r["when"] + dur,
                             anchors=[key2name[h] for h in hits], name=r.description, attendance=0))
    if skipped:
        print(f"[warn] {len(skipped)} von {len(cl)} Sperrungen uebersprungen, Beispiele:")
        for d, t, why in skipped[:5]:
            print(f"   [{why}] duration={d!r}  description={str(t)[:110]!r}")
    print(f"[info] Episoden: {sum(e['type'] != 'closure' for e in episodes)} Events, "
          f"{sum(e['type'] == 'closure' for e in episodes)} Sperrungen")
    return episodes


# ----------------------------------------------------------------------------- Synthetische Daten
# wahre log-Effekte pro Typ und Bin, je Hop 0/1/2
TRUE_ATT_SLOPE = 0.5      # wahre Abhaengigkeit von log(attendance) im synthetischen Test
TRUE = {
    "Sports":     dict(pre1=(.35, .12, .04), during=(.02, .01, 0), post1=(.9, .3, .1), post2=(.3, .1, .03)),
    "Music":      dict(pre1=(.25, .08, .02), post1=(.7, .25, .08), post2=(.25, .08, .02)),
    "Conference": dict(pre1=(.10, .03, 0), post1=(.15, .05, 0)),
    "closure":    dict(during=(np.log(.1), .25, .08)),
}


def make_synthetic(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-10", "2026-09-21 23:45", freq="15min")
    n = len(idx)
    names = [f"S{i:02d}" for i in range(24)]
    adj = defaultdict(set)

    def link(a, b):
        adj[a].add(b)
        adj[b].add(a)
    for i in range(11):
        link(names[i], names[i + 1])
        link(names[12 + i], names[13 + i])
    link("S05", "S17")

    # Wetter
    hod = idx.hour + idx.minute / 60
    day = np.arange(n) / 96
    prcp = np.zeros(n)
    for s0 in rng.integers(0, n - 30, 70):
        prcp[s0:s0 + rng.integers(8, 24)] = rng.uniform(.5, 4)
    weather = pd.DataFrame({"temp": 20 + 5 * np.sin(2 * np.pi * (hod - 9) / 24) + 3 * np.sin(2 * np.pi * day / 40),
                            "prcp": prcp, "wspd": 10 + rng.gamma(2, 2, n)}, index=idx)

    # Baseline
    dow = idx.dayofweek.values
    wk = dow < 5
    curve = np.where(wk,
                     .2 + np.exp(-(hod - 8) ** 2 / (2 * 1.2 ** 2)) + 1.1 * np.exp(-(hod - 17.5) ** 2 / (2 * 1.5 ** 2))
                     + .4 * np.exp(-(hod - 13) ** 2 / (2 * 3 ** 2)),
                     .15 + .6 * np.exp(-(hod - 15) ** 2 / (2 * 4 ** 2)))
    hol = (idx >= "2026-07-09") & (idx < "2026-08-23")
    scale = rng.lognormal(6.5, .6, len(names))
    L = (np.log(scale)[None, :] + np.log(curve)[:, None] - .05 * (prcp > .3)[:, None]
         - .15 * (hol & wk)[:, None] + rng.normal(0, .03, (n // 96 + 1, 1))[(np.arange(n) // 96)]
         + rng.normal(0, .07, (n, len(names))))

    # Episoden
    episodes = []
    spec = [("Sports", 15, ["S03", "S15"], 2.5), ("Music", 20, ["S08", "S20"], 3), ("Conference", 8, ["S10"], 6)]
    for typ, cnt, venues, dur in spec:
        for j in range(cnt):
            d = pd.Timestamp("2026-06-12") + pd.Timedelta(days=int(rng.integers(0, 95)))
            hr = 9 if typ == "Conference" else int(rng.choice([15, 19, 20]))
            st = d + pd.Timedelta(hours=hr, minutes=int(rng.choice([0, 30])))
            episodes.append(dict(id=f"ev_{typ}_{j}", type=typ, start=st, end=st + pd.Timedelta(hours=dur),
                                 anchors=[str(rng.choice(venues))], name=f"{typ} {j}",
                                 attendance=int(rng.integers(5000, 60000))))
    for j in range(12):
        st = pd.Timestamp("2026-06-12") + pd.Timedelta(days=int(rng.integers(0, 95)), hours=int(rng.integers(6, 20)))
        episodes.append(dict(id=f"cl_{j}", type="closure", start=st, end=st + pd.Timedelta(hours=int(rng.integers(2, 8))),
                             anchors=[str(rng.choice(names))], name="closure", attendance=0))

    # wahre Effekte einbauen (additiv im Log -> ueberlappende Episoden addieren sich)
    sidx = {s: i for i, s in enumerate(names)}
    for ep in episodes:
        # Effekt skaliert mit der Eventgroesse: TRUE_ATT_SLOPE pro log-Einheit, plus Streuung
        scale_e = 1.0
        if ep["type"] != "closure":
            z = np.log(ep["attendance"]) - np.log(20000)
            scale_e = 1 + TRUE_ATT_SLOPE * z + .15 * rng.normal()
        hops = bfs_hops(adj, ep["anchors"], MAX_HOP)
        for k, i0, i1 in ep_segments(ep, idx):
            eff = TRUE[ep["type"]].get(BIN_NAMES[k])
            if eff is None:
                continue
            for s, h in hops.items():
                L[i0:i1, sidx[s]] += eff[h] * scale_e
    flows = pd.DataFrame(np.round(np.exp(L)).astype(int), index=idx, columns=names)
    return flows, weather, adj, episodes


def aggregate(flows, weather, freq):
    """Flows aufsummieren, Wetter mitteln. Glaettet das Rauschen einzelner Viertelstunden."""
    f = flows.resample(freq).sum(min_count=1)
    w = weather.resample(freq).mean()
    print(f"[info] Aggregation auf {freq}: {len(flows)} -> {len(f)} Zeilen")
    return f, w


# ----------------------------------------------------------------------------- Schritt 1: Baseline
def baseline_design(idx, weather):
    """Designmatrix der Baseline. Gibt (X, n_level) zurueck; n_level = Anzahl der Level-Dummies."""
    n = len(idx)
    tod = idx.hour * 60 + idx.minute                 # Tageszeit-Slot, Raster-unabhaengig
    slots = np.unique(tod)
    slot = np.searchsorted(slots, tod)
    dow = idx.dayofweek
    dtype = np.select([dow <= 3, dow == 4, dow == 5], [0, 1, 2], 3)
    n_level = len(slots) * 4
    X1 = np.zeros((n, n_level))
    X1[np.arange(n), slot * 4 + dtype] = 1.0

    hol = np.zeros(n, bool)
    for a, b in SCHOOL_HOLIDAYS:
        hol |= (idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b) + pd.Timedelta(days=1))
    w = weather.reindex(idx).interpolate(limit_direction="both")
    prcp = w["prcp"].fillna(0)
    z = lambda x: ((x - x.mean()) / (x.std() + 1e-9)).values
    Z = np.column_stack([hol, hol & (dtype < 2),
                         prcp > .1, prcp.rolling(4, min_periods=1).sum() > .5,
                         z(w["temp"]), z(w["temp"]) ** 2, z(w["wspd"]) if "wspd" in w else np.zeros(n),
                         np.linspace(-1, 1, n)]).astype(float)
    return np.hstack([X1, Z]), n_level


def fit_baseline(Y, X, excluded, n_level, alpha=1.0):
    """Ridge pro Station; Zeilen mit NaN oder 'excluded' werden aus X'X / X'y herausgerechnet (schnell).
    Die Level-Dummies (erste n_level Spalten) werden NICHT geschrumpft, sonst wird die Baseline zu niedrig."""
    bad = np.isnan(Y) | excluded
    Y0 = np.where(np.isnan(Y), 0.0, Y)
    XtX, XtY = X.T @ X, X.T @ Y0
    pred = np.empty_like(Y0)
    I = np.diag(np.r_[np.full(n_level, 1e-6), np.full(X.shape[1] - n_level, alpha)])
    for s in range(Y.shape[1]):
        o = bad[:, s]
        Xo = X[o]
        beta = np.linalg.solve(XtX - Xo.T @ Xo + I, XtY[:, s] - Xo.T @ Y0[o, s])
        pred[:, s] = X @ beta
    return pred


# ----------------------------------------------------------------------------- Schritt 2+3: Effekt-Regression
def build_stage2(episodes, idx, stations, resid, att_ref=None):
    """Designmatrix der Effekt-Regression.

    Spalten: Typ-Effekt (Typ, Bin, Hop), Attendance-Steigung (Typ, Bin) und
    Einzelevent-Abweichung (Event, Bin); die letzten beiden nur auf Hop 0.
    att_ref: {Typ: mittleres log(attendance)} - zentriert die Steigung, damit der
    Typ-Effekt weiterhin fuer ein Event mittlerer Groesse gilt.
    """
    sidx = {s: i for i, s in enumerate(stations)}
    type_col, att_col, dev_col, meta = {}, {}, {}, []
    rows, R, C, V, y = {}, [], [], [], []

    def col(dic, key, m):
        if key not in dic:
            dic[key] = len(meta)
            meta.append(m)
        return dic[key]

    for ep in episodes:
        z = None
        if att_ref and ep["type"] in att_ref and ep.get("attendance", 0) > 0:
            z = np.log(ep["attendance"]) - att_ref[ep["type"]]
        for k, i0, i1 in ep_segments(ep, idx):
            for s_, h in ep["hops"].items():
                j = sidx[s_]
                cols = [(col(type_col, (ep["type"], k, h), ("type", ep["type"], None, k, h)), 1.0)]
                if h == 0:
                    cols.append((col(dev_col, (ep["id"], k), ("dev", ep["type"], ep["id"], k, 0)), 1.0))
                    if z is not None:
                        cols.append((col(att_col, (ep["type"], k), ("att", ep["type"], None, k, 0)), z))
                for t in range(i0, i1):
                    v = resid[t, j]
                    if np.isnan(v):
                        continue
                    r = rows.get((j, t))
                    if r is None:
                        r = rows[(j, t)] = len(rows)
                        y.append(v)
                    for c, val in cols:
                        R.append(r); C.append(c); V.append(val)
    X = sp.csr_matrix((V, (R, C)), shape=(len(rows), len(meta)))   # doppelte Eintraege addieren sich
    return X, np.array(y), meta


def fit_stage2(X, y, meta, n_em=20, tau_init=0.1):
    """Penalized LS. Event-Abweichungen u_e ~ N(0, tau^2[Typ,Bin]); tau^2 wird per EM geschaetzt."""
    XtX, Xty = (X.T @ X).toarray(), X.T @ y
    grp = [(m[1], m[3]) if m[0] == "dev" else None for m in meta]      # (Typ, Bin) je Dev-Spalte
    groups = sorted({g for g in grp if g is not None})
    tau2 = {g: tau_init ** 2 for g in groups}
    sigma2 = y.var()

    def lam_vec():
        return np.array([sigma2 / tau2[g] if g is not None else 1e-3 for g in grp])

    for _ in range(n_em):
        A = XtX + np.diag(lam_vec())
        beta = np.linalg.solve(A, Xty)
        sigma2 = np.mean((y - X @ beta) ** 2)
        pv = sigma2 * np.diag(np.linalg.inv(A))
        for g in groups:
            m = np.array([i for i in range(len(meta)) if grp[i] == g])
            tau2[g] = max(np.mean(beta[m] ** 2 + pv[m]), 1e-6)
    A = XtX + np.diag(lam_vec())
    beta = np.linalg.solve(A, Xty)
    se = np.sqrt(sigma2 * np.diag(np.linalg.inv(A)))   # optimistisch: ignoriert Autokorrelation der Residuen
    return beta, se, tau2, sigma2


def pct(b):
    return (np.exp(b) - 1) * 100


def tabulate(meta, beta, se, X, episodes):
    n_obs = np.asarray(X.sum(0)).ravel()
    n_ev = defaultdict(int)
    for ep in episodes:
        n_ev[ep["type"]] += 1
    eff, dev, att = [], [], []
    for i, (kind, typ, eid, k, h) in enumerate(meta):
        if kind == "att":
            # Steigung je log-Einheit -> Effekt einer Verdopplung der Besucherzahl
            att.append(dict(type=typ, bin=BIN_NAMES[k], per_doubling_pct=pct(beta[i] * np.log(2)),
                            ci_low_pct=pct((beta[i] - 1.96 * se[i]) * np.log(2)),
                            ci_high_pct=pct((beta[i] + 1.96 * se[i]) * np.log(2)),
                            n_events=n_ev[typ]))
        elif kind == "type":
            eff.append(dict(type=typ, bin=BIN_NAMES[k], hop=h, effect_pct=pct(beta[i]),
                            ci_low_pct=pct(beta[i] - 1.96 * se[i]), ci_high_pct=pct(beta[i] + 1.96 * se[i]),
                            n_events=n_ev[typ], n_obs=int(n_obs[i])))
        else:
            dev.append(dict(type=typ, event_id=eid, bin=BIN_NAMES[k], dev_log=beta[i], dev_se=se[i]))
    eff = pd.DataFrame(eff).sort_values(["type", "hop", "bin"])
    att = pd.DataFrame(att)
    if len(att):
        att = att.sort_values(["type", "bin"])
    dev = pd.DataFrame(dev)
    if len(dev):
        base = eff.set_index(["type", "bin"]).query("hop == 0").effect_pct
        base_log = np.log1p(base / 100)
        dev["total_effect_pct"] = [pct(base_log[(t, b)] + d) for t, b, d in zip(dev.type, dev.bin, dev.dev_log)]
        info = {e["id"]: (e["name"], e["start"], e["attendance"]) for e in episodes}
        dev["name"] = dev.event_id.map(lambda x: info[x][0])
        dev["start"] = dev.event_id.map(lambda x: info[x][1])
        dev["attendance"] = dev.event_id.map(lambda x: info[x][2])
    return eff, dev, att


# ----------------------------------------------------------------------------- Pipeline
def run(flows, weather, adj, episodes, outdir="out", placebo=False, truth=False):
    idx, stations = flows.index, list(flows.columns)
    sidx = {s: i for i, s in enumerate(stations)}
    if not episodes:
        raise SystemExit("Keine Episoden gefunden - Zuordnung Events/Sperrungen -> Stationen pruefen (siehe [warn] oben).")
    Y = np.log1p(flows.values.astype(float))

    for ep in episodes:
        ep["hops"] = {s: h for s, h in bfs_hops(adj, [a for a in ep["anchors"] if a in sidx], MAX_HOP).items()
                      if s in sidx}
    excluded = np.zeros_like(Y, bool)
    for ep in episodes:
        for k, i0, i1 in ep_segments(ep, idx):
            for s in ep["hops"]:
                excluded[i0:i1, sidx[s]] = True
    print(f"[info] Baseline-Fit ohne {excluded.mean():.1%} der Zellen (Event-/Sperrfenster)")

    Xb, n_level = baseline_design(idx, weather)
    pred = fit_baseline(Y, Xb, excluded, n_level)
    resid = Y - pred
    print(f"[info] Baseline: Std der Log-Residuen auf normalen Zeiten = {np.nanstd(resid[~excluded]):.3f} "
          f"(~{np.nanstd(resid[~excluded]) * 100:.0f}% Rauschen)")

    att_log = defaultdict(list)
    for ep in episodes:
        if ep.get("attendance", 0) > 0:
            att_log[ep["type"]].append(np.log(ep["attendance"]))
    att_ref = {t: float(np.mean(v)) for t, v in att_log.items() if len(v) >= MIN_EVENTS_FOR_ATTENDANCE}

    X, y, meta = build_stage2(episodes, idx, stations, resid, att_ref)
    beta, se, tau2, sigma2 = fit_stage2(X, y, meta)
    eff, dev, att = tabulate(meta, beta, se, X, episodes)
    print(f"[info] Regression: {X.shape[0]} Beobachtungen, {X.shape[1]} Koeffizienten, sigma={np.sqrt(sigma2):.3f}")
    print("[info] Streuung zwischen Einzelevents pro (Typ, Bin), tau auf Log-Skala (0.2 ~ +-20%):")
    for (t, k), v in tau2.items():
        if BIN_NAMES[k] in ("pre1", "post1", "post2"):
            print(f"         {str(t):<11}{BIN_NAMES[k]:<6} tau={np.sqrt(v):.3f}")

    os.makedirs(outdir, exist_ok=True)
    eff.to_csv(f"{outdir}/effects_profile.csv", index=False)
    dev.to_csv(f"{outdir}/event_effects.csv", index=False)
    if len(att):
        att.to_csv(f"{outdir}/attendance_slopes.csv", index=False)
        print("\nEffekt der Eventgroesse: Aenderung bei VERDOPPELTER Besucherzahl (Hop 0);")
        print("der Typ-Effekt oben gilt fuer ein Event mittlerer Groesse dieses Typs.")
        print(att.round(1).to_string(index=False))
        for t, v in sorted(att_ref.items()):
            n = len(att_log[t])
            print(f"   Referenz {t}: {np.exp(v):,.0f} Besucher (geom. Mittel, n={n}, "
                  f"Spanne {np.exp(min(att_log[t])):,.0f}-{np.exp(max(att_log[t])):,.0f})")

    if truth:  # nur Synthetik: Schaetzung vs. Wahrheit
        eff["true_pct"] = [pct(TRUE[t][b][h]) if b in TRUE[t] else 0.0 for t, b, h in zip(eff.type, eff.bin, eff.hop)]
        show = eff[eff.hop <= 1].copy()
        show["est (CI) / true"] = [f"{e:6.1f} ({l:6.1f}..{u:6.1f}) / {t:6.1f}" for e, l, u, t in
                                   zip(show.effect_pct, show.ci_low_pct, show.ci_high_pct, show.true_pct)]
        print("\nEffekt in % ueber Baseline, Schaetzung vs. Wahrheit:")
        print(show[["type", "hop", "bin", "est (CI) / true"]].to_string(index=False))
    else:
        print("\nTyp-Effekte, Hop 0 (Effekt in % ueber Baseline):")
        print(eff[eff.hop == 0].round(1).to_string(index=False))

    if placebo:
        # Verschobene Episoden als Negativkontrolle. Wichtig: Verschiebungen, die auf eine
        # echte Episode derselben Station fallen (Serien-Events!), sind KEIN Placebo -> raus.
        real = defaultdict(list)
        for ep in episodes:
            for s_ in ep["hops"]:
                real[s_].append((ep["start"] - 3 * H, ep["end"] + 2 * H))

        def collides(ep):
            lo, hi = ep["start"] - 3 * H, ep["end"] + 2 * H
            return any(a < hi and lo < b for s_ in ep["hops"] for a, b in real[s_])

        rng = np.random.default_rng(0)
        sh, n_coll = [], 0
        for ep in episodes:
            for _ in range(20):        # mehrere zufaellige Verschiebungen probieren
                off = pd.Timedelta(days=int(rng.choice([-1, 1])) * int(rng.integers(3, 60)))
                cand = dict(ep, start=ep["start"] + off, end=ep["end"] + off)
                if cand["start"] - 3 * H < idx[0] or cand["end"] + 2 * H > idx[-1]:
                    continue
                if not collides(cand):
                    sh.append(cand)
                    break
            else:
                n_coll += 1
        print(f"\nPLACEBO: {len(sh)} von {len(episodes)} Episoden zufaellig verschoben "
              f"(ohne Ueberschneidung mit echten Episoden derselben Station); {n_coll} nicht platzierbar")
        if sh:
            Xp, yp, mp = build_stage2(sh, idx, stations, resid, att_ref)
            bp, sp_, _, _ = fit_stage2(Xp, yp, mp)
            ep_eff, _, _ = tabulate(mp, bp, sp_, Xp, sh)
            print("Effekte auf verschobenen Terminen (Hop 0) - sollten nahe 0% liegen:")
            print(ep_eff[ep_eff.hop == 0].round(1)[["type", "bin", "effect_pct", "ci_low_pct", "ci_high_pct"]]
                  .to_string(index=False))
    return eff, dev


def main():
    global MIN_ATTENDANCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Ordner mit den Hackathon-CSVs")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--venue-map", help="CSV mit Spalten venue_name,address,station_name (Venue -> Station)")
    ap.add_argument("--geocode-cache", help="geocode_cache.json aus event_station_mapping.py "
                                            "(z.B. analysis_output/geocode_cache.json)")
    ap.add_argument("--freq", default="15min",
                    help="Zeitraster; '1h' summiert die Viertelstunden auf (deutlich weniger Rauschen)")
    ap.add_argument("--min-attendance", type=int,
                    help=f"Mindest-Besucherzahl eines Events (Standard {MIN_ATTENDANCE})")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    if a.min_attendance is not None:
        MIN_ATTENDANCE = a.min_attendance
    if a.synthetic:
        flows, weather, adj, eps = make_synthetic()
    elif a.data:
        flows, weather, adj, eps = load_real(a.data, a.venue_map, a.geocode_cache)
    else:
        ap.error("--data DIR oder --synthetic angeben")
    if a.freq != "15min":
        flows, weather = aggregate(flows, weather, a.freq)
    run(flows, weather, adj, eps, outdir=a.out, placebo=a.placebo, truth=a.synthetic)


if __name__ == "__main__":
    main()