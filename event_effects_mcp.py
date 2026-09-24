"""
event_effects_mcp.py - MCP-Server: Event- und Sperrungseffekte auf den U-Bahn-Flow

Kernkonzept (unveraendert aus event_effects.py):
  1) Baseline pro Station: log1p(flow) ~ Viertelstunde x Tagtyp + Ferien + Wetter + Trend,
     gefittet NUR auf Zeiten ohne Event/Sperrung. Pendlerwellen (Tageszeit x Tagtyp) und das
     Grundniveau jeder Station (Hotspots) stecken damit in der Baseline, nicht im Effekt.
  2) Residuen = log-Ist - log-Baseline  -> nur das Unnormale bleibt uebrig.
  3) Residuen ~ Typ-Effekt(Typ, Zeitfenster, Hop) + Attendance-Steigung + Einzelevent-Abweichung
     (Ridge auf die Abweichungen, Schrumpfungsstaerke tau^2 per EM).
  Blow-up: erwarteter Flow = expm1(log-Baseline + log-Effekt), pro Station im Umkreis und Zeitslot.

Tools
  effect_profile  Typ-Effekte in % ueber Baseline je Zeitfenster und Hop (ohne Argument: Uebersicht)
  blow_up         Szenario (Station, Typ, Zeit, Besucher) -> Baseline vs. erwarteter Flow im Umkreis
  event_effect    Event / Sperrung aus den Daten -> Baseline, erwarteter und tatsaechlicher Flow
  abnormal_flow   Ist vs. Baseline (nur Unnormales) fuer eine Station oder Top-N im ganzen Netz

Start
  DATA_DIR=./data python event_effects_mcp.py
  optional: GEOCODE_CACHE=analysis_output/geocode_cache.json   VENUE_MAP=venues.csv
            MIN_ATTENDANCE=2000   FREQ=15min
"""
import difflib
import json
import os
import re
import sys
import threading
import traceback
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import scipy.sparse as sp
try:                                       # mcp 1.x
    from mcp.server.fastmcp import FastMCP
except ImportError:                        # mcp 2.x (FastMCP heisst dort MCPServer)
    from mcp.server.mcpserver import MCPServer as FastMCP

# ----------------------------------------------------------------------------- Konfiguration
DATA_DIR = os.environ.get("DATA_DIR", "./data")
GEOCODE_CACHE = os.environ.get("GEOCODE_CACHE") or None
VENUE_MAP = os.environ.get("VENUE_MAP") or None
FREQ = os.environ.get("FREQ", "15min")
MIN_ATTENDANCE = int(os.environ.get("MIN_ATTENDANCE", "2000"))

TYPE_COL = "segment"             # Spalte in den Events, die den Event-Typ definiert
MIN_EVENTS_FOR_ATTENDANCE = 15   # Attendance-Steigung erst ab so vielen Events des Typs
MAX_HOP = 2                      # Einflussradius im Netz (Stationen vom Venue-Bahnhof)
DEFAULT_EVENT_HOURS = 3          # Dauer, falls estimated_end_local fehlt
MAX_DISTANCE_M = 1500            # weiter entfernte Venues gelten als "keine Station in der Naehe"
EARTH_RADIUS_M = 6_371_000
SCHOOL_HOLIDAYS = [("2026-07-09", "2026-08-22")]   # Berlin Sommerferien - BITTE PRUEFEN

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
STEP = pd.Timedelta(FREQ)
BIN_NAMES = ["pre3", "pre2", "pre1", "during", "post1", "post2"]
BIN_DESC = {"pre3": "3-2 h vor Beginn", "pre2": "2-1 h vor Beginn", "pre1": "letzte Stunde vor Beginn",
            "during": "Beginn bis Ende", "post1": "1. Stunde nach Ende", "post2": "2. Stunde nach Ende"}


def log(*a):
    """Nur nach stderr - stdout gehoert dem MCP-Protokoll."""
    print(*a, file=sys.stderr, flush=True)


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


def pct(b):
    return (np.exp(b) - 1) * 100


# ----------------------------------------------------------------------------- Netz / Namen
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
    return re.sub(r"str\.", "strasse", norm(t))


def station_key(n):
    """'S+U Warschauer Str. (Berlin)' -> 'warschauer strasse'."""
    k = text_key(n)
    k = re.sub(r"\s*\(berlin\)\s*$", "", k)
    k = re.sub(r"^(s\+u|s|u)\s+", "", k)
    return re.sub(r"\bstr\b", "strasse", k).strip()


def find_stations(text, keys):
    hits = [k for k in keys if len(k) > 3 and re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", text)]
    return sorted(hits, key=len, reverse=True)


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(np.asarray(lat2, float)), np.radians(np.asarray(lon2, float))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- Geocode-Cache
def cache_key(venue, address):
    def c(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return str(x).strip()
    return f"{c(venue)} | {c(address)}"


def extract_latlon(v):
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
    lat = next((v[k] for k in ("lat", "latitude", "y", "lat_deg") if v.get(k) is not None), None)
    lon = next((v[k] for k in ("lon", "lng", "long", "longitude", "x", "lon_deg") if v.get(k) is not None), None)
    if lat is None or lon is None:
        return None
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def key_variants(venue, address):
    def c(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return re.sub(r"\s+", " ", str(x)).strip().lower()
    v, a = c(venue), c(address)
    out = [f"{v}|{a}"]
    if a:
        out.append(f"|{a}")
    if v:
        out.append(f"{v}|")
    return out


def load_geocode_cache(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        items = [(cache_key(e.get("venue_name"), e.get("address")), e) for e in raw if isinstance(e, dict)]
    else:
        items = list(raw.items())
    cache = {}
    for k, v in items:
        hit = extract_latlon(v)
        if hit and not (np.isnan(hit[0]) or np.isnan(hit[1])):
            parts = str(k).split("|", 1)
            cache[key_variants(parts[0], parts[1] if len(parts) > 1 else "")[0]] = {"lat": hit[0], "lon": hit[1]}
            if len(parts) > 1 and parts[1].strip():
                cache.setdefault(key_variants("", parts[1])[0], {"lat": hit[0], "lon": hit[1]})
    log(f"[info] Geocode-Cache: {len(cache)} Schluessel aus {len(items)} Orten ({path})")
    return cache


def nearest_station_lookup(stations_df, station_names):
    st = stations_df[stations_df.station_name.isin(set(station_names))].dropna(subset=["latitude", "longitude"])
    if st.empty:
        raise RuntimeError("Keine Station mit Koordinaten und Flow-Daten - stations_with_ubahn.csv pruefen.")
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


# ----------------------------------------------------------------------------- Daten laden
def to_local(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert("Europe/Berlin").dt.tz_localize(None)


DATA_SUFFIX = os.environ.get("DATA_SUFFIX", "_pre_innotrans")   # flows_pre_innotrans.csv usw.


def pick(d, stem):
    """Nimmt bevorzugt '<stem>_pre_innotrans.csv', sonst die erste CSV, die mit stem beginnt."""
    exact = os.path.join(d, f"{stem}{DATA_SUFFIX}.csv")
    if os.path.exists(exact):
        return exact
    hits = sorted(f for f in os.listdir(d) if f.startswith(stem) and f.endswith(".csv"))
    if not hits:
        raise RuntimeError(f"Keine Datei '{stem}*.csv' in {d}")
    if len(hits) > 1:
        log(f"[warn] mehrere Dateien fuer '{stem}': {hits} -> nehme {hits[0]}")
    return os.path.join(d, hits[0])


def load_real(d, venue_map_path=None, geocode_cache_path=None):
    st = pd.read_csv(pick(d, "stations_with_ubahn"))
    con = pd.read_csv(pick(d, "berlin_ubahn_connections"))
    flows = pd.read_csv(pick(d, "flows"), parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    weather = pd.read_csv(pick(d, "weather_data"), index_col=0, parse_dates=True).sort_index()
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
            log(f"[warn] VENUE_TO_STATION: '{v}' nicht im Netz")
    pair_map = {}
    if venue_map is not None:
        for _, r in venue_map.iterrows():
            sk = station_key(r["station_name"])
            if sk in key2name:
                v = text_key(r["venue_name"]) if pd.notna(r.get("venue_name")) else ""
                a = text_key(r["address"]) if pd.notna(r.get("address")) else ""
                pair_map[(v, a)] = key2name[sk]
    nearest = nearest_station_lookup(stations_df, station_names) if geo_cache and stations_df is not None else None

    ev = ev.copy()
    ev["start"] = to_local(ev.began_local)
    ev["end"] = to_local(ev.estimated_end_local)
    bad = ev.end.isna() | (ev.end <= ev.start)
    ev.loc[bad, "end"] = ev.loc[bad, "start"] + pd.Timedelta(hours=DEFAULT_EVENT_HOURS)
    ev["estimated_attendance"] = ev.estimated_attendance.fillna(0)
    ev = ev[ev.estimated_attendance >= MIN_ATTENDANCE]

    episodes, unmapped, how = [], [], defaultdict(int)
    for i, r in ev.iterrows():
        v = text_key(r.venue_name) if pd.notna(r.venue_name) else ""
        ad = text_key(r.address) if pd.notna(r.address) else ""
        station = pair_map.get((v, ad))
        if station:
            how["venue_map"] += 1
        if station is None and nearest is not None:
            g = next((geo_cache[k] for k in key_variants(r.venue_name, r.address) if k in geo_cache), None)
            if g:
                cand, dist = nearest(g["lat"], g["lon"])
                if dist <= MAX_DISTANCE_M:
                    station = cand
                    how["geocode"] += 1
        if station is None:
            station = next((s for k, s in mapping.items() if k in v or k in ad), None)
            how["VENUE_TO_STATION"] += station is not None
        if station is None:
            hits = find_stations(v + " | " + ad, keys)
            station = key2name[hits[0]] if hits else None
            how["Name in Adresse"] += station is not None
        if station is None:
            unmapped.append((r.estimated_attendance, r.venue_name, r.address))
            continue
        episodes.append(dict(id=f"ev_{i}", type=str(r[TYPE_COL]), start=r.start, end=r.end,
                             anchors=[station], name=r.event_name, attendance=float(r.estimated_attendance)))
    log(f"[info] Events {len(ev)} (>= {MIN_ATTENDANCE} Besucher), zugeordnet: {dict(how)}, ohne Station: {len(unmapped)}")
    for att, v, a in sorted(unmapped, key=lambda x: -x[0])[:10]:
        log(f"   ohne Station: {int(att):>7}  {v} | {a}")

    n_skip = 0
    for i, r in cl.iterrows():
        dur = parse_duration(r.duration)
        hits = find_stations(text_key(r.description), keys)
        if dur is None or not hits:
            n_skip += 1
            continue
        episodes.append(dict(id=f"cl_{i}", type="closure", start=r["when"], end=r["when"] + dur,
                             anchors=[key2name[h] for h in hits], name=r.description, attendance=0))
    log(f"[info] Episoden: {sum(e['type'] != 'closure' for e in episodes)} Events, "
        f"{sum(e['type'] == 'closure' for e in episodes)} Sperrungen ({n_skip} Sperrungen uebersprungen)")
    return episodes


def aggregate(flows, weather, freq):
    return flows.resample(freq).sum(min_count=1), weather.resample(freq).mean()


# ----------------------------------------------------------------------------- Schritt 1: Baseline
def baseline_stats(idx, weather):
    """Normierungen aus den Trainingsdaten, damit die Baseline auch fuer neue Zeitpunkte identisch rechnet."""
    w = weather.reindex(idx).interpolate(limit_direction="both")
    ms = lambda c: (float(w[c].mean()), float(w[c].std())) if c in w else (0.0, 1.0)
    return dict(slots=np.unique(idx.hour * 60 + idx.minute), t0=idx[0], t1=idx[-1],
                temp=ms("temp"), wspd=ms("wspd"), has_wspd="wspd" in w)


def baseline_design(idx, weather, bs):
    """Designmatrix der Baseline. Gibt (X, n_level) zurueck; n_level = Anzahl der Level-Dummies.
    Auf den Trainingszeitpunkten identisch zur Originalversion; fuer neue Zeitpunkte ohne Wetter
    wird neutrales Wetter (Mittelwert, kein Regen) angenommen und der Trend bei +-1 gedeckelt."""
    n = len(idx)
    tod = idx.hour * 60 + idx.minute
    slot = np.clip(np.searchsorted(bs["slots"], tod), 0, len(bs["slots"]) - 1)
    dow = idx.dayofweek
    dtype = np.select([dow <= 3, dow == 4, dow == 5], [0, 1, 2], 3)
    n_level = len(bs["slots"]) * 4
    X1 = np.zeros((n, n_level))
    X1[np.arange(n), slot * 4 + dtype] = 1.0

    hol = np.zeros(n, bool)
    for a, b in SCHOOL_HOLIDAYS:
        hol |= (idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b) + pd.Timedelta(days=1))
    w = weather.reindex(idx).interpolate(limit_direction="both")
    prcp = w["prcp"].fillna(0)
    z = lambda c: ((w[c].fillna(bs[c][0]) - bs[c][0]) / (bs[c][1] + 1e-9)).values
    zt = z("temp")
    zw = z("wspd") if bs["has_wspd"] else np.zeros(n)
    trend = np.clip(-1 + 2 * np.asarray((idx - bs["t0"]) / (bs["t1"] - bs["t0"])), -1, 1)
    Z = np.column_stack([hol, hol & (dtype < 2),
                         prcp > .1, prcp.rolling(4, min_periods=1).sum() > .5,
                         zt, zt ** 2, zw, trend]).astype(float)
    return np.hstack([X1, Z]), n_level


def fit_baseline(Y, X, excluded, n_level, alpha=1.0):
    """Ridge pro Station; Event-/Sperrfenster und NaN werden aus X'X / X'y herausgerechnet.
    Level-Dummies werden nicht geschrumpft. Gibt (Vorhersage, Koeffizienten) zurueck."""
    bad = np.isnan(Y) | excluded
    Y0 = np.where(np.isnan(Y), 0.0, Y)
    XtX, XtY = X.T @ X, X.T @ Y0
    I = np.diag(np.r_[np.full(n_level, 1e-6), np.full(X.shape[1] - n_level, alpha)])
    B = np.empty((X.shape[1], Y.shape[1]))
    for s in range(Y.shape[1]):
        o = bad[:, s]
        Xo = X[o]
        B[:, s] = np.linalg.solve(XtX - Xo.T @ Xo + I, XtY[:, s] - Xo.T @ Y0[o, s])
    return X @ B, B


# ----------------------------------------------------------------------------- Schritt 2+3: Effekt-Regression
def build_stage2(episodes, idx, stations, resid, att_ref=None):
    """Spalten: Typ-Effekt (Typ, Bin, Hop), Attendance-Steigung (Typ, Bin) und
    Einzelevent-Abweichung (Event, Bin); die letzten beiden nur auf Hop 0."""
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
    X = sp.csr_matrix((V, (R, C)), shape=(len(rows), len(meta)))
    return X, np.array(y), meta


def fit_stage2(X, y, meta, n_em=20, tau_init=0.1):
    """Penalized LS. Event-Abweichungen u_e ~ N(0, tau^2[Typ,Bin]); tau^2 per EM."""
    XtX, Xty = (X.T @ X).toarray(), X.T @ y
    grp = [(m[1], m[3]) if m[0] == "dev" else None for m in meta]
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
    se = np.sqrt(sigma2 * np.diag(np.linalg.inv(A)))   # optimistisch: ignoriert Autokorrelation
    return beta, se


def tabulate(meta, beta, se, X, episodes):
    n_obs = np.asarray(X.sum(0)).ravel()
    n_ev = defaultdict(int)
    for ep in episodes:
        n_ev[ep["type"]] += 1
    eff, att = [], []
    for i, (kind, typ, _, k, h) in enumerate(meta):
        if kind == "att":
            att.append(dict(type=typ, window=BIN_NAMES[k], per_doubling_pct=pct(beta[i] * np.log(2)),
                            ci_low_pct=pct((beta[i] - 1.96 * se[i]) * np.log(2)),
                            ci_high_pct=pct((beta[i] + 1.96 * se[i]) * np.log(2))))
        elif kind == "type":
            eff.append(dict(type=typ, window=BIN_NAMES[k], k=k, hop=h, effect_pct=pct(beta[i]),
                            ci_low_pct=pct(beta[i] - 1.96 * se[i]), ci_high_pct=pct(beta[i] + 1.96 * se[i]),
                            n_events=n_ev[typ], n_obs=int(n_obs[i])))
    return pd.DataFrame(eff), pd.DataFrame(att)


# ----------------------------------------------------------------------------- Modell
def parse_ts(s):
    t = pd.Timestamp(s)
    return t.tz_convert("Europe/Berlin").tz_localize(None) if t.tzinfo is not None else t


class Model:
    def __init__(self):
        flows, weather, adj, episodes = load_real(DATA_DIR, VENUE_MAP, GEOCODE_CACHE)
        if FREQ != "15min":
            flows, weather = aggregate(flows, weather, FREQ)
        if not episodes:
            raise RuntimeError("Keine Episoden gefunden - Zuordnung Events/Sperrungen -> Stationen pruefen.")
        self.flows, self.weather, self.adj = flows, weather, adj
        self.idx, self.stations = flows.index, list(flows.columns)
        self.sidx = {s: i for i, s in enumerate(self.stations)}
        self.key2name = {station_key(n): n for n in self.stations}
        Y = np.log1p(flows.values.astype(float))

        for ep in episodes:
            ep["hops"] = self.hops(ep["anchors"])
        excluded = np.zeros_like(Y, bool)
        for ep in episodes:
            for _, i0, i1 in ep_segments(ep, self.idx):
                for s in ep["hops"]:
                    excluded[i0:i1, self.sidx[s]] = True
        log(f"[info] Baseline-Fit ohne {excluded.mean():.1%} der Zellen (Event-/Sperrfenster)")

        # 1) Baseline: Pendlerprofil + Stationsniveau + Ferien + Wetter + Trend
        self.bstats = baseline_stats(self.idx, weather)
        Xb, n_level = baseline_design(self.idx, weather, self.bstats)
        self.pred, self.B = fit_baseline(Y, Xb, excluded, n_level)
        # 2) Residuen = nur das Unnormale
        self.resid = Y - self.pred
        sd = np.nanstd(np.where(excluded, np.nan, self.resid), axis=0)
        self.resid_sd = np.where(sd > 0, sd, np.nan)

        # 3) Effekt-Regression
        att_log = defaultdict(list)
        for ep in episodes:
            if ep.get("attendance", 0) > 0:
                att_log[ep["type"]].append(np.log(ep["attendance"]))
        self.att_ref = {t: float(np.mean(v)) for t, v in att_log.items() if len(v) >= MIN_EVENTS_FOR_ATTENDANCE}
        X, y, meta = build_stage2(episodes, self.idx, self.stations, self.resid, self.att_ref)
        beta, se = fit_stage2(X, y, meta)
        self.coef = {(m[0], m[2] if m[0] == "dev" else m[1], m[3], m[4]): float(b) for m, b in zip(meta, beta)}
        self.eff, self.att = tabulate(meta, beta, se, X, episodes)
        self.episodes = episodes
        self.n_events = defaultdict(int)
        for ep in episodes:
            self.n_events[ep["type"]] += 1
        log(f"[info] Modell fertig: {len(self.stations)} Stationen, {len(episodes)} Episoden, "
            f"{X.shape[0]} Beobachtungen, {X.shape[1]} Koeffizienten")

    # ---- Hilfen
    def hops(self, anchors):
        return {s: h for s, h in bfs_hops(self.adj, [a for a in anchors if a in self.sidx], MAX_HOP).items()
                if s in self.sidx}

    def resolve_station(self, name):
        k = station_key(name)
        if k in self.key2name:
            return self.key2name[k]
        sub = [n for kk, n in self.key2name.items() if k and (k in kk or kk in k)]
        if len(sub) == 1:
            return sub[0]
        close = difflib.get_close_matches(k, list(self.key2name), n=3, cutoff=0.5)
        raise ValueError(f"Station '{name}' nicht gefunden. Vorschlaege: {[self.key2name[c] for c in close] or sub[:5]}")

    def resolve_type(self, name):
        types = list(self.n_events)
        hit = [t for t in types if t.lower() == str(name).lower()] or \
              [t for t in types if str(name).lower() in t.lower()]
        if not hit:
            raise ValueError(f"Event-Typ '{name}' unbekannt. Verfuegbar: {sorted(types)}")
        return hit[0]

    def size_z(self, typ, attendance):
        if typ in self.att_ref and attendance and attendance > 0:
            return float(np.log(attendance) - self.att_ref[typ])
        return None

    def log_effect(self, typ, k, h, z=None, eid=None):
        """Typ-Effekt + (auf Hop 0) Groessen-Steigung + Einzelevent-Abweichung, alles auf Log-Skala."""
        b = self.coef.get(("type", typ, k, h), 0.0)
        if h == 0:
            if z is not None:
                b += self.coef.get(("att", typ, k, 0), 0.0) * z
            if eid is not None:
                b += self.coef.get(("dev", eid, k, 0), 0.0)
        return b

    def baseline_log(self, ts):
        """log1p-Baseline fuer beliebige Zeitpunkte: bekannte aus dem Fit, neue aus den Koeffizienten."""
        pos = self.idx.get_indexer(ts)
        out = np.empty((len(ts), len(self.stations)))
        known = pos >= 0
        out[known] = self.pred[pos[known]]
        if (~known).any():
            X, _ = baseline_design(ts[~known], self.weather, self.bstats)
            out[~known] = X @ self.B
        return out

    def explaining(self, s, t):
        return [f"{ep['name']} ({ep['type']}, Hop {ep['hops'][s]})" for ep in self.episodes
                if s in ep["hops"] and ep["start"] - 3 * H <= t < ep["end"] + 2 * H]

    # ---- Blow-up
    def blow_up(self, anchors, typ, start, end, attendance=None, eid=None, granularity="window"):
        hops = self.hops(anchors)
        if not hops:
            raise ValueError("Keine der Ankerstationen hat Flow-Daten.")
        z = self.size_z(typ, attendance)
        bounds = bin_bounds(dict(start=start, end=end))
        grid = pd.date_range(bounds[0][0].ceil(FREQ), bounds[-1][1], freq=FREQ)
        grid = grid[grid < bounds[-1][1]]
        bins = np.full(len(grid), -1)
        for k, (lo, hi) in enumerate(bounds):
            bins[(grid >= lo) & (grid < hi)] = k
        grid, bins = grid[bins >= 0], bins[bins >= 0]

        base = self.baseline_log(grid)
        act = self.flows.reindex(grid)
        out = []
        for s, h in hops.items():
            j = self.sidx[s]
            eff = np.array([self.log_effect(typ, k, h, z, eid) for k in bins])
            b = np.maximum(np.expm1(base[:, j]), 0)
            e = np.maximum(np.expm1(base[:, j] + eff), 0)
            a = act[s].to_numpy(float)
            if granularity == "slot":
                rows = [dict(time=t, window=BIN_NAMES[k], baseline=bb, expected=ee,
                             actual=None if np.isnan(aa) else aa)
                        for t, k, bb, ee, aa in zip(grid, bins, b, e, a)]
            else:
                rows = []
                for k in np.unique(bins):
                    m = bins == k
                    sb, se_, ak = b[m].sum(), e[m].sum(), a[m]
                    sa = None if np.isnan(ak).all() else float(np.nansum(ak))
                    rows.append(dict(window=BIN_NAMES[k], desc=BIN_DESC[BIN_NAMES[k]],
                                     start=grid[m][0], end=grid[m][-1] + STEP,
                                     baseline=sb, expected=se_,
                                     effect_pct=(se_ / sb - 1) * 100 if sb > 0 else None,
                                     actual=sa, actual_vs_baseline_pct=(sa / sb - 1) * 100 if sa is not None and sb > 0 else None))
            out.append(dict(station=s, hop=h, extra_passengers_total=float((e - b).sum()),
                            peak_slot=grid[int(np.argmax(e))], peak_expected=float(e.max()), windows=rows))
        out.sort(key=lambda r: (r["hop"], -abs(r["extra_passengers_total"])))
        size = (f"Effekt auf {attendance:,.0f} Besucher skaliert (Referenz {np.exp(self.att_ref[typ]):,.0f})"
                if z is not None else "keine Groessenanpassung (Typ mit < 15 Events oder keine Besucherzahl)")
        return dict(anchors=anchors, type=typ, start=start, end=end, attendance=attendance,
                    size_adjustment=size,
                    note="baseline = normaler Flow ohne Event (Pendlerprofil, Stationsniveau, Wetter); "
                         "expected = Baseline x geschaetzter Effekt; Werte = Passagiere summiert pro Zeitfenster",
                    stations=out)


# ----------------------------------------------------------------------------- MCP
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if o is pd.NaT:
        return None
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d %H:%M")
    if isinstance(o, pd.Timedelta):
        return str(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return round(float(o), 2) if np.isfinite(o) else None
    return o


_model, _err, _ready = None, None, threading.Event()


def _fit():
    global _model, _err
    try:
        _model = Model()
    except BaseException as e:
        _err = e
        log(traceback.format_exc())
    finally:
        _ready.set()


def M():
    if not _ready.wait(timeout=30):
        raise RuntimeError("Modell wird noch gefittet - bitte in einer Minute erneut versuchen.")
    if _err is not None:
        raise RuntimeError(f"Modell-Fit fehlgeschlagen: {_err}")
    return _model


mcp = FastMCP("event-effects")


@mcp.tool()
def effect_profile(event_type: str | None = None) -> dict:
    """Geschaetzte Effekte eines Event-Typs in % ueber dem normalen Flow (Baseline ohne Pendler-/Stationseffekte).
    Zeitfenster: pre3/pre2/pre1 (Stunden vor Beginn), during, post1/post2 (Stunden nach Ende).
    hop = Abstand in Stationen vom Venue-Bahnhof (0 = Venue-Bahnhof). Typ 'closure' = Sperrungen
    (Hop 0 = gesperrte Station, Hop 1-2 = Nachbarn). Ohne event_type: Liste der Typen und Datenumfang."""
    m = M()
    if not event_type:
        return _clean(dict(types=dict(m.n_events), data_from=m.idx[0], data_to=m.idx[-1],
                           n_stations=len(m.stations), windows=BIN_DESC, max_hop=MAX_HOP))
    typ = m.resolve_type(event_type)
    eff = m.eff[m.eff.type == typ].sort_values(["hop", "k"]).drop(columns=["type", "k"])
    res = dict(type=typ, n_events=m.n_events[typ], effects=eff.to_dict("records"))
    if typ in m.att_ref and len(m.att):
        res["reference_attendance"] = float(np.exp(m.att_ref[typ]))
        a = m.att[m.att.type == typ].drop(columns=["type"])
        a["k"] = a.window.map(BIN_NAMES.index)
        res["attendance_slope_hop0"] = a.sort_values("k").drop(columns=["k"]).to_dict("records")
        res["attendance_note"] = "per_doubling_pct = zusaetzliche Aenderung am Venue-Bahnhof bei doppelter Besucherzahl"
    return _clean(res)


@mcp.tool()
def blow_up(stations: str, event_type: str, start: str, end: str | None = None,
            attendance: int | None = None, granularity: str = "window") -> dict:
    """Szenario hochrechnen: normaler Flow (Baseline) x geschaetzter Event-/Sperrungseffekt fuer die
    Station(en) und alle Nachbarn bis 2 Stationen Abstand.
    stations: Venue-Bahnhof oder bei Sperrung die gesperrten Stationen, komma-getrennt (z.B. 'Hallesches Tor, Kochstr.').
    event_type: z.B. 'Music', 'Sports' oder 'closure' (siehe effect_profile).
    start/end: Berliner Ortszeit 'YYYY-MM-DD HH:MM'; end fehlt -> start + 3 h.
    attendance: erwartete Besucher (skaliert den Effekt am Venue-Bahnhof).
    granularity: 'window' (Summen je Zeitfenster) oder 'slot' (jeder 15-min-Slot)."""
    m = M()
    anchors = [m.resolve_station(s) for s in stations.split(",") if s.strip()]
    typ = m.resolve_type(event_type)
    t0 = parse_ts(start)
    t1 = parse_ts(end) if end else t0 + pd.Timedelta(hours=DEFAULT_EVENT_HOURS)
    return _clean(m.blow_up(anchors, typ, t0, t1, attendance, None, granularity))


@mcp.tool()
def event_effect(query: str, date: str | None = None, granularity: str = "window") -> dict:
    """Ein Event oder eine Sperrung aus den Daten nachschlagen (Namens-/Beschreibungsteil, optional Datum
    'YYYY-MM-DD') und den Effekt zeigen: Baseline, vom Modell erwarteter Flow (inkl. der Abweichung genau
    dieses Events) und tatsaechlicher Flow, fuer den Bahnhof und seine Nachbarn. Liefert bei mehreren
    Treffern eine Kandidatenliste. Enthaelt Zeitraum, Dauer und Beschreibung (bei Sperrungen = Grund)."""
    m = M()
    q = norm(query)
    cands = [e for e in m.episodes if q in norm(e["name"])]
    if date:
        d = parse_ts(date).date()
        cands = [e for e in cands if e["start"].date() <= d <= e["end"].date()]
    if not cands:
        names = sorted({str(e["name"]) for e in m.episodes})
        close = difflib.get_close_matches(str(query), names, n=5, cutoff=0.3)
        return _clean(dict(error="kein Treffer", suggestions=close))
    uniq = {(str(e["name"]), e["start"]): e for e in cands}
    if len(uniq) > 1:
        return _clean(dict(message="mehrere Treffer - mit Datum oder genauerem Namen eingrenzen",
                           candidates=[dict(name=e["name"], type=e["type"], start=e["start"], end=e["end"],
                                            stations=e["anchors"], attendance=e["attendance"])
                                       for e in sorted(uniq.values(), key=lambda e: e["start"])[:15]]))
    ep = next(iter(uniq.values()))
    res = m.blow_up(ep["anchors"], ep["type"], ep["start"], ep["end"], ep["attendance"] or None,
                    ep["id"], granularity)
    res["event"] = dict(name=ep["name"], type=ep["type"], start=ep["start"], end=ep["end"],
                        duration=ep["end"] - ep["start"], stations=ep["anchors"], attendance=ep["attendance"])
    return _clean(res)


@mcp.tool()
def abnormal_flow(start: str, end: str, station: str | None = None, top: int = 10) -> dict:
    """Unnormaler Flow = Ist vs. Baseline (Pendlerprofil, Stationsniveau, Wetter herausgerechnet).
    Mit station: Zeitverlauf dieser Station. Ohne station: die top-N Stationen mit der staerksten
    Abweichung im Zeitraum (je Station der extremste Slot), mit Wetter und bekannten Events/Sperrungen,
    die sie erklaeren koennten (explained_by leer = unerklaerte Anomalie).
    z = Abweichung in Standardabweichungen des normalen Rauschens der Station.
    start/end: Berliner Ortszeit 'YYYY-MM-DD HH:MM'."""
    m = M()
    t0, t1 = parse_ts(start), parse_ts(end)
    pos = np.flatnonzero((m.idx >= t0) & (m.idx < t1))
    if not len(pos):
        raise ValueError(f"Keine Flow-Daten im Zeitraum (Daten: {m.idx[0]} bis {m.idx[-1]}).")

    def row(i, j, weather=False):
        t, s, r = m.idx[i], m.stations[j], m.resid[i, j]
        d = dict(time=t, station=s, actual=m.flows.iat[i, j], baseline=max(np.expm1(m.pred[i, j]), 0.0),
                 abnormal_pct=pct(r), z=r / m.resid_sd[j], explained_by=m.explaining(s, t))
        if weather:
            w = m.weather.reindex([t]).iloc[0]
            d["weather"] = {c: w[c] for c in ("temp", "prcp", "wspd", "coco") if c in w.index}
        return d

    if station:
        if len(pos) > 400:
            raise ValueError("Zu langer Zeitraum fuer eine Station (max. 400 Slots) - bitte kuerzen.")
        j = m.sidx[m.resolve_station(station)]
        return _clean(dict(station=m.stations[j], rows=[row(i, j) for i in pos]))
    Z = np.abs(np.nan_to_num(m.resid[pos] / m.resid_sd, nan=0.0))
    best = Z.argmax(axis=0)
    score = Z[best, np.arange(Z.shape[1])]
    order = np.argsort(-score)[:max(1, top)]
    return _clean(dict(rows=[row(pos[best[j]], j, weather=True) for j in order]))


if __name__ == "__main__":
    threading.Thread(target=_fit, daemon=True).start()
    mcp.run()