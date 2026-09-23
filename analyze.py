#!/usr/bin/env python3
"""
analyze_ubahn.py
=================
Explorations- und Analyse-Skript fuer den InnoTrans 2026 Hackathon
("Alstom Intelligence: Talk To My Train" - Berlin U-Bahn Datensatz).

Was das Skript macht
---------------------
1. Laedt alle Dateien gemaess `dataset_schema.md`.
2. Erstellt eine Reihe von Visualisierungen (Zeitreihen, Heatmaps,
   Netzwerkplot der Stationen, Wetter-Scatterplots, Energie, Events).
3. Berechnet Korrelationen:
   - Wetter <-> Gesamt-Fahrgastfluss (Pearson + Spearman)
   - Zeitversetzte (lagged) Kreuzkorrelation
   - Station-zu-Station Korrelationen, insbesondere Paare OHNE direkte
     U-Bahn-Verbindung, die trotzdem stark korrelieren (siehe
     Trainingsfrage 8 im Hackathon-Fragenkatalog)
4. Untersucht Kausalitaet / Wirkung (kein Beweis, aber belastbare Hinweise):
   - Granger-Kausalitaetstests (Wetter -> Fluss, Fluss <-> Fluss zwischen
     auffaelligen Stationspaaren)
   - Event-Impact-Analyse (Vorher/Waehrend/Nachher-Vergleich rund um
     bekannte Veranstaltungsorte, mit t-Test)
   - Closure-Impact-Analyse (Effekt einer Streckensperrung auf die
     gesperrten Stationen selbst und auf ihre Nachbarstationen)
5. Schreibt alle Kennzahlen + Diagramm-Pfade in einen Markdown-Report
   (`report.md`) im Output-Verzeichnis.

Wichtiger Hinweis zu "Kausalitaet"
-----------------------------------
Nichts hier ist ein kausaler Beweis im strengen Sinn. Granger-Kausalitaet
misst nur, ob vergangene Werte einer Zeitreihe die Vorhersage einer
anderen statistisch signifikant verbessern ("Vorhersage-Kausalitaet"),
nicht echte physikalische Verursachung. Die Event-/Closure-Analysen sind
Vorher/Nachher-Vergleiche (quasi-experimentell), keine kontrollierten
Experimente. Behandelt alle p-Werte als Hinweise, nicht als Beweise, und
kommuniziert das auch im fertigen Agenten (siehe Problem Statement:
"communicate uncertainty clearly").

Nutzung
-------
    python analyze_ubahn.py --data-dir data --output-dir analysis_output

Optionen: siehe `python analyze_ubahn.py --help`
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: funktioniert auch ohne Display / im Terminal
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    import inspect as _inspect

    from statsmodels.tsa.stattools import grangercausalitytests as _grangercausalitytests

    HAS_STATSMODELS = True
    _GRANGER_HAS_VERBOSE = "verbose" in _inspect.signature(_grangercausalitytests).parameters

    def run_granger(data, maxlag):
        """Wrapper kompatibel mit alten (verbose=...) und neuen statsmodels-Versionen
        (>=0.14 hat den `verbose`-Parameter entfernt)."""
        if _GRANGER_HAS_VERBOSE:
            return _grangercausalitytests(data, maxlag=maxlag, verbose=False)
        return _grangercausalitytests(data, maxlag=maxlag)
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False

    def run_granger(data, maxlag):  # noqa: ANN001
        raise RuntimeError("statsmodels ist nicht installiert.")

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# Bekannte Berliner Veranstaltungsorte (grobe Koordinaten) fuer die
# Event-Impact-Analyse. events.csv liefert keine lat/lon, nur eine Adresse.
# Fuer echtes Geocoding aller Venues koennte man z.B. die Adresse per
# Nominatim/OSM aufloesen; hier reicht eine kleine, erweiterbare Lookup-
# Tabelle fuer die grossen/bekannten Locations aus dem Fragenkatalog.
# Ergaenzt die Liste gern um weitere Venues aus eurem echten Datensatz.
# ---------------------------------------------------------------------------
KNOWN_VENUE_COORDS = {
    "uber arena": (52.5058, 13.4433),
    "mercedes-benz arena": (52.5058, 13.4433),
    "o2 world": (52.5058, 13.4433),
    "olympiastadion berlin": (52.5147, 13.2395),
    "olympiastadion": (52.5147, 13.2395),
    "velodrom": (52.5340, 13.4432),
    "waldbuehne": (52.5133, 13.2372),
    "waldbühne": (52.5133, 13.2372),
    "messe berlin": (52.5052, 13.2758),
    "messegelaende berlin": (52.5052, 13.2758),
    "alte foersterei": (52.4572, 13.5706),
    "max-schmeling-halle": (52.5423, 13.4128),
    "tempodrom": (52.4989, 13.3767),
    "columbiahalle": (52.4859, 13.4040),
    "admiralspalast": (52.5205, 13.3906),
}

STATION_COLOR = "#1f77b4"


# ===========================================================================
# Hilfsfunktionen
# ===========================================================================

def log(msg: str) -> None:
    print(f"[analyze_ubahn] {msg}")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distanz zwischen zwei WGS84-Punkten in Kilometern."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_duration(duration_str) -> pd.Timedelta:
    """Parst Strings wie '3h30min', '2h', '45min' zu einem Timedelta."""
    if pd.isna(duration_str):
        return pd.Timedelta(0)
    s = str(duration_str).strip().lower()
    m = re.match(r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*min)?", s)
    hours = int(m.group(1)) if m and m.group(1) else 0
    minutes = int(m.group(2)) if m and m.group(2) else 0
    return pd.Timedelta(hours=hours, minutes=minutes)


def lookup_venue_coords(venue_name: str, address: str = ""):
    """Sucht Koordinaten fuer einen Venue-Namen in der bekannten Lookup-Tabelle
    (schnell, offline, kein Netzwerk noetig)."""
    key = str(venue_name).strip().lower()
    if key in KNOWN_VENUE_COORDS:
        return KNOWN_VENUE_COORDS[key]
    for known_key, coords in KNOWN_VENUE_COORDS.items():
        if known_key in key or key in known_key:
            return coords
    return None


class GeoCache:
    """Persistenter Cache fuer Venue -> (lat, lon), damit echtes Geocoding
    (OpenStreetMap/Nominatim) nur einmal pro einzigartigem Venue passiert.
    Wird als JSON im Output-Verzeichnis gespeichert, ist also zwischen
    Skript-Laeufen wiederverwendbar."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                log(f"Konnte Geocoding-Cache nicht lesen ({exc}), starte leer.")

    def get(self, key: str):
        if key in self.data:
            val = self.data[key]
            return tuple(val) if val is not None else None
        return "__MISS__"

    def set(self, key: str, coords) -> None:
        self.data[key] = list(coords) if coords is not None else None

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log(f"Konnte Geocoding-Cache nicht speichern: {exc}")


def geocode_nominatim(query: str):
    """Fragt die kostenlose OpenStreetMap-Nominatim-API ab (kein API-Key
    noetig). Gibt (coords_or_None, request_ok) zurueck: request_ok=False
    bedeutet ein technisches Problem (offline, Timeout, blockiert) - dieser
    Fall wird NICHT gecacht, damit ein spaeterer Lauf mit funktionierendem
    Internet es erneut versucht. request_ok=True mit coords=None heisst:
    die Anfrage kam durch, aber es gab wirklich keinen Treffer - das wird
    gecacht, um nicht wiederholt sinnlos anzufragen."""
    try:
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({
            "q": query, "format": "json", "limit": 1,
        })
        req = urllib.request.Request(url, headers={"User-Agent": "innotrans-hackathon-analysis/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload:
            return (float(payload[0]["lat"]), float(payload[0]["lon"])), True
        return None, True  # Anfrage kam durch, aber kein Treffer
    except Exception as exc:  # noqa: BLE001
        log(f"Geocoding-Anfrage fehlgeschlagen fuer '{query}': {exc} (wird spaeter erneut versucht)")
        return None, False


def resolve_venue_coords(venue_name: str, address: str, city: str, country: str,
                          cache: "GeoCache", use_geocode: bool):
    """Loest einen Venue-Namen zu (lat, lon) auf: 1) bekannte Lookup-Tabelle
    (schnell), 2) Cache (schnell), 3) optional Live-Geocoding via Nominatim
    (langsam, braucht Internet, wird gecacht)."""
    known = lookup_venue_coords(venue_name, address)
    if known is not None:
        return known

    cache_key = f"{venue_name}|{address}".strip().lower()
    cached = cache.get(cache_key)
    if cached != "__MISS__":
        return cached

    if not use_geocode:
        return None

    query = ", ".join(p for p in [venue_name, address, city, country] if p and str(p) != "nan")
    coords, request_ok = geocode_nominatim(query)
    time.sleep(1.05)  # Nominatim Nutzungsrichtlinie: max. 1 Anfrage/Sekunde
    if request_ok:
        cache.set(cache_key, coords)
    return coords


def safe_savefig(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"Plot gespeichert: {path}")
    return str(path)


class Report:
    """Sammelt Markdown-Abschnitte und schreibt sie am Ende in eine Datei."""

    def __init__(self):
        self.sections: list[str] = []

    def add(self, markdown: str) -> None:
        self.sections.append(markdown.rstrip() + "\n")

    def add_table(self, df: pd.DataFrame, max_rows: int = 20) -> None:
        if df is None or df.empty:
            self.add("_Keine Daten verfuegbar._")
            return
        self.add(df.head(max_rows).to_markdown(index=False))

    def write(self, path: Path) -> None:
        path.write_text("\n\n".join(self.sections), encoding="utf-8")
        log(f"Report geschrieben: {path}")


# ===========================================================================
# 1. Daten laden & vorverarbeiten
# ===========================================================================

def load_data(data_dir: Path) -> dict:
    """Laedt die 8 Datensaetze. Sucht dabei robust nach der passenden Datei,
    egal ob sie exakt wie im Schema heisst (`flows.csv`) oder eine
    Variante mit Suffix traegt (z.B. `flows_pre_innotrans.csv`,
    `flows_pre_i....csv`, wie im Trainings-/Testdatensatz ueblich)."""
    # (key, [Kandidaten-Glob-Muster in Prioritaetsreihenfolge])
    patterns = {
        "stations": ["stations_with_ubahn*.csv", "*stations*ubahn*.csv"],
        "connections": ["berlin_ubahn_connections*.csv", "*connections*.csv"],
        "lines": ["berlin_ubahn_lines_used*.csv", "*lines_used*.csv", "*ubahn_lines*.csv"],
        "flows": ["flows*.csv"],
        "events": ["berlin_events_summer*.csv", "*events*.csv"],
        "closures": ["closures*.csv"],
        "weather": ["weather_data*.csv", "*weather*.csv"],
        "energy": ["energy_consumption*.csv", "*energy*.csv"],
    }
    data = {}
    for key, globs in patterns.items():
        match = None
        for pattern in globs:
            hits = sorted(data_dir.glob(pattern))
            if hits:
                match = hits[0]
                if len(hits) > 1:
                    log(f"Hinweis: mehrere Dateien passen zu '{key}' ({[h.name for h in hits]}), "
                        f"nehme '{match.name}'.")
                break
        if match is None:
            log(f"WARNUNG: keine Datei fuer '{key}' in {data_dir} gefunden "
                f"(gesucht: {globs}) -> wird uebersprungen.")
            continue
        try:
            data[key] = pd.read_csv(match)
            log(f"geladen: {key:<12s} <- {match.name:<45s} "
                f"{data[key].shape[0]:>7d} Zeilen x {data[key].shape[1]:>3d} Spalten")
        except Exception as exc:  # noqa: BLE001
            log(f"FEHLER beim Laden von {match}: {exc}")
    if "flows" not in data:
        sys.exit("Keine flows*.csv gefunden - das ist Pflicht fuer diese Analyse. Abbruch.")
    return data


def preprocess(data: dict) -> dict:
    if "flows" in data:
        f = data["flows"]
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        data["flows"] = f.sort_values("timestamp").reset_index(drop=True)

    if "weather" in data:
        w = data["weather"]
        first_col = w.columns[0]
        if first_col.lower() != "timestamp":
            w = w.rename(columns={first_col: "timestamp"})
        w["timestamp"] = pd.to_datetime(w["timestamp"])
        data["weather"] = w.sort_values("timestamp").reset_index(drop=True)

    if "events" in data:
        e = data["events"].copy()
        for col in ("began_local", "estimated_end_local"):
            if col in e.columns:
                e[col] = pd.to_datetime(e[col], utc=True, errors="coerce")
        # Wand-Uhrzeit Berlin, naiv gemacht, damit sie zu flows/weather passt
        if "began_local" in e.columns:
            e["began_naive"] = e["began_local"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
        if "estimated_end_local" in e.columns:
            e["end_naive"] = e["estimated_end_local"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
            e["end_naive"] = e["end_naive"].fillna(e["began_naive"] + pd.Timedelta(hours=2))
        data["events"] = e

    if "closures" in data:
        c = data["closures"].copy()
        c["when"] = pd.to_datetime(c["when"])
        c["duration_td"] = c["duration"].apply(parse_duration)
        c["end"] = c["when"] + c["duration_td"]
        data["closures"] = c

    if "energy" in data:
        en = data["energy"].copy()
        first_col = en.columns[0]
        if first_col.lower() != "timestamp":
            en = en.rename(columns={first_col: "timestamp"})
        # Datumsformat kann je nach Export variieren (MM-DD-YYYY laut Schema,
        # manche Exports liefern aber ISO o.ae.) - robust mit Fallback parsen.
        parsed = pd.to_datetime(en["timestamp"], format="%m-%d-%Y", errors="coerce")
        if parsed.isna().mean() > 0.5:  # Format passt offenbar nicht -> generisch parsen
            parsed = pd.to_datetime(en["timestamp"], errors="coerce")
        en["timestamp"] = parsed
        data["energy"] = en.sort_values("timestamp").reset_index(drop=True)

    return data


def get_station_columns(flows_df: pd.DataFrame) -> list:
    return [c for c in flows_df.columns if c != "timestamp"]


def build_graph(stations_df: pd.DataFrame, connections_df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for _, row in stations_df.iterrows():
        g.add_node(row["station_id"], name=row["station_name"],
                   lon=row["longitude"], lat=row["latitude"],
                   lines=row.get("u_bahn_lines", ""))
    if connections_df is not None:
        for _, row in connections_df.iterrows():
            if row["station_id_1"] in g.nodes and row["station_id_2"] in g.nodes:
                g.add_edge(row["station_id_1"], row["station_id_2"])
    return g


# ===========================================================================
# 2. Visualisierungen
# ===========================================================================

def plot_total_flow_timeseries(flows_df: pd.DataFrame, station_cols: list, out_dir: Path) -> str:
    total = flows_df[station_cols].sum(axis=1)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(flows_df["timestamp"], total, color=STATION_COLOR, linewidth=0.8)
    ax.set_title("Gesamter Fahrgastfluss ueber die Zeit (alle Stationen)")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Passagiere / 15 min")
    return safe_savefig(fig, out_dir / "01_total_flow_timeseries.png")


def plot_hourly_weekday_heatmap(flows_df: pd.DataFrame, station_cols: list, out_dir: Path) -> str:
    df = flows_df.copy()
    df["total"] = df[station_cols].sum(axis=1)
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.pivot_table(index="weekday", columns="hour", values="total", aggfunc="mean").reindex(order)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Ø Passagiere / 15 min"})
    ax.set_title("Durchschnittlicher Fluss nach Wochentag und Stunde")
    ax.set_xlabel("Stunde")
    ax.set_ylabel("")
    return safe_savefig(fig, out_dir / "02_hourly_weekday_heatmap.png")


def plot_top_stations(flows_df: pd.DataFrame, station_cols: list, out_dir: Path, top_n: int = 10) -> str:
    means = flows_df[station_cols].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 0.45 * top_n + 1.5))
    sns.barplot(x=means.values, y=means.index, ax=ax, color=STATION_COLOR)
    ax.set_title(f"Top {top_n} Stationen nach Ø Fahrgastfluss")
    ax.set_xlabel("Ø Passagiere / 15 min")
    return safe_savefig(fig, out_dir / "03_top_stations.png")


def plot_network_graph(g: nx.Graph, flows_df: pd.DataFrame, out_dir: Path) -> str:
    station_cols = get_station_columns(flows_df)
    mean_flow = flows_df[station_cols].mean()
    pos = {n: (attrs["lon"], attrs["lat"]) for n, attrs in g.nodes(data=True)}
    sizes, colors = [], []
    for n, attrs in g.nodes(data=True):
        m = mean_flow.get(attrs["name"], np.nan)
        sizes.append(60 if np.isnan(m) else 40 + m / 6)
        colors.append(0 if np.isnan(m) else m)
    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#999999", width=1.2, alpha=0.7)
    nodes = nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color=colors,
                                    cmap="YlOrRd", linewidths=0.5, edgecolors="black")
    labels = {n: attrs["name"] for n, attrs in g.nodes(data=True)}
    nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=6)
    ax.set_title("U-Bahn-Netz eingefaerbt nach durchschnittlichem Fahrgastfluss")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if len(colors) and np.nanmax(colors) > 0:
        fig.colorbar(nodes, ax=ax, label="Ø Passagiere / 15 min", shrink=0.8)
    return safe_savefig(fig, out_dir / "04_network_graph.png")


def plot_weather_vs_flow(merged: pd.DataFrame, out_dir: Path) -> str:
    vars_to_plot = [v for v in ["temp", "prcp", "rhum", "wspd"] if v in merged.columns]
    n = len(vars_to_plot)
    if n == 0:
        return ""
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, var in zip(axes, vars_to_plot):
        sns.regplot(data=merged, x=var, y="total_flow", ax=ax, scatter_kws={"alpha": 0.15, "s": 8},
                    line_kws={"color": "red"})
        ax.set_title(f"Fluss vs. {var}")
    fig.suptitle("Fahrgastfluss vs. Wettervariablen (mit linearer Trendlinie)")
    return safe_savefig(fig, out_dir / "05_weather_vs_flow.png")


def plot_energy_per_line(energy_df: pd.DataFrame, out_dir: Path) -> str:
    line_cols = [c for c in energy_df.columns if c != "timestamp"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col in line_cols:
        ax.plot(energy_df["timestamp"], energy_df[col], marker="o", markersize=2, label=col)
    ax.set_title("Taeglicher Energieverbrauch pro Linie")
    ax.set_xlabel("Datum")
    ax.set_ylabel("MWh")
    ax.legend(ncol=min(len(line_cols), 6), fontsize=8)
    return safe_savefig(fig, out_dir / "06_energy_per_line.png")


def plot_events_timeline(events_df: pd.DataFrame, out_dir: Path) -> str:
    df = events_df.dropna(subset=["began_naive"]).sort_values("began_naive")
    if df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(df) + 1.5))
    for i, (_, row) in enumerate(df.iterrows()):
        end = row.get("end_naive", row["began_naive"] + pd.Timedelta(hours=2))
        ax.barh(i, (end - row["began_naive"]).total_seconds() / 3600,
                left=mdates_to_num(row["began_naive"]), color="#d62728", alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{n[:35]}" for n in df["event_name"]], fontsize=8)
    ax.xaxis_date()
    ax.set_title("Event-Zeitleiste")
    ax.set_xlabel("Datum")
    return safe_savefig(fig, out_dir / "07_events_timeline.png")


def mdates_to_num(ts):
    import matplotlib.dates as mdates
    return mdates.date2num(ts)


# ===========================================================================
# 3. Korrelationsanalyse
# ===========================================================================
import pandas as pd
import numpy as np
from scipy.stats import trim_mean
from typing import Optional, Set


def merge_flows_weather(
    flows_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    station_cols: list,
    closures_df: Optional[pd.DataFrame] = None,
    seasonal: bool = True,
    robust: bool = True,
    trim_fraction: float = 0.2,
) -> pd.DataFrame:
    """
    Verbindet Fluss- und Wetterdaten und ergänzt eine intelligente 'Fluss-Anomalie':
    die Abweichung vom üblichen Fluss zu genau dieser Uhrzeit/diesem Wochentag.
    
    Die klassische Anomalie = (Ist-Fluss) - (Durchschnitt an diesem Wochentag/Uhrzeit).
    Das tägliche Pendlermuster überlagert Wettereffekte; Anomalie rechnet es heraus,
    damit der reine Wettereffekt sichtbar wird.
    
    VERBESSERUNGEN:
    1. **Robuste Baseline**: Nutzt trimmed mean statt simplem mean → weniger anfällig
       für Ausreißer (z.B. Event-Tage, Störungen).
    2. **Closure-Tage ausschließen**: Berechnet Baseline nur auf "sauberen" Tagen
       → repräsentativer für Normal-Betrieb.
    3. **Saisonalität**: Optional separate Baselines pro Monat → bessere Abbildung
       von Sommer vs. Winter, Schulferien etc.
    
    Args:
        flows_df: DataFrame mit Fluss-Daten (Spalten = Stationen).
        weather_df: DataFrame mit Wetter-Daten.
        station_cols: Liste der Stationsspalten zur Summation.
        closures_df: Optional — DataFrame mit Closures/Disruptions.
                      Columns: 'when' (datetime), 'duration' (string).
                      Wenn gegeben, werden diese Tage von Baseline-Berechnung ausgeschlossen.
        seasonal: bool (default True). Wenn True, separate Baselines pro Monat.
        robust: bool (default True). Wenn True, trimmed mean; sonst einfacher mean.
        trim_fraction: float (default 0.2). Fraktion der oberen/unteren Daten
                       zum Trimmen (nur wenn robust=True).
    
    Returns:
        merged DataFrame mit Spalten:
            - total_flow: Summe aller Stationen
            - flow_baseline: Erwartete Fluss-Norm für (weekday, hour, minute)
            - flow_anomaly: total_flow - flow_baseline
            - flow_anomaly_pct: Anomalie in Prozent
            - plus optional: month (wenn seasonal=True)
    """
    merged = flows_df.merge(weather_df, on="timestamp", how="inner")
    merged["total_flow"] = merged[station_cols].sum(axis=1)
    merged["hour"] = merged["timestamp"].dt.hour
    merged["minute"] = merged["timestamp"].dt.minute
    merged["weekday"] = merged["timestamp"].dt.dayofweek
    
    if seasonal:
        merged["month"] = merged["timestamp"].dt.month
    
    # --- SCHRITT 1: Bestimme Tage, die von Baseline-Berechnung ausgeschlossen sind ---
    affected_dates: Set = set()
    if closures_df is not None and not closures_df.empty:
        # Closures: nutze die 'when'-Spalte
        if "when" in closures_df.columns:
            affected_dates.update(closures_df["when"].dt.date)
    
    # Zusätzlich optional: Events können auch ausgeschlossen werden
    # affected_dates.update(events_df["began_local"].dt.date)  # Wenn Events-DF verfügbar
    
    # --- SCHRITT 2: Berechne Baseline auf "sauberen" Daten ---
    baseline_data = merged[~merged["timestamp"].dt.date.isin(affected_dates)].copy()
    
    # Gruppierungs-Spalten je nach Konfiguration
    group_cols = ["weekday", "hour", "minute"]
    if seasonal:
        group_cols.insert(0, "month")
    
    # Aggregations-Funktion
    if robust:
        def agg_func(x):
            if len(x) > 3:
                # Trimmed mean: Ignoriere untere/obere trim_fraction
                return trim_mean(x, trim_fraction)
            else:
                # Zu wenig Samples → fallback zu einfachem mean
                return x.mean()
    else:
        agg_func = "mean"
    
    # Berechne Baseline
    baseline_values = baseline_data.groupby(group_cols)["total_flow"].apply(agg_func)
    
    # --- SCHRITT 3: Joinde Baseline zurück auf alle Rows ---
    merged = merged.merge(
        baseline_values.rename("flow_baseline"),
        left_on=group_cols,
        right_index=True,
        how="left"
    )
    
    # Falls vereinzelte Slots keine Baseline haben (zu wenig/keine Daten),
    # nutze einen globalen Fallback
    global_fallback = baseline_data["total_flow"].mean()
    merged["flow_baseline"] = merged["flow_baseline"].fillna(global_fallback)
    
    # --- SCHRITT 4: Berechne Anomalien ---
    merged["flow_anomaly"] = merged["total_flow"] - merged["flow_baseline"]
    merged["flow_anomaly_pct"] = (
        100 * merged["flow_anomaly"] / merged["flow_baseline"].replace(0, np.nan)
    )
    
    return merged


# ============================================================================
# ALTERNATIVE: Wenn du auch Event-Ausschlag berücksichtigen möchtest
# ============================================================================

def merge_flows_weather_with_events(
    flows_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    station_cols: list,
    events_df: pd.DataFrame,
    closures_df: Optional[pd.DataFrame] = None,
    min_attendance_to_exclude: int = 20000,
    seasonal: bool = True,
    robust: bool = True,
    trim_fraction: float = 0.2,
    min_days_per_slot: int = 5,
) -> pd.DataFrame:
    merged = flows_df.merge(weather_df, on="timestamp", how="inner")
    merged["total_flow"] = merged[station_cols].sum(axis=1)
    merged["hour"] = merged["timestamp"].dt.hour
    merged["minute"] = merged["timestamp"].dt.minute
    merged["weekday"] = merged["timestamp"].dt.dayofweek
    merged["date"] = merged["timestamp"].dt.date

    if seasonal:
        merged["month"] = merged["timestamp"].dt.month

    group_cols = ["weekday", "hour", "minute"]
    if seasonal:
        group_cols.insert(0, "month")

    # --- Tage sammeln, die ausgeschlossen werden sollen ---
    affected_dates: Set = set()

    if closures_df is not None and not closures_df.empty and "when" in closures_df.columns:
        affected_dates.update(pd.to_datetime(closures_df["when"]).dt.date)

    if events_df is not None and not events_df.empty and "began_local" in events_df.columns:
        big_events = events_df
        if "estimated_attendance" in events_df.columns:
            big_events = events_df[events_df["estimated_attendance"] >= min_attendance_to_exclude]
        affected_dates.update(pd.to_datetime(big_events["began_local"]).dt.date)

    # --- Aggregationsfunktion ---
    if robust:
        def agg_func(x):
            return trim_mean(x, trim_fraction) if len(x) > 3 else x.mean()
    else:
        agg_func = "mean"

    # --- Baseline 1: ungefiltert (Fallback pro Slot, NIE eine globale Konstante) ---
    unfiltered_baseline = merged.groupby(group_cols)["total_flow"].apply(agg_func)
    unfiltered_baseline.name = "flow_baseline_unfiltered"

    # --- Baseline 2: gefiltert (ohne Closure-/Event-Tage) ---
    clean = merged[~merged["date"].isin(affected_dates)].copy()

    # Anzahl übriger eindeutiger Tage pro Slot prüfen
    days_per_slot = clean.groupby(group_cols)["date"].nunique()
    filtered_baseline = clean.groupby(group_cols)["total_flow"].apply(agg_func)
    filtered_baseline.name = "flow_baseline_filtered"

    baseline_df = pd.concat([unfiltered_baseline, filtered_baseline, days_per_slot], axis=1)
    baseline_df.columns = ["flow_baseline_unfiltered", "flow_baseline_filtered", "n_clean_days"]
    baseline_df["n_clean_days"] = baseline_df["n_clean_days"].fillna(0)

    # Nur nutzen, wenn genug saubere Tage übrig sind - sonst ungefiltert nehmen
    enough_data = baseline_df["n_clean_days"] >= min_days_per_slot
    baseline_df["flow_baseline"] = np.where(
        enough_data,
        baseline_df["flow_baseline_filtered"],
        baseline_df["flow_baseline_unfiltered"],
    )
    # Falls auch die ungefilterte Baseline für einen Slot fehlt (sollte kaum vorkommen)
    baseline_df["flow_baseline"] = baseline_df["flow_baseline"].fillna(
        baseline_df["flow_baseline_unfiltered"]
    )

    merged = merged.merge(
        baseline_df[["flow_baseline"]],
        left_on=group_cols,
        right_index=True,
        how="left",
    )

    # --- Anomalien berechnen ---
    merged["flow_anomaly"] = merged["total_flow"] - merged["flow_baseline"]
    merged["flow_anomaly_pct"] = (
        100 * merged["flow_anomaly"] / merged["flow_baseline"].replace(0, np.nan)
    )

    merged = merged.drop(columns=["date"])
    return merged


def correlation_weather_flow(merged: pd.DataFrame, report: Report) -> pd.DataFrame:
    """Korreliert jede Wettervariable sowohl mit dem rohen Gesamtfluss als
    auch mit der Fluss-Anomalie. Die Anomalie-Spalte ist meist
    aussagekraeftiger, weil sie den dominanten Tages-/Wochenrhythmus
    herausrechnet."""
    weather_vars = [v for v in ["temp", "rhum", "prcp", "wdir", "wspd", "pres", "cldc"] if v in merged.columns]
    rows = []
    for var in weather_vars:
        sub = merged[[var, "total_flow", "flow_anomaly"]].dropna()
        if len(sub) < 5:
            continue
        r_raw, p_raw = stats.pearsonr(sub[var], sub["total_flow"])
        r_ano, p_ano = stats.pearsonr(sub[var], sub["flow_anomaly"])
        rows.append({
            "variable": var,
            "r_roh": round(r_raw, 3), "p_roh": round(p_raw, 4),
            "r_bereinigt": round(r_ano, 3), "p_bereinigt": round(p_ano, 4),
            "n": len(sub),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("r_bereinigt", key=lambda s: s.abs(), ascending=False)

    report.add(
        "## Korrelation: Wetter <-> Fahrgastfluss\n\n"
        "**`r_roh`** korreliert die Wettervariable direkt mit dem summierten 15-Minuten-Fluss "
        "- meist klein, weil das tagesperiodische Pendlermuster (Stosszeiten) alles ueberlagert.\n\n"
        "**`r_bereinigt`** korreliert stattdessen mit der *Fluss-Anomalie* (Abweichung vom "
        "ueblichen Fluss zur selben Uhrzeit/demselben Wochentag) - hier zeigt sich der "
        "eigentliche Wettereffekt klarer. Das ist die aussagekraeftigere Spalte.\n\n"
        "**Faustregel fuer |r|:** < 0.1 sehr schwach · 0.1-0.3 schwach · 0.3-0.5 moderat · "
        "> 0.5 stark. `p < 0.05` gilt als statistisch signifikant."
    )
    report.add_table(result)
    return result


def plot_weather_bins(merged: pd.DataFrame, out_dir: Path) -> list:
    """Einfache, gut lesbare Balkendiagramme: durchschnittliche Fluss-Anomalie
    (in %) je Wetter-Kategorie (z.B. 'Kein Regen' / 'Regen' / 'Starker Regen').
    Fehlerbalken = 95%-Konfidenzintervall des Mittelwerts. Das macht den
    Wettereffekt auf einen Blick verstaendlich, ganz ohne Korrelationskoeffizient
    lesen zu muessen."""
    paths = []

    def _bin_and_plot(col, bins, labels, title, fname):
        if col not in merged.columns:
            return
        df = merged.dropna(subset=[col, "flow_anomaly_pct"]).copy()
        df["bin"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        grouped = df.groupby("bin", observed=True)["flow_anomaly_pct"]
        means, sems = grouped.mean(), grouped.sem()
        ci95 = sems.fillna(0) * 1.96
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ["#d62728" if m < 0 else "#2ca02c" for m in means]
        ax.bar(means.index.astype(str), means.values, yerr=ci95.values, capsize=4,
               color=colors, alpha=0.85)
        ax.axhline(0, color="grey", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Fluss-Anomalie ggue. Erwartungswert (%)")
        paths.append(safe_savefig(fig, out_dir / fname))

    if "prcp" in merged.columns:
        _bin_and_plot(
            "prcp", bins=[-0.01, 0, 1, 3, np.inf],
            labels=["Trocken", "Leichter Regen\n(0-1mm)", "Regen\n(1-3mm)", "Starker Regen\n(>3mm)"],
            title="Fluss-Anomalie nach Niederschlag", fname="05b_weather_bins_prcp.png",
        )
    if "temp" in merged.columns:
        _bin_and_plot(
            "temp", bins=[-np.inf, 5, 15, 25, np.inf],
            labels=["Kalt (<5°C)", "Kuehl (5-15°C)", "Mild (15-25°C)", "Heiss (>25°C)"],
            title="Fluss-Anomalie nach Temperatur", fname="05c_weather_bins_temp.png",
        )
    return paths


def correlation_weather_flow_daily(flows_df: pd.DataFrame, weather_df: pd.DataFrame, station_cols: list,
                                    report: Report, out_dir: Path) -> pd.DataFrame:
    """Aggregiert Fluss und Wetter auf Tagesebene (robuster/ruhiger als
    einzelne 15-Min-Werte, ein Punkt pro Tag) und korreliert die
    Tageswerte - eine einfache, ergaenzende Sicht neben der 15-Min-Analyse."""
    f = flows_df.copy()
    f["date"] = f["timestamp"].dt.date
    daily_flow = f.groupby("date")[station_cols].sum().sum(axis=1).rename("daily_total_flow")

    w = weather_df.copy()
    w["date"] = w["timestamp"].dt.date
    agg_map = {}
    if "temp" in w.columns:
        agg_map["temp_mean"] = ("temp", "mean")
    if "prcp" in w.columns:
        agg_map["prcp_sum"] = ("prcp", "sum")
    if "rhum" in w.columns:
        agg_map["rhum_mean"] = ("rhum", "mean")
    if "wspd" in w.columns:
        agg_map["wspd_mean"] = ("wspd", "mean")
    if "cldc" in w.columns:
        agg_map["cldc_mean"] = ("cldc", "mean")
    if not agg_map:
        return pd.DataFrame()
    daily_weather = w.groupby("date").agg(**agg_map)
    daily = daily_weather.join(daily_flow, how="inner").reset_index()

    rows = []
    for var in agg_map:
        sub = daily[[var, "daily_total_flow"]].dropna()
        if len(sub) < 5:
            continue
        r, p = stats.pearsonr(sub[var], sub["daily_total_flow"])
        rows.append({"tages_variable": var, "pearson_r": round(r, 3), "pearson_p": round(p, 4),
                     "n_tage": len(sub)})
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)

    report.add(
        "## Wetter <-> Fluss auf Tagesebene\n\n"
        "Robustere, einfachere Zusatz-Sicht: ein Punkt pro Tag statt 96 verrauschte "
        "15-Minuten-Werte. `temp_mean`/`rhum_mean`/`wspd_mean`/`cldc_mean` sind "
        "Tagesmittelwerte, `prcp_sum` ist die Tages-Niederschlagssumme."
    )
    report.add_table(table)

    plot_vars = [v for v in ["temp_mean", "prcp_sum"] if v in daily.columns]
    if plot_vars:
        fig, axes = plt.subplots(1, len(plot_vars), figsize=(5.5 * len(plot_vars), 4.5))
        if len(plot_vars) == 1:
            axes = [axes]
        for ax, var in zip(axes, plot_vars):
            sns.regplot(data=daily, x=var, y="daily_total_flow", ax=ax,
                        scatter_kws={"alpha": 0.6, "s": 25}, line_kws={"color": "red"})
            ax.set_title(f"Taeglicher Fluss vs. {var}")
        fig.suptitle("Fahrgastfluss vs. Wetter - Tagesebene")
        safe_savefig(fig, out_dir / "05d_weather_vs_flow_daily.png")

    return table


def correlation_matrix_heatmap(merged: pd.DataFrame, out_dir: Path) -> str:
    cols = [v for v in ["total_flow", "flow_anomaly", "temp", "rhum", "prcp", "wdir", "wspd", "pres", "cldc"]
            if v in merged.columns]
    corr = merged[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Korrelationsmatrix: Fluss (roh & Anomalie) & Wetter")
    return safe_savefig(fig, out_dir / "08_correlation_matrix.png")


def lagged_cross_correlation(merged: pd.DataFrame, var: str, out_dir: Path,
                              max_lag_steps: int = 16, y_col: str = "flow_anomaly") -> tuple:
    """Kreuzkorrelation zwischen `var` und der Fluss-Anomalie (Standard) fuer
    Lags von -max..+max Zeitschritten (Schrittweite = Grain der `merged`-Tabelle,
    i.d.R. 15 min). Positiver Lag bedeutet: `var` zur Zeit t korreliert mit dem
    Fluss zur Zeit t+lag (d.h. `var` geht dem Fluss voraus). Standardmaessig
    gegen die Anomalie statt den Rohfluss, damit das Ergebnis nicht durch den
    taeglichen Pendelrhythmus verzerrt wird (der sonst jede Kreuzkorrelation
    dominiert)."""
    if y_col not in merged.columns:
        y_col = "total_flow"
    x = merged[var].to_numpy()
    y = merged[y_col].to_numpy()
    lags = range(-max_lag_steps, max_lag_steps + 1)
    values = []
    for lag in lags:
        if lag < 0:
            a, b = x[:lag], y[-lag:]
        elif lag > 0:
            a, b = x[lag:], y[:-lag]
        else:
            a, b = x, y
        if len(a) < 10:
            values.append(np.nan)
            continue
        r = np.corrcoef(a, b)[0, 1]
        values.append(r)
    result = pd.DataFrame({"lag_steps": list(lags), "correlation": values})
    best = result.iloc[result["correlation"].abs().idxmax()]

    y_label = "Fluss-Anomalie" if y_col == "flow_anomaly" else "Gesamtfluss"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stem(result["lag_steps"], result["correlation"])
    ax.axvline(0, color="grey", linewidth=1)
    ax.set_title(f"Zeitversetzte Korrelation: {var} vs. {y_label}\n"
                 f"(stärkste Korrelation bei Lag={int(best['lag_steps'])}: r={best['correlation']:.2f})")
    ax.set_xlabel(f"Lag (Zeitschritte von '{var}' vor dem Fluss)")
    ax.set_ylabel("Korrelation")
    path = safe_savefig(fig, out_dir / f"09_lagged_corr_{var}.png")
    return path, result


def station_pair_correlation(flows_df: pd.DataFrame, station_cols: list, g: nx.Graph,
                              top_k: int = 15) -> pd.DataFrame:
    """Findet Stationspaare mit hoher Korrelation, die NICHT direkt per
    U-Bahn verbunden sind - potenzielle 'versteckte Abhaengigkeiten'
    (siehe Trainingsfrage 8)."""
    corr = flows_df[station_cols].corr(method="pearson")
    name_to_id = {attrs["name"]: n for n, attrs in g.nodes(data=True)}
    adjacent_pairs = set()
    for a, b in g.edges():
        name_a, name_b = g.nodes[a]["name"], g.nodes[b]["name"]
        adjacent_pairs.add(frozenset((name_a, name_b)))

    rows = []
    for s1, s2 in itertools.combinations(station_cols, 2):
        if frozenset((s1, s2)) in adjacent_pairs:
            continue
        r = corr.loc[s1, s2]
        if pd.isna(r):
            continue
        rows.append({"station_a": s1, "station_b": s2, "correlation": round(r, 3)})
    result = pd.DataFrame(rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False).head(top_k)
    return result.reset_index(drop=True)


def plot_top_pair_timeseries(flows_df: pd.DataFrame, pair_row: pd.Series, out_dir: Path) -> str:
    s1, s2 = pair_row["station_a"], pair_row["station_b"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax2 = ax.twinx()
    ax.plot(flows_df["timestamp"], flows_df[s1], color="#1f77b4", linewidth=0.8, label=s1)
    ax2.plot(flows_df["timestamp"], flows_df[s2], color="#d62728", linewidth=0.8, alpha=0.7, label=s2)
    ax.set_ylabel(s1, color="#1f77b4")
    ax2.set_ylabel(s2, color="#d62728")
    ax.set_title(f"Nicht direkt verbundenes, stark korrelierendes Paar: {s1} <-> {s2} "
                 f"(r={pair_row['correlation']:.2f})")
    return safe_savefig(fig, out_dir / "10_top_correlated_nonadjacent_pair.png")


# ===========================================================================
# 4. Kausalitaets- / Wirkungsanalyse
# ===========================================================================

def granger_causality_weather_to_flow(merged: pd.DataFrame, max_lag: int, report: Report) -> pd.DataFrame:
    report.add("## Granger-Kausalitaet: Wetter -> Fluss\n\n"
                "Testet, ob vergangene Werte einer Wettervariable die Vorhersage des "
                "Gesamtflusses statistisch signifikant verbessern (p < 0.05 = Hinweis auf "
                "Vorhersage-Kausalitaet in diese Richtung; kein Beweis fuer physikalische "
                "Verursachung).")
    if not HAS_STATSMODELS:
        report.add("_statsmodels nicht installiert - Granger-Tests uebersprungen. "
                    "Installieren mit `pip install statsmodels`._")
        return pd.DataFrame()

    weather_vars = [v for v in ["temp", "rhum", "prcp", "wspd", "pres", "cldc"] if v in merged.columns]
    rows = []
    for var in weather_vars:
        sub = merged[["total_flow", var]].dropna()
        if len(sub) < max_lag * 5:
            continue
        try:
            res = run_granger(sub[["total_flow", var]], max_lag)
            best_lag, best_p = min(
                ((lag, res[lag][0]["ssr_ftest"][1]) for lag in res), key=lambda t: t[1]
            )
            rows.append({"weather_var": var, "direction": f"{var} -> total_flow",
                         "best_lag": best_lag, "min_p_value": round(best_p, 4),
                         "significant_(p<0.05)": best_p < 0.05})
        except Exception as exc:  # noqa: BLE001
            log(f"Granger-Test fuer {var} fehlgeschlagen: {exc}")
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("min_p_value")
    report.add_table(result)
    return result


def granger_causality_station_pairs(flows_df: pd.DataFrame, pairs_df: pd.DataFrame,
                                     max_lag: int, report: Report, top_n: int = 5) -> pd.DataFrame:
    report.add("## Granger-Kausalitaet zwischen auffaelligen Stationspaaren\n\n"
                "Fuer die staerksten nicht-benachbarten korrelierten Paare: testet beide "
                "Richtungen, um zu sehen, welche Station 'vorausgeht'.")
    if not HAS_STATSMODELS or pairs_df.empty:
        report.add("_Uebersprungen (statsmodels fehlt oder keine Paare gefunden)._")
        return pd.DataFrame()

    rows = []
    for _, pair in pairs_df.head(top_n).iterrows():
        s1, s2 = pair["station_a"], pair["station_b"]
        sub = flows_df[[s1, s2]].dropna()
        for direction, cols in [(f"{s1} -> {s2}", [s2, s1]), (f"{s2} -> {s1}", [s1, s2])]:
            try:
                res = run_granger(sub[cols], max_lag)
                best_lag, best_p = min(
                    ((lag, res[lag][0]["ssr_ftest"][1]) for lag in res), key=lambda t: t[1]
                )
                rows.append({"direction": direction, "correlation": pair["correlation"],
                             "best_lag": best_lag, "min_p_value": round(best_p, 4),
                             "significant_(p<0.05)": best_p < 0.05})
            except Exception as exc:  # noqa: BLE001
                log(f"Granger-Test fuer {direction} fehlgeschlagen: {exc}")
    result = pd.DataFrame(rows)
    report.add_table(result)
    return result


def correlation_events_daily(flows_df: pd.DataFrame, events_df: pd.DataFrame, station_cols: list,
                              report: Report, out_dir: Path) -> pd.DataFrame:
    """Stadtweite Event<->Fluss-Korrelation OHNE Geocoding: fuer jeden Tag
    wird die Summe der `estimated_attendance` aller an diesem Tag
    stattfindenden Events mit dem gesamtstaedtischen Tages-Fluss
    korreliert. Erfasst damit ALLE Events (auch unbekannte Venues), nicht
    nur die per Koordinaten zuordenbaren - eine robuste Ergaenzung zur
    stationsgenauen Event-Impact-Analyse weiter unten.

    Wichtig: Events (v.a. Konzerte/Sport) liegen ueberproportional oft am
    Wochenende, das an sich schon einen anderen Fluss hat als Werktage.
    Ein naiver Vergleich 'Event-Tage vs. alle anderen Tage' wuerde diesen
    Wochentag-Effekt faelschlich dem Event zuschreiben. Deshalb wird hier
    - wie bei der Wetter-Anomalie - relativ zum ueblichen Fluss AM GLEICHEN
    WOCHENTAG (gemessen an Nicht-Event-Tagen) verglichen."""
    if "began_naive" not in events_df.columns:
        return pd.DataFrame()

    f = flows_df.copy()
    f["date"] = f["timestamp"].dt.date
    daily_flow = f.groupby("date")[station_cols].sum().sum(axis=1).rename("daily_total_flow")

    ev = events_df.dropna(subset=["began_naive"]).copy()
    ev["date"] = ev["began_naive"].dt.date
    attendance_col = "estimated_attendance" if "estimated_attendance" in ev.columns else None
    daily_events = ev.groupby("date").agg(
        n_events=("event_name", "count"),
        total_attendance=(attendance_col, "sum") if attendance_col else ("event_name", "count"),
    )

    daily = pd.DataFrame(index=daily_flow.index).join(daily_flow).join(daily_events, how="left")
    daily["n_events"] = daily["n_events"].fillna(0)
    daily["total_attendance"] = daily["total_attendance"].fillna(0)
    daily = daily.reset_index()
    daily["weekday"] = pd.to_datetime(daily["date"]).dt.dayofweek

    # Wochentag-Baseline NUR aus Nicht-Event-Tagen, um Event-Tage nicht in
    # ihre eigene Baseline einzurechnen.
    non_event_mask = daily["n_events"] == 0
    weekday_baseline = daily.loc[non_event_mask].groupby("weekday")["daily_total_flow"].mean()
    daily["weekday_baseline"] = daily["weekday"].map(weekday_baseline)
    daily["flow_anomaly_pct"] = (100 * (daily["daily_total_flow"] - daily["weekday_baseline"])
                                  / daily["weekday_baseline"])

    report.add(
        "## Event-Korrelation auf Tagesebene (alle Events, kein Geocoding noetig)\n\n"
        "Summe der erwarteten Besucherzahl aller Events pro Tag vs. gesamtstaedtischer "
        "Tages-Fluss. Das erfasst *alle* Events aus der Datei, unabhaengig davon, ob ihr "
        "Venue geocodiert werden konnte. **Wochentag-bereinigt:** verglichen wird die "
        "Abweichung vom ueblichen Fluss am selben Wochentag (Baseline aus Nicht-Event-Tagen), "
        "damit z.B. viele Wochenend-Konzerte nicht faelschlich als 'Events senken den Fluss' "
        "erscheinen, nur weil Wochenenden ohnehin ruhiger sind."
    )

    sub = daily[["total_attendance", "flow_anomaly_pct"]].dropna()
    rows = []
    if len(sub) >= 5 and sub["total_attendance"].std() > 0:
        r_raw, p_raw = stats.pearsonr(daily["total_attendance"], daily["daily_total_flow"])
        r_adj, p_adj = stats.pearsonr(sub["total_attendance"], sub["flow_anomaly_pct"])
        rows.append({"vergleich": "total_attendance vs. Fluss (roh)",
                     "r": round(r_raw, 3), "p_value": round(p_raw, 4), "n_tage": len(daily)})
        rows.append({"vergleich": "total_attendance vs. Fluss-Anomalie (wochentag-bereinigt)",
                     "r": round(r_adj, 3), "p_value": round(p_adj, 4), "n_tage": len(sub)})

    event_days = daily[daily["n_events"] > 0]
    non_event_days = daily[daily["n_events"] == 0]
    if len(event_days) >= 2 and len(non_event_days) >= 2:
        anomalies = event_days["flow_anomaly_pct"].dropna()
        if len(anomalies) >= 2:
            t_stat, p_val = stats.ttest_1samp(anomalies, 0)  # Test gegen Baseline=0
            rows.append({
                "vergleich": f"Tage MIT Event (n={len(event_days)}) vs. Wochentag-Baseline",
                "r": f"{anomalies.mean():+.1f}% Fluss-Anomalie",
                "p_value": round(p_val, 4), "n_tage": len(daily),
            })
    table = pd.DataFrame(rows)
    report.add_table(table)

    # Scatter: Attendance vs. wochentag-bereinigte Fluss-Anomalie
    if len(sub) >= 5:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.regplot(data=sub, x="total_attendance", y="flow_anomaly_pct", ax=ax,
                    scatter_kws={"alpha": 0.5, "s": 25}, line_kws={"color": "red"})
        ax.axhline(0, color="grey", linewidth=1)
        ax.set_title("Fluss-Anomalie (wochentag-bereinigt) vs. Summe erwarteter Event-Besucher")
        ax.set_xlabel("Summe estimated_attendance an diesem Tag")
        ax.set_ylabel("Fluss-Anomalie ggue. typischem Wochentag (%)")
        safe_savefig(fig, out_dir / "10b_event_attendance_vs_flow.png")

    # Balken: Event- vs. Nicht-Event-Tage (wochentag-bereinigt)
    if len(event_days) >= 2 and len(non_event_days) >= 2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        vals = [0.0, event_days["flow_anomaly_pct"].mean()]
        colors = ["#999999", "#d62728" if vals[1] < 0 else "#2ca02c"]
        ax.bar(["Ohne Event\n(Baseline)", "Mit Event(s)"], vals, color=colors)
        ax.axhline(0, color="grey", linewidth=1)
        ax.set_ylabel("Fluss-Anomalie ggue. typischem Wochentag (%)")
        ax.set_title("Tage mit vs. ohne Event (wochentag-bereinigt)")
        safe_savefig(fig, out_dir / "10c_event_days_vs_nonevent_days.png")

    # Aufschluesselung nach Event-Segment/Genre, falls vorhanden (ebenfalls wochentag-bereinigt)
    seg_col = "segment" if "segment" in ev.columns else ("genre" if "genre" in ev.columns else None)
    if seg_col:
        ev_seg = ev[["date", seg_col]].dropna()
        seg_rows = []
        for seg, grp in ev_seg.groupby(seg_col):
            seg_dates = set(grp["date"])
            seg_anomaly = daily[daily["date"].isin(seg_dates)]["flow_anomaly_pct"].dropna()
            if len(seg_anomaly) < 2:
                continue
            seg_rows.append({"segment": seg, "n_tage": len(seg_anomaly),
                             "uplift_vs_wochentag_baseline_%": round(seg_anomaly.mean(), 1)})
        if seg_rows:
            seg_table = pd.DataFrame(seg_rows).sort_values("uplift_vs_wochentag_baseline_%", ascending=False)
            report.add(f"### Fluss-Effekt nach Event-Kategorie (`{seg_col}`, wochentag-bereinigt)")
            report.add_table(seg_table, max_rows=25)
            fig, ax = plt.subplots(figsize=(8, 0.4 * len(seg_table) + 1.5))
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in seg_table["uplift_vs_wochentag_baseline_%"]]
            ax.barh(seg_table["segment"], seg_table["uplift_vs_wochentag_baseline_%"], color=colors)
            ax.axvline(0, color="grey", linewidth=1)
            ax.set_xlabel("Fluss-Anomalie ggue. typischem Wochentag (%)")
            ax.set_title(f"Fluss-Effekt nach Event-Kategorie ({seg_col}, wochentag-bereinigt)")
            safe_savefig(fig, out_dir / "10d_event_category_impact.png")

    return daily


def event_impact_analysis(flows_df: pd.DataFrame, events_df: pd.DataFrame, stations_df: pd.DataFrame,
                           radius_km: float, report: Report, out_dir: Path, cache: "GeoCache",
                           use_geocode: bool, pre_min: int = 60, post_min: int = 90,
                           show_top_n: int = 20) -> pd.DataFrame:
    """Stationsgenaue Event-Impact-Analyse: fuer jedes Event mit aufloesbarem
    Venue werden die Stationen im Umkreis `radius_km` bestimmt und ihr Fluss
    in der Stunde VOR Event-Beginn (Anreise-Fenster, Laenge `pre_min` Minuten)
    mit der Baseline (gleiche Uhrzeit, andere Tage) verglichen (Welch-t-Test).
    Venues werden zuerst gegen `KNOWN_VENUE_COORDS` und den Geocoding-Cache
    abgeglichen; ist `use_geocode` an, werden unbekannte Venues live ueber
    OpenStreetMap/Nominatim aufgeloest (braucht Internet, wird gecacht - kann
    bei vielen einzigartigen Venues einige Minuten dauern, da Nominatim
    max. 1 Anfrage/Sekunde erlaubt)."""
    report.add("## Event-Impact-Analyse (stationsgenau)\n\n"
                "Vergleicht den Fluss an Stationen nahe einem Venue **in der Stunde vor "
                "Event-Beginn** (Anreise-Fenster) mit dem typischen Fluss an denselben "
                "Stationen zur selben Tageszeit an anderen Tagen (Baseline, Welch-t-Test). "
                "Venues werden ueber eine bekannte Lookup-Tabelle" + (", einen Cache und "
                "Live-Geocoding (OpenStreetMap)" if use_geocode else " und einen Cache") +
                " aufgeloest.")
    rows = []
    ts = flows_df["timestamp"]
    unique_venues = events_df.get("venue_name", pd.Series(dtype=str)).dropna().nunique()
    log(f"Loese Koordinaten fuer bis zu {unique_venues} einzigartige Venues auf "
        f"({'mit' if use_geocode else 'ohne'} Live-Geocoding) ...")

    n_geocoded, n_matched = 0, 0
    for i, (_, ev) in enumerate(events_df.iterrows()):
        coords = resolve_venue_coords(ev.get("venue_name", ""), ev.get("address", ""),
                                       ev.get("city", ""), ev.get("country", ""), cache, use_geocode)
        if coords is not None:
            n_geocoded += 1
        if coords is None or pd.isna(ev.get("began_naive")):
            continue
        vlat, vlon = coords
        dists = stations_df.apply(lambda r: haversine_km(vlat, vlon, r["latitude"], r["longitude"]), axis=1)
        nearby = stations_df.loc[dists <= radius_km, "station_name"].tolist()
        if not nearby:
            nearby = [stations_df.loc[dists.idxmin(), "station_name"]]
        nearby = [s for s in nearby if s in flows_df.columns]
        if not nearby:
            continue

        start, end = ev["began_naive"], ev["end_naive"]
        # Anreise-Fenster: die `pre_min` Minuten VOR Event-Beginn - das ist
        # das relevante Fenster fuer die Impact-Analyse (statt "waehrend").
        pre_start = start - pd.Timedelta(minutes=pre_min)
        pre_mask = (ts >= pre_start) & (ts < start)
        during_mask = (ts >= start) & (ts <= end)
        post_mask = (ts > end) & (ts <= end + pd.Timedelta(minutes=post_min))

        pre_flow = flows_df.loc[pre_mask, nearby].sum(axis=1)
        during_flow = flows_df.loc[during_mask, nearby].sum(axis=1)
        post_flow = flows_df.loc[post_mask, nearby].sum(axis=1)
        # Baseline: gleiche Stunde wie das Anreise-Fenster (nicht die Event-Stunde
        # selbst), an anderen Tagen - damit der Vergleich zum Pre-Event-Fenster passt.
        same_hour_mask = (ts.dt.hour == pre_start.hour) & (~during_mask) & (~pre_mask) & (~post_mask)
        baseline_flow = flows_df.loc[same_hour_mask, nearby].sum(axis=1)

        if len(pre_flow) < 2 or len(baseline_flow) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(pre_flow, baseline_flow, equal_var=False)
        uplift_pct = 100 * (pre_flow.mean() - baseline_flow.mean()) / max(baseline_flow.mean(), 1e-9)
        n_matched += 1

        rows.append({
            "event": ev["event_name"], "venue": ev.get("venue_name", ""),
            "attendance": ev.get("estimated_attendance", np.nan),
            "nearby_stations": ", ".join(nearby[:4]) + ("..." if len(nearby) > 4 else ""),
            "pre_event_mean": round(pre_flow.mean(), 1),
            "baseline_mean": round(baseline_flow.mean(), 1),
            "uplift_vs_baseline_%": round(uplift_pct, 1),
            "p_value": round(p_val, 4),
            "significant_(p<0.05)": p_val < 0.05,
        })
        if (i + 1) % 25 == 0:
            log(f"  ... {i + 1}/{len(events_df)} Events verarbeitet")

    cache.save()
    result = pd.DataFrame(rows)
    n_events = len(events_df)
    report.add(f"**Zusammenfassung:** {n_geocoded}/{n_events} Venues aufgeloest, "
               f"{n_matched}/{n_events} Events tatsaechlich analysiert (Zeitfenster/Stationen vorhanden). "
               + (f"Davon {int(result['significant_(p<0.05)'].sum())} mit signifikantem Effekt "
                  f"(p<0.05), durchschnittlicher Uplift {result['uplift_vs_baseline_%'].mean():+.1f}%."
                  if not result.empty else "Keine Events konnten analysiert werden."))

    if not result.empty:
        top = result.reindex(result["uplift_vs_baseline_%"].abs().sort_values(ascending=False).index).head(show_top_n)
        report.add(f"### Top {min(show_top_n, len(result))} Events nach |Uplift|")
        report.add_table(top, max_rows=show_top_n)

        fig, ax = plt.subplots(figsize=(9, 0.4 * len(top) + 2))
        idx = np.arange(len(top))
        ax.barh(idx, top["baseline_mean"], color="#999999", label="Baseline (gleiche Uhrzeit)")
        ax.barh(idx, top["pre_event_mean"], color="#d62728", alpha=0.6, label="1h vor Event")
        ax.set_yticks(idx)
        ax.set_yticklabels(top["event"].str.slice(0, 40), fontsize=8)
        ax.set_xlabel("Ø Passagiere (Summe nahegelegener Stationen)")
        ax.set_title(f"Event-Impact: Top {len(top)} Events nach |Uplift| vs. Baseline")
        ax.legend()
        safe_savefig(fig, out_dir / "11_event_impact.png")

        # Attendance vs. Uplift - zeigt, ob groessere Events groesseren Effekt haben
        att = result.dropna(subset=["attendance", "uplift_vs_baseline_%"])
        if len(att) >= 5:
            r, p = stats.pearsonr(att["attendance"], att["uplift_vs_baseline_%"])
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.regplot(data=att, x="attendance", y="uplift_vs_baseline_%", ax=ax,
                        scatter_kws={"alpha": 0.5, "s": 25}, line_kws={"color": "red"})
            ax.set_title(f"Event-Groesse vs. Fluss-Uplift (r={r:.2f}, p={p:.3f})")
            ax.set_xlabel("Erwartete Besucherzahl (estimated_attendance)")
            ax.set_ylabel("Fluss-Uplift an nahegelegenen Stationen (%)")
            safe_savefig(fig, out_dir / "11b_event_attendance_vs_uplift.png")
    return result


def closure_impact_analysis(flows_df: pd.DataFrame, closures_df: pd.DataFrame, stations_df: pd.DataFrame,
                             g: nx.Graph, report: Report, out_dir: Path, pre_min: int = 60) -> pd.DataFrame:
    report.add("## Closure-Impact-Analyse\n\n"
                "Fuer jede Streckensperrung: vergleicht den Fluss an den (per Textabgleich "
                "erkannten) betroffenen Stationen und an deren direkten Nachbarn vor vs. "
                "waehrend der Sperrung. Ein Rueckgang an den gesperrten Stationen und ein "
                "Anstieg an Nachbarstationen ist ein Hinweis auf Umleitungsverhalten.")
    ts = flows_df["timestamp"]
    station_names = stations_df["station_name"].tolist()
    name_to_id = dict(zip(stations_df["station_name"], stations_df["station_id"]))

    rows = []
    for _, cl in closures_df.iterrows():
        desc = str(cl.get("description", ""))
        matched = [s for s in station_names if s and s.lower() in desc.lower()]
        if not matched:
            continue
        neighbor_names = set()
        for s in matched:
            sid = name_to_id.get(s)
            if sid in g:
                for nb in g.neighbors(sid):
                    nb_name = g.nodes[nb]["name"]
                    if nb_name not in matched:
                        neighbor_names.add(nb_name)
        matched = [s for s in matched if s in flows_df.columns]
        neighbor_names = [s for s in neighbor_names if s in flows_df.columns]
        if not matched:
            continue

        start, end = cl["when"], cl["end"]
        during_mask = (ts >= start) & (ts <= end)
        pre_mask = (ts >= start - pd.Timedelta(minutes=pre_min)) & (ts < start)

        closed_during = flows_df.loc[during_mask, matched].sum(axis=1)
        closed_pre = flows_df.loc[pre_mask, matched].sum(axis=1)
        closed_change = 100 * (closed_during.mean() - closed_pre.mean()) / max(closed_pre.mean(), 1e-9)

        if neighbor_names:
            nb_during = flows_df.loc[during_mask, neighbor_names].sum(axis=1)
            nb_pre = flows_df.loc[pre_mask, neighbor_names].sum(axis=1)
            nb_change = 100 * (nb_during.mean() - nb_pre.mean()) / max(nb_pre.mean(), 1e-9)
        else:
            nb_change = np.nan

        rows.append({
            "closure": desc[:60], "affected_stations": ", ".join(matched),
            "neighbor_stations": ", ".join(neighbor_names) if neighbor_names else "-",
            "closed_flow_change_%": round(closed_change, 1),
            "neighbor_flow_change_%": round(nb_change, 1) if not pd.isna(nb_change) else np.nan,
            "duration": cl.get("duration", ""),
        })

    result = pd.DataFrame(rows)
    report.add_table(result)
    if not result.empty:
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(result) + 2))
        idx = np.arange(len(result))
        width = 0.35
        ax.barh(idx - width / 2, result["closed_flow_change_%"], height=width, color="#1f77b4",
                label="gesperrte Station(en)")
        ax.barh(idx + width / 2, result["neighbor_flow_change_%"], height=width, color="#2ca02c",
                label="Nachbarstationen")
        ax.axvline(0, color="grey", linewidth=1)
        ax.set_yticks(idx)
        ax.set_yticklabels(result["closure"], fontsize=8)
        ax.set_xlabel("Veraenderung ggue. vorher (%)")
        ax.set_title("Closure-Impact: Flussveraenderung gesperrt vs. Nachbarn")
        ax.legend()
        safe_savefig(fig, out_dir / "12_closure_impact.png")
    return result


# ===========================================================================
# main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Visualisierung, Korrelations- und Kausalitaetsanalyse "
                                             "fuer den Berlin U-Bahn Hackathon-Datensatz.")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="Ordner mit den CSV-Dateien laut dataset_schema.md (Standard: ./data)")
    p.add_argument("--output-dir", type=Path, default=Path("analysis_output"),
                    help="Zielordner fuer Plots und Report (Standard: ./analysis_output)")
    p.add_argument("--top-n-stations", type=int, default=10, help="Anzahl Stationen im Top-N-Plot")
    p.add_argument("--top-k-pairs", type=int, default=15,
                    help="Anzahl nicht-benachbarter Stationspaare in der Korrelationstabelle")
    p.add_argument("--max-lag", type=int, default=4,
                    help="Max. Lag (in Zeitschritten der gewaehlten Aufloesung) fuer Granger-Tests")
    p.add_argument("--resample-freq", type=str, default="1h",
                    help="Aufloesung fuer Korrelation/Kausalitaet, z.B. '15min' (roh) oder '1h' (empfohlen)")
    p.add_argument("--event-radius-km", type=float, default=1.5,
                    help="Radius um ein Venue, innerhalb dessen Stationen als 'betroffen' gelten")
    p.add_argument("--geocode", dest="geocode", action="store_true", default=True,
                    help="Unbekannte Event-Venues live per OpenStreetMap/Nominatim geocodieren "
                         "(Standard: an; braucht Internet, Ergebnisse werden gecacht)")
    p.add_argument("--no-geocode", dest="geocode", action="store_false",
                    help="Live-Geocoding abschalten (nur KNOWN_VENUE_COORDS + vorhandener Cache)")
    return p.parse_args()


def main():
    args = parse_args()
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    report = Report()
    report.add(f"# Analyse-Report: Berlin U-Bahn Hackathon-Datensatz\n\n"
                f"Datenordner: `{args.data_dir}` | Aufloesung fuer Korrelation/Kausalitaet: `{args.resample_freq}`\n\n"
                "> Hinweis: Korrelation und Granger-Kausalitaet sind statistische Hinweise, kein Beweis "
                "physikalischer Verursachung. Alle Ergebnisse sollten im Agenten mit Unsicherheit "
                "kommuniziert werden.")

    log("Lade Daten ...")
    data = load_data(args.data_dir)
    data = preprocess(data)
    flows_df = data["flows"]
    station_cols = get_station_columns(flows_df)
    log(f"{len(station_cols)} Stationen, {len(flows_df)} Zeitstempel "
        f"({flows_df['timestamp'].min()} bis {flows_df['timestamp'].max()})")

    g = None
    if "stations" in data:
        g = build_graph(data["stations"], data.get("connections"))

    # ---------------------------------------------------------------- Plots
    log("Erzeuge Visualisierungen ...")
    plot_paths = []
    plot_paths.append(plot_total_flow_timeseries(flows_df, station_cols, plots_dir))
    plot_paths.append(plot_hourly_weekday_heatmap(flows_df, station_cols, plots_dir))
    plot_paths.append(plot_top_stations(flows_df, station_cols, plots_dir, args.top_n_stations))
    if g is not None:
        plot_paths.append(plot_network_graph(g, flows_df, plots_dir))

    merged = None
    if "weather" in data:
        merged = merge_flows_weather_with_events(flows_df, data["weather"], station_cols, data["events"])
        plot_paths.append(plot_weather_vs_flow(merged, plots_dir))
        plot_paths.extend(plot_weather_bins(merged, plots_dir))

    if "energy" in data:
        plot_paths.append(plot_energy_per_line(data["energy"], plots_dir))

    if "events" in data:
        plot_paths.append(plot_events_timeline(data["events"], plots_dir))

    report.add("## Erzeugte Visualisierungen\n\n" +
                "\n".join(f"- `{p}`" for p in plot_paths if p))

    # ------------------------------------------------------- Korrelationen
    log("Berechne Korrelationen ...")
    if merged is not None:
        correlation_weather_flow(merged, report)
        correlation_matrix_heatmap(merged, plots_dir)
        for var in [v for v in ["prcp", "temp"] if v in merged.columns]:
            lagged_cross_correlation(merged, var, plots_dir, max_lag_steps=16)
        correlation_weather_flow_daily(flows_df, data["weather"], station_cols, report, plots_dir)

    if "events" in data:
        correlation_events_daily(flows_df, data["events"], station_cols, report, plots_dir)

    pairs_df = pd.DataFrame()
    if g is not None:
        pairs_df = station_pair_correlation(flows_df, station_cols, g, top_k=args.top_k_pairs)
        report.add("## Stark korrelierte, NICHT direkt verbundene Stationspaare\n\n"
                    "Kandidaten fuer versteckte Abhaengigkeiten (z.B. Pendlerstrecken via "
                    "Umstieg, gemeinsame Zubringer-Buslinie, benachbarte Wohn-/Arbeitsgebiete). "
                    "Vgl. Trainingsfrage 8.")
        report.add_table(pairs_df)
        if not pairs_df.empty:
            plot_top_pair_timeseries(flows_df, pairs_df.iloc[0], plots_dir)

    # -------------------------------------------------------- Kausalitaet
    log("Fuehre Kausalitaets-/Wirkungsanalysen durch ...")
    if merged is not None:
        res_merged = merged.set_index("timestamp")[
            ["total_flow"] + [c for c in ["temp", "rhum", "prcp", "wspd", "pres", "cldc"] if c in merged.columns]
        ].resample(args.resample_freq).mean().dropna()
        res_merged = res_merged.reset_index()
        granger_causality_weather_to_flow(res_merged, args.max_lag, report)

    if g is not None and not pairs_df.empty:
        flows_resampled = flows_df.set_index("timestamp")[station_cols].resample(args.resample_freq).sum()
        flows_resampled = flows_resampled.reset_index()
        granger_causality_station_pairs(flows_resampled, pairs_df, args.max_lag, report)

    if "events" in data and "stations" in data:
        cache = GeoCache(args.output_dir / "geocode_cache.json")
        event_impact_analysis(flows_df, data["events"], data["stations"], args.event_radius_km,
                               report, plots_dir, cache, args.geocode)

    if "closures" in data and "stations" in data and g is not None:
        closure_impact_analysis(flows_df, data["closures"], data["stations"], g, report, plots_dir)

    report.write(args.output_dir / "report.md")
    log("Fertig. Alle Plots liegen in "
        f"'{plots_dir}', der Report in '{args.output_dir / 'report.md'}'.")


if __name__ == "__main__":
    main()