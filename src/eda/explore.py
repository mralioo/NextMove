"""
InnoTrans 2026 Hackathon -- Exploratory Data Analysis
=====================================================
Loads all provided datasets and generates visualizations and a summary report.

Usage:
    python src/eda/explore.py                          # uses default data path
    python src/eda/explore.py --data-dir data/test/   # point to new dataset
    python src/eda/explore.py --out-dir outputs/eda/  # custom output folder

Outputs (all saved to --out-dir):
    01_network_map.png              Station map coloured by U-Bahn line
    02_daily_flow_timeline.png      Network total flow over time + events + closures
    03_hourly_pattern.png           Average hourly flow pattern weekday vs weekend
    04_top_stations.png             Top-20 busiest stations bar chart
    05_flow_heatmap.png             Top-20 stations x hour-of-day heatmap
    06_event_flow_impact.png        Flow distribution on high-event vs normal days
    07_weather_flow.png             Temperature & precipitation vs daily flow
    08_energy_per_line.png          Energy consumption and efficiency per U-Bahn line
    09_closure_timeline.png         Gantt chart of all disruptions/closures
    10_flow_anomalies.png           Rolling baseline + Z-score anomaly spikes
    summary_report.txt              Key statistics for every dataset

Assumptions:
    - All timestamps are Berlin local time (CET/CEST).
    - flows.csv and weather_data.csv share the same 15-minute timestamp grain.
    - Events are mapped to stations via Haversine distance (nearest stations
      within 1.5 km); events without parseable coordinates use venue name lookup.
    - Passenger counts in flows.csv are simulated; energy_consumption.csv and
      closures.csv are also simulated (per dataset schema).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- safe on all systems
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default data directory relative to project root
DEFAULT_DATA_DIR = Path("data") / "training dataset"
DEFAULT_OUT_DIR  = Path("outputs") / "eda"

# Colour palette for U-Bahn lines (official BVG approximations)
LINE_COLOURS = {
    "U1": "#7DAF2E",   # green
    "U2": "#DA421E",   # red
    "U3": "#16683D",   # dark green
    "U5": "#7E4E04",   # brown
    "U6": "#6C4EA0",   # purple
    "U7": "#006CB5",   # blue
    "U8": "#224F9F",   # dark blue
    "U9": "#F3A019",   # orange
}

# Branding / style
BRAND_BG    = "#0D1117"   # dark background for "operator feel"
BRAND_TEXT  = "#E6EDF3"
BRAND_GRID  = "#21262D"
ACCENT      = "#58A6FF"

# Known venue -> approximate lat/lon for event geocoding
VENUE_COORDS = {
    "uber arena":           (52.5079, 13.4396),
    "mercedes-benz arena":  (52.5079, 13.4396),
    "mercedes benz arena":  (52.5079, 13.4396),
    "olympiastadion":       (52.5148, 13.2394),
    "waldbuehne":           (52.5144, 13.2321),
    "waldbühne":            (52.5144, 13.2321),
    "tempodrom":            (52.5026, 13.3794),
    "messe berlin":         (52.5080, 13.2782),
    "icm":                  (52.5080, 13.2782),
    "velodrom":             (52.5425, 13.4699),
    "columbiahalle":        (52.4798, 13.3879),
    "lido":                 (52.5040, 13.4479),
    "berghain":             (52.5111, 13.4432),
}


# ===========================================================================
# DATA LOADING
# ===========================================================================

def _find_file(data_dir: Path, keyword: str) -> Path | None:
    """Find the first CSV in data_dir whose stem contains keyword (case-insensitive)."""
    for p in sorted(data_dir.glob("*.csv")):
        if keyword.lower() in p.stem.lower():
            return p
    return None


class DataLoader:
    """
    Load all InnoTrans hackathon CSV files from a given directory.
    Handles both the 'pre_innotrans' training naming convention and any
    future dataset naming (e.g., test dataset released on Sep 25).

    All timestamps are coerced to timezone-naive Berlin local time.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._verify_dir()

    def _verify_dir(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def _load_csv(self, keyword: str, **read_kwargs) -> pd.DataFrame | None:
        path = _find_file(self.data_dir, keyword)
        if path is None:
            print(f"  [WARN] No file matching '{keyword}' found in {self.data_dir} -- skipping.")
            return None
        print(f"  Loading {path.name} …")
        return pd.read_csv(path, **read_kwargs)

    # ------------------------------------------------------------------
    def load_flows(self) -> pd.DataFrame:
        df = self._load_csv("flows", parse_dates=["timestamp"])
        if df is None:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return df

    def load_stations(self) -> pd.DataFrame:
        df = self._load_csv("stations_with_ubahn")
        if df is None:
            return pd.DataFrame()
        return df

    def load_connections(self) -> pd.DataFrame:
        df = self._load_csv("connections")
        if df is None:
            return pd.DataFrame()
        return df

    def load_events(self) -> pd.DataFrame:
        df = self._load_csv("events")
        if df is None:
            return pd.DataFrame()
        for col in ["began_local", "estimated_end_local"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None)
        return df

    def load_closures(self) -> pd.DataFrame:
        df = self._load_csv("closures")
        if df is None:
            return pd.DataFrame()
        df["when"] = pd.to_datetime(df["when"], errors="coerce")
        # Parse duration string like "2h30min" -> timedelta
        def parse_duration(s):
            if pd.isna(s):
                return pd.Timedelta(0)
            s = str(s).strip()
            h = m = 0
            if "h" in s:
                parts = s.split("h")
                h = int(parts[0])
                rest = parts[1].replace("min", "").strip()
                m = int(rest) if rest else 0
            elif "min" in s:
                m = int(s.replace("min", "").strip())
            return pd.Timedelta(hours=h, minutes=m)
        df["duration_td"] = df["duration"].apply(parse_duration)
        df["end"] = df["when"] + df["duration_td"]
        return df

    def load_weather(self) -> pd.DataFrame:
        df = self._load_csv("weather")
        if df is None:
            return pd.DataFrame()
        df = df.rename(columns={df.columns[0]: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
        return df

    def load_energy(self) -> pd.DataFrame:
        df = self._load_csv("energy")
        if df is None:
            return pd.DataFrame()
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def load_all(self) -> dict:
        print(f"\n[DATA]  Loading data from: {self.data_dir}\n")
        return {
            "flows":       self.load_flows(),
            "stations":    self.load_stations(),
            "connections": self.load_connections(),
            "events":      self.load_events(),
            "closures":    self.load_closures(),
            "weather":     self.load_weather(),
            "energy":      self.load_energy(),
        }


# ===========================================================================
# HELPER UTILITIES
# ===========================================================================

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in kilometres between two WGS-84 points."""
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(Δλ / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def nearest_stations(lat: float, lon: float, stations: pd.DataFrame, top_n: int = 3, max_km: float = 1.5) -> list[str]:
    """Return names of the nearest stations within max_km radius."""
    if stations.empty:
        return []
    dists = stations.apply(
        lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
    )
    close = stations[dists <= max_km].copy()
    close["_dist"] = dists[close.index]
    close = close.nsmallest(top_n, "_dist")
    return close["station_name"].tolist()


def flow_columns(flows: pd.DataFrame) -> list[str]:
    return [c for c in flows.columns if c != "timestamp"]


def daily_network_flow(flows: pd.DataFrame) -> pd.Series:
    """Total network passenger flow per day."""
    cols = flow_columns(flows)
    return flows.set_index("timestamp")[cols].resample("D").sum().sum(axis=1)


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply consistent dark style to an axes."""
    ax.set_facecolor(BRAND_BG)
    ax.spines[:].set_color(BRAND_GRID)
    ax.tick_params(colors=BRAND_TEXT, labelsize=9)
    ax.xaxis.label.set_color(BRAND_TEXT)
    ax.yaxis.label.set_color(BRAND_TEXT)
    ax.title.set_color(BRAND_TEXT)
    ax.grid(True, color=BRAND_GRID, linewidth=0.6, linestyle="--")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def _fig(nrows=1, ncols=1, figsize=(14, 5)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BRAND_BG)
    return fig, ax


def _save(fig, out_dir: Path, filename: str):
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BRAND_BG)
    plt.close(fig)
    print(f"  [OK]  Saved -> {path.name}")


# ===========================================================================
# PLOT 1 -- NETWORK MAP
# ===========================================================================

def plot_network_map(stations: pd.DataFrame, connections: pd.DataFrame,
                     flows: pd.DataFrame, out_dir: Path):
    """Station map coloured by U-Bahn line, sized by total passenger flow."""
    if stations.empty:
        return

    fig, ax = _fig(figsize=(13, 10))
    ax.set_facecolor(BRAND_BG)
    fig.patch.set_facecolor(BRAND_BG)

    # Draw connection edges first (grey)
    if not connections.empty:
        s_idx = stations.set_index("station_id")
        for _, row in connections.iterrows():
            for col_a, col_b in [("station_id_1", "station_id_2")]:
                if row[col_a] in s_idx.index and row[col_b] in s_idx.index:
                    x0, y0 = s_idx.loc[row[col_a], ["longitude", "latitude"]]
                    x1, y1 = s_idx.loc[row[col_b], ["longitude", "latitude"]]
                    ax.plot([x0, x1], [y0, y1], color="#3D444D", linewidth=1.0, zorder=1)

    # Station size = total flow (normalised)
    if not flows.empty:
        fcols = flow_columns(flows)
        total = flows[fcols].sum()
        # align by station name
        def get_size(name):
            if name in total.index:
                return total[name]
            return total.mean()
        stations = stations.copy()
        stations["_flow"] = stations["station_name"].apply(get_size)
        min_s, max_s = stations["_flow"].min(), stations["_flow"].max()
        stations["_size"] = 30 + 200 * (stations["_flow"] - min_s) / (max_s - min_s + 1)
    else:
        stations = stations.copy()
        stations["_size"] = 60

    # Plot by primary line (first in comma-separated list)
    stations["_primary_line"] = stations["u_bahn_lines"].str.split(",").str[0].str.strip()
    legend_patches = []
    for line, colour in LINE_COLOURS.items():
        subset = stations[stations["_primary_line"] == line]
        if subset.empty:
            continue
        ax.scatter(subset["longitude"], subset["latitude"],
                   c=colour, s=subset["_size"], zorder=3,
                   edgecolors="white", linewidths=0.4, alpha=0.92, label=line)
        legend_patches.append(mpatches.Patch(color=colour, label=line))

    ax.legend(handles=legend_patches, loc="upper left", framealpha=0.25,
              facecolor=BRAND_BG, edgecolor=BRAND_GRID,
              labelcolor=BRAND_TEXT, fontsize=9, title="Line",
              title_fontsize=9)

    ax.set_title("Berlin U-Bahn Network -- Station Map\n"
                 "(node size ∝ total passenger flow)",
                 fontsize=13, fontweight="bold", color=BRAND_TEXT, pad=10)
    ax.set_xlabel("Longitude", color=BRAND_TEXT, fontsize=9)
    ax.set_ylabel("Latitude",  color=BRAND_TEXT, fontsize=9)
    ax.tick_params(colors=BRAND_TEXT, labelsize=8)
    ax.spines[:].set_color(BRAND_GRID)
    ax.grid(True, color=BRAND_GRID, linewidth=0.5, linestyle="--")

    _save(fig, out_dir, "01_network_map.png")


# ===========================================================================
# PLOT 2 -- DAILY FLOW TIMELINE WITH EVENTS & CLOSURES
# ===========================================================================

def plot_daily_flow_timeline(flows: pd.DataFrame, events: pd.DataFrame,
                              closures: pd.DataFrame, out_dir: Path):
    """Total network flow per day, overlaid with event intensity and closures."""
    if flows.empty:
        return

    daily = daily_network_flow(flows)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                    facecolor=BRAND_BG,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.08)

    # --- top panel: flow ---
    ax1.fill_between(daily.index, daily.values, alpha=0.25, color=ACCENT)
    ax1.plot(daily.index, daily.values, color=ACCENT, linewidth=1.5, label="Network daily flow")

    # Rolling 7-day baseline
    baseline = daily.rolling(7, min_periods=1, center=True).mean()
    ax1.plot(baseline.index, baseline.values, color="#F78166",
             linewidth=1.0, linestyle="--", label="7-day rolling mean", alpha=0.8)

    # Mark closures as vertical red shading
    if not closures.empty:
        for _, cl in closures.iterrows():
            if pd.notna(cl["when"]):
                ax1.axvline(cl["when"], color="#FF4444", alpha=0.4, linewidth=0.8)

    _style_ax(ax1, title="Berlin U-Bahn -- Daily Passenger Flow (Network Total)",
              ylabel="Passenger count / day")
    ax1.legend(loc="upper right", framealpha=0.2, facecolor=BRAND_BG,
               labelcolor=BRAND_TEXT, fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    # --- bottom panel: event attendance per day ---
    if not events.empty and "began_local" in events.columns:
        ev_daily = events.groupby(events["began_local"].dt.normalize())["estimated_attendance"].sum()
        ax2.bar(ev_daily.index, ev_daily.values, color="#E8A319", alpha=0.8,
                width=0.8, label="Total event attendance")
        _style_ax(ax2, ylabel="Event attendance", xlabel="Date")
        ax2.legend(loc="upper right", framealpha=0.2, facecolor=BRAND_BG,
                   labelcolor=BRAND_TEXT, fontsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    else:
        ax2.set_visible(False)

    # Annotation: max flow day
    max_day = daily.idxmax()
    ax1.annotate(f"Peak: {daily.max()/1e6:.2f}M\n{max_day.strftime('%b %d')}",
                 xy=(max_day, daily.max()),
                 xytext=(max_day + pd.Timedelta(days=3), daily.max() * 0.98),
                 color=BRAND_TEXT, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=BRAND_TEXT, lw=0.8))

    _save(fig, out_dir, "02_daily_flow_timeline.png")


# ===========================================================================
# PLOT 3 -- HOURLY FLOW PATTERN (Weekday vs Weekend)
# ===========================================================================

def plot_hourly_pattern(flows: pd.DataFrame, out_dir: Path):
    """Average hourly flow pattern across the network, weekday vs weekend."""
    if flows.empty:
        return

    cols = flow_columns(flows)
    df = flows.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["total"] = df[cols].sum(axis=1)

    hourly_wd = df[~df["is_weekend"]].groupby("hour")["total"].mean()
    hourly_we = df[df["is_weekend"]].groupby("hour")["total"].mean()

    # Ensure all 24 hours present
    all_hours = pd.RangeIndex(24)
    hourly_wd = hourly_wd.reindex(all_hours, fill_value=0)
    hourly_we = hourly_we.reindex(all_hours, fill_value=0)

    fig, ax = _fig(figsize=(13, 5))
    x = np.arange(24)
    width = 0.38
    ax.bar(x - width / 2, hourly_wd.values, width, color=ACCENT,
           alpha=0.85, label="Weekday")
    ax.bar(x + width / 2, hourly_we.values, width, color="#E8A319",
           alpha=0.85, label="Weekend")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in x], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax.legend(framealpha=0.2, facecolor=BRAND_BG, labelcolor=BRAND_TEXT, fontsize=9)
    _style_ax(ax, title="Average Hourly Passenger Flow -- Weekday vs Weekend",
              xlabel="Hour of day (Berlin local time)",
              ylabel="Average passengers / 15-min interval")

    # Annotate peaks
    pk_wd = hourly_wd.idxmax()
    pk_we = hourly_we.idxmax()
    ax.annotate(f"AM peak\n{pk_wd}:00", xy=(pk_wd - width / 2, hourly_wd[pk_wd]),
                xytext=(pk_wd - 2, hourly_wd[pk_wd] * 1.05),
                color=BRAND_TEXT, fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="->", color=BRAND_TEXT, lw=0.6))

    _save(fig, out_dir, "03_hourly_pattern.png")


# ===========================================================================
# PLOT 4 -- TOP-20 BUSIEST STATIONS
# ===========================================================================

def plot_top_stations(flows: pd.DataFrame, stations: pd.DataFrame, out_dir: Path):
    """Horizontal bar chart of top-20 stations by total passenger flow."""
    if flows.empty:
        return

    cols = flow_columns(flows)
    total = flows[cols].sum().sort_values(ascending=True).tail(20)

    # Colour by line
    name_to_line = {}
    if not stations.empty:
        for _, row in stations.iterrows():
            primary = row["u_bahn_lines"].split(",")[0].strip()
            name_to_line[row["station_name"]] = primary

    bar_colours = [LINE_COLOURS.get(name_to_line.get(n, ""), ACCENT) for n in total.index]

    fig, ax = _fig(figsize=(13, 8))
    bars = ax.barh(range(len(total)), total.values, color=bar_colours, alpha=0.85)
    ax.set_yticks(range(len(total)))
    # Shorten station names for readability
    short_names = [n.replace(" (Berlin)", "").replace("S+U ", "").replace("U ", "") for n in total.index]
    ax.set_yticklabels(short_names, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))

    for bar, val in zip(bars, total.values):
        ax.text(val * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{val/1e6:.2f}M", va="center", ha="left",
                color=BRAND_TEXT, fontsize=7)

    # Legend for lines
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in LINE_COLOURS.items()]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.2,
              facecolor=BRAND_BG, labelcolor=BRAND_TEXT, fontsize=8, ncol=4)

    _style_ax(ax, title="Top-20 Busiest Stations -- Total Passenger Flow (Training Period)",
              xlabel="Total passenger count")
    _save(fig, out_dir, "04_top_stations.png")


# ===========================================================================
# PLOT 5 -- STATION x HOUR HEATMAP (top 20)
# ===========================================================================

def plot_flow_heatmap(flows: pd.DataFrame, out_dir: Path):
    """Heatmap of average flow by hour for the top-20 stations."""
    if flows.empty:
        return

    cols = flow_columns(flows)
    total = flows[cols].sum().sort_values(ascending=False).head(20)
    top_cols = total.index.tolist()

    df = flows[["timestamp"] + top_cols].copy()
    df["hour"] = df["timestamp"].dt.hour
    hourly_avg = df.groupby("hour")[top_cols].mean()

    # Shorten names
    short = {c: c.replace(" (Berlin)", "").replace("S+U ", "").replace("U ", "") for c in top_cols}
    hourly_avg = hourly_avg.rename(columns=short)

    fig, ax = plt.subplots(figsize=(15, 7), facecolor=BRAND_BG)
    matrix = hourly_avg.T.values  # shape: (stations, hours)
    norm = mcolors.Normalize(vmin=matrix.min(), vmax=matrix.max())
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", norm=norm, interpolation="nearest")

    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(list(short.values()), fontsize=8, color=BRAND_TEXT)
    all_hours = [h for h in hourly_avg.index]
    ax.set_xticks(range(len(all_hours)))
    ax.set_xticklabels([f"{h:02d}" for h in all_hours], fontsize=8, color=BRAND_TEXT)
    ax.set_xlabel("Hour of day", color=BRAND_TEXT, fontsize=9)
    ax.set_title("Average Passenger Flow by Station x Hour (Top 20 Stations)",
                 fontsize=12, fontweight="bold", color=BRAND_TEXT, pad=10)
    ax.set_facecolor(BRAND_BG)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Avg passengers / 15-min", color=BRAND_TEXT, fontsize=8)
    cbar.ax.tick_params(colors=BRAND_TEXT, labelsize=7)

    _save(fig, out_dir, "05_flow_heatmap.png")


# ===========================================================================
# PLOT 6 -- EVENT IMPACT ON FLOW
# ===========================================================================

def plot_event_impact(flows: pd.DataFrame, events: pd.DataFrame, out_dir: Path):
    """Boxplot: daily network flow on high-event days vs low-event days."""
    if flows.empty or events.empty:
        return

    daily = daily_network_flow(flows)

    if "began_local" not in events.columns:
        return
    ev_daily = events.groupby(events["began_local"].dt.normalize())["estimated_attendance"].sum()
    ev_daily.index = pd.to_datetime(ev_daily.index).tz_localize(None).normalize()
    daily.index = daily.index.normalize()

    threshold = ev_daily.quantile(0.6)
    high_event_days = set(ev_daily[ev_daily >= threshold].index)

    high = daily[[d in high_event_days for d in daily.index]].values
    low  = daily[[d not in high_event_days for d in daily.index]].values

    fig, ax = _fig(figsize=(10, 6))
    bp = ax.boxplot([low, high],
                    labels=["Low / No Events", "High Event Days"],
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=BRAND_TEXT),
                    capprops=dict(color=BRAND_TEXT),
                    flierprops=dict(marker="o", color=BRAND_TEXT, markersize=4, alpha=0.5))
    bp["boxes"][0].set_facecolor("#2D4A6E")
    bp["boxes"][1].set_facecolor("#7D2D2D")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    _style_ax(ax,
              title="Passenger Flow Distribution: High-Event vs Low/No-Event Days",
              ylabel="Daily network passenger count")

    # Annotate means
    for i, arr in enumerate([low, high], start=1):
        if len(arr):
            ax.text(i, np.mean(arr), f" mean: {np.mean(arr)/1e6:.2f}M",
                    va="center", color=BRAND_TEXT, fontsize=8)

    _save(fig, out_dir, "06_event_flow_impact.png")


# ===========================================================================
# PLOT 7 -- WEATHER vs FLOW
# ===========================================================================

def plot_weather_flow(flows: pd.DataFrame, weather: pd.DataFrame, out_dir: Path):
    """Scatter: daily temperature and precipitation vs daily network flow."""
    if flows.empty or weather.empty:
        return

    daily_flow = daily_network_flow(flows)

    wth = weather.copy()
    wth_daily_temp  = wth.set_index("timestamp")["temp"].resample("D").mean()
    wth_daily_prcp  = wth.set_index("timestamp")["prcp"].resample("D").sum()

    merged = pd.DataFrame({"flow": daily_flow,
                           "temp": wth_daily_temp,
                           "prcp": wth_daily_prcp}).dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=BRAND_BG)
    fig.subplots_adjust(wspace=0.3)

    # Temperature vs flow
    sc1 = ax1.scatter(merged["temp"], merged["flow"] / 1e6,
                      c=merged["temp"], cmap="RdYlBu_r",
                      alpha=0.7, s=40, edgecolors="none")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.02)
    cbar1.set_label("degC", color=BRAND_TEXT, fontsize=8)
    cbar1.ax.tick_params(colors=BRAND_TEXT, labelsize=7)

    # Linear trend
    if len(merged) > 5:
        coef = np.polyfit(merged["temp"], merged["flow"] / 1e6, 1)
        xline = np.linspace(merged["temp"].min(), merged["temp"].max(), 100)
        ax1.plot(xline, np.polyval(coef, xline), color="#F78166",
                 linewidth=1.5, linestyle="--", label="Trend")
        ax1.legend(framealpha=0.2, facecolor=BRAND_BG, labelcolor=BRAND_TEXT, fontsize=8)

    _style_ax(ax1, title="Temperature vs Daily Flow",
              xlabel="Mean daily temperature (degC)", ylabel="Daily flow (millions)")

    # Precipitation vs flow
    rain_days  = merged[merged["prcp"] > 0]
    clear_days = merged[merged["prcp"] == 0]
    ax2.scatter(clear_days["prcp"], clear_days["flow"] / 1e6,
                color=ACCENT, alpha=0.6, s=35, label="Dry", edgecolors="none")
    ax2.scatter(rain_days["prcp"], rain_days["flow"] / 1e6,
                color="#E8A319", alpha=0.7, s=40, label="Rain", edgecolors="none")
    ax2.legend(framealpha=0.2, facecolor=BRAND_BG, labelcolor=BRAND_TEXT, fontsize=8)
    _style_ax(ax2, title="Precipitation vs Daily Flow",
              xlabel="Daily precipitation (mm)", ylabel="Daily flow (millions)")

    _save(fig, out_dir, "07_weather_flow.png")


# ===========================================================================
# PLOT 8 -- ENERGY CONSUMPTION & EFFICIENCY
# ===========================================================================

def plot_energy(energy: pd.DataFrame, flows: pd.DataFrame,
                stations: pd.DataFrame, out_dir: Path):
    """Energy per line: total consumption and energy-per-passenger efficiency."""
    if energy.empty:
        return

    line_cols = [c for c in energy.columns if c != "date"]
    total_energy = energy[line_cols].sum()

    # Energy per passenger per line (MWh / total passengers on that line)
    energy_per_pass = {}
    if not flows.empty and not stations.empty:
        flow_cols_all = flow_columns(flows)
        for line in line_cols:
            # Stations on this line
            line_stations = stations[
                stations["u_bahn_lines"].str.contains(line, na=False)
            ]["station_name"].tolist()
            cols_on_line = [c for c in flow_cols_all if c in line_stations]
            if cols_on_line:
                total_pass = flows[cols_on_line].sum().sum()
                if total_pass > 0:
                    # MWh per 1000 passengers
                    energy_per_pass[line] = (total_energy[line] * 1000) / (total_pass / 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=BRAND_BG)
    fig.subplots_adjust(wspace=0.3)

    colours = [LINE_COLOURS.get(l, ACCENT) for l in total_energy.index]
    ax1.bar(total_energy.index, total_energy.values, color=colours, alpha=0.85)
    for i, (line, val) in enumerate(total_energy.items()):
        ax1.text(i, val + 50, f"{val:,.0f}", ha="center", va="bottom",
                 color=BRAND_TEXT, fontsize=8)
    _style_ax(ax1, title="Total Energy Consumption by Line (Training Period)",
              xlabel="U-Bahn Line", ylabel="Total energy (MWh)")

    if energy_per_pass:
        lines_eff = list(energy_per_pass.keys())
        vals_eff  = [energy_per_pass[l] for l in lines_eff]
        colours_eff = [LINE_COLOURS.get(l, ACCENT) for l in lines_eff]
        bars = ax2.bar(lines_eff, vals_eff, color=colours_eff, alpha=0.85)
        worst = lines_eff[np.argmax(vals_eff)]
        ax2.bar([worst], [max(vals_eff)], color="#FF4444", alpha=0.9,
                label=f"Worst: {worst}")
        ax2.legend(framealpha=0.2, facecolor=BRAND_BG, labelcolor=BRAND_TEXT, fontsize=8)
        for bar, val in zip(bars, vals_eff):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                     f"{val:.3f}", ha="center", va="bottom",
                     color=BRAND_TEXT, fontsize=7)
        _style_ax(ax2,
                  title="Energy Efficiency by Line\n(MWh per 1,000 passengers)",
                  xlabel="U-Bahn Line", ylabel="MWh / 1,000 passengers")
    else:
        ax2.set_visible(False)

    _save(fig, out_dir, "08_energy_per_line.png")


# ===========================================================================
# PLOT 9 -- CLOSURE TIMELINE (Gantt)
# ===========================================================================

def plot_closure_timeline(closures: pd.DataFrame, out_dir: Path):
    """Gantt chart showing all disruptions/closures over the training period."""
    if closures.empty:
        return

    df = closures.dropna(subset=["when"]).copy()
    df = df.sort_values("when").reset_index(drop=True)

    # Assign colour by type (section closure vs station closure)
    def closure_colour(desc):
        desc = str(desc).lower()
        if "suspended" in desc or "section" in desc:
            return "#FF4444"   # red: line suspension
        return "#E8A319"       # orange: station closure

    df["colour"] = df["description"].apply(closure_colour)

    fig, ax = plt.subplots(figsize=(15, max(5, len(df) * 0.45)), facecolor=BRAND_BG)

    for i, row in df.iterrows():
        start = row["when"]
        end   = row.get("end", start + pd.Timedelta(hours=2))
        colour = row["colour"]
        ax.barh(i, (end - start).total_seconds() / 3600,
                left=matplotlib.dates.date2num(start),
                height=0.6, color=colour, alpha=0.85, edgecolor="none")
        # Short label
        label = str(row["description"])[:55] + "…" if len(str(row["description"])) > 55 else str(row["description"])
        ax.text(matplotlib.dates.date2num(start) + 0.02, i,
                label, va="center", ha="left",
                color=BRAND_TEXT, fontsize=6.5)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"#{i+1}" for i in range(len(df))],
                       fontsize=7, color=BRAND_TEXT)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7, color=BRAND_TEXT)
    ax.set_facecolor(BRAND_BG)
    ax.spines[:].set_color(BRAND_GRID)
    ax.grid(True, color=BRAND_GRID, linewidth=0.4, linestyle="--", axis="x")
    ax.tick_params(axis="y", colors=BRAND_TEXT)

    # Legend
    red_patch    = mpatches.Patch(color="#FF4444", label="Line suspension")
    orange_patch = mpatches.Patch(color="#E8A319", label="Station closure")
    ax.legend(handles=[red_patch, orange_patch], loc="lower right",
              framealpha=0.2, facecolor=BRAND_BG, labelcolor=BRAND_TEXT, fontsize=8)

    ax.set_title("Disruption & Closure Timeline",
                 fontsize=12, fontweight="bold", color=BRAND_TEXT, pad=10)
    ax.set_xlabel("Date", color=BRAND_TEXT, fontsize=9)

    _save(fig, out_dir, "09_closure_timeline.png")


# ===========================================================================
# PLOT 10 -- FLOW ANOMALIES (rolling baseline + Z-score)
# ===========================================================================

def plot_flow_anomalies(flows: pd.DataFrame, closures: pd.DataFrame,
                         events: pd.DataFrame, out_dir: Path):
    """
    Network-total flow: rolling baseline +/- 2sigma band, with anomaly points
    and disruption markers.
    """
    if flows.empty:
        return

    cols = flow_columns(flows)
    ts = flows.set_index("timestamp")[cols].sum(axis=1).resample("H").sum()

    window = 24 * 7  # 1-week rolling window
    baseline = ts.rolling(window, min_periods=48, center=True).mean()
    sigma    = ts.rolling(window, min_periods=48, center=True).std()
    upper    = baseline + 2 * sigma
    lower    = baseline - 2 * sigma
    zscore   = (ts - baseline) / (sigma + 1)

    anomalies = ts[(zscore.abs() > 2.5) & ts.notna()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), facecolor=BRAND_BG,
                                    gridspec_kw={"height_ratios": [2.5, 1]})
    fig.subplots_adjust(hspace=0.08)

    ax1.fill_between(ts.index, lower, upper, alpha=0.15, color=ACCENT, label="+/-2sigma band")
    ax1.plot(ts.index, ts.values, color="#555F69", linewidth=0.6, alpha=0.8)
    ax1.plot(baseline.index, baseline.values, color=ACCENT, linewidth=1.2, label="Rolling mean (7-day)")
    ax1.scatter(anomalies.index, anomalies.values, color="#FF4444",
                zorder=5, s=18, label=f"Anomalies (|Z|>2.5) -- {len(anomalies)} pts")

    # Closure verticals
    if not closures.empty:
        for _, cl in closures.iterrows():
            if pd.notna(cl["when"]):
                ax1.axvline(cl["when"], color="#FF8C00", alpha=0.5, linewidth=0.8)

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax1.legend(framealpha=0.2, facecolor=BRAND_BG, labelcolor=BRAND_TEXT, fontsize=8, loc="upper right")
    _style_ax(ax1, title="Network Flow Anomaly Detection -- Rolling Baseline +/- 2sigma",
              ylabel="Passengers / hour")
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Z-score panel
    ax2.fill_between(zscore.index, zscore.values, 0,
                     where=zscore.values > 0, color="#58A6FF", alpha=0.4)
    ax2.fill_between(zscore.index, zscore.values, 0,
                     where=zscore.values < 0, color="#FF4444", alpha=0.4)
    ax2.plot(zscore.index, zscore.values, color=BRAND_TEXT, linewidth=0.5, alpha=0.6)
    ax2.axhline(2.5, color="#FF4444", linewidth=0.8, linestyle="--")
    ax2.axhline(-2.5, color="#FF4444", linewidth=0.8, linestyle="--")
    ax2.axhline(0, color=BRAND_GRID, linewidth=0.6)
    _style_ax(ax2, ylabel="Z-score", xlabel="Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    _save(fig, out_dir, "10_flow_anomalies.png")


# ===========================================================================
# SUMMARY REPORT
# ===========================================================================

def write_summary_report(data: dict, out_dir: Path):
    """Write a plain-text summary of key statistics for every dataset."""
    flows     = data["flows"]
    stations  = data["stations"]
    events    = data["events"]
    closures  = data["closures"]
    weather   = data["weather"]
    energy    = data["energy"]

    lines = ["=" * 60,
             "InnoTrans 2026 -- Dataset Summary Report",
             "=" * 60, ""]

    if not flows.empty:
        cols = flow_columns(flows)
        daily = daily_network_flow(flows)
        total_s = flows[cols].sum().sort_values(ascending=False)
        lines += [
            "PASSENGER FLOWS",
            f"  Date range  : {flows.timestamp.min():%Y-%m-%d %H:%M} -> {flows.timestamp.max():%Y-%m-%d %H:%M}",
            f"  Timestamps  : {len(flows):,} (15-min grain)",
            f"  Stations    : {len(cols)}",
            f"  Daily flow  : mean={daily.mean():,.0f}  max={daily.max():,.0f}  min={daily.min():,.0f}",
            f"  Peak day    : {daily.idxmax():%Y-%m-%d}  ({daily.max():,.0f} passengers)",
            f"  Top station : {total_s.index[0]} ({total_s.iloc[0]:,.0f} total)",
            f"  Quiet stn   : {total_s.index[-1]} ({total_s.iloc[-1]:,.0f} total)",
            "",
        ]

    if not stations.empty:
        lines += [
            "NETWORK",
            f"  Stations    : {len(stations)}",
            f"  Lines       : {', '.join(sorted(stations['u_bahn_lines'].str.split(',').explode().str.strip().unique()))}",
            "",
        ]

    if not events.empty:
        lines += [
            "EVENTS",
            f"  Total events: {len(events)}",
        ]
        if "began_local" in events.columns:
            ev_day = events.groupby(events["began_local"].dt.normalize())["estimated_attendance"].sum()
            lines += [
                f"  Peak day    : {ev_day.idxmax()} ({ev_day.max():,.0f} attendance)",
            ]
        lines += [
            f"  Attendance  : mean={events['estimated_attendance'].mean():.0f}  max={events['estimated_attendance'].max():,}",
            f"  Segments    : {dict(events['segment'].value_counts())}",
            "",
        ]

    if not closures.empty:
        lines += [
            "CLOSURES / DISRUPTIONS",
            f"  Total       : {len(closures)}",
            f"  Line suspensions: {closures['description'].str.contains('suspended', na=False, case=False).sum()}",
            f"  Station closures: {closures['description'].str.contains('closed', na=False, case=False).sum()}",
            "",
        ]

    if not weather.empty:
        lines += [
            "WEATHER",
            f"  Temp range  : {weather.temp.min():.1f}degC -> {weather.temp.max():.1f}degC",
            f"  Rain timestamps: {(weather.prcp > 0).sum():,} / {len(weather):,}  ({100*(weather.prcp>0).mean():.1f}%)",
            f"  Max precip  : {weather.prcp.max():.1f} mm",
            "",
        ]

    if not energy.empty:
        lcols = [c for c in energy.columns if c != "date"]
        total_e = energy[lcols].sum()
        lines += [
            "ENERGY",
            f"  Lines       : {', '.join(lcols)}",
            f"  Highest consumer : {total_e.idxmax()} ({total_e.max():,} MWh)",
            f"  Lowest consumer  : {total_e.idxmin()} ({total_e.min():,} MWh)",
            "",
        ]

    lines += ["=" * 60]
    report = "\n".join(lines)
    path = out_dir / "summary_report.txt"
    path.write_text(report, encoding="utf-8")
    print(f"  [OK]  Saved -> summary_report.txt")
    print()
    print(report)


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="InnoTrans 2026 -- Exploratory Data Analysis & Visualisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir", "-d",
        default=str(DEFAULT_DATA_DIR),
        help=f"Path to the CSV dataset directory (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--out-dir", "-o",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory for plots and report (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--plots", "-p",
        nargs="*",
        default=None,
        help="Which plots to generate (1-10). Default: all. Example: --plots 1 2 5",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(range(1, 11))
    if args.plots:
        wanted = set(int(p) for p in args.plots)

    loader = DataLoader(data_dir)
    data   = loader.load_all()

    flows     = data["flows"]
    stations  = data["stations"]
    connections = data["connections"]
    events    = data["events"]
    closures  = data["closures"]
    weather   = data["weather"]
    energy    = data["energy"]

    print(f"\n[ART]  Generating plots -> {out_dir}\n")

    if 1 in wanted:
        print("  [1/10] Network map …")
        plot_network_map(stations, connections, flows, out_dir)

    if 2 in wanted:
        print("  [2/10] Daily flow timeline …")
        plot_daily_flow_timeline(flows, events, closures, out_dir)

    if 3 in wanted:
        print("  [3/10] Hourly pattern …")
        plot_hourly_pattern(flows, out_dir)

    if 4 in wanted:
        print("  [4/10] Top-20 stations …")
        plot_top_stations(flows, stations, out_dir)

    if 5 in wanted:
        print("  [5/10] Flow heatmap …")
        plot_flow_heatmap(flows, out_dir)

    if 6 in wanted:
        print("  [6/10] Event impact …")
        plot_event_impact(flows, events, out_dir)

    if 7 in wanted:
        print("  [7/10] Weather vs flow …")
        plot_weather_flow(flows, weather, out_dir)

    if 8 in wanted:
        print("  [8/10] Energy per line …")
        plot_energy(energy, flows, stations, out_dir)

    if 9 in wanted:
        print("  [9/10] Closure timeline …")
        plot_closure_timeline(closures, out_dir)

    if 10 in wanted:
        print("  [10/10] Flow anomalies …")
        plot_flow_anomalies(flows, closures, events, out_dir)

    print("\n[RPT]  Writing summary report …")
    write_summary_report(data, out_dir)

    print(f"\n[*]  Done!  All outputs saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
