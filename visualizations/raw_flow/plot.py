"""
Raw passenger flow - what the untouched data looks like, before any normalization.

    python -m visualizations.raw_flow.plot
    python -m visualizations.raw_flow.plot --station "Alexanderplatz"

Figures (in visualizations/raw_flow/output/)
    01_network_total_over_time.png   passengers per day in the whole network over the period
    02_daily_profile_by_daytype.png  average passengers per hour of day: weekday / Saturday / Sunday
    03_top_stations.png              busiest 15 stations (mean passengers per day)
    04_weekday_hour_heatmap.png      average passengers per hour, for each weekday x hour of day
    05_station_week_<name>.png       one station, one week, hour by hour (with --station)
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualizations import common as c

OUT_DIR = Path(__file__).resolve().parent / "output"


def plot_network_total(flows, out):
    """Daily total over time. The weekly rhythm (weekend dips) is the dominant pattern."""
    daily = c.full_days(flows.sum(axis=1, min_count=1))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily / 1000, color=c.SERIES[0])
    ax.set_title("Passengers per day, whole network")
    c.subtitle(ax, "Raw counts, sum over all stations. The dips are weekends.")
    ax.set_ylabel("passengers per day (thousands)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylim(bottom=0)
    c.save(fig, out, "01_network_total_over_time")


def plot_daily_profile(flows, out):
    """Mean network flow per hour of the SERVICE day (5 am to midnight), split by day type.
    Shows the commuter waves. Midnight is drawn at the end (hour 24), where it belongs."""
    total = flows.sum(axis=1, min_count=1)
    service_hour = np.where(total.index.hour < 5, total.index.hour + 24, total.index.hour)
    # the 0-1 am slot belongs to the previous service day -> use that day's weekday
    dow = (total.index - pd.to_timedelta((total.index.hour < 5) * 24, unit="h")).dayofweek
    groups = {"Weekday (Mon-Fri)": dow <= 4, "Saturday": dow == 5, "Sunday": dow == 6}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    peaks = {}
    for (label, mask), color in zip(groups.items(), c.SERIES):
        prof = (total[mask].groupby(service_hour[mask]).mean() / 1000).dropna()
        prof = prof[prof.index >= 5]
        ax.plot(prof.index, prof.values, color=color, label=label)
        peaks[label] = prof.loc[18]
    # Direct labels at the evening peak (all lines end near 0 at midnight, labels would pile up there).
    ax.annotate("Weekday", (18, peaks["Weekday (Mon-Fri)"]), xytext=(8, 4), textcoords="offset points",
                fontsize=9)
    ax.annotate("Saturday = Sunday", (18, peaks["Saturday"]), xytext=(8, 4), textcoords="offset points",
                fontsize=9)
    ax.set_xlim(5, 24.3)
    ax.set_ylim(bottom=0)
    ax.set_title("Average day: passengers per hour")
    c.subtitle(ax, "Mean over all days of each type. Service runs 5 am to 1 am. Weekends are ~35 % lower.")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("passengers per hour (thousands)")
    ax.set_xticks(range(6, 25, 2), [f"{h % 24:02d}:00" for h in range(6, 25, 2)])
    ax.legend(loc="upper left")
    c.save(fig, out, "02_daily_profile_by_daytype")


def plot_top_stations(flows, out, n=15):
    """Busiest stations by mean passengers per day."""
    per_day = flows.resample("1D").sum(min_count=1).mean().sort_values().tail(n)
    fig, ax = plt.subplots(figsize=(8, 0.34 * n + 1.2))
    ax.barh([c.short_station(s) for s in per_day.index], per_day.values, color=c.SERIES[0], height=0.7)
    for y, v in enumerate(per_day.values):
        ax.text(v, y, f" {v:,.0f}", va="center", fontsize=8, color=c.TEXT_2)
    ax.set_title(f"Busiest {n} stations")
    c.subtitle(ax, "Mean passengers per day (raw counts)")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("passengers per day")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    c.save(fig, out, "03_top_stations")


def plot_weekday_hour_heatmap(flows, out):
    """Mean network flow for every weekday x hour of the SERVICE day (5 am to 1 am).
    Like the daily profile, the 0-1 am slot counts towards the previous day."""
    from matplotlib.colors import LinearSegmentedColormap
    total = flows.sum(axis=1, min_count=1)
    service_hour = np.where(total.index.hour < 5, total.index.hour + 24, total.index.hour)
    dow = (total.index - pd.to_timedelta((total.index.hour < 5) * 24, unit="h")).dayofweek
    grid = (total.groupby([dow, service_hour]).mean() / 1000).unstack()
    grid = grid.reindex(index=range(7), columns=range(5, 25))
    cmap = LinearSegmentedColormap.from_list("blues", [c.SURFACE, c.BLUE_LIGHT, c.SERIES[0], "#0d3a73"])
    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(grid.values, cmap=cmap, aspect="auto", vmin=0)
    vmax = np.nanmax(grid.values)
    for (y, x), v in np.ndenumerate(grid.values):
        if np.isfinite(v):
            ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 * vmax else c.TEXT)
    ax.set_xticks(range(grid.shape[1]), [f"{h % 24:02d}" for h in grid.columns])
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("hour of day (start of slot)")
    ax.set_title("Passengers per hour, by weekday and hour")
    c.subtitle(ax, "Mean over all weeks, whole network, in thousands. 00:00 counts towards the previous day.")
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.outline.set_visible(False)
    cb.set_label("passengers per hour (thousands)")
    c.save(fig, out, "04_weekday_hour_heatmap")


def plot_station_week(flows, station_query, out):
    """One station, the first full Monday-Sunday week, hour by hour."""
    matches = [s for s in flows.columns if station_query.lower() in s.lower()]
    if not matches:
        raise SystemExit(f"no station matches '{station_query}'")
    s = matches[0]
    first_monday = flows.index[flows.index.dayofweek == 0][0].normalize()
    week = flows.loc[first_monday:first_monday + pd.Timedelta(days=7), s]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(week.index, week.values, color=c.SERIES[0])
    ax.set_title(f"{c.short_station(s)}: one week of raw flow")
    c.subtitle(ax, f"Passengers per time slot, week starting {first_monday:%d %b %Y}")
    ax.set_ylabel("passengers per slot")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.set_ylim(bottom=0)
    safe = "".join(ch if ch.isalnum() else "_" for ch in c.short_station(s)).strip("_")
    c.save(fig, out, f"05_station_week_{safe}")


def main():
    ap = c.base_argparser("Plots of the raw passenger flow.")
    ap.add_argument("--freq", default="1h", help="time grid for the plots (default 1h)")
    ap.add_argument("--station", help="also plot one week for this station (substring match)")
    a = ap.parse_args()
    c.apply_style()
    out = Path(a.out) if a.out else OUT_DIR
    flows, _, _ = c.load_raw(a.data, a.freq)
    plot_network_total(flows, out)
    plot_daily_profile(flows, out)
    plot_top_stations(flows, out)
    plot_weekday_hour_heatmap(flows, out)
    if a.station:
        plot_station_week(flows, a.station, out)


if __name__ == "__main__":
    main()
