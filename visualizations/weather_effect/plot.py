"""
Effect of the weather on the NORMALIZED flow.

The normalized flow (normalized_flows.csv) is log1p(actual) - normal, i.e. the deviation from a
typical hour at that station, with the commuter waves already removed. Any remaining link to
rain or temperature is therefore the weather effect. For comparison, each figure also shows the
weather part the model assigns (normalized_weather.csv): if both agree, the model captures it.

    python -m visualizations.weather_effect.plot

Figures (in visualizations/weather_effect/output/)
    01_rain_bins.png                 flow vs normal by rain intensity
    02_temperature_bins.png          flow vs normal by temperature
    03_timeline.png                  daily rain and daily normalized flow over the whole period
    04_station_rain_sensitivity.png  stations that react most / least to rain (model coefficients)
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualizations import common as c

OUT_DIR = Path(__file__).resolve().parent / "output"
RAIN_BINS = [(-np.inf, 0.1, "dry\n< 0.1 mm"), (0.1, 0.5, "light\n0.1-0.5 mm"),
             (0.5, 2.0, "moderate\n0.5-2 mm"), (2.0, np.inf, "heavy\n> 2 mm")]
LABEL_OBS = "Observed (normalized flow)"
LABEL_MODEL = "Model weather part"


def network_series(normalized_dir):
    """Per timestamp: MEDIAN over all stations of the normalized flow and of the weather part.
    The median, because slots with 0 passengers pull the mean of log values down (~ -5 %), while
    the pipeline centres every station on its median. Night-pause timestamps are dropped."""
    total = c.read_table(normalized_dir, "normalized_flows")
    wpart = c.read_table(normalized_dir, "normalized_weather")
    df = pd.DataFrame({"total": total.median(axis=1), "weather": wpart.median(axis=1)}).dropna()
    return df, c.infer_freq(total.index)


def plot_rain_bins(df, weather, out):
    prcp = weather["prcp"].reindex(df.index)
    labels, obs, mod, n = [], [], [], []
    for lo, hi, label in RAIN_BINS:
        m = (prcp >= lo) & (prcp < hi)
        if m.sum() == 0:
            continue
        labels.append(label)
        obs.append(c.log_to_pct(df.total[m].mean()))
        mod.append(c.log_to_pct(df.weather[m].mean()))
        n.append(int(m.sum()))
    x, w = np.arange(len(labels)), 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2 - 0.01, obs, w, color=c.SERIES[0], label=LABEL_OBS)
    ax.bar(x + w / 2 + 0.01, mod, w, color=c.SERIES[1], label=LABEL_MODEL)
    for xi, v in zip(x, obs):     # value labels only on the observed bars
        ax.text(xi - w / 2, v, f"{v:+.0f} %", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color=c.TEXT_2)
    ax.set_xticks(x, [f"{l}\n({k} h)" for l, k in zip(labels, n)])
    c.zero_line(ax)
    c.pct_formatter(ax)
    ax.grid(axis="x", visible=False)
    ax.set_title("Rain: more passengers than normal")
    c.subtitle(ax, "Deviation from normal (median over stations), by precipitation in that hour. "
                   "The model treats rain as on/off, so it overstates light rain.")
    ax.set_ylabel("vs. normal")
    ax.legend(loc="upper left")
    c.save(fig, out, "01_rain_bins")


def plot_temperature_bins(df, weather, out, step=2, min_hours=15):
    temp = weather["temp"].reindex(df.index)
    dry = weather["prcp"].reindex(df.index) < 0.1      # dry hours only: keep rain out of the picture
    t_bin = (np.floor(temp / step) * step)[dry]
    g = df[dry].groupby(t_bin)
    stats = pd.DataFrame({"obs": c.log_to_pct(g.total.mean()), "mod": c.log_to_pct(g.weather.mean()),
                          "n": g.size()})
    stats = stats[stats.n >= min_hours]
    centers = stats.index + step / 2
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ends = []
    for col, color, label in [("obs", c.SERIES[0], LABEL_OBS), ("mod", c.SERIES[1], LABEL_MODEL)]:
        ax.plot(centers, stats[col], color=color, marker="o", markersize=5, label=label)
        ends.append((centers[-1], stats[col].iloc[-1], label.split(" (")[0], color))
    c.zero_line(ax)
    c.pct_formatter(ax)
    ax.set_xlim(centers.min() - 1, centers.max() + 7)
    c.label_line_ends(ax, ends)
    ax.set_title("Heat: fewer passengers than normal")
    c.subtitle(ax, f"Deviation from normal (median over stations), dry hours only, {step} °C bins (>= {min_hours} h each)")
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("vs. normal")
    ax.legend(loc="lower left")
    c.save(fig, out, "02_temperature_bins")


def plot_timeline(df, weather, out):
    """Two stacked panels on the same time axis (no dual y-axis): daily rain, daily deviation."""
    hours = df.total.resample("1D").count()
    daily = df.resample("1D").mean()[hours >= hours.median() * 0.9]     # drop partial days
    rain = weather["prcp"].resample("1D").sum().reindex(daily.index)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2], "hspace": 0.25})
    ax1.bar(daily.index, rain, width=0.8, color=c.SERIES[0])
    ax1.set_ylabel("rain (mm/day)")
    ax1.set_title("Rainy days lift the normalized flow")
    c.subtitle(ax1, "Top: daily precipitation. Bottom: daily mean of the hourly deviation (median over stations).")
    ax1.grid(axis="x", visible=False)
    ax2.plot(daily.index, c.log_to_pct(daily.total), color=c.SERIES[0], label=LABEL_OBS)
    ax2.plot(daily.index, c.log_to_pct(daily.weather), color=c.SERIES[1], label=LABEL_MODEL)
    c.zero_line(ax2)
    c.pct_formatter(ax2)
    ax2.set_ylabel("vs. normal")
    ax2.legend(loc="upper left", ncol=2)
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    c.save(fig, out, "03_timeline")


def plot_station_sensitivity(normalized_dir, out, n=10):
    coef = pd.read_csv(Path(normalized_dir) / "normal_flow_coefficients.csv").set_index("station")
    rain = coef["rain_both_pct"].sort_values()
    pick = pd.concat([rain.head(n), rain.tail(n)])
    colors = [c.SERIES[0] if v >= 0 else c.RED for v in pick]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(pick) + 1.4))
    y = np.arange(len(pick))
    ax.barh(y, pick.values, color=colors, height=0.7)
    ax.set_yticks(y, [c.short_station(s) for s in pick.index])
    for yi, v in zip(y, pick.values):
        ax.text(v, yi, f" {v:+.0f} % " if v >= 0 else f" {v:+.0f} % ", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=c.TEXT_2)
    ax.axvline(0, color=c.BASELINE, linewidth=1)
    ax.axhline(n - 0.5, color=c.GRID, linewidth=1, linestyle="--")
    c.pct_formatter(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Rain sensitivity per station ({n} lowest, {n} highest)")
    c.subtitle(ax, f"Model effect of rain now + in the previous hour. Network median {rain.median():+.0f} %.")
    c.save(fig, out, "04_station_rain_sensitivity")


def main():
    a = c.base_argparser("Weather effect on the normalized flow.").parse_args()
    c.apply_style()
    out = Path(a.out) if a.out else OUT_DIR
    df, freq = network_series(a.normalized)
    _, weather, _ = c.load_raw(a.data, freq)
    plot_rain_bins(df, weather, out)
    plot_temperature_bins(df, weather, out)
    plot_timeline(df, weather, out)
    plot_station_sensitivity(a.normalized, out)


if __name__ == "__main__":
    main()
