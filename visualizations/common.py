"""
Shared helpers for all plot scripts: paths, data loading, colors and chart style.

All plots read
    raw data         training_dataset/            (via nextmove_pipeline.loading)
    normalized data  data/normalized/             (output of nextmove_pipeline.pipeline)
and write PNGs into   visualizations/<topic>/output/.

Colors follow a validated, colorblind-safe palette (blue / orange / aqua). Aqua has low contrast
on white, so every line is direct-labeled instead of relying on color alone.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # render to files, no window needed
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from nextmove_pipeline import config as pipeline_config           # noqa: E402
from nextmove_pipeline.loading import (load_flows, load_network,   # noqa: E402
                                       load_stations, load_weather, resample)

DEFAULT_DATA_DIR = pipeline_config.DEFAULT_DATA_DIR
DEFAULT_NORMALIZED_DIR = pipeline_config.DEFAULT_OUT_DIR

# --------------------------------------------------------------------------- colors
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"         # secondary text: axis labels, annotations
GRID = "#e4e3df"
BASELINE = "#b5b4ae"       # the "0 % = normal" reference line
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, always in this order
BLUE_LIGHT = "#cde2fb"     # band / range fill (sequential step 100 of blue)
RED = "#e34948"            # negative pole (closures), blue is the positive pole


def apply_style():
    """Recessive grid and axes, thin marks, readable text."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 22,                   # leaves room for the subtitle line
        "axes.labelsize": 10, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.8, "legend.frameon": False, "legend.fontsize": 9,
    })


def subtitle(ax, text):
    """Grey one-line explanation under the (left-aligned) title."""
    ax.text(0, 1.01, text, transform=ax.transAxes, color=TEXT_2, fontsize=9, va="bottom")


def zero_line(ax):
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)


def label_line_ends(ax, items, min_gap_frac=0.06):
    """Direct labels at the right end of several lines: items = [(x, y, text, color), ...].
    A dot marks each line end; the texts are pushed apart vertically so they never overlap.
    Call AFTER setting the axis limits. Identity then never depends on color alone."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    dx = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
    last = None
    for x, y, text, color in sorted(items, key=lambda it: it[1]):
        ly = y if last is None else max(y, last + gap)
        last = ly
        ax.plot([x], [y], "o", color=color, markersize=4, zorder=4)
        ax.text(x + dx, ly, text, color=TEXT, fontsize=9, va="center", ha="left")


def full_days(series: pd.Series) -> pd.Series:
    """Daily sums, without partial days at the start/end of the data (they would look like a crash)."""
    counts = series.resample("1D").count()
    daily = series.resample("1D").sum(min_count=1)
    return daily[counts >= counts.median() * 0.9]


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(out_dir).resolve() / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    shown = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    print(f"[plot] {shown}")
    return path


# --------------------------------------------------------------------------- units
def log_to_pct(x):
    """Normalized tables are log1p differences: 0.1 -> +10.5 %, 0.69 -> +100 %."""
    return (np.exp(x) - 1) * 100


def log_to_factor(x):
    """Log difference -> multiplication factor: 0 -> 1x (normal), 0.69 -> 2x, 2.08 -> 8x."""
    return np.exp(x)


def factor_axis(ax, axis="y"):
    """Log-scaled axis in 'x normal' units, for effects that span 0.5x to 100x (events)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log", base=2)
    a.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: "normal" if v == 1 else f"{v:g}×"))


def pct_formatter(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:+.0f} %" if v else "0 %")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def short_station(name: str) -> str:
    """'S+U Warschauer Str. (Berlin)' -> 'Warschauer Str.'"""
    s = name.replace(" (Berlin)", "")
    for p in ("S+U ", "U ", "S "):
        if s.startswith(p):
            return s[len(p):]
    return s


# --------------------------------------------------------------------------- data loading
def read_table(normalized_dir, name: str) -> pd.DataFrame:
    """One of the pipeline outputs, e.g. 'normalized_flows' -> DataFrame (index: timestamp)."""
    path = Path(normalized_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python -m nextmove_pipeline.pipeline --freq 1h")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def infer_freq(index: pd.DatetimeIndex) -> str:
    """Time grid of the normalized tables ('1h' or '15min'), so raw data can be resampled to match."""
    step = pd.Series(index).diff().mode().iloc[0]
    return "1h" if step >= pd.Timedelta("1h") else "15min"


def load_raw(data_dir, freq):
    """Raw flows + weather on the given grid, restricted to stations with metadata."""
    stations = load_stations(data_dir)
    flows, weather = resample(load_flows(data_dir, stations.station_name), load_weather(data_dir), freq)
    return flows, weather, stations


def load_episodes(normalized_dir) -> pd.DataFrame:
    """Events and closures with their anchor stations (list), as written by the pipeline.
    Duplicate listings of the same show (e.g. 'Box seat' / 'Premium package' tickets: same
    station, same start) are merged, keeping the one with the highest attendance."""
    ep = pd.read_csv(Path(normalized_dir) / "episodes.csv", parse_dates=["start", "end"])
    ep = (ep.sort_values("attendance", ascending=False)
            .drop_duplicates(subset=["kind", "anchors", "start"]).sort_values("start"))
    ep["anchors"] = ep["anchors"].str.split("; ")
    return ep.reset_index(drop=True)


def base_argparser(description):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data", default=str(DEFAULT_DATA_DIR), help="folder with the raw CSV files")
    ap.add_argument("--normalized", default=str(DEFAULT_NORMALIZED_DIR), help="folder with the pipeline output")
    ap.add_argument("--out", help="output folder for the PNGs (default: the topic's output/ folder)")
    return ap
