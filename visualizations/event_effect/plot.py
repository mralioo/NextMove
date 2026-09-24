"""
Effect of events (and closures) on the NORMALIZED flow.

Uses normalized_flows.csv (deviation from a typical hour at that station) and the event /
closure -> station mapping of the pipeline (episodes.csv). Around every episode the values are
lined up on a common clock ("hours relative to the event") and summarized over all episodes.
Stations are grouped by network distance to the venue station: hop 0 = venue station itself,
hop 1 = direct neighbours, hop 2 = two stops away.

    python -m visualizations.event_effect.plot
    python -m visualizations.event_effect.plot --event "Guns"      # detail figure for one event
    python -m visualizations.event_effect.plot --table rest        # use the weather-free rest instead

Figures (in visualizations/event_effect/output/)
    01_event_profile.png          median deviation around events, aligned to start and to end
    02_event_peaks.png            per event: deviation at the venue station in the hour after the end
    03_attendance_vs_peak.png     does a bigger event give a bigger peak?
    04_closure_profile.png        closures: closed stations vs neighbours, aligned to the start
    05_event_detail_<name>.png    one event: actual vs normal passengers at the venue station
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nextmove_pipeline.episodes import hops_from
from visualizations import common as c

OUT_DIR = Path(__file__).resolve().parent / "output"
H = pd.Timedelta(hours=1)
HOP_LABELS = {0: "Venue station", 1: "1 stop away", 2: "2 stops away"}


# =========================================================================== data preparation
def hop_groups(adj, anchors, columns, max_hop=2):
    """{hop: [stations]} for stations with data, by network distance to the anchors."""
    dist = hops_from(adj, [a for a in anchors if a in adj or a in columns], max_hop)
    groups = {h: [] for h in range(max_hop + 1)}
    for s, h in dist.items():
        if s in columns:
            groups[h].append(s)
    return groups


def aligned_profile(table, adj, episodes, ref, offsets):
    """For every episode and hop: mean deviation (log) per whole hour relative to `ref`
    ('start' or 'end'). Returns a long DataFrame: id, hop, offset, value."""
    rows = []
    cols = set(table.columns)
    for _, ep in episodes.iterrows():
        groups = hop_groups(adj, ep.anchors, cols)
        t0 = ep[ref]
        window = table.loc[t0 + offsets[0] * H: t0 + (offsets[-1] + 1) * H - pd.Timedelta(seconds=1)]
        if window.empty:
            continue
        off = np.floor((window.index - t0) / H).astype(int)
        for h, stations in groups.items():
            if not stations:
                continue
            per_off = window[stations].mean(axis=1).groupby(off).mean()
            for o, v in per_off.items():
                rows.append(dict(id=ep.id, hop=h, offset=int(o), value=v))
    return pd.DataFrame(rows)


def summarize(profile):
    """Median and inter-quartile range over episodes, per hop and offset."""
    g = profile.dropna().groupby(["hop", "offset"])["value"]
    return pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
                         "n": g.size()}).reset_index()


# =========================================================================== figures
def _profile_panel(ax, summary, xlabel, band_hop=0):
    """Median line per hop (+ IQR band for the venue station), on a 'x normal' log axis."""
    for h, color in zip(sorted(summary.hop.unique()), c.SERIES):
        s = summary[summary.hop == h]
        if h == band_hop:
            ax.fill_between(s.offset, c.log_to_factor(s.q25), c.log_to_factor(s.q75),
                            color=c.BLUE_LIGHT, linewidth=0, zorder=0)
        ax.plot(s.offset, c.log_to_factor(s["median"]), color=color, marker="o", markersize=4,
                label=HOP_LABELS[h])
    ax.axhline(1, color=c.BASELINE, linewidth=1, zorder=1)
    ax.axvline(0, color=c.TEXT_2, linewidth=0.8, linestyle=":")
    c.factor_axis(ax)
    ax.set_xlabel(xlabel)


def _label_peaks(ax, summary):
    """Direct label next to the highest point of every line."""
    for h in sorted(summary.hop.unique()):
        s = summary[summary.hop == h]
        top = s.loc[s["median"].idxmax()]
        f = c.log_to_factor(top["median"])
        ax.annotate(f"{HOP_LABELS[h]}: {f:.1f}×", (top.offset, f), xytext=(8, 0),
                    textcoords="offset points", fontsize=9, va="center")


def plot_event_profile(table, adj, events, out, table_name):
    prof_s = summarize(aligned_profile(table, adj, events, "start", range(-4, 3)))
    prof_e = summarize(aligned_profile(table, adj, events, "end", range(-3, 4)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, gridspec_kw={"wspace": 0.12})
    _profile_panel(a1, prof_s, "hours relative to event START")
    _profile_panel(a2, prof_e, "hours relative to event END")
    _label_peaks(a2, prof_e)
    a1.set_xlim(-4.3, 2.3)
    a2.set_xlim(-3.3, 3.8)
    a1.set_ylabel("flow vs. normal (median over events)")
    a1.set_title("Events: the crowd arrives before the start and leaves after the end")
    c.subtitle(a1, f"{events.id.nunique()} events >= 2,000 visitors, {table_name}. "
                   "Band: middle 50 % of events at the venue station.")
    a1.legend(loc="upper left")
    c.save(fig, out, "01_event_profile")


def event_peaks(table, events):
    """Mean deviation (log) at the venue station in the first hour after the event end."""
    rows = []
    for _, ep in events.iterrows():
        st = [s for s in ep.anchors if s in table.columns]
        v = table.loc[ep.end: ep.end + H - pd.Timedelta(seconds=1), st].to_numpy()
        if st and v.size and not np.isnan(v).all():
            rows.append(dict(id=ep.id, name=ep["name"], station=st[0], start=ep.start,
                             attendance=ep.attendance, peak=np.nanmean(v)))
    return pd.DataFrame(rows)


def plot_event_peaks(peaks, out, table_name):
    p = peaks.sort_values("peak")
    labels = [f"{n[:38]}{'...' if len(n) > 38 else ''}  ·  {c.short_station(s)}, {d:%d %b}"
              for n, s, d in zip(p.name, p.station, p.start)]
    f = c.log_to_factor(p.peak.to_numpy())
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(p) + 1.4))
    y = np.arange(len(p))
    # bars grow from "normal" (1x) to the right (more) or left (less)
    ax.barh(y, f - 1, left=1, color=[c.SERIES[0] if v >= 1 else c.RED for v in f], height=0.7)
    ax.set_yticks(y, labels, fontsize=8)
    for yi, v in zip(y, f):
        ax.text(v, yi, f" {v:.1f}×" if v >= 1 else f"{v:.1f}× ", va="center",
                ha="left" if v >= 1 else "right", fontsize=8, color=c.TEXT_2)
    ax.axvline(1, color=c.BASELINE, linewidth=1)
    c.factor_axis(ax, "x")
    ax.set_xlim(0.5, f.max() * 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_title("Peak after each event at the venue station")
    c.subtitle(ax, f"First hour after the event end, {table_name}. "
                   f"Median {c.log_to_factor(p.peak.median()):.1f}× normal.")
    c.save(fig, out, "02_event_peaks")


def plot_attendance_vs_peak(peaks, out, table_name):
    f = c.log_to_factor(peaks.peak)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(peaks.attendance, f, s=48, color=c.SERIES[0], edgecolor=c.SURFACE, linewidth=1.5, zorder=3)
    for _, r in peaks.assign(f=f).nlargest(3, "f").iterrows():      # label the three biggest peaks
        name = r["name"] if len(r["name"]) <= 24 else r["name"][:24].rsplit(" ", 1)[0] + "..."
        ax.annotate(f"{name} ({c.short_station(r.station)})", (r.attendance, r.f),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, color=c.TEXT_2, va="center")
    ax.axhline(1, color=c.BASELINE, linewidth=1, zorder=1)
    c.factor_axis(ax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(right=peaks.attendance.max() + (peaks.attendance.max() - peaks.attendance.min()) * 0.45)
    corr = np.corrcoef(np.log(peaks.attendance), peaks.peak)[0, 1]
    ax.set_title("Bigger events, bigger peaks? Not in this data")
    c.subtitle(ax, f"Attendance only ranges {peaks.attendance.min():,.0f}-{peaks.attendance.max():,.0f}; "
                   f"r = {corr:.2f}, n = {len(peaks)}. The venue matters more than the count.")
    ax.set_xlabel("estimated attendance")
    ax.set_ylabel("flow vs. normal, hour after the end")
    c.save(fig, out, "03_attendance_vs_peak")


def plot_closure_profile(table, adj, closures, out, table_name):
    summary = summarize(aligned_profile(table, adj, closures, "start", range(-3, 6)))
    labels = {0: "Closed stations", 1: "1 stop away", 2: "2 stops away"}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ends = []
    for h, color in zip(sorted(summary.hop.unique()), c.SERIES):
        s = summary[summary.hop == h]
        y = c.log_to_pct(s["median"])
        ax.plot(s.offset, y, color=color, marker="o", markersize=4, label=labels[h])
        ends.append((s.offset.iloc[-1], y.iloc[-1], labels[h], color))
    c.zero_line(ax)
    ax.axvline(0, color=c.TEXT_2, linewidth=0.8, linestyle=":")
    ax.axvspan(0, np.median((closures.end - closures.start) / H), color=c.GRID, alpha=0.5, zorder=0)
    c.pct_formatter(ax)
    ax.set_xlim(-3.3, 8.5)
    c.label_line_ends(ax, ends)
    ax.set_xlabel("hours relative to closure START (shaded: median closure duration)")
    ax.set_ylabel("vs. normal (median over closures)")
    ax.set_title("Closures: flow drops at the closed stations")
    c.subtitle(ax, f"{closures.id.nunique()} closures, {table_name}")
    ax.legend(loc="lower right")
    c.save(fig, out, "04_closure_profile")


def plot_event_detail(table, events, normal_pass, raw, query, out):
    """Actual vs normal passengers at the venue station, plus the deviation, +-8 h around the event."""
    if query:
        hit = events[events["name"].str.contains(query, case=False, regex=False)]
        if hit.empty:
            raise SystemExit(f"no event matches '{query}'")
        ep = hit.sort_values("attendance").iloc[-1]
    else:
        ep = events.sort_values("attendance").iloc[-1]       # default: biggest event
    st = next(s for s in ep.anchors if s in table.columns)
    lo, hi = ep.start - 8 * H, ep.end + 3 * H
    actual, normal, dev = raw.loc[lo:hi, st], normal_pass.loc[lo:hi, st], table.loc[lo:hi, st]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2], "hspace": 0.3})
    for ax in (a1, a2):
        ax.axvspan(ep.start, ep.end, color=c.GRID, alpha=0.6, zorder=0)
    a1.plot(actual.index, actual, color=c.SERIES[0], label="Actual passengers")
    a1.plot(normal.index, normal, color=c.SERIES[1], linestyle="--", label="Normal (typical weather)")
    a1.set_ylabel("passengers per hour" if c.infer_freq(table.index) == "1h" else "passengers per 15 min")
    a1.set_ylim(bottom=0)
    a1.legend(loc="upper left")
    a1.set_title(f"{ep['name'][:60]}")
    c.subtitle(a1, f"{c.short_station(st)}, {ep.start:%a %d %b %Y}, "
                   f"{int(ep.attendance):,} visitors. Shaded: event time.")
    f = c.log_to_factor(dev)
    width = 0.8 * (dev.index[1] - dev.index[0]) / pd.Timedelta(days=1)       # bar width in days
    a2.bar(dev.index, f - 1, bottom=1, width=width, color=[c.SERIES[0] if v >= 1 else c.RED for v in f])
    a2.axhline(1, color=c.BASELINE, linewidth=1)
    c.factor_axis(a2)
    a2.set_ylabel("vs. normal")
    a2.grid(axis="x", visible=False)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    safe = "".join(ch if ch.isalnum() else "_" for ch in ep["name"][:30]).strip("_")
    c.save(fig, out, f"05_event_detail_{safe}")


# =========================================================================== main
def main():
    ap = c.base_argparser("Event and closure effects on the normalized flow.")
    ap.add_argument("--table", choices=["flows", "rest"], default="flows",
                    help="flows = normalized flow (default), rest = normalized flow minus weather part")
    ap.add_argument("--event", help="substring of an event name for the detail figure (default: biggest)")
    a = ap.parse_args()
    c.apply_style()
    out = Path(a.out) if a.out else OUT_DIR

    table = c.read_table(a.normalized, f"normalized_{a.table}")
    table_name = "normalized flow" if a.table == "flows" else "rest (weather removed)"
    episodes = c.load_episodes(a.normalized)
    events, closures = episodes[episodes.kind == "event"], episodes[episodes.kind == "closure"]
    raw, _, stations = c.load_raw(a.data, c.infer_freq(table.index))
    adj = c.load_network(a.data, stations)

    plot_event_profile(table, adj, events, out, table_name)
    peaks = event_peaks(table, events)
    plot_event_peaks(peaks, out, table_name)
    plot_attendance_vs_peak(peaks, out, table_name)
    plot_closure_profile(table, adj, closures, out, table_name)
    plot_event_detail(table, events, c.read_table(a.normalized, "normal_flow_passengers"), raw, a.event, out)


if __name__ == "__main__":
    main()
