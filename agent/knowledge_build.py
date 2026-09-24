"""Build the curated KNOWLEDGE BASE (ground truth, boundaries, insights) from the RAW data files.

    ./.venv/bin/python scripts/tasks.py kb-build        ->  knowledge/knowledge.json  (machine use: retrieval, sanity checks, MCP)
                             knowledge/knowledge.md    (human use + the text pushed to Cognee)

Every number is computed here from the CSVs with plain pandas — never typed by hand and never taken from the agent's own
tools — so the knowledge base can vouch for the tools. Three kinds of entries:

  boundary      what the data CANNOT support (no capacity data, dates outside the window, ...). Retrieved for every question
                the boundary applies to and used by the sanity checker.
  ground_truth  verified facts a good answer must agree with (closure records, Rudow's peak, energy ranking, resilience top 5,
                the Uber Arena effect, ...).
  insight       findings about the data / the models that shape how answers must be written (rain effect, events are local,
                the p95-exceedance ranking has no skill, ...).

The schema of an entry:  {id, kind, cats[A..X], text, value{...}, source}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "dashboard", REPO / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KB_DIR = REPO / "knowledge"


def _short(n: str) -> str:
    return re.sub(r"^(S\+U|U|S)\s+", "", re.sub(r"\s*\(Berlin\)\s*$", "", n)).strip()


def build(folder: str | None = None) -> list[dict]:
    from utils.data_loader import (DEFAULT_DATA_DIR, build_graph, load_closures, load_energy, load_events, load_flows, load_stations,
                                   load_weather, station_avg_flow)

    folder = folder or DEFAULT_DATA_DIR
    E: list[dict] = []

    def add(id_, kind, cats, text, value=None, source=""):
        E.append({"id": id_, "kind": kind, "cats": cats, "text": text, "value": value or {}, "source": source})

    fl = load_flows(folder)
    cols = [c for c in fl.columns if c != "timestamp" and not re.search(r"\.\d+$", c)]
    w = fl.set_index("timestamp")[cols].astype(float)
    cov0, cov1 = w.index.min(), w.index.max()
    st = load_stations(folder)
    lines = sorted({x.strip() for v in st["u_bahn_lines"] for x in str(v).split(",")})
    ev = load_events(folder)
    cl = load_closures(folder)
    wx = load_weather(folder).set_index("timestamp")

    # ------------------------------------------------------------------ boundaries
    add("B-COV", "boundary", list("ABCDEFGHPX"),
        f"The flow data covers {cov0:%Y-%m-%d %H:%M} to {cov1:%Y-%m-%d %H:%M} at 15-minute grain for {len(cols)} stations on the lines {', '.join(lines)} "
        "(there is no U4 data). Dates outside this window cannot be answered from data; only clearly labelled scenarios are possible.",
        {"start": str(cov0), "end": str(cov1), "n_stations": len(cols), "lines": lines}, "flows*.csv, stations_with_ubahn.csv")
    add("B-CAP", "boundary", list("ABCDEFGHPX"),
        "There is NO platform-capacity, headway, timetable or train-load data. 'Safe platform capacity' therefore cannot be measured or quoted; "
        "answers must say so and use a relative proxy (each station's own busiest-5% level, or predicted load) instead.",
        {"has_capacity_data": False}, "dataset schema")
    n_ven = int(ev["venue_name"].notna().sum())
    inn = int(ev["event_name"].str.contains("innotrans|messe", case=False, na=False).sum() + ev["venue_name"].fillna("").str.contains("messe|citycube", case=False).sum())
    add("B-EVT", "boundary", ["A", "P", "X"],
        f"The events file has {len(ev)} events ({n_ven} with a venue name) from {ev['began_local'].min():%Y-%m-%d} to {ev['began_local'].max():%Y-%m-%d}. Events carry a venue name and "
        f"address but NO station key or coordinates, and attendance is an organiser estimate. The file contains {inn} InnoTrans/Messe events.",
        {"n_events": len(ev), "n_with_venue": n_ven, "innotrans_events": inn}, "berlin_events*.csv")
    n500 = int((w.values == 500).sum())
    add("B-SIM", "boundary", list("ABCDEFGHP"),
        f"The passenger flows are simulated. {n500} readings equal exactly 500 (a clipping artefact, {n500 / w.size:.1%} of readings) although values up to {int(w.values.max())} exist.",
        {"n_at_500": n500, "max": int(w.values.max())}, "flows*.csv")
    reasons = cl["reason"].value_counts().to_dict()
    add("B-CLS", "boundary", ["C", "H"],
        f"There are {len(cl)} recorded closures (reasons: {', '.join(f'{k} x{v}' for k, v in reasons.items())}). They are simulated maintenance closures; "
        "closed stations read 0 and neighbouring stations show no measurable extra load.",
        {"n_closures": len(cl), "reasons": reasons}, "closures*.csv")
    add("B-OD", "boundary", ["H", "G", "X"],
        "The data has no origin-destination flows or route choices: which alternative routes passengers prefer, and why two stations are coupled, cannot be measured — only suggested.",
        {}, "dataset schema")
    add("B-ASSUME", "boundary", ["C"],
        "How many passengers divert around a closure, and to which stations, is ASSUMED (diversion shares 25/50/75 %, nearest open stations). Closure pressure is a model-based scenario, never a measurement.",
        {"shares": [25, 50, 75]}, "ml/disruption.py")
    add("B-NAME", "boundary", ["A", "P"],
        "The events file names the arena 'Uber Arena'; operators may say 'Mercedes-Benz Arena' (its earlier name). They are treated as the same venue — an assumption to be stated.",
        {"alias": {"mercedes-benz arena": "Uber Arena"}}, "berlin_events*.csv")
    add("B-EN", "boundary", ["E"],
        "Energy is daily MWh per line only; there is no rolling-stock, timetable or route-length data, so energy explanations are limited to ridership evidence.",
        {}, "energy_consumption*.csv")
    add("B-NEW", "boundary", list("ABCDEFGHPX"),
        "A new dataset (2026-09-22 to 2026-10-01) is provided on the final day; loaders merge every matching file (dedupe on timestamp) and coverage is read from the data.",
        {}, "dashboard/utils/data_loader.py")

    # ------------------------------------------------------------------ ground truth: closures
    for i, r in cl.reset_index(drop=True).iterrows():
        add(f"GT-CL-{i}", "ground_truth", ["C"],
            f"Closure #{i}: {r['description']} Start {r['when']:%Y-%m-%d %H:%M}, duration {r['duration']} ({r['duration_hours']:.2f} h), reason: {r['reason']}.",
            {"closure_id": int(i), "start": str(r["when"]), "hours": float(r["duration_hours"]), "reason": r["reason"], "line": r["affected_line"], "description": r["description"]},
            "closures*.csv")

    # ------------------------------------------------------------------ ground truth: station profile
    weekday = w[w.index.dayofweek < 5]
    hourly = weekday.groupby(weekday.index.hour).mean()
    peak_hour, peak_val = hourly.idxmax(), hourly.max()
    net_mean = float(peak_val.mean())
    for stn in ("U Rudow (Berlin)",):
        add("GT-D-" + _short(stn).upper(), "ground_truth", ["D"],
            f"{_short(stn)}: weekday commute peak at {int(peak_hour[stn])}:00 with {peak_val[stn]:.1f} passengers per 15 min; the mean of all stations' own weekday peaks is {net_mean:.1f}; "
            f"{_short(stn)} {'exceeds' if peak_val[stn] > net_mean else 'does not exceed'} it.",
            {"station": stn, "peak_hour": int(peak_hour[stn]), "peak_value": float(peak_val[stn]), "network_mean_peak": net_mean, "exceeds": bool(peak_val[stn] > net_mean)}, "flows*.csv")
    top_peak = peak_val.sort_values(ascending=False).head(5)
    add("GT-D-BUSIEST", "ground_truth", ["D", "P"],
        "Highest weekday hourly-average flows per 15 min: " + "; ".join(f"{_short(n)} {v:.0f} (peak {int(peak_hour[n])}:00)" for n, v in top_peak.items()) + ".",
        {"top": {n: float(v) for n, v in top_peak.items()}}, "flows*.csv")

    # ------------------------------------------------------------------ ground truth: energy
    en = load_energy(folder).set_index("timestamp")
    name_lines = st.groupby("station_name")["u_bahn_lines"].apply(lambda s: set(",".join(s).replace(" ", "").split(","))).to_dict()
    rows = []
    for line in en.columns:
        c_ = [c for c in cols if line in name_lines.get(c, set())]
        p_ = w[c_].sum(axis=1).resample("D").sum()
        idx = en.index.intersection(p_.index)
        rows.append((line, en.loc[idx, line].sum() * 1e6 / p_.loc[idx].sum(), p_.mean() / len(c_)))
    rows.sort(key=lambda x: -x[1])
    add("GT-E-RANK", "ground_truth", ["E"],
        f"Energy per passenger (Wh, total daily MWh / passengers at the line's stations): " + "; ".join(f"{l} {v:.0f}" for l, v, _ in rows)
        + f". Worst: {rows[0][0]} ({rows[0][1]:.0f} Wh), best: {rows[-1][0]} ({rows[-1][1]:.0f} Wh). Passengers are approximated per line (interchanges count on every line they serve).",
        {"ranking": {l: float(v) for l, v, _ in rows}, "worst": rows[0][0], "best": rows[-1][0]}, "energy_consumption*.csv, flows*.csv")

    # ------------------------------------------------------------------ ground truth: resilience
    g = build_graph(folder)
    id2n = dict(zip(st["station_id"], st["station_name"]))
    avg = station_avg_flow(folder).set_index("station_name")["avg_daily_passengers"].to_dict()
    import networkx as nx

    res = []
    for node in g.nodes():
        h = g.copy()
        h.remove_node(node)
        comps = sorted(nx.connected_components(h), key=len, reverse=True)
        cut = [n for c in comps[1:] for n in c]
        res.append((id2n.get(node, node), avg.get(id2n.get(node, node), 0) + sum(avg.get(id2n.get(n, n), 0) for n in cut), len(cut), len(comps)))
    res.sort(key=lambda x: -x[1])
    add("GT-F-TOP5", "ground_truth", ["F"],
        "Stations whose closure fragments the network most (passengers affected per day = own + cut-off stations): "
        + "; ".join(f"{_short(n)} {p:.0f} ({c} stations cut off, {f} parts)" for n, p, c, f in res[:5]) + ". Trains are assumed not to run through a closed station.",
        {"top": [{"station": n, "pax": float(p), "cut": int(c)} for n, p, c, _ in res[:5]]}, "berlin_ubahn_connections.csv, flows*.csv")

    # ------------------------------------------------------------------ ground truth: events
    ev = ev.copy()
    ev["start"] = ev["began_local"].dt.tz_localize(None) if getattr(ev["began_local"].dt, "tz", None) is not None else ev["began_local"]
    ev["end"] = pd.to_datetime(ev["estimated_end_local"], errors="coerce")
    if getattr(ev["end"].dt, "tz", None) is not None:
        ev["end"] = ev["end"].dt.tz_localize(None)
    ev["end"] = ev["end"].fillna(ev["start"] + pd.Timedelta(hours=2))
    arena = ev[ev["venue_name"].fillna("").str.contains("Uber Arena")]
    ratios = {}
    for stn in cols:
        rs = []
        for _, e in arena.iterrows():
            t0, t1 = e["end"], e["end"] + pd.Timedelta(hours=1)
            win = w.loc[(w.index >= t0) & (w.index < t1), stn].mean()
            ref = [w.loc[(w.index >= t0 + pd.Timedelta(weeks=k)) & (w.index < t1 + pd.Timedelta(weeks=k)), stn].mean() for k in (-2, -1, 1, 2)]
            ref = [x for x in ref if pd.notna(x)]
            if pd.notna(win) and ref and np.mean(ref) > 0:
                rs.append(win / np.mean(ref))
        if rs:
            ratios[stn] = float(np.median(rs))
    topr = sorted(ratios.items(), key=lambda kv: -kv[1])[:3]
    add("GT-A-UBER", "ground_truth", ["A"],
        f"Uber Arena: {len(arena)} events in the data (median attendance {arena['estimated_attendance'].median():.0f}). In the hour after they end the flow is, versus the same weekday hour in neighbouring weeks, "
        + "; ".join(f"{_short(n)} x{r:.1f}" for n, r in topr) + f"; Hermannplatz x{ratios.get('U Hermannplatz (Berlin)', float('nan')):.1f}. The event effect is local to the 1-2 stations next to the venue.",
        {"n_events": int(len(arena)), "top": {n: r for n, r in topr}, "hermannplatz": ratios.get("U Hermannplatz (Berlin)")}, "berlin_events*.csv, flows*.csv")

    # ------------------------------------------------------------------ insights: weather, topology
    rain = wx["prcp"].reindex(w.index) > 0.5
    tot = w.sum(axis=1)
    key = w.index.dayofweek * 96 + w.index.hour * 4 + w.index.minute // 15
    base = tot[~rain].groupby(key[~rain]).mean()
    ratio = float((tot[rain] / pd.Series(key[rain]).map(base).to_numpy()).mean())
    hit = (w == 500).sum(axis=1)
    add("I-RAIN", "insight", ["B", "P"],
        f"Rain (> 0.5 mm per 15 min, {int(rain.sum())} slots) lifts total network flow to x{ratio:.2f} of the dry value for the same weekday and time, and the number of stations pinned at 500 rises from "
        f"{hit[~rain].mean():.1f} to {hit[rain].mean():.1f} per slot.",
        {"rain_ratio": ratio, "hits_dry": float(hit[~rain].mean()), "hits_rain": float(hit[rain].mean())}, "weather_data*.csv, flows*.csv")
    hp = [n for n in id2n.values() if "Hermannplatz" in n]
    nb = sorted({_short(id2n.get(b if a in [k for k, v in id2n.items() if v in hp] else a)) for a, b in g.edges() if a in [k for k, v in id2n.items() if v in hp] or b in [k for k, v in id2n.items() if v in hp]} - {"Hermannplatz"})
    add("GT-U8", "ground_truth", ["C"],
        f"Hermannplatz is served by U7 and U8. Its direct neighbours are {', '.join(nb)}. No U8 station is named Neukölln: Rathaus Neukölln and Neukölln are U7 stations, so 'U8 between Hermannplatz and Neukölln' does not exist as such in the network data.",
        {"neighbours": nb}, "stations_with_ubahn.csv, berlin_ubahn_connections.csv")

    # ------------------------------------------------------------------ insights from stored validation runs
    val = KB_DIR / "validation.json"
    if val.exists():
        v = json.loads(val.read_text())
        pr = v.get("pressure")
        if pr:
            add("I-P-SKILL", "insight", ["P"],
                f"Ranking stations by the predicted busy-slot LOAD has skill: over {pr['n_days']} replay days the predicted top-3 stations had a mean observed peak of {pr['obs_peak_of_pred_top3']:.0f} per 15 min "
                f"versus {pr['obs_peak_mean_all']:.0f} for the average station, Spearman {pr['spearman_load']:.2f}. Ranking by the probability of exceeding a station's own p95 has NO skill "
                f"(Spearman {pr['spearman_p95']:.2f}): those exceedances are noise-driven. Load, not p95-exceedance, is the ranking to use.",
                pr, "knowledge/validation.json (evaluation/validate_pressure.py)")
        ml = v.get("ml")
        if ml:
            add("I-ML", "insight", ["C", "P", "D"],
                f"TabPFN demand model on held-out days: MAE {ml['mae_tabpfn']} vs {ml['mae_baseline']} for the naive station-slot mean, RMSE {ml['rmse_tabpfn']} vs {ml['rmse_baseline']}; pinball-loss skill "
                f"{ml['pinball_skill']:+.1%}. It adds context-conditioned probabilities more than a different decision.", ml, "ml/output/disruption_baseline_report.json")
    # ------------------------------------------------------------------ boundaries and ground truth from the QUALITY DATABASE (normalized data, see ml/quality_db.py)
    try:
        sys.path.insert(0, str(REPO / "ml"))
        import quality_db
        qdb = quality_db.QualityDB()
        if qdb.available:
            sb = qdb.sql("SELECT station, max_all, hard_ceiling FROM station_bounds ORDER BY max_all DESC")
            top = sb.iloc[0]
            eff = {e["group"]: e for e in qdb.event_effects()}
            issues = {i["topic"]: i["detail"] for i in qdb.data_issues()}
            wx_eff = qdb.weather_effects()["network_median_pct"]
            stt = qdb.status()
            add("Q-NORMAL", "boundary", ["A", "B", "C", "D", "P"],
                f"'Normal' flow is defined by the quality database: for every station, time of day and day type the flow that remains after removing event / closure windows and the effect of weather (typical weather). "
                f"Boundaries per station, day type and hour (p05-p95 of clean slots, highest value ever observed, hard ceiling = 1.5 x that maximum) are checked by the Inspector; a passenger figure above a station's "
                f"hard ceiling is impossible in this dataset.", {"train": stt["train_range"], "test": stt["test_range"], "hard_ceiling_factor": stt["hard_ceiling_factor"]}, "data/quality (ml/quality_db.py)")
            add("Q-CEILING", "boundary", ["A", "C", "P"],
                f"The highest passenger flow ever observed at any station is {top.max_all:.0f} per 15 minutes ({_short(top.station)}); the median station maximum is {sb.max_all.median():.0f}. "
                f"A predicted or claimed load above 1.5 x a station's own maximum is treated as impossible; above 1.15 x as extreme.", {"max": float(top.max_all), "station": top.station, "median_max": float(sb.max_all.median())}, "data/quality station_bounds")
            add("Q-EVENTS", "ground_truth", ["A"],
                "Measured peak uplift at the venue stations of the mapped events: " + "; ".join(f"{k}: median {v['median_ratio_peak']:.1f}x, p90 {v['p90_ratio_peak']:.1f}x (n={v['n']})" for k, v in eff.items() if k.startswith("event"))
                + ". Uplifts are peak slots at the anchor station and are noisy at the 15-minute grain.", {k: v for k, v in eff.items() if k.startswith("event")}, "data/quality event_effects")
            add("Q-WEATHER", "insight", ["B", "D", "P"],
                f"Weather explains very little of the flow: {issues.get('weather', '')} Median station effects: rain now/prev. hour {wx_eff.get('rain_both_pct')} %, +5 degC {wx_eff.get('temp_plus5C_vs_mean_pct')} %, school holiday weekday {wx_eff.get('school_holiday_weekday_pct')} %.",
                wx_eff, "data/quality coefficients")
            add("Q-OUTAGE", "boundary", ["B", "C", "H"], issues.get("outages", "") + " " + issues.get("noise", ""), {}, "data/quality outages")
            add("Q-TESTSHIFT", "boundary", ["A", "B", "C", "D", "P"], issues.get("test_shift", ""), {}, "data/quality meta")
    except Exception as e:                                        # the knowledge base must build without the quality database
        print(f"[warn] quality-database entries skipped: {type(e).__name__}: {e}")
    return E


def to_markdown(E: list[dict]) -> str:
    out = ["# NextMove knowledge base — ground truth, boundaries, insights", "",
           "Generated by `agent/knowledge_build.py` from the raw CSV files. Every number is computed, none typed by hand.", ""]
    for kind, title in (("boundary", "Boundaries (what the data cannot support)"), ("ground_truth", "Ground truth (verified facts)"), ("insight", "Insights (how answers must be written)")):
        out += [f"## {title}", ""]
        for e in E:
            if e["kind"] == kind:
                out.append(f"- **{e['id']}** [{', '.join(e['cats'])}] {e['text']} _(source: {e['source']})_")
        out.append("")
    return "\n".join(out)


def main() -> None:
    KB_DIR.mkdir(exist_ok=True)
    rep = REPO / "ml" / "output" / "disruption_baseline_report.json"
    val_p = KB_DIR / "validation.json"
    val = json.loads(val_p.read_text()) if val_p.exists() else {}
    if rep.exists():
        m = json.loads(rep.read_text())["heldout_metrics"]
        val["ml"] = {"mae_tabpfn": m["tabpfn_median"]["mae"], "mae_baseline": m["baseline_station_slot_mean"]["mae"], "rmse_tabpfn": m["tabpfn_mean"]["rmse"],
                     "rmse_baseline": m["baseline_station_slot_mean"]["rmse"], "pinball_skill": m["mean_pinball_loss"]["skill_vs_baseline"]}
        val_p.write_text(json.dumps(val, indent=1))
    E = build()
    (KB_DIR / "knowledge.json").write_text(json.dumps(E, indent=1, ensure_ascii=False, default=str))
    (KB_DIR / "knowledge.md").write_text(to_markdown(E))
    by = {}
    for e in E:
        by[e["kind"]] = by.get(e["kind"], 0) + 1
    print(f"knowledge base: {len(E)} entries {by} -> {KB_DIR.relative_to(REPO)}/knowledge.json + knowledge.md")


if __name__ == "__main__":
    main()
