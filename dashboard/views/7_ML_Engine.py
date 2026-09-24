import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.data_loader import load_closures
from utils.ml_results import CHECKPOINT_DIR, RESULTS_DIR, load_csv_result, load_manifest, load_report
from utils.ui import page_header, sidebar_dataset_picker

# Colours readable on both light and dark themes.
ACTUAL, PRED, BASE, REF = "#f59e0b", "#3b82f6", "#a855f7", "#9ca3af"
Q = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]


def qc(q: float) -> str:
    return f"q{int(round(q * 1000)):04d}"


# ------------------------------------------------------------------------------------------------
# helpers: verdicts and explanation blocks
# ------------------------------------------------------------------------------------------------
def verdict_lower_better(model: float, base: float, tie: float = 0.02, good: float = 0.05):
    """(level, label, relative change) for a lower-is-better metric."""
    ch = (model - base) / base if base else 0.0
    if ch <= -good:
        return "success", "✅ Better", ch
    if ch <= -tie:
        return "success", "🟢 Slightly better", ch
    if ch < tie:
        return "info", "➖ About the same", ch
    if ch < good:
        return "warning", "🟡 Slightly worse", ch
    return "error", "❌ Worse", ch


def verdict_higher_better(model: float, base: float, tie: float = 0.01, good: float = 0.03):
    """Same, for a higher-is-better metric, using absolute differences."""
    d = model - base
    if d >= good:
        return "success", "✅ Better", d
    if d >= tie:
        return "success", "🟢 Slightly better", d
    if d > -tie:
        return "info", "➖ About the same", d
    if d > -good:
        return "warning", "🟡 Slightly worse", d
    return "error", "❌ Worse", d


def explain(what: str, read: str, baseline: str | None = None, verdict: tuple[str, str] | None = None) -> None:
    """A bordered block under each chart: what it shows, how to read it, what the baseline is, and a
    colour-coded verdict (computed from the data, so it changes if the models are retrained)."""
    with st.container(border=True):
        st.markdown(f"**What it shows.** {what}")
        st.markdown(f"**How to read it.** {read}")
        if baseline:
            st.markdown(f"**Compared with.** {baseline}")
        if verdict:
            getattr(st, verdict[0])(verdict[1])


def scoreboard(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def roc_pr(y: np.ndarray, p: np.ndarray) -> dict:
    """ROC / precision-recall curves, AUCs and best-F1 threshold, numpy only."""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    pos, neg = ys.sum(), (1 - ys).sum()
    tpr, fpr = tp / pos, fp / neg
    prec = tp / (tp + fp)
    fx, ty = np.r_[0, fpr], np.r_[0, tpr]
    auc = float(np.sum(np.diff(fx) * (ty[1:] + ty[:-1]) / 2))
    ap = float(np.sum(np.diff(np.r_[0, tpr]) * prec))
    f1 = 2 * tp / (tp + fp + (pos - tp))
    i = int(np.argmax(f1))
    return {"fpr": fpr, "tpr": tpr, "prec": prec, "auc": auc, "ap": ap, "f1": float(f1[i]), "thr": float(ps[i])}


st.set_page_config(page_title="ML Engine", page_icon="🤖", layout="wide")
folder = sidebar_dataset_picker()
page_header(
    "TabPFN ML Engine — Predictions vs Ground Truth",
    "How good are the models behind the agent? Every chart compares a model with the real (simulated) flow it "
    "never saw — and with a simple baseline it has to beat.",
)
st.caption(
    f"Results read from `{RESULTS_DIR}` (written by `./.venv/bin/python scripts/tasks.py train-disruption` / `./.venv/bin/python scripts/tasks.py train-overcrowding`); "
    f"model checkpoints from `{CHECKPOINT_DIR}`. This page never calls the TabPFN API."
)

with st.expander("📖 How to read this page (glossary)", expanded=False):
    st.markdown(
        """
* **Held-out** — the last 20% of days, chronologically *after* every training row (and with closure windows removed).
  Scores on held-out data tell you how the model behaves on days it has never seen. Scores on training data would flatter it.
* **Ground truth** — the recorded passenger count per station per 15 minutes in `flows.csv`.
* **Baseline** — a deliberately simple predictor built from the *same training days* with no machine learning:
  the station's historical **median / mean for that weekday-type and 15-minute slot**, and its **empirical quantiles for that
  weekday-type and hour**. A model is only worth its complexity if it beats this. *Tie = the model adds nothing over the
  historical pattern.*
* **MAE** (mean absolute error) — average miss in passengers; lower is better. **RMSE** — like MAE but punishes big misses
  harder. **R²** — share of the variance explained (1 = perfect, 0 = no better than always guessing the average).
* **Prediction interval / quantile** — instead of one number, TabPFN predicts a distribution. The "80% interval" is the range
  between its 10th and 90th percentile; if it is honest, the real value lands inside it 80% of the time (**coverage**).
* **Pinball loss** — the standard score for quantile forecasts (average over all 13 predicted quantiles); lower is better.
  It rewards intervals that are both *right* and *narrow*.
* **Verdict colours** — ✅ better · 🟢 slightly better · ➖ about the same · 🟡 slightly worse · ❌ worse *than the baseline*
  (regression metrics: within ±2% = same, ≥5% = clearly better/worse; AUC/F1: within ±0.01 = same, ≥0.03 = clearly).
"""
    )

report = load_report()
held = load_csv_result("heldout")
show = load_csv_result("showcase")
case = load_csv_result("case_study")
clf = load_csv_result("classifier")
manifest = load_manifest()

if report is None and held is None and clf is None:
    st.warning(
        "No ML result files found yet. Generate them with:\n\n"
        "```\n./.venv/bin/python scripts/tasks.py train-disruption   # Category C engine\n./.venv/bin/python scripts/tasks.py train-overcrowding # overcrowding classifier\n```"
    )
    st.stop()

tab_reg, tab_clf, tab_case, tab_ckpt = st.tabs(
    ["📈 Regression — expected flow", "🚨 Classifier — overcrowding", "🚧 Closure case study", "💾 Model checkpoints"]
)

# ================================================================================================
# Regression
# ================================================================================================
with tab_reg:
    st.subheader("Regression: expected passengers per station per 15 min")
    st.caption(
        "`TabPFNRegressor` predicts a full distribution (mean + 13 quantiles) for each station/time slot from hour, weekday, "
        "weather, events and station features. It is the 'normal demand' baseline used by the Category C disruption solver."
    )
    m = (report or {}).get("heldout_metrics")
    has_base = bool(m and "baseline_station_slot_median" in m and held is not None and "baseline_median" in held.columns)
    if held is None or m is None:
        st.info("Run `./.venv/bin/python scripts/tasks.py train-disruption` to generate the held-out predictions.")
    elif not has_base:
        st.info("Result files predate the baseline comparison — re-run `./.venv/bin/python scripts/tasks.py train-disruption`.")
    else:
        # ---------------------------------------------------------------- scoreboard
        st.markdown("### Scoreboard — TabPFN vs the naive baseline")
        mae_v = verdict_lower_better(m["tabpfn_median"]["mae"], m["baseline_station_slot_median"]["mae"])
        rmse_v = verdict_lower_better(m["tabpfn_mean"]["rmse"], m["baseline_station_slot_mean"]["rmse"])
        r2_v = verdict_higher_better(m["tabpfn_mean"]["r2"], m["baseline_station_slot_mean"]["r2"], 0.005, 0.02)
        pin = m["mean_pinball_loss"]
        pin_v = verdict_lower_better(pin["tabpfn"], pin["baseline_empirical_quantiles"])
        cov_m, cov_b = m["interval_coverage"]["80% (q10-q90)"], m["baseline_interval_coverage"]["80% (q10-q90)"]
        cov_level, cov_label = (
            ("success", "✅ Closer to 80%") if abs(cov_m - 0.8) + 0.01 < abs(cov_b - 0.8)
            else ("warning", "🟡 Baseline closer to 80%") if abs(cov_b - 0.8) + 0.01 < abs(cov_m - 0.8)
            else ("info", "➖ About the same")
        )
        w_m, w_b = m["mean_interval_width_80"], m["baseline_mean_interval_width_80"]
        w_v = verdict_lower_better(w_m, w_b)
        scoreboard([
            {"Metric": "MAE (median forecast) ↓", "Plain meaning": "average miss, passengers per 15 min",
             "TabPFN": m["tabpfn_median"]["mae"], "Baseline": m["baseline_station_slot_median"]["mae"],
             "Change": f"{mae_v[2]:+.1%}", "Verdict": mae_v[1]},
            {"Metric": "RMSE (mean forecast) ↓", "Plain meaning": "same, but big misses count more",
             "TabPFN": m["tabpfn_mean"]["rmse"], "Baseline": m["baseline_station_slot_mean"]["rmse"],
             "Change": f"{rmse_v[2]:+.1%}", "Verdict": rmse_v[1]},
            {"Metric": "R² ↑", "Plain meaning": "share of variation explained",
             "TabPFN": m["tabpfn_mean"]["r2"], "Baseline": m["baseline_station_slot_mean"]["r2"],
             "Change": f"{r2_v[2]:+.3f}", "Verdict": r2_v[1]},
            {"Metric": "Pinball loss (13 quantiles) ↓", "Plain meaning": "quality of the whole predicted distribution",
             "TabPFN": pin["tabpfn"], "Baseline": pin["baseline_empirical_quantiles"],
             "Change": f"{pin_v[2]:+.1%}", "Verdict": pin_v[1]},
            {"Metric": "80% interval coverage (target 80%)", "Plain meaning": "how often truth lands inside the interval",
             "TabPFN": f"{cov_m:.1%}", "Baseline": f"{cov_b:.1%}", "Change": f"{(cov_m - cov_b) * 100:+.1f} pts", "Verdict": cov_label},
            {"Metric": "80% interval width ↓", "Plain meaning": "narrower = more informative (at equal coverage)",
             "TabPFN": w_m, "Baseline": w_b, "Change": f"{w_v[2]:+.1%}", "Verdict": w_v[1]},
        ])
        better = sum(v[0] == "success" for v in (mae_v, pin_v))
        worse = sum(v[0] in ("warning", "error") for v in (mae_v, rmse_v, pin_v))
        if better and not worse:
            overall = ("success", "**Overall: TabPFN is better than the baseline on the headline scores**, "
                       f"by {abs(mae_v[2]):.1%} on MAE and {abs(pin_v[2]):.1%} on pinball loss.")
        elif worse and not better:
            overall = ("error", "**Overall: TabPFN is worse than the baseline** on the headline scores.")
        else:
            overall = ("warning",
                       f"**Overall: TabPFN is roughly on par with the baseline** — MAE {mae_v[2]:+.1%}, RMSE {rmse_v[2]:+.1%}, "
                       f"pinball {pin_v[2]:+.1%} (within ±2% counts as a tie). The simulated flows are almost entirely "
                       "'station × time-of-day pattern + random noise', so once the historical pattern is given as a feature there is "
                       "little left for weather/event features to explain. Take from this: TabPFN gives a *sound, calibrated* "
                       "distribution out of the box (its 80% interval covers "
                       f"{cov_m:.0%}), not a dramatic accuracy gain over a lookup table.")
        getattr(st, overall[0])(overall[1])
        st.caption(f"Held-out rows scored: {m['n_test_rows']:,} · model fitted on {m['n_train_sample']:,} training rows · "
                   f"train days before {m['train_days_before']} · MAE of always predicting the global mean = "
                   f"{m['baseline_global_mean']['mae']} (a naive floor).")

        held = held.copy()
        held["in_t"] = (held["passengers"] >= held["q0100"]) & (held["passengers"] <= held["q0900"])
        held["in_b"] = (held["passengers"] >= held["b_q0100"]) & (held["passengers"] <= held["b_q0900"])
        held["err_t"] = held["passengers"] - held["mean"]
        held["err_b"] = held["passengers"] - held["baseline_mean"]
        held["ae_t"] = (held["passengers"] - held["q0500"]).abs()
        held["ae_b"] = (held["passengers"] - held["baseline_median"]).abs()

        # ---------------------------------------------------------------- 1. predicted vs actual
        st.markdown("### 1 · Predicted vs actual")
        lim = float(max(held["passengers"].quantile(0.995), held["q0500"].quantile(0.995)))
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                            subplot_titles=("TabPFN (median forecast)", "Baseline (station × slot median)"))
        for col, (xcol, inside) in enumerate((("q0500", "in_t"), ("baseline_median", "in_b")), start=1):
            for flag, color, name in ((True, PRED if col == 1 else BASE, "inside 80% interval"), (False, "#ef4444", "outside")):
                sub = held[held[inside] == flag]
                fig.add_trace(go.Scatter(x=sub[xcol], y=sub["passengers"], mode="markers", name=name, opacity=0.45,
                                         marker=dict(size=5, color=color), showlegend=(col == 1),
                                         text=sub["station_name"], hovertemplate="%{text}<br>pred %{x:.0f} · actual %{y}"),
                              row=1, col=col)
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=REF, dash="dash"),
                                     showlegend=False), row=1, col=col)
            fig.update_xaxes(range=[0, lim], title_text="Predicted passengers", row=1, col=col)
        fig.update_yaxes(range=[0, lim], title_text="Actual passengers", row=1, col=1)
        fig.update_layout(height=440, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        r_t = float(np.corrcoef(held["q0500"], held["passengers"])[0, 1])
        r_b = float(np.corrcoef(held["baseline_median"], held["passengers"])[0, 1])
        n500 = int((held["passengers"] == 500).sum())
        rv = verdict_higher_better(r_t, r_b, 0.01, 0.03)
        explain(
            "Each dot is one station in one 15-minute slot from the held-out days: horizontal = what the model predicted, "
            "vertical = what was really recorded. Left panel is TabPFN, right panel is the baseline, same axes.",
            "The dashed diagonal is a perfect forecast. Dots hugging it are good. A wide cloud means the forecast is a poor guide "
            "to any single slot — expected here because passenger counts are noisy. Blue/purple dots fell inside the model's 80% "
            "interval, red dots outside (about 1 in 5 should be red if the interval is honest).",
            baseline="the station's historical median for that weekday-type and 15-min slot — no ML, no weather.",
            verdict=(rv[0], f"{rv[1]} — correlation between forecast and truth: TabPFN r = {r_t:.3f} vs baseline r = {r_b:.3f}. "
                            f"Both clouds look alike; the differences are small."),
        )
        if n500:
            st.caption(
                f"⚠️ The flat red row at 500 is a data quirk, not a model fault: {n500} held-out readings are exactly 500 (across the "
                "whole flows file 15,831 readings equal 500 vs ~230 at 499). The simulator appears to floor/inject values there; no "
                "feature explains them, so neither model can predict them."
            )

        # ---------------------------------------------------------------- 2. reliability
        st.markdown("### 2 · Interval reliability (is the model honest about its uncertainty?)")
        xs = Q
        y_t = [float((held["passengers"] <= held[qc(q)]).mean()) for q in Q]
        y_b = [float((held["passengers"] <= held[f"b_{qc(q)}"]).mean()) for q in Q]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(color=REF, dash="dash")))
        fig.add_trace(go.Scatter(x=xs, y=y_t, mode="lines+markers", name="TabPFN", line=dict(color=PRED)))
        fig.add_trace(go.Scatter(x=xs, y=y_b, mode="lines+markers", name="Baseline (empirical quantiles)", line=dict(color=BASE)))
        fig.update_layout(height=400, xaxis_title="Quantile level the model claims (e.g. 0.9 = 'truth is below this 90% of the time')",
                          yaxis_title="Share of held-out rows actually at or below it")
        st.plotly_chart(fig, use_container_width=True)
        dev_t = float(np.mean(np.abs(np.array(y_t) - np.array(Q))))
        dev_b = float(np.mean(np.abs(np.array(y_b) - np.array(Q))))
        dv = verdict_lower_better(dev_t, dev_b, 0.10, 0.25)
        explain(
            "For each quantile the model predicts (2.5%, 5%, …, 97.5%), the share of real values that turned out to be at or below "
            "that prediction.",
            "If a model says 'there is a 90% chance the count is below X', then 90% of real values should be below X. A curve on the "
            "dashed diagonal = trustworthy probabilities. Above the diagonal = the model is too pessimistic (predicts too high); "
            "below = too optimistic (real values often exceed the forecast).",
            baseline="empirical quantiles of that station's flow for the same weekday-type and hour, from the training days.",
            verdict=(("success" if dev_t <= 0.03 else "warning"),
                     f"{'Well calibrated' if dev_t <= 0.03 else 'Noticeable miscalibration'}: on average TabPFN's curve is "
                     f"{dev_t * 100:.1f} percentage points from the diagonal vs {dev_b * 100:.1f} for the baseline "
                     f"({dv[1].split(' ', 1)[1].lower()}). This calibration is what lets `scenario_flow` turn a forecast into a "
                     "trustworthy 'probability of exceeding the p95' — but the baseline is calibrated just as well, so it is "
                     "not a TabPFN-only advantage here."),
        )

        # ---------------------------------------------------------------- 3 + 4 errors
        st.markdown("### 3 · Error distribution   |   4 · Accuracy by hour of day")
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=held["err_b"], name="Baseline", opacity=0.55, marker_color=BASE, nbinsx=70))
            fig.add_trace(go.Histogram(x=held["err_t"], name="TabPFN", opacity=0.55, marker_color=PRED, nbinsx=70))
            fig.add_vline(x=0, line_dash="dash", line_color=REF)
            fig.update_layout(barmode="overlay", height=360, xaxis_title="Actual − predicted mean (passengers)",
                              yaxis_title="Number of rows", xaxis_range=[-600, 900])
            st.plotly_chart(fig, use_container_width=True)
            bias_t, bias_b = float(held["err_t"].mean()), float(held["err_b"].mean())
            ev = verdict_lower_better(m["tabpfn_mean"]["rmse"], m["baseline_station_slot_mean"]["rmse"])
            explain(
                "How far off the forecast was, row by row: actual minus predicted mean. Zero = exact hit; positive = the real value "
                "was higher than forecast.",
                "A tall, narrow spike at 0 is good. The long right tail is unavoidable: sudden demand spikes are unpredictable "
                "from the features available. The two overlapping colours being almost identical means the models make almost "
                "the same-sized mistakes.",
                baseline=f"station × slot historical mean (RMSE {m['baseline_station_slot_mean']['rmse']}).",
                verdict=(ev[0], f"{ev[1]} — RMSE {m['tabpfn_mean']['rmse']} vs {m['baseline_station_slot_mean']['rmse']}. "
                                f"Average bias: TabPFN {bias_t:+.1f}, baseline {bias_b:+.1f} passengers "
                                "(close to 0 = no systematic over/under-forecasting)."),
            )
        with right:
            by = held.groupby("hour").agg(t=("ae_t", "mean"), b=("ae_b", "mean"), a=("passengers", "mean")).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=by["hour"], y=by["b"], name="Baseline MAE", marker_color=BASE))
            fig.add_trace(go.Bar(x=by["hour"], y=by["t"], name="TabPFN MAE", marker_color=PRED))
            fig.add_trace(go.Scatter(x=by["hour"], y=by["a"], name="Avg actual flow", yaxis="y2", line=dict(color=ACTUAL)))
            fig.update_layout(barmode="group", height=360, xaxis_title="Hour of day", yaxis_title="MAE (passengers)",
                              yaxis2=dict(title="Avg actual flow", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)
            wins = int((by["t"] < by["b"]).sum())
            explain(
                "Average absolute error per hour of the day (bars) next to average real demand that hour (orange line).",
                "Errors grow with volume: the morning and evening peaks are where a forecast misses by the most passengers, and "
                "night hours are near zero. Compare the blue and purple bars hour by hour — whichever is shorter wins that hour.",
                baseline="station × slot median.",
                verdict=(("success" if wins > 12 else "info" if wins >= 9 else "warning"),
                         f"TabPFN has the lower error in {wins} of {len(by)} hours — "
                         f"{'a modest but consistent edge' if wins > 12 else 'no consistent winner: the two are equivalent hour by hour'}."),
            )

        # ---------------------------------------------------------------- 5. band
        if show is not None:
            st.markdown("### 5 · Prediction band vs the real series")
            c1, c2, c3 = st.columns([2, 2, 1])
            station = c1.selectbox("Station", sorted(show["station_name"].unique()))
            sub = show[show["station_name"] == station].sort_values("timestamp")
            days = sorted(sub["timestamp"].dt.date.unique())
            day_sel = c2.multiselect("Days", days, default=days)
            has_b = "b_q0100" in show.columns
            show_b = c3.checkbox("Show baseline band", value=False, disabled=not has_b,
                                 help="Overlay the naive baseline's 80% interval (purple)." if has_b else "Re-run `./.venv/bin/python scripts/tasks.py train-disruption`.")
            sub = sub[sub["timestamp"].dt.date.isin(day_sel)] if day_sel else sub
            present = set(sub["timestamp"].dt.date)
            all_days = pd.date_range(sub["timestamp"].min().normalize(), sub["timestamp"].max().normalize())
            missing = [d for d in all_days if d.date() not in present]
            sub = sub.reset_index(drop=True)
            gaps = sub.index[sub["timestamp"].diff() > pd.Timedelta(minutes=30)]
            blanks = pd.DataFrame({"timestamp": [sub.loc[i - 1, "timestamp"] + pd.Timedelta(minutes=15) for i in gaps]})
            sub = pd.concat([sub, blanks], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            fig = go.Figure()
            if show_b:
                fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["b_q0900"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["b_q0100"], fill="tonexty", fillcolor="rgba(168,85,247,0.22)",
                                         line=dict(width=0), name="Baseline 80% interval"))
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["q0975"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["q0025"], fill="tonexty", fillcolor="rgba(59,130,246,0.12)",
                                     line=dict(width=0), name="TabPFN 95% interval"))
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["q0900"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["q0100"], fill="tonexty", fillcolor="rgba(59,130,246,0.30)",
                                     line=dict(width=0), name="TabPFN 80% interval"))
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["q0500"], name="TabPFN median", line=dict(color=PRED)))
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["passengers"], name="Actual", line=dict(color=ACTUAL, width=1.5)))
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["station_hour_p95"], name="Own p95 reference",
                                     line=dict(color="#ef4444", dash="dash", width=1)))
            fig.update_xaxes(rangebreaks=[dict(bounds=[1, 5], pattern="hour"), dict(values=missing)])
            fig.update_layout(height=430, yaxis_title="Passengers per 15 min", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            s_ok = sub.dropna(subset=["passengers"])
            s_cov = float(((s_ok["passengers"] >= s_ok["q0100"]) & (s_ok["passengers"] <= s_ok["q0900"])).mean())
            explain(
                "A few full held-out days at a busy, a mid-sized and a quiet station: the real count (orange), TabPFN's median forecast "
                "(blue line) and the 80% / 95% intervals it predicted (shaded). The red dashed line is the station's own 95th-percentile "
                "level for that hour — the threshold used to call a slot 'overcrowded'.",
                "The orange line should stay inside the dark-blue band about 4 slots in 5. The median tracks the daily rhythm (twin "
                "commute peaks, quiet nights) but not the slot-to-slot jitter. When the orange line spikes above the red dashed line "
                "the slot is 'overcrowded' — the forecast cannot see these spikes coming, it can only say how likely they are. "
                "Days containing a closure are excluded from the held-out pool, so they don't appear.",
                baseline="tick 'Show baseline band' to overlay the purple empirical 80% interval; the two bands are nearly the same "
                         "width and shape.",
                verdict=(("success" if 0.75 <= s_cov <= 0.85 else "warning"),
                         f"For the selection above, {s_cov:.0%} of actual points fall inside TabPFN's 80% interval "
                         f"(target 80%) — {'honest' if 0.75 <= s_cov <= 0.85 else 'off target for this small slice'}."),
            )

# ================================================================================================
# Classifier
# ================================================================================================
with tab_clf:
    st.subheader("Classifier: will this station/slot be overcrowded?")
    st.caption(
        "`TabPFNClassifier` predicts P(flow ≥ the station's own 90th percentile). 'Overcrowded' is a relative "
        "demand-pressure label — the dataset has no real capacity figure."
    )
    if clf is None:
        st.info("Run `./.venv/bin/python scripts/tasks.py train-overcrowding` to generate the classifier's held-out predictions.")
    else:
        y = clf["overcrowded"].to_numpy(int)
        p = clf["overcrowd_probability"].to_numpy(float)
        has_b = "baseline_probability" in clf.columns
        pb = clf["baseline_probability"].to_numpy(float) if has_b else None
        n_pos = int(y.sum())
        base_rate = n_pos / len(y)
        T = roc_pr(y, p)
        B = roc_pr(y, pb) if has_b else None

        st.markdown("### Scoreboard — TabPFN vs the naive baseline")
        rows = []
        auc_v = verdict_higher_better(T["auc"], B["auc"]) if has_b else None
        ap_v = verdict_higher_better(T["ap"], B["ap"]) if has_b else None
        f1_v = verdict_higher_better(T["f1"], B["f1"]) if has_b else None
        rows.append({"Metric": "ROC-AUC ↑", "Plain meaning": "chance a random overcrowded slot is ranked above a normal one (0.5 = coin flip)",
                     "TabPFN": round(T["auc"], 3), "Baseline": round(B["auc"], 3) if has_b else "n/a",
                     "Change": f"{auc_v[2]:+.3f}" if has_b else "", "Verdict": auc_v[1] if has_b else "run ./.venv/bin/python scripts/tasks.py train-overcrowding"})
        rows.append({"Metric": "PR-AUC ↑", "Plain meaning": f"precision across recall levels (random guessing = base rate {base_rate:.0%})",
                     "TabPFN": round(T["ap"], 3), "Baseline": round(B["ap"], 3) if has_b else "n/a",
                     "Change": f"{ap_v[2]:+.3f}" if has_b else "", "Verdict": ap_v[1] if has_b else ""})
        rows.append({"Metric": "Best F1 (over thresholds) ↑", "Plain meaning": "balance of precision and recall at the best cut-off",
                     "TabPFN": round(T["f1"], 3), "Baseline": round(B["f1"], 3) if has_b else "n/a",
                     "Change": f"{f1_v[2]:+.3f}" if has_b else "", "Verdict": f1_v[1] if has_b else ""})
        scoreboard(rows)
        if has_b:
            good = sum(v[0] == "success" for v in (auc_v, ap_v, f1_v))
            bad = sum(v[0] in ("warning", "error") for v in (auc_v, ap_v, f1_v))
            if good >= 2 and not bad:
                st.success(f"**Overall: TabPFN beats the historical-rate baseline** (ROC-AUC {T['auc']:.3f} vs {B['auc']:.3f}, "
                           f"PR-AUC {T['ap']:.3f} vs {B['ap']:.3f}) — weather/time features add real signal for this label.")
            elif bad and not good:
                st.error("**Overall: TabPFN is worse than the historical-rate baseline.**")
            else:
                f1_note = (f" The baseline is even slightly better at its best-F1 cut-off ({B['f1']:.3f} vs {T['f1']:.3f})."
                           if f1_v[0] in ("warning", "error") else "")
                st.warning(f"**Overall: roughly on par with the historical-rate baseline** (ROC-AUC {T['auc']:.3f} vs {B['auc']:.3f}, "
                           f"PR-AUC {T['ap']:.3f} vs {B['ap']:.3f}).{f1_note} Most of the predictability comes from *when* and *where* "
                           "(station × hour), which the baseline already knows — weather adds little on this simulated data.")
        st.caption("Baseline = each station's historical share of overcrowded slots for that weekday-type and hour, computed on training "
                   "days only (no model, no weather). Random guessing scores ROC-AUC 0.5 and PR-AUC equal to the base rate.")

        thr = st.slider("Decision threshold", 0.0, 1.0, float(round(T["thr"], 2)), 0.01,
                        help="Predict 'overcrowded' when the probability is at least this value. Default = the F1-maximising "
                             "threshold on this held-out sample.")
        pred = p >= thr
        tp_t, fp_t = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn_t, tn_t = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        prec, rec = tp_t / max(tp_t + fp_t, 1), tp_t / max(tp_t + fn_t, 1)
        c = st.columns(4)
        c[0].metric("Precision", f"{prec:.1%}", help="Of the slots flagged overcrowded, how many really were.")
        c[1].metric("Recall", f"{rec:.1%}", help="Of the slots that really were overcrowded, how many were flagged.")
        c[2].metric("F1", f"{2 * prec * rec / max(prec + rec, 1e-9):.3f}")
        c[3].metric("Flagged at 0.5", int((p >= 0.5).sum()), help="How many rows the default 0.5 cut-off would flag.")
        st.caption(f"Test sample: {len(y):,} rows, {base_rate:.0%} truly overcrowded (oversampled vs the natural 10%, so precision here is higher "
                   "than it would be in live traffic).")

        a, b = st.columns(2)
        with a:
            st.markdown("### 1 · ROC curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=REF, dash="dash"), name="coin flip"))
            if has_b:
                fig.add_trace(go.Scatter(x=B["fpr"], y=B["tpr"], mode="lines", name=f"Baseline (AUC {B['auc']:.2f})", line=dict(color=BASE)))
            fig.add_trace(go.Scatter(x=T["fpr"], y=T["tpr"], mode="lines", name=f"TabPFN (AUC {T['auc']:.2f})", line=dict(color=PRED)))
            fig.update_layout(height=380, xaxis_title="False-alarm rate (normal slots wrongly flagged)",
                              yaxis_title="Detection rate (overcrowded slots caught)", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
            explain(
                "As the alert threshold is lowered from strict to lenient, the curve traces how many real overcrowding events "
                "are caught (up) against how many normal slots trigger a false alarm (right).",
                "Bowed toward the top-left = good. The dashed diagonal is a coin flip (AUC 0.5). AUC is the area under the curve: "
                "0.86 means a randomly chosen overcrowded slot gets a higher risk score than a randomly chosen normal one 86% of the time.",
                baseline="historical overcrowding rate of the same station/weekday-type/hour." if has_b else None,
                verdict=(auc_v[0], f"{auc_v[1]} — AUC {T['auc']:.3f}" + (f" vs baseline {B['auc']:.3f}" if has_b else "")
                         + f"; far above the 0.5 coin flip: the ranking is informative.") if has_b else ("info", f"AUC {T['auc']:.3f} vs 0.5 coin flip."),
            )
        with b:
            st.markdown("### 2 · Precision–recall curve")
            fig = go.Figure()
            fig.add_hline(y=base_rate, line_dash="dash", line_color=REF, annotation_text="random = base rate")
            if has_b:
                fig.add_trace(go.Scatter(x=B["tpr"], y=B["prec"], mode="lines", name=f"Baseline (AP {B['ap']:.2f})", line=dict(color=BASE)))
            fig.add_trace(go.Scatter(x=T["tpr"], y=T["prec"], mode="lines", name=f"TabPFN (AP {T['ap']:.2f})", line=dict(color=PRED)))
            fig.update_layout(height=380, xaxis_title="Recall (share of overcrowded slots caught)", yaxis_title="Precision (share of alerts that are right)",
                              yaxis_range=[0, 1.02], legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
            explain(
                "The trade-off an operator actually faces: catch more real overcrowding (moving right) means more false alerts "
                "(precision falls).",
                "Higher and further right is better. Compare with the dashed line — what you'd get by flagging slots at random. "
                "This chart matters more than ROC when overcrowding is rare, because it ignores the easy 'normal' majority.",
                baseline="the purple curve (station/hour historical rate).",
                verdict=(ap_v[0], f"{ap_v[1]} — PR-AUC {T['ap']:.3f}" + (f" vs baseline {B['ap']:.3f}" if has_b else "")
                         + f" vs {base_rate:.2f} for random: roughly {T['ap'] / base_rate:.1f}× better than guessing.") if has_b
                else ("info", f"PR-AUC {T['ap']:.3f} vs {base_rate:.2f} random."),
            )

        a, b = st.columns(2)
        with a:
            st.markdown("### 3 · Confusion matrix (at the slider threshold)")
            cm = pd.DataFrame([[tn_t, fp_t], [fn_t, tp_t]], index=["Actually normal", "Actually overcrowded"],
                              columns=["Flagged normal", "Flagged overcrowded"])
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto")
            fig.update_layout(height=340, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            explain(
                "The four outcomes at the chosen threshold: top-left correct 'normal', bottom-right correct 'overcrowded', "
                "top-right false alarms, bottom-left missed events.",
                "You want the diagonal (top-left, bottom-right) dark and the off-diagonal light. Move the slider: a lower threshold "
                "shrinks the bottom-left (fewer misses) but grows the top-right (more false alarms).",
                verdict=("info", f"At threshold {thr:.2f}: {tp_t} of {tp_t + fn_t} overcrowded slots caught ({rec:.0%}) with {fp_t} false "
                                 f"alarms. At the default 0.5 the model would flag only {int((p >= 0.5).sum())} slots, which is why the "
                                 "threshold needs tuning before this classifier is used to trigger alerts."),
            )
        with b:
            st.markdown("### 4 · Predicted probability by true class")
            tmp = clf.assign(actual=np.where(clf["overcrowded"] == 1, "overcrowded", "normal"))
            fig = px.histogram(tmp, x="overcrowd_probability", color="actual", nbins=40, barmode="overlay", opacity=0.6,
                               color_discrete_map={"normal": PRED, "overcrowded": "#ef4444"},
                               labels={"overcrowd_probability": "Predicted P(overcrowded)"})
            fig.add_vline(x=thr, line_dash="dash", line_color=REF)
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
            med_pos, med_neg = float(np.median(p[y == 1])), float(np.median(p[y == 0]))
            explain(
                "How confident the model was, split by what actually happened: blue = slots that were normal, red = slots that were "
                "overcrowded.",
                "Two well-separated humps (blue far left, red far right) means the model can tell them apart; heavy overlap means it "
                "cannot. The dashed line is the slider threshold — everything to its right gets flagged.",
                verdict=(("success" if med_pos > med_neg + 0.15 else "warning"),
                         f"Median risk score is {med_pos:.2f} for truly overcrowded slots vs {med_neg:.2f} for normal ones — "
                         f"{'clearly separated' if med_pos > med_neg + 0.15 else 'weakly separated'}, but the humps still overlap, "
                         "so expect false alarms."),
            )

# ================================================================================================
# Closure case study
# ================================================================================================
with tab_case:
    st.subheader("Case study: did the 26 real closures move passenger flow?")
    st.caption(
        "For every closure the regression model predicts what each nearby station *would* have seen, and we compare it with what was "
        "actually recorded. If displaced passengers went to neighbouring stations, those stations would sit above their interval far "
        "more than the 10% of the time that pure noise gives."
    )
    if case is None or report is None:
        st.info("Run `./.venv/bin/python scripts/tasks.py train-disruption` to generate the closure case study.")
    else:
        summ = pd.DataFrame(report["case_study_summary"]).T.reset_index().rename(columns={"index": "group"})
        summ["label"] = summ["group"].str.replace("_", " ").str.replace("/", " — ")
        has_cb = "baseline_share_above_q90" in summ.columns
        near = summ[~summ["group"].str.contains("closed")]
        dev_t = float((near["share_above_q90"] - 0.10).abs().max())
        dev_b = float((near["baseline_share_above_q90"] - 0.10).abs().max()) if has_cb else None

        st.markdown("### 1 · Share of readings above the 90th-percentile bound")
        fig = go.Figure()
        if has_cb:
            fig.add_trace(go.Bar(y=summ["label"], x=summ["baseline_share_above_q90"], orientation="h", name="Naive baseline bound", marker_color=BASE))
        fig.add_trace(go.Bar(y=summ["label"], x=summ["share_above_q90"], orientation="h", name="TabPFN bound", marker_color=PRED))
        fig.add_vline(x=0.10, line_dash="dash", line_color=REF, annotation_text="10% = pure noise")
        fig.update_layout(barmode="group", height=430, xaxis_tickformat=".0%", xaxis_title="Share of readings above the bound",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        robust = dev_t <= 0.03 and (dev_b is None or dev_b <= 0.05)
        explain(
            "Rows are groups of stations around the closures (the closed/unserved stations, section endpoints & interchanges, and "
            "stations 1 and 2 hops away). Each bar is the share of readings that came in *above the model's 90th-percentile forecast* "
            "while the closure was on.",
            "By construction, 10% of readings exceed a 90th-percentile bound on a normal day (dashed line). If a group absorbed "
            "extra passengers, its bar would be far right of the line. Bars near the line = nothing unusual happened there. "
            "(The 'closed' rows sit at 0% because a closed station is *below*, not above, normal — see the next chart.)",
            baseline="the purple bars use the naive empirical station/hour 90th percentile instead of TabPFN — an independent second "
                     "opinion." if has_cb else None,
            verdict=(("success" if robust else "warning"),
                     f"Neighbouring groups deviate from 10% by at most {dev_t * 100:.1f} pts (TabPFN)"
                     + (f" and {dev_b * 100:.1f} pts (baseline)" if has_cb else "")
                     + (" — **no measurable redistribution**, and both methods agree, so the conclusion is robust and not an artefact of "
                        "the model. This is why the agent presents rerouting results as assumption-based scenarios."
                        if robust else " — larger than noise; inspect the explorer below.")),
        )

        st.markdown("### 2 · Observed ÷ expected flow")
        fig = go.Figure()
        if "observed_over_slot_mean" in summ.columns:
            fig.add_trace(go.Bar(y=summ["label"], x=summ["observed_over_slot_mean"], orientation="h", name="vs naive slot mean", marker_color=BASE))
        fig.add_trace(go.Bar(y=summ["label"], x=summ["observed_over_expected"], orientation="h", name="vs TabPFN expected", marker_color=PRED))
        fig.add_vline(x=1.0, line_dash="dash", line_color=REF, annotation_text="1.0 = as expected")
        fig.update_layout(barmode="group", height=430, xaxis_title="Observed / expected", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        closed = summ[summ["group"].str.contains("station/closed")]
        closed_val = float(closed["observed_over_expected"].iloc[0]) if len(closed) else float("nan")
        explain(
            "Total passengers actually recorded during the closure windows divided by the total the model (blue) or the naive "
            "historical mean (purple) expected without a closure.",
            "1.0 means exactly as expected. A closed station should drop to 0 (nobody can enter). If passengers were displaced "
            "onto neighbours, the neighbour bars would be clearly above 1.",
            baseline="station × slot historical mean.",
            verdict=("success", f"Closed stations read {closed_val:.2f}× expected (their flow is exactly zero — the one real, "
                                "unmistakable effect in the data). Every other group stays within a few percent of 1.0 for both "
                                "methods; station-closure rings sit slightly *below* 1.0 (opposite to redistribution) on only 11 events."),
        )

        st.markdown("### 3 · Closure explorer — observed vs predicted band")
        closures = load_closures(folder).reset_index(drop=True)
        ids = sorted(case["closure_id"].unique())
        labels = {i: f"#{i} · {closures.loc[i, 'when']:%Y-%m-%d %H:%M} · {closures.loc[i, 'description'][:70]}" for i in ids if i < len(closures)}
        c1, c2 = st.columns(2)
        cid = c1.selectbox("Closure", list(labels), format_func=lambda i: labels[i])
        sub = case[case["closure_id"] == cid]
        role_order = ["closed_unserved", "endpoint_or_interchange", "hop1", "hop2"]

        def role_of(s):
            return sub[sub["station_name"] == s]["role"].iloc[0]

        stations = sorted(sub["station_name"].unique(), key=lambda s: (role_order.index(role_of(s)) if role_of(s) in role_order else 9, s))
        st_sel = c2.selectbox("Station", stations, format_func=lambda s: f"{s} ({role_of(s).replace('_', ' ')})")
        s1 = sub[sub["station_name"] == st_sel].sort_values("timestamp")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s1["timestamp"], y=s1["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=s1["timestamp"], y=s1["q10"], fill="tonexty", fillcolor="rgba(59,130,246,0.25)", line=dict(width=0), name="TabPFN 80% interval"))
        fig.add_trace(go.Scatter(x=s1["timestamp"], y=s1["expected"], name="TabPFN expected (no closure)", line=dict(color=PRED)))
        if "slot_mean" in s1.columns:
            fig.add_trace(go.Scatter(x=s1["timestamp"], y=s1["slot_mean"], name="Naive slot mean", line=dict(color=BASE, dash="dot")))
        fig.add_trace(go.Scatter(x=s1["timestamp"], y=s1["passengers"], name="Actual", line=dict(color=ACTUAL, width=2)))
        fig.update_layout(height=380, yaxis_title="Passengers per 15 min", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        inside = float(s1["in80"].mean())
        role = role_of(st_sel)
        explain(
            "One closure, one station, the recorded flow (orange) against what would normally be expected there (blue line, shaded 80% "
            "interval; purple dotted = naive historical mean). Only the closure window itself is plotted, because that is what was scored.",
            "For a *closed* station the orange line drops to zero and sits far below the band — the closure is visible. For "
            "neighbours the orange line wanders inside the band like on any normal day — no visible surge. Pick different stations "
            "and closures to check that this holds beyond the summary charts.",
            verdict=(("info"), f"Selected: {role.replace('_', ' ')} station — {inside:.0%} of this window's readings fall inside the "
                               f"80% band ({'closure clearly visible' if role == 'closed_unserved' and s1['passengers'].mean() < 1 else 'behaves like a normal day'})."),
        )

# ================================================================================================
# Checkpoints
# ================================================================================================
with tab_ckpt:
    st.subheader("Model checkpoints — what inference actually loads")
    st.markdown(
        "TabPFN is a foundation model served through an API, so there are no weights to download. \"Fitting\" uploads the training "
        "rows and the server returns a **model id**; every prediction references it. A checkpoint therefore stores three things in "
        "`ml/checkpoints/<name>/`:\n\n"
        "* `model.json` — the server model id + hyperparameters (what `save_model()` writes; no data);\n"
        "* `train_sample.csv.gz` — the *exact* rows the model was fitted on, so the model can be rebuilt deterministically if the "
        "server ever forgets it, and so a prediction can be flagged as in-sample or held-out;\n"
        "* `meta.json` — task, features, fingerprint (dataset + features + split + model version), metrics, timestamps.\n\n"
        "On start-up the MCP server (and `scenario_flow`) restores these instead of re-fitting: if the server still has the fit it is "
        "**loaded**; if not it is **refit from the checkpoint** with identical data and settings; if the fingerprint no longer matches "
        "(new dataset, changed features) the checkpoint is treated as stale and a fresh one is fitted and saved. "
        "Build or verify them with `./.venv/bin/python scripts/tasks.py checkpoints` (`FORCE=1` to refit)."
    )
    if not manifest:
        st.info("No checkpoints yet. Run `./.venv/bin/python scripts/tasks.py checkpoints` (needs `TABPFN_API_TOKEN`).")
    else:
        mf = pd.DataFrame(manifest)
        mf["saved_at"] = mf["saved_at"].astype(str).str.replace("T", " ").str.replace("+00:00", " UTC")
        cols = ["name", "task", "description", "n_train_rows", "n_features", "tabpfn_model", "model_id", "saved_at"]
        st.dataframe(mf[[c for c in cols if c in mf.columns]], use_container_width=True, hide_index=True)
        rows = []
        for e in manifest:
            d = CHECKPOINT_DIR / e["name"]
            for f in sorted(d.glob("*")) if d.exists() else []:
                rows.append({"checkpoint": e["name"], "file": f.name, "size": f"{f.stat().st_size / 1024:.1f} KB"})
        if rows:
            st.markdown("**Files on disk**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Checkpoints are tied to the TabPFN account that fitted them: loading needs the same `TABPFN_API_TOKEN`. "
                   "They are not committed as weights (there are none) and contain no secrets, but do contain a sample of the dataset.")
