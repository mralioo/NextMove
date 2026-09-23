import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.obs_data import DB_PATH, db_exists, key, load_exp_runs, load_exp_turns
from utils.ui import page_header

st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")
page_header(
    "Experiments — what does each component choice change?",
    "Two questions, run through 12 configurations of the workflow. Each configuration ('arm') changes ONE component of the baseline "
    "(router, memory, MCP transport, ML engine, writer) so every difference can be attributed to that change.",
)
st.caption(f"Read from `{DB_PATH}` (tables `exp_runs`, `exp_turns`). Produced by `make experiments`. Design and hypotheses: `docs/experiments_plan.md`.")

PRED, BASE, ACTUAL, REF, GOOD, BAD = "#3b82f6", "#a855f7", "#f59e0b", "#9ca3af", "#22c55e", "#ef4444"
FACTOR_COLOR = {"-": REF, "router": PRED, "memory": BASE, "mcp": ACTUAL, "engine": GOOD, "writer": BAD}


def explain(what: str, read: str, verdict: tuple[str, str] | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**What it shows.** {what}")
        st.markdown(f"**How to read it.** {read}")
        if verdict:
            getattr(st, verdict[0])(verdict[1])


runs = load_exp_runs(key()) if db_exists() else pd.DataFrame()
if runs.empty:
    st.warning("No experiments yet. Run the suite with:\n\n```\nmake experiments ARGS=\"--list\"     # show the design\nmake experiments                   # run all 12 arms (~8 min)\n```")
    st.stop()

with st.expander("📖 The design in one minute", expanded=False):
    st.markdown(
        """
**Questions (same protocol in every arm).** **Q1** = the workbook's disruption question (U6 suspension; ground truth recomputed from the raw csv);
**Q2** = a follow-up that only makes sense after Q1 in the same session (*"Which of those stations should get staff first, and how sure are you?"*);
**Q1r** = Q1 again in a **new** session (exposes cross-session memory and run-to-run consistency).

**Factors and levels.** router: rules · llm · tfidf · jev — memory: session · none · episodic — MCP transport: stdio · in-memory · HTTP —
ML engine: TabPFN · empirical (no ML) — writer: small LLM · main LLM · template (no LLM).

**One factor at a time (OFAT).** Baseline **A00** (rules · session · stdio · TabPFN · small writer); each other arm changes exactly one factor.
**A01** repeats the baseline unchanged: the gap between A00 and A01 is the **noise floor** (LLM sampling, API jitter). A difference smaller than
the noise floor is not a finding.

**Quantitative:** latency (total and per stage), LLM/tool calls, tokens, quality score (route, LLM-judged completeness, fact accuracy vs ground truth, grounding,
readability, judge relevance / faithfulness / clarity), consistency between Q1 and Q1r, follow-up completeness, MCP start-up, ranking similarity of the ML engine.
**Qualitative:** the answers side by side plus the **LLM judge's** per-criterion verdicts and quoted justifications, and the failing checks.

**Limits.** One run per arm per question (n = 1): small effects are noise; interactions between factors are not measured; the judge is a small model
(possible bias toward its own style, some run-to-run variability); the quality score still saturates on easy items. See the plan for details.
"""
    )

runs = runs.sort_values("ts", ascending=False)
exps = list(dict.fromkeys(runs["exp_id"]))
labels = {e: f"{e} · {int((runs['exp_id'] == e).sum())} arms · {runs[runs['exp_id'] == e]['when'].iloc[0]:%Y-%m-%d %H:%M}" for e in exps}
eid = st.selectbox("Experiment run", exps, format_func=lambda e: labels[e])
R = runs[runs["exp_id"] == eid].sort_values("arm_id").set_index("arm_id")
turns = load_exp_turns(eid)
ok = R[R["status"] == "ok"]
S = {a: r["summary"] for a, r in ok.iterrows()}
if "A00" not in S:
    st.error("The baseline arm A00 is missing from this experiment run, so effects cannot be computed.")
    st.stop()
b = S["A00"]
REPS = [S[a] for a in S if R.loc[a, "factor"] == "-"]                    # baseline + its replicates


def base_value(metric_fn):
    """Baseline reference = mean over the identical replicates (A00, A01, A02)."""
    vals = [metric_fn(s) for s in REPS if metric_fn(s) is not None]
    return float(np.mean(vals)) if vals else None


def noise(metric_fn) -> float | None:
    """Noise floor = range (max - min) of a metric across the identical replicates."""
    vals = [metric_fn(s) for s in REPS if metric_fn(s) is not None]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else None


get_lat = lambda s: s["latency_mean_s"]
get_q = lambda s: s["quality"]
get_j = lambda s: s.get("judge")
n_lat, n_q, n_j = noise(get_lat), noise(get_q), noise(get_j)
B_LAT, B_Q, B_J = base_value(get_lat), base_value(get_q), base_value(get_j)

# ------------------------------------------------------------------------------------------ 1. arms table
st.markdown("### 1 · All arms at a glance")
rows = []
for a, r in R.iterrows():
    s = r["summary"]
    row = {"Arm": a, "Run name (full parameter set)": r["arm_name"], "Factor changed": r["factor"], "Status": r["status"]}
    if r["status"] == "ok":
        row.update({"Quality": s["quality"], "Judge": s.get("judge"), "Judge↔regex agreement": s.get("judge_agreement"), "Mean latency (s)": s["latency_mean_s"], "LLM calls": s["llm_calls"],
                    "Tool calls": s["tool_calls"], "Tokens (in+out)": s["tok_in"] + s["tok_out"], "Consistency Q1↔Q1r": s.get("consistency_q1_q1r"),
                    "Follow-up completeness": s.get("follow_up_completeness"), "Δ latency vs baseline": s["latency_mean_s"] - B_LAT,
                    "Δ quality vs baseline": s["quality"] - B_Q})
    else:
        row["Note"] = r["note"]
    rows.append(row)
tbl = pd.DataFrame(rows)
st.dataframe(tbl.round(3), use_container_width=True, hide_index=True)
st.caption(f"Baseline = mean of the {len(REPS)} identical replicates. Noise floor (range across replicates): latency **{'n/a' if n_lat is None else f'{n_lat:.2f} s'}**, quality **{'n/a' if n_q is None else f'{n_q:.3f}'}**, "
           f"judge **{'n/a' if n_j is None else f'{n_j:.3f}'}**. Differences below these are not findings.")

# ------------------------------------------------------------------------------------------ 2. latency
st.markdown("### 2 · Latency per turn, per arm")
lat_rows = [{"arm": f"{a}\n{ok.loc[a, 'label']}", "turn": t, "seconds": v} for a, s in S.items() for t, v in s["latency_s"].items()]
fig = px.bar(pd.DataFrame(lat_rows), x="arm", y="seconds", color="turn", barmode="group", color_discrete_map={"Q1": PRED, "Q2": BASE, "Q1r": ACTUAL})
if n_lat is not None:
    fig.add_hrect(y0=min(get_lat(s) for s in REPS), y1=max(get_lat(s) for s in REPS), fillcolor=REF, opacity=0.18, line_width=0, annotation_text="baseline replicates range")
fig.update_layout(height=420, xaxis_title="", yaxis_title="Seconds", legend=dict(orientation="h", y=-0.35))
st.plotly_chart(fig, use_container_width=True)
fast = min(S, key=lambda a: get_lat(S[a]))
explain(
    "Seconds per turn for every arm: Q1 (first question), Q2 (follow-up, same session) and Q1r (Q1 repeated in a new session). The grey band is the "
    "range spanned by the identical baseline replicates (A00, A01, A02) — the run-to-run noise.",
    "A bar clearly outside the band is a real speed difference. Q1r vs Q1 shows what memory and caches save on a repeated situation; Q2 is short "
    "because a follow-up needs no tools.",
    ("info", f"Fastest mean: **{fast}** ({ok.loc[fast, 'label']}) at {get_lat(S[fast]):.1f} s vs baseline {B_LAT:.1f} s."),
)

# ------------------------------------------------------------------------------------------ 3. effects
st.markdown("### 3 · Effect of each change against the baseline")
eff = []
for a, s in S.items():
    if R.loc[a, "factor"] == "-":
        continue
    eff.append({"arm": f"{a} {ok.loc[a, 'label']}", "factor": ok.loc[a, "factor"], "latency": get_lat(s) - B_LAT, "quality": s["quality"] - B_Q,
                "judge": None if get_j(s) is None or B_J is None else get_j(s) - B_J})
E = pd.DataFrame(eff)
c1, c2, c3 = st.columns(3)
for col, (m, title, nz) in zip((c1, c2, c3), (("latency", "Δ mean latency (s)  — lower is better", n_lat), ("quality", "Δ quality score  — higher is better", n_q),
                                              ("judge", "Δ judge score  — higher is better", n_j))):
    with col:
        d = E.dropna(subset=[m])
        fig = px.bar(d, x=m, y="arm", orientation="h", color="factor", color_discrete_map=FACTOR_COLOR, title=title)
        if nz is not None:
            fig.add_vrect(x0=-nz, x1=nz, fillcolor=REF, opacity=0.25, line_width=0)
        fig.update_layout(height=max(320, 30 * len(d) + 120), showlegend=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
explain(
    "The change in latency, quality and judge score of each arm relative to the baseline A00. The grey band is the noise floor.",
    "Only bars that leave the grey band are candidate effects. A bar inside the band is 'no measurable effect at n = 1'.",
)
verdicts = []
for _, r in E.iterrows():
    parts = []
    if n_lat is not None:
        parts.append(f"latency {r['latency']:+.1f} s ({'beyond' if abs(r['latency']) > n_lat else 'within'} noise)")
    if n_q is not None:
        parts.append(f"quality {r['quality']:+.2f} ({'beyond' if abs(r['quality']) > n_q else 'within'} noise)")
    verdicts.append(f"- **{r['arm']}** — " + "; ".join(parts))
st.markdown("**Verdicts vs noise floor**\n\n" + "\n".join(verdicts))

# ------------------------------------------------------------------------------------------ 4. factor panels
st.markdown("### 4 · Factor by factor")
t_router, t_mem, t_mcp, t_eng, t_wr = st.tabs(["Router", "Memory", "MCP transport", "ML engine", "Writer"])


def turn_val(arm, turn, key_):
    d = turns[(turns["arm_id"] == arm) & (turns["turn"] == turn)]
    return None if d.empty else d.iloc[0]["metrics"].get(key_)


with t_router:
    rr = []
    for a in ("A00", "R1", "R2", "R3"):
        if a not in R.index:
            continue
        r = R.loc[a]
        if r["status"] != "ok":
            rr.append({"Arm": a, "Router": r["params"]["router"], "Q1 route": "skipped", "Q2 route": "skipped", "Route latency Q1 (ms)": None, "Note": r["note"]})
            continue
        s = r["summary"]
        rr.append({"Arm": a, "Router": r["params"]["router"], "Q1 correct": turn_val(a, "Q1", "route_correct"), "Q2 correct (FOLLOW)": turn_val(a, "Q2", "route_correct"),
                   "Route latency Q1 (ms)": s["route_ms"].get("Q1"), "Route latency Q2 (ms)": s["route_ms"].get("Q2"), "Quality": s["quality"], "Judge": s.get("judge")})
    st.dataframe(pd.DataFrame(rr).round(3), use_container_width=True, hide_index=True)
    explain("Did each router send Q1 to the disruption playbook (C) and the follow-up Q2 to FOLLOW, and how long did the routing itself take?",
            "1.0 = routed correctly. Route latency is the router alone (the LLM router calls a model, the others do not). JEV shows 'skipped' unless JEV_API_KEY is set.")
with t_mem:
    mm = []
    for a in ("A00", "M1", "M2"):
        if a in S:
            s = S[a]
            mm.append({"Arm": a, "Memory": R.loc[a, "params"]["memory"], "Follow-up answered": s.get("follow_up_answered"), "Follow-up completeness": s.get("follow_up_completeness"),
                       "Q1 latency": s["latency_s"].get("Q1"), "Q1r latency": s["latency_s"].get("Q1r"), "Speed-up of Q1r": s.get("speedup_q1r"),
                       "Memory hit on Q1r": s.get("memory_hit_q1r"), "Consistency Q1↔Q1r": s.get("consistency_q1_q1r")})
    st.dataframe(pd.DataFrame(mm).round(3), use_container_width=True, hide_index=True)
    explain("With no memory the follow-up has nothing to refer to; with session memory it reuses the previous answer's facts; with episodic memory a repeated "
            "situation in a NEW session is served from a persistent store (tools skipped).",
            "Look at 'Follow-up completeness' (does the answer name the earlier stations, give an order, and say how sure it is) and at the speed-up of Q1r. "
            "Speed-ups smaller than the noise floor are not evidence.")
with t_mcp:
    cc = []
    for a in ("A00", "C1", "C2"):
        if a in S:
            s = S[a]
            cc.append({"Arm": a, "MCP transport": R.loc[a, "params"]["mcp"], "Server start-up (s)": s.get("mcp_startup_s"), "Tools time Q1 (s)": s["tools_s"].get("Q1"),
                       "Tools time Q1r (s)": s["tools_s"].get("Q1r"), "Tool calls": s["tool_calls"], "Mean latency (s)": s["latency_mean_s"]})
    st.dataframe(pd.DataFrame(cc).round(3), use_container_width=True, hide_index=True)
    explain("The same MCP tools reached three ways: a subprocess over stdio (production), inside the agent process (in-memory), or a separate process over HTTP.",
            "Tools time is the executor's wall time (tool calls run in parallel). Start-up is paid once per process. Answers should be identical; only speed differs.")
with t_eng:
    ee = []
    for a in ("A00", "E1"):
        if a in S:
            s = S[a]
            sim = s.get("vs_baseline_q1") or {}
            ee.append({"Arm": a, "Engine": R.loc[a, "params"]["engine"], "Top-3 stations": ", ".join(s.get("top3_q1", [])), "Same top station as A00": sim.get("top_station_same"),
                       "Top-3 overlap": sim.get("top3_overlap"), "Mean |Δ probability| (pts)": sim.get("mean_abs_prob_diff_pts"),
                       "Pair-order agreement": sim.get("pair_order_agreement"), "Tools time Q1 (s)": s["tools_s"].get("Q1")})
    st.dataframe(pd.DataFrame(ee).round(3), use_container_width=True, hide_index=True)
    if "A00" in S and "E1" in S:
        pa = turns[(turns["arm_id"] == "A00") & (turns["turn"] == "Q1")]
        pe = turns[(turns["arm_id"] == "E1") & (turns["turn"] == "Q1")]
        if not pa.empty and not pe.empty:
            fa, fe = pa.iloc[0]["facts"].get("press", []), pe.iloc[0]["facts"].get("press", [])
            comp = pd.DataFrame({"station": [p["s"] for p in fa], "TabPFN: % above own p95": [p["p"] for p in fa],
                                 "no closure": [p["p0"] for p in fa]}).merge(
                pd.DataFrame({"station": [p["s"] for p in fe], "Empirical: % above own p95": [p["p"] for p in fe]}), on="station", how="outer")
            st.dataframe(comp, use_container_width=True, hide_index=True)
    explain("The disruption scenario computed with the TabPFN demand model versus the same logic using the naive empirical station × hour quantiles (no ML).",
            "Same stations in the same order = the ML engine does not change WHO is under pressure; a large probability difference means it changes HOW SURE the "
            "system is. Earlier evaluation showed TabPFN ≈ baseline on point accuracy, so this arm tests whether that matters downstream.")
with t_wr:
    ww = []
    for a in ("A00", "W1", "W2"):
        if a in S:
            s = S[a]
            ww.append({"Arm": a, "Writer": R.loc[a, "params"]["writer"], "Write time Q1 (s)": s["write_s"].get("Q1"), "Tokens in+out": s["tok_in"] + s["tok_out"], "Quality": s["quality"],
                       "Judge": s.get("judge"), "Follow-up completeness": s.get("follow_up_completeness"), "Guard Q1": s["guard"].get("Q1"), "LLM calls": s["llm_calls"]})
    st.dataframe(pd.DataFrame(ww).round(3), use_container_width=True, hide_index=True)
    explain("Three ways to write the final brief: the small LLM (baseline), the shared main LLM, and a deterministic template with no LLM at all.",
            "Compare write time and judge score: the template is instant and always grounded but reads stiffer; the main model should follow the style and honesty "
            "rules best. The main-model arm uses 3 calls to the shared endpoint.")

# ------------------------------------------------------------------------------------------ 5. qualitative
st.divider()
st.markdown("### 5 · Qualitative comparison — the answers side by side")
turn_choice = st.radio("Turn", ["Q1", "Q2", "Q1r"], horizontal=True, help="Q1 = disruption question · Q2 = follow-up · Q1r = Q1 repeated in a new session")
chosen = st.multiselect("Arms to compare", list(ok.index), default=[a for a in ("A00", "M1", "W1", "W2") if a in ok.index][:4],
                        format_func=lambda a: f"{a} · {ok.loc[a, 'label']}")
if chosen:
    cols = st.columns(len(chosen))
    for col, a in zip(cols, chosen):
        d = turns[(turns["arm_id"] == a) & (turns["turn"] == turn_choice)]
        with col:
            st.markdown(f"**{a} · {ok.loc[a, 'label']}**")
            st.caption(ok.loc[a, "arm_name"])
            if d.empty:
                st.info("no data")
                continue
            r = d.iloc[0]
            st.markdown(r["answer"] or "_(empty)_")
            j = r["judge"]
            if j:
                st.caption(f"judge: relevance {j.get('relevance')}/5 · clarity {j.get('clarity')}/5 · faithfulness {j.get('faithfulness')}/5 — {j.get('note', '')}")
            fails = [c["name"] for c in r["checks"] if not c["ok"]]
            st.caption("✅ all checks passed" if not fails else "❌ " + "; ".join(fails))
    explain("The exact text the operator would have received in each selected arm, with the small-model judge's scores and the failing rubric checks.",
            "Read for what the numbers cannot show: wording, order of information, whether the caveat is clear, whether the follow-up really refers to the earlier stations. "
            "Differences in style between arms are qualitative findings; check them against the judge scores and the noise floor before drawing conclusions.")

st.markdown("### 6 · Setup of an arm")
arm_pick = st.selectbox("Arm", list(R.index), format_func=lambda a: f"{a} · {R.loc[a, 'label']}")
st.json({"run_name": R.loc[arm_pick, "arm_name"], "status": R.loc[arm_pick, "status"], "hypothesis_note": R.loc[arm_pick, "note"] or None,
         "parameters": R.loc[arm_pick, "params"], "setup": R.loc[arm_pick, "setup"]})
