import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.obs_data import BUDGET_S, DB_PATH, db_exists, key, load_runs, load_spans, load_tool_spans
from utils.ui import page_header

st.set_page_config(page_title="Observability", page_icon="🔭", layout="wide")
page_header(
    "Observability — traces and speed of the agent workflow",
    "Every question answered by the agent (CLI, `adk web`, benchmark, evaluation) is recorded: how long each stage took, "
    "which tools and models ran, whether the number guard intervened, and the full OpenTelemetry trace.",
)
st.caption(
    f"Read from `{DB_PATH}`. Written by ADK's own SQLite span exporter (traces) and an ADK plugin (one row per question). "
    "Speed budget for the charts: " f"**{BUDGET_S:.0f} s** per answer (`EVAL_LATENCY_BUDGET_S`)."
)

PRED, BASE, ACTUAL, REF, GOOD, BAD = "#3b82f6", "#a855f7", "#f59e0b", "#9ca3af", "#22c55e", "#ef4444"
STAGE_COLORS = {"route_s": "#22c55e", "tools_s": "#3b82f6", "write_s": "#a855f7", "other_s": "#9ca3af"}


def explain(what: str, read: str, verdict: tuple[str, str] | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**What it shows.** {what}")
        st.markdown(f"**How to read it.** {read}")
        if verdict:
            getattr(st, verdict[0])(verdict[1])


runs = load_runs(key()) if db_exists() else pd.DataFrame()
if runs.empty:
    st.warning(
        "No runs recorded yet. Ask the agent a question and it appears here:\n\n"
        "```\nmake agent-query Q=\"At what time does the commute flow peak at Rudow station usually take place?\"\n"
        "make bench          # 8 varied questions\nmake eval           # the workbook's TRAINING questions, scored\n```"
    )
    st.stop()

# ------------------------------------------------------------------------------------------ filters
with st.sidebar:
    st.markdown("### 🔭 Filters")
    src = st.multiselect("Source", sorted(runs["source"].dropna().unique()), default=sorted(runs["source"].dropna().unique()),
                         help="cli = make agent-query · web = adk web · bench = make bench · eval = make eval")
    mode = st.multiselect("Agent mode", sorted(runs["mode"].unique()), default=sorted(runs["mode"].unique()),
                          help="fast = router→executor→writer pipeline · llm = original supervisor loop")
    cats = st.multiselect("Category", sorted(runs["category"].dropna().unique()), default=sorted(runs["category"].dropna().unique()))

df = runs[runs["source"].isin(src) & runs["mode"].isin(mode) & (runs["category"].isin(cats) | runs["category"].isna())].copy()
if df.empty:
    st.info("No runs match the filters.")
    st.stop()

# ------------------------------------------------------------------------------------------ KPIs
guard_pass = df["guard"].fillna("").str.startswith("pass").mean()
within = (df["total_s"] <= BUDGET_S).mean()
c = st.columns(6)
c[0].metric("Questions answered", f"{len(df):,}")
c[1].metric("Mean latency", f"{df['total_s'].mean():.1f} s")
c[2].metric("Median (p50)", f"{df['total_s'].median():.1f} s")
c[3].metric("Slow tail (p95)", f"{df['total_s'].quantile(0.95):.1f} s")
c[4].metric("Within budget", f"{within:.0%}", help=f"Share of answers delivered in ≤ {BUDGET_S:.0f} s.")
c[5].metric("Guard: LLM answer kept", f"{guard_pass:.0%}", help="Share of answers where the writer's text passed the number/claim guard "
            "unchanged (the rest were replaced by the deterministic template).")
verdict = ("success", f"{within:.0%} of answers are inside the {BUDGET_S:.0f} s budget; median {df['total_s'].median():.1f} s.") \
    if within >= 0.95 else ("warning", f"Only {within:.0%} of answers are inside the {BUDGET_S:.0f} s budget — check the slowest runs below.")
getattr(st, verdict[0])(verdict[1])

# ------------------------------------------------------------------------------------------ 1. latency per run
st.markdown("### 1 · Latency of every answer, split by stage")
last = df.tail(80).reset_index(drop=True)
fig = go.Figure()
for col, name in (("route_s", "route (classify)"), ("tools_s", "tools (MCP + ML engine)"), ("write_s", "write (LLM + guard)"), ("other_s", "other / orchestration")):
    fig.add_trace(go.Bar(x=last.index, y=last[col], name=name, marker_color=STAGE_COLORS[col],
                         hovertext=last["question"].str[:80], hovertemplate="%{y:.2f}s — %{hovertext}"))
fig.add_hline(y=BUDGET_S, line_dash="dash", line_color=BAD, annotation_text=f"budget {BUDGET_S:.0f}s")
fig.update_layout(barmode="stack", height=380, xaxis_title="Run (oldest → newest, last 80)", yaxis_title="Seconds", legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, use_container_width=True)
mean_stage = df[["route_s", "tools_s", "write_s"]].mean()
dominant = mean_stage.idxmax().replace("_s", "")
explain(
    "One bar per question, stacked by where the time went: routing the question, running the tools (including the TabPFN ML engine), "
    "and the final LLM write + number guard. The dashed line is the response-time budget.",
    "Bars below the line are on time. A tall blue block means slow tools (usually a cold server or an uncached ML prediction); a tall "
    "purple block means the LLM was slow. The tiny green block is the deterministic router (about a hundredth of a second).",
    ("info", f"On average: route {mean_stage['route_s'] * 1000:.0f} ms · tools {mean_stage['tools_s']:.1f} s · write {mean_stage['write_s']:.1f} s. "
             f"The **{dominant}** stage dominates."),
)

# ------------------------------------------------------------------------------------------ 2 + 3
left, right = st.columns(2)
with left:
    st.markdown("### 2 · Latency by question category")
    fig = px.box(df.dropna(subset=["category"]), x="category", y="total_s", color="category", points="all",
                 labels={"total_s": "Seconds", "category": "Category"}, hover_data=["question"])
    fig.add_hline(y=BUDGET_S, line_dash="dash", line_color=BAD)
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    explain(
        "Answer time per question category (C = disruption response, D = station profile, A/B/E–H/X = not yet supported, answered as an "
        "honest decline).",
        "Each dot is one answer, the box is the middle 50%. Disruption answers (C) are the longest because they call four tools and "
        "write more text; declines are the quickest.",
    )
with right:
    st.markdown("### 3 · Which tools are slow?")
    ts = load_tool_spans(key())
    ts = ts[ts["trace_id"].isin(df["trace_id"])] if not ts.empty else ts
    if ts.empty:
        st.info("No tool spans recorded for the selected runs.")
    else:
        fig = px.box(ts, x="tool", y="duration_s", color="tool", points="all", labels={"duration_s": "Seconds", "tool": "MCP tool"})
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        slow = ts.groupby("tool")["duration_s"].median().sort_values(ascending=False)
        explain(
            "Duration of every MCP tool call (from the `mcp.tool` trace spans): closure lookup, graph reroute, the TabPFN scenario model, "
            "station profile, predictions.",
            "Tall boxes are the tools to optimise. The first call of a process is slow (server warm-up); repeat calls of cached "
            "closures take a fraction of a second.",
            ("info", f"Slowest median tool: **{slow.index[0]}** ({slow.iloc[0]:.2f} s)."),
        )

# ------------------------------------------------------------------------------------------ 4 + 5
left, right = st.columns(2)
with left:
    st.markdown("### 4 · Router tier and guard outcomes")
    df["guard_result"] = df["guard"].fillna("n/a").str.split(" ").str[0].replace({"": "n/a"})
    g = df.groupby(["guard_result"]).size().reset_index(name="runs")
    fig = px.bar(g, x="guard_result", y="runs", color="guard_result", labels={"guard_result": "Guard result", "runs": "Runs"})
    fig.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    tiers = df.groupby("tier").size()
    explain(
        "How the number/claim guard judged each written answer: `pass` = the LLM text was kept; `fallback` = it contained an ungrounded "
        "number or a capacity claim and was replaced by a template built from the facts; `template` = the LLM was too slow/failed.",
        "Mostly `pass` is healthy. Frequent `fallback` means the writer prompt needs tightening (answers stay correct but read stiffer).",
        ("info", f"Router: {int(tiers.get(0, 0))} questions routed deterministically (tier 0), {int(tiers.get(1, 0))} needed the LLM fallback (tier 1)."),
    )
with right:
    st.markdown("### 5 · Latency over time")
    fig = px.scatter(df, x="when", y="total_s", color="source", hover_data=["question", "category"], labels={"total_s": "Seconds", "when": ""})
    fig.add_hline(y=BUDGET_S, line_dash="dash", line_color=BAD)
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    explain(
        "Every recorded answer over time, coloured by where it was asked from.",
        "A drifting-up cloud after a change means a regression. Isolated spikes are usually a slow LLM call or a cold start.",
    )

# ------------------------------------------------------------------------------------------ trace explorer
st.divider()
st.markdown("### 6 · Trace explorer — follow one question through the workflow")
options = df.sort_values("ts", ascending=False)
choice = st.selectbox("Run", options["run_id"], format_func=lambda r: options.set_index("run_id").loc[r, "label"])
run = options.set_index("run_id").loc[choice]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", f"{run['total_s']:.2f} s")
c2.metric("Category", f"{run['category'] or '-'}", f"tier {int(run['tier']) if pd.notna(run['tier']) else '-'}", delta_color="off")
c3.metric("Tokens in → out", f"{int(run['tok_in']) if pd.notna(run['tok_in']) else 0} → {int(run['tok_out']) if pd.notna(run['tok_out']) else 0}")
c4.metric("LLM calls / tool calls", f"{int(run['n_llm_calls'] or 0)} / {int(run['n_tool_calls'] or 0)}")
st.markdown(f"**Question:** {run['question']}")
with st.expander("Answer delivered to the operator", expanded=True):
    st.markdown(run["answer"] or "_(none)_")

spans = load_spans(run["trace_id"]) if run["trace_id"] else pd.DataFrame()
if spans.empty:
    st.info("No spans stored for this run (traces are exported in the background; refresh in a moment).")
else:
    def kind(n: str) -> str:
        if n.startswith("invoke_agent") or n == "invocation":
            return "ADK agent"
        if n.startswith("llm."):
            return "LLM call"
        if n.startswith("mcp.tool"):
            return "MCP tool"
        if n.startswith("guard"):
            return "guard"
        if n.startswith("route") or n.startswith("executor"):
            return "pipeline step"
        return "MCP protocol / other"

    spans = spans.assign(kind=spans["name"].map(kind))
    t0 = pd.Timestamp("2000-01-01")
    spans["start"] = t0 + pd.to_timedelta(spans["start_ms"], unit="ms")
    spans["end"] = spans["start"] + pd.to_timedelta(spans["duration_ms"].clip(lower=1), unit="ms")
    spans["label"] = spans["name"] + "  (" + spans["duration_ms"].round(0).astype(int).astype(str) + " ms)"
    fig = px.timeline(spans, x_start="start", x_end="end", y="label", color="kind",
                      color_discrete_map={"ADK agent": REF, "LLM call": BASE, "MCP tool": PRED, "guard": GOOD, "pipeline step": ACTUAL,
                                          "MCP protocol / other": "#d1d5db"})
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(tickformat="%S.%L s", title="Time since the question arrived")
    fig.update_layout(height=max(360, 26 * len(spans) + 90), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    llm = spans[spans["name"].str.startswith("llm.")]
    inner = spans[~spans["name"].str.startswith(("invoke_agent", "invocation", "MCP send", "tools/call"))]   # skip wrapper spans
    explain(
        "The waterfall of this single question: the ADK agent stages (router → executor → writer), the tool calls the executor made "
        "(and the MCP protocol messages underneath), the LLM call and the guard check. Rows are nested top-to-bottom in call order.",
        "Read the bars left to right as time. Bars that start together ran in parallel (the executor calls several tools at once). "
        "The long bar is where this answer spent its time.",
        ("info", "Slowest step: **" + inner.loc[inner["duration_ms"].idxmax(), "name"] + f"** ({inner['duration_ms'].max():.0f} ms)."
                 + (f" LLM: {llm.iloc[0]['attrs'].get('gen_ai.request.model', '?')}." if len(llm) else "")),
    )
    with st.expander("Span details (attributes)"):
        tbl = spans[["name", "kind", "start_ms", "duration_ms"]].copy()
        tbl["attributes"] = spans["attrs"].apply(lambda a: {k: v for k, v in a.items() if k.startswith(("tmt.", "gen_ai.request", "gen_ai.usage", "gen_ai.tool", "gen_ai.agent.name"))})
        st.dataframe(tbl.round(1), use_container_width=True, hide_index=True)

t1, t2, t3 = st.tabs(["Plan (router output)", "Facts (executor output)", "Event timeline"])
with t1:
    st.json(json.loads(run["plan_json"]) if run["plan_json"] else {})
with t2:
    st.json(json.loads(run["facts_json"]) if run["facts_json"] else {})
with t3:
    ev = pd.DataFrame(json.loads(run["events_json"]) if run["events_json"] else [])
    st.dataframe(ev, use_container_width=True, hide_index=True)

st.divider()
st.markdown("### 7 · Recent runs")
show = df.sort_values("ts", ascending=False).head(50)[["when", "source", "mode", "category", "total_s", "route_ms", "tools_s", "write_s", "guard", "tok_in", "tok_out", "question"]]
st.dataframe(show.round(2), use_container_width=True, hide_index=True)
