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
        "```\n./.venv/bin/python scripts/tasks.py agent-query \"At what time does the commute flow peak at Rudow station usually take place?\"\n"
        "./.venv/bin/python scripts/tasks.py eval           # the workbook's TRAINING questions, scored\n```"
    )
    st.stop()

# ------------------------------------------------------------------------------------------ filters
with st.sidebar:
    st.markdown("### 🔭 Filters")
    src = st.multiselect("Source", sorted(runs["source"].dropna().unique()), default=sorted(runs["source"].dropna().unique()),
                         help="cli = ./.venv/bin/python scripts/tasks.py agent-query · web = adk web · eval = ./.venv/bin/python scripts/tasks.py eval")
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
        if n.startswith(("dispatcher", "supervisor", "route.", "guardrails.", "history.", "llm.route")):        # (old traces used the code names)
            return "Dispatcher"
        if n.startswith(("analyst", "worker")):
            return "Analyst"
        if n.startswith(("inspector", "evaluator")):
            return "Inspector"
        if n.startswith("mcp.tool"):
            return "MCP call"
        if n.startswith(("llm.", "gen_ai")):
            return "LLM call"
        if n.startswith(("guard", "writer.", "kb.", "kg.")):
            return "writer / checks / memory"
        if n.startswith("executor"):
            return "Analyst"
        return "MCP protocol"

    spans = spans.assign(kind=spans["name"].map(kind))
    t0 = pd.Timestamp("2000-01-01")
    spans["start"] = t0 + pd.to_timedelta(spans["start_ms"], unit="ms")
    spans["end"] = spans["start"] + pd.to_timedelta(spans["duration_ms"].clip(lower=1), unit="ms")
    spans["label"] = spans["name"] + "  (" + spans["duration_ms"].round(0).astype(int).astype(str) + " ms)"
    fig = px.timeline(spans, x_start="start", x_end="end", y="label", color="kind",
                      color_discrete_map={"ADK agent": REF, "Dispatcher": ACTUAL, "Analyst": "#0ea5e9", "Inspector": GOOD, "MCP call": PRED, "LLM call": BASE,
                                          "writer / checks / memory": "#f59e0b", "MCP protocol": "#d1d5db"})
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(tickformat="%S.%L s", title="Time since the question arrived")
    fig.update_layout(height=max(360, 26 * len(spans) + 90), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    llm = spans[spans["name"].str.startswith("llm.")]
    inner = spans[~spans["name"].str.startswith(("invoke_agent", "invocation", "MCP send", "tools/call"))]   # skip wrapper spans
    explain(
        "The waterfall of this single question, every step: the Dispatcher (rules, guardrails, history), the Analyst ⇄ Inspector rounds, each MCP call (and the MCP protocol "
        "messages underneath), the evaluator's checks and LLM, the writer's LLM call, the guard, sources, sanity check and knowledge-graph write. Rows are nested top-to-bottom in call order.",
        "Read the bars left to right as time. Bars that start together ran in parallel (the executor calls several tools at once). "
        "The long bar is where this answer spent its time.",
        ("info", "Slowest step: **" + inner.loc[inner["duration_ms"].idxmax(), "name"] + f"** ({inner['duration_ms'].max():.0f} ms)."
                 + (f" LLM: {llm.iloc[0]['attrs'].get('gen_ai.request.model', '?')}." if len(llm) else "")),
    )
    with st.expander("Span details (attributes)"):
        tbl = spans[["name", "kind", "start_ms", "duration_ms"]].copy()
        tbl["attributes"] = spans["attrs"].apply(lambda a: {k: v for k, v in a.items() if k.startswith(("tmt.", "gen_ai.request", "gen_ai.usage", "gen_ai.tool", "gen_ai.agent.name"))})
        st.dataframe(tbl.round(1), use_container_width=True, hide_index=True)

timing = json.loads(run["timing_json"]) if run.get("timing_json") else {}
hand = timing.get("handover", {})
t_time, t_steps, t_mcp, t_hand, t_plan, t_facts, t_ev = st.tabs(["Timing & LLM inference", "Steps & payloads", "MCP calls", "Hand-overs (route → result → verdict → writer)", "Plan", "Facts", "ADK events"])
with t_time:
    stg = timing.get("stages")
    if not stg:
        st.info("No stage timings stored for this run (recorded before this feature).")
    else:
        explain("Where the time of this question went, stage by stage, and every LLM inference (router, evaluator, writer) with its model, time and tokens.",
                "Stages run one after the other; the MCP calls inside the worker run in parallel, so their sum can exceed the worker time. LLM time is network + generation.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", f"{stg.get('total_s')} s")
        c2.metric("Dispatcher", f"{stg.get('supervisor_s')} s")
        c3.metric("Analyst ⇄ Inspector", f"{stg.get('worker_evaluator_s')} s")
        c4.metric("Writer", f"{stg.get('writer_s')} s")
        c5.metric("MCP calls (summed)", f"{stg.get('mcp_s')} s")
    llms = timing.get("llm") or []
    if llms:
        st.markdown("**LLM inferences**")
        st.dataframe(pd.DataFrame([{"role": r["role"], "model": r["model"], "inference (s)": r["seconds"], "tokens in": r.get("tok_in"), "tokens out": r.get("tok_out"),
                                    "tokens/s out": round(r["tok_out"] / r["seconds"], 1) if r.get("tok_out") and r["seconds"] else None, "error": r.get("error"),
                                    "prompt (start)": (r.get("prompt") or "")[:160], "response (start)": (r.get("response") or "")[:160]} for r in llms]), use_container_width=True, hide_index=True)
    elif stg:
        st.caption("No LLM was called for this question (deterministic route, deterministic evaluator, template or fixed-text answer).")
with t_steps:
    if spans.empty:
        st.info("No spans stored for this run.")
    else:
        explain("Every step in call order with its timing and the payload it carried: what the Dispatcher decided and routed, what each MCP call was asked and answered, what the evaluator checked and concluded, "
                "and the prompt / response of each LLM call.", "Pick a step to see all its attributes. `tmt.args` / `tmt.result` are the request and (truncated) response of an MCP call; `tmt.prompt` / `tmt.response` those of an LLM call.")
        core = spans[~spans["name"].str.startswith(("MCP send", "tools/call", "invocation"))].sort_values("start_ms")

        def summary(r) -> str:
            a = r["attrs"]
            n = r["name"]
            if n.startswith("mcp.tool"):
                return f"{a.get('tmt.server', '')} · args {a.get('tmt.args', '')[:70]} · {a.get('tmt.result_bytes', '?')} B" + (f" · waited {a['tmt.wait_ready_ms']} ms for the server" if a.get("tmt.wait_ready_ms", 0) > 200 else "")
            if n in ("dispatcher.plan", "supervisor.plan"):
                return f"{a.get('tmt.decision')} · {a.get('tmt.category')} · route {a.get('tmt.route', '')[:110]}"
            if n == "guardrails.input":
                return "in scope" if a.get("tmt.in_scope") else "BOUNCED"
            if n == "route.rules":
                return f"{a.get('tmt.category')} conf {a.get('tmt.confidence')} · {a.get('tmt.entities', '')[:90]}"
            if n == "history.lookup":
                return "hit " + a.get("tmt.hit_detail", "") if a.get("tmt.hit") else "miss"
            if n.startswith(("analyst.execute", "worker.execute")):
                return f"round {a.get('tmt.iteration')} · {a.get('tmt.specialist')} · planned {a.get('tmt.tools_planned', '')[:80]} · engine {a.get('tmt.ml_engine')} · {a.get('tmt.n_mcp_calls')} MCP calls"
            if n.startswith(("inspector.check", "evaluator.check")):
                return f"{a.get('tmt.verdict')} · score {a.get('tmt.score')} · {a.get('tmt.model')} · issues {a.get('tmt.issues', '[]')[:80]}"
            if n.startswith(("inspector.llm", "evaluator.llm")):
                return f"{a.get('gen_ai.request.model')} · prompt {a.get('tmt.prompt_chars')} chars · {a.get('gen_ai.usage.input_tokens')}→{a.get('gen_ai.usage.output_tokens')} tok"
            if n == "llm.write":
                return f"{a.get('gen_ai.request.model')} · {a.get('gen_ai.usage.input_tokens')}→{a.get('gen_ai.usage.output_tokens')} tok" + (f" · {a['tmt.fallback']}" if a.get("tmt.fallback") else "")
            if n == "guard.check":
                return "passed" if a.get("tmt.passed") else f"FAILED ungrounded {a.get('tmt.ungrounded')} banned {a.get('tmt.banned')}"
            if n == "writer.references":
                return a.get("tmt.sources_line", "")[:110]
            if n == "kb.sanity":
                return f"ok={a.get('tmt.sanity_ok')} fails {a.get('tmt.sanity_fails')}"
            if n == "kg.record_case":
                return f"actions {a.get('tmt.actions', '')[:80]} · Neo4j mirror {a.get('tmt.neo4j_mirror')}"
            return ""
        step_tbl = pd.DataFrame({"start (ms)": core["start_ms"].round(0), "duration (ms)": core["duration_ms"].round(0), "kind": core["kind"], "step": core["name"], "what happened": core.apply(summary, axis=1)})
        st.dataframe(step_tbl, use_container_width=True, hide_index=True, column_config={"what happened": st.column_config.TextColumn(width="large")})
        pick_step = st.selectbox("Inspect a step", [f"{int(r.start_ms):>6} ms · {r.name}" for r in core.itertuples()])
        srow = core.iloc[[f"{int(r.start_ms):>6} ms · {r.name}" for r in core.itertuples()].index(pick_step)]
        st.json({k: (json.loads(v) if isinstance(v, str) and v[:1] in "[{" and not v.endswith("chars]") else v) for k, v in srow["attrs"].items() if k.startswith(("tmt.", "gen_ai.request", "gen_ai.usage"))}, expanded=True)
with t_mcp:
    calls = timing.get("calls") or []
    if calls and "args" in calls[0]:
        explain("Every MCP call of this question — server, tool, arguments, a preview of the answer, its size and duration (which includes waiting for the server the first time).", "Calls in the same round run in parallel. A long `waited` means the MCP server was still warming up (the first question after a start).")
        st.dataframe(pd.DataFrame([{"round": c.get("round"), "server": c.get("server"), "tool": c["tool"], "seconds": c["s"], "waited for server (ms)": c.get("wait_ready_ms"), "bytes": c.get("bytes"),
                                    "ok": c.get("ok"), "arguments": json.dumps(c.get("args"), ensure_ascii=False)[:160], "result preview": (c.get("result_preview") or "")[:220]} for c in calls]), use_container_width=True, hide_index=True)
        pick_call = st.selectbox("Full arguments and result preview of", [f"round {c.get('round')} · {c['tool']} · {c['s']} s" for c in calls])
        cc = calls[[f"round {c.get('round')} · {c['tool']} · {c['s']} s" for c in calls].index(pick_call)]
        st.markdown("**Arguments**")
        st.json(cc.get("args") or {})
        st.markdown("**Result (preview, first 1 500 characters)**")
        st.code(cc.get("result_preview") or "", language="json")
    elif calls:
        st.dataframe(pd.DataFrame(calls), use_container_width=True, hide_index=True)
        st.caption("This run was recorded before per-call payloads were captured; only tool and seconds exist.")
    else:
        st.info("No MCP calls in this run (a bounce, a history answer, a follow-up explanation or a decline).")
with t_hand:
    if not hand:
        st.info("No hand-over payloads stored for this run (recorded before this feature).")
    else:
        explain("The typed messages that moved between the stages of this one question, in order: the Dispatcher's plan and route → each Analyst ⇄ evaluator round (what was tried, confidence, verdict, adjustments) → the "
                "writer's input, prompt and sources.", "This is the route the question took. Schemas: docs/agent_architecture_v3.md, docs/schemas/.")
        st.markdown(f"**1 · Dispatcher → Analyst**  ({timing.get('route_ms')} ms, decision `{timing.get('decision')}`)")
        st.json(hand.get("plan", {}), expanded=False)
        for rd in hand.get("rounds", []):
            st.markdown(f"**2.{rd['round']} · Analyst round {rd['round']} → Inspector**  (worker {rd.get('worker_s')} s, evaluator {rd.get('evaluator_s')} s · model `{rd.get('evaluator_model')}`)")
            st.json(rd, expanded=False)
        if hand.get("result"):
            st.markdown("**3 · Final worker result (confidence and why)**")
            st.json(hand["result"], expanded=False)
            st.markdown("**4 · Inspector verdict**")
            st.json({**hand.get("verdict", {}), "checks": hand.get("checks"), "similar_cases": hand.get("similar_cases")}, expanded=False)
        if hand.get("writer"):
            st.markdown(f"**5 · Writer**  ({timing.get('write_s')} s, source `{hand['writer'].get('source')}`, model `{hand['writer'].get('model')}`, guard: {hand['writer'].get('guard')})")
            w = dict(hand["writer"])
            prompt = w.pop("prompt", None)
            st.json(w, expanded=False)
            if prompt:
                st.markdown("Prompt sent to the writer LLM (question + facts, truncated):")
                st.code(prompt, language="text")
with t_plan:
    st.json(json.loads(run["plan_json"]) if run["plan_json"] else {})
with t_facts:
    st.json(json.loads(run["facts_json"]) if run["facts_json"] else {})
with t_ev:
    ev = pd.DataFrame(json.loads(run["events_json"]) if run["events_json"] else [])
    st.dataframe(ev, use_container_width=True, hide_index=True)

st.divider()
st.markdown("### 7 · Recent runs")
show = df.sort_values("ts", ascending=False).head(50)[["when", "source", "mode", "category", "total_s", "route_ms", "tools_s", "write_s", "guard", "tok_in", "tok_out", "question"]]
st.dataframe(show.round(2), use_container_width=True, hide_index=True)
