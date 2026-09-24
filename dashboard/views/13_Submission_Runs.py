import json

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import submissions as sub
from utils.ui import page_header

st.set_page_config(page_title="Submission runs", page_icon="📝", layout="wide")
page_header(
    "Submission runs — the organiser's questions answered by the agentic workflow",
    "Every run answers the workbook `team_answers_template v2.xlsx` (TRAINING, FINAL_TEST, TEAM_EVIDENCE) plus a fixed held-out stress run on the September 22 – October 1 data. "
    "Each run stores the answers, the tokens, the inference and tool time, the evaluator's verdicts and the exact system design, so runs with different components can be compared.",
)
st.caption(f"Read from `{sub.DB_PATH}` · produced by `./.venv/bin/python evaluation/submission_run.py --label <name> [--cheap] [--env KEY=VALUE] [--config JSON]` · report: `docs/submission_run_report.md`")

TIME_COLORS = {"supervisor_s": "#a855f7", "worker_s": "#3b82f6", "writer_s": "#f59e0b"}


def explain(what: str, read: str) -> None:
    with st.container(border=True):
        st.markdown(f"**What it shows.** {what}")
        st.markdown(f"**How to read it.** {read}")


if not sub.exists() or not sub.runs():
    st.warning("No submission run stored yet. Run:\n\n```\n./.venv/bin/python evaluation/submission_run.py --label baseline\n```")
    st.stop()

R = sub.runs()
by_id = {r["run_id"]: r for r in R}
tab_run, tab_cmp = st.tabs(["One run", "Compare runs"])

# ------------------------------------------------------------------------------------------------------------------ one run
with tab_run:
    pick = st.selectbox("Run", [r["run_id"] for r in R], format_func=lambda i: f"{i} · {by_id[i]['label']} · {by_id[i]['n_questions']} items · {by_id[i]['created_at']}")
    run = by_id[pick]
    m, cfg = run["metrics"], run["config"]
    A = sub.answers(pick)
    if run.get("note"):
        st.info(run["note"])

    st.markdown("### 1 · Headline metrics")
    c = st.columns(6)
    c[0].metric("Items", m["n_items"], f"{m['n_agent_answers']} by the agent")
    c[1].metric("Latency median / p95", f"{m['latency_s']['median']} / {m['latency_s']['p95']} s", f"max {m['latency_s']['max']} s", delta_color="off")
    c[2].metric("Tokens in / out", f"{m['tokens']['in']:,} / {m['tokens']['out']:,}", f"{m['tokens']['llm_calls']} LLM calls", delta_color="off")
    c[3].metric("LLM inference time", f"{m['llm_inference_s']} s", "summed over all calls", delta_color="off")
    c[4].metric("MCP calls", m["mcp_calls"], f"{m['mcp_s']} s summed", delta_color="off")
    c[5].metric("Run wall time", f"{m['wall_s']} s", f"errors: {m['n_errors']}", delta_color="off")
    c = st.columns(6)
    c[0].metric("Answered with data", m["answered_ok"])
    c[1].metric("Asked for missing input", m["asked_for_input"])
    c[2].metric("Declined (data cannot)", m["declined"])
    c[3].metric("Bounced (out of scope)", m["bounced"])
    c[4].metric("Evaluator accept / reject", f"{m['evaluator_accept']} / {m['evaluator_reject']}", f"{m['evaluator_revisions']} needed a 2nd round", delta_color="off")
    c[5].metric("Sanity check ok / failed", f"{m['sanity_ok']} / {m['sanity_failed']}", f"mean confidence {m['mean_confidence']}", delta_color="off")
    explain("The totals of this run over every question the agent answered (the TEAM_EVIDENCE rows are authored text and carry no time or tokens).",
            "Latency is the wall clock from the question to the final answer. 'Answered with data' counts questions where the worker returned facts and the evaluator accepted them; the rest are honest 'need more input', 'the data cannot answer' or 'out of scope' outcomes — they are not failures, but they are not answers either.")

    st.markdown("### 2 · System design of this run")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Pipeline**")
        st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(cfg["pipeline"], 1)))
        st.markdown("**Models per role**")
        st.dataframe(pd.DataFrame([{"role": k, "model": v or "— (falls back)"} for k, v in cfg["models"].items()]), hide_index=True, use_container_width=True)
        st.markdown("**Data**")
        st.json(cfg["data"], expanded=False)
    with d2:
        st.markdown("**Switches (components)**")
        st.dataframe(pd.DataFrame([{"switch": k, "value": str(v)} for k, v in cfg["switches"].items()]), hide_index=True, use_container_width=True)
        st.markdown("**Limits / failsafes**")
        st.dataframe(pd.DataFrame([{"limit": k, "value": str(v)} for k, v in cfg["limits"].items()]), hide_index=True, use_container_width=True)
    d3, d4 = st.columns(2)
    with d3:
        st.markdown("**ML checkpoints (TabPFN)**")
        st.dataframe(pd.DataFrame(cfg.get("ml_checkpoints", [])), hide_index=True, use_container_width=True)
    with d4:
        st.markdown("**Versions · git · overrides**")
        st.json({"versions": cfg["versions"], "git": cfg["git"], "env_overrides": cfg["env_overrides"], "cheap": cfg["cheap"]}, expanded=False)
    explain("Everything that defines this run's system: pipeline stages, the model behind each role, the component switches, the failsafe limits, the data window, the ML checkpoints, package versions and the git commit.",
            "Two runs with the same configuration should give the same decisions; a difference between runs is caused by whatever differs here. Compare them on the *Compare runs* tab.")

    st.markdown("### 3 · Time, tokens and tool use per question")
    ag = A[A["source"] == "agent"].copy()
    ag["id"] = ag["item_id"]
    if not ag.empty:
        t = ag.melt(id_vars=["id"], value_vars=["supervisor_s", "worker_s", "writer_s"], var_name="stage", value_name="seconds")
        fig = px.bar(t, x="id", y="seconds", color="stage", color_discrete_map=TIME_COLORS, title="Where the time went (supervisor · worker + evaluator incl. MCP calls · writer incl. LLM)")
        fig.update_layout(height=340, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        g1, g2 = st.columns(2)
        tk = ag.melt(id_vars=["id"], value_vars=["tok_in", "tok_out"], var_name="kind", value_name="tokens")
        f2 = px.bar(tk, x="id", y="tokens", color="kind", title="Tokens per question (all LLM calls)", color_discrete_map={"tok_in": "#3b82f6", "tok_out": "#f59e0b"})
        f2.update_layout(height=300, margin=dict(t=50, b=20))
        g1.plotly_chart(f2, use_container_width=True)
        f3 = px.scatter(ag, x="mcp_s", y="llm_s", size="latency_s", color="stage", hover_name="id", title="MCP tool time vs LLM inference time (bubble = total latency)")
        f3.update_layout(height=300, margin=dict(t=50, b=20))
        g2.plotly_chart(f3, use_container_width=True)
        explain("Per question: the stage times, the tokens, and how the time splits between the MCP tools (data + TabPFN) and the LLMs.",
                "A tall worker segment = tool/ML time (cold TabPFN scenarios are slow, cached ones fast); a tall writer segment = LLM inference. Tokens are summed over the router / evaluator / writer calls that actually happened.")
    st.dataframe(ag[["item_id", "stage", "category", "decision", "facts_status", "verdict", "confidence", "latency_s", "supervisor_s", "worker_s", "mcp_s", "writer_s", "llm_s", "llm_calls", "tok_in", "tok_out",
                     "mcp_calls", "rounds", "sanity_ok", "guard", "answer_words"]], hide_index=True, use_container_width=True)

    st.markdown("### 4 · Questions and answers")
    stages = [s for s in ["TRAINING", "FINAL_TEST", "HELDOUT", "TEAM_EVIDENCE"] if s in set(A["stage"])]
    tabs = st.tabs([f"{s} ({int((A['stage'] == s).sum())})" for s in stages])
    for tab, stage in zip(tabs, stages):
        with tab:
            if stage == "HELDOUT":
                st.caption("Fixed operator questions on the held-out period (Sept 22 – Oct 1), used as the STRESS evidence; the last rows show failure handling (beyond the data, injected instruction).")
            for r in A[A["stage"] == stage].itertuples():
                head = f"**{r.item_id}** · {r.question[:120]}{'…' if len(r.question) > 120 else ''}"
                tag = "authored + measured" if r.source != "agent" else f"{r.category or '-'} · {r.decision} · {r.facts_status} · conf {r.confidence} · {r.latency_s} s · {r.tok_in}+{r.tok_out} tok"
                with st.expander(f"{head}  —  {tag}"):
                    st.markdown("**Question**")
                    st.write(r.question)
                    st.markdown("**Answer**")
                    st.markdown(r.answer)
                    if r.source == "agent":
                        k = st.columns(6)
                        k[0].metric("Latency", f"{r.latency_s} s")
                        k[1].metric("LLM inference", f"{r.llm_s} s", f"{r.llm_calls} calls", delta_color="off")
                        k[2].metric("Tokens in/out", f"{r.tok_in}/{r.tok_out}")
                        k[3].metric("MCP", f"{r.mcp_s} s", f"{r.mcp_calls} calls", delta_color="off")
                        k[4].metric("Evaluator", f"{r.verdict}", f"{r.rounds} round(s)", delta_color="off")
                        k[5].metric("Sanity / guard", {1: "ok", 0: "FAILED", None: "n/a"}.get(r.sanity_ok, "n/a"), str(r.guard)[:30], delta_color="off")
                        s1, s2, s3 = st.tabs(["LLM calls", "MCP calls", "Route, checks, facts"])
                        with s1:
                            if r.llm_json:
                                st.dataframe(pd.DataFrame(r.llm_json), hide_index=True, use_container_width=True)
                            else:
                                st.caption("No LLM was called (deterministic route, template or fixed text).")
                        with s2:
                            if r.calls_json:
                                st.dataframe(pd.DataFrame([{**c, "args": json.dumps(c.get("args"), ensure_ascii=False)[:120]} for c in r.calls_json]), hide_index=True, use_container_width=True)
                            else:
                                st.caption("No MCP call.")
                        with s3:
                            st.json({"plan": r.plan_json, "evaluator_checks": r.checks_json, "facts": r.facts_json}, expanded=False)
                        if r.error:
                            st.error(r.error)

# ------------------------------------------------------------------------------------------------------------------ compare
with tab_cmp:
    if len(R) < 2:
        st.info("Only one run stored so far. Run `evaluation/submission_run.py` again with another `--label` and configuration (`--cheap`, `--env EVALUATOR_MODE=off`, `--config '{\"engine\":\"empirical\"}'`) to compare.")
    ids = [r["run_id"] for r in R]
    fmt = lambda i: f"{i} · {by_id[i]['label']}"
    st.markdown("### Runs overview")
    ov = pd.DataFrame([{"run": r["run_id"], "label": r["label"], "created": r["created_at"], "items": r["n_questions"], "median s": r["metrics"]["latency_s"]["median"], "p95 s": r["metrics"]["latency_s"]["p95"],
                        "tokens in": r["metrics"]["tokens"]["in"], "tokens out": r["metrics"]["tokens"]["out"], "LLM calls": r["metrics"]["tokens"]["llm_calls"], "MCP calls": r["metrics"]["mcp_calls"],
                        "answered": r["metrics"]["answered_ok"], "need input": r["metrics"]["asked_for_input"], "declined": r["metrics"]["declined"], "bounced": r["metrics"]["bounced"],
                        "writer model": r["config"]["models"].get("writer"), "evaluator": r["config"]["switches"].get("evaluator_mode")} for r in R])
    st.dataframe(ov, hide_index=True, use_container_width=True)
    if len(R) >= 2:
        a_id, b_id = st.columns(2)
        A_id = a_id.selectbox("Run A", ids, index=1 if len(ids) > 1 else 0, format_func=fmt)
        B_id = b_id.selectbox("Run B", ids, index=0, format_func=fmt)
        fa, fb = sub.flatten(by_id[A_id]["config"]), sub.flatten(by_id[B_id]["config"])
        keys = sorted(set(fa) | set(fb))
        diff = [{"setting": k, "A": fa.get(k), "B": fb.get(k)} for k in keys if fa.get(k) != fb.get(k) and not k.startswith(("git.", "ml_checkpoints"))]
        st.markdown("### What differs between the two system designs")
        if diff:
            st.dataframe(pd.DataFrame(diff), hide_index=True, use_container_width=True)
        else:
            st.success("Identical configuration (apart from git commit).")
        aa, bb = sub.answers(A_id), sub.answers(B_id)
        j = aa[aa["source"] == "agent"].merge(bb[bb["source"] == "agent"], on="item_id", suffixes=("_A", "_B"))
        if not j.empty:
            st.markdown("### Per question")
            cmp = pd.DataFrame({"item": j["item_id"], "decision A": j["decision_A"], "decision B": j["decision_B"], "latency A": j["latency_s_A"], "latency B": j["latency_s_B"],
                                "tokens A": j["tok_in_A"] + j["tok_out_A"], "tokens B": j["tok_in_B"] + j["tok_out_B"], "confidence A": j["confidence_A"], "confidence B": j["confidence_B"],
                                "same answer": j["answer_A"] == j["answer_B"]})
            st.dataframe(cmp, hide_index=True, use_container_width=True)
            lt = pd.concat([pd.DataFrame({"item": j["item_id"], "run": "A", "seconds": j["latency_s_A"]}), pd.DataFrame({"item": j["item_id"], "run": "B", "seconds": j["latency_s_B"]})])
            fig = px.bar(lt, x="item", y="seconds", color="run", barmode="group", title="Latency per question")
            fig.update_layout(height=320, margin=dict(t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            q = st.selectbox("Read the two answers side by side", list(j["item_id"]))
            row = j[j["item_id"] == q].iloc[0]
            l, r_ = st.columns(2)
            l.markdown(f"**A · {by_id[A_id]['label']}**")
            l.markdown(row["answer_A"])
            r_.markdown(f"**B · {by_id[B_id]['label']}**")
            r_.markdown(row["answer_B"])
