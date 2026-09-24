import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.obs_data import BUDGET_S, DB_PATH, db_exists, key, load_eval_items, load_eval_runs
from utils.ui import page_header

st.set_page_config(page_title="Evaluation", page_icon="✅", layout="wide")
page_header(
    "Evaluation — how good is the workflow on the hackathon questions?",
    "The organiser's answer workbook is the evaluation dataset: each question is run through the real agent and scored on the "
    "hackathon's own criteria. Every scored answer links back to its stored trace.",
)
st.caption(f"Read from `{DB_PATH}` (tables `eval_runs`, `eval_items`). Produced by `./.venv/bin/python scripts/tasks.py eval`.")

PRED, BASE, REF, GOOD, BAD = "#3b82f6", "#a855f7", "#9ca3af", "#22c55e", "#ef4444"
CRITERIA = {"relevance": 0.30, "reliability": 0.30, "stress": 0.20}
GROUP_OF = {"answered": "relevance", "completeness": "relevance", "route_correct": "relevance",
            "hallucination_free": "reliability", "fact_accuracy": "reliability", "honest_scope": "reliability",
            "decline_quality": "reliability", "assumption_disclosed": "reliability", "traceable": "reliability",
            "consistency": "reliability", "latency_ok": "stress", "readability_ok": "stress",
            "judge_relevance": "relevance", "judge_faithfulness": "reliability", "judge_grounded": "reliability", "judge_clarity": "stress"}
DEFS = {
    "answered": "The workflow produced a substantive answer (an honest decline is safe but gives the operator nothing).",
    "completeness": "Share of the required elements present in answered questions (e.g. reason, times, reroute, pressured stations, staff action, caveat).",
    "route_correct": "The router picked the expected question category, i.e. the right tools and datasets.",
    "hallucination_free": "Every number in the delivered answer traces to tool output or to the question; no capacity or 'bus service exists' claims.",
    "fact_accuracy": "Key facts equal ground truth recomputed from the RAW csv files, independently of the agent's tools.",
    "honest_scope": "Answered what the data supports and declined (instead of guessing) what it does not.",
    "decline_quality": "A decline names the limit and offers what IS possible (and reports the recorded finding for reroute behaviour).",
    "assumption_disclosed": "Disruption estimates state that they are assumption-based scenarios, not measurements.",
    "traceable": "A stored run and trace exist with the tool spans that produced the answer (explainability).",
    "consistency": "Repeated runs of the same question give the same category and the same key numbers.",
    "latency_ok": f"Answered within the latency budget ({BUDGET_S:.0f} s).",
    "readability_ok": "Short (≤150 words), plain, no JSON or internal keys leaking into the operator text.",
    "judge_relevance": "LLM judge (1–5, scaled to 0–1): does the answer address what was asked, or correctly decline what the system cannot answer?",
    "judge_faithfulness": "LLM judge: is every claim supported by the tool evidence / ground truth? Invented facts, 'measured' for a model estimate, capacity claims score low.",
    "judge_grounded": "LLM judge: 1.0 if it found NO unsupported claim in the answer, else 0.",
    "judge_clarity": "LLM judge: short, plain, actionable for an operator under pressure.",
}
DETERMINISTIC = ["answered", "route_correct", "fact_accuracy", "hallucination_free", "traceable", "consistency", "latency_ok", "readability_ok", "honest_scope"]


def explain(what: str, read: str, verdict: tuple[str, str] | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**What it shows.** {what}")
        st.markdown(f"**How to read it.** {read}")
        if verdict:
            getattr(st, verdict[0])(verdict[1])


ev = load_eval_runs(key()) if db_exists() else pd.DataFrame()
if ev.empty:
    st.warning("No evaluation runs yet. Run one with:\n\n```\n./.venv/bin/python scripts/tasks.py eval                                   # ONE brutal multi-part question, LLM-judged\n"
               "./.venv/bin/python scripts/tasks.py eval \"--suite training --allow-many\"   # the 11 workbook questions\n"
               "./.venv/bin/python scripts/tasks.py eval \"--suite training --allow-many --repeat 3 --export-xlsx\"   # + consistency, fill the workbook\n```")
    st.stop()

with st.expander("📖 The dataset and the criteria", expanded=False):
    st.markdown(
        """
**Dataset.** `evaluation/team_answers_template v1.xlsx` (sheet `TEAM_ANSWERS`): **TRAINING** = the 11 known questions (9 core + 2 bonus),
**FINAL_TEST** = 5 slots that are filled on the final day (loaded automatically once they contain questions), **TEAM_EVIDENCE** = team-level
prompts (not run through the agent). The **stress** suite adds Edge / Trap / typo / German / cross-cutting questions from `docs/test_questions.md`.

**How answers are scored — an LLM judge plus deterministic gates.** The *semantic* criteria (does it state the reason, the reroute, the staff action, the
caveat; does a decline name a concrete alternative; is anything unsupported; is it clear) are judged by an **LLM judge** (small model by default) that reads the
question, the criteria, the **ground truth recomputed from the raw csv files**, the tool evidence and the answer, and quotes its justification per criterion.
What a machine verifies better stays **deterministic**: facts equal to ground truth, every number traceable to tool output, latency, routing, stored trace, word
count. The old keyword rules are kept only as a **calibration signal** (judge-vs-regex agreement) and as the fallback when the judge is unreachable.

**Criteria (from the workbook's INSTRUCTIONS sheet).** Relevance 0.30 · Reliability 0.30 · Stress testing 0.20 are scored automatically below;
Innovation 0.10 and Impact 0.10 are team-level evidence (this observability and evaluation pipeline is part of that evidence).
Scores are in [0, 1]; the overall score is the weighted mean of the three automatic criteria.
"""
    )
    st.dataframe(pd.DataFrame([{"Metric": k, "Criterion": GROUP_OF[k], "Weight of criterion": CRITERIA[GROUP_OF[k]],
                                "Decided by": "deterministic check" if k in DETERMINISTIC else "LLM judge", "What it measures": v}
                               for k, v in DEFS.items()]), use_container_width=True, hide_index=True)

ev = ev.sort_values("ts", ascending=False)
labels = {r.eval_id: f"{r.when:%m-%d %H:%M} · {r.suite} · {int(r.n_items)} runs · {r.git_commit or ''}" for r in ev.itertuples()}
eid = st.selectbox("Evaluation run", list(labels), format_func=lambda i: labels[i])
cur = ev[ev["eval_id"] == eid].iloc[0]
S = cur["summary"]
same_suite = ev[(ev["suite"] == cur["suite"]) & (ev["ts"] < cur["ts"])]
prev = same_suite.iloc[0]["summary"] if len(same_suite) else None
items = load_eval_items(eid)
M = S["metrics"]

# ------------------------------------------------------------------------------------------ scoreboard
st.markdown("### Scoreboard")
c = st.columns(5)
for col, (name, w) in zip(c[:3], CRITERIA.items()):
    v = S.get(name)
    d = None if (prev is None or prev.get(name) is None or v is None) else f"{v - prev[name]:+.2f} vs previous"
    col.metric(f"{name.capitalize()} (weight {w})", "n/a" if v is None else f"{v:.2f}", d)
d = None if prev is None or prev.get("overall") is None else f"{S['overall'] - prev['overall']:+.2f} vs previous"
c[3].metric("Overall (weighted)", "n/a" if S.get("overall") is None else f"{S['overall']:.2f}", d)
cov = M.get("answered")
c[4].metric("Answer coverage", "n/a" if cov is None else f"{cov:.0%}", help="Share of questions the workflow actually answered (not declined).")
lat = S["latency"]
st.caption(f"Latency: mean **{lat['mean']:.1f} s** · p50 {lat['p50']:.1f} s · p95 {lat['p95']:.1f} s · max {lat['max']:.1f} s · "
           f"within {lat['budget_s']:.0f} s budget: **{lat['within_budget']:.0%}** · mode `{cur['mode']}` · commit `{cur['git_commit']}`"
           + (f" · note: {S['label']}" if S.get("label") else ""))
sc = S.get("scoring") or {}
if sc.get("judge_model"):
    agree = M.get("judge_agreement")
    st.info(f"**Scoring: {sc['method']}** · judge model `{sc['judge_model']}` (covered {sc['judge_coverage']:.0%} of runs)"
            + (f" · judge-vs-old-regex agreement **{agree:.0%}**" if agree is not None else "")
            + (f" · re-scored {sc['rescored_at']}" if sc.get("rescored_at") else "")
            + ". The judge is itself fallible (leniency, self-preference): read its quoted justifications in section 4 and treat disagreements with the deterministic gates as signals.")
else:
    st.warning("**Scoring: regex rubric only** — this run predates the LLM judge or the judge was disabled. Re-score it with `python evaluation/rejudge.py <eval_id>`.")
if cov is not None and cov < 0.5:
    st.warning(f"**Read the overall score with care:** it rewards honest, grounded answers, but the workflow only *answers* "
               f"{cov:.0%} of these questions — the rest are correct declines because no tools exist for those categories yet "
               "(A, B, E, F, G, H, X). Coverage is the main gap to close, not the quality of the answers it does give.")
elif S.get("overall") is not None:
    st.success(f"Overall {S['overall']:.2f} with {cov:.0%} of questions answered.")

# ------------------------------------------------------------------------------------------ metrics table
st.markdown("### 1 · Every metric, against the previous run")
rows = []
for k, g in GROUP_OF.items():
    if k in M:
        p = None if prev is None else prev["metrics"].get(k)
        rows.append({"Criterion": g, "Metric": k, "Score": round(M[k], 3), "Previous": None if p is None else round(p, 3),
                     "Δ": None if p is None else round(M[k] - p, 3), "What it measures": DEFS[k]})
tbl = pd.DataFrame(rows)
fig = px.bar(tbl.sort_values("Score"), x="Score", y="Metric", color="Criterion", orientation="h", range_x=[0, 1.02], text="Score",
             color_discrete_map={"relevance": PRED, "reliability": BASE, "stress": "#f59e0b"})
fig.update_layout(height=max(320, 34 * len(tbl) + 80))
st.plotly_chart(fig, use_container_width=True)
weak = tbl.sort_values("Score").iloc[0]
explain(
    "The average of each metric over all runs in this evaluation, coloured by the hackathon criterion it feeds. 1.0 = perfect.",
    "Short bars are the weak spots. A metric only counts the questions it applies to (e.g. fact accuracy exists only for the two questions "
    "with checkable ground truth; consistency only when `--repeat` > 1).",
    ("info", f"Weakest metric: **{weak['Metric']}** at {weak['Score']:.2f} — {weak['What it measures']}"),
)
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------------------ per question
if not items.empty:
    st.markdown("### 2 · Question × metric heatmap")
    mets = [k for k in GROUP_OF if any(k in m for m in items["metrics"])]
    mat = pd.DataFrame({
        iid: {k: np.nanmean([m.get(k) if m.get(k) is not None else np.nan for m in d["metrics"]]) for k in mets}
        for iid, d in items.groupby("item_id")}).T
    qtxt = items.drop_duplicates("item_id").set_index("item_id")["question"].str[:70]
    mat.index = [f"{i} · {qtxt[i]}" for i in mat.index]
    fig = px.imshow(mat, color_continuous_scale=["#ef4444", "#fde68a", "#22c55e"], zmin=0, zmax=1, aspect="auto", text_auto=".2f")
    fig.update_layout(height=max(320, 30 * len(mat) + 120), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    explain(
        "One row per question, one column per metric; green = 1.0, red = 0. Blank cells mean the metric does not apply to that question.",
        "Look for horizontal red stripes (a question that fails everywhere — usually a category without tools) and vertical red stripes "
        "(a metric that fails across questions — a systematic issue).",
    )

    st.markdown("### 3 · Latency per question against the budget")
    lat_df = items.assign(q=items["item_id"] + " · " + items["question"].str[:50])
    lat_df["budget"] = np.where(lat_df["latency_s"] > BUDGET_S, "over budget", "in budget")
    fig = px.bar(lat_df, x="q", y="latency_s", color="budget", color_discrete_map={"in budget": PRED, "over budget": BAD},
                 hover_data=["repeat_idx"], labels={"latency_s": "Seconds", "q": ""})
    fig.add_hline(y=BUDGET_S, line_dash="dash", line_color=REF, annotation_text=f"budget {BUDGET_S:.0f}s")
    fig.update_layout(height=360, xaxis_tickangle=-40, legend_title="")
    st.plotly_chart(fig, use_container_width=True)
    explain(
        "How long each evaluated answer took (repeats shown as separate bars).",
        "Supported questions (disruption, station profile) run tools and take longer; declines are near-instant. Compare with the dashed budget.",
        ("success" if lat["within_budget"] >= 0.95 else "warning",
         f"{lat['within_budget']:.0%} of runs within the {BUDGET_S:.0f} s budget; the slowest took {lat['max']:.1f} s."),
    )

    st.markdown("### 4 · Failed checks (with the answer that caused them)")
    shown = 0
    for iid, d in items.groupby("item_id"):
        fails: dict[str, int] = {}
        for chk in d["checks"]:
            for c_ in chk:
                if not c_["ok"]:
                    fails[c_["name"]] = fails.get(c_["name"], 0) + 1
        if not fails:
            continue
        shown += 1
        first = d.iloc[0]
        with st.expander(f"❌ {iid} — {first['question'][:95]}  ({len(fails)} failing check{'s' if len(fails) > 1 else ''})"):
            st.markdown("**Failing checks** (× = number of repeats affected): " + "; ".join(f"`{k}` ×{v}" for k, v in fails.items()))
            st.markdown("**Answer:**")
            st.markdown(first["answer"])
            jd = first.get("judge") if hasattr(first, "get") else None
            if isinstance(jd, dict):
                st.caption(f"LLM judge ({jd.get('model')}): relevance {jd.get('relevance')}/5 · faithfulness {jd.get('faithfulness')}/5 · clarity {jd.get('clarity')}/5 · "
                           f"usefulness {jd.get('usefulness')}/5 — {jd.get('comment', '')}")
            st.dataframe(pd.DataFrame(first["checks"]), use_container_width=True, hide_index=True)
            st.caption(f"run `{first['run_id']}` · trace `{first['trace_id']}` — pick it in the Observability page's trace explorer.")
    if not shown:
        st.success("No failing checks in this run.")

    if "judge_relevance" in M:
        st.markdown("### 5 · What the LLM judge said, per question")
        rows_j = []
        for iid, d in items.groupby("item_id"):
            js = [j for j in d["judge"] if isinstance(j, dict)] if "judge" in d else []
            if js:
                rows_j.append({"Question": f"{iid} · {d.iloc[0]['question'][:70]}", "Relevance": np.mean([j["relevance"] for j in js if j.get("relevance")]),
                               "Faithfulness": np.mean([j["faithfulness"] for j in js if j.get("faithfulness")]), "Clarity": np.mean([j["clarity"] for j in js if j.get("clarity")]),
                               "Unsupported claims": sum(len(j.get("unsupported_claims") or []) for j in js), "Judge's main criticism": js[0].get("comment", "")})
        if rows_j:
            st.dataframe(pd.DataFrame(rows_j).round(2), use_container_width=True, hide_index=True)
        explain("For every question: the judge's 1–5 scores (averaged over repeats), how many unsupported claims it found, and its one-line main criticism.",
                "Low faithfulness or unsupported claims mean the answer said something the tool evidence does not back. Compare with the deterministic gates in the heatmap: "
                "if the judge and the gates disagree (e.g. numbers grounded but faithfulness low), the wording overclaims even though the numbers are right.")

# ------------------------------------------------------------------------------------------ trend
st.divider()
st.markdown("### Trend across evaluation runs")
trend = ev[ev["suite"] == cur["suite"]].sort_values("ts")
if len(trend) < 2:
    st.info("Run the evaluation again after a change to see the trend.")
else:
    tr = pd.DataFrame({"when": trend["when"], **{k: [s.get(k) for s in trend["summary"]] for k in ("overall", "relevance", "reliability", "stress")},
                       "p95": [s["latency"]["p95"] for s in trend["summary"]]})
    fig = go.Figure()
    for k, color in (("overall", "#111827"), ("relevance", PRED), ("reliability", BASE), ("stress", "#f59e0b")):
        fig.add_trace(go.Scatter(x=tr["when"], y=tr[k], mode="lines+markers", name=k, line=dict(color=color, width=3 if k == "overall" else 2)))
    fig.add_trace(go.Bar(x=tr["when"], y=tr["p95"], name="p95 latency (s)", yaxis="y2", opacity=0.25, marker_color=REF))
    fig.update_layout(height=380, yaxis=dict(range=[0, 1.05], title="Score"), yaxis2=dict(title="p95 latency (s)", overlaying="y", side="right"))
    st.plotly_chart(fig, use_container_width=True)
    explain("Scores (lines) and p95 latency (bars) of every evaluation run of this suite.",
            "Use it to prove an improvement (or catch a regression) after each change to the router, executor, writer or tools.")
