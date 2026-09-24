"""LangSmith-style evaluation of the agent's answers: OFFLINE by default, LLM judges from `openevals`, run metrics like a trace UI.

    ./.venv/bin/python evaluation/langsmith_eval.py                      # the latest stored run of every suite (no agent re-run)
    ./.venv/bin/python evaluation/langsmith_eval.py ev-20260924-095110   # a specific stored evaluation run
    ./.venv/bin/python evaluation/langsmith_eval.py --upload             # ALSO send the runs + feedback to LangSmith (needs LANGSMITH_API_KEY)

What it does — the same machinery LangSmith uses, run locally:
  * `langsmith.evaluate(...)` with `upload_results=False`: every stored answer is an "example"; nothing leaves this machine unless --upload.
  * LLM-as-judge evaluators from `openevals` (LangSmith's prebuilt evaluator prompts), judged by the SMALL model:
        correctness   answer vs the reference (ground truth from the knowledge base)         [items that have a reference]
        groundedness  every claim supported by the FACTS the specialist produced (RAG groundedness)
        helpfulness   does the answer help the operator with the question (RAG helpfulness)
        relevance     is the answer on topic
        conciseness   short and to the point (an operator under pressure)
  * a deterministic evaluator `sanity` from the knowledge-base sanity check (stored with each run).
  * run metrics, as a tracing UI shows them: latency p50/p95/max, tokens, estimated cost, tool calls, error rate — plus per-category means.
Results go to the observability DB (ls_runs / ls_feedback) and the dashboard's Agent Workflow page. Judge = gpt-4o-mini (OPENAI_API_KEY); the shared main
model is never used. Judges are fallible (self-preference, leniency): read their comments and compare with the project judge (evaluation/judge.py).
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics as st
import sys
import time
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "agent"), str(REPO)]

PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6          # gpt-4o-mini list price per token (estimate; the shared main model differs)
JUDGE_MODEL = os.environ.get("LS_JUDGE_MODEL", "gpt-4o-mini")


def build_evaluators():
    """openevals LLM judges wrapped as LangSmith evaluators (inputs, outputs, reference_outputs) -> feedback."""
    from openai import OpenAI
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import (ANSWER_RELEVANCE_PROMPT, CONCISENESS_PROMPT, CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT, RAG_HELPFULNESS_PROMPT)

    client = OpenAI()
    mk = lambda prompt, key: create_llm_as_judge(prompt=prompt, judge=client, model=JUDGE_MODEL, feedback_key=key, continuous=True)
    correctness, grounded = mk(CORRECTNESS_PROMPT, "correctness"), mk(RAG_GROUNDEDNESS_PROMPT, "groundedness")
    helpful, relevant, concise = mk(RAG_HELPFULNESS_PROMPT, "helpfulness"), mk(ANSWER_RELEVANCE_PROMPT, "relevance"), mk(CONCISENESS_PROMPT, "conciseness")

    def ev_correctness(inputs, outputs, reference_outputs):
        if not (reference_outputs or {}).get("answer"):
            return {"key": "correctness", "score": None, "comment": "no reference for this item"}
        return correctness(inputs=inputs["question"], outputs=outputs["answer"], reference_outputs=reference_outputs["answer"])

    def ev_groundedness(inputs, outputs):
        return grounded(context=json.dumps(outputs["facts"], ensure_ascii=False)[:6000], outputs=outputs["answer"])

    def ev_helpfulness(inputs, outputs):
        return helpful(inputs=inputs["question"], outputs=outputs["answer"])

    def ev_relevance(inputs, outputs):
        return relevant(inputs=inputs["question"], outputs=outputs["answer"])

    def ev_conciseness(inputs, outputs):
        return concise(inputs=inputs["question"], outputs=outputs["answer"])

    def ev_sanity(inputs, outputs):
        s = outputs.get("sanity") or {}
        if s.get("ok") is None:
            return {"key": "sanity", "score": None, "comment": "no sanity result stored"}
        return {"key": "sanity", "score": bool(s["ok"]), "comment": f"knowledge-base checks {s.get('n')}, failed: {s.get('fails') or 'none'}"}

    return [ev_correctness, ev_groundedness, ev_helpfulness, ev_relevance, ev_conciseness, ev_sanity]


def load_examples(conn: sqlite3.Connection, eval_id: str):
    """Stored answers of one evaluation run -> LangSmith Examples (+ the run metrics kept aside)."""
    from langsmith.schemas import Example

    import dataset
    import metrics

    items = {i.id: i for i in dataset.load_workbook_items() + dataset.load_limit_items() + dataset.load_challenge_items() + dataset.load_bank_items()}
    ds_id = uuid.uuid4()
    examples, side = [], {}
    conn.row_factory = lambda cur, row: {d[0]: row[i] for i, d in enumerate(cur.description)}
    for r in conn.execute("SELECT * FROM eval_items WHERE eval_id=? ORDER BY item_id, repeat_idx", (eval_id,)).fetchall():
        run = conn.execute("SELECT * FROM runs WHERE run_id=?", (r["run_id"],)).fetchone() or {}
        it = items.get(r["item_id"])
        ref = ""
        if it is not None:
            try:
                ref = metrics.expectations(it, None).get("truth_text") or ""
            except Exception:
                ref = ""
            if not ref:                                              # T03 / T04: the reference comes from the knowledge base (built from the raw CSVs)
                import knowledge
                hit = knowledge.kb().search({"T03": "Hallesches Tor Kaiserin-Augusta-Str. closure", "T04": "Rudow weekday commute peak"}.get(it.id, ""), kinds=["ground_truth"], k=1) if it.id in ("T03", "T04") else []
                ref = hit[0]["text"] if hit else ""
        ex = Example(id=uuid.uuid4(), dataset_id=ds_id, inputs={"question": r["question"], "_key": f"{r['item_id']}#{r['repeat_idx']}"},
                     outputs={"answer": ref} if ref else {})
        examples.append(ex)
        timing = json.loads(run.get("timing_json") or "{}")
        side[ex.inputs["_key"]] = {"answer": r["answer"] or "", "facts": json.loads(run.get("facts_json") or "{}"), "sanity": timing.get("sanity"), "cat": r["category"],
                                   "latency_s": r["latency_s"] or run.get("total_s"), "tok_in": run.get("tok_in") or 0, "tok_out": run.get("tok_out") or 0,
                                   "n_tool_calls": run.get("n_tool_calls") or 0, "n_llm_calls": run.get("n_llm_calls") or 0, "status": run.get("status"), "item_id": r["item_id"],
                                   "run_id": r["run_id"], "trace_id": r.get("trace_id"), "question": r["question"]}
    return examples, side


def run_metrics(side: dict) -> dict:
    lat = sorted(v["latency_s"] for v in side.values() if v["latency_s"] is not None)
    pct = lambda q: lat[min(len(lat) - 1, int(round(q * (len(lat) - 1))))] if lat else None
    tin, tout = sum(v["tok_in"] for v in side.values()), sum(v["tok_out"] for v in side.values())
    errors = sum(1 for v in side.values() if v["status"] not in ("ok", "success", None) or v["facts"].get("status") == "error")
    per_cat: dict = {}
    for v in side.values():
        per_cat.setdefault(v["cat"] or "?", []).append(v["latency_s"] or 0)
    return {"n": len(side), "latency_p50_s": pct(0.5), "latency_p95_s": pct(0.95), "latency_max_s": lat[-1] if lat else None,
            "latency_mean_s": st.mean(lat) if lat else None, "tokens_in": tin, "tokens_out": tout, "est_cost_usd": round(tin * PRICE_IN + tout * PRICE_OUT, 5),
            "tool_calls": sum(v["n_tool_calls"] for v in side.values()), "llm_calls": sum(v["n_llm_calls"] for v in side.values()),
            "error_rate": errors / max(len(side), 1), "latency_by_category_s": {c: round(st.mean(x), 2) for c, x in per_cat.items()}}


def evaluate_run(conn: sqlite3.Connection, eval_id: str, upload: bool = False) -> dict:
    from langsmith import evaluate

    os.environ["LANGSMITH_TRACING"] = "true" if upload else "false"
    examples, side = load_examples(conn, eval_id)
    suite = conn.execute("SELECT suite FROM eval_runs WHERE eval_id=?", (eval_id,)).fetchone()["suite"]

    def target(inputs: dict) -> dict:                       # the "application": returns what the agent already answered
        v = side[inputs["_key"]]
        return {"answer": v["answer"], "facts": v["facts"], "sanity": v["sanity"]}

    res = evaluate(target, data=examples, evaluators=build_evaluators(), upload_results=upload, max_concurrency=3,
                   experiment_prefix=f"nextmove-{suite}", metadata={"eval_id": eval_id, "judge": JUDGE_MODEL})
    ls_id = time.strftime("ls-%Y%m%d-%H%M%S")
    conn.row_factory = None
    rows, scores = [], {}
    for r in res:
        key = r["example"].inputs["_key"]
        for fr in r["evaluation_results"]["results"]:
            val = None if fr.score is None else float(fr.score)
            rows.append((ls_id, key, fr.key, val, (fr.comment or "")[:600]))
            if val is not None:
                scores.setdefault(fr.key, []).append(val)
    means = {k: round(st.mean(v), 3) for k, v in scores.items()}
    rm = run_metrics(side)
    summary = {"eval_id": eval_id, "scores": means, "n_scored": {k: len(v) for k, v in scores.items()}, "run_metrics": rm, "judge": JUDGE_MODEL}
    conn.execute("INSERT OR REPLACE INTO ls_runs (ls_id, ts, eval_id, suite, judge_model, n_items, uploaded, summary_json) VALUES (?,?,?,?,?,?,?,?)",
                 (ls_id, time.time(), eval_id, suite, JUDGE_MODEL, len(examples), int(upload), json.dumps(summary)))
    conn.executemany("INSERT OR REPLACE INTO ls_feedback (ls_id, item_id, key, score, comment) VALUES (?,?,?,?,?)", rows)
    conn.commit()
    return {"ls_id": ls_id, **summary}


def main() -> None:
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    import observability as obs

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    upload = "--upload" in sys.argv
    if upload and not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit("--upload needs LANGSMITH_API_KEY (and sends questions, answers and facts to LangSmith). Not set: nothing was uploaded.")
    conn = obs.connect()
    conn.row_factory = sqlite3.Row
    ids = args or [r[0] for r in conn.execute("SELECT eval_id FROM eval_runs e WHERE ts=(SELECT MAX(ts) FROM eval_runs WHERE suite=e.suite) AND suite IN ('challenge','training','limit') ORDER BY ts")]
    conn.row_factory = None
    for eid in ids:
        conn.row_factory = sqlite3.Row
        out = evaluate_run(conn, eid, upload)
        rm = out["run_metrics"]
        print(f"\n== {out['ls_id']}  ({eid}, {rm['n']} answers, judge {out['judge']}, uploaded={upload})")
        print("   feedback (mean, 0-1): " + " · ".join(f"{k} {v:.2f}" for k, v in out["scores"].items()))
        print(f"   run metrics: latency p50 {rm['latency_p50_s']:.1f}s p95 {rm['latency_p95_s']:.1f}s max {rm['latency_max_s']:.1f}s · tokens {rm['tokens_in']}+{rm['tokens_out']} "
              f"(≈${rm['est_cost_usd']}) · tool calls {rm['tool_calls']} · errors {rm['error_rate']:.0%}")


if __name__ == "__main__":
    main()
