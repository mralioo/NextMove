"""Run the agent workflow over the evaluation dataset and score it.

    ./.venv/bin/python scripts/tasks.py eval                      # ONE brutal multi-part question (default): 1 call to the shared LLM endpoint
    ./.venv/bin/python scripts/tasks.py eval --suite training --allow-many     # the 11 workbook questions (~11 LLM calls)
    ./.venv/bin/python evaluation/run_eval.py --suite limit --cheap      # same, but with the small worker model

The LLM endpoint (SUPERVISOR/WRITER model) is shared, so the harness refuses to run more than MAX_CALLS (5)
questions x repeats unless you pass --allow-many; --cheap uses the small worker model for the answers. Answers are scored by an
LLM JUDGE (small model by default; --judge-model main uses the shared model, doubling the main-model calls; --no-judge = regex fallback).

Suites   limit     ONE complex message with six sub-asks (default) — see dataset.LIMIT_QUESTION
         training  the 11 TRAINING questions of `evaluation/team_answers_template v2.xlsx`
         challenge the 3 example questions of the problem statement (verbatim) + 3 date-grounded variants, scored on ideal-answer criteria
         final    the FINAL_TEST rows once they are filled in (Sept 25)
         stress   Edge + Trap + cross-cutting robustness questions from docs/test_questions.md
         bank     every non-follow-up question of docs/test_questions.md
         all      training + final + stress
Each question runs in a fresh session through the real ADK app, so it also produces a stored trace
(observability/agent_obs.db). Scores go to the `eval_runs` / `eval_items` tables and to
evaluation/output/eval_<id>.json; the dashboard's Evaluation page reads them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "agent"), str(REPO)]


MAX_CALLS = 5


def enforce_budget(n_questions: int, repeat: int, judge_main: bool, allow_many: bool) -> int:
    """Every question makes one call to the shared LLM endpoint (two if the judge is ALSO the main model; the default
    judge is the small model and does not count). Refuse big runs unless asked."""
    calls = n_questions * repeat * (2 if judge_main else 1)
    if calls > MAX_CALLS and not allow_many:
        raise SystemExit(f"Refusing to run: this would make ~{calls} calls to the shared LLM endpoint (limit {MAX_CALLS}). "
                         "Use the default single-question suite, --cheap, or pass --allow-many if you really mean it.")
    return calls


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="limit", choices=["limit", "training", "final", "stress", "bank", "all", "challenge"])
    ap.add_argument("--repeat", type=int, default=1, help="runs per question; >1 also scores consistency")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge and score with the regex rubric only (legacy / offline)")
    ap.add_argument("--judge-model", default="small", choices=["small", "main"], help="which model judges the answers (default: the small worker model)")
    ap.add_argument("--export-xlsx", action="store_true", help="write the answers into a COPY of the workbook")
    ap.add_argument("--team", default="", help="TEAM_NAME for the exported workbook")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (debugging)")
    ap.add_argument("--ids", default="", help="comma-separated question ids to run (e.g. CH1w,T04)")
    ap.add_argument("--no-warm", action="store_true", help="don't wait for the MCP server warm-up first")
    ap.add_argument("--label", default="", help="free-text note stored with the run")
    ap.add_argument("--allow-many", action="store_true", help=f"allow more than {MAX_CALLS} questions x repeats (uses the shared LLM endpoint)")
    ap.add_argument("--cheap", action="store_true", help="write answers with the small WORKER model instead of the shared main model")
    return ap.parse_args()


def build_items(suite: str):
    import dataset

    items = []
    if suite == "limit":
        items += dataset.load_limit_items()
    if suite == "challenge":
        items += dataset.load_challenge_items()
    if suite in ("training", "all"):
        items += [i for i in dataset.load_workbook_items() if i.stage == "TRAINING"]
    if suite in ("final", "all"):
        items += [i for i in dataset.load_workbook_items() if i.stage == "FINAL_TEST"]
    if suite in ("stress", "all"):
        items += dataset.load_bank_items(("Edge", "Trap")) + [i for i in dataset.load_bank_items() if i.id.startswith("R")]
    if suite == "bank":
        items += dataset.load_bank_items()
    seen, out = set(), []
    for i in items:
        if i.id not in seen:
            seen.add(i.id)
            out.append(i)
    return out


CATEGORY_OF_TRAINING = {"T01": "A", "T02": "B", "T03": "C", "T04": "D", "T05": "E", "T06": "F", "T07": "B",
                        "T08": "G", "T09": "H", "T10": "X", "T11": "X"}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


async def main() -> None:
    args = parse_args()
    os.environ["OBS_SOURCE"] = "eval"
    os.environ["WARM_LLM"] = "0"
    os.environ["TMT_HISTORY"] = "off"               # evaluations recompute: no answer-from-history, no cross-session restore                   # no warm-up ping to the shared endpoint during evaluation
    if args.cheap:
        os.environ["WRITER_LITELLM_MODEL"] = os.environ.get("WORKER_LITELLM_MODEL", "gpt-4o-mini")
        os.environ.pop("WRITER_API_BASE", None)
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    import ground_truth
    import judge as judge_mod
    import metrics
    import observability as obs
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    items = build_items(args.suite)
    if args.ids:
        want = {i.strip() for i in args.ids.split(",")}
        items = [i for i in items if i.id in want]
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit("No questions in this suite (FINAL_TEST rows are blank until the final day).")
    calls = enforce_budget(len(items), args.repeat, args.judge_model == "main" and not args.no_judge, args.allow_many)
    print(f"LLM budget: ~{calls} call(s) to the {'small worker' if args.cheap and calls == len(items) * args.repeat else 'shared main'} model; "
          f"judge: {'off (regex fallback)' if args.no_judge else args.judge_model + ' model'}")
    for it in items:
        it.expected_cat = it.expected_cat or CATEGORY_OF_TRAINING.get(it.id)
    truth = {"closure": ground_truth.closure_truth("Hallesches Tor", "Kaiserin-Augusta-Str."),
             "peak": ground_truth.commute_peak_truth("U Rudow (Berlin)")}

    from agent import app
    from mcp_runtime import get_runtime

    if not args.no_warm:
        rt = get_runtime()
        await asyncio.get_running_loop().run_in_executor(None, rt._ready.wait)
        await asyncio.sleep(9)                     # let the server's background warm-up finish (models, caches)
    ss = InMemorySessionService()
    runner = Runner(app=app, session_service=ss)
    eval_id = time.strftime("ev-%Y%m%d-%H%M%S")
    conn = obs.connect()
    conn.row_factory = lambda cur, row: {d[0]: row[i] for i, d in enumerate(cur.description)}
    print(f"eval {eval_id} · suite={args.suite} · {len(items)} questions × {args.repeat} · budget={metrics.BUDGET_S:.0f}s\n")

    rows, per_item, judges = [], {}, {}
    for it in items:
        exp = metrics.expectations(it, truth)
        runs_for_item = []
        for rep in range(args.repeat):
            sess = await ss.create_session(app_name=app.name, user_id="eval")
            answer = ""
            crash = None
            try:
                async for ev in runner.run_async(user_id="eval", session_id=sess.id,
                                                 new_message=types.Content(role="user", parts=[types.Part(text=it.question)])):
                    if ev.is_final_response() and ev.content and ev.content.parts:
                        answer = ev.content.parts[0].text or ""
            except Exception as e:          # one bad question must not abort the evaluation
                crash = f"{type(e).__name__}: {str(e)[:160]}"
            obs.flush_traces()
            run = conn.execute("SELECT * FROM runs WHERE session_id=? ORDER BY ts DESC LIMIT 1", (sess.id,)).fetchone()
            if run is None:
                print(f"  ! no run row recorded for {it.id}")
                continue
            sp = conn.execute("SELECT COUNT(*) n, SUM(name LIKE 'mcp.tool%') tools FROM spans WHERE trace_id=?", (run["trace_id"],)).fetchone()
            jd = None
            if not args.no_judge and not crash:
                jd = await judge_mod.judge_item(it.question, run["answer"] or "", json.loads(run["facts_json"] or "{}"), exp["criteria"],
                                                exp["truth_text"], model=args.judge_model)
                if jd is None:
                    print(f"  ! judge unavailable for {it.id} r{rep}: falling back to the regex rubric for this item")
            judges[(it.id, rep)] = jd
            m, checks = metrics.score_item(it, run, exp, {"n": sp["n"] or 0, "tools": sp["tools"] or 0}, judge=jd)
            if crash:                       # the run crashed: it counts as a failed answer, not as a skipped one
                checks.append({"name": "run completed without crashing", "ok": False, "detail": crash})
                m.update(answered=0.0, hallucination_free=0.0, readability_ok=0.0, traceable=0.0)
            runs_for_item.append((run, m, checks, rep))
            fails = [c["name"] for c in checks if not c["ok"]]
            print(f"  {it.id:4s} r{rep} {run['total_s']:5.1f}s  cat={run['category'] or '-':4s} {run['facts_status'] or '-':11s} "
                  f"{'PASS' if not fails else 'FAIL: ' + '; '.join(fails)[:110]}", flush=True)
        if len(runs_for_item) > 1:
            cons = metrics.consistency([r for r, *_ in runs_for_item])
            for _, m, _, _ in runs_for_item:
                m["consistency"] = cons
        for run, m, checks, rep in runs_for_item:
            rows.append(m)
            conn.execute("INSERT OR REPLACE INTO eval_items (eval_id, item_id, repeat_idx, stage, category, question, run_id, trace_id, answer, "
                         "latency_s, metrics_json, checks_json, judge_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                eval_id, it.id, rep, it.stage, exp["cat"], it.question, run["run_id"], run["trace_id"], run["answer"],
                run["total_s"], json.dumps(m), json.dumps(checks), json.dumps(judges.get((it.id, rep)))))
        conn.commit()
        per_item[it.id] = runs_for_item

    summary = metrics.group_scores(rows)
    lat = sorted(r["latency_s"] for r in rows)
    summary["latency"] = {"mean": sum(lat) / len(lat), "p50": lat[len(lat) // 2], "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
                          "max": lat[-1], "budget_s": metrics.BUDGET_S, "within_budget": sum(l <= metrics.BUDGET_S for l in lat) / len(lat)}
    summary["n_questions"], summary["n_runs"] = len(items), len(rows)
    judged = [j for j in judges.values() if j]
    summary["scoring"] = {"method": "llm-judge + deterministic gates" if judged else "regex rubric (no judge)",
                          "judge_model": judged[0]["model"] if judged else None, "judge_coverage": len(judged) / max(len(judges), 1),
                          "judge_tokens": sum(j.get("tokens", 0) for j in judged)}
    conn.execute("INSERT OR REPLACE INTO eval_runs VALUES (?,?,?,?,?,?,?,?)", (
        eval_id, time.time(), args.suite, "fast", len(rows), metrics.BUDGET_S, git_commit(),
        json.dumps({**summary, "label": args.label}, default=str)))
    conn.commit()

    fmt = lambda v: "  n/a" if v is None else f"{v:5.2f}"
    print(f"\n=== {eval_id}  ({args.suite}, {len(rows)} runs) ===")
    mm = summary["metrics"]
    print(f"  RELEVANCE   (0.30) {fmt(summary['relevance'])}   answered {fmt(mm.get('answered'))} · completeness {fmt(mm.get('completeness'))} · route {fmt(mm.get('route_correct'))}")
    print(f"  RELIABILITY (0.30) {fmt(summary['reliability'])}   grounded {fmt(mm.get('hallucination_free'))} · fact-accuracy {fmt(mm.get('fact_accuracy'))} · honest-scope {fmt(mm.get('honest_scope'))} · decline-quality {fmt(mm.get('decline_quality'))} · traceable {fmt(mm.get('traceable'))} · consistency {fmt(mm.get('consistency'))}")
    print(f"  STRESS      (0.20) {fmt(summary['stress'])}   latency-ok {fmt(summary['metrics'].get('latency_ok'))} · readable {fmt(summary['metrics'].get('readability_ok'))}")
    sc = summary["scoring"]
    print(f"  scoring: {sc['method']} (judge {sc['judge_model'] or 'none'}, coverage {sc['judge_coverage']:.0%}) · judge-vs-regex agreement {fmt(mm.get('judge_agreement'))} · "
          f"judge relevance {fmt(mm.get('judge_relevance'))} faithfulness {fmt(mm.get('judge_faithfulness'))} clarity {fmt(mm.get('judge_clarity'))}")
    print(f"  OVERALL (weighted, 0.8 of criteria) {fmt(summary['overall'])}")
    print(f"  latency: mean {summary['latency']['mean']:.1f}s · p50 {summary['latency']['p50']:.1f}s · p95 {summary['latency']['p95']:.1f}s · max {summary['latency']['max']:.1f}s · within budget {summary['latency']['within_budget']:.0%}")

    (HERE / "output").mkdir(exist_ok=True)
    (HERE / "output" / f"{eval_id}.json").write_text(json.dumps({
        "eval_id": eval_id, "args": vars(args), "summary": summary,
        "items": [{"id": i, "question": next(it.question for it in items if it.id == i),
                   "runs": [{"answer": r["answer"], "metrics": m, "checks": c} for r, m, c, _ in rs]} for i, rs in per_item.items()]},
        indent=2, default=str))
    if args.export_xlsx:
        export_xlsx(items, per_item, args.team)


def export_xlsx(items, per_item, team: str) -> None:
    """Write the agent's answers into a COPY of the organiser workbook (the original is never touched)."""
    import shutil

    import openpyxl
    import dataset

    out = HERE / "output" / "team_answers_filled.xlsx"
    shutil.copy(dataset.WORKBOOK, out)
    wb = openpyxl.load_workbook(out)
    ws = wb["TEAM_ANSWERS"]
    if team:
        ws["A2"] = team
    n = 0
    for it in items:
        if it.row and per_item.get(it.id):
            ws.cell(row=it.row, column=4, value=per_item[it.id][-1][0]["answer"])
            n += 1
    wb.save(out)
    print(f"\nWrote {n} answers to {out} (TEAM_EVIDENCE rows left blank for the team to author).")


if __name__ == "__main__":
    asyncio.run(main())
