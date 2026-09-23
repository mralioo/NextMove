"""Re-score STORED evaluation runs with the LLM judge (no agent re-run, no traces re-created).

    ./.venv/bin/python evaluation/rejudge.py                 # the latest run of every suite
    ./.venv/bin/python evaluation/rejudge.py ev-20260923-... # specific run(s)
    ./.venv/bin/python evaluation/rejudge.py --judge-model main    # judge with the shared main model (spot check)

Runs recorded before the judge existed were scored with the regex rubric only; this rewrites their scores using the
current evaluation (judge for the semantic criteria + deterministic gates) so old and new results are comparable.
Only the SMALL worker model is used unless --judge-model main is passed.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "agent"), str(REPO)]


def item_index():
    import dataset
    from run_eval import CATEGORY_OF_TRAINING

    items = {i.id: i for i in dataset.load_workbook_items()}
    items.update({i.id: i for i in dataset.load_limit_items()})
    items.update({i.id: i for i in dataset.load_bank_items()})
    for iid, it in items.items():
        it.expected_cat = it.expected_cat or CATEGORY_OF_TRAINING.get(iid)
    return items


async def rejudge_run(conn, eval_id: str, items: dict, truth: dict, model: str) -> None:
    import judge as judge_mod
    import metrics

    er = conn.execute("SELECT * FROM eval_runs WHERE eval_id=?", (eval_id,)).fetchone()
    rows = conn.execute("SELECT * FROM eval_items WHERE eval_id=? ORDER BY item_id, repeat_idx", (eval_id,)).fetchall()
    jobs, meta = [], []
    for r in rows:
        it = items.get(r["item_id"])
        if it is None:
            continue
        run = conn.execute("SELECT * FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()
        if run is None:
            continue
        exp = metrics.expectations(it, truth)
        jobs.append(dict(question=r["question"], answer=r["answer"] or "", facts=json.loads(run["facts_json"] or "{}"), criteria=exp["criteria"],
                         ground_truth=exp["truth_text"]))
        meta.append((r, dict(run), it, exp))
    print(f"  {eval_id} ({er['suite']}): judging {len(jobs)} answers with the {model} model ...", flush=True)
    verdicts = await judge_mod.judge_many(jobs, concurrency=4, model=model)

    per_item: dict[str, list] = {}
    all_m = []
    for (r, run, it, exp), jd in zip(meta, verdicts):
        sp = conn.execute("SELECT COUNT(*) n, SUM(name LIKE 'mcp.tool%') tools FROM spans WHERE trace_id=?", (run["trace_id"],)).fetchone()
        m, checks = metrics.score_item(it, run, exp, {"n": sp["n"] or 0, "tools": sp["tools"] or 0}, judge=jd)
        per_item.setdefault(r["item_id"], []).append((r, run, m, checks, jd))
    for iid, lst in per_item.items():
        cons = metrics.consistency([run for _, run, *_ in lst]) if len(lst) > 1 else None
        for r, run, m, checks, jd in lst:
            m["consistency"] = cons
            all_m.append(m)
            conn.execute("UPDATE eval_items SET metrics_json=?, checks_json=?, judge_json=? WHERE eval_id=? AND item_id=? AND repeat_idx=?",
                         (json.dumps(m), json.dumps(checks), json.dumps(jd), eval_id, iid, r["repeat_idx"]))
    old = json.loads(er["summary_json"])
    summary = metrics.group_scores(all_m)
    summary["latency"], summary["n_questions"], summary["n_runs"] = old["latency"], old["n_questions"], len(all_m)
    judged = [jd for lst in per_item.values() for *_, jd in lst if jd]
    summary["scoring"] = {"method": "llm-judge + deterministic gates" if judged else "regex rubric (no judge)",
                          "judge_model": judged[0]["model"] if judged else None, "judge_coverage": len(judged) / max(len(all_m), 1),
                          "judge_tokens": sum(j.get("tokens", 0) for j in judged), "rescored_at": time.strftime("%Y-%m-%d %H:%M")}
    summary["label"] = ((old.get("label") or "") + " [re-scored with LLM judge]").strip()
    conn.execute("UPDATE eval_runs SET summary_json=? WHERE eval_id=?", (json.dumps(summary, default=str), eval_id))
    conn.commit()
    f = lambda v: "n/a" if v is None else f"{v:.2f}"
    print(f"    overall {f(old.get('overall'))} -> {f(summary['overall'])} | relevance {f(old.get('relevance'))} -> {f(summary['relevance'])} | "
          f"reliability {f(old.get('reliability'))} -> {f(summary['reliability'])} | stress {f(old.get('stress'))} -> {f(summary['stress'])} | "
          f"judge-vs-regex agreement {f(summary['metrics'].get('judge_agreement'))} | judged {summary['scoring']['judge_coverage']:.0%}")


async def main() -> None:
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    import ground_truth
    import observability as obs

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = sys.argv[sys.argv.index("--judge-model") + 1] if "--judge-model" in sys.argv else "small"
    args = [a for a in args if a not in ("small", "main")]
    conn = obs.connect()
    conn.row_factory = sqlite3.Row
    ids = args or [r[0] for r in conn.execute("SELECT eval_id FROM eval_runs e WHERE ts=(SELECT MAX(ts) FROM eval_runs WHERE suite=e.suite) ORDER BY ts")]
    truth = {"closure": ground_truth.closure_truth("Hallesches Tor", "Kaiserin-Augusta-Str."),
             "peak": ground_truth.commute_peak_truth("U Rudow (Berlin)")}
    items = item_index()
    print(f"re-scoring {len(ids)} evaluation run(s) with the LLM judge ({model} model)")
    for eid in ids:
        await rejudge_run(conn, eid, items, truth, model)


if __name__ == "__main__":
    asyncio.run(main())
