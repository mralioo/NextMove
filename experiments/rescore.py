"""Re-score the RECORDED answers of an experiment with the current evaluation (LLM judge + deterministic gates).

    ./.venv/bin/python experiments/rescore.py [exp_id]        # default: the latest experiment run

The arms are NOT re-run (no main-model calls, no new traces): each stored turn's answer, facts and run row are re-judged by the
small judge model and the scores/summaries are rewritten. Use it after improving the evaluation, so old experiment data stays
comparable with new data.
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
sys.path[:0] = [str(HERE), str(REPO / "evaluation"), str(REPO / "agent"), str(REPO)]
import arms  # noqa: E402
import scoring  # noqa: E402
import questions  # noqa: E402


async def main() -> None:
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    import ground_truth
    import judge as judge_mod
    import observability as obs

    conn = obs.connect()
    conn.row_factory = sqlite3.Row
    exp_id = sys.argv[1] if len(sys.argv) > 1 else conn.execute("SELECT exp_id FROM exp_runs ORDER BY ts DESC LIMIT 1").fetchone()[0]
    truth = {"closure": ground_truth.closure_truth("Hallesches Tor", "Kaiserin-Augusta-Str."),
             "peak": ground_truth.commute_peak_truth("U Rudow (Berlin)")}
    arms_ok = [r for r in conn.execute("SELECT * FROM exp_runs WHERE exp_id=? AND status='ok' ORDER BY arm_id", (exp_id,))]
    print(f"re-scoring {exp_id}: {len(arms_ok)} arms x {len(questions.PROTOCOL)} turns with the LLM judge (small model)")
    t0 = time.time()
    for arm_row in arms_ok:
        aid = arm_row["arm_id"]
        turn_rows = {t["turn"]: t for t in conn.execute("SELECT * FROM exp_turns WHERE exp_id=? AND arm_id=?", (exp_id, aid))}
        q1_facts = json.loads(turn_rows["Q1"]["facts_json"]) if "Q1" in turn_rows else {}
        jobs, order = [], []
        for turn, _, _ in questions.PROTOCOL:
            tr = turn_rows.get(turn)
            if tr is None:
                continue
            item, exp = scoring.expectation_for(turn, truth, q1_facts)
            jobs.append(dict(question=tr["question"], answer=tr["answer"] or "", facts=json.loads(tr["facts_json"] or "{}"),
                             criteria=exp["criteria"], ground_truth=exp["truth_text"]))
            order.append(turn)
        verdicts = await judge_mod.judge_many(jobs, concurrency=3, model="small")
        turns, runs = {}, {}
        for turn, jd in zip(order, verdicts):
            tr = turn_rows[turn]
            run = dict(conn.execute("SELECT * FROM runs WHERE run_id=?", (tr["run_id"],)).fetchone())
            sp = conn.execute("SELECT COUNT(*) n, SUM(name LIKE 'mcp.tool%') tools FROM spans WHERE trace_id=?", (run["trace_id"],)).fetchone()
            m, checks = scoring.score_turn(turn, run, {"n": sp["n"] or 0, "tools": sp["tools"] or 0}, truth, q1_facts, judge=jd)
            turns[turn], runs[turn] = (m, checks), run
            conn.execute("UPDATE exp_turns SET metrics_json=?, checks_json=?, judge_json=? WHERE exp_id=? AND arm_id=? AND turn=?",
                         (json.dumps(m), json.dumps(checks), json.dumps(jd), exp_id, aid, turn))
        old = json.loads(arm_row["summary_json"] or "{}")
        new = scoring.arm_summary(turns, runs, old.get("mcp_startup_s"))
        for keep in ("vs_baseline_q1", "memory_store"):
            if keep in old:
                new[keep] = old[keep]
        conn.execute("UPDATE exp_runs SET summary_json=? WHERE exp_id=? AND arm_id=?", (json.dumps(new, default=str), exp_id, aid))
        conn.commit()
        print(f"  {aid:4s} quality {old.get('quality', 0):.2f} -> {new['quality']:.2f} | judge {new['judge'] if new['judge'] is None else round(new['judge'], 2)} | "
              f"judge-vs-regex agreement {new['judge_agreement'] if new['judge_agreement'] is None else round(new['judge_agreement'], 2)}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
