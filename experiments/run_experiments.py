"""Run the component-comparison experiment suite (2 questions x arms), one process per arm.

    make experiments                                   # every arm (~10 min; the main model is used by ONE arm, 3 calls)
    make experiments ARGS="--list"                     # show the design without running anything
    make experiments ARGS="--arms A00,A01,M1,M2"       # a subset
    make experiments ARGS="--arms A00,W2 --no-judge"

Results go to the `exp_runs` / `exp_turns` tables (dashboard page "Experiments") and to experiments/output/<exp_id>.json.
Design, hypotheses and how to read the results: docs/experiments_plan.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "agent"), str(REPO / "evaluation"), str(REPO)]
import arms  # noqa: E402
import questions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="", help="comma-separated arm ids (default: all)")
    ap.add_argument("--list", action="store_true", help="print the design and exit")
    ap.add_argument("--no-judge", action="store_true", help="skip the small-model qualitative judge")
    ap.add_argument("--note", default="", help="free-text note stored with the experiment")
    args = ap.parse_args()

    chosen = [arms.BY_ID[a.strip()] for a in args.arms.split(",") if a.strip()] if args.arms else arms.ARMS
    print(f"Two questions, protocol {[t for t, _, _ in questions.PROTOCOL]}, {len(chosen)} arm(s):\n")
    for a in chosen:
        print(f"  {a.arm_id:4s} {a.label:34s} {a.name}\n       H: {a.hypothesis}")
    main_calls = sum(3 for a in chosen if a.params["writer"] == "main")
    print(f"\nShared main-model calls this experiment will make: {main_calls}  (small-model calls for the other arms/judge are cheap)")
    if args.list:
        return

    import observability as obs

    exp_id = time.strftime("exp-%Y%m%d-%H%M%S")
    obs.connect().close()
    print(f"\n=== {exp_id} ===")
    for a in chosen:
        cmd = [sys.executable, str(HERE / "run_arm.py"), "--exp-id", exp_id, "--arm-id", a.arm_id] + (["--no-judge"] if args.no_judge else [])
        t = time.time()
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
        lines = [l for l in r.stdout.splitlines() if l.startswith(("[", "  ", " "))
                 and not any(x in l for x in ("warn", "INFO", "Starting", "stdio", "╭", "│", "╰"))]
        print("\n".join(l for l in lines if l.strip()))
        if r.returncode != 0:
            print(f"  !! arm {a.arm_id} exited with code {r.returncode}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
        print(f"  (arm wall time {time.time() - t:.0f}s)\n")

    # cross-arm comparison of the ML engine's Q1 facts against the baseline arm
    conn = obs.connect()
    conn.row_factory = lambda cur, row: {d[0]: row[i] for i, d in enumerate(cur.description)}
    import scoring
    base = conn.execute("SELECT facts_json FROM exp_turns WHERE exp_id=? AND arm_id='A00' AND turn='Q1'", (exp_id,)).fetchone()
    if base:
        base_facts = json.loads(base["facts_json"])
        for row in conn.execute("SELECT arm_id, summary_json FROM exp_runs WHERE exp_id=? AND status='ok'", (exp_id,)).fetchall():
            f = conn.execute("SELECT facts_json FROM exp_turns WHERE exp_id=? AND arm_id=? AND turn='Q1'", (exp_id, row["arm_id"])).fetchone()
            summ = json.loads(row["summary_json"])
            summ["vs_baseline_q1"] = scoring.facts_similarity(base_facts, json.loads(f["facts_json"])) if f else None
            conn.execute("UPDATE exp_runs SET summary_json=? WHERE exp_id=? AND arm_id=?", (json.dumps(summ), exp_id, row["arm_id"]))
        conn.commit()

    rows = conn.execute("SELECT * FROM exp_runs WHERE exp_id=? ORDER BY arm_id", (exp_id,)).fetchall()
    (HERE / "output").mkdir(exist_ok=True)
    (HERE / "output" / f"{exp_id}.json").write_text(json.dumps(
        {"exp_id": exp_id, "note": args.note, "arms": [{**r, "params": json.loads(r["params_json"]), "summary": json.loads(r["summary_json"] or "{}")} for r in rows]},
        indent=2, default=str))
    print(f"{'arm':5s}{'label':34s}{'quality':>8s}{'lat(s)':>8s}{'LLM':>5s}{'tools':>6s}{'consist.':>9s}{'Q2 done':>8s}")
    for r in rows:
        s = json.loads(r["summary_json"] or "{}")
        if r["status"] != "ok":
            print(f"{r['arm_id']:5s}{r['label']:34s}  {r['status']}: {r['note'][:70]}")
            continue
        f = lambda v, fmt="{:.2f}": "  n/a" if v is None else fmt.format(v)
        print(f"{r['arm_id']:5s}{r['label']:34s}{f(s['quality']):>8s}{f(s['latency_mean_s'], '{:.1f}'):>8s}{s['llm_calls']:>5d}{s['tool_calls']:>6d}"
              f"{f(s.get('consistency_q1_q1r')):>9s}{f(s.get('follow_up_completeness')):>8s}")
    print(f"\nSaved: experiments/output/{exp_id}.json · dashboard page 'Experiments'")


if __name__ == "__main__":
    main()
