"""Run ONE experiment arm in its own process (clean caches, clean memory, its own MCP server).

    ./.venv/bin/python experiments/run_arm.py --exp-id exp-... --arm-id R1

Normally started by experiments/run_experiments.py. The arm's parameters become environment variables BEFORE the agent
is imported (agent/config.py reads them), the two-question protocol runs through the real ADK app (so every turn is a
traced run), each turn is scored and stored in `exp_turns`, and the arm summary in `exp_runs`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "evaluation"), str(REPO / "agent"), str(REPO)]

import arms  # noqa: E402
import questions  # noqa: E402


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def apply_environment(arm: arms.Arm, exp_id: str) -> None:
    import config
    from config import RunConfig

    p = arm.params
    cfg = RunConfig(router=p["router"], memory=p["memory"], mcp=p["mcp"], engine=p["engine"],
                    writer="template" if p["writer"] == "template" else "llm").validate()
    os.environ.update(cfg.to_env())
    os.environ["OBS_SOURCE"] = f"exp:{arm.arm_id}"
    os.environ["WARM_LLM"] = "0"
    os.environ["TMT_HISTORY"] = "off"               # evaluations recompute: no answer-from-history, no cross-session restore
    os.environ["TMT_MEMORY_DB"] = str(REPO / "observability" / f"memory_{exp_id}_{arm.arm_id}.db")
    if p["writer"] == "small":
        os.environ["WRITER_LITELLM_MODEL"] = os.environ.get("WORKER_LITELLM_MODEL", "gpt-4o-mini")
        for k in ("WRITER_API_BASE", "WRITER_API_KEY", "WRITER_API_VERSION"):
            os.environ.pop(k, None)
    config.reload()                                        # `config` may already be imported: update the shared CONFIG in place
    expected = {"router": p["router"], "memory": p["memory"], "mcp": p["mcp"], "engine": p["engine"], "writer": "template" if p["writer"] == "template" else "llm"}
    got = {k: getattr(config.CONFIG, k) for k in expected}
    if got != expected:                                    # never run an experiment whose configuration did not take effect
        raise SystemExit(f"[{arm.arm_id}] effective config {got} != arm parameters {expected}")


def record_arm(conn, exp_id, arm, status, note, setup, summary):
    conn.execute("INSERT OR REPLACE INTO exp_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
        exp_id, arm.arm_id, arm.name, arm.label, arm.factor, json.dumps(arm.params), json.dumps(setup), time.time(), status, note,
        json.dumps(summary, default=str)))
    conn.commit()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--arm-id", required=True)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    arm = arms.BY_ID[args.arm_id]

    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    apply_environment(arm, args.exp_id)
    import observability as obs

    conn = obs.connect()
    conn.row_factory = lambda cur, row: {d[0]: row[i] for i, d in enumerate(cur.description)}
    setup = {"git": _git(), "python": platform.python_version(), "protocol": [t for t, _, _ in questions.PROTOCOL], "params": arm.params}

    import ground_truth
    import judge as judge_mod
    import scoring
    from llm_config import litellm_params

    import google.adk
    from agent import app
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from mcp_runtime import get_runtime

    import config as _config
    from dataclasses import asdict
    setup["effective_config"] = asdict(_config.CONFIG)
    setup["adk"] = getattr(google.adk, "__version__", "")
    wm = litellm_params("WRITER")
    setup["writer_model"] = "template (no LLM)" if arm.params["writer"] == "template" else (wm[0] if wm else "none")
    rm = litellm_params("ROUTER")
    setup["router_llm_model"] = rm[0] if rm else "none"
    setup["tabpfn_model"] = "v3.5_default" if arm.params["engine"] == "tabpfn" else "(not used)"

    truth = {"closure": ground_truth.closure_truth("Hallesches Tor", "Kaiserin-Augusta-Str."),
             "peak": ground_truth.commute_peak_truth("U Rudow (Berlin)")}
    rt = get_runtime()
    await asyncio.get_running_loop().run_in_executor(None, rt._ready.wait)
    await asyncio.sleep(9)                                  # let the server's background warm-up finish
    startup = rt.ready_after
    ss = InMemorySessionService()
    runner = Runner(app=app, session_service=ss)

    turns, runs_by_turn, q1_facts, session_id = {}, {}, {}, None
    print(f"[{arm.arm_id}] {arm.name}")
    for turn, text, sess_mode in questions.PROTOCOL:
        if sess_mode == "new" or session_id is None:
            session_id = (await ss.create_session(app_name=app.name, user_id="exp")).id
        answer = ""
        try:
            async for ev in runner.run_async(user_id="exp", session_id=session_id,
                                             new_message=types.Content(role="user", parts=[types.Part(text=text)])):
                if ev.is_final_response() and ev.content and ev.content.parts:
                    answer = ev.content.parts[0].text or ""
        except Exception as e:
            print(f"  {turn}: crashed: {type(e).__name__}: {str(e)[:120]}")
        obs.flush_traces()
        run = conn.execute("SELECT * FROM runs WHERE session_id=? ORDER BY ts DESC LIMIT 1", (session_id,)).fetchone()
        if run is None:
            print(f"  {turn}: no run recorded")
            continue
        sp = conn.execute("SELECT COUNT(*) n, SUM(name LIKE 'mcp.tool%') tools FROM spans WHERE trace_id=?", (run["trace_id"],)).fetchone()
        facts = json.loads(run["facts_json"] or "{}")
        item, exp = scoring.expectation_for(turn, truth, q1_facts)
        jd = None if args.no_judge else await judge_mod.judge_item(text, run["answer"] or "", facts, exp["criteria"], exp["truth_text"], model="small")
        m, checks = scoring.score_turn(turn, run, {"n": sp["n"] or 0, "tools": sp["tools"] or 0}, truth, q1_facts, judge=jd)
        if turn == "Q1":
            q1_facts = facts
        turns[turn], runs_by_turn[turn] = (m, checks), run
        conn.execute("INSERT OR REPLACE INTO exp_turns VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            args.exp_id, arm.arm_id, turn, text, run["answer"], run["run_id"], run["trace_id"], json.dumps(m), json.dumps(checks),
            json.dumps(facts, default=str)[:20000], json.dumps(jd) if jd else None))
        conn.commit()
        eff = (json.loads(run["timing_json"] or "{}").get("cfg") or {})
        if eff and (eff.get("router"), eff.get("memory"), eff.get("mcp"), eff.get("engine")) != (arm.params["router"], arm.params["memory"], arm.params["mcp"], arm.params["engine"]):
            raise SystemExit(f"[{arm.arm_id}] the recorded run used config {eff}, not the arm's parameters — experiment invalid")
        print(f"  {turn:3s} {run['total_s']:5.1f}s  route={run['category'] or '-':6s} status={run['facts_status'] or '-':8s} "
              f"quality={scoring.quality(m):.2f}" + (f" judge(rel/faith/clar)={m['judge_relevance']:.1f}/{m['judge_faithfulness']:.1f}/{m['judge_clarity']:.1f}" if m.get('judge_relevance') is not None else " judge=n/a"))

    summary = scoring.arm_summary(turns, runs_by_turn, startup)
    record_arm(conn, args.exp_id, arm, "ok", "", setup, summary)
    print(f"  => quality {summary['quality']:.2f} · mean latency {summary['latency_mean_s']:.1f}s · LLM calls {summary['llm_calls']} · tools {summary['tool_calls']}")
    rt.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
