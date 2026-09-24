"""Fill the RESULTS block of docs/experiments_plan.md from the observability database.

    make experiments-report              # latest experiment run
    make experiments-report EXP=exp-20260923-225649

Everything between <!-- RESULTS:START --> and <!-- RESULTS:END --> is regenerated (tables only — numbers come from the
database, never typed by hand). The human-written FINDINGS section of the document is left alone.
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "agent")]
import arms as A  # noqa: E402

DOC = REPO / "docs" / "experiments_plan.md"
DB = REPO / "observability" / "agent_obs.db"


def esc(s: str) -> str:
    return s.replace("|", "\\|")


def f(v, fmt="{:.2f}", none="–"):
    return none if v is None else fmt.format(v)


def build(exp_id: str | None) -> tuple[str, str]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    exp_id = exp_id or conn.execute("SELECT exp_id FROM exp_runs ORDER BY ts DESC LIMIT 1").fetchone()[0]
    rows = {r["arm_id"]: r for r in conn.execute("SELECT * FROM exp_runs WHERE exp_id=?", (exp_id,))}
    S = {a: json.loads(r["summary_json"] or "{}") for a, r in rows.items() if r["status"] == "ok"}
    reps = [S[a] for a in S if rows[a]["factor"] == "-"]
    lat = [s["latency_mean_s"] for s in reps]
    base, noise = st.mean(lat), max(lat) - min(lat)
    qs = [s["quality"] for s in reps]
    qb, qnoise = st.mean(qs), max(qs) - min(qs)

    out = [f"### Results of `{exp_id}`\n",
           f"Baseline = mean of the {len(reps)} identical replicates: **{base:.2f} s** mean latency per turn; "
           f"noise range across replicates **{noise:.2f} s** ({', '.join(f'{x:.2f}' for x in lat)} s); baseline quality {qb:.2f} "
           f"(replicates {', '.join(f'{x:.2f}' for x in qs)}; noise range **{qnoise:.2f}**). A latency or quality difference counts only if it exceeds the noise range. "
           "Quality and judge scores come from the **LLM judge + deterministic gates** (see section 4).\n",
           "#### All arms\n",
           "| Arm | Run name (full parameter set) | Mean latency (s) | Δ vs baseline | vs noise | Quality | Δ quality | vs noise | Judge | Judge↔regex | LLM calls | Tool calls | Tokens |",
           "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in A.ARMS:
        r = rows.get(arm.arm_id)
        if r is None:
            continue
        if r["status"] != "ok":
            out.append(f"| {arm.arm_id} | `{esc(arm.name)}` | – | – | {r['status']}: {esc((r['note'] or '')[:60])} | – | – | – | – | – | – | – | – |")
            continue
        s, d = S[arm.arm_id], S[arm.arm_id]["latency_mean_s"] - base
        dq = s["quality"] - qb
        tag = "baseline replicate" if arm.factor == "-" else ("**beyond noise**" if abs(d) > noise else "within noise")
        qtag = "replicate" if arm.factor == "-" else ("**beyond noise**" if abs(dq) > qnoise else "within noise")
        out.append(f"| {arm.arm_id} | `{esc(arm.name)}` | {s['latency_mean_s']:.2f} | {d:+.2f} | {tag} | {s['quality']:.2f} | {dq:+.2f} | {qtag} | {f(s.get('judge'))} | "
                   f"{f(s.get('judge_agreement'))} | {s['llm_calls']} | {s['tool_calls']} | {s['tok_in'] + s['tok_out']} |")

    def sec(title, ids, cols):
        out.append(f"\n#### {title}\n")
        out.append("| Arm | " + " | ".join(c for c, _ in cols) + " |")
        out.append("| --- | " + " | ".join("---:" for _ in cols) + " |")
        for a in ids:
            if a in S:
                out.append(f"| {a} | " + " | ".join(fn(S[a]) for _, fn in cols) + " |")

    sec("Router (RQ1)", ["A00", "R1", "R2"], [
        ("Route time Q1 (ms)", lambda s: f(s["route_ms"].get("Q1"), "{:.0f}")), ("Route time Q2 (ms)", lambda s: f(s["route_ms"].get("Q2"), "{:.0f}")),
        ("LLM calls", lambda s: str(s["llm_calls"])), ("Quality", lambda s: f(s["quality"])), ("Follow-up completeness", lambda s: f(s.get("follow_up_completeness")))])
    sec("Memory (RQ2)", ["A00", "M1", "M2"], [
        ("Follow-up completeness", lambda s: f(s.get("follow_up_completeness"))), ("Quality", lambda s: f(s["quality"])),
        ("Tools time Q1 (s)", lambda s: f(s["tools_s"].get("Q1"))), ("Tools time Q1r (s)", lambda s: f(s["tools_s"].get("Q1r"))),
        ("Tool calls", lambda s: str(s["tool_calls"])), ("Speed-up Q1r", lambda s: f(s.get("speedup_q1r"), "{:.0%}")),
        ("Memory hit on Q1r", lambda s: str(s.get("memory_hit_q1r")))])
    sec("MCP transport (RQ3)", ["A00", "C1", "C2"], [
        ("Server start-up (s)", lambda s: f(s.get("mcp_startup_s"), "{:.1f}")), ("Tools time Q1 (s)", lambda s: f(s["tools_s"].get("Q1"))),
        ("Tools time Q1r (s)", lambda s: f(s["tools_s"].get("Q1r"))), ("Mean latency (s)", lambda s: f(s["latency_mean_s"]))])
    sec("ML engine (RQ4)", ["A00", "E1"], [
        ("Top-3 stations", lambda s: ", ".join(s.get("top3_q1", []))),
        ("Same top station", lambda s: str((s.get("vs_baseline_q1") or {}).get("top_station_same", "–"))),
        ("Top-3 overlap", lambda s: f((s.get("vs_baseline_q1") or {}).get("top3_overlap"), "{:.0%}")),
        ("Mean abs Δ probability (pts)", lambda s: f((s.get("vs_baseline_q1") or {}).get("mean_abs_prob_diff_pts"), "{:.1f}")),
        ("Pair-order agreement", lambda s: f((s.get("vs_baseline_q1") or {}).get("pair_order_agreement"), "{:.0%}"))])
    sec("Writer (RQ5)", ["A00", "W1", "W2"], [
        ("Write time Q1 (s)", lambda s: f(s["write_s"].get("Q1"))), ("Tokens (in+out)", lambda s: str(s["tok_in"] + s["tok_out"])),
        ("Quality", lambda s: f(s["quality"])), ("Judge", lambda s: f(s.get("judge"))), ("Consistency Q1↔Q1r", lambda s: f(s.get("consistency_q1_q1r"))),
        ("Guard on Q1", lambda s: str(s["guard"].get("Q1")))])
    out.append("\n_Consistency Q1↔Q1r across arms that should be equivalent: "
               + ", ".join(f"{a} {f(S[a].get('consistency_q1_q1r'))}" for a in ("A00", "A01", "A02", "C1", "C2", "R2") if a in S)
               + " — a wide spread, i.e. this metric is noisy at n = 1._")
    return exp_id, "\n".join(out) + "\n"


def main() -> None:
    exp_id, block = build(sys.argv[1] if len(sys.argv) > 1 else None)
    text = DOC.read_text()
    pat = re.compile(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", re.S)
    if not pat.search(text):
        raise SystemExit(f"{DOC} has no RESULTS markers")
    DOC.write_text(pat.sub(lambda _: f"<!-- RESULTS:START -->\n{block}<!-- RESULTS:END -->", text))
    print(f"Updated the results block of {DOC.relative_to(REPO)} from {exp_id}")


if __name__ == "__main__":
    main()
