"""LangSmith: a small test dataset and LIVE experiments of the agent against it.

    ./.venv/bin/python evaluation/langsmith_run.py dataset          # create / update the dataset `nextmove-eval` in LangSmith
    ./.venv/bin/python evaluation/langsmith_run.py run              # run the agent on the CORE examples (3) and upload the experiment
    ./.venv/bin/python evaluation/langsmith_run.py run --all        # all 8 examples
    ./.venv/bin/python evaluation/langsmith_run.py run --offline    # same evaluators, nothing uploaded (works without a key)
    ./.venv/bin/python evaluation/langsmith_run.py status           # is the key valid, what exists in LangSmith

Needs LANGSMITH_API_KEY in .env (EU workspace: also LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com; optional LANGSMITH_PROJECT).
Uploading sends the questions, the agent's answers and its facts to LangSmith — the reason this is an explicit command and never automatic.

Dataset `nextmove-eval` — 8 examples, split `core` (the 3 of the ADK eval set) and `extended`; each has inputs {question}, a reference answer (approximate, from the raw data /
knowledge-base ground truth) and metadata {category, expected_decision}. Per run the experiment records, as LangSmith feedback:
    correctness · groundedness · helpfulness · relevance · conciseness   (openevals LLM judges, small model, 0-1)
    sanity (knowledge-base checks) · decision_correct (supervisor's decision = expected) · within_latency_budget · confidence
Models: the writer and the evaluator use the SMALL model here (the default roles point at the shared main model): this script sets it for you.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "agent"), str(REPO)]

DATASET = "nextmove-eval"
BUDGET_S = 20.0

# (id, split, question, reference answer, category, expected supervisor decision)
EXAMPLES = [
    ("c1_u6_closure", "core",
     "Line U6 is suspended on a section between Hallesches Tor and Kaiserin-Augusta-Strasse stations. What is the reason behind this closure and how long will it last? "
     "How should the passengers be rerouted, which stations would become overloaded, and where should additional staff be deployed?",
     "Closed for a safety inspection on 13 July 2026 from 13:50 to 15:20 (1.5 hours). No rail detour, so a replacement bus is needed. Kaiserin-Augusta-Str. is the most pressured station "
     "(about 78% chance of exceeding its own busiest-5% level versus 17% normally), then Mehringdamm (about 29% vs 10%): deploy staff there. The figures are assumption-based estimates, not capacity.",
     "C", "proceed"),
    ("d1_rudow_peak", "core",
     "At what time does the commute flow peak at Rudow station usually take place? Does it exceed the mean commute peak value across all stations?",
     "Rudow's weekday commute peak is at 18:00 with about 219 passengers per 15 minutes. That is below the network mean weekday peak of about 264 (roughly 17% lower), so it does not exceed it.",
     "D", "proceed"),
    ("a1_arena_concert", "core",
     "There is a sold-out concert at Mercedes-Benz Arena tonight at 21:00. What does the flow look like at Hermannplatz, and what should we do at 23:15 when it ends?",
     "Hermannplatz shows no measurable uplift from events at this venue. The data calls the arena Uber Arena (assumed to be the same venue). Warschauer Str. (about +114 passengers per 15 minutes, ~6x normal) "
     "and Schlesisches Tor (about +93, ~5x) feel it: put additional staff there from 23:15 until about 00:15. 'Tonight' has no date in the data, so the venue's past-event pattern is used; the venue-to-station "
     "link is inferred; there is no capacity data.",
     "A", "proceed"),
    ("e1_energy", "extended",
     "Which metro line has the worst energy-per-passenger efficiency ratio? What factors explain this inefficiency and what interventions would provide the largest improvement?",
     "U5 has the worst ratio: about 550 Wh per passenger, roughly twice the best lines (U9 about 255, U2 about 258). Its energy follows ridership (correlation ~0.98) and it carries fewer passengers per station "
     "than the best lines, so low ridership per station, not idle running, explains it. Interventions are suggestions; passengers per line are approximate and there is no rolling-stock data.",
     "E", "proceed"),
    ("f1_resilience", "extended",
     "Rank the five stations whose closure would fragment the network the most. For each station, estimate the number of passengers affected daily and suggest mitigation strategies.",
     "Alexanderplatz Bhf (about 179,500 passengers/day affected, 25 stations cut off), Bismarckstr. (about 133,700), Schillingstr. (about 131,100), Strausberger Platz (about 124,300), Weberwiese "
     "(about 117,500). Trains are assumed not to run through a closed station; mitigation ideas are suggestions from the graph (bypass via neighbouring stations).",
     "F", "proceed"),
    ("p1_innotrans", "extended",
     "During InnoTrans 2026, we expect major passenger flow and bad weather. Show me the 3 stations most likely to exceed safe platform capacity during the first day of the event.",
     "Highest predicted load on 22 September 2026 (scenario, rain assumed): Spichernstr., Berliner Str. and Kurfürstendamm. This is a load ranking, not a platform capacity: there is no capacity data, and the "
     "events file has no InnoTrans event (the first day is taken as 22 September).",
     "P", "proceed"),
    ("g1_capacity_decline", "extended",
     "How many passengers can the U6 platform at Mehringdamm safely hold?",
     "This cannot be answered: there is no platform-capacity data. It can offer closure analysis, station peaks or a demand-pressure proxy instead.",
     "OOS", "decline"),
    ("g2_unrelated_bounce", "extended",
     "What is the capital of France?",
     "Outside the assistant's scope: it only answers questions about Berlin U-Bahn passenger flows and operations.",
     "BOUNCE", "bounce"),
]


def setup_env() -> None:
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    small = os.environ.get("WORKER_LITELLM_MODEL", "gpt-4o-mini")
    os.environ["WRITER_LITELLM_MODEL"] = small                  # never the shared main model in an evaluation
    os.environ["EVALUATOR_LITELLM_MODEL"] = small
    for k in ("WRITER_API_BASE", "EVALUATOR_API_BASE"):
        os.environ.pop(k, None)
    os.environ["TMT_HISTORY"] = "off"                           # recompute every question
    os.environ["OBS_SOURCE"] = "langsmith"
    os.environ["WARM_LLM"] = "0"


def client():
    from langsmith import Client

    if not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY is not set. Add `LANGSMITH_API_KEY=...` to .env (EU workspace: also LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com), then re-run. "
                         "Nothing was uploaded. `run --offline` works without a key.")
    return Client()


def cmd_status() -> None:
    c = client()
    try:
        ds = [d.name for d in c.list_datasets(limit=50)]
        print(f"key OK · endpoint {c.api_url} · datasets: {ds}")
        if DATASET in ds:
            d = c.read_dataset(dataset_name=DATASET)
            n = len(list(c.list_examples(dataset_id=d.id)))
            print(f"'{DATASET}': {n} examples · {len(list(c.list_projects(reference_dataset_id=d.id)))} experiments")
    except Exception as e:
        raise SystemExit(f"LangSmith rejected the request: {type(e).__name__}: {str(e)[:200]}")


def cmd_dataset() -> None:
    c = client()
    try:
        d = c.read_dataset(dataset_name=DATASET)
        existing = {e.metadata.get("id"): e for e in c.list_examples(dataset_id=d.id) if e.metadata}
        print(f"dataset '{DATASET}' exists with {len(existing)} examples")
    except Exception:
        d = c.create_dataset(DATASET, description="NextMove: approximate-answer test set (3 core questions of the ADK eval set + 5 extended incl. a decline and a bounce). "
                                                  "Reference answers are approximate and come from the raw data / knowledge-base ground truth.")
        existing = {}
        print(f"created dataset '{DATASET}'")
    new = [e for e in EXAMPLES if e[0] not in existing]
    if new:
        c.create_examples(dataset_id=d.id, inputs=[{"question": e[2]} for e in new], outputs=[{"answer": e[3]} for e in new],
                          metadata=[{"id": e[0], "category": e[4], "expected_decision": e[5]} for e in new], splits=[e[1] for e in new])
    print(f"added {len(new)} examples · total {len(EXAMPLES)} · splits: core {sum(e[1] == 'core' for e in EXAMPLES)}, extended {sum(e[1] == 'extended' for e in EXAMPLES)}")
    print(f"open: https://smith.langchain.com  →  Datasets & Experiments  →  {DATASET}")


# ------------------------------------------------------------------------------------------ the live target
_RT: dict = {}


async def _ask(question: str) -> dict:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from agent import app

    ss = InMemorySessionService()
    s = await ss.create_session(app_name=app.name, user_id="langsmith")
    runner = Runner(app=app, session_service=ss)
    t0, final = time.time(), ""
    async for ev in runner.run_async(user_id="langsmith", session_id=s.id, new_message=types.Content(role="user", parts=[types.Part(text=question)])):
        if ev.is_final_response() and ev.content and ev.content.parts:
            final = ev.content.parts[0].text or ""
    st = (await ss.get_session(app_name=app.name, user_id="langsmith", session_id=s.id)).state
    tm, plan = st.get("timing", {}), st.get("plan", {})
    return {"answer": final, "facts": st.get("facts") or {}, "sanity": tm.get("sanity"), "decision": plan.get("decision"), "category": plan.get("cat"),
            "confidence": tm.get("confidence"), "verdict": tm.get("verdict"), "latency_s": round(time.time() - t0, 2), "rounds": len(tm.get("loop") or [])}


def target(inputs: dict) -> dict:
    if "warm" not in _RT:
        from mcp_runtime import get_runtime

        rt = get_runtime()
        rt._ready.wait()
        time.sleep(9)                                            # the MCP server's background model warm-up
        _RT["warm"] = True
    return asyncio.run(_ask(inputs["question"]))


def evaluators():
    import langsmith_eval

    base = langsmith_eval.build_evaluators()

    def decision_correct(run, example):
        want = (example.metadata or {}).get("expected_decision")
        got = (run.outputs or {}).get("decision")
        return {"key": "decision_correct", "score": bool(want and got == want), "comment": f"expected {want}, got {got}"}

    def within_latency_budget(run, example):
        lat = (run.outputs or {}).get("latency_s")
        return {"key": "within_latency_budget", "score": bool(lat is not None and lat <= BUDGET_S), "comment": f"{lat} s (budget {BUDGET_S:.0f} s)"}

    def confidence(run, example):
        c = (run.outputs or {}).get("confidence")
        return {"key": "confidence", "score": None if c is None else float(c), "comment": "worker confidence (deterministic formula)"}

    return base + [decision_correct, within_latency_budget, confidence]


def cmd_run(all_examples: bool, offline: bool) -> None:
    from langsmith import evaluate
    from langsmith.schemas import Example

    chosen = [e for e in EXAMPLES if all_examples or e[1] == "core"]
    prefix = f"nextmove-{time.strftime('%m%d-%H%M')}"
    meta = {"branch": os.popen("git rev-parse --abbrev-ref HEAD").read().strip(), "commit": os.popen("git rev-parse --short HEAD").read().strip(), "writer": os.environ["WRITER_LITELLM_MODEL"],
            "evaluator_llm": os.environ["EVALUATOR_LITELLM_MODEL"], "judge": "gpt-4o-mini (openevals)"}
    if offline:
        ds = uuid.uuid4()
        data = [Example(id=uuid.uuid4(), dataset_id=ds, inputs={"question": e[2]}, outputs={"answer": e[3]}, metadata={"id": e[0], "category": e[4], "expected_decision": e[5]}) for e in chosen]
        upload = False
        print(f"OFFLINE experiment '{prefix}' on {len(data)} examples (nothing uploaded)")
    else:
        c = client()
        try:
            c.read_dataset(dataset_name=DATASET)
        except Exception:
            raise SystemExit(f"dataset '{DATASET}' not found — run: python evaluation/langsmith_run.py dataset")
        data = [ex for ex in c.list_examples(dataset_name=DATASET, splits=None if all_examples else ["core"])]
        upload = True
        os.environ["LANGSMITH_TRACING"] = "true"
        print(f"experiment '{prefix}' on {len(data)} examples of '{DATASET}' — UPLOADING questions, answers and facts to {c.api_url}")
    res = evaluate(target, data=data, evaluators=evaluators(), experiment_prefix=prefix, metadata=meta, max_concurrency=1, upload_results=upload)
    rows = []
    for r in res:
        fb = {x.key: x.score for x in r["evaluation_results"]["results"]}
        out = r["run"].outputs or {}
        rows.append((r["example"].metadata.get("id") if r["example"].metadata else "?", out.get("decision"), out.get("latency_s"), fb))
    keys = sorted({k for *_, fb in rows for k in fb})
    print(f"\n{'example':22s} {'decision':10s} {'lat s':>6s}  " + " ".join(f"{k[:11]:>11s}" for k in keys))
    for eid, dec, lat, fb in rows:
        print(f"{eid:22s} {str(dec):10s} {lat or 0:>6.1f}  " + " ".join(f"{('' if fb.get(k) is None else format(float(fb[k]), '.2f')):>11s}" for k in keys))
    print("\nmeans: " + " · ".join(f"{k} {sum(float(fb[k]) for *_, fb in rows if fb.get(k) is not None) / max(1, sum(fb.get(k) is not None for *_, fb in rows)):.2f}" for k in keys))
    if upload:
        print(f"open: https://smith.langchain.com  →  Datasets & Experiments  →  {DATASET}  →  {prefix}*")


def main() -> None:
    setup_env()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        cmd_status()
    elif cmd == "dataset":
        cmd_dataset()
    elif cmd == "run":
        cmd_run("--all" in sys.argv, "--offline" in sys.argv)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
