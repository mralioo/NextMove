"""A scripted 6-turn conversation through the real ADK app: scope bounce, follow-up rerun, follow-up explanation with argument, history hit, decline.

    ./.venv/bin/python evaluation/conversation_demo.py            # small models only (writer + evaluator); ~6 questions

Shows, per turn: the supervisor's decision, the worker/evaluator loop, guardrail results, latency and the answer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "agent"), str(REPO)]
os.environ.setdefault("OBS_SOURCE", "conversation")
os.environ["WARM_LLM"] = "0"
import tempfile
os.environ["TMT_MEMORY_DB"] = os.path.join(tempfile.mkdtemp(), "demo_memory.db")     # a fresh turn history: the demo is reproducible
os.environ["WRITER_LITELLM_MODEL"] = os.environ.get("WORKER_LITELLM_MODEL", "gpt-4o-mini")     # never the shared main model in a demo
os.environ.pop("WRITER_API_BASE", None)
os.environ.setdefault("EVALUATOR_LITELLM_MODEL", os.environ["WRITER_LITELLM_MODEL"])
os.environ.pop("EVALUATOR_API_BASE", None)

TURNS = [
    "There is a sold-out concert at Mercedes-Benz Arena tonight at 21:00. What does the flow look like at Hermannplatz, and what should we do at 23:15 when it ends?",
    "What about if it ends at 22:30 instead?",
    "Why do you say Hermannplatz is not affected?",
    "There is a sold-out concert at Mercedes-Benz Arena tonight at 21:00. What does the flow look like at Hermannplatz, and what should we do at 23:15 when it ends?",
    "What is the capital of France?",
    "How many passengers can the U6 platform at Mehringdamm safely hold?",
]


async def main() -> None:
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from agent import app
    from mcp_runtime import get_runtime

    rt = get_runtime()
    await asyncio.get_running_loop().run_in_executor(None, rt._ready.wait)
    await asyncio.sleep(9)
    ss = InMemorySessionService()
    s = await ss.create_session(app_name=app.name, user_id="demo-operator")
    runner = Runner(app=app, session_service=ss)
    for i, q in enumerate(TURNS, 1):
        t = time.time()
        final = ""
        async for ev in runner.run_async(user_id="demo-operator", session_id=s.id, new_message=types.Content(role="user", parts=[types.Part(text=q)])):
            if ev.is_final_response() and ev.content and ev.content.parts:
                final = ev.content.parts[0].text or ""
        st = (await ss.get_session(app_name=app.name, user_id="demo-operator", session_id=s.id)).state
        plan, tm = st.get("plan", {}), st.get("timing", {})
        print(f"\n{'=' * 100}\nTURN {i}: {q}\n  decision={plan.get('decision')} category={plan.get('cat')} specialist={plan.get('route', {}).get('specialist')} "
              f"follow_up={plan.get('follow_up')} history={(plan.get('history') or {}).get('kind')}  |  {time.time() - t:.1f}s  llm_evals={tm.get('evaluator_llm_calls')}  confidence={tm.get('confidence')}  "
              f"verdict={tm.get('verdict')}\n  guardrails not passed: {[g['check'] + ':' + g['action'] for g in tm.get('guardrails', [])]}\n  loop: {json.dumps(tm.get('loop'), ensure_ascii=False)[:420]}\n{final}")


if __name__ == "__main__":
    asyncio.run(main())
