"""Latency benchmark for the fast pipeline: one warm process, a sequence of questions.

    make bench            # or: ./.venv/bin/python agent/bench.py [--llm]  (--llm = old supervisor loop)

Prints per-question wall time and the stage breakdown (route / tools / write), plus the guard result.
The first question includes the MCP server warm-up if it hasn't finished (as in a freshly started server);
'--wait' waits for warm-up first, which is what a long-running `adk web` / API process looks like.
"""
import asyncio
import os
import sys
import time
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

QUESTIONS = [
    ("C  training Q3", "Line U6 is suspended on a section between Hallesches Tor and Kaiserin-Augusta-Strasse stations. What is the reason behind this closure and how long will it last? How should the passengers be rerouted, which stations would become overloaded, and where should additional staff be deployed?"),
    ("C  follow-up", "How confident are you that those stations will really be overloaded?"),
    ("C  other closure", "U1 was suspended between Hallesches Tor and Schlesisches Tor on September 21st. Why and what are the rail alternatives?"),
    ("C  what-if", "What if the U2 were suspended between Alexanderplatz and Potsdamer Platz on September 15th from 17:00 for two hours?"),
    ("D  training Q4", "At what time does the commute flow peak at Rudow station usually take place? Does it exceed the mean commute peak value across all stations?"),
    ("D  prediction", "What was the flow at Rudow on July 15th at 08:00 and what would the model have predicted?"),
    ("A  unsupported", "There's a Guns N' Roses concert on June 23rd at the Uber Arena. What will the passenger flow look like at the neighboring stations?"),
    ("OOS capacity", "How many passengers can the U6 platform at Mehringdamm safely hold?"),
]


async def main() -> None:
    os.environ.setdefault("OBS_SOURCE", "bench")
    from agent import app
    from mcp_runtime import get_runtime

    t_boot = time.time()
    agent = app
    rt = get_runtime() if "--llm" not in sys.argv else None
    if rt and "--wait" in sys.argv:
        await asyncio.get_running_loop().run_in_executor(None, rt._ready.wait)
        await asyncio.sleep(9)   # let the server's background warm-up (models, caches) finish
    print(f"boot {time.time() - t_boot:.1f}s\n")
    ss = InMemorySessionService()
    s = await ss.create_session(app_name=agent.name, user_id="u")
    runner = Runner(app=agent, session_service=ss)
    total = 0.0
    for label, q in QUESTIONS:
        t = time.time()
        final = ""
        async for ev in runner.run_async(user_id="u", session_id=s.id, new_message=types.Content(role="user", parts=[types.Part(text=q)])):
            if ev.is_final_response() and ev.content and ev.content.parts:
                final = ev.content.parts[0].text or ""
        dt = time.time() - t
        total += dt
        tm = (await ss.get_session(app_name=agent.name, user_id="u", session_id=s.id)).state.get("timing", {})
        print(f"{label:16s} {dt:5.1f}s | route {tm.get('route_ms', '-')}ms · tools {tm.get('tools_s', '-')}s · write {tm.get('write_s', '-')}s | {tm.get('guard', '')}")
        if "--show" in sys.argv:
            print("   " + final.replace("\n", "\n   ") + "\n")
    print(f"\nmean {total / len(QUESTIONS):.1f}s per question")


asyncio.run(main())
