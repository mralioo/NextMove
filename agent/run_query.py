"""CLI entry point: run one operator question through the ADK agent.

Usage:
    ./.venv/bin/python agent/run_query.py "When does station Rudow's commute peak happen?"

Requires an LLM key for whichever backend agent.py is configured to use:
  - Gemini (default): GOOGLE_API_KEY or GEMINI_API_KEY
  - LiteLLM (if ADK_LITELLM_MODEL is set): whatever key that provider needs
    (e.g. ANTHROPIC_API_KEY for "anthropic/claude-...")

Put the key in the same .env this repo's other scripts read from (found by
walking up from the working directory — see ml/train_overcrowding_classifier.py).

The MCP server itself (mcp_server/server.py) is spawned automatically as a
subprocess by the ADK toolset — you don't need to start it separately.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


USER_ID = "operator"


async def run(question: str, trace: bool = False) -> None:
    import time

    os.environ.setdefault("OBS_SOURCE", "cli")
    t0 = time.time()
    agent = build_agent_for_cli()
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=agent.name, user_id=USER_ID)
    runner = Runner(app=agent, session_service=session_service)
    message = types.Content(role="user", parts=[types.Part(text=question)])
    final = ""
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=message):
        for part in (event.content.parts if event.content and event.content.parts else []):
            if part.function_call and trace:
                print(f"[tool call] {part.function_call.name}({dict(part.function_call.args or {})})")
            if part.function_response and trace:
                print(f"[tool result] {part.function_response.name} -> {str(part.function_response.response)[:300]}")
            if part.text:
                if event.is_final_response():
                    final = part.text
                elif trace:
                    print(f"[{event.author}] {part.text[:600]}")
    print(final)
    total = time.time() - t0
    st = (await session_service.get_session(app_name=agent.name, user_id=USER_ID, session_id=session.id)).state
    tm = st.get("timing")
    if tm:   # fast pipeline: per-stage breakdown
        calls = ", ".join(f"{c['tool']} {c['s']}s" for c in tm.get("calls", []))
        print(f"\n⏱ total {total:.1f}s · route {tm.get('route_ms')} ms (tier {tm.get('tier')}) · tools {tm.get('tools_s')}s "
              f"[{calls}] · write {tm.get('write_s')}s ({tm.get('tok_in')}→{tm.get('tok_out')} tok) · guard: {tm.get('guard')}")
    else:
        print(f"\n⏱ total {total:.1f}s")


def build_agent_for_cli():
    from agent import app   # App(root_agent + observability plugin)

    return app


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    args = [a for a in sys.argv[1:] if a != "--trace"]
    asyncio.run(run(" ".join(args), trace="--trace" in sys.argv))


if __name__ == "__main__":
    main()
