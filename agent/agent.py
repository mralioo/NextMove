"""ADK entry point for "NextMove".

`adk web agent/`, `adk run agent/` and every runner we build ourselves (run_query, bench, evaluation, dashboard) load `app`:
the supervisor -> worker <-> evaluator -> writer pipeline (fast_agent.py) with the observability plugin (traces, and one `runs` row per
question in observability/agent_obs.db).

Models are configured per role in .env (see llm_config.py): SUPERVISOR_* (also the evaluator's model unless EVALUATOR_LITELLM_MODEL is set),
WORKER_* (router fallback, small tasks), WRITER_* (falls back to SUPERVISOR_*).
"""
from __future__ import annotations

import sys
from pathlib import Path

from google.adk.apps import App

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_loader import load_all_dotenvs  # noqa: E402

load_all_dotenvs()

from fast_agent import build_fast_agent  # noqa: E402
from observability import build_plugin  # noqa: E402

root_agent = build_fast_agent()
app = App(name="agent", root_agent=root_agent, plugins=[build_plugin()])
