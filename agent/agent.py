"""ADK agent definition for "Talk To My Train" — the 4-role multi-agent slice.

Implements the role table from docs/Talk_To_My_Train_Engineering_Blueprint.md §3:

    Supervisor            -> classifies the question, delegates, checks the result
    Analysis specialist    -> station/flow/prediction questions (has the MCP tools)
    Scenario specialist    -> closure/event "what-if" questions (C, F, H, bonus)
    Verifier               -> reviews a specialist's draft for ungrounded claims

Wiring pattern: each specialist + the verifier is an ADK sub-agent with
mode="single_turn", attached to the Supervisor via `sub_agents=[...]`. ADK then
auto-exposes each one to the Supervisor as a callable tool (a `request: str` in,
final text out) and runs it inline in the Supervisor's own turn — the Supervisor
stays in control and can call the verifier, then repair, then finalize, all
without handing off the conversation. (ADK's own docs mark plain `AgentTool`
wrapping as the discouraged path now; `mode="single_turn"` + `sub_agents` is the
current recommendation for exactly this "call a sub-agent, get a result back"
shape — see AgentTool's docstring in tools/agent_tool.py.)

Tool surface today is Category D (station profiling), the two TabPFN predict
tools, and Category C (disruption response: resolve_closure -> apply_closure ->
alternate_paths -> scenario_flow, TabPFN-quantile-backed — see
mcp_server/disruption_tools.py and docs/disruption_case_study.md). The Scenario
specialist owns Category C and still honestly declines F/H (no tooling yet):
see docs/agentic_system_design.md.

Two-model setup: pass a strong model to the Supervisor (e.g. an externally
provided "luna" endpoint) and a lighter model to the specialists that actually
burn tool-calling turns. Configure per role — SUPERVISOR_* wins over WORKER_*
which wins over the generic ADK_* (so an unconfigured role just inherits the
single-model default that was already working):

    SUPERVISOR_LITELLM_MODEL=openai/<luna-model-name>   # litellm model string
    SUPERVISOR_API_BASE=https://<luna endpoint>          # only for a custom/
    SUPERVISOR_API_KEY=<luna key>                        # non-OpenAI/Anthropic host
    WORKER_LITELLM_MODEL=gpt-4o-mini                     # analysis/scenario/verifier
    ADK_LITELLM_MODEL=gpt-4o-mini                         # fallback for any unset role
    ADK_MODEL=gemini-2.5-flash                            # fallback if no LiteLLM model set

"openai/<name>" + a custom api_base is LiteLLM's generic way to hit any
OpenAI-compatible chat-completions endpoint — swap the prefix (see LiteLLM's
provider list) if luna speaks a different wire protocol.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from google.adk.agents import Agent

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from env_loader import load_all_dotenvs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_config import litellm_params  # noqa: E402

load_all_dotenvs()
MCP_SERVER_SCRIPT = REPO_ROOT / "mcp_server" / "server.py"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON_EXECUTABLE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

TOOL_SURFACE_NOTE = (
    "Tools available today (via MCP): describe_dataset, list_stations, resolve_station, "
    "station_profile, predict_overcrowding_risk, predict_expected_flow (categories D + point "
    "prediction), and resolve_closure, apply_closure, alternate_paths, scenario_flow "
    "(category C, disruption response). No event, weather, energy, resilience-ranking or "
    "reroute-behavior tools are wired up yet — see docs/agentic_system_design.md for the "
    "planned tool catalog per category."
)

SUPERVISOR_INSTRUCTION = f"""\
You are the supervisor for a Berlin U-Bahn operator decision-support system \
(InnoTrans 2026 hackathon, "Talk To My Train"). You never call data or prediction tools \
yourself — you classify the operator's question, delegate it to exactly one specialist, \
verify the result, and only then answer the operator.

{TOOL_SURFACE_NOTE}

You have three sub-agent tools:
- analysis_specialist: station profiling (peak hour, weekday/weekend rhythm, network-mean \
  benchmark) and point predictions (overcrowding-risk classification, expected-flow \
  regression) for an exact station + timestamp. Use this for most questions right now.
- scenario_specialist: disruption-response questions (category C: "line X suspended between A \
  and B — why, how long, how to reroute, who gets overloaded, where to deploy staff?"), for \
  closures in the dataset or hypothetical ones. Its answers are model-based estimates with \
  explicit assumptions. It still has NO tooling for network-resilience ranking (F), \
  actual-vs-theoretical reroute behavior (H) or event-surge questions — expect it to decline \
  those honestly and do not paper over that when it happens.
- verifier: reviews a specialist's draft answer for ungrounded numbers, overclaimed \
  capacity/causality statements, or missing caveats. It has no tools of its own.

PROCESS for every operator question:
1. Classify the question and pick ONE specialist (analysis_specialist or scenario_specialist).
2. Call that specialist with a clear, scoped restatement of the question as its `request`.
3. Call verifier with the specialist's full draft answer (and the original question) for review.
4. If verifier says PASS: relay the specialist's answer to the operator, substance unchanged.
5. If verifier requests a REPAIR: call the SAME specialist exactly once more with the \
   verifier's specific correction folded into the request, then finalize regardless of what \
   the second verifier pass says — never loop more than once.
6. If verifier says PARTIAL or ABSTAIN: pass that framing through to the operator plainly, \
   don't hide it or round it up to a confident answer.

Never answer from your own general knowledge — every number the operator sees must have come \
from a specialist that used a tool. Keep your own commentary to routing, not content.
"""

ANALYSIS_INSTRUCTION = f"""\
You are the analysis specialist. You receive one scoped question from the supervisor (not \
the raw operator conversation) and must answer it using ONLY the tools below — never from \
general knowledge or by inventing a number.

{TOOL_SURFACE_NOTE}

RULES:
- Call describe_dataset first if you're unsure of the coverage window.
- Always call resolve_station before station_profile / predict_* unless you were given an \
  exact station_name already (they look like 'U Rudow (Berlin)').
- If a tool returns an "error" key, say so plainly and explain what's missing — do not guess \
  a plausible-sounding number instead.
- predict_overcrowding_risk and predict_expected_flow only work for EXACT 15-minute \
  timestamps inside the dataset's coverage window (historical replay, not forecasting).
- When a prediction tool reports seen_in_training_sample=true, say this is an in-sample \
  sanity check, not a held-out evaluation.
- If the scoped question is actually outside station profiling / point predictions (event \
  impact, closures, energy, resilience, anomalies, correlations), say plainly this specialist \
  can't answer it and name the category if you can (see docs/agentic_system_design.md).
- Be concise: the number(s), the comparison, one sentence of grounding context.
"""

SCENARIO_INSTRUCTION = f"""\
You are the scenario specialist. You solve category C — disruption response — for the \
supervisor: a line section or station is closed (a closure from the dataset, or a hypothetical), \
and the operator wants the reason, duration, reroute options, which stations come under \
pressure, and where to deploy staff.

{TOOL_SURFACE_NOTE}

PROCEDURE (call in this order, using only tool output for facts):
1. resolve_closure(query) — if the question names a date/line/station/reason. Its `reason`, \
   `start`, `end` and `duration_hours` are the ONLY source for "why" and "how long". If no \
   closure matches, treat the closure as hypothetical (you need line + from_station + to_station, \
   or station, plus start 'YYYY-MM-DD HH:MM' and duration_minutes; ask the supervisor to supply \
   what's missing rather than inventing it).
2. apply_closure(closure_id | hypothetical args) — unserved stations, stations still served \
   by another line, groups cut off from the rail network.
3. alternate_paths(same args) — rail detours with transfers; if none, say "no rail detour — \
   replacement bus needed" and cite surface_link_candidates as geographic hints only.
4. scenario_flow(same args) — TabPFN demand estimates + redistribution. Only works for windows \
   inside the dataset coverage window; its first call takes about a minute.
Pass the same closure_id / hypothetical arguments to steps 2-4; nothing is remembered between calls.

NON-NEGOTIABLE FRAMING for anything from scenario_flow:
- These are MODEL-BASED ESTIMATES built on ASSUMPTIONS (the tool lists them), never measurements. \
  Say "estimated", and mention that the assumed diversion share is low/base/high (0.25/0.5/0.75) \
  and that the ranking depends on it.
- "Overloaded" / "under pressure" means: probability of exceeding that station's OWN historical \
  95th percentile for the same hour/day-type (prob_exceed_own_p95), compared with the \
  prob_exceed_without_closure baseline. Never call this a capacity limit — the dataset has no \
  platform, train or headway capacity data.
- If historical_reality_check is present, report that observed flow at the receiving stations \
  is consistent with normal variation (the dataset shows no measurable redistribution), so the \
  redistribution is scenario planning, not a replay of what happened.
- Staff deployment = the ranked_pressure_stations list, most-pressured first, with the added \
  load and probabilities. It's a prioritisation aid, not a staffing formula.
- Rail alternatives are valid graph paths, not timetable-feasible services.
- Do not add causes, times or numbers that no tool returned. If a tool returns an "error" key, \
  say so plainly (e.g. window outside the coverage window, ambiguous station) and stop.

Out of scope (say so plainly, name the category): network-resilience ranking (F), \
actual-vs-theoretical reroute behavior (H), event-surge / bonus questions — no tooling exists \
for them yet. Simple station-profile or point-prediction questions can be answered with \
those tools under the usual grounding rules.
"""

VERIFIER_INSTRUCTION = """\
You are the verifier. You have no tools. You will be given the original operator question \
and a specialist's draft answer. Check the draft against this list:

1. Every number in the draft should plausibly come from a tool call the specialist would \
   have made (station stats, prediction probabilities/values) — not invented.
2. No sentence should present a flow percentile or historical peak as a measured "safe \
   capacity" limit — that's a demand-pressure proxy, not a capacity figure (see \
   docs/agentic_system_design.md data-gap #2). Flag this if present.
3. No causal claim ("caused by X") should be stated as fact rather than a candidate \
   explanation, unless the draft is answering a pure station-profiling question with no \
   causal claim at all (in which case this check trivially passes).
4. If the draft says a tool returned an error or a capability isn't built, that should be \
   stated plainly to the operator, not glossed over or replaced with a guess.
5. The draft shouldn't claim precision the data can't support (e.g. minute-level precision \
   from 15-minute-bin data).
6. For disruption/closure answers (scenario_flow output): added load, exceedance probabilities \
   and pressure rankings must be framed as model-based ESTIMATES resting on stated assumptions \
   (diversion share, spill rule), not as measured or predicted-as-fact passenger counts; \
   "overload" must be defined relative to the station's own p95, not capacity; closure reason \
   and duration must come from the closure record, not be guessed. A number that is a literal \
   field of a tool result (e.g. `prob_exceed_own_p95`, `added_load`) needs no further \
   calculation — do not flag it for lacking one.

Respond with EXACTLY one line in one of these four forms, nothing else:
PASS: <one sentence why this is fine to relay as-is>
REPAIR: <one specific, actionable correction the specialist should make>
PARTIAL: <what's solid enough to relay, what's missing or unsupported>
ABSTAIN: <why nothing in this draft is safe to present to the operator>
"""


def _build_model(role: str):
    """role: "SUPERVISOR" or "WORKER". Role-specific env vars win; both fall back to the
    generic ADK_LITELLM_MODEL / ADK_MODEL so a single-model setup keeps working unchanged
    (e.g. before a role-specific "luna" endpoint is configured, SUPERVISOR just inherits
    whatever WORKER/ADK_* is already set).

    Azure OpenAI/Foundry endpoints (host contains "azure.com") get special handling,
    verified empirically against the real "luna" endpoint:
      - LiteLLM's `azure/` provider prefix is required — a plain `openai/<name>` string
        pointed at an Azure host 404s, because LiteLLM's azure/ adapter constructs the
        `/openai/...` request path itself. Auto-corrected here rather than left to fail.
      - `api_base` must be just the resource host (scheme + netloc), not a full URL with
        path/query — Azure portals often hand out the latter (e.g.
        ".../openai/responses?api-version=..."), which breaks LiteLLM's own path
        construction (it appends its own suffix, corrupting the URL). Trimmed here.
      - `api_version` must be passed as its own LiteLLM kwarg, not embedded in the URL —
        extracted from the api_base's query string automatically if `{role}_API_VERSION`
        isn't set explicitly.
    """
    cfg = litellm_params(role)
    if cfg is None:
        return os.environ.get("ADK_MODEL", "gemini-2.5-flash")
    model_name, kwargs = cfg
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=model_name, **kwargs)


def _build_mcp_toolset():
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from mcp import StdioServerParameters

    # The MCP SDK launches the child with a stripped-down environment and may resolve
    # the interpreter symlink, so a venv python can lose its site-packages
    # ("No module named 'fastmcp'" -> "Connection closed"). Pass the parent's env and
    # import path through explicitly so the server always sees the same packages.
    child_env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=PYTHON_EXECUTABLE,
                args=[str(MCP_SERVER_SCRIPT)],
                env=child_env,
            ),
            timeout=30,
        ),
    )


def build_analysis_specialist() -> Agent:
    return Agent(
        name="analysis_specialist",
        mode="single_turn",
        model=_build_model("WORKER"),
        instruction=ANALYSIS_INSTRUCTION,
        tools=[_build_mcp_toolset()],
    )


def build_scenario_specialist() -> Agent:
    return Agent(
        name="scenario_specialist",
        mode="single_turn",
        model=_build_model("WORKER"),
        instruction=SCENARIO_INSTRUCTION,
        tools=[_build_mcp_toolset()],
    )


def build_verifier() -> Agent:
    return Agent(
        name="verifier",
        mode="single_turn",
        model=_build_model("WORKER"),
        instruction=VERIFIER_INSTRUCTION,
    )


def build_agent() -> Agent:
    """Builds and returns the Supervisor — the root agent ADK's CLI/web UI discovers."""
    return Agent(
        name="supervisor",
        model=_build_model("SUPERVISOR"),
        instruction=SUPERVISOR_INSTRUCTION,
        sub_agents=[
            build_analysis_specialist(),
            build_scenario_specialist(),
            build_verifier(),
        ],
    )


def build_root_agent():
    """AGENT_MODE=fast (default): router -> executor -> writer pipeline (seconds).
    AGENT_MODE=llm: the original Supervisor + specialists + Verifier LLM loop (~1 min, open-ended)."""
    if os.environ.get("AGENT_MODE", "fast").lower() == "llm":
        return build_agent()
    from fast_agent import build_fast_agent

    return build_fast_agent()


root_agent = build_root_agent()
