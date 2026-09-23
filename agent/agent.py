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

Tool surface today is Category D (station profiling) + the two TabPFN predict
tools (see mcp_server/server.py) — so in practice the Scenario specialist will
mostly decline (honestly) until Category C/F/H tooling (graph/closure/reroute)
exists. That's intentional, not a bug: see docs/agentic_system_design.md.

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

load_all_dotenvs()
MCP_SERVER_SCRIPT = REPO_ROOT / "mcp_server" / "server.py"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON_EXECUTABLE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

TOOL_SURFACE_NOTE = (
    "Tools available today (via MCP): describe_dataset, list_stations, resolve_station, "
    "station_profile, predict_overcrowding_risk, predict_expected_flow. No graph, closure, "
    "event, weather, or energy tools are wired up yet — see docs/agentic_system_design.md "
    "for the planned tool catalog per category."
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
- scenario_specialist: closure/event "what-if" and rerouting questions (categories C, F, H, \
  and the bonus scenario questions). It has no graph/closure tooling in this MVP, so expect \
  it to decline most scenario questions honestly — do not paper over that when it happens.
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
You are the scenario specialist. Your remit is "what-if" and closure/event scenario \
questions: disruption rerouting, network fragmentation/resilience ranking, and actual-vs-\
theoretical reroute behavior (categories C, F, H, and the bonus InnoTrans-surge question).

{TOOL_SURFACE_NOTE}

IMPORTANT — read before answering: this MVP build has NO graph, closure, or rerouting tools \
wired up yet. If the supervisor's request is a genuine scenario question, you almost \
certainly cannot answer it — say so plainly: "Not supported yet in this MVP — no \
graph/closure tooling is wired up (see docs/agentic_system_design.md, categories C/F/H)." \
Do not attempt to reason your way to a plausible-sounding rerouting answer from general \
knowledge; a wrong-but-confident reroute recommendation is worse than an honest decline.

The one exception: if the request turns out to be simple enough that the station-profiling \
or prediction tools you DO have access to can actually answer it, use them and answer \
normally, with the same grounding rules as the analysis specialist (resolve_station first, \
never state an ungrounded number, report tool errors plainly).
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
    model_name = (
        os.environ.get(f"{role}_LITELLM_MODEL")
        or os.environ.get("WORKER_LITELLM_MODEL")
        or os.environ.get("ADK_LITELLM_MODEL")
    )
    if not model_name:
        return os.environ.get("ADK_MODEL", "gemini-2.5-flash")

    from google.adk.models.lite_llm import LiteLlm

    kwargs = {}
    api_base = os.environ.get(f"{role}_API_BASE")
    api_key = os.environ.get(f"{role}_API_KEY")
    api_version = os.environ.get(f"{role}_API_VERSION")

    if api_base:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(api_base)
        if "azure.com" in parsed.netloc:
            if not api_version:
                qs_version = parse_qs(parsed.query).get("api-version")
                api_version = qs_version[0] if qs_version else None
            api_base = f"{parsed.scheme}://{parsed.netloc}"
            if not model_name.startswith("azure/"):
                model_name = "azure/" + model_name.split("/", 1)[-1]
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if api_version:
        kwargs["api_version"] = api_version

    return LiteLlm(model=model_name, **kwargs)


def _build_mcp_toolset():
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from mcp import StdioServerParameters

    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=PYTHON_EXECUTABLE,
                args=[str(MCP_SERVER_SCRIPT)],
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


root_agent = build_agent()
