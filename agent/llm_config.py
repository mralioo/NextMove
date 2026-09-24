"""LiteLLM connection settings per role, shared by the ADK models (agent.py) and the fast pipeline
(router / writer call LiteLLM directly to keep latency and tokens minimal).

Roles: SUPERVISOR, WORKER, and the fast-pipeline roles ROUTER and WRITER. Each falls back
ROLE_* -> WORKER_* -> ADK_* so a single-model setup keeps working; see agent.py's docstring.

Azure OpenAI/Foundry hosts (host contains "azure.com") need the `azure/` provider prefix, an
`api_base` trimmed to scheme+host, and `api_version` as its own kwarg (extracted from the URL's
query string if not set explicitly) — verified against the real "luna" endpoint.
"""
from __future__ import annotations

import os
from urllib.parse import parse_qs, urlparse


# Which env roles a role falls back to when it has no model of its own. The writer is the one
# step where instruction-following matters most, so it prefers the strong SUPERVISOR model.
CHAINS = {
    "WRITER": ("WRITER", "SUPERVISOR", "WORKER", "ADK"),
    "SUPERVISOR": ("SUPERVISOR", "WORKER", "ADK"),
    "ROUTER": ("ROUTER", "WORKER", "ADK"),
    "WORKER": ("WORKER", "ADK"),
    # the evaluator is the "powerful" LLM, the same model as the supervisor's (EVALUATOR_LITELLM_MODEL overrides it)
    "EVALUATOR": ("EVALUATOR", "SUPERVISOR", "WORKER", "ADK"),
}


def litellm_params(role: str) -> tuple[str, dict] | None:
    """(model_name, kwargs) for litellm.completion, or None if no LiteLLM model is configured.
    Connection settings (api_base/key/version) come from the SAME env role that supplied the model."""
    src, model_name = None, None
    for r in CHAINS.get(role, (role, "WORKER", "ADK")):
        if os.environ.get(f"{r}_LITELLM_MODEL"):
            src, model_name = r, os.environ[f"{r}_LITELLM_MODEL"]
            break
    if not model_name:
        return None
    kwargs: dict = {}
    api_base = os.environ.get(f"{src}_API_BASE")
    api_key = os.environ.get(f"{src}_API_KEY")
    api_version = os.environ.get(f"{src}_API_VERSION")
    if api_base:
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
    return model_name, kwargs


def sampling_params(model_name: str, temperature: float = 0.2) -> dict:
    """Reasoning models (gpt-5*) reject `temperature`; give them a low reasoning effort instead (~3 s vs
    tens of seconds of hidden reasoning). Everything else gets a plain temperature."""
    if "gpt-5" in model_name or os.environ.get("WRITER_REASONING_EFFORT"):
        return {"reasoning_effort": os.environ.get("WRITER_REASONING_EFFORT", "low")}
    return {"temperature": temperature}
