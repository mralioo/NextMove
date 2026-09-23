"""Run configuration for the agent workflow — the knobs the experiment suite turns.

Read once from the TMT_CONFIG environment variable (a JSON object), so an experiment arm can run in its own
process with its own setup. Defaults reproduce the production workflow.

    router   rules    deterministic classifier (+ small-LLM fallback when unsure)   [production]
             llm      small LLM always classifies (no rules)
             tfidf    local TF-IDF + logistic-regression classifier for the category (rules still extract entities)
             jev      external JEV classification API (opt-in: needs JEV_API_KEY; sends the question to a third party)
    memory   session  the previous answer's facts are kept for follow-up questions in the same session   [production]
             none     no memory: every question is answered in isolation
             episodic session memory + a persistent store of past facts reused across sessions (skips the tools)
    mcp      stdio    MCP server as a subprocess over stdio                          [production]
             inmemory MCP server in the same process (no subprocess / pipes)
             http     MCP server as a separate process over streamable HTTP
    engine   tabpfn   disruption scenarios use the TabPFN demand model               [production]
             empirical  same scenario logic with the naive empirical baseline (no ML)
    writer   llm      the LLM writes the brief (model chosen by WRITER_LITELLM_MODEL)  [production]
             template deterministic template writer (no LLM at all)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

CHOICES = {
    "router": ("rules", "llm", "tfidf", "jev"),
    "memory": ("session", "none", "episodic"),
    "mcp": ("stdio", "inmemory", "http"),
    "engine": ("tabpfn", "empirical"),
    "writer": ("llm", "template"),
}


@dataclass
class RunConfig:
    router: str = "rules"
    memory: str = "session"
    mcp: str = "stdio"
    engine: str = "tabpfn"
    writer: str = "llm"

    def validate(self) -> "RunConfig":
        for k, allowed in CHOICES.items():
            if getattr(self, k) not in allowed:
                raise ValueError(f"config {k}={getattr(self, k)!r}; allowed: {allowed}")
        return self

    def to_env(self) -> dict[str, str]:
        """Environment for a child process (also carries the engine to the MCP server)."""
        return {"TMT_CONFIG": json.dumps(asdict(self)), "SCENARIO_ENGINE": self.engine}


def load() -> RunConfig:
    raw = os.environ.get("TMT_CONFIG")
    cfg = RunConfig(**{k: v for k, v in json.loads(raw).items() if k in CHOICES}) if raw else RunConfig()
    return cfg.validate()


CONFIG = load()


def reload() -> RunConfig:
    """Re-read TMT_CONFIG and update the shared CONFIG object IN PLACE, so modules that did
    `from config import CONFIG` before the environment was set still see the new values."""
    new = load()
    for k in CHOICES:
        setattr(CONFIG, k, getattr(new, k))
    return CONFIG
