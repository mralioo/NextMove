"""The experiment design: one baseline configuration and one-factor-at-a-time (OFAT) variations of it.

Factors (see agent/config.py for what each level means):
    router  rules | llm                        memory  session | none
    mcp     stdio | inmemory | http            engine  tabpfn | empirical
    writer  small | main | template            (small = gpt-4o-mini, main = the shared main model, template = no LLM)

Every arm gets an auto-generated NAME that spells out its full parameter set, e.g.
    router=llm|memory=session|mcp=stdio|engine=tabpfn|writer=small
so a result can never be separated from the setup that produced it. OFAT keeps the LLM bill small (the main model
is used by exactly one arm) and makes every difference attributable to a single change; interactions between
factors are NOT measured (that would need a factorial design — see docs/experiments_plan.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

BASELINE = {"router": "rules", "memory": "session", "mcp": "stdio", "engine": "tabpfn", "writer": "small"}
ORDER = list(BASELINE)


@dataclass
class Arm:
    arm_id: str
    label: str
    params: dict
    factor: str                       # which factor differs from the baseline ("-" for baseline / replicate)
    hypothesis: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "|".join(f"{k}={self.params[k]}" for k in ORDER)

    @property
    def title(self) -> str:
        return f"{self.arm_id} · {self.name}"


def _arm(arm_id: str, label: str, factor: str, hypothesis: str, **change) -> Arm:
    return Arm(arm_id, label, {**BASELINE, **change}, factor, hypothesis)


ARMS: list[Arm] = [
    _arm("A00", "baseline", "-", "Reference: production workflow with the small writer model."),
    _arm("A01", "baseline replicate 1 (noise floor)", "-",
         "Identical to A00. Differences between A00, A01 and A02 are run-to-run noise (LLM sampling, API jitter); their range sets the "
         "smallest effect we can claim for the other arms."),
    _arm("A02", "baseline replicate 2 (noise floor)", "-", "A third identical run: with A00 and A01 it gives the run-to-run range used as the noise floor."),
    _arm("R1", "router = LLM", "router", "An LLM router is slower than rules and no more accurate on these two questions; it may mis-route the "
         "follow-up.", router="llm"),
    _arm("M1", "memory = none", "memory", "Without session memory the follow-up question cannot be answered (it has no referent).", memory="none"),
    _arm("C1", "mcp = in-memory", "mcp", "Removing the subprocess/pipe layer saves start-up and per-call overhead; answers identical.", mcp="inmemory"),
    _arm("C2", "mcp = HTTP", "mcp", "HTTP adds per-call overhead vs stdio; answers identical.", mcp="http"),
    _arm("E1", "engine = empirical (no ML)", "engine", "Replacing TabPFN with the empirical baseline keeps the ranking similar but changes the "
         "pressure probabilities; speed similar or better.", engine="empirical"),
    _arm("W1", "writer = template (no LLM)", "writer", "A deterministic writer is fastest and fully grounded but reads stiffer and handles "
         "the follow-up less flexibly.", writer="template"),
    _arm("W2", "writer = main model", "writer", "The main model follows the style/honesty rules better than the small one, at the cost of latency "
         "and shared-API credit (3 calls).", writer="main"),
]
BY_ID = {a.arm_id: a for a in ARMS}
