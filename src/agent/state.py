"""
Agent State — supports clarification loop, structured intent, and feedback response ID.
"""
from typing import TypedDict, Optional


class AgentState(TypedDict):
    messages: list                 # list of HumanMessage / AIMessage
    query_type: str                # DISRUPTION|EVENT|ANOMALY|EFFICIENCY|NETWORK|FORECAST|COMBINED|GENERAL|CLARIFICATION_NEEDED
    parsed_context: dict           # structured intent extracted from query
    clarification_question: str    # question sent back to operator when info is missing
    clarification_fields: list     # which fields triggered clarification
    fingerprint: dict              # HCADE situation fingerprint
    tool_results: dict             # raw MCP tool outputs keyed by tool name
    synthesis: dict                # combined analysis result
    recommendation: dict           # single best action from HCADE
    final_response: str            # operator-readable formatted answer
    response_id: str               # UUID for this response — used to link feedback
    error: Optional[str]           # error message if something fails
