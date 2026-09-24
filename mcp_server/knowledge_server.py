"""MCP server for the knowledge base + memory (FastMCP, stdio or HTTP) — for OTHER agents, evaluators and sanity checkers.

    ./.venv/bin/python mcp_server/knowledge_server.py             # stdio
    MCP_TRANSPORT=http MCP_PORT=8766 ./.venv/bin/python mcp_server/knowledge_server.py

Deliberately separate from the data/TabPFN server (mcp_server/server.py): it starts in ~1 s (no model warm-up), is read-mostly,
and is what an external evaluator (LangSmith-style judge, Claude Code, another agent) connects to in order to
  * fetch the verified ground truth and the boundaries (`kb_search`, `kb_boundaries`, `kb_ground_truth`),
  * CHECK an answer or a facts JSON against them (`sanity_check`),
  * read what the agent said in a session (`session_history`) and what Cognee remembers (`cognee_recall`),
  * record a new insight (`add_insight`) — mirrored to Cognee.
All checks are deterministic; the ground truth is recomputed from the raw CSVs by agent/knowledge_build.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastmcp import FastMCP

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "agent")]
from env_loader import load_all_dotenvs  # noqa: E402

load_all_dotenvs()
import knowledge  # noqa: E402
import kgraph  # noqa: E402

mcp = FastMCP("nextmove-knowledge", instructions=(
    "Curated knowledge for the Berlin U-Bahn operator assistant: verified ground truth, boundaries (what the data cannot support) and insights, "
    "plus a deterministic sanity checker and the conversation history. Use kb_boundaries before asserting anything about capacity, dates outside "
    "the data, events or reroute behaviour; use sanity_check to verify an answer."))


@mcp.tool
def kb_search(query: str, category: str = "", kind: str = "", k: int = 3) -> list[dict]:
    """Search ground truth / boundaries / insights by free text. `category` A..X and `kind` boundary|ground_truth|insight are optional filters."""
    return knowledge.kb().search(query, cats=[category] if category else None, kinds=[kind] if kind else None, k=k)


@mcp.tool
def kb_boundaries(category: str = "") -> list[dict]:
    """The boundaries (what the data cannot support). With a category, only those that apply to it."""
    return [e for e in knowledge.kb().entries if e["kind"] == "boundary" and (not category or category in e["cats"])]


@mcp.tool
def kb_ground_truth(topic: str) -> list[dict]:
    """Verified facts matching `topic` (e.g. 'energy per passenger', 'Rudow peak', 'closure U6 Hallesches Tor'), recomputed from the raw CSVs."""
    return knowledge.kb().search(topic, kinds=["ground_truth"], k=3)


@mcp.tool
def sanity_check(question: str, answer: str, facts_json: str = "{}", category: str = "") -> dict:
    """Check an operator answer against the knowledge base: numbers grounded in `facts_json` (the specialist's facts), no capacity /
    bus-service / measured-pressure claims, station names exist, assumption-based results are labelled, and the facts equal ground truth
    recomputed from the raw data. Returns {ok, score, checks[{id, ok, detail}], evidence[kb ids]}."""
    try:
        facts = json.loads(facts_json) if facts_json else {}
    except json.JSONDecodeError:
        return {"ok": False, "checks": [{"id": "S-INPUT", "ok": False, "detail": "facts_json is not valid JSON"}]}
    return knowledge.kb().sanity_check(question, answer, facts, {"cat": category or facts.get("cat")})


@mcp.tool
def session_history(session_id: str, limit: int = 10) -> list[dict]:
    """The last turns (question, category, answer) of an agent session, from the local store."""
    return knowledge.kb().history(session_id, limit)


@mcp.tool
def add_insight(text: str, categories: str = "", source: str = "evaluator") -> dict:
    """Record a new insight (a finding worth remembering). `categories` is a comma-separated list of A..X. Mirrored to Cognee when enabled."""
    return {"id": knowledge.kb().add_insight(text, [c.strip() for c in categories.split(",") if c.strip()], source)}


@mcp.tool
def cognee_recall(query: str, top_k: int = 3, graph: bool = False) -> dict:
    """Semantic recall from the Cognee knowledge graph (`graph=True` returns an LLM-composed answer over the graph; slower). Recall takes
    seconds — this is for evaluators and follow-up analysis, not for the operator's hot path."""
    c = knowledge.cognee()
    if not c.available:
        return {"available": False, "note": "Cognee disabled or unreachable (COGNEE_ENABLED / COGNEE_API_BASE_URL / COGNEE_API_KEY)", "last_error": c.last_error}
    return {"available": True, "results": c.recall(query, top_k, "GRAPH_COMPLETION" if graph else "CHUNKS")}


@mcp.tool
def kg_similar(question: str, category: str = "", k: int = 3) -> list[dict]:
    """Knowledge graph: the past problems most similar to `question` with their accepted answer, the actions taken and the options offered."""
    from schemas import Entities

    import router
    ents = Entities(stations=router.find_stations(question), lines=[m.upper() for m in __import__("re").findall(r"\bU\s?[1-9]\b", question)])
    return [c.model_dump() for c in kgraph.kg().similar(question, category, ents, k=k)]


@mcp.tool
def kg_top_actions(category: str = "", n: int = 10) -> list[dict]:
    """The actions recommended most often in accepted answers (optionally for one category) — what the operators' assistant has proposed most."""
    return kgraph.kg().top_actions(category, n)


@mcp.tool
def kg_neighbors(label: str, key: str) -> dict:
    """Nodes linked to a node of the knowledge graph (label: Problem, Answer, Action, Option, Station, Line, Venue, Event, Concept, Category, Document)."""
    return kgraph.kg().neighbors(label, key)


@mcp.tool
def kg_add_case(question: str, category: str, answer: str, actions: str = "", options: str = "", verdict: str = "accept", source: str = "evaluator") -> dict:
    """Add a problem → answer → actions/options case (`actions` and `options` are ';'-separated). This is how the graph grows from outside the agent."""
    from schemas import Entities, GraphCase

    import router
    e = Entities(stations=router.find_stations(question))
    key = kgraph.kg().record_case(GraphCase(question=question, category=category, entities=e, answer=answer, verdict=verdict, source=source,
                                            actions=[a.strip() for a in actions.split(";") if a.strip()], options=[o.strip() for o in options.split(";") if o.strip()]))
    return {"problem_key": key}


@mcp.tool
def operator_kb_search(query: str, category: str = "", k: int = 3) -> list[dict]:
    """OPERATOR KNOWLEDGE BASE: past accepted answers that resemble `query` — brief, confidence, tools used, datasets, whether a full report exists. Use it to recall what was decided
    the last time a similar situation came up (returns turn ids for operator_kb_get)."""
    return knowledge.kb().find_artifacts(query, category, k)


@mcp.tool
def operator_kb_get(turn_id: int) -> dict:
    """The full artifact bundle of one answered turn: plan and objective, every MCP call with arguments / time / result preview, facts, confidence and reasons, the evaluator's checks,
    LLM calls (model, seconds, tokens), references, the brief and (if it was ever asked for) the full report."""
    return knowledge.kb().get_artifact(turn_id) or {"error": f"no artifact for turn {turn_id}"}


@mcp.tool
def operator_precedents(question: str, category: str = "", k: int = 2) -> list[dict]:
    """What operators DID in situations similar to `question`, and how it went: the reported actions (followed the advice? outcome worked / partly / did_not_work? score 1-5), the mean
    score of the answer given then, and the actions that answer recommended. Read this before recommending actions for a familiar situation; label it as precedent, never as a rule."""
    import feedback
    return feedback.precedents(question, category, None, k=k)


@mcp.tool
def operator_actions(category: str = "", station: str = "", min_score: float = 0, limit: int = 20) -> list[dict]:
    """Actions operators reported having taken (newest first), optionally filtered by category (A-H, P), a station name in the action text, or a minimum score of the report."""
    return kgraph.kg().operator_actions(category, station, min_score or None, limit)


@mcp.tool
def operator_feedback_stats(operator_id: str = "") -> dict:
    """Scores and action reports so far: number, mean score, score histogram, how often the advice was followed, outcomes, by category, and the share of action-requiring answers that got an action report."""
    return knowledge.kb().feedback_stats(operator_id)


@mcp.tool
def kg_stats() -> dict:
    """Size of the knowledge graph: nodes and relationships by type, problems by source."""
    return kgraph.kg().stats()


@mcp.tool
def memory_status() -> dict:
    """Knowledge-base size, stored turns/sessions, and Cognee connectivity + call statistics."""
    c = knowledge.cognee()
    return {"kb": knowledge.kb().stats(), "cognee": {"configured": c.configured, "available": c.available, "last_error": c.last_error,
                                                      "calls": c.stats["calls"], "errors": c.stats["errors"], "recall_ms": c.stats["recall_ms"][-5:]}}


if __name__ == "__main__":
    import os

    if os.environ.get("MCP_TRANSPORT") == "http":
        mcp.run(transport="http", host="127.0.0.1", port=int(os.environ.get("MCP_PORT", "8766")), show_banner=False)
    else:
        mcp.run()
