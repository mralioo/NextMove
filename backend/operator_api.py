"""OPERATOR API — the backend the UI talks to for the feedback loop and the operator knowledge base.

    ./.venv/bin/python backend/operator_api.py                 # http://localhost:8770   (OPERATOR_API_PORT), interactive docs at /docs, schema at /openapi.json
    make up                                                     # starts it with the other services (service name: operator-api)

The chat itself stays on the ADK server (`/run`, see docs/operator_feedback_api.md); this API adds what the UI needs around the answers: rate a response, report what was done
about the situation, see which answers still wait for an action report, read the stored artifact / full report of an answer, and look up precedents. Everything is written to the
operator knowledge base (SQLite), the knowledge graph (+ Neo4j mirror) and, when it is the configured memory, the memory agent — see agent/feedback.py.
No authentication in this prototype: put it behind the operator's gateway; `operator_id` is taken from the request (it is the ADK `user_id`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "agent")]
from env_loader import load_all_dotenvs  # noqa: E402

load_all_dotenvs()

import feedback  # noqa: E402
import knowledge  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

Followed = Literal["as_recommended", "modified", "different", "none"]
Outcome = Literal["worked", "partly", "did_not_work", "unknown"]

app = FastAPI(title="NextMove operator API", version="1.0",
              description="Feedback loop (score of a response, what the operator did) and operator knowledge base for the Talk To My Train UI.")
app.add_middleware(CORSMiddleware, allow_origins=[o for o in os.environ.get("OPERATOR_API_CORS", "*").split(",") if o], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------------------------------------ request bodies
class ScoreBody(BaseModel):
    operator_id: str = Field("", description="ADK user_id of the operator (defaults to the operator of the turn)")
    score: int = Field(..., ge=1, le=5, description="1 = useless / wrong … 5 = exactly what I needed. A thumbs-up/down UI sends 5 / 1")
    comment: str | None = Field(None, max_length=1000)


class ActionBody(BaseModel):
    operator_id: str = ""
    action_text: str = Field(..., min_length=3, max_length=1500, description="What the operator did about the situation, in their own words")
    followed: Followed | None = Field(None, description="Did the operator follow the advice? as_recommended | modified | different | none")
    outcome: Outcome | None = Field(None, description="How it went, if known (can be added later with PATCH)")
    occurred_at: str | None = Field(None, description="When it was done (ISO 8601 or free text); optional")
    score: int | None = Field(None, ge=1, le=5, description="Optionally rate the response in the same submission")
    comment: str | None = Field(None, max_length=1000)


class FeedbackBody(BaseModel):
    """Score and / or action in one call (at least one of them)."""
    turn_id: int
    operator_id: str = ""
    score: int | None = Field(None, ge=1, le=5)
    action_text: str | None = Field(None, max_length=1500)
    followed: Followed | None = None
    outcome: Outcome | None = None
    occurred_at: str | None = None
    comment: str | None = Field(None, max_length=1000)


class PatchBody(BaseModel):
    score: int | None = Field(None, ge=1, le=5)
    comment: str | None = None
    action_text: str | None = None
    followed: Followed | None = None
    outcome: Outcome | None = None
    occurred_at: str | None = None


def _run(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except feedback.FeedbackError as e:
        raise HTTPException(status_code=404 if str(e).startswith("unknown") else 422, detail=str(e))


# ------------------------------------------------------------------------------------------------ meta
@app.get("/api/v1/health", tags=["meta"])
def health() -> dict:
    """Is the API up, and how much is stored."""
    return {"status": "ok", "knowledge_base": knowledge.kb().stats(), "feedback": knowledge.kb().feedback_stats()}


@app.get("/api/v1/enums", tags=["meta"])
def enums() -> dict:
    """The allowed values, for building the form (labels are the UI's business)."""
    return {"score": {"min": 1, "max": 5, "meaning": {"1": "useless or wrong", "3": "partly useful", "5": "exactly what I needed"}}, "followed": list(feedback.FOLLOWED),
            "outcome": list(feedback.OUTCOMES), "action_text_max_chars": 1500}


# ------------------------------------------------------------------------------------------------ turns (answers)
@app.get("/api/v1/turns", tags=["turns"])
def list_turns(operator_id: str = "", session_id: str = "", limit: int = Query(30, ge=1, le=200), needs_action_report: bool = False) -> list[dict]:
    """Recent answers of an operator / session with `operator_score`, `n_feedback`, `requires_action`, `action_reported`, `has_report`, the brief-less summary fields the UI needs
    to show 'rate this answer' and 'tell us what you did'. `needs_action_report=true` = only accepted answers that asked for an action and have no action report yet."""
    return knowledge.kb().list_turns(operator_id, session_id, limit, needs_action_report)


@app.get("/api/v1/turns/{turn_id}", tags=["turns"])
def get_turn(turn_id: int) -> dict:
    """One answer: question, the answer text as stored, confidence, recommended actions, precedent line, tools used, and all its feedback records."""
    kb = knowledge.kb()
    info = kb.turn_info(turn_id)
    if info is None:
        raise HTTPException(404, f"unknown turn {turn_id}")
    return {**info, "feedback": kb.feedback_for_turn(turn_id)}


@app.get("/api/v1/turns/{turn_id}/artifact", tags=["turns"])
def get_artifact(turn_id: int) -> dict:
    """The whole stored bundle of an answer (plan, every tool call with arguments / time / result preview, facts, confidence + reasons, checks, LLM calls, brief, report, precedents)."""
    art = knowledge.kb().get_artifact(turn_id)
    if art is None:
        raise HTTPException(404, f"no artifact for turn {turn_id} (declines, questions for missing input and answers stored before artifacts existed have none)")
    return art


@app.get("/api/v1/turns/{turn_id}/report", tags=["turns"])
def get_report(turn_id: int) -> dict:
    """The full report of an answer IF it was generated before (the operator asked why / evidence / tools in chat). It is not generated here: the UI sends "why?" to the agent
    (same session) — then the report is stored and this endpoint returns it. The deterministic 'how this was worked out' section is always available as `method`."""
    import artifacts
    art = knowledge.kb().get_artifact(turn_id)
    if art is None:
        raise HTTPException(404, f"no artifact for turn {turn_id}")
    return {"turn_id": turn_id, "has_report": bool(art.get("report")), "report": art.get("report"), "method": artifacts.appendix(art), "brief": art.get("brief")}


# ------------------------------------------------------------------------------------------------ feedback
@app.post("/api/v1/feedback", tags=["feedback"], status_code=201)
def post_feedback(body: FeedbackBody) -> dict:
    """Score and / or action in one call. Feeds the knowledge base, the knowledge graph and (if configured) the memory agent. Returns the record and what it was fed to (`fed`)."""
    return _run(feedback.submit, body.turn_id, operator_id=body.operator_id, score=body.score, comment=body.comment, action_text=body.action_text, followed=body.followed,
                outcome=body.outcome, occurred_at=body.occurred_at)


@app.post("/api/v1/turns/{turn_id}/score", tags=["feedback"], status_code=201)
def post_score(turn_id: int, body: ScoreBody) -> dict:
    """Rate a response (1-5). A response rated 1-2 is no longer reused from history."""
    return _run(feedback.submit, turn_id, operator_id=body.operator_id, score=body.score, comment=body.comment)


@app.post("/api/v1/turns/{turn_id}/action", tags=["feedback"], status_code=201)
def post_action(turn_id: int, body: ActionBody) -> dict:
    """Report what was done about the situation (the field under an answer: 'what did you do?'). Stations and lines named in the text are extracted and linked in the graph; the action
    is matched against the actions the answer recommended. It becomes a precedent for similar situations."""
    return _run(feedback.submit, turn_id, operator_id=body.operator_id, score=body.score, comment=body.comment, action_text=body.action_text, followed=body.followed,
                outcome=body.outcome, occurred_at=body.occurred_at)


@app.get("/api/v1/turns/{turn_id}/feedback", tags=["feedback"])
def turn_feedback(turn_id: int) -> list[dict]:
    """All feedback records of one answer."""
    if knowledge.kb().turn_info(turn_id) is None:
        raise HTTPException(404, f"unknown turn {turn_id}")
    return knowledge.kb().feedback_for_turn(turn_id)


@app.patch("/api/v1/feedback/{feedback_id}", tags=["feedback"])
def patch_feedback(feedback_id: int, body: PatchBody) -> dict:
    """Edit a record — typically add the `outcome` once it is known ("it worked" / "did_not_work"), or correct the action text. The graph is refreshed."""
    return _run(feedback.update, feedback_id, **body.model_dump(exclude_none=True))


@app.delete("/api/v1/feedback/{feedback_id}", tags=["feedback"])
def delete_feedback(feedback_id: int) -> dict:
    """Withdraw a record (it no longer counts in scores or precedents)."""
    if not feedback.delete(feedback_id):
        raise HTTPException(404, f"unknown feedback {feedback_id}")
    return {"deleted": feedback_id}


@app.get("/api/v1/feedback/pending", tags=["feedback"])
def pending(operator_id: str = "", limit: int = Query(20, ge=1, le=100)) -> list[dict]:
    """Answers that asked for an action and still have no action report: the list behind a 'you have N situations to close' badge."""
    return knowledge.kb().list_turns(operator_id, "", limit, needs_action_report=True)


@app.get("/api/v1/feedback/stats", tags=["feedback"])
def stats(operator_id: str = "") -> dict:
    """Number and mean of scores, histogram, how often advice was followed, outcomes, by category, action-report rate."""
    return knowledge.kb().feedback_stats(operator_id)


# ------------------------------------------------------------------------------------------------ using the feedback
@app.get("/api/v1/precedents", tags=["knowledge"])
def get_precedents(question: str, category: str = "", k: int = Query(2, ge=1, le=5)) -> dict:
    """What operators did in situations similar to `question` and how it went, plus the ready-made one-line `precedent_line` the brief would show."""
    prec = feedback.precedents(question, category, None, k=k)
    return {"precedents": prec, "precedent_line": feedback.precedent_line(prec) or None}


@app.get("/api/v1/operator-actions", tags=["knowledge"])
def operator_actions(category: str = "", station: str = "", min_score: float | None = None, limit: int = Query(30, ge=1, le=200)) -> list[dict]:
    """Every action operators reported (newest first), filtered by category / a station in the text / minimum score."""
    import kgraph
    return kgraph.kg().operator_actions(category, station, min_score, limit)


@app.get("/api/v1/artifacts/search", tags=["knowledge"])
def search_artifacts(q: str, category: str = "", k: int = Query(5, ge=1, le=20)) -> list[dict]:
    """Past accepted answers that resemble `q` (operator knowledge base): brief, confidence, tools, whether a full report exists."""
    return knowledge.kb().find_artifacts(q, category, k)


@app.get("/api/v1/graph/stats", tags=["knowledge"])
def graph_stats() -> dict:
    """Size of the knowledge graph by node / relationship type (Feedback, OperatorAction, Artifact … included)."""
    import kgraph
    return kgraph.kg().stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("OPERATOR_API_HOST", "127.0.0.1"), port=int(os.environ.get("OPERATOR_API_PORT", "8770")), log_level="warning")
