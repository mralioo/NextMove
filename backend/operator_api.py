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
sys.path[:0] = [str(REPO), str(REPO / "agent"), str(REPO / "backend")]
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
              description="Feedback loop (score of a response, what the operator did) and operator knowledge base for the NextMove UI.")
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
# ------------------------------------------------------------------------------------------------ operator desktop: chat, topology, replay snapshot
class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=1500)
    operator_id: str = Field("operator", description="becomes the ADK user_id")
    session_id: str | None = Field(None, description="omit to start a new conversation; send the returned session_id to continue it (follow-ups, 'why?')")
    context_mode: Literal["auto", "continue", "new"] = Field("auto", description="auto = ask the operator when the message does not look connected to the conversation's situation; "
                                                             "continue = connect it anyway; new = start a new conversation (fresh context)")
    link_turn_id: int | None = Field(None, description="only with no session_id: start the new session CONNECTED to this earlier answer (resume a conversation from the history)")


_SESSION_TURN: dict[str, int] = {}          # session_id -> the turn whose situation the conversation is about (covers answers reused from history)


def _short(t: str, n: int = 90) -> str:
    import threads
    return threads.short_title(t or "", n)


def _prepare_chat(body: ChatBody):
    """Shared by /chat and /chat/stream: decide whether the message needs the operator's choice (topic switch), else which session / link to use.
    Returns (needs_choice_payload | None, sid, link, started_new, relation, anchor)."""
    import time as _t

    import threads
    kb = knowledge.kb()
    sid, link, mode = body.session_id, body.link_turn_id, body.context_mode
    anchor = (kb.session_anchor(sid) or (kb.turn_anchor(_SESSION_TURN[sid]) if sid in _SESSION_TURN else None)) if sid else (kb.turn_anchor(link) if link else None)
    rel = threads.relation(body.message, anchor) if anchor else {"relation": "related", "reason": "first question of the conversation", "signals": {}}
    if anchor and mode == "auto" and rel["relation"] != "related":
        qents = rel["signals"].get("new_entities")
        return ({"needs_choice": True, "relation": rel["relation"], "reason": rel["reason"], "message": body.message, "session_id": sid,
                 "anchor": {"question": _short(anchor["first_question"]), "current": _short(anchor["question"]), "category": anchor["category"], "turn_id": anchor["turn_id"]},
                 "choices": [{"id": "new", "label": "New conversation", "hint": "fresh context — recommended for a different topic"}, {"id": "continue", "label": "Continue this topic", "hint": "connect it to the previous question"}],
                 "recommended": "new" if rel["relation"] == "unrelated" or qents else "continue"}, None, None, None, rel, anchor)
    if mode == "new":
        sid, link = None, None
    started_new = sid is None
    sid = sid or f"ui-{int(_t.time() * 1000)}"
    return None, sid, link, started_new, rel, anchor


def _finish_chat(out: dict, body: ChatBody, sid: str, link, started_new: bool, rel: dict, anchor) -> dict:
    if out.get("artifact_turn_id"):                          # an answer reused from history stores no turn of its own: remember which situation the conversation is about
        _SESSION_TURN[sid] = out["artifact_turn_id"]
    mode = body.context_mode
    out["context"] = {"mode": "new" if started_new and not link else ("resumed" if started_new else mode), "relation": rel["relation"], "reason": rel["reason"],
                      "title": _short(anchor["first_question"] if anchor else body.message), "linked_turn_id": link if started_new else None}
    return out


@app.post("/api/v1/chat", tags=["chat"])
def chat(body: ChatBody) -> dict:
    """Ask the agent. Returns the answer (Markdown) plus everything the desktop shows next to it: `turn_id` / `artifact_turn_id` / `requires_action` / `precedent` (for feedback),
    `steps` (operations log), `tools`, `llm`, `tokens` {in,out,total,estimated}, `timing` {supervisor_s (Dispatcher), worker_evaluator_s (Analyst + Inspector), mcp_s, writer_s, total_s}, `counts`.

    **One conversation = one situation.** With an existing `session_id` and `context_mode = auto`, a message that is not clearly about the conversation's situation (another kind of question,
    other line / station / event, or a vague short message) is NOT sent to the agent. The response is `{"needs_choice": true, "relation", "reason", "anchor", "choices", "recommended", "message"}`;
    the UI shows two buttons and calls again with `context_mode = "continue"` (connect it to the previous topic) or `"new"` (start a new conversation: fresh session, no inherited context).
    The response's `context` says what happened: `{mode: new|continue|resumed|auto, relation, reason, title}`."""
    import chat_bridge
    choice, sid, link, started_new, rel, anchor = _prepare_chat(body)
    if choice:
        return choice
    try:
        out = chat_bridge.ask(body.operator_id, sid, body.message, link_turn_id=link if started_new else None)
    except chat_bridge.AgentUnavailable as e:
        raise HTTPException(503, str(e))
    return _finish_chat(out, body, sid, link, started_new, rel, anchor)


@app.post("/api/v1/chat/stream", tags=["chat"])
def chat_stream(body: ChatBody):
    """Same as `/chat`, streamed as Server-Sent Events so the UI can show the operations while the agent works. Events (`event:` name → JSON `data:`):
    `choice` (the topic-switch question, then the stream ends) · `step` {kind: route|tool|worker|evaluator|llm|writer, stage: dispatcher|analyst|inspector|writer, label, seconds, detail} ·
    `answer` (the complete response, identical to `/chat`) · `error` {message}."""
    import json as _json

    import chat_bridge
    from fastapi.responses import StreamingResponse
    choice, sid, link, started_new, rel, anchor = _prepare_chat(body)

    def gen():
        if choice:
            yield f"event: choice\ndata: {_json.dumps(choice)}\n\n"
            return
        for name, data in chat_bridge.ask_stream(body.operator_id, sid, body.message, link_turn_id=link if started_new else None):
            if name == "answer":
                data = _finish_chat(data, body, sid, link, started_new, rel, anchor)
            yield f"event: {name}\ndata: {_json.dumps(data, default=str)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"cache-control": "no-cache", "x-accel-buffering": "no"})


@app.get("/api/v1/conversations", tags=["chat"])
def conversations(operator_id: str = "", limit: int = Query(30, ge=1, le=100)) -> list[dict]:
    """The chat history: one entry per conversation (a conversation resumed from the history is still one entry), newest first — `title` = the first question, `category`, `n_turns`,
    `started`, `last` (epoch seconds), `mean_score`, `resumed`, `session_ids`, `last_session_id`, `last_turn_id`."""
    return knowledge.kb().conversation_chains(operator_id, limit)


@app.get("/api/v1/conversations/{session_id}", tags=["chat"])
def conversation(session_id: str) -> dict:
    """All questions and answers of a conversation (and of the sessions it was resumed in), oldest first, with what the UI needs to show rating / action controls again
    (`turn_id`, `has_artifact`, `requires_action`, `operator_score`, `action_reported`), and `resume_turn_id`: send it as `link_turn_id` (no session_id) to continue the conversation."""
    kb = knowledge.kb()
    chain = next((c for c in kb.conversation_chains("", 200) if session_id in c["session_ids"]), None)
    if chain is None:
        raise HTTPException(404, f"unknown conversation {session_id}")
    turns = [t for sid in chain["session_ids"] for t in kb.conversation_turns(sid)]
    anchor = kb.session_anchor(chain["last_session_id"]) or next((kb.turn_anchor(t["turn_id"]) for t in reversed(turns) if kb.turn_anchor(t["turn_id"])), None)
    return {**{k: chain[k] for k in ("conversation_id", "title", "category", "n_turns", "started", "last", "session_ids", "last_session_id", "resumed")}, "turns": turns,
            "anchor": anchor, "resume_turn_id": (anchor or {}).get("turn_id")}


# ------------------------------------------------------------------------------------------------ analytics pages (network, flows, cascade, energy, closures)
@app.get("/api/v1/status", tags=["analytics"])
def api_status() -> dict:
    """Liveness of the data layer and of the agent: {status, stations_count, flows_rows, data_window, lines, agent_up, models{role: model}, quality_db, knowledge_base}."""
    import analytics_data
    import chat_bridge
    out = analytics_data.status()
    try:
        import urllib.request
        out["agent_up"] = urllib.request.urlopen(chat_bridge.adk_url() + "/list-apps", timeout=2).status == 200
    except Exception:
        out["agent_up"] = False
    try:
        from llm_config import litellm_params
        out["models"] = {r.lower(): (litellm_params(r) or [None])[0] for r in ("SUPERVISOR", "ROUTER", "WORKER", "EVALUATOR", "WRITER")}
    except Exception:
        out["models"] = {}
    out["quality_db"] = (REPO / "data" / "quality" / "quality.db").exists()
    out["knowledge_base"] = knowledge.kb().stats()["entries"]
    return out


@app.get("/api/v1/stations", tags=["analytics"])
def api_stations() -> list[dict]:
    """The 167 stations: {name (exact), short_name, lat, lon, lines[], daily_flow (mean passengers per day), betweenness, degree}."""
    import analytics_data
    return analytics_data.stations()


@app.get("/api/v1/network/edges", tags=["analytics"])
def api_edges() -> list[dict]:
    """Track connections, one row per (edge, serving line): {line, from, to, from_lat, from_lon, to_lat, to_lon}."""
    import analytics_data
    return analytics_data.edges()


@app.get("/api/v1/flows/daily", tags=["analytics"])
def api_flows_daily() -> dict:
    """Network passengers per day (training + test window) with a 7-day rolling mean: {dates[], values[], rolling_mean[]}."""
    import analytics_data
    return analytics_data.flows_daily()


@app.get("/api/v1/flows/hourly-profile", tags=["analytics"])
def api_hourly_profile() -> list[dict]:
    """Network passengers per hour of the day, mean over weekdays and over weekends: [{hour: 'HH:00', weekday, weekend}]."""
    import analytics_data
    return analytics_data.hourly_profile()


@app.get("/api/v1/flows/heatmap", tags=["analytics"])
def api_heatmap(lines: str = "") -> dict:
    """Mean passengers per hour per station: all stations of ONE line, else the 20 busiest of the selected lines (`lines=U1,U2`): {stations[], station_ids[], values[][24], max_value, count, lines}."""
    import analytics_data
    return analytics_data.heatmap([x for x in lines.split(",") if x])


@app.get("/api/v1/flows/station/{name}", tags=["analytics"])
def api_flow_station(name: str) -> dict:
    """One station: hourly weekday / weekend profile, peak hours, daily totals."""
    import analytics_data
    r = analytics_data.station_flow(name)
    if "error" in r:
        raise HTTPException(404, r["error"])
    return r


@app.get("/api/v1/closures", tags=["analytics"])
def api_closures() -> list[dict]:
    """Recorded closures (training + test): {id, when, end, duration_hours, description, closure_type: line_suspension|station_closure, line, segment, reason}."""
    import analytics_data
    return analytics_data.closures()


@app.get("/api/v1/centrality", tags=["analytics"])
def api_centrality(n: int = Query(15, ge=1, le=167)) -> list[dict]:
    """The `n` busiest stations with daily flow and betweenness centrality."""
    import analytics_data
    return analytics_data.centrality(n)


@app.get("/api/v1/energy", tags=["analytics"])
def api_energy() -> dict:
    """Energy per passenger by line (Wh per passenger, worst first) with the evidence that explains it: {ranking:[{line, wh_per_pax, mwh_day, pax_day, n_stations, pax_per_station_day, corr_energy_pax,
    weekend_vs_weekday_*_pct, efficiency_rank, explanation}], unit, method, limits}. Same method as the agent's energy tool."""
    import analytics_data
    return analytics_data.energy()


class CascadeBody(BaseModel):
    closed_stations: list[str] = Field(..., min_length=1, description="station names (exact or loose, e.g. 'Hallesches Tor')")
    timestamp: str = Field("2026-07-13 14:00:00", description="Berlin local time; the typical flow of that weekday and 15-minute slot is used")
    share: float = Field(0.5, ge=0.05, le=1.0, description="share of the closed stations' typical flow that is diverted (the agent uses 0.25 / 0.5 / 0.75)")
    hops: int = Field(2, ge=1, le=4)


@app.post("/api/v1/cascade", tags=["analytics"])
def api_cascade(body: CascadeBody) -> dict:
    """Passenger cascade simulator: which open stations receive the closed stations' passengers and how far above their typical load they go (`overflow_ratio`, `at_risk` ≥ 1.3×).
    An assumption scenario (stated in `method` / `assumption`), not a measurement."""
    import analytics_data
    r = analytics_data.cascade(body.closed_stations, body.timestamp, body.share, body.hops)
    if "error" in r:
        raise HTTPException(422, r["error"])
    return r


@app.get("/api/v1/ops/topology", tags=["desktop"])
def ops_topology() -> dict:
    """Stations (id, short name, lon/lat, lines), edges (with the lines that serve them) and line colours — enough to draw the network map."""
    import ops_data
    return ops_data.topology()


@app.get("/api/v1/ops/timeline", tags=["desktop"])
def ops_timeline() -> dict:
    """The replay window (start, end, 15-minute step), a default time with an active closure, and all closures (for markers on the time bar). There is no live feed: the desktop replays recorded data."""
    import ops_data
    return ops_data.timeline()


@app.get("/api/v1/ops/snapshot", tags=["desktop"])
def ops_snapshot(at: str) -> dict:
    """The network at one 15-minute slot (ISO time): passengers per station vs the typical value for that weekday and slot, line loads, network total, busiest stations, closures active
    then (with the blocked edges and unserved stations), events around then, weather, and alerts."""
    import ops_data
    try:
        return ops_data.snapshot(at)
    except Exception as e:
        raise HTTPException(422, f"cannot build a snapshot for {at!r}: {type(e).__name__}: {str(e)[:120]}")


@app.get("/api/v1/ops/series", tags=["desktop"])
def ops_series(date: str) -> list[dict]:
    """Network passengers per 15 minutes for one day (`YYYY-MM-DD`) with the typical value — the sparkline under the map."""
    import ops_data
    return ops_data.series(date)


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


@app.get("/api/v1/graph/taxonomy", tags=["knowledge"])
def graph_taxonomy() -> dict:
    """The graph by category and domain (Events, Disruptions, Stations, Network, Energy, Diagnostics, Operations planning, Strategy): situations per category, and how often operators used each
    ActionType (staff_deployment, replacement_bus, passenger_information, rerouting, crowd_control, monitoring, service_change, coordination, other)."""
    import kgraph
    return kgraph.kg().taxonomy()


@app.get("/api/v1/graph/audit", tags=["knowledge"])
def graph_audit() -> dict:
    """Wrong or uncategorised branches: problems without situation / category / kind, follow-up-shaped problems, untyped actions, orphan feedback."""
    import kgraph
    return kgraph.kg().audit()


@app.post("/api/v1/graph/repair", tags=["knowledge"])
def graph_repair() -> dict:
    """Repair what the audit finds (nothing is deleted): kinds, situations, categories with domains, follow-up marking, action types. Returns before / fixed / after."""
    import kgraph
    return kgraph.kg().repair()


@app.get("/api/v1/graph/stats", tags=["knowledge"])
def graph_stats() -> dict:
    """Size of the knowledge graph by node / relationship type (Feedback, OperatorAction, Artifact … included)."""
    import kgraph
    return kgraph.kg().stats()


# the built React desktop (frontend/dist) is served at /app  — `./.venv/bin/python scripts/tasks.py ui-build`
_DIST = REPO / "frontend" / "dist"
if _DIST.exists():
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/app", StaticFiles(directory=_DIST, html=True), name="desktop")

    @app.get("/", include_in_schema=False)
    def _root():
        return RedirectResponse("/app/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("OPERATOR_API_HOST", "127.0.0.1"), port=int(os.environ.get("OPERATOR_API_PORT", "8770")), log_level="warning")
