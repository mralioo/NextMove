# Conversation threads, chat history and a categorised knowledge graph

**Date:** 2026-09-24 · **Branch:** `agentic-v3-supervisor-evaluator` · **Tests:** `tests/test_operator_feedback.py` (118 tests pass) · **Live check:** headless Chrome + API flow against scratch databases.

## 1. The problem

The agent answers a question *in the context of the previous answer* (follow-ups: "why?", "what about 22:30?", "and Neukölln?"). That is what makes it fast to use — and what makes it unsafe when an **unrelated question is typed into the same conversation**:

* a short question can inherit the wrong line, stations or date from the earlier situation (a "rerun" merged into the wrong context);
* the graph gets a **wrong branch**: the second question and the operator's feedback get attached to the first situation (or become a stray problem with no context);
* a new browser tab / new session used to silently inherit the operator's *last* turn from the knowledge base ("history survives restarts"), so even a "new" conversation was not clean.

## 2. The solution: one conversation = one situation, and the operator decides when it is unclear

| Piece | What it does |
| --- | --- |
| **Relation check** — `agent/threads.py::relation()` | Deterministic (the router's rules, ≈ 2 ms, no LLM). Compares a new message with the conversation's *situation* (its latest answered question: category, line, stations, event). **related** → refers to the last answer ("why?", "evidence"), changes only time / date / duration, or names the same line / place / event, or is off-topic (bounced on its own, situation untouched). **unrelated** → a complete question of its own with another category or other stations / lines. **unsure** → short and vague, not tied to the situation ("And Rudow?"). |
| **One question to the operator** — `POST /api/v1/chat`, `context_mode` | `auto` (default): *related* → answered at once, no interruption; *unrelated / unsure* → **the message is not sent to the agent**; the API returns `needs_choice` with the reason, the topic so far and two choices: **New conversation** (recommended for a different topic) or **Continue this topic**. `continue` connects it anyway; `new` starts a fresh session with no inherited context. |
| **Clean new sessions** — `agent/fast_agent.py` | A new session no longer inherits the operator's last turn (`TMT_RESTORE_LAST=on` brings the old behaviour back). A conversation is continued only **explicitly**: the session is created with `link_turn_id`, and the supervisor loads *that* turn's plan, facts and artifact (and its situation) — nothing else. |
| **Chat history** — `GET /api/v1/conversations`, `/conversations/{id}` | One entry per conversation, titled by its first question, with category, number of messages, time, mean score. A conversation resumed from the history stays **one entry** (sessions linked by `linked_from` are grouped). Answers reused from history are stored as messages of the conversation too, so they show up. |
| **UI** (`ChatPanel.jsx`) | **History** drawer (open an old conversation → its messages come back with rating / action controls; the next message continues it, connected to its last answer); a **"Topic"** pill above the composer shows what the conversation is about, with **Start a new topic**; the choice card (amber) appears only when needed; after "Continue" a small note says the message was connected; **New** starts a clean page. |

## 3. The knowledge graph: categorised, and follow-ups no longer branch

**Structure added** (`agent/kgraph.py`):

```
Domain (Events · Diagnostics · Disruptions · Stations · Energy · Network · Operations planning · Strategy)
  ▲ IN_DOMAIN
Category (A–H, P, X, with a human title)  ◀── IN_CATEGORY ── Situation ◀── PART_OF ── Problem ── VARIANT_OF ──▶ Problem (parent)
                                                        ▲ DISCUSSED                         │ RECEIVED_FEEDBACK
                                                 Conversation (session)              Feedback ── DESCRIBES ──▶ OperatorAction ── OF_TYPE ──▶ ActionType
                                                                                           └ IN_CATEGORY / ABOUT_SITUATION                      (staff_deployment · replacement_bus ·
                                                                                                                                                 passenger_information · rerouting · crowd_control ·
                                                                                                                                                 monitoring · service_change · coordination · other)
```

* **Situation** = the topic a conversation is about (its first accepted question). A **variant** ("what about 22:30?", a rerun with changed parameters) is a `Problem` of kind `variant`, linked `VARIANT_OF` its parent and `PART_OF` the **same Situation** — it is no longer a second, unrelated branch. A question of its own opens a new Situation.
* **Categories** carry a title and a **Domain**; the recommended `Action`s and the operators' `OperatorAction`s carry an **ActionType** (deterministic keyword rules, `kgraph.action_types`), so "which kinds of action did operators use for closures, and how did they go" is a query, and a precedent can be compared with what the answer recommended.
* **Feedback** (score, action report, outcome) attaches to the *situation's* Problem (the id travels with the answer), is placed in its Category and linked to its Situation. Feedback for an answer whose situation was never recorded (a decline, an answer that failed the sanity check) becomes a `feedback_only` Problem: kept for the audit trail but **excluded from similarity search and precedents** — it can no longer become a wrong branch.
* **Follow-up-shaped problems** (≤ 6 words, no entities, runtime) are marked `followup` by the audit and excluded from matching.
* **Precedents** already needed a shared line or station; they now also run on categorised, situation-level data.

**Housekeeping** — `kg-audit` (`./.venv/bin/python scripts/tasks.py kg-audit [--fix]`, also `GET /api/v1/graph/audit`, `POST /api/v1/graph/repair`, `GET /api/v1/graph/taxonomy`): finds Problems without kind / situation / category, follow-up-shaped Problems, untyped actions, orphan feedback; `--fix` repairs without deleting anything. **Run on the production graph today:** 109 problems, 297 issues (all legacy: no situation / kind / action types) → 0 after the repair; taxonomy now shows situations per category (e.g. Closure response 41, Station profile 9, Events 8 …); Neo4j re-synced.

## 4. Verified

* API flow (scratch databases): a U8 closure question → a *"what about 5 hours instead?"* rerun in the same conversation was answered without interruption and stored as a `variant` (`VARIANT_OF 1`, one Situation); an Adenauerplatz question in the same conversation returned `needs_choice` (unrelated, recommended *new*) and **was not sent to the agent**; choosing *new* answered it in a fresh session; an action report on the closure became `OperatorAction` typed `staff_deployment`, `passenger_information`, `replacement_bus`; taxonomy and stats consistent; audit clean.
* Browser (headless Chrome): closure answer → unrelated Rudow question → amber choice card → *New conversation* → fresh page with the new topic in the pill → *History* lists both conversations with category chips → opening the closure conversation and typing "What about if it lasts 3 hours instead?" continued it (no question, note "continuing …").

## 5. Limits

* The relation check is rule-based: it can call a paraphrase *unsure* (one extra click) and a message that reuses a station of the old situation *related* although the operator has changed their mind — the "Topic" pill and **Start a new topic** are the manual override.
* **Continue** on an unrelated question is allowed on purpose: the agent still treats a complete new question as a fresh one (its own standalone check), but a short message will inherit the old context — that is what the operator chose.
* The chat history reads the knowledge base: replies that were answered without a stored turn (bounces, off-topic) do not appear; conversations from before this change appear as they were stored (one per session).
* Resuming a conversation creates a **new ADK session linked to the last answer** (ADK sessions are in memory and disappear on restart); the history shows it as one conversation.
* ActionType is keyword-based (English); an unusual wording lands in `other`. Domains and category titles are a fixed table in `kgraph.CATEGORY_INFO`.
* The audit's "follow-up-shaped" rule is conservative (≤ 6 words, no entities, runtime source); it will not catch a follow-up that repeats a station name.
