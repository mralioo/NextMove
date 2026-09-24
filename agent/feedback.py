"""OPERATOR FEEDBACK LOOP.

Two kinds of feedback close the loop between an answer and what happened afterwards:

  score   the operator rates the response (1 = useless / wrong … 5 = exactly what I needed), optional comment
  action  what the operator actually DID about the situation the question was about (free text, e.g. "sent two staff to Neukölln, put a bus on U7") — with whether the
          advice was followed (`as_recommended` | `modified` | `different` | `none`) and, later, how it went (`worked` | `partly` | `did_not_work` | `unknown`)

Every submission is (1) stored in the operator knowledge base (`operator_feedback`, and the turn row carries the latest mean score — a badly rated answer (<= 2) is no
longer reused from history), (2) written to the knowledge graph (Feedback / OperatorAction nodes linked to the Problem, the Artifact, the Answer, the stations, the lines and
the recommended Actions), and (3) summarised into the memory agent (Cognee) when it is the configured memory. The next time a SIMILAR situation comes up, `precedents()` reads it
back: the writer adds one deterministic "precedent" line ("last time in a similar case operators … — rated 5/5, worked") or a caution when the outcome was bad.

Nothing here changes the numbers of an answer; precedents are context for the operator, labelled as such.
"""
from __future__ import annotations

import re
import threading

FOLLOWED = ("as_recommended", "modified", "different", "none")
OUTCOMES = ("worked", "partly", "did_not_work", "unknown")
KINDS = ("score", "action", "both")


class FeedbackError(ValueError):
    """A submission that cannot be accepted (unknown turn, score out of range, empty action, unknown value)."""


def _kb():
    from knowledge import kb
    return kb()


def _kg():
    from kgraph import kg
    return kg()


def _clean_score(score) -> int | None:
    if score is None or score == "":
        return None
    if isinstance(score, str):
        m = {"up": 5, "thumbs_up": 5, "good": 5, "down": 1, "thumbs_down": 1, "bad": 1}.get(score.strip().lower())
        if m is None and not score.strip().isdigit():
            raise FeedbackError(f"score must be 1-5 (or up/down), got {score!r}")
        score = m if m is not None else int(score)
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise FeedbackError(f"score must be an integer 1-5, got {score!r}")
    return score


def _extract(action_text: str) -> tuple[list[str], list[str]]:
    """Stations and lines named in the operator's own words (same matcher the router uses)."""
    try:
        import router
        stations = router.find_stations(action_text)
    except Exception:
        stations = []
    lines = ["U" + d for d in dict.fromkeys(re.findall(r"\b[uU]\s?([1-9])\b", action_text))]
    return list(stations), lines


def submit(turn_id: int, *, operator_id: str = "", score=None, comment: str | None = None, action_text: str | None = None, followed: str | None = None,
           outcome: str | None = None, occurred_at: str | None = None, kb=None, graph=None, mirror: bool = True) -> dict:
    """Store one feedback record and feed the memory + the knowledge graph. Returns the stored record (with `feedback_id`) and what was fed where."""
    kb = kb or _kb()
    info = kb.turn_info(int(turn_id))
    if info is None:
        raise FeedbackError(f"unknown turn {turn_id}")
    score = _clean_score(score)
    action_text = (action_text or "").strip() or None
    if score is None and action_text is None:
        raise FeedbackError("send a score, an action, or both")
    if followed is not None and followed not in FOLLOWED:
        raise FeedbackError(f"followed must be one of {FOLLOWED}")
    if outcome is not None and outcome not in OUTCOMES:
        raise FeedbackError(f"outcome must be one of {OUTCOMES}")
    if action_text and len(action_text) > 1500:
        raise FeedbackError("action text is limited to 1500 characters")
    stations, lines = _extract(action_text) if action_text else ([], [])
    kind = "both" if score is not None and action_text else ("action" if action_text else "score")
    fid = kb.add_feedback(int(turn_id), session_id=info["session_id"], user_id=operator_id or info["operator_id"], kind=kind, score=score, comment=(comment or "").strip() or None,
                          action_text=action_text, followed=followed, outcome=outcome, occurred_at=occurred_at, stations=stations, lines=lines, category=info["category"],
                          problem_key=info.get("problem_key"))
    rec = kb.get_feedback(fid)
    fed = {"knowledge_base": True, "knowledge_graph": False, "memory_agent": False}
    art = kb.get_artifact(int(turn_id))
    try:
        (graph or _kg()).record_feedback(rec, art, question=info["question"], category=info["category"] or "")
        fed["knowledge_graph"] = True
    except Exception as e:                                       # the record is safe in the knowledge base; the graph can be rebuilt with sync_graph()
        fed["knowledge_graph_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    if mirror:
        fed["memory_agent"] = _mirror_to_memory(rec, info)
    return {**rec, "fed": fed}


def _mirror_to_memory(rec: dict, info: dict) -> bool:
    """One entry in the memory agent (Cognee) — only when it is the configured memory (an upload to a third party, like the answers themselves)."""
    try:
        from config import CONFIG
        from knowledge import cognee
        if CONFIG.memory != "cognee" or not cognee().available:
            return False
        text = (f"Operator feedback on: {info['question'][:200]}\nScore: {rec.get('score')}/5. " + (f"Operator did: {rec['action_text']}. " if rec.get("action_text") else "")
                + (f"Followed advice: {rec.get('followed')}. " if rec.get("followed") else "") + (f"Outcome: {rec.get('outcome')}. " if rec.get("outcome") else "") + (rec.get("comment") or ""))
        threading.Thread(target=cognee().remember_qa, args=(rec["session_id"], f"[operator feedback] {info['question'][:200]}", text[:1500], f"category {info['category']}"), daemon=True,
                         name="cognee-feedback").start()
        return True
    except Exception:
        return False


def update(feedback_id: int, *, kb=None, graph=None, **fields) -> dict:
    """Change a record later (typically `outcome` once it is known, or a corrected action). The graph node is refreshed."""
    kb = kb or _kb()
    if fields.get("score") is not None:
        fields["score"] = _clean_score(fields["score"])
    if fields.get("followed") is not None and fields["followed"] not in FOLLOWED:
        raise FeedbackError(f"followed must be one of {FOLLOWED}")
    if fields.get("outcome") is not None and fields["outcome"] not in OUTCOMES:
        raise FeedbackError(f"outcome must be one of {OUTCOMES}")
    rec = kb.update_feedback(feedback_id, **fields)
    if rec is None:
        raise FeedbackError(f"unknown feedback {feedback_id}")
    info = kb.turn_info(rec["turn_id"]) or {}
    try:
        (graph or _kg()).record_feedback(rec, kb.get_artifact(rec["turn_id"]), question=info.get("question", ""), category=info.get("category") or "")
    except Exception:
        pass
    return rec


def delete(feedback_id: int, *, kb=None, graph=None) -> bool:
    """Withdraw a record: it disappears from the knowledge base views and from the scores; the graph node is marked withdrawn (kept for the audit trail, excluded from precedents)."""
    kb = kb or _kb()
    rec = kb.get_feedback(feedback_id)
    if rec is None or not kb.delete_feedback(feedback_id):
        return False
    try:
        g = graph or _kg()
        g.node("Feedback", f"fb{feedback_id}", withdrawn=True, score=None, action=None, comment=None)
    except Exception:
        pass
    return True


def sync_graph(kb=None, graph=None) -> int:
    """Re-feed the graph from the knowledge base records (after a graph reset / for records made while the graph was unavailable)."""
    kb = kb or _kb()
    n = 0
    from knowledge import KnowledgeBase  # noqa: F401
    with kb._conn() as c:
        ids = [r[0] for r in c.execute("SELECT id FROM operator_feedback WHERE deleted=0 ORDER BY id")]
    for fid in ids:
        rec = kb.get_feedback(fid)
        info = kb.turn_info(rec["turn_id"]) or {}
        (graph or _kg()).record_feedback(rec, kb.get_artifact(rec["turn_id"]), question=info.get("question", ""), category=info.get("category") or "")
        n += 1
    return n


# ------------------------------------------------------------------------------------------------ using the feedback in later decisions
def precedents(question: str, category: str = "", entities=None, k: int = 2, kb=None, graph=None, exclude_key: str | None = None) -> list[dict]:
    """What operators did in similar situations, and how it went (from the knowledge graph)."""
    return (graph or _kg()).operator_precedents(question, category, entities, k=k, exclude_key=exclude_key)


def precedent_line(prec: list[dict]) -> str:
    """One deterministic line for the brief, or "". A bad past outcome wins over a good one (the operator must see the caution first). Wording is fixed; the action text is the
    operator's own."""
    def _clip(t: str, n: int = 110) -> str:
        t = t.strip().rstrip(".")
        if len(t) > 1 and t[0].isupper() and not t[1].isupper():          # "Sent two staff…" -> "sent two staff…" (it continues a sentence); "U7 …" stays
            t = t[0].lower() + t[1:]
        return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"
    bad = good = None
    for p in prec:
        for a in p.get("actions") or []:
            is_bad = a.get("outcome") == "did_not_work" or (a.get("score") is not None and a["score"] <= 2)
            is_good = a.get("outcome") in ("worked",) or (a.get("score") is not None and a["score"] >= 4 and a.get("outcome") != "did_not_work")
            if is_bad and bad is None:
                bad = (p, a)
            if is_good and good is None:
                good = (p, a)
    if bad:
        p, a = bad
        return f"_Caution: in a similar case operators {_clip(a['action'])} — it did not work out ({'rated ' + str(a['score']) + '/5' if a.get('score') is not None else 'reported as not working'})._"
    if good:
        p, a = good
        tail = ", ".join(x for x in (f"rated {a['score']}/5" if a.get("score") is not None else "", "worked" if a.get("outcome") == "worked" else "") if x)
        return f"_Precedent: in a similar case operators {_clip(a['action'])}" + (f" ({tail})" if tail else "") + "._"
    return ""


def recommended_actions(brief: str, facts: dict) -> list[str]:
    """The actions the answer recommends (the brief's 'Do now' bullets, else the ones implied by the facts) — what an operator's action report is compared with."""
    acts = []
    m = re.search(r"\*\*Do now:\*\*(.*?)(?=\n\*\*|\n_|\Z)", brief or "", re.S)
    if m:
        acts = [x.strip(" -•*") for x in m.group(1).strip().splitlines() if x.strip(" -•*")]
    if not acts:
        try:
            from kgraph import actions_from_facts
            acts = actions_from_facts(facts)[0]
        except Exception:
            acts = []
    return acts[:5]
