"""
Feedback Store — persists operator feedback in SQLite at data/feedback.db.

Design notes:
- operator_feedback: thumbs-up/thumbs-down ratings on agent responses.
- actual_outcomes: what operationally happened after a recommendation (separate table).
- Feedback is stored for qualitative analysis ONLY.
  It does NOT automatically adjust HCADE recommendation scores.
  Any future use of feedback in scoring must be evaluated manually with sufficient data.
"""
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH: Path | None = None


def init_feedback_db(project_root: str | Path) -> None:
    """Initialize the SQLite database and create tables. Call once at startup."""
    global _DB_PATH
    _DB_PATH = Path(project_root) / "data" / "feedback.db"
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(_DB_PATH)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS operator_feedback (
            feedback_id     TEXT PRIMARY KEY,
            response_id     TEXT NOT NULL,
            query_text      TEXT,
            response_text   TEXT,
            query_type      TEXT,
            rating          INTEGER NOT NULL CHECK(rating IN (-1, 1)),
            feedback_text   TEXT,
            created_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS actual_outcomes (
            outcome_id      TEXT PRIMARY KEY,
            response_id     TEXT NOT NULL,
            outcome_text    TEXT NOT NULL,
            recorded_by     TEXT,
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_response
            ON operator_feedback(response_id);
        CREATE INDEX IF NOT EXISTS idx_outcomes_response
            ON actual_outcomes(response_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_qt
            ON operator_feedback(query_type);
    """)
    con.commit()
    con.close()
    print(f"Feedback DB initialised at {_DB_PATH}")


def _get_db() -> sqlite3.Connection:
    if _DB_PATH is None:
        raise RuntimeError("Feedback DB not initialised. Call init_feedback_db() first.")
    return sqlite3.connect(_DB_PATH)


def store_feedback(
    response_id: str,
    rating: int,              # +1 = helpful, -1 = not helpful
    feedback_text: str = "",
    query_text: str = "",
    response_text: str = "",
    query_type: str = "",
) -> str:
    """
    Store operator feedback for a given response.
    Returns the new feedback_id (UUID).

    NOTE: rating is stored for qualitative analysis only.
    It does NOT automatically adjust HCADE recommendation scores.
    """
    if rating not in (-1, 1):
        raise ValueError("rating must be +1 (helpful) or -1 (not helpful)")

    feedback_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    con = _get_db()
    try:
        con.execute(
            "INSERT INTO operator_feedback VALUES (?,?,?,?,?,?,?,?)",
            (feedback_id, response_id, query_text, response_text,
             query_type, rating, feedback_text, now)
        )
        con.commit()
    finally:
        con.close()
    return feedback_id


def store_actual_outcome(
    response_id: str,
    outcome_text: str,
    recorded_by: str = "",
) -> str:
    """
    Record what operationally happened after a recommendation was given.
    This is intentionally SEPARATE from operator ratings (thumbs up/down).
    Returns the new outcome_id (UUID).
    """
    outcome_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    con = _get_db()
    try:
        con.execute(
            "INSERT INTO actual_outcomes VALUES (?,?,?,?,?)",
            (outcome_id, response_id, outcome_text, recorded_by, now)
        )
        con.commit()
    finally:
        con.close()
    return outcome_id


def get_feedback_summary() -> dict:
    """
    Return aggregate feedback statistics for analysis purposes.
    Does NOT modify any HCADE scoring logic.
    """
    if _DB_PATH is None or not _DB_PATH.exists():
        return {"error": "Feedback DB not initialised", "total_feedback": 0}

    con = _get_db()
    try:
        cur = con.cursor()
        cur.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN rating=1  THEN 1 ELSE 0 END),
                   SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END)
            FROM operator_feedback
        """)
        total, positive, negative = cur.fetchone()

        cur.execute("""
            SELECT query_type, COUNT(*), AVG(rating)
            FROM operator_feedback
            GROUP BY query_type
        """)
        by_type = [
            {"query_type": r[0], "count": r[1], "avg_rating": round(r[2], 3)}
            for r in cur.fetchall()
        ]

        cur.execute("SELECT COUNT(*) FROM actual_outcomes")
        outcome_count = cur.fetchone()[0]

        return {
            "total_feedback": total or 0,
            "positive": positive or 0,
            "negative": negative or 0,
            "by_query_type": by_type,
            "actual_outcomes_recorded": outcome_count or 0,
            "note": (
                "Feedback is stored for qualitative analysis only. "
                "It does not automatically adjust HCADE recommendation scores."
            ),
        }
    finally:
        con.close()


def get_feedback_for_response(response_id: str) -> list:
    """Return all feedback records for a specific response_id."""
    if _DB_PATH is None or not _DB_PATH.exists():
        return []
    con = _get_db()
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT * FROM operator_feedback WHERE response_id=?",
            (response_id,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()
