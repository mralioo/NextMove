"""Write the 3 evaluation questions (with approximate reference answers) into the ADK UI's eval set `eval_set_1`.

    make adk-evalset            # then open the ADK UI (make up / make agent-web) -> Evals -> eval_set_1 -> run

The same questions and reference answers are documented in docs/test_questions.md ("ADK eval set"). The reference answers are APPROXIMATE on purpose:
ADK compares the agent's final response with them (response match score) and, because our tools run behind MCP inside the pipeline rather than as ADK
function calls, no tool trajectory is expected. Numbers were taken from the raw data / ground truth (knowledge base entries GT-CL-*, GT-D-RUDOW, GT-A-UBER).
Re-running replaces the three cases and leaves other cases of the set alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "agent"))

SET_ID, APP = "eval_set_1", "agent"

CASES = [
    ("c1_u6_closure",
     "Line U6 is suspended on a section between Hallesches Tor and Kaiserin-Augusta-Strasse stations. What is the reason behind this closure and how long will it last? "
     "How should the passengers be rerouted, which stations would become overloaded, and where should additional staff be deployed?",
     "The U6 section between Hallesches Tor and Kaiserin-Augusta-Str. is closed for a safety inspection on 13 July 2026 from 13:50 to 15:20, i.e. 1.5 hours. "
     "There is no rail detour, so a replacement bus is needed. Kaiserin-Augusta-Str. is the most pressured station (about a 78% chance of exceeding its own busiest-5% level "
     "versus 17% normally), followed by Mehringdamm (about 29% vs 10%); deploy additional staff there. The pressure figures are assumption-based estimates "
     "(the share of passengers who divert is assumed), not capacity measurements."),
    ("d1_rudow_peak",
     "At what time does the commute flow peak at Rudow station usually take place? Does it exceed the mean commute peak value across all stations?",
     "Rudow's weekday commute peak is at 18:00 with about 219 passengers per 15 minutes. That is below the network mean weekday peak of about 264 passengers "
     "(roughly 17% lower), so it does not exceed the mean."),
    ("a1_arena_concert",
     "There is a sold-out concert at Mercedes-Benz Arena tonight at 21:00. What does the flow look like at Hermannplatz, and what should we do at 23:15 when it ends?",
     "Hermannplatz shows no measurable uplift from events at this venue. In the data the arena is called Uber Arena and is assumed to be the same venue. The stations that feel "
     "concerts are Warschauer Str. (about +114 passengers per 15 minutes, roughly 6 times normal) and Schlesisches Tor (about +93, roughly 5 times normal). At 23:15, when it ends, "
     "put additional staff at those two stations until about 00:15. 'Tonight' has no calendar date in the data, so the numbers are the pattern of the venue's past events; "
     "the venue-to-station link is inferred from flows, and there is no capacity data."),
]


def main() -> None:
    from google.adk.evaluation.eval_case import EvalCase, Invocation, SessionInput
    from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
    from google.genai import types

    mgr = LocalEvalSetsManager(agents_dir=str(REPO))
    try:
        mgr.get_eval_set(APP, SET_ID)
    except Exception:
        mgr.create_eval_set(APP, SET_ID)
    existing = {c.eval_id for c in mgr.get_eval_set(APP, SET_ID).eval_cases}
    for eid, question, reference in CASES:
        if eid in existing:
            mgr.delete_eval_case(APP, SET_ID, eid)
        inv = Invocation(invocation_id=f"{eid}_inv", user_content=types.Content(role="user", parts=[types.Part(text=question)]),
                         final_response=types.Content(role="model", parts=[types.Part(text=reference)]))
        mgr.add_eval_case(APP, SET_ID, EvalCase(eval_id=eid, conversation=[inv], session_input=SessionInput(app_name=APP, user_id="user", state={})))
    es = mgr.get_eval_set(APP, SET_ID)
    print(f"{SET_ID}: {len(es.eval_cases)} cases -> {[c.eval_id for c in es.eval_cases]}")


if __name__ == "__main__":
    main()
