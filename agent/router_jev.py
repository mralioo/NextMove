"""JEV classification API adapter — OPT-IN, off by default.

Implements the request/response format documented at
https://huggingface.co/blog/sora-2/how-to-use-the-jev-ai-model-a-step-by-step-develop :

    POST https://thejevai.com/v1/systemone      Authorization: Bearer $JEV_API_KEY
    {"model": "jev-latest", "state": {...}, "questions": {"category": {"type": "choice", "instructions": "...", "criteria": {...}}}}
    -> {"answers": {"category": {"choice": "C", "confidence": 0.92, ...}}, "usage": {...}}

Caveats, stated plainly:
  * The post has no model card, weights or source repository, and no team member has verified the service; treat it
    as an unverified third-party API.
  * Using it SENDS THE OPERATOR'S QUESTION to that service. It therefore runs only when JEV_API_KEY is set in the
    environment (never by default), and the experiment suite records the arm as "skipped" when it is not.
  * The adapter follows the documented format but has only been tested against a mocked HTTP response (no key).
"""
from __future__ import annotations

import json
import os
import urllib.request

URL = os.environ.get("JEV_API_URL", "https://thejevai.com/v1/systemone")
CATEGORIES = {
    "A": "impact of an event/concert on passenger flow at nearby stations",
    "B": "anomalies or peaks in passenger flow and their root cause (weather, events, closures)",
    "C": "a line section or station is closed/suspended: reason, reroute, overloaded stations, staff",
    "D": "one station's flow profile: peak hour, weekday/weekend, comparison with the network mean, prediction at a time",
    "E": "energy consumption / efficiency per passenger of metro lines",
    "F": "network resilience: which stations' closure fragments the network",
    "G": "correlated demand between stations without a direct connection",
    "H": "how passengers actually reroute during disruptions versus the shortest path",
    "X": "investment recommendation or InnoTrans/Messe surge routing",
    "OOS": "cannot be answered from flow/closure/weather/event/energy data (capacity, delays, costs, forecasts, off-topic)",
}


class NotConfigured(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.environ.get("JEV_API_KEY"))


def build_request(question: str, has_history: bool) -> dict:
    return {"model": "jev-latest",
            "state": {"question": question, "has_previous_answer_in_session": has_history},
            "questions": {"category": {"type": "choice",
                                       "instructions": "Pick the single category that best describes what the metro operator is asking.",
                                       "criteria": CATEGORIES}}}


def classify(question: str, has_history: bool = False, timeout: float = 10.0) -> tuple[str, float]:
    if not configured():
        raise NotConfigured("JEV_API_KEY is not set: the JEV router is opt-in (it sends the question to a third-party API)")
    req = urllib.request.Request(URL, data=json.dumps(build_request(question, has_history)).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['JEV_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    try:
        ans = body["answers"]["category"]
        return str(ans["choice"]), float(ans.get("confidence", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"unexpected JEV response shape ({type(e).__name__}); body starts: {json.dumps(body)[:300]}") from e


def _check() -> int:
    """`make jev-check`: send ONE public training question to the API and print what came back. Use it once after
    setting JEV_API_KEY to confirm the key and the response format before running the experiment arm."""
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    if not configured():
        print("JEV_API_KEY is not set. Add it to your .env file (JEV_API_KEY=...) — optionally JEV_API_URL if the endpoint differs.\n"
              "Note: running this sends the question text below to a third-party service.")
        return 1
    q = " ".join(sys.argv[1:]) or "Line U6 is suspended between Hallesches Tor and Kaiserin-Augusta-Strasse. Why, and how should passengers be rerouted?"
    print(f"POST {URL}\nquestion: {q}")
    t = time.time()
    try:
        cat, conf = classify(q)
    except Exception as e:                       # HTTPError, timeout, bad shape: print the reason, do not crash
        body = getattr(e, "read", lambda: b"")()[:300]
        print(f"FAILED after {time.time() - t:.1f}s: {type(e).__name__}: {e} {body!r}")
        return 2
    print(f"OK in {time.time() - t:.2f}s -> category {cat} (confidence {conf:.2f}); expected C")
    return 0


if __name__ == "__main__":
    raise SystemExit(_check())
