"""Stage 1 of the fast pipeline: ROUTER.

Turns an operator question into a small JSON *plan* — category, and the entities the tools need
(lines, exact station names, dates, time, duration, what-if flag) — in milliseconds, with no LLM call.
An LLM (small, JSON-only, ~150 output tokens) is used only as a fallback when the deterministic
router is not confident. The plan is what gets forwarded to the executor: it names the category
(hence which tools/datasets to call) and pre-resolves the entities, so no downstream agent has to
re-read the prose question.

Plan schema (short keys on purpose — this JSON is what agents pass to each other):
    {"cat": "C", "conf": 0.9, "tier": 0,
     "lines": ["U6"], "stations": ["U Hallesches Tor (Berlin)", ...],   # exact station_name strings
     "data": ["closures", "flows", ...], "tools": ["resolve_closure", ...],   # what the executor will touch
     "raw": [],                                                          # unresolved place mentions (LLM tier)
     "dates": ["2026-07-13"], "time": "13:50", "dur_min": 90, "what_if": false, "follow": false}

Categories: A event impact · B anomalies · C disruption response · D station profile · E energy ·
F network resilience · G correlation · H reroute behaviour · X bonus (investment / InnoTrans) ·
OOS out of scope / unanswerable from data.
"""
from __future__ import annotations

import difflib
import glob
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

YEAR = 2026
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS.update({"januar": 1, "februar": 2, "märz": 3, "mai": 5, "juni": 6, "juli": 7, "oktober": 10, "dezember": 12})

# ---- category rules: (regex, weight). Highest total wins; ties broken by PRIORITY order. ----
RULES: dict[str, list[tuple[str, float]]] = {
    "F": [(r"fragment", 3), (r"single point(s)? of failure|articulation", 3), (r"resilien", 2.5),
          (r"most critical (station|node)|weakest link|isolate the most|cut off from the rest", 2),
          (r"rank .*stations? whose", 2), (r"how many stations would be cut", 2.5)],
    "H": [(r"deviation|significant", 1.5), (r"neighbou?ring stations?", 1.5), (r"actually (prefer|use|take|choose)", 3), (r"theoretical(ly)? shortest|shortest (route|path)s?", 2.5),
          (r"passenger behaviou?r|what does this reveal", 2.5), (r"walk(ed)? instead", 2.5),
          (r"avoid transfers?|prefer transfers?", 2.5), (r"alternative routes? .*(passengers|prefer)", 2.5),
          (r"extra passengers .*(neighbou?ring|u3|u\d) stations|did .* stations? (gain|carry)", 2.5)],
    "G": [(r"correlat", 3), (r"depend(ent|ency|s)? (strongly )?on another", 3), (r"despite no direct connection", 3),
          (r"tightly coupled|move together|lag\b|lead(s)? another", 2.5), (r"causes? the crowding", 2)],
    "E": [(r"energy", 3), (r"efficien", 2), (r"\bratio\b", 2), (r"\bmwh\b|kwh|rolling stock", 2), (r"per passenger", 1)],
    "B": [(r"anomal", 3), (r"root cause", 2.5), (r"caused by (bad )?(weather|heat|wind|rain)", 3),
          (r"peak caused", 3), (r"not explained|unexplained|cannot be explained", 2.5),
          (r"exactly 500|spike", 3), (r"signalling failure", 1.5)],
    "X": [(r"invest", 3), (r"infrastructure improvement", 3), (r"innotrans", 3), (r"messe", 2.5),
          (r"unconventional", 3), (r"not based on the shortest", 3)],
    "A": [(r"concert|festival|gig|match\b", 2), (r"\barena\b|olympiastadion|tempodrom|stadium|venue", 2.5),
          (r"\bevents?\b", 1.5), (r"guns n|doja cat|dikka|alligatoah|tour\b", 2), (r"ten events|events across", 2)],
    "C": [(r"what happened (on|to|at) the u\d", 2), (r"suspend|suspension", 2.5), (r"\bclos(ed|ure|ures|e)\b", 1.5), (r"disrupt", 1.5),
          (r"re-?rout", 2), (r"overload", 2), (r"deploy(ed)? .*staff|additional staff|staff", 1.2),
          (r"replacement bus|what if .*(stop|suspend|clos)", 2), (r"sperrung|gesperrt", 2.5),
          (r"closed on|was closed|is closed", 1)],
    "D": [(r"\bpeaks?\b|commute", 1.5), (r"usually|typical|rhythm|profile", 1.5), (r"busiest|quietest|quiet(est)? stations?", 2),
          (r"what time|when (does|is)", 1.2), (r"mean (commute )?peak|network mean|exceed", 1.5),
          (r"flow at .*(at|on) \d|predict", 1.5), (r"what was the flow|how (busy|crowded) is", 2), (r"weekday|weekend", 1)],
}
PRIORITY = ["F", "H", "G", "E", "B", "X", "A", "C", "D"]

# What each category needs — put in the plan so the executor (and anyone reading a trace) sees which
# datasets and MCP tools the question will touch. `tools` lists what is wired up today ([] = not yet).
CAT_DATA = {
    "A": {"data": ["events", "stations", "flows", "weather"], "tools": []},
    "B": {"data": ["flows", "weather", "events", "closures"], "tools": []},
    "C": {"data": ["closures", "connections", "stations", "flows", "weather", "events"],
          "tools": ["resolve_closure", "apply_closure", "alternate_paths", "scenario_flow"]},
    "D": {"data": ["flows"], "tools": ["resolve_station", "station_profile", "predict_expected_flow", "predict_overcrowding_risk"]},
    "E": {"data": ["energy", "flows", "stations"], "tools": []},
    "F": {"data": ["connections", "flows"], "tools": []},
    "G": {"data": ["flows", "connections", "stations"], "tools": []},
    "H": {"data": ["closures", "connections", "flows"], "tools": []},
    "X": {"data": ["flows", "connections", "energy", "closures"], "tools": []},
    "OOS": {"data": [], "tools": []}, "FOLLOW": {"data": [], "tools": []},
}
OOS = re.compile(r"capacity|safely hold|delayed trains?|cost of|how much would .* cost|currywurst|restaurant|"
                 r"ignore (your|all|previous)|next (monday|week|month)|tomorrow|tonight|\bu4\b|signalling failure", re.I)
FOLLOW = re.compile(r"^\s*(and|so|what about|how about)\b|\b(those|that|these|them|the top|the first)\b|"
                    r"how confident|which of (those|these)|measured and|explain (that|why)|what data would", re.I)

ALIASES = {"zoo": "zoologischer garten", "kotti": "kottbusser tor", "alex": "alexanderplatz",
           "hauptbahnhof": "berlin hauptbahnhof", "hbf": "berlin hauptbahnhof"}


# ---------------------------------------------------------------- station index
def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(berlin\)", " ", s)
    s = re.sub(r"straße|strasse|str\.?(?=\W|$)", "str", s)
    s = re.sub(r"^(s\+u|u|s)\s+", "", s.strip())
    s = re.sub(r"\b(bhf|bahnhof)\b", " ", s) if "hauptbahnhof" not in s else s
    return re.sub(r"[\s]+", " ", s.replace(".", " ")).strip()


@lru_cache(maxsize=1)
def station_index() -> dict[str, list[str]]:
    """normalised name -> exact station_name(s); read straight from the CSV (no streamlit import)."""
    base = os.environ.get("DATA_DIR", "data")
    repo = Path(__file__).resolve().parent.parent
    hits = glob.glob(str(Path(base) / "**" / "stations_with_ubahn.csv"), recursive=True) or \
        glob.glob(str(repo / "data" / "**" / "stations_with_ubahn.csv"), recursive=True)
    if not hits:
        return {}
    names = sorted(pd.read_csv(hits[0])["station_name"].unique())
    idx: dict[str, list[str]] = {}
    for n in names:
        idx.setdefault(_norm(n), []).append(n)
    for alias, target in ALIASES.items():
        if target in idx:
            idx.setdefault(alias, idx[target])
    return idx


def find_stations(text: str) -> list[str]:
    q = " " + _norm(text) + " "
    idx = station_index()
    found: list[tuple[int, str]] = []
    for norm_name in sorted(idx, key=len, reverse=True):          # longest first: "rathaus neukölln" before "neukölln"
        if len(norm_name) < 3:
            continue
        m = re.search(rf"(?<![\w-]){re.escape(norm_name)}(?![\w-])", q)
        if m:
            found.append((m.start(), norm_name))
            q = q[:m.start()] + " " * (m.end() - m.start()) + q[m.end():]   # blank the span
    out: list[str] = []
    for _, n in sorted(found):
        for exact in idx[n]:
            if exact not in out:
                out.append(exact)
    return out


STOP = set("when what time does the peak peaks usually commute station line between and at on in for how why which "
           "flow busiest busy quiet quietest closed closure suspended reason long will last stations network mean".split())


def fuzzy_stations(text: str) -> list[str]:
    """Typo-tolerant fallback ('Rudov' -> Rudow): close n-gram matches against station names."""
    idx = station_index()
    keys = list(idx)
    words = re.findall(r"[\wäöüß-]+", _norm(text))
    out: list[str] = []
    for n in (2, 1):
        for i in range(len(words) - n + 1):
            g = " ".join(words[i:i + n])
            if len(g) < 4 or g in STOP or any(w in STOP for w in g.split()):
                continue
            m = difflib.get_close_matches(g, keys, n=1, cutoff=0.78 if len(g) >= 5 else 0.9)
            if m:
                for exact in idx[m[0]]:
                    if exact not in out:
                        out.append(exact)
    return out


def find_month(text: str) -> str | None:
    """A month mentioned without a day ('in December') -> 'YYYY-MM'."""
    mon = "|".join(sorted(MONTHS, key=len, reverse=True))
    m = re.search(rf"\b(?:in|during|im)\s+({mon})\b(?!\.?\s*\d)", text.lower())
    return f"{YEAR}-{MONTHS[m.group(1)]:02d}" if m else None


# ---------------------------------------------------------------- dates / times / durations
def _iso(month: int, day: int) -> str:
    return f"{YEAR}-{month:02d}-{day:02d}"


def find_dates(text: str) -> list[str]:
    t = text.lower()
    out: list[str] = []
    out += re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    mon = "|".join(sorted(MONTHS, key=len, reverse=True))
    for m in re.finditer(rf"\b({mon})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*[-–]\s*(\d{{1,2}}))?", t):
        month, d1, d2 = MONTHS[m.group(1)], int(m.group(2)), m.group(3)
        out.append(_iso(month, d1))
        if d2:
            out.append(_iso(month, int(d2)))
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({mon})\b", t):
        out.append(_iso(MONTHS[m.group(2)], int(m.group(1))))
    for m in re.finditer(r"\b(\d{1,2})\.(\d{1,2})\.(?!\d)", t):
        out.append(_iso(int(m.group(2)), int(m.group(1))))
    return list(dict.fromkeys(out))


def find_time(text: str) -> str | None:
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    m = re.search(r"\bat (\d{1,2})\s?(am|pm)\b", text.lower())
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
        return f"{h:02d}:00"
    m = re.search(r"\b(?:at|um)\s+(\d{1,2})(?:\s?(?:h|uhr|o'clock))\b", text.lower())
    return f"{int(m.group(1)):02d}:00" if m else None


WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "half an": 0.5}


def find_duration_min(text: str) -> int | None:
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", t)
    if m:
        return int(float(m.group(1)) * 60)
    m = re.search(r"(\d+)\s*(minutes?|mins?)\b", t)
    if m:
        return int(m.group(1))
    m = re.search(rf"\b({'|'.join(WORDNUM)})\s+hours?\b", t)
    return int(WORDNUM[m.group(1)] * 60) if m else None


# ---------------------------------------------------------------- routing
def score_categories(q: str) -> dict[str, float]:
    ql = q.lower()
    return {c: sum(w for pat, w in rules if re.search(pat, ql)) for c, rules in RULES.items()}


def route(question: str, has_history: bool = False) -> dict:
    """Deterministic plan for `question` (tier 0). `conf` < ~0.55 means: ask the LLM router."""
    q = question.strip()
    scores = score_categories(q)
    ranked = sorted(scores, key=lambda c: (-scores[c], PRIORITY.index(c)))
    top, second = ranked[0], ranked[1]
    stations = find_stations(q)
    if not stations and top in ("C", "D"):
        stations = fuzzy_stations(q)
    lines = [m.upper().replace(" ", "") for m in re.findall(r"\bU\s?[1-9]\b", q, flags=re.I)]
    follow = bool(has_history and FOLLOW.search(q) and not stations)

    cat, conf = top, 0.0
    if scores[top] > 0:
        margin = scores[top] - scores[second]
        conf = min(1.0, scores[top] / 3.0) * (1.0 if margin >= 1.0 else 0.65)
    if OOS.search(q) and not (top in ("C", "D") and scores[top] >= 3 and not re.search(r"capacity|safely hold", q, re.I)):
        cat, conf = "OOS", 0.9
    if follow:
        cat, conf = "FOLLOW", 0.9
    if cat == "OOS" and re.search(r"\bu4\b", q, re.I):
        cat = "OOS"
    what_if = bool(re.search(r"\bwhat if\b|\bsuppose\b|\bimagine\b|\bwould be suspended\b", q, re.I))
    return {"cat": cat, "conf": round(conf, 2), "tier": 0, **CAT_DATA.get(cat, {}), "lines": list(dict.fromkeys(lines)),
            "stations": stations, "raw": [], "dates": find_dates(q), "month": find_month(q), "time": find_time(q),
            "dur_min": find_duration_min(q), "what_if": what_if, "follow": follow}


ROUTER_SYSTEM = (
    "Route a Berlin U-Bahn operator question. Reply with ONE JSON object, no prose:\n"
    '{"cat":"A|B|C|D|E|F|G|H|X|OOS","lines":["U6"],"places":["as written by user"],'
    '"dates":["YYYY-MM-DD"],"time":"HH:MM|null","dur_min":null,"what_if":false}\n'
    "A=event impact on stations, B=anomaly/root cause, C=line/station closure response (reroute, overload, staff), "
    "D=one station's flow profile/peak/prediction, E=energy per passenger, F=network fragmentation/critical stations, "
    "G=correlated stations, H=how passengers actually reroute, X=investment/InnoTrans/Messe, "
    "OOS=needs data we lack (capacity, delays, costs, forecasts beyond 2026-09-22, off-topic). Year is 2026."
)


async def llm_route(question: str, base_plan: dict) -> dict:
    """Tier 1: small JSON-only LLM call, used only when the deterministic router is unsure."""
    import litellm

    from llm_config import litellm_params, sampling_params

    cfg = litellm_params("ROUTER")
    if cfg is None:
        return base_plan
    model, kw = cfg
    resp = await litellm.acompletion(
        model=model, messages=[{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": question}],
        max_tokens=160, **sampling_params(model, 0), response_format={"type": "json_object"}, timeout=12, **kw)
    try:
        j = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return base_plan
    plan = dict(base_plan)
    plan.update(tier=1, cat=j.get("cat", plan["cat"]) if j.get("cat") in {*"ABCDEFGHX", "OOS"} else plan["cat"],
                conf=0.8, raw=[p for p in j.get("places", []) if isinstance(p, str)])
    plan["lines"] = plan["lines"] or [l for l in j.get("lines", []) if isinstance(l, str)]
    plan["dates"] = plan["dates"] or [d for d in j.get("dates", []) if isinstance(d, str)]
    plan["time"] = plan["time"] or j.get("time")
    plan["dur_min"] = plan["dur_min"] or j.get("dur_min")
    plan["what_if"] = plan["what_if"] or bool(j.get("what_if"))
    plan.update(CAT_DATA.get(plan["cat"], {}))
    return plan
