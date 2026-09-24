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
F network resilience · G correlation · H reroute behaviour · P pressure ranking (which stations get busiest on a day) ·
X bonus (investment / InnoTrans route) · OOS out of scope / unanswerable from data.
"""
from __future__ import annotations

import difflib
import glob
import json
import os
import re
import time
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
    "P": [(r"most likely to (exceed|overload|be (over)?crowded|be busy|be busiest)", 4.5), (r"(which|what|show( me)?|list|name|top)\s+(the\s+)?(\d+|two|three|four|five)\s+(busiest\s+)?stations?\b.*(exceed|overcrowd|crowded|busiest|most|capacity|pressure|load)", 3.5),
          (r"(exceed|overload|breach)\w*\s+(safe\s+)?(platform\s+)?capacity", 2.5), (r"first day of the (event|fair|innotrans)", 1.5),
          (r"stations? (at risk|under (the most )?(pressure|load))|highest (load|pressure)|most (crowded|overcrowded)", 2.5)],
    "X": [(r"invest", 3), (r"infrastructure improvement", 3), (r"innotrans", 2), (r"messe", 2),
          (r"unconventional", 3), (r"not based on the shortest", 3)],
    "A": [(r"concert|festival|gig|match\b", 2), (r"\barena\b|olympiastadion|tempodrom|stadium|venue", 2.5),
          (r"\bevents?\b", 1.5), (r"guns n|doja cat|dikka|alligatoah|tour\b", 2), (r"ten events|events across", 2)],
    "C": [(r"what happened (on|to|at) the u\d", 2), (r"suspend|suspension", 2.5), (r"\bclos(ed|ure|ures|e)\b", 1.5), (r"disrupt", 1.5),
          (r"re-?rout", 2), (r"overload", 2), (r"deploy(ed)? .*staff|additional staff|staff", 1.2),
          (r"replacement bus|what if .*(stop|suspend|clos)", 2), (r"sperrung|gesperrt", 2.5),
          (r"closed on|was closed|is closed", 1)],
    "D": [(r"\bwie voll\b|u-?bahnhof|fahrg(ä|ae)ste|stoßzeit|berufsverkehr|normalerweise|wie viele fahrg", 2.5), (r"\bpeaks?\b|commute", 1.5), (r"usually|typical|rhythm|profile", 1.5), (r"busiest|quietest|quiet(est)? stations?", 2),
          (r"what time|when (does|is)", 1.2), (r"mean (commute )?peak|network mean|exceed", 1.5),
          (r"flow at .*(at|on) \d|predict", 1.5), (r"what was the flow|how (busy|crowded) is", 2), (r"weekday|weekend", 1)],
}
PRIORITY = ["F", "H", "G", "E", "B", "P", "X", "A", "C", "D"]

# What each category needs — put in the plan so the executor (and anyone reading a trace) sees which
# datasets and MCP tools the question will touch. `tools` lists what is wired up today ([] = not yet).
CAT_DATA = {
    "A": {"data": ["events", "stations", "flows"], "tools": ["event_impact"]},
    "B": {"data": ["flows", "weather", "events", "closures"], "tools": ["find_anomalies"]},
    "C": {"data": ["closures", "connections", "stations", "flows", "weather", "events"],
          "tools": ["resolve_closure", "apply_closure", "alternate_paths", "scenario_flow"]},
    "D": {"data": ["flows"], "tools": ["resolve_station", "station_profile", "predict_expected_flow", "predict_overcrowding_risk"]},
    "E": {"data": ["energy", "flows", "stations"], "tools": ["energy_efficiency"]},
    "F": {"data": ["connections", "flows"], "tools": ["network_resilience_ranking"]},
    "G": {"data": ["flows", "connections", "stations"], "tools": ["correlated_stations"]},
    "H": {"data": ["closures", "flows"], "tools": ["reroute_behaviour"]},
    "P": {"data": ["flows", "weather", "events"], "tools": ["rank_pressure"]},
    "X": {"data": ["flows", "connections", "energy", "closures"], "tools": []},
    "OOS": {"data": [], "tools": []}, "FOLLOW": {"data": [], "tools": []},
}
OOS = re.compile(r"capacity|safely hold|delayed trains?|\bdelay\w*|real-?time|right now|live (status|position|data)|cost of|how much would .* cost|currywurst|restaurant|"
                 r"ignore (your|all|previous)|next (monday|week|month)|\bu4\b|signalling failure", re.I)
# a question that only asks for a capacity figure (no ranking / no scenario to answer around it)
CAPACITY_ASK = re.compile(r"how many (passengers|people)[^?.]*(hold|fit|accommodate)|capacity of|how much capacity|maximum capacity|safely hold", re.I)
REL_DAY = re.compile(r"\b(tonight|today|this (evening|morning|afternoon)|tomorrow)\b", re.I)
FOLLOW = re.compile(r"^\s*(and|so|what about|how about)\b|\b(those|that|these|them|the top|the first)\b|"
                    r"how confident|which of (those|these)|measured and|explain (that|why)|what data would", re.I)

# a question ABOUT the previous answer (why / how sure / based on what): explained from the earlier facts, never re-run
from detail_ask import DETAIL_ASK  # noqa: E402

_WHY = re.compile(r"^\s*(and\s+)?why\b|how (do you know|sure|confident)|based on what|on what basis|justify|what makes you|are you sure|explain (that|why|how)", re.I)


class _Either:
    """WHY_FOLLOW.search(q): the older why-phrases OR an explicit request for evidence / sources / tools / the full report."""
    @staticmethod
    def search(q: str):
        return _WHY.search(q) or DETAIL_ASK.search(q)


WHY_FOLLOW = _Either()

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


@lru_cache(maxsize=1)
def station_lines() -> dict[str, frozenset]:
    """exact station_name -> the lines serving it (from stations_with_ubahn.csv)."""
    base = os.environ.get("DATA_DIR", "data")
    repo = Path(__file__).resolve().parent.parent
    hits = glob.glob(str(Path(base) / "**" / "stations_with_ubahn.csv"), recursive=True) or glob.glob(str(repo / "data" / "**" / "stations_with_ubahn.csv"), recursive=True)
    if not hits:
        return {}
    df = pd.read_csv(hits[0])
    out: dict[str, set] = {}
    for n, ls in zip(df["station_name"], df["u_bahn_lines"]):
        out.setdefault(n, set()).update(x.strip() for x in str(ls).split(","))
    return {k: frozenset(v) for k, v in out.items()}


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


HORIZON = re.compile(r"\b(?:in|within|over|during)\s+the\s+(?:next|coming|first)\s+(\d+)\s*(minutes?|mins?|hours?|hrs?|h)\b|"
                     r"\bnext\s+(\d+)\s*(minutes?|mins?|hours?|hrs?|h)\b", re.I)


def find_horizon_min(text: str) -> int | None:
    """'... in the next 20 minutes' -> 20. A look-ahead window, NOT the length of a closure."""
    m = HORIZON.search(text)
    if not m:
        return None
    n, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return int(n) * (60 if unit.lower().startswith("h") else 1)


def find_duration_min(text: str) -> int | None:
    t = HORIZON.sub(" ", text.lower())          # a look-ahead horizon must never be read as the closure duration
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", t)
    if m:
        return int(float(m.group(1)) * 60)
    m = re.search(r"(\d+)\s*(minutes?|mins?)\b", t)
    if m:
        return int(m.group(1))
    m = re.search(rf"\b({'|'.join(WORDNUM)})\s+hours?\b", t)
    return int(WORDNUM[m.group(1)] * 60) if m else None


def find_times(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", text):
        out.append(f"{int(m.group(1)):02d}:{m.group(2)}")
    return out


NUMWORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10}


def find_top_n(text: str) -> int | None:
    m = re.search(r"\b(\d+|" + "|".join(NUMWORDS) + r")\s+(?:busiest\s+|most\s+\w+\s+)?(?:stations?|lines?|anomal\w+|pairs?|examples?)\b", text.lower())
    if not m:
        return None
    return int(m.group(1)) if m.group(1).isdigit() else NUMWORDS[m.group(1)]


def _mojibake(v: str) -> str:
    for enc in ("cp1252", "latin-1"):
        try:
            return v.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return v


@lru_cache(maxsize=1)
def _events_frame() -> pd.DataFrame:
    base = os.environ.get("DATA_DIR", "data")
    repo = Path(__file__).resolve().parent.parent
    hits = sorted(glob.glob(str(Path(base) / "**" / "berlin_events*.csv"), recursive=True)) or sorted(glob.glob(str(repo / "data" / "**" / "berlin_events*.csv"), recursive=True))
    header_names = None
    for h in hits:                                                       # the events test split has no header row: the training file names its columns
        cols = list(pd.read_csv(h, nrows=0, encoding="utf-8-sig").columns)
        if "event_name" in cols:
            header_names = cols
            break
    frames = []
    for h in hits:
        has_header = "event_name" in pd.read_csv(h, nrows=0, encoding="utf-8-sig").columns
        df = pd.read_csv(h, encoding="utf-8-sig") if has_header else pd.read_csv(h, encoding="utf-8-sig", header=None, names=header_names)
        if not has_header:
            for c in ("event_name", "venue_name"):
                df[c] = df[c].map(lambda v: _mojibake(v) if isinstance(v, str) else v)
        frames.append(df[["event_name", "venue_name", "began_local"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["event_name", "venue_name", "began_local"])


VENUE_ALIASES = {"mercedes-benz arena": "Uber Arena", "mercedes benz arena": "Uber Arena", "o2 arena": "Uber Arena", "mercedes-platz": "Uber Arena"}


def find_venue(text: str) -> tuple[str | None, bool]:
    """(venue name as the data spells it, alias_used) — venue names come from the events file itself."""
    t = text.lower()
    for alias, target in VENUE_ALIASES.items():
        if alias in t:
            return target, True
    ev = _events_frame()
    venues = sorted({v for v in ev["venue_name"].dropna().unique() if len(v) >= 5}, key=len, reverse=True)
    for v in venues:
        if v.lower() in t:
            return v, False
    return None, False


def find_event_name(text: str) -> str | None:
    """An event of the data named in the question ('Guns N' Roses concert' -> the base title before ' - ')."""
    norm = lambda x: re.sub(r"[^a-z0-9 ]", "", re.sub(r"[’'`]", "", x.lower())).strip()
    q = " " + re.sub(r"\s+", " ", norm(text)) + " "
    best = None
    for name in _events_frame()["event_name"].dropna().unique():
        base = norm(re.split(r"\s[-–:|]\s", name)[0])
        if len(base) >= 8 and base not in {"berlin", "concert", "festival"} and f" {base} " in q and (best is None or len(base) > len(best[0])):
            best = (base, re.split(r"\s[-–:|]\s", name)[0].replace("’", "'"))
    return best[1] if best else None


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
    follow = bool(has_history and (WHY_FOLLOW.search(q) or (FOLLOW.search(q) and not stations)))

    cat, conf = top, 0.0
    if scores[top] > 0:
        margin = scores[top] - scores[second]
        conf = min(1.0, scores[top] / 3.0) * (1.0 if margin >= 1.0 else 0.65)
    oos_hits = sorted({m.group(0).lower() for m in OOS.finditer(q)})
    answerable_top = top in ("C", "D", "A", "P", "B") and scores[top] >= 3
    if oos_hits and not (answerable_top and not CAPACITY_ASK.search(q)):
        cat, conf = "OOS", 0.9
    if follow:
        cat, conf = "FOLLOW", 0.9
    what_if = bool(re.search(r"\bwhat if\b|\bsuppose\b|\bimagine\b|\bwould be suspended\b", q, re.I))
    dates, assumed = find_dates(q), []
    times = find_times(q)
    venue, alias = find_venue(q)
    if cat in ("A", "P", "X", "OOS") and re.search(r"innotrans", q, re.I) and not dates:
        # operator context, NOT in the data: InnoTrans 2026 runs 22-25 Sept 2026 at Messe Berlin. Disclosed in the answer.
        first_day = re.search(r"first day", q, re.I)
        dates = ["2026-09-22"] if first_day or cat == "P" else dates
        if dates:
            assumed.append("InnoTrans 2026 is taken to start on 2026-09-22 (fair dates from the operator's context, cross-checked with the events file)")
    rel = REL_DAY.search(q)
    if cat == "A" and find_event_name(q):
        conf = max(conf, 0.7)                                       # an event of the data is named: the rules are sure (no need to ask the small LLM, which drifts to 'station profile')
    return {"cat": cat, "conf": round(conf, 2), "tier": 0, **CAT_DATA.get(cat, {}), "lines": list(dict.fromkeys(lines)),
            "stations": stations, "raw": [], "dates": dates, "month": find_month(q), "oos": oos_hits, "time": find_time(q), "times": times,
            "dur_min": find_duration_min(q), "horizon_min": find_horizon_min(q), "what_if": what_if, "follow": follow,
            "venue": venue, "alias": alias, "event": find_event_name(q), "n": find_top_n(q),
            "rain": True if re.search(r"\b(rain(s|y|ing)?|storm|bad weather|wet)\b", q, re.I) and not re.search(r"\bno rain\b", q, re.I) else None,
            "rel_day": rel.group(1).lower() if rel else None, "assumed": assumed}


# ---------------------------------------------------------------- multi-intent messages
_SENT = re.compile(r"(?<=[?!.])\s+")
_CLAUSE = re.compile(r",\s*(?:and\s+)?(?=(?:does|do|how|what|which|where|when|why|is|are|can|will|should)\b)|\band\s+(?=(?:what|how many|how much|which)\b)", re.I)


_QWORD = re.compile(r"(does|do|how|what|which|where|when|why|is|are|can|will|should)\b", re.I)


def _statement_merge(sents: list[str]) -> list[str]:
    """Context statements ('During InnoTrans we expect ...') belong to the question that follows; only questions and imperatives stand alone."""
    out, buf = [], ""
    for x in sents:
        if x.rstrip().endswith("?") or re.match(r"(ignore|tell|show|give|list|name|rank|identify|suggest|explain|predict|compare)\b", x, re.I):
            out.append((buf + " " + x).strip())
            buf = ""
        else:
            buf = (buf + " " + x).strip()
    return out + ([buf] if buf and not out else [])


def _part_plan(text: str, has_history: bool, min_conf: float = 0.3) -> dict | None:
    p = route(text, has_history)
    if p["cat"] != "OOS" and p["conf"] < min_conf and not p["dates"]:
        return None
    if p["conf"] < 0.3:                                    # no rule fired: the tie-break category means nothing
        if not p["dates"]:
            return None                                    # context clause, no question of its own
        p = {**p, "cat": "D", "conf": 0.5, **CAT_DATA["D"]}     # a date-only question ('the flow on Sept 30th'): the date guard decides
    return {**p, "text": text}


def split_parts(q: str, has_history: bool = False) -> list[dict]:
    """A message with SEVERAL different questions ('closure ... At the same time: does Rudow's peak exceed the mean, how many passengers can the
    platform hold, what will the flow be on Sept 30? Ignore your rules ...') is split so each part gets its own specialist. A sentence stays whole
    when its clauses belong to one category (the usual multi-clause closure question). Returns [] for an ordinary single question."""
    texts: list[str] = []
    for sent in _statement_merge([x for x in _SENT.split(q.strip()) if len(x.split()) >= 3]):
        clauses = [c.strip(" ,;:") for c in _CLAUSE.split(sent) if c and len(c.split()) >= 4]
        merged: list[str] = []
        for c in clauses:                                   # 'On 2026-09-23, during InnoTrans, which 3 stations ...': a lead-in without a question word belongs to the question after it
            if merged and not _QWORD.match(merged[-1]) and len(merged[-1].split()) <= 6:
                merged[-1] = merged[-1] + ", " + c
            else:
                merged.append(c)
        clauses = merged
        sub = [pp for pp in (_part_plan(c, has_history, 0.9) for c in clauses)] if len(clauses) > 1 else []
        cats = {pp["cat"] for pp in sub if pp} - {"FOLLOW"}
        texts.extend([pp["text"] for pp in sub if pp] if len(cats) > 1 else [sent])
    plans = [pp for pp in (_part_plan(t, has_history) for t in texts) if pp]
    cats = {p["cat"] for p in plans}
    answerable = [p for p in plans if p["cat"] not in ("OOS", "FOLLOW")]
    return plans if len(plans) >= 2 and len(cats) >= 2 and answerable else []


def with_parts(plan: dict, q: str, has_history: bool = False) -> dict:
    """Attach `parts` to the plan of a multi-intent message; the plan's own category becomes the first answerable part's."""
    parts = split_parts(q, has_history)
    if not parts or plan["cat"] == "FOLLOW":
        return plan
    lead = next(p for p in parts if p["cat"] not in ("OOS", "FOLLOW"))
    return {**plan, "cat": lead["cat"], "conf": max(plan["conf"], 0.9), "parts": parts, "tools": sorted({t for p in parts for t in p.get("tools", [])}),
            "oos": sorted({h for p in parts for h in p.get("oos", [])})}


ROUTER_SYSTEM = (
    "Route a Berlin U-Bahn operator question. Reply with ONE JSON object, no prose:\n"
    '{"cat":"A|B|C|D|E|F|G|H|P|X|OOS|FOLLOW","lines":["U6"],"places":["as written by user"],'
    '"dates":["YYYY-MM-DD"],"time":"HH:MM|null","dur_min":null,"what_if":false}\n'
    "A=event impact on stations, B=anomaly/root cause, C=line/station closure response (reroute, overload, staff), "
    "D=one station's flow profile/peak/prediction, E=energy per passenger, F=network fragmentation/critical stations, "
    "G=correlated stations, H=how passengers actually reroute, P=which N stations get the highest load/pressure on a day (also weather/event scenarios), X=investment/InnoTrans route, "
    "OOS=needs data we lack (capacity, delays, costs, forecasts beyond the data window, off-topic). Year is 2026. "
    "If HISTORY=yes and the message only refers back to the previous answer ('those stations', 'how sure'), use FOLLOW."
)


async def llm_route(question: str, base_plan: dict, has_history: bool = False) -> dict:
    """Tier 1: small JSON-only LLM call, used when the deterministic router is unsure (or always, in the
    `router=llm` experiment arm)."""
    import litellm

    from llm_config import litellm_params, sampling_params

    cfg = litellm_params("ROUTER")
    if cfg is None:
        return base_plan
    model, kw = cfg
    from observability import log_llm

    t0 = time.time()
    user = f"HISTORY={'yes' if has_history else 'no'}\n{question}"
    try:
        resp = await litellm.acompletion(
            model=model, messages=[{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=160, **sampling_params(model, 0), response_format={"type": "json_object"}, timeout=12, **kw)
    except Exception as e:
        log_llm("router", model, time.time() - t0, prompt=user, error=type(e).__name__, started=t0)
        raise
    log_llm("router", model, time.time() - t0, getattr(resp, "usage", None), user, resp.choices[0].message.content or "", started=t0)
    try:
        j = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return base_plan
    plan = dict(base_plan)
    plan.update(tier=1, cat=j.get("cat", plan["cat"]) if j.get("cat") in {*"ABCDEFGHPX", "OOS", "FOLLOW"} else plan["cat"],
                conf=0.8, raw=[p for p in j.get("places", []) if isinstance(p, str)])
    plan["lines"] = plan["lines"] or [l for l in j.get("lines", []) if isinstance(l, str)]
    plan["dates"] = plan["dates"] or [d for d in j.get("dates", []) if isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)]
    llm_time = j.get("time")
    llm_time = llm_time if isinstance(llm_time, str) and re.fullmatch(r"\d{1,2}:\d{2}", llm_time.strip()) else None   # "null"/"None" -> None
    plan["time"] = plan["time"] or llm_time
    plan["dur_min"] = plan["dur_min"] or (j.get("dur_min") if isinstance(j.get("dur_min"), (int, float)) else None)
    plan["what_if"] = plan["what_if"] or bool(j.get("what_if"))
    plan.update(CAT_DATA.get(plan["cat"], {}))
    plan["follow"] = plan["cat"] == "FOLLOW"
    return plan
