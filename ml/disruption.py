"""Category C (disruption response) — graph + closure logic, no ML in here.

This module is the deterministic half of the Category C solver:

    resolve_closure -> apply_closure -> alternate_paths -> scenario_flow
                                                          (needs a demand model,
                                                           see ml/demand_baseline.py)

Everything here works on station *names* (the `station_name` strings that are
also the flows.csv column headers), not VBB ids, because that is what the
operator, closures.csv and flows.csv all speak.

Modelling assumptions (all surfaced to the caller in `assumptions`, never
implied to be measured facts — the dataset has no capacity, headway, rolling
stock or per-line load data; see docs/agentic_system_design.md data-gap #2):

  * A "line section" closure (`Line U3 suspended between A and B`) removes the
    rail edges of THAT line along the section. Stations strictly between A and
    B that only serve that line become unserved; interchange stations keep
    their other lines. A and B stay open (served from their outer side).
  * A "station" closure means passengers cannot enter/exit there. Trains are
    assumed to run through without stopping (closures.csv doesn't say).
  * Edge-to-line attribution is inferred: an edge belongs to every line that
    serves BOTH endpoints. connections.csv carries no line label per edge.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from itertools import islice

import networkx as nx
import pandas as pd

DEFAULT_DIVERSION_SHARES = {"low": 0.25, "base": 0.5, "high": 0.75}
MAX_SPILL_HOPS = 2
MAX_ALT_PATHS = 3

ASSUMPTIONS = [
    "Demand figures are model-based baseline estimates (what the station would normally "
    "see in this time slot) plus an ASSUMED redistribution — not a measurement. The "
    "dataset shows no measurable redistribution during its historical closures.",
    "Only the closed/unserved stations' own baseline demand is redistributed "
    "(unserved: 100%; section endpoints and interchange stations still served: the "
    "`diversion_share` assumption).",
    "Displaced boardings spill to the nearest open stations (<=2 graph hops, weight 1/hops). "
    "Crossing passengers on a line-section suspension follow the alternate paths, "
    "split inversely to path length, and load the TRANSFER stations on those paths.",
    "'Overloaded' means exceeding the station's own historical 95th percentile for the same "
    "hour-of-day and weekday/weekend type — a relative demand-pressure proxy, NOT a "
    "measured platform or train capacity (no capacity data exists).",
    "Line/train loading on the alternate paths is not estimated (no per-line load data).",
]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(r"^(S\+U|U|S)\s+")
_SUFFIX_RE = re.compile(r"\s*\(Berlin\)\s*$")


def _norm(name: str) -> str:
    return _SUFFIX_RE.sub("", _PREFIX_RE.sub("", name.strip())).strip().lower()


@dataclass
class Network:
    """Station-name graph + line membership + flow-column names."""

    g: nx.Graph
    lines_of: dict[str, list[str]]
    flow_stations: list[str]
    coords: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (lat, lon)
    _norm_index: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for n in self.g.nodes:
            self._norm_index.setdefault(_norm(n), []).append(n)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_frames(cls, stations: pd.DataFrame, connections: pd.DataFrame,
                    flow_columns: list[str]) -> "Network":
        id_to_name = dict(zip(stations["station_id"], stations["station_name"]))
        lines_of: dict[str, set[str]] = {}
        for name, lines in zip(stations["station_name"], stations["u_bahn_lines"]):
            lines_of.setdefault(name, set()).update(x.strip() for x in lines.split(","))
        g = nx.Graph()
        g.add_nodes_from(lines_of)
        for a, b in connections[["station_id_1", "station_id_2"]].itertuples(index=False, name=None):
            if a in id_to_name and b in id_to_name and id_to_name[a] != id_to_name[b]:
                g.add_edge(id_to_name[a], id_to_name[b])
        coords = {n: (float(la), float(lo)) for n, la, lo in
                  zip(stations["station_name"], stations["latitude"], stations["longitude"])}
        return cls(g=g, lines_of={k: sorted(v) for k, v in lines_of.items()},
                   flow_stations=[c for c in flow_columns if c in g], coords=coords)

    # -- name resolution ---------------------------------------------------
    def resolve(self, query: str, max_results: int = 5) -> list[str]:
        """Exact station_name candidates for a loose reference, best first.
        Order: exact normalized match ('Neukölln' -> 'S+U Neukölln (Berlin)', NOT
        'U Rathaus Neukölln (Berlin)'), then substring, then fuzzy spelling."""
        q = _norm(query)
        exact = self._norm_index.get(q, [])
        names = list(self.g.nodes)
        sub = [n for n in names if q in n.lower() and n not in exact]
        fuzzy = difflib.get_close_matches(q, list(self._norm_index), n=max_results, cutoff=0.6)
        fuzzy_names = [n for f in fuzzy for n in self._norm_index[f]]
        return list(dict.fromkeys(exact + sub + fuzzy_names))[:max_results]

    def resolve_one(self, query: str) -> tuple[str | None, list[str]]:
        """(the single confident exact name, or None if ambiguous/unknown; plus all
        candidates). Confident = exactly one normalized-exact match, or no exact
        match and exactly one substring match ('Alexanderplatz' -> the one station
        containing it). Fuzzy spelling matches are suggestions only."""
        cands = self.resolve(query)
        if not cands:
            return None, []
        q = _norm(query)
        exact = self._norm_index.get(q, [])
        if len(exact) == 1:
            return exact[0], cands
        if not exact:
            sub = [n for n in self.g.nodes if q in n.lower()]
            if len(sub) == 1:
                return sub[0], cands
        return None, cands

    # -- line helpers ------------------------------------------------------
    def edge_lines(self, a: str, b: str) -> set[str]:
        return set(self.lines_of.get(a, [])) & set(self.lines_of.get(b, []))

    def line_subgraph(self, line: str) -> nx.Graph:
        nodes = [n for n, ls in self.lines_of.items() if line in ls]
        sub = nx.Graph()
        sub.add_nodes_from(nodes)
        sub.add_edges_from((a, b) for a, b in self.g.subgraph(nodes).edges
                           if line in self.edge_lines(a, b))
        return sub


# ---------------------------------------------------------------------------
# resolve_closure
# ---------------------------------------------------------------------------

def _closure_to_spec(net: Network, idx: int, row: pd.Series) -> dict:
    spec: dict = {
        "closure_id": int(idx),
        "source": "closures.csv",
        "start": row["when"].isoformat(sep=" "),
        "end": row["end"].isoformat(sep=" "),
        "duration_hours": float(row["duration_hours"]),
        "reason": row["reason"],
        "description": row["description"],
    }
    warnings: list[str] = []
    if row["closure_type"] == "Station closure":
        best, cands = net.resolve_one(row["affected_segment"])
        spec.update(kind="station", line=None, station=best, from_station=None, to_station=None)
        if best is None:
            warnings.append(f"station '{row['affected_segment']}' ambiguous/unknown: {cands}")
    elif row["closure_type"] == "Line suspension":
        a_txt, b_txt = row["affected_segment"].split(" ↔ ")
        a, ca = net.resolve_one(a_txt)
        b, cb = net.resolve_one(b_txt)
        spec.update(kind="line_section", line=row["affected_line"], station=None,
                    from_station=a, to_station=b)
        for txt, val, cands in ((a_txt, a, ca), (b_txt, b, cb)):
            if val is None:
                warnings.append(f"station '{txt}' ambiguous/unknown: {cands}")
    else:
        spec.update(kind="unknown", line=None, station=None, from_station=None, to_station=None)
        warnings.append("closure description not recognised")
    if warnings:
        spec["warnings"] = warnings
    return spec


def resolve_closures(net: Network, closures: pd.DataFrame, query: str = "",
                     max_results: int = 5) -> list[dict]:
    """Closures matching a free-text query: closure id, a date/date-time
    ('2026-09-14'), a line ('U7'), a station or a reason keyword. Empty query
    lists the most recent closures. Deterministic — reason/duration come
    straight from closures.csv."""
    df = closures.reset_index(drop=True)
    q = query.strip()
    if not q:
        hits = df.tail(max_results).index
    elif q.isdigit() and int(q) < len(df):
        hits = [int(q)]
    else:
        m = pd.Series(False, index=df.index)
        date_m = re.search(r"\d{4}-\d{2}-\d{2}", q)
        if date_m:
            m |= df["when"].dt.strftime("%Y-%m-%d") == date_m.group(0)
        line_m = re.findall(r"\bU\d+\b", q, flags=re.I)
        for ln in line_m:
            m |= df["affected_line"].fillna("").str.upper() == ln.upper()
        text = re.sub(r"\d{4}-\d{2}-\d{2}|\bU\d+\b", " ", q, flags=re.I).strip().lower()
        if text:
            m |= df["description"].str.lower().str.contains(re.escape(text), regex=True)
            for cand in net.resolve(text, 2)[:1]:  # station reference in the text
                m |= df["description"].str.lower().str.contains(re.escape(_norm(cand)), regex=True)
        hits = df.index[m][:max_results]
    return [_closure_to_spec(net, i, df.loc[i]) for i in hits]


def make_hypothetical_spec(net: Network, *, line: str | None, from_station: str | None,
                           to_station: str | None, station: str | None,
                           start: str, duration_minutes: int) -> dict | str:
    """Spec for a what-if closure that isn't in closures.csv. Returns an error
    string if a station reference can't be resolved unambiguously."""
    start_ts = pd.to_datetime(start)
    spec: dict = {
        "closure_id": None, "source": "hypothetical",
        "start": start_ts.isoformat(sep=" "),
        "end": (start_ts + pd.Timedelta(minutes=duration_minutes)).isoformat(sep=" "),
        "duration_hours": duration_minutes / 60, "reason": "hypothetical", "description": "what-if",
    }
    if station and not (from_station or to_station):
        best, cands = net.resolve_one(station)
        if best is None:
            return f"station '{station}' unknown/ambiguous: {cands}"
        spec.update(kind="station", line=None, station=best, from_station=None, to_station=None)
        return spec
    if not (line and from_station and to_station):
        return "need either `station`, or `line` + `from_station` + `to_station`"
    a, ca = net.resolve_one(from_station)
    b, cb = net.resolve_one(to_station)
    if a is None or b is None:
        return f"unresolved stations: from={from_station!r}->{ca}, to={to_station!r}->{cb}"
    spec.update(kind="line_section", line=line.upper(), station=None, from_station=a, to_station=b)
    return spec


# ---------------------------------------------------------------------------
# apply_closure
# ---------------------------------------------------------------------------

@dataclass
class ClosureEffect:
    spec: dict
    section_path: list[str]            # line_section: A..B along the line; station: [S]
    blocked_edges: list[tuple[str, str]]
    unserved: list[str]                # lose all rail service (or are closed)
    partially_served: list[str]        # in the section but keep another line (interchange)
    graph_after: nx.Graph
    notes: list[str] = field(default_factory=list)

    @property
    def affected_stations(self) -> list[str]:
        """Every station whose own flow the closure directly touches."""
        return list(dict.fromkeys(self.unserved + self.partially_served
                                  + ([self.spec["from_station"], self.spec["to_station"]]
                                     if self.spec["kind"] == "line_section" else [])))


def apply_closure(net: Network, spec: dict) -> ClosureEffect:
    g2 = net.g.copy()
    notes: list[str] = []

    if spec["kind"] == "station":
        s = spec["station"]
        if s not in net.g:
            raise ValueError(f"unknown station {s!r}")
        notes.append("Station closed to passengers; trains assumed to run through without "
                     "stopping. Alternate paths below are the conservative case where they don't.")
        g2.remove_node(s)
        return ClosureEffect(spec, [s], [], [s], [], g2, notes)

    if spec["kind"] != "line_section":
        raise ValueError(f"unsupported closure kind {spec['kind']!r}")

    line, a, b = spec["line"], spec["from_station"], spec["to_station"]
    if a is None or b is None:
        raise ValueError("closure endpoints not resolved: " + "; ".join(spec.get("warnings", [])))
    sub = net.line_subgraph(line)
    if a not in sub or b not in sub:
        raise ValueError(
            f"{a} (lines {net.lines_of.get(a)}) and {b} (lines {net.lines_of.get(b)}) are not both "
            f"on {line}. Check the line, or use resolve_closure / list_stations.")
    if not nx.has_path(sub, a, b):
        raise ValueError(f"{a} and {b} are on {line} but not connected along it (line branches?).")
    path = nx.shortest_path(sub, a, b)

    blocked = []
    shared_with: set[str] = set()
    for u, v in zip(path, path[1:]):
        if net.edge_lines(u, v) <= {line}:      # only this line runs here -> edge gone
            g2.remove_edge(u, v)
            blocked.append((u, v))
        else:
            shared_with |= net.edge_lines(u, v) - {line}
    if shared_with:
        notes.append(f"Some segments are also served by {sorted(shared_with)}; those rail edges "
                     f"stay in service (only edges served by {line} alone are removed).")
    interior = path[1:-1]
    unserved = [n for n in interior if set(net.lines_of[n]) <= {line}]
    partial = [n for n in interior if n not in unserved]
    return ClosureEffect(spec, path, blocked, unserved, partial, g2, notes)


def isolated_components(effect: ClosureEffect, net: Network) -> list[list[str]]:
    """Station groups cut off from the main network by the closure (no rail
    alternative at all — bus replacement territory). Excludes stations that
    are simply closed/unserved."""
    comps = sorted(nx.connected_components(effect.graph_after), key=len, reverse=True)
    if not comps:
        return []
    return [sorted(set(c) - set(effect.unserved)) for c in comps[1:]
            if set(c) - set(effect.unserved)]


# ---------------------------------------------------------------------------
# alternate_paths
# ---------------------------------------------------------------------------

def _min_transfers(net: Network, path: list[str]) -> tuple[list[str], list[str]]:
    """(line sequence, transfer stations) for a path, minimising line changes."""
    edge_sets = [net.edge_lines(u, v) or set(net.lines_of[u]) for u, v in zip(path, path[1:])]
    if not edge_sets:
        return [], []
    # DP over lines: cost[l] = (transfers, chosen sequence)
    best: dict[str, tuple[int, list[str]]] = {ln: (0, [ln]) for ln in edge_sets[0]}
    for es in edge_sets[1:]:
        nxt: dict[str, tuple[int, list[str]]] = {}
        for ln in es:
            cands = [(c + (0 if ln == p_last[-1] else 1), seq + [ln])
                     for c, seq in best.values() for p_last in [seq]]
            nxt[ln] = min(cands, key=lambda t: t[0])
        best = nxt
    _, seq = min(best.values(), key=lambda t: t[0])
    transfers = [path[i + 1] for i in range(len(seq) - 1) if seq[i] != seq[i + 1]]
    compressed = [seq[0]] + [ln for prev, ln in zip(seq, seq[1:]) if ln != prev]
    return compressed, transfers


def alternate_paths(net: Network, effect: ClosureEffect, k: int = MAX_ALT_PATHS) -> list[dict]:
    """Detour paths around the closure, fewest hops first.

    line_section: k simple paths from A to B in the graph with the section
    removed. station: for each pair of the station's neighbours (per line),
    the detours that avoid the station."""
    if effect.spec["kind"] == "line_section":
        pairs = [(effect.spec["from_station"], effect.spec["to_station"])]
    else:
        s = effect.spec["station"]
        pairs = []
        for line in net.lines_of.get(s, []):
            nbrs = [n for n in net.g.neighbors(s) if line in net.lines_of[n]
                    and line in net.edge_lines(s, n)]
            if len(nbrs) == 2:
                pairs.append((nbrs[0], nbrs[1]))

    out: list[dict] = []
    original_hops = len(effect.section_path) - 1 if effect.spec["kind"] == "line_section" else 2
    for a, b in pairs:
        if a not in effect.graph_after or b not in effect.graph_after \
                or not nx.has_path(effect.graph_after, a, b):
            out.append({"from": a, "to": b, "paths": [],
                        "note": "no rail alternative — endpoints disconnected (needs replacement bus)"})
            continue
        paths = []
        for p in islice(nx.shortest_simple_paths(effect.graph_after, a, b), k):
            lines, transfers = _min_transfers(net, p)
            paths.append({
                "stations": p, "hops": len(p) - 1,
                "extra_hops_vs_original": len(p) - 1 - original_hops,
                "lines": lines, "transfer_stations": transfers,
            })
        out.append({"from": a, "to": b, "paths": paths})
    return out


def surface_links(net: Network, effect: ClosureEffect, k: int = 3) -> list[dict]:
    """When a line-section suspension leaves NO rail path between the two
    sides (the common case: Berlin's U-Bahn graph is almost a tree), list the
    geographically closest station pairs across the cut as candidate
    bus/walk links. Straight-line distance only — it says a link is
    geometrically plausible, not that any service exists."""
    import numpy as np

    if effect.spec["kind"] != "line_section":
        return []
    a, b = effect.spec["from_station"], effect.spec["to_station"]
    g2 = effect.graph_after
    if nx.has_path(g2, a, b):
        return []
    side_a = [n for n in nx.node_connected_component(g2, a) if n not in effect.unserved]
    side_b = [n for n in nx.node_connected_component(g2, b) if n not in effect.unserved]
    ca = np.radians([net.coords[n] for n in side_a])
    cb = np.radians([net.coords[n] for n in side_b])
    dlat = ca[:, None, 0] - cb[None, :, 0]
    dlon = ca[:, None, 1] - cb[None, :, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(ca[:, None, 0]) * np.cos(cb[None, :, 0]) * np.sin(dlon / 2) ** 2
    km = 2 * 6371.0 * np.arcsin(np.sqrt(h))
    flat = np.argsort(km, axis=None)[:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, km.shape)
        out.append({"from": side_a[i], "to": side_b[j], "straight_line_km": round(float(km[i, j]), 2),
                    "from_lines": net.lines_of[side_a[i]], "to_lines": net.lines_of[side_b[j]]})
    return out


# ---------------------------------------------------------------------------
# scenario_flow — redistribution arithmetic, demand model injected
# ---------------------------------------------------------------------------

def spill_targets(net: Network, effect: ClosureEffect, source: str) -> dict[str, float]:
    """Nearest open stations to `source`, weights 1/hops (<= MAX_SPILL_HOPS), sum to 1."""
    blocked = set(effect.unserved)
    dist = nx.single_source_shortest_path_length(net.g, source, cutoff=MAX_SPILL_HOPS)
    cands = {n: 1.0 / d for n, d in dist.items() if d > 0 and n not in blocked}
    total = sum(cands.values())
    return {n: w / total for n, w in cands.items()} if total else {}


def path_weights(paths: list[dict]) -> list[float]:
    inv = [1.0 / max(p["hops"], 1) for p in paths]
    total = sum(inv)
    return [x / total for x in inv]


def redistribution_plan(net: Network, effect: ClosureEffect) -> dict:
    """The structure of the redistribution, independent of any numbers:
      sources:  {station: 'unserved' | 'partial' | 'endpoint'}
      spill:    {source: {receiver: weight}}
      transfer: {station: weight}   share of the crossing flow that transfers there
      crossing_pair: (A, B) or None
    """
    spec = effect.spec
    sources: dict[str, str] = {s: "unserved" for s in effect.unserved}
    sources.update({s: "partial" for s in effect.partially_served})
    crossing_pair = None
    transfer: dict[str, float] = {}
    if spec["kind"] == "line_section":
        a, b = spec["from_station"], spec["to_station"]
        sources.setdefault(a, "endpoint")
        sources.setdefault(b, "endpoint")
        crossing_pair = (a, b)
        alts = alternate_paths(net, effect)
        if alts and alts[0]["paths"]:
            for w, p in zip(path_weights(alts[0]["paths"]), alts[0]["paths"]):
                for t in p["transfer_stations"]:
                    transfer[t] = transfer.get(t, 0.0) + w
    spill = {s: spill_targets(net, effect, s) for s, kind in sources.items() if kind == "unserved"}
    return {"sources": sources, "spill": spill, "transfer": transfer, "crossing_pair": crossing_pair}


def exceed_probability(threshold: float, q_levels: list[float], q_values: list[float]) -> float:
    """P(X > threshold) from a quantile grid, linear interpolation of the CDF.
    Coarse in the tails (clamped to the outermost grid levels)."""
    import numpy as np

    vals = np.maximum.accumulate(np.asarray(q_values, float))  # enforce monotone
    cdf = float(np.interp(threshold, vals, q_levels, left=0.0, right=1.0))
    return round(1.0 - cdf, 3)
