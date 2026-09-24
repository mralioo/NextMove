"""A minimal LOCAL knowledge graph that maps problems (operator questions) to answers, the actions taken and the options offered — and grows.

Modelled on the Neo4j LLM Knowledge Graph Builder (https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/): documents are turned into a property
graph of typed nodes and relationships by an LLM (`LLMGraphTransformer`, the same engine the Builder uses), with a restricted schema, provenance
(`Document` nodes, `FROM_DOCUMENT` edges) and a graph that can be exported to Neo4j. What is different: it needs no server — nodes and edges live in
SQLite (observability/kgraph.db) with NetworkX for traversal — and the main way it grows is deterministic: every ACCEPTED answer adds a case.

Schema (see schemas.NODE_LABELS / REL_TYPES):

    (Problem)-[:IN_CATEGORY]->(Category)          (Problem)-[:INVOLVES]->(Station|Line|Venue|Event)
    (Problem)-[:ANSWERED_BY {verdict, confidence}]->(Answer)
    (Answer)-[:RECOMMENDS {weight}]->(Action)-[:AT]->(Station)        (Answer)-[:OFFERS_OPTION]->(Option)
    (Answer)-[:SUPPORTED_BY]->(Concept)            (Document)-[:FROM_DOCUMENT]-(*)   (LLM-extracted entities: label Concept or a schema label)

Edge `weight` counts how often that link was accepted, so frequently chosen actions rank first. `similar()` is what the evaluator uses to compare a new
result with what was answered and done before. `export_cypher()` / `export_csv()` write the graph for Neo4j (or the Graph Builder's import).
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KG_DB = Path(os.environ.get("TMT_KG_DB", REPO / "observability" / "kgraph.db"))
_STOP = set("the a an of to in on at is are was were be by for and or what which who how why when where do does did will would can could should this that it its as with from".split())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9äöüß ]", " ", s.lower())).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if len(t) > 2 and t not in _STOP}


def _short(n: str) -> str:
    return re.sub(r"^(S\+U|U|S)\s+", "", re.sub(r"\s*\(Berlin\)\s*$", "", n)).strip()


def _h(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


# ----------------------------------------------------------------------------------------------- actions from facts
def actions_from_facts(facts: dict) -> tuple[list[str], list[str]]:
    """(actions, options) an accepted answer implies — deterministic, straight from the specialist's facts."""
    acts: list[str] = []
    opts: list[str] = []
    cat = facts.get("cat")
    if cat == "C" and facts.get("status") == "ok":
        for r in (facts.get("press") or [])[:3]:
            acts.append(f"Deploy additional staff at {r['s']} around {r['at']}")
        un = (facts.get("cut") or {}).get("unserved") or []
        if un:
            acts.append("Inform passengers that " + ", ".join(un[:4]) + " have no service")
        for r in facts.get("reroute") or []:
            (acts if r.startswith("no rail detour") else opts).append(("Arrange " + re.sub(r" would be needed$", "", r.split("; ", 1)[-1])) if r.startswith("no rail detour") else "Use " + r)
        for a, b, km in (facts.get("alt") or {}).get("bus", [])[:2]:
            opts.append(f"Replacement bus between {a} and {b} (about {km} km)")
    elif cat == "A" and facts.get("status") == "ok":
        for a in facts.get("act") or []:
            acts.append(f"Put staff at {a['s']} from {a['from']} to {a['to']} after the event ends")
    elif cat == "P" and facts.get("status") == "ok":
        for r in facts.get("top") or []:
            acts.append(f"Pre-position staff and monitor {r['s']} (highest predicted load)")
    elif cat == "F" and facts.get("status") == "ok":
        for r in facts.get("top") or []:
            opts.append(f"If {r['s']} closes, bypass via " + ", ".join(r.get("nbr", [])[:2]))
    elif cat == "B" and facts.get("status") == "ok":
        for r in facts.get("found") or []:
            acts.append(f"Check {r['s']} around {r['at']} (anomaly)")
    return list(dict.fromkeys(acts)), list(dict.fromkeys(opts))


# ----------------------------------------------------------------------------------------------- the graph
class KnowledgeGraph:
    def __init__(self, path: Path | str = KG_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._neo_q: queue.Queue | None = None
        self._mirrors = Path(path).resolve() == Path(KG_DB).resolve()      # only the production graph is mirrored (a test / scratch graph must never write to Neo4j)
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS kg_nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, key TEXT, props TEXT, ts REAL, UNIQUE(label, key));
            CREATE TABLE IF NOT EXISTS kg_edges (id INTEGER PRIMARY KEY AUTOINCREMENT, src INTEGER, rel TEXT, dst INTEGER, props TEXT, weight REAL DEFAULT 1, ts REAL, UNIQUE(src, rel, dst));
            CREATE INDEX IF NOT EXISTS kg_n_label ON kg_nodes(label);
            CREATE INDEX IF NOT EXISTS kg_e_src ON kg_edges(src);
            """)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=15)

    # -- writes
    def node(self, label: str, key: str, _keep: tuple = (), **props) -> int:
        """Upsert. Properties named in `_keep` are never overwritten once set (provenance such as `source`)."""
        with self._lock, self._conn() as c:
            row = c.execute("SELECT id, props FROM kg_nodes WHERE label=? AND key=?", (label, key)).fetchone()
            if row:
                old = json.loads(row[1] or "{}")
                merged = {**old, **{k: v for k, v in props.items() if v is not None and not (k in _keep and k in old)}}
                c.execute("UPDATE kg_nodes SET props=? WHERE id=?", (json.dumps(merged, default=str), row[0]))
                self._mirror("node", label, key, merged)
                return row[0]
            nid = c.execute("INSERT INTO kg_nodes (label, key, props, ts) VALUES (?,?,?,?)", (label, key, json.dumps(props, default=str), time.time())).lastrowid
            self._mirror("node", label, key, props)
            return nid

    def edge(self, src: int, rel: str, dst: int, weight: float = 1.0, **props) -> None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT id, weight FROM kg_edges WHERE src=? AND rel=? AND dst=?", (src, rel, dst)).fetchone()
            if row:
                c.execute("UPDATE kg_edges SET weight=?, props=? WHERE id=?", (row[1] + weight, json.dumps(props, default=str), row[0]))
                w_total = row[1] + weight
            else:
                c.execute("INSERT INTO kg_edges (src, rel, dst, props, weight, ts) VALUES (?,?,?,?,?,?)", (src, rel, dst, json.dumps(props, default=str), weight, time.time()))
                w_total = weight
            s_row = c.execute("SELECT label, key FROM kg_nodes WHERE id=?", (src,)).fetchone()
            d_row = c.execute("SELECT label, key FROM kg_nodes WHERE id=?", (dst,)).fetchone()
        if s_row and d_row:
            self._mirror("edge", s_row, rel, d_row, w_total)

    def record_case(self, case) -> str:
        """Add / reinforce one problem → answer → actions/options record. Returns the Problem key."""
        pk = _h(_norm(case.question), 12)
        cat = self.node("Category", case.category)
        prob = self.node("Problem", pk, _keep=("source",), text=case.question, category=case.category, source=case.source, params=case.entities.model_dump(exclude_none=True, exclude_defaults=True))
        with self._conn() as c:
            c.execute("UPDATE kg_nodes SET props=json_set(props, '$.n_asked', COALESCE(json_extract(props, '$.n_asked'), 0) + 1) WHERE id=?", (prob,))
        self.edge(prob, "IN_CATEGORY", cat)
        ans = self.node("Answer", _h(case.answer, 12), text=case.answer[:900], confidence=case.confidence, verdict=case.verdict, source=case.source)
        self.edge(prob, "ANSWERED_BY", ans, verdict=case.verdict, confidence=case.confidence)
        for a in case.actions:
            aid = self.node("Action", _norm(a)[:120], text=a)
            self.edge(ans, "RECOMMENDS", aid)
            for st in case.entities.stations:
                if _short(st).lower() in a.lower():
                    self.edge(aid, "AT", self.node("Station", _short(st)))
        for o in case.options:
            self.edge(ans, "OFFERS_OPTION", self.node("Option", _norm(o)[:120], text=o))
        for st in case.entities.stations:
            self.edge(prob, "INVOLVES", self.node("Station", _short(st)))
        for ln in case.entities.lines:
            self.edge(prob, "INVOLVES", self.node("Line", ln))
        if case.entities.venue:
            self.edge(prob, "INVOLVES", self.node("Venue", case.entities.venue))
        if case.entities.event:
            self.edge(prob, "INVOLVES", self.node("Event", case.entities.event))
        return pk

    # -- reads
    def stats(self) -> dict:
        with self._conn() as c:
            nodes = dict(c.execute("SELECT label, COUNT(*) FROM kg_nodes GROUP BY label").fetchall())
            edges = dict(c.execute("SELECT rel, COUNT(*) FROM kg_edges GROUP BY rel").fetchall())
            src = dict(c.execute("SELECT json_extract(props, '$.source'), COUNT(*) FROM kg_nodes WHERE label='Problem' GROUP BY 1").fetchall())
        return {"nodes": sum(nodes.values()), "edges": sum(edges.values()), "by_label": nodes, "by_rel": edges, "problems_by_source": src}

    def _problems(self) -> list[tuple[int, str, dict]]:
        with self._conn() as c:
            return [(i, k, json.loads(p or "{}")) for i, k, p in c.execute("SELECT id, key, props FROM kg_nodes WHERE label='Problem'")]

    def case(self, problem_id: int, key: str, props: dict, similarity: float = 1.0):
        from schemas import KGCase

        with self._conn() as c:
            ans = c.execute("SELECT n.id, n.props, e.weight FROM kg_edges e JOIN kg_nodes n ON n.id=e.dst WHERE e.src=? AND e.rel='ANSWERED_BY' ORDER BY e.weight DESC LIMIT 1", (problem_id,)).fetchone()
            acts, opts = [], []
            if ans:
                acts = [json.loads(p)["text"] for (p,) in c.execute("SELECT n.props FROM kg_edges e JOIN kg_nodes n ON n.id=e.dst WHERE e.src=? AND e.rel='RECOMMENDS' ORDER BY e.weight DESC LIMIT 4", (ans[0],))]
                opts = [json.loads(p)["text"] for (p,) in c.execute("SELECT n.props FROM kg_edges e JOIN kg_nodes n ON n.id=e.dst WHERE e.src=? AND e.rel='OFFERS_OPTION' ORDER BY e.weight DESC LIMIT 3", (ans[0],))]
        return KGCase(problem_id=key, problem=props.get("text", "")[:300], category=props.get("category", "?"), similarity=round(similarity, 2),
                      answer=(json.loads(ans[1]).get("text", "")[:400] if ans else ""), actions=acts, options=opts, times_accepted=int(ans[2]) if ans else 0)

    def similar(self, question: str, category: str = "", entities=None, k: int = 3, min_sim: float = 0.2) -> list:
        """The `k` most similar past problems (token overlap + category + shared stations/lines) with their answer, actions and options."""
        q = _tokens(question)
        ents = {_short(s).lower() for s in (entities.stations if entities else [])} | {x.lower() for x in (entities.lines if entities else [])}
        scored = []
        for pid, key, props in self._problems():
            t = _tokens(props.get("text", ""))
            jac = len(q & t) / len(q | t) if q | t else 0.0
            pe = {_short(s).lower() for s in (props.get("params", {}).get("stations") or [])} | {x.lower() for x in (props.get("params", {}).get("lines") or [])}
            ov = len(ents & pe) / len(ents | pe) if ents | pe else 0.0
            sim = 0.6 * jac + 0.15 * (1.0 if category and props.get("category") == category else 0.0) + 0.25 * ov
            if sim >= min_sim:
                scored.append((sim, pid, key, props))
        return [self.case(pid, key, props, sim) for sim, pid, key, props in sorted(scored, key=lambda x: -x[0])[:k]]

    def neighbors(self, label: str, key: str, limit: int = 40) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT id, props FROM kg_nodes WHERE label=? AND key=?", (label, key)).fetchone()
            if not row:
                return {}
            out = [(r, l, k2, w, "out") for r, l, k2, w in c.execute("SELECT e.rel, n.label, n.key, e.weight FROM kg_edges e JOIN kg_nodes n ON n.id=e.dst WHERE e.src=? LIMIT ?", (row[0], limit))]
            out += [(r, l, k2, w, "in") for r, l, k2, w in c.execute("SELECT e.rel, n.label, n.key, e.weight FROM kg_edges e JOIN kg_nodes n ON n.id=e.src WHERE e.dst=? LIMIT ?", (row[0], limit))]
        return {"node": {"label": label, "key": key, "props": json.loads(row[1] or "{}")}, "edges": [{"rel": r, "label": l, "key": k2, "weight": w, "dir": d} for r, l, k2, w, d in out]}

    def top_actions(self, category: str = "", n: int = 10) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("""SELECT a.props, SUM(e.weight) w FROM kg_nodes a JOIN kg_edges e ON e.dst=a.id AND e.rel='RECOMMENDS'
                                JOIN kg_nodes ans ON ans.id=e.src JOIN kg_edges pe ON pe.dst=ans.id AND pe.rel='ANSWERED_BY'
                                JOIN kg_nodes p ON p.id=pe.src WHERE a.label='Action' AND (?='' OR json_extract(p.props,'$.category')=?)
                                GROUP BY a.id ORDER BY w DESC LIMIT ?""", (category, category, n)).fetchall()
        return [{"action": json.loads(p)["text"], "weight": w} for p, w in rows]

    def to_networkx(self):
        import networkx as nx

        g = nx.MultiDiGraph()
        with self._conn() as c:
            for i, l, k, p in c.execute("SELECT id, label, key, props FROM kg_nodes"):
                g.add_node(i, label=l, key=k, **{kk: vv for kk, vv in json.loads(p or "{}").items() if isinstance(vv, (str, int, float))})
            for s, r, d, w in c.execute("SELECT src, rel, dst, weight FROM kg_edges"):
                g.add_edge(s, d, rel=r, weight=w)
        return g

    # -- export (Neo4j / LLM Graph Builder import)
    def export_cypher(self, path: Path | str) -> int:
        esc = lambda s: str(s).replace("\\", "\\\\").replace("'", "\\'")
        lines = ["// NextMove knowledge graph — Neo4j import (Cypher). Generated by agent/kgraph.py", "CREATE INDEX node_label_key IF NOT EXISTS FOR (n:Node) ON (n.label, n.key);"]
        with self._conn() as c:
            for l, k, p in c.execute("SELECT label, key, props FROM kg_nodes"):
                props = json.loads(p or "{}")
                flat = ", ".join(f"n.{re.sub(r'[^A-Za-z0-9_]', '_', kk)} = '{esc(json.dumps(vv) if isinstance(vv, (dict, list)) else vv)}'" for kk, vv in props.items() if vv is not None)
                lines.append(f"MERGE (n:Node:{l} {{key: '{esc(k)}'}}) SET n.label = '{l}'" + (f", {flat}" if flat else "") + ";")
            for sl, sk, r, dl, dk, w in c.execute("SELECT s.label, s.key, e.rel, d.label, d.key, e.weight FROM kg_edges e JOIN kg_nodes s ON s.id=e.src JOIN kg_nodes d ON d.id=e.dst"):
                lines.append(f"MATCH (a:{sl} {{key: '{esc(sk)}'}}), (b:{dl} {{key: '{esc(dk)}'}}) MERGE (a)-[r:{r}]->(b) SET r.weight = {w};")
        Path(path).write_text("\n".join(lines) + "\n")
        return len(lines) - 2

    # -- Neo4j (live mirror + full sync). The SQLite graph stays the source of truth; Neo4j is a second, queryable copy (browser, Cypher, GDS).
    @staticmethod
    def neo4j_configured() -> bool:
        return bool(os.environ.get("NEO4J_URI") and os.environ.get("NEO4J_PASSWORD"))

    def _driver(self):
        if getattr(self, "_drv", None) is None:
            from neo4j import GraphDatabase

            self._drv = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]), connection_timeout=4)
        return self._drv

    @staticmethod
    def _flat(props: dict) -> dict:
        """Neo4j properties must be primitives: nested values become JSON strings."""
        return {re.sub(r"[^A-Za-z0-9_]", "_", k): (v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str)) for k, v in props.items() if v is not None}

    def _mirror(self, kind: str, *a) -> None:
        """Queue one node / edge for the live Neo4j copy; a daemon thread writes it. Never blocks or breaks the agent; a Neo4j outage only pauses the mirror."""
        if not self._mirrors or not self.neo4j_configured() or os.environ.get("NEO4J_MIRROR", "on") == "off":
            return
        if self._neo_q is None:
            self._neo_q = queue.Queue(maxsize=5000)
            threading.Thread(target=self._neo_worker, daemon=True, name="neo4j-mirror").start()
        try:
            self._neo_q.put_nowait((kind, a))
        except queue.Full:
            pass

    def _neo_worker(self) -> None:
        while True:
            kind, a = self._neo_q.get()
            try:
                with self._driver().session() as ses:
                    if kind == "node":
                        label, key, props = a
                        ses.run(f"MERGE (n:Node:`{label}` {{key: $key}}) SET n.label = $label, n += $props", key=key, label=label, props=self._flat(props))
                    else:
                        (sl, sk), rel, (dl, dk), w = a
                        ses.run(f"MATCH (a:Node {{label: $sl, key: $sk}}), (b:Node {{label: $dl, key: $dk}}) MERGE (a)-[r:`{rel}`]->(b) SET r.weight = $w", sl=sl, sk=sk, dl=dl, dk=dk, w=w)
            except Exception:
                time.sleep(2)                                    # Neo4j down: drop this item, keep the thread alive

    def sync_to_neo4j(self, clear: bool = False) -> dict:
        """Load the whole graph into Neo4j (idempotent MERGE; clear=True empties the Neo4j database first). Batched UNWIND, ~1 s for the seeded graph."""
        if not self.neo4j_configured():
            return {"ok": False, "reason": "NEO4J_URI / NEO4J_PASSWORD not set"}
        try:
            drv = self._driver()
            with drv.session() as ses:
                if clear:
                    ses.run("MATCH (n) DETACH DELETE n")
                ses.run("CREATE INDEX node_label_key IF NOT EXISTS FOR (n:Node) ON (n.label, n.key)")
                by_label: dict = {}
                with self._conn() as c:
                    for l, k, p in c.execute("SELECT label, key, props FROM kg_nodes"):
                        by_label.setdefault(l, []).append({"key": k, "props": {**self._flat(json.loads(p or "{}")), "label": l}})
                    edges_by_rel: dict = {}
                    for sl, sk, r, dl, dk, w in c.execute("SELECT s.label, s.key, e.rel, d.label, d.key, e.weight FROM kg_edges e JOIN kg_nodes s ON s.id=e.src JOIN kg_nodes d ON d.id=e.dst"):
                        edges_by_rel.setdefault(r, []).append({"sl": sl, "sk": sk, "dl": dl, "dk": dk, "w": w})
                n_nodes = n_edges = 0
                for label, rows in by_label.items():
                    for i in range(0, len(rows), 500):
                        ses.run(f"UNWIND $rows AS r MERGE (n:Node:`{label}` {{key: r.key}}) SET n += r.props", rows=rows[i:i + 500])
                    n_nodes += len(rows)
                for rel, rows in edges_by_rel.items():
                    for i in range(0, len(rows), 500):
                        ses.run(f"UNWIND $rows AS r MATCH (a:Node {{label: r.sl, key: r.sk}}), (b:Node {{label: r.dl, key: r.dk}}) MERGE (a)-[e:`{rel}`]->(b) SET e.weight = r.w", rows=rows[i:i + 500])
                    n_edges += len(rows)
            return {"ok": True, "nodes": n_nodes, "relationships": n_edges}
        except Exception as e:
            return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:160]}"}

    def neo4j_stats(self) -> dict:
        """Live counts from the Neo4j server (nodes per label, relationships per type) — proves the copy is there and queryable."""
        if not self.neo4j_configured():
            return {"configured": False}
        try:
            with self._driver().session() as ses:
                nodes = {r["l"]: r["n"] for r in ses.run("MATCH (n:Node) RETURN n.label AS l, count(*) AS n ORDER BY n DESC")}
                rels = {r["t"]: r["n"] for r in ses.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC")}
                top = [dict(r) for r in ses.run("MATCH (a:Action)<-[r:RECOMMENDS]-() RETURN a.text AS action, sum(r.weight) AS w ORDER BY w DESC LIMIT 5")]
            return {"configured": True, "reachable": True, "nodes": sum(nodes.values()), "relationships": sum(rels.values()), "by_label": nodes, "by_rel": rels, "top_actions": top,
                    "browser": "http://localhost:7474"}
        except Exception as e:
            return {"configured": True, "reachable": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}

    push_to_neo4j = sync_to_neo4j          # older name

    # -- LLM extraction (the LLM Graph Builder step)
    def extract_with_llm(self, texts: list[str], source: str, model: str | None = None) -> dict:
        """Turn unstructured text into nodes/relationships with LangChain's LLMGraphTransformer (the engine of the Neo4j LLM Graph Builder), restricted to
        this schema, and merge them here with Document provenance. Uses the small worker model. Returns counts."""
        import warnings

        warnings.filterwarnings("ignore")
        from langchain_core.documents import Document
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        from langchain_openai import ChatOpenAI

        from schemas import NODE_LABELS

        model = model or (os.environ.get("WORKER_LITELLM_MODEL") or "gpt-4o-mini").split("/")[-1]
        tr = LLMGraphTransformer(llm=ChatOpenAI(model=model, temperature=0), allowed_nodes=["Problem", "Answer", "Action", "Option", "Station", "Line", "Venue", "Event", "Concept"],
                                 allowed_relationships=["RECOMMENDS", "OFFERS_OPTION", "INVOLVES", "AT", "SUPPORTED_BY", "RELATED_TO"], strict_mode=True)
        docs = tr.convert_to_graph_documents([Document(page_content=t, metadata={"source": source}) for t in texts])
        n_nodes = n_edges = 0
        for gd, text in zip(docs, texts):
            doc = self.node("Document", f"{source}:{_h(text, 8)}", text=text[:400], source=source)
            ids: dict = {}
            for nd in gd.nodes:
                label = nd.type if nd.type in NODE_LABELS else "Concept"
                ids[(nd.id, nd.type)] = self.node(label, _norm(str(nd.id))[:120] or "x", text=str(nd.id), source="llm-graph-builder")
                self.edge(doc, "FROM_DOCUMENT", ids[(nd.id, nd.type)])
                n_nodes += 1
            for rel in gd.relationships:
                s, d = ids.get((rel.source.id, rel.source.type)), ids.get((rel.target.id, rel.target.type))
                if s and d:
                    self.edge(s, re.sub(r"[^A-Z_]", "_", rel.type.upper()) or "RELATED_TO", d, source="llm-graph-builder")
                    n_edges += 1
        return {"documents": len(docs), "nodes": n_nodes, "edges": n_edges, "model": model}


_KG: KnowledgeGraph | None = None


def kg() -> KnowledgeGraph:
    global _KG
    if _KG is None:
        _KG = KnowledgeGraph()
    return _KG


if __name__ == "__main__":
    import sys

    sys.path[:0] = [str(Path(__file__).resolve().parent), str(REPO)]
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    g = kg()
    if cmd == "stats":
        print(json.dumps({**g.stats(), "top_actions": g.top_actions(n=8)}, indent=1, ensure_ascii=False))
    elif cmd == "neo4j-sync":
        from env_loader import load_all_dotenvs
        load_all_dotenvs()
        print(json.dumps(g.sync_to_neo4j(clear="--clear" in sys.argv), indent=1))
        print(json.dumps(g.neo4j_stats(), indent=1, default=str))
    elif cmd == "export":
        out = REPO / "knowledge" / "kg_export.cypher"
        out.parent.mkdir(exist_ok=True)
        print(f"{g.export_cypher(out)} statements -> {out.relative_to(REPO)}")
