"""Seed the knowledge graph (cold start): history problems, their solutions and the actions taken.

    ./.venv/bin/python scripts/tasks.py kg-seed            # categories + 26 recorded closures + accepted training/challenge answers + question bank + LLM extraction
    ./.venv/bin/python scripts/tasks.py kg-seed --no-llm
    ./.venv/bin/python scripts/tasks.py kg-export          # knowledge/kg_export.cypher + knowledge/kg_csv/ for Neo4j / the LLM Graph Builder

Sources, each tagged on the Problem node (`source`):
  seed:closure   the 26 recorded closures — problem = the closure, solution = reason/duration + reroute (rail detour / replacement bus) + actions,
                 computed with the MCP tools apply_closure / alternate_paths (no ML)
  seed:training  ACCEPTED answers of the organiser's training questions from the stored evaluation runs (sanity check passed)
  seed:challenge the accepted answers to the problem statement's challenge questions
  seed:bank      the question bank (docs/test_questions.md): the question and what a good answer contains (guidance, no answer yet)
  llm-graph-builder  entities and relations extracted from the knowledge-base texts and accepted answers by LangChain's LLMGraphTransformer
The graph then grows at run time: every accepted answer adds/reinforces a case (agent/fast_agent.py).
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO)]


def _strip(a: str) -> str:
    return (a or "").split("\n**Sources:**")[0]


async def seed_closures(g) -> int:
    import executor
    import knowledge
    from mcp_runtime import get_runtime
    from schemas import Entities, GraphCase
    from kgraph import actions_from_facts

    rt = get_runtime()
    n = 0
    for e in knowledge.kb().entries:
        if not e["id"].startswith("GT-CL-"):
            continue
        v = e["value"]
        i = v["closure_id"]
        ap, alt = await asyncio.gather(rt.call("apply_closure", closure_id=i), rt.call("alternate_paths", closure_id=i, max_paths=2))
        if not isinstance(ap, dict) or "error" in ap:
            continue
        cl = ap.get("closure", {})
        facts = executor._compact_c(cl, ap, alt if isinstance(alt, dict) else {}, {"error": "seed"}, None)
        facts.pop("scenario_error", None)
        acts, opts = actions_from_facts(facts)
        line = cl.get("line") or ""
        a, b = executor.short(cl.get("from_station")), executor.short(cl.get("to_station"))
        st = executor.short(cl.get("station"))
        where = f"{line} between {a} and {b}" if a else f"station {st}"
        q = f"{where} is suspended on {v['start'][:10]} ({v['reason']}). Why, how long, how do we reroute and which stations are affected?"
        ans = (f"Closed for {v['reason']} from {v['start'][:16]} for {v['hours']:.2f} h. " + " ".join(facts.get("reroute") or []) +
               (f" No service at: {', '.join(facts['cut']['unserved'])}." if (facts.get("cut") or {}).get("unserved") else ""))
        ents = Entities(lines=[line] if line else [], stations=[s for s in (cl.get("from_station"), cl.get("to_station"), cl.get("station")) if s], dates=[v["start"][:10]])
        g.record_case(GraphCase(question=q, category="C", entities=ents, answer=ans, confidence=0.95, verdict="accept", actions=acts, options=opts, source="seed:closure"))
        n += 1
    return n


def seed_from_runs(g) -> dict:
    """Accepted answers from stored evaluation runs (latest run of each suite)."""
    import knowledge
    from schemas import Entities, GraphCase
    from kgraph import actions_from_facts

    kb = knowledge.kb()
    conn = sqlite3.connect(REPO / "observability" / "agent_obs.db")
    conn.row_factory = sqlite3.Row
    out = {"training": 0, "challenge": 0}
    for suite in ("training", "challenge"):
        ids = [r[0] for r in conn.execute("SELECT eval_id FROM eval_runs WHERE suite=? ORDER BY ts DESC LIMIT 3", (suite,))]
        seen = set()
        for eid in ids:
            for r in conn.execute("SELECT i.item_id, i.question, i.answer, i.run_id FROM eval_items i WHERE i.eval_id=?", (eid,)):
                if r["item_id"] in seen or not r["answer"]:
                    continue
                run = conn.execute("SELECT plan_json, facts_json, timing_json FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()
                if not run:
                    continue
                t, f, p = json.loads(run["timing_json"] or "{}"), json.loads(run["facts_json"] or "{}"), json.loads(run["plan_json"] or "{}")
                if f.get("status") not in ("ok", "multi") or (t.get("sanity") or {}).get("ok") is False:
                    continue
                seen.add(r["item_id"])
                acts, opts = actions_from_facts(f) if f.get("status") == "ok" else ([], [])
                g.record_case(GraphCase(question=r["question"], category=p.get("cat", "?"), entities=Entities.from_legacy(p), answer=_strip(r["answer"]), confidence=t.get("confidence"),
                                        verdict=t.get("verdict") or "accept", actions=acts, options=opts, source=f"seed:{suite}"))
                out[suite] += 1
    return out


def seed_bank(g) -> int:
    from schemas import Entities, GraphCase

    n = 0
    for line in (REPO / "docs" / "test_questions.md").read_text().splitlines():
        m = re.match(r"\|\s*([A-HXR]\d+)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if not m or m.group(2).startswith("Follow"):
            continue
        qid, kind, q, exp = m.groups()
        cat = qid[0] if qid[0] in "ABCDEFGHX" else "OOS"
        g.record_case(GraphCase(question=re.sub(r"[*`]", "", q), category=cat, entities=Entities(), answer="Guidance: " + re.sub(r"[*`]", "", exp)[:600], verdict="expected", source="seed:bank"))
        n += 1
    return n


def seed_llm(g, limit: int = 22) -> dict:
    import knowledge

    kb = knowledge.kb()
    texts = [e["text"] for e in kb.entries if e["kind"] in ("boundary", "insight")] + [e["text"] for e in kb.entries if e["id"] in ("GT-E-RANK", "GT-F-TOP5", "GT-A-UBER", "GT-U8", "GT-D-RUDOW")]
    with g._conn() as c:
        rows = c.execute("SELECT json_extract(props,'$.text') FROM kg_nodes WHERE label='Answer' AND json_extract(props,'$.source') IN ('seed:training','seed:challenge') LIMIT 6").fetchall()
    texts += [r[0] for r in rows if r[0]]
    return g.extract_with_llm(texts[:limit], source="knowledge-base")


def main() -> None:
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    import specialists
    from kgraph import kg

    g = kg()
    for cat, sp in specialists.SPECIALISTS.items():
        g.node("Category", cat, name=sp.title, role=sp.role, status=sp.status)
    print("categories:", len(specialists.SPECIALISTS))
    print("closures:", asyncio.run(seed_closures(g)))
    print("runs:", seed_from_runs(g))
    print("bank:", seed_bank(g))
    if "--no-llm" not in sys.argv:
        print("llm graph builder:", seed_llm(g))
    (REPO / "knowledge").mkdir(exist_ok=True)
    print("cypher statements:", g.export_cypher(REPO / "knowledge" / "kg_export.cypher"))
    print(json.dumps(g.stats(), indent=1))


if __name__ == "__main__":
    main()
