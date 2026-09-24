"""Inventory of every resource the system uses and whether it is working — Cognee, the knowledge base, the knowledge graph (SQLite + Neo4j),
the MCP servers and their tools, the models, the datasets and the TabPFN checkpoints.

    make resources          # printed report (also used by the dashboard's Resources page)

Every probe is read-only, short-timeout and failure-tolerant: a resource that is down is reported as down, never raised.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "agent")]


def _cognee() -> dict:
    import knowledge

    c = knowledge.cognee()
    out: dict = {"configured": c.configured, "base_url": c.base.split("//")[-1][:40] + "…" if c.base else None}
    if not c.configured:
        return out
    t = time.time()
    h = c.health()
    out["healthy"] = bool(h)
    out["health_ms"] = round((time.time() - t) * 1000)
    if not h:
        out["error"] = c.last_error
        return out
    out["version"] = h.get("version")
    out["components"] = {k: {"status": v.get("status"), "provider": v.get("provider")} for k, v in (h.get("components") or {}).items()}
    ds = c._req("GET", "/api/v1/datasets/") or []
    summ = {x["datasetId"]: x for x in (c._req("GET", "/api/v1/datasets/graph-summary") or [])}
    datasets = []
    for d in ds:
        items = c._req("GET", f"/api/v1/datasets/{d['id']}/data") or []
        datasets.append({"name": d["name"], "id": d["id"][:8], "documents": len(items), "graph_nodes": (summ.get(d["id"]) or {}).get("numNodes"), "graph_edges": (summ.get(d["id"]) or {}).get("numEdges")})
    out["datasets"] = datasets
    main = next((d for d in ds if d["name"] == knowledge.DATASET), None)
    if main:
        g = c._req("GET", f"/api/v1/datasets/{main['id']}/graph", timeout=40) or {}
        types: dict = {}
        for n in g.get("nodes", []):
            types[n.get("type")] = types.get(n.get("type"), 0) + 1
        out["knowledge_graph"] = {"dataset": knowledge.DATASET, "nodes": len(g.get("nodes", [])), "edges": len(g.get("edges", [])), "node_types": dict(sorted(types.items(), key=lambda kv: -kv[1])[:8])}
        out["status"] = c.status()
    ss = c._req("GET", "/api/v1/sessions") or {}
    out["sessions"] = {"total": ss.get("total"), "recent": [{"label": (s.get("label") or "")[:70], "messages": s.get("msg_count"), "status": s.get("status")} for s in (ss.get("sessions") or [])[:5]]}
    out["quota"] = c._req("GET", "/api/v1/quotas/usage")
    return out


def _mcp_tools() -> dict:
    """Tool names per MCP server, read from the source (no import: importing the data server would start its warm-up)."""
    out = {}
    for name, f in (("ubahn-flow-data", ["server.py", "disruption_tools.py", "analytics_tools.py"]), ("nextmove-knowledge", ["knowledge_server.py"])):
        tools = []
        for fn in f:
            tree = ast.parse((REPO / "mcp_server" / fn).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and any((isinstance(d, ast.Attribute) and d.attr == "tool") or (isinstance(d, ast.Name) and d.id == "tool") for d in node.decorator_list):
                    doc = (ast.get_docstring(node) or "").split("\n")[0][:90]
                    tools.append({"tool": node.name, "does": doc})
        out[name] = tools
    return out


def _docker(name: str) -> str | None:
    try:
        r = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Status}}"], capture_output=True, text=True, timeout=6)
        return r.stdout.strip() or None
    except Exception:
        return None


def _data() -> list[dict]:
    rows = []
    for p in sorted((REPO / "data").rglob("*.csv")):
        rows.append({"file": str(p.relative_to(REPO / "data")), "mb": round(p.stat().st_size / 1e6, 2)})
    return rows


def collect(cognee_deep: bool = True) -> dict:
    import kgraph
    import knowledge
    from llm_config import litellm_params

    kb = knowledge.kb()
    g = kgraph.kg()
    res: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    res["cognee"] = _cognee() if cognee_deep else {"configured": knowledge.cognee().configured}
    res["knowledge_base"] = {**kb.stats(), "boundaries": [e["id"] for e in kb.entries if e["kind"] == "boundary"], "ground_truth_sample": [e["id"] for e in kb.entries if e["kind"] == "ground_truth" and not e["id"].startswith("GT-CL")]}
    res["knowledge_graph_local"] = g.stats()
    res["neo4j"] = {**g.neo4j_stats(), "container": _docker("nextmove-neo4j")}
    res["mcp_servers"] = _mcp_tools()
    res["models"] = {role: (litellm_params(role) or ("not set", {}))[0] for role in ("SUPERVISOR", "EVALUATOR", "WRITER", "ROUTER", "WORKER")}
    res["datasets"] = _data()
    ck = REPO / "ml" / "checkpoints" / "manifest.json"
    man = json.loads(ck.read_text()) if ck.exists() else []
    res["tabpfn_checkpoints"] = [m if isinstance(m, str) else (m.get("name") or m.get("checkpoint") or str(m)[:40]) for m in (man.keys() if isinstance(man, dict) else man)]
    res["env_keys_present"] = {k: bool(os.environ.get(k)) for k in ("TABPFN_API_TOKEN", "OPENAI_API_KEY", "COGNEE_API_KEY", "NEO4J_PASSWORD", "LANGSMITH_API_KEY", "SUPERVISOR_API_KEY")}
    return res


def main() -> None:
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    r = collect()
    ok = lambda b: "OK " if b else "DOWN"
    c = r["cognee"]
    print(f"\nRESOURCES  ({r['generated']})\n")
    print(f"Cognee Cloud   [{ok(c.get('healthy'))}] configured={c['configured']} {c.get('base_url') or ''}  v{c.get('version')}  health {c.get('health_ms')} ms")
    for k, v in (c.get("components") or {}).items():
        print(f"    {k:14s} {v['status']:9s} {v['provider']}")
    for d in c.get("datasets", []):
        print(f"    dataset {d['name']:16s} {d['documents']:>3} documents · graph {d['graph_nodes']} nodes / {d['graph_edges']} edges (summary)")
    if c.get("knowledge_graph"):
        kg_ = c["knowledge_graph"]
        print(f"    Cognee knowledge graph of '{kg_['dataset']}': {kg_['nodes']} nodes, {kg_['edges']} edges · types {kg_['node_types']} · pipeline {c.get('status')}")
    if c.get("sessions"):
        print(f"    sessions {c['sessions']['total']} · quota used {c['quota']['storageUsedInBytes'] / 1e3:.0f} KB of {c['quota']['storageLimitInBytes'] / 1e6:.0f} MB")
    kb = r["knowledge_base"]
    print(f"\nKnowledge base [OK ] {kb['entries']} entries {kb['kinds']} · {kb['turns']} stored turns in {kb['sessions']} sessions")
    lg = r["knowledge_graph_local"]
    print(f"Knowledge graph, local SQLite [OK ] {lg['nodes']} nodes, {lg['edges']} relationships · problems by source {lg['problems_by_source']}")
    n = r["neo4j"]
    print(f"Knowledge graph, Neo4j        [{ok(n.get('reachable'))}] container: {n.get('container')}  browser {n.get('browser', '-')}  " + (f"{n['nodes']} nodes, {n['relationships']} relationships" if n.get("reachable") else str(n.get('error', 'not configured'))))
    if n.get("by_label"):
        print(f"    labels {n['by_label']}")
    print("\nMCP servers")
    for s, tools in r["mcp_servers"].items():
        print(f"    {s}: {len(tools)} tools — " + ", ".join(t["tool"] for t in tools))
    print("\nModels   " + " · ".join(f"{k} {v}" for k, v in r["models"].items()))
    print(f"Datasets {len(r['datasets'])} files · TabPFN checkpoints: {', '.join(r['tabpfn_checkpoints']) or 'none'}")
    print("Keys     " + " · ".join(f"{k} {'set' if v else 'MISSING'}" for k, v in r["env_keys_present"].items()) + "\n")


if __name__ == "__main__":
    main()
