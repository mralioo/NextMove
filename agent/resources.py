"""Inventory of every resource the system uses and whether it is working — Cognee, the knowledge base, the knowledge graph (SQLite + Neo4j),
the MCP servers and their tools, the models, the datasets and the TabPFN checkpoints.

    ./.venv/bin/python scripts/tasks.py resources          # printed report (also used by the dashboard's Resources page)

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


def _service_state() -> dict:
    f = REPO / ".run" / "services.json"
    try:
        st = json.loads(f.read_text())
    except Exception:
        return {}
    alive = {}
    for n, v in st.items():
        if v.get("docker"):
            alive[n] = v
            continue
        try:
            os.kill(v["pid"], 0)
            alive[n] = v
        except OSError:
            pass
    return alive


def _port_open(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def graph_snapshot_path() -> Path:
    return REPO / ".run" / "cognee_graph.html"


def links() -> list[dict]:
    """Every UI / dashboard / API page that can be opened for the resources, with whether it is reachable now.
    kind: ui (a web app) · api (interactive API docs) · snapshot (a local file) · command (something to run, no link)."""
    import knowledge

    st = _service_state()
    dash = st.get("dashboard", {}).get("port")
    adk = st.get("adk-web", {}).get("port")
    kn = st.get("mcp-knowledge", {}).get("port")
    opi = st.get("operator-api", {}).get("port")
    neo_up = _port_open(7474)
    c = knowledge.cognee()
    out: list[dict] = []

    def add(resource, label, url, kind, up, note=""):
        out.append({"resource": resource, "label": label, "url": url, "kind": kind, "available": bool(up), "note": note})

    for label, path in (("Dashboard", ""), ("Agent Workflow page", "/workflow"), ("Resources page", "/resources"), ("Observability (traces)", "/observability"), ("Evaluation", "/evaluation")):
        add("Dashboard (Streamlit)", label, f"http://localhost:{dash}{path}" if dash else None, "ui", dash and _port_open(dash), "" if dash else "not running: make up")
    add("ADK agent UI", "Chat with the agent · events · traces · Evals tab (eval_set_1)", f"http://localhost:{adk}/dev-ui/?app=agent" if adk else None, "ui", adk and _port_open(adk), "" if adk else "not running: make up")
    add("ADK agent UI", "ADK API documentation (Swagger)", f"http://localhost:{adk}/docs" if adk else None, "api", adk and _port_open(adk))
    add("Neo4j (knowledge graph)", "Neo4j Browser — explore the graph, run Cypher (user neo4j, password in .env)", "http://localhost:7474/browser/?connectURL=neo4j%3A%2F%2Flocalhost%3A7687", "ui", neo_up,
        "" if neo_up else "not running: ./.venv/bin/python scripts/tasks.py neo4j-up")
    snap = graph_snapshot_path()
    add("Cognee Cloud (memory)", "Cognee web app — memory, datasets, graph (sign-in required)", "https://platform.cognee.ai", "ui", c.configured, "" if c.configured else "COGNEE_* not set")
    add("Cognee Cloud (memory)", "Cognee API documentation for your tenant (Swagger)", f"{c.base}/docs" if c.base else None, "api", c.configured)
    add("Cognee Cloud (memory)", "Cognee API reference (ReDoc)", f"{c.base}/redoc" if c.base else None, "api", c.configured)
    add("Cognee Cloud (memory)", "Cognee knowledge-graph visualisation (local snapshot of the tenant's page)", str(snap), "snapshot", snap.exists(),
        "" if snap.exists() else "create it: ./.venv/bin/python scripts/tasks.py cognee-graph (or the button below)")
    add("Operator API (feedback loop)", "Interactive API documentation (Swagger) — score, action report, turns, precedents", f"http://127.0.0.1:{opi}/docs" if opi else None, "api", opi and _port_open(opi),
        "" if opi else "not running: make up")
    add("Knowledge MCP server", "MCP endpoint (for MCP clients; not a web page)", f"http://127.0.0.1:{kn}/mcp" if kn else None, "api", kn and _port_open(kn), "" if kn else "not running: make up")
    add("Knowledge MCP server", "Inspect the tools in a browser", "npx @modelcontextprotocol/inspector", "command", True, f"then connect to http://127.0.0.1:{kn or 8766}/mcp (streamable HTTP)")
    add("LangSmith", "LangSmith (only if you run `./.venv/bin/python scripts/tasks.py ls-eval --upload`)", "https://smith.langchain.com", "ui", bool(os.environ.get("LANGSMITH_API_KEY")), "" if os.environ.get("LANGSMITH_API_KEY") else "LANGSMITH_API_KEY not set: evaluations stay offline")
    return out


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
    res["links"] = links()
    res["env_keys_present"] = {k: bool(os.environ.get(k)) for k in ("TABPFN_API_TOKEN", "OPENAI_API_KEY", "COGNEE_API_KEY", "NEO4J_PASSWORD", "LANGSMITH_API_KEY", "SUPERVISOR_API_KEY")}
    return res


def save_graph_snapshot() -> tuple[bool, str]:
    """Fetch Cognee's interactive graph page for the dataset and save it under .run/ (the tenant's page needs the API key, so it is kept as a local file)."""
    import knowledge

    html = knowledge.cognee().visualize()
    if not html:
        return False, f"could not fetch the Cognee graph page: {knowledge.cognee().last_error or 'Cognee off'}"
    p = graph_snapshot_path()
    p.parent.mkdir(exist_ok=True)
    p.write_text(html)
    return True, f"{p} ({len(html) / 1e6:.1f} MB) — open it in a browser, or see the Resources page"


def main() -> None:
    from env_loader import load_all_dotenvs

    load_all_dotenvs()
    if len(sys.argv) > 1 and sys.argv[1] == "graph":
        print(save_graph_snapshot()[1])
        return
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
    print("\nOpen it (UIs and dashboards)")
    for l in r["links"]:
        mark = "OK " if l["available"] else "-- "
        print(f"  [{mark}] {l['resource']:26s} {l['label'][:70]}\n         {l['url'] or '(unavailable)'}" + (f"   ({l['note']})" if l["note"] else ""))
    print("\nModels   " + " · ".join(f"{k} {v}" for k, v in r["models"].items()))
    print(f"Datasets {len(r['datasets'])} files · TabPFN checkpoints: {', '.join(r['tabpfn_checkpoints']) or 'none'}")
    print("Keys     " + " · ".join(f"{k} {'set' if v else 'MISSING'}" for k, v in r["env_keys_present"].items()) + "\n")


if __name__ == "__main__":
    main()
