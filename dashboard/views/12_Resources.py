import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui import page_header

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "agent", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="Resources", page_icon="🧰", layout="wide")
page_header("Resources — what is connected and working",
            "Live inventory of Cognee, the knowledge base, the knowledge graph (local and Neo4j), the MCP servers and their tools, the models, the datasets and the TabPFN checkpoints. "
            "Every probe is read-only; a resource that is down is shown as down.")

try:
    from env_loader import load_all_dotenvs
    load_all_dotenvs()
    import resources as RS
except Exception as e:
    st.error(f"The agent modules could not be imported ({type(e).__name__}: {e}).")
    st.stop()


@st.cache_data(show_spinner="Probing Cognee, Neo4j and the stores …", ttl=60)
def collect() -> dict:
    return RS.collect()


if st.button("Refresh now"):
    collect.clear()
r = collect()
st.caption(f"Probed {r['generated']} (cached for 60 s).")
c, n, kb, lg = r["cognee"], r["neo4j"], r["knowledge_base"], r["knowledge_graph_local"]

m = st.columns(5)
m[0].metric("Cognee Cloud", "healthy" if c.get("healthy") else ("off" if not c.get("configured") else "DOWN"), help=f"health call {c.get('health_ms')} ms")
m[1].metric("Neo4j", "reachable" if n.get("reachable") else ("not configured" if not n.get("configured") else "DOWN"), help=str(n.get("container")))
m[2].metric("Knowledge base", f"{kb['entries']} entries")
m[3].metric("Graph (local / Neo4j)", f"{lg['nodes']} / {n.get('nodes', '–')} nodes")
m[4].metric("MCP tools", sum(len(v) for v in r["mcp_servers"].values()))

# ------------------------------------------------------------------------------------------ Cognee
st.header("Cognee Cloud")
st.markdown("Session memory (every answered turn is mirrored as a QA entry) and a knowledge graph Cognee built from our curated ground truth, boundaries and insights.")
if c.get("healthy"):
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Service** — version {c.get('version')}, tenant `{c.get('base_url')}`")
        st.dataframe(pd.DataFrame([{"Component": k, "Status": v["status"], "Provider": v["provider"]} for k, v in c["components"].items()]), use_container_width=True, hide_index=True)
        st.markdown("**Datasets**")
        st.dataframe(pd.DataFrame(c["datasets"]).rename(columns={"graph_nodes": "graph nodes (summary)", "graph_edges": "graph edges (summary)"}), use_container_width=True, hide_index=True)
    with b:
        g = c.get("knowledge_graph")
        if g:
            st.markdown(f"**Knowledge graph of dataset `{g['dataset']}`** — {g['nodes']} nodes, {g['edges']} edges (pipeline: {c.get('status')})")
            st.plotly_chart(px.bar(pd.DataFrame(list(g["node_types"].items()), columns=["Node type", "Count"]), x="Node type", y="Count", color="Node type").update_layout(height=280, showlegend=False, margin=dict(t=10, b=10)), use_container_width=True)
        s = c.get("sessions") or {}
        st.markdown(f"**Sessions** — {s.get('total')} stored")
        st.dataframe(pd.DataFrame(s.get("recent", [])), use_container_width=True, hide_index=True)
        if c.get("quota"):
            st.caption(f"Storage used: {c['quota']['storageUsedInBytes'] / 1e3:.0f} KB of {c['quota']['storageLimitInBytes'] / 1e6:.0f} MB")
else:
    st.warning(f"Cognee is not reachable or not configured. {c.get('error') or 'Set COGNEE_ENABLED, COGNEE_API_BASE_URL and COGNEE_API_KEY in .env.'}")

# ------------------------------------------------------------------------------------------ knowledge base
st.header("Knowledge base (local, mirrored to Cognee)")
st.markdown(f"{kb['entries']} entries — {kb['kinds']} — plus {kb['turns']} stored turns in {kb['sessions']} sessions. Boundaries: " + ", ".join(f"`{i}`" for i in kb["boundaries"]))
st.caption("Ground truth is recomputed from the raw CSVs by `make kb-build`; `make kb-sync` pushes it to Cognee.")

# ------------------------------------------------------------------------------------------ knowledge graph
st.header("Knowledge graph — local SQLite copy and Neo4j")
st.markdown("Problem → Answer → Action / Option, linked to stations, lines, venues and events. SQLite is the source of truth; every accepted answer is also written to Neo4j by a background mirror.")
a, b = st.columns(2)
with a:
    st.markdown(f"**Local (SQLite + NetworkX)** — {lg['nodes']} nodes, {lg['edges']} relationships")
    st.dataframe(pd.DataFrame(list(lg["by_label"].items()), columns=["Node type", "Count"]), use_container_width=True, hide_index=True)
with b:
    if n.get("reachable"):
        st.markdown(f"**Neo4j** — {n['nodes']} nodes, {n['relationships']} relationships · container: {n.get('container')}")
        st.dataframe(pd.DataFrame(list(n["by_label"].items()), columns=["Node type", "Count"]), use_container_width=True, hide_index=True)
        st.link_button("Open the Neo4j browser (user neo4j)", n.get("browser", "http://localhost:7474"))
        st.code("MATCH (p:Problem)-[:ANSWERED_BY]->(a:Answer)-[:RECOMMENDS]->(x:Action)\nRETURN p.text, x.text LIMIT 25", language="cypher")
        if n.get("top_actions"):
            st.markdown("**Most recommended actions (from Neo4j)**")
            st.dataframe(pd.DataFrame(n["top_actions"]), use_container_width=True, hide_index=True)
    else:
        st.warning(f"Neo4j is not reachable ({n.get('error') or 'not configured'}). Start it with `make neo4j-up`, then `make neo4j-sync`.")
if n.get("reachable") and (n["nodes"] != lg["nodes"] or n["relationships"] != lg["edges"]):
    st.info("The two copies differ in size: run `make neo4j-sync` to bring Neo4j level with the local graph.")

# ------------------------------------------------------------------------------------------ MCP
st.header("MCP servers and tools")
for server, tools in r["mcp_servers"].items():
    with st.expander(f"{server} — {len(tools)} tools", expanded=False):
        st.dataframe(pd.DataFrame(tools).rename(columns={"tool": "Tool", "does": "What it does"}), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------------------ rest
st.header("Models, datasets, checkpoints, keys")
x, y = st.columns(2)
with x:
    st.markdown("**Models per role**")
    st.dataframe(pd.DataFrame(list(r["models"].items()), columns=["Role", "Model"]), use_container_width=True, hide_index=True)
    st.markdown("**TabPFN checkpoints** — " + ", ".join(f"`{c_}`" for c_ in r["tabpfn_checkpoints"]))
with y:
    st.markdown("**API keys present** (values are never shown)")
    st.dataframe(pd.DataFrame([{"Key": k, "Present": "yes" if v else "MISSING"} for k, v in r["env_keys_present"].items()]), use_container_width=True, hide_index=True)
st.markdown("**Data files**")
st.dataframe(pd.DataFrame(r["datasets"]), use_container_width=True, hide_index=True)
