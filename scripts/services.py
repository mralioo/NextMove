"""Start / stop / inspect every service of the system with one command (used by `make up`, `make down`, `./.venv/bin/python scripts/tasks.py status`, `./.venv/bin/python scripts/tasks.py logs`).

    services.py up [--with-data-mcp] [--no-adk]     prepare (knowledge base, graph) and start everything in the background
    services.py down                                stop everything this script started
    services.py status                              what is running, on which port, healthy or not
    services.py logs [name]                         tail the log of one service (default: all)

Services
    dashboard       Streamlit dashboard incl. the Agent Workflow page          http://localhost:8501 (next free port if taken)
    adk-web         ADK dev UI: chat with the agent, see every event / tool    http://localhost:8000
    mcp-knowledge   MCP server: ground truth, sanity check, history, graph     http://127.0.0.1:8766/mcp   (streamable HTTP)
    neo4j           Neo4j (Docker container nextmove-neo4j, made by `./.venv/bin/python scripts/tasks.py neo4j-up`)   bolt://localhost:7687, browser :7474
    mcp-data        MCP server: datasets, analytics, TabPFN tools (optional)    http://127.0.0.1:8765/mcp   (--with-data-mcp; the agent itself
                    starts its own stdio copy of this server, so this one is only for external MCP clients)

State: pid + port files in .run/, logs in .run/logs/. Nothing here needs Docker.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / ".run"
LOGS = RUN / "logs"
PY = str(REPO / ".venv" / "bin" / "python")
BIN = REPO / ".venv" / "bin"
if not Path(PY).exists():
    PY = sys.executable


def free_port(start: int) -> int:
    for p in range(start, start + 200):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError(f"no free port from {start}")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def state() -> dict:
    f = RUN / "services.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save(st: dict) -> None:
    RUN.mkdir(exist_ok=True)
    (RUN / "services.json").write_text(json.dumps(st, indent=1))


def healthy(name: str, port: int) -> bool:
    if name in ("dashboard", "adk-web"):
        try:
            return urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3).status < 500
        except Exception:
            return False
    with socket.socket() as s:                       # MCP servers speak MCP, not plain HTTP: an open port is the check
        return s.connect_ex(("127.0.0.1", port)) == 0


def docker_neo4j(action: str) -> str:
    """start | stop the local Neo4j container (created once by `./.venv/bin/python scripts/tasks.py neo4j-up`). Returns a status word."""
    try:
        have = subprocess.run(["docker", "ps", "-a", "--filter", "name=^nextmove-neo4j$", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=8).stdout.strip()
        if action == "start":
            if not have:
                return "missing (run: ./.venv/bin/python scripts/tasks.py neo4j-up)"
            subprocess.run(["docker", "start", "nextmove-neo4j"], capture_output=True, timeout=30)
            return "started"
        if have:
            subprocess.run(["docker", "stop", "nextmove-neo4j"], capture_output=True, timeout=40)
            return "stopped"
        return "absent"
    except Exception as e:
        return f"docker unavailable ({type(e).__name__})"


def prepare() -> None:
    """Things the agent needs before it can answer well: the knowledge base and the seeded knowledge graph."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if not (REPO / "knowledge" / "knowledge.json").exists():
        print("· building the knowledge base from the raw CSVs (./.venv/bin/python scripts/tasks.py kb-build) ...")
        subprocess.run([PY, "agent/knowledge_build.py"], cwd=REPO, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    kg = REPO / "observability" / "kgraph.db"
    if not kg.exists():
        print("· seeding the knowledge graph without the LLM step (./.venv/bin/python scripts/tasks.py kg-seed --no-llm) ...")
        subprocess.run([PY, "agent/kgraph_build.py", "--no-llm"], cwd=REPO, env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not (REPO / ".env").exists():
        print("! no .env found: the agent needs TABPFN_API_TOKEN and LLM keys (see README); the dashboard and MCP servers still start.")


def spawn(name: str, cmd: list[str], port: int, cwd: Path = REPO, env_extra: dict | None = None) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / f"{name}.log", "ab")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "DATA_DIR": str(REPO / "data"), **(env_extra or {})}
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {"pid": p.pid, "port": port, "cmd": " ".join(cmd[-3:])}


def up(with_data: bool, adk: bool, neo: bool = True) -> None:
    st = {k: v for k, v in state().items() if v.get("docker") or alive(v["pid"])}
    running = set(st)
    prepare()
    if "dashboard" not in running:
        port = free_port(int(os.environ.get("PORT", "8501")))
        st["dashboard"] = spawn("dashboard", [PY, "-m", "streamlit", "run", "app.py", "--server.port", str(port), "--server.headless", "true"], port, cwd=REPO / "dashboard")
    if adk and "adk-web" not in running:
        port = free_port(int(os.environ.get("ADK_PORT", "8000")))
        st["adk-web"] = spawn("adk-web", [str(BIN / "adk"), "web", "--port", str(port), "agent/"], port)
    if "mcp-knowledge" not in running:
        port = free_port(int(os.environ.get("MCP_KNOWLEDGE_PORT", "8766")))
        st["mcp-knowledge"] = spawn("mcp-knowledge", [PY, "mcp_server/knowledge_server.py"], port, env_extra={"MCP_TRANSPORT": "http", "MCP_PORT": str(port)})
    if neo and "neo4j" not in running:
        r = docker_neo4j("start")
        if r == "started":
            st["neo4j"] = {"pid": 0, "port": 7687, "cmd": "docker: nextmove-neo4j", "docker": True}
        else:
            print(f"! Neo4j not started: {r}")
    if with_data and "mcp-data" not in running:
        port = free_port(int(os.environ.get("MCP_DATA_PORT", "8765")))
        st["mcp-data"] = spawn("mcp-data", [PY, "mcp_server/server.py"], port, env_extra={"MCP_TRANSPORT": "http", "MCP_PORT": str(port)})
    save(st)
    print("waiting for the services to answer ...")
    deadline = time.time() + 90
    pending = dict(st)
    while pending and time.time() < deadline:
        for n, v in list(pending.items()):
            if healthy(n, v["port"]):
                del pending[n]
        time.sleep(1.5)
    status()
    if pending:
        print(f"\n! not answering yet: {', '.join(pending)} — see `./.venv/bin/python scripts/tasks.py logs` (the data MCP server needs ~30 s to warm up).")


def status() -> None:
    st = state()
    if not st:
        print("no services started by `make up`.")
        return
    urls = {"dashboard": "http://localhost:{p}   (page: Agent Workflow)", "adk-web": "http://localhost:{p}   (select the app 'agent')",
            "neo4j": "bolt://localhost:{p}   browser http://localhost:7474 (user neo4j)", "mcp-knowledge": "http://127.0.0.1:{p}/mcp   (MCP, streamable HTTP)", "mcp-data": "http://127.0.0.1:{p}/mcp   (MCP, streamable HTTP)"}
    print(f"\n  {'service':15s} {'pid':>7s}  {'state':9s} address")
    for n, v in st.items():
        ok = True if v.get("docker") else alive(v["pid"])
        s = "healthy" if ok and healthy(n, v["port"]) else ("starting" if ok else "stopped")
        print(f"  {n:15s} {v['pid']:>7d}  {s:9s} {urls.get(n, '').format(p=v['port'])}")
    print(f"\n  logs: {LOGS.relative_to(REPO)}/<service>.log  ·  stop everything: make down\n")


def down() -> None:
    st = state()
    for n, v in st.items():
        if v.get("docker"):
            print(f"{n}: {docker_neo4j('stop')} (data volume kept)")
            continue
        if alive(v["pid"]):
            try:
                os.killpg(os.getpgid(v["pid"]), signal.SIGTERM)
            except OSError:
                pass
            print(f"stopped {n} (pid {v['pid']})")
    deadline = time.time() + 8
    while time.time() < deadline and any(alive(v["pid"]) for v in st.values() if not v.get("docker")):
        time.sleep(0.3)
    for n, v in st.items():
        if not v.get("docker") and alive(v["pid"]):
            try:
                os.killpg(os.getpgid(v["pid"]), signal.SIGKILL)
            except OSError:
                pass
    save({})


def logs(name: str | None) -> None:
    files = [LOGS / f"{name}.log"] if name else sorted(LOGS.glob("*.log"))
    if not files or not all(f.exists() for f in files):
        print("no logs yet.")
        return
    subprocess.run(["tail", "-n", "25", "-F", *map(str, files)])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "up":
        up("--with-data-mcp" in sys.argv, "--no-adk" not in sys.argv, "--no-neo4j" not in sys.argv)
    elif cmd == "down":
        down()
    elif cmd == "status":
        status()
    elif cmd == "logs":
        logs(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
