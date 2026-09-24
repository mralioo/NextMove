"""Small housekeeping helpers used by tasks.py: Neo4j container (Docker) and the observability database.

    neo4j.py up | down | backup-obs | clean-obs
"""
from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv" / "bin" / "python") if (REPO / ".venv" / "bin" / "python").exists() else sys.executable


def env_value(key: str) -> str:
    for line in (REPO / ".env").read_text().splitlines() if (REPO / ".env").exists() else []:
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return ""


def up() -> None:
    pw = env_value("NEO4J_PASSWORD")
    if not pw:
        raise SystemExit("Add NEO4J_URI=bolt://localhost:7687, NEO4J_USER=neo4j, NEO4J_PASSWORD=<pw> to .env first.")
    if subprocess.run(["docker", "start", "nextmove-neo4j"], capture_output=True).returncode != 0:
        subprocess.run(["docker", "run", "-d", "--name", "nextmove-neo4j", "--restart", "unless-stopped", "-p", "127.0.0.1:7474:7474", "-p", "127.0.0.1:7687:7687",
                        "-v", "nextmove_neo4j_data:/data", "-e", f"NEO4J_AUTH=neo4j/{pw}", "neo4j:5.26-community"], check=True, capture_output=True)
    print("waiting for Neo4j ...")
    for _ in range(60):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", 7687)) == 0:
                break
        time.sleep(2)
    time.sleep(6)
    subprocess.run([PY, "agent/kgraph.py", "neo4j-sync"], cwd=REPO)
    print("Neo4j browser -> http://localhost:7474 (user neo4j)")


def down() -> None:
    subprocess.run(["docker", "stop", "nextmove-neo4j"])


def backup_obs() -> None:
    d = REPO / "observability" / "backup"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"agent_obs_{time.strftime('%Y%m%d_%H%M%S')}.db"
    sqlite3.connect(REPO / "observability" / "agent_obs.db").backup(sqlite3.connect(dst))
    print("backed up ->", dst)


def clean_obs() -> None:
    for ext in ("", "-wal", "-shm"):
        (REPO / "observability" / f"agent_obs.db{ext}").unlink(missing_ok=True)
    print("observability database deleted")


if __name__ == "__main__":
    {"up": up, "down": down, "backup-obs": backup_obs, "clean-obs": clean_obs}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
