"""The Inspector's connection to the quality MCP server (mcp_server/quality_server.py).

An in-memory FastMCP client: the server runs in THIS process (full MCP protocol semantics, no subprocess, no pipes), so a check costs milliseconds. The database is opened lazily on the first
call and warmed in a background thread at import. If the database was never built (`./.venv/bin/python ml/quality_db.py build`) every call returns None and the Inspector simply has no
quality boundaries — it never blocks an answer.

    res = await quality_mcp.call("quality_check_facts", category="C", facts_json="{...}")
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_lock = threading.Lock()
_client = None
_ready = False


def available() -> bool:
    return os.environ.get("QUALITY_MCP", "on") != "off" and (REPO / "data" / "quality" / "quality.db").exists()


async def golden(tool: str, **args):
    """The same in-process client for the `golden_*` tools (the pre-processed files in data/normalized and data/processed): `await quality_mcp.golden("golden_series", station=..., start=..., end=...)`.
    Available whenever the normalized tables exist, even without the quality database."""
    if os.environ.get("QUALITY_MCP", "on") == "off" or not (REPO / "data" / "normalized" / "normalized_rest.csv").exists():
        return None
    return await _call(tool, **args)


def _server():
    sys.path[:0] = [str(REPO / "mcp_server"), str(REPO / "ml"), str(REPO)]
    import importlib
    return importlib.import_module("quality_server").mcp


async def call(tool: str, **args):
    """Call a quality tool; returns the decoded result or None when the database is not available / the call failed."""
    if not available():
        return None
    return await _call(tool, **args)


async def _call(tool: str, **args):
    global _client
    try:
        from fastmcp import Client
        with _lock:
            if _client is None:
                _client = Client(_server(), timeout=30)
        async with _client:
            r = await _client.call_tool(tool, args)
        return r.data if r.data is not None else r.structured_content
    except Exception:
        return None


def warm() -> None:
    """Load the cells in the background so the first check is instant."""
    def _w():
        try:
            sys.path[:0] = [str(REPO / "ml")]
            import quality_db
            if quality_db.load().available:
                quality_db.load().cells()
        except Exception:
            pass
    threading.Thread(target=_w, daemon=True, name="quality-warm").start()


def call_sync(tool: str, **args):
    return asyncio.run(call(tool, **args))
