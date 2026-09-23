"""A persistent, pre-warmed connection to the MCP server (dataset + TabPFN tools).

The slow, old design spawned a fresh MCP server process inside each specialist (cold start every
question). Here ONE server subprocess lives for the whole agent process:

  * it is started in a background thread as soon as this module is imported, so the server's own
    warm-up (feature-table cache, Category C model checkpoint, prediction cache) overlaps with
    everything else — including the operator typing their first question;
  * it runs on its own asyncio loop in a daemon thread, so it works the same under `adk web`,
    the CLI runner, or any other event loop;
  * tool calls from any loop are forwarded with `run_coroutine_threadsafe`, and several can be in
    flight at once (apply_closure / alternate_paths / scenario_flow are issued in parallel).

Use:  data = await mcp.call("scenario_flow", closure_id=8)      (returns the tool's JSON as python)
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER = REPO_ROOT / "mcp_server" / "server.py"


class McpRuntime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._client = None
        self._ready = threading.Event()
        self._error: Exception | None = None
        self.started_at = time.time()
        self.ready_after: float | None = None
        threading.Thread(target=self._run, name="mcp-runtime", daemon=True).start()

    # -- background loop -------------------------------------------------------------------
    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._start())
        self.loop.run_forever()

    async def _start(self) -> None:
        try:
            from fastmcp import Client
            from fastmcp.client.transports import PythonStdioTransport

            env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
            transport = PythonStdioTransport(SERVER, python_cmd=sys.executable, env=env, keep_alive=True)
            self._client = Client(transport, timeout=120)
            await self._client.__aenter__()
            self.ready_after = time.time() - self.started_at
        except Exception as e:  # surfaced on the first call
            self._error = e
        finally:
            self._ready.set()

    # -- public API ------------------------------------------------------------------------
    async def call(self, tool: str, **args):
        """Call an MCP tool; returns the decoded JSON result (dict/list)."""
        await asyncio.get_running_loop().run_in_executor(None, self._ready.wait)
        if self._error:
            raise RuntimeError(f"MCP server failed to start: {self._error}")
        fut = asyncio.run_coroutine_threadsafe(self._call(tool, args), self.loop)
        return await asyncio.wrap_future(fut)

    async def _call(self, tool: str, args: dict):
        args = {k: v for k, v in args.items() if v is not None}
        r = await self._client.call_tool(tool, args)
        if r.data is not None:
            return r.data
        return r.structured_content

    def call_sync(self, tool: str, **args):
        self._ready.wait()
        if self._error:
            raise RuntimeError(f"MCP server failed to start: {self._error}")
        return asyncio.run_coroutine_threadsafe(self._call(tool, args), self.loop).result()


_runtime: McpRuntime | None = None
_lock = threading.Lock()


def get_runtime() -> McpRuntime:
    """Process-wide singleton; the first call starts (and thereby warms) the server."""
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = McpRuntime()
        return _runtime
