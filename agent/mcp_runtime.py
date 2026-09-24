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
import json
import atexit
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG  # noqa: E402

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

            self.transport_kind = CONFIG.mcp
            if CONFIG.mcp == "inmemory":
                # server in THIS process: full MCP protocol semantics, but no subprocess and no pipes
                sys.path.insert(0, str(SERVER.parent))
                import importlib

                self._client = Client(importlib.import_module("server").mcp, timeout=120)
            elif CONFIG.mcp == "http":
                # server as a separate process, reached over streamable HTTP on localhost
                with socket.socket() as sk:
                    sk.bind(("127.0.0.1", 0))
                    port = sk.getsockname()[1]
                env = {**os.environ, "MCP_TRANSPORT": "http", "MCP_PORT": str(port),
                       "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
                self._proc = subprocess.Popen([sys.executable, str(SERVER)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                atexit.register(self.shutdown)
                for _ in range(240):                       # wait until the HTTP server accepts connections
                    try:
                        socket.create_connection(("127.0.0.1", port), timeout=0.25).close()
                        break
                    except OSError:
                        await asyncio.sleep(0.25)
                self._client = Client(f"http://127.0.0.1:{port}/mcp", timeout=120)
            else:
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

    def shutdown(self) -> None:
        """Stop a server process this runtime started (http mode); stdio children exit with their pipe."""
        proc = getattr(self, "_proc", None)
        if proc and proc.poll() is None:
            proc.terminate()

    # -- public API ------------------------------------------------------------------------
    async def call(self, tool: str, **args):
        """Call an MCP tool; returns the decoded JSON result (dict/list). Every call is a span (`mcp.tool <tool>`) carrying the server, the arguments, a preview of the
        result, its size and how long it waited for the server to be ready; the same record is appended to observability.CALL_LOG when the caller set one."""
        from observability import CALL_LOG, payload, set_attr, span

        clean = {k: v for k, v in args.items() if v is not None}
        with span(f"mcp.tool {tool}", **{"tmt.tool": tool, "tmt.server": "ubahn-flow-data", "tmt.transport": getattr(self, "transport_kind", CONFIG.mcp), "tmt.args": clean}) as sp:
            t0 = time.time()
            await asyncio.get_running_loop().run_in_executor(None, self._ready.wait)
            wait_ms = round((time.time() - t0) * 1000)
            rec = {"tool": tool, "server": "ubahn-flow-data", "args": clean, "wait_ready_ms": wait_ms, "ok": True, "start": t0}
            try:
                if self._error:
                    raise RuntimeError(f"MCP server failed to start: {self._error}")
                fut = asyncio.run_coroutine_threadsafe(self._call(tool, args), self.loop)
                result = await asyncio.wrap_future(fut)
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}", seconds=round(time.time() - t0, 3))
                set_attr(sp, "tmt.error", rec["error"])
                log = CALL_LOG.get()
                if log is not None:
                    log.append(rec)
                raise
            rec["seconds"] = round(time.time() - t0, 3)
            rec["result_preview"] = payload(result, 1500)
            rec["result_bytes"] = len(json.dumps(result, default=str)) if result is not None else 0
            set_attr(sp, "tmt.wait_ready_ms", wait_ms)
            set_attr(sp, "tmt.seconds", rec["seconds"])
            set_attr(sp, "tmt.result", payload(result, 3000))
            set_attr(sp, "tmt.result_bytes", rec["result_bytes"])
            if isinstance(result, dict) and "error" in result:
                rec["ok"] = False
                set_attr(sp, "tmt.tool_error", str(result["error"])[:300])
            log = CALL_LOG.get()
            if log is not None:
                log.append(rec)
            return result

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
