import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

import mcp_runtime  # noqa: E402
import observability as obs  # noqa: E402
from schemas import ToolCall  # noqa: E402
from worker import _tool_calls  # noqa: E402


def test_payload_truncates_and_reports_rest():
    out = obs.payload({"x": "a" * 500}, 100)
    assert len(out) < 140 and "… [+" in out
    assert obs.payload({"a": 1}) == '{"a":1}'


def test_toolcall_carries_payload_fields():
    t = ToolCall(tool="scenario_flow", server="ubahn-flow-data", seconds=1.2, args={"closure_id": 8}, result_preview="{}", result_bytes=2, ok=True, wait_ready_ms=5)
    assert t.args == {"closure_id": 8} and t.result_bytes == 2 and t.wait_ready_ms == 5


def test_mcp_call_fills_call_log_and_span_attrs():
    class Fake(mcp_runtime.McpRuntime):
        def __init__(self):
            self.loop = asyncio.new_event_loop()
            import threading
            self._ready = threading.Event()
            self._ready.set()
            self._error = None
            threading.Thread(target=self.loop.run_forever, daemon=True).start()

        async def _call(self, tool, args):
            return {"rows": [1, 2, 3]}

    log: list = []
    tok = obs.CALL_LOG.set(log)
    try:
        res = asyncio.run(Fake().call("alternate_paths", origin="X", none_arg=None))
    finally:
        obs.CALL_LOG.reset(tok)
    assert res == {"rows": [1, 2, 3]}
    assert len(log) == 1
    rec = log[0]
    assert rec["tool"] == "alternate_paths" and rec["args"] == {"origin": "X"} and rec["ok"]
    assert rec["result_bytes"] > 0 and "rows" in rec["result_preview"] and "wait_ready_ms" in rec


def test_tool_calls_conversion():
    rec = {"tool": "t", "server": "s", "args": {"a": 1}, "wait_ready_ms": 3, "ok": True, "seconds": 0.5, "result_preview": "{}", "result_bytes": 2}
    calls = _tool_calls([rec], [], 0.0)
    assert calls[0].tool == "t" and calls[0].args == {"a": 1} and calls[0].result_bytes == 2


def test_tool_calls_fallback_to_plain_trace():
    calls = _tool_calls([], [{"tool": "t", "s": 0.4}], 0.0)
    assert calls[0].tool == "t" and calls[0].seconds == 0.4 and calls[0].args == {}


def test_log_llm_records_time_tokens_and_truncates():
    class U:
        prompt_tokens, completion_tokens = 100, 20

    log: list = []
    tok = obs.LLM_LOG.set(log)
    try:
        rec = obs.log_llm("writer", "m", 1.23456, U(), "p" * 5000, "ok")
    finally:
        obs.LLM_LOG.reset(tok)
    assert log == [rec] and rec["seconds"] == 1.235 and rec["tok_in"] == 100 and rec["tok_out"] == 20 and len(rec["prompt"]) < 1600
    assert obs.log_llm("router", "m", 0.1)["tok_in"] is None      # no log set: still returns the record
