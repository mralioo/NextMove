"""Offline tests for the experiment suite's building blocks (no LLM, no MCP server, no network)."""
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent"), str(REPO / "evaluation"), str(REPO / "experiments"), str(REPO)]

import arms  # noqa: E402
import config  # noqa: E402
import executor  # noqa: E402
import memory  # noqa: E402
import questions  # noqa: E402
import router_jev  # noqa: E402
import router_tfidf  # noqa: E402
import scoring  # noqa: E402


def test_config_validates_levels_and_round_trips_through_env(monkeypatch):
    cfg = config.RunConfig(router="tfidf", memory="episodic", mcp="http", engine="empirical", writer="template").validate()
    monkeypatch.setenv("TMT_CONFIG", cfg.to_env()["TMT_CONFIG"])
    assert config.load() == cfg
    assert cfg.to_env()["SCENARIO_ENGINE"] == "empirical"
    import pytest
    with pytest.raises(ValueError):
        config.RunConfig(router="oracle").validate()


def test_design_is_one_factor_at_a_time_with_a_noise_floor():
    ids = [a.arm_id for a in arms.ARMS]
    assert len(ids) == len(set(ids))
    replicates = [a for a in arms.ARMS if a.factor == "-"]
    assert [a.arm_id for a in replicates] == ["A00", "A01", "A02"]              # baseline + 2 replicates -> noise range
    assert all(a.params == arms.BASELINE for a in replicates)                   # replicates are exact copies
    for a in (x for x in arms.ARMS if x.factor != "-"):
        differing = [k for k in arms.ORDER if a.params[k] != arms.BASELINE[k]]
        assert differing == [a.factor], f"{a.arm_id} must change exactly one factor, changes {differing}"
    names = [a.name for a in arms.ARMS if a.factor != "-"] + [arms.BY_ID["A00"].name]
    assert len(names) == len(set(names))                                        # every configuration has a unique, self-describing name
    assert arms.BY_ID["R1"].name == "router=llm|memory=session|mcp=stdio|engine=tabpfn|writer=small"
    assert sum(a.params["writer"] == "main" for a in arms.ARMS) == 1            # the shared main model is used by exactly one arm


def test_two_questions_and_protocol():
    assert [t for t, _, _ in questions.PROTOCOL] == ["Q1", "Q2", "Q1r"]
    assert questions.PROTOCOL[1][2] == "same" and questions.PROTOCOL[2][2] == "new"
    assert "those stations" in questions.Q2


def test_episodic_memory_keys_ignore_wording_and_expire(tmp_path):
    mem = memory.EpisodicMemory(tmp_path / "m.db", ttl_s=3600)
    plan = {"cat": "C", "lines": ["U6"], "stations": ["A", "B"], "dates": [], "time": None, "dur_min": None, "what_if": False, "raw": []}
    facts = {"status": "ok", "cat": "C", "press": [{"s": "X"}]}
    assert mem.lookup(plan, "2026-09-22") is None
    mem.store(plan, "2026-09-22", facts)
    hit = mem.lookup({**plan, "conf": 0.3, "tier": 1}, "2026-09-22")            # extra plan fields (wording-dependent) don't change the key
    assert hit["memory"] == "hit" and hit["press"] == facts["press"]
    assert mem.lookup(plan, "2026-09-30") is None                              # a new dataset never gets stale facts
    mem.store(plan, "2026-09-22", {"status": "need"})                          # only successful facts are remembered
    assert mem.lookup(plan, "2026-09-22")["status"] == "ok"
    assert memory.key_for({"cat": "A", "stations": ["A"]}) is None


class _CountingMcp:
    def __init__(self):
        self.calls = []

    async def call(self, tool, **kw):
        self.calls.append(tool)
        if tool == "describe_dataset":
            return {"coverage_start": "2026-06-10 05:00:00", "coverage_end": "2026-09-22 00:45:00"}
        if tool == "station_profile":
            return {"avg_daily_passengers": 6800, "weekday_peak_hour": 18, "weekday_peak_avg_passengers": 219, "weekend_peak_hour": 8,
                    "weekend_peak_avg_passengers": 146, "network_mean_weekday_peak_passengers": 265, "delta_vs_network_mean_pct": -17.3,
                    "exceeds_network_mean_weekday_peak": False}
        raise AssertionError(tool)


def test_executor_serves_repeat_situations_from_episodic_memory(tmp_path):
    executor._COVERAGE.clear()
    mem = memory.EpisodicMemory(tmp_path / "m.db")
    plan = {"cat": "D", "stations": ["U Rudow (Berlin)"], "raw": [], "dates": [], "time": None, "lines": [], "what_if": False, "month": None, "dur_min": None}
    mcp = _CountingMcp()
    f1, _ = asyncio.run(executor.execute(plan, "peak at Rudow?", None, mcp, mem))
    n_first = len(mcp.calls)
    f2, trace = asyncio.run(executor.execute(plan, "when does Rudow peak?", None, mcp, mem))
    assert f2["memory"] == "hit" and len(mcp.calls) == n_first and trace[0]["tool"].startswith("memory")
    assert f2["st"] == f1["st"]                                                 # identical facts, zero extra tool calls


def test_tfidf_router_is_trained_without_the_test_questions_and_routes_sanely():
    router_tfidf._MODEL.clear()
    texts, labels = router_tfidf._training_set(exclude=(questions.Q1, questions.Q2))
    assert len(texts) > 60 and not any(t.startswith(questions.Q1[:60]) for t in texts)
    cat, p = router_tfidf.classify("Rank the five stations whose closure would fragment the network the most.", exclude=(questions.Q1, questions.Q2))
    assert cat == "F" and 0 < p <= 1
    router_tfidf._MODEL.clear()


def test_jev_adapter_is_opt_in_and_follows_the_documented_format(monkeypatch):
    import pytest
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert router_jev.configured() is False
    with pytest.raises(router_jev.NotConfigured):
        router_jev.classify("U6 suspended?")
    req = router_jev.build_request("U6 suspended?", has_history=True)
    assert req["model"] == "jev-latest" and req["questions"]["category"]["type"] == "choice"
    assert "C" in req["questions"]["category"]["criteria"] and req["state"]["has_previous_answer_in_session"] is True

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"answers": {"category": {"choice": "C", "confidence": 0.92}}}).encode()
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    seen = {}
    def fake_urlopen(req, timeout):
        seen["auth"], seen["url"] = req.headers.get("Authorization"), req.full_url
        return _Resp()
    monkeypatch.setattr(router_jev.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_jev.json, "load", lambda r: json.loads(r.read()))
    assert router_jev.classify("U6 suspended?") == ("C", 0.92)
    assert seen["auth"] == "Bearer test-key" and seen["url"].endswith("/v1/systemone")


def test_facts_similarity_measures_ranking_agreement():
    a = {"press": [{"s": "K", "p": 78}, {"s": "M", "p": 29}, {"s": "U", "p": 21}]}
    same = scoring.facts_similarity(a, a)
    assert same["top_station_same"] and same["top3_overlap"] == 1.0 and same["mean_abs_prob_diff_pts"] == 0 and same["pair_order_agreement"] == 1.0
    lower = {"press": [{"s": "K", "p": 61}, {"s": "M", "p": 17}, {"s": "X", "p": 10}]}
    sim = scoring.facts_similarity(a, lower)
    assert sim["top_station_same"] and abs(sim["top3_overlap"] - 2 / 3) < 1e-9 and sim["mean_abs_prob_diff_pts"] == (17 + 12) / 2
    assert scoring.facts_similarity(a, {}) is None


def test_config_reload_updates_the_shared_object_in_place(monkeypatch):
    seen = config.CONFIG                                           # what modules hold via `from config import CONFIG`
    before = seen.router
    monkeypatch.setenv("TMT_CONFIG", json.dumps({"router": "tfidf", "memory": "none"}))
    assert config.reload() is seen and seen.router == "tfidf" and seen.memory == "none"
    monkeypatch.delenv("TMT_CONFIG")
    config.reload()
    assert seen.router == before == "rules" and seen.memory == "session"
