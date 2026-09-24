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
import questions  # noqa: E402
import scoring  # noqa: E402


def test_config_validates_levels_and_round_trips_through_env(monkeypatch):
    cfg = config.RunConfig(router="llm", memory="none", mcp="http", engine="empirical", writer="template").validate()
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
    monkeypatch.setenv("TMT_CONFIG", json.dumps({"router": "llm", "memory": "none"}))
    assert config.reload() is seen and seen.router == "llm" and seen.memory == "none"
    monkeypatch.delenv("TMT_CONFIG")
    config.reload()
    assert seen.router == before == "rules" and seen.memory in ("session", "cognee")     # cognee when COGNEE_ENABLED is set
