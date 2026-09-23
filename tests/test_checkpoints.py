"""Offline tests for checkpoint bookkeeping (no TabPFN API involved)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "ml")]

import checkpoints as C  # noqa: E402


def test_fingerprint_is_stable_and_sensitive():
    a = C.make_fingerprint(features=["x", "y"], n_rows=10, cutoff="2026-09-02")
    assert a == C.make_fingerprint(cutoff="2026-09-02", n_rows=10, features=["x", "y"])
    assert a != C.make_fingerprint(features=["x", "y"], n_rows=11, cutoff="2026-09-02")
    assert a != C.make_fingerprint(features=["y", "x"], n_rows=10, cutoff="2026-09-02")


def test_stale_or_missing_checkpoint_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CHECKPOINT_DIR", tmp_path)
    assert C.restore_checkpoint("nope", object, "fp", lambda d: d, "y") is None   # nothing saved
    d = tmp_path / "m"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"fingerprint": "old"}))
    (d / "model.json").write_text("{}")
    # fingerprint mismatch (dataset/features changed) -> None BEFORE any API call is attempted
    assert C.restore_checkpoint("m", object, "new", lambda d: d, "y") is None


def test_manifest_lists_saved_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CHECKPOINT_DIR", tmp_path)
    d = tmp_path / "m"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"name": "m", "task": "regression", "n_train_rows": 5}))
    path = C.write_manifest()
    assert json.loads(path.read_text())[0]["name"] == "m"
