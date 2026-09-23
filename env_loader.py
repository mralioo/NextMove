"""Loads every `.env` file found walking up from a starting directory, not just
the nearest one.

This repo ended up with secrets split across two `.env` files at different
directory levels (one in this repo's root, one in its parent folder).
`python-dotenv`'s `find_dotenv()` stops at the first `.env` it finds, so a
script run from the repo root only ever saw the repo-root file and silently
missed keys (like SUPERVISOR_LITELLM_MODEL) that only existed in the parent
one — the agent then fell through to its Gemini default and failed with no
GOOGLE_API_KEY. `load_all_dotenvs()` merges every `.env` on the way up
instead, nearest-first, so nothing is silently missed regardless of which
file a given key happens to live in.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_all_dotenvs(start_dir: str | Path | None = None) -> list[Path]:
    """Walk from start_dir (default: cwd) up to the filesystem root, loading
    every `.env` file found. Nearest file's values win on key conflicts —
    each is loaded with override=False, so a key already set by a nearer
    file (or already present in the real environment) is never clobbered by
    a farther one. Returns the list of paths actually loaded, nearest first.
    """
    current = Path(start_dir or Path.cwd()).resolve()
    loaded: list[Path] = []
    seen: set[Path] = set()
    for directory in [current, *current.parents]:
        if directory in seen:
            continue
        seen.add(directory)
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            loaded.append(candidate)
    return loaded
