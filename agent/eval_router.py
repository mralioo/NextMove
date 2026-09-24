"""Accuracy of the deterministic router on docs/test_questions.md (no LLM, no network).

    ./.venv/bin/python scripts/tasks.py eval-router

Expected category = the letter in the question ID (A1 -> A). Follow-ups may route to FOLLOW; Traps
may route to OOS; both count as correct. Cross-cutting R-questions are skipped (they mix categories).
"""
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router import route  # noqa: E402

DOC = Path(__file__).resolve().parent.parent / "docs" / "test_questions.md"
rows = []
for line in DOC.read_text().splitlines():
    m = re.match(r"\|\s*([A-HX]\d+)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*.+\|\s*$", line)
    if m:
        qid, typ, q = m.groups()
        rows.append((qid, typ, re.sub(r"^\(after [A-Z]\d+\)\s*", "", q)))

ok = defaultdict(list)
misses = []
t0 = time.time()
for qid, typ, q in rows:
    exp = qid[0]
    plan = route(q, has_history=typ.startswith("Follow"))
    good = plan["cat"] == exp or (typ.startswith("Follow") and plan["cat"] == "FOLLOW") or \
        (typ.startswith("Trap") and plan["cat"] == "OOS")
    ok[exp].append(good)
    if not good:
        misses.append((qid, typ, exp, plan["cat"], plan["conf"], q[:90]))
dt = (time.time() - t0) / max(len(rows), 1) * 1000
tot = sum(map(sum, ok.values()))
print(f"{tot}/{len(rows)} correct ({tot / len(rows):.0%}) · {dt:.2f} ms per question")
for c in sorted(ok):
    print(f"  {c}: {sum(ok[c])}/{len(ok[c])}")
print("\nMISSES:")
for m in misses:
    print("  %-4s %-10s exp=%s got=%-6s conf=%.2f  %s" % m)
low = [(qid, route(q)["conf"]) for qid, _, q in rows if route(q)["conf"] < 0.55]
print(f"\nLow-confidence (would call the LLM router): {len(low)}/{len(rows)}")
