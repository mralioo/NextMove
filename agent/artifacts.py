"""ARTIFACTS: everything one answered question produced, kept so the operator can ask "why / evidence / which tools" later without any recomputation.

The default answer is a brief (writer.py). Behind it, every turn stores ONE artifact bundle: the plan and objective, every MCP call (tool, arguments, time, size, result
preview), the facts, the confidence and its reasons, the evaluator's checks and verdict, the sanity check, the LLM calls (model, seconds, tokens), the references and the
brief. The bundle is (1) kept in the session, (2) written to the operator knowledge base (`turns.artifact_json`, searchable through the knowledge MCP server),
(3) linked in the knowledge graph (Problem -HAS_ARTIFACT-> Artifact -USED_TOOL/USED_DATASET/USED_MODEL/CITES-> ...) and (4) summarised into the memory agent (Cognee).

`full_report()` turns a bundle into the long answer: the writer's narrative (verdict, evidence, do now, caveat, argument) + a deterministic "How this was worked out" section
built ONLY from the stored bundle — so "which tools did you call and with what" is answered from the record, never from the model's memory.
"""
from __future__ import annotations

import json
import time

VERSION = "1.0"


def _tool_rows(calls: list[dict]) -> list[dict]:
    return [{"tool": c.get("tool"), "server": c.get("server", "ubahn-flow-data"), "args": c.get("args") or {}, "seconds": c.get("s", c.get("seconds")), "bytes": c.get("bytes", c.get("result_bytes")),
             "ok": c.get("ok", True), "round": c.get("round"), "result_preview": (c.get("result_preview") or "")[:600]} for c in (calls or [])]


def build(*, question: str, session_id: str, plan: dict, result: dict | None, verdict: dict | None, facts: dict, timing: dict, refs: list, brief: str, sources_line: str,
          sanity: dict | None, source: str, answer_mode: str, data_window: str) -> dict:
    """The bundle of one turn (plain dicts: JSON-able, schema-light on purpose so old bundles stay readable)."""
    sp = plan or {}
    res, ver = result or {}, verdict or {}
    ho = timing.get("handover", {})
    llm = [{k: r.get(k) for k in ("role", "model", "seconds", "tok_in", "tok_out", "error")} for r in (timing.get("llm") or [])]
    return {
        "artifact_version": VERSION, "turn_id": None, "session_id": session_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "question": question,
        "category": sp.get("category"), "specialist": (sp.get("route") or {}).get("specialist"), "decision": sp.get("decision"), "source": source, "answer_mode": answer_mode,
        "objective": (sp.get("objective") or {}), "entities": (sp.get("entities") or {}), "route": sp.get("route") or {},
        "assumptions": res.get("assumptions") or [], "confidence": res.get("confidence"), "confidence_reasons": res.get("confidence_reasons") or [],
        "verdict": {"verdict": ver.get("verdict"), "score": ver.get("score"), "objective_met": ver.get("objective_met"), "issues": ver.get("issues") or [], "rationale": ver.get("rationale"),
                    "model": ver.get("model"), "rounds": len(timing.get("loop") or []), "checks": ho.get("checks") or [], "ground_truth_ids": ver.get("ground_truth_ids") or [],
                    "boundary_ids": ver.get("boundary_ids") or [], "similar_cases": ho.get("similar_cases") or []},
        "tools": _tool_rows(timing.get("calls") or []), "datasets": res.get("datasets_used") or (sp.get("route") or {}).get("datasets") or [], "ml_engine": res.get("ml_engine_used"),
        "llm": llm, "facts": facts, "brief": brief, "report": None, "references": [r.model_dump() if hasattr(r, "model_dump") else r for r in refs], "sources_line": sources_line,
        "timing": timing.get("stages") or {}, "guard": timing.get("guard"), "sanity": sanity or {}, "data_window": data_window,
    }


def appendix(art: dict) -> str:
    """'How this was worked out' — deterministic, from the stored bundle only."""
    out = ["**How this was worked out**"]
    obj = art.get("objective") or {}
    ent = {k: v for k, v in (art.get("entities") or {}).items() if v}
    out.append(f"1. **Understood as:** category {art.get('category')} ({art.get('specialist') or 'specialist'}). Objective: {obj.get('statement', '—')}"
               + (f" Parameters: {json.dumps(ent, ensure_ascii=False)[:300]}." if ent else "")
               + (" Assumed: " + "; ".join(str(a) for a in art["assumptions"]) + "." if art.get("assumptions") else ""))
    tools = art.get("tools") or []
    if art.get("legacy"):
        return ("**How this was worked out**\nThe tool calls, checks and timings of this answer were not recorded (it was given before the operator knowledge base stored artifacts). "
                "Ask the question again to recompute it and keep the full record.")
    if tools:
        total = round(sum((t.get("seconds") or 0) for t in tools), 2)
        out.append(f"2. **Tools called ({len(tools)}, {total} s summed, some in parallel):**")
        for t in tools:
            args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (t.get("args") or {}).items())
            size = f" · {round(t['bytes'] / 1024, 1)} KB" if t.get("bytes") else ""
            out.append(f"   - `{t['tool']}({args})` — {t.get('server')} · {t.get('seconds')} s{size}" + ("" if t.get("ok", True) else " · FAILED") + (f" · round {t['round']}" if t.get("round") else ""))
    else:
        out.append("2. **Tools called:** none (answered without new data).")
    eng = {"tabpfn": "TabPFN quantile regression (foundation model for tables, predicts the normal-load distribution per station and 15 minutes)", "empirical": "empirical baseline, no ML",
           "none": "no forecasting model"}.get(art.get("ml_engine") or "none", art.get("ml_engine"))
    out.append(f"3. **Data and model:** datasets {', '.join(art.get('datasets') or []) or '—'}; engine: {eng}; data window {art.get('data_window')}.")
    v = art.get("verdict") or {}
    checks = v.get("checks") or []
    ok = sum(1 for c in checks if c.get("ok"))
    out.append(f"4. **Checks before you saw it:** evaluator **{v.get('verdict')}** (score {v.get('score')}, {v.get('rounds')} round(s), {v.get('model')}); {ok}/{len(checks)} checks passed"
               + ("; issues: " + "; ".join(v.get("issues") or []) if v.get("issues") else "") + ".")
    for c in checks:
        out.append(f"   - {'✓' if c.get('ok') else '✗'} {c.get('id')}: {str(c.get('detail', ''))[:140]}")
    san = art.get("sanity") or {}
    if san:
        out.append(f"   - knowledge-base sanity check: {'ok' if san.get('ok') else 'FAILED ' + str(san.get('fails')) if san.get('ok') is False else 'n/a'}; number guard: {art.get('guard')}")
    if v.get("ground_truth_ids") or v.get("boundary_ids"):
        out.append(f"5. **Knowledge base used:** ground truth {', '.join(v.get('ground_truth_ids') or []) or '—'}; boundaries {', '.join(v.get('boundary_ids') or []) or '—'}.")
    out.append(f"6. **Confidence {art.get('confidence')} because:** " + ("; ".join(art.get("confidence_reasons") or []) or "—") + ".")
    llm = art.get("llm") or []
    t = art.get("timing") or {}
    if t or llm:
        bits = [f"total {t.get('total_s')} s = supervisor {t.get('supervisor_s')} + worker/evaluator {t.get('worker_evaluator_s')} + writer {t.get('writer_s')}"] if t else []
        if llm:
            bits.append("LLM calls: " + "; ".join(f"{r['role']} {r['model']} {r['seconds']} s ({r.get('tok_in')}→{r.get('tok_out')} tokens)" for r in llm))
        out.append("7. **Time and models:** " + " · ".join(bits) + ".")
    if art.get("precedents"):
        out.append("8. **Operator precedents (what operators did in similar past situations):**")
        for p in art["precedents"]:
            out.append(f"   - similar case ({round(p.get('similarity', 0) * 100)}%): {str(p.get('problem'))[:110]} — mean score {p.get('mean_score')}")
            for a in p.get("actions") or []:
                out.append(f"     · operators {a.get('action')} (followed advice: {a.get('followed') or '?'}; outcome: {a.get('outcome') or '?'}; score {a.get('score')})")
    out.append(f"_Stored in the operator knowledge base{' as artifact #' + str(art['turn_id']) if art.get('turn_id') else ''} ({art.get('created_at')})._")
    return "\n".join(out)


def full_report(art: dict, narrative: str) -> str:
    """The long answer: narrative (the writer's detail mode) + the deterministic method section + sources."""
    parts = [narrative.strip(), appendix(art)]
    if art.get("sources_line") and "**Sources:**" not in narrative:
        parts.append(art["sources_line"])
    return "\n\n".join(p for p in parts if p)


def memory_summary(art: dict, limit: int = 1400) -> str:
    """A compact text of the bundle for the memory agent (Cognee entry context / operator knowledge base indexing)."""
    v = art.get("verdict") or {}
    txt = (f"[{art.get('category')}] {art.get('question')}\nDecision: {art.get('brief', '')[:400]}\nConfidence {art.get('confidence')} · evaluator {v.get('verdict')} · engine {art.get('ml_engine')}\n"
           f"Tools: {', '.join(t['tool'] for t in art.get('tools') or [])}\nDatasets: {', '.join(art.get('datasets') or [])}\n"
           f"Assumptions: {'; '.join(str(a) for a in art.get('assumptions') or [])}\nKB: {', '.join((v.get('ground_truth_ids') or []) + (v.get('boundary_ids') or []))}")
    return txt[:limit]
