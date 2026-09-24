import React from "react";
import { Wrench, Sparkles, Route, ShieldCheck, PenLine } from "lucide-react";

const ICON = { route: Route, worker: Wrench, tool: Wrench, evaluator: ShieldCheck, llm: Sparkles, writer: PenLine };
const fmt = (n, d = 2) => (n == null ? "—" : Number(n).toFixed(d));

/** The reference column next to the conversation: what was done for the last answer — steps with times, tool calls, inference time, tokens — and the session totals. */
export default function OpsLog({ last, totals }) {
  const t = last?.timing || {};
  const c = last?.counts || {};
  const llmS = (last?.llm || []).reduce((a, r) => a + (r.seconds || 0), 0);
  const maxS = Math.max(0.05, ...(last?.steps || []).map((s) => s.seconds || 0));
  return (
    <aside className="chat-ops">
      <h4>Operations <small>{last ? "last answer" : "waiting for a question"}</small></h4>
      <div className="chat-kpis">
        <div><b>{c.tool_calls ?? "—"}</b><span>tool calls</span></div>
        <div><b>{fmt(llmS)} s</b><span>inference time</span></div>
        <div><b>{last?.tokens ? last.tokens.total.toLocaleString() : "—"}</b><span>{last?.tokens?.estimated ? "tokens (est.)" : "tokens"}</span></div>
        <div><b>{fmt(t.total_s ?? last?.wall_s, 1)} s</b><span>total</span></div>
      </div>
      {last && (
        <>
          <div className="chat-split">
            {[["supervisor_s", "Dispatcher", "Understands the question, routes it and rejects off-topic or manipulative requests"],
              ["worker_evaluator_s", "Analyst + Inspector", "Analyst: pulls the data through MCP connectors and runs the load forecast · Inspector: recomputes key numbers from the raw data before anything is shown"],
              ["writer_s", "Writer", "Produces a one-screen brief: verdict, evidence, do-now, caveat, sources"]].map(([k, l, tip]) => (
              <div key={k} title={tip}><span>{l}</span><i style={{ width: `${Math.min(100, ((t[k] || 0) / (t.total_s || 1)) * 100)}%` }} /><em>{fmt(t[k], 2)} s</em></div>
            ))}
          </div>
          <div className="chat-tokrow">in {last.tokens.in.toLocaleString()} · out {last.tokens.out.toLocaleString()} · {c.llm_calls} LLM call{c.llm_calls === 1 ? "" : "s"} · {c.distinct_tools} distinct tool{c.distinct_tools === 1 ? "" : "s"}</div>
          <ol className="chat-steps">
            {last.steps.map((s, i) => {
              const I = ICON[s.kind] || Sparkles;
              return (
                <li key={i} className={s.kind}>
                  <span className="ic"><I size={14} /></span>
                  <div><b>{s.label}</b><small>{s.detail}</small></div>
                  <span className="sec">{s.seconds != null ? `${fmt(s.seconds, 2)} s` : ""}<i style={{ width: `${((s.seconds || 0) / maxS) * 100}%` }} /></span>
                </li>
              );
            })}
          </ol>
          <div className="chat-src">answer: <b>{last.answer_mode || last.source || "—"}</b>{last.source && last.source !== last.answer_mode ? ` · source ${last.source}` : ""}{last.guard ? ` · guard ${String(last.guard).slice(0, 40)}` : ""}</div>
        </>
      )}
      <h4>This session</h4>
      <div className="chat-kpis small">
        <div><b>{totals.questions}</b><span>questions</span></div>
        <div><b>{totals.tools}</b><span>tool calls</span></div>
        <div><b>{fmt(totals.inference, 1)} s</b><span>inference</span></div>
        <div><b>{totals.tokens.toLocaleString()}</b><span>tokens</span></div>
      </div>
    </aside>
  );
}
