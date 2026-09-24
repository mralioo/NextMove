import React from "react";
import { Route, Wrench, ShieldCheck, PenLine, Loader2, CheckCircle2, Settings } from "lucide-react";

const ICONS = { route: Route, tools: Wrench, check: ShieldCheck, write: PenLine };

/** Live progress of the four roles while the agent works; details (tool calls with seconds, verdicts) appear as the stream delivers them. */
export default function ThinkingPanel({ stages }) {
  const done = stages.filter((s) => s.status === "done").length;
  return (
    <div className="chat-think card">
      <div className="chat-think-head"><Settings size={16} className={done < stages.length ? "spin" : ""} /><h3>Agent team at work</h3></div>
      <div className="chat-think-bar"><i style={{ width: `${(done / stages.length) * 100}%` }} /></div>
      {stages.map((s) => {
        const Icon = ICONS[s.icon] || Settings;
        return (
          <div key={s.id} className={`chat-think-step ${s.status}`} title={s.text}>
            <div className="mark">{s.status === "pending" ? <span className="dot" /> : s.status === "active" ? <Loader2 size={16} className="spin" /> : <CheckCircle2 size={16} />}</div>
            <div className="body">
              <div className="ttl"><Icon size={14} /> <b>{s.title}</b>{s.seconds ? <em>{s.seconds.toFixed(2)} s</em> : null}</div>
              {s.status === "pending" && <small>{s.text}</small>}
              {s.details.map((d, i) => <small key={i}>{d.kind === "tool" ? `🛠 ${d.label}` : d.label}{d.seconds != null ? ` · ${Number(d.seconds).toFixed(2)} s` : ""}</small>)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
