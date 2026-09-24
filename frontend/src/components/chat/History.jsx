import React from "react";
import { Plus } from "lucide-react";
import { short, useChat } from "../../state/ChatContext";

const CAT = { A: "Events", B: "Anomalies", C: "Closures", D: "Stations", E: "Energy", F: "Network", G: "Correlations", H: "Reroute", P: "Pressure", X: "Strategy" };
const ago = (ts) => { const m = Math.max(0, (Date.now() / 1000 - ts) / 60); return m < 1 ? "just now" : m < 60 ? `${Math.round(m)} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`; };

/** The chat history: one entry per conversation (a resumed conversation stays one entry); click to reopen and continue it. */
export default function History({ onPicked }) {
  const { convs, sessionId, thread, newConversation, openConversation } = useChat();
  return (
    <nav className="chat-history">
      <h4>Conversations <button onClick={() => { newConversation(); onPicked?.(); }}><Plus size={12} /> New</button></h4>
      {convs.length === 0 && <div className="muted">No earlier conversation yet.</div>}
      {convs.map((c) => (
        <div key={c.conversation_id} className={"chat-conv" + ((c.session_ids.includes(sessionId) || (thread.linkTurn && c.last_turn_id >= thread.linkTurn && c.title === thread.title)) ? " on" : "")}
             onClick={async () => { await openConversation(c); onPicked?.(); }}>
          <div className="ttl">{short(c.title, 88)}</div>
          <div className="meta"><span className="pill-badge teal">{CAT[c.category] || c.category || "—"}</span><span>{c.n_turns} msg</span><span>{ago(c.last)}</span>{c.mean_score ? <span>★ {c.mean_score}</span> : null}{c.resumed ? <span>↻</span> : null}</div>
        </div>
      ))}
    </nav>
  );
}
