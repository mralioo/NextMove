import React, { useState } from "react";
import { Star, Wrench, Check, PenLine } from "lucide-react";
import { api } from "../../api/client";
import { useChat } from "../../state/ChatContext";

const FOLLOW = [["Why?", "why?"], ["Evidence", "show me the evidence"], ["Which tools?", "which tools did you call?"]];

function ActionForm({ msg, onSaved }) {
  const { operator, refreshPending } = useChat();
  const [text, setText] = useState("");
  const [followed, setFollowed] = useState("as_recommended");
  const [outcome, setOutcome] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const submit = async () => {
    setBusy(true); setErr("");
    try { await api.action(msg.artifact_turn_id || msg.turn_id, operator, { action_text: text, followed, outcome: outcome || undefined }); onSaved(); refreshPending(); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="chat-action">
      <label><PenLine size={12} /> What did you do about it?</label>
      <textarea rows={2} value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. sent two staff to Neukölln and ordered the replacement bus" />
      <div className="row">
        <select value={followed} onChange={(e) => setFollowed(e.target.value)}>
          <option value="as_recommended">as recommended</option><option value="modified">modified</option><option value="different">something else</option><option value="none">did nothing</option>
        </select>
        <select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
          <option value="">outcome: not yet known</option><option value="worked">it worked</option><option value="partly">partly</option><option value="did_not_work">did not work</option>
        </select>
        <button className="chat-primary" disabled={busy || text.trim().length < 3} onClick={submit}>{busy ? "Saving…" : "Save"}</button>
      </div>
      {err && <div className="chat-err">{err}</div>}
    </div>
  );
}

/** Rating (1-5), follow-up chips (why / evidence / which tools) and the action report — the operator options under every answer. */
export default function Feedback({ msg, index }) {
  const { operator, patchMsg, send } = useChat();
  const id = msg.artifact_turn_id || msg.turn_id;
  if (!id) return null;
  const rate = async (n) => { try { await api.score(id, operator, n); patchMsg(index, { score: n, err: null }); } catch (e) { patchMsg(index, { err: e.message }); } };
  return (
    <div className="chat-fb">
      <div className="chat-fb-row">
        <span className="muted">Rate</span>
        <span className="chat-stars">{[1, 2, 3, 4, 5].map((n) => (
          <button key={n} className={n <= (msg.score || 0) ? "on" : ""} onClick={() => rate(n)} title={`${n}/5`}><Star size={15} fill={n <= (msg.score || 0) ? "currentColor" : "none"} /></button>
        ))}</span>
        {FOLLOW.map(([l, q]) => <button key={l} className="chat-chip" onClick={() => send(q)}>{l}</button>)}
        {msg.requires_action && !msg.actionSaved && <button className="chat-chip accent" onClick={() => patchMsg(index, { showAction: !msg.showAction })}><Wrench size={12} /> I did something</button>}
        {msg.actionSaved && <span className="chat-saved"><Check size={13} /> action saved — it becomes a precedent</span>}
      </div>
      {msg.showAction && !msg.actionSaved && <ActionForm msg={msg} onSaved={() => patchMsg(index, { actionSaved: true, showAction: false })} />}
      {msg.err && <div className="chat-err">{msg.err}</div>}
    </div>
  );
}
