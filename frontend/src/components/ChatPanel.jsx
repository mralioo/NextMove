import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import { api } from "../api.js";
import OpsLog from "./OpsLog.jsx";

const PHASES = ["Dispatcher routing…", "Analyst calling the data tools…", "Inspector checking the numbers…", "Writer preparing the brief…"];
const FOLLOW = [["Why?", "why?"], ["Evidence", "show me the evidence"], ["Which tools?", "which tools did you call?"]];

function Stars({ value, onPick, disabled }) {
  return <span className="stars">{[1, 2, 3, 4, 5].map((n) => <button key={n} disabled={disabled} className={n <= (value || 0) ? "on" : ""} onClick={() => onPick(n)} title={`${n}/5`}>★</button>)}</span>;
}

function ActionForm({ msg, operator, onSaved }) {
  const [text, setText] = useState("");
  const [followed, setFollowed] = useState("as_recommended");
  const [outcome, setOutcome] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const submit = async () => {
    setBusy(true); setErr("");
    try { await api.action(msg.artifact_turn_id || msg.turn_id, operator, { action_text: text, followed, outcome: outcome || undefined }); onSaved(); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="action-form">
      <label>What did you do about it?</label>
      <textarea rows={2} value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. sent two staff to Neukölln and ordered the replacement bus" />
      <div className="row">
        <select value={followed} onChange={(e) => setFollowed(e.target.value)}>
          <option value="as_recommended">as recommended</option><option value="modified">modified</option><option value="different">something else</option><option value="none">did nothing</option>
        </select>
        <select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
          <option value="">outcome: not yet known</option><option value="worked">it worked</option><option value="partly">partly</option><option value="did_not_work">did not work</option>
        </select>
        <button className="primary" disabled={busy || text.trim().length < 3} onClick={submit}>{busy ? "Saving…" : "Save"}</button>
      </div>
      {err && <div className="err">{err}</div>}
    </div>
  );
}

function Feedback({ msg, operator, onAsk, patch }) {
  const id = msg.artifact_turn_id || msg.turn_id;
  if (!id) return null;
  const rate = async (n) => { try { await api.score(id, operator, n); patch({ score: n }); } catch (e) { patch({ err: e.message }); } };
  return (
    <div className="fb">
      <div className="fb-row">
        <span className="muted">Rate</span><Stars value={msg.score} onPick={rate} />
        {msg.answer_mode === "brief" && FOLLOW.map(([l, q]) => <button key={l} className="chipbtn" onClick={() => onAsk(q)}>{l}</button>)}
        {msg.requires_action && !msg.actionSaved && <button className="chipbtn accent" onClick={() => patch({ showAction: !msg.showAction })}>I did something</button>}
        {msg.actionSaved && <span className="saved">✓ action saved — it becomes a precedent</span>}
      </div>
      {msg.showAction && !msg.actionSaved && <ActionForm msg={msg} operator={operator} onSaved={() => patch({ actionSaved: true, showAction: false })} />}
      {msg.err && <div className="err">{msg.err}</div>}
    </div>
  );
}

export default function ChatPanel({ operator, sessionId, setSessionId, messages, setMessages, prefill, clearPrefill, onMinimize, onDockPick, thinking, setThinking, onAnswered, suggestions }) {
  const [text, setText] = useState("");
  const [phase, setPhase] = useState(0);
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, thinking]);
  useEffect(() => { if (thinking) { const t = setInterval(() => setPhase((p) => Math.min(p + 1, PHASES.length - 1)), 1400); return () => { clearInterval(t); setPhase(0); }; } }, [thinking]);
  useEffect(() => { if (prefill) { send(prefill); clearPrefill(); } }, [prefill]);

  const patchMsg = (i) => (p) => setMessages((ms) => ms.map((m, k) => (k === i ? { ...m, ...p } : m)));
  const send = async (raw) => {
    const q = (raw ?? text).trim();
    if (!q || thinking) return;
    setText("");
    setMessages((ms) => [...ms, { role: "user", text: q }]);
    setThinking(true);
    try {
      const r = await api.chat(q, operator, sessionId || undefined);
      setSessionId(r.session_id);
      setMessages((ms) => [...ms, { role: "agent", text: r.answer, ...r }]);
      onAnswered(r);
    } catch (e) {
      setMessages((ms) => [...ms, { role: "agent", error: true, text: e.message }]);
    } finally { setThinking(false); }
  };
  const lastAgent = [...messages].reverse().find((m) => m.role === "agent" && !m.error);

  return (
    <div className="chat-backdrop" onClick={onMinimize}>
      <div className="chat" onClick={(e) => e.stopPropagation()}>
        <header>
          <div><b>Toby</b><small>your control-room assistant · advisory only, you decide</small></div>
          <div className="hdr-actions">
            <button title="new conversation" onClick={() => { setMessages([]); setSessionId(null); }}>New</button>
            <button title="dock in the bottom-right corner" onClick={() => onDockPick("br")}>↘</button>
            <button title="dock in the top-left corner" onClick={() => onDockPick("tl")}>↖</button>
            <button title="minimise" onClick={onMinimize}>✕</button>
          </div>
        </header>
        <div className="chat-body">
          <div className="conv">
            <div className="msgs">
              {messages.length === 0 && (
                <div className="welcome">
                  <p>Ask in plain words. I answer with a short brief; say <i>why</i>, <i>evidence</i> or <i>which tools</i> for the full picture.</p>
                  <div className="sugg">{suggestions.map((s, i) => <button key={i} onClick={() => send(s.q)}>{s.label}</button>)}</div>
                </div>
              )}
              {messages.map((m, i) => (
                <div key={i} className={"msg " + m.role + (m.error ? " error" : "")}>
                  {m.role === "user" ? <div className="bubble">{m.text}</div> : (
                    <div className="bubble"><ReactMarkdown remarkPlugins={[remarkBreaks]}>{m.text}</ReactMarkdown>{!m.error && <Feedback msg={m} operator={operator} patch={patchMsg(i)} onAsk={send} />}</div>
                  )}
                </div>
              ))}
              {thinking && <div className="msg agent"><div className="bubble thinking"><span className="dots"><i /><i /><i /></span> {PHASES[phase]}</div></div>}
              <div ref={endRef} />
            </div>
            <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask about a closure, an event, a station…" autoFocus />
              <button className="primary" disabled={thinking || !text.trim()}>Send</button>
            </form>
          </div>
          <OpsLog last={lastAgent} totals={messages.filter((m) => m.role === "agent" && !m.error).reduce((a, m) => ({
            questions: a.questions + 1, tools: a.tools + (m.counts?.tool_calls || 0), inference: a.inference + (m.llm || []).reduce((s, r) => s + (r.seconds || 0), 0), tokens: a.tokens + (m.tokens?.total || 0),
          }), { questions: 0, tools: 0, inference: 0, tokens: 0 })} />
        </div>
      </div>
    </div>
  );
}
