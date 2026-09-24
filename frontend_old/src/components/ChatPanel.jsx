import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import { api } from "../api.js";
import OpsLog from "./OpsLog.jsx";

const CAT = { A: "Events", B: "Anomalies", C: "Closures", D: "Stations", E: "Energy", F: "Network", G: "Correlations", H: "Reroute", P: "Pressure", X: "Strategy" };
const ago = (ts) => { const m = Math.max(0, (Date.now() / 1000 - ts) / 60); return m < 1 ? "just now" : m < 60 ? `${Math.round(m)} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`; };
const short = (t, n = 64) => (t.length > n ? t.slice(0, n - 1).trimEnd() + "…" : t);
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

export default function ChatPanel({ operator, sessionId, setSessionId, messages, setMessages, prefill, clearPrefill, onMinimize, onDockPick, thinking, setThinking, onAnswered, suggestions, thread, setThread }) {
  const [text, setText] = useState("");
  const [histOpen, setHistOpen] = useState(false);
  const [convs, setConvs] = useState([]);
  const refreshConvs = () => api.conversations(operator).then(setConvs).catch(() => {});
  useEffect(() => { refreshConvs(); }, []);
  const [phase, setPhase] = useState(0);
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, thinking]);
  useEffect(() => { if (thinking) { const t = setInterval(() => setPhase((p) => Math.min(p + 1, PHASES.length - 1)), 1400); return () => { clearInterval(t); setPhase(0); }; } }, [thinking]);
  useEffect(() => { if (prefill) { send(prefill); clearPrefill(); } }, [prefill]);

  const patchMsg = (i) => (p) => setMessages((ms) => ms.map((m, k) => (k === i ? { ...m, ...p } : m)));
  const send = async (raw, mode = "auto", already = false) => {
    const q = (raw ?? text).trim();
    if (!q || thinking) return;
    setText("");
    if (!already) setMessages((ms) => [...ms, { role: "user", text: q }]);
    setThinking(true);
    try {
      const r = await api.chat(q, operator, sessionId || undefined, { context_mode: mode, link_turn_id: sessionId ? undefined : thread.linkTurn || undefined });
      if (r.needs_choice) { setMessages((ms) => [...ms, { role: "choice", ...r }]); return; }
      setSessionId(r.session_id);
      setThread((t) => ({ ...t, title: t.title || r.context?.title || short(q, 80) }));
      setMessages((ms) => [...ms, { role: "agent", text: r.answer, ...r }]);
      onAnswered(r); refreshConvs();
    } catch (e) {
      setMessages((ms) => [...ms, { role: "agent", error: true, text: e.message }]);
    } finally { setThinking(false); }
  };
  // the operator's choice for a message that does not look connected to the conversation
  const choose = async (i, choice) => {
    const c = messages[i];
    if (choice === "new") {
      setSessionId(null); setThread({ title: short(c.message, 80), linkTurn: null });
      setMessages([{ role: "user", text: c.message }]);                                    // a clean page: the old conversation stays in the history
      await send(c.message, "new", true);
    } else {
      setMessages((ms) => ms.map((m, k) => (k === i ? { role: "note", text: "Connected to the previous topic." } : m)));
      await send(c.message, "continue", true);
    }
  };
  const newConversation = () => { setMessages([]); setSessionId(null); setThread({ title: "", linkTurn: null }); };
  const openConversation = async (c) => {
    try {
      const d = await api.conversation(c.last_session_id);
      const ms = [];
      for (const t of d.turns) {
        ms.push({ role: "user", text: t.question });
        ms.push({ role: "agent", text: t.answer, turn_id: t.turn_id, artifact_turn_id: t.has_artifact ? t.turn_id : null, requires_action: t.requires_action, score: t.operator_score ? Math.round(t.operator_score) : undefined,
                  actionSaved: t.action_reported, answer_mode: "brief", loaded: true, counts: null });
      }
      setMessages(ms); setSessionId(null); setThread({ title: d.title, linkTurn: d.resume_turn_id }); setHistOpen(false);
    } catch (e) { setMessages((m) => [...m, { role: "agent", error: true, text: e.message }]); }
  };
  const lastAgent = [...messages].reverse().find((m) => m.role === "agent" && !m.error && m.tokens);

  return (
    <div className="chat-backdrop" onClick={onMinimize}>
      <div className="chat" onClick={(e) => e.stopPropagation()}>
        <header>
          <div><b>Toby</b><small>your control-room assistant · advisory only, you decide</small></div>
          <div className="hdr-actions">
            <button title="chat history" className={histOpen ? "on" : ""} onClick={() => { setHistOpen(!histOpen); refreshConvs(); }}>History</button>
            <button title="start a new conversation" onClick={newConversation}>New</button>
            <button title="dock in the bottom-right corner" onClick={() => onDockPick("br")}>↘</button>
            <button title="dock in the top-left corner" onClick={() => onDockPick("tl")}>↖</button>
            <button title="minimise" onClick={onMinimize}>✕</button>
          </div>
        </header>
        <div className={"chat-body" + (histOpen ? " hist" : "")}>
          {histOpen && (
            <nav className="history">
              <h4>Conversations <button onClick={newConversation}>＋ New</button></h4>
              {convs.length === 0 && <div className="muted">No earlier conversation yet.</div>}
              {convs.map((c) => (
                <div key={c.conversation_id} className={"conv-item" + ((c.session_ids.includes(sessionId) || (thread.linkTurn && c.last_turn_id >= thread.linkTurn && c.title === thread.title)) ? " on" : "")} onClick={() => openConversation(c)}>
                  <div className="ttl">{short(c.title, 88)}</div>
                  <div className="meta"><span className="chip cat">{CAT[c.category] || c.category || "—"}</span><span>{c.n_turns} msg</span><span>{ago(c.last)}</span>{c.mean_score ? <span>★ {c.mean_score}</span> : null}{c.resumed ? <span>↻</span> : null}</div>
                </div>
              ))}
            </nav>
          )}
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
                  {m.role === "note" ? <div className="note">{m.text}</div> : m.role === "choice" ? (
                    <div className="choice">
                      <b>{m.relation === "unrelated" ? "This looks like a different topic." : "I'm not sure this is about the same topic."}</b>
                      <span className="why">{m.reason}. The conversation so far is about: <i>{m.anchor?.question}</i></span>
                      <div className="btns">
                        {m.choices.map((c) => <button key={c.id} className={c.id === m.recommended ? "primary" : "chipbtn"} title={c.hint} disabled={thinking} onClick={() => choose(i, c.id)}>{c.label}</button>)}
                      </div>
                      <small>Mixing topics in one conversation can bend the answer — a new conversation starts with a clean context.</small>
                    </div>
                  ) : m.role === "user" ? <div className="bubble">{m.text}</div> : (
                    <div className="bubble"><ReactMarkdown remarkPlugins={[remarkBreaks]}>{m.text}</ReactMarkdown>
                      {m.context?.mode === "continue" && <div className="ctxnote">↳ connected to the previous topic</div>}
                      {m.context?.mode === "resumed" && <div className="ctxnote">↳ continuing “{short(m.context.title, 48)}”</div>}
                      {!m.error && <Feedback msg={m} operator={operator} patch={patchMsg(i)} onAsk={(q) => send(q)} />}</div>
                  )}
                </div>
              ))}
              {thinking && <div className="msg agent"><div className="bubble thinking"><span className="dots"><i /><i /><i /></span> {PHASES[phase]}</div></div>}
              <div ref={endRef} />
            </div>
            {thread.title && messages.length > 0 && (
              <div className="ctxpill"><span>Topic</span><b title={thread.title}>{short(thread.title, 70)}</b><button onClick={newConversation} disabled={thinking}>Start a new topic</button></div>
            )}
            <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask about a closure, an event, a station…" autoFocus />
              <button className="primary" disabled={thinking || !text.trim()}>Send</button>
            </form>
          </div>
          <OpsLog last={lastAgent} totals={messages.filter((m) => m.role === "agent" && !m.error && m.tokens).reduce((a, m) => ({
            questions: a.questions + 1, tools: a.tools + (m.counts?.tool_calls || 0), inference: a.inference + (m.llm || []).reduce((s, r) => s + (r.seconds || 0), 0), tokens: a.tokens + (m.tokens?.total || 0),
          }), { questions: 0, tools: 0, inference: 0, tokens: 0 })} />
        </div>
      </div>
    </div>
  );
}
