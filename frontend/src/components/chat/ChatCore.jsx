import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import { Bot, Check, ChevronRight, Clock, Copy, History as HistoryIcon, Mic, MicOff, Plus, Sparkles, Volume2, VolumeX, ArrowUp, X } from "lucide-react";
import { short, useChat } from "../../state/ChatContext";
import { useSpeech } from "../../hooks/useSpeech";
import Feedback from "./Feedback";
import OpsLog from "./OpsLog";
import ThinkingPanel from "./ThinkingPanel";
import History from "./History";

function TopicChoice({ m, index }) {
  const { choose, thinking } = useChat();
  return (
    <div className="chat-choice">
      <b>{m.relation === "unrelated" ? "This looks like a different topic." : "I'm not sure this is about the same topic."}</b>
      <span className="why">{m.reason}. The conversation so far is about: <i>{m.anchor?.question}</i></span>
      <div className="btns">
        {m.choices.map((c) => <button key={c.id} className={c.id === m.recommended ? "chat-primary" : "chat-chip"} title={c.hint} disabled={thinking} onClick={() => choose(index, c.id)}>{c.label}</button>)}
      </div>
      <small>Mixing topics in one conversation can bend the answer — a new conversation starts with a clean context.</small>
    </div>
  );
}

function AgentMessage({ m, i }) {
  const { speak, stopSpeaking, isSpeaking } = useSpeech();
  const [copied, setCopied] = useState(false);
  return (
    <div className="chat-msg agent">
      <div className="chat-who"><span className="av"><Bot size={13} /></span><b>Toby</b></div>
      <div className={"msg-agent" + (m.error ? " chat-error" : "")}>
        <div className="md"><ReactMarkdown remarkPlugins={[remarkBreaks]}>{m.text}</ReactMarkdown></div>
        {m.context?.mode === "continue" && <div className="chat-ctxnote">↳ connected to the previous topic</div>}
        {m.context?.mode === "resumed" && <div className="chat-ctxnote">↳ continuing “{short(m.context.title, 48)}”</div>}
      </div>
      {!m.error && (
        <div className="chat-bar">
          <button className="pill-tab" onClick={() => (isSpeaking ? stopSpeaking() : speak(m.text))} title="Speak the answer">{isSpeaking ? <VolumeX size={13} /> : <Volume2 size={13} />}<span>{isSpeaking ? "Stop" : "Speak"}</span></button>
          <button className="pill-tab" onClick={() => { navigator.clipboard?.writeText(m.text); setCopied(true); setTimeout(() => setCopied(false), 1500); }} title="Copy the answer">{copied ? <Check size={13} /> : <Copy size={13} />}<span>{copied ? "Copied" : "Copy"}</span></button>
          {m.wall_s != null && <span className="pill-badge active mono"><Clock size={11} /> {m.timing?.total_s ?? m.wall_s}s</span>}
          {m.tokens && <span className="pill-badge idle mono">{m.tokens.total.toLocaleString()} tok · {m.counts?.tool_calls ?? 0} tools</span>}
        </div>
      )}
      {!m.error && <Feedback msg={m} index={i} />}
    </div>
  );
}

/** The whole assistant: conversation, history drawer, composer (typing + voice), live team progress while working, operations column after. `variant` = overlay | page. */
export default function ChatCore({ variant = "page", onClose, onDock }) {
  const { messages, thinking, stages, thread, send, newConversation, suggestions } = useChat();
  const [text, setText] = useState("");
  const [histOpen, setHistOpen] = useState(variant === "page");
  const endRef = useRef(null);
  const { startListening, stopListening, isListening, transcript } = useSpeech();
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, thinking, stages]);
  useEffect(() => { if (isListening && transcript) setText(transcript); }, [transcript, isListening]);
  const submit = (q) => { const v = (q ?? text).trim(); if (!v) return; setText(""); send(v); };

  const agents = messages.filter((m) => m.role === "agent" && !m.error && m.tokens);
  const last = agents[agents.length - 1];
  const totals = agents.reduce((a, m) => ({ questions: a.questions + 1, tools: a.tools + (m.counts?.tool_calls || 0), inference: a.inference + (m.llm || []).reduce((s, r) => s + (r.seconds || 0), 0), tokens: a.tokens + (m.tokens?.total || 0) }),
                              { questions: 0, tools: 0, inference: 0, tokens: 0 });

  return (
    <div className={"chat-core " + variant}>
      <header className="chat-head">
        <div><b>NextMove</b><small>your control-room assistant · advisory only, you decide</small></div>
        <div className="actions">
          <button className={"pill-tab" + (histOpen ? " active" : "")} onClick={() => setHistOpen(!histOpen)}><HistoryIcon size={13} /> History</button>
          <button className="pill-tab" onClick={newConversation}><Plus size={13} /> New</button>
          {variant === "overlay" && <><button className="pill-tab" title="dock in the bottom-right corner" onClick={() => onDock("br")}>↘</button><button className="pill-tab" title="dock in the top-left corner" onClick={() => onDock("tl")}>↖</button><button className="pill-tab" onClick={onClose}><X size={13} /></button></>}
        </div>
      </header>
      <div className={"chat-body" + (histOpen ? " hist" : "")}>
        {histOpen && <History onPicked={() => variant === "overlay" && setHistOpen(false)} />}
        <div className="chat-conv-col">
          <div className="chat-msgs">
            {messages.length === 0 && (
              <div className="chat-welcome">
                <div className="orb"><Sparkles size={26} /></div>
                <h2>NextMove</h2>
                <p>Ask in plain words. I answer with a short brief; say <i>why</i>, <i>evidence</i> or <i>which tools</i> for the full picture.</p>
                <div className="grid">{suggestions.map((s, i) => (
                  <button key={i} onClick={() => submit(s.q)}><div><b>{s.label}</b><span>{short(s.q, 78)}</span></div><ChevronRight size={14} /></button>
                ))}</div>
              </div>
            )}
            {messages.map((m, i) => (
              m.role === "note" ? <div key={i} className="chat-note">{m.text}</div>
              : m.role === "choice" ? <div key={i} className="chat-msg agent"><TopicChoice m={m} index={i} /></div>
              : m.role === "user" ? <div key={i} className="chat-msg user"><div className="msg-user">{m.text}</div></div>
              : <AgentMessage key={i} m={m} i={i} />
            ))}
            {thinking && variant === "overlay" && <ThinkingInline stages={stages} />}
            <div ref={endRef} />
          </div>
          {thread.title && messages.length > 0 && (
            <div className="chat-pill"><span>Topic</span><b title={thread.title}>{short(thread.title, 70)}</b><button onClick={newConversation} disabled={thinking}>Start a new topic</button></div>
          )}
          <form className="chat-composer" onSubmit={(e) => { e.preventDefault(); submit(); }}>
            <button type="button" className={"icon" + (isListening ? " rec" : "")} title="Speech to text" onClick={() => (isListening ? stopListening() : startListening((r) => submit(r)))}>{isListening ? <MicOff size={18} /> : <Mic size={18} />}</button>
            <input value={text} onChange={(e) => setText(e.target.value)} disabled={thinking} placeholder={isListening ? "Listening…" : "Ask about a closure, an event, a station…"} autoFocus />
            <button className="send" disabled={thinking || !text.trim()}><ArrowUp size={19} strokeWidth={2.5} /></button>
          </form>
        </div>
        {variant === "page" && thinking ? <ThinkingPanel stages={stages} /> : (variant === "page" || !thinking) ? <OpsLog last={last} totals={totals} /> : <ThinkingPanel stages={stages} />}
      </div>
    </div>
  );
}

function ThinkingInline({ stages }) {
  const active = stages.find((s) => s.status === "active") || stages.find((s) => s.status === "pending");
  return (
    <div className="chat-msg agent"><div className="chat-thinking"><span className="spinner" /><span>{active ? `${active.title}: ${active.text}…` : "Working…"}</span></div></div>
  );
}
