import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";

export const OPERATOR = "operator-1";
export const STAGES = [
  { id: "dispatcher", icon: "route", title: "Dispatcher", text: "Understands the question, routes it, rejects off-topic or manipulative requests" },
  { id: "analyst", icon: "tools", title: "Analyst", text: "Pulls the data through MCP connectors and runs the load forecast" },
  { id: "inspector", icon: "check", title: "Inspector", text: "Recomputes key numbers from the raw data before anything is shown" },
  { id: "writer", icon: "write", title: "Writer", text: "Produces a one-screen brief: verdict, evidence, do-now, caveat, sources" },
];
const fresh = () => STAGES.map((s) => ({ ...s, status: "pending", details: [], seconds: null }));
export const short = (t, n = 64) => (t.length > n ? t.slice(0, n - 1).trimEnd() + "…" : t);

const Ctx = createContext(null);
export const useChat = () => useContext(Ctx);

/** One conversation state shared by the floating assistant (Toby) and the full-page Chat Copilot:
 *  messages, session / thread, streaming stages, history, topic-switch choice, feedback, pending action reports. */
export function ChatProvider({ children, snap }) {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [thread, setThread] = useState({ title: "", linkTurn: null });
  const [thinking, setThinking] = useState(false);
  const [stages, setStages] = useState(fresh());
  const [convs, setConvs] = useState([]);
  const [pending, setPending] = useState(0);
  const [open, setOpen] = useState(false);          // the floating overlay
  const stRef = useRef({ sessionId: null, thread: { title: "", linkTurn: null }, thinking: false });
  stRef.current = { sessionId, thread, thinking };

  const refreshConvs = useCallback(() => api.conversations(OPERATOR).then(setConvs).catch(() => {}), []);
  const refreshPending = useCallback(() => api.pending(OPERATOR).then((p) => setPending(p.length)).catch(() => {}), []);
  useEffect(() => { refreshConvs(); refreshPending(); }, [refreshConvs, refreshPending]);

  const onStep = (step) => setStages((prev) => {
    const idx = prev.findIndex((s) => s.id === step.stage);
    if (idx < 0) return prev;
    return prev.map((s, i) => {
      if (i < idx && s.status !== "done") return { ...s, status: "done" };
      if (i === idx) return { ...s, status: "active", details: [...s.details, step].slice(-12), seconds: (s.seconds || 0) + (step.seconds || 0) };
      return s;
    });
  });

  /** Send a message. mode: auto | continue | new. `already` = the user bubble is already in the list (after a topic choice). */
  const send = useCallback(async (raw, mode = "auto", already = false) => {
    const text = (raw || "").trim();
    if (!text || stRef.current.thinking) return;
    if (!already) setMessages((ms) => [...ms, { role: "user", text }]);
    setThinking(true);
    setStages(fresh());
    const { sessionId: sid, thread: th } = stRef.current;
    const body = { message: text, operator_id: OPERATOR, session_id: sid || undefined, context_mode: mode, link_turn_id: sid ? undefined : th.linkTurn || undefined };
    let answered = false;
    try {
      await api.chatStream(body, {
        choice: (c) => { answered = true; setMessages((ms) => [...ms, { role: "choice", ...c }]); },
        step: onStep,
        answer: (r) => {
          answered = true;
          setStages((prev) => prev.map((s) => ({ ...s, status: "done" })));
          setSessionId(r.session_id);
          setThread((t) => ({ ...t, title: t.title || r.context?.title || short(text, 80) }));
          setMessages((ms) => [...ms, { role: "agent", text: r.answer, ...r }]);
          refreshConvs(); refreshPending();
        },
        error: (e) => { answered = true; setMessages((ms) => [...ms, { role: "agent", error: true, text: e.message }]); },
      });
      if (!answered) throw new Error("the stream ended without an answer");
    } catch (e) {
      if (!answered) {                                            // streaming not available: fall back to the plain request
        try {
          const r = await api.chat(text, OPERATOR, sid || undefined, { context_mode: mode, link_turn_id: sid ? undefined : th.linkTurn || undefined });
          if (r.needs_choice) setMessages((ms) => [...ms, { role: "choice", ...r }]);
          else { setSessionId(r.session_id); setThread((t) => ({ ...t, title: t.title || r.context?.title || short(text, 80) })); setMessages((ms) => [...ms, { role: "agent", text: r.answer, ...r }]); refreshConvs(); refreshPending(); }
        } catch (e2) { setMessages((ms) => [...ms, { role: "agent", error: true, text: e2.message }]); }
      }
    } finally { setThinking(false); }
  }, [refreshConvs, refreshPending]);

  /** The operator's answer to "different topic — new conversation or continue?" */
  const choose = useCallback(async (index, choice) => {
    const c = messages[index];
    if (!c) return;
    if (choice === "new") {
      setSessionId(null);
      setThread({ title: short(c.message, 80), linkTurn: null });
      setMessages([{ role: "user", text: c.message }]);              // a clean page: the old conversation stays in the history
      stRef.current = { ...stRef.current, sessionId: null, thread: { title: "", linkTurn: null } };
      await send(c.message, "new", true);
    } else {
      setMessages((ms) => ms.map((m, k) => (k === index ? { role: "note", text: "Connected to the previous topic." } : m)));
      await send(c.message, "continue", true);
    }
  }, [messages, send]);

  const newConversation = useCallback(() => { setMessages([]); setSessionId(null); setThread({ title: "", linkTurn: null }); }, []);

  const openConversation = useCallback(async (c) => {
    try {
      const d = await api.conversation(c.last_session_id);
      const ms = [];
      for (const t of d.turns) {
        ms.push({ role: "user", text: t.question });
        ms.push({ role: "agent", text: t.answer, turn_id: t.turn_id, artifact_turn_id: t.has_artifact ? t.turn_id : null, requires_action: t.requires_action, score: t.operator_score ? Math.round(t.operator_score) : undefined,
                  actionSaved: t.action_reported, answer_mode: "brief", loaded: true });
      }
      setMessages(ms); setSessionId(null); setThread({ title: d.title, linkTurn: d.resume_turn_id });
    } catch (e) { setMessages((m) => [...m, { role: "agent", error: true, text: e.message }]); }
  }, []);

  const patchMsg = useCallback((i, p) => setMessages((ms) => ms.map((m, k) => (k === i ? { ...m, ...p } : m))), []);

  /** Ask from anywhere in the UI (map cards, closures): opens the assistant and sends the question. */
  const ask = useCallback((q) => { setOpen(true); setTimeout(() => send(q), 50); }, [send]);

  const suggestions = useMemo(() => {
    const s = [];
    for (const c of snap?.closures || []) s.push({ label: `Closure now: ${c.line || "station"} — what should we do?`, q: closureQuestion(c) });
    s.push({ label: "U7 closure on 25 Sept: where to deploy staff?", q: "Line U7 is suspended between Hermannplatz and Karl-Marx-Strasse on 2026-09-25 from 20:45 for 2 hours. What is the reason, how should passengers be rerouted, which stations would become overloaded, and where should additional staff be deployed?" });
    s.push({ label: "InnoTrans day: 3 highest-load stations", q: "On 2026-09-23, during InnoTrans, which 3 stations are most likely to see the highest load, and what should the control room do about it?" });
    s.push({ label: "Rudow: when is the commute peak?", q: "At what time does the commute flow peak at Rudow station usually take place? Does it exceed the mean commute peak value across all stations?" });
    s.push({ label: "Which line has the worst energy per passenger?", q: "Which metro line has the worst energy-per-passenger efficiency ratio? What factors explain this inefficiency?" });
    s.push({ label: "Anomalies on 19 July", q: "Identify three passenger-flow anomalies that cannot be explained by station closures on July 19th. Determine the most likely root causes using all of the available data." });
    return s.slice(0, 5);
  }, [snap]);

  const value = { messages, sessionId, thread, thinking, stages, convs, pending, open, setOpen, send, choose, newConversation, openConversation, patchMsg, ask, suggestions, refreshConvs, refreshPending, operator: OPERATOR };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

const shortName = (n) => (n || "?").replace(/ \(Berlin\)|^S\+U |^U /g, "");
export function closureQuestion(c) {
  const d = c.start.slice(0, 10), t = c.start.slice(11, 16);
  const hours = Math.max(1, Math.round((new Date(c.end) - new Date(c.start)) / 36e5));
  if (c.kind === "station") return `Station ${shortName(c.station)} is closed on ${d} from ${t} for ${hours} hours. Which neighbouring stations will be affected and where should additional staff be deployed?`;
  return `Line ${c.line} is suspended between ${shortName(c.from)} and ${shortName(c.to)} on ${d} from ${t} for ${hours} hours. How should passengers be rerouted, which stations would become overloaded, and where should additional staff be deployed?`;
}
