import React, { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api.js";
import MapView from "./components/MapView.jsx";
import TimeBar, { slotMs } from "./components/TimeBar.jsx";
import { LinePanel, RightPanel, closureQuestion } from "./components/Panels.jsx";
import Avatar, { loadPos } from "./components/Avatar.jsx";
import ChatPanel from "./components/ChatPanel.jsx";

const OPERATOR = "operator-1";

export default function App() {
  const [topo, setTopo] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [at, setAt] = useState(null);
  const [snap, setSnap] = useState(null);
  const [series, setSeries] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [lineFilter, setLineFilter] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");
  // assistant
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [thread, setThread] = useState({ title: "", linkTurn: null });
  const [thinking, setThinking] = useState(false);
  const [prefill, setPrefill] = useState(null);
  const [pending, setPending] = useState(0);
  const [pos, setPosState] = useState(loadPos);
  const setPos = (p) => { setPosState(p); try { localStorage.setItem("toby.pos", JSON.stringify(p)); } catch { /* private mode */ } };

  useEffect(() => { Promise.all([api.topology(), api.timeline()]).then(([t, tl]) => { setTopo(t); setTimeline(tl); setAt(tl.default_at.slice(0, 19)); }).catch((e) => setError(e.message)); }, []);
  useEffect(() => {
    if (!at) return;
    const h = setTimeout(() => api.snapshot(at).then(setSnap).catch((e) => setError(e.message)), 120);
    return () => clearTimeout(h);
  }, [at]);
  const day = at?.slice(0, 10);
  useEffect(() => { if (day) api.series(day).then(setSeries).catch(() => setSeries(null)); }, [day]);
  const refreshPending = useCallback(() => api.pending(OPERATOR).then((p) => setPending(p.length)).catch(() => {}), []);
  useEffect(() => { refreshPending(); }, [refreshPending]);

  const ask = (q) => { setPrefill(q); setOpen(true); };
  const suggestions = useMemo(() => {
    const s = [];
    for (const c of snap?.closures || []) s.push({ label: `Closure now: ${c.line || "station"} — what should we do?`, q: closureQuestion(c) });
    s.push({ label: "U7 closure on 25 Sept: where to deploy staff?", q: "Line U7 is suspended between Hermannplatz and Karl-Marx-Strasse on 2026-09-25 from 20:45 for 2 hours. What is the reason, how should passengers be rerouted, which stations would become overloaded, and where should additional staff be deployed?" });
    s.push({ label: "InnoTrans day: 3 highest-load stations", q: "On 2026-09-23, during InnoTrans, which 3 stations are most likely to see the highest load, and what should the control room do about it?" });
    s.push({ label: "Rudow: when is the commute peak?", q: "At what time does the commute flow peak at Rudow station usually take place? Does it exceed the mean commute peak value across all stations?" });
    s.push({ label: "Anomalies on 19 July", q: "Identify three passenger-flow anomalies that cannot be explained by station closures on July 19th. Determine the most likely root causes using all of the available data." });
    return s.slice(0, 5);
  }, [snap]);

  if (error && !topo) return <div className="boot err">Cannot reach the operator API: {error}<br />Start it with <code>make up</code> (http://127.0.0.1:8770).</div>;
  if (!topo || !timeline || !at) return <div className="boot">Loading the network…</div>;

  const clock = new Date(slotMs(at));
  const net = snap?.network;
  const w = snap?.weather;
  return (
    <div className="desk">
      <header className="top">
        <div className="brand"><b>Talk To My Train</b><span>Operator desktop · replay of recorded data</span></div>
        <div className="kpi"><small>Passengers / 15 min</small><b>{net ? net.total.toLocaleString() : "—"}</b><em className={net?.ratio > 1.3 ? "warn" : ""}>{net?.ratio ? `${net.ratio}× typical` : ""}</em></div>
        <div className="kpi"><small>Closures</small><b className={snap?.closures?.length ? "crit" : ""}>{snap?.closures?.length ?? "—"}</b></div>
        <div className="kpi"><small>Alerts</small><b>{snap?.alerts?.length ?? "—"}</b></div>
        <div className="kpi"><small>Weather</small><b>{w ? `${Math.round(w.temp)}°C` : "—"}</b><em>{w ? (w.prcp > 0 ? `rain ${w.prcp} mm` : "dry") : ""}</em></div>
        <div className="clock"><b>{clock.toISOString().slice(11, 16)}</b><small>{clock.toISOString().slice(0, 10)} · {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][clock.getUTCDay()]}</small></div>
      </header>
      <main className="grid">
        <div className="left"><LinePanel snap={snap} lineFilter={lineFilter} setLineFilter={setLineFilter} /></div>
        <div className="center">
          <MapView topo={topo} snap={snap} lineFilter={lineFilter} selected={selected} onSelect={setSelected} />
          <TimeBar timeline={timeline} at={at} setAt={setAt} playing={playing} setPlaying={setPlaying} speed={speed} setSpeed={setSpeed} series={series} />
        </div>
        <div className="right"><RightPanel snap={snap} topo={topo} selected={selected} onSelect={setSelected} onAsk={ask} /></div>
      </main>
      <Avatar open={open} onToggle={() => setOpen((o) => !o)} badge={pending} thinking={thinking} hint={pending ? `${pending} to close` : "Ask me"} pos={pos} setPos={setPos} />
      {open && (
        <ChatPanel operator={OPERATOR} sessionId={sessionId} setSessionId={setSessionId} messages={messages} setMessages={setMessages} prefill={prefill} clearPrefill={() => setPrefill(null)}
                   onMinimize={() => setOpen(false)} onDockPick={(c) => { setPos({ corner: c }); setOpen(false); }}
                   thinking={thinking} setThinking={setThinking} onAnswered={() => refreshPending()} suggestions={suggestions} thread={thread} setThread={setThread} />
      )}
    </div>
  );
}
