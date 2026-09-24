import React, { useEffect, useState } from "react";
import { api } from "../api/client";
import MapView from "../components/desk/MapView";
import TimeBar, { slotMs } from "../components/desk/TimeBar";
import { LinePanel, RightPanel } from "../components/desk/Panels";
import { useChat } from "../state/ChatContext";

/** The operator desk: replay of the recorded data with the network map, alerts, closures, events and the time bar. "Ask Toby" buttons open the assistant with a ready-made question. */
export default function Desk({ snap, setSnap }) {
  const { ask } = useChat();
  const [topo, setTopo] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [at, setAt] = useState(null);
  const [series, setSeries] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [lineFilter, setLineFilter] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => { Promise.all([api.topology(), api.timeline()]).then(([t, tl]) => { setTopo(t); setTimeline(tl); setAt(tl.default_at.slice(0, 19)); }).catch((e) => setError(e.message)); }, []);
  useEffect(() => {
    if (!at) return;
    const h = setTimeout(() => api.snapshot(at).then(setSnap).catch((e) => setError(e.message)), 120);
    return () => clearTimeout(h);
  }, [at]);
  const day = at?.slice(0, 10);
  useEffect(() => { if (day) api.series(day).then(setSeries).catch(() => setSeries(null)); }, [day]);

  if (error && !topo) return <div className="dk-boot err">Cannot reach the operator API: {error}<br />Start it with <code>make up</code>.</div>;
  if (!topo || !timeline || !at) return <div className="dk-boot">Loading the network…</div>;
  const clock = new Date(slotMs(at));
  const net = snap?.network, w = snap?.weather;
  return (
    <div className="dk-desk">
      <div className="dk-kpis">
        <div className="card metric-card"><span className="label">Passengers / 15 min</span><span className="value">{net ? net.total.toLocaleString() : "—"}</span><em className={net?.ratio > 1.3 ? "warn" : ""}>{net?.ratio ? `${net.ratio}× typical` : ""}</em></div>
        <div className="card metric-card"><span className="label">Closures now</span><span className="value" style={{ color: snap?.closures?.length ? "var(--danger)" : "var(--text)" }}>{snap?.closures?.length ?? "—"}</span></div>
        <div className="card metric-card"><span className="label">Alerts</span><span className="value" style={{ color: "var(--gold)" }}>{snap?.alerts?.length ?? "—"}</span></div>
        <div className="card metric-card"><span className="label">Weather</span><span className="value">{w ? `${Math.round(w.temp)}°C` : "—"}</span><em>{w ? (w.prcp > 0 ? `rain ${w.prcp} mm` : "dry") : ""}</em></div>
        <div className="card metric-card"><span className="label">Replay clock</span><span className="value mono">{clock.toISOString().slice(11, 16)}</span><em>{clock.toISOString().slice(0, 10)} · {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][clock.getUTCDay()]}</em></div>
      </div>
      <div className="dk-grid">
        <div className="dk-left"><LinePanel snap={snap} lineFilter={lineFilter} setLineFilter={setLineFilter} /></div>
        <div className="dk-center">
          <MapView topo={topo} snap={snap} lineFilter={lineFilter} selected={selected} onSelect={setSelected} />
          <TimeBar timeline={timeline} at={at} setAt={setAt} playing={playing} setPlaying={setPlaying} speed={speed} setSpeed={setSpeed} series={series} />
        </div>
        <div className="dk-right"><RightPanel snap={snap} topo={topo} selected={selected} onSelect={setSelected} onAsk={ask} /></div>
      </div>
    </div>
  );
}
