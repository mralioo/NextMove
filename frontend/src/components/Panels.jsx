import React from "react";
import { heat } from "./MapView.jsx";

export function LinePanel({ snap, lineFilter, setLineFilter }) {
  return (
    <section className="panel">
      <h3>Lines <small>load vs typical</small></h3>
      {(snap?.lines || []).map((l) => (
        <div key={l.line} className={"line-row" + (lineFilter === l.line ? " on" : "")} onClick={() => setLineFilter(lineFilter === l.line ? null : l.line)}>
          <span className="chip" style={{ background: l.color }}>{l.line}</span>
          <div className="bar"><i style={{ width: `${Math.min(100, ((l.ratio || 0) / 2) * 100)}%`, background: heat(l.ratio) }} /><b style={{ left: "50%" }} /></div>
          <span className="num">{l.load.toLocaleString()}<em>{l.ratio ? ` ${l.ratio}×` : ""}</em></span>
        </div>
      ))}
      {lineFilter && <button className="link" onClick={() => setLineFilter(null)}>show all lines</button>}
    </section>
  );
}

export function RightPanel({ snap, topo, selected, onAsk, onSelect }) {
  const st = selected ? topo.stations.find((s) => s.id === selected) : null;
  const d = st && snap?.stations?.[st.id];
  return (
    <>
      {st && (
        <section className="panel sel">
          <h3>{st.name} <small>{st.lines.join(" · ")}</small></h3>
          <div className="big">{d ? d.v : "—"}<small> passengers / 15 min</small></div>
          <div className="muted">typical {d?.base ?? "—"}{d?.ratio ? ` · ${d.ratio}× typical` : ""}</div>
          <button className="ask" onClick={() => onAsk(`Tell me about ${st.name}: at what time does its commute flow peak usually, and does it exceed the mean commute peak across all stations?`)}>Ask Toby about this station</button>
        </section>
      )}
      <section className="panel">
        <h3>Alerts <small>{snap?.alerts?.length || 0}</small></h3>
        {(snap?.alerts || []).length === 0 && <div className="muted">Nothing unusual at this time.</div>}
        {(snap?.alerts || []).map((a, i) => (
          <div key={i} className={"alert " + a.level} onClick={() => a.station && onSelect(a.station)}>{a.text}</div>
        ))}
      </section>
      <section className="panel">
        <h3>Closures now <small>{snap?.closures?.length || 0}</small></h3>
        {(snap?.closures || []).length === 0 && <div className="muted">No recorded closure is active.</div>}
        {(snap?.closures || []).map((c) => (
          <div key={c.id} className="closure">
            <b>{c.line || "Station"} · {c.kind === "station" ? c.station?.replace(/ \(Berlin\)|^S\+U |^U /g, "") : `${short(c.from)} ↔ ${short(c.to)}`}</b>
            <div className="muted">{c.reason} · {c.start.slice(11, 16)}–{c.end.slice(11, 16)}</div>
            <button className="ask" onClick={() => onAsk(closureQuestion(c))}>What should we do?</button>
          </div>
        ))}
      </section>
      <section className="panel">
        <h3>Events around <small>{snap?.events?.length || 0}</small></h3>
        {(snap?.events || []).length === 0 && <div className="muted">No larger event nearby in time.</div>}
        {(snap?.events || []).map((e, i) => (
          <div key={i} className="event"><b>{e.name.slice(0, 44)}</b><div className="muted">{e.venue || "venue n/a"} · {e.phase.replace("_", " ")}{e.attendance ? ` · ${e.attendance}` : ""}</div></div>
        ))}
      </section>
      <section className="panel">
        <h3>Busiest stations</h3>
        {(snap?.top || []).map((t) => (
          <div key={t.id} className="top-row" onClick={() => onSelect(t.id)}><span>{t.station}</span><span className="num">{t.v}<em>{t.ratio ? ` ${t.ratio}×` : ""}</em></span></div>
        ))}
      </section>
    </>
  );
}

const short = (n) => (n || "?").replace(/ \(Berlin\)|^S\+U |^U /g, "");

export function closureQuestion(c) {
  const d = c.start.slice(0, 10), t = c.start.slice(11, 16);
  const hours = Math.max(1, Math.round((new Date(c.end) - new Date(c.start)) / 36e5));
  if (c.kind === "station") return `Station ${short(c.station)} is closed on ${d} from ${t} for ${hours} hours. Which neighbouring stations will be affected and where should additional staff be deployed?`;
  return `Line ${c.line} is suspended between ${short(c.from)} and ${short(c.to)} on ${d} from ${t} for ${hours} hours. How should passengers be rerouted, which stations would become overloaded, and where should additional staff be deployed?`;
}
