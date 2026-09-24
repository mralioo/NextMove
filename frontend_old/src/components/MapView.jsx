import React, { useMemo, useRef, useState } from "react";

const W = 1000, H = 760, PAD = 36;

export function heat(ratio) {
  if (ratio == null) return "#6b7280";
  if (ratio < 0.7) return "#4b9cd3";
  if (ratio < 1.25) return "#3ddc97";
  if (ratio < 1.8) return "#f5c542";
  if (ratio < 2.5) return "#ff8c42";
  return "#ff4d4f";
}

/** The city topology: stations at their coordinates, edges in line colours, station dots coloured by load vs typical, active closures in red. Wheel = zoom, drag = pan. */
export default function MapView({ topo, snap, lineFilter, selected, onSelect }) {
  const [view, setView] = useState({ x: 0, y: 0, w: W, h: H });
  const [hover, setHover] = useState(null);
  const drag = useRef(null);
  const svgRef = useRef(null);

  const proj = useMemo(() => {
    const b = topo.bbox;
    const k = Math.cos(((b.min_lat + b.max_lat) / 2) * Math.PI / 180);
    const spanX = (b.max_lon - b.min_lon) * k, spanY = b.max_lat - b.min_lat;
    const s = Math.min((W - 2 * PAD) / spanX, (H - 2 * PAD) / spanY);
    const ox = (W - spanX * s) / 2, oy = (H - spanY * s) / 2;
    const map = {};
    for (const st of topo.stations) map[st.id] = [ox + (st.lon - b.min_lon) * k * s, oy + (b.max_lat - st.lat) * s];
    return map;
  }, [topo]);

  const color = useMemo(() => Object.fromEntries(topo.lines.map((l) => [l.line, l.color])), [topo]);
  const stById = useMemo(() => Object.fromEntries(topo.stations.map((s) => [s.id, s])), [topo]);
  const dim = (lines) => lineFilter && !lines.includes(lineFilter);
  const blocked = useMemo(() => new Set((snap?.closures || []).flatMap((c) => c.blocked_edges.map(([a, b]) => [a, b].sort().join("|")))), [snap]);
  const closedStations = useMemo(() => new Set((snap?.closures || []).flatMap((c) => [...(c.unserved || []), ...(c.kind === "station" && c.station ? [c.station] : [])])), [snap]);
  const topIds = useMemo(() => new Set((snap?.top || []).slice(0, 6).map((t) => t.id)), [snap]);

  const onWheel = (e) => {
    e.preventDefault();
    const rect = svgRef.current.getBoundingClientRect();
    const f = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    const px = view.x + ((e.clientX - rect.left) / rect.width) * view.w, py = view.y + ((e.clientY - rect.top) / rect.height) * view.h;
    const w = Math.min(W * 1.5, Math.max(120, view.w * f)), h = w * (H / W);
    setView({ x: px - ((px - view.x) / view.w) * w, y: py - ((py - view.y) / view.h) * h, w, h });
  };
  const onDown = (e) => { drag.current = { x: e.clientX, y: e.clientY, v: view, moved: false }; };
  const onMove = (e) => {
    const d = drag.current;
    if (!d) return;
    const rect = svgRef.current.getBoundingClientRect();
    const dx = ((e.clientX - d.x) / rect.width) * d.v.w, dy = ((e.clientY - d.y) / rect.height) * d.v.h;
    if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 4) d.moved = true;
    if (d.moved) setView({ ...d.v, x: d.v.x - dx, y: d.v.y - dy });
  };
  const onUp = () => { drag.current = null; };
  const zoomBy = (f) => setView((v) => { const w = Math.min(W * 1.5, Math.max(120, v.w * f)), h = w * (H / W); return { x: v.x + (v.w - w) / 2, y: v.y + (v.h - h) / 2, w, h }; });
  const scale = view.w / W;

  return (
    <div className="map-wrap">
      <svg ref={svgRef} viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`} onWheel={onWheel} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}
           onClick={() => { if (!drag.current?.moved) onSelect(null); }}>
        <g>
          {topo.edges.map((e, i) => {
            const [x1, y1] = proj[e.a] || [], [x2, y2] = proj[e.b] || [];
            if (x1 === undefined || x2 === undefined) return null;
            const isBlocked = blocked.has([e.a, e.b].sort().join("|"));
            const c = e.lines[0] ? color[e.lines[0]] : "#4b5563";
            return <g key={i} opacity={dim(e.lines) ? 0.12 : 1}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={c} strokeWidth={3.2 * Math.max(0.5, scale)} strokeLinecap="round" opacity={isBlocked ? 0.35 : 0.95} />
              {e.lines.length > 1 && <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color[e.lines[1]]} strokeWidth={1.4 * Math.max(0.5, scale)} strokeLinecap="round" />}
              {isBlocked && <line className="blocked" x1={x1} y1={y1} x2={x2} y2={y2} stroke="#ff4d4f" strokeWidth={5 * Math.max(0.5, scale)} strokeDasharray="6 5" strokeLinecap="round" />}
            </g>;
          })}
        </g>
        <g>
          {topo.stations.map((s) => {
            const p = proj[s.id];
            if (!p) return null;
            const d = snap?.stations?.[s.id];
            const r = (3.4 + Math.min(7, Math.sqrt(d?.v || 0) / 9) + (s.lines.length > 1 ? 1.2 : 0)) * Math.max(0.55, scale);
            const closed = closedStations.has(s.id);
            const isSel = selected === s.id;
            return <g key={s.id} opacity={dim(s.lines) ? 0.15 : 1} style={{ cursor: "pointer" }}
                      onPointerEnter={(e) => setHover({ s, d, x: e.clientX, y: e.clientY })} onPointerMove={(e) => setHover((h) => h && { ...h, x: e.clientX, y: e.clientY })} onPointerLeave={() => setHover(null)}
                      onClick={(e) => { e.stopPropagation(); onSelect(s.id); }}>
              {isSel && <circle cx={p[0]} cy={p[1]} r={r + 6 * Math.max(0.55, scale)} fill="none" stroke="#fff" strokeWidth={1.5} />}
              <circle cx={p[0]} cy={p[1]} r={r} fill={closed ? "#111827" : heat(d?.ratio)} stroke={s.lines.length > 1 ? "#fff" : "#0b1220"} strokeWidth={s.lines.length > 1 ? 1.6 : 1} />
              {closed && <text x={p[0]} y={p[1] + 4} textAnchor="middle" fontSize={11 * Math.max(0.6, scale)} fill="#ff4d4f" fontWeight="700">×</text>}
              {(isSel || topIds.has(s.id) || (d?.ratio || 0) >= 2 || scale < 0.45 && s.lines.length > 1) && <text x={p[0] + r + 3} y={p[1] + 3} fontSize={10.5 * Math.max(0.65, scale)} fill="#e5e7eb" className="lbl">{s.name}</text>}
            </g>;
          })}
        </g>
      </svg>
      <div className="map-legend">
        <span>load vs typical</span>
        {[["<0.7", 0.5], ["normal", 1], ["1.25×", 1.5], ["1.8×", 2], ["2.5×+", 3]].map(([t, v]) => <i key={t}><b style={{ background: heat(v) }} />{t}</i>)}
        <i><b style={{ background: "#111827", border: "1px solid #ff4d4f" }} />closed</i>
      </div>
      <div className="map-zoom"><button onClick={() => zoomBy(1 / 1.3)}>+</button><button onClick={() => zoomBy(1.3)}>−</button><button onClick={() => setView({ x: 0, y: 0, w: W, h: H })}>⟲</button></div>
      {hover && (
        <div className="tip" style={{ left: hover.x + 14, top: hover.y + 12 }}>
          <b>{hover.s.name}</b>
          <div className="chips">{hover.s.lines.map((l) => <span key={l} className="chip" style={{ background: color[l] }}>{l}</span>)}</div>
          {hover.d ? <div>{hover.d.v} passengers / 15 min<br />typical {hover.d.base}{hover.d.ratio ? ` · ${hover.d.ratio}×` : ""}</div> : <div>no flow data</div>}
        </div>
      )}
    </div>
  );
}
