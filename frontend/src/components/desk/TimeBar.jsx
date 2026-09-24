import React, { useEffect, useMemo, useRef } from "react";

const STEP = 15 * 60 * 1000;
const toLocal = (iso) => new Date(iso.length > 19 ? iso : iso + "Z");           // the API sends naive local times: keep them as wall-clock via UTC arithmetic
const fmtDate = (ms) => new Date(ms).toISOString().slice(0, 10);
const fmtTime = (ms) => new Date(ms).toISOString().slice(11, 16);

export function slotMs(iso) { return new Date(iso.slice(0, 19) + "Z").getTime(); }
export function isoFrom(ms) { return new Date(ms).toISOString().slice(0, 19); }

/** Replay clock: slider over the whole data window, date / time pickers, play, jump to a closure, sparkline of the network for the day. */
export default function TimeBar({ timeline, at, setAt, playing, setPlaying, speed, setSpeed, series }) {
  const t0 = slotMs(timeline.start), t1 = slotMs(timeline.end), cur = slotMs(at);
  const timer = useRef(null);
  useEffect(() => {
    clearInterval(timer.current);
    if (playing) timer.current = setInterval(() => setAt((a) => { const n = slotMs(a) + STEP; if (n > t1) { setPlaying(false); return a; } return isoFrom(n); }), 1000 / speed);
    return () => clearInterval(timer.current);
  }, [playing, speed, t1]);

  const spark = useMemo(() => {
    if (!series?.length) return null;
    const max = Math.max(...series.map((s) => Math.max(s.total, s.typical || 0))) || 1;
    const x = (i) => (i / (series.length - 1)) * 100;
    const y = (v) => 30 - (v / max) * 28;
    const path = (k) => series.map((s, i) => (s[k] == null ? null : `${i ? "L" : "M"}${x(i).toFixed(2)},${y(s[k]).toFixed(2)}`)).filter(Boolean).join(" ");
    const idx = series.findIndex((s) => slotMs(s.t) === cur);
    return { total: path("total"), typical: path("typical"), cx: idx >= 0 ? x(idx) : null };
  }, [series, cur]);

  return (
    <div className="dk-timebar">
      <div className="dk-tb-controls">
        <button onClick={() => setAt(isoFrom(cur - 4 * STEP))} title="−1 h">⏮</button>
        <button className="dk-play" onClick={() => setPlaying(!playing)}>{playing ? "❚❚" : "▶"}</button>
        <button onClick={() => setAt(isoFrom(cur + 4 * STEP))} title="+1 h">⏭</button>
        <select value={speed} onChange={(e) => setSpeed(+e.target.value)} title="replay speed"><option value={1}>1×</option><option value={3}>3×</option><option value={8}>8×</option></select>
        <input type="date" value={fmtDate(cur)} min={fmtDate(t0)} max={fmtDate(t1)} onChange={(e) => e.target.value && setAt(`${e.target.value}T${fmtTime(cur)}:00`)} />
        <input type="time" step={900} value={fmtTime(cur)} onChange={(e) => e.target.value && setAt(`${fmtDate(cur)}T${e.target.value}:00`)} />
        <select value="" onChange={(e) => { const c = timeline.closures.find((x) => String(x.id) === e.target.value); if (c) setAt(isoFrom(slotMs(c.when) + STEP)); }} title="jump to a recorded closure">
          <option value="">jump to a closure…</option>
          {timeline.closures.map((c) => <option key={c.id} value={c.id}>{c.when.slice(5, 16).replace("T", " ")} · {c.label.replace(/ due to.*/, "").slice(0, 60)}</option>)}
        </select>
      </div>
      <div className="dk-tb-slider">
        <input type="range" min={t0} max={t1} step={STEP} value={cur} onChange={(e) => setAt(isoFrom(+e.target.value))} />
        {timeline.closures.map((c) => <i key={c.id} className="dk-tick" style={{ left: `${((slotMs(c.when) - t0) / (t1 - t0)) * 100}%` }} title={c.label} />)}
      </div>
      {spark && (
        <div className="dk-spark" title="network passengers per 15 minutes on this day (line) vs typical (dashed)">
          <svg viewBox="0 0 100 32" preserveAspectRatio="none"><path d={spark.typical} className="dk-typ" /><path d={spark.total} className="dk-tot" />{spark.cx != null && <line x1={spark.cx} x2={spark.cx} y1="0" y2="32" className="dk-cur" />}</svg>
        </div>
      )}
    </div>
  );
}
