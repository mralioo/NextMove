import React, { useEffect, useRef, useState } from "react";

export const loadPos = () => { try { return JSON.parse(localStorage.getItem("toby.pos")) || { corner: "br" }; } catch { return { corner: "br" }; } };
const SIZE = 84, M = 18;
const corners = () => ({ tl: [M, 76], tr: [innerWidth - SIZE - M, 76], bl: [M, innerHeight - SIZE - M - 96], br: [innerWidth - SIZE - M, innerHeight - SIZE - M - 96] });

export function TobySvg({ talking, size = SIZE }) {
  return (
    <svg className={"toby" + (talking ? " talking" : "")} width={size} height={size} viewBox="0 0 100 100" aria-label="Toby, the assistant">
      <defs><linearGradient id="tb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#19f0de" /><stop offset="1" stopColor="#0a8f86" /></linearGradient></defs>
      <ellipse cx="50" cy="94" rx="30" ry="4" fill="rgba(0,0,0,.35)" />
      <rect x="14" y="18" width="72" height="66" rx="22" fill="url(#tb)" stroke="#c9fff9" strokeWidth="2.5" />
      <rect x="24" y="26" width="52" height="28" rx="12" fill="#031218" />
      <g className="eyes"><circle cx="40" cy="40" r="6.5" fill="#e6f4ff" /><circle cx="60" cy="40" r="6.5" fill="#e6f4ff" /><circle className="pupil" cx="41" cy="41" r="3" fill="#031218" /><circle className="pupil" cx="61" cy="41" r="3" fill="#031218" /></g>
      <path className="mouth" d="M40 62 Q50 70 60 62" fill="none" stroke="#e6f4ff" strokeWidth="3" strokeLinecap="round" />
      <circle cx="28" cy="74" r="4.5" fill="#fde68a" /><circle cx="72" cy="74" r="4.5" fill="#fde68a" />
      <rect x="30" y="8" width="40" height="12" rx="6" fill="#F5C518" stroke="#ffe680" strokeWidth="2" />
      <rect x="44" y="4" width="12" height="7" rx="3" fill="#ffd84d" />
      <rect x="36" y="84" width="10" height="6" rx="2" fill="#06454a" /><rect x="54" y="84" width="10" height="6" rx="2" fill="#06454a" />
    </svg>
  );
}

/** The floating assistant. Drag it anywhere, drop it near a corner and it snaps there; click (no drag) to bring it to the middle of the page as the conversation. */
export default function Toby({ open, onToggle, badge, thinking, hint, pos, setPos }) {
  const [drag, setDrag] = useState(null);
  const [, force] = useState(0);
  const ref = useRef(null);
  useEffect(() => { const f = () => force((n) => n + 1); addEventListener("resize", f); return () => removeEventListener("resize", f); }, []);

  const xy = pos.corner ? corners()[pos.corner] : [Math.min(innerWidth - SIZE, Math.max(0, pos.x)), Math.min(innerHeight - SIZE, Math.max(0, pos.y))];
  const [x, y] = drag ? [drag.x, drag.y] : xy;

  const down = (e) => {
    if (open) return;
    ref.current.setPointerCapture(e.pointerId);
    setDrag({ sx: e.clientX, sy: e.clientY, ox: x, oy: y, x, y, moved: false });
  };
  const move = (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
    setDrag({ ...drag, x: drag.ox + dx, y: drag.oy + dy, moved: drag.moved || Math.abs(dx) + Math.abs(dy) > 5 });
  };
  const up = () => {
    if (!drag) return;
    if (!drag.moved) { setDrag(null); onToggle(); return; }
    const c = corners();
    let best = null, bd = 1e9;
    for (const [k, [cx, cy]] of Object.entries(c)) { const d = Math.hypot(cx - drag.x, cy - drag.y); if (d < bd) { bd = d; best = k; } }
    setPos(bd < 170 ? { corner: best } : { x: drag.x, y: drag.y });
    setDrag(null);
  };

  const style = open ? { left: "50%", top: "calc(50% - 300px)", transform: "translate(-50%, -50%)" } : { left: x, top: y };
  return (
    <div ref={ref} className={"avatar" + (open ? " open" : "") + (drag?.moved ? " dragging" : "")} style={style} onPointerDown={down} onPointerMove={move} onPointerUp={up} title={open ? "" : "Ask Toby — drag me to a corner"}>
      <TobySvg talking={thinking} size={open ? 96 : SIZE} />
      {!open && badge > 0 && <span className="badge" title="situations waiting for your action report">{badge}</span>}
      {!open && !drag?.moved && <span className="hint">{hint || "Ask me"}</span>}
    </div>
  );
}
