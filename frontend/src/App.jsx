import React, { useEffect, useState } from "react";
import { MessageSquare, MapPin, BarChart2, AlertTriangle, Zap, Wifi, LayoutDashboard } from "lucide-react";
import Desk from "./pages/Desk";
import Network from "./pages/Network";
import Analytics from "./pages/Analytics";
import Chat from "./pages/Chat";
import Disruptions from "./pages/Disruptions";
import Energy from "./pages/Energy";
import ChatOverlay from "./components/ChatOverlay";
import { ChatProvider } from "./state/ChatContext";
import { api } from "./api/client";

const TABS = [
  { id: "desk", label: "Operator Desk", icon: LayoutDashboard },
  { id: "chat", label: "Chat Copilot", icon: MessageSquare },
  { id: "network", label: "Network & Heatmap", icon: MapPin },
  { id: "analytics", label: "Flow Analytics", icon: BarChart2 },
  { id: "disruptions", label: "Cascade Simulator", icon: AlertTriangle },
  { id: "energy", label: "Energy Efficiency", icon: Zap },
];

export default function App() {
  const [tab, setTab] = useState(() => localStorage.getItem("ttmt.tab") || "desk");
  const [status, setStatus] = useState(null);
  const [snap, setSnap] = useState(null);          // the desk's replay snapshot: the assistant suggests the closure that is active there
  useEffect(() => { try { localStorage.setItem("ttmt.tab", tab); } catch { /* private mode */ } }, [tab]);
  useEffect(() => {
    const load = () => api.status().then(setStatus).catch(() => setStatus({ status: "down" }));
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);
  const up = status?.status === "ok";
  return (
    <ChatProvider snap={snap}>
      <div className="app-shell">
        <header className="app-top">
          <div className="brand">
            <div className="logo">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M4 14.5C4 10.5 7.5 5 12 5C16.5 5 20 10.5 20 14.5C20 17 18 18 12 18C6 18 4 17 4 14.5Z" stroke="var(--teal)" strokeWidth="1.8" fill="rgba(0,229,212,0.12)" /><path d="M7 10C7.5 8 9.5 7 12 7C14.5 7 16.5 8 17 10" stroke="var(--teal)" strokeWidth="1.6" strokeLinecap="round" /><circle cx="8" cy="15" r="1.3" fill="var(--gold)" /><circle cx="16" cy="15" r="1.3" fill="var(--gold)" /><path d="M2 21H22" stroke="var(--teal)" strokeWidth="1.5" strokeLinecap="round" strokeOpacity="0.4" /></svg>
            </div>
            <div>
              <div className="title">NEXTMOVE <span className="pill-badge teal">INNOTRANS 2026</span></div>
              <p>Dispatcher · Analyst · Inspector · Writer — an AI team for the control room</p>
            </div>
          </div>
          <div className="tabs">
            {TABS.map((t) => { const I = t.icon; return <button key={t.id} onClick={() => setTab(t.id)} className={`pill-tab ${tab === t.id ? "active" : ""}`}><I size={14} />{t.label}</button>; })}
          </div>
          <div className="right">
            <div className={`pill-badge ${up ? "active" : "idle"}`}><span className="pulse" />{up ? "API UP" : "API DOWN"}</div>
            <div className="model" title={status?.models ? Object.entries(status.models).map(([k, v]) => `${k}: ${v}`).join("\n") : ""}><Wifi size={13} color={status?.agent_up ? "var(--teal)" : "var(--danger)"} /><span>{status?.models?.writer || "agent"}</span></div>
          </div>
        </header>
        <main className="app-main">
          <div className={"page" + (tab === "desk" ? " on" : "")}><Desk snap={snap} setSnap={setSnap} /></div>
          <div className={"page" + (tab === "chat" ? " on" : "")}><Chat /></div>
          <div className={"page pad" + (tab === "network" ? " on" : "")}>{tab === "network" && <Network />}</div>
          <div className={"page pad" + (tab === "analytics" ? " on" : "")}>{tab === "analytics" && <Analytics />}</div>
          <div className={"page pad" + (tab === "disruptions" ? " on" : "")}>{tab === "disruptions" && <Disruptions />}</div>
          <div className={"page pad" + (tab === "energy" ? " on" : "")}>{tab === "energy" && <Energy />}</div>
        </main>
        {tab !== "chat" && <ChatOverlay />}
      </div>
    </ChatProvider>
  );
}
