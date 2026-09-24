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
import logo from "../assets/logo.webp";

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
            <img className="logo" src={logo} alt="NextMove" />
            <div>
              <div className="title"><span className="pill-badge teal">INNOTRANS 2026</span></div>
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
