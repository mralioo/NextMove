import React, { useState, useEffect } from 'react';
import { 
  MessageSquare, MapPin, BarChart2, AlertTriangle, Zap, 
  Wifi, Activity
} from 'lucide-react';
import Network from './pages/Network';
import Analytics from './pages/Analytics';
import Chat from './pages/Chat';
import Disruptions from './pages/Disruptions';
import Energy from './pages/Energy';
import { api } from './api/client';

export default function App() {
  const [tab, setTab] = useState('chat');
  const [apiStatus, setApiStatus] = useState({ loaded: false, stations: 168, flows: 8320 });

  useEffect(() => {
    api.status()
      .then(res => {
        if (res.status === 'ok') {
          setApiStatus({ loaded: true, stations: res.stations_count, flows: res.flows_rows });
        }
      })
      .catch(() => setApiStatus({ loaded: false, stations: 168, flows: 8320 }));
  }, []);

  const navTabs = [
    { id: 'chat', label: 'Chat Copilot', icon: MessageSquare },
    { id: 'network', label: 'Network & Heatmap', icon: MapPin },
    { id: 'analytics', label: 'Flow Analytics', icon: BarChart2 },
    { id: 'disruptions', label: 'Cascade Simulator', icon: AlertTriangle },
    { id: 'energy', label: 'Energy Efficiency', icon: Zap },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      
      {/* Top Capsule Navigation Bar */}
      <header style={{
        height: '64px',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(2, 11, 14, 0.85)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid var(--border-subtle)',
        zIndex: 50,
        flexShrink: 0
      }}>
        {/* Brand / Logo with sleek modern Metro icon */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '38px', height: '38px', borderRadius: '10px',
            background: 'linear-gradient(135deg, rgba(0, 229, 212, 0.22) 0%, rgba(245, 197, 24, 0.12) 100%)',
            border: '1px solid rgba(0, 229, 212, 0.35)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 14px rgba(0, 229, 212, 0.2)'
          }}>
            {/* Aerodynamic bullet train emblem */}
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path d="M4 14.5C4 10.5 7.5 5 12 5C16.5 5 20 10.5 20 14.5C20 17 18 18 12 18C6 18 4 17 4 14.5Z" stroke="var(--teal)" strokeWidth="1.8" fill="rgba(0, 229, 212, 0.12)" />
              <path d="M7 10C7.5 8 9.5 7 12 7C14.5 7 16.5 8 17 10" stroke="var(--teal)" strokeWidth="1.6" strokeLinecap="round" />
              <circle cx="8" cy="15" r="1.3" fill="var(--gold)" />
              <circle cx="16" cy="15" r="1.3" fill="var(--gold)" />
              <line x1="12" y1="13" x2="12" y2="16" stroke="var(--teal)" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M2 21H22" stroke="var(--teal)" strokeWidth="1.5" strokeLinecap="round" strokeOpacity="0.4" />
            </svg>
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontFamily: 'Space Grotesk', fontWeight: 700, fontSize: '1.05rem', letterSpacing: '0.04em', color: 'var(--text)' }}>
                TALK TO MY TRAIN
              </span>
              <span style={{ fontSize: '0.62rem', background: 'rgba(0,229,212,0.15)', color: 'var(--teal)', border: '1px solid rgba(0,229,212,0.3)', padding: '1px 7px', borderRadius: 'var(--radius-pill)', fontWeight: 600 }}>
                INNOTRANS 2026
              </span>
            </div>
            <p style={{ margin: 0, fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              Historical-Context-Aware Decision Engine (HCADE)
            </p>
          </div>
        </div>

        {/* Center Pill Tabs */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          background: 'rgba(6, 22, 29, 0.9)',
          padding: '4px 6px',
          borderRadius: 'var(--radius-pill)',
          border: '1px solid var(--border-subtle)'
        }}>
          {navTabs.map(t => {
            const active = tab === t.id;
            const Icon = t.icon;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`pill-tab ${active ? 'active' : ''}`}
                style={{
                  fontSize: '0.82rem',
                  padding: '7px 16px',
                  fontWeight: active ? 600 : 400
                }}
              >
                <Icon size={14} />
                {t.label}
              </button>
            );
          })}
        </div>

        {/* Right Info Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="pill-badge active" style={{ fontSize: '0.72rem', padding: '4px 12px' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--gold)', animation: 'pulseGlow 2s infinite' }} />
            ACTIVE
          </div>
          
          <div style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            fontSize: '0.75rem', color: 'var(--text-muted)',
            background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)',
            padding: '4px 12px', borderRadius: 'var(--radius-pill)'
          }}>
            <Wifi size={13} color="var(--teal)" />
            <span>gpt-5.6-luna</span>
          </div>
        </div>
      </header>

      {/* Main Workspace — all tabs always mounted, hidden via display:none so Chat WS stays alive */}
      <main style={{
        flex: 1,
        overflow: 'hidden',
        padding: '20px 24px',
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        maxWidth: '1600px',
        margin: '0 auto',
        position: 'relative'
      }}>
        <div style={{ display: tab === 'chat' ? 'flex' : 'none', flex: 1, flexDirection: 'column', overflow: 'hidden', height: '100%' }}>
          <Chat />
        </div>
        <div style={{ display: tab === 'network' ? 'flex' : 'none', flex: 1, flexDirection: 'column', overflow: 'hidden', height: '100%' }}>
          <Network />
        </div>
        <div style={{ display: tab === 'analytics' ? 'flex' : 'none', flex: 1, flexDirection: 'column', overflow: 'hidden', height: '100%' }}>
          <Analytics />
        </div>
        <div style={{ display: tab === 'disruptions' ? 'flex' : 'none', flex: 1, flexDirection: 'column', overflow: 'hidden', height: '100%' }}>
          <Disruptions />
        </div>
        <div style={{ display: tab === 'energy' ? 'flex' : 'none', flex: 1, flexDirection: 'column', overflow: 'hidden', height: '100%' }}>
          <Energy />
        </div>
      </main>

    </div>
  );
}
