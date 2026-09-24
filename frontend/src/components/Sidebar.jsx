import React, { useEffect, useState } from 'react';
import { Rocket, MapPin, BarChart2, MessageSquare, AlertTriangle, Zap } from 'lucide-react';

export default function Sidebar({ activeTab, onTabChange }) {
  const [apiUp, setApiUp] = useState(false);

  useEffect(() => {
    fetch('/api/status')
      .then(res => setApiUp(res.ok))
      .catch(() => setApiUp(false));
  }, []);

  const navItems = [
    { id: 'network', icon: MapPin, label: 'Network' },
    { id: 'analytics', icon: BarChart2, label: 'Analytics' },
    { id: 'chat', icon: MessageSquare, label: 'Chat' },
    { id: 'disruptions', icon: AlertTriangle, label: 'Disruptions' },
    { id: 'energy', icon: Zap, label: 'Energy' },
  ];

  return (
    <div className="sidebar">
      <div style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Rocket size={28} color="var(--teal)" />
          <div>
            <h1 style={{ fontSize: '1.25rem', color: 'var(--teal)', margin: 0, fontWeight: 700 }}>Talk To My Train</h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>InnoTrans 2026</p>
          </div>
        </div>
      </div>

      <nav style={{ flex: 1, padding: '0 12px' }}>
        {navItems.map(item => {
          const active = activeTab === item.id;
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px 16px',
                marginBottom: '8px',
                background: active ? 'var(--teal-dim)' : 'transparent',
                border: 'none',
                borderLeft: active ? '3px solid var(--teal)' : '3px solid transparent',
                borderRadius: '0 8px 8px 0',
                color: active ? 'var(--teal)' : 'var(--text-muted)',
                cursor: 'pointer',
                textAlign: 'left',
                fontSize: '0.9rem',
                fontWeight: 500,
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => !active && (e.currentTarget.style.background = 'var(--surface2)')}
              onMouseLeave={(e) => !active && (e.currentTarget.style.background = 'transparent')}
            >
              <Icon size={20} />
              {item.label}
            </button>
          );
        })}
      </nav>

      <div style={{ padding: '24px', borderTop: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: apiUp ? 'var(--teal)' : 'var(--danger)' }} />
          API Status
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <span style={{ fontSize: '0.7rem', padding: '2px 6px', background: 'var(--surface2)', borderRadius: '4px', color: 'var(--text-muted)' }}>gpt-5.6-luna</span>
          <span style={{ fontSize: '0.7rem', padding: '2px 6px', background: 'var(--surface2)', borderRadius: '4px', color: 'var(--text-muted)' }}>HCADE v1.0</span>
        </div>
      </div>
    </div>
  );
}
