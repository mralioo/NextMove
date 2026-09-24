import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, Cell, ReferenceLine 
} from 'recharts';
import { api } from '../api/client';
import { Zap, AlertCircle, CheckCircle, Info, ChevronDown, ChevronUp } from 'lucide-react';

const LINE_COLORS = {
  U1: '#55B09E', U2: '#DA4A2F', U3: '#006B35',
  U4: '#FFCC00', U5: '#7D4C3C', U6: '#7B2481',
  U7: '#009BD2', U8: '#005FAD', U9: '#F5A623'
};

export default function Energy() {
  const [energyData, setEnergyData] = useState([]);
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [expandedLine, setExpandedLine] = useState(null);

  useEffect(() => {
    api.energy()
      .then(data => {
        const r = Array.isArray(data?.ranking) ? data.ranking : [];
        setEnergyData(r);
        setMeta(data);
        setExpandedLine(r[0]?.line || null);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const worst = energyData[0];
  const best = energyData[energyData.length - 1];
  const mean = energyData.length ? Math.round(energyData.reduce((acc, curr) => acc + (curr.wh_per_pax || 0), 0) / energyData.length) : 0;

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '10px', color: 'var(--teal)' }}>
        <div style={{ width: '22px', height: '22px', border: '2px solid var(--teal)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
        <span>Computing energy efficiency metrics...</span>
      </div>
    );
  }

  if (error || !worst) {
    return <div className="card" style={{ color: 'var(--danger)' }}>Energy data not available{error ? `: ${error}` : ''}.</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      
      {/* 4 KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
        <div className="card metric-card">
          <span className="label">Least Efficient Line</span>
          <span className="value" style={{ color: 'var(--danger)' }}>{worst.line}</span>
        </div>
        <div className="card metric-card">
          <span className="label">Worst Wh / passenger</span>
          <span className="value" style={{ color: 'var(--danger)' }}>{Math.round(worst.wh_per_pax)}</span>
        </div>
        <div className="card metric-card">
          <span className="label">Most Efficient Line</span>
          <span className="value" style={{ color: 'var(--teal)' }}>{best.line}</span>
        </div>
        <div className="card metric-card">
          <span className="label">Best Wh / passenger</span>
          <span className="value" style={{ color: 'var(--teal)' }}>{Math.round(best.wh_per_pax)}</span>
        </div>
      </div>

      {/* Main Bar Chart */}
      <div className="card" style={{ height: '340px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <div>
            <h3 style={{ margin: 0, color: 'var(--text)', fontSize: '0.95rem' }}>
              Energy per Passenger (Wh / passenger)
            </h3>
            <p style={{ margin: 0, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Ranked from worst (left) to best (right); lower is better. {meta?.method}
            </p>
          </div>
          <span className="pill-badge active">Line mean: {mean} Wh/pax</span>
        </div>

        <div style={{ flex: 1, width: '100%', minHeight: 0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={energyData} margin={{ top: 15, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,212,0.06)" />
              <XAxis dataKey="line" stroke="#678A96" tick={{ fill: '#E0EEF4', fontSize: 11, fontWeight: 600 }} />
              <YAxis stroke="#678A96" tick={{ fill: '#678A96', fontSize: 10 }} />
              <Tooltip formatter={(val) => [`${val} Wh per passenger`, 'Energy']} />
              <ReferenceLine y={mean} stroke="var(--gold)" strokeDasharray="3 3" label={{ value: 'Mean', fill: 'var(--gold)', fontSize: 10 }} />
              <Bar dataKey="wh_per_pax" radius={[6, 6, 0, 0]}>
                {energyData.map((entry, index) => {
                  const color = index === 0 ? 'var(--danger)' : (index === energyData.length - 1 ? 'var(--teal)' : 'rgba(0, 229, 212, 0.45)');
                  return <Cell key={`cell-${index}`} fill={color} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Line Details & Intervention Recommendations */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <h3 style={{ margin: 0, color: 'var(--text)', fontSize: '0.95rem' }}>
          Line-by-Line Evidence
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {energyData.map((item) => {
            const isExpanded = expandedLine === item.line;
            const lineCol = LINE_COLORS[item.line] || 'var(--teal)';

            return (
              <div
                key={item.line}
                style={{
                  background: isExpanded ? 'rgba(8, 26, 34, 0.85)' : 'rgba(5, 20, 26, 0.5)',
                  border: `1px solid ${isExpanded ? 'rgba(0, 229, 212, 0.3)' : 'var(--border-subtle)'}`,
                  borderRadius: 'var(--radius-sm)',
                  padding: '12px 16px',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onClick={() => setExpandedLine(isExpanded ? null : item.line)}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{
                      background: lineCol,
                      color: '#000',
                      padding: '2px 10px',
                      borderRadius: 'var(--radius-pill)',
                      fontWeight: 800,
                      fontSize: '0.75rem'
                    }}>
                      {item.line}
                    </span>
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text)' }}>
                      Rank #{item.efficiency_rank} · {Math.round(item.wh_per_pax)} Wh / pax · {item.mwh_day} MWh / day
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono' }}>
                      {item.pax_day ? `${item.pax_day.toLocaleString()} pax / day` : ''}
                    </span>
                    {isExpanded ? <ChevronUp size={16} color="var(--teal)" /> : <ChevronDown size={16} color="var(--text-muted)" />}
                  </div>
                </div>

                {isExpanded && (
                  <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid rgba(255, 255, 255, 0.06)', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    <div style={{ marginBottom: '6px' }}>
                      <strong style={{ color: 'var(--text)' }}>What the data shows: </strong>
                      {item.explanation}
                    </div>
                    {meta?.limits?.length ? <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Limits: {meta.limits.join('; ')}</div> : null}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}
