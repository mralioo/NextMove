import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, ReferenceLine, Cell 
} from 'recharts';
import { AlertTriangle, Play, ShieldAlert, CheckCircle2 } from 'lucide-react';

const LINE_COLORS = {
  U1: '#55B09E', U2: '#DA4A2F', U3: '#006B35',
  U4: '#FFCC00', U5: '#7D4C3C', U6: '#7B2481',
  U7: '#009BD2', U8: '#005FAD', U9: '#F5A623'
};

export default function Disruptions() {
  const [closures, setClosures] = useState([]);
  const [stations, setStations] = useState([]);
  const [selectedStations, setSelectedStations] = useState(['U Hermannplatz (Berlin)']);
  const [timestamp, setTimestamp] = useState('2026-09-25T20:45:00')
  const [share, setShare] = useState(0.5);
  const [hops, setHops] = useState(2);
  const [filter, setFilter] = useState('');
  const [error, setError] = useState('');;
  const [cascadeResult, setCascadeResult] = useState(null);
  const [loadingCascade, setLoadingCascade] = useState(false);

  useEffect(() => {
    api.closures().then(data => setClosures(Array.isArray(data) ? data : [])).catch(() => {});
    api.stations().then(data => setStations(Array.isArray(data) ? data : [])).catch(() => {});
  }, []);

  const runCascade = async () => {
    if (!selectedStations.length) return;
    setLoadingCascade(true);
    try {
      const res = await api.cascade({
        closed_stations: selectedStations,
        timestamp,
        share,
        hops
      });
      setError('');
      
      const stationsList = (res.all_stations || [])
        .filter(s => !s.is_closed)
        .slice(0, 15)
        .map(s => ({
          name: s.station.replace(" (Berlin)", "").replace("U ", ""),
          overflowRatio: Number((s.overflow_ratio || 1).toFixed(2)),
          extraPax: Math.round(s.extra_passengers || 0),
          atRisk: s.at_risk
        }));

      setCascadeResult({
        ...res,
        stationsList
      });
    } catch (e) {
      setError(e.message);
      setCascadeResult(null);
    }
    setLoadingCascade(false);
  };

  const toggleStation = (name) => {
    setSelectedStations(prev => 
      prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    );
  };

  const suspensionsCount = closures.filter(c => c.closure_type === 'line_suspension').length;
  const stationClosuresCount = closures.length - suspensionsCount;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      
      {/* 3 Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
        <div className="card metric-card">
          <span className="label">Total Recorded Disruptions</span>
          <span className="value" style={{ color: 'var(--gold)' }}>
            {closures.length}
          </span>
        </div>
        <div className="card metric-card">
          <span className="label">Line Suspensions</span>
          <span className="value" style={{ color: 'var(--danger)' }}>
            {suspensionsCount}
          </span>
        </div>
        <div className="card metric-card">
          <span className="label">Station Closures</span>
          <span className="value" style={{ color: 'var(--teal)' }}>
            {stationClosuresCount}
          </span>
        </div>
      </div>

      {/* Simulator Section */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ margin: 0, color: 'var(--text)', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <ShieldAlert size={18} color="var(--gold)" />
              Passenger Cascade & Crowd Redistribution Simulator
            </h3>
            <p style={{ margin: '2px 0 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              What-if: a chosen share of each closed station's typical flow (same weekday and 15-minute slot) is diverted to the open stations within a few hops of the network. The share is an assumption you set — the recorded data has no diversion behaviour.
            </p>
          </div>
          <button
            onClick={runCascade}
            disabled={loadingCascade || !selectedStations.length}
            className="pill-tab active"
            style={{
              background: 'linear-gradient(135deg, rgba(0, 229, 212, 0.25) 0%, rgba(245, 197, 24, 0.15) 100%)',
              border: '1px solid var(--teal)',
              padding: '8px 20px',
              whiteSpace: 'nowrap',
              fontSize: '0.82rem',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            <Play size={14} fill="var(--teal)" />
            {loadingCascade ? 'Computing Cascade...' : 'Run Simulation'}
          </button>
        </div>

        {/* Simulator Controls & Output */}
        <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '20px' }}>
          
          {/* Controls Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                Incident time (local Berlin time, e.g. 2026-09-25T20:45:00):
              </label>
              <input
                type="text"
                value={timestamp}
                onChange={e => setTimestamp(e.target.value)}
                style={{
                  width: '100%',
                  background: 'rgba(255, 255, 255, 0.04)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '8px 12px',
                  color: 'var(--text)',
                  fontSize: '0.85rem',
                  fontFamily: 'JetBrains Mono, monospace'
                }}
              />
            </div>

            <div style={{ display: 'flex', gap: '10px' }}>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', flex: 1 }}>Share diverted
                <select value={share} onChange={e => setShare(+e.target.value)} style={{ width: '100%', marginTop: '4px', background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '6px' }}>
                  {[0.25, 0.5, 0.75].map(v => <option key={v} value={v}>{Math.round(v * 100)} %</option>)}
                </select>
              </label>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', flex: 1 }}>Reach (hops)
                <select value={hops} onChange={e => setHops(+e.target.value)} style={{ width: '100%', marginTop: '4px', background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '6px' }}>
                  {[1, 2, 3].map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            </div>
            <div>
              <input placeholder="filter stations…" value={filter} onChange={e => setFilter(e.target.value)} style={{ width: '100%', background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '6px 10px', fontSize: '0.8rem', marginBottom: '8px' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Selected Closed Stations ({selectedStations.length}):
                </label>
                {selectedStations.length > 0 && (
                  <button
                    onClick={() => setSelectedStations([])}
                    style={{ background: 'none', border: 'none', color: 'var(--danger)', fontSize: '0.7rem', cursor: 'pointer' }}
                  >
                    Clear
                  </button>
                )}
              </div>

              {/* Station Selection Pills */}
              <div style={{
                maxHeight: '220px',
                overflowY: 'auto',
                display: 'flex',
                flexWrap: 'wrap',
                gap: '6px',
                padding: '8px',
                background: 'rgba(0,0,0,0.25)',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)'
              }}>
                {[...stations].filter(st => selectedStations.includes(st.name) || (st.name + st.short_name).toLowerCase().includes(filter.toLowerCase())).sort((a, b) => (selectedStations.includes(b.name) - selectedStations.includes(a.name)) || a.name.localeCompare(b.name)).map(st => {
                  const isSelected = selectedStations.includes(st.name);
                  return (
                    <button
                      key={st.name}
                      onClick={() => toggleStation(st.name)}
                      style={{
                        background: isSelected ? 'rgba(255, 92, 92, 0.25)' : 'rgba(255, 255, 255, 0.03)',
                        border: `1px solid ${isSelected ? 'var(--danger)' : 'var(--border-subtle)'}`,
                        color: isSelected ? '#FFFFFF' : 'var(--text-muted)',
                        padding: '3px 10px',
                        borderRadius: 'var(--radius-pill)',
                        fontSize: '0.7rem',
                        cursor: 'pointer'
                      }}
                    >
                      {st.short_name || st.name.replace(" (Berlin)", "").replace("U ", "")}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Results Bar Chart */}
          <div style={{ height: '320px', background: 'rgba(0,0,0,0.2)', borderRadius: 'var(--radius-md)', padding: '12px', border: '1px solid var(--border-subtle)' }}>
            {error ? (
              <div style={{ color: 'var(--danger)', fontSize: '0.85rem' }}>{error}</div>
            ) : !cascadeResult ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '8px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                <AlertTriangle size={24} color="var(--gold)" />
                <span>Select stations and click "Run Simulation" to model crowd redistribution</span>
              </div>
            ) : (
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', fontSize: '0.78rem' }}>
                  <span title={cascadeResult.assumption}>Load vs typical after diversion (1.0 = normal) · {cascadeResult.method}</span>
                  <span className="pill-badge active">
                    Max Overflow: {cascadeResult.max_overflow_ratio?.toFixed(2)}x
                  </span>
                </div>
                <div style={{ flex: 1, minHeight: 0 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={cascadeResult.stationsList} layout="vertical" margin={{ left: 10, right: 30, top: 5, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,212,0.06)" />
                      <XAxis type="number" domain={[0, 'dataMax + 0.3']} stroke="#678A96" tick={{ fill: '#678A96', fontSize: 10 }} />
                      <YAxis dataKey="name" type="category" stroke="#678A96" tick={{ fill: '#E0EEF4', fontSize: 9 }} width={100} />
                      <Tooltip formatter={(value, n, p) => [`${value}× typical (+${p.payload.extraPax} passengers / 15 min)`, 'Load']} />
                      <ReferenceLine x={1.0} stroke="#678A96" strokeDasharray="3 3" label={{ value: 'Normal', fill: '#678A96', fontSize: 9 }} />
                      <ReferenceLine x={1.3} stroke="var(--danger)" strokeDasharray="3 3" label={{ value: 'Risk > 1.3', fill: 'var(--danger)', fontSize: 9 }} />
                      <Bar dataKey="overflowRatio" radius={[0, 4, 4, 0]}>
                        {cascadeResult.stationsList?.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.atRisk ? 'var(--danger)' : 'var(--teal)'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

        </div>
      </div>

      {/* Historical Disruption Records */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <h3 style={{ margin: 0, color: 'var(--text)', fontSize: '0.95rem' }}>Recent Logged Network Closures</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '10px', maxHeight: '240px', overflowY: 'auto' }}>
          {closures.map((c, i) => (
            <div
              key={c.id || i}
              style={{
                background: 'rgba(6, 22, 29, 0.65)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 12px',
                fontSize: '0.75rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '4px'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{
                  background: `${LINE_COLORS[c.line] || 'var(--teal)'}25`,
                  color: LINE_COLORS[c.line] || 'var(--teal)',
                  padding: '2px 8px', borderRadius: 'var(--radius-pill)', fontWeight: 700, fontSize: '0.7rem'
                }}>
                  {c.line || 'U-Bahn'}
                </span>
                <span style={{ color: 'var(--text-muted)', fontFamily: 'JetBrains Mono', fontSize: '0.68rem' }}>
                  {c.duration_hours}h duration
                </span>
              </div>
              <div style={{ color: 'var(--text)', fontWeight: 500, lineHeight: 1.3, marginTop: '2px' }}>
                {c.description}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', marginTop: 'auto' }}>
                {c.when?.replace('T', ' ')}
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
