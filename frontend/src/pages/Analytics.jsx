import React, { useEffect, useState } from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, BarChart, Bar, LineChart, Line, Cell 
} from 'recharts';
import { api } from '../api/client';
import { TrendingUp, Award, Calendar, Users } from 'lucide-react';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: 'rgba(5, 20, 26, 0.95)',
        border: '1px solid var(--border)',
        padding: '10px 14px',
        borderRadius: 'var(--radius-md)',
        color: '#E0EEF4',
        boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
        backdropFilter: 'blur(10px)',
        fontSize: '0.8rem'
      }}>
        <p style={{ margin: '0 0 6px 0', fontWeight: 'bold', color: 'var(--teal)' }}>{label}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ margin: '2px 0', color: entry.color }}>
            {entry.name}: <strong>{Math.round(entry.value || 0).toLocaleString()}</strong>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function Analytics() {
  const [dailyFlow, setDailyFlow] = useState([]);
  const [topStations, setTopStations] = useState([]);
  const [hourlyPattern, setHourlyPattern] = useState([]);
  const [status, setStatus] = useState(null);
  const [unit, setUnit] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.flowsDaily().catch(() => ({ dates: [], values: [], rolling_mean: [] })),
      api.centrality().catch(() => []),
      api.hourlyProfile().catch(() => []),
      api.status().catch(() => null)
    ]).then(([flowData, cents, hourly, st]) => {
      setHourlyPattern(Array.isArray(hourly) ? hourly : []);
      setStatus(st);
      setUnit(flowData.unit || '');
      if (flowData.dates) {
        const formatted = flowData.dates.map((d, i) => ({
          date: d,
          flow: flowData.values[i] || 0,
          rollingMean: Math.round(flowData.rolling_mean[i] || 0)
        }));
        setDailyFlow(formatted);
      }

      if (Array.isArray(cents) && cents.length) {
        setTopStations(cents.map(c => ({
          name: c.short_name || c.station,
          value: Math.round(c.daily_flow || 0),
          betweenness: c.betweenness
        })));
      }
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const totalPax = dailyFlow.reduce((acc, curr) => acc + (curr.flow || 0), 0);
  const peakDay = dailyFlow.length 
    ? dailyFlow.reduce((max, curr) => curr.flow > max.flow ? curr : max, dailyFlow[0])
    : { date: 'N/A', flow: 0 };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '10px', color: 'var(--teal)' }}>
        <div style={{ width: '22px', height: '22px', border: '2px solid var(--teal)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
        <span>Loading passenger analytics...</span>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto' }}>
      
      {/* 4 Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
        <div className="card metric-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="label">Total Network Passengers</span>
            <Users size={16} color="var(--teal)" />
          </div>
          <span className="value" style={{ color: 'var(--teal)' }}>
            {(totalPax / 1_000_000).toFixed(1)}M
          </span>
        </div>

        <div className="card metric-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="label">Peak Station</span>
            <Award size={16} color="var(--gold)" />
          </div>
          <span className="value" style={{ fontSize: '1.25rem' }}>
            {topStations[0]?.name || '—'}
          </span>
        </div>

        <div className="card metric-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="label">Peak Flow Date</span>
            <TrendingUp size={16} color="var(--teal)" />
          </div>
          <span className="value" style={{ fontSize: '1.25rem' }}>
            {peakDay.date}
          </span>
        </div>

        <div className="card metric-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="label">Observed Range</span>
            <Calendar size={16} color="var(--text-muted)" />
          </div>
          <span className="value" style={{ fontSize: '1.15rem' }}>
            {status?.data_window ? `${status.data_window.start.slice(5, 10)} → ${status.data_window.end.slice(5, 10)}` : '—'}
          </span>
        </div>
      </div>

      {/* Daily Flow Area Chart */}
      <div className="card" style={{ height: '320px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
          <div>
            <h3 style={{ margin: 0, color: 'var(--text)', fontSize: '0.95rem' }}>Network-Wide Daily Flow</h3>
            <p style={{ margin: 0, fontSize: '0.72rem', color: 'var(--text-muted)' }}>Daily volume (teal) vs 7-day rolling mean (dashed gold) — {unit}</p>
          </div>
          <span className="pill-badge active">{(status?.flows_rows ?? 0).toLocaleString()} timestamps</span>
        </div>

        <div style={{ flex: 1, width: '100%', minHeight: 0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={dailyFlow}>
              <defs>
                <linearGradient id="tealGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--teal)" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="var(--teal)" stopOpacity={0.0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,212,0.06)" />
              <XAxis dataKey="date" stroke="#678A96" tick={{ fill: '#678A96', fontSize: 10 }} />
              <YAxis stroke="#678A96" tick={{ fill: '#678A96', fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" name="Daily Flow" dataKey="flow" stroke="var(--teal)" strokeWidth={2} fillOpacity={1} fill="url(#tealGrad)" />
              <Area type="monotone" name="7-Day Rolling" dataKey="rollingMean" stroke="var(--gold)" strokeDasharray="4 4" strokeWidth={1.5} fill="none" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Two Column Section */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        
        {/* Top Stations Bar Chart */}
        <div className="card" style={{ height: '360px', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ margin: '0 0 12px 0', color: 'var(--text)', fontSize: '0.95rem' }}>Top 15 Stations by Daily Demand (mean passengers / day)</h3>
          <div style={{ flex: 1, width: '100%', minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topStations} layout="vertical" margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,212,0.06)" />
                <XAxis type="number" stroke="#678A96" tick={{ fill: '#678A96', fontSize: 10 }} />
                <YAxis dataKey="name" type="category" stroke="#678A96" tick={{ fill: '#E0EEF4', fontSize: 9 }} width={110} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" name="Daily Passengers" radius={[0, 4, 4, 0]}>
                  {topStations.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={`rgba(0, 229, 212, ${Math.max(0.25, 1 - index * 0.05)})`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Weekday vs Weekend Profile */}
        <div className="card" style={{ height: '360px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ margin: 0, color: 'var(--text)', fontSize: '0.95rem' }}>Commute Curves (weekday vs weekend, whole network per hour)</h3>
            <div style={{ display: 'flex', gap: '12px', fontSize: '0.72rem' }}>
              <span style={{ color: 'var(--teal)' }}>● Weekday Peak</span>
              <span style={{ color: 'var(--gold)' }}>● Weekend</span>
            </div>
          </div>
          <div style={{ flex: 1, width: '100%', minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={hourlyPattern}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,212,0.06)" />
                <XAxis dataKey="hour" stroke="#678A96" tick={{ fill: '#678A96', fontSize: 10 }} />
                <YAxis stroke="#678A96" tick={{ fill: '#678A96', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" name="Weekday (Mon-Fri)" dataKey="weekday" stroke="var(--teal)" strokeWidth={2.5} dot={{ r: 3 }} />
                <Line type="monotone" name="Weekend (Sat-Sun)" dataKey="weekend" stroke="var(--gold)" strokeDasharray="5 5" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

    </div>
  );
}
