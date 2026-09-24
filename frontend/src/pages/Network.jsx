import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { api } from '../api/client';
import { Layers, Sun, Moon, RefreshCw } from 'lucide-react';

const LINE_COLORS = {
  U1: '#00A389', U2: '#E3342F', U3: '#007A3D',
  U4: '#F6B900', U5: '#7E3F11', U6: '#8B1E8B',
  U7: '#009EE0', U8: '#005CA9', U9: '#F37920'
};

export default function Network() {
  const [stations, setStations] = useState([]);
  const [edges, setEdges] = useState([]);
  const [heatmapData, setHeatmapData] = useState(null);
  const [activeLines, setActiveLines] = useState(Object.keys(LINE_COLORS));
  const [darkTiles, setDarkTiles] = useState(true);
  const [tiles, setTiles] = useState(true);            // background map tiles come from openstreetmap.org (third party); off = only our own data
  const [loading, setLoading] = useState(true);
  const [loadingHeatmap, setLoadingHeatmap] = useState(false);
  const [error, setError] = useState(null);

  // Load map topology once
  const loadMapTopology = () => {
    setLoading(true);
    setError(null);
    Promise.all([
      api.stations().catch(err => { console.error('Stations error', err); return []; }),
      api.edges().catch(err => { console.error('Edges error', err); return []; })
    ]).then(([s, e]) => {
      setStations(Array.isArray(s) ? s : []);
      setEdges(Array.isArray(e) ? e : []);
      setLoading(false);
    }).catch(err => {
      setError(err.message);
      setLoading(false);
    });
  };

  // Load heatmap dynamically whenever active lines change
  const fetchHeatmapForLines = (lines) => {
    if (!lines || lines.length === 0) {
      setHeatmapData(null);
      return;
    }
    setLoadingHeatmap(true);
    api.flowsHeatmap(lines.join(','))
      .then(h => {
        setHeatmapData(h);
        setLoadingHeatmap(false);
      })
      .catch(err => {
        console.error('Heatmap error', err);
        setLoadingHeatmap(false);
      });
  };

  useEffect(() => {
    loadMapTopology();
  }, []);

  useEffect(() => {
    fetchHeatmapForLines(activeLines);
  }, [activeLines]);

  const toggleLine = (line) => {
    setActiveLines(prev => {
      if (prev.includes(line)) {
        const next = prev.filter(l => l !== line);
        return next.length === 0 ? [line] : next;
      } else {
        return [...prev, line];
      }
    });
  };

  const selectSingleLineOnly = (line) => {
    setActiveLines([line]);
  };

  const selectAllLines = () => setActiveLines(Object.keys(LINE_COLORS));
  const clearAllLines = () => setActiveLines(Object.keys(LINE_COLORS).slice(0, 1));

  // Valid edges filtered by active lines and non-null coordinates
  const validEdges = edges.filter(e => 
    e && activeLines.includes(e.line) &&
    e.from_lat != null && e.from_lat !== 0 &&
    e.from_lon != null && e.from_lon !== 0 &&
    e.to_lat != null && e.to_lat !== 0 &&
    e.to_lon != null && e.to_lon !== 0
  );

  // Valid stations filtered by active lines and valid coordinates
  const validStations = stations.filter(s =>
    s && s.lat != null && s.lat !== 0 &&
    s.lon != null && s.lon !== 0 &&
    s.lines?.some(l => activeLines.includes(l))
  );

  const isSingleLine = activeLines.length === 1;
  const singleLineName = activeLines[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%', overflowY: 'auto' }}>
      
      {/* Top Header & Capsule Line Filters */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '14px', padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '32px', height: '32px', borderRadius: '50%',
              background: 'rgba(0, 229, 212, 0.12)', border: '1px solid rgba(0, 229, 212, 0.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              <Layers size={16} color="var(--teal)" />
            </div>
            <div>
              <h2 style={{ fontSize: '1.05rem', margin: 0, color: 'var(--text)' }}>U-Bahn Network Topology</h2>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
                {validStations.length} Stations Active · {validEdges.length} Track Connections
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            {/* Dark / Light OSM Map Toggle */}
            <button
              onClick={() => setDarkTiles(prev => !prev)}
              className="pill-tab"
              style={{ fontSize: '0.75rem', padding: '5px 12px', border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', gap: '6px' }}
              title="Toggle Dark / Normal OpenStreetMap style"
            >
              {darkTiles ? <Moon size={13} color="var(--teal)" /> : <Sun size={13} color="var(--gold)" />}
              <span>{darkTiles ? 'Dark Map' : 'Light Map'}</span>
            </button>

            <button onClick={() => setTiles(prev => !prev)} className="pill-tab" style={{ fontSize: '0.75rem', padding: '5px 12px', border: '1px solid var(--border-subtle)' }} title="Background tiles are loaded from openstreetmap.org (third party). Off = only our own data is drawn.">
              {tiles ? 'OSM tiles: on' : 'OSM tiles: off'}
            </button>

            <button onClick={loadMapTopology} className="pill-tab" style={{ fontSize: '0.75rem', padding: '5px 10px', border: '1px solid var(--border-subtle)' }} title="Reload data">
              <RefreshCw size={13} />
            </button>

            <button onClick={selectAllLines} className="pill-tab" style={{ fontSize: '0.75rem', padding: '5px 12px' }}>
              Select All
            </button>
          </div>
        </div>

        {/* Line Capsule Badges */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginRight: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Lines:</span>
          {Object.entries(LINE_COLORS).map(([line, color]) => {
            const isActive = activeLines.includes(line);
            return (
              <button
                key={line}
                onClick={() => toggleLine(line)}
                onDoubleClick={() => selectSingleLineOnly(line)}
                title={`Click to toggle ${line} (Double-click to select only ${line})`}
                style={{
                  background: isActive ? `${color}25` : 'rgba(255, 255, 255, 0.03)',
                  color: isActive ? color : 'var(--text-muted)',
                  border: `1.5px solid ${isActive ? color : 'rgba(255, 255, 255, 0.08)'}`,
                  padding: '4px 14px',
                  borderRadius: 'var(--radius-pill)',
                  cursor: 'pointer',
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  boxShadow: isActive ? `0 0 10px ${color}35` : 'none',
                  transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)'
                }}
              >
                {line}
              </button>
            );
          })}
        </div>
      </div>

      {/* Map Section */}
      <div className="card" style={{ padding: '0', overflow: 'hidden', height: '480px', minHeight: '480px', position: 'relative' }}>
        {loading && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 1000,
            background: 'rgba(2, 11, 14, 0.85)', backdropFilter: 'blur(8px)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px'
          }}>
            <div style={{ width: '28px', height: '28px', border: '2px solid var(--teal)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Loading network topology & map tiles...</span>
          </div>
        )}

        {error && (
          <div style={{ padding: '24px', color: 'var(--danger)', fontSize: '0.9rem' }}>
            Failed to load network data: {error} — is the operator API running (make up)?
          </div>
        )}

        {/* Leaflet Map */}
        <div style={{
          height: '100%',
          width: '100%',
          filter: darkTiles && tiles ? 'invert(98%) hue-rotate(185deg) brightness(88%) contrast(110%)' : 'none',
          transition: 'filter 0.3s ease'
        }}>
          <MapContainer
            key={`berlin-osm-map-${validStations.length}`}
            center={[52.518, 13.395]}
            zoom={12}
            scrollWheelZoom={true}
            style={{ height: '100%', width: '100%' }}
          >
            {tiles && <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              maxZoom={19}
            />}

            {/* U-Bahn Line Edges (Track Polylines) */}
            {validEdges.map((edge, i) => (
              <Polyline
                key={`edge-${edge.line}-${i}`}
                positions={[
                  [edge.from_lat, edge.from_lon],
                  [edge.to_lat, edge.to_lon]
                ]}
                pathOptions={{
                  color: LINE_COLORS[edge.line] || 'var(--teal)',
                  weight: 5,
                  opacity: 0.95,
                  lineCap: 'round',
                  lineJoin: 'round'
                }}
              />
            ))}

            {/* Station Circle Markers with High Contrast */}
            {validStations.map((station, i) => {
              const primaryLine = station.lines?.[0];
              const color = LINE_COLORS[primaryLine] || 'var(--teal)';
              const radius = Math.max(5, Math.min(12, (station.daily_flow || 0) / 10000 + 4.5));

              return (
                <CircleMarker
                  key={`st-${station.name}-${i}`}
                  center={[station.lat, station.lon]}
                  radius={radius}
                  pathOptions={{
                    color: '#020B0E',
                    fillColor: color,
                    fillOpacity: 1.0,
                    weight: 2
                  }}
                >
                  <Popup>
                    <div style={{ padding: '6px', minWidth: '170px' }}>
                      <h4 style={{ margin: '0 0 6px 0', fontSize: '0.95rem', color: 'var(--teal)', fontWeight: 700 }}>
                        {station.short_name || station.name}
                      </h4>
                      <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginBottom: '8px' }}>
                        {station.lines?.map(l => (
                          <span key={l} style={{
                            background: LINE_COLORS[l] || 'var(--teal)',
                            color: '#000',
                            padding: '1px 6px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: 700
                          }}>
                            {l}
                          </span>
                        ))}
                      </div>
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                        <div>Daily Flow: <strong style={{ color: 'var(--text)' }}>{Math.round(station.daily_flow || 0).toLocaleString()}</strong> pax</div>
                        <div>Betweenness: <strong style={{ color: 'var(--text)' }}>{station.betweenness ? station.betweenness.toFixed(4) : '0.0000'}</strong></div>
                        <div>Degree: <strong style={{ color: 'var(--text)' }}>{station.degree || 0}</strong></div>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>
        </div>
      </div>

      {/* Dynamic Hourly Flow Heatmap Matrix */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--teal)' }}>
              Horizontal Passenger Flow Heatmap
            </h3>
            <p style={{ margin: '2px 0 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {isSingleLine
                ? `Showing ALL stations on Line ${singleLineName} across 24 hours (00:00 - 23:00)`
                : `Showing the 20 busiest stations across the selected lines (00:00 - 23:00) · passengers per hour, mean over all days · gold = peak`}
            </p>
          </div>

          <div>
            {isSingleLine ? (
              <span
                className="pill-badge"
                style={{
                  background: `${LINE_COLORS[singleLineName] || 'var(--teal)'}25`,
                  color: LINE_COLORS[singleLineName] || 'var(--teal)',
                  border: `1px solid ${LINE_COLORS[singleLineName] || 'var(--teal)'}`,
                  fontSize: '0.72rem',
                  fontWeight: 700
                }}
              >
                Line {singleLineName} · All {heatmapData?.count || 0} Stations
              </span>
            ) : (
              <span className="pill-badge teal" style={{ fontSize: '0.72rem' }}>
                Top 20 Stations ({activeLines.length} Lines Selected)
              </span>
            )}
          </div>
        </div>

        <div style={{ overflowX: 'auto', maxHeight: '340px', overflowY: 'auto' }}>
          {loadingHeatmap ? (
            <div style={{ color: 'var(--teal)', textAlign: 'center', padding: '30px', fontSize: '0.85rem' }}>
              Updating heatmap...
            </div>
          ) : (!heatmapData || !heatmapData.stations?.length ? (
            <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '30px', fontSize: '0.85rem' }}>
              No heatmap data available for selected line(s)
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '170px repeat(24, 1fr)', gap: '2px', minWidth: '760px' }}>
              {/* Header hours */}
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700 }}>Station</div>
              {Array.from({ length: 24 }, (_, h) => (
                <div key={h} style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textAlign: 'center', fontWeight: 600 }}>
                  {h}h
                </div>
              ))}

              {/* Station rows */}
              {heatmapData.stations.map((stName, idx) => {
                const hourlyValues = heatmapData.values?.[idx] || [];
                const maxVal = heatmapData.max_value || 1000;

                return (
                  <React.Fragment key={`${stName}-${idx}`}>
                    <div style={{
                      fontSize: '0.72rem', color: 'var(--text)', whiteSpace: 'nowrap',
                      overflow: 'hidden', textOverflow: 'ellipsis', alignSelf: 'center', paddingRight: '8px'
                    }}>
                      {stName}
                    </div>
                    {Array.from({ length: 24 }, (_, h) => {
                      const val = hourlyValues[h] || 0;
                      const ratio = maxVal > 0 ? Math.min(1, val / maxVal) : 0;
                      // Smooth gradient from deep dark teal to electric cyan to golden amber
                      const bg = ratio > 0.65
                        ? `rgba(245, 197, 24, ${0.45 + ratio * 0.55})`
                        : (ratio > 0.2 ? `rgba(0, 229, 212, ${0.15 + ratio * 0.75})` : 'rgba(255, 255, 255, 0.03)');

                      return (
                        <div
                          key={h}
                          title={`${stName} @ ${h}:00 - ${Math.round(val)} pax`}
                          style={{
                            height: '14px',
                            background: bg,
                            borderRadius: '2px',
                            cursor: 'pointer'
                          }}
                        />
                      );
                    })}
                  </React.Fragment>
                );
              })}
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
