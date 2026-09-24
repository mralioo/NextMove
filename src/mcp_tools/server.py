"""
FastMCP server exposing all 14 analytical tools for the U-Bahn agent.
All tools import from DataStore which must be initialized before server starts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
try:
    from fastmcp import FastMCP
except ImportError:
    print("Error: fastmcp is not installed. Please install it using 'pip install fastmcp'.")
    sys.exit(1)

from data_pipeline.loader import DataStore
from data_pipeline.graph_builder import get_k_shortest_paths
from data_pipeline.flow_baseline import detect_station_anomalies
from data_pipeline.event_mapper import get_events_near_station
from analytics.cascade_sim import simulate_cascade
from analytics.stress_score import compute_stress_score
from analytics.behavior_model import analyze_route_preference
from analytics.energy import compute_energy_efficiency
from analytics.surge_forecast import forecast_event_surge
from hcade.fingerprint import build_fingerprint
from hcade.similarity import find_similar_situations
from hcade.outcome_analyzer import extract_historical_outcomes
from hcade.recommender import recommend_action

mcp = FastMCP('ubahn-intelligence', instructions='Berlin U-Bahn operator intelligence agent for InnoTrans 2026.')

@mcp.tool()
def get_station_flow(station_name: str, start_dt: str, end_dt: str) -> dict:
    """Get passenger flow for a station within a time window."""
    try:
        if station_name not in DataStore.flows.columns:
            return {"error": f"Station {station_name} not found in flows data."}
        
        mask = (DataStore.flows['timestamp'] >= start_dt) & (DataStore.flows['timestamp'] <= end_dt)
        filtered = DataStore.flows[mask]
        
        if filtered.empty:
            return {"error": "No flow data found for the given time window."}
            
        values = filtered[station_name].tolist()
        timestamps = filtered['timestamp'].astype(str).tolist()
        mean_val = float(np.mean(values))
        peak_val = float(np.max(values))
        peak_idx = int(np.argmax(values))
        peak_time = timestamps[peak_idx]
        
        return {
            "station": station_name,
            "timestamps": timestamps,
            "values": values,
            "mean": mean_val,
            "peak": peak_val,
            "peak_time": peak_time,
            "source": "flows.csv"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_flow_baseline(station_name: str, hour: int, is_weekend: bool) -> dict:
    """Get baseline flow statistics for a station at a specific hour and day type."""
    try:
        # Assuming DataStore has a baselines property or we compute it
        # This is a stub for the logic assuming precomputed baseline or simple calculation
        df = DataStore.flows.copy()
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['is_weekend'] = pd.to_datetime(df['timestamp']).dt.dayofweek >= 5
        
        mask = (df['hour'] == hour) & (df['is_weekend'] == is_weekend)
        if station_name in df.columns:
            vals = df.loc[mask, station_name].dropna()
            if not vals.empty:
                return {
                    "station": station_name,
                    "hour": hour,
                    "is_weekend": is_weekend,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "p75": float(vals.quantile(0.75)),
                    "p95": float(vals.quantile(0.95)),
                    "source": "flows.csv (computed baseline)"
                }
        return {"error": f"No baseline data for {station_name} at hour {hour} (weekend={is_weekend})"}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def detect_flow_anomaly(station_name: str, start_dt: str, end_dt: str, z_threshold: float = 2.0) -> dict:
    """Detect passenger flow anomalies for a station."""
    try:
        anomalies = detect_station_anomalies(station_name, start_dt, end_dt, z_threshold)
        # anomalies might be a df, convert to dict
        if isinstance(anomalies, pd.DataFrame):
            anomalies = anomalies.to_dict(orient='records')
        return {
            "anomalies": anomalies,
            "count": len(anomalies) if anomalies else 0,
            "source": "flows.csv / anomaly_detection"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_event_context(station_name: str, date: str, radius_km: float = 1.5) -> dict:
    """Find events near a station on a given date."""
    try:
        events = get_events_near_station(station_name, date, radius_km)
        if isinstance(events, pd.DataFrame):
            events = events.to_dict(orient='records')
        return {
            "events": events,
            "count": len(events) if events else 0,
            "source": "events.csv"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_disruption_info(start_dt: str, end_dt: str, line: str = None, station: str = None) -> dict:
    """Get disruption and closure information for the network."""
    try:
        closures = DataStore.closures
        mask = (closures['start_time'] <= end_dt) & (closures['end_time'] >= start_dt)
        filtered = closures[mask]
        
        if line:
            filtered = filtered[filtered['line'] == line]
        if station:
            filtered = filtered[(filtered['start_station'] == station) | (filtered['end_station'] == station)]
            
        disruptions = filtered.to_dict(orient='records')
        return {
            "disruptions": disruptions,
            "count": len(disruptions),
            "source": "closures.csv"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_weather_at_time(dt: str) -> dict:
    """Get weather conditions at a specific time."""
    try:
        weather = DataStore.weather
        weather_dt = pd.to_datetime(weather['timestamp'])
        target_dt = pd.to_datetime(dt)
        
        # Asof lookup
        weather = weather.sort_values('timestamp')
        idx = np.abs(weather_dt - target_dt).argmin()
        row = weather.iloc[idx]
        
        return {
            "temp_c": float(row.get('temp_c', 0)),
            "precipitation": float(row.get('precipitation', 0)),
            "humidity": float(row.get('humidity', 0)),
            "wind_speed_kph": float(row.get('wind_speed', 0)),
            "coco": float(row.get('coco', 0)) if 'coco' in row else None,
            "condition_bucket": row.get('condition_bucket', 'unknown'),
            "source": "weather.csv"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def compute_rerouting(from_station: str, to_station: str, closed_stations: list = None, k: int = 3) -> dict:
    """Compute K shortest paths between two stations accounting for closures."""
    try:
        routes = get_k_shortest_paths(from_station, to_station, closed_stations, k)
        return {
            "routes": routes,
            "source": "network_graph"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def simulate_crowd_cascade(closed_stations: list, disruption_timestamp: str, closed_segment_from: str = None, closed_segment_to: str = None) -> dict:
    """Simulate passenger flow cascading effects during a closure."""
    try:
        all_stations, at_risk_stations, max_overflow_ratio = simulate_cascade(
            closed_stations, disruption_timestamp, closed_segment_from, closed_segment_to
        )
        return {
            "all_stations": all_stations,
            "at_risk_stations": at_risk_stations,
            "max_overflow_ratio": max_overflow_ratio,
            "source": "cascade_simulation"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_network_centrality(top_n: int = 10) -> dict:
    """Get the most central stations in the network."""
    try:
        # Assumes DataStore.centrality is available
        ranking = DataStore.centrality[:top_n]
        return {
            "ranking": ranking,
            "source": "network_centrality"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def compute_stress_scores(stations: list, timestamp: str, active_event: dict = None, active_closure: dict = None) -> dict:
    """Compute stress scores for multiple stations at a specific time."""
    try:
        scores = []
        for st in stations:
            sc = compute_stress_score(st, timestamp, active_event, active_closure)
            scores.append(sc)
        
        scores.sort(key=lambda x: x.get('stress_score', 0), reverse=True)
        return {
            "scores": scores,
            "source": "stress_scoring"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def find_historical_analogues(situation: dict, top_k: int = 5) -> dict:
    """Find historically similar situations based on a situation fingerprint."""
    try:
        fingerprint = build_fingerprint(situation)
        matches = find_similar_situations(fingerprint, top_k)
        outcomes = extract_historical_outcomes(matches)
        
        return {
            "fingerprint": fingerprint,
            "matches": matches,
            "outcomes": outcomes,
            "source": "historical_analogues"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def analyze_route_preferences(disrupted_line: str, disruption_start: str, disruption_end: str) -> dict:
    """Analyze actual passenger route preferences during a disruption."""
    try:
        res = analyze_route_preference(disrupted_line, disruption_start, disruption_end)
        res['source'] = 'route_preference_analysis'
        return res
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_energy_efficiency(line: str = None, start_date: str = None, end_date: str = None) -> dict:
    """Compute energy efficiency (energy per passenger) for lines."""
    try:
        res = compute_energy_efficiency(line, start_date, end_date)
        res['source'] = 'energy.csv and flows.csv'
        return res
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def forecast_surge(station_name: str, event_dt: str, event_attendance: int, event_type: str = 'concert') -> dict:
    """Forecast passenger flow surges due to planned events."""
    try:
        forecast = forecast_event_surge(station_name, event_dt, event_attendance, event_type)
        forecast['source'] = 'surge_forecast_model'
        return forecast
    except Exception as e:
        return {"error": str(e)}
