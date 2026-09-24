import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

def flow_columns(flows: pd.DataFrame) -> list[str]:
    return [col for col in flows.columns if col != 'timestamp']

def compute_station_baselines(flows: pd.DataFrame) -> pd.DataFrame:
    if flows is None or flows.empty:
        return pd.DataFrame()
        
    flows_work = flows.copy()
    if 'timestamp' in flows_work.columns:
        if not pd.api.types.is_datetime64_any_dtype(flows_work['timestamp']):
            flows_work['timestamp'] = pd.to_datetime(flows_work['timestamp'], errors='coerce')
        flows_work['hour'] = flows_work['timestamp'].dt.hour
        flows_work['day_type'] = np.where(flows_work['timestamp'].dt.dayofweek < 5, 'weekday', 'weekend')
    else:
        return pd.DataFrame()
        
    stations = flow_columns(flows)
    
    # Fast vectorized aggregation without melting 1.4 million rows with Python lambdas
    grouped = flows_work.groupby(['hour', 'day_type'])[stations]
    means = grouped.mean().unstack(level=['hour', 'day_type'])
    stds = grouped.std().unstack(level=['hour', 'day_type'])
    q25 = grouped.quantile(0.25).unstack(level=['hour', 'day_type'])
    q75 = grouped.quantile(0.75).unstack(level=['hour', 'day_type'])
    q95 = grouped.quantile(0.95).unstack(level=['hour', 'day_type'])
    
    records = []
    for (hour, day_type), col_means in grouped.mean().iterrows():
        col_stds = stds[hour][day_type] if (hour in stds and day_type in stds[hour]) else pd.Series(0, index=stations)
        col_q25 = q25[hour][day_type] if (hour in q25 and day_type in q25[hour]) else col_means
        col_q75 = q75[hour][day_type] if (hour in q75 and day_type in q75[hour]) else col_means
        col_q95 = q95[hour][day_type] if (hour in q95 and day_type in q95[hour]) else col_means
        
        for st in stations:
            records.append({
                'station': st,
                'hour': hour,
                'day_type': day_type,
                'mean': float(col_means.get(st, 0.0) or 0.0),
                'std': float(col_stds.get(st, 0.0) or 0.0),
                'p25': float(col_q25.get(st, 0.0) or 0.0),
                'p75': float(col_q75.get(st, 0.0) or 0.0),
                'p95': float(col_q95.get(st, 0.0) or 0.0)
            })
            
    return pd.DataFrame(records)

def compute_rolling_zscore(flows: pd.DataFrame, station: str, window_days: int = 28) -> pd.Series:
    if station not in flows.columns or 'timestamp' not in flows.columns:
        return pd.Series(dtype=float)
        
    ts = flows[['timestamp', station]].copy()
    ts = ts.set_index('timestamp').sort_index()
    
    window_periods = window_days * 96
    
    rolling_mean = ts[station].rolling(window=window_periods, min_periods=96, center=True).mean()
    rolling_std = ts[station].rolling(window=window_periods, min_periods=96, center=True).std()
    
    z_score = (ts[station] - rolling_mean) / (rolling_std + 1e-6)
    return z_score

def detect_station_anomalies(flows: pd.DataFrame, station: str, start_dt, end_dt, z_threshold: float = 2.0) -> pd.DataFrame:
    if station not in flows.columns or 'timestamp' not in flows.columns:
        return pd.DataFrame()
        
    ts = flows[['timestamp', station]].copy().set_index('timestamp').sort_index()
    
    window_periods = 28 * 96
    rolling_mean = ts[station].rolling(window=window_periods, min_periods=96, center=True).mean()
    rolling_std = ts[station].rolling(window=window_periods, min_periods=96, center=True).std()
    
    z_score = (ts[station] - rolling_mean) / (rolling_std + 1e-6)
    
    df = pd.DataFrame({
        'timestamp': ts.index,
        'flow': ts[station],
        'z_score': z_score,
        'baseline_mean': rolling_mean
    })
    
    mask = (df['timestamp'] >= pd.to_datetime(start_dt)) & (df['timestamp'] <= pd.to_datetime(end_dt))
    df = df[mask]
    
    anomalies = df[df['z_score'].abs() >= z_threshold].copy()
    
    return anomalies.reset_index(drop=True)

def get_network_daily_flow(flows: pd.DataFrame) -> pd.Series:
    if flows is None or flows.empty or 'timestamp' not in flows.columns:
        return pd.Series(dtype=float)
        
    flows_work = flows.copy()
    flows_work = flows_work.set_index('timestamp')
    network_flow = flows_work.sum(axis=1)
    return network_flow.resample('D').sum()
