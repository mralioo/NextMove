"""
Historical Outcome Analyzer.
For each matched historical timestamp, extracts what actually happened
to passenger flows in the following hours vs 7-day-prior baseline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

def extract_historical_outcomes(matched_rows, flows, window_hours=2) -> list:
    if matched_rows.empty or flows.empty:
        return []
        
    if 'timestamp' in flows.columns:
        flows = flows.set_index('timestamp')
        
    outcomes = []
    
    for idx, row in matched_rows.iterrows():
        ts = pd.to_datetime(row.get('timestamp'))
        if pd.isnull(ts):
            continue
            
        window_end = ts + pd.Timedelta(hours=window_hours)
        prior_ts = ts - pd.Timedelta(days=7)
        prior_end = prior_ts + pd.Timedelta(hours=window_hours)
        
        current_window = flows.loc[ts:window_end]
        prior_window = flows.loc[prior_ts:prior_end]
        
        if current_window.empty or prior_window.empty:
            continue
            
        current_mean = current_window.mean()
        prior_mean = prior_window.mean()
        
        delta_pct = ((current_mean - prior_mean) / (prior_mean.abs() + 1)) * 100.0
        delta_pct = delta_pct.dropna()
        
        if delta_pct.empty:
            continue
            
        sorted_delta = delta_pct.sort_values(ascending=False)
        top_surge = sorted_delta.head(5).to_dict()
        top_surge = {str(k): float(v) for k, v in top_surge.items()}
        
        peak_station = str(sorted_delta.index[0]) if len(sorted_delta) > 0 else None
        peak_pct = float(sorted_delta.iloc[0]) if len(sorted_delta) > 0 else 0.0
        net_pct = float(delta_pct.mean()) if not delta_pct.empty else 0.0
        
        outcomes.append({
            'timestamp': str(ts),
            'top_surge_stations': top_surge,
            'peak_surge_station': peak_station,
            'peak_surge_pct': peak_pct,
            'network_delta_pct': net_pct
        })
        
    return outcomes
