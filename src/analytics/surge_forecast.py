"""
Surge Forecaster: estimates expected passenger flow increase at a station
for an upcoming event, using HCADE historical similarity matching.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from hcade.similarity import find_similar_situations
from hcade.outcome_analyzer import extract_historical_outcomes
from hcade.fingerprint import hour_to_bucket, coco_to_bucket

def forecast_event_surge(station_name, event_dt, event_attendance, event_type, historical_index, flows, baselines) -> dict:
    dt = pd.to_datetime(event_dt, errors='coerce')
    if pd.isnull(dt):
        return {}
        
    hour_bucket = hour_to_bucket(dt.hour)
    is_weekend = dt.weekday() >= 5
    
    if event_attendance < 500:
        att_bucket = 'SMALL'
    elif event_attendance <= 1500:
        att_bucket = 'MEDIUM'
    else:
        att_bucket = 'LARGE'
        
    fingerprint = {
        'hour_bucket': hour_bucket,
        'is_weekend': is_weekend,
        'has_event': True,
        'event_attendance_bucket': att_bucket,
        'has_disruption': False,
        'weather_bucket': 'CLEAR',
        'disruption_line': None
    }
    
    matched = find_similar_situations(fingerprint, historical_index, top_k=5)
    outcomes = extract_historical_outcomes(matched, flows, window_hours=2)
    
    surges = []
    for out in outcomes:
        st_surge = out.get('top_surge_stations', {}).get(station_name)
        if st_surge is not None:
            surges.append(st_surge)
        else:
            net_surge = out.get('network_delta_pct', 0.0)
            surges.append(net_surge)
            
    if surges:
        expected_delta_pct = float(sum(surges) / len(surges))
    else:
        expected_delta_pct = 0.0
        
    pre_surge_start = dt - pd.Timedelta(minutes=75)
    post_dispersal = dt + pd.Timedelta(hours=2)
    
    actions = []
    if expected_delta_pct > 20:
        actions.append("Deploy extra crowd control staff")
    if expected_delta_pct > 50:
        actions.append("Coordinate with local police for station entry metering")
    if not actions:
        actions.append("Standard monitoring")
        
    return {
        'station_name': station_name,
        'event_dt': str(dt),
        'expected_delta_pct': expected_delta_pct,
        'timing': {
            'pre_surge_start': str(pre_surge_start),
            'post_dispersal': str(post_dispersal)
        },
        'recommended_actions': actions,
        'confidence_level': 'HIGH' if len(surges) >= 3 else ('MEDIUM' if len(surges) > 0 else 'LOW')
    }
