"""
Situation Fingerprinter: converts parsed query context into a structured
numeric/categorical fingerprint for HCADE similarity search.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

def hour_to_bucket(hour: int) -> str:
    if hour in (6, 7, 8):
        return 'PEAK_AM'
    elif 9 <= hour <= 14:
        return 'MID_DAY'
    elif 15 <= hour <= 19:
        return 'PEAK_PM'
    else:
        return 'OFF_PEAK'

def coco_to_bucket(coco: float, temp: float, prcp: float) -> str:
    if pd.isna(coco): coco = 0
    if pd.isna(temp): temp = 15
    if pd.isna(prcp): prcp = 0
    
    if coco >= 14:
        return 'STORM'
    if temp > 32:
        return 'HEAT'
    if prcp > 5:
        return 'HEAVY_RAIN'
    if prcp > 0:
        return 'RAIN'
    return 'CLEAR'

def build_fingerprint(parsed_context: dict) -> dict:
    dt_str = parsed_context.get('datetime', '2026-07-17 08:00')
    dt = pd.to_datetime(dt_str, errors='coerce')
    if pd.isnull(dt):
        dt = pd.to_datetime('2026-07-17 08:00')
        
    hour_bucket = hour_to_bucket(dt.hour)
    is_weekend = dt.weekday() >= 5
    
    att = parsed_context.get('event_attendance')
    if att is not None and not pd.isna(att):
        has_event = True
        if att < 500:
            att_bucket = 'SMALL'
        elif att <= 1500:
            att_bucket = 'MEDIUM'
        else:
            att_bucket = 'LARGE'
    else:
        has_event = False
        att_bucket = None
        
    weather_coco = parsed_context.get('weather_coco', 0.0)
    weather_temp = parsed_context.get('weather_temp', 15.0)
    weather_prcp = parsed_context.get('weather_prcp', 0.0)
    
    weather_bucket = coco_to_bucket(weather_coco, weather_temp, weather_prcp)
    
    disrupted_line = parsed_context.get('disrupted_line')
    has_disruption = disrupted_line is not None
    
    closure_type = parsed_context.get('closure_type', 'any')
    
    return {
        'datetime': str(dt),
        'hour_bucket': hour_bucket,
        'is_weekend': is_weekend,
        'has_event': has_event,
        'event_attendance_bucket': att_bucket,
        'has_disruption': has_disruption,
        'disruption_line': disrupted_line,
        'weather_bucket': weather_bucket,
        'closure_type': closure_type
    }
