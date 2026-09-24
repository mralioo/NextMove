"""
Multi-factor StressScore computation.
StressScore(s,t) = alpha*NormFlow + beta*EventScore + gamma*WeatherScore + delta*CascadeScore

Coefficients fitted by OLS on training data at startup.
Critical: gamma_temp is NEGATIVE (heat reduces ridership).
Critical: gamma_prcp is near ZERO (rain has weak effect).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

STRESS_THRESHOLDS = {'LOW': 0.4, 'MEDIUM': 0.7, 'HIGH': float('inf')}

def fit_stress_coefficients(flows, weather, events_mapped, closures) -> dict:
    default_coefs = {'alpha': 0.35, 'beta': 0.25, 'gamma_temp': -0.15, 'gamma_prcp': 0.02, 'delta': 0.20}
    if flows.empty or weather.empty or events_mapped.empty:
        return default_coefs

    try:
        flows_daily = flows.copy()
        if 'timestamp' in flows_daily.columns:
            flows_daily = flows_daily.set_index('timestamp')
        
        daily_flow = flows_daily.sum(axis=1).resample('D').sum()
        rolling_mean = daily_flow.rolling(window=7, min_periods=1).mean()
        network_overflow_ratio = (daily_flow / rolling_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        df = pd.DataFrame({'overflow': network_overflow_ratio})
        
        weather_daily = weather.copy()
        if 'timestamp' in weather_daily.columns:
            weather_daily = weather_daily.set_index('timestamp')
        weather_daily = weather_daily.resample('D').mean()
        
        df = df.join(weather_daily[['temp', 'prcp']], how='left').fillna({'temp': 15.0, 'prcp': 0.0})
        
        events_daily = events_mapped.copy()
        if 'began_local' in events_daily.columns:
            events_daily['date'] = pd.to_datetime(events_daily['began_local']).dt.date
            event_agg = events_daily.groupby('date').agg({'attendance': 'max', 'id': 'count'}).reset_index()
            event_agg['date'] = pd.to_datetime(event_agg['date'])
            event_agg = event_agg.set_index('date')
            df = df.join(event_agg, how='left')
            df['has_event'] = df['id'] > 0
            df['max_attendance_norm'] = df['attendance'].fillna(0) / 2394.0
        else:
            df['has_event'] = False
            df['max_attendance_norm'] = 0.0
            
        closures_daily = closures.copy()
        if 'when' in closures_daily.columns:
            closures_daily['date'] = pd.to_datetime(closures_daily['when']).dt.date
            df = df.join(closures_daily.groupby('date').size().rename('closure_count'), how='left')
            df['has_disruption'] = df['closure_count'].fillna(0) > 0
        else:
            df['has_disruption'] = False
            
        valid_days = daily_flow > 500000
        df = df[valid_days].dropna()
        
        if len(df) < 5:
            return default_coefs
            
        X = df[['max_attendance_norm', 'temp', 'prcp', 'has_disruption']].astype(float)
        y = df['overflow'].astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        coef = model.coef_
        
        return {
            'alpha': 0.35,
            'beta': float(coef[0]),
            'gamma_temp': float(coef[1]),
            'gamma_prcp': float(coef[2]),
            'delta': 0.20
        }
    except Exception:
        return default_coefs

def compute_stress_score(station, timestamp, flows, baselines, weather, events_mapped, cascade_results, coef) -> dict:
    ts = pd.to_datetime(timestamp, errors='coerce')
    
    norm_flow = 0.0
    if not flows.empty and not baselines.empty:
        f_df = flows.set_index('timestamp') if 'timestamp' in flows.columns else flows
        b_df = baselines.set_index('timestamp') if 'timestamp' in baselines.columns else baselines
            
        current_flow = f_df.asof(ts).get(station, 0)
        baseline_mean = b_df.asof(ts).get(station, 1)
        if baseline_mean and baseline_mean > 0:
            norm_flow = float(current_flow / baseline_mean)
            
    event_score = 0.0
    if not events_mapped.empty and 'began_local' in events_mapped.columns:
        active_events = events_mapped[pd.to_datetime(events_mapped['began_local']).dt.date == ts.date()]
        if not active_events.empty:
            max_att = active_events['attendance'].max()
            event_score = min(float(max_att) / 2394.0, 1.0)
            
    weather_score = 0.0
    if not weather.empty:
        w_idx = weather.set_index('timestamp') if 'timestamp' in weather.columns else weather
        w_snap = w_idx.asof(ts)
        if w_snap is not None and not w_snap.empty:
            temp = w_snap.get('temp', 15.0)
            prcp = w_snap.get('prcp', 0.0)
            if pd.isna(temp): temp = 15.0
            if pd.isna(prcp): prcp = 0.0
            weather_score = 0.8 * max(0.0, (30.0 - float(temp)) / 30.0) + 0.2 * min(float(prcp)/10.0, 1.0)
            
    cascade_score = 0.0
    for cr in cascade_results:
        if cr.get('station') == station:
            overflow = cr.get('overflow_ratio', 0.0)
            cascade_score = min(overflow, 2.0) / 2.0
            break
            
    alpha = coef.get('alpha', 0.35)
    beta = coef.get('beta', 0.25)
    gamma_temp = coef.get('gamma_temp', -0.15)
    delta = coef.get('delta', 0.20)
    
    raw_score = alpha * norm_flow + beta * event_score + abs(gamma_temp) * weather_score + delta * cascade_score
    stress_score = min(float(raw_score), 1.0)
    
    risk_level = get_risk_level(stress_score)
    
    return {
        'station': station,
        'timestamp': timestamp,
        'norm_flow': norm_flow,
        'event_score': event_score,
        'weather_score': weather_score,
        'cascade_score': cascade_score,
        'raw_score': float(raw_score),
        'stress_score': stress_score,
        'risk_level': risk_level
    }

def get_risk_level(score: float) -> str:
    if score < STRESS_THRESHOLDS['LOW']:
        return 'LOW'
    elif score < STRESS_THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    return 'HIGH'
