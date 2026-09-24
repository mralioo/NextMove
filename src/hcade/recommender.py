"""
HCADE Evidence-Grounded Recommender.
Evaluates candidate interventions and outputs ONE best-supported action.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

INTERVENTIONS = [
    {
        'id': 'bus_bridge',
        'name': 'Deploy bus bridge on suspended section',
        'description': 'Provide bus replacement service covering the closed segment. Absorbs ~30% of stranded passengers based on historical data.',
        'overflow_reduction': 0.30,
        'applicable_types': ['line_suspension'],
    },
    {
        'id': 'increased_frequency',
        'name': 'Increase train frequency on parallel alternative lines',
        'description': 'Boost service frequency by ~25% on lines U1, U2, U3, U9 connecting to affected area.',
        'overflow_reduction': 0.20,
        'applicable_types': ['line_suspension', 'station_closure', 'any'],
    },
    {
        'id': 'staff_redeployment',
        'name': 'Deploy crowd management staff to overflow stations',
        'description': 'Redeploy platform staff to top-3 stations at risk of overcrowding.',
        'overflow_reduction': 0.0,
        'applicable_types': ['any'],
    },
    {
        'id': 'passenger_information',
        'name': 'Activate real-time passenger information campaign',
        'description': 'Push alternative route guidance via BVG app and platform displays to distribute load.',
        'overflow_reduction': 0.15,
        'applicable_types': ['any'],
    },
]

def recommend_action(fingerprint, cascade_results, historical_outcomes) -> dict:
    at_risk = [cr for cr in cascade_results if cr.get('at_risk')]
    if at_risk:
        max_base_overflow = max([cr.get('overflow_ratio', 0.0) for cr in at_risk])
    else:
        max_base_overflow = 0.0
        
    closure_type = fingerprint.get('closure_type', 'any')
    
    avg_hist_surge = 0.0
    if historical_outcomes:
        avg_hist_surge = sum(out.get('network_delta_pct', 0.0) for out in historical_outcomes) / len(historical_outcomes)
        
    best_score = -float('inf')
    best_intervention = None
    
    for inv in INTERVENTIONS:
        if closure_type not in inv['applicable_types'] and 'any' not in inv['applicable_types']:
            continue
            
        remaining_overflow = max_base_overflow * (1.0 - inv['overflow_reduction'])
        score = -remaining_overflow
        
        if avg_hist_surge > 40.0 and inv['id'] == 'bus_bridge':
            score += 0.10
            
        if not cascade_results and inv['id'] == 'passenger_information':
            score += 1.0
            
        if score > best_score:
            best_score = score
            best_intervention = inv
            
    if best_intervention is None:
        best_intervention = INTERVENTIONS[-1]
        
    evidence_from_history = []
    for out in historical_outcomes[:3]:
        evidence_from_history.append(f"On {out.get('timestamp')}, peak surge was {out.get('peak_surge_pct', 0.0):.1f}% at {out.get('peak_surge_station', 'Unknown')}.")
        
    matches = len(historical_outcomes)
    if matches >= 3:
        confidence = 'HIGH'
    elif matches >= 1:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
        
    return {
        'recommended_action': best_intervention['name'],
        'action_description': best_intervention['description'],
        'stations_at_risk': [cr.get('station') for cr in at_risk],
        'max_predicted_overflow': float(max_base_overflow),
        'evidence_from_history': evidence_from_history,
        'confidence': confidence,
        'num_historical_matches': matches,
        'data_source': 'HCADE Historical Similarity'
    }
