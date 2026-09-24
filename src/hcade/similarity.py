"""
HCADE Similarity Search.
Weighted composite scoring to find top-K similar historical situations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

SIMILARITY_WEIGHTS = {
    'hour_bucket': 0.30,
    'is_weekend': 0.20,
    'disruption_line': 0.25,
    'weather_bucket': 0.15,
    'has_event': 0.10,
}

def find_similar_situations(fingerprint, historical_index, top_k=5) -> pd.DataFrame:
    if historical_index.empty:
        return pd.DataFrame()
        
    df = historical_index.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['similarity_score'] = 0.0
    
    for feature, weight in SIMILARITY_WEIGHTS.items():
        if feature == 'disruption_line':
            query_disruption = fingerprint.get('has_disruption', False)
            query_line = fingerprint.get('disruption_line')
            
            if query_disruption:
                match = (df['disruption_line'] == query_line) & (df['has_disruption'] == True)
            else:
                match = df['has_disruption'] == False
            df['similarity_score'] += match.astype(float) * weight
        else:
            val = fingerprint.get(feature)
            if val is not None:
                match = df[feature] == val
                df['similarity_score'] += match.astype(float) * weight
                
    df = df.sort_values(by='similarity_score', ascending=False).head(top_k)
    return df
