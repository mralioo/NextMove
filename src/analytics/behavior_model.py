"""
Behavioral Route Preference Model.
Compares ACTUAL post-disruption flow increases at alternative stations
vs THEORETICAL shortest-path redistribution volumes.
Implements revealed preference theory from behavioral economics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import networkx as nx
import random

def analyze_route_preference(disrupted_line, disruption_start, disruption_end, flows, G, stations) -> dict:
    if flows.empty or G is None or len(G.nodes) == 0:
        return {'error': 'Empty data or graph'}
        
    start_ts = pd.to_datetime(disruption_start, errors='coerce')
    end_ts = pd.to_datetime(disruption_end, errors='coerce')
    
    if pd.isnull(start_ts) or pd.isnull(end_ts):
        return {'error': 'Invalid timestamps'}
        
    if 'timestamp' in flows.columns:
        flows = flows.set_index('timestamp')
        
    window = flows.loc[start_ts:end_ts]
    prior_start = start_ts - pd.Timedelta(days=7)
    prior_end = end_ts - pd.Timedelta(days=7)
    prior_window = flows.loc[prior_start:prior_end]
    
    if window.empty or prior_window.empty:
        return {'error': 'Disruption window not in data'}
        
    line_stations = []
    if not stations.empty and 'u_bahn_lines' in stations.columns:
        for idx, row in stations.iterrows():
            lines = row['u_bahn_lines']
            if isinstance(lines, str) and disrupted_line in lines:
                line_stations.append(row['name'])
            elif isinstance(lines, list) and disrupted_line in lines:
                line_stations.append(row['name'])
                
    mean_window = window.mean()
    mean_prior = prior_window.mean()
    actual_delta = mean_window - mean_prior
    
    for ls in line_stations:
        if ls in actual_delta:
            actual_delta = actual_delta.drop(ls)
            
    actual_delta = actual_delta.dropna()
    max_abs_delta = float(actual_delta.abs().max()) if len(actual_delta) > 0 else 0.0
    actual_norm = actual_delta / max_abs_delta if max_abs_delta > 0 else actual_delta
    
    G_open = G.copy()
    G_open.remove_nodes_from([n for n in line_stations if n in G_open])
    
    if len(G_open.nodes) == 0:
        return {'error': 'Empty disrupted graph'}
        
    closed_avail = [n for n in line_stations if n in G]
    closed_sample = random.sample(closed_avail, min(20, len(closed_avail)))
    open_avail = list(G_open.nodes)
    open_sample = random.sample(open_avail, min(10, len(open_avail)))
    
    path_counts = {n: 0 for n in G_open.nodes}
    
    for c in closed_sample:
        for o in open_sample:
            try:
                paths = list(nx.all_shortest_paths(G, source=c, target=o))
                for path in paths:
                    for node in path:
                        if node in G_open:
                            path_counts[node] += 1
            except nx.NetworkXNoPath:
                continue
                
    max_count = max(path_counts.values()) if path_counts else 0
    theoretical_norm = {k: v / max_count if max_count > 0 else 0 for k, v in path_counts.items()}
    
    preferred_routes = []
    avoided_routes = []
    
    for st, actual in actual_norm.items():
        if st in theoretical_norm:
            theo = theoretical_norm[st]
            if actual > theo + 0.15:
                preferred_routes.append(str(st))
            elif actual < theo - 0.15:
                avoided_routes.append(str(st))
                
    insights = f"Found {len(preferred_routes)} preferred stations and {len(avoided_routes)} avoided stations. Preferred stations absorb more traffic than shortest-path topology predicts, indicating behavioral preference possibly due to station amenities, transfer convenience, or perceived reliability."
    
    return {
        'preferred_routes': preferred_routes,
        'avoided_routes': avoided_routes,
        'behavioral_insights': insights,
        'data_window': {'start': str(start_ts), 'end': str(end_ts)}
    }
