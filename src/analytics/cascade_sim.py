"""
Graph-based passenger cascade simulation.
When stations or line segments are closed, redistributes stranded passenger flow
to nearest open stations proportional to inverse hop distance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import pandas as pd
import numpy as np

def simulate_cascade(
    G: nx.Graph,
    closed_stations: list,
    closed_segment: tuple,  # (from_station, to_station) or None
    flows: pd.DataFrame,
    timestamp: str,
    redistribution_factor: float = 0.85,
    stations_df: pd.DataFrame = None,
    line: str = None,
) -> list:
    """
    Simulate passenger redistribution after closure.
    """
    ts = pd.to_datetime(timestamp, errors='coerce')
    if pd.isnull(ts) or flows.empty:
        return []
    
    if 'timestamp' in flows.columns:
        flows = flows.set_index('timestamp')
        
    snap = flows.asof(ts)
    if snap is None or snap.empty:
        return []
        
    all_closed = set(closed_stations)
    
    if closed_segment and stations_df is not None and line is not None:
        from_st, to_st = closed_segment
        line_nodes = [n for n, d in G.nodes(data=True) if line in d.get('u_bahn_lines', [])]
        subG = G.subgraph(line_nodes)
        if from_st in subG and to_st in subG:
            try:
                path = nx.shortest_path(subG, source=from_st, target=to_st)
                all_closed.update(path)
            except nx.NetworkXNoPath:
                pass
                
    G_open = G.copy()
    G_open.remove_nodes_from([n for n in all_closed if n in G_open])
    
    stranded_passengers = {}
    for st in all_closed:
        if st in snap and not pd.isna(snap[st]):
            stranded_passengers[st] = float(snap[st]) * redistribution_factor
            
    extra_passengers = {node: 0.0 for node in G_open.nodes()}
    
    for st, stranded in stranded_passengers.items():
        if stranded <= 0:
            continue
        
        open_neighbors = []
        if st in G:
            neighbors = list(G.neighbors(st))
            open_neighbors = [n for n in neighbors if n in G_open]
            
        if not open_neighbors:
            if st in G:
                edges = nx.bfs_edges(G, source=st, depth_limit=2)
                for u, v in edges:
                    if v in G_open:
                        open_neighbors.append(v)
            open_neighbors = list(set(open_neighbors))
            
        if not open_neighbors:
            continue
            
        dist_weights = {}
        for n in open_neighbors:
            try:
                length = nx.shortest_path_length(G, source=st, target=n)
                dist_weights[n] = 1.0 / length if length > 0 else 1.0
            except nx.NetworkXNoPath:
                dist_weights[n] = 0.0
                
        total_weight = sum(dist_weights.values())
        if total_weight > 0:
            for n, w in dist_weights.items():
                extra_passengers[n] += stranded * (w / total_weight)
                
    result = []
    for st in snap.index:
        if pd.isna(snap[st]) or not isinstance(st, str):
            continue
            
        is_closed = st in all_closed
        baseline_flow = float(snap[st])
        extra = extra_passengers.get(st, 0.0) if not is_closed else 0.0
        total_expected = baseline_flow + extra
        
        overflow_ratio = float(total_expected / baseline_flow) if baseline_flow > 0 and not is_closed else 0.0
        at_risk = overflow_ratio > 1.3
        
        result.append({
            'station': st,
            'baseline_flow': baseline_flow,
            'extra_passengers': extra,
            'total_expected': total_expected,
            'overflow_ratio': overflow_ratio,
            'at_risk': at_risk,
            'is_closed': is_closed
        })
        
    result.sort(key=lambda x: (not x['is_closed'], -x['overflow_ratio']))
    return result
