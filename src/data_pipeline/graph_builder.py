import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import pandas as pd

def build_graph(stations: pd.DataFrame, connections: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    if stations is None or connections is None:
        return G
        
    id_to_name = {}
    for _, row in stations.iterrows():
        name = row.get('station_name')
        station_id = row.get('station_id')
        lat = row.get('lat')
        lon = row.get('lon')
        lines = row.get('u_bahn_lines')
        if pd.notna(name):
            id_to_name[station_id] = name
            G.add_node(name, station_id=station_id, lat=lat, lon=lon, lines=lines)
            
    for _, row in connections.iterrows():
        id1 = row.get('station_id_1')
        id2 = row.get('station_id_2')
        if id1 in id_to_name and id2 in id_to_name:
            G.add_edge(id_to_name[id1], id_to_name[id2])
            
    return G

def compute_centrality(G: nx.Graph) -> dict:
    if not G or G.number_of_nodes() == 0:
        return {}
    
    bc = nx.betweenness_centrality(G, normalized=True)
    deg = dict(G.degree())
    cc = nx.closeness_centrality(G)
    
    result = {}
    for node in G.nodes():
        result[node] = {
            'betweenness': bc.get(node, 0.0),
            'degree': deg.get(node, 0),
            'closeness': cc.get(node, 0.0)
        }
    return result

def get_k_shortest_paths(G: nx.Graph, source: str, target: str, k: int = 3, closed_nodes: list = None, closed_edges: list = None) -> list[list[str]]:
    G_work = G.copy()
    if closed_nodes:
        G_work.remove_nodes_from([n for n in closed_nodes if n in G_work])
    if closed_edges:
        G_work.remove_edges_from([e for e in closed_edges if G_work.has_edge(*e)])
        
    if source not in G_work or target not in G_work:
        return []
        
    paths = []
    try:
        path_gen = nx.shortest_simple_paths(G_work, source, target)
        for _ in range(k):
            paths.append(next(path_gen))
    except (nx.NetworkXNoPath, StopIteration):
        pass
    return paths

def stations_on_line_segment(G: nx.Graph, line: str, from_station: str, to_station: str, stations_df: pd.DataFrame) -> list[str]:
    if line is None:
        G_sub = G
    else:
        if stations_df is None or 'u_bahn_lines' not in stations_df.columns:
            return []
        
        valid_stations = []
        for _, row in stations_df.iterrows():
            name = row.get('station_name')
            lines = str(row.get('u_bahn_lines', ''))
            if line in lines:
                valid_stations.append(name)
                
        G_sub = G.subgraph([n for n in valid_stations if n in G])
        
    if from_station not in G_sub or to_station not in G_sub:
        return []
        
    try:
        return nx.shortest_path(G_sub, from_station, to_station)
    except nx.NetworkXNoPath:
        return []
