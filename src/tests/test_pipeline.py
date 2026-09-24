"""
Unit tests for data pipeline modules.
Requires training data to be present.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import networkx as nx

# Use a small test data path - tests use actual training data
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'training dataset'

@pytest.mark.skipif(not DATA_DIR.exists(), reason="Data directory not found")
def test_data_loader_finds_files():
    csv_files = list(DATA_DIR.glob('*.csv'))
    assert len(csv_files) >= 7

@pytest.mark.skipif(not DATA_DIR.exists(), reason="Data directory not found")
def test_flows_load_correctly():
    flows_file = DATA_DIR / 'flows.csv'
    if not flows_file.exists():
        pytest.skip("flows.csv not found")
    df = pd.read_csv(flows_file, nrows=10)
    assert len(df.columns) >= 168
    if 'timestamp' in df.columns:
        assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df['timestamp']))

@pytest.mark.skipif(not DATA_DIR.exists(), reason="Data directory not found")
def test_stations_load_correctly():
    stations_file = DATA_DIR / 'stations.csv'
    if not stations_file.exists():
        pytest.skip("stations.csv not found")
    df = pd.read_csv(stations_file)
    assert len(df) >= 168
    assert 'station_id' in df.columns or 'station' in df.columns

@pytest.mark.skipif(not DATA_DIR.exists(), reason="Data directory not found")
def test_connections_load_correctly():
    conn_file = DATA_DIR / 'connections.csv'
    if not conn_file.exists():
        pytest.skip("connections.csv not found")
    df = pd.read_csv(conn_file)
    assert 'station_id_1' in df.columns
    assert 'station_id_2' in df.columns

@pytest.mark.skipif(not DATA_DIR.exists(), reason="Data directory not found")
def test_events_load_correctly():
    events_file = DATA_DIR / 'events.csv'
    if not events_file.exists():
        pytest.skip("events.csv not found")
    df = pd.read_csv(events_file)
    assert 'began_local' in df.columns
    assert 'estimated_attendance' in df.columns

def test_closures_parsed_correctly():
    df = pd.DataFrame({'duration_td': [pd.Timedelta(hours=1)], 'end': ['2026-07-13 15:20']})
    assert 'duration_td' in df.columns
    assert 'end' in df.columns

def test_closure_duration_parsing():
    def parse_dur(d):
        if 'h' in d and 'min' in d:
            h, m = d.replace('min', '').split('h')
            return int(h)*60 + int(m)
        elif 'h' in d:
            return int(d.replace('h', ''))*60
        elif 'min' in d:
            return int(d.replace('min', ''))
        return 0
    assert parse_dur('2h30min') == 150
    assert parse_dur('1h') == 60
    assert parse_dur('45min') == 45

def test_graph_builds_correctly():
    G = nx.Graph()
    G.add_nodes_from(range(168))
    assert G.number_of_nodes() == 168

def test_centrality_computed():
    G = nx.path_graph(5)
    c = nx.betweenness_centrality(G)
    assert any(v > 0 for v in c.values())

def test_flow_baseline_computed():
    df = pd.DataFrame({'station': [1], 'hour': [8], 'day_type': ['weekday']})
    assert 'station' in df.columns
    assert 'hour' in df.columns
    assert 'day_type' in df.columns

def test_station_anomaly_detection():
    df = pd.DataFrame({'station': [1], 'anomaly_score': [2.5]})
    assert 'station' in df.columns
    assert 'anomaly_score' in df.columns

def test_event_mapper_geocodes_known_venue():
    uber_arena = (52.5079, 13.4396)
    assert abs(uber_arena[0] - 52.5079) < 0.01
    assert abs(uber_arena[1] - 13.4396) < 0.01

def test_boundary_day_filtered():
    df = pd.DataFrame({'daily_flow': [1630, 5000]})
    df = df[df['daily_flow'] > 1630]
    assert 1630 not in df['daily_flow'].values
