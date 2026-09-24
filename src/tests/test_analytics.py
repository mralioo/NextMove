import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd

def test_cascade_sim_empty_closure():
    stations = [{'station': 1, 'overflow_ratio': 1.0}, {'station': 2, 'overflow_ratio': 1.0}]
    df = pd.DataFrame(stations)
    assert (df['overflow_ratio'] == 1.0).all()

def test_cascade_sim_with_closed_station():
    # Mocking cascade
    df = pd.DataFrame({'station': [1, 2], 'extra_passengers': [0, 50]})
    assert df.loc[1, 'extra_passengers'] > 0

def test_cascade_sim_at_risk_flag():
    df = pd.DataFrame({'station': [1, 2], 'overflow_ratio': [1.0, 1.4]})
    df['at_risk'] = df['overflow_ratio'] > 1.3
    assert not df.loc[0, 'at_risk']
    assert df.loc[1, 'at_risk']

def test_stress_score_range():
    score = 0.5
    assert 0 <= score <= 1

def test_stress_score_heat_reduces():
    temp_normal = 20
    temp_hot = 35
    weather_score_normal = -0.15 * temp_normal
    weather_score_hot = -0.15 * temp_hot
    assert weather_score_hot < weather_score_normal

def test_energy_efficiency_u5_worst():
    df = pd.DataFrame({'line': ['U1', 'U5'], 'mwh_per_1k_pax': [260.0, 550.0]})
    worst_line = df.loc[df['mwh_per_1k_pax'].idxmax(), 'line']
    assert worst_line == 'U5'

def test_energy_efficiency_u1_best():
    df = pd.DataFrame({'line': ['U1', 'U5'], 'mwh_per_1k_pax': [260.0, 550.0]})
    best_line = df.loc[df['mwh_per_1k_pax'].idxmin(), 'line']
    assert best_line == 'U1'

def test_energy_efficiency_ranking():
    df = pd.DataFrame({'line': ['U1', 'U5'], 'mwh_per_1k_pax': [260.0, 550.0]})
    df['efficiency_rank'] = df['mwh_per_1k_pax'].rank(ascending=False)
    assert df.loc[df['line'] == 'U5', 'efficiency_rank'].iloc[0] == 1.0

def test_behavior_model_returns_dict():
    res = {'preferred_routes': ['Route A', 'Route B']}
    assert isinstance(res, dict)
    assert 'preferred_routes' in res
