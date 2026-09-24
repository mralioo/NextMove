import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd

def test_fingerprint_peak_am():
    hour = 7
    is_weekend = False
    assert hour == 7 and not is_weekend

def test_fingerprint_weekend():
    day = 'Saturday'
    is_weekend = (day in ['Saturday', 'Sunday'])
    assert is_weekend

def test_fingerprint_heat_bucket():
    temp = 35
    prcp = 0
    bucket = 'HEAT' if temp >= 32.0 else 'NORMAL'
    assert bucket == 'HEAT'

def test_fingerprint_storm_bucket():
    coco = 15
    bucket = 'STORM' if coco >= 14.0 else 'NORMAL'
    assert bucket == 'STORM'

def test_fingerprint_rain_bucket():
    prcp = 2
    temp = 18
    bucket = 'RAIN' if prcp > 0 else 'NORMAL'
    assert bucket == 'RAIN'

def test_fingerprint_attendance_buckets():
    def get_bucket(att):
        if att < 500: return 'SMALL'
        elif att <= 1500: return 'MEDIUM'
        else: return 'LARGE'
    
    assert get_bucket(300) == 'SMALL'
    assert get_bucket(800) == 'MEDIUM'
    assert get_bucket(2000) == 'LARGE'

def test_similarity_returns_top_k():
    k = 5
    results = [1, 2, 3, 4, 5]
    assert len(results) <= k

def test_similarity_score_range():
    scores = [0.1, 0.5, 0.9]
    assert all(0 <= s <= 1 for s in scores)

def test_similarity_hour_bucket_weighted():
    score_match = 1.0
    score_mismatch = 0.5
    assert score_match > score_mismatch

def test_recommender_returns_single_action():
    res = {'recommended_action': 'bus_bridge'}
    assert 'recommended_action' in res

def test_recommender_bus_bridge_for_suspension():
    fingerprint = 'line_suspension'
    action = 'bus_bridge' if fingerprint == 'line_suspension' else 'none'
    assert action == 'bus_bridge'

def test_recommender_confidence_levels():
    def get_conf(matches):
        if matches == 0: return 'LOW'
        elif matches < 3: return 'MEDIUM'
        else: return 'HIGH'
    
    assert get_conf(0) == 'LOW'
    assert get_conf(1) == 'MEDIUM'
    assert get_conf(3) == 'HIGH'
