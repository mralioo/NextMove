"""
Central configuration for InnoTrans 2026 Hackathon agent.
All constants, paths, and thresholds in one place.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR_DEFAULT = PROJECT_ROOT / 'data' / 'training dataset'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
EVAL_DIR = PROJECT_ROOT / 'evaluation'

# LLM configuration (from environment variables)
LLM_BASE_URL = os.getenv('LLM_BASE_URL', None)  # None = use default OpenAI
LLM_API_KEY  = os.getenv('LLM_API_KEY', 'dummy')
LLM_MODEL    = os.getenv('LLM_MODEL', 'gpt-4o')

# Data thresholds
MIN_TIMESTAMPS_PER_DAY = 96        # < 96 = incomplete day, filter out
EVENT_ATTENDANCE_MAX   = 2394      # max observed in training data
EVENT_BUCKET_SMALL     = 500       # < 500 = SMALL
EVENT_BUCKET_MEDIUM    = 1500      # 500-1500 = MEDIUM; > 1500 = LARGE

# Flow analysis
ANOMALY_ZSCORE_NETWORK  = 2.5     # Z-score threshold for network-level anomalies
ANOMALY_ZSCORE_STATION  = 2.0     # Z-score threshold for station-level anomalies
ROLLING_BASELINE_DAYS   = 28      # 4-week rolling baseline
OVERFLOW_RISK_THRESHOLD = 1.3     # overflow_ratio > 1.3 = at risk
REDISTRIBUTION_FACTOR   = 0.85   # fraction of stranded pax who reroute

# StressScore default coefficients (overridden by OLS fit)
STRESS_ALPHA      = 0.35   # normalised flow weight
STRESS_BETA       = 0.25   # event proximity weight
STRESS_GAMMA_TEMP = -0.15  # temperature effect (NEGATIVE: heat reduces flow)
STRESS_GAMMA_PRCP = 0.02   # precipitation effect (near zero)
STRESS_DELTA      = 0.20   # cascade pressure weight

# StressScore risk thresholds
STRESS_LOW_THRESHOLD    = 0.4
STRESS_MEDIUM_THRESHOLD = 0.7

# Network
EVENT_STATION_RADIUS_KM = 1.5     # radius to find nearest stations for events
DEFAULT_REROUTE_K       = 3       # number of alternative routes to compute
MINUTES_PER_STOP        = 2.5     # estimated travel time per stop (minutes)

# HCADE
HCADE_TOP_K             = 5       # number of historical analogues to retrieve
HCADE_OUTCOME_WINDOW_H  = 2       # hours to analyze after matched incident

# Weather buckets (coco codes)
STORM_COCO_MIN = 14.0
HEAT_TEMP_MIN  = 32.0
HEAVY_RAIN_MM  = 5.0
RAIN_MM_MIN    = 0.0

# Training data date range
TRAIN_START = '2026-06-10'
TRAIN_END   = '2026-09-21'

# Known pre-computed answers for validation
KNOWN_ANSWERS = {
    'Q5_worst_efficiency_line': 'U5',
    'Q5_best_efficiency_line': 'U1',
    'Q3_closure_line': 'U6',
    'Q3_closure_date': '2026-07-13',
    'Q3_closure_duration': '1h30min',
    'Q3_closure_reason': 'safety inspection',
}
