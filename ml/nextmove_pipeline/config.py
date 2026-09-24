"""
Central configuration for the NextMove normalization pipeline.

Everything that is a "judgement call" (thresholds, radii, holiday dates, manual
venue -> station overrides) lives here, so the rest of the code stays free of
magic numbers.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]            # repository root (this package lives in ml/nextmove_pipeline/)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "training dataset"   # the model is FITTED on this split ...
TEST_DATA_DIR = PROJECT_ROOT / "data" / "testing dataset"      # ... and applied to this one (the organisers' held-out days, 22 Sept - 1 Oct)
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "normalized"

# Our own geocode cache (read + write). Written as {"venue | address": {lat, lon, source, query}}.
DEFAULT_GEOCODE_CACHE = PROJECT_ROOT / "data" / "processed" / "geocode_cache_pipeline.json"
# Older caches from earlier scripts. They are only READ (never modified) to avoid re-querying OSM.
SEED_GEOCODE_CACHES = [
    PROJECT_ROOT / "analysis_output" / "geocode_cache.json",
    PROJECT_ROOT / "data" / "processed" / "geocode_cache.json",
]     # (absent in this repository: only read if they exist)

# Suffix of the raw input files, e.g. flows_pre_innotrans.csv. If a file with this
# suffix does not exist, the first CSV starting with the stem is used instead.
DATA_SUFFIX = "_pre_innotrans"
TEST_SUFFIX = "_rest"          # the test split's files (flows_rest.csv ...): tried second, see loading.find_file

# --------------------------------------------------------------------------- geography
EARTH_RADIUS_M = 6_371_000
# Rough bounding box of Berlin; geocoding hits outside of it are rejected as wrong matches.
BERLIN_BBOX = dict(lat_min=52.33, lat_max=52.68, lon_min=13.08, lon_max=13.77)
# A venue farther away than this from every U-Bahn station counts as "no station nearby".
MAX_VENUE_DISTANCE_M = 1500
# Nominatim usage policy: at most one request per second, and an identifying user agent.
NOMINATIM_USER_AGENT = "nextmove-hackathon-innotrans2026"
NOMINATIM_MIN_INTERVAL_S = 1.1

# --------------------------------------------------------------------------- events / closures
MIN_EVENT_ATTENDANCE = 2000   # smaller events are treated as noise (not excluded from the fit)
DEFAULT_EVENT_HOURS = 3       # used when estimated_end_local is missing or invalid
MAX_HOP = 2                   # an episode influences stations up to this many hops away
# Time window around an episode that is excluded from the baseline fit.
EXCLUDE_HOURS_BEFORE = 3
EXCLUDE_HOURS_AFTER = 2

# Manual venue -> station overrides. Checked BEFORE geocoding, because domain knowledge beats
# the straight-line nearest station (e.g. Waldbuehne crowds use U2 Olympia-Stadion).
# Keys are matched as substrings of the normalized "venue | address" text.
# Values must be exact station names from stations_with_ubahn.csv (validated at load time).
VENUE_TO_STATION = {
    "uber arena": "S+U Warschauer Str. (Berlin)",
    "mercedes-benz arena": "S+U Warschauer Str. (Berlin)",
    "uber-platz": "S+U Warschauer Str. (Berlin)",
    "olympiastadion": "U Olympia-Stadion (Berlin)",
    "olympischer platz": "U Olympia-Stadion (Berlin)",
    "olympiapark": "U Olympia-Stadion (Berlin)",
    "waldbühne": "U Olympia-Stadion (Berlin)",
    "admiralspalast": "S+U Friedrichstr. Bhf (Berlin)",
    "columbiahalle": "U Platz der Luftbrücke (Berlin)",
    "tempodrom": "U Möckernbrücke (Berlin)",
}

# --------------------------------------------------------------------------- baseline model
# Berlin school summer holidays 2026 (inclusive). PLEASE VERIFY against the official calendar.
SCHOOL_HOLIDAYS = [("2026-07-09", "2026-08-22")]
RAIN_THRESHOLD_MM = 0.1   # a slot counts as "rainy" above this precipitation
RIDGE_ALPHA = 1.0         # shrinkage of the weather / holiday / trend terms (level dummies: none)
