import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import math

VENUE_COORDS = {
    'uber arena': (52.5079, 13.4396),
    'mercedes-benz arena': (52.5079, 13.4396),
    'mercedes benz arena': (52.5079, 13.4396),
    'olympiastadion': (52.5148, 13.2394),
    'waldbuehne': (52.5144, 13.2321),
    'waldbuhne': (52.5144, 13.2321),
    'tempodrom': (52.5026, 13.3794),
    'messe berlin': (52.5080, 13.2782),
    'velodrom': (52.5425, 13.4699),
    'columbiahalle': (52.4798, 13.3879),
    'admiralspalast': (52.5236, 13.3876),
    'arena berlin': (52.4971, 13.4609),
    'huxleys': (52.4791, 13.3872),
    'so36': (52.5027, 13.4344),
    'volksbuehne': (52.5268, 13.4115),
    'berliner philharmonie': (52.5097, 13.3697),
    'zitadelle': (52.5404, 13.2054),
    'kindl-buehne': (52.4831, 13.3977),
    'parkbuehne': (52.5144, 13.2321),
}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def resolve_event_coords(event_row: pd.Series) -> tuple | None:
    venue = str(event_row.get('venue_name', '')).lower()
    addr = str(event_row.get('address', '')).lower()
    
    for key, coords in VENUE_COORDS.items():
        if key in venue or key in addr:
            return coords
    return None

def map_events_to_stations(events: pd.DataFrame, stations: pd.DataFrame, radius_km: float = 1.5) -> pd.DataFrame:
    events_mapped = events.copy()
    if stations is None or stations.empty or events_mapped.empty:
        events_mapped['resolved_lat'] = np.nan
        events_mapped['resolved_lon'] = np.nan
        events_mapped['nearest_stations'] = [[] for _ in range(len(events_mapped))]
        events_mapped['nearest_station'] = None
        return events_mapped

    resolved_lats = []
    resolved_lons = []
    nearest_stations_list = []
    nearest_station_single = []

    for _, event_row in events_mapped.iterrows():
        coords = resolve_event_coords(event_row)
        if coords is None:
            resolved_lats.append(np.nan)
            resolved_lons.append(np.nan)
            nearest_stations_list.append([])
            nearest_station_single.append(None)
            continue
            
        lat, lon = coords
        resolved_lats.append(lat)
        resolved_lons.append(lon)
        
        near_stats = []
        closest_st = None
        min_dist = float('inf')
        
        for _, st_row in stations.iterrows():
            st_lat = st_row.get('latitude')
            st_lon = st_row.get('longitude')
            st_name = st_row.get('station_name')
            if pd.notna(st_lat) and pd.notna(st_lon) and pd.notna(st_name):
                dist = haversine_km(lat, lon, st_lat, st_lon)
                if dist <= radius_km:
                    near_stats.append(st_name)
                if dist < min_dist:
                    min_dist = dist
                    closest_st = st_name
                    
        nearest_stations_list.append(near_stats)
        if min_dist <= radius_km:
            nearest_station_single.append(closest_st)
        else:
            nearest_station_single.append(None)

    events_mapped['resolved_lat'] = resolved_lats
    events_mapped['resolved_lon'] = resolved_lons
    events_mapped['nearest_stations'] = nearest_stations_list
    events_mapped['nearest_station'] = nearest_station_single

    return events_mapped

def get_events_near_station(station_name: str, date, events_mapped: pd.DataFrame, radius_km: float = 1.5) -> pd.DataFrame:
    if events_mapped is None or events_mapped.empty or 'began_local' not in events_mapped.columns:
        return pd.DataFrame()
        
    date_str = pd.to_datetime(date).date()
    
    mask = []
    for _, row in events_mapped.iterrows():
        event_date = pd.to_datetime(row['began_local']).date()
        near = row.get('nearest_stations', [])
        if event_date == date_str and station_name in near:
            mask.append(True)
        else:
            mask.append(False)
            
    return events_mapped[mask].copy()
