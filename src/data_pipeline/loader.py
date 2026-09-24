import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import re
import warnings

def flow_columns(flows: pd.DataFrame) -> list[str]:
    return [col for col in flows.columns if col != 'timestamp']

class DataLoader:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directory {self.data_dir} does not exist.")
            
    def _find_file(self, keyword: str) -> Path | None:
        keyword = keyword.lower()
        for p in self.data_dir.iterdir():
            if p.is_file() and keyword in p.stem.lower() and p.suffix.lower() == '.csv':
                return p
        return None
        
    def _load_csv(self, keyword: str, **read_kwargs) -> pd.DataFrame | None:
        file_path = self._find_file(keyword)
        if file_path:
            print(f"Loading {keyword} from {file_path.name}...")
            return pd.read_csv(file_path, **read_kwargs)
        print(f"Warning: No file found containing '{keyword}' in stem.")
        return None

    def _validate_dataframe(self, df: pd.DataFrame, name: str,
                             timestamp_col: str | None = None,
                             required_cols: list | None = None,
                             id_cols: list | None = None) -> pd.DataFrame:
        """
        Run data-quality checks on a loaded DataFrame and print a concise report.
        Checks: missing values, invalid/NaT timestamps, duplicate rows, missing identifiers.
        Returns the DataFrame unchanged (no records silently dropped).
        """
        if df is None or df.empty:
            print(f"  [QA] {name}: DataFrame is empty or None - skipping validation.")
            return df

        issues = []

        # 1. Missing values per column
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if not null_cols.empty:
            summary = ", ".join(f"{c}={n}" for c, n in null_cols.items())
            issues.append(f"Missing values: {summary}")

        # 2. Invalid timestamps
        if timestamp_col and timestamp_col in df.columns:
            n_nat = df[timestamp_col].isna().sum()
            if n_nat > 0:
                issues.append(f"Invalid/NaT timestamps in '{timestamp_col}': {n_nat} rows")

        # 3. Duplicate rows
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            issues.append(f"Duplicate rows: {n_dupes}")

        # 4. Required columns present
        if required_cols:
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")

        # 5. Missing identifier values (station/line columns must not be null)
        if id_cols:
            for col in id_cols:
                if col in df.columns:
                    n_missing_id = df[col].isna().sum()
                    if n_missing_id > 0:
                        issues.append(f"Missing identifier in '{col}': {n_missing_id} rows")

        if issues:
            print(f"  [QA] {name} ({len(df)} rows): " + " | ".join(issues))
        else:
            print(f"  [QA] {name} ({len(df)} rows): OK")

        return df

    def load_flows(self) -> pd.DataFrame | None:
        flows = self._load_csv('flows')
        if flows is not None and 'timestamp' in flows.columns:
            flows['timestamp'] = pd.to_datetime(flows['timestamp'], errors='coerce').dt.tz_localize(None)
        self._validate_dataframe(flows, 'flows', timestamp_col='timestamp')
        return flows

    def load_stations(self) -> pd.DataFrame | None:
        st = self._load_csv('stations_with_ubahn')
        self._validate_dataframe(
            st, 'stations',
            required_cols=['station_id', 'station_name', 'latitude', 'longitude', 'u_bahn_lines'],
            id_cols=['station_id', 'station_name']
        )
        return st

    def load_connections(self) -> pd.DataFrame | None:
        conn = self._load_csv('berlin_ubahn_connections')
        self._validate_dataframe(
            conn, 'connections',
            required_cols=['station_id_1', 'station_id_2'],
            id_cols=['station_id_1', 'station_id_2']
        )
        return conn

    def load_events(self) -> pd.DataFrame | None:
        events = self._load_csv('events')
        if events is not None:
            if 'began_local' in events.columns:
                events['began_local'] = pd.to_datetime(events['began_local'], utc=True, errors='coerce').dt.tz_localize(None)
            if 'estimated_end_local' in events.columns:
                events['estimated_end_local'] = pd.to_datetime(events['estimated_end_local'], utc=True, errors='coerce').dt.tz_localize(None)
            if 'estimated_attendance' in events.columns and 'attendance' not in events.columns:
                events['attendance'] = events['estimated_attendance']
        self._validate_dataframe(events, 'events', timestamp_col='began_local')
        return events

    def load_closures(self) -> pd.DataFrame | None:
        closures = self._load_csv('closures')
        if closures is not None:
            if 'when' in closures.columns:
                closures['when'] = pd.to_datetime(closures['when'], errors='coerce').dt.tz_localize(None)
            
            if 'duration' in closures.columns:
                def parse_duration(d):
                    if pd.isna(d):
                        return pd.NaT
                    d_str = str(d)
                    hours, mins = 0, 0
                    h_match = re.search(r'(\d+)h', d_str)
                    m_match = re.search(r'(\d+)min', d_str)
                    if h_match:
                        hours = int(h_match.group(1))
                    if m_match:
                        mins = int(m_match.group(1))
                    if not h_match and not m_match and d_str.isdigit():
                        mins = int(d_str)
                    return pd.Timedelta(hours=hours, minutes=mins)
                    
                closures['duration_td'] = closures['duration'].apply(parse_duration)
                if 'when' in closures.columns:
                    closures['end'] = closures['when'] + closures['duration_td']
            
            if 'description' in closures.columns:
                def extract_type(desc):
                    if not isinstance(desc, str):
                        return 'station_closure'
                    if 'suspended' in desc.lower():
                        return 'line_suspension'
                    return 'station_closure'
                    
                def extract_line(desc):
                    if not isinstance(desc, str):
                        return None
                    m = re.search(r'Line (U\d+)', desc)
                    if m:
                        return m.group(1)
                    return None
                    
                closures['closure_type'] = closures['description'].apply(extract_type)
                closures['_line'] = closures['description'].apply(extract_line)
        self._validate_dataframe(closures, 'closures', timestamp_col='when')
        return closures

    def load_weather(self) -> pd.DataFrame | None:
        weather = self._load_csv('weather')
        if weather is not None:
            weather = weather.rename(columns={weather.columns[0]: 'timestamp'})
            weather['timestamp'] = pd.to_datetime(weather['timestamp'], errors='coerce').dt.tz_localize(None)
        self._validate_dataframe(weather, 'weather', timestamp_col='timestamp')
        return weather

    def load_energy(self) -> pd.DataFrame | None:
        energy = self._load_csv('energy')
        if energy is not None:
            energy = energy.rename(columns={energy.columns[0]: 'date'})\
 
            energy['date'] = pd.to_datetime(energy['date'], errors='coerce').dt.tz_localize(None)
        self._validate_dataframe(energy, 'energy', timestamp_col='date')
        return energy
        
    def load_all(self) -> dict:
        print("--- Data Quality Report ---")
        result = {
            'flows':       self.load_flows(),
            'stations':    self.load_stations(),
            'connections': self.load_connections(),
            'events':      self.load_events(),
            'closures':    self.load_closures(),
            'weather':     self.load_weather(),
            'energy':      self.load_energy()
        }
        print("--- End Data Quality Report ---")
        return result


class DataStore:
    _data: dict = None
    
    @classmethod
    def initialize(cls, data_dir: str):
        loader = DataLoader(data_dir)
        raw_data = loader.load_all()
        cls._data = cls._build(raw_data, Path(data_dir))
        print("DataStore initialized successfully.")
        
    @classmethod
    def get(cls) -> dict:
        if cls._data is None:
            raise RuntimeError("DataStore is not initialized. Call DataStore.initialize(data_dir) first.")
        return cls._data
        
    @classmethod
    def _build(cls, raw: dict, data_dir: Path) -> dict:
        data = raw.copy()
        
        # filter flows
        if data.get('flows') is not None:
            flows = data['flows']
            flows['date'] = flows['timestamp'].dt.date
            counts = flows.groupby('date')['timestamp'].count()
            valid_dates = counts[counts >= 70].index
            data['flows'] = flows[flows['date'].isin(valid_dates)].drop(columns=['date']).copy()
            print(f"Flows filtered: {len(valid_dates)} days remaining.")
            
        from .graph_builder import build_graph, compute_centrality
        from .event_mapper import map_events_to_stations
        from .flow_baseline import compute_station_baselines
        
        if data.get('stations') is not None and data.get('connections') is not None:
            data['graph'] = build_graph(data['stations'], data['connections'])
            data['centrality'] = compute_centrality(data['graph'])
        else:
            data['graph'] = None
            data['centrality'] = None
            
        if data.get('events') is not None and data.get('stations') is not None:
            data['events_mapped'] = map_events_to_stations(data['events'], data['stations'])
        else:
            data['events_mapped'] = None
            
        if data.get('flows') is not None:
            data['baselines'] = compute_station_baselines(data['flows'])
        else:
            data['baselines'] = None
            
        data['historical_index'] = None
        try:
            from hcade.index_builder import build_historical_index
            if data.get('flows') is not None:
                w  = data.get('weather')   if data.get('weather')   is not None else pd.DataFrame()
                em = data.get('events_mapped') if data.get('events_mapped') is not None else pd.DataFrame()
                cl = data.get('closures')  if data.get('closures')  is not None else pd.DataFrame()
                data['historical_index'] = build_historical_index(data['flows'], w, em, cl)
                print(f"Historical index built: {len(data['historical_index'])} rows.")
        except Exception as e:
            print(f"Warning: Could not build historical index: {e}")

        data['stress_coef'] = None
        try:
            from analytics.stress_score import fit_stress_coefficients
            if data.get('flows') is not None:
                w  = data.get('weather')       if data.get('weather')       is not None else pd.DataFrame()
                em = data.get('events_mapped') if data.get('events_mapped') is not None else pd.DataFrame()
                cl = data.get('closures')      if data.get('closures')      is not None else pd.DataFrame()
                data['stress_coef'] = fit_stress_coefficients(data['flows'], w, em, cl)
                print("Stress coefficients fitted.")
        except Exception as e:
            print(f"Warning: Could not fit stress coefficients: {e}")

        return data
