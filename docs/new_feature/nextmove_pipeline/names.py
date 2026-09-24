"""
Text normalization and station-name matching.

Station names in the data look like 'S+U Warschauer Str. (Berlin)', while event
addresses and closure descriptions say 'Warschauer Straße' or 'Warschauer Str.'.
Everything is therefore reduced to a canonical *key* before comparing:

    'S+U Warschauer Str. (Berlin)'   -> 'warschauer strasse'
    'S+U Friedrichstr. Bhf (Berlin)' -> 'friedrichstrasse'
"""
import difflib
import re


def normalize_text(text) -> str:
    """Lowercase, 'ß' -> 'ss', abbreviation 'str' / 'str.' at a word end -> 'strasse',
    collapse whitespace. ('Spichernstr', 'Spichernstr.' and 'Spichernstraße' become equal.)"""
    t = str(text).lower().replace("ß", "ss")
    t = re.sub(r"str\b\.?", "strasse", t)
    return re.sub(r"\s+", " ", t).strip()


def station_key(name: str) -> str:
    """Canonical key of a station name: no '(Berlin)', no 'S+U'/'U'/'S' prefix, no 'Bhf'."""
    k = normalize_text(name)
    k = re.sub(r"\s*\(berlin\)\s*$", "", k)
    k = re.sub(r"^(s\+u|s|u)\s+", "", k)
    k = re.sub(r"\s+bhf$", "", k)
    return k.strip()


class StationNameIndex:
    """Resolves free-text station mentions to the exact station names used in the data."""

    def __init__(self, station_names):
        self.key_to_name = {station_key(n): n for n in station_names}
        # Longest keys first, so 'kaiserin-augusta-strasse' wins over a shorter overlapping key.
        self._keys = sorted(self.key_to_name, key=len, reverse=True)

    def resolve(self, mention: str, cutoff: float = 0.85):
        """Single station mention (e.g. 'Hallesches Tor') -> station name, or None.
        Exact key match first, then a fuzzy match to tolerate typos like 'Cottbusser Platz'."""
        key = station_key(mention)
        if key in self.key_to_name:
            return self.key_to_name[key]
        close = difflib.get_close_matches(key, self._keys, n=1, cutoff=cutoff)
        return self.key_to_name[close[0]] if close else None

    def find_in_text(self, text: str):
        """All stations whose key occurs as a whole word in `text` (longest match first,
        overlapping shorter matches are dropped)."""
        t = normalize_text(text)
        hits, taken = [], []
        for k in self._keys:
            if len(k) <= 3:
                continue
            for m in re.finditer(r"(?<!\w)" + re.escape(k) + r"(?!\w)", t):
                if not any(m.start() < e and s < m.end() for s, e in taken):
                    taken.append((m.start(), m.end()))
                    hits.append(self.key_to_name[k])
                    break
        return hits
