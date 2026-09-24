"""
NextMove normalization pipeline.

Modules (in data-flow order):
    config.py    all thresholds, paths and manual overrides
    loading.py   read the raw CSV files
    names.py     station-name normalization and matching
    geo.py       address -> coordinates (cached Nominatim) -> nearest stations  [also a CLI]
    episodes.py  events / closures -> affected stations and time windows
    baseline.py  normal-flow model, decomposition into weather part and rest
    pipeline.py  end-to-end run that writes the tables                          [CLI]
"""
