# Hackathon 2026 Berlin U-Bahn Dataset - Schema and Data Descriptions

Contains data organization, schema, and descriptions of the provided information which will be used for the hackathon challenge. All timestamps in the files are presented as Berlin local time.
Note that `flows.csv` and `weather_data.csv` share the same 15-minute timestamp grain.

## stations_with_ubahn.csv
Constains real U-Bahn stations (for 8 out of 9 lines).
Note that one U-Bahn line is not available frp, the real network.

| Column | Type | Description |
| --- | --- | --- |
| `station_id` | string, primary key | VBB (Verkehrsverbund Berlin-Brandenburg) station identifier. |
| `station_name` | string | Station display name; used as a station-flow column name in `flows.csv`. |
| `longitude` | decimal | WGS84 longitude of the station. |
| `latitude` | decimal | WGS84 latitude of the station. |
| `u_bahn_lines` | string | Serving U-Bahn line or lines (comma-separated when multiple). |

## berlin_ubahn_connections.csv
Contains real bidirectional connections between the U-Bahn stations.
Note that station adjecency matrix can be created using this file.

| Column | Type | Description |
| --- | --- | --- |
| `station_id_1` | string, foreign key | VBB station endpoint; references `stations_with_ubahn.station_id`. |
| `station_id_2` | string, foreign key | VBB station other endpoint; references `stations_with_ubahn.station_id`. |

## berlin_ubahn_lines_used.csv
Contains Berlin U-Bahn metadata.

| Column | Type | Description |
| --- | --- | --- |
| `line_id` | string, primary key | VBB line identifier. |
| `line_name` | string | Public line name, for example `U1`. |
| `operator` | string/integer identifier | Transit operator identifier. |
| `mode` | string | Transport mode; currently `train`. |
| `product` | string | Product type; currently `subway`. |
| `n_variants` | integer | Number of alternative routes during the operations. |

## flows.csv
Contains 168 station-level passenger flow simulated time series. One row every 15 minutes, number of passengers per corresponding station (column-wise).
Station names match values in `stations_with_ubahn.station_name`.

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime | 15-minute observation timestamp. |
| One column per station name | integer | Simulated passenger flow/count at that station for the timestamp. |

## berlin_events_summer_2026.csv
Contains real events in the city of Berlin, from 2026-06-10 until 2026-09-21.

| Column | Type | Description |
| --- | --- | --- |
| `event_name` | string | Event name / title. |
| `began_local` | ISO 8601 datetime with UTC offset | Scheduled local start time. |
| `estimated_end_local` | ISO 8601 datetime with UTC offset, nullable | Estimated local end time. |
| `venue_name` | string, nullable | Venue name. |
| `address` | string | Venue or street address. |
| `city` | string | City. |
| `country` | string | Country. |
| `segment` | string | Broad event category. |
| `genre` | string | Event genre. |
| `description` | string | Event category description. |
| `estimated_attendance` | integer | Estimated attendee count. |
| `event_url` | URL/string | Event source URL. |

## closures.csv
Contains station closures, simulated data.

| Column | Type | Description |
| --- | --- | --- |
| `when` | datetime | Closure start time. |
| `duration` | string | Human-readable duration, for example `3h30min`. |
| `description` | string | Closure or disruption description. |

## weather_data.csv
Contains the real weather data for the Berlin metro area, from 2026-06-10 to 2026-09-21.

| Column | Type | Description |
| --- | --- | --- |
| Unnamed first column | datetime | Timestamp; logically `timestamp`. |
| `temp` | decimal | Air temperature, degrees Celsius. |
| `rhum` | decimal | Relative humidity, percent. |
| `prcp` | decimal | Precipitation amount. |
| `wdir` | decimal | Wind direction in degrees (reference clockwise true north). |
| `wspd` | decimal | Wind speed, kph. |
| `pres` | decimal | Atmospheric pressure, hPa. |
| `cldc` | decimal | Cloud cover, percent. |
| `coco` | decimal/integer code | Weather-condition category code, such as clear, cloudy, thunderstorms, etc. |

## energy_consumption.csv
Contains daily energy consumption per line, simulated data.
| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime | Date in MM-DD-YYYY , MM - month, DD - day in the month, YYYY - year|
| One column per line | integer | Simulated energy consumption in MWh per line for the date. |


## Relationships

- `stations_with_ubahn.station_id` is the station key.
- `flows.csv` maps to stations through its station-name column headers.
- `flows.timestamp` joins to the timestamp in `weather_data.csv`.
- Events have no direct station key in the CSV output; they can be associated to stations through venue or address geocoding.