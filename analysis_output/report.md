# Analyse-Report: Berlin U-Bahn Hackathon-Datensatz

Datenordner: `training_dataset` | Aufloesung fuer Korrelation/Kausalitaet: `1h`

> Hinweis: Korrelation und Granger-Kausalitaet sind statistische Hinweise, kein Beweis physikalischer Verursachung. Alle Ergebnisse sollten im Agenten mit Unsicherheit kommuniziert werden.


## Erzeugte Visualisierungen

- `analysis_output/plots/01_total_flow_timeseries.png`
- `analysis_output/plots/02_hourly_weekday_heatmap.png`
- `analysis_output/plots/03_top_stations.png`
- `analysis_output/plots/04_network_graph.png`
- `analysis_output/plots/05_weather_vs_flow.png`
- `analysis_output/plots/05b_weather_bins_prcp.png`
- `analysis_output/plots/05c_weather_bins_temp.png`
- `analysis_output/plots/06_energy_per_line.png`
- `analysis_output/plots/07_events_timeline.png`


## Korrelation: Wetter <-> Fahrgastfluss

**`r_roh`** korreliert die Wettervariable direkt mit dem summierten 15-Minuten-Fluss - meist klein, weil das tagesperiodische Pendlermuster (Stosszeiten) alles ueberlagert.

**`r_bereinigt`** korreliert stattdessen mit der *Fluss-Anomalie* (Abweichung vom ueblichen Fluss zur selben Uhrzeit/demselben Wochentag) - hier zeigt sich der eigentliche Wettereffekt klarer. Das ist die aussagekraeftigere Spalte.

**Faustregel fuer |r|:** < 0.1 sehr schwach · 0.1-0.3 schwach · 0.3-0.5 moderat · > 0.5 stark. `p < 0.05` gilt als statistisch signifikant.


| variable   |   r_roh |   p_roh |   r_bereinigt |   p_bereinigt |    n |
|:-----------|--------:|--------:|--------------:|--------------:|-----:|
| temp       |  -0.018 |  0.1102 |        -0.345 |             0 | 8320 |
| rhum       |   0.012 |  0.278  |         0.323 |             0 | 8320 |
| prcp       |   0.071 |  0      |         0.3   |             0 | 8320 |
| wdir       |   0.006 |  0.5559 |         0.13  |             0 | 8320 |
| pres       |  -0.063 |  0      |        -0.121 |             0 | 8320 |
| cldc       |   0.016 |  0.1499 |         0.116 |             0 | 8320 |
| wspd       |  -0.01  |  0.351  |         0.096 |             0 | 8320 |


## Wetter <-> Fluss auf Tagesebene

Robustere, einfachere Zusatz-Sicht: ein Punkt pro Tag statt 96 verrauschte 15-Minuten-Werte. `temp_mean`/`rhum_mean`/`wspd_mean`/`cldc_mean` sind Tagesmittelwerte, `prcp_sum` ist die Tages-Niederschlagssumme.


| tages_variable   |   pearson_r |   pearson_p |   n_tage |
|:-----------------|------------:|------------:|---------:|
| wspd_mean        |       0.198 |      0.043  |      105 |
| prcp_sum         |       0.159 |      0.1055 |      105 |
| temp_mean        |      -0.152 |      0.1215 |      105 |
| rhum_mean        |       0.137 |      0.1639 |      105 |
| cldc_mean        |       0.045 |      0.6454 |      105 |


## Event-Korrelation auf Tagesebene (alle Events, kein Geocoding noetig)

Summe der erwarteten Besucherzahl aller Events pro Tag vs. gesamtstaedtischer Tages-Fluss. Das erfasst *alle* Events aus der Datei, unabhaengig davon, ob ihr Venue geocodiert werden konnte. **Wochentag-bereinigt:** verglichen wird die Abweichung vom ueblichen Fluss am selben Wochentag (Baseline aus Nicht-Event-Tagen), damit z.B. viele Wochenend-Konzerte nicht faelschlich als 'Events senken den Fluss' erscheinen, nur weil Wochenenden ohnehin ruhiger sind.


| vergleich                                                 | r                     |   p_value |   n_tage |
|:----------------------------------------------------------|:----------------------|----------:|---------:|
| total_attendance vs. Fluss (roh)                          | -0.266                |    0.006  |      105 |
| total_attendance vs. Fluss-Anomalie (wochentag-bereinigt) | 0.388                 |    0.1533 |       15 |
| Tage MIT Event (n=102) vs. Wochentag-Baseline             | +55.0% Fluss-Anomalie |    0      |      105 |


### Fluss-Effekt nach Event-Kategorie (`segment`, wochentag-bereinigt)


| segment        |   n_tage |   uplift_vs_wochentag_baseline_% |
|:---------------|---------:|---------------------------------:|
| Music          |        9 |                             55.3 |
| Arts & Theatre |        5 |                             53.4 |


## Stark korrelierte, NICHT direkt verbundene Stationspaare

Kandidaten fuer versteckte Abhaengigkeiten (z.B. Pendlerstrecken via Umstieg, gemeinsame Zubringer-Buslinie, benachbarte Wohn-/Arbeitsgebiete). Vgl. Trainingsfrage 8.


| station_a                       | station_b                        |   correlation |
|:--------------------------------|:---------------------------------|--------------:|
| U Uhlandstr. (Berlin)           | U Oskar-Helene-Heim (Berlin)     |         0.4   |
| U Vinetastr. (Berlin)           | U Friedrichsfelde (Berlin)       |         0.397 |
| U Konstanzer Str. (Berlin)      | U Schwartzkopffstr. (Berlin)     |         0.396 |
| S+U Gesundbrunnen Bhf (Berlin)  | U Friedrichsfelde (Berlin)       |         0.395 |
| U Uhlandstr. (Berlin)           | U Kaiserin-Augusta-Str. (Berlin) |         0.393 |
| S+U Yorckstr. (Berlin)          | U Hellersdorf (Berlin)           |         0.392 |
| U Tierpark (Berlin)             | U Hönow (Berlin)                 |         0.391 |
| U Uhlandstr. (Berlin)           | U Rotes Rathaus (Berlin)         |         0.391 |
| S+U Gesundbrunnen Bhf (Berlin)  | U Stadtmitte (Berlin)            |         0.391 |
| U Pankstr. (Berlin)             | U Elsterwerdaer Platz (Berlin)   |         0.391 |
| U Reinickendorfer Str. (Berlin) | U Eberswalder Str. (Berlin)      |         0.391 |
| U Paulsternstr. (Berlin)        | U Friedrichsfelde (Berlin)       |         0.391 |
| U Bundestag (Berlin)            | U Paracelsus-Bad (Berlin)        |         0.39  |
| U Platz der Luftbrücke (Berlin) | S+U Brandenburger Tor (Berlin)   |         0.39  |
| U Südstern (Berlin)             | U Paulsternstr. (Berlin)         |         0.39  |


## Granger-Kausalitaet: Wetter -> Fluss

Testet, ob vergangene Werte einer Wettervariable die Vorhersage des Gesamtflusses statistisch signifikant verbessern (p < 0.05 = Hinweis auf Vorhersage-Kausalitaet in diese Richtung; kein Beweis fuer physikalische Verursachung).


| weather_var   | direction          |   best_lag |   min_p_value | significant_(p<0.05)   |
|:--------------|:-------------------|-----------:|--------------:|:-----------------------|
| temp          | temp -> total_flow |          4 |        0      | True                   |
| rhum          | rhum -> total_flow |          4 |        0      | True                   |
| wspd          | wspd -> total_flow |          4 |        0      | True                   |
| pres          | pres -> total_flow |          4 |        0      | True                   |
| prcp          | prcp -> total_flow |          3 |        0.0081 | True                   |
| cldc          | cldc -> total_flow |          4 |        0.0278 | True                   |


## Granger-Kausalitaet zwischen auffaelligen Stationspaaren

Fuer die staerksten nicht-benachbarten korrelierten Paare: testet beide Richtungen, um zu sehen, welche Station 'vorausgeht'.


| direction                                                    |   correlation |   best_lag |   min_p_value | significant_(p<0.05)   |
|:-------------------------------------------------------------|--------------:|-----------:|--------------:|:-----------------------|
| U Uhlandstr. (Berlin) -> U Oskar-Helene-Heim (Berlin)        |         0.4   |          3 |             0 | True                   |
| U Oskar-Helene-Heim (Berlin) -> U Uhlandstr. (Berlin)        |         0.4   |          3 |             0 | True                   |
| U Vinetastr. (Berlin) -> U Friedrichsfelde (Berlin)          |         0.397 |          3 |             0 | True                   |
| U Friedrichsfelde (Berlin) -> U Vinetastr. (Berlin)          |         0.397 |          3 |             0 | True                   |
| U Konstanzer Str. (Berlin) -> U Schwartzkopffstr. (Berlin)   |         0.396 |          3 |             0 | True                   |
| U Schwartzkopffstr. (Berlin) -> U Konstanzer Str. (Berlin)   |         0.396 |          3 |             0 | True                   |
| S+U Gesundbrunnen Bhf (Berlin) -> U Friedrichsfelde (Berlin) |         0.395 |          3 |             0 | True                   |
| U Friedrichsfelde (Berlin) -> S+U Gesundbrunnen Bhf (Berlin) |         0.395 |          3 |             0 | True                   |
| U Uhlandstr. (Berlin) -> U Kaiserin-Augusta-Str. (Berlin)    |         0.393 |          3 |             0 | True                   |
| U Kaiserin-Augusta-Str. (Berlin) -> U Uhlandstr. (Berlin)    |         0.393 |          3 |             0 | True                   |


## Event-Impact-Analyse (stationsgenau)

Vergleicht den Fluss an Stationen nahe einem Venue **in der Stunde vor Event-Beginn** (Anreise-Fenster) mit dem typischen Fluss an denselben Stationen zur selben Tageszeit an anderen Tagen (Baseline, Welch-t-Test). Venues werden ueber eine bekannte Lookup-Tabelle, einen Cache und Live-Geocoding (OpenStreetMap) aufgeloest.


**Zusammenfassung:** 358/417 Venues aufgeloest, 358/417 Events tatsaechlich analysiert (Zeitfenster/Stationen vorhanden). Davon 76 mit signifikantem Effekt (p<0.05), durchschnittlicher Uplift +14.3%.


### Top 20 Events nach |Uplift|


| event                                                                 | venue                 |   attendance | nearby_stations                                                                                                                          |   pre_event_mean |   baseline_mean |   uplift_vs_baseline_% |   p_value | significant_(p<0.05)   |
|:----------------------------------------------------------------------|:----------------------|-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|-----------------:|----------------:|-----------------------:|----------:|:-----------------------|
| MAGMA 2026 | 08.+09.08.2026 WEEKEND TICKET                            | RSO.BERLIN            |         2057 | U Blaschkoallee (Berlin)                                                                                                                 |             96.2 |             7.5 |                 1182.1 |    0.1191 | False                  |
| Omar76 - Tour 2026 - Tour 2026 | Premium Upgrade (no Ticket included) | nan                   |         2145 | U Karl-Marx-Str. (Berlin), S+U Neukölln (Berlin), U Leinestr. (Berlin), S+U Hermannstr. (Berlin)...                                      |            289.5 |            42.3 |                  583.9 |    0.0154 | True                   |
| Berlin | Old but Gold Ü30 Hip Hop Festival @ Zitadelle Spandau        | nan                   |         1789 | U Altstadt Spandau (Berlin), S+U Rathaus Spandau (Berlin), U Zitadelle (Berlin)                                                          |            128.8 |            22.5 |                  473.3 |    0.263  | False                  |
| MAGMA 2026 | 08.08.2026 DAY 1 TICKET                                  | RSO.BERLIN            |         1964 | U Blaschkoallee (Berlin)                                                                                                                 |            104   |            18.7 |                  456.1 |    0.0609 | False                  |
| Klassentreffen Festival                                               | nan                   |         1914 | U Jakob-Kaiser-Platz (Berlin)                                                                                                            |             91   |            17.7 |                  414.4 |    0.2761 | False                  |
| TFELD 2026                                                            | nan                   |         2090 | U Hallesches Tor (Berlin), U Gneisenaustr. (Berlin), U Mehringdamm (Berlin), U Platz der Luftbrücke (Berlin)...                          |            292.5 |            58.5 |                  400.2 |    0.0267 | True                   |
| MAGMA 2026 | 09.08.2026 DAY 2 TICKET                                  | RSO.BERLIN            |         2038 | U Blaschkoallee (Berlin)                                                                                                                 |             91.8 |            18.8 |                  387.5 |    0.1878 | False                  |
| FKA twigs - Body High Tour                                            | nan                   |         1636 | U Frankfurter Tor (Berlin)                                                                                                               |            427.8 |           191.4 |                  123.4 |    0.0312 | True                   |
| Cirque du Soleil ALIZÉ                                                | nan                   |          627 | U Bundestag (Berlin), U Kurfürstenstr. (Berlin), U Mendelssohn-Bartholdy-Park (Berlin), U Kochstr. (Checkpoint Charlie) (Berlin)...      |            262.5 |           128.1 |                  104.9 |    0.1115 | False                  |
| Hayley Williams                                                       | Tempodrom Berlin      |         1485 | U Kurfürstenstr. (Berlin), U Mendelssohn-Bartholdy-Park (Berlin), U Kochstr. (Checkpoint Charlie) (Berlin), U Hallesches Tor (Berlin)... |            250.5 |           124.1 |                  101.9 |    0.0188 | True                   |
| Cirque du Soleil ALIZÉ                                                | nan                   |          694 | U Bundestag (Berlin), U Kurfürstenstr. (Berlin), U Mendelssohn-Bartholdy-Park (Berlin), U Kochstr. (Checkpoint Charlie) (Berlin)...      |           2791.2 |          1396.9 |                   99.8 |    0.0159 | True                   |
| System Of A Down | Business Seat Packages                             | nan                   |         1746 | U Ruhleben (Berlin), U Olympia-Stadion (Berlin), U Neu-Westend (Berlin)                                                                  |           1055.8 |           537.7 |                   96.4 |    0.021  | True                   |
| System Of A Down | VIP Packages                                       | nan                   |         1911 | U Ruhleben (Berlin), U Olympia-Stadion (Berlin), U Neu-Westend (Berlin)                                                                  |           1055.8 |           537.7 |                   96.4 |    0.021  | True                   |
| System Of A Down                                                      | nan                   |         2149 | U Ruhleben (Berlin), U Olympia-Stadion (Berlin), U Neu-Westend (Berlin)                                                                  |           1055.8 |           537.7 |                   96.4 |    0.021  | True                   |
| ivri                                                                  | nan                   |         1886 | U Kottbusser Tor (Berlin), U Görlitzer Bahnhof (Berlin), U Schlesisches Tor (Berlin), U Schönleinstr. (Berlin)...                        |           1589.2 |           817.2 |                   94.5 |    0.0172 | True                   |
| The Swingin' Hermlins                                                 | nan                   |         2270 | U Mierendorffplatz (Berlin), S+U Jungfernheide Bhf (Berlin), U Sophie-Charlotte-Platz (Berlin), U Deutsche Oper (Berlin)...              |           2507   |          1312.5 |                   91   |    0.0103 | True                   |
| Chezile - Wish You Were Here EU/UK Tour                               | nan                   |         2343 | U Kottbusser Tor (Berlin), U Görlitzer Bahnhof (Berlin), U Schlesisches Tor (Berlin), U Schönleinstr. (Berlin)...                        |           1283.2 |           820.2 |                   56.5 |    0.1917 | False                  |
| Lachkater - die Stand Up Comedy Show                                  | Tati Goes Underground |          589 | U Bernauer Str. (Berlin), S+U Alexanderplatz Bhf (Berlin), U Rosa-Luxemburg-Platz (Berlin), U Schillingstr. (Berlin)...                  |           1291.2 |           834.5 |                   54.7 |    0.1685 | False                  |
| A night with Sinatra in Berlin                                        | nan                   |          682 | U Hansaplatz (Berlin), U Ernst-Reuter-Platz (Berlin), S+U Zoologischer Garten Bhf (Berlin), U Augsburger Str. (Berlin)...                |           2351.8 |          5040   |                  -53.3 |    0.0072 | True                   |
| Cirque du Soleil ALIZÉ                                                | nan                   |          653 | U Bundestag (Berlin), U Kurfürstenstr. (Berlin), U Mendelssohn-Bartholdy-Park (Berlin), U Kochstr. (Checkpoint Charlie) (Berlin)...      |            664.5 |          1417.6 |                  -53.1 |    0.0417 | True                   |


## Closure-Impact-Analyse

Fuer jede Streckensperrung: vergleicht den Fluss an den (per Textabgleich erkannten) betroffenen Stationen und an deren direkten Nachbarn vor vs. waehrend der Sperrung. Ein Rueckgang an den gesperrten Stationen und ein Anstieg an Nachbarstationen ist ein Hinweis auf Umleitungsverhalten.


_Keine Daten verfuegbar._
