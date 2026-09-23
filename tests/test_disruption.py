"""Offline tests for the Category C graph logic (no TabPFN, no dataset needed).

Synthetic network:  A - B - C - D  on line U1, plus a U2 detour A - X - D.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "ml")]

import disruption as D  # noqa: E402


@pytest.fixture
def net():
    stations = pd.DataFrame({
        "station_id": list("abcdx"),
        "station_name": ["U Alpha (Berlin)", "U Beta (Berlin)", "S+U Gamma Bhf (Berlin)",
                         "U Delta (Berlin)", "U Xray (Berlin)"],
        "latitude": [52.50, 52.51, 52.52, 52.53, 52.51],
        "longitude": [13.40, 13.41, 13.42, 13.43, 13.45],
        "u_bahn_lines": ["U1,U2", "U1", "U1", "U1,U2", "U2"],
    })
    conn = pd.DataFrame({"station_id_1": list("abcax"), "station_id_2": list("bcdxd")})
    return D.Network.from_frames(stations, conn, list(stations["station_name"]))


def _spec(net, a, b, line="U1"):
    return D.make_hypothetical_spec(net, line=line, from_station=a, to_station=b, station=None,
                                    start="2026-07-01 10:00", duration_minutes=60)


def test_resolve_unique_substring_and_prefixes(net):
    assert net.resolve_one("Gamma")[0] == "S+U Gamma Bhf (Berlin)"
    assert net.resolve_one("alpha")[0] == "U Alpha (Berlin)"
    assert net.resolve_one("nowhere")[0] is None


def test_section_closure_marks_interior_unserved_and_finds_detour(net):
    eff = D.apply_closure(net, _spec(net, "Alpha", "Delta"))
    assert eff.section_path == ["U Alpha (Berlin)", "U Beta (Berlin)",
                                "S+U Gamma Bhf (Berlin)", "U Delta (Berlin)"]
    assert set(eff.unserved) == {"U Beta (Berlin)", "S+U Gamma Bhf (Berlin)"}
    alts = D.alternate_paths(net, eff)
    assert alts[0]["paths"][0]["stations"] == ["U Alpha (Berlin)", "U Xray (Berlin)", "U Delta (Berlin)"]
    assert D.surface_links(net, eff) == []          # rail detour exists


def test_no_detour_gives_surface_links():
    # pure line A-B-C-D (a tree): suspending B..C cuts it in two, no rail detour possible
    stations = pd.DataFrame({
        "station_id": list("abcd"),
        "station_name": ["U Alpha (Berlin)", "U Beta (Berlin)", "U Gamma (Berlin)", "U Delta (Berlin)"],
        "latitude": [52.50, 52.51, 52.52, 52.53], "longitude": [13.40, 13.41, 13.42, 13.43],
        "u_bahn_lines": ["U1"] * 4,
    })
    conn = pd.DataFrame({"station_id_1": list("abc"), "station_id_2": list("bcd")})
    tree = D.Network.from_frames(stations, conn, list(stations["station_name"]))
    eff = D.apply_closure(tree, _spec(tree, "Beta", "Gamma"))
    alts = D.alternate_paths(tree, eff)
    assert alts[0]["paths"] == []
    links = D.surface_links(tree, eff)
    assert links and links[0]["straight_line_km"] > 0
    assert {links[0]["from"], links[0]["to"]} == {"U Beta (Berlin)", "U Gamma (Berlin)"}


def test_wrong_line_is_rejected(net):
    with pytest.raises(ValueError):
        D.apply_closure(net, _spec(net, "Alpha", "Xray", line="U1"))


def test_spill_weights_sum_to_one_and_skip_closed(net):
    eff = D.apply_closure(net, _spec(net, "Alpha", "Delta"))
    w = D.spill_targets(net, eff, "U Beta (Berlin)")
    assert abs(sum(w.values()) - 1) < 1e-9
    assert "S+U Gamma Bhf (Berlin)" not in w and "U Beta (Berlin)" not in w


def test_exceed_probability_monotone_and_clamped():
    lv, qv = [0.1, 0.5, 0.9], [50, 90, 130]
    assert D.exceed_probability(90, lv, qv) == 0.5
    assert D.exceed_probability(10, lv, qv) == 1.0
    assert D.exceed_probability(999, lv, qv) == 0.0
    assert D.exceed_probability(70, lv, qv) > D.exceed_probability(110, lv, qv)
