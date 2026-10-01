"""Inspection guidance: time-of-day deviations, .inp parsing and the benchmark rank check."""
import numpy as np
import pandas as pd
import pytest

from conftest import needs_inp
from waterguard.inspection import (STD_FLOOR_M, fit_typical_pressure, link_segments, parse_inp,
                                   pressure_deviation, sensor_rank_check, slot_of_day)

INP = """[TITLE]
toy
[JUNCTIONS]
 n1 10 0 ;comment
 n2 10 0
 n3 10 0
[PIPES]
;ID node1 node2
 p1 n1 n2 100 0.1 100 0 Open
 p2 n2 n3 100 0.1 100 0 Open
[PUMPS]
 PUMP_1 n1 n3 HEAD 1
[COORDINATES]
;Node X Y
 n1 0 0
 n2 10 0
 n3 10 10
[END]
"""


def _pressures(days=3):
    ts = pd.date_range("2018-01-01", periods=288 * days, freq="5min")
    slot = slot_of_day(ts)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Timestamp": ts,
                       "a": 50 + np.sin(slot / 288 * 2 * np.pi) + rng.normal(0, 0.2, len(ts)),
                       "flat": np.full(len(ts), 39.09)})
    return df


def test_slot_of_day():
    ts = pd.DatetimeIndex(["2018-01-01 00:00", "2018-01-01 00:05", "2018-03-02 23:55"])
    assert slot_of_day(ts).tolist() == [0, 1, 287]


def test_typical_uses_only_reference_rows_and_floors_std():
    P = _pressures()
    ref = P["Timestamp"].iloc[:288 * 2]
    P_mod = P.copy()
    P_mod.loc[288 * 2:, "a"] += 100  # changing non-reference rows must not change the typical values
    m1, s1 = fit_typical_pressure(P, ref)
    m2, s2 = fit_typical_pressure(P_mod, ref)
    pd.testing.assert_frame_equal(m1, m2)
    assert (s1["flat"] == STD_FLOOR_M).all()  # constant sensor -> floor, not zero
    assert len(m1) == 288


def test_pressure_deviation_sign_and_finiteness():
    P = _pressures()
    med, spread = fit_typical_pressure(P, P["Timestamp"].iloc[:288 * 2])
    P.loc[288 * 2 + 10, "a"] -= 5  # a pressure drop
    dev = pressure_deviation(P, P["Timestamp"].iloc[288 * 2:], med, spread)
    assert np.isfinite(dev[["a", "flat"]].to_numpy()).all()
    assert dev.loc[10, "a"] < -5
    assert dev["a"].idxmin() == 10


def test_parse_inp_and_segments(tmp_path):
    f = tmp_path / "toy.inp"
    f.write_text(INP)
    nodes, links = parse_inp(f)
    assert nodes["id"].tolist() == ["n1", "n2", "n3"]
    assert set(links["id"]) == {"p1", "p2", "PUMP_1"}
    seg = link_segments(nodes, links).set_index("id")
    assert tuple(seg.loc["p2", ["xm", "ym"]]) == (10.0, 5.0)


def test_sensor_rank_check():
    window = pd.DataFrame({"s_near": [-3.0, -2.0], "s_far": [-1.0, -1.0], "s_up": [1.0, 2.0]})
    xy = pd.DataFrame({"x": [0, 100, 50], "y": [0, 0, 50]}, index=["s_near", "s_far", "s_up"])
    r = sensor_rank_check(window, xy, (1.0, 0.0))
    assert r["top_sensor"] == "s_near" and r["nearest_sensor"] == "s_near" and r["nearest_sensor_rank"] == 1


@needs_inp
def test_real_ltown_inp_has_all_pressure_sensors_and_leak_pipes():
    from waterguard.config import DATA_DIR
    from waterguard.data import EXPECTED_LEAK_LOCATIONS
    nodes, links = parse_inp(DATA_DIR / "L-TOWN.inp")
    sensors = pd.read_csv(DATA_DIR.parent / "outputs" / "network_sensors.csv")["id"]
    assert set(sensors) <= set(nodes["id"]) and len(sensors) == 33
    assert set(EXPECTED_LEAK_LOCATIONS) <= set(links["id"])


@pytest.mark.parametrize("col", ["nearest_sensor_rank"])
def test_inspection_check_output(col):
    from waterguard.config import OUTPUTS_DIR
    c = pd.read_csv(OUTPUTS_DIR / "inspection_check_v2.csv")
    assert len(c) == 2 and set(c["leaking_pipe"]) == {"p158", "p369"}
    assert c[col].between(1, 33).all()
