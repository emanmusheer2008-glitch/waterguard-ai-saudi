"""Inspection guidance: which pressure sensors look most unusual right now?

This is a *descriptive* aid, separate from the Random Forest. For every
pressure sensor and every 5-minute slot of the day, the "usual" value is the
median of NON-severe TRAINING periods at that time of day. A sensor's
deviation is (current - usual) / spread, where spread is the training
standard deviation for that slot, floored at ``STD_FLOOR_M`` because some
L-Town sensors (e.g. n215) are almost constant and are recorded in 0.01 m
steps, which would otherwise give near-infinite scores.

A leak lets water escape, so pressure near it tends to fall. Sensors with
the most negative deviation are therefore a reasonable place to START an
investigation. It is NOT leak localisation: sensors are sparse, deviations
spread through the network, and the benchmark has only two severe test
episodes to check against.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

STD_FLOOR_M = 0.05  # metres; see module docstring
SLOTS_PER_DAY = 288


def slot_of_day(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """0..287 index of the 5-minute slot within the day."""
    ts = pd.DatetimeIndex(ts)
    return np.asarray(ts.hour * 12 + ts.minute // 5)


def fit_typical_pressure(pressures: pd.DataFrame, reference_ts) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-slot median and floored std of each pressure sensor, using only ``reference_ts`` rows.

    ``pressures`` has a Timestamp column plus one column per sensor.
    Returns (median, spread), each indexed by slot (0..287) with sensor columns.
    """
    P = pressures.set_index("Timestamp")
    ref = P.loc[P.index.isin(pd.DatetimeIndex(reference_ts))]
    slots = slot_of_day(ref.index)
    med = ref.groupby(slots).median()
    spread = ref.groupby(slots).std().clip(lower=STD_FLOOR_M)
    return med, spread


def pressure_deviation(pressures: pd.DataFrame, timestamps, med: pd.DataFrame, spread: pd.DataFrame) -> pd.DataFrame:
    """Deviation score for each sensor at each requested timestamp (negative = lower than usual)."""
    P = pressures.set_index("Timestamp").loc[pd.DatetimeIndex(timestamps)]
    s = slot_of_day(P.index)
    z = (P.to_numpy() - med.loc[s].to_numpy()) / spread.loc[s].to_numpy()
    out = pd.DataFrame(z, columns=P.columns)
    out.insert(0, "Timestamp", P.index.to_numpy())
    return out


# ---------------------------------------------------------------- network model
def _section(text: str, name: str) -> list[list[str]]:
    m = re.search(r"\[" + re.escape(name) + r"\](.*?)(?=\n\[)", text, re.S)
    if not m:
        return []
    rows = []
    for line in m.group(1).splitlines():
        line = line.split(";", 1)[0].strip()
        if line:
            rows.append(line.split())
    return rows


def parse_inp(path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read node coordinates and link end-points from an EPANET .inp file.

    Returns (nodes[id, x, y], links[id, kind, node1, node2]) for pipes, pumps and valves.
    """
    text = Path(path).read_text(errors="replace")
    nodes = pd.DataFrame([r[:3] for r in _section(text, "COORDINATES")], columns=["id", "x", "y"])
    nodes[["x", "y"]] = nodes[["x", "y"]].astype(float)
    links = []
    for kind, sec in [("pipe", "PIPES"), ("pump", "PUMPS"), ("valve", "VALVES")]:
        links += [[r[0], kind, r[1], r[2]] for r in _section(text, sec)]
    return nodes, pd.DataFrame(links, columns=["id", "kind", "node1", "node2"])


def link_segments(nodes: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Links as straight segments x0,y0 -> x1,y1 plus the midpoint (for plotting / distances)."""
    xy = nodes.set_index("id")[["x", "y"]]
    seg = links.join(xy, on="node1").join(xy, on="node2", rsuffix="1").rename(columns={"x": "x0", "y": "y0"})
    seg = seg.dropna(subset=["x0", "y0", "x1", "y1"])
    seg["xm"] = (seg["x0"] + seg["x1"]) / 2
    seg["ym"] = (seg["y0"] + seg["y1"]) / 2
    return seg[["id", "kind", "x0", "y0", "x1", "y1", "xm", "ym"]].reset_index(drop=True)


def sensor_rank_check(dev_window: pd.DataFrame, sensor_xy: pd.DataFrame, leak_xy: tuple[float, float]) -> dict:
    """Evaluation only: where does the sensor nearest the true leak rank in mean deviation?

    ``dev_window`` = deviation rows for one episode (sensor columns only).
    ``sensor_xy`` = DataFrame indexed by sensor id with x, y.
    """
    mean_dev = dev_window.mean().sort_values()  # most negative first
    d = np.hypot(sensor_xy["x"] - leak_xy[0], sensor_xy["y"] - leak_xy[1]).reindex(mean_dev.index)
    nearest = d.idxmin()
    return {
        "top_sensor": mean_dev.index[0],
        "top_sensor_distance": float(d[mean_dev.index[0]]),
        "nearest_sensor": nearest,
        "nearest_sensor_distance": float(d[nearest]),
        "nearest_sensor_rank": int(list(mean_dev.index).index(nearest) + 1),
        "n_sensors": int(len(mean_dev)),
        "top5": ", ".join(mean_dev.index[:5]),
    }
