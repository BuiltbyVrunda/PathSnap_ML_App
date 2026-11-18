"""Inference loader: reads ONLY the prediction inputs (.track, .nodes, .arcs).

The ground-truth .route file is deliberately NOT read here. It is read only by
ground_truth.py, after the prediction is made.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class TripDataError(Exception):
    """A trip file is missing or cannot be parsed."""


@dataclass
class RoadNetwork:
    nodes: np.ndarray  # (M, 2) float: longitude, latitude. Row index = node_id
    arcs: np.ndarray   # (K, 2) int: from_node_id, to_node_id. Row index = arc_id (arcs are directed)


def read_table(folder: Path, trip_id: str, ext: str, ncols: int, dtype) -> np.ndarray:
    """Read a header-less, whitespace-separated file <trip_id><ext> into an (n, ncols) array."""
    path = folder / f"{trip_id}{ext}"
    if not path.is_file():
        raise TripDataError(f"Trip {trip_id}: missing file {path.name}")
    if path.stat().st_size == 0:
        raise TripDataError(f"Trip {trip_id}: file {path.name} is empty")
    try:
        data = np.loadtxt(path, dtype=dtype, ndmin=2)
    except ValueError as err:
        raise TripDataError(f"Trip {trip_id}: cannot parse {path.name} ({err})") from err
    if data.shape[1] != ncols:
        raise TripDataError(f"Trip {trip_id}: {path.name} has {data.shape[1]} columns, expected {ncols}")
    return data


def load_points(folder: Path, trip_id: str) -> np.ndarray:
    """GPS observations as an (N, 3) array: longitude, latitude, time in seconds. Row = obs_idx."""
    return read_table(folder, trip_id, ".track", 3, float)


def load_road_network(folder: Path, trip_id: str) -> RoadNetwork:
    """Road nodes and directed arcs for one trip, with a check that arcs point at real nodes."""
    nodes = read_table(folder, trip_id, ".nodes", 2, float)
    arcs = read_table(folder, trip_id, ".arcs", 2, np.int64)
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise TripDataError(f"Trip {trip_id}: .arcs refers to a node_id outside 0..{len(nodes) - 1}")
    return RoadNetwork(nodes=nodes, arcs=arcs)
