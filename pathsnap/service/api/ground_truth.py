"""SERVER-SIDE SCORING ONLY. The only module that reads the ground-truth .route file.

Rules:
  * Call score_trip() only AFTER predict_trip() has produced its prediction.
  * Per-point reference arcs stay inside this module. score_trip() returns only
    aggregate counts, so ground truth never reaches the predictor or the UI.

==========================  PLACEHOLDER  ==========================
TODO(Member 1): replace align_points_to_route() with the team's alignment module
(pathsnap/data/reference_alignment), including "ambiguous" handling.
====================================================================
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .geo import LocalProjection, point_segment_distances
from .loader import RoadNetwork, read_table

# A point farther than this from its aligned reference arc is "unaligned" and not scored.
ALIGN_MAX_DIST_M = 50.0


def load_reference_route(folder: Path, trip_id: str) -> Optional[np.ndarray]:
    """Ordered reference arc_ids from <trip_id>.route, or None if the trip has no .route file."""
    if not (folder / f"{trip_id}.route").is_file():
        return None
    return read_table(folder, trip_id, ".route", 1, np.int64)[:, 0]


def align_points_to_route(points: np.ndarray, route: np.ndarray, net: RoadNetwork) -> np.ndarray:
    """Give each GPS point one reference arc_id (-1 = unaligned).

    PLACEHOLDER, order-aware ("forward-only") alignment:
    the .route lists arcs in driving order, so as we walk through the GPS points in time,
    the matched position in the route may stay the same or move forward, never backwards.
    Among all such forward-only assignments we pick the one with the smallest total
    point-to-arc distance (dynamic programming, like DTW).
    """
    proj = LocalProjection.around(points[:, :2])
    gps_xy = proj.to_xy(points[:, :2])
    node_xy = proj.to_xy(net.nodes)
    starts = node_xy[net.arcs[route, 0]]
    ends = node_xy[net.arcs[route, 1]]

    # dist[i, j] = metres from GPS point i to the j-th arc of the route (in chunks to limit memory).
    dist = np.vstack([point_segment_distances(gps_xy[i:i + 500], starts, ends)
                      for i in range(0, len(gps_xy), 500)])

    # cost[i, j] = best total distance for points 0..i when point i sits at route position j.
    # Forward-only: point i-1 must sit at a position <= j, hence the running minimum.
    cost = np.empty_like(dist)
    cost[0] = dist[0]
    for i in range(1, len(dist)):
        cost[i] = dist[i] + np.minimum.accumulate(cost[i - 1])

    # Walk back from the last point to recover the chosen route position of every point.
    pos = np.empty(len(dist), dtype=np.int64)
    pos[-1] = int(np.argmin(cost[-1]))
    for i in range(len(dist) - 2, -1, -1):
        pos[i] = int(np.argmin(cost[i, :pos[i + 1] + 1]))

    ref_arcs = route[pos].copy()
    too_far = dist[np.arange(len(dist)), pos] > ALIGN_MAX_DIST_M
    ref_arcs[too_far] = -1
    return ref_arcs


def score_trip(folder: Path, trip_id: str, points: np.ndarray, net: RoadNetwork,
               predicted: List[Optional[int]]) -> Optional[Dict]:
    """Pointwise (observation-level) accuracy = correct / aligned.

    An aligned point with no prediction (None) counts as incorrect.
    Returns None when the trip has no reference or no point could be aligned.
    """
    route = load_reference_route(folder, trip_id)
    if route is None or len(route) == 0:
        return None
    ref_arcs = align_points_to_route(points, route, net)

    aligned = ref_arcs >= 0
    pred = np.array([-2 if a is None else a for a in predicted])  # -2 never equals a reference arc
    n_aligned = int(aligned.sum())
    if n_aligned == 0:
        return None
    n_correct = int((pred[aligned] == ref_arcs[aligned]).sum())
    return {"accuracy": n_correct / n_aligned, "correct": n_correct, "aligned": n_aligned}
