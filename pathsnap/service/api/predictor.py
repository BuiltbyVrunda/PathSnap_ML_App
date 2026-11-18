"""Model call used by the API.

==========================  PLACEHOLDER  ==========================
TODO(team): replace the body of predict_trip() with the real pipeline before the demo:
  1. candidate generation       (Member 1, pathsnap/data/candidates)
  2. trained classifier scores  (Member 2, pathsnap/models)
  3. HMM + Viterbi decoding     (Member 3, pathsnap/decoder)
Keep the same signature so the API does not need to change.
====================================================================

The predictor receives ONLY the GPS points and the road network.
It never sees the ground-truth route.
"""
from typing import List, Optional

import numpy as np
import shapely

from .geo import LocalProjection
from .loader import RoadNetwork

# Points farther than this from every arc get no prediction (None).
MAX_MATCH_DISTANCE_M = 100.0


def predict_trip(points: np.ndarray, road_network: RoadNetwork) -> List[Optional[int]]:
    """Return one predicted arc_id per GPS point (None = no arc found).

    PLACEHOLDER method: nearest arc by straight-line distance in metres.
    It ignores arc direction and road connectivity, which the real HMM will use.
    """
    # Lon/lat -> metres, centred on this trip.
    proj = LocalProjection.around(points[:, :2])
    gps_xy = proj.to_xy(points[:, :2])
    node_xy = proj.to_xy(road_network.nodes)

    # One straight line per arc: from its from-node to its to-node.
    starts = node_xy[road_network.arcs[:, 0]]
    ends = node_xy[road_network.arcs[:, 1]]
    arc_lines = shapely.linestrings(np.stack([starts, ends], axis=1))

    # R-tree spatial index: finds the closest arc quickly, even with 50k+ arcs.
    tree = shapely.STRtree(arc_lines)
    point_idx, arc_idx = tree.query_nearest(
        shapely.points(gps_xy), max_distance=MAX_MATCH_DISTANCE_M, all_matches=False
    )

    predicted: List[Optional[int]] = [None] * len(points)
    for p, a in zip(point_idx, arc_idx):
        predicted[int(p)] = int(a)
    return predicted
