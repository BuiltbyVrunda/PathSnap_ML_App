"""Shared metric geometry for inference and independent reference alignment."""

import numpy as np
import shapely
from shapely import STRtree

from pathsnap.data.loading import Trip


class SegmentIndex:
    def __init__(self, trip: Trip, arc_ids: np.ndarray):
        self.start, self.end = trip.node_xy[trip.arc_nodes[arc_ids]].transpose(1, 0, 2)
        self.vector = self.end - self.start
        self.length_squared = np.einsum("ij,ij->i", self.vector, self.vector)
        segments = shapely.linestrings(np.stack((self.start, self.end), axis=1))
        points = self.length_squared == 0
        segments[points] = shapely.points(self.start[points])
        self.tree = STRtree(segments)

    def query(self, xy: np.ndarray, radius_m: float):
        obs, segments = self.tree.query(shapely.points(xy), predicate="dwithin", distance=radius_m)
        offset = xy[obs] - self.start[segments]
        vector = self.vector[segments]
        lengths = self.length_squared[segments]
        fraction = np.divide(
            np.einsum("ij,ij->i", offset, vector),
            lengths,
            out=np.zeros(len(obs)),
            where=lengths > 0,
        ).clip(0, 1)
        distance = np.linalg.norm(offset - fraction[:, None] * vector, axis=1)
        return obs, segments, fraction, distance

    def heading_difference(self, segments: np.ndarray, heading_deg: np.ndarray) -> np.ndarray:
        dx, dy = self.vector[segments].T
        bearing_deg = np.rad2deg(np.arctan2(dx, dy))
        difference = np.abs((bearing_deg - heading_deg + 180) % 360 - 180)
        difference[self.length_squared[segments] == 0] = np.nan
        return difference


def observation_motion(trip: Trip, min_heading_move_m: float, max_heading_gap_s: float):
    """Compute causal movement context using only the preceding GPS observation."""
    delta_t = np.r_[np.nan, np.diff(trip.track[:, 2])]
    movement = np.vstack(([np.nan, np.nan], np.diff(trip.observation_xy, axis=0)))
    shift = np.linalg.norm(movement, axis=1)
    speed = np.divide(shift, delta_t, out=np.full_like(shift, np.nan), where=delta_t > 0)
    heading = np.rad2deg(np.arctan2(movement[:, 0], movement[:, 1]))
    reliable = (delta_t > 0) & (delta_t <= max_heading_gap_s) & (shift >= min_heading_move_m)
    heading[~reliable] = np.nan
    return delta_t, speed, heading
