"""Generate inference candidates using only GPS observations and the road graph."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from pathsnap.data.loading import Trip

from .spatial import SegmentIndex, observation_motion

FEATURE_COLUMNS = ("along_arc_fraction", "distance_m", "heading_diff_deg", "speed_mps", "delta_t_s")


@dataclass(frozen=True, slots=True)
class Candidate:
    trip_id: str
    obs_idx: int
    arc_id: int
    along_arc_fraction: float
    distance_m: float
    heading_diff_deg: float | None
    speed_mps: float | None
    delta_t_s: float | None


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 60.0
    max_candidates: int = 24
    min_heading_move_m: float = 3.0
    max_heading_gap_s: float = 10.0

    def __post_init__(self):
        thresholds = (self.radius_m, self.min_heading_move_m, self.max_heading_gap_s)
        if any(not isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("candidate distances and times must be positive and finite")
        if type(self.max_candidates) is not int or self.max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")


def select_candidates(obs: np.ndarray, arc: np.ndarray, distance: np.ndarray, config: CandidateConfig):
    """Return row indices within the radius and nearest cap, sorted by the shared key."""
    eligible = np.flatnonzero(distance <= config.radius_m)
    if not eligible.size:
        return eligible
    ranked = eligible[np.lexsort((arc[eligible], distance[eligible], obs[eligible]))]
    groups = obs[ranked]
    row = np.arange(len(ranked))
    group_start = np.maximum.accumulate(np.where(np.r_[True, groups[1:] != groups[:-1]], row, 0))
    keep = ranked[row - group_start < config.max_candidates]
    return keep[np.lexsort((arc[keep], obs[keep]))]


def _optional_number(value: float) -> float | None:
    return float(value) if isfinite(value) else None


def generate_candidates(trip: Trip, config: CandidateConfig = CandidateConfig()) -> list[Candidate]:
    """Return at most max_candidates rows per observation, ordered by (obs_idx, arc_id)."""
    index = SegmentIndex(trip, np.arange(len(trip.arc_nodes)))
    delta_t, speed, heading = observation_motion(trip, config.min_heading_move_m, config.max_heading_gap_s)
    candidates = []
    for start in range(0, len(trip.track), 1024):
        obs, arc, fraction, distance = index.query(trip.observation_xy[start:start + 1024], config.radius_m)
        keep = select_candidates(obs, arc, distance, config)
        obs_idx, arc_id = obs[keep] + start, arc[keep]
        differences = index.heading_difference(arc_id, heading[obs_idx])
        for i, j, position, dist, angle in zip(obs_idx, arc_id, fraction[keep], distance[keep], differences, strict=True):
            candidates.append(Candidate(
                trip.trip_id, int(i), int(j), float(position), float(dist), _optional_number(angle),
                _optional_number(speed[i]), _optional_number(delta_t[i]),
            ))
    return candidates
