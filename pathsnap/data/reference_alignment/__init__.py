"""Read and align a supplied reference route independently of inference candidates."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

import numpy as np

from pathsnap.data.candidates.spatial import SegmentIndex, observation_motion
from pathsnap.data.loading import Trip, read_numeric, require_rows


@dataclass(frozen=True, slots=True)
class Reference:
    trip_id: str
    obs_idx: int
    ref_arc_id: int | None
    alignment_status: Literal["aligned", "ambiguous", "unaligned"]


@dataclass(frozen=True)
class AlignmentConfig:
    radius_m: float = 75.0
    heading_cost_m: float = 12.0
    ambiguity_margin_m: float = 3.0
    min_heading_move_m: float = 3.0
    max_heading_gap_s: float = 10.0

    def __post_init__(self):
        positive = (self.radius_m, self.min_heading_move_m, self.max_heading_gap_s)
        nonnegative = (self.heading_cost_m, self.ambiguity_margin_m)
        if any(not isfinite(value) or value <= 0 for value in positive):
            raise ValueError("alignment distances and times must be positive and finite")
        if any(not isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("alignment heading cost and ambiguity margin must be nonnegative and finite")


def read_reference_route(trip: Trip, trip_dir: str | Path) -> np.ndarray:
    path = Path(trip_dir) / f"{trip.trip_id}.route"
    route = read_numeric(path, 1, np.int64)[:, 0]
    require_rows(path, (route >= 0) & (route < len(trip.arc_nodes)), "invalid reference arc ID")
    return route


def reference_break_positions(trip: Trip, route: np.ndarray) -> np.ndarray:
    """Return zero-based reference positions that start immediately after a break."""
    endpoints = trip.arc_nodes[route]
    return np.flatnonzero(endpoints[:-1, 1] != endpoints[1:, 0]) + 1


def _ordered_alignment(positions: np.ndarray, reward: np.ndarray, offsets: np.ndarray, route_size: int):
    """Return one optimal monotone path and the best total score through each state."""
    # Index 0 is the start sentinel. Route position p occupies slot p+1.
    best_end = np.zeros(route_size + 1)
    ending_state = np.full(route_size + 1, -1, dtype=np.int64)
    predecessors = np.full(len(reward), -1, dtype=np.int64)
    through = np.empty(len(reward))
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        if start == end:
            continue
        slots = positions[start:end] + 1
        prefix = np.maximum.accumulate(best_end)
        before = prefix[slots]
        scores = before + reward[start:end]
        through[start:end] = scores
        predecessors[start:end] = ending_state[np.searchsorted(prefix, before)]
        improve = scores > best_end[slots]
        best_end[slots[improve]] = scores[improve]
        ending_state[slots[improve]] = np.flatnonzero(improve) + start

    last = int(np.argmax(best_end))
    optimum = best_end[last]
    state = ending_state[last]
    path = []
    while state >= 0:
        path.append(state)
        state = predecessors[state]

    # A combined prefix and suffix through a state quantifies ordering-aware ambiguity.
    best_start = np.zeros(route_size)
    for start, end in zip(offsets[-2::-1], offsets[:0:-1], strict=True):
        if start == end:
            continue
        slots = positions[start:end]
        later = np.maximum.accumulate(best_start[::-1])[::-1][slots]
        through[start:end] += later
        best_start[slots] = np.maximum(best_start[slots], later + reward[start:end])
    return path, optimum, through


def align_references(trip: Trip, route: np.ndarray, config: AlignmentConfig = AlignmentConfig()) -> list[Reference]:
    """Align observations to ordered route occurrences, using no inference candidates."""
    route = np.asarray(route)
    if route.ndim != 1 or route.dtype.kind not in "iu" or np.any((route < 0) | (route >= len(trip.arc_nodes))):
        raise ValueError(f"{trip.trip_id}: expected an array of valid reference arc IDs")
    labels = [Reference(trip.trip_id, i, None, "unaligned") for i in range(len(trip.track))]
    if not route.size:
        return labels

    index = SegmentIndex(trip, route)
    _, _, heading = observation_motion(trip, config.min_heading_move_m, config.max_heading_gap_s)
    groups = []
    for start in range(0, len(trip.track), 1024):
        obs, positions, _, distance = index.query(trip.observation_xy[start:start + 1024], config.radius_m)
        obs += start
        heading_diff = index.heading_difference(positions, heading[obs])
        heading_cost = config.heading_cost_m * np.nan_to_num(heading_diff) / 180
        reward = config.radius_m + 1.0 - distance - heading_cost
        keep = np.flatnonzero(reward > 0)
        order = keep[np.lexsort((positions[keep], obs[keep]))]
        groups.append((obs[order], positions[order], reward[order]))

    obs, positions, reward = [np.concatenate(group) for group in zip(*groups, strict=True)]
    offsets = np.r_[0, np.bincount(obs, minlength=len(trip.track)).cumsum()]
    path, optimum, through = _ordered_alignment(positions, reward, offsets, len(route))
    for state in path:
        i = int(obs[state])
        arc_id = int(route[positions[state]])
        start, end = offsets[i:i + 2]
        alternatives = (route[positions[start:end]] != arc_id) & (
            through[start:end] >= optimum - config.ambiguity_margin_m - 1e-9
        )
        labels[i] = (
            Reference(trip.trip_id, i, None, "ambiguous")
            if alternatives.any()
            else Reference(trip.trip_id, i, arc_id, "aligned")
        )
    return labels
