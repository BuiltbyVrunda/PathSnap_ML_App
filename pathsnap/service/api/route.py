"""Turn per-point predicted arcs into drawable route parts and a continuity score.

Uses only the prediction and the road network (no ground truth).
TODO(Member 3): the decoder may later return route parts directly, including
intermediate arcs between matches. Then build_route_parts() can be replaced.
"""
from typing import Dict, List, Optional

from .loader import RoadNetwork


def _connected(net: RoadNetwork, a: int, b: int) -> bool:
    """True if arc b starts where arc a ends (a directed, drivable link)."""
    return bool(net.arcs[a, 1] == net.arcs[b, 0])


def build_route_parts(predicted: List[Optional[int]], net: RoadNetwork) -> List[Dict]:
    """Group the predicted arcs into contiguous parts. A new part starts when the next arc
    does not connect to the previous one. Same shape as the shared decoder format."""
    parts: List[Dict] = []
    current: Optional[Dict] = None
    skipped_unmatched = False  # did we pass points with no prediction since the last arc?

    for obs_idx, arc in enumerate(predicted):
        if arc is None:
            skipped_unmatched = True
            continue
        if current is not None:
            last = current["arc_ids"][-1]
            if arc == last:                      # still on the same arc
                current["end_obs_idx"] = obs_idx
                skipped_unmatched = False
                continue
            if _connected(net, last, arc):       # moved onto a connected arc
                current["arc_ids"].append(arc)
                current["end_obs_idx"] = obs_idx
                skipped_unmatched = False
                continue
        # Start a new part. The first part has no break; later parts say why they broke.
        reason = None if current is None else ("no_candidates" if skipped_unmatched else "invalid_transition")
        current = {"start_obs_idx": obs_idx, "end_obs_idx": obs_idx, "arc_ids": [arc], "break_before": reason}
        parts.append(current)
        skipped_unmatched = False
    return parts


def parts_to_geometry(parts: List[Dict], net: RoadNetwork) -> Optional[Dict]:
    """GeoJSON MultiLineString with one line per part, coordinates as [lon, lat].
    Separate lines mean the map shows a gap at each break instead of a false straight line."""
    if not parts:
        return None
    lines = []
    for part in parts:
        arcs = part["arc_ids"]
        first_node = net.arcs[arcs[0], 0]
        node_ids = [first_node] + [net.arcs[a, 1] for a in arcs]   # from-node of first arc, then each to-node
        lines.append([[float(net.nodes[n, 0]), float(net.nodes[n, 1])] for n in node_ids])
    return {"type": "MultiLineString", "coordinates": lines}


def route_continuity(predicted: List[Optional[int]], net: RoadNetwork) -> Dict:
    """PLACEHOLDER continuity metric (TODO: replace with Member 3's definition).

    Take the matched arcs in order, drop repeats of the same arc, then count how many
    consecutive pairs are connected (to_node of one == from_node of the next).
    """
    sequence: List[int] = []
    for arc in predicted:
        if arc is not None and (not sequence or sequence[-1] != arc):
            sequence.append(arc)
    transitions = len(sequence) - 1
    connected = sum(_connected(net, a, b) for a, b in zip(sequence, sequence[1:]))
    return {
        "value": connected / transitions if transitions > 0 else None,  # None = not measurable
        "connected": int(connected),
        "transitions": max(transitions, 0),
    }
