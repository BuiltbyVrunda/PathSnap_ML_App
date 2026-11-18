"""Unit tests for the Viterbi decoder edge cases and robustness."""

import pytest
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.models import (
    DecodedTrip,
    Match,
    Observation,
    RoadArc,
    RoadNode,
    RoutePart,
    ScoredCandidate,
    Trip,
)
from pathsnap.decoder.viterbi.decoder import decode_trip


def test_deterministic_tie_breaking_smaller_arc_id():
    """Verify that ties in emission or path probabilities break deterministically to smaller arc_id."""
    nodes = [
        RoadNode(node_id=0, lon=0.0, lat=0.0),
        RoadNode(node_id=1, lon=0.001, lat=0.0),
    ]
    # Two parallel arcs between same endpoints with identical length
    arcs = [
        RoadArc(arc_id=5, from_node_id=0, to_node_id=1, length_m=100.0),
        RoadArc(arc_id=2, from_node_id=0, to_node_id=1, length_m=100.0),
    ]
    observations = [
        Observation(obs_idx=0, lon=0.0005, lat=0.0, time_s=0.0),
    ]
    trip = Trip(trip_id="tie_trip", observations=observations, nodes=nodes, arcs=arcs)

    # Identical emissions 0.5
    candidates = [
        ScoredCandidate(
            trip_id="tie_trip",
            obs_idx=0,
            arc_id=5,
            along_arc_fraction=0.5,
            distance_m=0.0,
            emission_prob=0.5,
        ),
        ScoredCandidate(
            trip_id="tie_trip",
            obs_idx=0,
            arc_id=2,
            along_arc_fraction=0.5,
            distance_m=0.0,
            emission_prob=0.5,
        ),
    ]

    decoded = decode_trip(trip, candidates)
    # Must choose arc_id=2 because 2 < 5
    assert decoded.matches[0].arc_id == 2


def test_empty_trip():
    """Verify handling of an empty trip with 0 observations."""
    trip = Trip(trip_id="empty_trip", observations=[], nodes=[], arcs=[])
    decoded = decode_trip(trip, [])
    assert decoded.trip_id == "empty_trip"
    assert decoded.matches == []
    assert decoded.route_parts == []


def test_single_observation_trip():
    """Verify decoding a trip with exactly 1 observation."""
    nodes = [RoadNode(node_id=0, lon=0.0, lat=0.0), RoadNode(node_id=1, lon=0.001, lat=0.0)]
    arcs = [RoadArc(arc_id=10, from_node_id=0, to_node_id=1, length_m=100.0)]
    observations = [Observation(obs_idx=0, lon=0.0005, lat=0.0, time_s=0.0)]
    trip = Trip(trip_id="single_obs", observations=observations, nodes=nodes, arcs=arcs)

    candidates = [
        ScoredCandidate(
            trip_id="single_obs",
            obs_idx=0,
            arc_id=10,
            along_arc_fraction=0.5,
            distance_m=2.0,
            emission_prob=1.0,
        )
    ]

    decoded = decode_trip(trip, candidates)
    assert len(decoded.matches) == 1
    assert decoded.matches[0].arc_id == 10
    assert len(decoded.route_parts) == 1
    assert decoded.route_parts[0].arc_ids == [10]
    assert decoded.route_parts[0].break_before is None


def test_same_arc_forward_movement():
    """Verify that multiple consecutive observations along the same arc stay on that arc."""
    nodes = [RoadNode(node_id=0, lon=0.0, lat=0.0), RoadNode(node_id=1, lon=0.01, lat=0.0)]
    arcs = [RoadArc(arc_id=0, from_node_id=0, to_node_id=1, length_m=1000.0)]
    observations = [
        Observation(obs_idx=0, lon=0.002, lat=0.0, time_s=0.0),
        Observation(obs_idx=1, lon=0.005, lat=0.0, time_s=10.0),
        Observation(obs_idx=2, lon=0.008, lat=0.0, time_s=20.0),
    ]
    trip = Trip(trip_id="same_arc_trip", observations=observations, nodes=nodes, arcs=arcs)

    candidates = [
        ScoredCandidate(
            trip_id="same_arc_trip",
            obs_idx=0,
            arc_id=0,
            along_arc_fraction=0.2,
            distance_m=0.0,
            emission_prob=1.0,
        ),
        ScoredCandidate(
            trip_id="same_arc_trip",
            obs_idx=1,
            arc_id=0,
            along_arc_fraction=0.5,
            distance_m=0.0,
            emission_prob=1.0,
        ),
        ScoredCandidate(
            trip_id="same_arc_trip",
            obs_idx=2,
            arc_id=0,
            along_arc_fraction=0.8,
            distance_m=0.0,
            emission_prob=1.0,
        ),
    ]

    decoded = decode_trip(trip, candidates)
    assert [m.arc_id for m in decoded.matches] == [0, 0, 0]
    assert len(decoded.route_parts) == 1
    # Road segment traversed is arc 0
    assert decoded.route_parts[0].arc_ids == [0]
    assert decoded.route_parts[0].start_obs_idx == 0
    assert decoded.route_parts[0].end_obs_idx == 2


def test_speed_limit_filtering():
    """Verify that excessive speeds exceeding max_speed_mps trigger an invalid_transition break."""
    nodes = [
        RoadNode(node_id=0, lon=0.0, lat=0.0),
        RoadNode(node_id=1, lon=0.01, lat=0.0),
        RoadNode(node_id=2, lon=0.02, lat=0.0),
    ]
    arcs = [
        RoadArc(arc_id=0, from_node_id=0, to_node_id=1, length_m=1000.0),
        RoadArc(arc_id=1, from_node_id=1, to_node_id=2, length_m=1000.0),
    ]
    # Obs 0 at 0s, Obs 1 at 1s -> distance ~1000m in 1s = 1000 m/s > 45 m/s
    observations = [
        Observation(obs_idx=0, lon=0.001, lat=0.0, time_s=0.0),
        Observation(obs_idx=1, lon=0.015, lat=0.0, time_s=1.0),
    ]
    trip = Trip(trip_id="speed_trip", observations=observations, nodes=nodes, arcs=arcs)

    candidates = [
        ScoredCandidate(
            trip_id="speed_trip",
            obs_idx=0,
            arc_id=0,
            along_arc_fraction=0.1,
            distance_m=0.0,
            emission_prob=1.0,
        ),
        ScoredCandidate(
            trip_id="speed_trip",
            obs_idx=1,
            arc_id=1,
            along_arc_fraction=0.5,
            distance_m=0.0,
            emission_prob=1.0,
        ),
    ]

    config = DecoderConfig(max_speed_mps=45.0)
    decoded = decode_trip(trip, candidates, config)

    # Because 1000 m/s > 45 m/s, transition is impossible -> invalid_transition
    assert len(decoded.route_parts) == 2
    assert decoded.route_parts[1].break_before == "invalid_transition"


def test_dict_adapter_compatibility():
    """Verify that plain dict inputs from raw JSON are handled cleanly."""
    trip_dict = {
        "trip_id": "dict_trip",
        "observations": [
            {"obs_idx": 0, "lon": 0.0, "lat": 0.0, "time_s": 0.0},
            {"obs_idx": 1, "lon": 0.001, "lat": 0.0, "time_s": 5.0},
        ],
        "nodes": [
            {"node_id": 0, "lon": 0.0, "lat": 0.0},
            {"node_id": 1, "lon": 0.001, "lat": 0.0},
        ],
        "arcs": [
            {"arc_id": 0, "from_node_id": 0, "to_node_id": 1, "length_m": 100.0},
        ],
    }

    cands_dict = [
        {
            "trip_id": "dict_trip",
            "obs_idx": 0,
            "arc_id": 0,
            "along_arc_fraction": 0.1,
            "distance_m": 1.0,
            "emission_prob": 1.0,
        },
        {
            "trip_id": "dict_trip",
            "obs_idx": 1,
            "arc_id": 0,
            "along_arc_fraction": 0.9,
            "distance_m": 1.0,
            "emission_prob": 1.0,
        },
    ]

    decoded = decode_trip(trip_dict, cands_dict)
    assert decoded.trip_id == "dict_trip"
    assert len(decoded.matches) == 2
    assert decoded.to_dict()["matches"][0]["arc_id"] == 0
