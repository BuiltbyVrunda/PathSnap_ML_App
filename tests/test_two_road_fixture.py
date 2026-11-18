"""Unit tests for the shared two-road fixture and scenarios 1-5 from handoffs/03."""

import pytest
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.evaluation.comparator import compare_classifier_vs_viterbi
from pathsnap.decoder.evaluation.metrics import compute_top1_accuracy
from pathsnap.decoder.models import (
    ClassifierMatch,
    Observation,
    Reference,
    RoadArc,
    RoadNode,
    ScoredCandidate,
    Trip,
)
from pathsnap.decoder.viterbi.decoder import decode_trip
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph


def make_two_road_fixture_trip() -> Trip:
    """Create the two-road fixture described in handoffs/03-decoder-and-route-evaluation.md.

    Nodes:
      0: (0, 0)
      1: (0.001, 0)
      2: (0.002, 0)
      3: (0, 0.0001)
      4: (0.001, 0.0001)
      5: (0.002, 0.0001)

    Arcs (approx 111m each):
      0: 0 -> 1
      1: 1 -> 2
      2: 3 -> 4
      3: 4 -> 5
    """
    nodes = [
        RoadNode(node_id=0, lon=0.0, lat=0.0),
        RoadNode(node_id=1, lon=0.001, lat=0.0),
        RoadNode(node_id=2, lon=0.002, lat=0.0),
        RoadNode(node_id=3, lon=0.0, lat=0.0001),
        RoadNode(node_id=4, lon=0.001, lat=0.0001),
        RoadNode(node_id=5, lon=0.002, lat=0.0001),
    ]

    arcs = [
        RoadArc(arc_id=0, from_node_id=0, to_node_id=1, length_m=111.0),
        RoadArc(arc_id=1, from_node_id=1, to_node_id=2, length_m=111.0),
        RoadArc(arc_id=2, from_node_id=3, to_node_id=4, length_m=111.0),
        RoadArc(arc_id=3, from_node_id=4, to_node_id=5, length_m=111.0),
    ]

    observations = [
        Observation(obs_idx=0, lon=0.00025, lat=0.0, time_s=1.0),
        Observation(obs_idx=1, lon=0.00175, lat=0.0, time_s=11.0),
    ]

    return Trip(trip_id="00000001", observations=observations, nodes=nodes, arcs=arcs)


def test_scenario_1_two_disconnected_parallel_roads():
    """Scenario 1: Two disconnected parallel roads.

    Emissions:
      Obs 0: arc 0 = 0.9, arc 2 = 0.1
      Obs 1: arc 1 = 0.2, arc 3 = 0.8

    Classifier-only chooses [0, 3].
    Viterbi chooses [0, 1] because [0, 1] is continuous while [0, 3] is disconnected.
    Accuracy improves from 1/2 to 2/2.
    """
    trip = make_two_road_fixture_trip()

    candidates = [
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=0,
            arc_id=0,
            along_arc_fraction=0.25,
            distance_m=0.0,
            emission_prob=0.9,
        ),
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=0,
            arc_id=2,
            along_arc_fraction=0.25,
            distance_m=11.1,
            emission_prob=0.1,
        ),
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=1,
            arc_id=1,
            along_arc_fraction=0.75,
            distance_m=0.0,
            emission_prob=0.2,
        ),
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=1,
            arc_id=3,
            along_arc_fraction=0.75,
            distance_m=11.1,
            emission_prob=0.8,
        ),
    ]

    references = [
        Reference(trip_id="00000001", obs_idx=0, ref_arc_id=0, alignment_status="aligned"),
        Reference(trip_id="00000001", obs_idx=1, ref_arc_id=1, alignment_status="aligned"),
    ]

    # Equal start priors and equal transition weights for feasible connections
    config = DecoderConfig(use_equal_transition_weights=True)

    # 1. Greedy baseline matches
    classifier_matches = [
        ClassifierMatch(trip_id="00000001", obs_idx=0, arc_id=0),
        ClassifierMatch(trip_id="00000001", obs_idx=1, arc_id=3),
    ]
    base_acc = compute_top1_accuracy(classifier_matches, references)
    assert base_acc["accuracy"] == 0.5  # 1/2 correct
    assert base_acc["correct"] == 1
    assert base_acc["aligned"] == 2

    # 2. Viterbi decoding
    decoded = decode_trip(trip, candidates, config)

    # Matches must be [0, 1]
    match_arcs = [m.arc_id for m in decoded.matches]
    assert match_arcs == [0, 1]
    assert all(m.unmatched_reason is None for m in decoded.matches)

    # Route parts: exactly one part containing [0, 1]
    assert len(decoded.route_parts) == 1
    assert decoded.route_parts[0].start_obs_idx == 0
    assert decoded.route_parts[0].end_obs_idx == 1
    assert decoded.route_parts[0].arc_ids == [0, 1]
    assert decoded.route_parts[0].break_before is None

    # Accuracy must improve to 2/2 (1.0)
    dec_acc = compute_top1_accuracy(decoded.matches, references)
    assert dec_acc["accuracy"] == 1.0
    assert dec_acc["correct"] == 2

    # Verify through comparator
    comp = compare_classifier_vs_viterbi(
        trip=trip,
        scored_candidates=candidates,
        classifier_matches=classifier_matches,
        references=references,
        reference_route=[0, 1],
        config=config,
    )
    assert comp["baseline"]["top1_accuracy"] == 0.5
    assert comp["viterbi"]["top1_accuracy"] == 1.0
    assert comp["delta"]["accuracy_improvement"] == 0.5


def test_scenario_2_intermediate_connecting_arc():
    """Scenario 2: Intermediate connecting arc.

    Construct a 3-arc chain: arc 0 (0->1), arc 1 (1->2), arc 2 (2->3).
    Observations match on arc 0 and arc 2.
    Expect route_parts to contain [0, 1, 2] and exactly 2 observation matches.
    """
    nodes = [
        RoadNode(node_id=0, lon=0.0, lat=0.0),
        RoadNode(node_id=1, lon=0.001, lat=0.0),
        RoadNode(node_id=2, lon=0.002, lat=0.0),
        RoadNode(node_id=3, lon=0.003, lat=0.0),
    ]
    arcs = [
        RoadArc(arc_id=0, from_node_id=0, to_node_id=1, length_m=100.0),
        RoadArc(arc_id=1, from_node_id=1, to_node_id=2, length_m=100.0),
        RoadArc(arc_id=2, from_node_id=2, to_node_id=3, length_m=100.0),
    ]
    observations = [
        Observation(obs_idx=0, lon=0.0005, lat=0.0, time_s=1.0),
        Observation(obs_idx=1, lon=0.0025, lat=0.0, time_s=10.0),
    ]
    trip = Trip(trip_id="00000002", observations=observations, nodes=nodes, arcs=arcs)

    candidates = [
        ScoredCandidate(
            trip_id="00000002",
            obs_idx=0,
            arc_id=0,
            along_arc_fraction=0.5,
            distance_m=0.0,
            emission_prob=1.0,
        ),
        ScoredCandidate(
            trip_id="00000002",
            obs_idx=1,
            arc_id=2,
            along_arc_fraction=0.5,
            distance_m=0.0,
            emission_prob=1.0,
        ),
    ]

    config = DecoderConfig(use_equal_transition_weights=True)
    decoded = decode_trip(trip, candidates, config)

    # Must produce 2 matches
    assert len(decoded.matches) == 2
    assert decoded.matches[0].obs_idx == 0 and decoded.matches[0].arc_id == 0
    assert decoded.matches[1].obs_idx == 1 and decoded.matches[1].arc_id == 2

    # Route parts must contain 3 arcs: [0, 1, 2]
    assert len(decoded.route_parts) == 1
    assert decoded.route_parts[0].arc_ids == [0, 1, 2]
    assert decoded.route_parts[0].start_obs_idx == 0
    assert decoded.route_parts[0].end_obs_idx == 1


def test_scenario_3_gps_gap():
    """Scenario 3: GPS gap split.

    Set the second timestamp to 121 seconds with a 30-second split threshold.
    Expect two route parts, second marked break_before = 'gps_gap'.
    Independent matches across the break are [0, 3].
    """
    trip = make_two_road_fixture_trip()
    # Change obs 1 timestamp to 121.0 (delta = 120s > 30s)
    trip.observations[1].time_s = 121.0

    candidates = [
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=0,
            arc_id=0,
            along_arc_fraction=0.25,
            distance_m=0.0,
            emission_prob=0.9,
        ),
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=0,
            arc_id=2,
            along_arc_fraction=0.25,
            distance_m=11.1,
            emission_prob=0.1,
        ),
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=1,
            arc_id=1,
            along_arc_fraction=0.75,
            distance_m=0.0,
            emission_prob=0.2,
        ),
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=1,
            arc_id=3,
            along_arc_fraction=0.75,
            distance_m=11.1,
            emission_prob=0.8,
        ),
    ]

    config = DecoderConfig(gps_gap_threshold_s=30.0, use_equal_transition_weights=True)
    decoded = decode_trip(trip, candidates, config)

    # Independent matches across the gap: [0, 3]
    assert [m.arc_id for m in decoded.matches] == [0, 3]

    # Two route parts: second part marked break_before = 'gps_gap'
    assert len(decoded.route_parts) == 2
    assert decoded.route_parts[0].break_before is None
    assert decoded.route_parts[0].arc_ids == [0]
    assert decoded.route_parts[0].start_obs_idx == 0
    assert decoded.route_parts[0].end_obs_idx == 0

    assert decoded.route_parts[1].break_before == "gps_gap"
    assert decoded.route_parts[1].arc_ids == [3]
    assert decoded.route_parts[1].start_obs_idx == 1
    assert decoded.route_parts[1].end_obs_idx == 1


def test_scenario_4_invalid_transition():
    """Scenario 4: Invalid road transition.

    Only candidate 0 at obs 0 and candidate 3 at obs 1. Emissions = 1.0.
    Expect break marked 'invalid_transition' and no invented connector.
    Both original observations preserved with matches [0, 3].
    """
    trip = make_two_road_fixture_trip()

    candidates = [
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=0,
            arc_id=0,
            along_arc_fraction=0.25,
            distance_m=0.0,
            emission_prob=1.0,
        ),
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=1,
            arc_id=3,
            along_arc_fraction=0.75,
            distance_m=11.1,
            emission_prob=1.0,
        ),
    ]

    config = DecoderConfig(use_equal_transition_weights=True)
    decoded = decode_trip(trip, candidates, config)

    # Both observations matched independently
    assert [m.arc_id for m in decoded.matches] == [0, 3]
    assert len(decoded.matches) == 2

    # Two route parts, second has break_before = 'invalid_transition'
    assert len(decoded.route_parts) == 2
    assert decoded.route_parts[0].break_before is None
    assert decoded.route_parts[0].arc_ids == [0]

    assert decoded.route_parts[1].break_before == "invalid_transition"
    assert decoded.route_parts[1].arc_ids == [3]


def test_scenario_5_empty_candidates_and_reverse_arc():
    """Scenario 5: Empty candidate set and reverse arc rejection.

    1. Remove all candidates for observation 1:
       Expect obs 1 match arc_id=None, unmatched_reason='no_candidates'.
    2. All candidate sets empty:
       Expect route_parts to be empty list.
    3. Test that a reverse-only arc cannot satisfy a forward transition.
    """
    trip = make_two_road_fixture_trip()

    # Part 1: only obs 0 has candidate 0
    cands_single = [
        ScoredCandidate(
            trip_id="00000001",
            obs_idx=0,
            arc_id=0,
            along_arc_fraction=0.25,
            distance_m=0.0,
            emission_prob=1.0,
        ),
    ]

    config = DecoderConfig(use_equal_transition_weights=True)
    decoded = decode_trip(trip, cands_single, config)

    assert len(decoded.matches) == 2
    assert decoded.matches[0].obs_idx == 0
    assert decoded.matches[0].arc_id == 0
    assert decoded.matches[0].unmatched_reason is None

    assert decoded.matches[1].obs_idx == 1
    assert decoded.matches[1].arc_id is None
    assert decoded.matches[1].unmatched_reason == "no_candidates"

    # Part 2: All candidate sets empty
    decoded_empty = decode_trip(trip, [], config)
    assert len(decoded_empty.matches) == 2
    assert all(m.arc_id is None for m in decoded_empty.matches)
    assert all(m.unmatched_reason == "no_candidates" for m in decoded_empty.matches)
    assert decoded_empty.route_parts == []

    # Part 3: Reverse-only arc rejection
    # Create an arc from node 1 -> 0 (reverse direction of 0 -> 1)
    rev_nodes = [
        RoadNode(node_id=0, lon=0.0, lat=0.0),
        RoadNode(node_id=1, lon=0.001, lat=0.0),
    ]
    # One-way eastbound: 0 -> 1 (arc 0). One-way westbound: 1 -> 0 (arc 1).
    rev_arcs = [
        RoadArc(arc_id=0, from_node_id=0, to_node_id=1, length_m=100.0),
        RoadArc(arc_id=1, from_node_id=1, to_node_id=0, length_m=100.0),
    ]
    graph = RoadNetworkGraph(rev_nodes, rev_arcs)

    # Attempt transition backwards along one-way eastbound arc 0: frac 0.8 -> 0.2
    dist_back, _ = graph.get_candidate_distance_and_path(
        from_arc_id=0, from_frac=0.8, to_arc_id=0, to_frac=0.2
    )
    # A cycle 0->1 and 1->0 exists, so total network distance is remaining on arc 0 (0.2*100)
    # + arc 1 (100) + to_frac on arc 0 (0.2*100) = 140m.
    assert dist_back == pytest.approx(140.0)

    # Now test with NO reverse arc available (strictly one-way 0 -> 1):
    strict_oneway_arcs = [RoadArc(arc_id=0, from_node_id=0, to_node_id=1, length_m=100.0)]
    graph_oneway = RoadNetworkGraph(rev_nodes, strict_oneway_arcs)
    dist_impossible, _ = graph_oneway.get_candidate_distance_and_path(
        from_arc_id=0, from_frac=0.8, to_arc_id=0, to_frac=0.2
    )
    # Backward movement on strictly one-way road without reverse path is impossible
    assert dist_impossible is None
