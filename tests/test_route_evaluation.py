"""Unit tests for the route evaluation and baseline comparator metrics."""

import pytest
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.evaluation.comparator import (
    compare_classifier_vs_viterbi,
    evaluate_decoded_trip,
)
from pathsnap.decoder.evaluation.metrics import (
    compute_candidate_miss_rate,
    compute_macro_f1,
    compute_route_continuity,
    compute_sequence_edit_distance,
    compute_top1_accuracy,
)
from pathsnap.decoder.models import (
    ClassifierMatch,
    DecodedTrip,
    Match,
    Observation,
    Reference,
    RoadArc,
    RoadNode,
    RoutePart,
    ScoredCandidate,
    Trip,
)
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph


def test_top1_accuracy_aligned_only():
    """Verify that only aligned observations contribute to Top-1 accuracy."""
    references = [
        Reference(trip_id="T1", obs_idx=0, ref_arc_id=10, alignment_status="aligned"),
        Reference(trip_id="T1", obs_idx=1, ref_arc_id=20, alignment_status="aligned"),
        Reference(trip_id="T1", obs_idx=2, ref_arc_id=None, alignment_status="ambiguous"),
        Reference(trip_id="T1", obs_idx=3, ref_arc_id=None, alignment_status="unaligned"),
    ]

    # Obs 0 correct (10), Obs 1 wrong (99), Obs 2 predicted (30 - ignored), Obs 3 null
    matches = [
        Match(obs_idx=0, arc_id=10),
        Match(obs_idx=1, arc_id=99),
        Match(obs_idx=2, arc_id=30),
        Match(obs_idx=3, arc_id=None, unmatched_reason="no_candidates"),
    ]

    res = compute_top1_accuracy(matches, references)
    assert res["aligned"] == 2
    assert res["ambiguous"] == 1
    assert res["unaligned"] == 1
    assert res["correct"] == 1
    assert res["accuracy"] == pytest.approx(0.5)


def test_top1_accuracy_unresolved_as_incorrect():
    """Verify that an aligned observation with an unresolved/null match counts as incorrect."""
    references = [
        Reference(trip_id="T1", obs_idx=0, ref_arc_id=10, alignment_status="aligned"),
        Reference(trip_id="T1", obs_idx=1, ref_arc_id=20, alignment_status="aligned"),
    ]
    matches = [
        Match(obs_idx=0, arc_id=10),
        Match(obs_idx=1, arc_id=None, unmatched_reason="no_candidates"),
    ]
    res = compute_top1_accuracy(matches, references)
    assert res["aligned"] == 2
    assert res["correct"] == 1
    assert res["accuracy"] == pytest.approx(0.5)


def test_top1_accuracy_zero_aligned():
    """Verify that 0 aligned observations returns accuracy=None without ZeroDivisionError."""
    references = [
        Reference(trip_id="T1", obs_idx=0, ref_arc_id=None, alignment_status="unaligned"),
    ]
    matches = [Match(obs_idx=0, arc_id=5)]
    res = compute_top1_accuracy(matches, references)
    assert res["aligned"] == 0
    assert res["accuracy"] is None


def test_candidate_miss_rate():
    """Verify calculation of candidate miss rate against aligned references."""
    references = [
        Reference(trip_id="T1", obs_idx=0, ref_arc_id=10, alignment_status="aligned"),
        Reference(trip_id="T1", obs_idx=1, ref_arc_id=20, alignment_status="aligned"),
        Reference(trip_id="T1", obs_idx=2, ref_arc_id=30, alignment_status="aligned"),
        Reference(trip_id="T1", obs_idx=3, ref_arc_id=None, alignment_status="ambiguous"),
    ]
    # Obs 0 has arc 10 (hit)
    # Obs 1 has arcs [21, 22] (miss!)
    # Obs 2 has no candidates (miss!)
    candidates = [
        {"trip_id": "T1", "obs_idx": 0, "arc_id": 10},
        {"trip_id": "T1", "obs_idx": 1, "arc_id": 21},
        {"trip_id": "T1", "obs_idx": 1, "arc_id": 22},
    ]

    miss_rate = compute_candidate_miss_rate(candidates, references)
    # Aligned: 3. Misses: 2 (obs 1 and obs 2). Rate: 2/3.
    assert miss_rate == pytest.approx(2.0 / 3.0)


def test_macro_f1():
    """Verify Macro F1 calculation over trip-qualified arc IDs."""
    references = [
        Reference(trip_id="T1", obs_idx=0, ref_arc_id=1, alignment_status="aligned"),
        Reference(trip_id="T1", obs_idx=1, ref_arc_id=1, alignment_status="aligned"),
        Reference(trip_id="T1", obs_idx=2, ref_arc_id=2, alignment_status="aligned"),
    ]
    # Matches: obs 0 -> 1 (TP for 1), obs 1 -> 2 (FP for 2, FN for 1), obs 2 -> 2 (TP for 2)
    matches = [
        Match(obs_idx=0, arc_id=1),
        Match(obs_idx=1, arc_id=2),
        Match(obs_idx=2, arc_id=2),
    ]
    # For class 1: TP=1, FP=0, FN=1 -> F1 = 2*1 / (2*1 + 0 + 1) = 2/3
    # For class 2: TP=1, FP=1, FN=0 -> F1 = 2*1 / (2*1 + 1 + 0) = 2/3
    # Macro F1 = (2/3 + 2/3) / 2 = 2/3
    macro_f1 = compute_macro_f1(matches, references, trip_id="T1")
    assert macro_f1 == pytest.approx(2.0 / 3.0)


def test_route_continuity_metric():
    """Verify route continuity metric calculations on road network."""
    nodes = [
        RoadNode(node_id=0, lon=0.0, lat=0.0),
        RoadNode(node_id=1, lon=0.001, lat=0.0),
        RoadNode(node_id=2, lon=0.002, lat=0.0),
        RoadNode(node_id=3, lon=0.01, lat=0.01),
        RoadNode(node_id=4, lon=0.02, lat=0.02),
    ]
    arcs = [
        RoadArc(arc_id=0, from_node_id=0, to_node_id=1, length_m=100.0),
        RoadArc(arc_id=1, from_node_id=1, to_node_id=2, length_m=100.0),
        RoadArc(arc_id=2, from_node_id=3, to_node_id=4, length_m=100.0),
    ]
    graph = RoadNetworkGraph(nodes, arcs)

    observations = [
        Observation(obs_idx=0, lon=0.0, lat=0.0, time_s=0.0),
        Observation(obs_idx=1, lon=0.001, lat=0.0, time_s=5.0),
        Observation(obs_idx=2, lon=0.002, lat=0.0, time_s=10.0),
        Observation(obs_idx=3, lon=0.01, lat=0.01, time_s=100.0),  # Gap: 90s > 30s
        Observation(obs_idx=4, lon=0.02, lat=0.02, time_s=105.0),
    ]

    matches = [
        Match(obs_idx=0, arc_id=0),
        Match(obs_idx=1, arc_id=1),     # Continuous from 0 (0 -> 1)
        Match(obs_idx=2, arc_id=None),  # Unresolved point
        Match(obs_idx=3, arc_id=2),     # Interval 2->3 has unresolved point; interval 1->2 has unresolved
        Match(obs_idx=4, arc_id=2),     # Interval 3->4: dt=5s, same arc 2 -> continuous!
    ]

    config = DecoderConfig(gps_gap_threshold_s=30.0)
    cont = compute_route_continuity(matches, observations, graph, config)

    # Intervals:
    # 0->1: eligible, continuous (0 to 1 connected)
    # 1->2: unresolved (obs 2 has arc_id=None)
    # 2->3: unresolved (obs 2 has arc_id=None)
    # 3->4: eligible, continuous (arc 2 to arc 2)
    assert cont["eligible_transitions"] == 2
    assert cont["continuous_transitions"] == 2
    assert cont["breaks"] == 0
    assert cont["unresolved_points"] == 2
    assert cont["continuity_rate"] == pytest.approx(1.0)


def test_sequence_edit_distance():
    """Verify Levenshtein edit distance between decoded route parts and reference route."""
    route_parts = [
        RoutePart(start_obs_idx=0, end_obs_idx=2, arc_ids=[0, 1, 2]),
        RoutePart(start_obs_idx=3, end_obs_idx=4, arc_ids=[5, 6], break_before="gps_gap"),
    ]
    # Reference route: [0, 1, 2, 4, 5, 6] (one insertion needed: arc 4)
    ref_route = [0, 1, 2, 4, 5, 6]

    res = compute_sequence_edit_distance(route_parts, ref_route)
    assert res["predicted_length"] == 5
    assert res["reference_length"] == 6
    assert res["raw_edit_distance"] == 1
    assert res["normalized_edit_distance"] == pytest.approx(1.0 / 6.0)


def test_evaluate_decoded_trip():
    """Verify server-side evaluate_decoded_trip output format."""
    decoded = DecodedTrip(
        trip_id="00000001",
        matches=[
            Match(obs_idx=0, arc_id=0),
            Match(obs_idx=1, arc_id=1),
        ],
        route_parts=[
            RoutePart(start_obs_idx=0, end_obs_idx=1, arc_ids=[0, 1], break_before=None)
        ],
    )
    references = [
        Reference(trip_id="00000001", obs_idx=0, ref_arc_id=0, alignment_status="aligned"),
        Reference(trip_id="00000001", obs_idx=1, ref_arc_id=1, alignment_status="aligned"),
    ]

    res = evaluate_decoded_trip(decoded, references)
    assert res["trip_id"] == "00000001"
    assert res["accuracy"] == 1.0
    assert res["correct"] == 2
    assert res["aligned"] == 2
    assert res["matched"] == 2
    assert res["route_breaks"] == 0
