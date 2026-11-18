"""Comparative evaluation harness between classifier baseline and HMM/Viterbi decoding."""

from typing import Any, Dict, List, Optional, Union

from pathsnap.decoder.config import DecoderConfig
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
    Reference,
    ScoredCandidate,
    Trip,
)
from pathsnap.decoder.viterbi.decoder import decode_trip
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph


def evaluate_decoded_trip(
    decoded_trip: Union[DecodedTrip, Dict[str, Any]],
    references: List[Union[Reference, Dict[str, Any]]],
    graph: Optional[RoadNetworkGraph] = None,
    trip: Optional[Union[Trip, Dict[str, Any]]] = None,
    reference_route: Optional[List[int]] = None,
    config: Optional[DecoderConfig] = None,
) -> Dict[str, Any]:
    """Evaluate a decoded trip against ground truth reference data.

    Returns observation-level accuracy, route breaks, and optional continuity/edit distance.
    Used for server-side evaluation on Laptop 1.
    """
    if config is None:
        config = DecoderConfig()

    trip_obj = (
        (trip if isinstance(trip, Trip) else Trip.from_dict(trip))
        if trip is not None
        else None
    )
    dec_obj = (
        decoded_trip
        if isinstance(decoded_trip, DecodedTrip)
        else DecodedTrip.from_dict(decoded_trip)
    )

    acc_stats = compute_top1_accuracy(dec_obj.matches, references)

    # Count route breaks: parts with a non-null break_before
    route_breaks = sum(1 for rp in dec_obj.route_parts if rp.break_before is not None)
    matched_count = sum(1 for m in dec_obj.matches if m.arc_id is not None)

    result: Dict[str, Any] = {
        "trip_id": dec_obj.trip_id,
        "accuracy": acc_stats["accuracy"],
        "correct": acc_stats["correct"],
        "aligned": acc_stats["aligned"],
        "ambiguous": acc_stats["ambiguous"],
        "unaligned": acc_stats["unaligned"],
        "matched": matched_count,
        "total_observations": len(dec_obj.matches),
        "route_breaks": route_breaks,
        "route_parts_count": len(dec_obj.route_parts),
    }

    # Macro F1
    result["macro_f1"] = compute_macro_f1(dec_obj.matches, references, dec_obj.trip_id)

    # Optional continuity if graph and observations are available
    if graph is not None and trip_obj is not None:
        cont_stats = compute_route_continuity(
            dec_obj.matches, trip_obj.observations, graph, config
        )
        result["continuity"] = cont_stats

    # Optional edit distance if reference route is provided
    if reference_route is not None:
        edit_stats = compute_sequence_edit_distance(dec_obj.route_parts, reference_route)
        result["edit_distance"] = edit_stats

    return result


def compare_classifier_vs_viterbi(
    trip: Union[Trip, Dict[str, Any]],
    scored_candidates: List[Union[ScoredCandidate, Dict[str, Any]]],
    classifier_matches: List[Union[ClassifierMatch, Dict[str, Any]]],
    references: List[Union[Reference, Dict[str, Any]]],
    reference_route: Optional[List[int]] = None,
    config: Optional[DecoderConfig] = None,
) -> Dict[str, Any]:
    """Run an identical, fair comparison between classifier baseline and HMM/Viterbi.

    Ensures both pipelines are evaluated against the exact same aligned reference
    denominators, continuity reachability rules, and network graphs.

    Args:
        trip: Trip object containing observations, nodes, and arcs.
        scored_candidates: Observation-normalized scored candidate records.
        classifier_matches: Top-1 greedy classifier baseline matches.
        references: Ground truth reference records for evaluation.
        reference_route: Optional ground-truth sequence of reference arc IDs.
        config: Decoder configuration parameters.

    Returns:
        Structured dictionary comparing baseline vs decoded metrics.
    """
    if config is None:
        config = DecoderConfig()

    trip_obj = trip if isinstance(trip, Trip) else Trip.from_dict(trip)

    # Build road network graph
    graph = RoadNetworkGraph(
        nodes=trip_obj.nodes,
        arcs=trip_obj.arcs,
        max_search_distance_m=config.max_network_search_m,
        same_arc_reverse_tolerance_m=config.same_arc_reverse_tolerance_m,
    )

    # 1. Run HMM/Viterbi decoding
    decoded_trip = decode_trip(trip_obj, scored_candidates, config)

    # 2. Evaluate Baseline Classifier
    base_acc = compute_top1_accuracy(classifier_matches, references)
    base_f1 = compute_macro_f1(classifier_matches, references, trip_obj.trip_id)
    base_cont = compute_route_continuity(
        classifier_matches, trip_obj.observations, graph, config
    )

    # 3. Evaluate Decoded Trip
    dec_acc = compute_top1_accuracy(decoded_trip.matches, references)
    dec_f1 = compute_macro_f1(decoded_trip.matches, references, trip_obj.trip_id)
    dec_cont = compute_route_continuity(
        decoded_trip.matches, trip_obj.observations, graph, config
    )

    # 4. Candidate Miss Rate
    miss_rate = compute_candidate_miss_rate(scored_candidates, references)

    # 5. Route breaks
    route_breaks = sum(1 for rp in decoded_trip.route_parts if rp.break_before is not None)

    comparison: Dict[str, Any] = {
        "trip_id": trip_obj.trip_id,
        "candidate_miss_rate": miss_rate,
        "aligned_count": base_acc["aligned"],
        "baseline": {
            "top1_accuracy": base_acc["accuracy"],
            "correct_count": base_acc["correct"],
            "macro_f1": base_f1,
            "continuity_rate": base_cont["continuity_rate"],
            "continuous_transitions": base_cont["continuous_transitions"],
            "eligible_transitions": base_cont["eligible_transitions"],
            "breaks": base_cont["breaks"],
        },
        "viterbi": {
            "top1_accuracy": dec_acc["accuracy"],
            "correct_count": dec_acc["correct"],
            "macro_f1": dec_f1,
            "continuity_rate": dec_cont["continuity_rate"],
            "continuous_transitions": dec_cont["continuous_transitions"],
            "eligible_transitions": dec_cont["eligible_transitions"],
            "breaks": dec_cont["breaks"],
            "route_parts_count": len(decoded_trip.route_parts),
            "route_breaks_count": route_breaks,
        },
        "delta": {
            "accuracy_improvement": (
                (dec_acc["accuracy"] - base_acc["accuracy"])
                if dec_acc["accuracy"] is not None and base_acc["accuracy"] is not None
                else None
            ),
            "macro_f1_improvement": (
                (dec_f1 - base_f1)
                if dec_f1 is not None and base_f1 is not None
                else None
            ),
            "continuity_improvement": (
                (dec_cont["continuity_rate"] - base_cont["continuity_rate"])
                if dec_cont["continuity_rate"] is not None and base_cont["continuity_rate"] is not None
                else None
            ),
        },
        "decoded_trip": decoded_trip,
    }

    if reference_route is not None:
        edit_dist = compute_sequence_edit_distance(decoded_trip.route_parts, reference_route)
        comparison["viterbi"]["sequence_edit_distance"] = edit_dist

    return comparison
