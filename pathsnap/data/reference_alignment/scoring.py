"""Measure exact-arc coverage and scores against independent reference labels."""

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from pathsnap.data.candidates import Candidate
from pathsnap.data.loading import Trip

from . import AlignmentConfig, Reference, align_references, read_reference_route


def candidate_coverage(references: Sequence[Reference], candidates: Sequence[Candidate]) -> dict:
    keys = {(c.trip_id, c.obs_idx, c.arc_id) for c in candidates}
    observed = {(c.trip_id, c.obs_idx) for c in candidates}
    counts = Counter(ref.alignment_status for ref in references)
    aligned = [ref for ref in references if ref.alignment_status == "aligned"]
    misses = sum((ref.trip_id, ref.obs_idx, ref.ref_arc_id) not in keys for ref in aligned)
    return {
        "observations": len(references),
        "aligned": len(aligned),
        "ambiguous": counts["ambiguous"],
        "unaligned": counts["unaligned"],
        "zero_candidates": sum((ref.trip_id, ref.obs_idx) not in observed for ref in references),
        "aligned_zero_candidates": sum((ref.trip_id, ref.obs_idx) not in observed for ref in aligned),
        "candidate_rows": len(candidates),
        "candidate_misses": misses,
        "candidate_miss_rate": misses / len(aligned) if aligned else None,
    }


def score_predictions(
    references: Sequence[Reference], predictions: Mapping[tuple[str, int], int | None]
) -> dict:
    """Score trip-qualified arc classes, including missing predictions as false negatives."""
    status = Counter(ref.alignment_status for ref in references)
    actual_counts = Counter()
    predicted_counts = Counter()
    true_positives = Counter()
    for ref in references:
        if ref.alignment_status != "aligned":
            continue
        actual = (ref.trip_id, ref.ref_arc_id)
        actual_counts[actual] += 1
        arc_id = predictions.get((ref.trip_id, ref.obs_idx))
        if arc_id is not None:
            predicted = (ref.trip_id, arc_id)
            predicted_counts[predicted] += 1
            if predicted == actual:
                true_positives[actual] += 1

    aligned = status["aligned"]
    correct = sum(true_positives.values())
    classes = actual_counts.keys() | predicted_counts.keys()
    macro_f1 = (
        sum(2 * true_positives[key] / (actual_counts[key] + predicted_counts[key]) for key in classes)
        / len(classes)
        if classes
        else None
    )
    return {
        "observations": len(references),
        "aligned": aligned,
        "ambiguous": status["ambiguous"],
        "unaligned": status["unaligned"],
        "correct": correct,
        "accuracy": correct / aligned if aligned else None,
        "macro_f1": macro_f1,
        "macro_f1_classes": len(classes),
    }


def score_trip(
    trip: Trip,
    reference_dir: str | Path,
    predictions: Mapping[int, int | None],
    config: AlignmentConfig = AlignmentConfig(),
) -> dict:
    """Laptop 1 entry point. Call after prediction and expose only aggregate metrics."""
    for obs_idx, arc_id in predictions.items():
        if not 0 <= obs_idx < len(trip.track) or (arc_id is not None and not 0 <= arc_id < len(trip.arc_nodes)):
            raise ValueError(f"{trip.trip_id}: prediction uses an invalid observation or arc ID")
    references = align_references(trip, read_reference_route(trip, reference_dir), config)
    return score_predictions(references, {(trip.trip_id, obs_idx): arc_id for obs_idx, arc_id in predictions.items()})
