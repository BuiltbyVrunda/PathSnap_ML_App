"""Evaluation and baseline comparison subpackage for route decoding."""

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

__all__ = [
    "compare_classifier_vs_viterbi",
    "evaluate_decoded_trip",
    "compute_top1_accuracy",
    "compute_candidate_miss_rate",
    "compute_macro_f1",
    "compute_route_continuity",
    "compute_sequence_edit_distance",
]
