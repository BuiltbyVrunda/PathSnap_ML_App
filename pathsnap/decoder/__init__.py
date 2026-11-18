"""PathSnap HMM/Viterbi route decoder and evaluation package."""

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
from pathsnap.decoder.viterbi.decoder import decode_trip
from pathsnap.decoder.viterbi.graph import RoadNetworkGraph
from pathsnap.decoder.viterbi.probabilities import (
    compute_log_emission,
    compute_log_transition,
    haversine_distance_m,
)

__all__ = [
    "DecoderConfig",
    "Trip",
    "Observation",
    "RoadNode",
    "RoadArc",
    "ScoredCandidate",
    "ClassifierMatch",
    "Reference",
    "Match",
    "RoutePart",
    "DecodedTrip",
    "RoadNetworkGraph",
    "decode_trip",
    "evaluate_decoded_trip",
    "compare_classifier_vs_viterbi",
    "compute_log_emission",
    "compute_log_transition",
    "haversine_distance_m",
    "compute_top1_accuracy",
    "compute_candidate_miss_rate",
    "compute_macro_f1",
    "compute_route_continuity",
    "compute_sequence_edit_distance",
]
