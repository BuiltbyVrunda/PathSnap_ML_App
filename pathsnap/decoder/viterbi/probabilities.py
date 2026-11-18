"""HMM emission and transition probability functions."""

import math
from typing import Optional

from pathsnap.decoder.config import DecoderConfig


def compute_log_emission(emission_prob: float, floor: float = 1e-6) -> float:
    """Compute the natural log emission probability with a safety floor.

    Formula:
        ln(max(emission_prob, floor))

    Args:
        emission_prob: Observation-normalized emission probability in [0, 1].
        floor: Minimum probability floor to avoid taking ln(0).

    Returns:
        Natural logarithm of floored emission probability.
    """
    floored = max(float(emission_prob), float(floor))
    return math.log(floored)


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute the great-circle distance between two GPS coordinates in metres.

    Uses the Haversine formula with mean Earth radius R = 6,371,000 metres.
    """
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c


def compute_log_transition(
    d_net: Optional[float],
    d_gps: float,
    delta_t: Optional[float],
    config: Optional[DecoderConfig] = None,
) -> float:
    """Compute the log transition probability between two candidate road states.

    Formula:
        If d_net is None (disconnected / unreachable) -> -infinity
        If delta_t > 0 and (d_net / delta_t) > max_speed_mps -> -infinity
        If use_equal_transition_weights is True -> 0.0 (for fixture compatibility)
        Otherwise (exponential distance difference penalty):
            ln P(c_t | c_{t-1}) = - |d_net - d_gps| / beta

    Args:
        d_net: Shortest directed road network distance between projected candidate positions in metres.
        d_gps: Great-circle / spatial distance between the two GPS observations in metres.
        delta_t: Elapsed time in seconds between consecutive observations.
        config: Decoder configuration containing thresholds, beta, and max speed.

    Returns:
        Log transition probability (float). Returns float("-inf") for impossible transitions.
    """
    if config is None:
        config = DecoderConfig()

    # Disconnected or unreachable road segments
    if d_net is None:
        return float("-inf")

    # Plausible physical speed check
    if delta_t is not None and delta_t > 0:
        speed = d_net / delta_t
        if speed > config.max_speed_mps:
            return float("-inf")

    # Equal transition weights mode (mandated by test fixtures / standard graph reachability)
    if config.use_equal_transition_weights:
        return 0.0

    # Exponential decay based on discrepancy between network distance and Euclidean/GPS displacement
    dist_diff = abs(d_net - d_gps)
    beta = max(1e-3, config.transition_beta)
    return -dist_diff / beta
