"""Shared column names, feature order and constants for the candidate scorer (contract v1)."""
from __future__ import annotations

KEY_COLS = ["trip_id", "obs_idx", "arc_id"]
OBS_KEY = ["trip_id", "obs_idx"]

# Raw candidate features exactly as delivered by the candidate generator (Candidate record).
BASE_FEATURES = [
    "along_arc_fraction",
    "distance_m",
    "heading_diff_deg",   # nullable
    "speed_mps",          # nullable
    "delta_t_s",          # nullable
]

# Context features derived *within one observation's candidate set* only (no labels, no other
# observations), so they are available at inference. Computed by features.add_context_features.
CONTEXT_FEATURES = [
    "distance_rank",            # 0 = nearest candidate of the observation
    "distance_gap_to_min_m",    # distance_m - min distance_m in the observation
    "n_candidates",             # size of the observation's candidate set
]

# Explicit, fixed feature order used by every model and stored with the saved artifact.
FEATURE_ORDER = BASE_FEATURES + CONTEXT_FEATURES

CANDIDATE_COLS = KEY_COLS + BASE_FEATURES
SCORED_COLS = CANDIDATE_COLS + ["emission_prob"]
MATCH_COLS = ["trip_id", "obs_idx", "arc_id"]
REFERENCE_COLS = ["trip_id", "obs_idx", "ref_arc_id", "alignment_status"]
OBSERVATION_COLS = ["trip_id", "obs_idx", "lon", "lat", "time_s"]

ALIGNED, AMBIGUOUS, UNALIGNED = "aligned", "ambiguous", "unaligned"
EPSILON = 1e-12  # floor used inside log loss
RANDOM_SEED = 42
