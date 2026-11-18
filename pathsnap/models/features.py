"""Candidate feature matrix construction (inference-safe: uses only candidate rows)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BASE_FEATURES, FEATURE_ORDER, OBS_KEY


def add_context_features(cands: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `cands` with the within-observation context features appended."""
    out = cands.copy()
    if out.empty:
        for c in ("distance_rank", "distance_gap_to_min_m", "n_candidates"):
            out[c] = pd.Series(dtype="float64")
        return out
    g = out.groupby(OBS_KEY, sort=False)["distance_m"]
    out["distance_rank"] = g.rank(method="first").astype("float64") - 1.0
    out["distance_gap_to_min_m"] = out["distance_m"] - g.transform("min")
    out["n_candidates"] = g.transform("size").astype("float64")
    return out


def feature_matrix(cands: pd.DataFrame, feature_order: list[str] = FEATURE_ORDER) -> np.ndarray:
    """Float matrix with columns in `feature_order`; nulls become NaN (imputed inside the pipeline)."""
    missing = [c for c in BASE_FEATURES if c not in cands.columns]
    if missing:
        raise KeyError(f"candidate table lacks columns {missing}")
    full = add_context_features(cands)
    return full[feature_order].astype("float64").to_numpy()
