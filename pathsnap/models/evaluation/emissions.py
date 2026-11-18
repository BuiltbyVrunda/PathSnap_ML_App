"""Score -> emission convention and classifier Top-1 matches (model-to-decoder contract)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import CANDIDATE_COLS, MATCH_COLS, OBS_KEY


def normalise_emissions(cands: pd.DataFrame, probs: np.ndarray, mode: str = "proportional") -> np.ndarray:
    """Convert candidate scores to emissions that sum to 1 within each observation.

    mode="proportional" (probabilistic classifiers): divide each positive-class probability by the sum
    within its observation. mode="softmax" (rankers): softmax of the raw ranking scores within the
    observation (shifted by the observation maximum for numerical stability).

    A zero (or non-finite) sum gives uniform emissions. Emissions sum to 1 per non-empty
    candidate set. Input order is preserved.
    """
    p = np.nan_to_num(np.asarray(probs, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    if len(p) == 0:
        return p
    key = cands.groupby(OBS_KEY, sort=False).ngroup().to_numpy()
    if mode == "softmax":
        mx = np.full(key.max() + 1, -np.inf)
        np.maximum.at(mx, key, p)
        p = np.exp(p - mx[key])
    elif mode != "proportional":
        raise ValueError(f"unknown emission mode {mode!r}")
    p = np.clip(p, 0.0, None)
    sums = np.bincount(key, weights=p)
    sizes = np.bincount(key).astype("float64")
    s, n = sums[key], sizes[key]
    with np.errstate(divide="ignore", invalid="ignore"):
        em = np.where(s > 0, p / s, 1.0 / n)
    return em


def make_scored(cands: pd.DataFrame, probs: np.ndarray, mode: str = "proportional") -> pd.DataFrame:
    """ScoredCandidate table: Candidate columns unchanged + emission_prob, sorted by key."""
    scored = cands[CANDIDATE_COLS].copy()
    scored["emission_prob"] = normalise_emissions(cands, probs, mode)
    return scored.sort_values(["trip_id", "obs_idx", "arc_id"], kind="mergesort").reset_index(drop=True)


def classifier_matches(scored: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """One ClassifierMatch per observation: highest emission, smaller arc id on ties, null if empty."""
    if len(scored):
        # sorted by arc_id within observation => first row at the max emission is the smaller arc id
        s = scored.sort_values(["trip_id", "obs_idx", "arc_id"], kind="mergesort")
        mx = s.groupby(OBS_KEY, sort=False)["emission_prob"].transform("max")
        top = s[s["emission_prob"] >= mx].drop_duplicates(OBS_KEY, keep="first")[OBS_KEY + ["arc_id"]]
    else:
        top = pd.DataFrame({"trip_id": [], "obs_idx": [], "arc_id": []})
    obs = observations[OBS_KEY].drop_duplicates()
    out = obs.merge(top, on=OBS_KEY, how="left")
    out["arc_id"] = out["arc_id"].astype("Int64")
    return out.sort_values(OBS_KEY, kind="mergesort").reset_index(drop=True)[MATCH_COLS]
