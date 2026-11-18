"""Supervised labels from the (separate) development reference table."""
from __future__ import annotations

import pandas as pd

from ..schema import ALIGNED, OBS_KEY


def label_candidates(cands: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Join references on (trip_id, obs_idx); keep only candidates of *aligned* observations.

    Adds integer column `label` (1 when arc_id == ref_arc_id). Ambiguous and unaligned
    observations are excluded from supervised training. Used for fitting only.
    """
    ref = references.loc[references["alignment_status"] == ALIGNED, OBS_KEY + ["ref_arc_id"]]
    j = cands.merge(ref, on=OBS_KEY, how="inner")
    j["label"] = (j["arc_id"] == j["ref_arc_id"]).astype("int8")
    return j.drop(columns=["ref_arc_id"]).sort_values(["trip_id", "obs_idx", "arc_id"],
                                                      kind="mergesort").reset_index(drop=True)
