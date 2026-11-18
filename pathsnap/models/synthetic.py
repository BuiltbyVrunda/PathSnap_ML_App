"""Synthetic candidate fixtures from the Member 2 handoff (trips 00000001-00000005)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CANDIDATE_COLS, REFERENCE_COLS

_ROWS = [
    (0, 0, 0.25, 1.0, 5.0, np.nan, np.nan),
    (0, 2, 0.25, 30.0, 100.0, np.nan, np.nan),
    (1, 1, 0.75, 2.0, 8.0, 12.0, 10.0),
    (1, 3, 0.75, 25.0, 90.0, 12.0, 10.0),
]
_COLS = ["obs_idx", "arc_id", "along_arc_fraction", "distance_m", "heading_diff_deg", "speed_mps", "delta_t_s"]


def _trip(trip_id: str, extra_obs: bool):
    c = pd.DataFrame(_ROWS, columns=_COLS)
    c.insert(0, "trip_id", trip_id)
    n_obs = 3 if extra_obs else 2
    obs = pd.DataFrame({"trip_id": trip_id, "obs_idx": range(n_obs),
                        "lon": 8.0 + 0.001 * np.arange(n_obs), "lat": 49.0,
                        "time_s": np.arange(1.0, n_obs + 1)})
    refs = pd.DataFrame({"trip_id": trip_id, "obs_idx": range(n_obs),
                         "ref_arc_id": [0, 1, 1][:n_obs], "alignment_status": "aligned"})
    return c[CANDIDATE_COLS], obs, refs[REFERENCE_COLS]


def synthetic_split(train_ids=("00000001", "00000002", "00000003", "00000004"), held_out="00000005"):
    """Return dict with train/held-out candidates, observations and references (separate tables)."""
    tr = [_trip(t, False) for t in train_ids]
    ho = _trip(held_out, True)
    cat = lambda i: pd.concat([x[i] for x in tr], ignore_index=True)  # noqa: E731
    return {
        "train": {"candidates": cat(0), "observations": cat(1), "references": cat(2)},
        "heldout": {"candidates": ho[0], "observations": ho[1], "references": ho[2]},
    }
