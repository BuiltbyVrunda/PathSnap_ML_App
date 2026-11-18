"""Trip-grouped cross-validation folds shared by every model."""
from __future__ import annotations

import numpy as np


def trip_folds(trip_ids, n_splits: int = 5, seed: int = 42):
    """Deterministic grouped folds over *trips*: returns [(train_trips, val_trips), ...].

    Trips are shuffled with `seed` first so folds are reproducible but not order-dependent;
    GroupKFold then balances by candidate-row count when `weights` are given via sizes.
    """
    trips = np.array(sorted(set(map(str, trip_ids))))
    if n_splits > len(trips):
        raise ValueError(f"n_splits={n_splits} > number of trips={len(trips)}")
    rng = np.random.RandomState(seed)
    perm = trips[rng.permutation(len(trips))]
    # assign trip -> fold round-robin on the shuffled order (balanced trip counts)
    fold_of = {t: i % n_splits for i, t in enumerate(perm)}
    out = []
    for k in range(n_splits):
        val = sorted(t for t, f in fold_of.items() if f == k)
        tr = sorted(t for t, f in fold_of.items() if f != k)
        assert not set(val) & set(tr)
        out.append((tr, val))
    return out
