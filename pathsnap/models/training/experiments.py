"""Shared training / evaluation harness: grouped CV, held-out evaluation, logging.

All models use identical trip-grouped folds, identical preprocessing code (fitted per training
fold) and identical scoring rules. Data tables (separate inputs, per the contract):
    candidates   : Candidate columns                  (inference input)
    observations : trip_id, obs_idx, lon, lat, time_s (inference input)
    references   : Reference columns                  (development labels only)

Inference-time boundary: candidate table in -> ScoredCandidate + ClassifierMatch out
(feature construction + model scoring + emission normalisation + Top-1 selection). Candidate
generation, file I/O and reference reading are excluded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..evaluation.emissions import classifier_matches, make_scored
from ..evaluation.metrics import evaluate
from ..features import feature_matrix
from ..schema import RANDOM_SEED
from .folds import trip_folds
from .labels import label_candidates
from .models import MODEL_NAMES, default_params, make_pipeline
from .scorer import CandidateScorer


class Data:
    """Bundle of the three tables with a cached, label-free feature matrix."""

    def __init__(self, candidates: pd.DataFrame, observations: pd.DataFrame, references: pd.DataFrame):
        self.candidates = candidates.sort_values(["trip_id", "obs_idx", "arc_id"],
                                                 kind="mergesort").reset_index(drop=True)
        self.observations = observations
        self.references = references
        self._X = None
        self._y = None

    @property
    def X(self) -> np.ndarray:
        if self._X is None:  # derived features use one observation's candidates only -> no leakage
            self._X = feature_matrix(self.candidates)
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Label per candidate row; -1 for candidates of non-aligned observations (never trained on)."""
        if self._y is None:
            keys = ["trip_id", "obs_idx", "arc_id"]
            lab = label_candidates(self.candidates, self.references)[keys + ["label"]]
            m = self.candidates[keys].merge(lab, how="left", on=keys)
            self._y = m["label"].fillna(-1).astype("int8").to_numpy()
        return self._y

    def trips(self) -> list[str]:
        return sorted(self.observations["trip_id"].astype(str).unique())

    def subset(self, trips) -> "Data":
        t = set(trips)
        d = Data(self.candidates[self.candidates["trip_id"].isin(t)],
                 self.observations[self.observations["trip_id"].isin(t)],
                 self.references[self.references["trip_id"].isin(t)])
        if self._X is not None:  # candidates are sorted identically, so a boolean slice is aligned
            d._X = self._X[self.candidates["trip_id"].isin(t).to_numpy()]
        return d


def _score_with(pipe, data: Data):
    t0 = time.perf_counter()
    X = data.X
    p = pipe.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
    scored = make_scored(data.candidates, p)
    matches = classifier_matches(scored, data.observations)
    return scored, matches, time.perf_counter() - t0


def cross_validate(name: str, data: Data, params: dict | None = None, n_splits: int = 5,
                   seed: int = RANDOM_SEED) -> dict:
    """Trip-grouped CV on the training partition. Preprocessing is fitted per training fold."""
    _ = data.X  # materialise once so subset() reuses it
    folds = trip_folds(data.trips(), n_splits, seed)
    trip_arr = data.candidates["trip_id"].to_numpy()
    y_all = data.y
    rows = []
    for k, (tr, va) in enumerate(folds):
        assert not set(tr) & set(va), "a trip appeared on both sides of a split"
        tr_mask = np.isin(trip_arr, tr) & (y_all >= 0)
        pipe = make_pipeline(name, params, seed)
        t0 = time.perf_counter()
        pipe.fit(data.X[tr_mask], y_all[tr_mask])
        fit_s = time.perf_counter() - t0
        vd = data.subset(va)
        scored, matches, inf_s = _score_with(pipe, vd)
        m = evaluate(vd.references, matches, vd.candidates, scored)
        rows.append({"fold": k, "n_train_trips": len(tr), "n_val_trips": len(va),
                     "train_seconds": fit_s, "inference_seconds": inf_s, **m})
    f1 = [r["macro_f1"] for r in rows if r["macro_f1"] is not None]
    acc = [r["top1_accuracy"] for r in rows if r["top1_accuracy"] is not None]
    return {"model": name, "params": {**default_params(name, seed), **(params or {})}, "seed": seed,
            "n_splits": n_splits, "folds": rows,
            "cv_macro_f1_mean": float(np.mean(f1)) if f1 else None,
            "cv_macro_f1_std": float(np.std(f1)) if f1 else None,
            "cv_top1_mean": float(np.mean(acc)) if acc else None}


def fit_scorer(name: str, train: Data, params: dict | None = None, seed: int = RANDOM_SEED) -> CandidateScorer:
    mask = train.y >= 0
    lab = train.candidates[mask].copy()
    lab["label"] = train.y[mask]
    return CandidateScorer.fit(make_pipeline(name, params, seed), lab, name,
                               {**default_params(name, seed), **(params or {})}, seed)


def evaluate_scorer(scorer: CandidateScorer, test: Data,
                    continuity_fn: Callable[[pd.DataFrame], dict] | None = None, repeats: int = 3) -> dict:
    """Held-out evaluation. The reference table is read only after prediction."""
    times = []
    scored = matches = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        scored, matches = scorer.score(test.candidates, test.observations)
        times.append(time.perf_counter() - t0)
    m = evaluate(test.references, matches, test.candidates, scored)
    m["inference_seconds_median"] = float(np.median(times))
    m["n_candidate_rows"] = int(len(test.candidates))
    m["continuity"] = continuity_fn(matches) if continuity_fn else None
    return {"metrics": m, "matches": matches, "scored": scored}


def summary_row(name: str, seed: int, cv: dict | None, scorer: CandidateScorer, m: dict) -> dict:
    return {
        "model": name, "seed": seed,
        "cv_macro_f1_mean": cv and cv["cv_macro_f1_mean"], "cv_macro_f1_std": cv and cv["cv_macro_f1_std"],
        "cv_top1_mean": cv and cv["cv_top1_mean"],
        "test_top1": m["top1_accuracy"], "test_macro_f1": m["macro_f1"],
        "test_macro_precision": m["macro_precision"], "test_macro_recall": m["macro_recall"],
        "test_candidate_miss_rate": m.get("candidate_miss_rate"),
        "test_log_loss": m.get("log_loss"), "test_log_loss_n": m.get("log_loss_n"),
        "test_log_loss_coverage": m.get("log_loss_coverage"),
        "n_aligned": m["n_aligned"], "n_ambiguous": m["n_ambiguous"], "n_unaligned": m["n_unaligned"],
        "train_seconds": scorer.extra["train_seconds"], "inference_seconds": m["inference_seconds_median"],
        "continuity": json.dumps(m["continuity"]) if m["continuity"] else None,
    }


def run_comparison(train: Data, test: Data, outdir: str | Path, models=MODEL_NAMES, n_splits: int = 5,
                   seed: int = RANDOM_SEED, param_overrides: dict | None = None,
                   continuity_fn=None) -> pd.DataFrame:
    """CV on train, refit on train, evaluate on test for every model; write logs + predictions."""
    out = Path(outdir)
    for sub in ("predictions", "cv", "models"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for name in models:
        params = (param_overrides or {}).get(name)
        cv = cross_validate(name, train, params, n_splits, seed)
        (out / "cv" / f"{name}.json").write_text(json.dumps(cv, indent=2, default=str))
        scorer = fit_scorer(name, train, params, seed)
        ev = evaluate_scorer(scorer, test, continuity_fn)
        ev["matches"].to_csv(out / "predictions" / f"{name}_classifier_matches.csv", index=False)
        (out / "cv" / f"{name}_test_metrics.json").write_text(json.dumps(ev["metrics"], indent=2, default=str))
        rows.append(summary_row(name, seed, cv, scorer, ev["metrics"]))
        scorer.save(out / "models" / f"{name}.joblib")
    df = pd.DataFrame(rows)
    df.to_csv(out / "model_comparison.csv", index=False)
    return df
