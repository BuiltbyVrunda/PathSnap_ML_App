"""Fitted candidate scorer: save / load / score (inference inputs only).

Entry points
------------
    from pathsnap.models.training.scorer import CandidateScorer
    scorer = CandidateScorer.load("artifacts/models/pathsnap_scorer.joblib")
    scored, matches = scorer.score(candidates_df, observations_df)

CLI (reproducible load-and-predict):
    python -m pathsnap.models.training.scorer --model artifacts/models/pathsnap_scorer.joblib \
        --candidates cands.csv --observations obs.csv --out out_dir
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from ..evaluation.emissions import classifier_matches, make_scored
from ..features import feature_matrix
from .models import emission_mode, fit_kwargs, positive_scores
from ..schema import CANDIDATE_COLS, FEATURE_ORDER, OBS_KEY, SCORED_COLS


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, dtype={"trip_id": str})


class CandidateScorer:
    """Wraps a fitted pipeline plus the metadata needed to reproduce its scores."""

    def __init__(self, pipeline: Pipeline, model_name: str, params: dict, seed: int,
                 feature_order: list[str] | None = None, extra: dict | None = None):
        self.pipeline = pipeline
        self.model_name = model_name
        self.params = params
        self.seed = seed
        self.feature_order = list(feature_order or FEATURE_ORDER)
        self.extra = extra or {}

    # --- fitting -------------------------------------------------------------------------
    @classmethod
    def fit(cls, pipeline: Pipeline, labelled: pd.DataFrame, model_name: str, params: dict,
            seed: int, extra: dict | None = None) -> "CandidateScorer":
        X = feature_matrix(labelled)
        t0 = time.perf_counter()
        pipeline.fit(X, labelled["label"].to_numpy(), **fit_kwargs(model_name, labelled))
        extra = {**(extra or {}), "train_seconds": time.perf_counter() - t0,
                 "n_train_rows": int(len(labelled)),
                 "n_train_trips": int(labelled["trip_id"].nunique())}
        return cls(pipeline, model_name, params, seed, extra=extra)

    # --- inference -----------------------------------------------------------------------
    def positive_proba(self, candidates: pd.DataFrame) -> np.ndarray:
        if len(candidates) == 0:
            return np.zeros(0)
        X = feature_matrix(candidates, self.feature_order)
        return positive_scores(self.pipeline, X)

    def score(self, candidates: pd.DataFrame, observations: pd.DataFrame):
        """Inference only: never receives reference columns. Returns (ScoredCandidate, ClassifierMatch)."""
        leaked = {"label", "ref_arc_id", "alignment_status"} & set(candidates.columns)
        if leaked:
            raise ValueError(f"reference/label columns must not reach the scorer: {sorted(leaked)}")
        cands = candidates[CANDIDATE_COLS].sort_values(["trip_id", "obs_idx", "arc_id"], kind="mergesort")
        cands = cands.reset_index(drop=True)
        if cands.duplicated(["trip_id", "obs_idx", "arc_id"]).any():
            raise ValueError("candidate keys (trip_id, obs_idx, arc_id) must be unique")
        scored = make_scored(cands, self.positive_proba(cands), emission_mode(self.model_name))
        matches = classifier_matches(scored, observations)
        return scored[SCORED_COLS], matches

    # --- persistence ---------------------------------------------------------------------
    def metadata(self) -> dict:
        return {
            "model_name": self.model_name, "params": self.params, "seed": self.seed,
            "feature_order": self.feature_order, "extra": self.extra,
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metadata": self.metadata()}, path)
        path.with_suffix(".json").write_text(json.dumps(self.metadata(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CandidateScorer":
        blob = joblib.load(path)
        m = blob["metadata"]
        return cls(blob["pipeline"], m["model_name"], m["params"], m["seed"], m["feature_order"], m["extra"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score candidates with a saved PathSnap scorer")
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", required=True, help="csv/parquet with Candidate columns")
    ap.add_argument("--observations", required=True, help="csv/parquet with trip_id,obs_idx,...")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    scorer = CandidateScorer.load(a.model)
    scored, matches = scorer.score(_read_table(a.candidates), _read_table(a.observations))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "scored_candidates.csv", index=False)
    matches.to_csv(out / "classifier_matches.csv", index=False)
    print(f"scored {len(scored)} candidates, {len(matches)} observations "
          f"({int(matches['arc_id'].isna().sum())} null) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
