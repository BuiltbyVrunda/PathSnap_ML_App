"""Model registry: five distinct classifiers behind one preprocessing front-end.

Preprocessing (fitted inside each training fold, never on validation/test rows):
median imputation with missing-indicator columns for the nullable features, then standard
scaling for the scale-sensitive logistic regression only.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ..schema import RANDOM_SEED

MODEL_NAMES = ["logistic_regression", "gaussian_nb", "decision_tree", "random_forest", "hist_gradient_boosting"]
EXTRA_MODEL_NAMES = ["lightgbm_ranker", "catboost"]
ALL_MODEL_NAMES = MODEL_NAMES + EXTRA_MODEL_NAMES
RANKERS = {"lightgbm_ranker"}  # trained on observation groups; scores -> softmax emissions


def default_params(name: str, seed: int = RANDOM_SEED) -> dict:
    return {
        "logistic_regression": dict(C=1.0, max_iter=1000, class_weight=None, random_state=seed),
        "gaussian_nb": dict(var_smoothing=1e-9),
        "decision_tree": dict(max_depth=8, min_samples_leaf=20, random_state=seed),
        "random_forest": dict(n_estimators=150, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                       l2_regularization=0.0, random_state=seed),
        "lightgbm_ranker": dict(n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=20,
                                reg_lambda=0.0, random_state=seed),
        "catboost": dict(iterations=300, depth=6, learning_rate=0.1, l2_leaf_reg=3.0, random_seed=seed),
    }[name]


class LGBMRankerModel(BaseEstimator):
    """LightGBM LambdaMART ranker. One query group = one GPS observation's candidate set.

    Relevance label = 1 for the reference arc, 0 otherwise. `score_kind == "ranker"` tells the scorer to
    use raw scores and softmax them within each observation. No early stopping: n_estimators is tuned by the
    grouped-CV Bayesian search instead of an inner validation split.
    """

    score_kind = "ranker"

    def __init__(self, n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=20,
                 reg_lambda=0.0, random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, X, y, group=None):
        from lightgbm import LGBMRanker
        if group is None:
            raise ValueError("LGBMRankerModel needs group sizes (candidates per observation)")
        self.model_ = LGBMRanker(objective="lambdarank", n_estimators=self.n_estimators,
                                 learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                                 min_child_samples=self.min_child_samples, reg_lambda=self.reg_lambda,
                                 random_state=self.random_state, n_jobs=-1, verbose=-1)
        self.model_.fit(X, y, group=group)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def positive_scores(pipe, X):
    """Per-candidate score: positive-class probability, or raw ranking score for rankers."""
    clf = pipe.steps[-1][1]
    if getattr(clf, "score_kind", None) == "ranker":
        return pipe.predict(X)
    return pipe.predict_proba(X)[:, 1]


def emission_mode(name: str) -> str:
    return "softmax" if name in RANKERS else "proportional"


def fit_kwargs(name: str, cands) -> dict:
    """Extra fit arguments: query-group sizes (contiguous rows of one observation) for rankers."""
    if name not in RANKERS:
        return {}
    codes = cands.groupby(["trip_id", "obs_idx"], sort=False).ngroup().to_numpy()
    return {"clf__group": np.bincount(codes)}


def make_pipeline(name: str, params: dict | None = None, seed: int = RANDOM_SEED) -> Pipeline:
    p = {**default_params(name, seed), **(params or {})}
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if name == "logistic_regression":
        steps += [("scale", StandardScaler()), ("clf", LogisticRegression(**p))]
    elif name == "gaussian_nb":
        steps += [("clf", GaussianNB(**p))]
    elif name == "decision_tree":
        steps += [("clf", DecisionTreeClassifier(**p))]
    elif name == "random_forest":
        steps += [("clf", RandomForestClassifier(**p))]
    elif name == "hist_gradient_boosting":
        steps += [("clf", HistGradientBoostingClassifier(**p))]
    elif name == "lightgbm_ranker":
        steps += [("clf", LGBMRankerModel(**p))]
    elif name == "catboost":
        from catboost import CatBoostClassifier
        steps += [("clf", CatBoostClassifier(loss_function="Logloss", verbose=0, thread_count=-1,
                                          allow_writing_files=False, **p))]
    else:
        raise KeyError(name)
    return Pipeline(steps)
