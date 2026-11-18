"""Model registry: five distinct classifiers behind one preprocessing front-end.

Preprocessing (fitted inside each training fold, never on validation/test rows):
median imputation with missing-indicator columns for the nullable features, then standard
scaling for the scale-sensitive logistic regression only.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ..schema import RANDOM_SEED

MODEL_NAMES = ["logistic_regression", "gaussian_nb", "decision_tree", "random_forest", "hist_gradient_boosting"]


def default_params(name: str, seed: int = RANDOM_SEED) -> dict:
    return {
        "logistic_regression": dict(C=1.0, max_iter=1000, class_weight=None, random_state=seed),
        "gaussian_nb": dict(var_smoothing=1e-9),
        "decision_tree": dict(max_depth=8, min_samples_leaf=20, random_state=seed),
        "random_forest": dict(n_estimators=150, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                       l2_regularization=0.0, random_state=seed),
    }[name]


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
    else:
        raise KeyError(name)
    return Pipeline(steps)
