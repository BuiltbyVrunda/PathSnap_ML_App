"""Bayesian hyper-parameter search (Optuna TPE) using only trip-grouped CV folds on `train`.

Objective: mean macro F1 over the grouped folds (higher is better), identical to the CV metric in the
model comparison. The held-out student_test and reserved evaluator trips are never passed in here.
"""
from __future__ import annotations

import json
from pathlib import Path

import optuna

from ..schema import RANDOM_SEED
from ..training.experiments import Data, cross_validate

SEARCH_SPACES = {
    "hist_gradient_boosting": {
        "learning_rate": ("float_log", 0.02, 0.3),
        "max_iter": ("int", 80, 400),
        "max_leaf_nodes": ("int", 8, 96),
        "min_samples_leaf": ("int", 10, 200),
        "l2_regularization": ("float_log", 1e-6, 10.0),
    },
    "lightgbm_ranker": {
        "n_estimators": ("int", 80, 500),
        "learning_rate": ("float_log", 0.02, 0.3),
        "num_leaves": ("int", 8, 96),
        "min_child_samples": ("int", 10, 200),
        "reg_lambda": ("float_log", 1e-6, 10.0),
    },
    "catboost": {
        "iterations": ("int", 150, 600),
        "depth": ("int", 4, 9),
        "learning_rate": ("float_log", 0.03, 0.3),
        "l2_leaf_reg": ("float_log", 1.0, 20.0),
    },
    "random_forest": {
        "n_estimators": ("int", 60, 300),
        "max_depth": ("int", 6, 24),
        "min_samples_leaf": ("int", 1, 40),
        "max_features": ("float", 0.3, 1.0),
    },
}


def _suggest(trial: optuna.Trial, space: dict) -> dict:
    out = {}
    for k, (kind, lo, hi) in space.items():
        if kind == "int":
            out[k] = trial.suggest_int(k, lo, hi)
        elif kind == "float":
            out[k] = trial.suggest_float(k, lo, hi)
        else:
            out[k] = trial.suggest_float(k, lo, hi, log=True)
    return out


def bayes_search(name: str, train: Data, n_trials: int, outdir: str | Path, n_splits: int = 5,
                 seed: int = RANDOM_SEED, space: dict | None = None) -> dict:
    space = space or SEARCH_SPACES[name]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = _suggest(trial, space)
        cv = cross_validate(name, train, params, n_splits, seed)
        trial.set_user_attr("cv_macro_f1_std", cv["cv_macro_f1_std"])
        trial.set_user_attr("cv_top1_mean", cv["cv_top1_mean"])
        trial.set_user_attr("fit_seconds", sum(f["train_seconds"] for f in cv["folds"]))
        return cv["cv_macro_f1_mean"] if cv["cv_macro_f1_mean"] is not None else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    study.trials_dataframe().to_csv(out / f"{name}_trials.csv", index=False)
    result = {"model": name, "method": "Optuna TPE (Bayesian)",
              "objective": "mean grouped-CV macro F1 on train (maximise)",
              "n_trials": n_trials, "n_splits": n_splits, "seed": seed,
              "search_space": {k: list(v) for k, v in space.items()},
              "best_value": study.best_value, "best_params": study.best_params}
    (out / f"{name}_search.json").write_text(json.dumps(result, indent=2, default=str))
    return result
