"""Independent tests from the Member 2 handoff, run on the synthetic trips 00000001-00000005."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pathsnap.models.evaluation.emissions import classifier_matches, make_scored, normalise_emissions
from pathsnap.models.evaluation.metrics import evaluate
from pathsnap.models.schema import FEATURE_ORDER, SCORED_COLS
from pathsnap.models.synthetic import synthetic_split
from pathsnap.models.training.experiments import Data, cross_validate, fit_scorer
from pathsnap.models.training.folds import trip_folds
from pathsnap.models.training.models import ALL_MODEL_NAMES, MODEL_NAMES
from pathsnap.models.training.scorer import CandidateScorer
from pathsnap.models.tuning.bayes import bayes_search

SMALL = {
    "decision_tree": dict(min_samples_leaf=1),
    "random_forest": dict(n_estimators=10, min_samples_leaf=1, n_jobs=1),
    "hist_gradient_boosting": dict(max_iter=10, min_samples_leaf=1),
    "lightgbm_ranker": dict(n_estimators=10, min_child_samples=1, num_leaves=4),
    "catboost": dict(iterations=10, depth=2),
}


@pytest.fixture(scope="module")
def syn():
    return synthetic_split()


@pytest.fixture(scope="module")
def train(syn):
    return Data(**syn["train"])


def test_grouped_folds_never_split_a_trip(syn):
    trips = sorted(syn["train"]["observations"]["trip_id"].unique())
    folds = trip_folds(trips, n_splits=2, seed=1)
    assert len(folds) == 2
    for tr, va in folds:
        assert tr and va and not set(tr) & set(va)
    assert sorted(sum((va for _, va in folds), [])) == trips  # every trip validated exactly once


@pytest.mark.parametrize("name", ALL_MODEL_NAMES)
def test_all_five_models_train_and_emit_scores(name, train, syn):
    scorer = fit_scorer(name, train, SMALL.get(name))
    held = syn["heldout"]
    scored, matches = scorer.score(held["candidates"], held["observations"])
    assert list(scored.columns) == SCORED_COLS
    sums = scored.groupby(["trip_id", "obs_idx"])["emission_prob"].sum()
    assert np.allclose(sums, 1.0)
    assert len(matches) == 3 and pd.isna(matches.loc[2, "arc_id"])  # empty candidate set -> null
    cv = cross_validate(name, train, SMALL.get(name), n_splits=2, seed=1)
    assert cv["cv_macro_f1_mean"] is not None and len(cv["folds"]) == 2


def test_normalisation_and_tie_break():
    c = pd.DataFrame({"trip_id": "00000001", "obs_idx": 0, "arc_id": [5, 9]})
    assert np.allclose(normalise_emissions(c, np.array([0.6, 0.2])), [0.75, 0.25])
    assert np.allclose(normalise_emissions(c, np.array([0.0, 0.0])), [0.5, 0.5])
    cands = c.assign(along_arc_fraction=0.5, distance_m=1.0, heading_diff_deg=1.0, speed_mps=1.0, delta_t_s=1.0)
    scored = make_scored(cands, np.array([0.0, 0.0]))
    obs = pd.DataFrame({"trip_id": ["00000001"], "obs_idx": [0]})
    assert classifier_matches(scored, obs).loc[0, "arc_id"] == 5  # smaller arc id wins the tie


def test_evaluator_perfect_predictions_with_empty_candidate_obs(syn):
    held = syn["heldout"]
    preds = pd.DataFrame({"trip_id": "00000005", "obs_idx": [0, 1], "arc_id": [0, 1]})  # none for index 2
    scored = make_scored(held["candidates"], np.array([0.9, 0.1, 0.9, 0.1]))
    m = evaluate(held["references"], preds, held["candidates"], scored)
    assert m["n_aligned"] == 3 and m["n_ambiguous"] == 0 and m["n_unaligned"] == 0
    assert m["top1_accuracy"] == pytest.approx(2 / 3)
    assert m["candidate_miss_rate"] == pytest.approx(1 / 3) and m["n_candidate_miss"] == 1
    assert m["n_null_predictions"] == 1
    assert m["log_loss_n"] == 2 and m["log_loss_coverage"] == pytest.approx(2 / 3)
    assert m["log_loss"] == pytest.approx(-np.log(0.9))
    # macro F1 over trip-qualified arcs {0, 1}: arc 0 perfect, arc 1 recall 1/2 precision 1
    assert m["macro_f1"] == pytest.approx((1.0 + 2 * 1 * 0.5 / 1.5) / 2)


def test_metrics_unavailable_with_zero_eligible_observations():
    refs = pd.DataFrame({"trip_id": ["1"], "obs_idx": [0], "ref_arc_id": [None], "alignment_status": ["ambiguous"]})
    m = evaluate(refs, pd.DataFrame({"trip_id": [], "obs_idx": [], "arc_id": []}))
    assert m["top1_accuracy"] is None and m["macro_f1"] is None and m["n_ambiguous"] == 1


def test_ambiguous_observations_excluded_from_training(syn):
    from pathsnap.models.training.labels import label_candidates
    refs = syn["train"]["references"].copy()
    refs.loc[refs["obs_idx"] == 1, "alignment_status"] = "ambiguous"
    lab = label_candidates(syn["train"]["candidates"], refs)
    assert set(lab["obs_idx"]) == {0}
    assert lab.groupby("obs_idx")["label"].sum().iloc[0] == 4  # one positive per trip


def test_save_reload_in_fresh_process_matches(train, syn, tmp_path):
    scorer = fit_scorer("hist_gradient_boosting", train, SMALL["hist_gradient_boosting"])
    path = scorer.save(tmp_path / "scorer.joblib")
    held = syn["heldout"]
    ref_scored, ref_matches = scorer.score(held["candidates"], held["observations"])
    held["candidates"].to_csv(tmp_path / "c.csv", index=False)
    held["observations"].to_csv(tmp_path / "o.csv", index=False)  # inference inputs only, no references
    subprocess.run([sys.executable, "-m", "pathsnap.models.training.scorer", "--model", str(path),
                    "--candidates", str(tmp_path / "c.csv"), "--observations", str(tmp_path / "o.csv"),
                    "--out", str(tmp_path / "out")], check=True)
    got = pd.read_csv(tmp_path / "out" / "scored_candidates.csv", dtype={"trip_id": str})
    gm = pd.read_csv(tmp_path / "out" / "classifier_matches.csv", dtype={"trip_id": str})
    assert list(got.columns) == SCORED_COLS
    assert got[["trip_id", "obs_idx", "arc_id"]].equals(ref_scored[["trip_id", "obs_idx", "arc_id"]])
    assert np.allclose(got["emission_prob"], ref_scored["emission_prob"], atol=1e-9)
    assert gm["arc_id"].fillna(-1).astype(int).tolist() == ref_matches["arc_id"].fillna(-1).astype(int).tolist()
    assert CandidateScorer.load(path).feature_order == FEATURE_ORDER


def test_scorer_rejects_reference_columns(train, syn):
    scorer = fit_scorer("decision_tree", train, SMALL["decision_tree"])
    leaky = syn["heldout"]["candidates"].assign(ref_arc_id=0)
    with pytest.raises(ValueError):
        scorer.score(leaky, syn["heldout"]["observations"])


def test_small_bayes_search_logs_and_refits(train, tmp_path):
    space = {"max_iter": ("int", 5, 15), "learning_rate": ("float_log", 0.05, 0.3), "min_samples_leaf": ("int", 1, 2)}
    res = bayes_search("hist_gradient_boosting", train, n_trials=3, outdir=tmp_path, n_splits=2, space=space)
    assert res["n_trials"] == 3 and set(res["best_params"]) == set(space)
    hist = pd.read_csv(tmp_path / "hist_gradient_boosting_trials.csv")
    assert len(hist) == 3 and (tmp_path / "hist_gradient_boosting_search.json").exists()
    scorer = fit_scorer("hist_gradient_boosting", train, res["best_params"])  # refit on train
    assert scorer.params["max_iter"] == res["best_params"]["max_iter"]


def test_softmax_emissions_for_rankers():
    c = pd.DataFrame({"trip_id": "00000001", "obs_idx": [0, 0, 1], "arc_id": [5, 9, 2]})
    em = normalise_emissions(c, np.array([1.0, 1.0 + np.log(3.0), 500.0]), mode="softmax")
    assert np.allclose(em, [0.25, 0.75, 1.0])  # stable even for large scores


def test_ranker_learns_to_prefer_the_reference_arc(train, syn):
    scorer = fit_scorer("lightgbm_ranker", train, dict(n_estimators=30, min_child_samples=1, num_leaves=4))
    held = syn["heldout"]
    _, matches = scorer.score(held["candidates"], held["observations"])
    assert matches["arc_id"].iloc[:2].tolist() == [0, 1]
