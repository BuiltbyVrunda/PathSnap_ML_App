# Member 2: candidate scorer, model comparison, Bayesian tuning

## Conventions (agree with Members 1, 3, 4)
- **Features (fixed order):** `along_arc_fraction, distance_m, heading_diff_deg, speed_mps, delta_t_s` plus three
  within-observation context features `distance_rank, distance_gap_to_min_m, n_candidates` (`schema.FEATURE_ORDER`).
- **Nulls:** median imputation + missing-indicator columns, fitted inside each training fold / the saved pipeline.
- **Emission:** positive-class probability divided by the sum within the observation; uniform if the sum is 0;
  sums to 1 per non-empty candidate set. Output = Candidate columns + `emission_prob`.
- **Classifier match:** one row per observation, highest emission, smaller `arc_id` on ties, `null` if no candidates.
- **Metrics:** see `evaluation/metrics.py` docstring (Top-1 over aligned obs, candidate miss rate, trip-qualified
  macro F1 with sentinel excluded, log loss on the covered subset with size/coverage reported).
- **Inference timing boundary:** candidate table in -> scored candidates + matches out.

## Extra challengers
- `lightgbm_ranker`: LightGBM LambdaMART (`lambdarank`), one query group per GPS observation, label 1 for the
  reference arc. Raw scores become emissions with a **softmax within each observation** (`emission_mode`
  in `training/models.py`). No inner early stopping; `n_estimators` is tuned by the grouped-CV search.
- `catboost`: `CatBoostClassifier` (Logloss); `predict_proba` goes through the usual proportional normalisation.
- Both use the same features, preprocessing, grouped folds and metrics as the five baselines. Run with
  `python -m pathsnap.models.run_extra_models --data runs/dev_tables --promote` after `run_member2`.

## Notebook and saved results
- `notebooks/member2_training_and_testing.ipynb` loads the saved models/results, re-scores `student_test` with every
  saved model, checks the numbers against the saved runs, and shows folds, Bayesian trials and the final model.
  Its outputs are written to `reports/model-experiments/notebook_outputs/` (`NBnn_` prefix).
- The full runs were executed by `run_member2.py` and `run_extra_models.py` on a local CPU laptop; the notebook's
  retraining cells (`RUN_TRAINING`) call the same library functions and are off by default.
- `reports/model-experiments/results_archive/` holds descriptively named copies of every run output;
  `MANIFEST.md` lists source file, producer and timestamp for each.

## Commands
```bash
python -m pytest tests/models -q                       # synthetic handoff checks
python -m pathsnap.models.run_member2 --data runs/dev_tables --out reports/model-experiments --trials 25
# load-and-predict (inference inputs only):
python -m pathsnap.models.training.scorer --model artifacts/models/pathsnap_scorer.joblib \
    --candidates cands.csv --observations obs.csv --out out_dir
```
In Python: `CandidateScorer.load(path).score(candidates_df, observations_df) -> (scored, matches)`.

## Status
Seven models were cross-validated and compared on the real 80 train / 10 student_test trips; LightGBM ranker and
CatBoost were tuned (20 Optuna trials each); the final model is the tuned LightGBM ranker. Random forest was not
tuned. Route continuity is pending Member 3's evaluator (`continuity_fn` hook in `evaluate_scorer`).
