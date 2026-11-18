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
Harness, scorer, metrics and Bayesian (Optuna TPE) search are verified on the synthetic trips only. Real
80-trip CV, tuning and student-test comparison are pending Member 1's candidate and reference tables, and the
continuity metric is pending Member 3's evaluator (`continuity_fn` hook in `evaluate_scorer`).
