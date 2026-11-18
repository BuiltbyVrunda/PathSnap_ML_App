# PathSnap handoff · Member 2
## Model experiments and Bayesian optimisation

You own the candidate scorer, comparisons, and the saved model on Laptop 1. Build against synthetic candidate records while Member 1 prepares the real data.

## What to build

1. Join development references to candidate rows on `(trip_id, obs_idx)`. For aligned observations, a candidate is positive when `arc_id == ref_arc_id`. Exclude ambiguous and unaligned observations from supervised training. Retain all aligned observations in evaluation, including those with zero candidates.
2. Train and compare at least five distinct models. Start with logistic regression, Gaussian naive Bayes, a decision tree, a random forest, and histogram gradient boosting. Use the same grouped folds by trip ID on the 80 training trips. Fit any feature transforms within each training fold.
3. Use named candidate features with an explicit feature order. Keep raw trip, observation, arc and node IDs as identifiers. Handle nullable features in the fitted preprocessing. Save model configurations, seeds, fold results, training time, inference time, and predictions.
4. Compare models on the ten `student_test` trips. Record CV macro F1, student-test Top-1 accuracy and macro F1, precision and recall, candidate miss rate, route continuity using Member 3's evaluator, and inference time. Where applicable, log observation-level log loss as the mean of `-log(max(reference emission, epsilon))` on aligned observations with the reference in the candidate set. Report that subset's size and coverage so comparisons show candidate misses.
5. Run Bayesian optimisation for the selected model using only grouped folds on `train`. Record the objective, search ranges, trial count, best parameters, and trial history. Refit using `train` and compare tuned and untuned performance on the same `student_test` trips. Keep the optimiser and reserved evaluator trips separate.

### Shared trip and candidate formats

These v1 record shapes describe the fields passed between modules. The examples use JSON. In-process implementations may use typed records or tables with the same fields.

```text
Trip {
  trip_id: string
  observations: [{obs_idx: int, lon: float, lat: float, time_s: float}]
  nodes: [{node_id: int, lon: float, lat: float}]
  arcs: [{arc_id: int, from_node_id: int, to_node_id: int, length_m: float}]
}

Candidate {
  trip_id: string
  obs_idx: int
  arc_id: int
  along_arc_fraction: float
  distance_m: float
  heading_diff_deg: float | null
  speed_mps: float | null
  delta_t_s: float | null
}
```

- Keep trip IDs as strings with leading zeros. Observation, node, and arc IDs are zero-based raw row numbers scoped to the trip. Keep observations in source order and preserve IDs even if a point has no candidates.
- Use longitude first, latitude second, and metres for distances. Arcs are directed. `along_arc_fraction` runs from 0 at the from-node to 1 at the to-node. Heading difference is the smallest unsigned angle in degrees, in `[0, 180]`. Use `null` for unavailable features.
- Candidate keys `(trip_id, obs_idx, arc_id)` are unique. Sort candidate rows by `obs_idx`, then `arc_id` within a trip. Keep the full observation list alongside candidate rows so an empty candidate set is visible.

### Shared reference format and scoring

```text
Reference {
  trip_id: string
  obs_idx: int
  ref_arc_id: int | null
  alignment_status: "aligned" | "ambiguous" | "unaligned"
}
```

Create one reference row per observation. Only `aligned` rows have a reference arc ID. The alignment uses the ordered reference route and observations independently of the generated candidate set. Identical directed endpoint pairs can be ambiguous. Preserve their arc IDs and assign an exact reference ID only when the alignment supports it. Fit and run inference with references kept in a separate data flow. The reserved evaluator reference reader runs on Laptop 1 after prediction.

For development, compute Top-1 accuracy over all aligned observations. An aligned observation with a candidate miss or unresolved prediction counts as incorrect. Report the number of aligned, ambiguous, and unaligned observations. Candidate miss rate is the number of aligned observations whose reference arc is missing from their candidate set divided by the total aligned observations. Use trip-qualified arc IDs for macro F1 and exclude the unmatched sentinel from the class average while counting its false negatives for true classes. A metric with zero eligible observations is unavailable.

### Shared model-to-decoder format

```text
ScoredCandidate = Candidate + {emission_prob: float}

ClassifierMatch {
  trip_id: string
  obs_idx: int
  arc_id: int | null
}
```

Keep candidate fields and keys unchanged. For the initial probabilistic models, divide each candidate's positive-class probability by the sum of those probabilities within the observation. Use uniform emissions if the sum is zero. Emissions sum to 1 for each nonempty candidate set. Emit one classifier match for every observation, using the highest emission and the smaller arc ID on a tie. Use `null` for an empty candidate set. The decoder can floor emission probabilities before taking logs. Keep reference fields out of both output records.

## Test independently

Create four synthetic training trips with IDs `"00000001"` through `"00000004"`. Repeat these rows for each trip. Send candidate records and reference records to the scorer as separate inputs during evaluation.

| obs_idx | arc_id | along_arc_fraction | distance_m | heading_diff_deg | speed_mps | delta_t_s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.25 | 1 | 5 | null | null |
| 0 | 2 | 0.25 | 30 | 100 | null | null |
| 1 | 1 | 0.75 | 2 | 8 | 12 | 10 |
| 1 | 3 | 0.75 | 25 | 90 | 12 | 10 |

Set reference arcs to `0` and `1` with `alignment_status: "aligned"` for each trip. Build a held-out synthetic trip `"00000005"` with those same rows and an extra aligned observation at index 2, reference arc `1`, and zero candidate rows.

1. Run a two-fold grouped smoke test and assert that each trip appears on only one side of each split. Check that all five model configurations can train and emit scores.
2. Unit-test normalisation with stub probabilities `[0.6, 0.2]`. Expect emissions `[0.75, 0.25]`. For `[0, 0]` expect `[0.5, 0.5]` and the smaller arc ID to win the tie.
3. Feed perfect predictions for indices 0 and 1 and no prediction for index 2 into the evaluator. Expect Top-1 accuracy `2/3`, candidate miss rate `1/3`, and a null classifier match for index 2. If testing log loss, expect two covered observations.
4. Fit and serialise a real smoke-test model. Reload it in a fresh process and score the same input. Verify the keys, ordering, match IDs, and probabilities within numerical tolerance. Run the scorer with only inference inputs.
5. Run a small Bayesian search against grouped synthetic folds to verify logging and refit. Replace it with the documented full training run before reporting experimental results.

## Hand off

Send Member 3 the score contract, sample output, and the classifier's baseline matches. Send Member 4 the saved fitted pipeline, loader and scoring entry point, dependency versions, feature order, model configuration, and a reproducible load-and-predict command. Supply actual comparison tables, plots, metrics definitions, tuning trials, and findings for report Sections 3 and 4.
