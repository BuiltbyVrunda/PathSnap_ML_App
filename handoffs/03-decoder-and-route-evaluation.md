# PathSnap handoff · Member 3
## HMM/Viterbi and route evaluation

You own the sequential decoder and its comparison against the classifier baseline. Build with a hand-made directed graph and fixed emissions while Member 2 trains the models.

## What to build

1. Use candidate directed arc IDs as hidden states. Consume the supplied observation-normalised emissions in log space. Preserve observation and arc IDs throughout decoding.
2. Compute transitions using directed network travel between the two projected positions. Consider movement and elapsed time, including stays on the same arc and feasible connecting paths. Recover intermediate arcs for the route output.
3. Implement Viterbi with backpointers and document the recurrence, initialisation, tie handling, and score floor. Break the sequence when the gap policy or graph prevents a defensible connection. Keep original observations visible when a candidate set is empty.
4. Use the same reference alignment and denominators for classifier-only and decoded scores. Measure continuity on consecutive matched observation intervals with the same reachability, time and gap rules for both. Report eligible transition counts, excluded gaps and unresolved points separately.
5. Define a sequence metric, such as ordered arc edit distance on a comparable route representation, and state how repeated arcs, intermediate arcs, and route breaks are handled. Provide per-trip and aggregate comparisons on the ten student-test trips. Explain measured improvements and regressions.

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

### Shared decoder-to-API format

```text
DecodedTrip {
  trip_id: string
  matches: [{
    obs_idx: int, arc_id: int | null,
    unmatched_reason: "no_candidates" | "invalid_transition" | null
  }]
  route_parts: [{
    start_obs_idx: int, end_obs_idx: int, arc_ids: [int],
    break_before: "gps_gap" | "no_candidates" | "invalid_transition" | null
  }]
}
```

Return one match per observation in source order. A matched observation has a null `unmatched_reason`. Each route part is a contiguous directed path and may include intermediate arc IDs between observation matches. The first part has `break_before: null`. Later parts give the reason for the preceding break. Geometry is assembled one part at a time so the UI can display breaks without drawing a line across them.

## Test independently

### Shared two-road fixture

The fixture uses two disconnected eastbound roads. In the table, `.nodes` and `.arcs` contain only the raw columns listed beside the row IDs.

| Node row ID | Longitude | Latitude |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 0.001 | 0 |
| 2 | 0.002 | 0 |
| 3 | 0 | 0.0001 |
| 4 | 0.001 | 0.0001 |
| 5 | 0.002 | 0.0001 |

| Arc row ID | From node | To node |
| ---: | ---: | ---: |
| 0 | 0 | 1 |
| 1 | 1 | 2 |
| 2 | 3 | 4 |
| 3 | 4 | 5 |

Use trip ID `"00000001"`. The `.track` rows are `0.00025 0 1` and `0.00175 0 11`. The `.route` rows are `0` and `1`. Each arc length is about 111 m. For fixtures that supply candidates directly, use candidate arc sets `{0, 2}` for observation 0 and `{1, 3}` for observation 1. The two candidate positions for each observation have along-arc fractions close to `0.25` and `0.75`, respectively.

Use the following emissions and test with equal start priors and equal transition weight for feasible connections. Treat disconnected transitions as impossible.

| obs_idx | arc_id | emission_prob |
| ---: | ---: | ---: |
| 0 | 0 | 0.9 |
| 0 | 2 | 0.1 |
| 1 | 1 | 0.2 |
| 1 | 3 | 0.8 |

1. Expect classifier matches `[0, 3]`. Viterbi should choose `[0, 1]` because its feasible path has a greater product of emissions than `[2, 3]`. Expect one route part from observation 0 to 1 with arc IDs `[0, 1]`. This fixture's aligned accuracy should move from `1/2` to `2/2`.
2. Add an intermediate connecting arc between a pair of observed arcs in a separate tiny graph. Expect three arcs in the route part and two per-observation matches.
3. Set the second timestamp to 121 and configure a 30-second split threshold. Expect two route parts, the second marked `gps_gap`. With the same fixed emissions the independent matches across the break are `[0, 3]`.
4. Restore the original timestamps and keep only candidate 0 at the first observation and candidate 3 at the second. Set the remaining emissions to 1. Expect a break marked `invalid_transition` and no invented connector.
5. Remove all candidates for observation 1. Expect two observation records, with the second arc ID null and reason `no_candidates`. Repeat with all candidate sets empty and expect an empty `route_parts` list. Test that a reverse-only arc cannot satisfy a forward transition.

## Hand off

Send Member 4 the decoder entry point, configuration, decoded-trip records, and a minimal fixture with expected output. Send Member 2 the shared continuity evaluator for the five-model comparison. Supply the classifier-only and decoder comparison, gap cases, and algorithm explanation for report Section 5. Choose decoder parameters on permitted development data only.
