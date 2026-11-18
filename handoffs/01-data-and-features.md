# PathSnap handoff · Member 1
## Data, candidates, and reference alignment

You own the input pipeline. Members 2 and 3 need candidate rows and an independent reference table. Member 4 needs an inference loader that works from GPS and graph data alone.

## What to build

1. Read raw data from `Student folder/Student folder/{train,student_test}/{trip_id}/`. The filename stem matches the trip ID. Files are whitespace separated and headerless.
2. Inventory the 80 training trips and 10 student-test trips. Summarise file sizes, GPS counts, missing or invalid values, time gaps, geographic extent, graph size and weak connectivity, and reference-route discontinuities. Retain a representative trip and coordinates for the EDA map.
3. Build an inference loader and a spatial search for nearby directed road segments. Choose and document how you calculate distances in metres across trip locations. Make candidate radius and limit configurable. Preserve duplicate endpoint pairs with their original arc IDs.
4. Separately read the ordered reference arcs and implement observation-level alignment, including ambiguous cases and route breaks. Route position and GPS row position are different quantities. Retain alignment and candidate-miss counts by trip and split.
5. Publish the schema and one small parsed sample first so the other members can wire their modules before the dataset pipeline is finished.

| Raw file | Columns |
| --- | --- |
| `.track` | Longitude, latitude, timestamp in seconds |
| `.nodes` | Longitude, latitude |
| `.arcs` | Directed from-node ID, to-node ID |
| `.route` | Ordered reference arc ID |

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

## Test independently

### Shared two-road fixture

The fixture uses two disconnected eastbound roads. The [parsed fixture](fixtures/two_road.json) gives team members concrete trip, candidate and reference records to use when wiring their modules. In the table, `.nodes` and `.arcs` contain only the raw columns listed beside the row IDs.

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

1. Save the fixture as four files under a temporary `train/00000001/` folder. Load the trip and assert that all row IDs, timestamps, and directions survive parsing.
2. Set a 30 m search radius and allow at least two candidates. Expect arc IDs `[0, 2]` at observation 0 and `[1, 3]` at observation 1. The lower-road candidates have near-zero distance and the upper-road candidates are about 11 m away. Check the projection fractions within a small numerical tolerance.
3. Enable reference alignment. Expect aligned arcs `[0, 1]` and a candidate miss rate of `0/2`. Then remove the `.route` file and rerun inference. Expect identical trip and candidate output.
4. Remove the candidate row for arc `1` at observation 1 before computing coverage. Expect `1/2` candidate misses with reference rows unchanged. Rerun candidate generation with an observation outside the search radius and verify its ID stays in the trip with zero candidates.
5. Add a duplicate endpoint row to `.arcs`, try a reference route with a broken link, and try an out-of-range node ID. Check that duplicates keep distinct IDs, the break is visible, and the invalid graph is rejected with a trip and file locator.

## Hand off

Send the loader and candidate generator to Members 2–4. Send the alignment module and labelled development sample to Members 2 and 3, and the server-side scoring entry point to Member 4. Record the feature units, alignment policy, candidate settings, miss rates, and EDA evidence for report Section 2. Keep the ten reserved evaluator trips out of development.

## Implementation and reproduction

Run the commands from the extracted bundle or the repository root with Python 3.12. Install `requirements-data.txt` for loading, candidates, export, and scoring. Install `requirements-eda.txt` for the audit, plots, and Folium map.

```powershell
python -m pip install -r requirements-data.txt
python -m unittest discover -s tests/integration -v
```

```python
from pathsnap.data.loading import load_trip
from pathsnap.data.candidates import FEATURE_COLUMNS, CandidateConfig, generate_candidates

trip = load_trip(trip_dir)
candidates = generate_candidates(trip, CandidateConfig(radius_m=60, max_candidates=24))
shared_trip_record = trip.to_record()
```

`trip.track[obs_idx]` provides longitude, latitude and timestamp. `trip.arc_nodes[arc_id]` provides the directed node pair, `trip.arc_lengths_m[arc_id]` gives its projected length, and `trip.node_lonlat[node_id]` provides map coordinates in longitude-first order. Arrays are read-only. Candidate and reference rows are dataclasses and can be serialized with `dataclasses.asdict`. Candidate ordering within an observation is by original arc ID. To rank candidates, use the model scores.

### Member 2 and Member 3 development data

```powershell
python -m pathsnap.data.development --dataset "Student folder/Student folder" --output runs/member1/development --split train
python -m pathsnap.data.development --dataset "Student folder/Student folder" --output runs/member1/development --split student_test
```

Add `--trip-id 00000001` to regenerate the sample provided in the handoff. Each trip has `candidates.csv.gz`, `references.csv.gz`, and `coverage.json`. The exporter reserves a trip folder for one writer and publishes the coverage file after both tables finish so readers can identify complete exports. If an interrupted process leaves an `.exporting` folder, confirm the writer has exited before clearing it and rerunning the trip. Read `trip_id` as text and blank numeric cells as missing values. Train using the five fields in `FEATURE_COLUMNS`. Join candidates to references only in the development workflow, keep aligned rows for supervision, and label a candidate positive when its arc ID equals the reference arc ID. Fit preprocessing inside trip-grouped training folds. Use the full reference table as the evaluation denominator and count a candidate miss or unresolved match as incorrect.

Use `candidate_coverage(references, candidates)` for retrieval coverage and `score_predictions(references, predictions)` for pointwise scores. Both live in `pathsnap.data.reference_alignment.scoring`. The second function takes a mapping from `(trip_id, obs_idx)` to arc ID or `None`. Macro F1 averages the trip-qualified union of true and predicted arc classes on aligned observations. A missing prediction counts as a false negative for the true class. Decode over the complete ordered `trip.track`, including observations absent from the candidate table. The decoder owns its own transition and GPS-gap policy.

### Member 4 server and report

Load the trip and generate candidates before prediction. Once the decoder returns, call `score_trip(trip, reference_dir, predictions)` on Laptop 1 using the same trip object and a mapping from `obs_idx` to arc ID or `None`. The function opens the expected `<trip_id>.route` in `reference_dir` and returns aggregate counts, accuracy, and macro F1. A missing file raises `FileNotFoundError`. Return `metrics: null` when references are unavailable or the aligned count is zero. The API may select the aggregate fields defined in the response contract. Keep the development Folium output on the team side.

```powershell
python -m pip install -r requirements-eda.txt
python -m pathsnap.data.eda --dataset "Student folder/Student folder" --output runs/member1/train-audit --split train --train-sweep
python -m pathsnap.data.eda --dataset "Student folder/Student folder" --output runs/member1/student-test-audit --split student_test
python -m pathsnap.data.eda_plots --dataset "Student folder/Student folder" --output runs/member1/plots --split train --trip-id 00000048 --inventory runs/member1/train-audit/trip_inventory.csv --inventory runs/member1/student-test-audit/trip_inventory.csv
```

The Folium map uses training trip `00000048` and supplies GPS, nearby roads and the reference route with layer controls. Use it for EDA and use the integrated prediction map for the final demonstration. For a report image, open the generated HTML in a browser and capture the initial view with the legend, controls and map attribution visible. Save it as `runs/member1/plots/development_map_preview.png` to include it in the Member 4 ZIP. The static plots, audit tables, and the [Section 2 draft](../reports/eda/section-2.md) accompany the report bundle.

### ZIP recipients

| Archive | Send to | Purpose |
| --- | --- | --- |
| `PathSnap-Member1-for-Member2.zip` | Member 2 | Candidate features, label generation, training-trip example, and coverage |
| `PathSnap-Member1-for-Member3.zip` | Member 3 | Directed graph and candidate contracts, alignment and pointwise scoring, and training-trip example |
| `PathSnap-Member1-for-Member4.zip` | Member 4 | Inference and server scoring, report Section 2, inventory, plots, and Folium EDA map |

The ZIPs contain the implementation, tests, installation instructions and role-specific evidence. The supplied raw dataset is used from each team member's local copy.

The local ZIPs in `runs/member1/handoffs/` use export tables checked against the audit at the default settings. Keep transfer archives under `runs/` so repository clones stay focused on the source and reproducible workflows.
