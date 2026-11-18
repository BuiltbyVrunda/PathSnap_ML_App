# PathSnap team plan

PathSnap takes a noisy GPS trip and its road network, scores nearby road segments for each GPS observation, and uses HMM/Viterbi decoding to reconstruct a coherent route. Laptop 1 runs the trained pipeline, FastAPI, and server-side scoring. Laptop 2 displays the ten evaluator trip options and shows the predicted route and scores returned by the API.

## What is in the supplied data

The dataset root is `Student folder/Student folder/`. The ten reserved evaluator trips described in the guidelines are absent from the supplied folders.

| Partition | Trips | GPS observations | Intended use |
| --- | ---: | ---: | --- |
| `train` | 80 | 208,417 | Fit preprocessing and models, run trip-grouped cross-validation and Bayesian optimisation |
| `student_test` | 10 | 19,160 | Compare models, assess tuning and decoding, and rehearse the demo |
| Reserved evaluator testing | 10 per guide | Unavailable | Final evaluation only |

Each supplied trip has four headerless, whitespace-separated files. `.track` contains longitude, latitude, and time in seconds. `.nodes` contains node longitude and latitude, with the zero-based row number as the node ID. `.arcs` contains directed from/to node IDs, with the zero-based row number as the arc ID. `.route` contains the chronological sequence of reference arc IDs. Preserve trip IDs as strings and use `(trip_id, arc_id)` to identify segments across trips.

All 90 trips passed basic column, coordinate-bound, index-bound, and timestamp-order checks. GPS tracks ranged from 262 to 19,313 observations and road networks reached 307,392 arcs in one trip. There were 7,550 time steps longer than one second. Sixty trips contained at least one duplicate directed endpoint pair, and 11 reference routes had at least one adjacent pair of arcs whose endpoints failed to join. Preserve original arc identities and account for gaps and reference discontinuities when interpreting scores.

The main dependency is label alignment. A reference route lists traversed arcs, and GPS observations arrive on their own timeline. Document and test an order-aware alignment policy before trusting pointwise metrics. Keep track of observations with ambiguous alignment and observations whose reference arc was missed during candidate generation.

## Four parallel assignments

### Member 1 · Data, candidates, and reference alignment

- **Start immediately:** Parse the raw trip files, produce the dataset inventory, and inspect missing values, coordinate ranges, sampling gaps, graph connectivity, and representative routes. Keep the inference loader for GPS and road graph separate from the reference reader.
- **Build:** Use distances in metres and a spatial search to retrieve nearby directed arcs. Produce candidate features such as point-to-segment distance, heading difference, and time or movement context. Preserve original observation and arc IDs. Implement reference-to-observation alignment in a separate development labelling and scoring module. Use the route order and document how you handle duplicates, ambiguous points, and disconnected reference arcs.
- **Handoff:** An inference-ready candidate generator, repeatable labelled development data or a generator, a small labelled development example, and data-quality and candidate-coverage summaries. Keep labelled tables separate from inference inputs. Compute candidate miss rate against the aligned references, including observations where candidate generation misses the reference.
- **Report:** Section 2, dataset understanding and feature engineering. Supply the EDA data and plots and coordinate the EDA map with Member 4.

### Member 2 · Model experiments and Bayesian tuning

- **Start immediately:** Build a shared training and evaluation harness using a tiny synthetic candidate table. Prepare five distinct model runs. An initial candidate set is logistic regression, Gaussian naive Bayes, a decision tree, a random forest, and histogram gradient boosting.
- **Build:** Learn to score candidate rows using match labels and rank the candidates within each observation. Use the same trip-grouped folds and fair evaluation rules across all five models. Log configurations, seeds, fold results, candidate coverage, inference time, and predictions. Run Bayesian search on the chosen model using training-trip folds, refit on the training partition, and compare tuned and untuned variants on the same student-test trips.
- **Handoff:** Reproducible scripts for every attempted model, comparison tables and plots, search space and trial history, the final saved model and preprocessing, and candidate-aligned scores Member 3 can use for decoding. Document the score-to-emission convention and observation-level log loss convention where applicable.
- **Report:** Sections 3 and 4, including per-model analysis, the consolidated comparison, and Bayesian optimisation. Include CV macro F1, student-test Top-1 accuracy and macro F1, route continuity, log loss where applicable, and inference time. Use Member 3's continuity code once it is ready.

### Member 3 · HMM/Viterbi and route evaluation

- **Start immediately:** Use a hand-built directed graph with candidate scores to implement and test decoding. Include a parallel road, a disconnected candidate, an empty candidate set, and a long GPS gap.
- **Build:** Treat candidate arcs as hidden states and classifier scores as emissions. Use feasible directed network travel between successive candidates to set transition scores. Handle missing candidates, invalid transitions, GPS gaps, and intermediate road segments explicitly. Return a matched arc for each relevant observation and flag any unresolved points or breaks. Distinguish the observation matches from any intermediate arcs added to build the route.
- **Handoff:** A decoder, transition configuration, continuity metrics, and a comparison of classifier-only versus classifier-plus-HMM on the same student-test trips. Report pointwise accuracy and sequence performance with denominators, and explain examples where decoding helped or hurt. Provide a route result that Member 4 can serialize.
- **Report:** Section 5, including state, emission and transition choices, Viterbi recurrence, gap handling, and measured impact.

### Member 4 · FastAPI, evaluator UI, Folium, and release

- **Start immediately:** Build the FastAPI endpoints and UI against a stub predictor with the agreed response shape. Test Laptop 2 calling Laptop 1 over the network using development fixtures. Build Folium layers using a development trip.
- **Build:** Load the real artifacts and wire candidate generation, model scoring, and decoding into the API. Keep reserved reference labels in a scoring component on Laptop 1 and read them after prediction. Laptop 2 lists the evaluator trip IDs and displays predicted geometry and aggregate metrics returned by the API. Capture inference request logs and the successful response on the receiving laptop.
- **Handoff:** Working API and UI, setup and deployment instructions, run commands, screenshots, and the demo checklist. Produce a development Folium map with GPS, roads, prediction, and a reference layer where supplied for development. Keep the evaluator UI limited to allowed layers and aggregate scores. Include a map screenshot in the PDF and commit the map-generation code.
- **Report and release:** Own sections 6 and 7 and assemble the other members' sections into one PDF. Prepare the public GitHub repository with source, dependency and deployment instructions, artifacts, and experiment evidence. Each member supplies their code and verifies their report claims against actual runs.

## Interfaces to agree before implementation

| Boundary | Proposed shared contract |
| --- | --- |
| Data → model and decoder | Each trip has an ordered observation list with `obs_idx`, coordinates, and timestamp, and a directed graph with original arc IDs. A candidate row has `trip_id`, `obs_idx`, `arc_id`, a projected position or fraction along the segment, and named numeric features. A separate development reference table marks aligned arc IDs and alignment status. |
| Model → decoder | Scores are keyed by `trip_id`, `obs_idx`, and `arc_id`. Agree whether each field contains a raw candidate score or an observation-normalized emission. Keep the base classifier's Top-1 arc so decoded results can be compared against the same inputs. |
| Decoder → API | Return ordered per-observation matches, reconstructed route segments with explicit break markers, and diagnostics for gaps or missing candidates. Preserve blank observations so metrics can account for them. |
| API → UI | Suggested endpoints are `GET /trips` and `POST /predict` with a string `trip_id`. Return predicted geometry, aggregate evaluation metrics with counts, and status or error details. Keep reference routes and per-observation reference labels on Laptop 1. |

Agree on a shared scoring policy before comparing results. Specify alignment and duplicate-arc handling, how to count candidate misses and unlabelled observations, macro F1 aggregation across trip-local arc IDs, a continuity definition that accounts for traversable intermediate arcs, and an inference timing boundary. Record score coverage alongside accuracy.

## Integration and hand-in

1. **Parallel start.** Member 1 publishes the schema and a small development fixture. Member 2 uses synthetic candidate rows to build the training harness, Member 3 uses synthetic graph and emission inputs, and Member 4 builds the UI and API against mock predictions.
2. **First end-to-end run.** Bring one supplied development trip through candidates, model scores, Viterbi, API, and UI over the two laptops. Use this as a suggested preparation target for the first evaluation before lunch. The supplied brief gives the timing without specifying a phase-one checklist.
3. **Full development validation.** Compare the five trained models, retain the Bayesian search record, run the before/after decoder comparison, and exercise all ten supplied student-test trips through the full integration. Record any unresolved observations and failure cases.
4. **Final evaluation.** Point the API at the organiser-provided trip set when available. Support selection of all ten reserved trips, live prediction on Laptop 1, server-side scoring, visible request logs, and display on Laptop 2. Keep reserved data out of training, tuning, and development decisions.
5. **Submission.** Hand in exactly two deliverables, a public GitHub repository and one consolidated technical report PDF. Include the actual plots, trial results, reproducible code and artifacts, a Folium map screenshot, working UI screenshots, and a successful API response captured on the receiving laptop. Each team member should be able to explain the entire pipeline in the viva.

## Source materials

- [Main Guidelines](PathSnap_student_materials/Main%20Guidelines.pdf), project scope and two-laptop deployment
- [Dataset Exploration](PathSnap_student_materials/Dataset%20Exploration.pdf), file definitions, data partitions, and leakage rules
- [Models and Training Guidelines](PathSnap_student_materials/Models%20and%20Training%20Guidelines.pdf), experiments, Bayesian search, decoding, and final evaluation
- [Deliverables](PathSnap_student_materials/Deliverables.pdf), report sections and submission requirements
- [Assessment and Plagiarism Policy](PathSnap_student_materials/Assessment%20and%20Plagiarism%20Policy.pdf), independent work, verifiable evidence, and viva
