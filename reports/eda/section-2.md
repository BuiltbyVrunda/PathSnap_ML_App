# 2.2 Dataset understanding and exploratory data analysis

PathSnap's development data comprises 80 training trips and 10 student-test trips. Each trip supplies a GPS track, a directed road graph, and an ordered reference route. The data audit read all 360 expected files. No trip failed the file, column, finite-value, coordinate, timestamp-order or index checks. The audit and candidate-setting comparison used only the supplied development data.

| Measured quantity | Training | Student test |
| --- | ---: | ---: |
| Trips and valid files | 80 trips, 320 files | 10 trips, 40 files |
| Total GPS observations | 208,417 | 19,160 |
| GPS observations per trip, min / median / max | 361 / 1,529 / 19,313 | 262 / 1,420 / 3,958 |
| Total road nodes / directed arcs | 1,933,485 / 3,876,904 | 90,689 / 185,041 |
| Maximum arcs in a single trip | 307,392 | 56,027 |
| Median weak components per graph | 20 | 11 |
| Isolated nodes | 0 | 0 |
| One-second sampling intervals | 201,048 / 208,337 (96.50%) | 18,886 / 19,150 (98.62%) |
| GPS intervals over 1 s / over 10 s | 7,286 / 590 | 264 / 27 |
| Longest recorded GPS interval | 153,064 s, trip `00000090` | 210 s, trip `00000065` |
| Trips with duplicate directed endpoint pairs | 53 | 7 |
| Zero-length arcs | 100 | 6 |
| Trips with at least one reference break | 10 | 1 |

Training GPS coordinates span longitude −100.418538° to 137.237029° and latitude −34.640291° to 61.267258°. Student-test coordinates span longitude −93.424337° to 77.271115° and latitude 42.645529° to 61.699590°. These are split-wide bounds across trips in different locations. Per-trip extents and file sizes are in the inventory. Weak connectivity treats directions as undirected for the component count. Even without isolated nodes, some graphs contain several components and direction still governs travel. The longest timestamps are reported as observed. Their cause was not established from the files.

## Preprocessing and candidate features

| Practice | Reason |
| --- | --- |
| Parse headerless whitespace rows strictly and check counts, types, coordinates, and graph/route ID bounds. Reject malformed rows with the file locator. | Dropping or reordering a row could shift every later source ID and silently connect the wrong roads. |
| Preserve string trip IDs and zero-based source IDs, original directions, duplicate endpoint pairs and zero-length arcs. Allow equal timestamps and retain positive gaps. | Joining tables requires stable, trip-qualified IDs. The graph and route reference those exact rows. Irregular time steps carry movement and decoder context. |
| Use a separate WGS84 azimuthal equidistant projection in metres for each trip, centered on the GPS mean latitude and circular mean longitude. Keep the original coordinates for maps. | Longitude degrees vary in physical scale with latitude. A local projection supports fast, consistent segment geometry and avoids assuming every trip lies in one projection zone. |
| Use an STRtree over each directed endpoint-to-endpoint segment. Use each zero-length segment as a point. | A spatial index avoids scanning the full graph per observation and still finds long arcs when neither endpoint lies inside the search radius. Point handling preserves degenerate arc identity. |
| Retrieve within a configurable radius, rank by distance with arc ID as the tie-breaker, cap the results, and emit rows ordered by `(obs_idx, arc_id)`. Keep all GPS rows in the trip. | This bounds model and decoder work while preserving reproducibility and leaving no-candidate observations visible. |
| Compute features from GPS and road geometry. Read reference files in a separate module. | Feature generation is the same with or without a route file. The reference remains available for training labels and post-prediction evaluation. |
| Export compressed CSVs per trip, with reference records in a separate table and a per-trip coverage record. | Per-trip shards preserve split boundaries and can be regenerated without transferring the raw files. The separate reference table retains observations with zero candidates. |

Point-to-segment geometry uses straight segments formed from the two supplied nodes. For those segments, the largest absolute difference between projected length and WGS84 inverse geodesic endpoint length was 0.0114 m on training graphs and 0.0186 m on student-test graphs. This check measures projection error on arc lengths. It cannot validate road shape between endpoints or GPS measurement accuracy.

| Feature | Units and calculation | Reason |
| --- | --- | --- |
| `along_arc_fraction` | Clamped projection fraction, 0 at the from-node and 1 at the to-node. Zero for a point arc. | Provides direction-aware position within the arc for route transitions and model scoring. |
| `distance_m` | Euclidean distance in metres to the closest location on the projected segment. | A closer road is generally more plausible. |
| `heading_diff_deg` | Smallest unsigned difference, 0°–180°, between the directed arc and the previous GPS displacement. Missing for the first observation, a point arc, displacement under 3 m, elapsed time above 10 s, or elapsed time at or below zero. | Differentiates roads with opposing directions. A minimum displacement and maximum elapsed time reduce unreliable bearings from jitter and long gaps. |
| `speed_mps` | Straight-line displacement from the previous GPS point divided by actual elapsed seconds. Missing at the first point or when elapsed time is zero. | Gives movement context. After a gap this is an interval average based on net displacement. |
| `delta_t_s` | Elapsed seconds since the preceding observation. Missing for the first observation. | Identifies irregular sampling and lets models interpret the corresponding movement features. |

At the selected candidate settings the heading feature was missing for 124,225 of 1,978,331 training candidate rows and 12,015 of 229,773 student-test candidate rows. Speed and time delta were each missing for 1,099 training rows and 99 student-test rows. These are candidate-row counts, so an observation can contribute multiple missing values. Keep nullable fields as missing through export so the model can handle them inside its fitted preprocessing.

## Reference alignment and scoring

The route file lists original arc IDs in traversal order. The aligner indexes **route occurrences** and searches within 75 m of each observation independently of the inference candidate set. For an eligible occurrence, the match reward is `76 - distance_m - 12 * heading_diff_deg / 180`. When heading is unavailable, its cost is zero. Nonpositive rewards are excluded. Dynamic programming maximizes the total reward subject to nondecreasing route positions. The same occurrence may support multiple observations, and skipping an observation or an unused route occurrence is allowed. The 75 m reference radius provides room for noisy observations and permits the narrower inference radius to miss a valid reference. The bounded heading term helps distinguish direction while allowing geometry to dominate. These are heuristics on the supplied track and route.

A backward pass computes the best complete ordered path through each possible match. A matched observation is **ambiguous** if using a different arc ID is within 3 reward units of the optimum. The margin avoids giving near ties a hard label. Ambiguity is scored across distinct IDs, so repeated occurrences of one arc count as one possible ID. A duplicate arc outside the reference route remains eligible as an inference candidate. Two plausible duplicate IDs explicitly in the reference are treated as ambiguous. A skipped observation is **unaligned**. Only **aligned** rows carry an arc ID. Broken links are recorded at their supplied route positions and the order on either side is kept. The map draws each contiguous reference part separately.

Training trips `00000054` and `00000033` account for 20,183 of the 22,096 unaligned training observations. An independent 75 m reference-only spatial query found no reference segment within range for 16,475 of 19,313 observations in `00000054`, and for 3,012 of 5,844 in `00000033`. Some observations near a reference arc were also skipped by the reward and order rules. These distances explain much of the low alignment yield on these trips. The cause of the divergence remains undetermined.

## Train-only candidate comparison

| Radius | Cap | Candidate rows | Candidates per GPS observation | Aligned misses out of 158,564 | Miss rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 m | 12 | 1,185,108 | 5.69 | 2,135 | 1.3465% |
| 60 m | 12 | 1,467,111 | 7.04 | 1,546 | 0.9750% |
| **60 m** | **24** | **1,978,331** | **9.49** | **1,024** | **0.6458%** |
| 75 m | 32 | 2,624,043 | 12.59 | 149 | 0.0940% |

The sweep reused the 75 m / 32 candidate superset and applied narrower distance and rank filters to hold the graph, observations, and labels fixed. The default 60 m / 24 setting retained 99.35% of aligned references on training data. The 75 m / 32 alternative retrieved another 875 references at the cost of 645,712 additional candidate rows, about 33% more than the default. The default limits the work passed to the model and decoder while keeping training retrieval above 99%. The student-test split was then audited at the selected settings.

| Split | GPS observations | Aligned | Ambiguous | Unaligned | Zero candidates (all observations) | Aligned misses | Miss rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 208,417 | 158,564 | 27,757 | 22,096 | 17,493 | 1,024 | 0.6458% |
| Student test | 19,160 | 15,979 | 3,160 | 21 | 0 | 4 | 0.0250% |

The miss-rate denominator includes every aligned observation. Of the 1,024 training misses, 321 had zero candidates. The other misses had candidates but lacked the exact reference arc. Alignment yield was 76.08% on training and 83.40% on student test, so the miss rates describe the scorable subsets. Exact-arc accuracy and macro F1 use the same aligned-observation eligibility. Missing predictions count as errors, and macro F1 uses trip-qualified classes without a class for missing predictions. These results measure candidate coverage against derived labels. Model accuracy and route continuity require the later experiments.

## Figures and verification

- `development_split_eda.png` plots trajectory sizes, graph sizes and sampling gaps for all 90 development trips.
- `gps_and_road_geometry.png` and `development_map.html` show training trip `00000048` with GPS, nearby roads and its supplied reference. The trip's compact GPS extent and 2,592-arc graph made the layers readable at overview scale. The Folium map provides a basemap and layer toggles. The browser-rendered `development_map_preview.png` captures these layers and the OpenStreetMap attribution for the EDA report figure. The map uses a training example before integration with model predictions.
- The inventory, per-trip coverage, split totals and environment metadata are included in the Member 4 ZIP. Reproduction commands and interface examples are in the [data handoff](../../handoffs/01-data-and-features.md#implementation-and-reproduction).
- Verification covers inference with the route file removed, directed geometry and distance units, duplicate and degenerate arcs, gap handling, exact-arc misses, malformed input, trip-qualified scores, and comparison of the alignment recurrence against exhaustive small cases.

Projection and spatial-index references: [PROJ azimuthal equidistant projection](https://proj.org/en/stable/operations/projections/aeqd.html), [Shapely STRtree](https://shapely.readthedocs.io/en/stable/strtree.html).
