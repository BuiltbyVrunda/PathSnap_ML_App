# PathSnap handoff · Member 4
## FastAPI, evaluator UI, Folium, and release

You own the live two-laptop demonstration and the assembly of the final submission. Build the API and UI against a stub while the other three members develop the inference pipeline.

## What to build

1. On Laptop 1, create a FastAPI app with configurable paths for the trip registry and model artifacts. Keep a predictor interface that accepts the chosen trip and returns a decoded trip. Implement it first with a development stub.
2. On Laptop 2, build the trip selection UI. Get its options from the API. Send the chosen trip ID, show loading and error states, render route parts on a map, and display accuracy and counts. Make the Laptop 1 base URL configurable. Configure CORS for the actual UI origin for direct browser requests.
3. Connect Member 1's inference loader and candidate generator, Member 2's model scorer, and Member 3's decoder on Laptop 1. Invoke server-side scoring after decoding. The registry, inference pipeline and scoring code should support all ten reserved trips when supplied. Keep reference labels and route geometry in the server-side scoring flow.
4. Build a separate development Folium map with raw GPS, roads, predicted route, and reference route where available. Give it a legend and layer controls. Capture a representative screenshot for the PDF and include generation code in the repository.
5. Add a short runbook with startup commands, model artifact paths, how to set the API URL from Laptop 2, and the command to verify the connection. Record real inference request logs and capture a successful response on the receiving laptop.

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

### Shared API-to-UI format

`GET /trips` returns the configured trip list. `POST /predict` takes a string `trip_id`. Keep the endpoint contract identical on both laptops.

```json
{"trips":[{"trip_id":"00000001"}]}
```

```json
{"trip_id":"00000001"}
```

Example prediction response for the two-road test trip

```json
{
  "trip_id": "00000001",
  "status": "ok",
  "route_geometry": {
    "type": "MultiLineString",
    "coordinates": [[[0.0, 0.0], [0.001, 0.0], [0.002, 0.0]]]
  },
  "counts": {"observations": 2, "matched": 2, "route_breaks": 0},
  "metrics": {"accuracy": 1.0, "correct": 2, "aligned": 2}
}
```

- `status` is `ok` for a fully matched continuous trip, `partial` when some observations are unresolved or the route has breaks, or `no_match` when no observations match. A no-match result has null geometry. Use one LineString per route part with GeoJSON coordinate order `[longitude, latitude]`. Count route breaks as the number of parts with a non-null `break_before`.
- `accuracy` is a fraction from 0 to 1. Display it as a percentage with its denominator. `metrics` is null when there are no scorable references. When references exist, include the counts and score every aligned observation, including unresolved matches as incorrect.
- Unknown trip IDs return HTTP 404. Invalid payloads return HTTP 422. For failures, show a readable UI message. Resolve IDs through the configured trip list.

## Test independently

1. In development, configure the API with the ten supplied `student_test` trip IDs, or generate ten synthetic IDs. Return the example response above from the predictor stub for `"00000001"` in a synthetic registry. The UI should populate ten options, render the lower road for this fixture, and display `100% (2/2)`.
2. For a second synthetic trip, return two non-touching route parts, status `partial`, and a metrics object with fewer correct observations than aligned ones. Confirm two visible polylines with a gap and the corresponding score. Test no-match, a null metrics field, timeout, unknown trip, and invalid JSON body shape.
3. Inspect the response body and browser network traffic. Verify the JSON contains no reference geometry or per-observation reference labels. Check receive-side logs for the request ID or trip ID and successful status. For the final pipeline, repeat with the real predictor and keep the reference reader callable only by server-side scoring after a prediction is produced.
4. Once the app is implemented, start it on Laptop 1 with Uvicorn bound to `0.0.0.0`. The example uses `pathsnap.service.api:app` as the app module. On Laptop 2, replace `LAPTOP1_IP` with Laptop 1's LAN address and try both endpoints from PowerShell.

```powershell
# Laptop 1
python -m uvicorn pathsnap.service.api:app --host 0.0.0.0 --port 8000

# Laptop 2
$pathSnapApi = 'http://LAPTOP1_IP:8000'
$pathSnapTrips = Invoke-RestMethod -Uri "$pathSnapApi/trips"
$pathSnapTrips.trips.Count
$pathSnapBody = @{ trip_id = $pathSnapTrips.trips[0].trip_id } | ConvertTo-Json
Invoke-RestMethod -Uri "$pathSnapApi/predict" -Method Post -ContentType 'application/json' -Body $pathSnapBody
```

5. Open the UI on Laptop 2 against the same API. Select each development trip and verify that it triggers a request on Laptop 1. Repeat with live inference and the reserved registry during final evaluation once those trips are available.

## Hand off

Ship source, dependencies, run commands, and the two-laptop verification checklist. Own report Sections 6 and 7 and assemble one consolidated PDF with each member's actual results, a Folium screenshot, working UI screenshots, and a successful API response captured on the receiving laptop. Prepare the public GitHub repository with reproducible source, experiment logs, and required model/configuration artifacts. Keep credentials and reserved evaluator data out of the public repository.
