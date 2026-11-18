"""PathSnap API endpoints and request logging (runs on Laptop 1)."""
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from . import ground_truth, route, trips
from .config import TRIPS_DIR
from .loader import TripDataError, load_points, load_road_network
from .predictor import predict_trip

# ---- Logging: one clean line per request, shown to the evaluator during the demo ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("pathsnap.api")
# Uvicorn prints its own access line too; hide it so each request is logged once, in our format.
logging.getLogger("uvicorn.access").disabled = True

app = FastAPI(title="PathSnap API")
log.info("PathSnap API starting | trip folder: %s | %d trips found", TRIPS_DIR, len(trips.list_trip_ids()))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Time every request and log: method, path, trip_id, status code, duration, caller IP."""
    start = time.perf_counter()
    status = 500  # used if the endpoint crashes
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        ms = (time.perf_counter() - start) * 1000
        trip_id = getattr(request.state, "trip_id", "-")  # endpoints set this when a trip is involved
        caller = request.client.host if request.client else "-"
        log.info("%-4s %-9s | trip=%-8s | status=%d | %8.1f ms | from %s",
                 request.method, request.url.path, trip_id, status, ms, caller)


@app.get("/health")
def health():
    """Simple liveness check: Laptop 2 calls this to see if the API is reachable."""
    return {"status": "ok"}


@app.get("/trips")
def get_trips():
    """List the trip IDs in the configured trip folder (contract: {"trips": [{"trip_id": ...}]})."""
    if not TRIPS_DIR.is_dir():
        raise HTTPException(status_code=500,
                            detail=f"Trip folder not found: {TRIPS_DIR}. Set PATHSNAP_TRIPS_DIR.")
    return {"trips": [{"trip_id": t} for t in trips.list_trip_ids()]}


class PredictRequest(BaseModel):
    trip_id: str  # a body without a string trip_id is rejected automatically with HTTP 422


@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    """Run the full pipeline for one trip and score it on the server."""
    trip_id = body.trip_id
    request.state.trip_id = trip_id  # shown in the request log line

    # 1. Resolve the trip through the configured trip list.
    try:
        folder = trips.trip_folder(trip_id)
    except trips.TripNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown trip_id '{trip_id}'. See GET /trips.")

    # 2. Load the prediction inputs only (GPS + road network; no ground truth).
    try:
        points = load_points(folder, trip_id)
        network = load_road_network(folder, trip_id)
    except TripDataError as err:
        raise HTTPException(status_code=500, detail=str(err))

    # 3. Predict one arc per GPS point. The predictor never sees ground truth.
    start = time.perf_counter()
    predicted = predict_trip(points, network)
    inference_ms = (time.perf_counter() - start) * 1000
    if len(predicted) != len(points):
        raise HTTPException(status_code=500, detail="Predictor returned the wrong number of arcs.")

    # 4. Build the drawable route from the prediction.
    parts = route.build_route_parts(predicted, network)
    geometry = route.parts_to_geometry(parts, network)
    continuity = route.route_continuity(predicted, network)

    # 5. Score AFTER predicting, on the server. Only aggregate counts come back.
    try:
        metrics = ground_truth.score_trip(folder, trip_id, points, network, predicted)
    except TripDataError as err:
        raise HTTPException(status_code=500, detail=str(err))

    matched = sum(a is not None for a in predicted)
    breaks = sum(p["break_before"] is not None for p in parts)
    if matched == 0:
        status = "no_match"
    elif matched < len(points) or breaks > 0:
        status = "partial"
    else:
        status = "ok"

    acc_text = (f"{metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['aligned']})"
                if metrics else "n/a")
    cont_text = (f"{continuity['value']:.1%} ({continuity['connected']}/{continuity['transitions']})"
                 if continuity["value"] is not None else "n/a")
    log.info("     predict %s | %d points | matched %d | breaks %d | accuracy %s | continuity %s | inference %.0f ms",
             trip_id, len(points), matched, breaks, acc_text, cont_text, inference_ms)

    return {
        # --- shared API-to-UI contract (handoffs/04) ---
        "trip_id": trip_id,
        "status": status,
        "route_geometry": geometry,
        "counts": {"observations": len(points), "matched": matched, "route_breaks": breaks},
        "metrics": metrics,
        # --- extras for the UI ---
        "route_continuity": continuity,
        "gps_points": points[:, :2].tolist(),  # [[lon, lat], ...] observed input, not ground truth
        "matches": [{"obs_idx": i, "arc_id": a, "unmatched_reason": None if a is not None else "no_candidates"}
                    for i, a in enumerate(predicted)],
        "inference_ms": round(inference_ms, 1),
    }
