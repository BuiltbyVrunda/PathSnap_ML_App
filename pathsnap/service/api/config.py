"""Server settings. Read from environment variables so no path is hardcoded."""
import os
from pathlib import Path

# Repo root = four levels up from this file (pathsnap/service/api/config.py).
REPO_ROOT = Path(__file__).resolve().parents[3]

# Folder with one sub-folder per trip: <TRIPS_DIR>/<trip_id>/<trip_id>.track, .nodes, .arcs, .route
# Development default: the student-test trips. On demo day, point it at the evaluator trips:
#   export PATHSNAP_TRIPS_DIR=/path/to/evaluator_trips
TRIPS_DIR = Path(
    os.environ.get("PATHSNAP_TRIPS_DIR", REPO_ROOT / "data" / "student_test")
).resolve()
