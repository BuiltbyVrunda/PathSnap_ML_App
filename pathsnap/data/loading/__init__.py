"""Read the inference inputs and preserve their source row IDs."""

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class Trip:
    trip_id: str
    track: NDArray[np.float64]
    node_lonlat: NDArray[np.float64]
    arc_nodes: NDArray[np.int64]
    observation_xy: NDArray[np.float64] = field(repr=False)
    node_xy: NDArray[np.float64] = field(repr=False)
    arc_lengths_m: NDArray[np.float64] = field(repr=False)

    def to_record(self) -> dict:
        """Convert array-backed data to the shared v1 trip record."""
        return {
            "trip_id": self.trip_id,
            "observations": [
                dict(obs_idx=i, lon=lon, lat=lat, time_s=time_s)
                for i, (lon, lat, time_s) in enumerate(self.track.tolist())
            ],
            "nodes": [
                dict(node_id=i, lon=lon, lat=lat)
                for i, (lon, lat) in enumerate(self.node_lonlat.tolist())
            ],
            "arcs": [
                dict(arc_id=i, from_node_id=start, to_node_id=end, length_m=length)
                for i, ((start, end), length) in enumerate(
                    zip(self.arc_nodes.tolist(), self.arc_lengths_m.tolist(), strict=True)
                )
            ],
        }


def read_numeric(path: Path, columns: int, dtype: type) -> NDArray:
    """Parse strict whitespace-separated rows and report physical file locations."""
    source = path.read_text(encoding="utf-8")
    # loadtxt ignores blank lines by design. Reject them to preserve row identity.
    for line_number, line in enumerate(source.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank row")
    if not source:
        raise ValueError(f"{path}: empty file")
    try:
        converter = int if dtype == np.int64 else None
        rows = np.loadtxt(StringIO(source), dtype=dtype, comments=None, ndmin=2, converters=converter)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if rows.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, found {rows.shape[1]}")
    require_rows(path, np.isfinite(rows).all(axis=1), "non-finite value")
    return rows


def require_rows(path: Path, valid: NDArray, message: str) -> None:
    invalid = np.flatnonzero(~valid)
    if invalid.size:
        raise ValueError(f"{path}:{invalid[0] + 1}: {message} ({invalid.size} invalid rows)")


def _check_lonlat(path: Path, rows: NDArray) -> None:
    valid = (np.abs(rows[:, 0]) <= 180) & (np.abs(rows[:, 1]) <= 90)
    require_rows(path, valid, "longitude or latitude out of range")


def load_trip(trip_dir: str | Path) -> Trip:
    """Read only .track, .nodes and .arcs from a named trip directory."""
    folder = Path(trip_dir)
    trip_id = folder.name
    track_path, nodes_path, arcs_path = [
        folder / f"{trip_id}.{suffix}" for suffix in ("track", "nodes", "arcs")
    ]
    track = read_numeric(track_path, 3, np.float64)
    nodes = read_numeric(nodes_path, 2, np.float64)
    arcs = read_numeric(arcs_path, 2, np.int64)
    _check_lonlat(track_path, track)
    _check_lonlat(nodes_path, nodes)
    require_rows(track_path, np.r_[True, np.diff(track[:, 2]) >= 0], "timestamp decreased")
    require_rows(arcs_path, ((arcs >= 0) & (arcs < len(nodes))).all(axis=1), "invalid node ID")

    # Circular mean keeps a trip near the antimeridian centered on its own data.
    radians = np.deg2rad(track[:, 0])
    lon0 = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    lat0 = track[:, 1].mean()
    crs = CRS.from_proj4(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +datum=WGS84 +units=m")
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    observation_xy = np.column_stack(project.transform(track[:, 0], track[:, 1], errcheck=True))
    node_xy = np.column_stack(project.transform(nodes[:, 0], nodes[:, 1], errcheck=True))
    start, end = node_xy[arcs].transpose(1, 0, 2)
    length_m = np.linalg.norm(end - start, axis=1)
    for array in (track, nodes, arcs, observation_xy, node_xy, length_m):
        array.flags.writeable = False
    return Trip(trip_id, track, nodes, arcs, observation_xy, node_xy, length_m)
