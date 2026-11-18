"""Audit supplied development trips and generate EDA evidence."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from time import perf_counter

import numpy as np
from pyproj import Geod
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from pathsnap.data.candidates import CandidateConfig, generate_candidates, select_candidates
from pathsnap.data.development import DEVELOPMENT_SPLITS, trip_directories, write_csv, write_json
from pathsnap.data.loading import load_trip
from pathsnap.data.reference_alignment import (
    AlignmentConfig, align_references, read_reference_route, reference_break_positions,
)
from pathsnap.data.reference_alignment.scoring import candidate_coverage

TRAIN_SETTINGS = ((40.0, 12), (60.0, 12), (60.0, 24), (75.0, 32))


def inventory_row(folder, split, trip, route):
    from_node, to_node = trip.arc_nodes.T
    node_count = len(trip.node_lonlat)
    graph = csr_array((np.ones(len(from_node), dtype=np.int32), (from_node, to_node)),
                      shape=(node_count, node_count))
    components, labels = connected_components(graph, directed=True, connection="weak")
    connected_nodes = np.unique(trip.arc_nodes).size
    _, endpoint_counts = np.unique(from_node * node_count + to_node, return_counts=True)
    gaps = np.diff(trip.track[:, 2])
    breaks = reference_break_positions(trip, route)
    lon, lat = trip.track[:, :2].T
    start, end = trip.node_lonlat[trip.arc_nodes].transpose(1, 0, 2)
    _, _, ellipsoidal = Geod(ellps="WGS84").inv(*start.T, *end.T)
    arc_error = np.abs(trip.arc_lengths_m - ellipsoidal)
    row = {
        "split": split, "trip_id": trip.trip_id, "valid": True, "validation_error": "",
        "gps_rows": len(trip.track), "nodes": node_count, "arcs": len(from_node), "route_arcs": len(route),
        "track_lon_min": lon.min(), "track_lon_max": lon.max(), "track_lat_min": lat.min(),
        "track_lat_max": lat.max(),
        "track_duration_s": float(gaps.sum()), "sampling_steps": gaps.size,
        "equal_timestamps": int((gaps == 0).sum()), "one_second_steps": int((gaps == 1).sum()),
        "gaps_gt_1s": int((gaps > 1).sum()), "gaps_gt_10s": int((gaps > 10).sum()),
        "max_gap_s": float(gaps.max(initial=0)),
        "weak_components": components, "isolated_nodes": node_count - connected_nodes,
        "largest_weak_component_nodes": int(np.bincount(labels).max()),
        "duplicate_directed_pairs": int((endpoint_counts > 1).sum()),
        "extra_directed_arcs": int((endpoint_counts - 1).sum()),
        "zero_length_arcs": int((trip.arc_lengths_m == 0).sum()),
        "self_loop_arcs": int((from_node == to_node).sum()),
        "projection_arc_length_abs_error_p95_m": float(np.percentile(arc_error, 95)),
        "projection_arc_length_abs_error_max_m": float(arc_error.max()),
        "route_breaks": len(breaks), "route_break_positions": " ".join(map(str, breaks)),
        "repeated_reference_entries": int(len(route) - np.unique(route).size),
    }
    row.update(file_sizes(folder))
    return row


def file_sizes(folder: Path) -> dict:
    files = {suffix: folder / f"{folder.name}.{suffix}" for suffix in ("track", "nodes", "arcs", "route")}
    return {f"{suffix}_bytes": path.stat().st_size if path.is_file() else None for suffix, path in files.items()}


def _summaries(coverage):
    groups = {}
    category = ("split", "radius_m", "max_candidates")
    metrics = ("observations", "aligned", "ambiguous", "unaligned", "zero_candidates",
               "aligned_zero_candidates", "candidate_rows", "candidate_misses", "null_heading_rows",
               "null_speed_rows", "null_delta_t_rows")
    for row in coverage:
        key = tuple(row[name] for name in category)
        totals = groups.setdefault(key, dict(zip(category, key, strict=True), trips=0, **dict.fromkeys(metrics, 0)))
        totals["trips"] += 1
        for field in metrics:
            totals[field] += row[field]
    for totals in groups.values():
        aligned = totals["aligned"]
        totals["candidate_miss_rate"] = totals["candidate_misses"] / aligned if aligned else None
        totals["candidates_per_observation"] = totals["candidate_rows"] / totals["observations"]
    return list(groups.values())


def run_audit(dataset: Path, output: Path, splits, train_sweep: bool) -> None:
    inventory, coverage, timings = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    default = CandidateConfig()
    for split in splits:
        settings = TRAIN_SETTINGS if train_sweep and split == "train" else ((default.radius_m, default.max_candidates),)
        for folder in trip_directories(dataset, split):
            started = perf_counter()
            try:
                trip = load_trip(folder)
                route = read_reference_route(trip, folder)
            except (OSError, ValueError) as exc:
                inventory.append(dict(split=split, trip_id=folder.name, valid=False, validation_error=str(exc),
                                      **file_sizes(folder)))
                print(f"{split}/{folder.name}: {exc}", flush=True)
                continue
            inventory.append(inventory_row(folder, split, trip, route))
            refs = align_references(trip, route)
            generation_started = perf_counter()
            candidates = generate_candidates(trip, CandidateConfig(*settings[-1]))
            elapsed = perf_counter() - generation_started
            timings.append(dict(split=split, trip_id=trip.trip_id,
                                radius_m=settings[-1][0], max_candidates=settings[-1][1],
                                candidate_generation_s=elapsed))
            obs = np.array([row.obs_idx for row in candidates], dtype=int)
            arcs = np.array([row.arc_id for row in candidates], dtype=int)
            distance = np.array([row.distance_m for row in candidates])
            for radius, cap in settings:
                indices = select_candidates(obs, arcs, distance, CandidateConfig(radius, cap))
                rows = [candidates[i] for i in indices]
                coverage.append(dict(
                    split=split, trip_id=trip.trip_id, radius_m=radius, max_candidates=cap,
                    **candidate_coverage(refs, rows),
                    null_heading_rows=sum(c.heading_diff_deg is None for c in rows),
                    null_speed_rows=sum(c.speed_mps is None for c in rows),
                    null_delta_t_rows=sum(c.delta_t_s is None for c in rows),
                ))
            print(f"Audited {split}/{trip.trip_id} in {perf_counter() - started:.2f}s", flush=True)
    columns = sorted({key for row in inventory for key in row} - {"split", "trip_id", "valid", "validation_error"})
    write_csv(output / "trip_inventory.csv", ["split", "trip_id", "valid", "validation_error", *columns], inventory)
    status = {"trips": len(inventory), "valid_trips": sum(row["valid"] for row in inventory)}
    if coverage:
        write_csv(output / "candidate_coverage_by_trip.csv", coverage[0].keys(), coverage)
        summary = _summaries(coverage)
        write_csv(output / "candidate_coverage_by_split.csv", summary[0].keys(), summary)
        write_csv(output / "candidate_generation_times.csv", timings[0].keys(), timings)
    write_json(output / "audit_metadata.json", {
        "splits": splits, "training_sweep": train_sweep, **status,
        "default_candidate_config": asdict(default), "alignment_config": asdict(AlignmentConfig()),
        "python": python_version(), "packages": {
            name: version(name) for name in ("numpy", "scipy", "pyproj", "shapely")
        },
    })
    if status["valid_trips"] != status["trips"]:
        raise SystemExit("Audit found invalid trips. See trip_inventory.csv.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=DEVELOPMENT_SPLITS, action="append", required=True)
    parser.add_argument("--train-sweep", action="store_true")
    args = parser.parse_args()
    run_audit(args.dataset, args.output, tuple(dict.fromkeys(args.split)), args.train_sweep)


if __name__ == "__main__":
    main()
