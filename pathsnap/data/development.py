"""Export reproducible development candidate and reference tables one trip at a time."""

import argparse
import csv
import gzip
import json
from dataclasses import asdict, fields
from pathlib import Path

from pathsnap.data.candidates import Candidate, CandidateConfig, generate_candidates
from pathsnap.data.loading import load_trip
from pathsnap.data.reference_alignment import AlignmentConfig, Reference, align_references, read_reference_route
from pathsnap.data.reference_alignment.scoring import candidate_coverage

DEVELOPMENT_SPLITS = ("train", "student_test")


def trip_directories(root: Path, split: str) -> list[Path]:
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError(f"expected a development split from {DEVELOPMENT_SPLITS}")
    folders = sorted(path for path in (root / split).iterdir() if path.is_dir())
    if not folders:
        raise ValueError(f"{root / split}: no trip directories")
    return folders


def write_csv(path: Path, columns, rows) -> None:
    open_file = gzip.open if path.suffix == ".gz" else Path.open
    with open_file(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=DEVELOPMENT_SPLITS, required=True)
    parser.add_argument("--trip-id", help="export one trip from the chosen split")
    parser.add_argument("--radius-m", type=float, default=CandidateConfig.radius_m)
    parser.add_argument("--max-candidates", type=int, default=CandidateConfig.max_candidates)
    args = parser.parse_args()
    config = CandidateConfig(args.radius_m, args.max_candidates)
    alignment = AlignmentConfig()
    folders = trip_directories(args.dataset, args.split)
    folders = [folder for folder in folders if args.trip_id is None or folder.name == args.trip_id]
    if not folders:
        parser.error(f"trip {args.trip_id!r} was not found in {args.split}")
    for folder in folders:
        trip = load_trip(folder)
        candidates = generate_candidates(trip, config)
        references = align_references(trip, read_reference_route(trip, folder), alignment)
        output = args.output / args.split / trip.trip_id
        output.mkdir(parents=True, exist_ok=True)
        summary = output / "coverage.json"
        writer_lock = output / ".exporting"
        try:
            writer_lock.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(f"{output}: another writer holds the export folder") from exc
        try:
            # Publish the completion record only after both tables have been written.
            summary.unlink(missing_ok=True)
            write_csv(output / "candidates.csv.gz", [f.name for f in fields(Candidate)], map(asdict, candidates))
            write_csv(output / "references.csv.gz", [f.name for f in fields(Reference)], map(asdict, references))
            staged_summary = writer_lock / "coverage.json"
            write_json(staged_summary, {
                "split": args.split, "trip_id": trip.trip_id,
                "candidate_config": asdict(config), "alignment_config": asdict(alignment),
                **candidate_coverage(references, candidates),
            })
            staged_summary.replace(summary)
        finally:
            (writer_lock / "coverage.json").unlink(missing_ok=True)
            writer_lock.rmdir()
        print(f"Exported {args.split}/{trip.trip_id}: {len(candidates):,} candidate rows", flush=True)


if __name__ == "__main__":
    main()
