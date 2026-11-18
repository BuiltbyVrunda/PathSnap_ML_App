import tempfile
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

from pathsnap.data.candidates import CandidateConfig, generate_candidates
from pathsnap.data.loading import load_trip
from pathsnap.data.reference_alignment import (
    AlignmentConfig, Reference, _ordered_alignment, align_references, read_reference_route,
    reference_break_positions,
)
from pathsnap.data.reference_alignment.scoring import candidate_coverage, score_predictions, score_trip


class MemberOneTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.folder = Path(workspace.name) / "train" / "00000001"
        self.folder.mkdir(parents=True)
        self.write("track", "0.00025 0 1\n0.00175 0 11\n")
        self.write("nodes", "0 0\n0.001 0\n0.002 0\n0 0.0001\n0.001 0.0001\n0.002 0.0001\n")
        self.write("arcs", "0 1\n1 2\n3 4\n4 5\n")
        self.write("route", "0\n1\n")

    def write(self, suffix, data):
        (self.folder / f"00000001.{suffix}").write_text(data, encoding="utf-8")

    def test_full_contract_without_reference_during_inference(self):
        trip = load_trip(self.folder)
        config = CandidateConfig(30.0, 4)
        candidates = generate_candidates(trip, config)
        self.assertEqual(trip.to_record()["observations"][0],
                         dict(obs_idx=0, lon=0.00025, lat=0.0, time_s=1.0))
        self.assertAlmostEqual(trip.arc_lengths_m[0], 111.319, places=2)
        self.assertEqual([(c.obs_idx, c.arc_id) for c in candidates], [(0, 0), (0, 2), (1, 1), (1, 3)])
        np.testing.assert_allclose([c.along_arc_fraction for c in candidates], [.25, .25, .75, .75], atol=1e-5)
        self.assertAlmostEqual(candidates[1].distance_m, 11.057, places=2)
        self.assertIsNone(candidates[0].heading_diff_deg)
        self.assertAlmostEqual(candidates[2].heading_diff_deg, 0, places=5)
        self.assertAlmostEqual(candidates[2].speed_mps, 16.698, places=2)
        references = align_references(trip, read_reference_route(trip, self.folder))
        self.assertEqual([r.ref_arc_id for r in references], [0, 1])
        self.assertEqual(candidate_coverage(references, candidates)["candidate_miss_rate"], 0)
        self.assertEqual(candidate_coverage(references, [c for c in candidates if c.arc_id != 1])["candidate_miss_rate"], .5)
        self.assertEqual(candidate_coverage(references, candidates[:2])["aligned_zero_candidates"], 1)
        self.assertEqual(score_trip(trip, self.folder, {0: 0, 1: None})["accuracy"], .5)
        (self.folder / "00000001.route").unlink()
        self.assertEqual(trip.to_record(), load_trip(self.folder).to_record())
        self.assertEqual(candidates, generate_candidates(load_trip(self.folder), config))

    def test_duplicate_arcs_route_break_and_zero_length(self):
        self.write("arcs", "0 1\n1 2\n3 4\n4 5\n0 1\n5 5\n")
        self.write("route", "0\n4\n1\n")
        trip = load_trip(self.folder)
        route = read_reference_route(trip, self.folder)
        np.testing.assert_array_equal(reference_break_positions(trip, route), [1])
        candidates = generate_candidates(trip, CandidateConfig(35, 8))
        self.assertEqual([c.arc_id for c in candidates if c.obs_idx == 0], [0, 2, 4])
        self.assertEqual(generate_candidates(trip, CandidateConfig(35, 1))[0].arc_id, 0)
        degenerate = next(c for c in candidates if c.arc_id == 5)
        self.assertEqual(degenerate.along_arc_fraction, 0)
        self.assertIsNone(degenerate.heading_diff_deg)
        labels = align_references(trip, route)
        self.assertEqual(labels[0].alignment_status, "ambiguous")
        self.assertIsNone(labels[0].ref_arc_id)
        self.assertEqual(labels[1].ref_arc_id, 1)
        labels = align_references(trip, np.array([0, 0, 1]))
        self.assertEqual([ref.ref_arc_id for ref in labels], [0, 1])

    def test_no_candidates_preserves_ids_and_handles_gaps(self):
        self.write("track", "0.00025 0 1\n0.00175 0 27\n0.010 0.010 27\n")
        trip = load_trip(self.folder)
        candidates = generate_candidates(trip, CandidateConfig(30, 4))
        self.assertEqual({c.obs_idx for c in candidates}, {0, 1})
        self.assertEqual(trip.to_record()["observations"][2]["obs_idx"], 2)
        self.assertIsNone(candidates[2].heading_diff_deg)
        self.assertEqual(candidates[2].delta_t_s, 26)
        refs = align_references(trip, read_reference_route(trip, self.folder), AlignmentConfig(radius_m=50))
        self.assertEqual([r.alignment_status for r in refs], ["aligned", "aligned", "unaligned"])
        self.assertEqual(candidate_coverage(refs, candidates)["zero_candidates"], 1)

    def test_validation_errors_include_file_locator(self):
        for invalid in ("0 99", "0 1.5", "0 1\n\n1 2"):
            with self.subTest(invalid=invalid):
                self.write("arcs", invalid)
                with self.assertRaisesRegex(ValueError, r"00000001\.arcs:"):
                    load_trip(self.folder)
        self.write("arcs", "0 1\n")
        trip = load_trip(self.folder)
        with self.assertRaisesRegex(ValueError, r"00000001\.route:2:"):
            read_reference_route(trip, self.folder)
        for invalid in ("nan 0 1", "181 0 1", "0 0 2\n0 0 1"):
            with self.subTest(invalid=invalid):
                self.write("track", invalid)
                with self.assertRaisesRegex(ValueError, r"00000001\.track:"):
                    load_trip(self.folder)

    def test_one_observation_and_trip_qualified_metrics(self):
        self.write("track", "0.00025 0 1\n")
        trip = load_trip(self.folder)
        candidates = generate_candidates(trip)
        self.assertIsNone(candidates[0].speed_mps)
        references = align_references(trip, read_reference_route(trip, self.folder))
        second_trip = replace(references[0], trip_id="00000002")
        scores = score_predictions([*references, second_trip], {("00000001", 0): 0})
        self.assertEqual((scores["accuracy"], scores["macro_f1"], scores["macro_f1_classes"]), (.5, .5, 2))
        empty = score_predictions([Reference("00000003", 0, None, "ambiguous")], {})
        self.assertIsNone(empty["accuracy"])
        self.assertIsNone(empty["macro_f1"])

    def test_dynamic_program_matches_exhaustive_monotone_paths(self):
        rng = np.random.default_rng(17)
        for _ in range(24):
            positions = np.tile(np.arange(3), 4)
            rewards = rng.integers(1, 12, size=len(positions)).astype(float)
            offsets = np.arange(5) * 3
            path, score, through = _ordered_alignment(positions, rewards, offsets, 3)
            best = np.zeros(len(positions))
            for choices in product(range(-1, 3), repeat=4):
                matched = [pos for pos in choices if pos >= 0]
                if matched != sorted(matched):
                    continue
                states = [i * 3 + pos for i, pos in enumerate(choices) if pos >= 0]
                best[states] = np.maximum(best[states], rewards[states].sum())
            self.assertEqual(score, best.max())
            self.assertEqual(rewards[path].sum(), score)
            np.testing.assert_array_equal(through, best)


if __name__ == "__main__":
    unittest.main()
