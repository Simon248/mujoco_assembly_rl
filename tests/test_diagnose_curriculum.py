"""Tests du diagnostic de sampling, sans simulation ni modèle SAC."""
from types import SimpleNamespace
import unittest

from src.config import load_config
from src.diagnose_curriculum import (
    format_diagnostic,
    lineage_diagnostic,
    virtual_sampling_diagnostic,
)


class CurriculumSamplingDiagnosticTest(unittest.TestCase):
    @staticmethod
    def state(
        state_id: int, parent_id: int | None, depth: int,
        *, pose_distance: float,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            state_id=state_id,
            parent_id=parent_id,
            generation_depth=depth,
            pose_distance=pose_distance,
            position_error=pose_distance,
            rotation_error=pose_distance / 10.0,
        )

    def test_lineage_reports_boundary_depths_and_non_monotonic_branch(self):
        # La profondeur suit A -> B -> C bien que 9 -> 6 se rapproche du goal.
        a = self.state(1, None, 1, pose_distance=4.0)
        b = self.state(2, 1, 2, pose_distance=9.0)
        c = self.state(3, 2, 3, pose_distance=6.0)
        d = self.state(4, 1, 2, pose_distance=50.0)
        pools = {
            "mastered": [a, b],
            "frontier": [c],
            "too_hard": [d],
        }

        result = lineage_diagnostic(pools)

        self.assertEqual(result["total_states"], 4)
        self.assertEqual(result["max_generation_depth"], 3)
        self.assertEqual(result["mastered_boundary"]["count"], 1)
        self.assertEqual(result["mastered_boundary"]["depth"]["max"], 2.0)
        deepest = result["deepest_lineages"][0]
        self.assertEqual(deepest["path"], "goal → state_1 → state_2 → state_3")
        self.assertEqual(deepest["depth"], 3)
        self.assertEqual(deepest["classification"], "frontier")

    def test_lineage_marks_legacy_roots_and_limits_deepest_branches(self):
        pools = {"mastered": [], "frontier": [], "too_hard": []}
        for state_id in range(7):
            pools["too_hard"].append(self.state(
                state_id, None, 0, pose_distance=float(state_id),
            ))

        result = lineage_diagnostic(pools)

        self.assertEqual(len(result["deepest_lineages"]), 5)
        self.assertTrue(all(
            item["path"].startswith("legacy_root → state_")
            for item in result["deepest_lineages"]
        ))

    def test_virtual_sampling_reports_sources_and_every_historical_bin(self):
        config = load_config("configs/test1V21.yaml")["curriculum"]
        frontier = [self.state(1, None, 1, pose_distance=.006)]
        historical = [
            self.state(index, None, depth, pose_distance=value)
            for index, (depth, value) in enumerate(
                zip(
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    [.040, .001, .030, .002, .020, .003, .010, .004],
                ),
                10,
            )
        ]
        result = virtual_sampling_diagnostic(
            config, frontier, historical, draws=10_000, seed=21,
        )
        self.assertAlmostEqual(result["fractions"]["frontier"], .50, delta=.02)
        self.assertAlmostEqual(result["fractions"]["historical"], .30, delta=.02)
        self.assertAlmostEqual(result["fractions"]["true_start"], .20, delta=.02)
        self.assertEqual(len(result["historical_bins"]), 4)
        self.assertTrue(all(
            item["size"] > 0
            and item["selection_fraction_within_historical"] > .20
            for item in result["historical_bins"]
        ))
        self.assertEqual(
            [(item["depth_min"], item["depth_max"])
             for item in result["historical_bins"]],
            [(0, 0), (1, 1), (2, 2), (3, 3)],
        )

    def test_human_output_contains_required_sections(self):
        fixture = {
            "generated": 1,
            "valid": 1,
            "restoration_failures": 0,
            "expansion_efficiency": {
                "expansion_branches": 3,
                "expansion_candidates": 8,
                "expansion_hops": 10,
                "new_mastered": 5,
                "new_frontier": 2,
                "new_too_hard": 1,
                "expansion_rollouts": 40,
                "revalidation_mastered_rollouts": 40,
                "revalidation_too_hard_rollouts": 60,
                "expansion_scale_mean": 1.25,
                "expansion_scale_max": 1.5625,
                "frontier_found_per_candidate": .25,
                "expansion_wall_time": 2.5,
                "revalidation_wall_time": 1.5,
            },
            "sampling_pool_source": "fixture",
            "pool_sizes": {"frontier": 1, "mastered": 1, "too_hard": 0},
            "frontier_distance": {
                "min": .002, "q25": .002, "median": .002,
                "q75": .002, "max": .002,
            },
            "historical_distance": {
                "min": .001, "q25": .001, "median": .001,
                "q75": .001, "max": .001,
            },
            "too_hard_distance": {
                "min": None, "q25": None, "median": None,
                "q75": None, "max": None,
            },
            "lineage": {
                "total_states": 2,
                "max_generation_depth": 2,
                "pools": {
                    "mastered": {
                        "count": 1,
                        "depth": {"min": 1.0, "median": 1.0, "max": 1.0},
                    },
                    "frontier": {
                        "count": 1,
                        "depth": {"min": 2.0, "median": 2.0, "max": 2.0},
                    },
                    "too_hard": {
                        "count": 0,
                        "depth": {"min": None, "median": None, "max": None},
                    },
                },
                "mastered_boundary": {
                    "count": 1,
                    "depth": {"min": 1.0, "median": 1.0, "max": 1.0},
                },
                "deepest_lineages": [{
                    "path": "goal → state_1 → state_2",
                    "depth": 2,
                    "classification": "frontier",
                    "position_error": .002,
                    "rotation_error": .01,
                }],
            },
            "sampling_test": {
                "draws": 10_000,
                "fractions": {
                    "frontier": .5, "historical": .3, "true_start": .2,
                },
                "historical_bins": [{
                    "bin": 1, "size": 1, "min": .001, "max": .001,
                    "depth_min": 1, "depth_max": 1,
                    "selection_fraction_within_historical": 1.0,
                }],
            },
        }
        output = format_diagnostic(fixture)
        for heading in (
            "EXPANSION EFFICIENCY", "POOL SIZES", "LINEAGE",
            "DEEPEST LINEAGES",
            "GEOMETRY DIAGNOSTICS", "FRONTIER DISTANCE",
            "MASTERED/HISTORICAL DISTANCE", "TOO HARD DISTANCE",
            "SAMPLING TEST", "HISTORICAL DEPTH BINS",
        ):
            self.assertIn(heading, output)
        self.assertIn("max generation depth: 2", output)
        self.assertIn("goal → state_1 → state_2", output)
        self.assertIn("branches expanded: 3", output)
        self.assertIn("candidates generated: 8", output)
        self.assertIn("new frontier: 2", output)
        self.assertIn("frontier found per candidate: 0.25", output)
        self.assertIn("mastered revalidation: 40", output)
        self.assertIn("too-hard revalidation: 60", output)

    def test_expansion_efficiency_is_explicitly_unavailable_for_old_results(self):
        fixture = {
            "generated": 0,
            "valid": 0,
            "restoration_failures": 0,
            "sampling_pool_source": "fixture",
            "pool_sizes": {"frontier": 0, "mastered": 0, "too_hard": 0},
            "frontier_distance": {
                key: None for key in ("min", "q25", "median", "q75", "max")
            },
            "historical_distance": {
                key: None for key in ("min", "q25", "median", "q75", "max")
            },
            "too_hard_distance": {
                key: None for key in ("min", "q25", "median", "q75", "max")
            },
            "lineage": {
                "total_states": 0,
                "max_generation_depth": 0,
                "pools": {
                    name: {
                        "count": 0,
                        "depth": {"min": None, "median": None, "max": None},
                    }
                    for name in ("mastered", "frontier", "too_hard")
                },
                "mastered_boundary": {
                    "count": 0,
                    "depth": {"min": None, "median": None, "max": None},
                },
                "deepest_lineages": [],
            },
            "sampling_test": {
                "draws": 0,
                "fractions": {
                    "frontier": 0.0, "historical": 0.0, "true_start": 1.0,
                },
                "historical_bins": [],
            },
        }

        output = format_diagnostic(fixture)

        self.assertIn("EXPANSION EFFICIENCY", output)
        self.assertIn("branches expanded: unavailable (not persisted)", output)


if __name__ == "__main__":
    unittest.main()
