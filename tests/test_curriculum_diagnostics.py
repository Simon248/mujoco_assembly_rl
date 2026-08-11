import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from src.curriculum_diagnostics import (
    ExpansionDiagnostics, curriculum_state_rows, write_curriculum_diagnostics,
)


class CurriculumDiagnosticsTest(unittest.TestCase):
    def _manager(self):
        state = SimpleNamespace(
            state_id=7, parent_id=3, generation_depth=2, success_rate=.6,
            position_error=.01, rotation_error=.02, pose_distance=.012,
        )
        report = SimpleNamespace(
            expansion_branches=1, expansion_hops=2, expansion_attempts=3,
            raw_candidates_generated=2, valid_candidates=2,
            nonduplicate_candidates=1, qualified_candidates=1,
            safe_prefix_candidates=1, full_walk_candidates=1,
            attempt_duplicate=1, attempt_no_candidate=1,
            attempts_per_hop=[1, 2], stop_reasons={"frontier": 1},
            safe_prefix_steps=[2],
            raw_parent_translation_mm=[1.0, 3.0],
            raw_parent_rotation_deg=[2.0, 4.0],
            persistent_attempts=3, independent_attempts=0,
            branch_heading_changes=[.1, .3],
            attempt_to_heading_deviations=[.2, .4],
            successive_hop_heading_opposition=1,
            guided_memory_insertions=1,
            guided_memory_rejected_duplicates=1,
            rejected_force_max=[80.0], rejected_torque_max=[],
            candidate_final_force=[4.0], candidate_final_torque=[.3],
            new_mastered=0, new_frontier=1, new_too_hard=0,
            expansion_wall_time=.5,
            **{
                f"proposal_{kind}_{suffix}": 0
                for kind in ("uniform", "guided")
                for suffix in (
                    "attempts", "candidates", "unique", "safe_prefix",
                    "attempt_budget_failures",
                )
            },
        )
        report.proposal_uniform_attempts = 3
        report.proposal_uniform_candidates = 2
        report.proposal_uniform_unique = 1
        lifecycle = SimpleNamespace(
            created_update=1, last_revalidated_update=2,
            revalidation_count=2, frontier_since_update=1,
            consecutive_frontier_updates=2,
        )
        return SimpleNamespace(
            last_generation_report=report,
            last_revalidation_report=SimpleNamespace(
                wall_time=.2, frontier_promoted_to_mastered=0,
                frontier_remained_frontier=1,
                frontier_demoted_to_too_hard=0,
            ),
            pools={"mastered": [], "frontier": [state], "too_hard": []},
            state_lifecycle={7: lifecycle}, update_count=2,
            mastered_boundary_states=lambda: [],
        )

    def test_one_expansion_row_and_one_row_per_state_per_update(self):
        manager = self._manager()
        targets = SimpleNamespace(
            true_start=.2, frontier=.5, historical=.3,
            historical_fraction_effective=.375,
        )
        next_targets = SimpleNamespace(
            true_start=.7, frontier=.2, historical=.1,
        )
        diagnostics = ExpansionDiagnostics.build(
            manager, 1000, targets,
            {"true_start": .2, "frontier": .5, "historical": .3},
            next_targets,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = curriculum_state_rows(manager, 1000)
            write_curriculum_diagnostics(output, diagnostics, rows)
            write_curriculum_diagnostics(output, diagnostics, rows)
            with (output / "curriculum_expansion.csv").open() as stream:
                expansion = list(csv.DictReader(stream))
            with (output / "curriculum_states.csv").open() as stream:
                states = list(csv.DictReader(stream))
        self.assertEqual(len(expansion), 2)
        self.assertEqual(len(states), 2)
        self.assertEqual(expansion[0]["qualified_candidates"], "1")
        self.assertEqual(expansion[0]["sampling_target_used_frontier"], "0.5")
        self.assertEqual(expansion[0]["sampling_observed_frontier"], "0.5")
        self.assertEqual(expansion[0]["sampling_target_next_frontier"], "0.2")
        self.assertNotIn("sampling_target_frontier", expansion[0])
        self.assertEqual(expansion[0]["persistent_attempts"], "3")
        self.assertEqual(expansion[0]["branch_heading_changes_mean"], "0.2")
        self.assertEqual(
            expansion[0]["attempt_to_heading_deviation_max"], "0.4",
        )
        self.assertEqual(
            expansion[0]["successive_hop_heading_opposition"], "1",
        )
        self.assertEqual(expansion[0]["frontier_position_max"], "0.01")
        self.assertEqual(expansion[0]["guided_memory_insertions"], "1")
        self.assertEqual(
            expansion[0]["reverse_candidate_parent_delta_position_mm_mean"],
            "2.0",
        )
        self.assertEqual(expansion[0]["safe_prefix_steps_max"], "2.0")
        self.assertEqual(states[0]["state_id"], "7")


if __name__ == "__main__":
    unittest.main()
