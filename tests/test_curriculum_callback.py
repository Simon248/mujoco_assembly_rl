"""Tests purs du logging et de la diffusion RCG dans le callback SB3."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

from stable_baselines3.common.logger import HumanOutputFormat, Logger

from src.train import ReverseCurriculumCallback


class _Logger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.dumps: list[int] = []

    def record(
        self, key: str, value: float,
        exclude: str | tuple[str, ...] | None = None,
    ) -> None:
        self.values[key] = value

    def dump(self, step: int) -> None:
        self.dumps.append(step)


class _Manager:
    worker_rng_states = None
    next_update_timesteps = 50_000

    def __init__(self) -> None:
        state = lambda distance, depth: SimpleNamespace(
            pose_distance=distance, generation_depth=depth,
        )
        self.pools = {
            "too_hard": [state(.040, 7)],
            "frontier": [state(.006, 5)],
            "mastered": [state(.001, 2), state(.010, 4)],
        }
        self.last_expansion_seed_distances = [.008, .009, .010]
        self.last_expansion_seed_depths = [2, 4, 6]
        self.last_generation_report = SimpleNamespace(
            expansion_candidates=8,
            expansion_hops=10,
            expansion_branches=3,
            expansion_rollouts=40,
            new_mastered=5,
            new_frontier=2,
            new_too_hard=1,
            mean_hops_per_branch=10 / 3,
            max_hops_reached=4,
            expansion_scale_mean=1.25,
            expansion_scale_max=1.5625,
            frontier_found_per_candidate=.25,
            expansion_wall_time=2.5,
            stop_reasons={"duplicate": 2, "frontier": 1},
            raw_candidates_generated=7,
            valid_candidates=6,
            nonduplicate_candidates=4,
            qualified_candidates=4,
            raw_parent_translation_mm=[.2, .4],
            raw_parent_rotation_deg=[1.0, 3.0],
            duplicate_parent_translation_mm=[.2],
            duplicate_parent_rotation_deg=[1.0],
            duplicate_nearest_position_mm=[.1],
            duplicate_nearest_rotation_deg=[.5],
            reverse_steps=[2, 4],
            safe_prefix_steps=[2],
            persistent_attempts=6, independent_attempts=4,
            branch_heading_changes=[.1, .3],
            attempt_to_heading_deviations=[.2, .4],
            successive_hop_heading_opposition=1,
            guided_memory_insertions=4,
            guided_memory_rejected_duplicates=2,
        )
        self.last_revalidation_report = SimpleNamespace(
            too_hard_revalidated=12,
            too_hard_to_frontier=3,
            too_hard_to_mastered=1,
            too_hard_remained_hard=8,
            mastered_rollouts=40,
            too_hard_rollouts=60,
            wall_time=1.5,
        )

    def training_reset_pools(self):
        return {
            "frontier": list(self.pools["frontier"]),
            "historical": list(self.pools["mastered"]),
        }

    def pool_sizes(self):
        return {name: len(states) for name, states in self.pools.items()}

    def frontier_success_rate_mean(self):
        return .6

    def mastered_boundary_states(self):
        return [self.pools["mastered"][-1]]


class _Workers:
    def __init__(self) -> None:
        self.calls = []

    def env_method(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return []


class ReverseCurriculumCallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = _Manager()
        self.workers = _Workers()
        self.callback = ReverseCurriculumCallback(
            self.manager, self.workers, Path("/tmp"), "sac", 50_000,
        )
        self.logger = _Logger()
        self.callback.model = SimpleNamespace(
            logger=self.logger, num_timesteps=0,
        )

    def _step(self, sources, dones, lengths=None, successes=None) -> None:
        lengths = lengths or [None] * len(sources)
        successes = successes or [False] * len(sources)
        infos = []
        for source, done, length, success in zip(
            sources, dones, lengths, successes,
        ):
            info = {
                "reset_source": source,
                "safe_success": success,
            }
            if source == "curriculum_frontier":
                info["curriculum_start_pose_distance"] = .006
            elif source == "curriculum_historical":
                info["curriculum_start_pose_distance"] = .001
            if done:
                info["episode"] = {"l": length}
            infos.append(info)
        self.callback.locals = {"dones": dones, "infos": infos}
        self.callback._on_step()

    def test_broadcast_keeps_frontier_and_historical_separate(self):
        self.callback._broadcast_pool()
        method, args, kwargs = self.workers.calls[-1]
        self.assertEqual(method, "set_curriculum_reset_pools")
        self.assertEqual(args[0], self.manager.pools["frontier"])
        self.assertEqual(args[1], self.manager.pools["mastered"])
        self.assertNotIn(self.manager.pools["too_hard"][0], args[0] + args[1])
        self.assertEqual(kwargs, {})

    def test_episode_and_transition_metrics_are_distinct(self):
        sources = [
            "curriculum_frontier", "curriculum_historical", "true_start",
        ]
        self._step(sources, [False, False, False])
        self._step(sources, [True, False, False], [2, None, None])
        self._step(
            ["curriculum_historical", "true_start"], [True, False],
            [3, None], [True, False],
        )
        self._step(["true_start"], [True], [4])
        self.callback._record_metrics(
            include_pool_metrics=True, include_update_metrics=True,
        )

        values = self.logger.values
        self.assertAlmostEqual(values["curriculum/reset_fraction_total"], 2 / 3)
        self.assertAlmostEqual(values["curriculum/reset_fraction_frontier"], 1 / 3)
        self.assertAlmostEqual(values["curriculum/reset_fraction_historical"], 1 / 3)
        self.assertAlmostEqual(values["curriculum/reset_fraction_true_start"], 1 / 3)
        self.assertAlmostEqual(
            values["curriculum/transition_fraction_frontier"], 2 / 9,
        )
        self.assertAlmostEqual(
            values["curriculum/transition_fraction_historical"], 3 / 9,
        )
        self.assertAlmostEqual(
            values["curriculum/transition_fraction_true_start"], 4 / 9,
        )
        self.assertEqual(values["curriculum/episode_length_frontier_mean"], 2)
        self.assertEqual(values["curriculum/episode_length_historical_mean"], 3)
        self.assertEqual(values["curriculum/episode_length_true_start_mean"], 4)
        self.assertEqual(values["curriculum/success_rate_historical"], 1)
        self.assertEqual(values["curriculum/historical_pool_size"], 2)
        self.assertEqual(values["curriculum/historical_pose_distance_min"], .001)
        self.assertEqual(values["curriculum/historical_pose_distance_max"], .010)
        self.assertEqual(values["curriculum/mastered_max_pose_distance"], .010)
        self.assertEqual(values["curriculum/mastered_pose_distance_max"], .010)
        self.assertEqual(values["curriculum/historical_pose_distance_median"], .0055)
        self.assertEqual(values["curriculum/frontier_pose_distance_q25"], .006)
        self.assertEqual(values["curriculum/too_hard_pose_distance_q75"], .040)
        self.assertEqual(values["curriculum/max_generation_depth"], 7)
        self.assertEqual(values["curriculum/mastered_max_depth"], 4)
        self.assertEqual(values["curriculum/frontier_max_depth"], 5)
        self.assertEqual(values["curriculum/too_hard_max_depth"], 7)
        self.assertEqual(values["curriculum/mastered_boundary_count"], 1)
        self.assertEqual(
            values["curriculum/used_start_distance_frontier_mean"], .006,
        )
        self.assertEqual(
            values["curriculum/used_start_distance_historical_median"], .001,
        )
        self.assertEqual(values["curriculum/success_rate_frontier"], 0)
        self.assertEqual(values["curriculum/success_rate_true_start"], 0)
        self.assertEqual(values["curriculum/expansion_seed_distance_min"], .008)
        self.assertAlmostEqual(
            values["curriculum/expansion_seed_distance_mean"], .009,
        )
        self.assertEqual(values["curriculum/expansion_seed_distance_max"], .010)
        self.assertEqual(values["curriculum/expansion_seed_depth_mean"], 4)
        self.assertEqual(values["curriculum/expansion_seed_depth_max"], 6)
        self.assertEqual(values["curriculum/too_hard_revalidated"], 12)
        self.assertEqual(values["curriculum/too_hard_to_frontier"], 3)
        self.assertEqual(values["curriculum/too_hard_to_mastered"], 1)
        self.assertEqual(values["curriculum/too_hard_remained_hard"], 8)
        expected_update_metrics = {
            "expansion_candidates": 8,
            "expansion_hops": 10,
            "expansion_branches": 3,
            "expansion_rollouts": 40,
            "revalidation_mastered_rollouts": 40,
            "revalidation_too_hard_rollouts": 60,
            "new_mastered": 5,
            "new_frontier": 2,
            "new_too_hard": 1,
            "mean_hops_per_branch": 10 / 3,
            "max_hops_reached": 4,
            "expansion_scale_mean": 1.25,
            "expansion_scale_max": 1.5625,
            "frontier_found_per_candidate": .25,
            "expansion_wall_time": 2.5,
            "revalidation_wall_time": 1.5,
        }
        for name, expected in expected_update_metrics.items():
            self.assertAlmostEqual(values[f"curriculum/{name}"], expected)
        self.assertEqual(values["curriculum/stop_duplicate"], 2)
        self.assertEqual(values["curriculum/stop_frontier"], 1)
        self.assertEqual(values["curriculum/stop_workspace"], 0)
        self.assertEqual(values["curriculum/raw_candidates_generated"], 7)
        self.assertEqual(values["curriculum/valid_candidates"], 6)
        self.assertEqual(values["curriculum/nonduplicate_candidates"], 4)
        self.assertEqual(values["curriculum/qualified_candidates"], 4)
        self.assertAlmostEqual(values["curriculum/raw_candidate_rate"], .7)
        self.assertAlmostEqual(values["curriculum/qualification_rate"], .4)
        self.assertAlmostEqual(
            values["curriculum/raw_parent_translation_mm_mean"], .3,
        )
        self.assertEqual(values["curriculum/raw_parent_rotation_deg_max"], 3)
        self.assertEqual(values["curriculum/reverse_steps_min"], 2)
        self.assertEqual(values["curriculum/persistent_attempts"], 6)
        self.assertEqual(values["curriculum/independent_attempts"], 4)
        self.assertAlmostEqual(
            values["curriculum/branch_heading_changes_mean"], .2,
        )
        self.assertEqual(
            values["curriculum/attempt_to_heading_deviation_max"], .4,
        )
        self.assertEqual(
            values["curriculum/successive_hop_heading_opposition"], 1,
        )
        self.assertEqual(values["curriculum/guided_memory_insertions"], 4)
        self.assertEqual(
            values["curriculum/guided_memory_rejected_duplicates"], 2,
        )
        self.assertAlmostEqual(
            values[
                "curriculum/reverse_candidate_parent_delta_position_mm_mean"
            ],
            .3,
        )

    def test_training_end_flushes_the_last_tensorboard_window(self):
        self.callback._process_due_work = lambda: None
        self.callback._save_curriculum = lambda path: None
        self.callback.model.num_timesteps = 12_345
        self.callback._on_training_end()
        self.assertEqual(self.logger.dumps, [12_345])

    def test_true_start_progress_uses_simultaneous_pose_and_milestones(self):
        self.callback.locals = {
            "dones": [True],
            "infos": [{
                "reset_source": "true_start", "safe_success": False,
                "episode": {"l": 12}, "best_position_error": .006,
                "best_pose_metric": .009,
                "position_error_at_best_pose": .007,
                "rotation_error_at_best_pose": .02,
                "reached_20mm": 1.0, "reached_10mm": 1.0,
                "reached_5mm": 0.0, "reached_2mm": 0.0,
            }],
        }
        self.callback._on_step()
        self.callback._record_metrics()
        self.assertEqual(self.logger.values["true_start/best_position_error"], .006)
        self.assertEqual(self.logger.values["true_start/best_pose_metric"], .009)
        self.assertEqual(self.logger.values["true_start/reached_10mm"], 1.0)
        self.assertEqual(self.logger.values["true_start/reached_5mm"], 0.0)

    def test_frontier_start_repetition_is_counted_by_state_id_per_window(self):
        infos = [
            {
                "reset_source": "curriculum_frontier", "safe_success": False,
                "curriculum_start_state_id": state_id, "episode": {"l": 2},
            }
            for state_id in (7, 7, 8)
        ]
        self.callback.locals = {"dones": [True] * 3, "infos": infos}
        self.callback._on_step()
        self.callback._record_metrics()
        self.assertEqual(
            self.logger.values["curriculum/frontier_unique_states_sampled"], 2,
        )
        self.assertEqual(
            self.logger.values["curriculum/frontier_resets_per_state_max"], 2,
        )
        self.assertEqual(
            self.logger.values["curriculum/frontier_resets_per_state_mean"], 1.5,
        )

    def test_sampling_targets_are_bound_to_the_correct_update_window(self):
        self.manager.pools["frontier"] = []
        self.manager.pools["mastered"] = []
        self.manager.mastered_boundary_states = lambda: []
        self.manager.config = {
            "curriculum_reset_probability": .95,
            "start_sampling": {
                "strategy": "adaptive_three_way",
                "historical_bins": 4,
                "frontier": {
                    "fraction_per_state": .10, "fraction_max": .45,
                },
                "historical": {
                    "fraction_per_state": .01, "fraction_max": .25,
                },
                "true_start": {"fraction_min": .30},
            },
        }
        self.callback.sampling_targets_used = self.callback._sampling_targets()
        self.callback.source_episode_counts["true_start"] = 5
        captured = {}

        def update(model):
            self.manager.pools["frontier"] = [
                SimpleNamespace(pose_distance=.01, generation_depth=index)
                for index in (1, 2)
            ]
            self.manager.next_update_timesteps = 100_000

        def write(targets_used, targets_next):
            captured["used"] = targets_used
            captured["next"] = targets_next

        self.manager.update = update
        self.callback._write_curriculum_diagnostics = write
        self.callback.model.num_timesteps = 50_000
        self.callback._process_due_work()

        self.assertEqual(captured["used"].frontier, 0.0)
        self.assertEqual(captured["used"].true_start, 1.0)
        self.assertEqual(captured["next"].frontier, .20)
        self.assertEqual(captured["next"].true_start, .80)
        self.assertEqual(
            self.logger.values["curriculum/sampling/target_used/frontier"],
            0.0,
        )
        self.assertEqual(
            self.logger.values["curriculum/sampling/observed/frontier"],
            0.0,
        )
        self.assertEqual(
            self.logger.values["curriculum/sampling/target_next/frontier"],
            .20,
        )
        self.assertEqual(self.callback.sampling_targets_used.frontier, .20)

    def test_long_used_start_metrics_do_not_collide_in_human_logger(self):
        stream = io.StringIO()
        logger = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(stream, max_length=36)],
        )
        self.callback.model.logger = logger
        self.callback.used_start_distances["curriculum_historical"] = [
            .001, .002, .003,
        ]
        self.callback._record_metrics(
            include_pool_metrics=True, include_update_metrics=True,
        )

        keys = [
            f"curriculum/used_start_distance_historical_{statistic}"
            for statistic in ("min", "median", "max", "mean")
        ]
        self.assertTrue(all(key in logger.name_to_value for key in keys))
        self.assertTrue(all(
            logger.name_to_excluded[key] == ("stdout",) for key in keys
        ))
        expansion_keys = [
            f"curriculum/expansion_seed_distance_{statistic}"
            for statistic in ("min", "mean", "max")
        ]
        self.assertTrue(all(key in logger.name_to_value for key in expansion_keys))
        self.assertTrue(all(
            logger.name_to_excluded[key] == ("stdout",)
            for key in expansion_keys
        ))
        lineage_update_keys = [
            f"curriculum/expansion_seed_depth_{statistic}"
            for statistic in ("mean", "max")
        ]
        self.assertTrue(all(
            key in logger.name_to_value for key in lineage_update_keys
        ))
        self.assertTrue(all(
            logger.name_to_excluded[key] == ("stdout",)
            for key in lineage_update_keys
        ))
        cost_and_efficiency_keys = [
            "curriculum/revalidation_mastered_rollouts",
            "curriculum/revalidation_too_hard_rollouts",
            "curriculum/revalidation_wall_time",
            *[
                f"curriculum/{name}" for name in (
                    "expansion_candidates", "expansion_hops",
                    "expansion_branches", "expansion_rollouts",
                    "new_mastered", "new_frontier", "new_too_hard",
                    "mean_hops_per_branch", "max_hops_reached",
                    "expansion_scale_mean", "expansion_scale_max",
                    "frontier_found_per_candidate", "expansion_wall_time",
                )
            ],
        ]
        self.assertTrue(all(
            key in logger.name_to_value for key in cost_and_efficiency_keys
        ))
        self.assertTrue(all(
            logger.name_to_excluded[key] == ("stdout",)
            for key in cost_and_efficiency_keys
        ))
        self.assertEqual(
            logger.name_to_value["curriculum/max_generation_depth"], 7,
        )
        self.assertEqual(
            logger.name_to_value["curriculum/mastered_boundary_count"], 1,
        )

        # Une valeur non exclue force le vrai HumanOutputFormat à rendre la
        # table; les quatre longues clés restent disponibles aux autres formats.
        logger.record("train/sentinel", 1.0)
        logger.dump(step=1)
        self.assertIn("sentinel", stream.getvalue())
        self.assertNotIn("used_start_distance", stream.getvalue())
        self.assertNotIn("expansion_seed_distance", stream.getvalue())
        self.assertNotIn("expansion_seed_depth", stream.getvalue())
        self.assertNotIn("expansion_candidates", stream.getvalue())
        self.assertNotIn("revalidation_mastered", stream.getvalue())

    def test_update_logging_accepts_legacy_reports_without_cost_fields(self):
        self.manager.last_generation_report = SimpleNamespace(generated=3)
        self.manager.last_revalidation_report = SimpleNamespace(
            too_hard_revalidated=0,
        )

        self.callback._record_metrics(include_update_metrics=True)

        self.assertEqual(
            self.logger.values["curriculum/too_hard_revalidated"], 0,
        )
        self.assertNotIn(
            "curriculum/expansion_candidates", self.logger.values,
        )
        self.assertNotIn(
            "curriculum/revalidation_mastered_rollouts", self.logger.values,
        )


if __name__ == "__main__":
    unittest.main()
