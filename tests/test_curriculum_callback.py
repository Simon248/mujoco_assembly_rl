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
        self.last_revalidation_report = SimpleNamespace(
            too_hard_revalidated=12,
            too_hard_to_frontier=3,
            too_hard_to_mastered=1,
            too_hard_remained_hard=8,
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

    def test_training_end_flushes_the_last_tensorboard_window(self):
        self.callback._process_due_work = lambda: None
        self.callback._save_curriculum = lambda path: None
        self.callback.model.num_timesteps = 12_345
        self.callback._on_training_end()
        self.assertEqual(self.logger.dumps, [12_345])

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


if __name__ == "__main__":
    unittest.main()
