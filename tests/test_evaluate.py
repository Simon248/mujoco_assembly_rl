from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.evaluate import (
    checkpoint_steps, configured_stochastic_sources, resolve_models,
    evaluate_model, summarize_episodes, trajectory_row,
)


class EvaluateTest(unittest.TestCase):
    def test_all_checkpoints_are_sorted_by_training_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory); checkpoints = run / "checkpoints"
            checkpoints.mkdir()
            for name in ("sac_100_steps.zip", "sac_20_steps.zip", "sac_3_steps.zip"):
                (checkpoints / name).touch()
            models = resolve_models(run, requested=None, all_checkpoints=True)
            self.assertEqual([path.name for path in models], [
                "sac_3_steps.zip", "sac_20_steps.zip", "sac_100_steps.zip",
            ])
            self.assertEqual(checkpoint_steps(models[-1]), 100)

    def test_requested_models_keep_order_and_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory); checkpoints = run / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "sac_20_steps.zip").touch()
            (checkpoints / "sac_100_steps.zip").touch()
            models = resolve_models(
                run,
                [Path("sac_100_steps.zip"), Path("sac_20_steps.zip"),
                 Path("sac_100_steps.zip")],
                all_checkpoints=False,
            )
            self.assertEqual([path.name for path in models], [
                "sac_100_steps.zip", "sac_20_steps.zip",
            ])

    def test_episode_summary_contains_rates_means_and_medians(self):
        rows = [
            self._episode(steps=10, safe_success=True, geometric_success=True,
                          unsafe=False, position=0.001, rotation=0.01),
            self._episode(steps=20, safe_success=False, geometric_success=True,
                          unsafe=True, position=0.003, rotation=0.03,
                          unsafe_force=True),
            self._episode(steps=30, safe_success=False, geometric_success=False,
                          unsafe=False, position=0.005, rotation=0.05,
                          reason="timeout"),
        ]
        summary = summarize_episodes(rows)
        self.assertAlmostEqual(summary["safe_success_rate"], 1 / 3)
        self.assertAlmostEqual(summary["geometric_success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["unsafe_rate"], 1 / 3)
        self.assertEqual(summary["mean_episode_length"], 20.0)
        self.assertEqual(summary["median_episode_length"], 20.0)
        self.assertEqual(summary["median_final_position_error"], 0.003)
        self.assertEqual(summary["median_final_rotation_error"], 0.03)

    def test_deterministic_configuration_has_no_stochastic_sources(self):
        config = {
            "randomization": {
                "mobile_translation": [0, 0, 0], "mobile_rotation_deg": [0, 0, 0],
                "fixed_translation": [0, 0, 0], "fixed_rotation_deg": [0, 0, 0],
                "friction_scale": [1, 1],
            },
            "perception": {
                "translation_noise_std": 0, "rotation_noise_std_deg": 0,
                "wrench_noise_std": [0, 0, 0, 0, 0, 0],
            },
        }
        self.assertEqual(configured_stochastic_sources(config), [])
        config["randomization"]["mobile_translation"][0] = 0.001
        config["perception"]["wrench_noise_std"][2] = 0.5
        self.assertEqual(configured_stochastic_sources(config), [
            "randomization.mobile_translation", "perception.wrench_noise_std",
        ])

    def test_trajectory_row_contains_axes_action_and_safety(self):
        info = {
            **{key: float(index) for index, key in enumerate((
                "position_error_x", "position_error_y", "position_error_z",
                "rotation_error_x", "rotation_error_y", "rotation_error_z",
                "action_x", "action_y", "action_z", "action_rx", "action_ry", "action_rz",
            ))},
            "force": 4.0, "torque": 0.2, "unsafe": False,
            "termination_reason": "running",
        }
        row = trajectory_row(0, 7, info, terminated=False, truncated=False)
        self.assertEqual(row["step"], 7)
        self.assertEqual(row["action_rz"], 11.0)
        self.assertTrue(row["safe"])
        self.assertFalse(row["unsafe"])

    @patch("src.evaluate.load_evaluation_model")
    @patch("src.evaluate.TenonMortaiseEnv")
    def test_evaluate_model_explicitly_forces_true_start_role(
        self, environment_constructor, load_model,
    ):
        class FakeEnvironment:
            cfg = {
                "training": {"algorithm": "sac"},
                "action": {"action_frame": "task"},
                "randomization": {}, "perception": {},
            }

            def reset(self, *, seed):
                return np.zeros(18), {"reset_source": "true_start"}

            def step(self, action):
                return np.zeros(18), 1.0, True, False, {
                    "success": True, "safe_success": True,
                    "geometric_success": True, "unsafe": False,
                    "unsafe_force": False, "unsafe_torque": False,
                    "unsafe_workspace": False, "termination_reason": "success",
                    "position_error": 0.0, "rotation_error": 0.0,
                    "episode_max_force": 0.0, "episode_max_torque": 0.0,
                }

            def close(self):
                pass

        class FakeModel:
            num_timesteps = 10

            def __init__(self):
                self.deterministic = []

            def predict(self, observation, deterministic):
                self.deterministic.append(deterministic)
                return np.zeros(6), None

        fake_environment = FakeEnvironment()
        fake_model = FakeModel()
        environment_constructor.return_value = fake_environment
        load_model.return_value = fake_model
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model_path = run / "model.zip"
            model_path.write_bytes(b"model")
            summary = evaluate_model(
                run=run, model_path=model_path, result_name="true_start",
                episode_count=2, seed=100, render=False, render_speed=1.0,
                write_trajectory=False,
            )
        self.assertEqual(summary["evaluation_reset_source"], "true_start")
        self.assertEqual(fake_model.deterministic, [True, True])
        self.assertFalse(
            environment_constructor.call_args.kwargs["allow_curriculum_resets"]
        )

    @staticmethod
    def _episode(
        *, steps, safe_success, geometric_success, unsafe, position, rotation,
        unsafe_force=False, reason="success",
    ):
        return {
            "steps": steps, "safe_success": safe_success,
            "geometric_success": geometric_success, "unsafe": unsafe,
            "final_position_error": position, "final_rotation_error": rotation,
            "episode_max_force": 5.0, "episode_max_torque": 0.5,
            "unsafe_force": unsafe_force, "unsafe_torque": False,
            "unsafe_workspace": False, "termination_reason": reason,
        }


if __name__ == "__main__":
    unittest.main()
