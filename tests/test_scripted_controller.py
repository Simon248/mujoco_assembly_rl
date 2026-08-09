from __future__ import annotations

import unittest

import numpy as np

from src.assembly_env import AssemblyEnv
from src.config import load_config
from src.evaluate_scripted import proportional_action


class ScriptedControllerTest(unittest.TestCase):
    def test_v14_initial_pose_commands_toward_target(self):
        config = load_config("configs/test1V14.yaml")
        env = AssemblyEnv("configs/test1V14.yaml")
        try:
            _, info = env.reset(seed=100)
            error = info["true_error"]
            action = proportional_action(
                error,
                max_translation_step=config["action"]["max_translation_step"],
                max_rotation_step_deg=config["action"]["max_rotation_step_deg"],
            )
        finally:
            env.close()
        self.assertGreater(error[2], 0.0)
        self.assertLess(action[2], 0.0)
        np.testing.assert_allclose(action[[0, 1, 3, 4, 5]], 0.0, atol=1e-6)

    def test_initial_positive_z_error_commands_only_toward_target(self):
        action = proportional_action(
            np.array([0, 0, 0.040, 0, 0, 0]),
            max_translation_step=0.001, max_rotation_step_deg=1.0,
        )
        np.testing.assert_allclose(action, [0, 0, -1, 0, 0, 0])

    def test_small_translation_error_is_proportional(self):
        action = proportional_action(
            np.array([0.00025, 0, 0, 0, 0, 0]),
            max_translation_step=0.001, max_rotation_step_deg=1.0,
        )
        self.assertAlmostEqual(float(action[0]), -0.25)
        np.testing.assert_allclose(action[1:], 0)

    def test_large_translation_error_is_saturated(self):
        action = proportional_action(
            np.array([-0.040, 0, 0, 0, 0, 0]),
            max_translation_step=0.001, max_rotation_step_deg=1.0,
        )
        self.assertEqual(float(action[0]), 1.0)

    def test_rotation_limit_is_converted_from_degrees_to_radians(self):
        action = proportional_action(
            np.array([0, 0, 0, np.deg2rad(0.25), 0, np.deg2rad(-2.0)]),
            max_translation_step=0.001, max_rotation_step_deg=1.0,
        )
        np.testing.assert_allclose(action[3:], [-0.25, 0, 1], atol=1e-7)

    def test_zero_error_produces_zero_action(self):
        action = proportional_action(
            np.zeros(6), max_translation_step=0.001,
            max_rotation_step_deg=1.0,
        )
        np.testing.assert_array_equal(action, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
