from __future__ import annotations

import unittest
import numpy as np

from src.task_logic import assess_status, reward_components


SAFETY = {
    "position_tolerance": 0.003,
    "rotation_tolerance_deg": 4.0,
    "max_force": 100.0,
    "max_torque": 10.0,
    "workspace_radius": 0.2,
}
REWARD = {
    "position_weight": 20.0, "orientation_weight": 2.0,
    "progress_weight": 1.0, "force_weight": 0.01,
    "action_weight": 0.01, "success_bonus": 100.0,
    "unsafe_penalty": 300.0,
}
ACTION = {
    "max_translation_step": 0.001,
    "max_rotation_step_deg": 1.0,
}


def status(**overrides):
    values = dict(position_error=0.002, rotation_error=0.02, max_force=0.0,
                  max_torque=0.0, workspace_error=0.002, step_count=1,
                  config=SAFETY, max_episode_steps=300)
    values.update(overrides)
    return assess_status(**values)


class TaskLogicTest(unittest.TestCase):
    def test_safe_geometric_success_receives_bonus(self):
        result = status()
        self.assertTrue(result.success)
        self.assertFalse(result.unsafe)
        self.assertEqual(result.termination_reason, "success")
        reward = reward_components(
            position_error=.002, rotation_error=.02,
            previous_position_error=.003, previous_rotation_error=.03,
            max_force=0, action=np.zeros(6), status=result, config=REWARD,
            action_config=ACTION,
        )
        self.assertEqual(reward["reward_success"], 100.0)
        self.assertEqual(reward["reward_unsafe"], 0.0)

    def test_geometric_success_at_force_limit_is_unsafe_only(self):
        result = status(max_force=100.0)
        self.assertTrue(result.geometric_success)
        self.assertTrue(result.unsafe_force)
        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "unsafe_force")
        reward = reward_components(
            position_error=.002, rotation_error=.02,
            previous_position_error=.003, previous_rotation_error=.03,
            max_force=100, action=np.zeros(6), status=result, config=REWARD,
            action_config=ACTION,
        )
        self.assertEqual(reward["reward_success"], 0.0)
        self.assertEqual(reward["reward_unsafe"], -300.0)

    def test_force_and_torque_reason_is_unique(self):
        result = status(max_force=100.0, max_torque=10.0)
        self.assertEqual(result.termination_reason, "unsafe_force_and_torque")
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)

    def test_timeout_is_truncated_but_not_terminated(self):
        result = status(position_error=.04, rotation_error=.1, step_count=300)
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertEqual(result.termination_reason, "timeout")

    def test_translation_progress_is_normalized_by_maximum_action_step(self):
        cases = (
            (0.009, 1.0),
            (0.0095, 0.5),
            (0.010, 0.0),
            (0.011, -1.0),
        )
        reward_config = {**REWARD, "progress_weight": 1.0}
        for current_position_error, expected in cases:
            with self.subTest(current_position_error=current_position_error):
                reward = reward_components(
                    position_error=current_position_error,
                    rotation_error=0.1,
                    previous_position_error=0.010,
                    previous_rotation_error=0.1,
                    max_force=0.0,
                    action=np.zeros(6),
                    status=status(position_error=0.010, rotation_error=0.1),
                    config=reward_config,
                    action_config=ACTION,
                )
                self.assertAlmostEqual(reward["reward_progress"], expected)

    def test_progress_weight_scales_normalized_translation_progress(self):
        reward = reward_components(
            position_error=0.009,
            rotation_error=0.1,
            previous_position_error=0.010,
            previous_rotation_error=0.1,
            max_force=0.0,
            action=np.zeros(6),
            status=status(position_error=0.010, rotation_error=0.1),
            config={**REWARD, "progress_weight": 2.0},
            action_config=ACTION,
        )
        self.assertAlmostEqual(reward["reward_progress"], 2.0)

    def test_progress_is_not_clipped(self):
        reward = reward_components(
            position_error=0.008,
            rotation_error=0.1,
            previous_position_error=0.010,
            previous_rotation_error=0.1,
            max_force=0.0,
            action=np.zeros(6),
            status=status(position_error=0.010, rotation_error=0.1),
            config={**REWARD, "progress_weight": 1.0},
            action_config=ACTION,
        )
        self.assertAlmostEqual(reward["reward_progress"], 2.0)

    def test_rotation_progress_is_normalized_by_maximum_action_step(self):
        reward_config = {**REWARD, "progress_weight": 1.0}
        previous_rotation_error = np.deg2rad(10.0)
        for current_degrees, expected in ((9.0, 1.0), (11.0, -1.0)):
            with self.subTest(current_degrees=current_degrees):
                reward = reward_components(
                    position_error=0.010,
                    rotation_error=np.deg2rad(current_degrees),
                    previous_position_error=0.010,
                    previous_rotation_error=previous_rotation_error,
                    max_force=0.0,
                    action=np.zeros(6),
                    status=status(position_error=0.010, rotation_error=0.1),
                    config=reward_config,
                    action_config=ACTION,
                )
                self.assertAlmostEqual(reward["reward_progress"], expected)

    def test_translation_and_rotation_progress_have_equal_weight(self):
        reward = reward_components(
            position_error=0.009,
            rotation_error=np.deg2rad(9.0),
            previous_position_error=0.010,
            previous_rotation_error=np.deg2rad(10.0),
            max_force=0.0,
            action=np.zeros(6),
            status=status(position_error=0.010, rotation_error=0.1),
            config=REWARD,
            action_config=ACTION,
        )
        self.assertAlmostEqual(reward["reward_progress"], 2.0)

    def test_opposite_translation_and_rotation_progress_cancel(self):
        reward = reward_components(
            position_error=0.009,
            rotation_error=np.deg2rad(11.0),
            previous_position_error=0.010,
            previous_rotation_error=np.deg2rad(10.0),
            max_force=0.0,
            action=np.zeros(6),
            status=status(position_error=0.010, rotation_error=0.1),
            config=REWARD,
            action_config=ACTION,
        )
        self.assertAlmostEqual(reward["reward_progress"], 0.0)


if __name__ == "__main__":
    unittest.main()
