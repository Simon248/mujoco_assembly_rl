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
    "progress_weight": 10.0, "force_weight": 0.01,
    "action_weight": 0.01, "success_bonus": 100.0,
    "unsafe_penalty": 300.0,
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


if __name__ == "__main__":
    unittest.main()
