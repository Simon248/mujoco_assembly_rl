from __future__ import annotations

import unittest
import numpy as np

from src.task_logic import assess_status, pose_distance, pose_potential, reward_components


SAFETY = {
    "position_tolerance": 0.0005, "rotation_tolerance_deg": 2.0,
    "max_force": 80.0, "max_torque": 10.0, "workspace_radius": 0.15,
}
REWARD = {
    "rotation_length_scale": 0.05,
    "potential_scale": 10.0, "potential_distance_scale": 0.010,
    "step_penalty": 0.05, "force_weight": 0.01,
    "torque_weight": 0.08, "action_weight": 0.01,
    "success_bonus": 300.0, "unsafe_penalty": 300.0,
    "timeout_penalty": 150.0,
}


def status(**overrides):
    values = dict(
        position_error=0.01, rotation_error=0.1, max_force=0.0,
        max_torque=0.0, workspace_error=0.01, step_count=1,
        config=SAFETY, max_episode_steps=300,
    )
    values.update(overrides)
    return assess_status(**values)


def components(current, following, task_status=None, **overrides):
    values = dict(
        current_pose_distance=current, next_pose_distance=following, gamma=0.99,
        max_force=0.0, max_torque=0.0, action=np.zeros(6),
        status=task_status or status(), config=REWARD,
    )
    values.update(overrides)
    return reward_components(**values)


class TaskLogicTest(unittest.TestCase):
    def test_pose_distance_position_rotation_and_combined(self):
        self.assertAlmostEqual(pose_distance(0.010, 0.0, 0.05), 0.010)
        rotation_only = pose_distance(0.0, np.deg2rad(1.0), 0.05)
        self.assertAlmostEqual(rotation_only, 0.0008726646)
        self.assertAlmostEqual(
            pose_distance(0.010, np.deg2rad(5.0), 0.05),
            np.hypot(0.010, 0.05 * np.deg2rad(5.0)),
        )

    def test_potential_is_bounded_and_monotone(self):
        self.assertEqual(pose_potential(0.0, 10.0, 0.010), 10.0)
        self.assertGreater(
            pose_potential(0.005, 10.0, 0.010),
            pose_potential(0.020, 10.0, 0.010),
        )
        self.assertLess(pose_potential(1.0, 10.0, 0.010), 1e-20)

    def test_potential_reward_improvement_degradation_and_stationarity(self):
        improvement = components(0.040, 0.030)["reward_potential"]
        degradation = components(0.040, 0.050)["reward_potential"]
        stationary = components(0.040, 0.040)["reward_potential"]
        phi = pose_potential(0.040, 10.0, 0.010)
        self.assertGreater(improvement, 0.0)
        self.assertLess(degradation, 0.0)
        self.assertAlmostEqual(stationary, (0.99 - 1.0) * phi)

    def test_terminal_potential_and_separate_terminal_events(self):
        success = status(position_error=0.0, rotation_error=0.0)
        timeout = status(step_count=300)
        unsafe = status(max_torque=10.0)
        success_reward = components(0.001, 0.0, success)
        timeout_reward = components(0.010, 0.010, timeout)
        unsafe_reward = components(0.010, 0.010, unsafe)
        self.assertAlmostEqual(success_reward["reward_potential"], -pose_potential(0.001, 10, .01))
        self.assertEqual(success_reward["reward_success"], 300.0)
        self.assertEqual(success_reward["reward_timeout"], 0.0)
        self.assertEqual(timeout_reward["reward_timeout"], -150.0)
        self.assertEqual(timeout_reward["reward_success"], 0.0)
        self.assertEqual(unsafe_reward["reward_unsafe"], -300.0)
        self.assertEqual(unsafe_reward["reward_timeout"], 0.0)

    def test_timeout_is_terminal_exactly_at_step_300_with_priorities(self):
        before = status(step_count=299)
        timeout = status(step_count=300)
        success = status(step_count=300, position_error=0.0, rotation_error=0.0)
        unsafe = status(step_count=300, max_torque=10.0)
        self.assertEqual((before.terminated, before.truncated, before.termination_reason),
                         (False, False, "running"))
        self.assertEqual((timeout.terminated, timeout.truncated, timeout.termination_reason),
                         (True, False, "timeout"))
        self.assertEqual(success.termination_reason, "success")
        self.assertEqual(unsafe.termination_reason, "unsafe_torque")

    def test_force_torque_action_and_step_formulas_are_unchanged(self):
        reward = components(
            0.02, 0.019, max_force=5.0, max_torque=2.0,
            action=np.ones(6),
        )
        self.assertEqual(reward["reward_force"], -0.05)
        self.assertEqual(reward["reward_torque"], -0.16)
        self.assertEqual(reward["reward_action"], -0.06)
        self.assertEqual(reward["reward_step"], -0.05)

    def test_nominal_z_descent_beats_stationarity_and_lateral_motion(self):
        start = pose_distance(0.040, 0.0, 0.05)
        down = pose_distance(0.0395, 0.0, 0.05)
        still = pose_distance(0.040, 0.0, 0.05)
        lateral = pose_distance(np.hypot(0.040, 0.0005), 0.0, 0.05)
        # Ne sommer que les rewards, pas pose_distance/phi utilisés pour le logging.
        self.z_rewards = {
            name: sum(value for key, value in components(start, distance).items()
                      if key.startswith("reward_"))
            for name, distance in (("down", down), ("still", still), ("lateral", lateral))
        }
        self.assertGreater(self.z_rewards["down"], self.z_rewards["still"])
        self.assertGreater(self.z_rewards["down"], self.z_rewards["lateral"])

    def test_position_and_orientation_improvements_share_one_distance(self):
        a = pose_distance(0.010, np.deg2rad(5), 0.05)
        b = pose_distance(0.008, np.deg2rad(5), 0.05)
        c = pose_distance(0.010, np.deg2rad(3), 0.05)
        tradeoff = pose_distance(0.008, np.deg2rad(7), 0.05)
        self.assertLess(b, a)
        self.assertLess(c, a)
        self.assertAlmostEqual(tradeoff, np.hypot(0.008, 0.05 * np.deg2rad(7)))


if __name__ == "__main__":
    unittest.main()
