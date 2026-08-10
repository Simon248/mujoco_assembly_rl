from __future__ import annotations

import unittest
import numpy as np

from src.task_logic import assess_status, pose_distance, reward_components


SAFETY = {
    "position_tolerance": 0.0005, "rotation_tolerance_deg": 2.0,
    "max_force": 80.0, "max_torque": 10.0, "workspace_radius": 0.15,
}
REWARD = {
    "rotation_length_scale": 0.05, "pose_weight": 50.0,
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


def components(position, rotation=0.0, task_status=None, **overrides):
    values = dict(
        position_error=position, rotation_error=rotation,
        max_force=0.0, max_torque=0.0, action=np.zeros(6),
        status=task_status or status(), config=REWARD,
    )
    values.update(overrides)
    return reward_components(**values)


def total(values):
    return sum(value for key, value in values.items() if key.startswith("reward_"))


class TaskLogicTest(unittest.TestCase):
    def test_pose_distance_is_additive(self):
        self.assertAlmostEqual(pose_distance(0.010, 0.0, 0.05), 0.010)
        self.assertAlmostEqual(pose_distance(0.0, np.deg2rad(1), 0.05), 0.0008726646)
        self.assertAlmostEqual(
            pose_distance(0.010, np.deg2rad(5), 0.05),
            0.010 + 0.05 * np.deg2rad(5),
        )

    def test_reward_uses_reached_pose_and_logs_equivalent_rotation(self):
        result = components(0.010, np.deg2rad(5))
        expected_distance = 0.010 + 0.05 * np.deg2rad(5)
        self.assertAlmostEqual(result["rotation_equivalent_distance"], 0.05 * np.deg2rad(5))
        self.assertAlmostEqual(result["pose_distance"], expected_distance)
        self.assertAlmostEqual(result["reward_pose"], -50.0 * expected_distance)

    def test_real_actions_rank_down_stay_lateral_and_up(self):
        action_down = np.array([0, 0, -1, 0, 0, 0], dtype=float)
        action_stay = np.zeros(6)
        action_lateral = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        action_up = np.array([0, 0, 1, 0, 0, 0], dtype=float)
        rewards = {
            "down": total(components(0.0395, action=action_down)),
            "stay": total(components(0.0400, action=action_stay)),
            "lateral": total(components(np.hypot(0.040, 0.0005), action=action_lateral)),
            "up": total(components(0.0405, action=action_up)),
        }
        self.assertAlmostEqual(rewards["down"], -2.035)
        self.assertAlmostEqual(rewards["stay"], -2.050)
        self.assertGreater(rewards["down"], rewards["stay"])
        self.assertGreater(rewards["stay"], rewards["lateral"])
        self.assertGreater(rewards["stay"], rewards["up"])

    def test_all_reasonable_down_amplitudes_beat_staying(self):
        stay = total(components(0.040))
        for amplitude in (0.25, 0.50, 1.0):
            with self.subTest(amplitude=amplitude):
                action = np.array([0, 0, -amplitude, 0, 0, 0])
                reached = 0.040 - 0.0005 * amplitude
                self.assertGreater(total(components(reached, action=action)), stay)

    def test_unnecessary_rotation_is_worse_without_axis_specific_penalty(self):
        down = total(components(0.0395, action=np.array([0, 0, -1, 0, 0, 0])))
        down_rotate = total(components(
            0.0395, np.deg2rad(0.25), action=np.array([0, 0, -1, 1, 0, 0]),
        ))
        self.assertGreater(down, down_rotate)

    def test_lateral_detour_is_penalized_but_neither_forbidden_nor_terminal(self):
        detour_status = status(position_error=np.hypot(0.040, 0.0005))
        self.assertFalse(detour_status.terminated)
        self.assertEqual(detour_status.termination_reason, "running")

    def test_success_is_dominant_and_terminal_events_remain_separate(self):
        success_status = status(position_error=0.0004, rotation_error=np.deg2rad(1))
        success_reward = components(0.0004, np.deg2rad(1), success_status)
        near_reward = components(0.0006, 0.0)
        self.assertEqual(success_reward["reward_success"], 300.0)
        self.assertGreater(total(success_reward), total(near_reward) + 299.0)
        timeout_reward = components(0.040, task_status=status(step_count=300))
        unsafe_reward = components(0.040, task_status=status(max_torque=10.0))
        self.assertEqual(timeout_reward["reward_timeout"], -150.0)
        self.assertEqual(unsafe_reward["reward_unsafe"], -300.0)
        self.assertEqual(unsafe_reward["reward_timeout"], 0.0)

    def test_immobile_episode_times_out_at_300(self):
        running_total = 0.0
        for step in range(1, 301):
            task_status = status(step_count=step)
            running_total += total(components(0.040, task_status=task_status))
        self.assertAlmostEqual(running_total, 299 * -2.05 + -152.05)
        self.assertEqual(
            (task_status.terminated, task_status.truncated, task_status.termination_reason),
            (True, False, "timeout"),
        )

    def test_force_torque_action_and_step_formulas_are_unchanged(self):
        reward = components(0.019, max_force=5.0, max_torque=2.0, action=np.ones(6))
        self.assertEqual(reward["reward_force"], -0.05)
        self.assertEqual(reward["reward_torque"], -0.16)
        self.assertEqual(reward["reward_action"], -0.06)
        self.assertEqual(reward["reward_step"], -0.05)


if __name__ == "__main__":
    unittest.main()
