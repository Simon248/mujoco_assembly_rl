from __future__ import annotations

import unittest
import numpy as np

from src.task_logic import (
    assess_status, newly_reached_milestones, prepare_proximity_milestones,
    reward_components, satisfied_milestone_indices,
)


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
    OLD_MILESTONES = prepare_proximity_milestones([
        {"threshold": 0.010, "bonus": 5.0},
    ])
    POSE_MILESTONES = prepare_proximity_milestones([
        {"position_threshold": 0.010, "orientation_threshold_deg": 5, "bonus": 5},
        {"position_threshold": 0.006, "orientation_threshold_deg": 4, "bonus": 10},
        {"position_threshold": 0.004, "orientation_threshold_deg": 3, "bonus": 20},
        {"position_threshold": 0.002, "orientation_threshold_deg": 2, "bonus": 40},
    ])

    def test_proximity_milestones_are_unique_and_can_be_crossed_together(self):
        reached: set[int] = set()
        bonus, reached = newly_reached_milestones(
            0.009, np.deg2rad(90), self.OLD_MILESTONES, reached,
        )
        self.assertEqual(bonus, 5.0)
        bonus, reached = newly_reached_milestones(
            0.011, 0.0, self.OLD_MILESTONES, reached,
        )
        self.assertEqual(bonus, 0.0)
        bonus, reached = newly_reached_milestones(
            0.009, 0.0, self.OLD_MILESTONES, reached,
        )
        self.assertEqual(bonus, 0.0)

    def test_pose_milestones_require_both_conditions_and_are_unique(self):
        reached: set[int] = set()
        bonus, reached = newly_reached_milestones(
            0.005, np.deg2rad(7), self.POSE_MILESTONES, reached,
        )
        self.assertEqual(bonus, 0.0)  # Position bonne, orientation mauvaise.
        bonus, reached = newly_reached_milestones(
            0.0055, np.deg2rad(3), self.POSE_MILESTONES, reached,
        )
        self.assertEqual(bonus, 15.0)  # 10 mm/5° et 6 mm/4° deviennent valides.
        bonus, reached = newly_reached_milestones(
            0.008, np.deg2rad(7), self.POSE_MILESTONES, reached,
        )
        self.assertEqual(bonus, 0.0)
        bonus, reached = newly_reached_milestones(
            0.0055, np.deg2rad(3), self.POSE_MILESTONES, reached,
        )
        self.assertEqual(bonus, 0.0)  # Aucun double bonus.

    def test_position_too_far_and_multiple_pose_milestones(self):
        bonus, reached = newly_reached_milestones(
            0.008, np.deg2rad(1), self.POSE_MILESTONES, set(),
        )
        self.assertEqual(bonus, 5.0)
        self.assertNotIn(1, reached)  # 6 mm/4° reste hors zone.
        bonus, reached = newly_reached_milestones(
            0.0035, np.deg2rad(2.5), self.POSE_MILESTONES, set(),
        )
        self.assertEqual(bonus, 35.0)
        self.assertEqual(reached, {0, 1, 2})
        self.assertNotIn(3, reached)

    def test_initially_satisfied_pose_milestone_is_marked_without_bonus(self):
        reached = satisfied_milestone_indices(
            0.008, np.deg2rad(2), self.POSE_MILESTONES,
        )
        self.assertEqual(reached, {0})
        bonus, reached = newly_reached_milestones(
            0.007, np.deg2rad(2), self.POSE_MILESTONES, reached,
        )
        self.assertEqual(bonus, 0.0)

    def test_step_timeout_and_backward_compatible_reward_components(self):
        running = status(position_error=.04, rotation_error=.1, step_count=299)
        timeout = status(position_error=.04, rotation_error=.1, step_count=300)
        success = status(step_count=300)
        unsafe = status(position_error=.04, rotation_error=.1,
                        step_count=300, max_torque=10.0)
        configured = {
            **REWARD, "step_penalty": 0.02, "timeout_penalty": 100.0,
        }
        common = dict(
            position_error=.04, rotation_error=.1,
            previous_position_error=.04, previous_rotation_error=.1,
            max_force=0.0, max_torque=0.0, action=np.zeros(6),
            action_config=ACTION,
        )
        before = reward_components(**common, status=running, config=configured)
        at_timeout = reward_components(**common, status=timeout, config=configured)
        at_success = reward_components(**common, status=success, config=configured)
        at_unsafe = reward_components(**common, status=unsafe, config=configured)
        self.assertEqual(before["reward_step"], -0.02)
        self.assertEqual(before["reward_timeout"], 0.0)
        self.assertEqual(at_timeout["reward_step"], -0.02)
        self.assertEqual(at_timeout["reward_timeout"], -100.0)
        self.assertEqual(at_success["reward_timeout"], 0.0)
        self.assertEqual(at_unsafe["reward_timeout"], 0.0)
        historical = reward_components(**common, status=running, config=REWARD)
        self.assertEqual(historical["reward_proximity"], 0.0)
        self.assertEqual(historical["reward_step"], 0.0)
        self.assertEqual(historical["reward_timeout"], 0.0)

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

    def test_timeout_is_terminal_and_not_truncated(self):
        result = status(position_error=.04, rotation_error=.1, step_count=300)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.termination_reason, "timeout")

    def test_step_before_timeout_is_running(self):
        result = status(position_error=.04, rotation_error=.1, step_count=299)
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.termination_reason, "running")

    def test_success_and_unsafe_keep_priority_at_last_step(self):
        success = status(step_count=300)
        self.assertTrue(success.terminated)
        self.assertFalse(success.truncated)
        self.assertEqual(success.termination_reason, "success")
        unsafe = status(step_count=300, max_torque=10.0)
        self.assertTrue(unsafe.terminated)
        self.assertFalse(unsafe.truncated)
        self.assertEqual(unsafe.termination_reason, "unsafe_torque")

    def test_continuous_torque_reward_and_unsafe_penalty_coexist(self):
        for torque, expected in ((0.0, 0.0), (5.0, -0.4), (9.0, -0.72)):
            with self.subTest(torque=torque):
                result = status(position_error=.04, rotation_error=.1, max_torque=torque)
                reward = reward_components(
                    position_error=.04, rotation_error=.1,
                    previous_position_error=.04, previous_rotation_error=.1,
                    max_force=0.0, max_torque=torque, action=np.zeros(6),
                    status=result, config={**REWARD, "torque_weight": 0.08},
                    action_config=ACTION,
                )
                self.assertAlmostEqual(reward["reward_torque"], expected)
                self.assertFalse(result.unsafe_torque)
        result = status(position_error=.04, rotation_error=.1, max_torque=10.0)
        reward = reward_components(
            position_error=.04, rotation_error=.1,
            previous_position_error=.04, previous_rotation_error=.1,
            max_force=0.0, max_torque=10.0, action=np.zeros(6),
            status=result, config={**REWARD, "torque_weight": 0.08},
            action_config=ACTION,
        )
        self.assertAlmostEqual(reward["reward_torque"], -0.8)
        self.assertTrue(result.unsafe_torque)
        self.assertEqual(reward["reward_unsafe"], -300.0)
        self.assertTrue(result.terminated)

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
