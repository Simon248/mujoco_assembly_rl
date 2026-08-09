from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from src.assembly_env import TenonMortaiseEnv


class EnvironmentHelpersTest(unittest.TestCase):
    def test_environment_timeout_is_terminal_on_exact_last_step(self):
        env = TenonMortaiseEnv("configs/test1V14.yaml")
        try:
            env.reset(seed=100)
            env.cfg["simulation"]["max_episode_steps"] = 1
            _, _, terminated, truncated, info = env.step(np.zeros(6))
        finally:
            env.close()
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "timeout")

    def test_reset_clears_proximity_milestone_state(self):
        env = TenonMortaiseEnv("configs/test1V18.yaml")
        try:
            env.reset(seed=100)
            self.assertEqual(env.proximity_milestones_reached, set())
            env.proximity_milestones_reached = {0, 1, 2, 3}
            env.reset(seed=100)
            self.assertEqual(env.proximity_milestones_reached, set())
            _, _, _, _, info = env.step(np.zeros(6))
            self.assertEqual(info["reward_step"], -0.02)
            self.assertEqual(info["reward_proximity"], 0.0)
            self.assertEqual(info["reward_timeout"], 0.0)
            self.assertEqual(info["proximity_milestones_reached"], 0)
            self.assertIn("episode_reward_proximity", info)
            self.assertIn("episode_reward_step", info)
            self.assertIn("episode_reward_timeout", info)
        finally:
            env.close()

    def _substep_env(self, wrenches):
        env = object.__new__(TenonMortaiseEnv)
        env.model = object(); env.data = object(); env.frame_skip = 10
        sequence = iter(wrenches)
        env._true_wrench = lambda: np.asarray(next(sequence), dtype=float)
        env._error = lambda: np.zeros(6)
        return env

    @patch("src.assembly_env.mujoco.mj_step")
    def test_substep_peak_is_latched_even_if_final_value_is_low(self, mj_step):
        values = [[1, 0, 0, 0, 0, 0], [80, 0, 0, 0, 0, 0]] + [[2, 0, 0, 0, 0, 0]] * 8
        env = self._substep_env(values)
        force, torque = env._run_control_substeps({"max_force": 100, "max_torque": 10, "workspace_radius": .2})
        self.assertEqual(mj_step.call_count, 10)
        self.assertEqual(force, 80.0); self.assertEqual(torque, 0.0)

    @patch("src.assembly_env.mujoco.mj_step")
    def test_substeps_stop_when_torque_reaches_limit(self, mj_step):
        values = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 10]]
        env = self._substep_env(values)
        force, torque = env._run_control_substeps({"max_force": 100, "max_torque": 10, "workspace_radius": .2})
        self.assertEqual(mj_step.call_count, 3)
        self.assertEqual(force, 0.0); self.assertEqual(torque, 10.0)

    def test_friction_scale_is_applied_from_base_values(self):
        env = object.__new__(TenonMortaiseEnv)
        env.np_random = np.random.default_rng(4)
        env.model = SimpleNamespace(geom_friction=np.ones((3, 3)))
        env._contact_geom_ids = np.array([0, 2])
        env._base_contact_friction = np.array([[.8, .01, .001], [.8, .01, .001]])
        scale = env._randomize_friction({"friction_scale": [1.5, 1.5]})
        self.assertEqual(scale, 1.5)
        np.testing.assert_allclose(env.model.geom_friction[[0, 2]], env._base_contact_friction * 1.5)


if __name__ == "__main__":
    unittest.main()
