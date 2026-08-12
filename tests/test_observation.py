from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np

from src.assembly_env import TenonMortaiseEnv
from src.config import load_config, save_resolved_config


class ObservationTest(unittest.TestCase):
    def setUp(self):
        self.env = TenonMortaiseEnv("configs/test1.yaml")

    def tearDown(self):
        self.env.close()

    def test_shape_matches_explicit_eighteen_dimensional_space(self):
        observation, _ = self.env.reset(seed=17)
        self.assertEqual(observation.shape, (18,))
        self.assertEqual(observation.shape, self.env.observation_space.shape)

    def test_reset_exposes_a_zero_admittance_offset(self):
        observation, _ = self.env.reset(seed=17)
        np.testing.assert_allclose(observation[12:18], np.zeros(6), atol=1e-7)

    def test_known_offset_uses_max_offset_scaling_and_rotvec_axes(self):
        self.env.reset(seed=17)
        known_offset = np.array([0.004, -0.002, 0.008, 0.075, -0.15, 0.03])
        self.env.admittance.offset = known_offset.copy()
        observation = self.env._observation()
        np.testing.assert_allclose(
            observation[12:18],
            known_offset / self.env.admittance.offset_limit,
            atol=1e-7,
        )

    def test_simulator_and_admittance_velocities_are_not_observed(self):
        self.env.reset(seed=17)
        before = self.env._observation().copy()
        self.env.data.qvel[:] = np.linspace(-1.0, 1.0, self.env.data.qvel.size)
        self.env.admittance.velocity = np.array([0.01, -0.01, 0.02, 0.1, -0.2, 0.25])
        after = self.env._observation()
        np.testing.assert_array_equal(after, before)

    def _configured_env(self, previous: bool, admittance: bool):
        config = deepcopy(load_config("configs/test1V42.yaml"))
        config["observation"]["include_previous_pose_error"] = previous
        config["observation"]["include_admittance_position"] = admittance
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "config.yaml"
        save_resolved_config(config, path)
        env = TenonMortaiseEnv(path)
        env._temporary_config_directory = directory
        return env

    def test_all_observation_dimension_combinations(self):
        for previous, admittance, expected in (
            (False, False, 12), (True, False, 18),
            (False, True, 18), (True, True, 24),
        ):
            with self.subTest(previous=previous, admittance=admittance):
                env = self._configured_env(previous, admittance)
                try:
                    observation, _ = env.reset(seed=17)
                    self.assertEqual(env.observation_space.shape, (expected,))
                    self.assertEqual(observation.shape, (expected,))
                finally:
                    env.close()
                    env._temporary_config_directory.cleanup()

    def test_first_observation_repeats_current_error_as_previous(self):
        env = self._configured_env(True, True)
        try:
            observation, _ = env.reset(seed=17)
            np.testing.assert_array_equal(observation[:6], observation[6:12])
            np.testing.assert_allclose(observation[18:24], 0.0, atol=1e-7)
        finally:
            env.close()
            env._temporary_config_directory.cleanup()

    def test_temporal_sequence_is_current_then_previous(self):
        env = self._configured_env(True, True)
        try:
            env.reset(seed=17)
            errors = [
                np.array([.001, .002, .003, .01, .02, .03]),
                np.array([.002, .003, .004, .02, .03, .04]),
                np.array([.003, .004, .005, .03, .04, .05]),
            ]
            env._observed_wrench = lambda: np.zeros(6)
            env.current_observed_pose_error = errors[0].copy()
            env.previous_pose_error = errors[0].copy()
            observations = [env._observation()]
            for error in errors[1:]:
                env._begin_physical_transition()
                env.current_observed_pose_error = error.copy()
                observations.append(env._observation())
            normalized = [
                np.r_[error[:3] / env.cfg["observation"]["position_scale"],
                      error[3:] / env.cfg["observation"]["rotation_scale"]]
                for error in (
                    np.array([.001, .002, .003, .01, .02, .03]),
                    np.array([.002, .003, .004, .02, .03, .04]),
                    np.array([.003, .004, .005, .03, .04, .05]),
                )
            ]
            np.testing.assert_allclose(observations[0][:6], normalized[0])
            np.testing.assert_allclose(observations[0][6:12], normalized[0])
            np.testing.assert_allclose(observations[1][:6], normalized[1])
            np.testing.assert_allclose(observations[1][6:12], normalized[0])
            np.testing.assert_allclose(observations[2][:6], normalized[2])
            np.testing.assert_allclose(observations[2][6:12], normalized[1])
        finally:
            env.close()
            env._temporary_config_directory.cleanup()

    def test_multiple_reads_do_not_advance_history(self):
        env = self._configured_env(True, True)
        try:
            first, _ = env.reset(seed=17)
            second = env._observation()
            third = env._observation()
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(second, third)
            np.testing.assert_array_equal(env.previous_pose_error,
                                          env.current_observed_pose_error)
        finally:
            env.close()
            env._temporary_config_directory.cleanup()

    def test_new_episode_resets_but_curriculum_restore_preserves_history(self):
        env = self._configured_env(True, True)
        try:
            env.reset(seed=17)
            state = env.capture_curriculum_state()
            env.step(np.array([0, 0, -1, 0, 0, 0], dtype=float))
            reset_observation, _ = env.reset(seed=18)
            np.testing.assert_array_equal(reset_observation[:6], reset_observation[6:12])
            env.step(np.array([1, 0, 0, 0, 0, 0], dtype=float))
            restored_observation, _ = env.restore_curriculum_state(
                state, reset_episode=True, reset_source="curriculum_frontier",
            )
            scales = env.cfg["observation"]
            expected_previous = np.r_[
                state.previous_pose_error[:3] / scales["position_scale"],
                state.previous_pose_error[3:] / scales["rotation_scale"],
            ]
            np.testing.assert_allclose(restored_observation[6:12], expected_previous)
        finally:
            env.close()
            env._temporary_config_directory.cleanup()

    def test_curriculum_snapshot_keeps_previous_error_used_by_observation(self):
        env = self._configured_env(True, True)
        try:
            initial, _ = env.reset(seed=29)
            stepped, *_ = env.step(np.array([0, 0, -1, 0, 0, 0], dtype=float))
            state = env.capture_curriculum_state()
            scales = env.cfg["observation"]
            saved_normalized = np.r_[
                state.previous_pose_error[:3] / scales["position_scale"],
                state.previous_pose_error[3:] / scales["rotation_scale"],
            ]
            np.testing.assert_allclose(saved_normalized, stepped[6:12])
            restored_a, _ = env.restore_curriculum_state(state)
            env.step(np.array([1, 0, 0, 0, 0, 0], dtype=float))
            restored_b, _ = env.restore_curriculum_state(state)
            np.testing.assert_array_equal(restored_a, restored_b)
            np.testing.assert_allclose(restored_a[6:12], saved_normalized)
            self.assertFalse(np.shares_memory(
                state.previous_pose_error, env.previous_pose_error,
            ))
            self.assertEqual(initial.shape, (24,))
        finally:
            env.close()
            env._temporary_config_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
