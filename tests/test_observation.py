from __future__ import annotations

import unittest

import numpy as np

from src.assembly_env import TenonMortaiseEnv


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


if __name__ == "__main__":
    unittest.main()
