from __future__ import annotations

import unittest
import numpy as np

from src.admittance import AdmittanceController


def config() -> dict:
    return {
        "mass": [2.0] * 6,
        "damping": [0.0] * 6,
        "stiffness": [10.0] * 6,
        "max_offset": [0.008] * 6,
        "max_velocity": [0.02] * 6,
    }


class AdmittanceTest(unittest.TestCase):
    def test_zero_wrench_stays_at_zero_and_reset_clears_state(self):
        controller = AdmittanceController(config())
        np.testing.assert_allclose(controller.step(np.zeros(6), 0.02), 0.0)
        controller.step(np.array([10, 0, 0, 0, 0, 0]), 0.02)
        controller.reset()
        np.testing.assert_allclose(controller.offset, 0.0)
        np.testing.assert_allclose(controller.velocity, 0.0)

    def test_velocity_is_bounded_before_offset_integration(self):
        controller = AdmittanceController(config())
        result = controller.step(np.array([100, 0, 0, 0, 0, 0]), 0.02)
        self.assertAlmostEqual(controller.velocity[0], 0.02)
        self.assertAlmostEqual(result[0], 0.0004)

    def test_saturation_cancels_only_outward_velocity(self):
        controller = AdmittanceController(config())
        controller.offset[0] = controller.offset_limit[0]
        controller.velocity[0] = 0.01
        controller.step(np.array([100, 0, 0, 0, 0, 0]), 0.02)
        self.assertEqual(controller.offset[0], controller.offset_limit[0])
        self.assertEqual(controller.velocity[0], 0.0)
        controller.step(np.array([-100, 0, 0, 0, 0, 0]), 0.02)
        self.assertLess(controller.offset[0], controller.offset_limit[0])
        self.assertLess(controller.velocity[0], 0.0)

    def test_stationary_absolute_offset_is_not_an_increment(self):
        controller = AdmittanceController(config())
        controller.offset[0] = 0.005
        equilibrium_wrench = np.zeros(6)
        equilibrium_wrench[0] = controller.stiffness[0] * controller.offset[0]
        first = controller.step(equilibrium_wrench, 0.02)
        second = controller.step(equilibrium_wrench, 0.02)
        self.assertAlmostEqual(first[0], 0.005)
        self.assertAlmostEqual(second[0], 0.005)


if __name__ == "__main__":
    unittest.main()
