from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.assembly_env import TenonMortaiseEnv, admittance_change_pose
from src.transforms import quat_to_rotvec, relative


class ReactiveAdmittanceIntegrationTest(unittest.TestCase):
    """Tests du raccordement admittance -> mocap -> weld MuJoCo réel."""

    def setUp(self):
        self.env = TenonMortaiseEnv("configs/test1V10.yaml")
        self.env.reset(seed=23)
        self.assertEqual(
            self.env.cfg["action"]["control_mode"], "reactive_actual_pose",
        )
        self.zero_action = np.zeros(6)
        self.mocap_id = self.env.model.body_mocapid[self.env.target_mocap]

    def tearDown(self):
        self.env.close()

    def grasp_pose(self):
        return (
            self.env.data.site_xpos[self.env.grasp_site].copy(),
            self.env._site_quat(),
        )

    def commanded_pose(self):
        return (
            self.env.data.mocap_pos[self.mocap_id].copy(),
            self.env.data.mocap_quat[self.mocap_id].copy(),
        )

    @staticmethod
    def pose_error(reference, pose):
        error = relative(reference, pose)
        return np.r_[error[0], quat_to_rotvec(error[1])]

    def test_constant_absolute_offset_is_not_reapplied_by_real_environment(self):
        constant_offset = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.env.admittance.offset = constant_offset.copy()
        self.env.admittance.velocity[:] = 0.0
        measurements = []

        equilibrium_wrench = self.env.admittance.stiffness * constant_offset

        with patch.object(
            self.env, "_observed_wrench", return_value=equilibrium_wrench,
        ):
            for step_index in range(3):
                actual_before = self.grasp_pose()
                previous_offset = self.env.admittance.offset.copy()
                self.env.step(self.zero_action)
                new_offset = self.env.admittance.offset.copy()
                delta_pose = admittance_change_pose(previous_offset, new_offset)
                commanded = self.commanded_pose()
                actual_after = self.grasp_pose()
                measurements.append({
                    "step": step_index,
                    "actual_before": actual_before,
                    "actual_after": actual_after,
                    "commanded": commanded,
                    "previous_offset": previous_offset,
                    "new_offset": new_offset,
                    "delta": self.pose_error((np.zeros(3), np.array([1., 0., 0., 0.])), delta_pose),
                    "command_from_actual": self.pose_error(actual_before, commanded),
                    "tracking_error": self.pose_error(commanded, actual_after),
                })

        for measurement in measurements:
            np.testing.assert_allclose(
                measurement["previous_offset"], constant_offset, atol=1e-15,
            )
            np.testing.assert_allclose(
                measurement["new_offset"], constant_offset, atol=1e-15,
            )
            np.testing.assert_allclose(measurement["delta"], np.zeros(6), atol=1e-12)
            np.testing.assert_allclose(
                measurement["command_from_actual"], np.zeros(6), atol=1e-12,
            )

    def test_unreached_admittance_correction_is_not_reapplied_next_cycle(self):
        initial_offset = np.zeros(6)
        # Avec max_velocity.x=20 mm/s et dt=20 ms, le vrai intégrateur produit
        # au maximum 0.4 mm pendant ce premier cycle.
        changed_offset = np.array([0.0004, 0.0, 0.0, 0.0, 0.0, 0.0])
        measurements = []

        def controlled_wrench():
            if self.env.steps == 0:
                return np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # Après le premier cycle, fixer proprement l'état dynamique à son
            # équilibre pour que le vrai AdmittanceController.step reste stable.
            self.env.admittance.velocity[:] = 0.0
            return self.env.admittance.stiffness * self.env.admittance.offset

        self.env.admittance.offset = initial_offset.copy()
        self.env.admittance.velocity[:] = 0.0
        with patch.object(self.env, "_observed_wrench", side_effect=controlled_wrench):
            for step_index in range(2):
                actual_before = self.grasp_pose()
                previous_offset = self.env.admittance.offset.copy()
                self.env.step(self.zero_action)
                new_offset = self.env.admittance.offset.copy()
                delta_pose = admittance_change_pose(previous_offset, new_offset)
                commanded = self.commanded_pose()
                actual_after = self.grasp_pose()
                measurements.append({
                    "step": step_index,
                    "actual_before": actual_before,
                    "actual_after": actual_after,
                    "commanded": commanded,
                    "previous_offset": previous_offset,
                    "new_offset": new_offset,
                    "delta": self.pose_error((np.zeros(3), np.array([1., 0., 0., 0.])), delta_pose),
                    "command_from_actual": self.pose_error(actual_before, commanded),
                    "tracking_error": self.pose_error(commanded, actual_after),
                })

        first, second = measurements
        np.testing.assert_allclose(first["previous_offset"], initial_offset, atol=1e-15)
        np.testing.assert_allclose(first["new_offset"], changed_offset, atol=1e-15)
        self.assertAlmostEqual(first["delta"][0], changed_offset[0], places=12)
        self.assertAlmostEqual(first["command_from_actual"][0], changed_offset[0], places=12)

        np.testing.assert_allclose(second["previous_offset"], changed_offset, atol=1e-15)
        np.testing.assert_allclose(second["new_offset"], changed_offset, atol=1e-15)
        np.testing.assert_allclose(second["delta"], np.zeros(6), atol=1e-12)
        np.testing.assert_allclose(second["command_from_actual"], np.zeros(6), atol=1e-12)

        # Valeurs conservées pour le compte rendu diagnostique du test.
        self.first_cycle_tracking_translation = float(np.linalg.norm(first["tracking_error"][:3]))
        self.first_cycle_tracking_rotation = float(np.linalg.norm(first["tracking_error"][3:]))
        self.max_tracking_translation = max(
            float(np.linalg.norm(item["tracking_error"][:3])) for item in measurements
        )
        self.max_tracking_rotation = max(
            float(np.linalg.norm(item["tracking_error"][3:])) for item in measurements
        )


if __name__ == "__main__":
    unittest.main()
