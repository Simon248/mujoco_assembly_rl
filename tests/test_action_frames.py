from __future__ import annotations

import unittest
import numpy as np

from src.assembly_env import TenonMortaiseEnv, advance_grasp_reference
from src.transforms import (
    compose, euler_xyz_to_quat, inverse, quat_to_rotvec, relative,
    rotate, rotvec_to_quat,
)


class ActionFrameGeometryTest(unittest.TestCase):
    def setUp(self):
        self.task_target = (
            np.array([0.3, -0.2, 0.4]),
            euler_xyz_to_quat(np.deg2rad([20.0, -15.0, 35.0])),
        )
        self.task_pose = compose(self.task_target, (
            np.array([0.02, -0.01, 0.03]),
            euler_xyz_to_quat(np.deg2rad([8.0, -6.0, 12.0])),
        ))
        self.task_to_grasp = (
            np.array([0.127, 0.0045, 0.1925]),
            euler_xyz_to_quat(np.deg2rad([15.0, -10.0, 25.0])),
        )
        self.grasp_pose = compose(self.task_pose, self.task_to_grasp)

    def recovered_task(self, grasp_pose):
        return compose(grasp_pose, inverse(self.task_to_grasp))

    def assert_pose_close(self, actual, expected, atol=1e-12):
        error = relative(expected, actual)
        np.testing.assert_allclose(error[0], np.zeros(3), atol=atol)
        np.testing.assert_allclose(quat_to_rotvec(error[1]), np.zeros(3), atol=atol)

    def test_task_translation_follows_each_cad_axis_without_cross_axis_error(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                translation = np.zeros(3); translation[axis] = 0.001
                desired_grasp = advance_grasp_reference(
                    self.grasp_pose,
                    self.task_to_grasp,
                    (translation, rotvec_to_quat(np.zeros(3))),
                    "task",
                    self.task_target,
                )
                desired_task = self.recovered_task(desired_grasp)
                before = relative(self.task_target, self.task_pose)
                after = relative(self.task_target, desired_task)
                np.testing.assert_allclose(after[0] - before[0], translation, atol=1e-12)
                np.testing.assert_allclose(
                    desired_task[0] - self.task_pose[0],
                    rotate(self.task_target[1], translation),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    quat_to_rotvec(relative(self.task_pose, desired_task)[1]),
                    np.zeros(3), atol=1e-12,
                )

    def test_task_rotation_keeps_cad_origin_fixed(self):
        for axis in (0, 1):
            with self.subTest(axis=axis):
                rotation = np.zeros(3); rotation[axis] = np.deg2rad(1.0)
                desired_grasp = advance_grasp_reference(
                    self.grasp_pose,
                    self.task_to_grasp,
                    (np.zeros(3), rotvec_to_quat(rotation)),
                    "task",
                    self.task_target,
                )
                desired_task = self.recovered_task(desired_grasp)
                before = relative(self.task_target, self.task_pose)
                after = relative(self.task_target, desired_task)
                expected_orientation = compose(
                    (np.zeros(3), rotvec_to_quat(rotation)),
                    (np.zeros(3), before[1]),
                )[1]
                np.testing.assert_allclose(desired_task[0], self.task_pose[0], atol=1e-12)
                np.testing.assert_allclose(after[0], before[0], atol=1e-12)
                np.testing.assert_allclose(
                    quat_to_rotvec(relative(
                        (np.zeros(3), expected_orientation),
                        (np.zeros(3), after[1]),
                    )[1]),
                    np.zeros(3), atol=1e-12,
                )

    def test_object_to_grasp_to_object_round_trip(self):
        grasp = compose(self.task_pose, self.task_to_grasp)
        self.assert_pose_close(self.recovered_task(grasp), self.task_pose)

    def test_grasp_mode_is_exactly_the_historical_right_composition(self):
        translation = np.array([0.001, -0.0005, 0.00025])
        rotation = np.deg2rad([0.5, -0.25, 1.0])
        delta = (translation, rotvec_to_quat(rotation))
        expected = compose(self.grasp_pose, delta)
        actual = advance_grasp_reference(
            self.grasp_pose, self.task_to_grasp, delta, "grasp",
        )
        self.assert_pose_close(actual, expected)

    def test_historical_grasp_rotation_moves_cad_origin_through_lever_arm(self):
        rotation = np.array([0.0, np.deg2rad(1.0), 0.0])
        desired_grasp = advance_grasp_reference(
            self.grasp_pose,
            self.task_to_grasp,
            (np.zeros(3), rotvec_to_quat(rotation)),
            "grasp",
        )
        displacement = np.linalg.norm(self.recovered_task(desired_grasp)[0] - self.task_pose[0])
        self.assertGreater(displacement, 0.001)

    def test_task_environment_smoke_logs_clipped_policy_action_and_error_axes(self):
        env = TenonMortaiseEnv("configs/test1V5.yaml")
        try:
            observation, _ = env.reset(seed=17)
            self.assertEqual(observation.shape, (12,))
            _, _, _, _, info = env.step(np.array([2.0, -2.0, 0.25, 1.5, -1.5, 0.5]))
            np.testing.assert_allclose(
                [info["position_error_x"], info["position_error_y"], info["position_error_z"],
                 info["rotation_error_x"], info["rotation_error_y"], info["rotation_error_z"]],
                info["true_error"],
            )
            np.testing.assert_allclose(
                [info["action_x"], info["action_y"], info["action_z"],
                 info["action_rx"], info["action_ry"], info["action_rz"]],
                [1.0, -1.0, 0.25, 1.0, -1.0, 0.5],
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
