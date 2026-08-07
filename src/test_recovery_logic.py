from __future__ import annotations

from collections import deque
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.assembly_env import AssemblyEnv, ResidualConfig
from src.evaluate_nominal import nominal_config


def recovery_env() -> AssemblyEnv:
    """Build only the Python state needed by recovery logic, not MuJoCo/SDF."""
    env = object.__new__(AssemblyEnv)
    env.config = replace(
        ResidualConfig(),
        recovery_force_steps=2,
        recovery_torque_steps=2,
        recovery_min_steps=2,
        recovery_clear_steps=2,
        recovery_effort_persistence_steps=2,
        stall_window_steps=2,
    )
    env._control_mode = "tracking"
    env._contact_search_latched = False
    env._contact_search_trigger = "none"
    env._contact_search_count = 0
    env._recovery_count = 0
    env._recovery_from_contact_search_count = 0
    env._last_recovery_trigger = "none"
    env._recovery_trigger_contact = False
    env._recovery_trigger_force = 0.0
    env._recovery_trigger_torque = 0.0
    env._recovery_steps = 0
    env._recovery_attempt_duration = 0.0
    env._recovery_duration = 0.0
    env._soft_force_steps = 0
    env._soft_torque_steps = 0
    env._recovery_effort_steps = 0
    env._forced_retreat = False
    env._clear_steps = 0
    env._stuck_detected = False
    env._stagnation_errors = deque(maxlen=3)
    env._mode_steps = {"tracking": 0, "contact_search": 0, "recovery": 0}
    env._progress_action_sum = 0.0
    env._effective_progress_sum = 0.0
    env._advance_steps = 0
    env._hold_steps = 0
    env._retreat_steps = 0
    env._progress = 0.9
    env._max_progress = 0.9
    env._progress_rate = 0.0
    env._previous_action = np.zeros(7)
    env._residual_offset = np.zeros(6)
    env._admittance_offset = np.zeros(6)
    env._admittance_velocity = np.zeros(6)
    return env


class RecoveryLogicTest(unittest.TestCase):
    def test_contact_threshold_defaults_are_calibrated(self) -> None:
        config = ResidualConfig()
        self.assertEqual(config.soft_torque, 4.5)
        self.assertEqual(config.recovery_torque_steps, 5)
        self.assertEqual(config.hard_torque, 8.0)
        self.assertEqual(
            config.actuator_force_limits,
            (250.0, 250.0, 300.0, 30.0, 30.0, 30.0),
        )

    def test_nominal_baseline_has_no_variability_or_recovery(self) -> None:
        config = nominal_config(progress_speed=0.1)
        self.assertEqual(config.progress_speed, 0.1)
        self.assertFalse(config.contact_search_enabled)
        self.assertFalse(config.recovery_enabled)
        for field in (
            "initial_linear_error",
            "initial_angular_error",
            "fixture_linear_error",
            "fixture_angular_error",
            "grasp_linear_error",
            "grasp_angular_error",
        ):
            self.assertEqual(getattr(config, field), 0.0)

    def test_zero_action_means_nominal_progress_while_tracking(self) -> None:
        env = recovery_env()
        for progress in (0.0, 0.4, 0.80, 0.825, 0.85, 1.0):
            with self.subTest(progress=progress):
                env._progress = progress
                env._control_mode = "tracking"
                self.assertEqual(env._effective_progress_request(0.0), 1.0)
                self.assertEqual(env._effective_progress_request(-1.0), 0.0)
                self.assertEqual(env._effective_progress_request(1.0), 1.5)

    def test_all_six_residual_actions_remain_available_at_every_progress(self) -> None:
        env = recovery_env()
        action = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.0])

        for progress in (0.0, 0.2, 0.79, 0.80, 0.825, 0.85, 1.0):
            with self.subTest(progress=progress):
                env._progress = progress
                env._control_mode = "tracking"
                np.testing.assert_allclose(
                    env._residual_action_for_mode(action, "tracking"),
                    action[:6],
                )

    def test_contact_at_any_progress_enters_and_latches_contact_search(self) -> None:
        for progress in (0.0, 0.2, 0.79, 0.85, 1.0):
            with self.subTest(progress=progress):
                env = recovery_env()
                env._progress = progress
                env._control_mode = "tracking"
                env._update_recovery_state(
                    force=1.0,
                    torque=0.1,
                    contact=True,
                    requested_progress=1.0,
                    position_error=0.05,
                    rotation_error=0.2,
                )

                self.assertEqual(env._control_mode, "contact_search")
                self.assertTrue(env._contact_search_latched)
                self.assertEqual(env._contact_search_count, 1)

                # Losing contact later must not silently return to tracking.
                env._update_recovery_state(
                    force=0.0,
                    torque=0.0,
                    contact=False,
                    requested_progress=0.0,
                    position_error=0.05,
                    rotation_error=0.2,
                )
                self.assertEqual(env._control_mode, "contact_search")
                self.assertTrue(env._contact_search_latched)
                self.assertEqual(env._contact_search_count, 1)

    def test_inertial_wrench_without_contact_cannot_latch_or_recover(self) -> None:
        env = recovery_env()
        env._progress = 0.2
        env._max_progress = 0.2

        for _ in range(10):
            env._update_recovery_state(
                force=30.0,
                torque=5.0,
                contact=False,
                requested_progress=0.0,
                position_error=0.1,
                rotation_error=0.2,
            )

        self.assertEqual(env._control_mode, "tracking")
        self.assertFalse(env._contact_search_latched)
        self.assertEqual(env._recovery_count, 0)

    def test_progress_frontier_rewards_each_interval_only_once(self) -> None:
        env = recovery_env()
        env._max_progress = 0.4

        self.assertAlmostEqual(env._update_progress_frontier(0.5), 0.1)
        self.assertEqual(env._update_progress_frontier(0.3), 0.0)
        self.assertEqual(env._update_progress_frontier(0.5), 0.0)
        self.assertAlmostEqual(env._update_progress_frontier(0.55), 0.05)
        self.assertAlmostEqual(env._max_progress, 0.55)

    def test_contact_search_progress_semantics_allow_slow_advance_and_retreat(self) -> None:
        env = recovery_env()
        env._enter_contact_search()
        self.assertEqual(env._effective_progress_request(0.0), 0.25)
        self.assertEqual(env._effective_progress_request(-0.25), 0.0)
        self.assertEqual(env._effective_progress_request(-1.0), -0.75)
        self.assertEqual(env._effective_progress_request(1.0), 0.5)

    def test_contact_search_progress_scale_is_bounded_and_monotonic(self) -> None:
        env = recovery_env()
        self.assertEqual(env._contact_progress_scale(0.0, 0.0), 1.0)
        self.assertEqual(env._contact_progress_scale(20.0, 4.5), 1.0)
        middle = env._contact_progress_scale(50.0, 6.25)
        self.assertGreater(middle, 0.0)
        self.assertLess(middle, 1.0)
        self.assertEqual(env._contact_progress_scale(80.0, 8.0), 0.0)

    def test_contact_search_and_recovery_keep_all_residual_actions(self) -> None:
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])
        np.testing.assert_allclose(
            AssemblyEnv._residual_action_for_mode(action, "contact_search"),
            action[:6],
        )
        np.testing.assert_allclose(
            AssemblyEnv._residual_action_for_mode(action, "recovery"),
            action[:6],
        )

    def test_tactile_latch_enables_large_residual_limit_with_admittance(self) -> None:
        env = recovery_env()
        env._progress = 1.0

        # Progress alone must not enlarge the Cartesian search authority.
        self.assertEqual(
            env._linear_limit_for_state("tracking"),
            env.config.residual_linear_limit,
        )

        env._enter_contact_search("contact")
        residual_limit = env._linear_limit_for_state("contact_search")
        self.assertEqual(
            residual_limit,
            env.config.terminal_residual_linear_limit,
        )
        self.assertAlmostEqual(
            residual_limit + env.config.admittance_offset_limit,
            0.030,
        )

        # The tactile latch, rather than a spatial threshold, preserves the
        # larger limit through a retreat and a recovery mode.
        env._progress = 0.5
        self.assertEqual(env._linear_limit_for_state("recovery"), residual_limit)

    def test_non_tactile_recovery_limit_closes_after_return_to_tracking(self) -> None:
        env = recovery_env()
        self.assertFalse(env._contact_search_latched)
        self.assertEqual(
            env._linear_limit_for_state("recovery"),
            env.config.terminal_residual_linear_limit,
        )
        self.assertEqual(
            env._linear_limit_for_state("tracking"),
            env.config.residual_linear_limit,
        )

    def test_admittance_unloads_opposite_to_measured_wrench(self) -> None:
        env = recovery_env()
        env.config = replace(
            env.config,
            decision_dt=0.1,
            admittance_mass=1.0,
            admittance_damping=0.0,
            admittance_stiffness=0.0,
            admittance_offset_limit=1.0,
        )
        env._admittance_velocity = np.zeros(6)
        env._admittance_offset = np.zeros(6)
        env._update_admittance(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertLess(env._admittance_velocity[0], 0.0)
        self.assertLess(env._admittance_offset[0], 0.0)

    def test_wrench_is_rotated_from_wrist_frame_to_world_frame(self) -> None:
        env = recovery_env()
        env.force_sensor_id = 1
        env.torque_sensor_id = 2
        env.wrist_site_id = 0
        env._wrench_bias = np.zeros(6)
        env.data = SimpleNamespace(
            site_xmat=np.array(
                [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
            )
        )
        env._sensor = lambda sensor_id: (
            np.array([1.0, 0.0, 0.0])
            if sensor_id == env.force_sensor_id
            else np.array([0.0, 1.0, 0.0])
        )

        np.testing.assert_allclose(
            env._wrench(),
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_admittance_ignores_inertial_wrench_without_tactile_evidence(self) -> None:
        env = recovery_env()
        env.config = replace(
            env.config,
            decision_dt=0.1,
            admittance_mass=1.0,
            admittance_damping=0.0,
            admittance_stiffness=0.0,
            admittance_offset_limit=1.0,
        )
        env._admittance_velocity = np.zeros(6)
        env._admittance_offset = np.zeros(6)

        env._update_admittance(
            np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            tactile_active=False,
        )

        np.testing.assert_allclose(env._admittance_velocity, 0.0)
        np.testing.assert_allclose(env._admittance_offset, 0.0)

    def test_normalized_controller_state_contains_offsets_and_velocity(self) -> None:
        env = recovery_env()
        env._progress = 0.5
        linear_limit = env.config.residual_linear_limit
        angular_limit = env.config.residual_angular_limit
        admittance_limit = env.config.admittance_offset_limit
        admittance_velocity_limit = getattr(
            env.config,
            "admittance_velocity_limit",
            0.04,
        )
        env._residual_offset = np.array([
            0.5 * linear_limit,
            -linear_limit,
            0.0,
            0.5 * angular_limit,
            0.0,
            -angular_limit,
        ])
        env._admittance_offset = admittance_limit * np.array([
            0.5,
            -1.0,
            0.0,
            1.0,
            -0.5,
            0.0,
        ])
        env._admittance_velocity = admittance_velocity_limit * np.array([
            -1.0,
            0.5,
            0.0,
            1.0,
            0.0,
            -0.5,
        ])

        state = env._normalized_controller_state()

        self.assertEqual(state.shape, (18,))
        np.testing.assert_allclose(
            state[:6],
            [0.5, -1.0, 0.0, 0.5, 0.0, -1.0],
        )
        np.testing.assert_allclose(
            state[6:12],
            [0.5, -1.0, 0.0, 1.0, -0.5, 0.0],
        )
        np.testing.assert_allclose(
            state[12:18],
            [-1.0, 0.5, 0.0, 1.0, 0.0, -0.5],
        )

    def test_base_observation_appends_normalized_controller_state(self) -> None:
        env = recovery_env()
        env._progress = 0.5
        env._contact_search_latched = True
        env.data = SimpleNamespace(
            qpos=np.zeros(6),
            qvel=np.zeros(6),
        )
        env.qpos_adr = np.arange(6)
        env.dof_adr = np.arange(6)
        env._path_qpos = lambda progress: np.zeros(6)
        env._has_contact = lambda: False
        env._residual_offset[0] = 0.5 * env.config.residual_linear_limit
        env._admittance_offset[1] = -env.config.admittance_offset_limit
        env._admittance_velocity[2] = 0.02

        observation = env._base_obs(np.zeros(6))

        self.assertEqual(observation.shape, (56,))
        np.testing.assert_allclose(
            observation[-19:-1],
            env._normalized_controller_state(),
        )
        self.assertEqual(observation[-1], 1.0)

    def test_residual_offset_cost_is_symmetric_and_monotonic(self) -> None:
        env = recovery_env()
        linear_limit = env.config.residual_linear_limit

        env._residual_offset.fill(0.0)
        zero_cost = env._residual_offset_cost(linear_limit)
        env._residual_offset[0] = 0.5 * linear_limit
        half_cost = env._residual_offset_cost(linear_limit)
        env._residual_offset[0] = linear_limit
        full_cost = env._residual_offset_cost(linear_limit)
        env._residual_offset[0] = -linear_limit
        negative_full_cost = env._residual_offset_cost(linear_limit)
        env._residual_offset[1] = linear_limit
        two_axis_cost = env._residual_offset_cost(linear_limit)

        self.assertEqual(zero_cost, 0.0)
        self.assertGreater(half_cost, zero_cost)
        self.assertGreater(full_cost, half_cost)
        self.assertAlmostEqual(negative_full_cost, full_cost)
        self.assertGreater(two_axis_cost, full_cost)

    def test_residual_offset_cost_uses_fraction_of_active_linear_limit(self) -> None:
        env = recovery_env()
        base_limit = env.config.residual_linear_limit
        terminal_limit = env.config.terminal_residual_linear_limit

        env._residual_offset.fill(0.0)
        env._residual_offset[0] = 0.5 * base_limit
        base_cost = env._residual_offset_cost(base_limit)
        env._residual_offset[0] = 0.5 * terminal_limit
        terminal_cost = env._residual_offset_cost(terminal_limit)

        self.assertAlmostEqual(base_cost, terminal_cost)

    def test_persistent_soft_force_enters_recovery(self) -> None:
        env = recovery_env()
        env._progress = 0.5
        for _ in range(2):
            env._update_recovery_state(
                force=21.0, torque=0.0, contact=True, requested_progress=1.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "recovery")
        self.assertEqual(env._recovery_count, 1)
        self.assertTrue(env._stuck_detected)
        self.assertTrue(env._recovery_trigger_contact)
        self.assertEqual(env._recovery_trigger_force, 21.0)

    def test_progress_alone_does_not_enter_contact_search(self) -> None:
        env = recovery_env()
        env._progress = env.config.stall_progress
        env._update_recovery_state(
            force=0.0,
            torque=0.0,
            contact=False,
            requested_progress=1.0,
            position_error=0.02,
            rotation_error=0.2,
        )

        self.assertEqual(env._control_mode, "tracking")
        self.assertFalse(env._contact_search_latched)
        self.assertEqual(env._contact_search_count, 0)

    def test_tracking_stagnation_without_contact_enters_recovery(self) -> None:
        env = recovery_env()
        env._progress = env.config.stall_progress
        for _ in range(env._stagnation_errors.maxlen):
            env._update_recovery_state(
                force=0.0,
                torque=0.0,
                contact=False,
                requested_progress=1.0,
                position_error=0.02,
                rotation_error=0.2,
            )

        self.assertEqual(env._control_mode, "recovery")
        self.assertEqual(env._last_recovery_trigger, "stagnation")
        self.assertFalse(env._contact_search_latched)
        self.assertEqual(env._contact_search_count, 0)

    def test_expected_contact_torque_does_not_trigger_recovery(self) -> None:
        env = recovery_env()
        env._enter_contact_search()
        for index in range(10):
            env._update_recovery_state(
                force=5.0, torque=4.07, contact=True, requested_progress=0.25,
                position_error=0.03 - index * 0.001,
                rotation_error=0.3 - index * 0.01,
            )
        self.assertEqual(env._control_mode, "contact_search")
        self.assertEqual(env._recovery_count, 0)

    def test_persistent_soft_effort_enters_recovery_during_contact_search(self) -> None:
        env = recovery_env()
        env._enter_contact_search()
        for index in range(env.config.recovery_force_steps):
            env._update_recovery_state(
                force=25.0, torque=4.6, contact=True,
                requested_progress=0.25,
                position_error=0.03 - index * 0.001,
                rotation_error=0.3 - index * 0.01,
            )
        self.assertEqual(env._control_mode, "recovery")
        self.assertEqual(env._recovery_count, 1)

    def test_improving_weak_contact_stays_in_contact_search(self) -> None:
        env = recovery_env()
        env._enter_contact_search()
        for index in range(6):
            env._update_recovery_state(
                force=10.0, torque=2.0, contact=True,
                requested_progress=0.25,
                position_error=0.03 - index * 0.001,
                rotation_error=0.3 - index * 0.01,
            )
        self.assertEqual(env._control_mode, "contact_search")
        self.assertEqual(env._recovery_count, 0)

    def test_stagnation_near_goal_enters_recovery(self) -> None:
        env = recovery_env()
        for _ in range(3):
            env._update_recovery_state(
                force=5.0, torque=1.0, contact=True, requested_progress=1.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "recovery")

    def test_unresolved_endpoint_without_contact_recovers_without_tactile_latch(self) -> None:
        env = recovery_env()
        env._progress = 1.0
        for _ in range(3):
            env._update_recovery_state(
                force=0.0, torque=0.0, contact=False, requested_progress=0.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "recovery")
        self.assertEqual(env._last_recovery_trigger, "stagnation")
        self.assertFalse(env._contact_search_latched)

    def test_recovery_blocks_forward_progress_and_exits_after_clearance(self) -> None:
        env = recovery_env()
        env._enter_contact_search()
        env._enter_recovery()
        self.assertEqual(env._effective_progress_request(0.7), 0.0)
        self.assertEqual(env._effective_progress_request(-0.7), -0.7)
        for _ in range(2):
            env._record_control_step("recovery", 0.0, 0.0)
            env._update_recovery_state(
                force=0.0, torque=0.0, contact=True, requested_progress=0.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "recovery")
        for _ in range(2):
            env._record_control_step("recovery", 0.0, 0.0)
            env._update_recovery_state(
                force=0.0, torque=0.0, contact=False, requested_progress=0.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "contact_search")

    def test_early_recovery_returns_to_tracking(self) -> None:
        env = recovery_env()
        env._progress = 0.5
        env._enter_recovery()
        for _ in range(2):
            env._record_control_step("recovery", 0.0, 0.0)
            env._update_recovery_state(
                force=0.0, torque=0.0, contact=False, requested_progress=0.0,
                position_error=0.1, rotation_error=0.2,
            )
        self.assertEqual(env._control_mode, "tracking")

    def test_persistent_soft_torque_enters_recovery_and_forces_retreat(self) -> None:
        env = recovery_env()
        env._progress = 0.5
        for _ in range(2):
            env._update_recovery_state(
                force=0.0, torque=4.6, contact=True, requested_progress=1.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "recovery")
        self.assertTrue(env._forced_retreat)
        self.assertEqual(env._effective_progress_request(0.0), -1.0)

    def test_soft_torque_persistence_resets_below_threshold(self) -> None:
        env = recovery_env()
        env._enter_contact_search()
        env._update_recovery_state(
            force=0.0, torque=4.6, contact=True, requested_progress=0.0,
            position_error=0.01, rotation_error=0.1,
        )
        env._update_recovery_state(
            force=0.0,
            torque=env.config.soft_torque - 0.1,
            contact=True,
            requested_progress=0.0,
            position_error=0.01, rotation_error=0.1,
        )
        env._update_recovery_state(
            force=0.0, torque=4.6, contact=True, requested_progress=0.0,
            position_error=0.01, rotation_error=0.1,
        )
        self.assertEqual(env._control_mode, "contact_search")

    def test_recovery_fails_after_its_bounded_duration(self) -> None:
        env = recovery_env()
        env._enter_recovery()
        env._recovery_attempt_duration = env.config.recovery_max_duration_s
        self.assertTrue(env._recovery_has_failed())

    def test_new_recovery_resets_only_attempt_duration(self) -> None:
        env = recovery_env()
        env._recovery_duration = 1.5
        env._recovery_attempt_duration = 1.5
        env._enter_recovery()
        self.assertEqual(env._recovery_attempt_duration, 0.0)
        self.assertEqual(env._recovery_duration, 1.5)

    def test_cumulative_recovery_duration_does_not_fail_a_new_attempt(self) -> None:
        env = recovery_env()
        env._control_mode = "recovery"
        env._recovery_duration = env.config.recovery_max_duration_s + 1.0
        env._recovery_attempt_duration = 0.2
        self.assertFalse(env._recovery_has_failed())

    def test_recovery_can_be_disabled_for_nominal_baseline(self) -> None:
        env = recovery_env()
        env.config = replace(
            env.config,
            contact_search_enabled=False,
            recovery_enabled=False,
        )
        env._update_recovery_state(
            force=100.0, torque=10.0, contact=True, requested_progress=1.0,
            position_error=0.01, rotation_error=0.1,
        )
        self.assertEqual(env._control_mode, "tracking")
        self.assertEqual(env._recovery_count, 0)

    def test_mode_metrics_credit_the_mode_that_executed_the_action(self) -> None:
        env = recovery_env()
        env._record_control_step("tracking", 0.0, 1.0)
        env._record_control_step("contact_search", -0.25, 0.0)
        env._record_control_step("recovery", -1.0, -1.0)
        self.assertEqual(env._mode_steps, {
            "tracking": 1,
            "contact_search": 1,
            "recovery": 1,
        })
        self.assertEqual(env._advance_steps, 1)
        self.assertEqual(env._hold_steps, 1)
        self.assertEqual(env._retreat_steps, 1)
        self.assertAlmostEqual(env._recovery_duration, env.config.decision_dt)

    def test_soft_torque_does_not_interrupt_physics_substeps(self) -> None:
        env = recovery_env()
        env.frame_skip = 7
        env.model = object()
        env.data = object()
        env._wrench = lambda: np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        env._has_contact = lambda: False
        with patch("src.assembly_env.mujoco.mj_step") as mj_step:
            _, peak_torque, contact = env._run_control_substeps()
        self.assertEqual(mj_step.call_count, 7)
        self.assertEqual(peak_torque, 5.0)
        self.assertFalse(contact)

    def test_hard_torque_interrupts_physics_substeps(self) -> None:
        env = recovery_env()
        env.frame_skip = 7
        env.model = object()
        env.data = object()
        wrenches = iter([
            np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 7.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 8.1, 0.0, 0.0]),
        ])
        env._wrench = lambda: next(wrenches)
        env._has_contact = lambda: False
        with patch("src.assembly_env.mujoco.mj_step") as mj_step:
            _, peak_torque, _ = env._run_control_substeps()
        self.assertEqual(mj_step.call_count, 3)
        self.assertEqual(peak_torque, 8.1)

    def test_hard_safety_thresholds_are_inclusive(self) -> None:
        env = recovery_env()

        self.assertEqual(
            env._is_hard_unsafe(
                env.config.hard_force - 1e-9,
                env.config.hard_torque - 1e-9,
            ),
            (False, False),
        )
        self.assertEqual(
            env._is_hard_unsafe(env.config.hard_force, 0.0),
            (True, False),
        )
        self.assertEqual(
            env._is_hard_unsafe(0.0, env.config.hard_torque),
            (False, True),
        )
        self.assertEqual(
            env._is_hard_unsafe(
                env.config.hard_force,
                env.config.hard_torque,
            ),
            (True, True),
        )

    def test_exact_hard_torque_interrupts_physics_substeps(self) -> None:
        env = recovery_env()
        env.frame_skip = 7
        env.model = object()
        env.data = object()
        env._wrench = lambda: np.array([
            0.0,
            0.0,
            0.0,
            env.config.hard_torque,
            0.0,
            0.0,
        ])
        env._has_contact = lambda: False

        with patch("src.assembly_env.mujoco.mj_step") as mj_step:
            _, peak_torque, _ = env._run_control_substeps()

        self.assertEqual(mj_step.call_count, 1)
        self.assertEqual(peak_torque, env.config.hard_torque)

    def test_brief_substep_contact_is_not_lost(self) -> None:
        env = recovery_env()
        env.frame_skip = 3
        env.model = object()
        env.data = object()
        env._wrench = lambda: np.zeros(6)
        contacts = iter([False, True, False])
        env._has_contact = lambda: next(contacts)

        with patch("src.assembly_env.mujoco.mj_step"):
            _, _, contact = env._run_control_substeps()

        self.assertTrue(contact)


if __name__ == "__main__":
    unittest.main()
