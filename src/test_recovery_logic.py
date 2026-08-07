from __future__ import annotations

from collections import deque
from dataclasses import replace
import unittest

from src.assembly_env import AssemblyEnv, ResidualConfig


def recovery_env() -> AssemblyEnv:
    """Build only the Python state needed by recovery logic, not MuJoCo/SDF."""
    env = object.__new__(AssemblyEnv)
    env.config = replace(
        ResidualConfig(),
        recovery_force_steps=2,
        recovery_torque_steps=1,
        recovery_min_steps=2,
        recovery_clear_steps=2,
        recovery_effort_persistence_steps=2,
        stall_window_steps=2,
    )
    env._control_mode = "tracking"
    env._recovery_count = 0
    env._recovery_steps = 0
    env._recovery_duration = 0.0
    env._soft_force_steps = 0
    env._soft_torque_steps = 0
    env._recovery_effort_steps = 0
    env._forced_retreat = False
    env._clear_steps = 0
    env._stuck_detected = False
    env._stagnation_errors = deque(maxlen=3)
    env._progress = 0.9
    return env


class RecoveryLogicTest(unittest.TestCase):
    def test_persistent_soft_force_enters_recovery(self) -> None:
        env = recovery_env()
        for _ in range(2):
            env._update_recovery_state(
                force=21.0, torque=0.0, contact=True, requested_progress=1.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "recovery")
        self.assertEqual(env._recovery_count, 1)
        self.assertTrue(env._stuck_detected)

    def test_stagnation_near_goal_enters_recovery(self) -> None:
        env = recovery_env()
        for _ in range(3):
            env._update_recovery_state(
                force=0.0, torque=0.0, contact=False, requested_progress=1.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "recovery")

    def test_unresolved_endpoint_enters_recovery_without_forward_request(self) -> None:
        env = recovery_env()
        env._progress = 1.0
        for _ in range(3):
            env._update_recovery_state(
                force=0.0, torque=0.0, contact=False, requested_progress=0.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "recovery")

    def test_recovery_blocks_forward_progress_and_exits_after_clearance(self) -> None:
        env = recovery_env()
        env._enter_recovery()
        self.assertEqual(env._effective_progress_request(0.7), 0.0)
        self.assertEqual(env._effective_progress_request(-0.7), -0.7)
        for _ in range(2):
            env._update_recovery_state(
                force=0.0, torque=0.0, contact=False, requested_progress=0.0,
                position_error=0.01, rotation_error=0.1,
            )
        self.assertEqual(env._control_mode, "tracking")

    def test_persistent_soft_torque_enters_recovery_and_forces_retreat(self) -> None:
        env = recovery_env()
        env._update_recovery_state(
            force=0.0, torque=2.1, contact=True, requested_progress=1.0,
            position_error=0.01, rotation_error=0.1,
        )
        self.assertEqual(env._control_mode, "recovery")
        self.assertTrue(env._forced_retreat)
        self.assertEqual(env._effective_progress_request(0.0), -1.0)

    def test_recovery_fails_after_its_bounded_duration(self) -> None:
        env = recovery_env()
        env._enter_recovery()
        env._recovery_duration = env.config.recovery_max_duration_s
        self.assertTrue(env._recovery_has_failed())


if __name__ == "__main__":
    unittest.main()
