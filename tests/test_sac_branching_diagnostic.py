from __future__ import annotations

import unittest

import numpy as np

from src.assembly_env import AssemblyEnv
from src.config import load_config
from src.diagnose_sac_branching import (
    assert_matching_states, discounted_return, physical_state, replay_to_step,
)


class SacBranchingDiagnosticTest(unittest.TestCase):
    def test_discounted_return(self):
        self.assertAlmostEqual(discounted_return([1.0, 2.0, 3.0], 0.5), 2.75)

    def test_replay_is_deterministic_and_branches_are_independent(self):
        config = load_config("configs/test1V14.yaml")
        first = AssemblyEnv("configs/test1V14.yaml")
        second = AssemblyEnv("configs/test1V14.yaml")
        try:
            obs_a = replay_to_step(first, config, 2, 100)
            obs_b = replay_to_step(second, config, 2, 100)
            state_a = physical_state(first, obs_a)
            state_b = physical_state(second, obs_b)
            assert_matching_states([state_a, state_b])
            snapshot_b = {name: value.copy() for name, value in state_b.items()}
            first.step(np.zeros(6, dtype=np.float32))
            for name, expected in snapshot_b.items():
                np.testing.assert_array_equal(physical_state(second, obs_b)[name], expected)
        finally:
            first.close(); second.close()

    def test_replay_preserves_remaining_episode_budget(self):
        config = load_config("configs/test1V14.yaml")
        env = AssemblyEnv("configs/test1V14.yaml")
        branch_step = 2
        try:
            obs = replay_to_step(env, config, branch_step, 100)
            remaining_steps = 0
            done = False
            while not done:
                obs, _, terminated, truncated, _ = env.step(np.zeros(6))
                remaining_steps += 1; done = terminated or truncated
        finally:
            env.close()
        self.assertLessEqual(
            remaining_steps,
            config["simulation"]["max_episode_steps"] - branch_step,
        )


if __name__ == "__main__":
    unittest.main()
