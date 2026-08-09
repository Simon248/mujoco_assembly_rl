from __future__ import annotations

from functools import partial
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from src.diagnose_sac_q import compare_actions, critic_values
from src.evaluate_scripted import proportional_action


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(18,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(18, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(18, dtype=np.float32), 0.0, False, True, {}


class SacQDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.env = DummyVecEnv([partial(TinyEnv)])
        self.model = SAC("MlpPolicy", self.env, seed=1, device="cpu",
                         policy_kwargs={"net_arch": [8]})
        self.obs = np.zeros(18, dtype=np.float32)
        self.p_action = proportional_action(
            np.array([0, 0, 0.04, 0, 0, 0]),
            max_translation_step=0.001, max_rotation_step_deg=1.0,
        )

    def tearDown(self):
        self.env.close()

    def test_diagnostic_does_not_modify_actor_or_critic_parameters(self):
        before = {name: value.detach().clone()
                  for name, value in self.model.policy.state_dict().items()}
        compare_actions(self.model, self.obs, self.p_action)
        for name, value in self.model.policy.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_q_dimensions_and_batch_consistency(self):
        single = critic_values(self.model, self.obs, self.p_action)
        batch = critic_values(
            self.model, np.stack([self.obs, self.obs]),
            np.stack([self.p_action, self.p_action]),
        )
        self.assertEqual(single.shape, (1, 2))
        self.assertEqual(batch.shape, (2, 2))
        np.testing.assert_allclose(batch, np.repeat(single, 2, axis=0))

    def test_p_and_sac_actions_match_environment_shape_and_bounds(self):
        result = compare_actions(self.model, self.obs, self.p_action)
        p = np.array([result[f"p_action_{i}"] for i in range(6)])
        sac = np.array([result[f"sac_action_{i}"] for i in range(6)])
        self.assertEqual(p.shape, self.env.action_space.shape)
        self.assertEqual(sac.shape, self.env.action_space.shape)
        self.assertTrue(np.all(np.abs(p) <= 1))
        self.assertTrue(np.all(np.abs(sac) <= 1))


if __name__ == "__main__":
    unittest.main()
