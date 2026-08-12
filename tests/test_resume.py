from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
import unittest
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from src.resume import (
    apply_resume_configuration, next_future_curriculum_update,
    rebuild_empty_replay_buffer, validate_effective_resume_configuration,
)
from src.train import create_sac_model


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(18,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(18, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(18, dtype=np.float32), 0.0, False, True, {}


def config(**training_overrides):
    training = {
        "algorithm": "sac", "buffer_size": 100, "learning_rate": 3e-4,
        "gamma": .99, "tau": .005, "batch_size": 256,
        "train_freq": [1, "step"], "gradient_steps": -1,
        "learning_starts": 5_000, "target_update_interval": 1,
        "network": [256, 256], "ent_coef": "auto", "target_entropy": "auto",
    }
    training.update(training_overrides)
    return {
        "training": training,
        "resume": {"replay_buffer_policy": "auto", "apply_current_yaml": True,
                   "fail_on_structural_change": True, "log_parameter_diff": True},
        "reward": {"pose_weight": 50}, "success": {"position_tolerance": .0005},
        "simulation": {"control_dt": .02}, "admittance": {}, "action": {},
        "randomization": {}, "perception": {}, "observation": {}, "curriculum": {},
    }


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.env = DummyVecEnv([partial(TinyEnv)])
        self.old = config()
        self.model = create_sac_model(
            self.env, self.old["training"], base_seed=7,
            tensorboard_log=Path("tensorboard"), device="cpu",
        )

    def tearDown(self):
        self.env.close()

    def test_next_future_curriculum_update_is_strictly_future(self):
        self.assertEqual(next_future_curriculum_update(2_780_000, 20_000), 2_800_000)
        self.assertEqual(next_future_curriculum_update(2_781_000, 20_000), 2_800_000)
        self.assertEqual(next_future_curriculum_update(2_800_000, 20_000), 2_820_000)

    def test_runtime_overrides_preserve_weights_and_optimizer_state(self):
        current = config(
            learning_rate=1e-4, gamma=.995, tau=.01, batch_size=32,
            train_freq=[4, "step"], gradient_steps=3, learning_starts=100,
            target_update_interval=2,
        )
        actor_before = [parameter.detach().clone() for parameter in self.model.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in self.model.critic.parameters()]
        optimizer_ids = (
            id(self.model.actor.optimizer), id(self.model.critic.optimizer),
            id(self.model.ent_coef_optimizer),
        )
        result = apply_resume_configuration(
            self.model, self.env, current, previous_config=self.old,
        )
        validate_effective_resume_configuration(self.model, current)
        for before, after in zip(actor_before, self.model.actor.parameters()):
            np.testing.assert_array_equal(before.numpy(), after.detach().numpy())
        for before, after in zip(critic_before, self.model.critic.parameters()):
            np.testing.assert_array_equal(before.numpy(), after.detach().numpy())
        self.model._logger = Mock()
        observation = np.zeros((1, 18), dtype=np.float32)
        action = np.zeros((1, 6), dtype=np.float32)
        for _ in range(40):
            self.model.replay_buffer.add(
                observation, observation, action, np.zeros(1), np.zeros(1), [{}],
            )
        self.model.train(gradient_steps=3, batch_size=16)
        self.assertEqual(result.replay_action, "keep")
        self.assertEqual(self.model.gamma, .995)
        self.assertEqual(self.model.tau, .01)
        self.assertEqual(self.model.batch_size, 32)
        self.assertEqual((self.model.train_freq.frequency, self.model.train_freq.unit.value), (4, "step"))
        self.assertEqual(self.model.gradient_steps, 3)
        self.assertEqual(self.model.learning_starts, 100)
        self.assertEqual(self.model.target_update_interval, 2)
        self.assertEqual(optimizer_ids, (
            id(self.model.actor.optimizer), id(self.model.critic.optimizer),
            id(self.model.ent_coef_optimizer),
        ))
        for optimizer in (self.model.actor.optimizer, self.model.critic.optimizer,
                          self.model.ent_coef_optimizer):
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_reward_change_auto_discards_and_rebuilds_buffer(self):
        current = deepcopy(self.old)
        current["reward"]["pose_weight"] = 40
        result = apply_resume_configuration(
            self.model, self.env, current, previous_config=self.old,
        )
        self.assertEqual(result.replay_action, "discard")
        rebuild_empty_replay_buffer(self.model, 100)
        self.assertEqual(self.model.replay_buffer.size(), 0)

    def test_buffer_size_change_rebuilds_to_effective_capacity(self):
        current = config(buffer_size=1_000)
        result = apply_resume_configuration(
            self.model, self.env, current, previous_config=self.old,
        )
        self.assertEqual(result.replay_action, "discard")
        rebuild_empty_replay_buffer(self.model, 1_000)
        self.assertEqual(self.model.replay_buffer.buffer_size, 1_000)

    def test_structural_network_and_spaces_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "training.network"):
            apply_resume_configuration(
                self.model, self.env, config(network=[512, 512]),
                previous_config=self.old,
            )
        incompatible = DummyVecEnv([lambda: TinyObservationEnv()])
        try:
            with self.assertRaisesRegex(ValueError, "observation_space"):
                apply_resume_configuration(
                    self.model, incompatible, self.old, previous_config=self.old,
                )
        finally:
            incompatible.close()


class TinyObservationEnv(TinyEnv):
    observation_space = gym.spaces.Box(-1, 1, shape=(24,), dtype=np.float32)


if __name__ == "__main__":
    unittest.main()
