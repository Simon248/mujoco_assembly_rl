from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import load_config, save_resolved_config
from src.evaluate import load_evaluation_model
from src.train import create_model


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(18,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(18, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(18, dtype=np.float32), 0.0, False, True, {}


class RlAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.env = DummyVecEnv([partial(TinyEnv)])
        self.base_training = {
            "algorithm": "sac", "buffer_size": 100,
            "learning_rate": 1e-4, "gamma": 0.97, "ent_coef": "auto",
            "target_entropy": "auto", "td3": {
                "action_noise_std": 0.1, "policy_delay": 2,
                "target_policy_noise": 0.2, "target_noise_clip": 0.5,
            },
        }

    def tearDown(self):
        self.env.close()

    def _create(self, algorithm):
        training = deepcopy(self.base_training); training["algorithm"] = algorithm
        return create_model(
            self.env, training, base_seed=7,
            tensorboard_log=Path("tensorboard"), device="cpu",
        )

    def test_factory_builds_sac_and_td3_with_same_network(self):
        sac = self._create("sac")
        td3 = self._create("td3")
        self.assertIsInstance(sac, SAC)
        self.assertIsInstance(td3, TD3)
        self.assertEqual(sac.policy.net_arch, [256, 256])
        self.assertEqual(td3.policy.net_arch, [256, 256])
        self.assertEqual(sac.gamma, 0.97)
        self.assertEqual(td3.gamma, 0.97)
        self.assertIsNone(sac.action_noise)
        self.assertEqual(td3.action_noise._sigma.shape, (6,))
        np.testing.assert_allclose(td3.action_noise._sigma, 0.1)

    def test_old_config_defaults_to_sac_and_resolved_yaml_is_explicit(self):
        config = load_config("configs/test0.yaml")
        config["training"].pop("algorithm")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "old.yaml"
            save_resolved_config(config, source)
            resolved = load_config(source)
            archived = Path(directory) / "resolved.yaml"
            save_resolved_config(resolved, archived)
            self.assertEqual(resolved["training"]["algorithm"], "sac")
            self.assertIn("algorithm: sac", archived.read_text(encoding="utf-8"))

    def test_algorithm_is_case_insensitive_and_rejects_unsupported_value(self):
        config = load_config("configs/test0.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            config["training"]["algorithm"] = "TD3"
            save_resolved_config(config, path)
            self.assertEqual(load_config(path)["training"]["algorithm"], "td3")
            config["training"]["algorithm"] = "ppo"
            save_resolved_config(config, path)
            with self.assertRaisesRegex(
                ValueError, "Unsupported RL algorithm: ppo.*sac, td3"
            ):
                load_config(path)

    @patch("src.evaluate.TD3.load")
    @patch("src.evaluate.SAC.load")
    def test_evaluation_loader_selects_algorithm_and_old_run_sac_default(
        self, sac_load, td3_load,
    ):
        env = Mock()
        load_evaluation_model(Path("model.zip"), env, "sac")
        sac_load.assert_called_once_with(Path("model.zip"), env=env)
        td3_load.assert_not_called()
        load_evaluation_model(Path("model.zip"), env, "td3")
        td3_load.assert_called_once_with(Path("model.zip"), env=env)

    def test_v15_only_changes_algorithm_and_training_fields_from_v14(self):
        v14 = load_config("configs/test1V14.yaml")
        v15 = load_config("configs/test1V15.yaml")
        self.assertEqual(v15["training"]["algorithm"], "td3")
        self.assertEqual(v15["training"]["buffer_size"], 250_000)
        v14["training"].update({
            "algorithm": "td3",
            "total_timesteps": v15["training"]["total_timesteps"],
            "buffer_size": v15["training"]["buffer_size"],
            "learning_rate": v15["training"]["learning_rate"],
            "td3": v15["training"]["td3"],
        })
        self.assertEqual(v15, v14)


if __name__ == "__main__":
    unittest.main()
