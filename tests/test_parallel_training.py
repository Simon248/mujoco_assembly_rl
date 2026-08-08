from __future__ import annotations

import csv
from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from src.train import (
    MONITOR_FIELDS, build_vec_env, create_sac_model, scaled_callback_freq,
)
from src.config import load_config


AXIS_FIELDS = (
    "position_error_x", "position_error_y", "position_error_z",
    "rotation_error_x", "rotation_error_y", "rotation_error_z",
    "action_x", "action_y", "action_z", "action_rx", "action_ry", "action_rz",
)


class OneStepEpisodeEnv(gym.Env):
    """Petit env picklable pour vérifier uniquement l'agrégation VecMonitor."""
    def __init__(self, rank: int):
        self.rank = rank
        self.observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        info = {key: 0.0 for key in MONITOR_FIELDS}
        info.update({
            "geometric_success": True, "success": True, "safe_success": True,
            "unsafe": False, "unsafe_force": False, "unsafe_torque": False,
            "unsafe_workspace": False, "termination_reason": "success",
            "position_error": float(self.rank),
        })
        info.update({
            key: float(self.rank * 100 + index)
            for index, key in enumerate(AXIS_FIELDS)
        })
        return np.zeros(1, dtype=np.float32), 1.0, True, False, info


class ParallelTrainingTest(unittest.TestCase):
    @patch("src.train.SAC")
    def test_sac_receives_configured_entropy_parameters(self, sac_constructor):
        cases = (
            ({}, "auto", "auto"),
            (load_config("configs/test1V5.yaml")["training"], "auto", "auto"),
            (load_config("configs/test1V6.yaml")["training"], "auto", -3.0),
        )
        for training, expected_coef, expected_target in cases:
            with self.subTest(target_entropy=expected_target):
                create_sac_model(
                    object(), training, base_seed=7,
                    tensorboard_log=Path("tensorboard"), device="cpu",
                )
                kwargs = sac_constructor.call_args.kwargs
                self.assertEqual(kwargs["ent_coef"], expected_coef)
                self.assertEqual(kwargs["target_entropy"], expected_target)
                sac_constructor.reset_mock()

    def test_callback_frequency_is_expressed_in_transitions(self):
        self.assertEqual(scaled_callback_freq(50_000, 1), 50_000)
        self.assertEqual(scaled_callback_freq(50_000, 4), 12_500)
        self.assertEqual(scaled_callback_freq(2, 8), 1)

    def test_one_env_uses_dummy_vec_env(self):
        with tempfile.TemporaryDirectory() as directory:
            env = build_vec_env(
                Path("configs/test1.yaml"), 1, 41,
                Path(directory) / "monitor.csv",
            )
            try:
                self.assertIsInstance(env.venv, DummyVecEnv)
                env.reset()
                self.assertEqual(env.get_attr("worker_seed"), [41])
            finally:
                env.close()

    def test_two_subprocess_envs_are_seeded_independently_and_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = build_vec_env(
                Path("configs/test1.yaml"), 2, 73, root / "monitor.csv"
            )
            try:
                self.assertIsInstance(env.venv, SubprocVecEnv)
                obs = env.reset()
                self.assertEqual(obs.shape, (2, 12))
                self.assertEqual(env.get_attr("worker_seed"), [73, 74])
                self.assertEqual(len(set(env.get_attr("worker_pid"))), 2)

                model = SAC(
                    "MlpPolicy", env, seed=73, device="cpu", verbose=0,
                    learning_starts=100, buffer_size=100, batch_size=2,
                    train_freq=(1, "step"), gradient_steps=-1,
                    policy_kwargs={"net_arch": [16, 16]},
                )
                model.learn(total_timesteps=2)
                self.assertEqual(model.num_timesteps, 2)
                self.assertEqual(model.gradient_steps, -1)
            finally:
                env.close()

            monitor_files = list(root.glob("*monitor.csv"))
            self.assertEqual(len(monitor_files), 1)
            with monitor_files[0].open(encoding="utf-8") as stream:
                next(stream)  # en-tête JSON du Monitor SB3
                columns = next(csv.reader(stream))
            for field in MONITOR_FIELDS:
                self.assertIn(field, columns)

    def test_four_subprocess_envs_collect_one_vector_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = build_vec_env(
                Path("configs/test1.yaml"), 4, 100,
                root / "monitor.csv",
            )
            try:
                obs = env.reset()
                self.assertEqual(obs.shape, (4, 12))
                self.assertEqual(env.get_attr("worker_seed"), [100, 101, 102, 103])
                self.assertEqual(len(set(env.get_attr("worker_pid"))), 4)
                model = SAC(
                    "MlpPolicy", env, seed=100, device="cpu", verbose=0,
                    learning_starts=100, buffer_size=100, batch_size=4,
                    train_freq=(1, "step"), gradient_steps=-1,
                    policy_kwargs={"net_arch": [16, 16]},
                )
                callback = CheckpointCallback(
                    save_freq=scaled_callback_freq(4, 4),
                    save_path=str(root / "checkpoints"),
                    name_prefix="sac",
                )
                model.learn(total_timesteps=4, callback=callback)
                self.assertEqual(model.num_timesteps, 4)
                self.assertTrue((root / "checkpoints" / "sac_4_steps.zip").is_file())
            finally:
                env.close()

    def test_vec_monitor_writes_all_workers_to_one_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "monitor.csv"
            workers = SubprocVecEnv(
                [partial(OneStepEpisodeEnv, rank) for rank in range(2)],
                start_method="spawn",
            )
            env = VecMonitor(workers, filename=str(path), info_keywords=MONITOR_FIELDS)
            try:
                env.reset()
                env.step(np.zeros((2, 1), dtype=np.float32))
            finally:
                env.close()

            monitor_files = list(Path(directory).glob("*monitor.csv"))
            self.assertEqual(len(monitor_files), 1)
            with monitor_files[0].open(encoding="utf-8") as stream:
                next(stream)
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual({float(row["position_error"]) for row in rows}, {0.0, 1.0})
            rows_by_rank = {int(float(row["position_error"])): row for row in rows}
            self.assertEqual(float(rows_by_rank[0]["action_rz"]), 11.0)
            self.assertEqual(float(rows_by_rank[1]["action_rz"]), 111.0)


if __name__ == "__main__":
    unittest.main()
