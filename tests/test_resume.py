from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from src.resume import (
    apply_resume_configuration, devices_are_compatible,
    next_future_curriculum_update,
    expected_replay_buffer_internal_size, prepare_resume_replay_buffer,
    rebuild_empty_replay_buffer, validate_effective_resume_configuration,
    validate_replay_buffer_capacity,
)
from src.train import create_sac_model, create_td3_model


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(18,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(18, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(18, dtype=np.float32), 0.0, False, True, {}


class TinyScalarEnv(TinyEnv):
    observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, True, {}


def config(**training_overrides):
    training = {
        "algorithm": "sac", "buffer_size": 100, "learning_rate": 3e-4,
        "gamma": .99, "tau": .005, "batch_size": 256,
        "train_freq": [1, "step"], "gradient_steps": -1,
        "learning_starts": 5_000, "target_update_interval": 1,
        "network": [256, 256], "ent_coef": "auto", "target_entropy": "auto",
        "n_envs": 1, "base_seed": 7, "optimize_memory_usage": False,
        "td3": {
            "action_noise_std": .1, "policy_delay": 2,
            "target_policy_noise": .2, "target_noise_clip": .5,
        },
    }
    training.update(training_overrides)
    return {
        "training": training,
        "resume": {"replay_buffer_policy": "auto", "apply_current_yaml": True,
                   "fail_on_structural_change": True, "log_parameter_diff": True},
        "case": "tenon_1",
        "target_pose_fixed_to_mobile": {
            "position": [0.0, 0.0, 0.0],
            "orientation_quat": [1.0, 0.0, 0.0, 0.0],
        },
        "initial_pose_fixed_to_mobile": {
            "position": [0.0, 0.0, 0.04],
            "orientation_quat": [1.0, 0.0, 0.0, 0.0],
        },
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

    def test_device_validation_normalizes_unspecified_cuda_index(self):
        self.assertTrue(devices_are_compatible("cpu", "cpu"))
        self.assertTrue(devices_are_compatible("cuda", "cuda:0"))
        self.assertTrue(devices_are_compatible("cuda:1", "cuda:1"))
        self.assertFalse(devices_are_compatible("cuda:1", "cuda:0"))
        self.assertFalse(devices_are_compatible("cpu", "cuda:0"))

    def test_runtime_overrides_preserve_weights_and_optimizer_state(self):
        current = config(
            learning_rate=1e-4, gamma=.995, tau=.01, batch_size=32,
            train_freq=[4, "step"], gradient_steps=3, learning_starts=100,
            target_update_interval=2, target_entropy=-3.0,
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
        self.assertEqual(self.model.target_entropy, -3.0)
        self.assertEqual(optimizer_ids, (
            id(self.model.actor.optimizer), id(self.model.critic.optimizer),
            id(self.model.ent_coef_optimizer),
        ))
        for optimizer in (self.model.actor.optimizer, self.model.critic.optimizer,
                          self.model.ent_coef_optimizer):
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_actual_sac_load_applies_and_validates_runtime_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sac.zip"
            self.model.save(checkpoint)
            loaded = SAC.load(checkpoint, device="cpu")
        current = config(learning_rate=1e-4, gamma=.996)
        result = apply_resume_configuration(
            loaded, self.env, current, previous_config=self.old,
        )
        validate_effective_resume_configuration(loaded, current)
        self.assertEqual(result.replay_action, "keep")
        self.assertAlmostEqual(loaded.gamma, .996)
        for optimizer in (
            loaded.actor.optimizer, loaded.critic.optimizer,
            loaded.ent_coef_optimizer,
        ):
            self.assertTrue(all(
                np.isclose(group["lr"], 1e-4)
                for group in optimizer.param_groups
            ))

    def test_unchanged_fixed_and_auto_initial_entropy_modes_are_supported(self):
        for ent_coef in (.2, "auto_0.5"):
            with self.subTest(ent_coef=ent_coef):
                entropy_config = config(ent_coef=ent_coef)
                model = create_sac_model(
                    self.env, entropy_config["training"], base_seed=7,
                    tensorboard_log=Path("tensorboard"), device="cpu",
                )
                apply_resume_configuration(
                    model, self.env, entropy_config,
                    previous_config=entropy_config,
                )
                validate_effective_resume_configuration(
                    model, entropy_config,
                )

    def test_reward_change_auto_discards_and_rebuilds_buffer(self):
        current = deepcopy(self.old)
        current["reward"]["pose_weight"] = 40
        result = apply_resume_configuration(
            self.model, self.env, current, previous_config=self.old,
        )
        self.assertEqual(result.replay_action, "discard")
        rebuild_empty_replay_buffer(self.model, 100)
        self.assertEqual(self.model.replay_buffer.size(), 0)

    def test_target_change_discards_replay_and_marks_curriculum_incompatible(self):
        current = deepcopy(self.old)
        current["target_pose_fixed_to_mobile"]["position"][2] = -.005
        with self.assertWarnsRegex(RuntimeWarning, "target_pose"):
            result = apply_resume_configuration(
                self.model, self.env, current, previous_config=self.old,
            )
        self.assertEqual(result.replay_action, "discard")
        self.assertIn("target_pose_fixed_to_mobile", result.sensitive_changes)
        self.assertIn(
            "target_pose_fixed_to_mobile",
            result.curriculum_incompatible_changes,
        )

    def test_keep_policy_rejects_semantically_incompatible_replay(self):
        current = deepcopy(self.old)
        current["reward"]["pose_weight"] = 40
        current["resume"]["replay_buffer_policy"] = "keep"
        with self.assertRaisesRegex(
            ValueError, "replay_buffer_policy=keep incompatible.*reward",
        ):
            apply_resume_configuration(
                self.model, self.env, current, previous_config=self.old,
            )

    def test_unknown_source_config_discards_in_auto_and_rejects_keep(self):
        with self.assertWarnsRegex(RuntimeWarning, "config unavailable"):
            result = apply_resume_configuration(
                self.model, self.env, self.old, previous_config=None,
            )
        self.assertEqual(result.replay_action, "discard")
        self.assertFalse(result.semantic_compatibility_known)

        current = deepcopy(self.old)
        current["resume"]["replay_buffer_policy"] = "keep"
        with self.assertRaisesRegex(
            ValueError, "replay_buffer_policy=keep.*config source introuvable",
        ):
            apply_resume_configuration(
                self.model, self.env, current, previous_config=None,
            )

    def test_seed_change_is_explicitly_rejected(self):
        current = deepcopy(self.old)
        current["training"]["base_seed"] = 8
        with self.assertRaisesRegex(ValueError, "base_seed cannot be changed"):
            apply_resume_configuration(
                self.model, self.env, current, previous_config=self.old,
            )

    def test_resume_safety_bypasses_are_rejected_only_on_resume(self):
        for key in ("apply_current_yaml", "fail_on_structural_change"):
            with self.subTest(key=key):
                current = deepcopy(self.old)
                current["resume"][key] = False
                with self.assertRaisesRegex(ValueError, f"{key}=false"):
                    apply_resume_configuration(
                        self.model, self.env, current,
                        previous_config=self.old,
                    )

    def test_buffer_size_change_rebuilds_to_effective_capacity(self):
        current = config(buffer_size=1_000)
        result = apply_resume_configuration(
            self.model, self.env, current, previous_config=self.old,
        )
        self.assertEqual(result.replay_action, "discard")
        rebuild_empty_replay_buffer(self.model, 1_000)
        self.assertEqual(self.model.replay_buffer.buffer_size, 1_000)

    def test_sb3_290_vectorized_buffer_capacity_uses_integer_division(self):
        observation_space = gym.spaces.Box(
            -1, 1, shape=(1,), dtype=np.float32,
        )
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        replay = ReplayBuffer(
            250_000, observation_space, action_space, device="cpu", n_envs=16,
        )
        model = SimpleNamespace(
            buffer_size=250_000, replay_buffer=replay, n_envs=16,
            optimize_memory_usage=False,
            observation_space=observation_space, action_space=action_space,
            device=replay.device, replay_buffer_class=ReplayBuffer,
        )
        self.assertEqual(replay.buffer_size, 15_625)
        self.assertEqual(
            expected_replay_buffer_internal_size(250_000, 16), 15_625,
        )
        validate_replay_buffer_capacity(model, 250_000)

        # SB3 floors instead of rounding up: the effective capacity can be
        # lower than the model-level request when it is not divisible.
        replay_non_divisible = ReplayBuffer(
            250_007, observation_space, action_space,
            device="cpu", n_envs=16,
        )
        model.buffer_size = 250_007
        model.replay_buffer = replay_non_divisible
        self.assertEqual(replay_non_divisible.buffer_size, 15_625)
        validate_replay_buffer_capacity(model, 250_007)

    def test_replay_capacity_validation_detects_a_false_override(self):
        self.model.buffer_size = 250
        with self.assertRaisesRegex(
            RuntimeError,
            "(?s)requested total:.*250.*model-level value:.*250.*"
            "expected internal:.*250.*replay internal:.*100",
        ):
            validate_replay_buffer_capacity(self.model, 250)

    def test_runtime_validation_detects_a_false_gamma_override(self):
        current = config(gamma=.996)
        apply_resume_configuration(
            self.model, self.env, current, previous_config=self.old,
        )
        self.model.gamma = .99
        with self.assertRaisesRegex(
            RuntimeError, "gamma; requested=0.996, effective=0.99",
        ):
            validate_effective_resume_configuration(
                self.model, current, validate_replay_buffer=False,
            )

    def test_n_envs_change_is_explicitly_rejected(self):
        current = config(n_envs=16)
        with self.assertRaisesRegex(
            ValueError, "training.n_envs.*checkpoint=1.*requested=16",
        ):
            apply_resume_configuration(
                self.model, self.env, current, previous_config=self.old,
            )

    def test_ent_coef_specification_change_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "ent_coef cannot be changed"):
            apply_resume_configuration(
                self.model, self.env, config(ent_coef="auto_0.1"),
                previous_config=self.old,
            )

    def test_td3_runtime_overrides_reach_runtime_objects(self):
        old = config(algorithm="td3")
        model = create_td3_model(
            self.env, old["training"], base_seed=7,
            tensorboard_log=Path("tensorboard"), device="cpu",
        )
        current = deepcopy(old)
        current["training"]["learning_rate"] = 1e-4
        current["training"]["gamma"] = .996
        current["training"]["td3"] = {
            "action_noise_std": .25,
            "policy_delay": 4,
            "target_policy_noise": .35,
            "target_noise_clip": .7,
        }
        apply_resume_configuration(
            model, self.env, current, previous_config=old,
        )
        validate_effective_resume_configuration(
            model, current, validate_replay_buffer=False,
        )
        self.assertEqual(model.policy_delay, 4)
        self.assertAlmostEqual(model.target_policy_noise, .35)
        self.assertAlmostEqual(model.target_noise_clip, .7)
        np.testing.assert_allclose(model.action_noise._sigma, .25)

    def test_same_capacity_replay_is_loaded_and_preserved(self):
        observation = np.zeros((1, 18), dtype=np.float32)
        action = np.zeros((1, 6), dtype=np.float32)
        for index in range(7):
            self.model.replay_buffer.add(
                observation, observation, action,
                np.array([float(index)]), np.zeros(1), [{}],
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.pkl"
            self.model.save_replay_buffer(path)
            rebuilt = create_sac_model(
                self.env, self.old["training"], base_seed=7,
                tensorboard_log=Path("tensorboard"), device="cpu",
            )
            report = prepare_resume_replay_buffer(
                rebuilt, path, "keep", requested_total=100,
                checkpoint_requested_total=100,
            )
        self.assertEqual(report.transitions_preserved, 7)
        self.assertEqual(report.transitions_discarded, 0)
        self.assertEqual(rebuilt.replay_buffer.size(), 7)

    def test_resize_up_and_down_explicitly_discard_loaded_transitions(self):
        scalar_env = DummyVecEnv([partial(TinyScalarEnv) for _ in range(16)])
        observation = np.zeros((16, 1), dtype=np.float32)
        action = np.zeros((16, 1), dtype=np.float32)
        try:
            for old_size, new_size in (
                (250_000, 1_000_000), (1_000_000, 250_000),
            ):
                with self.subTest(old_size=old_size, new_size=new_size):
                    old_training = deepcopy(self.old["training"])
                    old_training["buffer_size"] = old_size
                    old_training["n_envs"] = 16
                    old_training["network"] = [8]
                    source = create_sac_model(
                        scalar_env, old_training, base_seed=7,
                        tensorboard_log=Path("tensorboard"), device="cpu",
                    )
                    for _ in range(9):
                        source.replay_buffer.add(
                            observation, observation, action,
                            np.zeros(16), np.zeros(16), [{} for _ in range(16)],
                        )
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "replay.pkl"
                        source.save_replay_buffer(path)
                        report = prepare_resume_replay_buffer(
                            source, path, "discard", requested_total=new_size,
                            checkpoint_requested_total=old_size,
                        )
                    self.assertEqual(report.transitions_preserved, 0)
                    self.assertEqual(report.transitions_discarded, 9 * 16)
                    self.assertEqual(source.replay_buffer.size(), 0)
                    self.assertEqual(
                        source.replay_buffer.buffer_size,
                        expected_replay_buffer_internal_size(new_size, 16),
                    )
        finally:
            scalar_env.close()

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
