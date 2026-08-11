from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from src.config import load_config, save_resolved_config
from src.train import learn_model, resolve_total_timesteps


class ConfigTest(unittest.TestCase):
    def test_test1_is_fully_resolved_and_exported_without_extends(self):
        config = load_config("configs/test1.yaml")
        self.assertNotIn("extends", config)
        self.assertIn("unsafe_penalty", config["reward"])
        self.assertIn("max_velocity", config["admittance"])
        self.assertEqual(config["training"]["n_envs"], 16)
        self.assertEqual(config["training"]["base_seed"], 7)
        self.assertEqual(config["training"]["checkpoint_freq"], 50_000)
        self.assertEqual(config["training"]["total_timesteps"], 500_000)
        self.assertEqual(config["training"]["buffer_size"], 50_000)
        self.assertEqual(config["training"]["learning_rate"], 3e-4)
        self.assertTrue(config["observation"]["include_admittance_position"])
        self.assertEqual(config["evaluation"], {
            "enabled": True, "eval_freq": 25_000, "n_eval_episodes": 1,
            "deterministic": True, "seed": 10_007,
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "config.yaml"
            save_resolved_config(config, output)
            self.assertNotIn("extends:", output.read_text(encoding="utf-8"))
            self.assertEqual(load_config(output), config)

    def test_action_steps_must_be_strictly_positive(self):
        config = load_config("configs/test0.yaml")
        for field in ("max_translation_step", "max_rotation_step_deg"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                invalid = deepcopy(config)
                invalid["action"][field] = 0.0
                path = Path(directory) / "config.yaml"
                save_resolved_config(invalid, path)
                with self.assertRaisesRegex(ValueError, rf"action\.{field}"):
                    load_config(path)

    def test_test1_v4_only_overrides_progress_weight(self):
        baseline = load_config("configs/test1.yaml")
        v4 = load_config("configs/test1V4.yaml")

        self.assertEqual(baseline["reward"]["progress_weight"], 10.0)
        self.assertEqual(v4["reward"]["progress_weight"], 2.5)
        baseline["reward"]["progress_weight"] = 2.5
        self.assertEqual(v4, baseline)

    def test_missing_action_frame_defaults_to_historical_grasp_mode(self):
        config = load_config("configs/test1V4.yaml")
        config["action"].pop("action_frame")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archived_config.yaml"
            save_resolved_config(config, path)
            self.assertEqual(load_config(path)["action"]["action_frame"], "grasp")

    def test_action_frame_rejects_unknown_values(self):
        config = load_config("configs/test1V4.yaml")
        config["action"]["action_frame"] = "world"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save_resolved_config(config, path)
            with self.assertRaisesRegex(ValueError, "action.action_frame"):
                load_config(path)

    def test_missing_control_mode_defaults_to_historical_behavior(self):
        config = load_config("configs/test1V5.yaml")
        config["action"].pop("control_mode")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archived_config.yaml"
            save_resolved_config(config, path)
            self.assertEqual(
                load_config(path)["action"]["control_mode"],
                "accumulated_reference",
            )

    def test_control_mode_rejects_unknown_values(self):
        config = load_config("configs/test1V5.yaml")
        config["action"]["control_mode"] = "integrate_everything"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save_resolved_config(config, path)
            with self.assertRaisesRegex(ValueError, "action.control_mode"):
                load_config(path)

    def test_test1_v10_only_switches_control_mode_from_test1_v9(self):
        # V9 n'est pas un artefact suivi; sa configuration était celle de V5.
        reference = load_config("configs/test1V5.yaml")
        reactive = load_config("configs/test1V10.yaml")
        self.assertEqual(reference["action"]["control_mode"], "accumulated_reference")
        self.assertEqual(reactive["action"]["control_mode"], "reactive_actual_pose")
        reference["action"]["control_mode"] = "reactive_actual_pose"
        self.assertEqual(reactive, reference)

    def test_test1_v5_only_switches_the_action_frame(self):
        v4 = load_config("configs/test1V4.yaml")
        v5 = load_config("configs/test1V5.yaml")

        self.assertEqual(v4["action"]["action_frame"], "grasp")
        self.assertEqual(v5["action"]["action_frame"], "task")
        v4["action"]["action_frame"] = "task"
        self.assertEqual(v5, v4)

    def test_entropy_defaults_are_explicit_and_keep_yaml_types(self):
        config = load_config("configs/test1V5.yaml")
        self.assertEqual(config["training"]["ent_coef"], "auto")
        self.assertIsInstance(config["training"]["ent_coef"], str)
        self.assertEqual(config["training"]["target_entropy"], "auto")
        self.assertIsInstance(config["training"]["target_entropy"], str)

    def test_old_config_without_entropy_fields_uses_historical_defaults(self):
        config = load_config("configs/test1V5.yaml")
        config["training"].pop("ent_coef")
        config["training"].pop("target_entropy")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archived_config.yaml"
            save_resolved_config(config, path)
            resolved = load_config(path)
        self.assertEqual(resolved["training"]["ent_coef"], "auto")
        self.assertEqual(resolved["training"]["target_entropy"], "auto")

    def test_old_config_gets_historical_sac_optimizer_defaults(self):
        config = load_config("configs/test1V10.yaml")
        config["training"].pop("buffer_size")
        config["training"].pop("learning_rate")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archived_config.yaml"
            save_resolved_config(config, path)
            resolved = load_config(path)
        self.assertEqual(resolved["training"]["buffer_size"], 50_000)
        self.assertEqual(resolved["training"]["learning_rate"], 3e-4)

    def test_old_config_gets_zero_torque_penalty_and_v16_enables_it(self):
        config = load_config("configs/test1V15.yaml")
        config["reward"].pop("torque_weight")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archived_config.yaml"
            save_resolved_config(config, path)
            self.assertEqual(load_config(path)["reward"]["torque_weight"], 0.0)
        v14 = load_config("configs/test1V14.yaml")
        v16 = load_config("configs/test1V16.yaml")
        self.assertEqual(v16["reward"]["torque_weight"], 0.08)
        v14["reward"]["torque_weight"] = 0.08
        self.assertEqual(v16["reward"], v14["reward"])

    def test_v19_resolves_to_only_the_new_reward_terms(self):
        v19 = load_config("configs/test1V19.yaml")
        self.assertEqual(v19["reward"], {
            "rotation_length_scale": 0.05,
            "potential_scale": 10.0,
            "potential_distance_scale": 0.010,
            "pose_weight": 50.0,
            "step_penalty": 0.05,
            "force_weight": 0.01,
            "torque_weight": 0.08,
            "action_weight": 0.01,
            "success_bonus": 300.0,
            "unsafe_penalty": 300.0,
            "timeout_penalty": 150.0,
        })
        self.assertEqual(v19["training"]["gamma"], 0.99)

    def test_v20_replaces_v19_reward_without_potential_parameters(self):
        v20 = load_config("configs/test1V20.yaml")
        self.assertEqual(v20["reward"], {
            "rotation_length_scale": 0.05,
            "pose_weight": 50.0,
            "step_penalty": 0.05,
            "force_weight": 0.01,
            "torque_weight": 0.08,
            "action_weight": 0.01,
            "success_bonus": 300.0,
            "unsafe_penalty": 300.0,
            "timeout_penalty": 150.0,
        })
        self.assertEqual(v20["training"]["gamma"], 0.99)

    def test_v21_adds_only_the_explicit_reverse_curriculum_block(self):
        v20 = load_config("configs/test1V20.yaml")
        v21 = load_config("configs/test1V21.yaml")
        self.assertTrue(v21["curriculum"]["enabled"])
        self.assertEqual(v21["curriculum"], {
            "enabled": True,
            "curriculum_reset_probability": 0.80,
            "success_rate_low": 0.10,
            "success_rate_high": 0.90,
            "update_interval_timesteps": 50_000,
            "candidates_per_update": 32,
            "evaluation_rollouts_per_candidate": 5,
            "max_pool_size": 2_000,
            "start_sampling": {
                "frontier_fraction": 0.625,
                "historical_fraction": 0.375,
                "historical_bins": 4,
                "strategy": "legacy",
                "adaptive_historical": False,
                "historical_fraction_per_state": 0.01,
                "historical_fraction_max": 0.375,
            },
            "revalidation": {
                "mastered_samples_per_update": 8,
                "too_hard_samples_per_update": 12,
                "every_n_curriculum_updates": 1,
            },
            "expansion": {
                "max_hops_per_seed": 4,
                "max_attempts_per_hop": 8,
                "max_candidates_per_update": 24,
                "initial_scale": 1.0,
                "scale_up_factor": 1.25,
                "scale_down_factor": 0.7,
                "min_scale": 0.5,
                "max_scale": 3.0,
            },
            "reverse_random_walk": {
                "walks_per_seed": 8, "max_steps": 20,
                "action_scale": 0.5,
                "proposal_mode": "independent",
                "persistent_proposal": {
                    "attempt_direction_noise_std": 0.20,
                    "hop_direction_noise_std": 0.15,
                    "step_noise_std": 0.10,
                },
                "proposal": {
                    "guided_fraction": 0.0,
                    "guided_noise_std": 0.20,
                    "memory_size_per_parent": 16,
                },
            },
            "deduplication": {
                "position_tolerance": 0.0005,
                "rotation_tolerance_deg": 0.5,
            },
        })
        v20["curriculum"] = deepcopy(v21["curriculum"])
        self.assertEqual(v20, v21)

    def test_v21_curi_max_only_keeps_its_explicit_reset_probability(self):
        baseline = load_config("configs/test1V21.yaml")
        curi_max = load_config("configs/test1V21-curi_max.yaml")

        self.assertEqual(curi_max["curriculum"]["curriculum_reset_probability"], .95)
        self.assertEqual(curi_max["curriculum"]["expansion"], {
            "max_hops_per_seed": 4,
            "max_attempts_per_hop": 8,
            "max_candidates_per_update": 24,
            "initial_scale": 1.0,
            "scale_up_factor": 1.25,
            "scale_down_factor": 0.7,
            "min_scale": 0.5,
            "max_scale": 3.0,
        })
        self.assertNotIn(
            "min_pose_distance_increase",
            curi_max["curriculum"]["reverse_random_walk"],
        )
        baseline["curriculum"]["curriculum_reset_probability"] = .95
        self.assertEqual(curi_max, baseline)

    def test_old_config_without_curriculum_defaults_to_disabled(self):
        config = load_config("configs/test1V20.yaml")
        config.pop("curriculum")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archived_config.yaml"
            save_resolved_config(config, path)
            resolved = load_config(path)
        self.assertEqual(resolved["curriculum"], {"enabled": False})

    def test_archived_v21_gets_backward_compatible_sampling_defaults(self):
        config = load_config("configs/test1V21.yaml")
        config["curriculum"].pop("start_sampling")
        config["curriculum"].pop("revalidation")
        config["curriculum"].pop("expansion")
        config["curriculum"]["evaluation_rollouts_per_candidate"] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archived_v21.yaml"
            save_resolved_config(config, path)
            resolved = load_config(path)
        self.assertEqual(resolved["curriculum"]["start_sampling"], {
            "frontier_fraction": .625,
            "historical_fraction": .375,
            "historical_bins": 4,
            "strategy": "legacy",
            "adaptive_historical": False,
            "historical_fraction_per_state": .01,
            "historical_fraction_max": .375,
        })
        self.assertEqual(resolved["curriculum"]["expansion"], {
            "max_hops_per_seed": 4,
            "max_attempts_per_hop": 8,
            "max_candidates_per_update": 24,
            "initial_scale": 1.0,
            "scale_up_factor": 1.25,
            "scale_down_factor": 0.7,
            "min_scale": 0.5,
            "max_scale": 3.0,
        })
        self.assertEqual(resolved["curriculum"]["revalidation"], {
            "mastered_samples_per_update": 8,
            "too_hard_samples_per_update": 12,
            "every_n_curriculum_updates": 1,
        })
        self.assertEqual(
            resolved["curriculum"]["evaluation_rollouts_per_candidate"], 3,
        )

    def test_historical_bin_count_defaults_to_four_in_partial_sampling_block(self):
        config = load_config("configs/test1V21.yaml")
        config["curriculum"]["start_sampling"].pop("historical_bins")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial_sampling.yaml"
            save_resolved_config(config, path)
            resolved = load_config(path)
        self.assertEqual(
            resolved["curriculum"]["start_sampling"]["historical_bins"], 4,
        )

    def test_adaptive_three_way_caps_preserve_true_start_minimum(self):
        config = load_config("configs/test1V29.yaml")
        config["curriculum"]["start_sampling"]["frontier"][
            "fraction_max"
        ] = .60
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_sampling.yaml"
            save_resolved_config(config, path)
            with self.assertRaisesRegex(ValueError, "fraction_max"):
                load_config(path)

    def test_persistent_proposal_mode_and_noise_are_explicitly_validated(self):
        base = load_config("configs/test1V34.yaml")
        self.assertEqual(
            base["curriculum"]["reverse_random_walk"]["proposal_mode"],
            "persistent",
        )
        for value, message in (("unknown", "proposal_mode"),):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                invalid = deepcopy(base)
                invalid["curriculum"]["reverse_random_walk"][
                    "proposal_mode"
                ] = value
                path = Path(directory) / "config.yaml"
                save_resolved_config(invalid, path)
                with self.assertRaisesRegex(ValueError, message):
                    load_config(path)

        persistent = base["curriculum"]["reverse_random_walk"][
            "persistent_proposal"
        ]
        self.assertEqual(persistent, {
            "attempt_direction_noise_std": 0.20,
            "hop_direction_noise_std": 0.15,
            "step_noise_std": 0.10,
        })
        for key in persistent:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                invalid = deepcopy(base)
                invalid["curriculum"]["reverse_random_walk"][
                    "persistent_proposal"
                ][key] = -0.1
                path = Path(directory) / "config.yaml"
                save_resolved_config(invalid, path)
                with self.assertRaisesRegex(ValueError, key):
                    load_config(path)

    def test_v33_v34_are_an_independent_persistent_ab_pair(self):
        v33 = load_config("configs/test1V33.yaml")
        v34 = load_config("configs/test1V34.yaml")
        walk33 = deepcopy(v33["curriculum"]["reverse_random_walk"])
        walk34 = deepcopy(v34["curriculum"]["reverse_random_walk"])
        self.assertEqual(walk33.pop("proposal_mode"), "independent")
        self.assertEqual(walk34.pop("proposal_mode"), "persistent")
        self.assertEqual(walk33, walk34)
        v33["curriculum"]["reverse_random_walk"] = walk33
        v34["curriculum"]["reverse_random_walk"] = walk34
        self.assertEqual(v33, v34)

    def test_v35_enables_branch_persistent_proposal_explicitly(self):
        walk = load_config("configs/test1V35.yaml")["curriculum"][
            "reverse_random_walk"
        ]
        self.assertEqual(walk["proposal_mode"], "persistent")
        self.assertEqual(walk["persistent_proposal"], {
            "attempt_direction_noise_std": 0.20,
            "hop_direction_noise_std": 0.15,
            "step_noise_std": 0.10,
        })

    def test_enabled_curriculum_is_strictly_validated(self):
        base = load_config("configs/test1V21.yaml")
        mutations = (
            (lambda cfg: cfg["curriculum"].__setitem__(
                "curriculum_reset_probability", 1.0),
             "curriculum_reset_probability"),
            (lambda cfg: cfg["curriculum"].__setitem__(
                "success_rate_low", 0.95), "success_rate_low"),
            (lambda cfg: cfg["curriculum"]["reverse_random_walk"].__setitem__(
                "action_scale", 1.5), "action_scale"),
            (lambda cfg: cfg["curriculum"]["deduplication"].__setitem__(
                "position_tolerance", 0.0), "position_tolerance"),
            (lambda cfg: cfg["curriculum"]["start_sampling"].__setitem__(
                "historical_fraction", 0.5), "frontier_fraction"),
            (lambda cfg: cfg["curriculum"]["start_sampling"].__setitem__(
                "historical_bins", 0), "historical_bins"),
            (lambda cfg: cfg["curriculum"]["revalidation"].__setitem__(
                "mastered_samples_per_update", -1),
             "mastered_samples_per_update"),
            (lambda cfg: cfg["curriculum"]["revalidation"].__setitem__(
                "too_hard_samples_per_update", -1),
             "too_hard_samples_per_update"),
            (lambda cfg: cfg["curriculum"]["revalidation"].__setitem__(
                "every_n_curriculum_updates", 0),
             "every_n_curriculum_updates"),
            (lambda cfg: cfg["curriculum"]["expansion"].__setitem__(
                "max_hops_per_seed", 0), "max_hops_per_seed"),
            (lambda cfg: cfg["curriculum"]["expansion"].__setitem__(
                "max_candidates_per_update", True),
             "max_candidates_per_update"),
            (lambda cfg: cfg["curriculum"]["expansion"].__setitem__(
                "initial_scale", 0.1), "initial_scale"),
            (lambda cfg: cfg["curriculum"]["expansion"].__setitem__(
                "scale_up_factor", 0.9), "scale_up_factor"),
            (lambda cfg: cfg["curriculum"]["expansion"].__setitem__(
                "scale_down_factor", 1.1), "scale_down_factor"),
            (lambda cfg: cfg["curriculum"]["expansion"].__setitem__(
                "min_scale", 0.0), "min_scale"),
            (lambda cfg: cfg["curriculum"]["expansion"].__setitem__(
                "max_scale", 0.4), "max_scale"),
        )
        for mutate, expected_message in mutations:
            with self.subTest(field=expected_message), tempfile.TemporaryDirectory() as directory:
                invalid = deepcopy(base); mutate(invalid)
                path = Path(directory) / "config.yaml"
                save_resolved_config(invalid, path)
                with self.assertRaisesRegex(ValueError, expected_message):
                    load_config(path)

    def test_deprecated_geometry_fields_are_accepted_without_validation(self):
        config = load_config("configs/test1V21.yaml")
        config["curriculum"]["expansion"] = {
            "mastered_edge_fraction": "deprecated-value",
        }
        config["curriculum"]["reverse_random_walk"][
            "min_pose_distance_increase"
        ] = "deprecated-value"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_config.yaml"
            save_resolved_config(config, path)
            resolved = load_config(path)

        self.assertEqual(
            resolved["curriculum"]["expansion"]["mastered_edge_fraction"],
            "deprecated-value",
        )
        self.assertEqual(
            resolved["curriculum"]["expansion"]["max_hops_per_seed"], 4,
        )
        self.assertEqual(
            resolved["curriculum"]["expansion"]["max_attempts_per_hop"], 8,
        )
        self.assertEqual(
            resolved["curriculum"]["reverse_random_walk"][
                "min_pose_distance_increase"
            ],
            "deprecated-value",
        )

    def test_total_timesteps_resolution_and_archived_effective_value(self):
        training = {"total_timesteps": 1_000_000}
        self.assertEqual(resolve_total_timesteps(training, None), 1_000_000)
        self.assertEqual(resolve_total_timesteps({}, None), 500_000)
        self.assertEqual(resolve_total_timesteps(training, 100_000), 100_000)

        config = load_config("configs/test0.yaml")
        config["training"]["total_timesteps"] = resolve_total_timesteps(
            config["training"], 100_000
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save_resolved_config(config, path)
            archived = load_config(path)
        self.assertEqual(archived["training"]["total_timesteps"], 100_000)

    def test_model_learn_receives_yaml_total_timesteps(self):
        model = Mock()
        callbacks = Mock()
        training = {"total_timesteps": 1_000_000}
        learn_model(model, resolve_total_timesteps(training, None), callbacks)
        model.learn.assert_called_once_with(
            total_timesteps=1_000_000,
            callback=callbacks,
            progress_bar=True,
        )

    def test_test1_v13_only_changes_budget_and_learning_rate_from_v12(self):
        v12 = load_config("configs/test1V12.yaml")
        v13 = load_config("configs/test1V13.yaml")
        self.assertEqual(v13["training"]["total_timesteps"], 6_000_000)
        self.assertEqual(v13["training"]["buffer_size"], 250_000)
        self.assertEqual(v13["training"]["learning_rate"], 1e-4)
        v12["training"]["total_timesteps"] = 6_000_000
        v12["training"]["learning_rate"] = 1e-4
        self.assertEqual(v13, v12)

    def test_test1_v11_only_increases_replay_buffer(self):
        v10 = load_config("configs/test1V10.yaml")
        v11 = load_config("configs/test1V11.yaml")
        self.assertEqual(v10["training"]["buffer_size"], 50_000)
        self.assertEqual(v11["training"]["buffer_size"], 500_000)
        self.assertEqual(v11["training"]["learning_rate"], 3e-4)
        v10["training"]["buffer_size"] = 500_000
        self.assertEqual(v11, v10)

    def test_sac_buffer_and_learning_rate_must_be_positive(self):
        config = load_config("configs/test1V10.yaml")
        for field, invalid_value in (("buffer_size", 0), ("learning_rate", 0.0)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                invalid = deepcopy(config)
                invalid["training"][field] = invalid_value
                path = Path(directory) / "config.yaml"
                save_resolved_config(invalid, path)
                with self.assertRaisesRegex(ValueError, rf"training\.{field}"):
                    load_config(path)

    def test_test1_v6_only_overrides_target_entropy(self):
        v5 = load_config("configs/test1V5.yaml")
        v6 = load_config("configs/test1V6.yaml")

        self.assertEqual(v6["training"]["ent_coef"], "auto")
        self.assertEqual(v6["training"]["target_entropy"], -3.0)
        self.assertIsInstance(v6["training"]["target_entropy"], float)
        v5["training"]["target_entropy"] = -3.0
        self.assertEqual(v6, v5)

    def test_numeric_entropy_values_remain_numeric_after_export(self):
        config = load_config("configs/test1V5.yaml")
        config["training"]["ent_coef"] = 0.1
        config["training"]["target_entropy"] = -3.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save_resolved_config(config, path)
            resolved = load_config(path)
        self.assertEqual(resolved["training"]["ent_coef"], 0.1)
        self.assertIsInstance(resolved["training"]["ent_coef"], float)
        self.assertEqual(resolved["training"]["target_entropy"], -3.0)
        self.assertIsInstance(resolved["training"]["target_entropy"], float)

    def test_old_config_gets_backward_compatible_observation_and_eval_defaults(self):
        config = load_config("configs/test1.yaml")
        config["observation"].pop("include_admittance_position")
        config.pop("evaluation")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archived_config.yaml"
            save_resolved_config(config, path)
            resolved = load_config(path)
        self.assertFalse(resolved["observation"]["include_admittance_position"])
        self.assertFalse(resolved["evaluation"]["enabled"])
        self.assertTrue(resolved["evaluation"]["deterministic"])


if __name__ == "__main__":
    unittest.main()
