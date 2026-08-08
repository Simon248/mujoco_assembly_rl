from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from src.config import load_config, save_resolved_config


class ConfigTest(unittest.TestCase):
    def test_test1_is_fully_resolved_and_exported_without_extends(self):
        config = load_config("configs/test1.yaml")
        self.assertNotIn("extends", config)
        self.assertIn("unsafe_penalty", config["reward"])
        self.assertIn("max_velocity", config["admittance"])
        self.assertEqual(config["training"]["n_envs"], 12)
        self.assertEqual(config["training"]["base_seed"], 7)
        self.assertEqual(config["training"]["checkpoint_freq"], 50_000)
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


if __name__ == "__main__":
    unittest.main()
