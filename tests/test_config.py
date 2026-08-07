from __future__ import annotations

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
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "config.yaml"
            save_resolved_config(config, output)
            self.assertNotIn("extends:", output.read_text(encoding="utf-8"))
            self.assertEqual(load_config(output), config)


if __name__ == "__main__":
    unittest.main()
