from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from src.place_path import load_place_path


def paths_root() -> Path:
    return Path(
        os.environ.get("PATHS_DIR", str(Path(__file__).parents[1] / "data/input/chemin"))
    )


class PlacePathTest(unittest.TestCase):
    def test_each_recorded_place_path_has_normalized_progress(self) -> None:
        root = paths_root()
        for part in ("part_1", "part_2", "part_3"):
            path = load_place_path(root, part)
            self.assertEqual(path.progress[0], 0.0)
            self.assertEqual(path.progress[-1], 1.0)
            self.assertTrue(np.all(np.diff(path.progress) > 0.0))

    def test_endpoint_interpolation_returns_recorded_endpoints(self) -> None:
        root = paths_root()
        path = load_place_path(root, "part_1")
        start_p, start_q = path.pose_at(0.0)
        end_p, end_q = path.pose_at(1.0)
        np.testing.assert_allclose(start_p, path.positions[0])
        np.testing.assert_allclose(end_p, path.positions[-1])
        np.testing.assert_allclose(start_q, path.quaternions[0])
        np.testing.assert_allclose(end_q, path.quaternions[-1])

    def test_pose_is_clamped_to_path_bounds(self) -> None:
        root = paths_root()
        path = load_place_path(root, "part_2")
        np.testing.assert_allclose(path.pose_at(-1.0)[0], path.pose_at(0.0)[0])
        np.testing.assert_allclose(path.pose_at(2.0)[0], path.pose_at(1.0)[0])


if __name__ == "__main__":
    unittest.main()
