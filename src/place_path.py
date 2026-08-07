"""Loading and geometric interpolation of recorded ``place`` paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _pose(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = entry["pose"]["position"]
    orientation = entry["pose"]["orientation"]
    p = np.array([position["x"], position["y"], position["z"]], dtype=np.float64)
    q = np.array(
        [orientation["w"], orientation["x"], orientation["y"], orientation["z"]],
        dtype=np.float64,
    )
    return p, q / np.linalg.norm(q)


def slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path spherical interpolation for MuJoCo ``wxyz`` quaternions."""
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    if dot < 0.0:
        end, dot = -end, -dot
    if dot > 0.9995:
        result = start + fraction * (end - start)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    return (
        np.sin((1.0 - fraction) * angle) / sin_angle * start
        + np.sin(fraction * angle) / sin_angle * end
    )


@dataclass(frozen=True)
class PlacePath:
    """A path parameterized by normalized arc length instead of wall-clock time."""

    part_name: str
    source: Path
    positions: np.ndarray
    quaternions: np.ndarray
    progress: np.ndarray

    @property
    def final_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions[-1].copy(), self.quaternions[-1].copy()

    def pose_at(self, progress: float) -> tuple[np.ndarray, np.ndarray]:
        value = float(np.clip(progress, 0.0, 1.0))
        if value <= 0.0:
            return self.positions[0].copy(), self.quaternions[0].copy()
        if value >= 1.0:
            return self.positions[-1].copy(), self.quaternions[-1].copy()
        index = int(np.searchsorted(self.progress, value, side="right") - 1)
        index = int(np.clip(index, 0, len(self.progress) - 2))
        begin, end = self.progress[index], self.progress[index + 1]
        fraction = 0.0 if end <= begin else (value - begin) / (end - begin)
        return (
            (1.0 - fraction) * self.positions[index] + fraction * self.positions[index + 1],
            slerp(self.quaternions[index], self.quaternions[index + 1], fraction),
        )


def find_path(paths_dir: str | Path, part_name: str) -> Path:
    matches = sorted(Path(paths_dir).glob(f"chandelier_{part_name}_place*.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one place YAML for {part_name!r} in {paths_dir}; found {matches}"
        )
    return matches[0]


def load_place_path(paths_dir: str | Path, part_name: str) -> PlacePath:
    source = find_path(paths_dir, part_name)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    tracked = str(document.get("tracked_frame"))
    expected = f"chandelier_{part_name}"
    if tracked != expected:
        raise ValueError(f"{source} tracks {tracked!r}, expected {expected!r}")
    points = document.get("segments", {}).get("place")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError(f"{source} has no usable place segment")
    poses = [_pose(point) for point in points]
    positions = np.stack([item[0] for item in poses])
    quaternions = np.stack([item[1] for item in poses])
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    # Keep pure-rotation waypoints ordered even when their translation is zero.
    distances = np.maximum(distances, 1e-9)
    progress = np.concatenate([[0.0], np.cumsum(distances)])
    progress /= progress[-1]
    return PlacePath(part_name, source, positions, quaternions, progress)
