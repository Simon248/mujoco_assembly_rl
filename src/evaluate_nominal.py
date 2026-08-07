"""Evaluate the recorded place path without SAC residual actions."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from src.assembly_env import AssemblyEnv, ResidualConfig


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nominal place-path tracking with a zero RL action."
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=_path_from_env("MUJOCO_XML_PATH", "/data/input/scene.xml"),
    )
    parser.add_argument(
        "--paths-dir",
        type=Path,
        default=_path_from_env("PATHS_DIR", "/data/input/chemin"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_path_from_env("OUTPUT_DIR", "/data/output"),
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        default=None,
        help=(
            "Chemin explicite du rapport JSON. Par défaut : "
            "<output-dir>/<part>/nominal_evaluation.json."
        ),
    )
    parser.add_argument(
        "--part",
        choices=("part_1", "part_2", "part_3"),
        default=os.environ.get("ASSEMBLY_PART", "part_1"),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get("NOMINAL_EPISODES", "5")),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("NOMINAL_SEED", "100")),
    )
    parser.add_argument(
        "--render",
        choices=("none", "human"),
        default=os.environ.get("NOMINAL_RENDER", "none"),
    )
    parser.add_argument(
        "--progress-speed",
        type=float,
        default=float(os.environ.get("NOMINAL_PROGRESS_SPEED", "0.25")),
        help="Vitesse ds/dt utilisée par le suivi nominal.",
    )
    return parser.parse_args()


def nominal_config(progress_speed: float = 0.25) -> ResidualConfig:
    """Exact geometry and no recovery: isolate the nominal controller itself."""
    return replace(
        ResidualConfig(),
        initial_linear_error=0.0,
        initial_angular_error=0.0,
        fixture_linear_error=0.0,
        fixture_angular_error=0.0,
        grasp_linear_error=0.0,
        grasp_angular_error=0.0,
        progress_speed=progress_speed,
        contact_search_enabled=False,
        recovery_enabled=False,
    )


def run_episode(env: AssemblyEnv, seed: int) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    del observation
    action = np.zeros(7, dtype=np.float32)
    terminated = truncated = False
    total_reward = 0.0
    steps = 0
    final_info: dict[str, Any] = {}
    terminal_trace: deque[dict[str, Any]] = deque(maxlen=25)
    while not (terminated or truncated):
        _, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if env._progress >= 0.85:
            terminal_trace.append(
                {
                    "step": steps,
                    "path_progress": env._progress,
                    "true_relative_position_m": env._sensor(
                        env.true_rel_pos_sensor_id
                    ).tolist(),
                    "path_position_m": env.path.pose_at(env._progress)[0].tolist(),
                    "controller_target": env.data.ctrl[env.actuator_ids].tolist(),
                    "wrench": env._wrench().tolist(),
                    "contact": env._has_contact(),
                    "admittance_offset": env._admittance_offset.tolist(),
                }
            )
    target_position, target_quaternion = env.path.final_pose
    return {
        "seed": seed,
        "steps": steps,
        "duration_s": steps * env.config.decision_dt,
        "reward": total_reward,
        "success": bool(final_info["is_success"]),
        "termination_reason": final_info["termination_reason"],
        "terminated": bool(final_info["terminated"]),
        "truncated": bool(final_info["truncated"]),
        "path_progress": float(final_info["path_progress"]),
        "final_position_error_m": float(final_info["final_position_error_m"]),
        "final_rotation_error_rad": float(final_info["final_rotation_error_rad"]),
        "max_force_N": float(final_info["max_force_N"]),
        "max_torque_Nm": float(final_info["max_torque_Nm"]),
        "contact_duration_s": float(final_info["contact_duration_s"]),
        "contact_impulse_Ns": float(final_info["contact_impulse_Ns"]),
        "recovery_count": int(final_info["recovery_count"]),
        "final_wrench": env._wrench().tolist(),
        "true_relative_position_m": env._sensor(env.true_rel_pos_sensor_id).tolist(),
        "true_relative_quaternion_wxyz": env._sensor(
            env.true_rel_quat_sensor_id
        ).tolist(),
        "target_relative_position_m": target_position.tolist(),
        "target_relative_quaternion_wxyz": target_quaternion.tolist(),
        "controller_qpos": env.data.qpos[env.qpos_adr].tolist(),
        "controller_target": env.data.ctrl[env.actuator_ids].tolist(),
        "admittance_offset": env._admittance_offset.tolist(),
        "residual_offset": env._residual_offset.tolist(),
        "terminal_trace": list(terminal_trace),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero")
    if args.progress_speed <= 0.0:
        raise ValueError("--progress-speed must be greater than zero")
    xml_path = args.xml.resolve()
    paths_dir = args.paths_dir.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    if not paths_dir.is_dir():
        raise FileNotFoundError(f"Recorded paths directory not found: {paths_dir}")

    render_mode = "human" if args.render == "human" else None
    config = nominal_config(args.progress_speed)
    env = AssemblyEnv(
        xml_path,
        render_mode=render_mode,
        part_name=args.part,
        paths_dir=paths_dir,
        config=config,
    )
    try:
        episodes = []
        for index in range(args.episodes):
            episode = run_episode(env, args.seed + index)
            episodes.append(episode)
            print(
                f"[nominal] épisode {index + 1}/{args.episodes} : "
                f"raison={episode['termination_reason']}, "
                f"s={episode['path_progress']:.3f}, "
                f"erreur={episode['final_position_error_m'] * 1_000:.2f} mm, "
                f"couple_max={episode['max_torque_Nm']:.3f} Nm",
                flush=True,
            )
    finally:
        env.close()
    success_count = sum(int(item["success"]) for item in episodes)
    termination_reasons = Counter(item["termination_reason"] for item in episodes)
    result = {
        "mode": "nominal_zero_residual",
        "part_name": args.part,
        "xml_path": str(xml_path),
        "paths_dir": str(paths_dir),
        "path_file": str(env.path.source),
        "action": [0.0] * 7,
        "action_semantics": (
            "zero Cartesian residual and nominal forward path progression"
        ),
        "config": asdict(config),
        "episodes_requested": args.episodes,
        "success_count": success_count,
        "success_rate": success_count / args.episodes,
        "baseline_passed": success_count == args.episodes,
        "termination_reasons": dict(termination_reasons),
        "mean_steps": mean(item["steps"] for item in episodes),
        "mean_path_progress": mean(item["path_progress"] for item in episodes),
        "mean_final_position_error_m": mean(item["final_position_error_m"] for item in episodes),
        "mean_final_rotation_error_rad": mean(item["final_rotation_error_rad"] for item in episodes),
        "max_force_N": max(item["max_force_N"] for item in episodes),
        "max_torque_Nm": max(item["max_torque_Nm"] for item in episodes),
        "max_contact_impulse_Ns": max(item["contact_impulse_Ns"] for item in episodes),
        "episodes": episodes,
    }
    result_path = (
        args.result_file.expanduser().resolve()
        if args.result_file is not None
        else args.output_dir.resolve() / args.part / "nominal_evaluation.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    public_result = {
        key: value
        for key, value in result.items()
        if key not in {"episodes", "config"}
    }
    print(json.dumps(public_result, indent=2), flush=True)
    print(f"[nominal] rapport : {result_path}", flush=True)


if __name__ == "__main__":
    main()
