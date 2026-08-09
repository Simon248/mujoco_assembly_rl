"""Évaluation diagnostique avec un contrôleur proportionnel 6D sans RL."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.assembly_env import AssemblyEnv
from src.config import load_config
from src.transforms import quat_to_rotvec, relative


EPISODE_FIELDS = (
    "episode", "success", "unsafe", "unsafe_reason", "termination_reason",
    "terminated", "truncated",
    "length", "reward", "final_position_error_mm", "final_rotation_error_deg",
    "min_position_error_mm", "min_rotation_error_deg", "max_force_N",
    "max_torque_Nm", "max_tracking_position_error_mm",
    "final_tracking_position_error_mm", "max_tracking_rotation_error_deg",
    "final_tracking_rotation_error_deg",
)


def proportional_action(
    pose_error: np.ndarray,
    *,
    max_translation_step: float,
    max_rotation_step_deg: float,
    position_gain: float = 1.0,
    rotation_gain: float = 1.0,
) -> np.ndarray:
    """Convertit une erreur [m, rad] en action normalisée saturée."""
    error = np.asarray(pose_error, dtype=float)
    if error.shape != (6,):
        raise ValueError("pose_error doit avoir la forme (6,)")
    limits = np.r_[
        np.full(3, float(max_translation_step)),
        np.full(3, np.deg2rad(float(max_rotation_step_deg))),
    ]
    gains = np.r_[np.full(3, position_gain), np.full(3, rotation_gain)]
    if np.any(limits <= 0) or np.any(gains <= 0):
        raise ValueError("les limites et gains du contrôleur doivent être positifs")
    return np.clip(-gains * error / limits, -1.0, 1.0).astype(np.float32)


def observed_pose_error(obs: np.ndarray, config: dict) -> np.ndarray:
    """Décode exclusivement l'erreur de pose des six premières observations."""
    observation = np.asarray(obs, dtype=float)
    if observation.shape[0] < 6:
        raise ValueError("l'observation ne contient pas d'erreur de pose 6D")
    scales = config["observation"]
    return np.r_[
        observation[:3] * float(scales["position_scale"]),
        observation[3:6] * float(scales["rotation_scale"]),
    ]


def tracking_error(env: AssemblyEnv) -> tuple[float, float]:
    """Retourne l'écart grasp_target -> grasp réel en mètres et radians."""
    mocap_id = env.model.body_mocapid[env.target_mocap]
    target = (
        env.data.mocap_pos[mocap_id].copy(),
        env.data.mocap_quat[mocap_id].copy(),
    )
    actual = (env.data.site_xpos[env.grasp_site].copy(), env._site_quat())
    error = relative(target, actual)
    return float(np.linalg.norm(error[0])), float(np.linalg.norm(quat_to_rotvec(error[1])))


def _unsafe_reason(info: dict) -> str:
    if info["unsafe_force"]:
        return "force"
    if info["unsafe_torque"]:
        return "torque"
    if info["unsafe_workspace"]:
        return "workspace"
    return ""


def _trajectory_row(env, episode, step, info, action, reward) -> dict:
    true_error = np.asarray(info["true_error"])
    wrench = env._true_wrench()
    tracking_position, tracking_rotation = tracking_error(env)
    target_id = env.model.body_mocapid[env.target_mocap]
    return {
        "episode": episode, "step": step,
        **{f"position_error_{axis}_m": true_error[i] for i, axis in enumerate("xyz")},
        **{f"rotation_error_{axis}_rad": true_error[i + 3] for i, axis in enumerate("xyz")},
        **{f"action_{name}": action[i] for i, name in enumerate(("x", "y", "z", "rx", "ry", "rz"))},
        **{f"wrench_{name}": wrench[i] for i, name in enumerate(("fx", "fy", "fz", "tx", "ty", "tz"))},
        **{f"admittance_offset_{name}": env.admittance.offset[i] for i, name in enumerate(("x", "y", "z", "rx", "ry", "rz"))},
        **{f"grasp_target_{axis}": env.data.mocap_pos[target_id][i] for i, axis in enumerate("xyz")},
        **{f"grasp_actual_{axis}": env.data.site_xpos[env.grasp_site][i] for i, axis in enumerate("xyz")},
        "tracking_position_error_mm": tracking_position * 1e3,
        "tracking_rotation_error_deg": np.rad2deg(tracking_rotation),
        "reward": reward,
    }


def evaluate_scripted(
    config_path: Path, output_path: Path, *, episode_count: int,
    seed: int, render: bool, render_speed: float, write_trajectory: bool,
) -> list[dict]:
    config = load_config(config_path)
    controller = config.get("scripted_controller", {})
    action_config = config["action"]
    env = AssemblyEnv(config_path, "human" if render else None, render_speed)
    rows: list[dict] = []
    trajectory: list[dict] = []
    try:
        for episode in range(episode_count):
            obs, reset_info = env.reset(seed=seed + episode)
            initial_error = np.asarray(reset_info["true_error"])
            min_position = float(np.linalg.norm(initial_error[:3]))
            min_rotation = float(np.linalg.norm(initial_error[3:]))
            tracking_position, tracking_rotation = tracking_error(env)
            max_tracking_position = tracking_position
            max_tracking_rotation = tracking_rotation
            total_reward = 0.0
            step = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = proportional_action(
                    observed_pose_error(obs, config),
                    max_translation_step=action_config["max_translation_step"],
                    max_rotation_step_deg=action_config["max_rotation_step_deg"],
                    position_gain=float(controller.get("position_gain", 1.0)),
                    rotation_gain=float(controller.get("rotation_gain", 1.0)),
                )
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                total_reward += reward
                min_position = min(min_position, float(info["position_error"]))
                min_rotation = min(min_rotation, float(info["rotation_error"]))
                tracking_position, tracking_rotation = tracking_error(env)
                max_tracking_position = max(max_tracking_position, tracking_position)
                max_tracking_rotation = max(max_tracking_rotation, tracking_rotation)
                if write_trajectory and episode == 0:
                    trajectory.append(_trajectory_row(
                        env, episode, step, info, action, reward,
                    ))
            row = {
                "episode": episode + 1,
                "success": bool(info["safe_success"]),
                "unsafe": bool(info["unsafe"]),
                "unsafe_reason": _unsafe_reason(info),
                "termination_reason": info["termination_reason"],
                "terminated": bool(terminated), "truncated": bool(truncated),
                "length": step, "reward": total_reward,
                "final_position_error_mm": float(info["position_error"]) * 1e3,
                "final_rotation_error_deg": float(np.rad2deg(info["rotation_error"])),
                "min_position_error_mm": min_position * 1e3,
                "min_rotation_error_deg": float(np.rad2deg(min_rotation)),
                "max_force_N": float(info["episode_max_force"]),
                "max_torque_Nm": float(info["episode_max_torque"]),
                "max_tracking_position_error_mm": max_tracking_position * 1e3,
                "final_tracking_position_error_mm": tracking_position * 1e3,
                "max_tracking_rotation_error_deg": float(np.rad2deg(max_tracking_rotation)),
                "final_tracking_rotation_error_deg": float(np.rad2deg(tracking_rotation)),
            }
            rows.append(row)
            print(
                f"Episode {episode + 1}: {info['termination_reason']}; "
                f"success={row['success']} unsafe={row['unsafe']} "
                f"reason={row['unsafe_reason'] or '-'} steps={step} "
                f"final={row['final_position_error_mm']:.3f} mm/"
                f"{row['final_rotation_error_deg']:.3f} deg "
                f"best={row['min_position_error_mm']:.3f} mm/"
                f"{row['min_rotation_error_deg']:.3f} deg "
                f"max={row['max_force_N']:.2f} N/{row['max_torque_Nm']:.3f} Nm "
                f"reward={total_reward:.3f}"
            )
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    if write_trajectory:
        trajectory_path = output_path.with_name(f"{output_path.stem}_trajectory.csv")
        with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=trajectory[0].keys())
            writer.writeheader(); writer.writerows(trajectory)
    return rows


def _stats(rows: list[dict], field: str) -> str:
    values = np.asarray([row[field] for row in rows], dtype=float)
    return (f"median={np.median(values):.3f}, mean={np.mean(values):.3f}, "
            f"min={np.min(values):.3f}, max={np.max(values):.3f}")


def print_summary(rows: list[dict], output_path: Path) -> None:
    successes = sum(row["success"] for row in rows)
    print("\nSCRIPTED CONTROLLER EVALUATION")
    print(f"Episodes: {len(rows)}")
    print(f"Safe successes: {successes} / {len(rows)} ({100 * successes / len(rows):.1f} %)")
    print("Unsafe: force={}, torque={}, workspace={}".format(
        sum(row["unsafe_reason"] == "force" for row in rows),
        sum(row["unsafe_reason"] == "torque" for row in rows),
        sum(row["unsafe_reason"] == "workspace" for row in rows),
    ))
    print(f"Timeouts: {sum(row['termination_reason'] == 'timeout' for row in rows)}")
    for label, field in (
        ("Final position error [mm]", "final_position_error_mm"),
        ("Final rotation error [deg]", "final_rotation_error_deg"),
        ("Best position error [mm]", "min_position_error_mm"),
        ("Best rotation error [deg]", "min_rotation_error_deg"),
        ("Max force [N]", "max_force_N"), ("Max torque [Nm]", "max_torque_Nm"),
        ("Max tracking position error [mm]", "max_tracking_position_error_mm"),
        ("Final tracking position error [mm]", "final_tracking_position_error_mm"),
        ("Max tracking rotation error [deg]", "max_tracking_rotation_error_deg"),
        ("Final tracking rotation error [deg]", "final_tracking_rotation_error_deg"),
    ):
        print(f"{label}: {_stats(rows, field)}")
    print(f"CSV: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/test1V14.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes doit être strictement positif")
    if args.render_speed <= 0:
        parser.error("--render-speed doit être strictement positif")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    episode_count = args.episodes or int(
        config.get("scripted_evaluation", {}).get("n_episodes", 20)
    )
    output = args.output or Path("data/output") / args.config.stem / "scripted_eval.csv"
    rows = evaluate_scripted(
        args.config, output, episode_count=episode_count, seed=args.seed,
        render=args.render, render_speed=args.render_speed,
        write_trajectory=args.trajectory,
    )
    print_summary(rows, output)


if __name__ == "__main__":
    main()
