"""Diagnostic contrefactuel par replay déterministe dans des envs indépendants."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import SAC

from src.assembly_env import AssemblyEnv
from src.config import load_config
from src.diagnose_sac_q import compare_actions
from src.evaluate import find_model
from src.evaluate_scripted import observed_pose_error, proportional_action


def discounted_return(rewards: list[float], gamma: float) -> float:
    return float(sum(reward * gamma ** index for index, reward in enumerate(rewards)))


def p_action(obs: np.ndarray, config: dict) -> np.ndarray:
    action = config["action"]
    controller = config.get("scripted_controller", {})
    return proportional_action(
        observed_pose_error(obs, config),
        max_translation_step=action["max_translation_step"],
        max_rotation_step_deg=action["max_rotation_step_deg"],
        position_gain=float(controller.get("position_gain", 1.0)),
        rotation_gain=float(controller.get("rotation_gain", 1.0)),
    )


def replay_to_step(env: AssemblyEnv, config: dict, step: int, seed: int) -> np.ndarray:
    obs, _ = env.reset(seed=seed)
    for replay_step in range(step):
        obs, _, terminated, truncated, _ = env.step(p_action(obs, config))
        if terminated or truncated:
            raise ValueError(
                f"La trajectoire P termine au step {replay_step + 1}, "
                f"avant le point de branchement {step}"
            )
    return obs


def physical_state(env: AssemblyEnv, obs: np.ndarray) -> dict[str, np.ndarray]:
    mocap_id = env.model.body_mocapid[env.target_mocap]
    return {
        "observation": np.asarray(obs).copy(),
        "pose_error": env._error().copy(),
        "observed_wrench": np.asarray(obs[6:12]).copy(),
        "admittance_offset": env.admittance.offset.copy(),
        "grasp_position": env.data.site_xpos[env.grasp_site].copy(),
        "grasp_quaternion": env._site_quat(),
        "target_position": env.data.mocap_pos[mocap_id].copy(),
        "target_quaternion": env.data.mocap_quat[mocap_id].copy(),
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
    }


def assert_matching_states(states: list[dict[str, np.ndarray]], atol: float = 1e-10) -> None:
    reference = states[0]
    for branch_index, state in enumerate(states[1:], start=1):
        for name, expected in reference.items():
            if not np.allclose(state[name], expected, rtol=0.0, atol=atol):
                difference = float(np.max(np.abs(state[name] - expected)))
                raise RuntimeError(
                    f"Branch state mismatch: branch={branch_index}, field={name}, "
                    f"max_abs_difference={difference:.3e}"
                )


def run_branch(
    env: AssemblyEnv,
    obs: np.ndarray,
    first_action: np.ndarray,
    continuation: Callable[[np.ndarray], np.ndarray],
    gamma: float,
) -> dict:
    before = env._error()
    rewards: list[float] = []
    obs, reward, terminated, truncated, info = env.step(first_action)
    rewards.append(float(reward))
    after = np.asarray(info["true_error"])
    first = {
        "immediate_reward": float(reward),
        "position_error_before_mm": float(np.linalg.norm(before[:3]) * 1e3),
        "position_error_after_mm": float(np.linalg.norm(after[:3]) * 1e3),
        "position_progress_mm": float(
            (np.linalg.norm(before[:3]) - np.linalg.norm(after[:3])) * 1e3
        ),
        "rotation_error_before_deg": float(np.rad2deg(np.linalg.norm(before[3:]))),
        "rotation_error_after_deg": float(np.rad2deg(np.linalg.norm(after[3:]))),
        "rotation_progress_deg": float(np.rad2deg(
            np.linalg.norm(before[3:]) - np.linalg.norm(after[3:])
        )),
        "force_after_N": float(info["force"]),
        "torque_after_Nm": float(info["torque"]),
        "first_unsafe": bool(info["unsafe"]),
        "first_success": bool(info["safe_success"]),
        "first_terminated": bool(terminated),
        "first_truncated": bool(truncated),
        "first_reward_progress": float(info["reward_progress"]),
    }
    length = 1
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(continuation(obs))
        rewards.append(float(reward)); length += 1
    return {
        **first,
        "total_reward": float(sum(rewards)),
        "discounted_return": discounted_return(rewards, gamma),
        "success": bool(info["safe_success"]), "unsafe": bool(info["unsafe"]),
        "termination_reason": info["termination_reason"], "length": length,
    }


def _flatten_branch(prefix: str, result: dict) -> dict:
    return {f"{prefix}_{name}": value for name, value in result.items()}


def diagnose_branch_point(
    *, model: SAC, model_path: Path, config_path: Path, branch_step: int, seed: int,
) -> dict:
    config = load_config(config_path)
    envs = [AssemblyEnv(config_path) for _ in range(3)]
    try:
        observations = [
            replay_to_step(env, config, branch_step, seed) for env in envs
        ]
        assert_matching_states([
            physical_state(env, obs) for env, obs in zip(envs, observations)
        ])
        obs = observations[0]
        branch_error = envs[0]._error()
        action_p = p_action(obs, config)
        action_sac, _ = model.predict(obs, deterministic=True)
        action_sac = np.asarray(action_sac, dtype=np.float32).reshape(6)
        comparison = compare_actions(model, obs, action_p)
        gamma = float(model.gamma)
        use_p = lambda current_obs: p_action(current_obs, config)
        use_sac = lambda current_obs: np.asarray(
            model.predict(current_obs, deterministic=True)[0], dtype=np.float32
        ).reshape(6)
        p_to_p = run_branch(envs[0], observations[0], action_p, use_p, gamma)
        p_to_sac = run_branch(envs[1], observations[1], action_p, use_sac, gamma)
        sac_to_sac = run_branch(envs[2], observations[2], action_sac, use_sac, gamma)
        critic_preference = "SAC" if comparison["qmin_sac"] > comparison["qmin_p"] else "P"
        actual_preference = (
            "SAC" if sac_to_sac["discounted_return"] > p_to_sac["discounted_return"]
            else "P"
        )
        return {
            "checkpoint": model_path.name, "branch_step": branch_step,
            "position_error_mm": float(np.linalg.norm(branch_error[:3]) * 1e3),
            "rotation_error_deg": float(np.rad2deg(np.linalg.norm(branch_error[3:]))),
            **comparison,
            "critic_preference": critic_preference,
            **_flatten_branch("p_to_p", p_to_p),
            **_flatten_branch("p_to_sac", p_to_sac),
            **_flatten_branch("sac_to_sac", sac_to_sac),
            "actual_preference": actual_preference,
            "ranking_agreement": critic_preference == actual_preference,
            "q_error_p": comparison["qmin_p"] - p_to_sac["discounted_return"],
            "q_error_sac": comparison["qmin_sac"] - sac_to_sac["discounted_return"],
            "gamma": gamma,
        }
    finally:
        for env in envs:
            env.close()


def print_result(row: dict) -> None:
    print("\n" + "=" * 58)
    print(f"BRANCH POINT — STEP {row['branch_step']}")
    print(f"Pose error: {row['position_error_mm']:.3f} mm / "
          f"{row['rotation_error_deg']:.3f} deg")
    print("P action:  ", [round(row[f"p_action_{i}"], 4) for i in range(6)])
    print("SAC action:", [round(row[f"sac_action_{i}"], 4) for i in range(6)])
    print("\nCRITICS              Q1          Q2        Qmin")
    print(f"P action      {row['q1_p']:10.4f}  {row['q2_p']:10.4f}  {row['qmin_p']:10.4f}")
    print(f"SAC action    {row['q1_sac']:10.4f}  {row['q2_sac']:10.4f}  {row['qmin_sac']:10.4f}")
    print(f"Critic prefers: {row['critic_preference']}")
    for prefix, label in (("p_to_p", "P → P"), ("p_to_sac", "P → SAC"),
                          ("sac_to_sac", "SAC → SAC")):
        print(f"\nBRANCH {label}")
        print(f"Immediate reward: {row[prefix + '_immediate_reward']:.4f}")
        print(f"After first step: {row[prefix + '_position_error_after_mm']:.3f} mm / "
              f"{row[prefix + '_rotation_error_after_deg']:.3f} deg")
        print(f"Discounted/total return: {row[prefix + '_discounted_return']:.4f} / "
              f"{row[prefix + '_total_reward']:.4f}")
        print(f"Success={row[prefix + '_success']} unsafe={row[prefix + '_unsafe']} "
              f"length={row[prefix + '_length']} reason={row[prefix + '_termination_reason']}")
    print("\nRESULT")
    print(f"Critic ranking: {row['critic_preference']} > "
          f"{'P' if row['critic_preference'] == 'SAC' else 'SAC'}")
    print(f"Actual P→SAC return: {row['p_to_sac_discounted_return']:.4f}")
    print(f"Actual SAC→SAC return: {row['sac_to_sac_discounted_return']:.4f}")
    print(f"Actual ranking: {row['actual_preference']} > "
          f"{'P' if row['actual_preference'] == 'SAC' else 'SAC'}")
    print(f"Ranking agreement: {str(row['ranking_agreement']).upper()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 60, 70, 80])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if any(step < 0 for step in args.steps):
        parser.error("--steps doit contenir des entiers positifs ou nuls")
    return args


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    model_path = find_model(run, args.model)
    config_path = args.config or run / "config.yaml"
    model = SAC.load(model_path, device="auto")
    rows = [
        diagnose_branch_point(
            model=model, model_path=model_path, config_path=config_path,
            branch_step=step, seed=args.seed,
        )
        for step in args.steps
    ]
    output = args.output or run / "sac_branching_diagnostic.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print_result(row)
    mismatches = sum(not row["ranking_agreement"] for row in rows)
    print(f"\nRanking disagreements: {mismatches} / {len(rows)}")
    if mismatches:
        print("The critic ranking disagrees with the observed return at one or more "
              "states. This is evidence of critic estimation/extrapolation error, "
              "not absolute proof.")
    print("P→P is a behavioral baseline; Q(s, action_P) instead corresponds to P "
          "followed by the SAC policy and is compared with P→SAC.")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
