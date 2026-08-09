"""Compare les actions P/SAC selon les critics, sans aucun apprentissage."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from src.assembly_env import AssemblyEnv
from src.config import load_config
from src.evaluate import resolve_models
from src.evaluate_scripted import observed_pose_error, proportional_action


def critic_values(model: SAC, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Retourne Q1/Q2 online par élément, sans gradient ni changement de poids."""
    obs = np.asarray(observations, dtype=np.float32)
    act = np.asarray(actions, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    if act.ndim == 1:
        act = act[None, :]
    if obs.shape[0] != act.shape[0]:
        raise ValueError("observations et actions doivent avoir le même batch")
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    action_tensor = torch.as_tensor(act, device=model.device)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        critics = model.critic(obs_tensor, action_tensor)
    return np.column_stack([
        critic.detach().cpu().numpy().reshape(-1) for critic in critics
    ])


def compare_actions(model: SAC, obs: np.ndarray, p_action: np.ndarray) -> dict:
    sac_action, _ = model.predict(obs, deterministic=True)
    sac_action = np.asarray(sac_action, dtype=np.float32).reshape(6)
    p_action = np.asarray(p_action, dtype=np.float32).reshape(6)
    q = critic_values(model, np.stack([obs, obs]), np.stack([p_action, sac_action]))
    q1_p, q2_p = map(float, q[0]); q1_sac, q2_sac = map(float, q[1])
    p_norm = float(np.linalg.norm(p_action))
    sac_norm = float(np.linalg.norm(sac_action))
    cosine = (
        float(np.dot(p_action, sac_action) / (p_norm * sac_norm))
        if p_norm > 0 and sac_norm > 0 else float("nan")
    )
    return {
        **{f"p_action_{i}": float(value) for i, value in enumerate(p_action)},
        **{f"sac_action_{i}": float(value) for i, value in enumerate(sac_action)},
        "p_translation_magnitude": float(np.linalg.norm(p_action[:3])),
        "p_rotation_magnitude": float(np.linalg.norm(p_action[3:])),
        "sac_translation_magnitude": float(np.linalg.norm(sac_action[:3])),
        "sac_rotation_magnitude": float(np.linalg.norm(sac_action[3:])),
        "action_distance": float(np.linalg.norm(sac_action - p_action)),
        "action_cosine_similarity": cosine,
        "q1_p": q1_p, "q2_p": q2_p, "qmin_p": min(q1_p, q2_p),
        "q1_sac": q1_sac, "q2_sac": q2_sac,
        "qmin_sac": min(q1_sac, q2_sac),
        "delta_qmin_sac_minus_p": min(q1_sac, q2_sac) - min(q1_p, q2_p),
        "critic_disagreement_p": abs(q1_p - q2_p),
        "critic_disagreement_sac": abs(q1_sac - q2_sac),
    }


def observation_fields(obs: np.ndarray, config: dict) -> dict:
    error = observed_pose_error(obs, config)
    scales = config["observation"]
    wrench = np.r_[
        np.asarray(obs[6:9]) * float(scales["force_scale"]),
        np.asarray(obs[9:12]) * float(scales["torque_scale"]),
    ]
    fields = {
        **{f"position_error_{axis}_mm": float(error[i] * 1e3)
           for i, axis in enumerate("xyz")},
        **{f"rotation_error_{axis}_deg": float(np.rad2deg(error[i + 3]))
           for i, axis in enumerate("xyz")},
        **{f"force_{axis}_N": float(wrench[i]) for i, axis in enumerate("xyz")},
        **{f"torque_{axis}_Nm": float(wrench[i + 3]) for i, axis in enumerate("xyz")},
    }
    offset = np.asarray(obs[12:18]) if len(obs) >= 18 else np.zeros(6)
    if len(obs) >= 18:
        offset = offset * np.asarray(config["admittance"]["max_offset"], dtype=float)
    fields.update({f"admittance_offset_{i}": float(value) for i, value in enumerate(offset)})
    return fields


def print_state(row: dict, label: str) -> None:
    print(f"\nSTATE {row['step']} — {label}")
    print("Pose error xyz [mm]:", [round(row[f"position_error_{a}_mm"], 3) for a in "xyz"])
    print("Pose error rot [deg]:", [round(row[f"rotation_error_{a}_deg"], 3) for a in "xyz"])
    print("Wrench force [N]:", [round(row[f"force_{a}_N"], 3) for a in "xyz"])
    print("Wrench torque [Nm]:", [round(row[f"torque_{a}_Nm"], 4) for a in "xyz"])
    print("P action:  ", [round(row[f"p_action_{i}"], 4) for i in range(6)])
    print("SAC action:", [round(row[f"sac_action_{i}"], 4) for i in range(6)])
    print("                 Q1          Q2        Qmin")
    print(f"P action   {row['q1_p']:10.4f}  {row['q2_p']:10.4f}  {row['qmin_p']:10.4f}")
    print(f"SAC action {row['q1_sac']:10.4f}  {row['q2_sac']:10.4f}  {row['qmin_sac']:10.4f}")
    print(f"Delta Qmin SAC-P: {row['delta_qmin_sac_minus_p']:.4f}")
    print(f"Critic disagreement P/SAC: {row['critic_disagreement_p']:.4f} / "
          f"{row['critic_disagreement_sac']:.4f}")


def diagnose_model(
    model_path: Path, config_path: Path, *, seed: int, interval: int,
) -> tuple[list[dict], bool]:
    config = load_config(config_path)
    action_config = config["action"]
    controller = config.get("scripted_controller", {})
    env = AssemblyEnv(config_path)
    model = SAC.load(model_path, env=env, device="auto")
    rows: list[dict] = []
    try:
        obs, _ = env.reset(seed=seed)
        step = 0
        terminated = truncated = False
        while not (terminated or truncated):
            p_action = proportional_action(
                observed_pose_error(obs, config),
                max_translation_step=action_config["max_translation_step"],
                max_rotation_step_deg=action_config["max_rotation_step_deg"],
                position_gain=float(controller.get("position_gain", 1.0)),
                rotation_gain=float(controller.get("rotation_gain", 1.0)),
            )
            if step % interval == 0:
                row = {
                    "checkpoint": model_path.name, "step": step,
                    **observation_fields(obs, config),
                    **compare_actions(model, obs, p_action),
                }
                rows.append(row)
                print_state(row, "RESET" if step == 0 else "P TRAJECTORY")
            obs, _, terminated, truncated, info = env.step(p_action)
            step += 1
        # Toujours inclure le dernier état observé, notamment près du succès.
        if not rows or rows[-1]["step"] != step:
            p_action = proportional_action(
                observed_pose_error(obs, config),
                max_translation_step=action_config["max_translation_step"],
                max_rotation_step_deg=action_config["max_rotation_step_deg"],
                position_gain=float(controller.get("position_gain", 1.0)),
                rotation_gain=float(controller.get("rotation_gain", 1.0)),
            )
            row = {"checkpoint": model_path.name, "step": step,
                   **observation_fields(obs, config),
                   **compare_actions(model, obs, p_action)}
            rows.append(row); print_state(row, "TERMINAL")
        return rows, bool(info["safe_success"])
    finally:
        env.close()


def print_summary(rows: list[dict], success: bool) -> None:
    deltas = np.asarray([row["delta_qmin_sac_minus_p"] for row in rows])
    prefer_p = int(np.sum(deltas < 0))
    prefer_sac = int(np.sum(deltas > 0))
    print("\nP-CONTROLLER SUCCESS TRAJECTORY")
    print(f"P trajectory safe success: {success}")
    print(f"States analyzed: {len(rows)}")
    print(f"Critics prefer P action: {prefer_p} / {len(rows)}")
    print(f"Critics prefer SAC action: {prefer_sac} / {len(rows)}")
    print(f"Median Qmin P: {np.median([r['qmin_p'] for r in rows]):.4f}")
    print(f"Median Qmin SAC: {np.median([r['qmin_sac'] for r in rows]):.4f}")
    print(f"Median delta Q: {np.median(deltas):.4f}")
    print(f"Mean action distance SAC↔P: {np.mean([r['action_distance'] for r in rows]):.4f}")
    print("At initial state critics prefer:", "SAC" if deltas[0] > 0 else "P")
    print("Near target critics prefer:", "SAC" if deltas[-1] > 0 else "P")
    ratio = prefer_sac / len(rows)
    if ratio >= 0.6:
        print("Possible critic miscalibration / overestimation: critics often rank the "
              "SAC action above the known successful P action.")
    elif prefer_p / len(rows) >= 0.6:
        print("Critics often prefer the P action, but the deterministic actor does not "
              "produce it. Investigate actor optimization / policy update.")
    else:
        print("The critic ranking is mixed along the trajectory; no single mechanism dominates.")
    print("Q(s,a) estimates future return; it is not the immediate reward.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", dest="models", type=Path, action="append")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval doit être strictement positif")
    return args


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    config = args.config or run / "config.yaml"
    models = resolve_models(run, args.models, args.all_checkpoints)
    all_rows: list[dict] = []
    checkpoint_preferences: list[int] = []
    for model_path in models:
        print(f"\n===== CHECKPOINT {model_path.name} =====")
        rows, success = diagnose_model(
            model_path, config, seed=args.seed, interval=args.interval,
        )
        print_summary(rows, success)
        all_rows.extend(rows)
        checkpoint_preferences.append(int(np.median(
            [row["delta_qmin_sac_minus_p"] for row in rows]
        ) > 0))
    output = args.output or run / "sac_q_diagnostic.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys())
        writer.writeheader(); writer.writerows(all_rows)
    if len(set(checkpoint_preferences)) > 1:
        print("Critic/action ranking changes significantly across checkpoints; "
              "possible training instability.")
    print(f"\nCSV: {output}")


if __name__ == "__main__":
    main()
