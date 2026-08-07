"""Évaluation déterministe, isolée par modèle, avec métriques de sécurité."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from stable_baselines3 import SAC

from src.assembly_env import TenonMortaiseEnv


def _candidate_paths(run: Path, requested: Path) -> list[Path]:
    paths = [requested, run / requested, run / "checkpoints" / requested]
    return paths + [path.with_suffix(".zip") for path in paths if path.suffix != ".zip"]


def find_model(run: Path, requested: Path | None = None) -> Path:
    """Resolve an explicit model, or prefer final then latest checkpoint."""
    if requested is not None:
        for path in _candidate_paths(run, requested):
            if path.is_file():
                return path.resolve()
        raise FileNotFoundError(f"Modèle demandé introuvable: {requested}")

    final_model = run / "model.zip"
    interrupted_model = run / "model_interrupted.zip"
    if final_model.is_file():
        return final_model.resolve()
    checkpoints = list((run / "checkpoints").glob("*.zip"))
    if interrupted_model.is_file():
        checkpoints.append(interrupted_model)
    if checkpoints:
        checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
        print(f"Modèle final absent; évaluation de: {checkpoint.name}")
        return checkpoint.resolve()
    raise FileNotFoundError(
        f"Aucun modèle à évaluer dans {run}. Attendu: model.zip, "
        "model_interrupted.zip ou checkpoints/*.zip."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    model_default = os.environ.get("MODEL_PATH") or None
    parser.add_argument("--model", type=Path, default=Path(model_default) if model_default else None)
    parser.add_argument("--result-name", default=os.environ.get("EVAL_RESULT_NAME") or None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-speed", type=float, default=1.0,
                        help="1=temps réel, 0.25=quatre fois plus lent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes doit être strictement positif")
    run = args.run.resolve()
    model_path = find_model(run, args.model)
    result_name = args.result_name or model_path.stem
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", result_name):
        raise ValueError("result-name accepte uniquement lettres, chiffres, '.', '_' et '-'")

    env = TenonMortaiseEnv(
        run / "config.yaml", "human" if args.render else None, args.render_speed
    )
    model = SAC.load(model_path, env=env)
    episodes: list[dict] = []
    trajectory: list[dict] = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            done = False; episode_reward = 0.0; step = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward; step += 1; done = terminated or truncated
                trajectory.append({
                    "episode": episode, "step": step,
                    "position_error": info["position_error"],
                    "rotation_error": info["rotation_error"],
                    "force": info["force"], "torque": info["torque"],
                    "max_force_substep": info["max_force_substep"],
                    "max_torque_substep": info["max_torque_substep"],
                })
            episodes.append({
                "episode": episode, "seed": args.seed + episode,
                "success": info["success"], "safe_success": info["safe_success"],
                "geometric_success": info["geometric_success"],
                "unsafe": info["unsafe"],
                "unsafe_force": info["unsafe_force"],
                "unsafe_torque": info["unsafe_torque"],
                "unsafe_workspace": info["unsafe_workspace"],
                "termination_reason": info["termination_reason"],
                "steps": step, "reward": episode_reward,
                "final_position_error": info["position_error"],
                "final_rotation_error": info["rotation_error"],
                "episode_max_force": info["episode_max_force"],
                "episode_max_torque": info["episode_max_torque"],
            })
    finally:
        env.close()

    output = run / "evaluations"
    output.mkdir(exist_ok=True)
    episode_path = output / f"{result_name}_episodes.csv"
    trajectory_path = output / f"{result_name}_trajectory.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=episodes[0].keys())
        writer.writeheader(); writer.writerows(episodes)
    with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=trajectory[0].keys())
        writer.writeheader(); writer.writerows(trajectory)

    successful_steps = [row["steps"] for row in episodes if row["safe_success"]]
    summary = {
        "model_path": str(model_path), "model_sha256": sha256(model_path),
        "episodes": args.episodes,
        "safe_success_rate": float(np.mean([row["safe_success"] for row in episodes])),
        "geometric_success_rate": float(np.mean([row["geometric_success"] for row in episodes])),
        "mean_steps_to_success": float(np.mean(successful_steps)) if successful_steps else None,
        "mean_final_position_error": float(np.mean([row["final_position_error"] for row in episodes])),
        "mean_final_rotation_error": float(np.mean([row["final_rotation_error"] for row in episodes])),
        "max_force": float(max(row["episode_max_force"] for row in episodes)),
        "max_torque": float(max(row["episode_max_torque"] for row in episodes)),
        "effort_terminations": sum(
            row["unsafe_force"] or row["unsafe_torque"] for row in episodes
        ),
        "workspace_terminations": sum(row["unsafe_workspace"] for row in episodes),
        "timeouts": sum(row["termination_reason"] == "timeout" for row in episodes),
        "episodes_csv": str(episode_path), "trajectory_csv": str(trajectory_path),
    }
    summary_path = output / f"{result_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
