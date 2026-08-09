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
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from src.assembly_env import TenonMortaiseEnv


def load_evaluation_model(
    model_path: Path, env: TenonMortaiseEnv, algorithm: str,
) -> BaseAlgorithm:
    """Charge le type de modèle archivé avec le run, SAC par défaut historique."""
    if algorithm == "sac":
        return SAC.load(model_path, env=env)
    if algorithm == "td3":
        return TD3.load(model_path, env=env)
    raise ValueError(
        f"Unsupported RL algorithm: {algorithm}. Supported algorithms: sac, td3"
    )


CHECKPOINT_SUMMARY_FIELDS = (
    "checkpoint", "checkpoint_steps", "trained_steps", "model_sha256",
    "episodes", "safe_success_rate", "geometric_success_rate", "unsafe_rate",
    "mean_episode_length", "median_episode_length",
    "median_final_position_error", "median_final_rotation_error",
    "action_frame", "evaluation_design",
)


def _candidate_paths(run: Path, requested: Path) -> list[Path]:
    paths = [requested, run / requested, run / "checkpoints" / requested]
    return paths + [path.with_suffix(".zip") for path in paths if path.suffix != ".zip"]


def checkpoint_steps(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps$", path.stem)
    return int(match.group(1)) if match else None


def find_model(run: Path, requested: Path | None = None) -> Path:
    """Résout un modèle explicite, ou préfère le final puis le dernier checkpoint."""
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


def resolve_models(
    run: Path,
    requested: list[Path] | None,
    all_checkpoints: bool,
) -> list[Path]:
    if requested and all_checkpoints:
        raise ValueError("--model et --all-checkpoints sont mutuellement exclusifs")
    if all_checkpoints:
        models = sorted(
            (run / "checkpoints").glob("*.zip"),
            key=lambda path: (
                checkpoint_steps(path) is None,
                checkpoint_steps(path) or 0,
                path.name,
            ),
        )
        if not models:
            raise FileNotFoundError(f"Aucun checkpoint dans {run / 'checkpoints'}")
        return [path.resolve() for path in models]
    if not requested:
        return [find_model(run)]

    models: list[Path] = []
    seen: set[Path] = set()
    for item in requested:
        path = find_model(run, item)
        if path not in seen:
            seen.add(path); models.append(path)
    return models


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_stochastic_sources(config: dict) -> list[str]:
    """Liste les sources configurées qui rendent les resets dépendants du seed."""
    sources: list[str] = []
    randomization = config.get("randomization", {})
    for field in (
        "mobile_translation", "mobile_rotation_deg",
        "fixed_translation", "fixed_rotation_deg",
    ):
        if np.any(np.asarray(randomization.get(field, 0.0), dtype=float) != 0.0):
            sources.append(f"randomization.{field}")
    friction = np.asarray(randomization.get("friction_scale", [1.0, 1.0]), dtype=float)
    if friction.shape == (2,) and friction[0] != friction[1]:
        sources.append("randomization.friction_scale")

    perception = config.get("perception", {})
    for field in ("translation_noise_std", "rotation_noise_std_deg"):
        if float(perception.get(field, 0.0)) != 0.0:
            sources.append(f"perception.{field}")
    if np.any(np.asarray(perception.get("wrench_noise_std", 0.0), dtype=float) != 0.0):
        sources.append("perception.wrench_noise_std")
    return sources


def summarize_episodes(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Impossible de résumer une évaluation sans épisode")
    lengths = np.asarray([row["steps"] for row in rows], dtype=float)
    final_position = np.asarray([row["final_position_error"] for row in rows], dtype=float)
    final_rotation = np.asarray([row["final_rotation_error"] for row in rows], dtype=float)
    successful_steps = [row["steps"] for row in rows if row["safe_success"]]
    return {
        "episodes": len(rows),
        "safe_success_rate": float(np.mean([row["safe_success"] for row in rows])),
        "geometric_success_rate": float(np.mean([row["geometric_success"] for row in rows])),
        "unsafe_rate": float(np.mean([row["unsafe"] for row in rows])),
        "mean_episode_length": float(np.mean(lengths)),
        "median_episode_length": float(np.median(lengths)),
        "mean_steps_to_success": float(np.mean(successful_steps)) if successful_steps else None,
        "mean_final_position_error": float(np.mean(final_position)),
        "median_final_position_error": float(np.median(final_position)),
        "mean_final_rotation_error": float(np.mean(final_rotation)),
        "median_final_rotation_error": float(np.median(final_rotation)),
        "max_force": float(max(row["episode_max_force"] for row in rows)),
        "max_torque": float(max(row["episode_max_torque"] for row in rows)),
        "effort_terminations": int(sum(
            row["unsafe_force"] or row["unsafe_torque"] for row in rows
        )),
        "workspace_terminations": int(sum(row["unsafe_workspace"] for row in rows)),
        "timeouts": int(sum(row["termination_reason"] == "timeout" for row in rows)),
    }


def trajectory_row(
    episode: int,
    step: int,
    info: dict,
    terminated: bool,
    truncated: bool,
) -> dict:
    keys = (
        "position_error_x", "position_error_y", "position_error_z",
        "rotation_error_x", "rotation_error_y", "rotation_error_z",
        "action_x", "action_y", "action_z", "action_rx", "action_ry", "action_rz",
        "force", "torque", "unsafe", "termination_reason",
    )
    return {
        "episode": episode,
        "step": step,
        **{key: info[key] for key in keys},
        "safe": not bool(info["unsafe"]),
        "terminated": terminated,
        "truncated": truncated,
    }


def evaluate_model(
    *,
    run: Path,
    model_path: Path,
    result_name: str,
    episode_count: int,
    seed: int,
    render: bool,
    render_speed: float,
    write_trajectory: bool,
) -> dict:
    env = TenonMortaiseEnv(
        run / "config.yaml", "human" if render else None, render_speed
    )
    algorithm = env.cfg["training"].get("algorithm", "sac").lower()
    model = load_evaluation_model(model_path, env, algorithm)
    episodes: list[dict] = []
    trajectory: list[dict] = []
    try:
        for episode in range(episode_count):
            obs, _ = env.reset(seed=seed + episode)
            done = False; episode_reward = 0.0; step = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward; step += 1; done = terminated or truncated
                if write_trajectory and episode == 0:
                    trajectory.append(trajectory_row(
                        episode, step, info, terminated, truncated,
                    ))
            episodes.append({
                "episode": episode, "seed": seed + episode,
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
        trained_steps = int(model.num_timesteps)
        action_frame = env.cfg["action"]["action_frame"]
        stochastic_sources = configured_stochastic_sources(env.cfg)
    finally:
        env.close()

    output = run / "evaluations"
    output.mkdir(exist_ok=True)
    episode_path = output / f"{result_name}_episodes.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=episodes[0].keys())
        writer.writeheader(); writer.writerows(episodes)

    trajectory_path: Path | None = None
    if write_trajectory:
        trajectory_path = output / f"{result_name}_trajectory.csv"
        with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=trajectory[0].keys())
            writer.writeheader(); writer.writerows(trajectory)

    deterministic_setup = not stochastic_sources
    summary = {
        "checkpoint": model_path.name,
        "checkpoint_steps": checkpoint_steps(model_path),
        "trained_steps": trained_steps,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        **summarize_episodes(episodes),
        "deterministic_policy": True,
        "configured_stochastic_sources": stochastic_sources,
        "evaluation_design": (
            "single_deterministic_scenario_repeated"
            if deterministic_setup else "seeded_randomized_scenarios"
        ),
        "evaluation_note": (
            f"{episode_count} répétition{'s' if episode_count != 1 else ''} "
            "du même scénario déterministe; "
            "les taux sont descriptifs et ne représentent pas des essais indépendants."
            if deterministic_setup else
            "Politique déterministe évaluée sur des scénarios pseudo-aléatoires seedés."
        ),
        "action_frame": action_frame,
        "episodes_csv": str(episode_path),
        "trajectory_csv": str(trajectory_path) if trajectory_path else None,
    }
    summary_path = output / f"{result_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", dest="models", type=Path, action="append",
                        help="Modèle à évaluer; option répétable")
    parser.add_argument("--all-checkpoints", action="store_true",
                        default=_env_flag("EVAL_ALL_CHECKPOINTS"))
    parser.add_argument("--result-name", default=os.environ.get("EVAL_RESULT_NAME") or None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--trajectory", action="store_true",
                        default=_env_flag("EVAL_TRAJECTORY"),
                        help="Archive la trajectoire détaillée du premier épisode")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-speed", type=float, default=1.0,
                        help="1=temps réel, 0.25=quatre fois plus lent")
    args = parser.parse_args()
    if args.models is None:
        configured_models: list[Path] = []
        if os.environ.get("MODEL_PATH"):
            configured_models.append(Path(os.environ["MODEL_PATH"]))
        configured_models.extend(
            Path(item.strip()) for item in os.environ.get("MODEL_PATHS", "").split(",")
            if item.strip()
        )
        args.models = configured_models or None
    return args


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes doit être strictement positif")
    if args.render_speed <= 0:
        raise ValueError("render-speed doit être strictement positif")
    run = args.run.resolve()
    models = resolve_models(run, args.models, args.all_checkpoints)
    if args.result_name and len(models) != 1:
        raise ValueError("--result-name ne peut être utilisé qu'avec un seul modèle")

    summaries: list[dict] = []
    for model_path in models:
        result_name = args.result_name or model_path.stem
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", result_name):
            raise ValueError("result-name accepte uniquement lettres, chiffres, '.', '_' et '-'")
        summaries.append(evaluate_model(
            run=run, model_path=model_path, result_name=result_name,
            episode_count=args.episodes, seed=args.seed,
            render=args.render, render_speed=args.render_speed,
            write_trajectory=args.trajectory,
        ))

    table_path = run / "evaluations" / "checkpoints_summary.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CHECKPOINT_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({key: summary[key] for key in CHECKPOINT_SUMMARY_FIELDS}
                         for summary in summaries)
    print(json.dumps({
        "checkpoints": summaries,
        "checkpoints_summary_csv": str(table_path),
    }, indent=2))


if __name__ == "__main__":
    main()
