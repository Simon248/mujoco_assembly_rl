from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from stable_baselines3 import SAC

from src.assembly_env import AssemblyEnv
import numpy as np


STEP_REWARD_METRICS = (
    "offset_cost",
    "dense_reward",
    "terminal_reward",
)

EVALUATION_METRICS = {
    "path_progress": "mean_final_path_progress",
    "max_path_progress": "mean_max_path_progress",
    "final_position_error_m": "mean_final_position_error_m",
    "final_rotation_error_rad": "mean_final_rotation_error_rad",
    "max_force_N": "mean_max_force_N",
    "max_torque_Nm": "mean_max_torque_Nm",
    "contact_impulse_Ns": "mean_contact_impulse_Ns",
    "contact_duration_s": "mean_contact_duration_s",
    "recovery_count": "mean_recovery_count",
    "recovery_duration_s": "mean_recovery_duration_s",
    "contact_search_latched": "contact_search_latched_rate",
    "residual_linear_offset_m": "mean_final_residual_linear_offset_m",
    "residual_angular_offset_rad": "mean_final_residual_angular_offset_rad",
}

def _path_from_env(name: str, default: str) -> Path:
    return Path(
        os.environ.get(name, default)
    ).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SAC policy."
    )

    parser.add_argument(
        "--xml",
        type=Path,
        default=_path_from_env(
            "MUJOCO_XML_PATH",
            "/data/input/scene.xml",
        ),
    )
    parser.add_argument(
        "--part",
        choices=("part_1", "part_2", "part_3"),
        default=os.environ.get("ASSEMBLY_PART", "part_1"),
    )
    parser.add_argument(
        "--paths-dir",
        type=Path,
        default=_path_from_env("PATHS_DIR", "/data/input/chemin"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_path_from_env(
            "OUTPUT_DIR",
            "/data/output",
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=_path_from_env(
            "MODEL_PATH",
            "/data/output/models/assembly_sac.zip",
        ),
    )

    parser.add_argument(
        "--result-file",
        type=Path,
        default=_path_from_env(
            "EVAL_RESULT_PATH",
            "/data/output/evaluation.json",
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=int(
            os.environ.get(
                "EVAL_EPISODES",
                "10",
            )
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=int(
            os.environ.get(
                "EVAL_SEED",
                "100",
            )
        ),
    )

    parser.add_argument(
        "--render",
        choices=(
            "none",
            "human",
        ),
        default=os.environ.get(
            "EVAL_RENDER",
            "human",
        ),
    )

    return parser.parse_args()


def find_latest_model(
    output_dir: Path,
) -> Path | None:
    candidates: list[Path] = []

    candidates.extend(
        (output_dir / "models").glob("*.zip")
    )

    candidates.extend(
        (output_dir / "checkpoints").glob("*.zip")
    )

    candidates = [
        path
        for path in candidates
        if path.is_file()
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )

def json_default(value):
    """Convertit les types NumPy en types JSON standards."""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Type non sérialisable en JSON : {type(value).__name__}"
    )


def _mean_final_metric(
    episodes: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        float(item["final_info"][key])
        for item in episodes
        if isinstance(item["final_info"].get(key), (int, float, np.number))
    ]
    return mean(values) if values else None


def _write_episode_csv(
    path: Path,
    episodes: list[dict[str, Any]],
) -> None:
    """Write one flat, inspectable row per deterministic evaluation episode."""
    base_fields = ["episode", "seed", "reward", "steps", "success"]
    total_fields = [f"{key}_sum" for key in STEP_REWARD_METRICS]
    info_fields = sorted(
        {
            key
            for episode in episodes
            for key, value in episode["final_info"].items()
            if key != "config" and not isinstance(value, dict)
        }
    )
    fieldnames = base_fields + total_fields + info_fields

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for episode in episodes:
            row = {key: episode[key] for key in base_fields}
            row.update(episode["reward_metric_sums"])
            for key in info_fields:
                value = episode["final_info"].get(key)
                if isinstance(value, (list, tuple, np.ndarray)):
                    value = json.dumps(value, default=json_default)
                row[key] = value
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    xml_path = args.xml.resolve()
    output_dir = args.output_dir.resolve() / args.part
    requested_model_path = args.model.resolve()
    raw_result_file = args.result_file.resolve()
    result_file = raw_result_file.parent / args.part / raw_result_file.name
    paths_dir = args.paths_dir.resolve()

    if not xml_path.is_file():
        raise FileNotFoundError(
            f"MuJoCo XML not found: {xml_path}"
        )

    model_path = requested_model_path

    if not model_path.is_file():
        fallback = find_latest_model(
            output_dir
        )

        if fallback is None:
            raise FileNotFoundError(
                "Aucun modèle entraîné trouvé. "
                "Le modèle final est créé à la fin de "
                "l'entraînement et les checkpoints sont "
                "créés après CHECKPOINT_FREQ pas. "
                "Vérifie data/output/checkpoints et "
                "data/output/models."
            )

        model_path = fallback

        print(
            "[evaluate] modèle final absent ; "
            "utilisation du modèle le plus récent : "
            f"{model_path}",
            flush=True,
        )

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be greater than zero"
        )

    render_mode = "human" if args.render == "human" else None

    env = AssemblyEnv(
        xml_path,
        render_mode=render_mode,
        part_name=args.part,
        paths_dir=paths_dir,
    )


    model = SAC.load(
        str(model_path),
        env=env,
    )

    episodes: list[dict[str, Any]] = []

    try:
        for episode_index in range(
            args.episodes
        ):
            obs, _ = env.reset(
                seed=args.seed + episode_index
            )

            terminated = False
            truncated = False
            total_reward = 0.0
            reward_metric_sums = {
                f"{key}_sum": 0.0
                for key in STEP_REWARD_METRICS
            }
            steps = 0
            final_info: dict[str, Any] = {}

            while not (
                terminated or truncated
            ):
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    final_info,
                ) = env.step(action)

                total_reward += float(reward)
                for key in STEP_REWARD_METRICS:
                    value = final_info.get(key)
                    if isinstance(value, (int, float, np.number)):
                        reward_metric_sums[f"{key}_sum"] += float(value)
                steps += 1

            success = bool(
                final_info.get(
                    "is_success",
                    False,
                )
            )

            episodes.append(
                {
                    "episode": episode_index,
                    "seed": (
                        args.seed
                        + episode_index
                    ),
                    "reward": total_reward,
                    "steps": steps,
                    "success": success,
                    "reward_metric_sums": reward_metric_sums,
                    "final_info": final_info,
                }
            )

            print(
                f"[evaluate] épisode "
                f"{episode_index + 1}/"
                f"{args.episodes} : "
                f"reward={total_reward:.3f}, "
                f"steps={steps}, "
                f"success={success}",
                flush=True,
            )

    finally:
        env.close()

    successes = sum(
        int(item["success"])
        for item in episodes
    )

    evaluation_metrics = {
        summary_key: value
        for key, summary_key in EVALUATION_METRICS.items()
        if (value := _mean_final_metric(episodes, key)) is not None
    }
    evaluation_metrics.update(
        {
            f"mean_{key}_sum": mean(
                float(item["reward_metric_sums"][f"{key}_sum"])
                for item in episodes
            )
            for key in STEP_REWARD_METRICS
        }
    )
    evaluation_metrics["contact_search_trigger_counts"] = dict(
        Counter(
            str(item["final_info"].get("contact_search_trigger", "unknown"))
            for item in episodes
        )
    )

    episode_csv_file = result_file.with_suffix(".csv")

    summary = {
        "model_path": str(model_path),
        "xml_path": str(xml_path),
        "part_name": args.part,
        "paths_dir": str(paths_dir),
        "episode_count": len(episodes),
        "success_count": successes,
        "success_rate": (
            successes / len(episodes)
        ),
        "mean_reward": mean(
            float(item["reward"])
            for item in episodes
        ),
        "mean_steps": mean(
            int(item["steps"])
            for item in episodes
        ),
        "evaluation_metrics": evaluation_metrics,
        "episode_csv_path": str(episode_csv_file),
        "episodes": episodes,
    }

    result_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_file.write_text(
        json.dumps(
            summary,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    _write_episode_csv(episode_csv_file, episodes)

    public_summary = {
        key: value
        for key, value in summary.items()
        if key != "episodes"
    }

    print(
        json.dumps(
            public_summary,
            indent=2,
            default=json_default,
        )
    )

    print(
        f"[evaluate] résultat détaillé : "
        f"{result_file}"
    )
    print(
        f"[evaluate] résumé CSV par épisode : "
        f"{episode_csv_file}"
    )


if __name__ == "__main__":
    main()
