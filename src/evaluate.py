from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from stable_baselines3 import SAC

from src.assembly_env import AssemblyEnv
import numpy as np

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
            "none",
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
def main() -> None:
    args = parse_args()

    xml_path = args.xml.resolve()
    output_dir = args.output_dir.resolve()
    requested_model_path = args.model.resolve()
    result_file = args.result_file.resolve()

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
        curriculum_enabled=False,
        disassembly_probability=0.0,
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

    summary = {
        "model_path": str(model_path),
        "xml_path": str(xml_path),
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


if __name__ == "__main__":
    main()
