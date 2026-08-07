"""Entraînement SAC et persistance complète d'un essai reproductible."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import tarfile

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.assembly_env import TenonMortaiseEnv
from src.config import load_config, save_resolved_config


MONITOR_FIELDS = (
    "geometric_success", "success", "safe_success", "unsafe",
    "unsafe_force", "unsafe_torque", "unsafe_workspace",
    "termination_reason", "position_error", "rotation_error",
    "force", "torque", "max_force_substep", "max_torque_substep",
    "episode_max_force", "episode_max_torque", "friction_scale",
    "reward_position", "reward_orientation", "reward_progress",
    "reward_force", "reward_action", "reward_success", "reward_unsafe",
    "episode_reward_position", "episode_reward_orientation",
    "episode_reward_progress", "episode_reward_force",
    "episode_reward_action", "episode_reward_success", "episode_reward_unsafe",
)


class EpisodeMetricsCallback(BaseCallback):
    """Expose les métriques terminales du Monitor dans TensorBoard."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if not done:
                continue
            for key in MONITOR_FIELDS:
                value = info.get(key)
                if isinstance(value, (bool, int, float, np.number)):
                    self.logger.record(f"assembly/{key}", float(value))
        return True


def archive_run_context(output: Path, args: argparse.Namespace) -> None:
    """Archive versions, inputs and a source snapshot without invoking Git."""
    def source_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(member.name).parts
        return None if "__pycache__" in parts or member.name.endswith((".pyc", ".pyo")) else member

    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for source in (
            "src", "tests", "configs", "requirements.txt", "Dockerfile",
            "docker-compose.yml", "Makefile", "README.md",
        ):
            path = Path(source)
            if path.exists():
                archive.add(path, arcname=path.as_posix(), filter=source_filter)
    input_hashes = {}
    for root in (
        Path("data/input/cad/tenon-mortaise"),
        Path("data/input/grasp_poses/tenon"),
    ):
        for path in sorted(root.glob("*")):
            if path.is_file():
                input_hashes[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": vars(args),
        "input_sha256": input_hashes,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test1.yaml")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0 or args.checkpoint_freq <= 0:
        raise ValueError("timesteps et checkpoint-freq doivent être strictement positifs")

    resolved_config = load_config(args.config)
    name = args.run or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("data/output") / name
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(resolved_config, output / "config.yaml")
    archive_run_context(output, args)

    # L'environnement lit la copie archivée : entraînement et évaluation
    # utilisent ainsi exactement le même document autonome.
    env = Monitor(
        TenonMortaiseEnv(output / "config.yaml"),
        filename=str(output / "monitor.csv"),
        info_keywords=MONITOR_FIELDS,
    )
    model = SAC(
        "MlpPolicy", env, seed=args.seed, verbose=1,
        tensorboard_log=str(output / "tensorboard"), device=args.device,
        learning_starts=5_000, buffer_size=50_000, batch_size=256,
    )
    callback = CallbackList([
        CheckpointCallback(
            args.checkpoint_freq, str(output / "checkpoints"), name_prefix="sac"
        ),
        EpisodeMetricsCallback(),
    ])
    try:
        model.learn(args.timesteps, callback=callback, progress_bar=True)
        model.save(output / "model")
        print(f"Essai sauvegardé: {output}")
    except KeyboardInterrupt:
        model.save(output / "model_interrupted")
        print(f"Entraînement interrompu; modèle partiel sauvegardé: {output / 'model_interrupted.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
