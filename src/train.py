"""Entraînement SAC vectorisé et persistance d'un run reproductible."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import tarfile
from typing import Callable

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from src.assembly_env import TenonMortaiseEnv
from src.config import load_config, save_resolved_config


MONITOR_FIELDS = (
    "geometric_success", "success", "safe_success", "unsafe",
    "unsafe_force", "unsafe_torque", "unsafe_workspace",
    "termination_reason", "position_error", "rotation_error",
    "position_error_x", "position_error_y", "position_error_z",
    "rotation_error_x", "rotation_error_y", "rotation_error_z",
    "action_x", "action_y", "action_z", "action_rx", "action_ry", "action_rz",
    "force", "torque", "max_force_substep", "max_torque_substep",
    "episode_max_force", "episode_max_torque", "friction_scale",
    "reward_position", "reward_orientation", "reward_progress",
    "reward_force", "reward_action", "reward_success", "reward_unsafe",
    "episode_reward_position", "episode_reward_orientation",
    "episode_reward_progress", "episode_reward_force",
    "episode_reward_action", "episode_reward_success", "episode_reward_unsafe",
)


class EpisodeMetricsCallback(BaseCallback):
    """Expose les métriques terminales centralisées dans TensorBoard."""

    def _on_step(self) -> bool:
        terminal_infos = [
            info for done, info in zip(
                self.locals.get("dones", []), self.locals.get("infos", [])
            ) if done
        ]
        for key in MONITOR_FIELDS:
            values = [
                float(info[key]) for info in terminal_infos
                if isinstance(info.get(key), (bool, int, float, np.number))
            ]
            if values:
                self.logger.record(f"assembly/{key}", float(np.mean(values)))
        return True


def make_env(config_path: Path, rank: int, base_seed: int) -> Callable[[], TenonMortaiseEnv]:
    """Retourne une factory picklable créant une simulation MuJoCo indépendante."""
    env_seed = base_seed + rank

    def initialize() -> TenonMortaiseEnv:
        env = TenonMortaiseEnv(config_path)
        # Attributs de diagnostic utiles aux tests, sans effet sur step().
        env.worker_rank = rank
        env.worker_seed = env_seed
        env.worker_pid = os.getpid()
        env.action_space.seed(env_seed)
        return env

    return initialize


def build_vec_env(
    config_path: Path,
    n_envs: int,
    base_seed: int,
    monitor_path: Path,
) -> VecMonitor:
    """Construit les workers puis un unique writer VecMonitor dans le parent."""
    if n_envs <= 0:
        raise ValueError("n_envs doit être strictement positif")
    config_path = config_path.resolve()
    factories = [make_env(config_path, rank, base_seed) for rank in range(n_envs)]
    if n_envs == 1:
        vector_env: VecEnv = DummyVecEnv(factories)
    else:
        vector_env = SubprocVecEnv(factories, start_method="spawn")
    vector_env.seed(base_seed)
    return VecMonitor(
        vector_env,
        filename=str(monitor_path),
        info_keywords=MONITOR_FIELDS,
    )


def scaled_callback_freq(transition_freq: int, n_envs: int) -> int:
    """Convertit une fréquence en transitions vers les appels vectorisés SB3."""
    return max(transition_freq // n_envs, 1)


def create_sac_model(
    env: VecEnv,
    training: dict,
    *,
    base_seed: int,
    tensorboard_log: Path,
    device: str,
) -> SAC:
    """Construit SAC avec les hyperparamètres effectifs du YAML résolu."""
    return SAC(
        "MlpPolicy", env, seed=base_seed, verbose=1,
        tensorboard_log=str(tensorboard_log), device=device,
        learning_starts=5_000, buffer_size=50_000, batch_size=256,
        train_freq=(1, "step"), gradient_steps=-1,
        ent_coef=training.get("ent_coef", "auto"),
        target_entropy=training.get("target_entropy", "auto"),
    )


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
            "python": platform.python_version(), "numpy": np.__version__,
            "mujoco": mujoco.__version__, "gymnasium": gymnasium.__version__,
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
    parser.add_argument("--seed", type=int, default=None,
                        help="Remplace training.base_seed pour ce run")
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None,
                        help="Remplace training.checkpoint_freq pour ce run")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("timesteps doit être strictement positif")

    resolved_config = load_config(args.config)
    training = resolved_config["training"]
    if args.seed is not None:
        training["base_seed"] = args.seed
    if args.checkpoint_freq is not None:
        training["checkpoint_freq"] = args.checkpoint_freq
    n_envs = int(training["n_envs"])
    base_seed = int(training["base_seed"])
    checkpoint_freq = int(training["checkpoint_freq"])
    if base_seed < 0 or checkpoint_freq <= 0:
        raise ValueError("base_seed doit être >= 0 et checkpoint_freq > 0")

    # Les valeurs effectives sont archivées à la fois dans le YAML et le manifeste.
    args.n_envs = n_envs
    args.base_seed = base_seed
    args.effective_checkpoint_freq = scaled_callback_freq(checkpoint_freq, n_envs)
    name = args.run or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("data/output") / name
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(resolved_config, output / "config.yaml")
    archive_run_context(output, args)

    env = build_vec_env(output / "config.yaml", n_envs, base_seed, output / "monitor.csv")
    model = create_sac_model(
        env, training, base_seed=base_seed,
        tensorboard_log=output / "tensorboard", device=args.device,
    )
    callbacks = CallbackList([
        CheckpointCallback(
            scaled_callback_freq(checkpoint_freq, n_envs),
            str(output / "checkpoints"), name_prefix="sac",
        ),
        EpisodeMetricsCallback(),
    ])
    try:
        model.learn(args.timesteps, callback=callbacks, progress_bar=True)
        model.save(output / "model")
        print(f"Essai sauvegardé: {output}")
    except KeyboardInterrupt:
        model.save(output / "model_interrupted")
        print(f"Entraînement interrompu; modèle partiel sauvegardé: {output / 'model_interrupted.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
