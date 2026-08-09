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
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.noise import NormalActionNoise
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
    "reward_force", "reward_torque", "reward_action", "reward_proximity",
    "reward_step", "reward_success", "reward_unsafe", "reward_timeout",
    "proximity_milestones_reached", "proximity_milestones_total",
    "episode_reward_position", "episode_reward_orientation",
    "episode_reward_progress", "episode_reward_force", "episode_reward_torque",
    "episode_reward_action", "episode_reward_proximity", "episode_reward_step",
    "episode_reward_success", "episode_reward_unsafe", "episode_reward_timeout",
)

EVAL_MONITOR_FIELDS = MONITOR_FIELDS + (
    "final_position_error", "final_rotation_error", "max_force", "max_torque",
    "training_timesteps",
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


class TrainingTimestepEvalCallback(EvalCallback):
    """Attach the evaluated model timestep to every evaluation episode."""

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.set_attr("training_timesteps", self.num_timesteps)
        return super()._on_step()


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
    *,
    monitor_fields: tuple[str, ...] = MONITOR_FIELDS,
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
        info_keywords=monitor_fields,
    )


def scaled_callback_freq(transition_freq: int, n_envs: int) -> int:
    """Convertit une fréquence en transitions vers les appels vectorisés SB3."""
    return max(transition_freq // n_envs, 1)


def resolve_total_timesteps(training: dict, cli_timesteps: int | None) -> int:
    """Resolve CLI override over YAML while retaining the historical default."""
    total_timesteps = (
        cli_timesteps
        if cli_timesteps is not None
        else training.get("total_timesteps", 500_000)
    )
    if (isinstance(total_timesteps, bool)
            or not isinstance(total_timesteps, int)
            or total_timesteps <= 0):
        raise ValueError("timesteps doit être un entier strictement positif")
    return total_timesteps


def learn_model(model: BaseAlgorithm, total_timesteps: int, callbacks: CallbackList) -> None:
    """Start SB3 with the already resolved transition budget."""
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )


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
        learning_starts=5_000,
        buffer_size=training.get("buffer_size", 50_000),
        learning_rate=training.get("learning_rate", 3e-4),
        batch_size=256,
        train_freq=(1, "step"), gradient_steps=-1,
        ent_coef=training.get("ent_coef", "auto"),
        target_entropy=training.get("target_entropy", "auto"),
        policy_kwargs={"net_arch": [256, 256]},
    )


def create_td3_model(
    env: VecEnv,
    training: dict,
    *,
    base_seed: int,
    tensorboard_log: Path,
    device: str,
) -> TD3:
    """Construit TD3 avec les paramètres communs et son bruit d'exploration."""
    td3 = training.get("td3", {})
    action_dim = int(env.action_space.shape[-1])
    noise_std = float(td3.get("action_noise_std", 0.1))
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim), sigma=noise_std * np.ones(action_dim),
    )
    return TD3(
        "MlpPolicy", env, seed=base_seed, verbose=1,
        tensorboard_log=str(tensorboard_log), device=device,
        learning_starts=5_000,
        buffer_size=training.get("buffer_size", 50_000),
        learning_rate=training.get("learning_rate", 3e-4),
        batch_size=256,
        train_freq=(1, "step"), gradient_steps=-1,
        action_noise=action_noise,
        policy_delay=int(td3.get("policy_delay", 2)),
        target_policy_noise=float(td3.get("target_policy_noise", 0.2)),
        target_noise_clip=float(td3.get("target_noise_clip", 0.5)),
        policy_kwargs={"net_arch": [256, 256]},
    )


def create_model(
    env: VecEnv, training: dict, *, base_seed: int,
    tensorboard_log: Path, device: str,
) -> BaseAlgorithm:
    """Sélectionne l'un des deux constructeurs explicites depuis le YAML résolu."""
    common = dict(
        env=env, training=training, base_seed=base_seed,
        tensorboard_log=tensorboard_log, device=device,
    )
    if training["algorithm"] == "sac":
        return create_sac_model(**common)
    if training["algorithm"] == "td3":
        return create_td3_model(**common)
    raise ValueError(
        f"Unsupported RL algorithm: {training['algorithm']}. "
        "Supported algorithms: sac, td3"
    )


def archive_run_context(
    output: Path, args: argparse.Namespace, total_timesteps: int, algorithm: str,
) -> None:
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
        "total_timesteps": total_timesteps,
        "algorithm": algorithm,
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
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Remplace training.total_timesteps pour ce run")
    parser.add_argument("--seed", type=int, default=None,
                        help="Remplace training.base_seed pour ce run")
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None,
                        help="Remplace training.checkpoint_freq pour ce run")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_config = load_config(args.config)
    training = resolved_config["training"]
    algorithm = training["algorithm"]
    total_timesteps = resolve_total_timesteps(training, args.timesteps)
    training["total_timesteps"] = total_timesteps
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
    archive_run_context(output, args, total_timesteps, algorithm)

    env = build_vec_env(output / "config.yaml", n_envs, base_seed, output / "monitor.csv")
    evaluation = resolved_config["evaluation"]
    eval_env = None
    eval_callback = None
    if evaluation["enabled"]:
        eval_dir = output / "eval"
        eval_dir.mkdir()
        eval_env = build_vec_env(
            output / "config.yaml", 1, int(evaluation["seed"]),
            eval_dir / "monitor.csv", monitor_fields=EVAL_MONITOR_FIELDS,
        )
        eval_callback = TrainingTimestepEvalCallback(
            eval_env,
            eval_freq=scaled_callback_freq(int(evaluation["eval_freq"]), n_envs),
            n_eval_episodes=int(evaluation["n_eval_episodes"]),
            deterministic=bool(evaluation["deterministic"]),
            best_model_save_path=str(eval_dir),
            log_path=str(eval_dir),
        )
    model = create_model(
        env, training, base_seed=base_seed,
        tensorboard_log=output / "tensorboard", device=args.device,
    )
    print(f"RL algorithm: {algorithm.upper()}")
    print(f"total_timesteps: {total_timesteps}")
    print(f"buffer_size: {training['buffer_size']}")
    print(f"learning_rate: {training['learning_rate']}")
    print("network: [256, 256]")
    if algorithm == "sac":
        print(f"ent_coef: {training['ent_coef']}")
        print(f"target_entropy: {training['target_entropy']}")
    else:
        print(f"action_noise_std: {training['td3']['action_noise_std']}")
    callback_items = [
        CheckpointCallback(
            scaled_callback_freq(checkpoint_freq, n_envs),
            str(output / "checkpoints"), name_prefix=algorithm,
        ),
        EpisodeMetricsCallback(),
    ]
    if eval_callback is not None:
        callback_items.append(eval_callback)
    callbacks = CallbackList(callback_items)
    try:
        learn_model(model, total_timesteps, callbacks)
        model.save(output / "model")
        print(f"Essai sauvegardé: {output}")
    except KeyboardInterrupt:
        model.save(output / "model_interrupted")
        print(f"Entraînement interrompu; modèle partiel sauvegardé: {output / 'model_interrupted.zip'}")
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
