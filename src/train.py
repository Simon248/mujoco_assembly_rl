from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.assembly_env import AssemblyEnv


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


class ConsoleProgressCallback(BaseCallback):
    """Affiche périodiquement la progression dans les logs Docker."""

    def __init__(
        self,
        total_timesteps: int,
        print_freq: int = 250,
    ) -> None:
        super().__init__(verbose=0)

        self.total_timesteps_target = total_timesteps
        self.print_freq = max(1, print_freq)

        self._next_print = self.print_freq
        self._start_time = 0.0
        self._episodes = 0
        self._last_info: dict[str, Any] = {}

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()

        print(
            f"[train] démarrage : objectif={self.total_timesteps_target} pas, "
            f"affichage tous les {self.print_freq} pas",
            flush=True,
        )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")

        if dones is not None:
            self._episodes += int(
                np.asarray(dones, dtype=np.int32).sum()
            )

        infos = self.locals.get("infos")

        if infos:
            self._last_info = dict(infos[-1])
            for key in (
                "position_error_m",
                "lateral_error_m",
                "rotation_error_rad",
                "force_norm_N",
                "curriculum_stage",
                "curriculum_success_rate",
                "is_disassembly",
            ):
                if key in self._last_info:
                    self.logger.record(f"assembly/{key}", self._last_info[key])

        if (
            self.num_timesteps >= self._next_print
            or self.num_timesteps >= self.total_timesteps_target
        ):
            elapsed = max(
                time.monotonic() - self._start_time,
                1e-9,
            )

            speed = self.num_timesteps / elapsed

            percent = (
                100.0
                * self.num_timesteps
                / self.total_timesteps_target
            )

            extra = ""

            if self._last_info:
                position_error = self._last_info.get(
                    "position_error_m",
                    float("nan"),
                )

                force_norm = self._last_info.get(
                    "force_norm_N",
                    float("nan"),
                )

                success = bool(
                    self._last_info.get(
                        "is_success",
                        False,
                    )
                )

                extra = (
                    f" | erreur={position_error:.4f} m"
                    f" | force={force_norm:.1f} N"
                    f" | succès={success}"
                )

            print(
                f"[train] {self.num_timesteps}/"
                f"{self.total_timesteps_target} "
                f"({percent:.1f} %) "
                f"| {speed:.1f} pas/s "
                f"| épisodes={self._episodes}"
                f"{extra}",
                flush=True,
            )

            while self._next_print <= self.num_timesteps:
                self._next_print += self.print_freq

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC on the MuJoCo assembly task."
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
        "--timesteps",
        type=int,
        default=int(
            os.environ.get(
                "TOTAL_TIMESTEPS",
                "500000",
            )
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=int(
            os.environ.get(
                "SEED",
                "7",
            )
        ),
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=int(
            os.environ.get(
                "CHECKPOINT_FREQ",
                "50000",
            )
        ),
    )

    parser.add_argument(
        "--learning-starts",
        type=int,
        default=int(
            os.environ.get(
                "LEARNING_STARTS",
                "5000",
            )
        ),
    )

    parser.add_argument(
        "--log-freq",
        type=int,
        default=int(
            os.environ.get(
                "LOG_FREQ",
                "250",
            )
        ),
    )

    parser.add_argument(
        "--skip-env-check",
        action="store_true",
    )

    return parser.parse_args()


def write_metadata(
    output_dir: Path,
    *,
    xml_path: Path,
    timesteps_requested: int,
    timesteps_completed: int,
    seed: int,
    model_path: Path,
    completed: bool,
) -> None:
    metadata = {
        "algorithm": "SAC",
        "task": "chandelier_cad_assembly",
        "action_space": "[dx, dy, dz, droll, dpitch, dyaw]",
        "cad_collision": "MuJoCo SDF generated from the original STL meshes",
        "xml_path": str(xml_path),
        "timesteps_requested": timesteps_requested,
        "timesteps_completed": timesteps_completed,
        "seed": seed,
        "completed": completed,
        "model_path": str(model_path),
    }

    metadata_path = output_dir / "training_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    xml_path = args.xml.resolve()
    output_dir = args.output_dir.resolve()

    if not xml_path.is_file():
        raise FileNotFoundError(
            f"MuJoCo XML not found: {xml_path}"
        )

    if args.timesteps <= 0:
        raise ValueError(
            "--timesteps must be greater than zero"
        )

    if args.checkpoint_freq <= 0:
        raise ValueError(
            "--checkpoint-freq must be greater than zero"
        )

    if args.learning_starts < 0:
        raise ValueError(
            "--learning-starts cannot be negative"
        )

    if args.log_freq <= 0:
        raise ValueError(
            "--log-freq must be greater than zero"
        )

    model_dir = output_dir / "models"
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"
    monitor_dir = output_dir / "monitor"

    for directory in (
        model_dir,
        checkpoint_dir,
        tensorboard_dir,
        monitor_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("[train] configuration", flush=True)
    print(f"[train] XML             : {xml_path}", flush=True)
    print(f"[train] sortie          : {output_dir}", flush=True)
    print(f"[train] timesteps       : {args.timesteps}", flush=True)
    print(
        f"[train] learning_starts : {args.learning_starts}",
        flush=True,
    )
    print(
        f"[train] checkpoint_freq : {args.checkpoint_freq}",
        flush=True,
    )

    if args.learning_starts >= args.timesteps:
        print(
            "[train] ATTENTION : learning_starts >= timesteps. "
            "Le buffer sera rempli, mais aucune mise à jour "
            "du réseau ne sera effectuée.",
            flush=True,
        )

    if not args.skip_env_check:
        print(
            "[train] validation Gymnasium...",
            flush=True,
        )

        test_env = AssemblyEnv(xml_path)

        check_env(
            test_env,
            warn=True,
        )

        test_env.close()

        print(
            "[train] environnement valide",
            flush=True,
        )

    env = Monitor(
        AssemblyEnv(xml_path),
        filename=str(
            monitor_dir / "train"
        ),
        info_keywords=(
            "is_success",
            "position_error_m",
            "lateral_error_m",
            "rotation_error_rad",
            "force_norm_N",
            "is_disassembly",
        ),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="assembly_sac",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=2,
    )

    progress_callback = ConsoleProgressCallback(
        total_timesteps=args.timesteps,
        print_freq=args.log_freq,
    )

    callbacks = CallbackList(
        [
            checkpoint_callback,
            progress_callback,
        ]
    )

    import os
    import torch

    device = os.environ.get("SB3_DEVICE", "cpu")
    buffer_size = int(os.environ.get("BUFFER_SIZE", "10000"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))

    print(f"[train] torch           : {torch.__version__}", flush=True)
    print(f"[train] device demandé  : {device}", flush=True)
    print(f"[train] CUDA disponible : {torch.cuda.is_available()}", flush=True)
    print(f"[train] buffer_size     : {buffer_size}", flush=True)
    print(f"[train] batch_size      : {batch_size}", flush=True)
    print("[train] création du modèle SAC...", flush=True)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=buffer_size,
        learning_starts=args.learning_starts,
        batch_size=batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={
            "net_arch": [256, 256],
        },
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
        device=device,
    )

    print("[train] modèle SAC créé", flush=True)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,

            # Pour SAC, affiche les statistiques à chaque épisode.
            log_interval=1,

            # Désactivé car la barre Rich/TQDM est parfois mal rendue
            # dans les logs Docker.
            progress_bar=False,
        )

        final_model_path = (
            model_dir / "assembly_sac.zip"
        )

        model.save(
            str(final_model_path.with_suffix(""))
        )

        write_metadata(
            output_dir,
            xml_path=xml_path,
            timesteps_requested=args.timesteps,
            timesteps_completed=model.num_timesteps,
            seed=args.seed,
            model_path=final_model_path,
            completed=True,
        )

        print(
            f"[train] modèle final sauvegardé : "
            f"{final_model_path}",
            flush=True,
        )

    except KeyboardInterrupt:
        interrupted_model_path = (
            model_dir
            / "assembly_sac_interrupted.zip"
        )

        model.save(
            str(
                interrupted_model_path.with_suffix("")
            )
        )

        write_metadata(
            output_dir,
            xml_path=xml_path,
            timesteps_requested=args.timesteps,
            timesteps_completed=model.num_timesteps,
            seed=args.seed,
            model_path=interrupted_model_path,
            completed=False,
        )

        print(
            "\n[train] interruption reçue ; "
            f"modèle partiel sauvegardé : "
            f"{interrupted_model_path}",
            flush=True,
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
