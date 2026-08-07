from __future__ import annotations

import argparse
from dataclasses import asdict
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

from src.assembly_env import AssemblyEnv, ResidualConfig


SAC_GAMMA = 0.999


MONITOR_INFO_KEYWORDS = (
    "is_success",
    "path_progress",
    "max_path_progress",
    "progress_action",
    "progress_intention",
    "progress_scale",
    "effective_progress_request",
    "mean_progress_action",
    "mean_effective_progress_request",
    "advance_fraction",
    "hold_fraction",
    "retreat_fraction",
    "final_position_error_m",
    "final_rotation_error_rad",
    "force_norm_N",
    "torque_norm_Nm",
    "contact_impulse_Ns",
    "unsafe_contact",
    "max_force_N",
    "max_torque_Nm",
    "terminated",
    "truncated",
    "termination_reason",
    "control_mode",
    "next_control_mode",
    "contact_search_count",
    "contact_search_duration_s",
    "contact_search_fraction",
    "contact_search_latched",
    "contact_search_trigger",
    "tracking_duration_s",
    "tracking_fraction",
    "recovery_count",
    "recovery_from_contact_search_count",
    "recovery_duration_s",
    "recovery_fraction",
    "recovery_attempt_duration_s",
    "recovery_trigger",
    "recovery_trigger_contact",
    "recovery_trigger_force_N",
    "recovery_trigger_torque_Nm",
    "stuck_detected",
    "forced_retreat",
    "recovery_failed",
    "soft_effort_exceeded",
    "terminal_linear_limit_m",
    "residual_linear_offset_m",
    "residual_angular_offset_rad",
    "admittance_linear_offset_m",
    "admittance_angular_offset_rad",
    "offset_cost",
    "dense_reward",
    "terminal_reward",
    "episode_offset_cost",
    "episode_dense_reward",
    "episode_terminal_reward",
)


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
                "path_progress",
                "final_position_error_m",
                "final_rotation_error_rad",
                "force_norm_N",
                "torque_norm_Nm",
                "contact_impulse_Ns",
                "unsafe_contact",
                "contact_search_fraction",
                "recovery_fraction",
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
                    "final_position_error_m",
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
                    f" | err. finale={position_error:.4f} m"
                    f" | force={force_norm:.1f} N"
                    f" | s={self._last_info.get('path_progress', float('nan')):.2f}"
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
        "--part",
        choices=("part_1", "part_2", "part_3"),
        default=os.environ.get("ASSEMBLY_PART", "part_1"),
        help="Pièce active ; les modèles et métriques sont séparés par pièce.",
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
        default=os.environ.get("SKIP_ENV_CHECK", "0") == "1",
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
    part_name: str,
    paths_dir: Path,
) -> None:
    metadata = {
        "algorithm": "SAC",
        "task": "chandelier_residual_tactile_place",
        "control_semantics_version": 6,
        "part_name": part_name,
        "paths_dir": str(paths_dir),
        "action_space": "[vx_res, vy_res, vz_res, wx_res, wy_res, wz_res, progress_residual]",
        "progress_action_semantics": {
            "tracking": "0=nominal speed, -1=stop, +1=1.5x nominal speed",
            "contact_search": "0=0.25x nominal, negative=hold/retreat, positive<=0.5x nominal",
            "recovery": "positive=hold, zero=hold, negative=retreat",
        },
        "residual_action_authority": (
            "all six Cartesian residual actions remain active over the whole path; "
            "there is no progress-based action gate"
        ),
        "control_mode": "tracking + contact_search + bounded_recovery",
        "contact_search_latch": (
            "contact_search is activated only by simulated contact and remains "
            "latched for the episode; inertial wrench alone cannot trigger it"
        ),
        "stall_detection": (
            "final-pose stagnation is monitored from s=0.85; this progress "
            "threshold does not activate contact_search"
        ),
        "terminal_residual_limit": (
            "24 mm only after the tactile latch, in contact_search or in recovery; "
            "it is not activated by path progress"
        ),
        "admittance_activation": (
            "measured wrench is applied to admittance only during current "
            "simulated contact; free-space inertial loads do not bend the path"
        ),
        "recovery_residual_actions": (
            "all three translations and all three rotations remain available"
        ),
        "recovery_triggers": (
            "with tactile context, 5 persistent decisions above 20 N/4.5 Nm; "
            "or final-pose stagnation from s=0.85"
        ),
        "observation": (
            "8 frames x 56 values (448 total), including current and maximum "
            "progress, controller modes, tactile latch, residual offset, "
            "admittance offset and admittance velocity"
        ),
        "reward": {
            "dense": (
                "clipped to [-0.1, 0.1], including accumulated-offset cost; "
                "progress is rewarded only when exceeding the episode maximum"
            ),
            "success": 250.0,
            "unsafe": -800.0,
            "unsafe_force_and_torque": -900.0,
            "recovery_failed": -300.0,
            "time_limit": "final-pose quality penalty, bounded to -60",
        },
        "discount_factor": SAC_GAMMA,
        "time_limit_handling": (
            "700-step truncations are stored as terminal transitions; the "
            "critic does not bootstrap beyond an episode reset"
        ),
        "residual_config": asdict(ResidualConfig()),
        "cad_collision": "MuJoCo SDF using chandelier_assembly_table_collision.stl",
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
    output_dir = args.output_dir.resolve() / args.part
    paths_dir = args.paths_dir.resolve()

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
    print(f"[train] pièce           : {args.part}", flush=True)
    print(f"[train] chemins          : {paths_dir}", flush=True)
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

        test_env = AssemblyEnv(xml_path, part_name=args.part, paths_dir=paths_dir)

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
        AssemblyEnv(xml_path, part_name=args.part, paths_dir=paths_dir),
        filename=str(
            monitor_dir / "train"
        ),
        info_keywords=MONITOR_INFO_KEYWORDS,
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
    print(f"[train] gamma           : {SAC_GAMMA}", flush=True)
    print("[train] création du modèle SAC...", flush=True)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=buffer_size,
        learning_starts=args.learning_starts,
        batch_size=batch_size,
        tau=0.005,
        gamma=SAC_GAMMA,
        replay_buffer_kwargs={
            "handle_timeout_termination": False,
        },
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
            part_name=args.part,
            paths_dir=paths_dir,
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
            part_name=args.part,
            paths_dir=paths_dir,
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
