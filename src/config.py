"""Chargement des configurations d'essais, avec héritage YAML minimal."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration and return its fully resolved mapping.

    The returned dictionary never contains ``extends`` and is consequently safe
    to archive with a training run.
    """
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"La configuration doit être un mapping YAML: {path}")
    explicit_potential_reward = (
        isinstance(cfg.get("reward"), dict)
        and "potential_scale" in cfg["reward"]
    )
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = path.parent / parent
        if not parent_path.is_file():
            raise ValueError(
                f"Configuration archivée non autonome: {path} hérite de "
                f"{parent!r}, introuvable à {parent_path}. "
                "Cet essai utilise l'ancien format et doit être relancé."
            )
        parent_config = load_config(parent_path)
        if explicit_potential_reward:
            # La nouvelle formulation remplace intégralement l'ancien bloc reward.
            parent_config["reward"] = {}
        cfg = _merge(parent_config, cfg)
    required = {"case", "target_pose_fixed_to_mobile", "initial_pose_fixed_to_mobile"}
    missing = required - cfg.keys()
    if missing: raise ValueError(f"Configuration incomplète ({path}): {sorted(missing)}")
    if cfg["case"] not in {"tenon_1", "tenon_2"}: raise ValueError("case doit être tenon_1 ou tenon_2")
    required_fields = {
        "action": {"max_translation_step", "max_rotation_step_deg"},
        "admittance": {"mass", "damping", "stiffness", "max_offset", "max_velocity"},
        "reward": {
            "force_weight", "action_weight", "success_bonus", "unsafe_penalty",
        },
        "randomization": {"friction_scale"},
    }
    for section, fields in required_fields.items():
        values = cfg.get(section)
        absent = fields if not isinstance(values, dict) else fields - values.keys()
        if absent:
            raise ValueError(
                f"Configuration obsolète ou incomplète ({path}): "
                f"{section} doit définir {sorted(absent)}. Relancez un nouvel essai."
            )
    # Compatibilité des runs archivés avant l'introduction de cette composante.
    torque_weight = cfg["reward"].setdefault("torque_weight", 0.0)
    if (isinstance(torque_weight, bool)
            or not isinstance(torque_weight, (int, float))
            or torque_weight < 0 or not np.isfinite(torque_weight)):
        raise ValueError("reward.torque_weight doit être positif ou nul")
    reward = cfg["reward"]
    reward.setdefault("rotation_length_scale", 0.05)
    reward.setdefault("potential_scale", 10.0)
    reward.setdefault("potential_distance_scale", 0.010)
    reward.setdefault("step_penalty", 0.0)
    reward.setdefault("timeout_penalty", 0.0)
    for key in (
        "rotation_length_scale", "potential_scale", "potential_distance_scale",
    ):
        value = reward[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or value <= 0 or not np.isfinite(value)):
            raise ValueError(f"reward.{key} doit être strictement positif")
    for key in ("step_penalty", "timeout_penalty"):
        value = reward[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or value < 0 or not np.isfinite(value)):
            raise ValueError(f"reward.{key} doit être positif ou nul")
    for field in ("max_translation_step", "max_rotation_step_deg"):
        value = cfg["action"][field]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 < value < float("inf")):
            raise ValueError(f"action.{field} doit être strictement positif")
    action_frame = cfg["action"].setdefault("action_frame", "grasp")
    if action_frame not in {"task", "grasp"}:
        raise ValueError("action.action_frame doit être 'task' ou 'grasp'")
    control_mode = cfg["action"].setdefault("control_mode", "accumulated_reference")
    if control_mode not in {"accumulated_reference", "reactive_actual_pose"}:
        raise ValueError(
            "action.control_mode doit être 'accumulated_reference' ou "
            "'reactive_actual_pose'"
        )
    friction_range = cfg["randomization"]["friction_scale"]
    if len(friction_range) != 2 or friction_range[0] <= 0 or friction_range[0] > friction_range[1]:
        raise ValueError("randomization.friction_scale doit être [min, max] avec 0 < min <= max")
    training = cfg.setdefault("training", {})
    algorithm = training.setdefault("algorithm", "sac")
    if not isinstance(algorithm, str) or algorithm.lower() not in {"sac", "td3"}:
        raise ValueError(
            f"Unsupported RL algorithm: {algorithm}. Supported algorithms: sac, td3"
        )
    training["algorithm"] = algorithm.lower()
    training.setdefault("n_envs", 1)
    training.setdefault("base_seed", 7)
    training.setdefault("checkpoint_freq", 50_000)
    training.setdefault("ent_coef", "auto")
    training.setdefault("target_entropy", "auto")
    training.setdefault("total_timesteps", 500_000)
    training.setdefault("buffer_size", 50_000)
    training.setdefault("learning_rate", 3e-4)
    training.setdefault("gamma", 0.99)
    td3 = training.setdefault("td3", {})
    if not isinstance(td3, dict):
        raise ValueError("training.td3 doit être un mapping")
    td3.setdefault("action_noise_std", 0.1)
    td3.setdefault("policy_delay", 2)
    td3.setdefault("target_policy_noise", 0.2)
    td3.setdefault("target_noise_clip", 0.5)
    for key in ("n_envs", "checkpoint_freq", "total_timesteps", "buffer_size"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} doit être un entier strictement positif")
    learning_rate = training["learning_rate"]
    if (isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not 0 < learning_rate < float("inf")):
        raise ValueError("training.learning_rate doit être strictement positif")
    gamma = training["gamma"]
    if (isinstance(gamma, bool) or not isinstance(gamma, (int, float))
            or not 0 < gamma <= 1 or not np.isfinite(gamma)):
        raise ValueError("training.gamma doit être dans ]0, 1]")
    for key in ("action_noise_std", "target_policy_noise", "target_noise_clip"):
        value = td3[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or value < 0 or not np.isfinite(value)):
            raise ValueError(f"training.td3.{key} doit être positif ou nul")
    policy_delay = td3["policy_delay"]
    if isinstance(policy_delay, bool) or not isinstance(policy_delay, int) or policy_delay <= 0:
        raise ValueError("training.td3.policy_delay doit être un entier strictement positif")
    base_seed = training["base_seed"]
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("training.base_seed doit être un entier positif ou nul")
    observation = cfg.setdefault("observation", {})
    observation.setdefault("include_admittance_position", False)
    if not isinstance(observation["include_admittance_position"], bool):
        raise ValueError("observation.include_admittance_position doit être un booléen")
    evaluation = cfg.setdefault("evaluation", {})
    evaluation.setdefault("enabled", False)
    evaluation.setdefault("eval_freq", 25_000)
    evaluation.setdefault("n_eval_episodes", 1)
    evaluation.setdefault("deterministic", True)
    evaluation.setdefault("seed", 10_007)
    for key in ("enabled", "deterministic"):
        if not isinstance(evaluation[key], bool):
            raise ValueError(f"evaluation.{key} doit être un booléen")
    for key in ("eval_freq", "n_eval_episodes"):
        value = evaluation[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"evaluation.{key} doit être un entier strictement positif")
    eval_seed = evaluation["seed"]
    if isinstance(eval_seed, bool) or not isinstance(eval_seed, int) or eval_seed < 0:
        raise ValueError("evaluation.seed doit être un entier positif ou nul")
    return cfg


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    """Archive one self-contained, human-readable configuration YAML."""
    if "extends" in config:
        raise ValueError("Une configuration résolue ne doit pas contenir 'extends'")
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
