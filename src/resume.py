"""Application sûre et vérifiable de la configuration YAML lors d'une reprise."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit


RUNTIME_OVERRIDABLE = (
    "learning_rate", "gamma", "tau", "batch_size", "train_freq",
    "gradient_steps", "learning_starts", "target_update_interval",
    "target_entropy",
)
ENVIRONMENT_OVERRIDABLE = (
    "reward", "success", "simulation", "admittance", "action",
    "randomization", "perception", "observation", "curriculum",
)
REPLAY_BUFFER_SENSITIVE = (
    "reward", "success", "simulation", "admittance", "action",
    "randomization", "perception", "observation",
)
STRUCTURAL_INCOMPATIBLE = (
    "training.algorithm", "training.network", "observation_space",
    "action_space", "policy architecture",
)
OPERATIONAL_PATHS = {
    "training.total_timesteps", "training.checkpoint_freq", "training.base_seed",
    "training.n_envs", "evaluation", "resume",
}
AUTO_INVALIDATING_SECTIONS = {"reward", "success", "action", "observation"}


@dataclass(frozen=True)
class ResumeResult:
    replay_action: str
    changes: tuple[str, ...]
    sensitive_changes: tuple[str, ...]
    warnings: tuple[str, ...]


def next_future_curriculum_update(current_timesteps: int, update_interval: int) -> int:
    """Première échéance strictement postérieure au timestep courant."""
    if current_timesteps < 0 or update_interval <= 0:
        raise ValueError("timesteps doit être >= 0 et update_interval > 0")
    return (current_timesteps // update_interval + 1) * update_interval


def _training_value(training: dict[str, Any], key: str) -> Any:
    defaults = {
        "learning_rate": 3e-4, "gamma": .99, "tau": .005,
        "batch_size": 256, "train_freq": [1, "step"], "gradient_steps": -1,
        "learning_starts": 5_000, "target_update_interval": 1,
        "target_entropy": "auto", "network": [256, 256],
    }
    return training.get(key, defaults[key])


def _normalized_train_freq(value: Any) -> TrainFreq:
    if isinstance(value, TrainFreq):
        return value
    if isinstance(value, int):
        value = (value, "step")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("training.train_freq doit être un entier ou [fréquence, unité]")
    frequency, unit = value
    return TrainFreq(int(frequency), TrainFrequencyUnit(str(unit)))


def _optimizer_lrs(model: Any) -> dict[str, float]:
    optimizers = {
        "actor": getattr(getattr(model, "actor", None), "optimizer", None),
        "critic": getattr(getattr(model, "critic", None), "optimizer", None),
        "entropy": getattr(model, "ent_coef_optimizer", None),
    }
    return {
        name: float(optimizer.param_groups[0]["lr"])
        for name, optimizer in optimizers.items() if optimizer is not None
    }


def _set_learning_rate(model: Any, value: float) -> None:
    model.learning_rate = value
    model.lr_schedule = lambda _: value
    for optimizer in (
        getattr(getattr(model, "actor", None), "optimizer", None),
        getattr(getattr(model, "critic", None), "optimizer", None),
        getattr(model, "ent_coef_optimizer", None),
    ):
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = value


def _spaces_equal(checkpoint_space: Any, current_space: Any) -> bool:
    return (
        type(checkpoint_space) is type(current_space)
        and checkpoint_space.shape == current_space.shape
        and np.allclose(checkpoint_space.low, current_space.low)
        and np.allclose(checkpoint_space.high, current_space.high)
    )


def _changed_sections(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    if previous is None:
        return []
    return [
        section for section in ENVIRONMENT_OVERRIDABLE
        if previous.get(section) != current.get(section)
    ]


def _leaf_changes(old: Any, new: Any, prefix: str) -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        paths = []
        for key in sorted(old.keys() | new.keys()):
            paths.extend(_leaf_changes(old.get(key), new.get(key), f"{prefix}.{key}"))
        return paths
    return [] if old == new else [f"{prefix}: {old!r} -> {new!r}"]


def apply_resume_configuration(
    model: Any, env: Any, current_config: dict[str, Any], *,
    previous_config: dict[str, Any] | None = None,
) -> ResumeResult:
    """Valide la structure, applique les overrides et décide du replay."""
    training = current_config["training"]
    resume = current_config.get("resume", {})
    if training["algorithm"] != ("sac" if isinstance(model, SAC) else "td3"):
        raise ValueError("structural parameter changed during resume: training.algorithm")
    if not _spaces_equal(model.observation_space, env.observation_space):
        raise ValueError(
            "structural parameter changed during resume: observation_space; "
            f"checkpoint shape={model.observation_space.shape}, "
            f"current environment shape={env.observation_space.shape}"
        )
    if not _spaces_equal(model.action_space, env.action_space):
        raise ValueError(
            "structural parameter changed during resume: action_space; "
            f"checkpoint shape={model.action_space.shape}, "
            f"current environment shape={env.action_space.shape}"
        )
    configured_network = list(_training_value(training, "network"))
    effective_network = list(model.policy.net_arch)
    if configured_network != effective_network:
        raise ValueError(
            "structural parameter changed during resume: training.network "
            f"{effective_network} -> {configured_network}"
        )

    changes: list[str] = []
    checkpoint = {
        key: getattr(model, key) for key in RUNTIME_OVERRIDABLE
        if key != "target_entropy" or isinstance(model, SAC)
    }
    configured_ent_coef = training.get("ent_coef", "auto")
    checkpoint_auto_entropy = getattr(model, "ent_coef_optimizer", None) is not None
    configured_auto_entropy = isinstance(configured_ent_coef, str) and configured_ent_coef.startswith("auto")
    if isinstance(model, SAC) and checkpoint_auto_entropy != configured_auto_entropy:
        raise ValueError("structural parameter changed during resume: training.ent_coef auto/fixed")
    previous_ent = None if previous_config is None else previous_config.get("training", {}).get("ent_coef", "auto")
    if isinstance(model, SAC) and previous_ent is not None and previous_ent != configured_ent_coef:
        raise ValueError("training.ent_coef cannot be changed safely during resume")

    learning_rate = float(_training_value(training, "learning_rate"))
    _set_learning_rate(model, learning_rate)
    for key in RUNTIME_OVERRIDABLE:
        if key == "learning_rate" or (key == "target_entropy" and not isinstance(model, SAC)):
            continue
        value = _training_value(training, key)
        if key == "train_freq":
            value = _normalized_train_freq(value)
        elif key == "target_entropy" and value == "auto":
            value = -float(np.prod(model.action_space.shape))
        setattr(model, key, value)
    for key, old in checkpoint.items():
        new = getattr(model, key)
        if old != new:
            changes.append(f"training.{key}: {old} -> {new} [APPLIED]")

    environment_changes = _changed_sections(previous_config, current_config)
    sensitive = [section for section in environment_changes if section in REPLAY_BUFFER_SENSITIVE]
    for section in environment_changes:
        leaf_changes = _leaf_changes(
            previous_config.get(section), current_config.get(section), section,
        ) if previous_config is not None else [section]
        changes.extend(f"{change} [ENVIRONMENT UPDATED]" for change in leaf_changes)
    policy = resume.get("replay_buffer_policy", "auto")
    old_buffer_size = int(getattr(model, "buffer_size", 0))
    new_buffer_size = int(training["buffer_size"])
    size_changed = old_buffer_size != new_buffer_size
    if size_changed:
        changes.append(f"training.buffer_size: {old_buffer_size} -> {new_buffer_size} [REBUILT]")
    invalidating = [section for section in sensitive if section in AUTO_INVALIDATING_SECTIONS]
    if "perception" in sensitive:
        invalidating.append("perception")
    if (previous_config is not None and
            previous_config.get("simulation", {}).get("max_episode_steps")
            != current_config.get("simulation", {}).get("max_episode_steps")):
        invalidating.append("simulation.max_episode_steps")
    messages = []
    if sensitive:
        messages.append("Replay-buffer-sensitive YAML changes: " + ", ".join(sensitive))
    if policy == "error" and (sensitive or size_changed):
        raise ValueError("replay buffer incompatible with current YAML: " + ", ".join(sensitive))
    if policy == "discard" or size_changed or (policy == "auto" and invalidating):
        replay_action = "discard"
    else:
        replay_action = "keep"
    if policy == "keep" and size_changed:
        raise ValueError("replay_buffer_policy=keep incompatible avec un changement de buffer_size")
    model.buffer_size = new_buffer_size
    validate_effective_resume_configuration(model, current_config)
    for message in messages:
        warnings.warn(message, RuntimeWarning)
    return ResumeResult(replay_action, tuple(changes), tuple(sensitive), tuple(messages))


def validate_effective_resume_configuration(model: Any, config: dict[str, Any]) -> None:
    """Lit les objets SB3 effectifs et refuse toute divergence."""
    training = config["training"]
    expected = {
        "gamma": float(_training_value(training, "gamma")),
        "tau": float(_training_value(training, "tau")),
        "batch_size": int(_training_value(training, "batch_size")),
        "gradient_steps": int(_training_value(training, "gradient_steps")),
        "learning_starts": int(_training_value(training, "learning_starts")),
        "target_update_interval": int(_training_value(training, "target_update_interval")),
    }
    for key, value in expected.items():
        if getattr(model, key) != value:
            raise RuntimeError(f"Resume override ineffective: {key}")
    expected_freq = _normalized_train_freq(_training_value(training, "train_freq"))
    if model.train_freq != expected_freq:
        raise RuntimeError("Resume override ineffective: train_freq")
    lr = float(_training_value(training, "learning_rate"))
    if model.learning_rate != lr or model.lr_schedule(1.0) != lr:
        raise RuntimeError("Resume override ineffective: learning_rate schedule")
    for name, effective in _optimizer_lrs(model).items():
        if not np.isclose(effective, lr):
            raise RuntimeError(f"Resume override ineffective: {name} optimizer lr")


def rebuild_empty_replay_buffer(model: Any, buffer_size: int) -> None:
    """Reconstruit réellement un buffer vide sans toucher aux réseaux."""
    model.replay_buffer = model.replay_buffer_class(
        buffer_size,
        model.observation_space,
        model.action_space,
        device=model.device,
        n_envs=model.n_envs,
        optimize_memory_usage=model.optimize_memory_usage,
        **model.replay_buffer_kwargs,
    )
    model.buffer_size = buffer_size


def effective_resume_summary(model: Any) -> dict[str, Any]:
    return {
        "learning_rates": _optimizer_lrs(model), "gamma": model.gamma,
        "tau": model.tau, "batch_size": model.batch_size,
        "train_freq": model.train_freq, "gradient_steps": model.gradient_steps,
        "learning_starts": model.learning_starts,
        "target_update_interval": model.target_update_interval,
        "replay_buffer_capacity": getattr(getattr(model, "replay_buffer", None), "buffer_size", None),
    }
