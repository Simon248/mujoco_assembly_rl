"""Application sûre et vérifiable de la configuration YAML lors d'une reprise."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit


COMMON_RUNTIME_OVERRIDABLE = (
    "learning_rate", "gamma", "tau", "batch_size", "train_freq",
    "gradient_steps", "learning_starts", )
SAC_RUNTIME_OVERRIDABLE = (
    "target_update_interval", "target_entropy",
)
TD3_RUNTIME_OVERRIDABLE = (
    "policy_delay", "target_policy_noise", "target_noise_clip",
)
ENVIRONMENT_OVERRIDABLE = (
    "case", "target_pose_fixed_to_mobile", "initial_pose_fixed_to_mobile",
    "reward", "success", "simulation", "admittance", "action",
    "randomization", "perception", "observation", "curriculum",
)
REPLAY_BUFFER_SENSITIVE = (
    "case", "target_pose_fixed_to_mobile", "initial_pose_fixed_to_mobile",
    "reward", "success", "simulation", "admittance", "action",
    "randomization", "perception", "observation",
)


@dataclass(frozen=True)
class ResumeResult:
    replay_action: str
    changes: tuple[str, ...]
    sensitive_changes: tuple[str, ...]
    warnings: tuple[str, ...]
    curriculum_incompatible_changes: tuple[str, ...]
    semantic_compatibility_known: bool
    checkpoint_values: dict[str, Any]
    checkpoint_buffer_size: int
    requested_buffer_size: int


@dataclass(frozen=True)
class ReplayResumeReport:
    action: str
    checkpoint_requested_total: int
    old_internal_size: int
    old_n_envs: int
    old_effective_capacity: int
    requested_total: int
    expected_internal_size: int
    actual_internal_size: int
    n_envs: int
    effective_capacity: int
    transitions_preserved: int
    transitions_discarded: int
    source_loaded: bool


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
        "n_envs": 1, "optimize_memory_usage": False,
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


def expected_replay_buffer_internal_size(
    requested_total: int, n_envs: int,
) -> int:
    """Mirror exactly ReplayBuffer.__init__ in Stable-Baselines3 2.9.0."""
    if requested_total <= 0 or n_envs <= 0:
        raise ValueError("requested_total et n_envs doivent être strictement positifs")
    return max(int(requested_total) // int(n_envs), 1)


def replay_buffer_transition_count(replay_buffer: Any) -> int:
    """Number of transitions stored, not the number of vectorized time slots."""
    return int(replay_buffer.size()) * int(replay_buffer.n_envs)


def devices_are_compatible(requested: Any, effective: Any) -> bool:
    """Treat an unspecified CUDA index as the runtime-selected CUDA device."""
    requested_device = torch.device(requested)
    effective_device = torch.device(effective)
    return (
        requested_device.type == effective_device.type
        and (requested_device.index is None
             or requested_device.index == effective_device.index)
    )


def _canonical_ent_coef(value: Any) -> str | float:
    if isinstance(value, str):
        if not value.startswith("auto"):
            raise ValueError(f"ent_coef SAC invalide: {value!r}")
        return value
    return float(value)


def _effective_target_entropy(model: Any, configured: Any) -> float:
    if configured == "auto":
        return -float(np.prod(model.action_space.shape))
    return float(configured)


def _td3_noise_std(model: Any) -> float | None:
    noise = getattr(model, "action_noise", None)
    sigma = getattr(noise, "_sigma", None)
    if sigma is None:
        return None
    values = np.asarray(sigma, dtype=float)
    if not values.size or not np.allclose(values, values.flat[0]):
        return None
    return float(values.flat[0])


def _optimizer_lrs(model: Any) -> dict[str, tuple[float, ...]]:
    optimizers = {
        "actor": getattr(getattr(model, "actor", None), "optimizer", None),
        "critic": getattr(getattr(model, "critic", None), "optimizer", None),
        "entropy": getattr(model, "ent_coef_optimizer", None),
    }
    return {
        name: tuple(float(group["lr"]) for group in optimizer.param_groups)
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
    if not resume.get("apply_current_yaml", True):
        raise ValueError(
            "resume.apply_current_yaml=false n'est pas supporté pendant une "
            "reprise: l'environnement vient nécessairement du YAML courant"
        )
    if not resume.get("fail_on_structural_change", True):
        raise ValueError(
            "resume.fail_on_structural_change=false n'est pas supporté pendant "
            "une reprise: les incompatibilités structurelles ne peuvent pas "
            "être ignorées sûrement"
        )
    if isinstance(model, SAC):
        algorithm = "sac"
    elif isinstance(model, TD3):
        algorithm = "td3"
    else:
        raise ValueError(
            f"Algorithme de checkpoint non supporté: {type(model).__name__}"
        )
    if training["algorithm"] != algorithm:
        raise ValueError("structural parameter changed during resume: training.algorithm")
    requested_n_envs = int(_training_value(training, "n_envs"))
    checkpoint_n_envs = int(model.n_envs)
    environment_n_envs = int(env.num_envs)
    if requested_n_envs != checkpoint_n_envs or environment_n_envs != checkpoint_n_envs:
        raise ValueError(
            "structural parameter changed during resume: training.n_envs; "
            f"checkpoint={checkpoint_n_envs}, requested={requested_n_envs}, "
            f"environment={environment_n_envs}. Changer n_envs rend la disposition "
            "du replay buffer incompatible."
        )
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

    requested_optimization = bool(
        _training_value(training, "optimize_memory_usage")
    )
    if requested_optimization != model.optimize_memory_usage:
        raise ValueError(
            "training.optimize_memory_usage cannot be changed safely during "
            "resume; it changes the replay-buffer storage layout"
        )
    if model.seed is None:
        raise ValueError(
            "Checkpoint sans seed SB3: la continuité des flux RNG ne peut pas "
            "être validée"
        )
    requested_seed = int(training.get("base_seed", model.seed))
    previous_seed = (
        int(previous_config["training"].get("base_seed", requested_seed))
        if previous_config is not None else int(model.seed)
    )
    if requested_seed != previous_seed or requested_seed != int(model.seed):
        raise ValueError(
            "training.base_seed cannot be changed safely during resume; "
            f"checkpoint/model={model.seed}, previous YAML={previous_seed}, "
            f"requested={requested_seed}. Restarting all SB3, environment and "
            "curriculum RNG streams is not equivalent to a resume."
        )
    if bool(getattr(model, "use_sde", False)):
        raise ValueError(
            "Checkpoint uses gSDE but this project does not expose a resume-safe "
            "gSDE configuration"
        )
    if int(getattr(model, "n_steps", 1)) != 1:
        raise ValueError(
            "Checkpoint uses n-step replay but this project only supports n_steps=1"
        )

    changes: list[str] = []
    checkpoint: dict[str, Any] = {
        key: getattr(model, key) for key in COMMON_RUNTIME_OVERRIDABLE
    }
    if algorithm == "sac":
        checkpoint.update({
            key: getattr(model, key) for key in SAC_RUNTIME_OVERRIDABLE
        })
        configured_ent_coef = _canonical_ent_coef(
            training.get("ent_coef", "auto")
        )
        checkpoint_ent_coef = _canonical_ent_coef(model.ent_coef)
        if checkpoint_ent_coef != configured_ent_coef:
            raise ValueError(
                "training.ent_coef cannot be changed safely during resume; "
                f"checkpoint={checkpoint_ent_coef!r}, "
                f"requested={configured_ent_coef!r}"
            )
        checkpoint["ent_coef"] = checkpoint_ent_coef
    else:
        td3 = training["td3"]
        checkpoint.update({
            key: getattr(model, key) for key in TD3_RUNTIME_OVERRIDABLE
        })
        checkpoint["action_noise_std"] = _td3_noise_std(model)

    learning_rate = float(_training_value(training, "learning_rate"))
    _set_learning_rate(model, learning_rate)
    for key in COMMON_RUNTIME_OVERRIDABLE:
        if key == "learning_rate":
            continue
        value = _training_value(training, key)
        if key == "train_freq":
            value = _normalized_train_freq(value)
        setattr(model, key, value)
    if algorithm == "sac":
        model.target_update_interval = int(
            _training_value(training, "target_update_interval")
        )
        model.target_entropy = _effective_target_entropy(
            model, _training_value(training, "target_entropy")
        )
    else:
        td3 = training["td3"]
        model.policy_delay = int(td3["policy_delay"])
        model.target_policy_noise = float(td3["target_policy_noise"])
        model.target_noise_clip = float(td3["target_noise_clip"])
        noise_std = float(td3["action_noise_std"])
        action_dim = int(model.action_space.shape[-1])
        model.action_noise = NormalActionNoise(
            mean=np.zeros(action_dim), sigma=noise_std * np.ones(action_dim),
        )
    for key, old in checkpoint.items():
        if key == "action_noise_std":
            new = _td3_noise_std(model)
        elif key == "ent_coef":
            new = _canonical_ent_coef(model.ent_coef)
        else:
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
    # Every listed section changes rewards, transitions, terminal flags,
    # observations or the task itself. Keeping old tuples would mix MDPs.
    invalidating = list(sensitive)
    messages = []
    if sensitive:
        messages.append("Replay-buffer-sensitive YAML changes: " + ", ".join(sensitive))
    if previous_config is None:
        messages.append(
            "Previous resolved config unavailable; semantic replay compatibility "
            "cannot be established"
        )
        if policy in {"error", "keep"}:
            raise ValueError(
                f"replay_buffer_policy={policy} et config source introuvable: "
                "la compatibilité sémantique ne peut pas être prouvée"
            )
    if policy == "error" and (sensitive or size_changed):
        incompatibilities = list(sensitive)
        if size_changed:
            incompatibilities.append("buffer_size")
        raise ValueError(
            "replay buffer incompatible with current YAML: "
            + ", ".join(incompatibilities)
        )
    if (policy == "discard" or size_changed
            or (policy == "auto" and (invalidating or previous_config is None))):
        replay_action = "discard"
    else:
        replay_action = "keep"
    if policy == "keep" and (size_changed or sensitive):
        reasons = []
        if size_changed:
            reasons.append("buffer_size")
        reasons.extend(sensitive)
        raise ValueError(
            "replay_buffer_policy=keep incompatible avec: "
            + ", ".join(reasons)
        )
    model.buffer_size = new_buffer_size
    validate_effective_resume_configuration(
        model, current_config, validate_replay_buffer=False,
    )
    for message in messages:
        warnings.warn(message, RuntimeWarning)
    return ResumeResult(
        replay_action, tuple(changes), tuple(sensitive), tuple(messages),
        curriculum_incompatible_changes=tuple(sensitive),
        semantic_compatibility_known=previous_config is not None,
        checkpoint_values=checkpoint,
        checkpoint_buffer_size=old_buffer_size,
        requested_buffer_size=new_buffer_size,
    )


def validate_replay_buffer_capacity(
    model: Any, requested_total: int,
) -> None:
    """Validate model-level and vectorized ReplayBuffer capacity semantics."""
    replay_buffer = getattr(model, "replay_buffer", None)
    if replay_buffer is None:
        raise RuntimeError("Resume override ineffective: replay_buffer absent")
    model_n_envs = int(model.n_envs)
    replay_n_envs = int(replay_buffer.n_envs)
    expected_internal = expected_replay_buffer_internal_size(
        int(requested_total), model_n_envs,
    )
    actual_internal = int(replay_buffer.buffer_size)
    model_level = int(model.buffer_size)
    effective_capacity = actual_internal * replay_n_envs
    if (model_level != int(requested_total)
            or replay_n_envs != model_n_envs
            or actual_internal != expected_internal):
        raise RuntimeError(
            "Resume override ineffective: replay_buffer capacity\n"
            f"  requested total:          {int(requested_total)}\n"
            f"  model-level value:        {model_level}\n"
            f"  model n_envs:             {model_n_envs}\n"
            f"  replay n_envs:            {replay_n_envs}\n"
            f"  expected internal:        {expected_internal}\n"
            f"  replay internal:          {actual_internal}\n"
            f"  effective replay capacity:{effective_capacity}"
        )
    if bool(replay_buffer.optimize_memory_usage) != bool(
        model.optimize_memory_usage
    ):
        raise RuntimeError(
            "Resume override ineffective: replay_buffer.optimize_memory_usage; "
            f"model={model.optimize_memory_usage}, "
            f"replay={replay_buffer.optimize_memory_usage}"
        )
    if not devices_are_compatible(model.device, replay_buffer.device):
        raise RuntimeError(
            "Resume replay buffer device incompatible with model; "
            f"model={model.device}, replay={replay_buffer.device}"
        )
    if not isinstance(replay_buffer, model.replay_buffer_class):
        raise RuntimeError(
            "Resume replay buffer class incompatible with model; "
            f"expected={model.replay_buffer_class.__name__}, "
            f"effective={type(replay_buffer).__name__}"
        )
    if not _spaces_equal(replay_buffer.observation_space, model.observation_space):
        raise RuntimeError(
            "Resume replay buffer observation_space incompatible with model"
        )
    if not _spaces_equal(replay_buffer.action_space, model.action_space):
        raise RuntimeError("Resume replay buffer action_space incompatible with model")


def validate_effective_resume_configuration(
    model: Any, config: dict[str, Any], *,
    validate_replay_buffer: bool = True,
) -> None:
    """Read the exact SB3 runtime objects and reject every divergence."""
    training = config["training"]
    expected = {
        "gamma": float(_training_value(training, "gamma")),
        "tau": float(_training_value(training, "tau")),
        "batch_size": int(_training_value(training, "batch_size")),
        "gradient_steps": int(_training_value(training, "gradient_steps")),
        "learning_starts": int(_training_value(training, "learning_starts")),
    }
    if isinstance(model, SAC):
        expected["target_update_interval"] = int(
            _training_value(training, "target_update_interval")
        )
    elif isinstance(model, TD3):
        td3 = training["td3"]
        expected.update({
            "policy_delay": int(td3["policy_delay"]),
            "target_policy_noise": float(td3["target_policy_noise"]),
            "target_noise_clip": float(td3["target_noise_clip"]),
        })
    for key, value in expected.items():
        effective = getattr(model, key)
        if (isinstance(value, float) and not np.isclose(effective, value)
                or not isinstance(value, float) and effective != value):
            raise RuntimeError(
                f"Resume override ineffective: {key}; "
                f"requested={value!r}, effective={effective!r}"
            )
    expected_freq = _normalized_train_freq(_training_value(training, "train_freq"))
    if model.train_freq != expected_freq:
        raise RuntimeError(
            "Resume override ineffective: train_freq; "
            f"requested={expected_freq!r}, effective={model.train_freq!r}"
        )
    learning_rate = float(_training_value(training, "learning_rate"))
    schedule_values = tuple(
        float(model.lr_schedule(progress)) for progress in (1.0, .5, 0.0)
    )
    if (not np.isclose(float(model.learning_rate), learning_rate)
            or any(not np.isclose(value, learning_rate) for value in schedule_values)):
        raise RuntimeError(
            "Resume override ineffective: learning_rate schedule; "
            f"requested={learning_rate}, model={model.learning_rate}, "
            f"schedule={schedule_values}"
        )
    for name, group_rates in _optimizer_lrs(model).items():
        if any(not np.isclose(rate, learning_rate) for rate in group_rates):
            raise RuntimeError(
                f"Resume override ineffective: {name} optimizer lr; "
                f"requested={learning_rate}, effective={group_rates}"
            )
    requested_n_envs = int(_training_value(training, "n_envs"))
    if int(model.n_envs) != requested_n_envs:
        raise RuntimeError(
            "Resume override ineffective: n_envs; "
            f"requested={requested_n_envs}, effective={model.n_envs}"
        )
    configured_network = list(_training_value(training, "network"))
    if list(model.policy.net_arch) != configured_network:
        raise RuntimeError(
            "Resume override ineffective: policy_kwargs.net_arch; "
            f"requested={configured_network}, effective={model.policy.net_arch}"
        )
    if bool(getattr(model, "use_sde", False)):
        raise RuntimeError("Resume effective configuration unexpectedly enables gSDE")
    if int(getattr(model, "n_steps", 1)) != 1:
        raise RuntimeError("Resume effective configuration unexpectedly uses n_steps != 1")
    parameter_devices = {parameter.device for parameter in model.policy.parameters()}
    incompatible_devices = {
        device for device in parameter_devices
        if not devices_are_compatible(model.device, device)
    }
    if incompatible_devices:
        raise RuntimeError(
            "Resume model parameters are not all on the configured device; "
            f"model={model.device}, "
            f"parameters={sorted(map(str, parameter_devices))}"
        )
    requested_memory_optimization = bool(
        _training_value(training, "optimize_memory_usage")
    )
    if bool(model.optimize_memory_usage) != requested_memory_optimization:
        raise RuntimeError(
            "Resume override ineffective: optimize_memory_usage; "
            f"requested={requested_memory_optimization}, "
            f"effective={model.optimize_memory_usage}"
        )
    if isinstance(model, SAC):
        if model.action_noise is not None:
            raise RuntimeError(
                "Resume effective SAC configuration unexpectedly has action_noise"
            )
        configured_ent_coef = _canonical_ent_coef(training.get("ent_coef", "auto"))
        if _canonical_ent_coef(model.ent_coef) != configured_ent_coef:
            raise RuntimeError("Resume override ineffective: ent_coef specification")
        entropy_is_auto = isinstance(configured_ent_coef, str)
        if (getattr(model, "ent_coef_optimizer", None) is not None) != entropy_is_auto:
            raise RuntimeError("Resume override ineffective: ent_coef optimizer mode")
        expected_entropy = _effective_target_entropy(
            model, _training_value(training, "target_entropy")
        )
        if not np.isclose(float(model.target_entropy), expected_entropy):
            raise RuntimeError(
                "Resume override ineffective: target_entropy; "
                f"requested={expected_entropy}, effective={model.target_entropy}"
            )
        if not entropy_is_auto:
            tensor_value = float(model.ent_coef_tensor.detach().cpu().item())
            if not np.isclose(tensor_value, float(configured_ent_coef)):
                raise RuntimeError(
                    "Resume override ineffective: fixed ent_coef tensor; "
                    f"requested={configured_ent_coef}, effective={tensor_value}"
                )
    else:
        expected_noise = float(training["td3"]["action_noise_std"])
        effective_noise = _td3_noise_std(model)
        if effective_noise is None or not np.isclose(effective_noise, expected_noise):
            raise RuntimeError(
                "Resume override ineffective: TD3 action_noise_std; "
                f"requested={expected_noise}, effective={effective_noise}"
            )
    if validate_replay_buffer:
        validate_replay_buffer_capacity(model, int(training["buffer_size"]))


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


def prepare_resume_replay_buffer(
    model: Any, replay_path: Path, replay_action: str, *,
    requested_total: int, checkpoint_requested_total: int,
) -> ReplayResumeReport:
    """Load or explicitly discard a coordinated replay, then validate layout."""
    if replay_action not in {"keep", "discard"}:
        raise ValueError(f"Action replay inconnue: {replay_action!r}")
    source_loaded = replay_path.is_file()
    if replay_action == "keep" and not source_loaded:
        raise FileNotFoundError(
            "Replay buffer à conserver mais fichier coordonné introuvable: "
            f"{replay_path}"
        )
    if source_loaded:
        model.load_replay_buffer(replay_path)
        old_buffer = model.replay_buffer
        old_internal_size = int(old_buffer.buffer_size)
        old_n_envs = int(old_buffer.n_envs)
        old_effective_capacity = old_internal_size * old_n_envs
        old_transitions = replay_buffer_transition_count(old_buffer)
    else:
        old_internal_size = expected_replay_buffer_internal_size(
            checkpoint_requested_total, int(model.n_envs),
        )
        old_n_envs = int(model.n_envs)
        old_effective_capacity = old_internal_size * old_n_envs
        old_transitions = 0
    if replay_action == "discard":
        rebuild_empty_replay_buffer(model, requested_total)
        transitions_preserved = 0
        transitions_discarded = old_transitions
    else:
        model.buffer_size = int(requested_total)
        transitions_preserved = old_transitions
        transitions_discarded = 0
    validate_replay_buffer_capacity(model, requested_total)
    replay_buffer = model.replay_buffer
    actual_internal = int(replay_buffer.buffer_size)
    n_envs = int(replay_buffer.n_envs)
    return ReplayResumeReport(
        action=replay_action,
        checkpoint_requested_total=int(checkpoint_requested_total),
        old_internal_size=old_internal_size,
        old_n_envs=old_n_envs,
        old_effective_capacity=old_effective_capacity,
        requested_total=int(requested_total),
        expected_internal_size=expected_replay_buffer_internal_size(
            requested_total, int(model.n_envs),
        ),
        actual_internal_size=actual_internal,
        n_envs=n_envs,
        effective_capacity=actual_internal * n_envs,
        transitions_preserved=transitions_preserved,
        transitions_discarded=transitions_discarded,
        source_loaded=source_loaded,
    )


def effective_resume_summary(
    model: Any, config: dict[str, Any], result: ResumeResult,
    replay: ReplayResumeReport,
) -> dict[str, dict[str, Any]]:
    """One launch-time audit: checkpoint, requested, and effective values."""
    training = config["training"]
    checkpoint = result.checkpoint_values
    summary: dict[str, dict[str, Any]] = {}
    for key in COMMON_RUNTIME_OVERRIDABLE:
        requested = _training_value(training, key)
        if key == "train_freq":
            requested = _normalized_train_freq(requested)
        effective = (
            tuple(model.lr_schedule(progress) for progress in (1.0, .5, 0.0))
            if key == "learning_rate" else getattr(model, key)
        )
        summary[key] = {
            "checkpoint/model": checkpoint[key],
            "requested": requested,
            "effective": effective,
            "status": "OK",
        }
    algorithm_keys = (
        SAC_RUNTIME_OVERRIDABLE if isinstance(model, SAC)
        else TD3_RUNTIME_OVERRIDABLE
    )
    for key in algorithm_keys:
        requested = (
            _training_value(training, key) if isinstance(model, SAC)
            else training["td3"][key]
        )
        if key == "target_entropy":
            requested = _effective_target_entropy(model, requested)
        summary[key] = {
            "checkpoint/model": checkpoint[key],
            "requested": requested,
            "effective": getattr(model, key),
            "status": "OK",
        }
    if isinstance(model, SAC):
        summary["ent_coef"] = {
            "checkpoint/model": checkpoint["ent_coef"],
            "requested": training["ent_coef"],
            "effective": (
                "learned/optimizer-preserved"
                if model.ent_coef_optimizer is not None
                else float(model.ent_coef_tensor.detach().cpu().item())
            ),
            "status": "OK",
        }
    else:
        summary["action_noise_std"] = {
            "checkpoint/model": checkpoint["action_noise_std"],
            "requested": training["td3"]["action_noise_std"],
            "effective": _td3_noise_std(model),
            "status": "OK",
        }
    summary["buffer_size"] = {
        "checkpoint requested total": replay.checkpoint_requested_total,
        "checkpoint internal": replay.old_internal_size,
        "checkpoint n_envs": replay.old_n_envs,
        "checkpoint effective capacity": replay.old_effective_capacity,
        "requested total": replay.requested_total,
        "model-level value": model.buffer_size,
        "n_envs": replay.n_envs,
        "expected internal": replay.expected_internal_size,
        "replay internal": replay.actual_internal_size,
        "effective capacity": replay.effective_capacity,
        "preserved transitions": replay.transitions_preserved,
        "discarded transitions": replay.transitions_discarded,
        "source replay loaded": replay.source_loaded,
        "status": "OK",
    }
    summary["runtime_structure"] = {
        "algorithm": training["algorithm"],
        "n_envs": model.n_envs,
        "network": list(model.policy.net_arch),
        "device": str(model.device),
        "model seed (checkpoint)": model.seed,
        "environment base_seed": training["base_seed"],
        "optimize_memory_usage": model.optimize_memory_usage,
        "status": "OK",
    }
    summary["optimizer_learning_rates"] = {
        **_optimizer_lrs(model), "status": "OK",
    }
    return summary
