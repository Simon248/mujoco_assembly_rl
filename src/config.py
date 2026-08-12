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
    explicit_reward_replacement = (
        isinstance(cfg.get("reward"), dict)
        and "rotation_length_scale" in cfg["reward"]
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
        if explicit_reward_replacement:
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
    reward.setdefault("pose_weight", 50.0)
    reward.setdefault("step_penalty", 0.0)
    reward.setdefault("timeout_penalty", 0.0)
    for key in (
        "rotation_length_scale", "pose_weight",
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
    training.setdefault("tau", 0.005)
    training.setdefault("batch_size", 256)
    training.setdefault("train_freq", [1, "step"])
    training.setdefault("gradient_steps", -1)
    training.setdefault("learning_starts", 5_000)
    training.setdefault("target_update_interval", 1)
    training.setdefault("network", [256, 256])
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
    if not 0 < float(training["tau"]) <= 1:
        raise ValueError("training.tau doit être dans ]0, 1]")
    for key in ("batch_size", "learning_starts", "target_update_interval"):
        if isinstance(training[key], bool) or not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError(f"training.{key} doit être un entier strictement positif")
    if isinstance(training["gradient_steps"], bool) or not isinstance(training["gradient_steps"], int):
        raise ValueError("training.gradient_steps doit être un entier")
    train_freq = training["train_freq"]
    if (not isinstance(train_freq, (list, tuple)) or len(train_freq) != 2
            or not isinstance(train_freq[0], int) or train_freq[0] <= 0
            or train_freq[1] not in {"step", "episode"}):
        raise ValueError("training.train_freq doit être [entier positif, step|episode]")
    network = training["network"]
    if not isinstance(network, list) or not network or any(
        isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in network
    ):
        raise ValueError("training.network doit être une liste d'entiers positifs")
    resume = cfg.setdefault("resume", {})
    if not isinstance(resume, dict):
        raise ValueError("resume doit être un mapping")
    resume.setdefault("replay_buffer_policy", "auto")
    resume.setdefault("apply_current_yaml", True)
    resume.setdefault("fail_on_structural_change", True)
    resume.setdefault("log_parameter_diff", True)
    if resume["replay_buffer_policy"] not in {"keep", "discard", "auto", "error"}:
        raise ValueError("resume.replay_buffer_policy doit être keep, discard, auto ou error")
    for key in ("apply_current_yaml", "fail_on_structural_change", "log_parameter_diff"):
        if not isinstance(resume[key], bool):
            raise ValueError(f"resume.{key} doit être un booléen")
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
    observation.setdefault("include_previous_pose_error", False)
    for key in ("include_admittance_position", "include_previous_pose_error"):
        if not isinstance(observation[key], bool):
            raise ValueError(f"observation.{key} doit être un booléen")
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

    curriculum = cfg.setdefault("curriculum", {"enabled": False})
    if not isinstance(curriculum, dict):
        raise ValueError("curriculum doit être un mapping")
    enabled = curriculum.setdefault("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("curriculum.enabled doit être un booléen")
    if enabled:
        required_curriculum = {
            "curriculum_reset_probability", "success_rate_low",
            "success_rate_high", "update_interval_timesteps",
            "candidates_per_update", "evaluation_rollouts_per_candidate",
            "max_pool_size", "reverse_random_walk", "deduplication",
        }
        missing_curriculum = required_curriculum - curriculum.keys()
        if missing_curriculum:
            raise ValueError(
                "curriculum activé mais incomplet: champs manquants "
                f"{sorted(missing_curriculum)}"
            )
        walk = curriculum["reverse_random_walk"]
        deduplication = curriculum["deduplication"]
        # Valeurs injectées pour charger les archives V21 antérieures aux
        # stratégies de sampling, revalidation et expansion actuelles. Le YAML
        # source courant les déclare explicitement pour rester reproductible.
        start_sampling = curriculum.setdefault("start_sampling", {})
        revalidation = curriculum.setdefault("revalidation", {})
        expansion = curriculum.setdefault("expansion", {})
        if not isinstance(walk, dict):
            raise ValueError("curriculum.reverse_random_walk doit être un mapping")
        if not isinstance(deduplication, dict):
            raise ValueError("curriculum.deduplication doit être un mapping")
        if not isinstance(start_sampling, dict):
            raise ValueError("curriculum.start_sampling doit être un mapping")
        if not isinstance(revalidation, dict):
            raise ValueError("curriculum.revalidation doit être un mapping")
        if not isinstance(expansion, dict):
            raise ValueError("curriculum.expansion doit être un mapping")
        start_sampling.setdefault("frontier_fraction", 0.625)
        start_sampling.setdefault("historical_fraction", 0.375)
        start_sampling.setdefault("historical_bins", 4)
        start_sampling.setdefault("strategy", "legacy")
        start_sampling.setdefault("balance_unit", "episodes")
        transition_balance = start_sampling.setdefault("transition_balance", {})
        if not isinstance(transition_balance, dict):
            raise ValueError(
                "curriculum.start_sampling.transition_balance doit être un mapping"
            )
        transition_balance.setdefault("ema_alpha", 0.25)
        transition_balance.setdefault("min_episode_length", 1.0)
        transition_balance.setdefault("min_completed_episodes", 1)
        start_sampling.setdefault("adaptive_historical", False)
        start_sampling.setdefault("historical_fraction_per_state", 0.01)
        start_sampling.setdefault(
            "historical_fraction_max", start_sampling["historical_fraction"],
        )
        frontier_sampling = start_sampling.get("frontier")
        historical_sampling = start_sampling.get("historical")
        true_start_sampling = start_sampling.get("true_start")
        fallback_sampling = start_sampling.get("fallback")
        curriculum_diagnostics = curriculum.get("diagnostics", {})
        if not isinstance(curriculum_diagnostics, dict):
            raise ValueError("curriculum.diagnostics doit être un mapping")
        proposal = walk.setdefault("proposal", {})
        if not isinstance(proposal, dict):
            raise ValueError(
                "curriculum.reverse_random_walk.proposal doit être un mapping"
            )
        proposal_mode = walk.setdefault("proposal_mode", "independent")
        persistent_proposal = walk.setdefault("persistent_proposal", {})
        if not isinstance(persistent_proposal, dict):
            raise ValueError(
                "curriculum.reverse_random_walk.persistent_proposal doit être "
                "un mapping"
            )
        persistent_proposal.setdefault("attempt_direction_noise_std", 0.20)
        persistent_proposal.setdefault("hop_direction_noise_std", 0.15)
        persistent_proposal.setdefault("step_noise_std", 0.10)
        proposal.setdefault("guided_fraction", 0.0)
        proposal.setdefault("guided_noise_std", 0.20)
        proposal.setdefault("memory_size_per_parent", 16)
        revalidation.setdefault("mastered_samples_per_update", 8)
        revalidation.setdefault("too_hard_samples_per_update", 12)
        revalidation.setdefault("every_n_curriculum_updates", 1)
        expansion.setdefault("max_hops_per_seed", 4)
        expansion.setdefault("max_attempts_per_hop", 8)
        expansion.setdefault("max_candidates_per_update", 24)
        expansion.setdefault("initial_scale", 1.0)
        expansion.setdefault("scale_up_factor", 1.25)
        expansion.setdefault("scale_down_factor", 0.7)
        expansion.setdefault("min_scale", 0.5)
        expansion.setdefault("max_scale", 3.0)
        missing_walk = {
            "walks_per_seed", "max_steps", "action_scale",
        } - walk.keys()
        missing_deduplication = {
            "position_tolerance", "rotation_tolerance_deg",
        } - deduplication.keys()
        if missing_walk:
            raise ValueError(
                "curriculum.reverse_random_walk incomplet: "
                f"{sorted(missing_walk)}"
            )
        if missing_deduplication:
            raise ValueError(
                "curriculum.deduplication incomplet: "
                f"{sorted(missing_deduplication)}"
            )
        def finite_number(value: Any, name: str) -> float:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value)):
                raise ValueError(f"{name} doit être un nombre fini")
            return float(value)

        reset_probability = finite_number(
            curriculum["curriculum_reset_probability"],
            "curriculum.curriculum_reset_probability",
        )
        if not 0.0 < reset_probability < 1.0:
            raise ValueError(
                "curriculum.curriculum_reset_probability doit être dans ]0, 1["
            )
        success_rate_low = finite_number(
            curriculum["success_rate_low"], "curriculum.success_rate_low",
        )
        success_rate_high = finite_number(
            curriculum["success_rate_high"], "curriculum.success_rate_high",
        )
        if not 0.0 <= success_rate_low < success_rate_high <= 1.0:
            raise ValueError(
                "curriculum exige 0 <= success_rate_low < success_rate_high <= 1"
            )
        frontier_fraction = finite_number(
            start_sampling["frontier_fraction"],
            "curriculum.start_sampling.frontier_fraction",
        )
        historical_fraction = finite_number(
            start_sampling["historical_fraction"],
            "curriculum.start_sampling.historical_fraction",
        )
        if not (0.0 <= frontier_fraction <= 1.0
                and 0.0 <= historical_fraction <= 1.0):
            raise ValueError(
                "curriculum.start_sampling fractions doivent être dans [0, 1]"
            )
        if not np.isclose(
            frontier_fraction + historical_fraction, 1.0,
            rtol=0.0, atol=1e-12,
        ):
            raise ValueError(
                "curriculum.start_sampling exige "
                "frontier_fraction + historical_fraction = 1.0"
            )
        historical_bins = start_sampling["historical_bins"]
        if (isinstance(historical_bins, bool)
                or not isinstance(historical_bins, int)
                or historical_bins <= 0):
            raise ValueError(
                "curriculum.start_sampling.historical_bins doit être un entier "
                "strictement positif"
            )
        strategy = start_sampling["strategy"]
        if strategy not in {
            "legacy", "adaptive_three_way", "adaptive_diverse_fallback",
        }:
            raise ValueError(
                "curriculum.start_sampling.strategy doit être 'legacy', "
                "'adaptive_three_way' ou 'adaptive_diverse_fallback'"
            )
        balance_unit = start_sampling["balance_unit"]
        if balance_unit not in {"episodes", "transitions"}:
            raise ValueError(
                "curriculum.start_sampling.balance_unit doit être 'episodes' "
                "ou 'transitions'"
            )
        ema_alpha = finite_number(
            transition_balance["ema_alpha"],
            "curriculum.start_sampling.transition_balance.ema_alpha",
        )
        min_episode_length = finite_number(
            transition_balance["min_episode_length"],
            "curriculum.start_sampling.transition_balance.min_episode_length",
        )
        min_completed_episodes = transition_balance["min_completed_episodes"]
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(
                "curriculum.start_sampling.transition_balance.ema_alpha "
                "doit être dans ]0, 1]"
            )
        if min_episode_length <= 0.0:
            raise ValueError(
                "curriculum.start_sampling.transition_balance."
                "min_episode_length doit être strictement positif"
            )
        if (isinstance(min_completed_episodes, bool)
                or not isinstance(min_completed_episodes, int)
                or min_completed_episodes < 1):
            raise ValueError(
                "curriculum.start_sampling.transition_balance."
                "min_completed_episodes doit être un entier >= 1"
            )
        if strategy in {"adaptive_three_way", "adaptive_diverse_fallback"}:
            for name, section in (
                ("frontier", frontier_sampling),
                ("historical", historical_sampling),
                ("true_start", true_start_sampling),
            ):
                if not isinstance(section, dict):
                    raise ValueError(
                        f"curriculum.start_sampling.{name} doit être un mapping"
                    )
            frontier_per_state = finite_number(
                frontier_sampling.get("fraction_per_state"),
                "curriculum.start_sampling.frontier.fraction_per_state",
            )
            frontier_max = finite_number(
                frontier_sampling.get("fraction_max"),
                "curriculum.start_sampling.frontier.fraction_max",
            )
            adaptive_historical_per_state = finite_number(
                historical_sampling.get("fraction_per_state"),
                "curriculum.start_sampling.historical.fraction_per_state",
            )
            adaptive_historical_max = finite_number(
                historical_sampling.get("fraction_max"),
                "curriculum.start_sampling.historical.fraction_max",
            )
            true_start_min = finite_number(
                true_start_sampling.get("fraction_min"),
                "curriculum.start_sampling.true_start.fraction_min",
            )
            if (frontier_per_state < 0.0 or frontier_max < 0.0
                    or adaptive_historical_per_state < 0.0
                    or adaptive_historical_max < 0.0
                    or not 0.0 <= true_start_min <= 1.0):
                raise ValueError(
                    "Les fractions adaptatives doivent être positives et "
                    "true_start.fraction_min dans [0, 1]"
                )
            if frontier_max + adaptive_historical_max > 1.0 - true_start_min + 1e-12:
                raise ValueError(
                    "frontier.fraction_max + historical.fraction_max doit être <= "
                    "1 - true_start.fraction_min"
                )
            if strategy == "adaptive_diverse_fallback":
                if not isinstance(fallback_sampling, dict):
                    raise ValueError(
                        "curriculum.start_sampling.fallback doit être un mapping"
                    )
                for name in (
                    "mastered_boundary", "too_hard_near", "historical_boost",
                ):
                    section = fallback_sampling.get(name)
                    if not isinstance(section, dict):
                        raise ValueError(
                            f"curriculum.start_sampling.fallback.{name} "
                            "doit être un mapping"
                        )
                    per_state = finite_number(
                        section.get("fraction_per_state"),
                        "curriculum.start_sampling.fallback."
                        f"{name}.fraction_per_state",
                    )
                    maximum = finite_number(
                        section.get("fraction_max"),
                        "curriculum.start_sampling.fallback."
                        f"{name}.fraction_max",
                    )
                    if per_state < 0.0 or not 0.0 <= maximum <= 1.0:
                        raise ValueError(
                            "Les fractions fallback doivent avoir "
                            "fraction_per_state >= 0 et fraction_max dans [0, 1]"
                        )
        for name, default in (
            ("near_ancestor_position_m", 0.001),
            ("near_ancestor_rotation_deg", 1.0),
        ):
            if finite_number(
                curriculum_diagnostics.get(name, default),
                f"curriculum.diagnostics.{name}",
            ) < 0.0:
                raise ValueError(f"curriculum.diagnostics.{name} doit être positif")
        for key in (
            "mastered_samples_per_update", "too_hard_samples_per_update",
        ):
            value = revalidation[key]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < 0):
                raise ValueError(
                    f"curriculum.revalidation.{key} doit être un entier "
                    "positif ou nul"
                )
        if not isinstance(start_sampling["adaptive_historical"], bool):
            raise ValueError(
                "curriculum.start_sampling.adaptive_historical doit être un booléen"
            )
        historical_per_state = finite_number(
            start_sampling["historical_fraction_per_state"],
            "curriculum.start_sampling.historical_fraction_per_state",
        )
        historical_max = finite_number(
            start_sampling["historical_fraction_max"],
            "curriculum.start_sampling.historical_fraction_max",
        )
        if historical_per_state < 0.0 or not 0.0 <= historical_max <= 1.0:
            raise ValueError(
                "historical_fraction_per_state doit être positif et "
                "historical_fraction_max dans [0, 1]"
            )
        guided_fraction = finite_number(
            proposal["guided_fraction"],
            "curriculum.reverse_random_walk.proposal.guided_fraction",
        )
        guided_noise = finite_number(
            proposal["guided_noise_std"],
            "curriculum.reverse_random_walk.proposal.guided_noise_std",
        )
        if not 0.0 <= guided_fraction <= 1.0 or guided_noise < 0.0:
            raise ValueError(
                "guided_fraction doit être dans [0, 1] et guided_noise_std positif"
            )
        memory_size = proposal["memory_size_per_parent"]
        if (isinstance(memory_size, bool) or not isinstance(memory_size, int)
                or memory_size <= 0):
            raise ValueError("memory_size_per_parent doit être strictement positif")
        if proposal_mode not in {"independent", "persistent"}:
            raise ValueError(
                "curriculum.reverse_random_walk.proposal_mode doit être "
                "'independent' ou 'persistent'"
            )
        for key in (
            "attempt_direction_noise_std", "hop_direction_noise_std",
            "step_noise_std",
        ):
            if finite_number(
                persistent_proposal[key],
                f"curriculum.reverse_random_walk.persistent_proposal.{key}",
            ) < 0.0:
                raise ValueError(
                    "curriculum.reverse_random_walk.persistent_proposal."
                    f"{key} doit être positif ou nul"
                )
        revalidation_frequency = revalidation["every_n_curriculum_updates"]
        if (isinstance(revalidation_frequency, bool)
                or not isinstance(revalidation_frequency, int)
                or revalidation_frequency <= 0):
            raise ValueError(
                "curriculum.revalidation.every_n_curriculum_updates doit être "
                "un entier strictement positif"
            )
        for key in (
            "max_hops_per_seed", "max_attempts_per_hop",
            "max_candidates_per_update",
        ):
            value = expansion[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"curriculum.expansion.{key} doit être un entier "
                    "strictement positif"
                )
        initial_scale = finite_number(
            expansion["initial_scale"],
            "curriculum.expansion.initial_scale",
        )
        scale_up_factor = finite_number(
            expansion["scale_up_factor"],
            "curriculum.expansion.scale_up_factor",
        )
        scale_down_factor = finite_number(
            expansion["scale_down_factor"],
            "curriculum.expansion.scale_down_factor",
        )
        min_scale = finite_number(
            expansion["min_scale"],
            "curriculum.expansion.min_scale",
        )
        max_scale = finite_number(
            expansion["max_scale"],
            "curriculum.expansion.max_scale",
        )
        if min_scale <= 0.0:
            raise ValueError(
                "curriculum.expansion.min_scale doit être strictement positif"
            )
        if max_scale < min_scale:
            raise ValueError(
                "curriculum.expansion exige min_scale <= max_scale"
            )
        if not min_scale <= initial_scale <= max_scale:
            raise ValueError(
                "curriculum.expansion.initial_scale doit être dans "
                "[min_scale, max_scale]"
            )
        if scale_up_factor < 1.0:
            raise ValueError(
                "curriculum.expansion.scale_up_factor doit être supérieur "
                "ou égal à 1"
            )
        if not 0.0 < scale_down_factor <= 1.0:
            raise ValueError(
                "curriculum.expansion.scale_down_factor doit être dans ]0, 1]"
            )
        for key in (
            "update_interval_timesteps", "candidates_per_update",
            "evaluation_rollouts_per_candidate", "max_pool_size",
        ):
            value = curriculum[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"curriculum.{key} doit être un entier strictement positif"
                )
        for key in ("walks_per_seed", "max_steps"):
            value = walk[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    "curriculum.reverse_random_walk."
                    f"{key} doit être un entier strictement positif"
                )
        action_scale = finite_number(
            walk["action_scale"],
            "curriculum.reverse_random_walk.action_scale",
        )
        if not 0.0 < action_scale <= 1.0:
            raise ValueError(
                "curriculum.reverse_random_walk.action_scale doit être dans ]0, 1]"
            )
        for key in ("position_tolerance", "rotation_tolerance_deg"):
            name = f"curriculum.deduplication.{key}"
            if finite_number(deduplication[key], name) <= 0.0:
                raise ValueError(f"{name} doit être strictement positif")
    return cfg


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    """Archive one self-contained, human-readable configuration YAML."""
    if "extends" in config:
        raise ValueError("Une configuration résolue ne doit pas contenir 'extends'")
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
