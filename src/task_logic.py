"""Logique pure de terminaison et de reward de la tâche d'assemblage."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TaskStatus:
    geometric_success: bool
    success: bool
    unsafe: bool
    unsafe_force: bool
    unsafe_torque: bool
    unsafe_workspace: bool
    terminated: bool
    truncated: bool
    termination_reason: str


def prepare_proximity_milestones(
    configured: list[dict],
) -> list[tuple[float, float | None, float]]:
    """Convertit une fois les seuils configurés vers mètres/radians/bonus."""
    return [
        (
            float(item.get("position_threshold", item.get("threshold"))),
            (np.deg2rad(float(item["orientation_threshold_deg"]))
             if "orientation_threshold_deg" in item else None),
            float(item["bonus"]),
        )
        for item in configured
    ]


def newly_reached_milestones(
    position_error: float,
    rotation_error: float,
    milestones: list[tuple[float, float | None, float]],
    reached: set[int],
) -> tuple[float, set[int]]:
    """Calcule tous les bonus encore disponibles atteints pendant ce step."""
    satisfied = satisfied_milestone_indices(
        position_error, rotation_error, milestones,
    )
    new_indices = satisfied - reached
    bonus = sum(milestones[index][2] for index in new_indices)
    return bonus, reached | new_indices


def satisfied_milestone_indices(
    position_error: float,
    rotation_error: float,
    milestones: list[tuple[float, float | None, float]],
) -> set[int]:
    """Retourne les milestones dont les conditions sont simultanément vraies."""
    return {
        index for index, (position_threshold, orientation_threshold, _) in enumerate(milestones)
        if (position_error < position_threshold
            and (orientation_threshold is None or rotation_error < orientation_threshold))
    }


def assess_status(
    *,
    position_error: float,
    rotation_error: float,
    max_force: float,
    max_torque: float,
    workspace_error: float,
    step_count: int,
    config: dict,
    max_episode_steps: int,
) -> TaskStatus:
    """Classe un état avec priorité absolue aux contraintes de sécurité."""
    geometric_success = (
        position_error < float(config["position_tolerance"])
        and rotation_error < np.deg2rad(float(config["rotation_tolerance_deg"]))
    )
    force_unsafe = max_force >= float(config["max_force"])
    torque_unsafe = max_torque >= float(config["max_torque"])
    workspace_unsafe = workspace_error >= float(config["workspace_radius"])
    unsafe = force_unsafe or torque_unsafe or workspace_unsafe
    success = geometric_success and not unsafe
    timeout = step_count >= max_episode_steps and not success and not unsafe
    terminated = success or unsafe or timeout
    truncated = False

    if force_unsafe and torque_unsafe:
        reason = "unsafe_force_and_torque"
    elif force_unsafe:
        reason = "unsafe_force"
    elif torque_unsafe:
        reason = "unsafe_torque"
    elif workspace_unsafe:
        reason = "unsafe_workspace"
    elif success:
        reason = "success"
    elif timeout:
        reason = "timeout"
    else:
        reason = "running"
    return TaskStatus(
        geometric_success, success, unsafe,
        force_unsafe, torque_unsafe, workspace_unsafe,
        terminated, truncated, reason,
    )


def reward_components(
    *,
    position_error: float,
    rotation_error: float,
    previous_position_error: float,
    previous_rotation_error: float,
    max_force: float,
    action: np.ndarray,
    status: TaskStatus,
    config: dict,
    action_config: dict,
    max_torque: float = 0.0,
    proximity_bonus: float = 0.0,
) -> dict[str, float]:
    """Calcule les composantes interprétables du reward sur vérité terrain."""
    position_progress = previous_position_error - position_error
    normalized_position_progress = (
        position_progress / float(action_config["max_translation_step"])
    )
    rotation_progress = previous_rotation_error - rotation_error
    normalized_rotation_progress = (
        rotation_progress
        / np.deg2rad(float(action_config["max_rotation_step_deg"]))
    )
    return {
        "reward_position": -float(config["position_weight"]) * position_error,
        "reward_orientation": -float(config["orientation_weight"]) * rotation_error,
        "reward_progress": float(config["progress_weight"]) * (
            normalized_position_progress + normalized_rotation_progress
        ),
        "reward_force": -float(config["force_weight"]) * max_force,
        "reward_torque": -float(config.get("torque_weight", 0.0)) * max_torque,
        "reward_action": -float(config["action_weight"]) * float(np.dot(action, action)),
        "reward_proximity": float(proximity_bonus),
        "reward_step": -float(config.get("step_penalty", 0.0)),
        "reward_success": float(config["success_bonus"]) if status.success else 0.0,
        "reward_unsafe": -float(config["unsafe_penalty"]) if status.unsafe else 0.0,
        "reward_timeout": (
            -float(config.get("timeout_penalty", 0.0))
            if status.termination_reason == "timeout" else 0.0
        ),
    }
