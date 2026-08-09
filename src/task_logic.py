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


def pose_distance(
    position_error: float, rotation_error: float, rotation_length_scale: float,
) -> float:
    """Distance de pose 6D en mètres, avec rotation convertie en longueur."""
    return float(np.hypot(position_error, rotation_length_scale * rotation_error))


def pose_potential(
    distance: float, potential_scale: float, potential_distance_scale: float,
) -> float:
    """Potentiel positif, borné et strictement décroissant avec la distance."""
    return float(potential_scale * np.exp(-distance / potential_distance_scale))


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
    current_pose_distance: float,
    next_pose_distance: float,
    gamma: float,
    max_force: float,
    action: np.ndarray,
    status: TaskStatus,
    config: dict,
    max_torque: float = 0.0,
) -> dict[str, float]:
    """Calcule la reward potentielle et les seuls coûts/événements conservés."""
    phi_current = pose_potential(
        current_pose_distance, config["potential_scale"],
        config["potential_distance_scale"],
    )
    phi_next = pose_potential(
        next_pose_distance, config["potential_scale"],
        config["potential_distance_scale"],
    )
    # Dans ce MDP épisodique, aucun potentiel ne subsiste après un terminal.
    phi_next_for_shaping = 0.0 if status.terminated else phi_next
    return {
        "pose_distance": next_pose_distance,
        "phi_current": phi_current,
        "phi_next": phi_next,
        "reward_potential": gamma * phi_next_for_shaping - phi_current,
        "reward_force": -float(config["force_weight"]) * max_force,
        "reward_torque": -float(config.get("torque_weight", 0.0)) * max_torque,
        "reward_action": -float(config["action_weight"]) * float(np.dot(action, action)),
        "reward_step": -float(config.get("step_penalty", 0.0)),
        "reward_success": float(config["success_bonus"]) if status.success else 0.0,
        "reward_unsafe": -float(config["unsafe_penalty"]) if status.unsafe else 0.0,
        "reward_timeout": (
            -float(config.get("timeout_penalty", 0.0))
            if status.termination_reason == "timeout" else 0.0
        ),
    }
