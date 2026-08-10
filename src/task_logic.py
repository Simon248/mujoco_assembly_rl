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
    """Distance additive de pose 6D, en mètres équivalents."""
    return float(position_error + rotation_length_scale * rotation_error)


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
    max_force: float,
    action: np.ndarray,
    status: TaskStatus,
    config: dict,
    max_torque: float = 0.0,
) -> dict[str, float]:
    """Calcule le coût dense de l'état atteint et les coûts/événements."""
    rotation_equivalent_distance = (
        float(config["rotation_length_scale"]) * rotation_error
    )
    distance = pose_distance(
        position_error, rotation_error, float(config["rotation_length_scale"]),
    )
    return {
        "rotation_equivalent_distance": rotation_equivalent_distance,
        "pose_distance": distance,
        "reward_pose": -float(config["pose_weight"]) * distance,
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
