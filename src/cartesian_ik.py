from __future__ import annotations

import mujoco
import numpy as np


def differential_ik_position_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    joint_ids: np.ndarray,
    delta_pose: np.ndarray,
    damping: float = 0.05,
    max_joint_step: float = 0.04,
) -> np.ndarray:
    """Convert a small Cartesian increment into joint position targets.

    delta_pose is [dx, dy, dz, dRx, dRy, dRz] in the world frame. This is a
    local differential IK step, not a trajectory planner. It assumes scalar
    hinge/slide joints and should be followed by a joint position or impedance
    controller.
    """
    delta_pose = np.asarray(delta_pose, dtype=np.float64)
    if delta_pose.shape != (6,):
        raise ValueError("delta_pose must have shape (6,)")

    joint_ids = np.asarray(joint_ids, dtype=np.int32)
    dof_ids = model.jnt_dofadr[joint_ids]
    qpos_ids = model.jnt_qposadr[joint_ids]

    jac_pos = np.zeros((3, model.nv), dtype=np.float64)
    jac_rot = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, int(ee_site_id))
    jacobian = np.vstack([jac_pos[:, dof_ids], jac_rot[:, dof_ids]])

    # Damped least-squares inverse: J^T (J J^T + lambda^2 I)^-1 dx.
    regularized = jacobian @ jacobian.T + (damping**2) * np.eye(6)
    joint_delta = jacobian.T @ np.linalg.solve(regularized, delta_pose)
    joint_delta = np.clip(joint_delta, -max_joint_step, max_joint_step)

    q_target = data.qpos[qpos_ids].copy() + joint_delta
    limited = model.jnt_limited[joint_ids].astype(bool)
    q_target[limited] = np.clip(
        q_target[limited],
        model.jnt_range[joint_ids[limited], 0],
        model.jnt_range[joint_ids[limited], 1],
    )
    return q_target
