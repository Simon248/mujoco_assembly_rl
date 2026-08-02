from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class AssemblyEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal MuJoCo peg-in-hole environment.

    The policy controls small increments of [x, y, z, yaw]. The example uses a
    Cartesian stage so the complete RL pipeline is runnable without a robot
    model. For a real arm, replace _apply_action() with the differential IK
    controller in cartesian_ik.py.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 250,
        frame_skip: int = 20,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.xml_path = Path(xml_path or Path(__file__).with_name("scene.xml"))

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.joint_names = ["joint_x", "joint_y", "joint_z", "joint_yaw"]
        self.actuator_names = ["act_x", "act_y", "act_z", "act_yaw"]
        self.sensor_names = {
            "force": "wrist_force",
            "torque": "wrist_torque",
            "ee_pos": "ee_pos",
            "ee_quat": "ee_quat",
            "rel_pos": "part_rel_pos",
            "rel_quat": "part_rel_quat",
        }

        self.joint_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names],
            dtype=np.int32,
        )
        self.actuator_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.actuator_names],
            dtype=np.int32,
        )
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.dof_adr = self.model.jnt_dofadr[self.joint_ids]

        # Policy actions are normalized. Physical increments are applied at 50 Hz.
        self.action_scale = np.array(
            [0.0008, 0.0008, 0.0008, np.deg2rad(0.6)], dtype=np.float64
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Observation = q, dq, EE pose, part pose relative to hole, wrench, previous action.
        observation_size = 4 + 4 + 3 + 4 + 3 + 4 + 6 + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.goal_rel_pos = np.array([0.0, 0.0, -0.045], dtype=np.float64)
        self.goal_rel_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.max_force = 120.0

        self._step_count = 0
        self._previous_action = np.zeros(4, dtype=np.float64)
        self._ctrl_target = np.zeros(4, dtype=np.float64)
        self._wrench_bias = np.zeros(6, dtype=np.float64)
        self._previous_pos_error = 0.0
        self._previous_rot_error = 0.0
        self._viewer = None

    def _name2id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _sensor(self, name: str) -> np.ndarray:
        sensor_id = self._name2id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return self.data.sensordata[address : address + dimension].copy()

    @staticmethod
    def _quat_angle_wxyz(q: np.ndarray) -> float:
        q = q / max(np.linalg.norm(q), 1e-12)
        # q and -q represent the same orientation.
        scalar = float(np.clip(abs(q[0]), 0.0, 1.0))
        return 2.0 * np.arccos(scalar)

    def _raw_state(self) -> dict[str, np.ndarray]:
        force = self._sensor(self.sensor_names["force"])
        torque = self._sensor(self.sensor_names["torque"])
        return {
            "qpos": self.data.qpos[self.qpos_adr].copy(),
            "qvel": self.data.qvel[self.dof_adr].copy(),
            "ee_pos": self._sensor(self.sensor_names["ee_pos"]),
            "ee_quat": self._sensor(self.sensor_names["ee_quat"]),
            "rel_pos": self._sensor(self.sensor_names["rel_pos"]),
            "rel_quat": self._sensor(self.sensor_names["rel_quat"]),
            "wrench": np.concatenate([force, torque]) - self._wrench_bias,
        }

    def _get_obs(self) -> np.ndarray:
        s = self._raw_state()

        # Fixed, physically meaningful scaling. Reuse exactly the same scaling in ROS 2.
        q_low = self.model.jnt_range[self.joint_ids, 0]
        q_high = self.model.jnt_range[self.joint_ids, 1]
        q_mid = 0.5 * (q_low + q_high)
        q_half_range = np.maximum(0.5 * (q_high - q_low), 1e-6)
        qpos_scaled = (s["qpos"] - q_mid) / q_half_range
        qvel_scaled = s["qvel"] / np.array([0.20, 0.20, 0.20, 2.0])
        ee_pos_scaled = s["ee_pos"] / 0.20
        rel_pos_scaled = s["rel_pos"] / 0.05
        wrench_scaled = s["wrench"] / np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])

        obs = np.concatenate(
            [
                qpos_scaled,
                qvel_scaled,
                ee_pos_scaled,
                s["ee_quat"],
                rel_pos_scaled,
                s["rel_quat"],
                wrench_scaled,
                self._previous_action,
            ]
        )
        return obs.astype(np.float32)

    def _errors(self) -> tuple[float, float, float]:
        rel_pos = self._sensor(self.sensor_names["rel_pos"])
        rel_quat = self._sensor(self.sensor_names["rel_quat"])
        position_error = float(np.linalg.norm(rel_pos - self.goal_rel_pos))
        lateral_error = float(np.linalg.norm(rel_pos[:2] - self.goal_rel_pos[:2]))
        rotation_error = self._quat_angle_wxyz(rel_quat)
        return position_error, lateral_error, rotation_error

    def _apply_action(self, action: np.ndarray) -> None:
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._ctrl_target += clipped * self.action_scale

        ctrl_range = self.model.actuator_ctrlrange[self.actuator_ids]
        self._ctrl_target = np.clip(
            self._ctrl_target, ctrl_range[:, 0], ctrl_range[:, 1]
        )
        self.data.ctrl[self.actuator_ids] = self._ctrl_target
        self._previous_action = clipped

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        pos_error, lateral_error, rot_error = self._errors()
        wrench = self._raw_state()["wrench"]
        force_norm = float(np.linalg.norm(wrench[:3]))

        # Dense progress reward plus penalties that discourage impacts and chatter.
        progress = self._previous_pos_error - pos_error
        rotation_progress = self._previous_rot_error - rot_error
        reward = 250.0 * progress + 0.5 * rotation_progress
        reward -= 0.01 * float(np.dot(self._previous_action, self._previous_action))
        reward -= 0.0005 * max(force_norm - 15.0, 0.0) ** 2

        rel_pos = self._sensor(self.sensor_names["rel_pos"])
        if lateral_error < 0.0025:
            reward += 0.05
        if rel_pos[2] < -0.005:
            reward += 0.10

        success = (
            lateral_error < 0.0015
            and abs(rel_pos[2] - self.goal_rel_pos[2]) < 0.002
            and rot_error < np.deg2rad(2.0)
        )
        unsafe_contact = force_norm > self.max_force
        terminated = bool(success or unsafe_contact)
        truncated = self._step_count >= self.max_episode_steps

        if success:
            reward += 20.0
        elif unsafe_contact:
            reward -= 10.0

        self._previous_pos_error = pos_error
        self._previous_rot_error = rot_error

        info = {
            "is_success": success,
            "position_error_m": pos_error,
            "lateral_error_m": lateral_error,
            "rotation_error_rad": rot_error,
            "force_norm_N": force_norm,
        }

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        mujoco.mj_resetData(self.model, self.data)

        initial_q = np.array(
            [
                self.np_random.uniform(-0.006, 0.006),
                self.np_random.uniform(-0.006, 0.006),
                0.0,
                self.np_random.uniform(-np.deg2rad(8), np.deg2rad(8)),
            ],
            dtype=np.float64,
        )
        self.data.qpos[self.qpos_adr] = initial_q
        self.data.qvel[self.dof_adr] = 0.0
        self._ctrl_target = initial_q.copy()
        self.data.ctrl[self.actuator_ids] = self._ctrl_target
        self._previous_action.fill(0.0)
        self._step_count = 0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        # Tare the simulated wrist sensor after the initial pose has settled.
        self._wrench_bias = np.concatenate(
            [
                self._sensor(self.sensor_names["force"]),
                self._sensor(self.sensor_names["torque"]),
            ]
        )
        self._previous_pos_error, _, self._previous_rot_error = self._errors()

        return self._get_obs(), {"is_success": False}

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
