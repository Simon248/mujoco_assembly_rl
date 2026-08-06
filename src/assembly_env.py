from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from src.mujoco_plugins import load_sdf_plugin


class AssemblyEnv(gym.Env[np.ndarray, np.ndarray]):
    """Six-DOF CAD assembly environment.

    The policy controls small Cartesian increments of
    ``[x, y, z, roll, pitch, yaw]``. The two STL files share the assembly
    frame: the identity relative pose is the assembled target.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 600,
        frame_skip: int = 20,
        curriculum_enabled: bool = True,
        disassembly_probability: float = 0.75,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.curriculum_enabled = curriculum_enabled
        if not 0.0 <= disassembly_probability <= 1.0:
            raise ValueError("disassembly_probability must be in [0, 1]")
        self.disassembly_probability = disassembly_probability
        self.xml_path = Path(xml_path or Path(__file__).with_name("scene.xml"))

        load_sdf_plugin()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.joint_names = [
            "joint_x", "joint_y", "joint_z", "joint_roll", "joint_pitch", "joint_yaw"
        ]
        self.actuator_names = [
            "act_x", "act_y", "act_z", "act_roll", "act_pitch", "act_yaw"
        ]
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
            [0.0008, 0.0008, 0.0008, np.deg2rad(0.6), np.deg2rad(0.6), np.deg2rad(0.6)],
            dtype=np.float64,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        # Observation includes both achieved and desired CAD poses.
        observation_size = 6 + 6 + 3 + 4 + 3 + 4 + 3 + 4 + 6 + 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.goal_rel_pos = np.zeros(3, dtype=np.float64)
        self.goal_rel_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._episode_mode = "assembly"
        self.max_force = 120.0
        self.table_collision_id = self._name2id(
            mujoco.mjtObj.mjOBJ_GEOM, "assembly_table_collision"
        )
        self.part_collision_id = self._name2id(
            mujoco.mjtObj.mjOBJ_GEOM, "part_1_collision"
        )

        # (final-distribution probability, xy range, z range, rotation range).
        # A stage advances only after 21 successes in a window of 30 episodes.
        self._curriculum = (
            (0.1, 0.002, (0.003, 0.008), np.deg2rad(0.1)),
            (0.20, 0.010, (0.020, 0.040), np.deg2rad(2.0)),
            (0.50, 0.030, (0.050, 0.100), np.deg2rad(5.0)),
            (0.80, 0.060, (0.100, 0.150), np.deg2rad(10.0)),
            (1.00, 0.100, (0.200, 0.200), np.deg2rad(15.0)),
        )
        self._training_decisions = 0
        self._curriculum_stage = 0
        self._curriculum_outcomes: deque[float] = deque(maxlen=30)
        self._last_curriculum_stage = 0
        self._max_reset_attempts = 100

        self._step_count = 0
        self._previous_action = np.zeros(6, dtype=np.float64)
        self._ctrl_target = np.zeros(6, dtype=np.float64)
        self._wrench_bias = np.zeros(6, dtype=np.float64)
        self._previous_pos_error = 0.0
        self._previous_rot_error = 0.0
        self._viewer = None
        # Cadencement temps réel du rendu "human".
        # Durée simulée par appel à step() = timestep * frame_skip.
        self._step_wall_time = float(self.model.opt.timestep) * self.frame_skip
        self._next_render_time: float | None = None


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

    @staticmethod
    def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * ry - lz * rz,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )

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
        qvel_scaled = s["qvel"] / np.array([0.20, 0.20, 0.20, 2.0, 2.0, 2.0])
        ee_pos_scaled = s["ee_pos"] / 0.20
        rel_pos_scaled = s["rel_pos"] / 0.20
        wrench_scaled = s["wrench"] / np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])

        obs = np.concatenate(
            [
                qpos_scaled,
                qvel_scaled,
                ee_pos_scaled,
                s["ee_quat"],
                rel_pos_scaled,
                s["rel_quat"],
                self.goal_rel_pos / 0.20,
                self.goal_rel_quat,
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
        goal_inverse = self.goal_rel_quat * np.array([1.0, -1.0, -1.0, -1.0])
        rotation_error = self._quat_angle_wxyz(
            self._quat_multiply_wxyz(goal_inverse, rel_quat)
        )
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

    def _sample_initial_q(self) -> tuple[np.ndarray, int, bool]:
        """Sample a curriculum pose; all samples remain collision-free."""
        stage = (
            len(self._curriculum) - 1 if not self.curriculum_enabled else self._curriculum_stage
        )
        final_probability, xy_range, z_range, angle_range = self._curriculum[stage]
        use_final_distribution = bool(self.np_random.random() < final_probability)
        if use_final_distribution:
            xy_range, z_range, angle_range = 0.10, (0.200, 0.200), np.deg2rad(15.0)

        height = self.np_random.uniform(*z_range)
        q = np.array(
            [
                self.np_random.uniform(-xy_range, xy_range),
                self.np_random.uniform(-xy_range, xy_range),
                height - 0.20,
                self.np_random.uniform(-angle_range, angle_range),
                self.np_random.uniform(-angle_range, angle_range),
                self.np_random.uniform(-angle_range, angle_range),
            ],
            dtype=np.float64,
        )
        return q, stage, use_final_distribution

    def _update_curriculum(self, success: bool) -> None:
        if not self.curriculum_enabled or self._curriculum_stage >= len(self._curriculum) - 1:
            return
        self._curriculum_outcomes.append(float(success))
        if len(self._curriculum_outcomes) == self._curriculum_outcomes.maxlen and np.mean(
            self._curriculum_outcomes
        ) >= 0.70:
            self._curriculum_stage += 1
            self._curriculum_outcomes.clear()

    def _has_assembly_contact(self) -> bool:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if {
                int(contact.geom1),
                int(contact.geom2),
            } == {self.table_collision_id, self.part_collision_id}:
                return True
        return False

    def _set_collision_free_initial_pose(self) -> tuple[np.ndarray, int, bool]:
        for _ in range(self._max_reset_attempts):
            initial_q, stage, is_final = self._sample_initial_q()
            self.data.qpos[self.qpos_adr] = initial_q
            self.data.qvel[self.dof_adr] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if not self._has_assembly_contact():
                return initial_q, stage, is_final
        raise RuntimeError(
            "Unable to sample a collision-free assembly start pose after "
            f"{self._max_reset_attempts} attempts. Check CAD overlap and curriculum ranges."
        )

    def _sample_free_goal(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample a collision-free target in the final free-space domain."""
        for _ in range(self._max_reset_attempts):
            q = np.array(
                [
                    self.np_random.uniform(-0.10, 0.10),
                    self.np_random.uniform(-0.10, 0.10),
                    0.0,
                    self.np_random.uniform(-np.deg2rad(15), np.deg2rad(15)),
                    self.np_random.uniform(-np.deg2rad(15), np.deg2rad(15)),
                    self.np_random.uniform(-np.deg2rad(15), np.deg2rad(15)),
                ],
                dtype=np.float64,
            )
            self.data.qpos[self.qpos_adr] = q
            self.data.qvel[self.dof_adr] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if not self._has_assembly_contact():
                return (
                    self._sensor(self.sensor_names["rel_pos"]),
                    self._sensor(self.sensor_names["rel_quat"]),
                )
        raise RuntimeError("Unable to sample a collision-free disassembly goal.")

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        self._training_decisions += 1
        pos_error, lateral_error, rot_error = self._errors()
        wrench = self._raw_state()["wrench"]
        force_norm = float(np.linalg.norm(wrench[:3]))

        # Dense progress reward plus penalties that discourage impacts and chatter.
        progress = self._previous_pos_error - pos_error
        rotation_progress = self._previous_rot_error - rot_error
        reward = 250.0 * progress + 40.0 * rotation_progress
        reward -= 0.01 * float(np.dot(self._previous_action, self._previous_action))
        reward -= 0.0005 * max(force_norm - 15.0, 0.0) ** 2

        rel_pos = self._sensor(self.sensor_names["rel_pos"])
        if self._episode_mode == "assembly" and lateral_error < 0.0025:
            reward += 0.05
        if self._episode_mode == "assembly" and rel_pos[2] < 0.015:
            reward += 0.10

        position_tolerance = 0.005 if self._episode_mode == "disassembly" else 0.0015
        rotation_tolerance = np.deg2rad(5.0 if self._episode_mode == "disassembly" else 2.0)
        success = (
            pos_error < position_tolerance
            and rot_error < rotation_tolerance
        )
        unsafe_contact = force_norm > self.max_force
        terminated = bool(success or unsafe_contact)
        truncated = self._step_count >= self.max_episode_steps

        if success:
            reward += 100.0
        elif unsafe_contact:
            reward -= 10.0

        if (terminated or truncated) and self._episode_mode == "assembly":
            self._update_curriculum(success)

        self._previous_pos_error = pos_error
        self._previous_rot_error = rot_error

        info = {
            "is_success": success,
            "position_error_m": pos_error,
            "lateral_error_m": lateral_error,
            "rotation_error_rad": rot_error,
            "force_norm_N": force_norm,
            "is_disassembly": self._episode_mode == "disassembly",
            "curriculum_stage": self._last_curriculum_stage + 1,
            "curriculum_success_rate": float(np.mean(self._curriculum_outcomes))
            if self._curriculum_outcomes
            else 0.0,
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
        self._episode_mode = (
            "disassembly"
            if self.curriculum_enabled and self.np_random.random() < self.disassembly_probability
            else "assembly"
        )
        if self._episode_mode == "disassembly":
            self.goal_rel_pos, self.goal_rel_quat = self._sample_free_goal()
            mujoco.mj_resetData(self.model, self.data)
            initial_q = np.array([0.0, 0.0, -0.20, 0.0, 0.0, 0.0], dtype=np.float64)
            self.data.qpos[self.qpos_adr] = initial_q
            self.data.qvel[self.dof_adr] = 0.0
            self._last_curriculum_stage = -1
            is_final_distribution = True
        else:
            self.goal_rel_pos.fill(0.0)
            self.goal_rel_quat[:] = [1.0, 0.0, 0.0, 0.0]
            initial_q, self._last_curriculum_stage, is_final_distribution = (
                self._set_collision_free_initial_pose()
            )
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

        return self._get_obs(), {
            "is_success": False,
            "curriculum_stage": self._last_curriculum_stage + 1,
            "is_final_distribution": is_final_distribution,
            "episode_mode": self._episode_mode,
        }

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._next_render_time = time.monotonic()
        self._viewer.sync()

        # Cadencement temps réel : on attend que l'horloge murale rattrape
        # le temps simulé écoulé depuis le dernier rendu. Sans cela, la
        # simulation défile trop vite pour être observable.
        if self._next_render_time is not None:
            self._next_render_time += self._step_wall_time
            delay = self._next_render_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # On a pris du retard : on resynchronise plutôt que
                # d'accumuler la dette temporelle.
                self._next_render_time = time.monotonic()

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
            self._next_render_time = None
