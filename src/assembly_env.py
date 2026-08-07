from __future__ import annotations

import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from src.mujoco_plugins import load_sdf_plugin
from src.place_path import PlacePath, load_place_path


@dataclass(frozen=True)
class ResidualConfig:
    """Safety-relevant values for the tactile residual controller (SI units)."""

    decision_dt: float = 0.02
    residual_linear_speed: float = 0.020
    residual_angular_speed: float = np.deg2rad(20.0)
    progress_speed: float = 0.25
    residual_linear_limit: float = 0.015
    residual_angular_limit: float = np.deg2rad(12.0)
    corridor_linear_limit: float = 0.020
    soft_force: float = 20.0
    soft_torque: float = 1.5
    hard_force: float = 80.0
    hard_torque: float = 8.0
    history_length: int = 8
    initial_linear_error: float = 0.000
    initial_angular_error: float = np.deg2rad(0.0)
    fixture_linear_error: float = 0.000
    fixture_angular_error: float = np.deg2rad(0.0)
    grasp_linear_error: float = 0.000
    grasp_angular_error: float = np.deg2rad(0.0)
    admittance_mass: float = 3.0
    admittance_damping: float = 90.0
    admittance_stiffness: float = 900.0
    admittance_offset_limit: float = 0.006
    success_position: float = 0.003
    success_rotation: float = np.deg2rad(4.0)
    recovery_force_steps: int = 5
    recovery_torque_steps: int = 1
    recovery_min_steps: int = 10
    recovery_clear_steps: int = 10
    recovery_clear_force: float = 15.0
    recovery_clear_torque: float = 1.5
    recovery_effort_persistence_steps: int = 25
    recovery_max_duration_s: float = 2.0
    stall_progress: float = 0.85
    stall_window_steps: int = 50
    stall_position_improvement: float = 0.0005
    stall_rotation_improvement: float = np.deg2rad(0.5)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array([lw * rw - lx * rx - ly * ry - lz * rz, lw * rx + lx * rw + ly * rz - lz * ry,
                     lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw])


def _quat_inverse(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_angle(quat: np.ndarray) -> float:
    return 2.0 * np.arccos(np.clip(abs(quat[0]) / max(np.linalg.norm(quat), 1e-12), 0.0, 1.0))


def _rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.array([0.0, *vector], dtype=np.float64)
    return _quat_multiply(_quat_multiply(quat, pure), _quat_inverse(quat))[1:]


def _quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Euler XYZ for the small rotations used by the Cartesian fixture."""
    w, x, y, z = q / np.linalg.norm(q)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


class AssemblyEnv(gym.Env[np.ndarray, np.ndarray]):
    """Tactile residual-RL environment for one recorded CAD ``place`` path.

    The actor receives no true piece/fixture relative pose.  Fixture and grasp
    errors are sampled physically at reset and remain hidden for the episode.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, xml_path: str | Path | None = None, render_mode: str | None = None,
                 max_episode_steps: int = 700, frame_skip: int = 20,
                 part_name: str = "part_1", paths_dir: str | Path | None = None,
                 config: ResidualConfig | None = None) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if part_name not in {"part_1", "part_2", "part_3"}:
            raise ValueError("part_name must be part_1, part_2, or part_3")
        self.render_mode, self.max_episode_steps, self.frame_skip = render_mode, max_episode_steps, frame_skip
        self.part_name = part_name
        self.config = config or ResidualConfig()
        self.xml_path = Path(xml_path or Path(__file__).parents[1] / "data/input/scene.xml")
        self.paths_dir = Path(paths_dir or self.xml_path.parent / "chemin")
        self.cad_dir = self.xml_path.parent / "cad"
        self.path: PlacePath = load_place_path(self.paths_dir, part_name)

        load_sdf_plugin()
        self.model = self._make_model(np.zeros(6), np.zeros(6))
        self.data = mujoco.MjData(self.model)
        self.joint_names = ["joint_x", "joint_y", "joint_z", "joint_roll", "joint_pitch", "joint_yaw"]
        self.actuator_names = ["act_x", "act_y", "act_z", "act_roll", "act_pitch", "act_yaw"]
        self.joint_ids = np.array([self._name2id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names])
        self.actuator_ids = np.array([self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.actuator_names])
        self.qpos_adr, self.dof_adr = self.model.jnt_qposadr[self.joint_ids], self.model.jnt_dofadr[self.joint_ids]
        self.force_sensor_id = self._name2id(mujoco.mjtObj.mjOBJ_SENSOR, "wrist_force")
        self.torque_sensor_id = self._name2id(mujoco.mjtObj.mjOBJ_SENSOR, "wrist_torque")
        self.true_rel_pos_sensor_id = self._name2id(mujoco.mjtObj.mjOBJ_SENSOR, "part_rel_pos")
        self.true_rel_quat_sensor_id = self._name2id(mujoco.mjtObj.mjOBJ_SENSOR, "part_rel_quat")
        self.active_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, f"{part_name}_collision")
        self.obstacle_geom_ids = {self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "assembly_table_collision")}
        self.obstacle_geom_ids.update(self._name2id(mujoco.mjtObj.mjOBJ_GEOM, f"part_{i}_collision") for i in range(1, int(part_name[-1])))
        self.fixture_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "fixture")
        self.grasp_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "grasp_error")
        self.fixed_body_ids = {
            name: self._name2id(mujoco.mjtObj.mjOBJ_BODY, f"{name}_fixed")
            for name in (f"part_{i}" for i in range(1, int(part_name[-1])))
        }

        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self._base_observation_size = 34
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self._base_observation_size * self.config.history_length,), dtype=np.float32)
        self._history: deque[np.ndarray] = deque(maxlen=self.config.history_length)
        self._viewer = None
        self._next_render_time: float | None = None
        self._step_wall_time = float(self.model.opt.timestep) * self.frame_skip
        self._wrench_bias = np.zeros(6)
        self._previous_action = np.zeros(7)
        self._residual_offset = np.zeros(6)
        self._admittance_offset = np.zeros(6)
        self._admittance_velocity = np.zeros(6)
        self._progress = self._previous_progress = 0.0
        self._progress_rate = 0.0
        self._step_count = 0
        self._contact_duration = 0.0
        self._impulse = 0.0
        self._max_force = 0.0
        self._max_torque = 0.0
        self._control_mode = "tracking"
        self._recovery_count = 0
        self._recovery_steps = 0
        self._recovery_duration = 0.0
        self._soft_force_steps = 0
        self._soft_torque_steps = 0
        self._recovery_effort_steps = 0
        self._forced_retreat = False
        self._clear_steps = 0
        self._stuck_detected = False
        self._stagnation_errors: deque[tuple[float, float]] = deque(
            maxlen=self.config.stall_window_steps + 1
        )
        self._previous_true_position_error = 0.0
        self._previous_true_rotation_error = 0.0
        self._previous_force = 0.0

    def _make_model(self, fixture_error: np.ndarray, grasp_error: np.ndarray) -> mujoco.MjModel:
        fixture_p, fixture_q = fixture_error[:3], self._euler_quat(fixture_error[3:])
        grasp_p, grasp_q = grasp_error[:3], self._euler_quat(grasp_error[3:])
        parts = [f"part_{i}" for i in range(1, int(self.part_name[-1]))]
        ext = ['<mujoco model="residual_place">', '<compiler angle="radian" autolimits="true" meshdir="' + escape(str(self.cad_dir)) + '"/>',
               '<option timestep="0.001" integrator="implicitfast" cone="elliptic" sdf_iterations="10" sdf_initpoints="20"/>',
               '<default><joint damping="2" armature="0.02"/><geom friction="0.9 0.01 0.001" condim="4" solref="0.004 1" solimp="0.90 0.95 0.001"/><position kp="700" kv="45"/></default>',
               '<extension><plugin plugin="mujoco.sdf.sdflib"><instance name="table_sdf"><config key="aabb" value="0"/></instance>']
        ext += [f'<instance name="{name}_sdf"><config key="aabb" value="0"/></instance>' for name in parts + [self.part_name]]
        ext += ['</plugin></extension><asset>', '<mesh name="table_mesh" file="chandelier_assembly_table_visual.stl"/>']
        ext += [f'<mesh name="{name}_mesh" file="chandelier_{name}.stl"/>' for name in parts + [self.part_name]]
        ext += ['</asset><worldbody><light pos="0 -1 1.5" dir="0 1 -1"/><geom type="plane" size="0 0 .05" contype="0" conaffinity="0"/>',
                f'<body name="fixture" pos="{" ".join(map(str, fixture_p))}" quat="{" ".join(map(str, fixture_q))}"><geom type="mesh" mesh="table_mesh" contype="0" conaffinity="0"/><geom name="assembly_table_collision" type="sdf" mesh="table_mesh"><plugin instance="table_sdf"/></geom><site name="assembly_frame" pos="0 0 0" size=".001"/></body>',
                '<body name="x_stage" pos="0 0 .20"><inertial pos="0 0 0" mass="1" diaginertia=".001 .001 .001"/><joint name="joint_x" type="slide" axis="1 0 0" range="-.15 .15"/><body><inertial pos="0 0 0" mass=".8" diaginertia=".0008 .0008 .0008"/><joint name="joint_y" type="slide" axis="0 1 0" range="-.15 .15"/><body><inertial pos="0 0 0" mass=".6" diaginertia=".0006 .0006 .0006"/><joint name="joint_z" type="slide" axis="0 0 1" range="-.24 .05"/><body><inertial pos="0 0 0" mass=".4" diaginertia=".0004 .0004 .0004"/><joint name="joint_roll" type="hinge" axis="1 0 0" range="-.7 .7"/><body><inertial pos="0 0 0" mass=".3" diaginertia=".0003 .0003 .0003"/><joint name="joint_pitch" type="hinge" axis="0 1 0" range="-.7 .7"/><body><inertial pos="0 0 0" mass=".2" diaginertia=".0002 .0002 .0002"/><joint name="joint_yaw" type="hinge" axis="0 0 1" range="-.7 .7"/><site name="wrist_ft_site" size=".001"/><site name="ee_site" size=".001"/>',
                f'<body name="grasp_error" pos="{" ".join(map(str, grasp_p))}" quat="{" ".join(map(str, grasp_q))}"><geom name="{self.part_name}_collision" type="sdf" mesh="{self.part_name}_mesh" mass=".2"><plugin instance="{self.part_name}_sdf"/></geom><site name="part_frame" pos="0 0 0" size=".001"/></body>', '</body></body></body></body></body></body>']
        for name in parts:
            path = load_place_path(self.paths_dir, name)
            p, q = path.final_pose
            # Fixed assembled pieces move with the physically shifted fixture.
            p = fixture_p + _rotate(fixture_q, p)
            q = _quat_multiply(fixture_q, q)
            ext += [f'<body name="{name}_fixed" pos="{" ".join(map(str, p))}" quat="{" ".join(map(str, q))}"><geom name="{name}_collision" type="sdf" mesh="{name}_mesh"><plugin instance="{name}_sdf"/></geom></body>']
        ext += ['</worldbody><actuator>']
        for name, joint, low, high in zip(["act_x", "act_y", "act_z", "act_roll", "act_pitch", "act_yaw"], self.joint_names if hasattr(self, "joint_names") else ["joint_x", "joint_y", "joint_z", "joint_roll", "joint_pitch", "joint_yaw"], ["-.15", "-.15", "-.24", "-.7", "-.7", "-.7"], [".15", ".15", ".05", ".7", ".7", ".7"]):
            ext += [f'<position name="{name}" joint="{joint}" ctrlrange="{low} {high}" forcerange="-300 300"/>']
        ext += ['</actuator><sensor><force name="wrist_force" site="wrist_ft_site"/><torque name="wrist_torque" site="wrist_ft_site"/><framepos name="part_rel_pos" objtype="site" objname="part_frame" reftype="site" refname="assembly_frame"/><framequat name="part_rel_quat" objtype="site" objname="part_frame" reftype="site" refname="assembly_frame"/></sensor></mujoco>']
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as file:
            file.write("\n".join(ext)); temporary = Path(file.name)
        try:
            return mujoco.MjModel.from_xml_path(str(temporary))
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _euler_quat(euler: np.ndarray) -> np.ndarray:
        cr, sr = np.cos(euler[0] / 2), np.sin(euler[0] / 2)
        cp, sp = np.cos(euler[1] / 2), np.sin(euler[1] / 2)
        cy, sy = np.cos(euler[2] / 2), np.sin(euler[2] / 2)
        return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])

    def _name2id(self, object_type: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.model, object_type, name)
        if result < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return result

    def _sensor(self, sensor_id: int) -> np.ndarray:
        address, dimension = int(self.model.sensor_adr[sensor_id]), int(self.model.sensor_dim[sensor_id])
        return self.data.sensordata[address:address + dimension].copy()

    def _wrench(self) -> np.ndarray:
        return np.concatenate([self._sensor(self.force_sensor_id), self._sensor(self.torque_sensor_id)]) - self._wrench_bias

    def _true_final_errors(self) -> tuple[float, float]:
        """Physical piece-to-fixture error, unavailable to the actor."""
        target_pos, target_quat = self.path.final_pose
        relative_pos = self._sensor(self.true_rel_pos_sensor_id)
        relative_quat = self._sensor(self.true_rel_quat_sensor_id)
        position_error = float(np.linalg.norm(relative_pos - target_pos))
        rotation_error = _quat_angle(
            _quat_multiply(_quat_inverse(target_quat), relative_quat)
        )
        return position_error, rotation_error

    def _path_qpos(self, progress: float) -> np.ndarray:
        p, q = self.path.pose_at(progress)
        return np.concatenate([p - np.array([0.0, 0.0, 0.20]), _quat_to_euler_xyz(q)])

    def _sample_error(self, linear: float, angular: float) -> np.ndarray:
        return np.concatenate([self.np_random.normal(0.0, linear, 3), self.np_random.normal(0.0, angular, 3)])

    def _apply_physical_errors(self, fixture_error: np.ndarray, grasp_error: np.ndarray) -> None:
        """Update static model transforms without recompiling expensive SDF meshes."""
        fixture_p, fixture_q = fixture_error[:3], self._euler_quat(fixture_error[3:])
        self.model.body_pos[self.fixture_body_id] = fixture_p
        self.model.body_quat[self.fixture_body_id] = fixture_q
        self.model.body_pos[self.grasp_body_id] = grasp_error[:3]
        self.model.body_quat[self.grasp_body_id] = self._euler_quat(grasp_error[3:])
        for name, body_id in self.fixed_body_ids.items():
            p, q = load_place_path(self.paths_dir, name).final_pose
            self.model.body_pos[body_id] = fixture_p + _rotate(fixture_q, p)
            self.model.body_quat[body_id] = _quat_multiply(fixture_q, q)
        # ``mj_forward`` consumes these static body transforms directly.  Do
        # not call ``mj_setConst`` here: it reinitializes SDF plugins and would
        # make every episode pay the mesh-preprocessing cost again.

    def _has_contact(self) -> bool:
        for contact in self.data.contact[:self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.active_geom_id in pair and pair & self.obstacle_geom_ids:
                return True
        return False

    def _update_admittance(self, wrench: np.ndarray) -> None:
        c = self.config
        acceleration = (wrench - c.admittance_damping * self._admittance_velocity - c.admittance_stiffness * self._admittance_offset) / c.admittance_mass
        self._admittance_velocity += np.clip(acceleration, -2.0, 2.0) * c.decision_dt
        self._admittance_velocity = np.clip(self._admittance_velocity, -0.04, 0.04)
        self._admittance_offset += self._admittance_velocity * c.decision_dt
        self._admittance_offset = np.clip(self._admittance_offset, -c.admittance_offset_limit, c.admittance_offset_limit)

    def _base_obs(self, wrench: np.ndarray) -> np.ndarray:
        qpos = self.data.qpos[self.qpos_adr].copy()
        qvel = self.data.qvel[self.dof_adr].copy()
        reference, final = self._path_qpos(self._progress), self._path_qpos(1.0)
        contact = float(self._has_contact())
        return np.concatenate([np.clip((qpos - reference) / np.array([.02, .02, .02, .25, .25, .25]), -5, 5),
                               np.clip((qpos - final) / np.array([.15, .15, .25, .7, .7, .7]), -5, 5),
                               [self._progress, self._progress_rate], qvel / np.array([.10, .10, .10, 1., 1., 1.]),
                               wrench / np.array([50., 50., 50., 5., 5., 5.]), self._previous_action, [contact]])

    def _get_obs(self, wrench: np.ndarray) -> np.ndarray:
        current = self._base_obs(wrench).astype(np.float32)
        if not self._history:
            self._history.extend(current.copy() for _ in range(self.config.history_length))
        else:
            self._history.append(current)
        return np.concatenate(self._history).astype(np.float32)

    def _enter_recovery(self) -> None:
        if self._control_mode == "recovery":
            return
        self._control_mode = "recovery"
        self._recovery_count += 1
        self._recovery_steps = 0
        self._clear_steps = 0
        self._recovery_effort_steps = 0
        self._forced_retreat = False
        self._stuck_detected = True

    def _effective_progress_request(self, requested_progress: float) -> float:
        """Recovery permits holding or retreating, never advancing the path."""
        if self._control_mode != "recovery":
            return requested_progress
        return -1.0 if self._forced_retreat else min(requested_progress, 0.0)

    def _update_recovery_state(
        self,
        *,
        force: float,
        torque: float,
        contact: bool,
        requested_progress: float,
        position_error: float,
        rotation_error: float,
    ) -> None:
        c = self.config
        self._soft_force_steps = (
            self._soft_force_steps + 1 if force > c.soft_force else 0
        )
        self._soft_torque_steps = (
            self._soft_torque_steps + 1 if torque > c.soft_torque else 0
        )
        terminal_pose_unresolved = (
            self._progress >= 0.98
            and (
                position_error >= c.success_position
                or rotation_error >= c.success_rotation
            )
        )
        # At the path endpoint, do not wait for SAC to ask for more forward
        # progress: it may be stuck and holding still with a bad final pose.
        if terminal_pose_unresolved or (
            self._progress >= c.stall_progress and requested_progress > 0.1
        ):
            self._stagnation_errors.append((position_error, rotation_error))
        else:
            self._stagnation_errors.clear()
        stalled = False
        if len(self._stagnation_errors) == self._stagnation_errors.maxlen:
            old_pos, old_rot = self._stagnation_errors[0]
            stalled = (
                old_pos - position_error < c.stall_position_improvement
                and old_rot - rotation_error < c.stall_rotation_improvement
            )
        if self._control_mode == "tracking" and (
            self._soft_force_steps >= c.recovery_force_steps
            or self._soft_torque_steps >= c.recovery_torque_steps
            or stalled
        ):
            self._enter_recovery()
        if self._control_mode != "recovery":
            return
        self._recovery_steps += 1
        self._recovery_duration += c.decision_dt
        effort_is_high = force > c.soft_force or torque > c.soft_torque
        self._recovery_effort_steps = self._recovery_effort_steps + 1 if effort_is_high else 0
        self._forced_retreat = (
            torque > c.soft_torque
            or self._recovery_effort_steps >= c.recovery_effort_persistence_steps
        )
        self._clear_steps = self._clear_steps + 1 if (
            force < c.recovery_clear_force
            and torque < c.recovery_clear_torque
            and not contact
        ) else 0
        if (
            self._recovery_steps >= c.recovery_min_steps
            and self._clear_steps >= c.recovery_clear_steps
        ):
            self._control_mode = "tracking"
            self._soft_force_steps = 0
            self._soft_torque_steps = 0
            self._recovery_effort_steps = 0
            self._forced_retreat = False
            self._stagnation_errors.clear()

    def _recovery_has_failed(self) -> bool:
        return (
            self._control_mode == "recovery"
            and self._recovery_duration >= self.config.recovery_max_duration_s
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        c = self.config
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        mode_used = self._control_mode
        self._previous_progress = self._progress
        requested_progress = float(action[6])
        # A recovery must first free the part; only a retreat or hold is allowed.
        requested_progress = self._effective_progress_request(requested_progress)
        self._progress_rate = requested_progress * c.progress_speed
        self._progress = float(np.clip(self._progress + self._progress_rate * c.decision_dt, 0.0, 1.0))
        velocity_scale = np.array([c.residual_linear_speed] * 3 + [c.residual_angular_speed] * 3)
        residual_action = action[:6].copy()
        if mode_used == "recovery":
            residual_action[3:] = 0.0
        self._residual_offset += residual_action * velocity_scale * c.decision_dt
        limits = np.array([c.residual_linear_limit] * 3 + [c.residual_angular_limit] * 3)
        self._residual_offset = np.clip(self._residual_offset, -limits, limits)
        wrench_before = self._wrench()
        self._update_admittance(wrench_before)
        target = self._path_qpos(self._progress) + self._residual_offset + self._admittance_offset
        ctrl_range = self.model.actuator_ctrlrange[self.actuator_ids]
        self.data.ctrl[self.actuator_ids] = np.clip(target, ctrl_range[:, 0], ctrl_range[:, 1])
        peak_force = 0.0
        peak_torque = 0.0
        soft_torque_stop = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            substep_wrench = self._wrench()
            peak_force = max(peak_force, float(np.linalg.norm(substep_wrench[:3])))
            peak_torque = max(peak_torque, float(np.linalg.norm(substep_wrench[3:])))
            # Do not continue to rotate into a contact for the remaining
            # substeps once even the soft torque threshold is reached.
            soft_torque_stop = peak_torque > c.soft_torque
            if peak_force > c.hard_force or peak_torque > c.hard_torque or soft_torque_stop:
                break
        self._step_count += 1
        wrench = self._wrench()
        force, torque = float(np.linalg.norm(wrench[:3])), float(np.linalg.norm(wrench[3:]))
        self._max_force = max(self._max_force, peak_force)
        self._max_torque = max(self._max_torque, peak_torque)
        contact = self._has_contact()
        self._contact_duration += c.decision_dt if contact else 0.0
        self._impulse += max(0.0, force - c.soft_force) * c.decision_dt
        qpos = self.data.qpos[self.qpos_adr]
        final_error, final_rot_error = self._true_final_errors()
        corridor_error = float(np.linalg.norm(qpos[:3] - self._path_qpos(self._progress)[:3]))
        progress_delta = self._progress - self._previous_progress
        progress_weight = 15.0 * np.clip((0.98 - self._previous_progress) / 0.13, 0.0, 1.0)
        position_improvement = self._previous_true_position_error - final_error
        rotation_improvement = self._previous_true_rotation_error - final_rot_error
        reward = progress_weight * max(progress_delta, 0.0)
        reward += 100.0 * position_improvement + 5.0 * rotation_improvement
        reward -= 2.0 * final_error + 0.10 * final_rot_error
        reward -= 8.0 * max(force - c.soft_force, 0.0) ** 2 / c.soft_force**2
        reward -= 4.0 * max(torque - c.soft_torque, 0.0) ** 2 / c.soft_torque**2
        reward -= 2.0 * max(corridor_error - c.corridor_linear_limit, 0.0) ** 2 / c.corridor_linear_limit**2
        action_cost = .01 * float(np.dot(action, action))
        action_change_cost = .02 * float(np.dot(action - self._previous_action, action - self._previous_action))
        if mode_used == "recovery":
            reward += 0.05 * max(self._previous_force - force, 0.0)
        elif self._progress >= 0.90:
            action_cost *= 2.0
            action_change_cost *= 2.0
        reward -= action_cost + action_change_cost
        hard_force = peak_force > c.hard_force
        hard_torque = peak_torque > c.hard_torque
        unsafe = hard_force or hard_torque
        success = self._progress >= 0.999 and final_error < c.success_position and final_rot_error < c.success_rotation and not unsafe
        if success: reward += 100.0
        if unsafe: reward -= 30.0
        self._update_recovery_state(
            force=force,
            torque=torque,
            contact=contact,
            requested_progress=float(action[6]),
            position_error=final_error,
            rotation_error=final_rot_error,
        )
        self._previous_action = action
        self._previous_true_position_error = final_error
        self._previous_true_rotation_error = final_rot_error
        self._previous_force = force
        recovery_failed = self._recovery_has_failed()
        if recovery_failed:
            reward -= 15.0
        terminated, truncated = bool(success or unsafe or recovery_failed), self._step_count >= self.max_episode_steps
        if success:
            termination_reason = "success"
        elif hard_force and hard_torque:
            termination_reason = "unsafe_force_and_torque"
        elif hard_force:
            termination_reason = "unsafe_force"
        elif hard_torque:
            termination_reason = "unsafe_torque"
        elif recovery_failed:
            termination_reason = "recovery_failed"
        elif truncated:
            termination_reason = "time_limit"
        else:
            termination_reason = "running"
        info = {"is_success": success, "part_name": self.part_name, "path_progress": self._progress, "progress_rate": self._progress_rate,
                "final_position_error_m": final_error, "final_rotation_error_rad": final_rot_error, "force_norm_N": force, "torque_norm_Nm": torque,
                "contact": contact, "contact_duration_s": self._contact_duration, "contact_impulse_Ns": self._impulse, "unsafe_contact": unsafe,
                "max_force_N": self._max_force, "max_torque_Nm": self._max_torque,
                "terminated": terminated, "truncated": truncated, "termination_reason": termination_reason,
                "control_mode": mode_used, "recovery_count": self._recovery_count,
                "recovery_duration_s": self._recovery_duration, "stuck_detected": self._stuck_detected,
                "forced_retreat": self._forced_retreat, "recovery_failed": recovery_failed,
                "soft_torque_stop": soft_torque_stop,
                "residual_offset_norm": float(np.linalg.norm(self._residual_offset)), "config": asdict(c)}
        if self.render_mode == "human": self.render()
        return self._get_obs(wrench), float(reward), terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        fixture_error = self._sample_error(self.config.fixture_linear_error, self.config.fixture_angular_error)
        grasp_error = self._sample_error(self.config.grasp_linear_error, self.config.grasp_angular_error)
        mujoco.mj_resetData(self.model, self.data)
        self._apply_physical_errors(fixture_error, grasp_error)
        self._progress = self._previous_progress = 0.0
        self._progress_rate = 0.0
        self._residual_offset.fill(0); self._admittance_offset.fill(0); self._admittance_velocity.fill(0); self._previous_action.fill(0)
        self._step_count = 0; self._contact_duration = 0.0; self._impulse = 0.0; self._max_force = 0.0; self._max_torque = 0.0; self._history.clear()
        self._control_mode = "tracking"; self._recovery_count = 0; self._recovery_steps = 0; self._recovery_duration = 0.0
        self._soft_force_steps = 0; self._soft_torque_steps = 0; self._recovery_effort_steps = 0; self._forced_retreat = False
        self._clear_steps = 0; self._stuck_detected = False; self._stagnation_errors.clear()
        initial = self._path_qpos(0.0) + self._sample_error(self.config.initial_linear_error, self.config.initial_angular_error)
        self.data.qpos[self.qpos_adr] = initial
        self.data.ctrl[self.actuator_ids] = initial
        mujoco.mj_forward(self.model, self.data)
        for _ in range(30): mujoco.mj_step(self.model, self.data)
        self._wrench_bias = np.concatenate([self._sensor(self.force_sensor_id), self._sensor(self.torque_sensor_id)])
        wrench = self._wrench()
        self._previous_true_position_error, self._previous_true_rotation_error = self._true_final_errors()
        self._previous_force = float(np.linalg.norm(wrench[:3]))
        return self._get_obs(wrench), {"part_name": self.part_name, "path_file": str(self.path.source), "is_success": False}

    def render(self) -> None:
        if self.render_mode != "human": return
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data); self._next_render_time = time.monotonic()
        self._viewer.sync()
        if self._next_render_time is not None:
            self._next_render_time += self._step_wall_time
            delay = self._next_render_time - time.monotonic()
            if delay > 0: time.sleep(delay)
            else: self._next_render_time = time.monotonic()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close(); self._viewer = None
