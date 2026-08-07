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
    terminal_residual_linear_limit: float = 0.024
    residual_angular_limit: float = np.deg2rad(12.0)
    corridor_linear_limit: float = 0.020
    terminal_corridor_linear_limit: float = 0.032
    soft_force: float = 20.0
    soft_torque: float = 4.5
    hard_force: float = 80.0
    hard_torque: float = 8.0
    actuator_force_limits: tuple[float, ...] = (
        250.0,
        250.0,
        300.0,
        30.0,
        30.0,
        30.0,
    )
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
    # MuJoCo's wrist sensor reports the wrench transmitted by the child body
    # to the wrist.  The compliant target must move in the opposite direction
    # to unload that contact.
    admittance_wrench_sign: float = -1.0
    admittance_offset_limit: float = 0.006
    admittance_velocity_limit: float = 0.040
    admittance_acceleration_limit: float = 2.0
    success_position: float = 0.003
    success_rotation: float = np.deg2rad(4.0)
    contact_search_enabled: bool = True
    contact_search_nominal_request: float = 0.25
    contact_search_max_forward_request: float = 0.50
    recovery_enabled: bool = True
    recovery_force_steps: int = 5
    recovery_torque_steps: int = 5
    recovery_min_steps: int = 10
    recovery_clear_steps: int = 10
    recovery_clear_force: float = 15.0
    recovery_clear_torque: float = 3.5
    recovery_effort_persistence_steps: int = 25
    recovery_max_duration_s: float = 2.0
    stall_progress: float = 0.85
    stall_window_steps: int = 50
    stall_position_improvement: float = 0.0005
    stall_rotation_improvement: float = np.deg2rad(0.5)
    dense_reward_limit: float = 0.10
    success_reward: float = 250.0
    unsafe_reward: float = -800.0
    unsafe_force_and_torque_reward: float = -900.0
    recovery_failed_reward: float = -300.0
    time_limit_position_penalty: float = 40.0
    time_limit_rotation_penalty: float = 20.0


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
        self.wrist_site_id = self._name2id(mujoco.mjtObj.mjOBJ_SITE, "wrist_ft_site")
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
        # The controller's integrator states and tactile-search latch are
        # observable. Hidden physical errors and the true part-to-fixture pose
        # deliberately remain unavailable to SAC.
        self._base_observation_size = 56
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
        self._max_progress = 0.0
        self._progress_rate = 0.0
        self._step_count = 0
        self._contact_duration = 0.0
        self._impulse = 0.0
        self._max_force = 0.0
        self._max_torque = 0.0
        self._control_mode = "tracking"
        self._contact_search_count = 0
        self._contact_search_latched = False
        self._contact_search_trigger = "none"
        self._recovery_count = 0
        self._recovery_from_contact_search_count = 0
        self._last_recovery_trigger = "none"
        self._recovery_trigger_contact = False
        self._recovery_trigger_force = 0.0
        self._recovery_trigger_torque = 0.0
        self._recovery_steps = 0
        self._recovery_attempt_duration = 0.0
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
        self._mode_steps = {"tracking": 0, "contact_search": 0, "recovery": 0}
        self._progress_action_sum = 0.0
        self._effective_progress_sum = 0.0
        self._advance_steps = 0
        self._hold_steps = 0
        self._retreat_steps = 0
        self._previous_true_position_error = 0.0
        self._previous_true_rotation_error = 0.0
        self._previous_force = 0.0
        self._last_offset_cost = 0.0
        self._last_dense_reward = 0.0
        self._last_terminal_reward = 0.0
        self._episode_offset_cost = 0.0
        self._episode_dense_reward = 0.0
        self._episode_terminal_reward = 0.0

    def _make_model(self, fixture_error: np.ndarray, grasp_error: np.ndarray) -> mujoco.MjModel:
        fixture_p, fixture_q = fixture_error[:3], self._euler_quat(fixture_error[3:])
        grasp_p, grasp_q = grasp_error[:3], self._euler_quat(grasp_error[3:])
        parts = [f"part_{i}" for i in range(1, int(self.part_name[-1]))]
        ext = ['<mujoco model="residual_place">', '<compiler angle="radian" autolimits="true" meshdir="' + escape(str(self.cad_dir)) + '"/>',
               '<option timestep="0.001" integrator="implicitfast" cone="elliptic" sdf_iterations="10" sdf_initpoints="20"/>',
               '<default><joint damping="2" armature="0.02"/><geom friction="0.9 0.01 0.001" condim="4" solref="0.004 1" solimp="0.90 0.95 0.001"/><position kp="700" kv="45"/></default>',
               '<extension><plugin plugin="mujoco.sdf.sdflib"><instance name="table_sdf"><config key="aabb" value="0"/></instance>']
        ext += [f'<instance name="{name}_sdf"><config key="aabb" value="0"/></instance>' for name in parts + [self.part_name]]
        ext += [
            '</plugin></extension><asset>',
            '<mesh name="table_visual_mesh" file="chandelier_assembly_table_visual.stl"/>',
            '<mesh name="table_collision_mesh" file="chandelier_assembly_table_collision.stl"/>',
        ]
        ext += [f'<mesh name="{name}_mesh" file="chandelier_{name}.stl"/>' for name in parts + [self.part_name]]
        ext += ['</asset><worldbody><light pos="0 -1 1.5" dir="0 1 -1"/><geom type="plane" size="0 0 .05" contype="0" conaffinity="0"/>',
                f'<body name="fixture" pos="{" ".join(map(str, fixture_p))}" quat="{" ".join(map(str, fixture_q))}"><geom type="mesh" mesh="table_visual_mesh" contype="0" conaffinity="0"/><geom name="assembly_table_collision" type="sdf" mesh="table_collision_mesh"><plugin instance="table_sdf"/></geom><site name="assembly_frame" pos="0 0 0" size=".001"/></body>',
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
        actuator_specs = zip(
            ["act_x", "act_y", "act_z", "act_roll", "act_pitch", "act_yaw"],
            self.joint_names if hasattr(self, "joint_names") else ["joint_x", "joint_y", "joint_z", "joint_roll", "joint_pitch", "joint_yaw"],
            ["-.15", "-.15", "-.24", "-.7", "-.7", "-.7"],
            [".15", ".15", ".05", ".7", ".7", ".7"],
            self.config.actuator_force_limits,
        )
        for name, joint, low, high, force_limit in actuator_specs:
            ext += [
                f'<position name="{name}" joint="{joint}" '
                f'ctrlrange="{low} {high}" '
                f'forcerange="-{force_limit} {force_limit}"/>'
            ]
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
        local_wrench = np.concatenate(
            [
                self._sensor(self.force_sensor_id),
                self._sensor(self.torque_sensor_id),
            ]
        ) - self._wrench_bias
        site_to_world = self.data.site_xmat[self.wrist_site_id].reshape(3, 3)
        return np.concatenate(
            [
                site_to_world @ local_wrench[:3],
                site_to_world @ local_wrench[3:],
            ]
        )

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

    def _update_admittance(
        self,
        wrench: np.ndarray,
        *,
        tactile_active: bool = True,
    ) -> None:
        c = self.config
        # In free space the sensor also sees inertial loads from nominal path
        # acceleration. They must not bend the path as if they were contact.
        applied_wrench = wrench if tactile_active else np.zeros(6)
        acceleration = (c.admittance_wrench_sign * applied_wrench - c.admittance_damping * self._admittance_velocity - c.admittance_stiffness * self._admittance_offset) / c.admittance_mass
        self._admittance_velocity += np.clip(
            acceleration,
            -c.admittance_acceleration_limit,
            c.admittance_acceleration_limit,
        ) * c.decision_dt
        self._admittance_velocity = np.clip(
            self._admittance_velocity,
            -c.admittance_velocity_limit,
            c.admittance_velocity_limit,
        )
        self._admittance_offset += self._admittance_velocity * c.decision_dt
        self._admittance_offset = np.clip(self._admittance_offset, -c.admittance_offset_limit, c.admittance_offset_limit)

    def _normalized_controller_state(self) -> np.ndarray:
        """Return observable integrator states, normalized to their limits."""
        c = self.config
        linear_limit = (
            c.terminal_residual_linear_limit
            if self._contact_search_latched
            or self._control_mode in {"contact_search", "recovery"}
            else c.residual_linear_limit
        )
        residual_scale = np.array(
            [linear_limit] * 3 + [c.residual_angular_limit] * 3,
            dtype=np.float64,
        )
        admittance_offset_scale = np.full(6, c.admittance_offset_limit)
        admittance_velocity_scale = np.full(6, c.admittance_velocity_limit)
        return np.concatenate(
            [
                np.clip(self._residual_offset / residual_scale, -1.0, 1.0),
                np.clip(
                    self._admittance_offset / admittance_offset_scale,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    self._admittance_velocity / admittance_velocity_scale,
                    -1.0,
                    1.0,
                ),
            ]
        )

    def _residual_offset_cost(self, linear_limit: float) -> float:
        """Penalize persistent path displacement, not only the latest action."""
        linear = self._residual_offset[:3] / max(linear_limit, 1e-9)
        angular = self._residual_offset[3:] / max(
            self.config.residual_angular_limit,
            1e-9,
        )
        return float(0.01 * np.dot(linear, linear) + 0.005 * np.dot(angular, angular))

    def _is_hard_unsafe(self, force: float, torque: float) -> tuple[bool, bool]:
        """Use inclusive limits so equality cannot slip through one more step."""
        return force >= self.config.hard_force, torque >= self.config.hard_torque

    def _base_obs(self, wrench: np.ndarray) -> np.ndarray:
        qpos = self.data.qpos[self.qpos_adr].copy()
        qvel = self.data.qvel[self.dof_adr].copy()
        reference, final = self._path_qpos(self._progress), self._path_qpos(1.0)
        contact = float(self._has_contact())
        mode = [
            float(self._control_mode == "contact_search"),
            float(self._control_mode == "recovery"),
        ]
        return np.concatenate([np.clip((qpos - reference) / np.array([.02, .02, .02, .25, .25, .25]), -5, 5),
                               np.clip((qpos - final) / np.array([.15, .15, .25, .7, .7, .7]), -5, 5),
                               [self._progress, self._max_progress, self._progress_rate], qvel / np.array([.10, .10, .10, 1., 1., 1.]),
                               wrench / np.array([50., 50., 50., 5., 5., 5.]), self._previous_action, [contact], mode,
                               self._normalized_controller_state(), [float(self._contact_search_latched)]])

    def _get_obs(self, wrench: np.ndarray) -> np.ndarray:
        current = self._base_obs(wrench).astype(np.float32)
        if not self._history:
            self._history.extend(current.copy() for _ in range(self.config.history_length))
        else:
            self._history.append(current)
        return np.concatenate(self._history).astype(np.float32)

    def _enter_contact_search(self, trigger: str = "unspecified") -> bool:
        if not self.config.contact_search_enabled:
            return False
        newly_latched = not self._contact_search_latched
        if newly_latched:
            self._contact_search_latched = True
            self._contact_search_trigger = trigger
        entered_mode = self._control_mode == "tracking"
        if entered_mode:
            self._control_mode = "contact_search"
            self._contact_search_count += 1
            self._stagnation_errors.clear()
        return newly_latched or entered_mode

    def _enter_recovery(
        self,
        trigger: str = "unspecified",
        *,
        contact: bool = False,
        force: float = 0.0,
        torque: float = 0.0,
    ) -> bool:
        if not self.config.recovery_enabled or self._control_mode == "recovery":
            return False
        if self._control_mode == "contact_search":
            self._recovery_from_contact_search_count += 1
        self._control_mode = "recovery"
        self._recovery_count += 1
        self._recovery_steps = 0
        self._recovery_attempt_duration = 0.0
        self._clear_steps = 0
        self._recovery_effort_steps = 0
        self._forced_retreat = False
        self._stuck_detected = True
        self._last_recovery_trigger = trigger
        self._recovery_trigger_contact = contact
        self._recovery_trigger_force = force
        self._recovery_trigger_torque = torque
        return True

    def _effective_progress_request(self, requested_progress: float) -> float:
        """Map the progress action according to the observable control mode.

        In tracking, a zero RL action follows the path at nominal speed. The
        negative half slows to a stop and the positive half can accelerate up
        to 1.5 times nominal speed. Contact search advances more slowly and
        gives most of the action range to holding or retreating. SAC remains
        active at every path position; there is no progress-based action gate.
        """
        if self._control_mode == "tracking":
            if requested_progress <= 0.0:
                return 1.0 + requested_progress
            return 1.0 + 0.5 * requested_progress
        if self._control_mode == "contact_search":
            return float(
                np.clip(
                    self.config.contact_search_nominal_request + requested_progress,
                    -1.0,
                    self.config.contact_search_max_forward_request,
                )
            )
        return -1.0 if self._forced_retreat else min(requested_progress, 0.0)

    def _update_progress_frontier(self, progress: float) -> float:
        """Reward a path interval at most once, even after a retreat."""
        new_progress = max(float(progress) - self._max_progress, 0.0)
        self._max_progress = max(self._max_progress, float(progress))
        return new_progress

    def _contact_progress_scale(self, force: float, torque: float) -> float:
        """Slow only forward contact-search motion between soft and hard limits."""
        c = self.config
        force_ratio = np.clip(
            (force - c.soft_force) / max(c.hard_force - c.soft_force, 1e-9),
            0.0,
            1.0,
        )
        torque_ratio = np.clip(
            (torque - c.soft_torque) / max(c.hard_torque - c.soft_torque, 1e-9),
            0.0,
            1.0,
        )
        return float(1.0 - max(force_ratio, torque_ratio))

    @staticmethod
    def _residual_action_for_mode(action: np.ndarray, mode: str) -> np.ndarray:
        del mode
        return np.asarray(action[:6], dtype=np.float64).copy()

    def _linear_limit_for_state(self, mode: str) -> float:
        if (
            mode in {"contact_search", "recovery"}
            or self._contact_search_latched
        ):
            return self.config.terminal_residual_linear_limit
        return self.config.residual_linear_limit

    def _corridor_limit_for_state(self, mode: str) -> float:
        if (
            mode in {"contact_search", "recovery"}
            or self._contact_search_latched
        ):
            return self.config.terminal_corridor_linear_limit
        return self.config.corridor_linear_limit

    def _record_control_step(
        self,
        mode: str,
        progress_action: float,
        effective_progress: float,
    ) -> None:
        c = self.config
        self._mode_steps[mode] += 1
        self._progress_action_sum += progress_action
        self._effective_progress_sum += effective_progress
        if effective_progress > 0.05:
            self._advance_steps += 1
        elif effective_progress < -0.05:
            self._retreat_steps += 1
        else:
            self._hold_steps += 1
        if mode == "recovery":
            self._recovery_steps += 1
            self._recovery_attempt_duration += c.decision_dt
            self._recovery_duration += c.decision_dt

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
        if not (c.recovery_enabled or c.contact_search_enabled):
            return

        # Tactile evidence can start local search anywhere along the recorded
        # path. The latch deliberately survives loss of contact and retreat.
        if contact:
            self._enter_contact_search("contact")
        tactile_context = contact
        self._soft_force_steps = (
            self._soft_force_steps + 1
            if tactile_context and force >= c.soft_force
            else 0
        )
        self._soft_torque_steps = (
            self._soft_torque_steps + 1
            if tactile_context and torque >= c.soft_torque
            else 0
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
        persistent_force = self._soft_force_steps >= c.recovery_force_steps
        persistent_torque = self._soft_torque_steps >= c.recovery_torque_steps
        trigger = ""
        if persistent_force:
            trigger = "force"
        elif persistent_torque:
            trigger = "torque"
        elif stalled:
            trigger = "stagnation"
        entered_recovery = False
        if self._control_mode != "recovery" and trigger:
            entered_recovery = self._enter_recovery(
                trigger,
                contact=contact,
                force=force,
                torque=torque,
            )
        if self._control_mode != "recovery":
            return

        effort_is_high = tactile_context and (
            force >= c.soft_force or torque >= c.soft_torque
        )
        if entered_recovery:
            self._forced_retreat = effort_is_high
            return
        self._recovery_effort_steps = (
            self._recovery_effort_steps + 1 if effort_is_high else 0
        )
        self._forced_retreat = (
            effort_is_high
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
            self._control_mode = (
                "contact_search"
                if c.contact_search_enabled and self._contact_search_latched
                else "tracking"
            )
            self._soft_force_steps = 0
            self._soft_torque_steps = 0
            self._recovery_effort_steps = 0
            self._forced_retreat = False
            self._stagnation_errors.clear()

    def _recovery_has_failed(self) -> bool:
        return (
            self._control_mode == "recovery"
            and self._recovery_attempt_duration >= self.config.recovery_max_duration_s
        )

    def _run_control_substeps(self) -> tuple[float, float, bool]:
        """Advance a full decision unless a hard safety threshold is reached."""
        c = self.config
        peak_force = 0.0
        peak_torque = 0.0
        contact_detected = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            contact_detected = contact_detected or self._has_contact()
            substep_wrench = self._wrench()
            peak_force = max(
                peak_force,
                float(np.linalg.norm(substep_wrench[:3])),
            )
            peak_torque = max(
                peak_torque,
                float(np.linalg.norm(substep_wrench[3:])),
            )
            hard_force, hard_torque = self._is_hard_unsafe(
                peak_force,
                peak_torque,
            )
            if hard_force or hard_torque:
                break
        return peak_force, peak_torque, contact_detected

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        c = self.config
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        mode_used = self._control_mode
        self._previous_progress = self._progress
        progress_action = float(action[6])
        wrench_before = self._wrench()
        force_before = float(np.linalg.norm(wrench_before[:3]))
        torque_before = float(np.linalg.norm(wrench_before[3:]))
        progress_intention = self._effective_progress_request(progress_action)
        progress_scale = 1.0
        if mode_used == "contact_search" and progress_intention > 0.0:
            progress_scale = self._contact_progress_scale(
                force_before,
                torque_before,
            )
        requested_progress = progress_intention * progress_scale
        self._record_control_step(mode_used, progress_action, requested_progress)
        self._progress_rate = requested_progress * c.progress_speed
        self._progress = float(np.clip(self._progress + self._progress_rate * c.decision_dt, 0.0, 1.0))
        previous_progress_frontier = self._max_progress
        new_progress = self._update_progress_frontier(self._progress)
        velocity_scale = np.array([c.residual_linear_speed] * 3 + [c.residual_angular_speed] * 3)
        residual_action = self._residual_action_for_mode(action, mode_used)
        self._residual_offset += residual_action * velocity_scale * c.decision_dt
        linear_limit = self._linear_limit_for_state(mode_used)
        limits = np.array([linear_limit] * 3 + [c.residual_angular_limit] * 3)
        self._residual_offset = np.clip(self._residual_offset, -limits, limits)
        self._update_admittance(
            wrench_before,
            tactile_active=self._has_contact(),
        )
        target = self._path_qpos(self._progress) + self._residual_offset + self._admittance_offset
        ctrl_range = self.model.actuator_ctrlrange[self.actuator_ids]
        self.data.ctrl[self.actuator_ids] = np.clip(target, ctrl_range[:, 0], ctrl_range[:, 1])
        peak_force, peak_torque, contact_during_substeps = self._run_control_substeps()
        self._step_count += 1
        wrench = self._wrench()
        force, torque = float(np.linalg.norm(wrench[:3])), float(np.linalg.norm(wrench[3:]))
        self._max_force = max(self._max_force, peak_force)
        self._max_torque = max(self._max_torque, peak_torque)
        contact = contact_during_substeps or self._has_contact()
        self._contact_duration += c.decision_dt if contact else 0.0
        # A hard/brief contact can disappear before the end-of-decision sensor
        # sample, so use the substep peak for the conservative impulse metric.
        self._impulse += max(0.0, peak_force - c.soft_force) * c.decision_dt
        qpos = self.data.qpos[self.qpos_adr]
        final_error, final_rot_error = self._true_final_errors()
        corridor_error = float(np.linalg.norm(qpos[:3] - self._path_qpos(self._progress)[:3]))
        corridor_limit = self._corridor_limit_for_state(mode_used)
        progress_weight = 15.0 * np.clip(
            (0.98 - previous_progress_frontier) / (0.98 - 0.85),
            0.0,
            1.0,
        )
        position_improvement = self._previous_true_position_error - final_error
        rotation_improvement = self._previous_true_rotation_error - final_rot_error
        # Early in the path, true-pose improvement merely duplicates nominal
        # progress and would saturate the clipped reward. It takes over only
        # near the goal, leaving room for action/offset costs in free space.
        pose_focus = float(
            np.clip(
                (self._previous_progress - c.stall_progress)
                / max(0.98 - c.stall_progress, 1e-9),
                0.0,
                1.0,
            )
        )
        task_reward = progress_weight * new_progress
        task_reward += pose_focus * (
            500.0 * position_improvement + 20.0 * rotation_improvement
        )
        force_excess = np.clip(
            (peak_force - c.soft_force) / max(c.hard_force - c.soft_force, 1e-9),
            0.0,
            1.0,
        )
        torque_excess = np.clip(
            (peak_torque - c.soft_torque) / max(c.hard_torque - c.soft_torque, 1e-9),
            0.0,
            1.0,
        )
        effort_weight = 0.25 if mode_used in {"contact_search", "recovery"} else 0.5
        effort_cost = effort_weight * (force_excess**2 + torque_excess**2)
        corridor_excess = max(corridor_error - corridor_limit, 0.0)
        corridor_cost = 0.1 * (
            corridor_excess / max(corridor_limit, 1e-9)
        ) ** 2
        action_cost = 0.002 * float(np.dot(action, action))
        action_delta = action - self._previous_action
        action_change_cost = 0.005 * float(np.dot(action_delta, action_delta))
        offset_cost = self._residual_offset_cost(linear_limit)
        if mode_used == "recovery":
            task_reward += 0.05 * float(
                np.clip((self._previous_force - force) / c.soft_force, 0.0, 1.0)
            )
        task_reward = float(
            np.clip(task_reward, -c.dense_reward_limit, c.dense_reward_limit)
        )
        dense_reward = task_reward - (
            effort_cost
            + corridor_cost
            + action_cost
            + action_change_cost
            + offset_cost
        )
        dense_reward = float(
            np.clip(dense_reward, -c.dense_reward_limit, c.dense_reward_limit)
        )
        hard_force, hard_torque = self._is_hard_unsafe(
            peak_force,
            peak_torque,
        )
        unsafe = hard_force or hard_torque
        success = self._progress >= 0.999 and final_error < c.success_position and final_rot_error < c.success_rotation and not unsafe
        self._update_recovery_state(
            force=max(force, peak_force),
            torque=max(torque, peak_torque),
            contact=contact,
            requested_progress=progress_intention,
            position_error=final_error,
            rotation_error=final_rot_error,
        )
        self._previous_action = action
        self._previous_true_position_error = final_error
        self._previous_true_rotation_error = final_rot_error
        self._previous_force = force
        recovery_failed = self._recovery_has_failed()
        terminated = bool(success or unsafe or recovery_failed)
        truncated = bool(
            not terminated and self._step_count >= self.max_episode_steps
        )
        terminal_reward = 0.0
        if success:
            terminal_reward = c.success_reward
        elif hard_force and hard_torque:
            terminal_reward = c.unsafe_force_and_torque_reward
        elif unsafe:
            terminal_reward = c.unsafe_reward
        elif recovery_failed:
            terminal_reward = c.recovery_failed_reward
        elif truncated:
            position_quality = np.clip(
                (final_error - c.success_position) / max(0.030 - c.success_position, 1e-9),
                0.0,
                1.0,
            )
            rotation_quality = np.clip(
                (final_rot_error - c.success_rotation)
                / max(np.deg2rad(20.0) - c.success_rotation, 1e-9),
                0.0,
                1.0,
            )
            terminal_reward = -float(
                c.time_limit_position_penalty * position_quality
                + c.time_limit_rotation_penalty * rotation_quality
            )
        reward = dense_reward + terminal_reward
        self._last_offset_cost = offset_cost
        self._last_dense_reward = dense_reward
        self._last_terminal_reward = terminal_reward
        self._episode_offset_cost += offset_cost
        self._episode_dense_reward += dense_reward
        self._episode_terminal_reward += terminal_reward
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
        recorded_steps = max(sum(self._mode_steps.values()), 1)
        tracking_duration = self._mode_steps["tracking"] * c.decision_dt
        contact_search_duration = self._mode_steps["contact_search"] * c.decision_dt
        soft_effort_exceeded = peak_force >= c.soft_force or peak_torque >= c.soft_torque
        info = {"is_success": success, "part_name": self.part_name, "path_progress": self._progress, "max_path_progress": self._max_progress, "progress_rate": self._progress_rate,
                "progress_action": progress_action, "progress_intention": progress_intention,
                "progress_scale": progress_scale, "effective_progress_request": requested_progress,
                "mean_progress_action": self._progress_action_sum / recorded_steps,
                "mean_effective_progress_request": self._effective_progress_sum / recorded_steps,
                "advance_fraction": self._advance_steps / recorded_steps,
                "hold_fraction": self._hold_steps / recorded_steps,
                "retreat_fraction": self._retreat_steps / recorded_steps,
                "final_position_error_m": final_error, "final_rotation_error_rad": final_rot_error, "force_norm_N": force, "torque_norm_Nm": torque,
                "contact": contact, "contact_duration_s": self._contact_duration, "contact_impulse_Ns": self._impulse, "unsafe_contact": unsafe,
                "max_force_N": self._max_force, "max_torque_Nm": self._max_torque,
                "terminated": terminated, "truncated": truncated, "termination_reason": termination_reason,
                "control_mode": mode_used, "next_control_mode": self._control_mode,
                "contact_search_count": self._contact_search_count,
                "contact_search_latched": self._contact_search_latched,
                "contact_search_trigger": self._contact_search_trigger,
                "contact_search_duration_s": contact_search_duration,
                "contact_search_fraction": self._mode_steps["contact_search"] / recorded_steps,
                "tracking_duration_s": tracking_duration,
                "tracking_fraction": self._mode_steps["tracking"] / recorded_steps,
                "recovery_count": self._recovery_count,
                "recovery_from_contact_search_count": self._recovery_from_contact_search_count,
                "recovery_duration_s": self._recovery_duration,
                "recovery_fraction": self._mode_steps["recovery"] / recorded_steps,
                "recovery_trigger": self._last_recovery_trigger,
                "recovery_trigger_contact": self._recovery_trigger_contact,
                "recovery_trigger_force_N": self._recovery_trigger_force,
                "recovery_trigger_torque_Nm": self._recovery_trigger_torque,
                "stuck_detected": self._stuck_detected,
                "recovery_attempt_duration_s": self._recovery_attempt_duration,
                "forced_retreat": self._forced_retreat, "recovery_failed": recovery_failed,
                "soft_effort_exceeded": soft_effort_exceeded,
                "terminal_linear_limit_m": linear_limit,
                "residual_linear_offset_m": float(np.linalg.norm(self._residual_offset[:3])),
                "residual_angular_offset_rad": float(np.linalg.norm(self._residual_offset[3:])),
                "admittance_linear_offset_m": float(np.linalg.norm(self._admittance_offset[:3])),
                "admittance_angular_offset_rad": float(np.linalg.norm(self._admittance_offset[3:])),
                "offset_cost": offset_cost,
                "dense_reward": dense_reward,
                "terminal_reward": terminal_reward,
                "episode_offset_cost": self._episode_offset_cost,
                "episode_dense_reward": self._episode_dense_reward,
                "episode_terminal_reward": self._episode_terminal_reward,
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
        self._max_progress = 0.0
        self._progress_rate = 0.0
        self._residual_offset.fill(0); self._admittance_offset.fill(0); self._admittance_velocity.fill(0); self._previous_action.fill(0)
        self._step_count = 0; self._contact_duration = 0.0; self._impulse = 0.0; self._max_force = 0.0; self._max_torque = 0.0; self._history.clear()
        self._control_mode = "tracking"; self._contact_search_count = 0
        self._contact_search_latched = False; self._contact_search_trigger = "none"
        self._recovery_count = 0; self._recovery_from_contact_search_count = 0
        self._last_recovery_trigger = "none"; self._recovery_steps = 0
        self._recovery_trigger_contact = False
        self._recovery_trigger_force = 0.0; self._recovery_trigger_torque = 0.0
        self._recovery_attempt_duration = 0.0; self._recovery_duration = 0.0
        self._soft_force_steps = 0; self._soft_torque_steps = 0; self._recovery_effort_steps = 0; self._forced_retreat = False
        self._clear_steps = 0; self._stuck_detected = False; self._stagnation_errors.clear()
        self._mode_steps = {"tracking": 0, "contact_search": 0, "recovery": 0}
        self._progress_action_sum = 0.0; self._effective_progress_sum = 0.0
        self._advance_steps = 0; self._hold_steps = 0; self._retreat_steps = 0
        self._last_offset_cost = 0.0; self._last_dense_reward = 0.0; self._last_terminal_reward = 0.0
        self._episode_offset_cost = 0.0; self._episode_dense_reward = 0.0; self._episode_terminal_reward = 0.0
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
