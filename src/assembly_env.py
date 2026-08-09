"""Environnement Gymnasium unique pour les essais tenon--mortaise.

Convention centrale: T_fixed_to_mobile décrit la pose du repère CAD du tenon
dans le repère CAD de la mortaise. L'erreur est T_target^-1 T_relative.
"""
from __future__ import annotations
from pathlib import Path
import time
from xml.sax.saxutils import escape
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from src.admittance import AdmittanceController
from src.config import load_config
from src.mujoco_plugins import load_sdf_plugin
from src.task_logic import (
    assess_status, newly_reached_milestones, prepare_proximity_milestones,
    reward_components, satisfied_milestone_indices,
)
from src.transforms import compose, euler_xyz_to_quat, inv, inverse, quat_to_rotvec, relative, rotvec_to_quat, rotate
from src.wrench import contact_wrench_at_site

ROOT = Path(__file__).resolve().parents[1]


def apply_action_delta(
    base_grasp_pose: tuple[np.ndarray, np.ndarray],
    task_to_grasp: tuple[np.ndarray, np.ndarray],
    delta_pose: tuple[np.ndarray, np.ndarray],
    action_frame: str,
    task_target: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Applique le delta d'action à une pose grasp, sans choisir sa provenance."""
    if action_frame == "grasp":
        return compose(base_grasp_pose, delta_pose)
    if action_frame == "task":
        if task_target is None:
            raise ValueError("task_target est requis avec action_frame='task'")
        task_reference = compose(base_grasp_pose, inverse(task_to_grasp))
        reference_in_target = relative(task_target, task_reference)
        desired_in_target = (
            reference_in_target[0] + delta_pose[0],
            compose(
                (np.zeros(3), delta_pose[1]),
                (np.zeros(3), reference_in_target[1]),
            )[1],
        )
        task_desired = compose(task_target, desired_in_target)
        return compose(task_desired, task_to_grasp)
    raise ValueError("action.action_frame doit être 'task' ou 'grasp'")


# Nom historique conservé pour les utilisateurs externes et les anciens tests.
advance_grasp_reference = apply_action_delta


def admittance_change_pose(
    previous_offset: np.ndarray, new_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne T_old_offset_to_new_offset, rotations incluses via SE(3)."""
    old_pose = (previous_offset[:3], rotvec_to_quat(previous_offset[3:]))
    new_pose = (new_offset[:3], rotvec_to_quat(new_offset[3:]))
    return relative(old_pose, new_pose)


class TenonMortaiseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    def __init__(self, config_path: str | Path = "configs/test1.yaml", render_mode: str | None = None,
                 render_speed: float = 1.0):
        if render_mode not in (None, "human"): raise ValueError("render_mode doit être None ou human")
        if render_speed <= 0: raise ValueError("render_speed doit être strictement positif")
        self.cfg, self.render_mode = load_config(config_path), render_mode
        self.render_speed = float(render_speed)
        self.control_dt = float(self.cfg["simulation"]["control_dt"])
        self.frame_skip = round(self.control_dt / self.cfg["simulation"]["timestep"])
        if self.frame_skip < 1: raise ValueError("control_dt doit être >= timestep")
        self.task_to_grasp = self._load_grasp_pose()
        load_sdf_plugin()
        self.model = mujoco.MjModel.from_xml_string(self._mjcf())
        self.data = mujoco.MjData(self.model)
        self.mobile_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "tenon_collision")
        self.fixed_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "mortaise_collision")
        self.fixed_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "mortaise")
        self.mobile_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "tenon")
        self.grasp_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "grasp_frame")
        self.target_mocap = self._id(mujoco.mjtObj.mjOBJ_BODY, "grasp_target")
        self.free_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "tenon_free")
        self.qadr = self.model.jnt_qposadr[self.free_joint]
        self.admittance = AdmittanceController(self.cfg["admittance"])
        self.proximity_milestones = prepare_proximity_milestones(
            self.cfg["reward"].get("proximity_milestones", [])
        )
        self._contact_geom_ids = np.array([self.fixed_geom, self.mobile_geom])
        self._base_contact_friction = self.model.geom_friction[self._contact_geom_ids].copy()
        self.action_space = spaces.Box(-1., 1., (6,), dtype=np.float32)
        # pose error 6D + wrench 6D + optional admittance position offset 6D.
        # The offset is controller-owned and can also be reconstructed on the
        # real robot; no simulator or admittance velocity is observed.
        self.include_admittance_position = bool(
            self.cfg["observation"]["include_admittance_position"]
        )
        observation_size = 12 + (6 if self.include_admittance_position else 0)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (observation_size,), dtype=np.float32
        )
        self.viewer = None; self.steps = 0; self.last_true_error = np.zeros(6)
        self.perception_bias = (np.zeros(3), np.array([1., 0., 0., 0.]))
        # État réservé au mode historique; le mode réactif n'en dépend pas.
        self.reference_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.episode_max_force = 0.0; self.episode_max_torque = 0.0
        self.episode_reward_components: dict[str, float] = {}
        self.proximity_milestones_reached: set[int] = set()
        self.friction_scale = 1.0
        # Set by the periodic evaluation callback; zero during training.
        self.training_timesteps = 0

    def _id(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0: raise RuntimeError(f"MJCF invalide: {name} est absent")
        return result

    def _mjcf(self) -> str:
        cad = ROOT / "data/input/cad/tenon-mortaise"
        mobile = "tenon_visual.stl" if self.cfg["case"] == "tenon_1" else "tenon-2.stl"
        # Les STL sont déjà exprimés en mètres. Le CSV de grasp reste en
        # millimètres et sa position est convertie dans _load_grasp_pose().
        grasp = self.task_to_grasp
        p, q = grasp
        weld_p, weld_q = inverse(grasp)  # T_grasp_to_mobile pour le weld mocap.
        return f'''<mujoco model="tenon_mortaise">
 <compiler angle="radian" meshdir="{escape(str(cad))}" autolimits="true"/>
 <option timestep="{self.cfg['simulation']['timestep']}" gravity="0 0 -9.81" integrator="implicitfast"/>
 <default><geom friction="0.8 0.01 0.001" condim="4" solref="0.004 1"/><joint damping="1"/></default>
 <extension><plugin plugin="mujoco.sdf.sdflib"><instance name="mortaise_sdf"><config key="aabb" value="0"/></instance><instance name="tenon_sdf"><config key="aabb" value="0"/></instance></plugin></extension>
 <asset><mesh name="mortaise_mesh" file="mortaise_visual.stl"/><mesh name="tenon_mesh" file="{mobile}"/></asset>
 <worldbody><light pos="0 -0.5 0.8"/><camera name="overview" pos=".34 -.42 .28" xyaxes=".78 .62 0 -.30 .38 .87"/><geom type="plane" size="1 1 .1" pos="0 0 -.1"/>
 <body name="mortaise"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/><geom type="mesh" mesh="mortaise_mesh" contype="0" conaffinity="0" rgba=".45 .45 .5 1"/><geom name="mortaise_collision" type="sdf" mesh="mortaise_mesh" rgba="0 0 0 0"><plugin instance="mortaise_sdf"/></geom><site name="fixed_frame" size=".008" rgba="0 1 0 1"/></body>
 <body name="grasp_target" mocap="true"/>
 <body name="tenon"><freejoint name="tenon_free"/><geom type="mesh" mesh="tenon_mesh" contype="0" conaffinity="0" rgba=".9 .55 .1 1"/><geom name="tenon_collision" type="sdf" mesh="tenon_mesh" mass=".2" rgba="0 0 0 0"><plugin instance="tenon_sdf"/></geom><site name="mobile_frame" size=".008" rgba="1 1 0 1"/><site name="grasp_frame" pos="{p[0]} {p[1]} {p[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}" size=".012" rgba="1 0 0 1"/></body>
 </worldbody><equality><weld name="grasp_weld" body1="grasp_target" body2="tenon" relpose="{weld_p[0]} {weld_p[1]} {weld_p[2]} {weld_q[0]} {weld_q[1]} {weld_q[2]} {weld_q[3]}" solref=".004 1"/></equality></mujoco>'''

    def _load_grasp_pose(self):
        csv = ROOT / "data/input/grasp_poses/tenon/valid_poses.csv"
        row = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8", max_rows=1)
        position = np.array([row["pos_x"], row["pos_y"], row["pos_z"]], float) * .001
        matrix = np.array([[row[f"ori_{i}{j}"] for j in range(3)] for i in range(3)])
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, matrix.ravel())
        return position, quat

    def _pose(self, body): return self.data.xpos[body].copy(), self.data.xquat[body].copy()
    def _target(self):
        d = self.cfg["target_pose_fixed_to_mobile"]; return np.array(d["position"],float), np.array(d["orientation_quat"],float)
    def _error(self, observed=False):
        fixed_mobile = relative(self._pose(self.fixed_body), self._pose(self.mobile_body))
        if observed:
            p = self.cfg["perception"]
            noise = (self.np_random.normal(0, p["translation_noise_std"], 3),
                     rotvec_to_quat(self.np_random.normal(0, np.deg2rad(p["rotation_noise_std_deg"]), 3)))
            fixed_mobile = compose(compose(fixed_mobile, self.perception_bias), noise)
        err = relative(self._target(), fixed_mobile)
        return np.r_[err[0], quat_to_rotvec(err[1])]
    def _true_wrench(self):
        return contact_wrench_at_site(self.model, self.data, self.mobile_geom, self.grasp_site)
    def _observed_wrench(self):
        return self._true_wrench() + self.np_random.normal(
            0, np.asarray(self.cfg["perception"]["wrench_noise_std"], float)
        )
    def _observation(self):
        error, wrench, n = self._error(observed=True), self._observed_wrench(), self.cfg["observation"]
        values = [
            error[:3] / n["position_scale"],
            error[3:] / n["rotation_scale"],
            wrench[:3] / n["force_scale"],
            wrench[3:] / n["torque_scale"],
        ]
        if self.include_admittance_position:
            # Translation and rotation-vector use the existing reference frame;
            # max_offset gives a natural unit scale without changing its limits.
            values.append(self.admittance.offset / self.admittance.offset_limit)
        return np.clip(np.concatenate(values), -20, 20).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.steps = 0; self.admittance.reset()
        self.episode_max_force = 0.0; self.episode_max_torque = 0.0
        self.episode_reward_components = {}
        mujoco.mj_resetData(self.model, self.data)
        r = self.cfg["randomization"]; initial = self.cfg["initial_pose_fixed_to_mobile"]
        ip = np.array(initial["position"],float) + self.np_random.uniform(-np.array(r["mobile_translation"]), np.array(r["mobile_translation"]))
        iq = compose((np.zeros(3), np.array(initial["orientation_quat"],float)), (np.zeros(3), euler_xyz_to_quat(np.deg2rad(self.np_random.uniform(-np.array(r["mobile_rotation_deg"]), np.array(r["mobile_rotation_deg"]))))))[1]
        fp = self.np_random.uniform(-np.array(r["fixed_translation"]), np.array(r["fixed_translation"]))
        fq = euler_xyz_to_quat(np.deg2rad(self.np_random.uniform(-np.array(r["fixed_rotation_deg"]), np.array(r["fixed_rotation_deg"]))))
        self.model.body_pos[self.fixed_body], self.model.body_quat[self.fixed_body] = fp, fq
        self.friction_scale = self._randomize_friction(r)
        self.data.qpos[self.qadr:self.qadr+3], self.data.qpos[self.qadr+3:self.qadr+7] = compose((fp,fq),(ip,iq))
        mujoco.mj_forward(self.model, self.data)
        # Target mocap is explicitly the grasp pose, not the CAD origin.
        grasp = (self.data.site_xpos[self.grasp_site].copy(), self._site_quat())
        self.reference_pose = (
            (grasp[0].copy(), grasp[1].copy())
            if self.cfg["action"]["control_mode"] == "accumulated_reference"
            else None
        )
        self.data.mocap_pos[self.model.body_mocapid[self.target_mocap]], self.data.mocap_quat[self.model.body_mocapid[self.target_mocap]] = grasp
        p = self.cfg["perception"]
        self.perception_bias = (np.array(p["translation_bias"],float), euler_xyz_to_quat(np.deg2rad(p["rotation_bias_deg"])))
        self.last_true_error = self._error()
        initial_position_error = float(np.linalg.norm(self.last_true_error[:3]))
        initial_rotation_error = float(np.linalg.norm(self.last_true_error[3:]))
        self.proximity_milestones_reached = satisfied_milestone_indices(
            initial_position_error, initial_rotation_error,
            self.proximity_milestones,
        )
        return self._observation(), {
            "true_error": self.last_true_error.copy(), "friction_scale": self.friction_scale
        }

    def _site_quat(self):
        q=np.zeros(4); mujoco.mju_mat2Quat(q, self.data.site_xmat[self.grasp_site]); return q

    def _randomize_friction(self, randomization: dict) -> float:
        low, high = map(float, randomization["friction_scale"])
        scale = float(self.np_random.uniform(low, high))
        self.model.geom_friction[self._contact_geom_ids] = self._base_contact_friction * scale
        return scale

    def _run_control_substeps(self, safety: dict) -> tuple[float, float]:
        """Avance MuJoCo en latchant les pics vrais et en stoppant au seuil dur."""
        max_force = 0.0; max_torque = 0.0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            wrench = self._true_wrench()
            max_force = max(max_force, float(np.linalg.norm(wrench[:3])))
            max_torque = max(max_torque, float(np.linalg.norm(wrench[3:])))
            if (max_force >= safety["max_force"] or
                    max_torque >= safety["max_torque"] or
                    np.linalg.norm(self._error()[:3]) >= safety["workspace_radius"]):
                break
        return max_force, max_torque

    def step(self, action):
        action = np.clip(np.asarray(action,float), -1, 1); a=self.cfg["action"]
        nominal = np.r_[action[:3]*a["max_translation_step"], action[3:]*np.deg2rad(a["max_rotation_step_deg"])]
        delta_pose = (nominal[:3], rotvec_to_quat(nominal[3:]))
        actual_grasp_pose = (
            self.data.site_xpos[self.grasp_site].copy(), self._site_quat(),
        )
        task_target = compose(self._pose(self.fixed_body), self._target())
        historical_mode = a["control_mode"] == "accumulated_reference"
        if historical_mode and self.reference_pose is None:
            raise RuntimeError("reset() doit être appelé avant step()")
        command_base = self.reference_pose if historical_mode else actual_grasp_pose
        nominal_target = apply_action_delta(
            command_base, self.task_to_grasp, delta_pose, a["action_frame"],
            task_target,
        )
        if historical_mode:
            self.reference_pose = nominal_target

        observed_wrench = self._observed_wrench()
        previous_admittance_offset = self.admittance.offset.copy()
        if historical_mode:
            # Compatibilité exacte : le wrench du grasp réel est exprimé dans
            # la référence nominale historique qui porte l'offset absolu.
            reference_q = self.reference_pose[1]
            wrench_for_admittance = np.r_[
                rotate(inv(reference_q), rotate(actual_grasp_pose[1], observed_wrench[:3])),
                rotate(inv(reference_q), rotate(actual_grasp_pose[1], observed_wrench[3:])),
            ]
        else:
            # Le capteur et l'admittance utilisent les axes observables du
            # grasp réel courant : aucune référence historique cachée.
            wrench_for_admittance = observed_wrench
        new_admittance_offset = self.admittance.step(wrench_for_admittance, self.control_dt)
        if historical_mode:
            admittance_pose = (
                new_admittance_offset[:3], rotvec_to_quat(new_admittance_offset[3:]),
            )
        else:
            admittance_pose = admittance_change_pose(
                previous_admittance_offset, new_admittance_offset,
            )
        target = compose(nominal_target, admittance_pose)
        mocap = self.model.body_mocapid[self.target_mocap]; self.data.mocap_pos[mocap], self.data.mocap_quat[mocap] = target
        safety = self.cfg["success"]
        step_max_force, step_max_torque = self._run_control_substeps(safety)

        self.steps += 1
        self.episode_max_force = max(self.episode_max_force, step_max_force)
        self.episode_max_torque = max(self.episode_max_torque, step_max_torque)
        true_error = self._error(); pos = float(np.linalg.norm(true_error[:3])); rot = float(np.linalg.norm(true_error[3:]))
        previous_pos = float(np.linalg.norm(self.last_true_error[:3])); previous_rot = float(np.linalg.norm(self.last_true_error[3:])); self.last_true_error = true_error
        status = assess_status(
            position_error=pos, rotation_error=rot,
            max_force=step_max_force, max_torque=step_max_torque,
            workspace_error=pos, step_count=self.steps,
            config=safety, max_episode_steps=self.cfg["simulation"]["max_episode_steps"],
        )
        proximity_bonus, self.proximity_milestones_reached = newly_reached_milestones(
            pos, rot, self.proximity_milestones,
            self.proximity_milestones_reached,
        )
        components = reward_components(
            position_error=pos, rotation_error=rot,
            previous_position_error=previous_pos, previous_rotation_error=previous_rot,
            max_force=step_max_force, action=action,
            status=status, config=self.cfg["reward"], action_config=a,
            max_torque=step_max_torque,
            proximity_bonus=proximity_bonus,
        )
        for key, value in components.items():
            self.episode_reward_components[key] = self.episode_reward_components.get(key, 0.0) + value
        final_wrench = self._true_wrench()
        info = {
            **components,
            "geometric_success": status.geometric_success,
            "success": status.success,
            "safe_success": status.success,
            "unsafe": status.unsafe,
            "unsafe_force": status.unsafe_force,
            "unsafe_torque": status.unsafe_torque,
            "unsafe_workspace": status.unsafe_workspace,
            "termination_reason": status.termination_reason,
            "true_error": true_error,
            "position_error": pos,
            "rotation_error": rot,
            "position_error_x": float(true_error[0]),
            "position_error_y": float(true_error[1]),
            "position_error_z": float(true_error[2]),
            "rotation_error_x": float(true_error[3]),
            "rotation_error_y": float(true_error[4]),
            "rotation_error_z": float(true_error[5]),
            "action_x": float(action[0]),
            "action_y": float(action[1]),
            "action_z": float(action[2]),
            "action_rx": float(action[3]),
            "action_ry": float(action[4]),
            "action_rz": float(action[5]),
            "force": float(np.linalg.norm(final_wrench[:3])),
            "torque": float(np.linalg.norm(final_wrench[3:])),
            "max_force_substep": step_max_force,
            "max_torque_substep": step_max_torque,
            "episode_max_force": self.episode_max_force,
            "episode_max_torque": self.episode_max_torque,
            "final_position_error": pos,
            "final_rotation_error": rot,
            "max_force": self.episode_max_force,
            "max_torque": self.episode_max_torque,
            "training_timesteps": self.training_timesteps,
            "friction_scale": self.friction_scale,
            "proximity_milestones_reached": len(self.proximity_milestones_reached),
            "proximity_milestones_total": len(self.proximity_milestones),
        }
        info.update({f"episode_{key}": value for key, value in self.episode_reward_components.items()})
        if self.render_mode=="human": self.render()
        return self._observation(), float(sum(components.values())), status.terminated, status.truncated, info
    def render(self):
        if self.viewer is None:
            import mujoco.viewer; self.viewer=mujoco.viewer.launch_passive(self.model,self.data)
            # Start every evaluation with a useful view of the complete CAD assembly.
            self.viewer.cam.lookat[:] = self.data.xpos[self.fixed_body]
            self.viewer.cam.distance = .45
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25
        self.viewer.sync()
        # A human evaluation is a diagnostic tool: keep MuJoCo at the control rate.
        time.sleep(self.control_dt / self.render_speed)
    def close(self):
        if self.viewer is not None: self.viewer.close(); self.viewer=None

# Alias explicite pour les scripts externes.
AssemblyEnv = TenonMortaiseEnv
