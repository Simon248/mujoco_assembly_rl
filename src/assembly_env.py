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
from src.transforms import compose, euler_xyz_to_quat, inverse, quat_to_rotvec, relative, rotvec_to_quat, rotate
from src.wrench import contact_wrench_at_site

ROOT = Path(__file__).resolve().parents[1]

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
        load_sdf_plugin()
        self.model = mujoco.MjModel.from_xml_string(self._mjcf())
        self.data = mujoco.MjData(self.model)
        self.mobile_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "tenon_collision")
        self.fixed_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "mortaise")
        self.mobile_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "tenon")
        self.grasp_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "grasp_frame")
        self.target_mocap = self._id(mujoco.mjtObj.mjOBJ_BODY, "grasp_target")
        self.free_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "tenon_free")
        self.qadr = self.model.jnt_qposadr[self.free_joint]
        self.admittance = AdmittanceController(self.cfg["admittance"])
        self.action_space = spaces.Box(-1., 1., (6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)
        self.viewer = None; self.steps = 0; self.last_true_error = np.zeros(6)
        self.perception_bias = (np.zeros(3), np.array([1., 0., 0., 0.]))

    def _id(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0: raise RuntimeError(f"MJCF invalide: {name} est absent")
        return result

    def _mjcf(self) -> str:
        cad = ROOT / "data/input/cad/tenon-mortaise"
        mobile = "tenon_visual.stl" if self.cfg["case"] == "tenon_1" else "tenon-2.stl"
        # Les STL sont déjà exprimés en mètres. Le CSV de grasp reste en
        # millimètres et sa position est convertie dans _load_grasp_pose().
        grasp = self._load_grasp_pose()
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
 </worldbody><equality><weld name="grasp_weld" body1="grasp_target" body2="tenon" relpose="{weld_p[0]} {weld_p[1]} {weld_p[2]} {weld_q[0]} {weld_q[1]} {weld_q[2]} {weld_q[3]}" solref=".02 1"/></equality></mujoco>'''

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
    def _wrench(self):
        wrench = contact_wrench_at_site(self.model, self.data, self.mobile_geom, self.grasp_site)
        return wrench + self.np_random.normal(0, np.asarray(self.cfg["perception"]["wrench_noise_std"],float))
    def _observation(self):
        error, wrench, n = self._error(observed=True), self._wrench(), self.cfg["observation"]
        return np.clip(np.r_[error[:3]/n["position_scale"], error[3:]/n["rotation_scale"], wrench[:3]/n["force_scale"], wrench[3:]/n["torque_scale"]], -20, 20).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.steps = 0; self.admittance.reset()
        mujoco.mj_resetData(self.model, self.data)
        r = self.cfg["randomization"]; initial = self.cfg["initial_pose_fixed_to_mobile"]
        ip = np.array(initial["position"],float) + self.np_random.uniform(-np.array(r["mobile_translation"]), np.array(r["mobile_translation"]))
        iq = compose((np.zeros(3), np.array(initial["orientation_quat"],float)), (np.zeros(3), euler_xyz_to_quat(np.deg2rad(self.np_random.uniform(-np.array(r["mobile_rotation_deg"]), np.array(r["mobile_rotation_deg"]))))))[1]
        fp = self.np_random.uniform(-np.array(r["fixed_translation"]), np.array(r["fixed_translation"]))
        fq = euler_xyz_to_quat(np.deg2rad(self.np_random.uniform(-np.array(r["fixed_rotation_deg"]), np.array(r["fixed_rotation_deg"]))))
        self.model.body_pos[self.fixed_body], self.model.body_quat[self.fixed_body] = fp, fq
        self.data.qpos[self.qadr:self.qadr+3], self.data.qpos[self.qadr+3:self.qadr+7] = compose((fp,fq),(ip,iq))
        mujoco.mj_forward(self.model, self.data)
        # Target mocap is explicitly the grasp pose, not the CAD origin.
        grasp = (self.data.site_xpos[self.grasp_site].copy(), self._site_quat())
        self.data.mocap_pos[self.model.body_mocapid[self.target_mocap]], self.data.mocap_quat[self.model.body_mocapid[self.target_mocap]] = grasp
        p = self.cfg["perception"]
        self.perception_bias = (np.array(p["translation_bias"],float), euler_xyz_to_quat(np.deg2rad(p["rotation_bias_deg"])))
        self.last_true_error = self._error(); return self._observation(), {"true_error": self.last_true_error.copy()}

    def _site_quat(self):
        q=np.zeros(4); mujoco.mju_mat2Quat(q, self.data.site_xmat[self.grasp_site]); return q
    def step(self, action):
        action = np.clip(np.asarray(action,float), -1, 1); a=self.cfg["action"]
        nominal = np.r_[action[:3]*a["max_translation_step"], action[3:]*np.deg2rad(a["max_rotation_step_deg"])]
        delta = self.admittance.step(nominal, self._wrench(), self.control_dt)
        gp, gq = self.data.site_xpos[self.grasp_site].copy(), self._site_quat()
        target = (gp + rotate(gq, delta[:3]), compose((np.zeros(3),gq),(np.zeros(3),rotvec_to_quat(delta[3:])))[1])
        mocap = self.model.body_mocapid[self.target_mocap]; self.data.mocap_pos[mocap], self.data.mocap_quat[mocap] = target
        for _ in range(self.frame_skip): mujoco.mj_step(self.model, self.data)
        self.steps += 1; true_error=self._error(); wrench=self._wrench(); pos=np.linalg.norm(true_error[:3]); rot=np.linalg.norm(true_error[3:])
        previous=np.linalg.norm(self.last_true_error[:3])+0.1*np.linalg.norm(self.last_true_error[3:]); current=pos+0.1*rot; self.last_true_error=true_error
        rw=self.cfg["reward"]; components={"reward_position":-rw["position_weight"]*pos, "reward_orientation":-rw["orientation_weight"]*rot, "reward_progress":rw["progress_weight"]*(previous-current), "reward_force":-rw["force_weight"]*np.linalg.norm(wrench[:3]), "reward_action":-rw["action_weight"]*np.dot(action,action), "reward_success":0.}
        s=self.cfg["success"]; success=pos<s["position_tolerance"] and rot<np.deg2rad(s["rotation_tolerance_deg"]); unsafe=np.linalg.norm(wrench[:3])>s["max_force"] or np.linalg.norm(wrench[3:])>s["max_torque"] or np.linalg.norm(self._pose(self.mobile_body)[0])>s["workspace_radius"]
        if success: components["reward_success"]=rw["success_bonus"]
        terminated=success or unsafe; truncated=self.steps>=self.cfg["simulation"]["max_episode_steps"]
        info={**components,"success":success,"unsafe":unsafe,"true_error":true_error,"position_error":pos,"rotation_error":rot,"force":np.linalg.norm(wrench[:3]),"torque":np.linalg.norm(wrench[3:])}
        if self.render_mode=="human": self.render()
        return self._observation(), float(sum(components.values())), terminated, truncated, info
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
