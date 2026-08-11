"""Environnement Gymnasium unique pour les essais tenon--mortaise.

Convention centrale: T_fixed_to_mobile décrit la pose du repère CAD du tenon
dans le repère CAD de la mortaise. L'erreur est T_target^-1 T_relative.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import time
from xml.sax.saxutils import escape
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from src.admittance import AdmittanceController
from src.config import load_config
from src.curriculum import (
    CurriculumGenerationResult, CurriculumResetSelection, CurriculumState,
    PhysicsStepResult, configured_start_sampling_probabilities,
    historical_quantile_bins, select_training_start,
)
from src.mujoco_plugins import load_sdf_plugin
from src.task_logic import assess_status, pose_distance, reward_components
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
                 render_speed: float = 1.0, *,
                 allow_curriculum_resets: bool = False):
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
        self.viewer = None; self.steps = 0
        self.perception_bias = (np.zeros(3), np.array([1., 0., 0., 0.]))
        # État réservé au mode historique; le mode réactif n'en dépend pas.
        self.reference_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.episode_max_force = 0.0; self.episode_max_torque = 0.0
        self.best_position_error = np.inf; self.best_rotation_error = np.inf
        self.best_pose_metric = np.inf
        self.position_error_at_best_pose = np.inf
        self.rotation_error_at_best_pose = np.inf
        self.episode_reward_components: dict[str, float] = {}
        self.friction_scale = 1.0
        # Set by the periodic evaluation callback; zero during training.
        self.training_timesteps = 0
        # Le défaut sûr est le vrai départ. Seule la factory d'entraînement
        # active ce rôle; evaluate.py et EvalCallback restent donc immunisés.
        self.allow_curriculum_resets = bool(allow_curriculum_resets)
        self.curriculum_frontier_pool: list[CurriculumState] = []
        self.curriculum_historical_pool: list[CurriculumState] = []
        self.curriculum_historical_bins: list[list[CurriculumState]] = []
        self.curriculum_rng = np.random.default_rng(0)
        self.reset_source = "true_start"
        self.curriculum_start_position_error = np.nan
        self.curriculum_start_rotation_error = np.nan
        self.curriculum_start_pose_distance = np.nan
        self.curriculum_start_success_rate = np.nan
        self.is_curriculum_reset = False

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

    def _curriculum_contact_categories(self) -> tuple[str, ...]:
        """Coarse, observation-only roles for reverse-walk contact diagnostics.

        The model has no existing forbidden-contact policy. In particular,
        mobile--fixed contact is normal assembly contact, never a new reject
        condition introduced by this diagnostic.
        """
        categories: set[str] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair == {self.mobile_geom, self.fixed_geom}:
                categories.add("piece_fixture")
            elif self.mobile_geom in pair:
                categories.add("piece_other")
            elif self.fixed_geom in pair:
                categories.add("fixture_other")
            else:
                categories.add("unknown")
        return tuple(sorted(categories))

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

    def _reset_episode_statistics(self) -> None:
        self.steps = 0
        self.episode_max_force = 0.0; self.episode_max_torque = 0.0
        self.episode_reward_components = {}
        self.best_position_error = np.inf; self.best_rotation_error = np.inf
        self.best_pose_metric = np.inf
        self.position_error_at_best_pose = np.inf
        self.rotation_error_at_best_pose = np.inf

    def _update_best_pose(self, position_error: float, rotation_error: float) -> None:
        metric = pose_distance(
            position_error, rotation_error,
            float(self.cfg["reward"]["rotation_length_scale"]),
        )
        if metric < self.best_pose_metric:
            self.best_pose_metric = metric
            self.position_error_at_best_pose = position_error
            self.rotation_error_at_best_pose = rotation_error

    def _set_episode_start_metadata(
        self, reset_source: str, state: CurriculumState | None = None,
    ) -> None:
        self.reset_source = reset_source
        self.is_curriculum_reset = reset_source in {
            "curriculum_frontier", "curriculum_historical",
        }
        if state is None:
            self.curriculum_start_state_id = -1
            self.curriculum_start_generation_depth = -1
            self.curriculum_start_position_error = np.nan
            self.curriculum_start_rotation_error = np.nan
            self.curriculum_start_pose_distance = np.nan
            self.curriculum_start_success_rate = np.nan
        else:
            self.curriculum_start_state_id = int(state.state_id)
            self.curriculum_start_generation_depth = int(state.generation_depth)
            self.curriculum_start_position_error = float(state.position_error)
            self.curriculum_start_rotation_error = float(state.rotation_error)
            self.curriculum_start_pose_distance = float(state.pose_distance)
            self.curriculum_start_success_rate = float(state.success_rate)

    def _start_info(self, true_error: np.ndarray) -> dict:
        return {
            "true_error": true_error,
            "friction_scale": self.friction_scale,
            "reset_source": self.reset_source,
            "is_curriculum_reset": self.is_curriculum_reset,
            "curriculum_start_position_error": self.curriculum_start_position_error,
            "curriculum_start_rotation_error": self.curriculum_start_rotation_error,
            "curriculum_start_pose_distance": self.curriculum_start_pose_distance,
            "curriculum_start_success_rate": self.curriculum_start_success_rate,
            "curriculum_start_state_id": self.curriculum_start_state_id,
            "curriculum_start_generation_depth": (
                self.curriculum_start_generation_depth
            ),
        }

    def _initialize_true_start(self) -> np.ndarray:
        self._reset_episode_statistics(); self.admittance.reset()
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
        true_error = self._error()
        self.best_position_error = float(np.linalg.norm(true_error[:3]))
        self.best_rotation_error = float(np.linalg.norm(true_error[3:]))
        self._update_best_pose(
            self.best_position_error, self.best_rotation_error,
        )
        self._set_episode_start_metadata("true_start")
        return true_error

    def _select_reset(self, requested: str) -> CurriculumResetSelection:
        if requested not in {
            "auto", "curriculum", "true_start",
            "curriculum_frontier", "curriculum_historical",
        }:
            raise ValueError(
                "options.reset_source doit être 'auto', 'curriculum', "
                "'true_start', 'curriculum_frontier' ou "
                "'curriculum_historical'"
            )
        if (not self.allow_curriculum_resets
                or not bool(self.cfg["curriculum"]["enabled"])):
            # Le rôle d'évaluation reste prioritaire, même si des pools ont été
            # installés ou qu'un appelant demande explicitement le curriculum.
            return CurriculumResetSelection("true_start", None)
        curriculum = self.cfg["curriculum"]
        sampling = curriculum["start_sampling"]
        probabilities = configured_start_sampling_probabilities(
            frontier_pool_size=len(self.curriculum_frontier_pool),
            historical_pool_size=len(self.curriculum_historical_pool),
            curriculum_probability=float(
                curriculum["curriculum_reset_probability"]
            ),
            config=sampling,
        )
        return select_training_start(
            self.curriculum_rng,
            curriculum_probability=float(
                curriculum["curriculum_reset_probability"]
            ),
            frontier_fraction=float(sampling.get("frontier_fraction", .625)),
            historical_fraction=float(sampling.get("historical_fraction", .375)),
            historical_bins=int(sampling["historical_bins"]),
            frontier=self.curriculum_frontier_pool,
            historical=self.curriculum_historical_pool,
            requested=requested,
            historical_bin_groups=self.curriculum_historical_bins,
            probabilities=(
                probabilities
                if sampling.get("strategy", "legacy") == "adaptive_three_way"
                else None
            ),
        )

    def _choose_reset_source(self, requested: str) -> str:
        """Compatibilité pour les diagnostics/tests qui ne restaurent pas l'état."""
        return self._select_reset(requested).source

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.curriculum_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), 22])
            )
        requested = (options or {}).get("reset_source", "auto")
        selection = self._select_reset(requested)
        if selection.state is not None:
            return self.restore_curriculum_state(
                selection.state, reset_episode=True,
                restore_rng=False, reset_source=selection.source,
            )
        true_error = self._initialize_true_start()
        return self._observation(), self._start_info(true_error)

    def set_curriculum_reset_pools(
        self, frontier: list[CurriculumState], historical: list[CurriculumState],
    ) -> None:
        """Remplace les deux mémoires de reset; too-hard n'est jamais diffusé."""
        self.curriculum_frontier_pool = list(frontier)
        self.curriculum_historical_pool = list(historical)
        self.curriculum_historical_bins = historical_quantile_bins(
            self.curriculum_historical_pool,
            int(self.cfg["curriculum"]["start_sampling"]["historical_bins"]),
        )

    def set_curriculum_reset_pool(self, states: list[CurriculumState]) -> None:
        """Compatibilité avec l'ancienne API : le pool fourni devient frontier."""
        self.set_curriculum_reset_pools(states, [])

    def get_curriculum_rng_state(self) -> dict:
        return deepcopy(self.curriculum_rng.bit_generator.state)

    def set_curriculum_rng_state(self, state: dict) -> None:
        self.curriculum_rng.bit_generator.state = deepcopy(state)

    def get_worker_rng_state(self) -> dict:
        return {
            "curriculum": self.get_curriculum_rng_state(),
            "environment": deepcopy(self.np_random.bit_generator.state),
        }

    def set_worker_rng_state(self, state: dict) -> None:
        # Compatibilité avec les premiers snapshots ne contenant que le RCG.
        if "curriculum" not in state:
            self.set_curriculum_rng_state(state)
            return
        self.set_curriculum_rng_state(state["curriculum"])
        self.np_random.bit_generator.state = deepcopy(state["environment"])

    def _site_quat(self):
        q=np.zeros(4); mujoco.mju_mat2Quat(q, self.data.site_xmat[self.grasp_site]); return q

    def _randomize_friction(self, randomization: dict) -> float:
        low, high = map(float, randomization["friction_scale"])
        scale = float(self.np_random.uniform(low, high))
        self.model.geom_friction[self._contact_geom_ids] = self._base_contact_friction * scale
        return scale

    def _integration_state(self) -> np.ndarray:
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, specification), dtype=float)
        mujoco.mj_getState(self.model, self.data, state, specification)
        return state

    def capture_curriculum_state(
        self, *, success_rate: float = np.nan,
    ) -> CurriculumState:
        """Capture l'état physique restaurable, sans statistiques d'épisode."""
        true_error = self._error()
        position_error = float(np.linalg.norm(true_error[:3]))
        rotation_error = float(np.linalg.norm(true_error[3:]))
        task_pose = relative(
            self._pose(self.fixed_body), self._pose(self.mobile_body),
        )
        reference_position = (
            None if self.reference_pose is None else self.reference_pose[0].copy()
        )
        reference_quaternion = (
            None if self.reference_pose is None else self.reference_pose[1].copy()
        )
        rng_state = (
            deepcopy(self.np_random.bit_generator.state)
            if hasattr(self, "np_random") else None
        )
        return CurriculumState(
            mj_state=self._integration_state(),
            fixed_body_position=self.model.body_pos[self.fixed_body].copy(),
            fixed_body_quaternion=self.model.body_quat[self.fixed_body].copy(),
            contact_friction=self.model.geom_friction[
                self._contact_geom_ids
            ].copy(),
            friction_scale=float(self.friction_scale),
            admittance_offset=self.admittance.offset.copy(),
            admittance_velocity=self.admittance.velocity.copy(),
            reference_position=reference_position,
            reference_quaternion=reference_quaternion,
            perception_bias_position=self.perception_bias[0].copy(),
            perception_bias_quaternion=self.perception_bias[1].copy(),
            environment_rng_state=rng_state,
            task_position=task_pose[0].copy(),
            task_quaternion=task_pose[1].copy(),
            position_error=position_error,
            rotation_error=rotation_error,
            pose_distance=pose_distance(
                position_error, rotation_error,
                float(self.cfg["reward"]["rotation_length_scale"]),
            ),
            success_rate=float(success_rate),
        )

    def restore_curriculum_state(
        self, state: CurriculumState, *, reset_episode: bool = True,
        restore_rng: bool = True, reset_source: str = "curriculum_frontier",
    ) -> tuple[np.ndarray, dict]:
        """Restaure exactement un snapshot puis recalcule les quantités dérivées."""
        self.model.body_pos[self.fixed_body] = state.fixed_body_position
        self.model.body_quat[self.fixed_body] = state.fixed_body_quaternion
        self.model.geom_friction[self._contact_geom_ids] = state.contact_friction
        self.friction_scale = float(state.friction_scale)
        mujoco.mj_setState(
            self.model, self.data, state.mj_state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        self.admittance.offset = state.admittance_offset.copy()
        self.admittance.velocity = state.admittance_velocity.copy()
        self.reference_pose = (
            None
            if state.reference_position is None
            else (
                state.reference_position.copy(),
                state.reference_quaternion.copy(),
            )
        )
        self.perception_bias = (
            state.perception_bias_position.copy(),
            state.perception_bias_quaternion.copy(),
        )
        if restore_rng and state.environment_rng_state is not None:
            self.np_random.bit_generator.state = deepcopy(
                state.environment_rng_state
            )
        mujoco.mj_forward(self.model, self.data)
        # mj_forward recalcule correctement les poses/contact dérivés mais
        # remplace qacc_warmstart. Le second setState remet ce warm-start exact;
        # les quantités dérivées restent valides puisque qpos n'a pas changé.
        mujoco.mj_setState(
            self.model, self.data, state.mj_state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        true_error = self._error()
        if reset_episode:
            self._reset_episode_statistics()
            position_error = float(np.linalg.norm(true_error[:3]))
            rotation_error = float(np.linalg.norm(true_error[3:]))
            self.best_position_error = position_error
            self.best_rotation_error = rotation_error
            self._update_best_pose(position_error, rotation_error)
            self._set_episode_start_metadata(reset_source, state)
        return self._observation(), self._start_info(true_error)

    def build_goal_seed(self, *, seed: int | None = None) -> CurriculumState:
        """Construit et valide la pose finale exacte, réservée à l'expansion."""
        super().reset(seed=seed)
        self._reset_episode_statistics(); self.admittance.reset()
        mujoco.mj_resetData(self.model, self.data)
        fixed_pose = (
            np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.model.body_pos[self.fixed_body] = fixed_pose[0]
        self.model.body_quat[self.fixed_body] = fixed_pose[1]
        self.model.geom_friction[self._contact_geom_ids] = self._base_contact_friction
        self.friction_scale = 1.0
        mobile_pose = compose(fixed_pose, self._target())
        self.data.qpos[self.qadr:self.qadr + 3] = mobile_pose[0]
        self.data.qpos[self.qadr + 3:self.qadr + 7] = mobile_pose[1]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        grasp = (
            self.data.site_xpos[self.grasp_site].copy(), self._site_quat(),
        )
        self.reference_pose = (
            (grasp[0].copy(), grasp[1].copy())
            if self.cfg["action"]["control_mode"] == "accumulated_reference"
            else None
        )
        mocap = self.model.body_mocapid[self.target_mocap]
        self.data.mocap_pos[mocap] = grasp[0]
        self.data.mocap_quat[mocap] = grasp[1]
        perception = self.cfg["perception"]
        self.perception_bias = (
            np.asarray(perception["translation_bias"], dtype=float),
            euler_xyz_to_quat(np.deg2rad(perception["rotation_bias_deg"])),
        )
        mujoco.mj_forward(self.model, self.data)

        true_error = self._error()
        position_error = float(np.linalg.norm(true_error[:3]))
        rotation_error = float(np.linalg.norm(true_error[3:]))
        wrench = self._true_wrench()
        safety = self.cfg["success"]
        status = assess_status(
            position_error=position_error, rotation_error=rotation_error,
            max_force=float(np.linalg.norm(wrench[:3])),
            max_torque=float(np.linalg.norm(wrench[3:])),
            workspace_error=position_error, step_count=0,
            config=safety,
            max_episode_steps=self.cfg["simulation"]["max_episode_steps"],
        )
        if not status.success or status.unsafe:
            raise RuntimeError(
                "Goal seed RCG invalide: "
                f"position_error={position_error:.9g}, "
                f"rotation_error={rotation_error:.9g}, "
                f"unsafe={status.unsafe}, reason={status.termination_reason}"
            )
        self.best_position_error = position_error
        self.best_rotation_error = rotation_error
        self._update_best_pose(position_error, rotation_error)
        self._set_episode_start_metadata("goal_seed")
        goal_seed = self.capture_curriculum_state(success_rate=1.0)

        # Vérification de stabilité sur un cycle de contrôle, puis restauration
        # de la pose exacte servant de seed.
        stable_force, stable_torque = self._run_control_substeps(safety)
        stable_error = self._error()
        stable_status = assess_status(
            position_error=float(np.linalg.norm(stable_error[:3])),
            rotation_error=float(np.linalg.norm(stable_error[3:])),
            max_force=stable_force, max_torque=stable_torque,
            workspace_error=float(np.linalg.norm(stable_error[:3])),
            step_count=0, config=safety,
            max_episode_steps=self.cfg["simulation"]["max_episode_steps"],
        )
        if not stable_status.success or stable_status.unsafe:
            raise RuntimeError(
                "Goal seed RCG instable après un cycle de contrôle: "
                f"reason={stable_status.termination_reason}"
            )
        self.restore_curriculum_state(
            goal_seed, reset_episode=False, restore_rng=True,
            reset_source="goal_seed",
        )
        return goal_seed

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

    def _advance_physics(self, action) -> PhysicsStepResult:
        """Chaîne physique commune aux épisodes RL et aux marches RCG."""
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
        return PhysicsStepResult(
            action=action,
            true_error=self._error(),
            final_wrench=self._true_wrench(),
            max_force=step_max_force,
            max_torque=step_max_torque,
        )

    def step_for_curriculum_generation(
        self, action,
    ) -> CurriculumGenerationResult:
        """Avance la vraie physique sans reward, statistiques RL ni timeout.

        Le succès est rapporté mais n'interrompt pas la marche; l'orchestrateur
        n'abandonne que sur une violation de sécurité.
        """
        result = self._advance_physics(action)
        position_error = float(np.linalg.norm(result.true_error[:3]))
        rotation_error = float(np.linalg.norm(result.true_error[3:]))
        status = assess_status(
            position_error=position_error, rotation_error=rotation_error,
            max_force=result.max_force, max_torque=result.max_torque,
            workspace_error=position_error, step_count=0,
            config=self.cfg["success"],
            max_episode_steps=self.cfg["simulation"]["max_episode_steps"],
        )
        distance = pose_distance(
            position_error, rotation_error,
            float(self.cfg["reward"]["rotation_length_scale"]),
        )
        state = self.capture_curriculum_state()
        final_force = float(np.linalg.norm(result.final_wrench[:3]))
        final_torque = float(np.linalg.norm(result.final_wrench[3:]))
        return CurriculumGenerationResult(
            state=state,
            geometric_success=status.geometric_success,
            success=status.success,
            unsafe=status.unsafe,
            unsafe_force=status.unsafe_force,
            unsafe_torque=status.unsafe_torque,
            unsafe_workspace=status.unsafe_workspace,
            position_error=position_error,
            rotation_error=rotation_error,
            pose_distance=distance,
            max_force=result.max_force,
            max_torque=result.max_torque,
            final_force=final_force,
            final_torque=final_torque,
            contact_categories=self._curriculum_contact_categories(),
        )

    def step(self, action):
        result = self._advance_physics(action)
        action = result.action
        step_max_force = result.max_force
        step_max_torque = result.max_torque

        self.steps += 1
        self.episode_max_force = max(self.episode_max_force, step_max_force)
        self.episode_max_torque = max(self.episode_max_torque, step_max_torque)
        true_error = result.true_error
        pos = float(np.linalg.norm(true_error[:3])); rot = float(np.linalg.norm(true_error[3:]))
        self.best_position_error = min(self.best_position_error, pos)
        self.best_rotation_error = min(self.best_rotation_error, rot)
        self._update_best_pose(pos, rot)
        status = assess_status(
            position_error=pos, rotation_error=rot,
            max_force=step_max_force, max_torque=step_max_torque,
            workspace_error=pos, step_count=self.steps,
            config=self.cfg["success"], max_episode_steps=self.cfg["simulation"]["max_episode_steps"],
        )
        components = reward_components(
            position_error=pos, rotation_error=rot,
            max_force=step_max_force, action=action,
            status=status, config=self.cfg["reward"],
            max_torque=step_max_torque,
        )
        for key, value in components.items():
            if key.startswith("reward_"):
                self.episode_reward_components[key] = self.episode_reward_components.get(key, 0.0) + value
        final_wrench = result.final_wrench
        info = {
            **components,
            "geometric_success": status.geometric_success,
            "success": status.success,
            "safe_success": status.success,
            "is_success": status.success,
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
            "best_position_error": self.best_position_error,
            "best_rotation_error": self.best_rotation_error,
            "best_pose_metric": self.best_pose_metric,
            "position_error_at_best_pose": self.position_error_at_best_pose,
            "rotation_error_at_best_pose": self.rotation_error_at_best_pose,
            "max_force": self.episode_max_force,
            "max_torque": self.episode_max_torque,
            "training_timesteps": self.training_timesteps,
            "friction_scale": self.friction_scale,
            "reset_source": self.reset_source,
            "is_curriculum_reset": self.is_curriculum_reset,
            "curriculum_start_position_error": self.curriculum_start_position_error,
            "curriculum_start_rotation_error": self.curriculum_start_rotation_error,
            "curriculum_start_pose_distance": self.curriculum_start_pose_distance,
            "curriculum_start_success_rate": self.curriculum_start_success_rate,
            "curriculum_start_state_id": self.curriculum_start_state_id,
            "curriculum_start_generation_depth": (
                self.curriculum_start_generation_depth
            ),
        }
        milestones = self.cfg.get("diagnostics", {}).get(
            "true_start_position_milestones_m", (.020, .010, .005, .002),
        )
        for threshold in milestones:
            millimetres = int(round(float(threshold) * 1000.0))
            info[f"reached_{millimetres}mm"] = float(
                self.best_position_error <= float(threshold)
            )
        info.update({f"episode_{key}": value for key, value in self.episode_reward_components.items()})
        if self.render_mode=="human": self.render()
        total_reward = sum(
            value for key, value in components.items() if key.startswith("reward_")
        )
        return self._observation(), float(total_reward), status.terminated, status.truncated, info
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
