from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings

import numpy as np

from src.assembly_env import TenonMortaiseEnv
from src.config import load_config
from src.curriculum import (
    CurriculumGenerationResult, CurriculumState, GenerationReport,
    ReverseCurriculumManager, StateLifecycleStats,
    classify_success_rate,
    compute_adaptive_diverse_fallback_probabilities,
    compute_adaptive_three_way_probabilities,
    compute_start_sampling_probabilities, effective_historical_fraction,
    historical_quantile_bins, mastered_boundary_states,
    reset_probabilities_for_transition_targets,
    mastered_edge_states, select_too_hard_by_lineage,
    select_training_start, too_hard_near_states,
    StartSamplingProbabilities, update_sampling_episode_length_ema,
)
from src.task_logic import assess_status, reward_components
from src.train import make_env


class ReverseCurriculumPhysicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/test1V21.yaml")
        cls.env = TenonMortaiseEnv(
            "configs/test1V21.yaml", allow_curriculum_resets=True,
        )
        cls.manager = ReverseCurriculumManager(
            cls.env, cls.config["curriculum"], seed=7,
        )
        cls.candidates, cls.report = cls.manager.generate_candidates(
            [cls.manager.goal_seed]
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_a_goal_seed_is_successful_safe_and_aligned(self):
        state = self.manager.goal_seed
        self.env.restore_curriculum_state(state, reset_episode=False)
        error = self.env._error()
        wrench = self.env._true_wrench()
        status = assess_status(
            position_error=float(np.linalg.norm(error[:3])),
            rotation_error=float(np.linalg.norm(error[3:])),
            max_force=float(np.linalg.norm(wrench[:3])),
            max_torque=float(np.linalg.norm(wrench[3:])),
            workspace_error=float(np.linalg.norm(error[:3])),
            step_count=0, config=self.config["success"],
            max_episode_steps=self.config["simulation"]["max_episode_steps"],
        )
        self.assertTrue(status.success)
        self.assertFalse(status.unsafe)
        mocap = self.env.model.body_mocapid[self.env.target_mocap]
        np.testing.assert_allclose(
            self.env.data.mocap_pos[mocap],
            self.env.data.site_xpos[self.env.grasp_site], atol=1e-12,
        )

    def test_b_generation_produces_safe_non_successful_goal_children(self):
        self.assertGreater(len(self.candidates), 0)
        self.assertEqual(
            len({state.state_id for state in self.candidates}),
            len(self.candidates),
        )
        for state in self.candidates:
            self.assertIsNone(state.parent_id)
            self.assertEqual(state.generation_depth, 1)
            self.env.restore_curriculum_state(state, reset_episode=False)
            wrench = self.env._true_wrench()
            status = assess_status(
                position_error=state.position_error,
                rotation_error=state.rotation_error,
                max_force=float(np.linalg.norm(wrench[:3])),
                max_torque=float(np.linalg.norm(wrench[3:])),
                workspace_error=state.position_error, step_count=0,
                config=self.config["success"],
                max_episode_steps=self.config["simulation"]["max_episode_steps"],
            )
            self.assertFalse(status.success)
            self.assertFalse(status.unsafe)

    def test_lineage_depth_counts_successive_curriculum_expansions(self):
        manager = object.__new__(ReverseCurriculumManager)
        manager.next_state_id = 1
        a = manager._assign_lineage_to_candidate(
            self.candidates[0], self.manager.goal_seed,
        )
        manager.next_state_id += 1
        b = manager._assign_lineage_to_candidate(self.candidates[1], a)
        manager.next_state_id += 1
        c = manager._assign_lineage_to_candidate(self.candidates[2], b)

        self.assertEqual(
            [(state.state_id, state.parent_id, state.generation_depth)
             for state in (a, b, c)],
            [(1, None, 1), (2, 1, 2), (3, 2, 3)],
        )

    def test_generation_accepts_non_monotone_pose_distance(self):
        parent = replace(
            self.candidates[0], state_id=90, parent_id=80,
            generation_depth=2, pose_distance=.009,
        )
        child_snapshot = replace(
            self.candidates[1], state_id=-1, parent_id=None,
            generation_depth=0, pose_distance=.006,
        )
        result = SimpleNamespace(
            state=child_snapshot, unsafe=False, success=False,
            pose_distance=.006,
        )

        class GenerationEnv:
            def restore_curriculum_state(self, *args, **kwargs):
                return None

            def step_for_curriculum_generation(self, action):
                return result

        manager = object.__new__(ReverseCurriculumManager)
        manager.env = GenerationEnv()
        manager.config = {"candidates_per_update": 1}
        manager.walk = {
            "walks_per_seed": 1, "max_steps": 1, "action_scale": .1,
        }
        manager.deduplication = {
            "position_tolerance": 1e-9, "rotation_tolerance_deg": 1e-9,
        }
        manager.rng = np.random.default_rng(7)
        manager.pools = {
            "too_hard": [], "frontier": [], "mastered": [],
        }
        manager.next_state_id = 91
        manager.last_expansion_seed_distances = []
        manager.last_expansion_seed_depths = []

        candidates, report = manager.generate_candidates([parent])

        self.assertEqual(report.valid, 1)
        self.assertEqual(report.not_outward_rejected, 0)
        self.assertLess(candidates[0].pose_distance, parent.pose_distance)
        self.assertEqual(candidates[0].parent_id, parent.state_id)
        self.assertEqual(
            candidates[0].generation_depth, parent.generation_depth + 1,
        )

    def test_c_generation_does_not_touch_rl_counters_or_replay_proxy(self):
        class ReplayProxy:
            def size(self):
                return 123

        class ModelProxy:
            num_timesteps = 456
            replay_buffer = ReplayProxy()

        model = ModelProxy()
        self.env.steps = 17
        self.env.episode_max_force = 3.0
        self.env.episode_reward_components = {"reward_pose": -2.0}
        before_model = (model.num_timesteps, model.replay_buffer.size())
        before_episode = (
            self.env.steps, self.env.episode_max_force,
            deepcopy(self.env.episode_reward_components),
        )
        self.manager.generate_candidates([self.manager.goal_seed])
        self.assertEqual(
            (model.num_timesteps, model.replay_buffer.size()), before_model,
        )
        self.assertEqual(
            (self.env.steps, self.env.episode_max_force,
             self.env.episode_reward_components), before_episode,
        )

    def test_candidate_qualification_is_stochastic_and_outside_replay(self):
        class ReplayProxy:
            def size(self):
                return 12

        class ModelProxy:
            def __init__(self):
                self.num_timesteps = 34
                self.replay_buffer = ReplayProxy()
                self.deterministic_arguments = []

            def predict(self, observation, deterministic):
                self.deterministic_arguments.append(deterministic)
                return np.zeros(6), None

        class QualificationEnv:
            def restore_curriculum_state(self, *args, **kwargs):
                return np.zeros(18, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(18, dtype=np.float32), 0.0, True, False,
                    {"safe_success": False},
                )

        model = ModelProxy()
        original_env = self.manager.env
        # L'environnement dédié n'est pas enveloppé par Monitor/VecMonitor :
        # aucun writer d'entraînement n'est donc accessible aux rollouts RCG.
        self.assertIs(original_env, self.env)
        self.assertFalse(hasattr(original_env, "results_writer"))
        self.manager.env = QualificationEnv()
        try:
            qualified = self.manager.qualify_candidates(
                model, [self.candidates[0]],
            )
        finally:
            self.manager.env = original_env
        self.assertEqual(qualified[0].success_rate, 0.0)
        self.assertEqual(
            model.deterministic_arguments,
            [False] * self.config["curriculum"][
                "evaluation_rollouts_per_candidate"
            ],
        )
        self.assertEqual(model.num_timesteps, 34)
        self.assertEqual(model.replay_buffer.size(), 12)

    def test_curriculum_state_pickle_restores_pools_rng_and_schedule(self):
        worker_states = [self.env.get_worker_rng_state()]
        expected_next_id = self.manager.next_state_id
        expected_update = self.manager.next_update_timesteps
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curriculum_state.pkl"
            self.manager.save(path, worker_states)
            self.manager.next_state_id = 99999
            self.manager.next_update_timesteps = 99999
            self.manager.rng.random()
            self.manager.load(path)
        self.assertEqual(self.manager.next_state_id, expected_next_id)
        self.assertEqual(self.manager.next_update_timesteps, expected_update)
        self.assertEqual(self.manager.worker_rng_states, worker_states)

    def test_d_snapshot_restore_and_next_action_are_reproducible(self):
        state = self.candidates[0]
        observation_a, _ = self.env.restore_curriculum_state(
            state, reset_episode=True, restore_rng=True,
        )
        restored_a = self.env.capture_curriculum_state()
        action = np.array([.2, -.1, .3, -.2, .1, -.3])
        next_a = self.env.step_for_curriculum_generation(action)

        observation_b, _ = self.env.restore_curriculum_state(
            state, reset_episode=True, restore_rng=True,
        )
        restored_b = self.env.capture_curriculum_state()
        next_b = self.env.step_for_curriculum_generation(action)

        np.testing.assert_allclose(observation_a, observation_b, atol=1e-12)
        for field in (
            "mj_state", "fixed_body_position", "fixed_body_quaternion",
            "contact_friction", "admittance_offset", "admittance_velocity",
            "task_position", "task_quaternion",
        ):
            np.testing.assert_allclose(
                getattr(restored_a, field), getattr(restored_b, field), atol=1e-12,
            )
        np.testing.assert_allclose(
            next_a.state.mj_state, next_b.state.mj_state, atol=1e-10,
        )
        np.testing.assert_allclose(
            next_a.state.admittance_offset,
            next_b.state.admittance_offset, atol=1e-12,
        )

    def test_e_forced_true_reset_matches_v20_nominal_40_mm_start(self):
        baseline = TenonMortaiseEnv("configs/test1V20.yaml")
        try:
            _, baseline_info = baseline.reset(
                seed=99, options={"reset_source": "true_start"},
            )
            _, curriculum_info = self.env.reset(
                seed=99, options={"reset_source": "true_start"},
            )
            self.assertEqual(curriculum_info["reset_source"], "true_start")
            self.assertEqual(curriculum_info["curriculum_start_state_id"], -1)
            self.assertEqual(
                curriculum_info["curriculum_start_generation_depth"], -1,
            )
            np.testing.assert_allclose(
                baseline_info["true_error"], curriculum_info["true_error"],
                atol=1e-12,
            )
            self.assertAlmostEqual(
                np.linalg.norm(curriculum_info["true_error"][:3]), 0.04,
                places=12,
            )
        finally:
            baseline.close()

    def test_f_forced_curriculum_reset_restores_pool_snapshot(self):
        state = self.candidates[0]
        self.env.set_curriculum_reset_pools([state], [])
        _, info = self.env.reset(
            seed=101, options={"reset_source": "curriculum_frontier"},
        )
        self.assertEqual(info["reset_source"], "curriculum_frontier")
        self.assertTrue(info["is_curriculum_reset"])
        self.assertEqual(info["curriculum_start_state_id"], state.state_id)
        self.assertEqual(
            info["curriculum_start_generation_depth"], state.generation_depth,
        )
        np.testing.assert_allclose(
            self.env._integration_state(), state.mj_state, atol=1e-10,
        )
        np.testing.assert_allclose(
            self.env.admittance.offset, state.admittance_offset, atol=1e-12,
        )

    def test_new_forced_reset_sources_preserve_episode_metadata(self):
        boundary, too_hard = self.candidates[:2]
        self.env.set_curriculum_reset_pools(
            [], [], [boundary], [too_hard],
        )
        for source, state in (
            ("curriculum_mastered_boundary", boundary),
            ("curriculum_too_hard_near", too_hard),
        ):
            with self.subTest(source=source):
                _, info = self.env.reset(
                    seed=102, options={"reset_source": source},
                )
                self.assertEqual(info["reset_source"], source)
                self.assertTrue(info["is_curriculum_reset"])
                self.assertEqual(
                    info["curriculum_start_state_id"], state.state_id,
                )
                self.assertEqual(
                    info["curriculum_start_pose_distance"], state.pose_distance,
                )

    def test_g_reset_source_mixture_is_consistent_with_50_30_20(self):
        self.env.set_curriculum_reset_pools(
            [self.candidates[0]], [self.candidates[1]],
        )
        self.env.curriculum_rng = np.random.default_rng(2021)
        draws = [self.env._choose_reset_source("auto") for _ in range(20_000)]
        expected = {
            "curriculum_frontier": .50,
            "curriculum_historical": .30,
            "true_start": .20,
        }
        for source, probability in expected.items():
            self.assertAlmostEqual(
                draws.count(source) / len(draws), probability, delta=.02,
            )

    def test_transition_mode_uses_callback_probabilities_in_environment(self):
        self.env.set_curriculum_reset_pools(
            [self.candidates[0]], [self.candidates[1]],
        )
        effective = StartSamplingProbabilities(
            true_start=.80, frontier=.05, historical=.15,
            historical_fraction_effective=.15,
        )
        sampling = self.env.cfg["curriculum"]["start_sampling"]
        original_unit = sampling["balance_unit"]
        try:
            self.env.set_curriculum_sampling_probabilities(effective)
            self.env.curriculum_rng = np.random.default_rng(216)
            historical_path = [
                self.env._choose_reset_source("auto") for _ in range(100)
            ]
            self.env.set_curriculum_sampling_probabilities(None)
            self.env.curriculum_rng = np.random.default_rng(216)
            self.assertEqual(historical_path, [
                self.env._choose_reset_source("auto") for _ in range(100)
            ])
            sampling["balance_unit"] = "transitions"
            self.env.set_curriculum_sampling_probabilities(effective)
            self.env.curriculum_rng = np.random.default_rng(216)
            draws = [
                self.env._choose_reset_source("auto") for _ in range(30_000)
            ]
        finally:
            sampling["balance_unit"] = original_unit
            self.env.set_curriculum_sampling_probabilities(None)
        for source, probability in {
            "true_start": .80,
            "curriculum_frontier": .05,
            "curriculum_historical": .15,
        }.items():
            self.assertAlmostEqual(
                draws.count(source) / len(draws), probability, delta=.015,
            )

    def test_i_evaluation_role_persistently_forces_true_start(self):
        factory = make_env(Path("configs/test1V21.yaml"), 0, 123)
        evaluation_env = factory()
        try:
            evaluation_env.set_curriculum_reset_pool([self.candidates[0]])
            self.assertFalse(evaluation_env.allow_curriculum_resets)
            for seed in range(10):
                _, info = evaluation_env.reset(seed=seed)
                self.assertEqual(info["reset_source"], "true_start")
                self.assertFalse(info["is_curriculum_reset"])
                self.assertAlmostEqual(
                    np.linalg.norm(info["true_error"][:3]), 0.04, places=12,
                )
        finally:
            evaluation_env.close()

    def test_frontier_requalification_five_of_five_moves_to_mastered(self):
        manager = object.__new__(ReverseCurriculumManager)
        manager.config = deepcopy(self.config["curriculum"])
        manager.config["revalidation"]["mastered_samples_per_update"] = 0
        manager.rng = np.random.default_rng(1)
        first = replace(
            self.candidates[0], state_id=10_001, success_rate=.6,
        )
        second = replace(
            self.candidates[1], state_id=10_003, success_rate=.4,
        )
        manager.pools = {
            "too_hard": [], "frontier": [first, second], "mastered": [],
        }
        seen = []
        def qualify(model, states):
            seen.extend(candidate.state_id for candidate in states)
            return [replace(candidate, success_rate=1.0) for candidate in states]
        manager.qualify_candidates = qualify
        self.assertEqual(manager.revalidate_existing(object()), 2)
        self.assertEqual(seen, [10_001, 10_003])
        self.assertEqual(manager.pools["frontier"], [])
        self.assertEqual(
            [candidate.state_id for candidate in manager.pools["mastered"]],
            [10_001, 10_003],
        )

    def test_mastered_revalidation_three_of_five_returns_to_frontier(self):
        manager = object.__new__(ReverseCurriculumManager)
        manager.config = deepcopy(self.config["curriculum"])
        manager.config["revalidation"]["mastered_samples_per_update"] = 1
        manager.rng = np.random.default_rng(2)
        state = replace(
            self.candidates[0], state_id=10_002, success_rate=1.0,
        )
        manager.pools = {
            "too_hard": [], "frontier": [], "mastered": [state],
        }
        manager.qualify_candidates = lambda model, states: [
            replace(candidate, success_rate=3 / 5) for candidate in states
        ]
        self.assertEqual(manager.revalidate_existing(object()), 1)
        self.assertEqual(manager.pools["mastered"], [])
        self.assertEqual(
            [candidate.state_id for candidate in manager.pools["frontier"]],
            [10_002],
        )

    def test_legacy_v1_minimal_pickle_migrates_all_pools_and_schedule(self):
        source = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=107,
        )
        source.pools = {
            "too_hard": [replace(
                self.candidates[0], state_id=20_001, success_rate=0.0,
            )],
            "frontier": [replace(
                self.candidates[1], state_id=20_002, success_rate=.6,
            )],
            "mastered": [replace(
                self.candidates[2], state_id=20_003, success_rate=1.0,
            )],
        }
        payload = source.state_dict()
        legacy_states = [
            payload["goal_seed"],
            *(state for pool in payload["pools"].values() for state in pool),
        ]
        for state in legacy_states:
            for field in ("state_id", "parent_id", "generation_depth"):
                vars(state).pop(field, None)
        legacy_keys = {
            "version", "goal_seed", "pools", "numpy_rng_state",
            "torch_rng_state", "torch_seed", "torch_cuda_rng_states",
            "worker_rng_states", "next_state_id", "update_count",
            "next_update_timesteps",
        }
        payload = {key: value for key, value in payload.items() if key in legacy_keys}
        payload["version"] = 1
        payload["worker_rng_states"] = [self.env.get_curriculum_rng_state()]
        restored = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=108,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored.load_state_dict(payload)
        self.assertTrue(any("V1 ancien" in str(item.message) for item in caught))
        self.assertEqual(restored.pool_sizes(), {
            "too_hard": 1, "frontier": 1, "mastered": 1,
        })
        restored_states = restored.all_states()
        state_ids = [state.state_id for state in restored_states]
        self.assertEqual(len(set(state_ids)), 3)
        self.assertTrue(all(state_id >= 0 for state_id in state_ids))
        self.assertTrue(all(
            state.parent_id is None and state.generation_depth == 0
            for state in restored_states
        ))
        self.assertGreater(restored.next_state_id, max(state_ids))
        self.assertEqual(
            restored.next_update_timesteps, source.next_update_timesteps,
        )
        self.assertEqual(restored.worker_rng_states, payload["worker_rng_states"])
        legacy_mastered = restored.pools["mastered"][0]
        child = restored._assign_lineage_to_candidate(
            self.candidates[0], legacy_mastered,
        )
        self.assertEqual(child.parent_id, legacy_mastered.state_id)
        self.assertEqual(child.generation_depth, 1)
        resaved = restored.state_dict()
        self.assertTrue(all(
            {"state_id", "parent_id", "generation_depth"} <= vars(state).keys()
            for pool in resaved["pools"].values() for state in pool
        ))

    def test_v2_pickle_without_expansion_config_keeps_existing_pools(self):
        source = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=109,
        )
        state = replace(
            self.candidates[0], state_id=21_001, success_rate=0.0,
        )
        source.pools["too_hard"] = [state]
        source.next_state_id = 21_002
        source.update_count = 3
        source.next_update_timesteps = 200_000
        payload = source.state_dict(training_timesteps=150_000)
        payload["curriculum_config"].pop("expansion", None)
        payload["curriculum_config"]["revalidation"].pop(
            "too_hard_samples_per_update", None,
        )
        legacy_state = payload["pools"]["too_hard"][0]
        vars(legacy_state).pop("parent_id")
        vars(legacy_state).pop("generation_depth")

        restored = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=110,
        )
        restored.load_state_dict(payload)
        self.assertEqual(restored.pool_sizes(), {
            "too_hard": 1, "frontier": 0, "mastered": 0,
        })
        self.assertEqual(restored.pools["too_hard"][0].state_id, 21_001)
        self.assertIsNone(restored.pools["too_hard"][0].parent_id)
        self.assertEqual(restored.pools["too_hard"][0].generation_depth, 0)
        self.assertEqual(restored.loaded_training_timesteps, 150_000)
        self.assertEqual(restored.next_state_id, 21_002)
        self.assertEqual(restored.update_count, 3)
        self.assertEqual(restored.next_update_timesteps, 200_000)

    def test_v4_pickle_preserves_sampling_episode_length_ema(self):
        source = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=211,
        )
        source.sampling_episode_length_ema = {
            "true_start": 283.0,
            "frontier": 1.0,
            "historical": 2.0,
            "mastered_boundary": 3.0,
            "too_hard_near": 4.4,
        }
        payload = source.state_dict()
        self.assertEqual(payload["version"], 5)
        restored = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=212,
        )
        restored.load_state_dict(payload)
        self.assertEqual(
            restored.sampling_episode_length_ema,
            source.sampling_episode_length_ema,
        )

    def test_v4_snapshot_without_pose_history_migrates_once(self):
        source = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=215,
        )
        payload = source.state_dict()
        payload["version"] = 4
        payload["task_config_sha256"] = source._task_config_sha256(
            legacy_observation_history=True,
        )
        states = [
            payload["goal_seed"],
            *(state for pool in payload["pools"].values() for state in pool),
        ]
        for state in states:
            vars(state).pop("previous_pose_error", None)
        restored = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=216,
        )
        with self.assertWarnsRegex(RuntimeWarning, "legacy curriculum state"):
            restored.load_state_dict(payload)
        self.assertIsNone(restored.goal_seed.previous_pose_error)
        self.env.include_previous_pose_error = True
        observation, _ = self.env.restore_curriculum_state(restored.goal_seed)
        np.testing.assert_array_equal(observation[:6], observation[6:12])

    def test_v3_pickle_without_sampling_ema_loads_unit_bootstrap(self):
        source = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=213,
        )
        payload = source.state_dict()
        payload["version"] = 3
        payload.pop("sampling_episode_length_ema")
        restored = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=214,
        )
        restored.load_state_dict(payload)
        self.assertEqual(
            restored.sampling_episode_length_ema,
            {name: 1.0 for name in (
                "true_start", "frontier", "historical",
                "mastered_boundary", "too_hard_near",
            )},
        )

    def test_save_reload_preserves_lineage_and_classification(self):
        source = ReverseCurriculumManager(
            self.env, self.config["curriculum"], seed=115,
        )
        a = replace(
            self.candidates[0], state_id=23_001, parent_id=None,
            generation_depth=1, success_rate=1.0,
        )
        b = replace(
            self.candidates[1], state_id=23_002, parent_id=a.state_id,
            generation_depth=2, success_rate=.6,
        )
        c = replace(
            self.candidates[2], state_id=23_003, parent_id=b.state_id,
            generation_depth=3, success_rate=0.0,
        )
        source.pools = {
            "too_hard": [c], "frontier": [b], "mastered": [a],
        }
        source.next_state_id = 23_004
        source.state_lifecycle = {
            23_002: StateLifecycleStats(
                created_update=2, last_revalidated_update=4,
                revalidation_count=3, frontier_since_update=3,
                consecutive_frontier_updates=2,
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curriculum_state.pkl"
            source.save(path, training_timesteps=1_600_000)
            restored = ReverseCurriculumManager(
                self.env, self.config["curriculum"], seed=116,
            )
            restored.load(path)

        expected = {
            23_001: (None, 1, "mastered"),
            23_002: (23_001, 2, "frontier"),
            23_003: (23_002, 3, "too_hard"),
        }
        actual = {
            state.state_id: (state.parent_id, state.generation_depth, pool)
            for pool, states in restored.pools.items()
            for state in states
        }
        self.assertEqual(actual, expected)
        self.assertEqual(restored.next_state_id, 23_004)
        self.assertEqual(restored.loaded_training_timesteps, 1_600_000)
        self.assertEqual(restored.state_lifecycle[23_002].revalidation_count, 3)
        self.assertEqual(restored.state_lifecycle[23_002].frontier_since_update, 3)
        self.assertEqual(restored.state_lifecycle[23_001].created_update, -1)

    def test_save_reload_preserves_a_graph_produced_by_multihop_expansion(self):
        curriculum_config = deepcopy(self.config["curriculum"])
        curriculum_config.setdefault("expansion", {}).update({
            "max_hops_per_seed": 4,
            "max_candidates_per_update": 24,
            "initial_scale": 1.0,
            "scale_up_factor": 1.25,
            "scale_down_factor": .7,
            "min_scale": .5,
            "max_scale": 3.0,
        })
        source = ReverseCurriculumManager(
            self.env, curriculum_config, seed=117,
        )
        root = replace(
            self.candidates[0], state_id=24_001, parent_id=None,
            generation_depth=1, success_rate=1.0,
        )
        source.pools["mastered"] = [root]
        source.next_state_id = 24_002
        snapshots = iter(self.candidates[1:4])
        rates = iter([1.0, 1.0, .6])

        def generate(seed, scale, report):
            report.generated += 1
            return next(snapshots), None

        source._generate_hop_snapshot = generate
        source._is_duplicate = lambda candidate, additional: False
        source.qualify_candidates = lambda model, states: [
            replace(states[0], success_rate=next(rates))
        ]
        report = source._expand_branches(object())
        self.assertEqual(report.expansion_candidates, 3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curriculum_state.pkl"
            source.save(path, training_timesteps=1_650_000)
            restored = ReverseCurriculumManager(
                self.env, curriculum_config, seed=118,
            )
            restored.load(path)

        expected = {
            24_001: (None, 1, "mastered"),
            24_002: (24_001, 2, "mastered"),
            24_003: (24_002, 3, "mastered"),
            24_004: (24_003, 4, "frontier"),
        }
        actual = {
            state.state_id: (state.parent_id, state.generation_depth, pool)
            for pool, states in restored.pools.items()
            for state in states
        }
        self.assertEqual(actual, expected)
        self.assertEqual(restored.loaded_training_timesteps, 1_650_000)

    def test_resume_with_new_reset_probability_restores_pools_and_schedule(self):
        saved_config = deepcopy(self.config["curriculum"])
        saved_config["curriculum_reset_probability"] = .80
        source = ReverseCurriculumManager(self.env, saved_config, seed=111)
        source.pools = {
            "too_hard": [replace(
                self.candidates[0], state_id=22_001, success_rate=0.0,
            )],
            "frontier": [replace(
                self.candidates[1], state_id=22_002, success_rate=.6,
            )],
            "mastered": [replace(
                self.candidates[2], state_id=22_003, success_rate=1.0,
            )],
        }
        source.next_state_id = 22_004
        source.update_count = 31
        source.next_update_timesteps = 1_650_000
        payload = source.state_dict(training_timesteps=1_600_000)

        current_config = deepcopy(saved_config)
        current_config["curriculum_reset_probability"] = .95
        restored = ReverseCurriculumManager(
            self.env, current_config, seed=112,
        )
        restored.load_state_dict(payload)

        self.assertEqual(
            payload["curriculum_config"]["curriculum_reset_probability"],
            .80,
        )
        self.assertEqual(
            restored.config["curriculum_reset_probability"], .95,
        )
        self.assertEqual(
            {
                name: [state.state_id for state in restored.pools[name]]
                for name in ("too_hard", "frontier", "mastered")
            },
            {
                "too_hard": [22_001],
                "frontier": [22_002],
                "mastered": [22_003],
            },
        )
        self.assertEqual(restored.next_state_id, 22_004)
        self.assertEqual(restored.update_count, 31)
        self.assertEqual(restored.next_update_timesteps, 1_650_000)
        self.assertEqual(restored.loaded_training_timesteps, 1_600_000)

        worker = object.__new__(TenonMortaiseEnv)
        worker.cfg = {"curriculum": deepcopy(restored.config)}
        worker.allow_curriculum_resets = True
        reset_pools = restored.training_reset_pools()
        worker.set_curriculum_reset_pools(
            reset_pools["frontier"], reset_pools["historical"],
        )
        worker.curriculum_rng = np.random.default_rng(4)
        self.assertTrue(
            worker._choose_reset_source("auto").startswith("curriculum_"),
        )
        worker.cfg["curriculum"]["curriculum_reset_probability"] = .80
        worker.curriculum_rng = np.random.default_rng(4)
        self.assertEqual(worker._choose_reset_source("auto"), "true_start")

    def test_resume_rejects_structural_change_at_load(self):
        saved_config = deepcopy(self.config["curriculum"])
        saved_config["curriculum_reset_probability"] = .80
        source = ReverseCurriculumManager(self.env, saved_config, seed=113)
        payload = source.state_dict()

        current_config = deepcopy(saved_config)
        current_config["curriculum_reset_probability"] = .95
        current_config["success_rate_low"] = .2
        restored = ReverseCurriculumManager(
            self.env, current_config, seed=114,
        )
        with self.assertRaisesRegex(
            ValueError, "paramètres structurels incompatibles",
        ):
            restored.load_state_dict(payload)

    def test_j_v20_and_v21_reward_are_identical_on_same_transition(self):
        baseline = TenonMortaiseEnv("configs/test1V20.yaml")
        curriculum = TenonMortaiseEnv("configs/test1V21.yaml")
        action = np.array([.1, -.2, .3, -.4, .5, -.6])
        try:
            baseline.reset(seed=77, options={"reset_source": "true_start"})
            curriculum.reset(seed=77, options={"reset_source": "true_start"})
            _, reward_a, terminated_a, truncated_a, info_a = baseline.step(action)
            _, reward_b, terminated_b, truncated_b, info_b = curriculum.step(action)
        finally:
            baseline.close(); curriculum.close()
        self.assertEqual(self.config["reward"], load_config("configs/test1V20.yaml")["reward"])
        self.assertEqual((terminated_a, truncated_a), (terminated_b, truncated_b))
        self.assertAlmostEqual(reward_a, reward_b, places=12)
        for key in info_a:
            if key.startswith("reward_"):
                self.assertAlmostEqual(info_a[key], info_b[key], places=12)


class ReverseCurriculumPureLogicTest(unittest.TestCase):
    @staticmethod
    def _states(distances, *, depths=None, parents=None, start_id=1):
        distances = list(distances)
        depths = list(range(len(distances))) if depths is None else list(depths)
        parents = [None] * len(distances) if parents is None else list(parents)
        return [
            SimpleNamespace(
                pose_distance=float(distance), state_id=start_id + index,
                parent_id=parents[index], generation_depth=int(depths[index]),
            )
            for index, distance in enumerate(distances)
        ]

    @staticmethod
    def _select(frontier, historical, *, seed=7, requested="auto"):
        return select_training_start(
            np.random.default_rng(seed), curriculum_probability=.8,
            frontier_fraction=.625, historical_fraction=.375,
            historical_bins=4, frontier=frontier, historical=historical,
            requested=requested,
        )

    @classmethod
    def _manager(cls, *, seed=7):
        manager = object.__new__(ReverseCurriculumManager)
        manager.config = {
            "candidates_per_update": 1,
            "evaluation_rollouts_per_candidate": 5,
            "update_interval_timesteps": 50_000,
            "success_rate_low": .10,
            "success_rate_high": .90,
            "max_pool_size": 100,
            "expansion": {
                "max_hops_per_seed": 4,
                "max_candidates_per_update": 24,
                "initial_scale": 1.0,
                "scale_up_factor": 1.25,
                "scale_down_factor": .7,
                "min_scale": .5,
                "max_scale": 3.0,
            },
            "revalidation": {
                "mastered_samples_per_update": 0,
                "too_hard_samples_per_update": 1,
            },
        }
        manager.walk = {
            "walks_per_seed": 1, "max_steps": 2, "action_scale": .5,
        }
        manager.rng = np.random.default_rng(seed)
        manager.pools = {
            "too_hard": [], "frontier": [], "mastered": [],
        }
        manager.goal_seed = SimpleNamespace(
            pose_distance=0.0, state_id=-1,
            parent_id=None, generation_depth=0,
        )
        manager.last_expansion_seed_distances = []
        manager.last_expansion_seed_depths = []
        return manager

    @staticmethod
    def _with_success_rate(state, success_rate):
        return SimpleNamespace(**{
            **vars(state), "success_rate": float(success_rate),
        })

    @staticmethod
    def _snapshot(
        state_id: int, *, parent_id: int | None = None,
        depth: int = 0, x: float = 0.0, success_rate: float = np.nan,
    ) -> CurriculumState:
        """Construit un snapshot numérique minimal pour les tests multi-hop."""
        return CurriculumState(
            mj_state=np.array([x], dtype=float),
            fixed_body_position=np.zeros(3),
            fixed_body_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            contact_friction=np.ones((1, 3)), friction_scale=1.0,
            admittance_offset=np.zeros(6), admittance_velocity=np.zeros(6),
            reference_position=None, reference_quaternion=None,
            perception_bias_position=np.zeros(3),
            perception_bias_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            previous_pose_error=np.zeros(6),
            environment_rng_state=None,
            task_position=np.array([x, 0.0, 0.0]),
            task_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            position_error=abs(x), rotation_error=0.0,
            pose_distance=abs(x), success_rate=success_rate,
            state_id=state_id, parent_id=parent_id,
            generation_depth=depth,
        )

    def _multihop_manager(
        self, success_rates, *, seed_count=1, max_hops=4,
        candidate_budget=24, scale_up=1.25, max_scale=3.0,
    ):
        manager = self._manager(seed=37)
        manager.config["expansion"].update({
            "max_hops_per_seed": max_hops,
            "max_candidates_per_update": candidate_budget,
            "scale_up_factor": scale_up,
            "max_scale": max_scale,
        })
        seeds = [
            self._snapshot(
                index + 1, depth=1, x=.01 * (index + 1), success_rate=1.0,
            )
            for index in range(seed_count)
        ]
        manager.pools["mastered"] = list(seeds)
        manager.goal_seed = self._snapshot(-1)
        manager.next_state_id = 100
        rates = iter(success_rates)
        generated_from: list[int] = []
        used_scales: list[float] = []
        raw_index = 0

        def generate(seed, scale, report):
            nonlocal raw_index
            generated_from.append(seed.state_id)
            used_scales.append(scale)
            raw_index += 1
            report.generated += 1
            report.raw_candidates_generated += 1
            report.reverse_steps.append(1)
            return self._snapshot(-1, x=1.0 + raw_index), None

        manager._generate_hop_snapshot = generate
        manager._is_duplicate = lambda candidate, additional: False
        manager.qualify_candidates = lambda model, states: [
            replace(states[0], success_rate=float(next(rates)))
        ]
        return manager, seeds, generated_from, used_scales

    def test_a_sampling_distribution_is_derived_as_50_30_20(self):
        rng = np.random.default_rng(21)
        frontier = self._states([.005, .006])
        historical = self._states([.001, .002, .003, .004])
        counts = {
            "curriculum_frontier": 0,
            "curriculum_historical": 0,
            "true_start": 0,
        }
        for _ in range(20_000):
            selection = select_training_start(
                rng, curriculum_probability=.8,
                frontier_fraction=.625, historical_fraction=.375,
                historical_bins=4, frontier=frontier, historical=historical,
            )
            counts[selection.source] += 1
        self.assertAlmostEqual(counts["curriculum_frontier"] / 20_000, .50, delta=.02)
        self.assertAlmostEqual(counts["curriculum_historical"] / 20_000, .30, delta=.02)
        self.assertAlmostEqual(counts["true_start"] / 20_000, .20, delta=.02)

    def test_b_empty_frontier_falls_back_to_historical(self):
        selection = self._select(
            [], self._states([.001]), requested="curriculum_frontier",
        )
        self.assertEqual(selection.source, "true_start")

    def test_adaptive_historical_ramp_and_empty_pool_probabilities(self):
        expected = {0: 0.0, 3: .03, 20: .20, 38: .375, 100: .375}
        for size, fraction in expected.items():
            self.assertAlmostEqual(effective_historical_fraction(
                size, adaptive=True, fixed_fraction=.375,
                fraction_per_state=.01, fraction_max=.375,
            ), fraction)
        empty_frontier = compute_start_sampling_probabilities(
            curriculum_probability=.95, historical_fraction=.10,
            frontier_available=False, historical_available=True,
        )
        self.assertAlmostEqual(empty_frontier.historical, .095)
        self.assertAlmostEqual(empty_frontier.true_start, .905)
        empty_historical = compute_start_sampling_probabilities(
            curriculum_probability=.95, historical_fraction=.10,
            frontier_available=True, historical_available=False,
        )
        self.assertEqual(empty_historical.frontier, .95)
        both_empty = compute_start_sampling_probabilities(
            curriculum_probability=.95, historical_fraction=.10,
            frontier_available=False, historical_available=False,
        )
        self.assertEqual(both_empty.true_start, 1.0)

    def test_adaptive_historical_disabled_keeps_fixed_fraction(self):
        self.assertEqual(effective_historical_fraction(
            3, adaptive=False, fixed_fraction=.375,
            fraction_per_state=.01, fraction_max=.375,
        ), .375)

    def test_auto_sampling_does_not_reassign_empty_frontier_budget(self):
        rng = np.random.default_rng(41)
        historical = self._states([.001, .002, .003])
        historical_count = sum(
            select_training_start(
                rng, curriculum_probability=.95,
                frontier_fraction=.90, historical_fraction=.10,
                historical_bins=4, frontier=[], historical=historical,
            ).source == "curriculum_historical"
            for _ in range(20_000)
        )
        self.assertAlmostEqual(historical_count / 20_000, .095, delta=.01)

        rng = np.random.default_rng(42)
        frontier = self._states([.005])
        frontier_count = sum(
            select_training_start(
                rng, curriculum_probability=.95,
                frontier_fraction=.90, historical_fraction=.10,
                historical_bins=4, frontier=frontier, historical=[],
            ).source == "curriculum_frontier"
            for _ in range(20_000)
        )
        self.assertAlmostEqual(frontier_count / 20_000, .95, delta=.01)

    def test_adaptive_three_way_probabilities_scale_and_cap_independently(self):
        def probabilities(frontier, historical):
            return compute_adaptive_three_way_probabilities(
                frontier, historical,
                frontier_fraction_per_state=.10, frontier_fraction_max=.45,
                historical_fraction_per_state=.01,
                historical_fraction_max=.25,
            )

        for size, expected in {0: 0, 1: .1, 2: .2, 3: .3, 5: .45, 100: .45}.items():
            self.assertAlmostEqual(probabilities(size, 0).frontier, expected)
        for size, expected in {0: 0, 3: .03, 10: .1, 20: .2, 25: .25, 100: .25}.items():
            self.assertAlmostEqual(probabilities(0, size).historical, expected)
        early = probabilities(2, 3)
        self.assertAlmostEqual(early.frontier, .20)
        self.assertAlmostEqual(early.historical, .03)
        self.assertAlmostEqual(early.true_start, .77)
        mature = probabilities(100, 100)
        self.assertAlmostEqual(mature.frontier, .45)
        self.assertAlmostEqual(mature.historical, .25)
        self.assertAlmostEqual(mature.true_start, .30)
        empty = probabilities(0, 0)
        self.assertAlmostEqual(empty.frontier, 0.0)
        self.assertAlmostEqual(empty.historical, 0.0)
        self.assertAlmostEqual(empty.true_start, 1.0)
        no_frontier = probabilities(0, 10)
        self.assertAlmostEqual(no_frontier.frontier, 0.0)
        self.assertAlmostEqual(no_frontier.historical, .10)
        self.assertAlmostEqual(no_frontier.true_start, .90)
        no_historical = probabilities(3, 0)
        self.assertAlmostEqual(no_historical.frontier, .30)
        self.assertAlmostEqual(no_historical.historical, 0.0)
        self.assertAlmostEqual(no_historical.true_start, .70)
        self.assertAlmostEqual(
            early.frontier + early.historical + early.true_start, 1.0,
        )

    def test_adaptive_three_way_sampling_matches_target_with_one_draw(self):
        probabilities = compute_adaptive_three_way_probabilities(
            2, 3, frontier_fraction_per_state=.10,
            frontier_fraction_max=.45, historical_fraction_per_state=.01,
            historical_fraction_max=.25,
        )
        rng = np.random.default_rng(43)
        counts = {source: 0 for source in (
            "curriculum_frontier", "curriculum_historical", "true_start",
        )}
        for _ in range(20_000):
            selection = select_training_start(
                rng, curriculum_probability=.95,
                frontier_fraction=.625, historical_fraction=.375,
                historical_bins=4, frontier=self._states([.005, .006]),
                historical=self._states([.001, .002, .003]),
                probabilities=probabilities,
            )
            counts[selection.source] += 1
        self.assertAlmostEqual(counts["curriculum_frontier"] / 20_000, .20, delta=.015)
        self.assertAlmostEqual(counts["curriculum_historical"] / 20_000, .03, delta=.01)
        self.assertAlmostEqual(counts["true_start"] / 20_000, .77, delta=.015)

    @staticmethod
    def _sampling_probabilities(**overrides):
        values = {
            "true_start": .20,
            "frontier": .30,
            "historical": .10,
            "historical_fraction_effective": .10,
            "mastered_boundary": .25,
            "too_hard_near": .15,
        }
        values.update(overrides)
        return StartSamplingProbabilities(**values)

    def test_transition_targets_are_identity_with_equal_lengths(self):
        targets = self._sampling_probabilities()
        effective = reset_probabilities_for_transition_targets(
            targets, {name: 7.0 for name in (
                "true_start", "frontier", "historical",
                "mastered_boundary", "too_hard_near",
            )},
        )
        self.assertEqual(effective, targets)

    def test_transition_balancing_reconstructs_v39_example(self):
        targets = self._sampling_probabilities(
            true_start=.71, frontier=.10, historical=.04,
            historical_fraction_effective=.04,
            mastered_boundary=.10, too_hard_near=.05,
        )
        lengths = {
            "true_start": 283.0,
            "frontier": 1.0,
            "historical": 1.0,
            "mastered_boundary": 1.0,
            "too_hard_near": 4.4,
        }
        effective = reset_probabilities_for_transition_targets(
            targets, lengths,
        )
        expected = {
            "true_start": .0099,
            "frontier": .3939,
            "historical": .1576,
            "mastered_boundary": .3939,
            "too_hard_near": .0448,
        }
        for name, probability in expected.items():
            self.assertAlmostEqual(
                getattr(effective, name), probability, delta=5e-5,
            )
        denominator = sum(
            getattr(effective, name) * lengths[name]
            for name in expected
        )
        for name in expected:
            reconstructed = (
                getattr(effective, name) * lengths[name] / denominator
            )
            self.assertAlmostEqual(
                reconstructed, getattr(targets, name), places=12,
            )
        self.assertAlmostEqual(
            sum(getattr(effective, name) for name in expected),
            1.0, places=12,
        )

    def test_transition_balancing_handles_long_short_and_inactive_sources(self):
        targets = self._sampling_probabilities(
            true_start=.50, frontier=.50, historical=0.0,
            historical_fraction_effective=0.0,
            mastered_boundary=0.0, too_hard_near=0.0,
        )
        lengths = {
            "true_start": 100.0,
            "frontier": 1.0,
            "historical": 3.0,
            "mastered_boundary": 4.0,
            "too_hard_near": 5.0,
        }
        effective = reset_probabilities_for_transition_targets(
            targets, lengths,
        )
        self.assertAlmostEqual(effective.true_start, 1 / 101, places=12)
        self.assertAlmostEqual(effective.frontier, 100 / 101, places=12)
        self.assertEqual(effective.historical, 0.0)
        self.assertEqual(effective.mastered_boundary, 0.0)
        self.assertEqual(effective.too_hard_near, 0.0)
        transition_mass = (
            effective.true_start * lengths["true_start"]
            + effective.frontier * lengths["frontier"]
        )
        self.assertAlmostEqual(
            effective.true_start * lengths["true_start"] / transition_mass,
            .5, places=12,
        )

    def test_transition_balanced_probabilities_drive_empirical_reset_sampling(self):
        targets = self._sampling_probabilities(
            true_start=.71, frontier=.10, historical=.04,
            historical_fraction_effective=.04,
            mastered_boundary=.10, too_hard_near=.05,
        )
        effective = reset_probabilities_for_transition_targets(targets, {
            "true_start": 283.0, "frontier": 1.0, "historical": 1.0,
            "mastered_boundary": 1.0, "too_hard_near": 4.4,
        })
        pools = {
            "frontier": self._states([.01]),
            "historical": self._states([.005]),
            "mastered_boundary": self._states([.02]),
            "too_hard_near": self._states([.03]),
        }
        reset_to_label = {
            "true_start": "true_start",
            "curriculum_frontier": "frontier",
            "curriculum_historical": "historical",
            "curriculum_mastered_boundary": "mastered_boundary",
            "curriculum_too_hard_near": "too_hard_near",
        }
        counts = {name: 0 for name in reset_to_label.values()}
        rng = np.random.default_rng(215)
        for _ in range(50_000):
            selection = select_training_start(
                rng, curriculum_probability=.95,
                frontier_fraction=.625, historical_fraction=.375,
                historical_bins=4, probabilities=effective, **pools,
            )
            counts[reset_to_label[selection.source]] += 1
        for name, count in counts.items():
            self.assertAlmostEqual(
                count / 50_000, getattr(effective, name), delta=.01,
            )

    def test_transition_balancing_rejects_invalid_episode_lengths(self):
        targets = self._sampling_probabilities()
        valid = {name: 1.0 for name in (
            "true_start", "frontier", "historical",
            "mastered_boundary", "too_hard_near",
        )}
        for invalid in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                lengths = dict(valid)
                lengths["frontier"] = invalid
                with self.assertRaisesRegex(ValueError, "frontier"):
                    reset_probabilities_for_transition_targets(
                        targets, lengths,
                    )

    def test_episode_length_ema_updates_exactly_and_keeps_missing_source(self):
        previous = {
            name: value for name, value in zip(
                ("true_start", "frontier", "historical",
                 "mastered_boundary", "too_hard_near"),
                (100.0, 4.0, 3.0, 2.0, 1.0),
            )
        }
        completed = {
            "true_start": [200.0, 300.0],
            "frontier": [2.0],
            "historical": [],
            "mastered_boundary": [8.0, 12.0],
            "too_hard_near": [],
        }
        updated = update_sampling_episode_length_ema(
            previous, completed, ema_alpha=.25,
            min_completed_episodes=2,
        )
        self.assertEqual(updated["true_start"], .25 * 250.0 + .75 * 100.0)
        self.assertEqual(updated["mastered_boundary"], .25 * 10.0 + .75 * 2.0)
        self.assertEqual(updated["frontier"], previous["frontier"])
        self.assertEqual(updated["historical"], previous["historical"])
        self.assertEqual(updated["too_hard_near"], previous["too_hard_near"])

    @staticmethod
    def _diverse_probabilities(frontier, historical, boundary, too_hard):
        return compute_adaptive_diverse_fallback_probabilities(
            frontier, historical, boundary, too_hard,
            frontier_fraction_per_state=.10, frontier_fraction_max=.45,
            historical_fraction_per_state=.01,
            historical_fraction_max=.25,
            mastered_boundary_fraction_per_state=.05,
            mastered_boundary_fraction_max=.20,
            too_hard_near_fraction_per_state=.05,
            too_hard_near_fraction_max=.20,
            historical_boost_fraction_per_state=.01,
            historical_boost_fraction_max=.05,
            true_start_fraction_min=.30,
        )

    def test_diverse_fallback_keeps_frontier_cap_and_vanishes_when_saturated(self):
        for frontier_count, expected in ((0, 0.0), (1, .10), (2, .20),
                                         (4, .40), (5, .45), (20, .45)):
            probabilities = self._diverse_probabilities(
                frontier_count, 25, 4, 4,
            )
            self.assertAlmostEqual(probabilities.frontier, expected)
            self.assertAlmostEqual(
                probabilities.missing_frontier_budget, .45 - expected,
            )
        saturated = self._diverse_probabilities(5, 25, 4, 4)
        legacy = compute_adaptive_three_way_probabilities(
            5, 25, frontier_fraction_per_state=.10,
            frontier_fraction_max=.45, historical_fraction_per_state=.01,
            historical_fraction_max=.25,
        )
        self.assertEqual(saturated.mastered_boundary, 0.0)
        self.assertEqual(saturated.too_hard_near, 0.0)
        self.assertEqual(saturated.fallback_budget_used, 0.0)
        self.assertEqual(
            (saturated.true_start, saturated.frontier, saturated.historical),
            (legacy.true_start, legacy.frontier, legacy.historical),
        )

    def test_diverse_fallback_uses_missing_budget_without_inflating_frontier(self):
        one_frontier = self._diverse_probabilities(1, 10, 4, 4)
        self.assertAlmostEqual(one_frontier.frontier, .10)
        self.assertGreater(one_frontier.mastered_boundary, 0.0)
        self.assertGreater(one_frontier.too_hard_near, 0.0)
        self.assertGreater(one_frontier.historical, .10)
        scale = .35 / .45
        self.assertAlmostEqual(one_frontier.mastered_boundary, .20 * scale)
        self.assertAlmostEqual(one_frontier.too_hard_near, .20 * scale)
        self.assertAlmostEqual(one_frontier.historical - .10, .05 * scale)
        self.assertAlmostEqual(one_frontier.fallback_budget_used, .35)

        no_frontier = self._diverse_probabilities(0, 25, 4, 4)
        self.assertEqual(no_frontier.frontier, 0.0)
        self.assertAlmostEqual(no_frontier.mastered_boundary, .20)
        self.assertAlmostEqual(no_frontier.too_hard_near, .20)
        self.assertAlmostEqual(no_frontier.historical, .30)
        self.assertAlmostEqual(no_frontier.true_start, .30)

    def test_diverse_fallback_low_diversity_leaves_unused_budget_at_true_start(self):
        probabilities = self._diverse_probabilities(0, 0, 1, 1)
        self.assertAlmostEqual(probabilities.mastered_boundary, .05)
        self.assertAlmostEqual(probabilities.too_hard_near, .05)
        self.assertAlmostEqual(probabilities.fallback_budget_used, .10)
        self.assertAlmostEqual(probabilities.true_start, .90)

    def test_diverse_fallback_always_preserves_minimum_and_unit_sum(self):
        for frontier in (0, 1, 2, 4, 5, 20):
            for historical in (0, 1, 10, 25, 100):
                for boundary in (0, 1, 4, 20):
                    for too_hard in (0, 1, 4, 20):
                        probabilities = self._diverse_probabilities(
                            frontier, historical, boundary, too_hard,
                        )
                        values = (
                            probabilities.true_start,
                            probabilities.frontier,
                            probabilities.historical,
                            probabilities.mastered_boundary,
                            probabilities.too_hard_near,
                        )
                        self.assertGreaterEqual(
                            probabilities.true_start, .30 - 1e-12,
                        )
                        self.assertAlmostEqual(sum(values), 1.0, places=12)

    def test_diverse_fallback_sampling_matches_five_source_targets(self):
        probabilities = self._diverse_probabilities(1, 10, 2, 1)
        pools = {
            "frontier": self._states([.01]),
            "historical": self._states(np.linspace(.001, .01, 10)),
            "mastered_boundary": self._states([.02, .021]),
            "too_hard_near": self._states([.03]),
        }
        counts = {source: 0 for source in (
            "true_start", "curriculum_frontier", "curriculum_historical",
            "curriculum_mastered_boundary", "curriculum_too_hard_near",
        )}
        rng = np.random.default_rng(44)
        for _ in range(40_000):
            selection = select_training_start(
                rng, curriculum_probability=.95,
                frontier_fraction=.625, historical_fraction=.375,
                historical_bins=4, probabilities=probabilities, **pools,
            )
            counts[selection.source] += 1
        expected = {
            "true_start": probabilities.true_start,
            "curriculum_frontier": probabilities.frontier,
            "curriculum_historical": probabilities.historical,
            "curriculum_mastered_boundary": probabilities.mastered_boundary,
            "curriculum_too_hard_near": probabilities.too_hard_near,
        }
        for source, target in expected.items():
            self.assertAlmostEqual(
                counts[source] / 40_000, target, delta=.012,
            )

    def test_c_empty_historical_falls_back_to_frontier(self):
        selection = self._select(
            self._states([.005]), [], requested="curriculum_historical",
        )
        self.assertEqual(selection.source, "curriculum_frontier")

    def test_d_all_training_pools_empty_falls_back_to_true_start(self):
        for requested in ("auto", "curriculum", "curriculum_frontier",
                          "curriculum_historical"):
            self.assertEqual(
                self._select([], [], requested=requested).source, "true_start",
            )

    def test_e_historical_quantile_bins_are_nonempty_and_all_sampled(self):
        for count in (1, 2, 3, 17):
            states = self._states(np.linspace(.001, .020, count))
            bins = historical_quantile_bins(states, 4)
            self.assertEqual(len(bins), min(count, 4))
            self.assertTrue(all(bins))
        non_monotone = self._states(
            [.001, 100.0, .010, 50.0],
            depths=[3, 0, 2, 1], start_id=20,
        )
        self.assertEqual(
            [state.state_id
             for group in historical_quantile_bins(non_monotone, 4)
             for state in group],
            [21, 23, 22, 20],
        )
        historical = self._states(
            [.001, .0011, .0012, .010, .0101, .0102, .020, .030, .040],
        )
        rng = np.random.default_rng(31)
        counts = [0, 0, 0, 0]
        for _ in range(8_000):
            selection = select_training_start(
                rng, curriculum_probability=.8,
                frontier_fraction=.625, historical_fraction=.375,
                historical_bins=4, frontier=[], historical=historical,
                requested="curriculum_historical",
            )
            counts[selection.historical_bin] += 1
        for count in counts:
            self.assertAlmostEqual(count / 8_000, .25, delta=.03)

    def test_h_success_band_boundaries(self):
        expected = {
            0.0: "too_hard", 0.05: "too_hard",
            0.10: "frontier", 0.50: "frontier", 0.90: "frontier",
            0.95: "mastered", 1.0: "mastered",
        }
        for success_rate, category in expected.items():
            with self.subTest(success_rate=success_rate):
                self.assertEqual(
                    classify_success_rate(success_rate, 0.10, 0.90), category,
                )

    def test_multihop_crosses_mastered_states_until_frontier_with_exact_lineage(self):
        manager, seeds, generated_from, scales = self._multihop_manager(
            [1.0, 1.0, .6], max_hops=4,
        )

        report = manager._expand_branches(object())

        self.assertEqual(report.expansion_candidates, 3)
        self.assertEqual(report.expansion_hops, 3)
        self.assertEqual(report.expansion_branches, 1)
        self.assertEqual((report.new_mastered, report.new_frontier), (2, 1))
        self.assertEqual(report.stop_reasons, {"frontier": 1})
        self.assertEqual(generated_from, [seeds[0].state_id, 100, 101])
        self.assertEqual(scales, [1.0, 1.25, 1.5625])
        states = {state.state_id: state for state in manager.all_states()}
        self.assertEqual(
            [(states[state_id].parent_id, states[state_id].generation_depth)
             for state_id in (100, 101, 102)],
            [(seeds[0].state_id, 2), (100, 3), (101, 4)],
        )

    def test_multihop_too_hard_stops_without_a_third_candidate(self):
        manager, _, generated_from, _ = self._multihop_manager(
            [1.0, 0.0, 1.0], max_hops=4,
        )

        report = manager._expand_branches(object())

        self.assertEqual(report.expansion_candidates, 2)
        self.assertEqual(report.expansion_hops, 2)
        self.assertEqual(report.new_too_hard, 1)
        self.assertEqual(report.stop_reasons, {"too_hard": 1})
        self.assertEqual(generated_from, [1, 100])

    def test_multihop_max_hops_bounds_an_all_mastered_branch(self):
        manager, _, _, scales = self._multihop_manager(
            [1.0] * 10, max_hops=4,
        )

        report = manager._expand_branches(object())

        self.assertEqual(report.expansion_candidates, 4)
        self.assertEqual(report.expansion_hops, 4)
        self.assertEqual(report.max_hops_reached, 4)
        self.assertEqual(report.stop_reasons, {"max_hops": 1})
        self.assertEqual(scales, [1.0, 1.25, 1.5625, 1.953125])

    def test_multihop_global_budget_is_checked_before_qualification(self):
        manager, _, _, _ = self._multihop_manager(
            [1.0] * 20, seed_count=3, max_hops=4, candidate_budget=5,
        )
        qualification_calls = 0
        original_qualify = manager.qualify_candidates

        def qualify(model, states):
            nonlocal qualification_calls
            qualification_calls += 1
            return original_qualify(model, states)

        manager.qualify_candidates = qualify
        report = manager._expand_branches(object())

        self.assertEqual(report.expansion_candidates, 5)
        self.assertEqual(report.expansion_rollouts, 25)
        self.assertEqual(qualification_calls, 5)
        self.assertEqual(report.expansion_hops, 5)
        self.assertIn("candidate_budget", report.stop_reasons)

    def test_round_robin_attempts_every_branch_before_any_second_hop(self):
        manager, seeds, generated_from, _ = self._multihop_manager(
            [1.0] * 20, seed_count=3, max_hops=4, candidate_budget=5,
        )

        manager._expand_branches(object())

        self.assertEqual(set(generated_from[:3]), {
            state.state_id for state in seeds
        })
        self.assertEqual(len(generated_from[:3]), 3)
        self.assertTrue(all(state_id < 100 for state_id in generated_from[:3]))
        self.assertTrue(all(state_id >= 100 for state_id in generated_from[3:]))

    def test_expansion_scale_is_capped_and_is_not_a_difficulty_signal(self):
        manager, _, _, scales = self._multihop_manager(
            [1.0, 1.0, 1.0, .5], max_hops=4,
            scale_up=2.0, max_scale=1.5,
        )

        report = manager._expand_branches(object())

        self.assertEqual(scales, [1.0, 1.5, 1.5, 1.5])
        self.assertEqual(report.expansion_scale_max, 1.5)
        self.assertEqual(report.new_frontier, 1)

    def test_too_hard_scale_down_is_bounded_but_not_retried_in_v1(self):
        manager = self._manager()
        self.assertAlmostEqual(
            manager._next_expansion_scale(1.0, "too_hard"), .7,
        )
        self.assertEqual(
            manager._next_expansion_scale(.5, "too_hard"), .5,
        )

    def test_invalid_and_duplicate_each_stop_the_branch_without_descendant(self):
        for stop_kind in ("snapshot_invalid", "duplicate"):
            with self.subTest(stop_kind=stop_kind):
                manager, seeds, _, _ = self._multihop_manager([1.0])
                manager.config["expansion"]["max_attempts_per_hop"] = 1
                qualification_calls = 0

                def qualify(model, states):
                    nonlocal qualification_calls
                    qualification_calls += 1
                    return [replace(states[0], success_rate=1.0)]

                manager.qualify_candidates = qualify
                if stop_kind == "snapshot_invalid":
                    manager._generate_hop_snapshot = (
                        lambda seed, scale, report: (None, "snapshot_invalid")
                    )
                else:
                    manager._is_duplicate = lambda candidate, additional: True

                report = manager._expand_branches(object())

                self.assertEqual(report.expansion_candidates, 0)
                self.assertEqual(report.expansion_hops, 1)
                self.assertEqual(qualification_calls, 0)
                self.assertEqual(report.stop_reasons, {"attempt_budget": 1})
                self.assertEqual(manager.pools["mastered"], seeds)
                self.assertEqual(manager.next_state_id, 100)

    def test_reverse_unsafe_reasons_preserve_force_torque_priority(self):
        seed = self._snapshot(1, x=.01)
        for flags, expected in (
            ((True, False, False), "force"),
            ((False, True, False), "torque"),
            ((True, True, False), "force_and_torque"),
            ((False, False, True), "workspace"),
        ):
            with self.subTest(flags=flags):
                manager = self._manager()
                manager.walk["max_steps"] = 1
                result = CurriculumGenerationResult(
                    state=seed, geometric_success=False, success=False,
                    unsafe=True, unsafe_force=flags[0], unsafe_torque=flags[1],
                    unsafe_workspace=flags[2], position_error=.01,
                    rotation_error=0.0, pose_distance=.01,
                    max_force=91.0, max_torque=9.0,
                )
                manager.env = SimpleNamespace(
                    restore_curriculum_state=lambda *args, **kwargs: None,
                    step_for_curriculum_generation=lambda action: result,
                )
                report = GenerationReport()
                snapshot, reason = manager._generate_hop_snapshot(seed, 1.0, report)
                self.assertIsNone(snapshot)
                self.assertEqual(reason, expected)
                self.assertEqual(len(report.rejected_force_max), int(flags[0]))
                self.assertEqual(len(report.rejected_torque_max), int(flags[1]))

    def test_unsafe_walk_returns_last_valid_non_success_prefix(self):
        manager = self._manager()
        manager.walk.update({
            "proposal_mode": "persistent", "max_steps": 5,
            "persistent_proposal": {
                "attempt_direction_noise_std": 0.0,
                "hop_direction_noise_std": 0.0,
                "step_noise_std": 0.0,
            },
        })
        manager._active_proposal_direction = np.array([1, 0, 0, 0, 0, 0.])
        parent = self._snapshot(1, x=.01)
        snapshots = [self._snapshot(-1, x=value) for value in (.02, .03, .04)]
        outcomes = [
            CurriculumGenerationResult(
                state=state, geometric_success=False, success=False,
                unsafe=False, unsafe_force=False, unsafe_torque=False,
                unsafe_workspace=False, position_error=state.position_error,
                rotation_error=0.0, pose_distance=state.pose_distance,
            ) for state in snapshots
        ]
        outcomes.append(CurriculumGenerationResult(
            state=self._snapshot(-1, x=.05), geometric_success=False,
            success=False, unsafe=True, unsafe_force=False, unsafe_torque=True,
            unsafe_workspace=False, position_error=.05, rotation_error=0.0,
            pose_distance=.05, max_torque=9.0,
        ))
        calls = 0
        def step(action):
            nonlocal calls
            value = outcomes[calls]
            calls += 1
            return value
        manager.env = SimpleNamespace(
            restore_curriculum_state=lambda *args, **kwargs: None,
            step_for_curriculum_generation=step,
        )
        report = GenerationReport()
        candidate, reason = manager._generate_hop_snapshot(parent, 1.0, report)
        self.assertIs(candidate, snapshots[-1])
        self.assertIsNone(reason)
        self.assertEqual(calls, 4)
        self.assertEqual(report.safe_prefix_candidates, 1)
        self.assertEqual(report.safe_prefix_steps, [3])
        self.assertEqual(report.persistent_attempts, 1)

    def test_expansion_funnel_and_parent_displacement_are_diagnostic_only(self):
        manager, _, _, _ = self._multihop_manager([.6])
        report = manager._expand_branches(object())

        self.assertEqual(report.expansion_hops, 1)
        self.assertEqual(report.raw_candidates_generated, 1)
        self.assertEqual(report.valid_candidates, 1)
        self.assertEqual(report.nonduplicate_candidates, 1)
        self.assertEqual(report.qualified_candidates, 1)
        # Parent x=10 mm and synthetic raw candidate x=2 m: delta is 1990 mm.
        self.assertAlmostEqual(report.raw_parent_translation_mm[0], 1990.0)
        self.assertAlmostEqual(report.raw_parent_rotation_deg[0], 0.0)

    def test_workspace_and_generation_failure_are_exclusive_stops(self):
        for reason in ("workspace", "generation_failed"):
            with self.subTest(reason=reason):
                manager, _, _, _ = self._multihop_manager([1.0])
                manager.config["expansion"]["max_attempts_per_hop"] = 1
                manager._generate_hop_snapshot = (
                    lambda seed, scale, report: (None, reason)
                )
                report = manager._expand_branches(object())
                self.assertEqual(report.stop_reasons, {"attempt_budget": 1})
                self.assertEqual(report.attempt_no_candidate, 1)
                self.assertEqual(report.qualified_candidates, 0)

    def test_attempts_retry_from_same_parent_without_consuming_candidate_budget(self):
        manager, seeds, generated_from, _ = self._multihop_manager([.6])
        manager.config["expansion"]["max_attempts_per_hop"] = 4
        attempts = iter((None, self._snapshot(-1, x=.01), self._snapshot(-1, x=2.0)))

        def generate(seed, scale, report):
            generated_from.append(seed.state_id)
            report.raw_candidates_generated += 1
            snapshot = next(attempts)
            return snapshot, "generation_failed" if snapshot is None else None

        manager._generate_hop_snapshot = generate
        duplicate_calls = 0
        def duplicate(candidate, additional):
            nonlocal duplicate_calls
            duplicate_calls += 1
            return duplicate_calls == 1
        manager._is_duplicate = duplicate
        report = manager._expand_branches(object())

        self.assertEqual(report.expansion_attempts, 3)
        self.assertEqual(report.attempt_no_candidate, 1)
        self.assertEqual(report.attempt_duplicate, 1)
        self.assertEqual(report.attempt_candidate_found, 1)
        self.assertEqual(report.expansion_candidates, 1)
        self.assertEqual(generated_from, [seeds[0].state_id] * 3)
        self.assertEqual(report.guided_memory_rejected_duplicates, 1)
        self.assertEqual(report.guided_memory_insertions, 1)
        self.assertEqual(len(manager.proposal_memory[seeds[0].state_id]), 1)

    def test_attempt_budget_stops_branch_after_exact_limit(self):
        manager, _, _, _ = self._multihop_manager([1.0])
        manager.config["expansion"]["max_attempts_per_hop"] = 3
        manager._generate_hop_snapshot = lambda seed, scale, report: (
            None, "generation_failed",
        )
        report = manager._expand_branches(object())
        self.assertEqual(report.expansion_attempts, 3)
        self.assertEqual(report.stop_reasons, {"attempt_budget": 1})

    def test_guided_fraction_zero_preserves_rng_and_uniform_fallback(self):
        manager = self._manager(seed=19)
        manager.walk["proposal"] = {
            "guided_fraction": 0.0, "guided_noise_std": .2,
            "memory_size_per_parent": 4,
        }
        parent = self._snapshot(1, x=.01)
        manager.proposal_memory = {1: [np.ones(6)]}
        before = deepcopy(manager.rng.bit_generator.state)
        kind, direction = manager._choose_reverse_proposal(parent)
        self.assertEqual(kind, "uniform")
        self.assertIsNone(direction)
        self.assertEqual(manager.rng.bit_generator.state, before)

        manager.walk["proposal"]["guided_fraction"] = 1.0
        manager.proposal_memory = {}
        kind, direction = manager._choose_reverse_proposal(parent)
        self.assertEqual(kind, "uniform")
        self.assertIsNone(direction)

    def test_proposal_memory_is_bounded_per_parent(self):
        manager = self._manager()
        manager.walk["proposal"] = {
            "guided_fraction": 1.0, "guided_noise_std": .2,
            "memory_size_per_parent": 4,
        }
        manager.proposal_memory = {}
        parent = self._snapshot(1, x=.01)
        for index in range(10):
            manager._remember_proposal(
                parent, self._snapshot(-1, x=.02 + index * .01),
            )
        self.assertEqual(len(manager.proposal_memory[1]), 4)

    def test_guided_direction_keeps_direction_not_candidate_magnitude(self):
        manager = self._manager()
        manager.env = SimpleNamespace(cfg={"action": {
            "max_translation_step": .001,
            "max_rotation_step_deg": 10.0,
        }})
        parent = self._snapshot(1)
        translated = self._snapshot(-1, x=.002)
        direction_x = manager._proposal_direction(parent, translated)
        np.testing.assert_allclose(direction_x, [1, 0, 0, 0, 0, 0])

        angle = np.deg2rad(5.0)
        rotated = replace(
            self._snapshot(-1),
            task_quaternion=np.array([
                np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2),
            ]),
        )
        direction_rz = manager._proposal_direction(parent, rotated)
        np.testing.assert_allclose(
            direction_rz, [0, 0, 0, 0, 0, 1], atol=1e-12,
        )

        farther = self._snapshot(-1, x=.020)
        np.testing.assert_allclose(
            manager._proposal_direction(parent, farther), direction_x,
        )

    def test_independent_mode_preserves_the_historical_uniform_rng_path(self):
        manager = self._manager(seed=23)
        manager.walk.update({
            "proposal_mode": "independent", "max_steps": 3,
            "action_scale": .5,
        })
        parent = self._snapshot(1)
        actions = []
        expected_rng = np.random.default_rng()
        expected_rng.bit_generator.state = deepcopy(manager.rng.bit_generator.state)

        class Env:
            def restore_curriculum_state(self, *args, **kwargs):
                return None

            def step_for_curriculum_generation(inner_self, action):
                actions.append(np.asarray(action).copy())
                return CurriculumGenerationResult(
                    state=self._snapshot(-1, x=.01),
                    geometric_success=False, success=False, unsafe=False,
                    unsafe_force=False, unsafe_torque=False,
                    unsafe_workspace=False, position_error=.01,
                    rotation_error=0.0, pose_distance=.01,
                )

        manager.env = Env()
        report = GenerationReport()
        manager._generate_hop_snapshot(parent, 1.0, report)
        expected = np.asarray([
            expected_rng.uniform(-.5, .5, size=6) for _ in range(3)
        ])
        np.testing.assert_array_equal(np.asarray(actions), expected)
        self.assertEqual(report.independent_attempts, 1)
        self.assertEqual(report.persistent_attempts, 0)

    def test_persistent_guided_attempt_reuses_one_direction(self):
        manager = self._manager(seed=24)
        manager.walk.update({
            "proposal_mode": "persistent", "max_steps": 4,
            "action_scale": .5,
            "persistent_proposal": {
                "attempt_direction_noise_std": 0.0,
                "hop_direction_noise_std": 0.0,
                "step_noise_std": 0.0,
            },
            "proposal": {
                "guided_fraction": 1.0, "guided_noise_std": 0.0,
                "memory_size_per_parent": 4,
            },
        })
        manager._active_proposal_kind = "guided"
        manager._active_proposal_direction = np.array([1, 0, 0, 0, 0, 0.])
        parent = self._snapshot(1)
        actions = []

        class Env:
            def restore_curriculum_state(self, *args, **kwargs):
                return None

            def step_for_curriculum_generation(inner_self, action):
                actions.append(np.asarray(action).copy())
                return CurriculumGenerationResult(
                    state=self._snapshot(-1, x=.01),
                    geometric_success=False, success=False, unsafe=False,
                    unsafe_force=False, unsafe_torque=False,
                    unsafe_workspace=False, position_error=.01,
                    rotation_error=0.0, pose_distance=.01,
                )

        manager.env = Env()
        report = GenerationReport()
        manager._generate_hop_snapshot(parent, 1.0, report)
        np.testing.assert_array_equal(
            np.asarray(actions), np.tile([.5, 0, 0, 0, 0, 0], (4, 1)),
        )
        self.assertEqual(report.persistent_attempts, 1)
        self.assertEqual(report.independent_attempts, 0)

    def test_persistent_step_noise_stays_centered_and_clipped(self):
        manager = self._manager(seed=25)
        manager.walk.update({
            "proposal_mode": "persistent", "max_steps": 500,
            "action_scale": .5,
            "persistent_proposal": {
                "attempt_direction_noise_std": 0.0,
                "hop_direction_noise_std": 0.0,
                "step_noise_std": .05,
            },
        })
        direction = np.array([.25, -.20, .10, 0.0, .15, -.10])
        manager._active_proposal_direction = direction
        parent = self._snapshot(1)
        actions = []

        class Env:
            def restore_curriculum_state(self, *args, **kwargs):
                return None

            def step_for_curriculum_generation(inner_self, action):
                actions.append(np.asarray(action).copy())
                return CurriculumGenerationResult(
                    state=self._snapshot(-1, x=.01),
                    geometric_success=False, success=False, unsafe=False,
                    unsafe_force=False, unsafe_torque=False,
                    unsafe_workspace=False, position_error=.01,
                    rotation_error=0.0, pose_distance=.01,
                )

        manager.env = Env()
        manager._generate_hop_snapshot(parent, 1.0, GenerationReport())
        samples = np.asarray(actions)
        self.assertGreater(float(np.std(samples[:, 0])), 0.0)
        self.assertLessEqual(float(np.max(np.abs(samples))), .5)
        np.testing.assert_allclose(
            np.mean(samples, axis=0) / .5, direction, atol=.015,
        )

    def test_persistent_branch_keeps_one_guided_heading_across_mastered_hops(self):
        manager, seeds, _, _ = self._multihop_manager([1.0, 1.0, .6])
        manager.walk.update({
            "proposal_mode": "persistent",
            "persistent_proposal": {
                "attempt_direction_noise_std": 0.0,
                "hop_direction_noise_std": 0.0,
                "step_noise_std": 0.0,
            },
            "proposal": {
                "guided_fraction": 1.0, "guided_noise_std": 0.0,
                "memory_size_per_parent": 4,
            },
        })
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        manager.proposal_memory = {int(seeds[0].state_id): [expected.copy()]}
        choose = manager._choose_reverse_proposal
        queried_parents = []

        def choose_once(parent):
            queried_parents.append(int(parent.state_id))
            return choose(parent)

        manager._choose_reverse_proposal = choose_once
        generated = manager._generate_hop_snapshot
        headings = []

        def capture_heading(seed, scale, report):
            headings.append(manager._active_proposal_direction.copy())
            return generated(seed, scale, report)

        manager._generate_hop_snapshot = capture_heading
        report = manager._expand_branches(object())

        np.testing.assert_array_equal(headings, np.tile(expected, (3, 1)))
        np.testing.assert_array_equal(report.branch_heading_changes, [0.0, 0.0])
        self.assertEqual(report.proposal_guided_attempts, 3)
        self.assertEqual(report.successive_hop_heading_opposition, 0)
        self.assertEqual(queried_parents, [int(seeds[0].state_id)])

    def test_hop_heading_noise_is_local_and_clipped(self):
        manager = self._manager(seed=27)
        manager.walk["persistent_proposal"] = {
            "attempt_direction_noise_std": 0.0,
            "hop_direction_noise_std": 0.15,
            "step_noise_std": 0.0,
        }
        heading = np.array([.95, -.90, .2, 0.0, .1, -.1])
        expected_rng = np.random.default_rng()
        expected_rng.bit_generator.state = deepcopy(manager.rng.bit_generator.state)
        expected = np.clip(
            heading + expected_rng.normal(0.0, .15, size=6), -1.0, 1.0,
        )

        actual = manager._next_branch_heading(heading)

        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual, heading))
        self.assertLessEqual(float(np.max(np.abs(actual))), 1.0)

    def test_retries_vary_around_the_same_heading_and_restore_the_parent(self):
        manager = self._manager(seed=28)
        manager.walk.update({
            "proposal_mode": "persistent", "max_steps": 1,
            "action_scale": .5,
            "persistent_proposal": {
                "attempt_direction_noise_std": .20,
                "hop_direction_noise_std": 0.0,
                "step_noise_std": 0.0,
            },
        })
        heading = np.array([.5, -.25, .1, 0.0, .2, -.1])
        manager._active_proposal_direction = heading
        parent = self._snapshot(1)
        actions = []
        restored = []
        expected_rng = np.random.default_rng()
        expected_rng.bit_generator.state = deepcopy(manager.rng.bit_generator.state)

        class Env:
            def restore_curriculum_state(inner_self, state, **kwargs):
                restored.append(state)

            def step_for_curriculum_generation(inner_self, action):
                actions.append(np.asarray(action).copy())
                return CurriculumGenerationResult(
                    state=self._snapshot(-1, x=.01),
                    geometric_success=False, success=False, unsafe=False,
                    unsafe_force=False, unsafe_torque=False,
                    unsafe_workspace=False, position_error=.01,
                    rotation_error=0.0, pose_distance=.01,
                )

        manager.env = Env()
        report = GenerationReport()
        for _ in range(3):
            manager._generate_hop_snapshot(parent, 1.0, report)

        expected = np.asarray([
            np.clip(
                heading + expected_rng.normal(0.0, .20, size=6), -1.0, 1.0,
            ) * .5
            for _ in range(3)
        ])
        np.testing.assert_array_equal(actions, expected)
        self.assertTrue(all(state is parent for state in restored))
        self.assertEqual(len({tuple(action) for action in actions}), 3)
        self.assertTrue(all(value > 0.0 for value in report.attempt_to_heading_deviations))

    def test_two_branches_from_the_same_parent_get_independent_headings(self):
        manager = self._manager(seed=29)
        manager.walk.update({
            "proposal_mode": "persistent",
            "persistent_proposal": {
                "attempt_direction_noise_std": 0.0,
                "hop_direction_noise_std": 0.0,
                "step_noise_std": 0.0,
            },
            "proposal": {
                "guided_fraction": 0.0, "guided_noise_std": 0.0,
                "memory_size_per_parent": 4,
            },
        })
        parent = self._snapshot(1, depth=1, x=.01, success_rate=1.0)
        manager.pools["mastered"] = [parent]
        manager.next_state_id = 100
        headings = []

        def generate(seed, scale, report):
            headings.append(manager._active_proposal_direction.copy())
            return self._snapshot(-1, x=1.0 + len(headings)), None

        manager._generate_hop_snapshot = generate
        manager._is_duplicate = lambda candidate, additional: False
        manager.qualify_candidates = lambda model, states: [
            replace(states[0], success_rate=.6)
        ]

        report = manager._expand_branches(object(), seeds=[parent, parent])

        self.assertEqual(report.expansion_branches, 2)
        self.assertEqual(len(headings), 2)
        self.assertFalse(np.array_equal(headings[0], headings[1]))

    def test_branch_can_curve_across_hops_without_a_direction_constraint(self):
        manager, seeds, _, _ = self._multihop_manager([1.0, 1.0, .6])
        manager.walk.update({
            "proposal_mode": "persistent",
            "persistent_proposal": {
                "attempt_direction_noise_std": 0.0,
                "hop_direction_noise_std": 0.15,
                "step_noise_std": 0.0,
            },
            "proposal": {
                "guided_fraction": 1.0, "guided_noise_std": 0.0,
                "memory_size_per_parent": 4,
            },
        })
        x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        xy = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        manager.proposal_memory = {int(seeds[0].state_id): [x.copy()]}
        turns = iter((xy.copy(), y.copy()))
        manager._next_branch_heading = lambda heading: next(turns)
        generated = manager._generate_hop_snapshot
        headings = []

        def capture_heading(seed, scale, report):
            headings.append(manager._active_proposal_direction.copy())
            return generated(seed, scale, report)

        manager._generate_hop_snapshot = capture_heading
        report = manager._expand_branches(object())

        np.testing.assert_array_equal(headings, np.asarray([x, xy, y]))
        self.assertEqual(report.expansion_candidates, 3)
        self.assertEqual(report.successive_hop_heading_opposition, 0)

    def test_legacy_bootstrap_honours_persistent_proposal_mode(self):
        manager = self._manager(seed=26)
        manager.config["candidates_per_update"] = 3
        manager.walk.update({
            "proposal_mode": "persistent", "walks_per_seed": 1,
            "max_steps": 3, "action_scale": .5,
            "persistent_proposal": {
                "attempt_direction_noise_std": 0.0,
                "hop_direction_noise_std": 0.0,
                "step_noise_std": 0.0,
            },
        })
        manager.next_state_id = 10
        manager._is_duplicate = lambda candidate, additional: False
        parent = self._snapshot(1)
        actions = []
        step = 0

        class Env:
            def restore_curriculum_state(self, *args, **kwargs):
                return None

            def step_for_curriculum_generation(inner_self, action):
                nonlocal step
                step += 1
                actions.append(np.asarray(action).copy())
                return CurriculumGenerationResult(
                    state=self._snapshot(-1, x=.01 * step),
                    geometric_success=False, success=False, unsafe=False,
                    unsafe_force=False, unsafe_torque=False,
                    unsafe_workspace=False, position_error=.01 * step,
                    rotation_error=0.0, pose_distance=.01 * step,
                )

        manager.env = Env()
        candidates, report = manager.generate_candidates([parent])
        self.assertEqual(len(candidates), 3)
        np.testing.assert_array_equal(
            np.asarray(actions), np.tile(actions[0], (3, 1)),
        )
        self.assertEqual(report.persistent_attempts, 1)
        self.assertEqual(report.independent_attempts, 0)

    def test_proposal_metrics_distinguish_uniform_and_guided_attempts(self):
        uniform, _, _, _ = self._multihop_manager([.6])
        uniform.walk["proposal"] = {
            "guided_fraction": 0.0, "guided_noise_std": .2,
            "memory_size_per_parent": 4,
        }
        uniform_report = uniform._expand_branches(object())
        self.assertEqual(uniform_report.proposal_uniform_attempts, 1)
        self.assertEqual(uniform_report.proposal_uniform_unique, 1)

        guided, seeds, _, _ = self._multihop_manager([.6])
        guided.walk["proposal"] = {
            "guided_fraction": 1.0, "guided_noise_std": .2,
            "memory_size_per_parent": 4,
        }
        guided.proposal_memory = {
            int(seeds[0].state_id): [np.zeros(6)],
        }
        guided_report = guided._expand_branches(object())
        self.assertEqual(guided_report.proposal_guided_attempts, 1)
        self.assertEqual(guided_report.proposal_guided_unique, 1)

    def test_frontier_lifecycle_tracks_remain_promote_and_demote(self):
        manager = self._manager()
        manager.update_count = 4
        manager.state_lifecycle = {
            1: StateLifecycleStats(
                created_update=1, frontier_since_update=2,
                consecutive_frontier_updates=2,
            ),
        }
        manager._update_lifecycle_after_revalidation(
            {1: "frontier"}, {1: "frontier"},
        )
        stats = manager.state_lifecycle[1]
        self.assertEqual(stats.revalidation_count, 1)
        self.assertEqual(stats.last_revalidated_update, 5)
        self.assertEqual(stats.frontier_since_update, 2)
        self.assertEqual(stats.consecutive_frontier_updates, 3)

        manager.update_count = 5
        manager._update_lifecycle_after_revalidation(
            {1: "frontier"}, {1: "mastered"},
        )
        self.assertIsNone(stats.frontier_since_update)
        self.assertEqual(stats.consecutive_frontier_updates, 0)

        manager.update_count = 6
        manager._update_lifecycle_after_revalidation(
            {1: "mastered"}, {1: "frontier"},
        )
        self.assertEqual(stats.frontier_since_update, 7)
        self.assertEqual(stats.consecutive_frontier_updates, 1)

    def test_nearest_ancestor_diagnostic_is_observational(self):
        manager = self._manager()
        manager.config["diagnostics"] = {
            "near_ancestor_position_m": .001,
            "near_ancestor_rotation_deg": 1.0,
        }
        ancestor = self._snapshot(1, x=.010)
        parent = self._snapshot(2, parent_id=1, depth=2, x=.020)
        candidate = self._snapshot(-1, parent_id=2, depth=3, x=.0105)
        before = (candidate.parent_id, candidate.generation_depth)
        position, rotation, near = manager._ancestor_diagnostics(
            candidate, [ancestor, parent],
        )
        self.assertAlmostEqual(position, .0005)
        self.assertEqual(rotation, 0.0)
        self.assertTrue(near)
        self.assertEqual((candidate.parent_id, candidate.generation_depth), before)

    def test_deduplication_remembers_an_accepted_state_pruned_during_update(self):
        manager, seeds, _, _ = self._multihop_manager(
            [1.0], max_hops=4,
        )
        manager.config["max_pool_size"] = 1
        manager.config["expansion"]["max_attempts_per_hop"] = 1
        manager.deduplication = {
            "position_tolerance": 1e-6,
            "rotation_tolerance_deg": 1e-6,
        }
        repeated_snapshot = self._snapshot(-1, x=2.0)
        qualification_calls = 0

        def generate(seed, scale, report):
            report.generated += 1
            return repeated_snapshot, None

        def qualify(model, states):
            nonlocal qualification_calls
            qualification_calls += 1
            return [replace(states[0], success_rate=1.0)]

        manager._generate_hop_snapshot = generate
        manager._is_duplicate = ReverseCurriculumManager._is_duplicate.__get__(
            manager, ReverseCurriculumManager,
        )
        manager.qualify_candidates = qualify

        report = manager._expand_branches(object())

        # Le premier candidat est accepté puis immédiatement pruné par la
        # limite du pool. Sa pose reste néanmoins connue et stoppe le hop 2.
        self.assertEqual(report.expansion_candidates, 1)
        self.assertEqual(report.expansion_hops, 2)
        self.assertEqual(report.deduplicated_rejected, 1)
        self.assertEqual(report.stop_reasons, {"attempt_budget": 1})
        self.assertEqual(qualification_calls, 1)
        self.assertEqual(manager.next_state_id, 101)
        self.assertEqual(manager.pools["mastered"], seeds)

    def test_multihop_does_not_touch_model_or_replay_counters(self):
        class Replay:
            def size(self):
                return 12

        model = SimpleNamespace(num_timesteps=34, replay_buffer=Replay())
        manager, _, _, _ = self._multihop_manager([1.0, .6])
        before = (model.num_timesteps, model.replay_buffer.size())

        manager._expand_branches(model)

        self.assertEqual(
            (model.num_timesteps, model.replay_buffer.size()), before,
        )

    def test_one_hop_keeps_only_the_last_safe_non_success_snapshot(self):
        manager = self._manager(seed=91)
        manager.walk = {
            "walks_per_seed": 99, "max_steps": 3, "action_scale": .5,
        }
        snapshots = [
            self._snapshot(-1, x=.1),
            self._snapshot(-1, x=.2),
            self._snapshot(-1, x=.3),
        ]
        results = iter([
            SimpleNamespace(state=snapshots[0], unsafe=False, success=True),
            SimpleNamespace(state=snapshots[1], unsafe=False, success=False),
            SimpleNamespace(state=snapshots[2], unsafe=False, success=False),
        ])
        actions = []

        class Env:
            def restore_curriculum_state(self, *args, **kwargs):
                return None

            def step_for_curriculum_generation(self, action):
                actions.append(action)
                return next(results)

        manager.env = Env()
        report = GenerationReport()
        state, reason = manager._generate_hop_snapshot(
            self._snapshot(1, depth=1), 1.5, report,
        )

        self.assertIs(state, snapshots[2])
        self.assertIsNone(reason)
        self.assertEqual(report.generated, 3)
        self.assertEqual(report.successful_excluded, 1)
        self.assertEqual(len(actions), 3)  # walks_per_seed n'est pas utilisé ici.
        self.assertLessEqual(float(np.max(np.abs(actions))), .75)

    def test_non_finite_snapshot_is_invalid(self):
        candidate = self._snapshot(-1)
        candidate.task_position[0] = np.nan
        self.assertFalse(
            ReverseCurriculumManager._candidate_snapshot_is_valid(candidate)
        )

    def test_revalidation_frequency_runs_first_update_then_every_n_updates(self):
        manager = self._manager()
        manager.config["revalidation"]["every_n_curriculum_updates"] = 2
        manager.update_count = 0
        manager.next_update_timesteps = 50_000
        calls = []

        def revalidate(model):
            calls.append(manager.update_count)
            manager.last_revalidation_report = SimpleNamespace()
            return 0

        manager.revalidate_existing = revalidate
        manager._expand_branches = lambda model: GenerationReport()

        manager.update(object())
        manager.update(object())
        self.assertEqual(
            manager.last_revalidation_report.total_revalidated, 0,
        )
        self.assertEqual(manager.last_revalidation_report.mastered_rollouts, 0)
        self.assertEqual(manager.last_revalidation_report.too_hard_rollouts, 0)
        self.assertEqual(manager.last_revalidation_report.wall_time, 0.0)
        for _ in range(3):
            manager.update(object())

        self.assertEqual(calls, [0, 2, 4])
        self.assertEqual(manager.update_count, 5)

    def test_mastered_boundary_follows_current_mastered_children(self):
        manager = self._manager()
        a, b, c = self._states(
            [4.0, 9.0, 6.0], depths=[1, 2, 3],
            parents=[None, 1, 2], start_id=1,
        )
        manager.pools["mastered"] = [a, b]
        manager.pools["frontier"] = [c]
        self.assertEqual(manager.mastered_boundary_states(), [b])
        states_by_id, children_by_parent = manager._build_lineage_index()
        self.assertEqual(set(states_by_id), {1, 2, 3})
        self.assertEqual(
            [state.state_id for state in children_by_parent[1]], [2],
        )
        self.assertEqual(
            [state.state_id for state in children_by_parent[2]], [3],
        )

        manager.pools["mastered"].append(c)
        manager.pools["frontier"] = []
        self.assertEqual(manager.mastered_boundary_states(), [c])

    def test_branched_boundary_sampling_is_uniform_and_ignores_pose_distance(self):
        manager = self._manager(seed=21)
        a = self._states([4.0], depths=[1], start_id=10)[0]
        b, c = self._states(
            [50.0, 2.0], depths=[2, 2], parents=[10, 10], start_id=11,
        )
        manager.pools["mastered"] = [a, b, c]
        self.assertEqual(
            {state.state_id for state in mastered_boundary_states(
                manager.pools["mastered"],
            )},
            {11, 12},
        )
        sampled = {
            manager._expansion_seeds()[0].state_id for _ in range(200)
        }
        self.assertEqual(sampled, {11, 12})
        self.assertEqual(
            manager.training_reset_pools()["historical"],
            manager.pools["mastered"],
        )

    def test_mastered_boundary_and_expansion_fallbacks_handle_small_pools(self):
        manager = self._manager()
        only_mastered = self._states([2.0])[0]
        manager.pools["mastered"] = [only_mastered]
        self.assertEqual(manager.mastered_edge_states(), [only_mastered])
        self.assertEqual(manager._expansion_seeds(), [only_mastered])

        only_frontier = self._states([3.0])[0]
        manager.pools["mastered"] = []
        manager.pools["frontier"] = [only_frontier]
        self.assertEqual(manager._expansion_seeds(), [only_frontier])

        manager.pools["frontier"] = []
        only_too_hard = self._states([5.0])[0]
        manager.pools["too_hard"] = [only_too_hard]
        self.assertEqual(manager._expansion_seeds(), [manager.goal_seed])
        self.assertEqual(manager.last_expansion_seed_distances, [0.0])
        self.assertEqual(manager.last_expansion_seed_depths, [0])

    def test_deprecated_mastered_edge_fraction_is_ignored(self):
        a, b = self._states(
            [100.0, 1.0], depths=[1, 2], parents=[None, 1], start_id=1,
        )
        for old_fraction in (0, .25, 1.0, "ancienne-valeur"):
            with self.subTest(old_fraction=old_fraction):
                self.assertEqual(
                    mastered_edge_states([a, b], old_fraction), [b],
                )

    def test_duplicate_keeps_the_existing_lineage_without_multi_parent(self):
        manager = self._manager()
        manager.deduplication = {
            "position_tolerance": 1e-3,
            "rotation_tolerance_deg": 1.0,
        }
        existing = SimpleNamespace(
            state_id=10, parent_id=4, generation_depth=3,
            task_position=np.array([.01, .02, .03]),
            task_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        candidate = SimpleNamespace(
            state_id=99, parent_id=80, generation_depth=9,
            task_position=existing.task_position.copy(),
            task_quaternion=existing.task_quaternion.copy(),
        )
        manager.pools["frontier"] = [existing]

        self.assertTrue(manager._is_duplicate(candidate, []))
        self.assertEqual(
            (existing.state_id, existing.parent_id, existing.generation_depth),
            (10, 4, 3),
        )

    def test_too_hard_selection_prioritizes_children_of_mastered(self):
        mastered = self._states([4.0], depths=[1], start_id=100)
        preferred = self._states(
            [50.0], depths=[2], parents=[100], start_id=200,
        )[0]
        unrelated = self._states(
            [.001], depths=[8], parents=[999], start_id=201,
        )[0]
        selected = select_too_hard_by_lineage(
            [unrelated, preferred], mastered, 1, np.random.default_rng(1),
        )
        self.assertEqual([state.state_id for state in selected], [200])
        completed = select_too_hard_by_lineage(
            [unrelated, preferred], mastered, 2, np.random.default_rng(1),
        )
        self.assertEqual(
            [state.state_id for state in completed], [200, 201],
        )

    def test_too_hard_near_pool_contains_only_direct_mastered_children(self):
        mastered = self._states([4.0], depths=[1], start_id=100)
        direct, unrelated, grandchild = self._states(
            [4.1, 4.2, 4.3], depths=[2, 2, 3],
            parents=[100, 999, 200], start_id=200,
        )
        eligible = too_hard_near_states(
            [unrelated, grandchild, direct], mastered,
        )
        self.assertEqual(eligible, [direct])

        manager = self._manager()
        manager.pools = {
            "mastered": mastered,
            "frontier": [],
            "too_hard": [unrelated, grandchild, direct],
        }
        reset_pools = manager.training_reset_pools()
        self.assertEqual(reset_pools["mastered_boundary"], mastered)
        self.assertEqual(reset_pools["too_hard_near"], [direct])

    def test_too_hard_selection_falls_back_uniformly_without_mastered(self):
        too_hard = self._states([50.0, .001], start_id=300)
        selected = select_too_hard_by_lineage(
            too_hard, [], 1, np.random.default_rng(3),
        )
        self.assertEqual(len(selected), 1)
        self.assertIn(selected[0].state_id, {300, 301})

    def _manager_with_revalidated_too_hard(self, success_rate):
        manager = self._manager(seed=5)
        mastered = self._with_success_rate(self._states([4.0])[0], 1.0)
        too_hard = self._with_success_rate(self._states([4.1])[0], 0.0)
        too_hard.state_id = 101
        too_hard.parent_id = mastered.state_id
        too_hard.generation_depth = mastered.generation_depth + 1
        manager.pools = {
            "too_hard": [too_hard], "frontier": [],
            "mastered": [mastered],
        }
        manager.qualify_candidates = lambda model, states: [
            self._with_success_rate(state, success_rate) for state in states
        ]
        return manager, too_hard

    def test_too_hard_revalidation_moves_state_to_frontier(self):
        manager, state = self._manager_with_revalidated_too_hard(.6)
        self.assertEqual(manager.revalidate_existing(object()), 1)
        self.assertEqual(manager.pools["too_hard"], [])
        self.assertEqual(manager.pools["mastered"][0].pose_distance, 4.0)
        self.assertEqual(
            [(item.state_id, item.parent_id, item.generation_depth,
              item.success_rate)
             for item in manager.pools["frontier"]],
            [(state.state_id, state.parent_id, state.generation_depth, .6)],
        )
        self.assertEqual(manager.last_revalidation_report.too_hard_revalidated, 1)
        self.assertEqual(manager.last_revalidation_report.too_hard_to_frontier, 1)
        self.assertEqual(manager.last_revalidation_report.too_hard_rollouts, 5)
        self.assertGreaterEqual(manager.last_revalidation_report.wall_time, 0.0)

    def test_too_hard_revalidation_moves_state_to_mastered(self):
        manager, state = self._manager_with_revalidated_too_hard(1.0)
        manager.revalidate_existing(object())
        self.assertEqual(manager.pools["too_hard"], [])
        migrated = [
            item for item in manager.pools["mastered"]
            if item.state_id == state.state_id
        ]
        self.assertEqual(len(migrated), 1)
        self.assertEqual(migrated[0].success_rate, 1.0)
        self.assertEqual(manager.last_revalidation_report.too_hard_to_mastered, 1)

    def test_too_hard_revalidation_can_remain_hard_without_duplication(self):
        manager, state = self._manager_with_revalidated_too_hard(0.0)
        manager.revalidate_existing(object())
        self.assertEqual(manager.pools["frontier"], [])
        remaining = [
            item for item in manager.pools["too_hard"]
            if item.state_id == state.state_id
        ]
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].success_rate, 0.0)
        self.assertEqual(
            manager.last_revalidation_report.too_hard_remained_hard, 1,
        )

    def test_revalidation_rollout_failure_leaves_pools_unchanged(self):
        manager, _ = self._manager_with_revalidated_too_hard(.6)
        before = {name: list(states) for name, states in manager.pools.items()}

        def fail(model, states):
            raise RuntimeError("rollout failure")

        manager.qualify_candidates = fail
        with self.assertRaisesRegex(RuntimeError, "rollout failure"):
            manager.revalidate_existing(object())
        self.assertEqual(manager.pools, before)

    def test_revalidation_cannot_modify_lineage(self):
        manager, _ = self._manager_with_revalidated_too_hard(.6)
        before = {name: list(states) for name, states in manager.pools.items()}

        def change_parent(model, states):
            changed = [self._with_success_rate(state, .6) for state in states]
            changed[0].parent_id = 999
            return changed

        manager.qualify_candidates = change_parent
        with self.assertRaisesRegex(RuntimeError, "lineage"):
            manager.revalidate_existing(object())
        self.assertEqual(manager.pools, before)

    def test_expansion_config_migration_is_targeted_and_backward_compatible(self):
        old = {
            "success_rate_low": .1,
            "reverse_random_walk": {"min_pose_distance_increase": 1e-5},
        }
        current = {
            "success_rate_low": .1,
            "expansion": {
                "mastered_edge_fraction": .25,
                "max_hops_per_seed": 4,
                "max_candidates_per_update": 24,
                "initial_scale": 1.0,
                "scale_up_factor": 1.25,
                "scale_down_factor": .7,
                "min_scale": .5,
                "max_scale": 3.0,
            },
            "revalidation": {"every_n_curriculum_updates": 3},
        }
        self.assertTrue(
            ReverseCurriculumManager._curriculum_configs_compatible(
                old, current,
            )
        )
        saved_with_strategy = deepcopy(current)
        changed_strategy = deepcopy(current)
        changed_strategy["expansion"].update({
            "max_hops_per_seed": 9,
            "max_candidates_per_update": 7,
            "initial_scale": .8,
            "scale_up_factor": 1.1,
            "scale_down_factor": .6,
            "min_scale": .3,
            "max_scale": 2.0,
        })
        self.assertTrue(
            ReverseCurriculumManager._curriculum_configs_compatible(
                saved_with_strategy, changed_strategy,
            )
        )
        saved_with_strategy["reverse_random_walk"] = {
            "proposal_mode": "independent",
            "persistent_proposal": {"step_noise_std": .10},
        }
        changed_strategy["reverse_random_walk"] = {
            "proposal_mode": "persistent",
            "persistent_proposal": {"step_noise_std": .20},
        }
        self.assertTrue(
            ReverseCurriculumManager._curriculum_configs_compatible(
                saved_with_strategy, changed_strategy,
            )
        )
        current["success_rate_low"] = .2
        self.assertFalse(
            ReverseCurriculumManager._curriculum_configs_compatible(
                old, current,
            )
        )

    def test_resume_allows_curriculum_reset_probability_change(self):
        saved = {
            "success_rate_low": .1,
            "curriculum_reset_probability": .80,
        }
        current = deepcopy(saved)
        current["curriculum_reset_probability"] = .95
        self.assertTrue(
            ReverseCurriculumManager._curriculum_configs_compatible(
                saved, current,
            )
        )

    def test_resume_still_rejects_structural_curriculum_change(self):
        saved = {
            "success_rate_low": .1,
            "curriculum_reset_probability": .80,
        }
        structurally_different = deepcopy(saved)
        structurally_different["curriculum_reset_probability"] = .95
        structurally_different["success_rate_low"] = .2
        self.assertFalse(
            ReverseCurriculumManager._curriculum_configs_compatible(
                saved, structurally_different,
            )
        )

    def test_v21_only_adds_curriculum_to_v20(self):
        v20 = load_config("configs/test1V20.yaml")
        v21 = load_config("configs/test1V21.yaml")
        v20["curriculum"] = v21["curriculum"]
        self.assertEqual(v20, v21)

    def test_pool_pruning_preserves_depth_extremes_not_pose_extremes(self):
        manager = object.__new__(ReverseCurriculumManager)
        manager.config = {"max_pool_size": 2}
        shallow, middle, deep = self._states(
            [50.0, .001, 2.0], depths=[0, 5, 10],
            parents=[None, 1, 2], start_id=1,
        )
        for state in (shallow, middle, deep):
            state.success_rate = 1.0
        manager.pools = {
            "too_hard": [], "frontier": [],
            "mastered": [middle, deep, shallow],
        }
        manager._prune()
        self.assertIn(shallow, manager.pools["mastered"])
        self.assertIn(deep, manager.pools["mastered"])
        self.assertNotIn(middle, manager.pools["mastered"])

    def test_pool_pruning_keeps_deep_leaves_from_distinct_branches(self):
        manager = object.__new__(ReverseCurriculumManager)
        manager.config = {"max_pool_size": 3}
        root, middle, leaf_a, leaf_b = self._states(
            [8.0, 1.0, 100.0, .001], depths=[0, 5, 10, 10],
            parents=[None, 1, 2, 2], start_id=1,
        )
        for state in (root, middle, leaf_a, leaf_b):
            state.success_rate = 1.0
        manager.pools = {
            "too_hard": [], "frontier": [],
            "mastered": [leaf_a, middle, leaf_b, root],
        }

        manager._prune()

        kept_ids = {state.state_id for state in manager.pools["mastered"]}
        self.assertEqual(kept_ids, {1, 3, 4})

    def test_reward_function_is_independent_of_curriculum(self):
        v20 = load_config("configs/test1V20.yaml")
        v21 = load_config("configs/test1V21.yaml")
        status = assess_status(
            position_error=.01, rotation_error=.02,
            max_force=3.0, max_torque=.4, workspace_error=.01,
            step_count=5, config=v20["success"],
            max_episode_steps=v20["simulation"]["max_episode_steps"],
        )
        arguments = dict(
            position_error=.01, rotation_error=.02, max_force=3.0,
            max_torque=.4, action=np.full(6, .2), status=status,
        )
        self.assertEqual(
            reward_components(**arguments, config=v20["reward"]),
            reward_components(**arguments, config=v21["reward"]),
        )


if __name__ == "__main__":
    unittest.main()
