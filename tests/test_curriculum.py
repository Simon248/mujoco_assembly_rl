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
    ReverseCurriculumManager, classify_success_rate,
    historical_quantile_bins, mastered_boundary_states,
    mastered_edge_states, select_too_hard_by_lineage,
    select_training_start,
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
        np.testing.assert_allclose(
            self.env._integration_state(), state.mj_state, atol=1e-10,
        )
        np.testing.assert_allclose(
            self.env.admittance.offset, state.admittance_offset, atol=1e-12,
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
            "success_rate_low": .10,
            "success_rate_high": .90,
            "max_pool_size": 100,
            "expansion": {"mastered_edge_fraction": .25},
            "revalidation": {
                "mastered_samples_per_update": 0,
                "too_hard_samples_per_update": 1,
            },
        }
        manager.walk = {"walks_per_seed": 1}
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
        self.assertEqual(selection.source, "curriculum_historical")

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
            "expansion": {"mastered_edge_fraction": .25},
        }
        self.assertTrue(
            ReverseCurriculumManager._curriculum_configs_compatible(
                old, current,
            )
        )
        current["expansion"]["future_semantic_change"] = True
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
