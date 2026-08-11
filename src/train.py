"""Entraînement SAC vectorisé et persistance d'un run reproductible."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tarfile
from typing import Callable
import warnings

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from src.assembly_env import TenonMortaiseEnv
from src.config import load_config, save_resolved_config
from src.curriculum import (
    RESET_SOURCES as CURRICULUM_RESET_SOURCES, SAMPLING_SOURCE_NAMES,
    ReverseCurriculumManager, configured_start_sampling_probabilities,
    reset_probabilities_for_transition_targets,
    update_sampling_episode_length_ema,
)
from src.curriculum_diagnostics import (
    ExpansionDiagnostics, curriculum_state_rows, write_curriculum_diagnostics,
)


MONITOR_FIELDS = (
    "geometric_success", "success", "safe_success", "unsafe",
    "unsafe_force", "unsafe_torque", "unsafe_workspace",
    "termination_reason", "position_error", "rotation_error",
    "position_error_x", "position_error_y", "position_error_z",
    "rotation_error_x", "rotation_error_y", "rotation_error_z",
    "action_x", "action_y", "action_z", "action_rx", "action_ry", "action_rz",
    "force", "torque", "max_force_substep", "max_torque_substep",
    "episode_max_force", "episode_max_torque", "friction_scale",
    "rotation_equivalent_distance", "pose_distance", "reward_pose",
    "reward_force", "reward_torque", "reward_action",
    "reward_step", "reward_success", "reward_unsafe", "reward_timeout",
    "episode_reward_pose", "episode_reward_force", "episode_reward_torque",
    "episode_reward_action", "episode_reward_step",
    "episode_reward_success", "episode_reward_unsafe", "episode_reward_timeout",
    "reset_source", "curriculum_start_position_error",
    "curriculum_start_rotation_error", "curriculum_start_pose_distance",
    "curriculum_start_success_rate", "curriculum_start_state_id",
    "curriculum_start_generation_depth", "is_curriculum_reset",
    "best_pose_metric", "position_error_at_best_pose",
    "rotation_error_at_best_pose", "reached_20mm", "reached_10mm",
    "reached_5mm", "reached_2mm",
)

EVAL_MONITOR_FIELDS = MONITOR_FIELDS + (
    "final_position_error", "final_rotation_error",
    "best_position_error", "best_rotation_error", "max_force", "max_torque",
    "training_timesteps",
)


class EpisodeMetricsCallback(BaseCallback):
    """Expose les métriques terminales centralisées dans TensorBoard."""

    def _on_step(self) -> bool:
        terminal_infos = [
            info for done, info in zip(
                self.locals.get("dones", []), self.locals.get("infos", [])
            ) if done
        ]
        for key in MONITOR_FIELDS:
            values = [
                float(info[key]) for info in terminal_infos
                if isinstance(info.get(key), (bool, int, float, np.number))
                and np.isfinite(float(info[key]))
            ]
            if values:
                self.logger.record(f"assembly/{key}", float(np.mean(values)))
        return True


class TrainingTimestepEvalCallback(EvalCallback):
    """Attach the evaluated model timestep to every evaluation episode."""

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.set_attr("training_timesteps", self.num_timesteps)
            try:
                curriculum_roles = self.eval_env.get_attr(
                    "allow_curriculum_resets"
                )
            except AttributeError:
                # Environnements synthétiques des tests, sans logique RCG.
                curriculum_roles = []
            if any(curriculum_roles):
                raise RuntimeError("EvalCallback ne doit jamais autoriser le curriculum")
            self.logger.record("eval/reset_source_true_start", 1.0)
        return super()._on_step()


class ReverseCurriculumCallback(BaseCallback):
    """Met à jour le RCG hors workers et coordonne ses checkpoints avec SAC."""

    RESET_SOURCES = CURRICULUM_RESET_SOURCES
    SOURCE_LABELS = {
        "true_start": "true_start",
        "curriculum_frontier": "frontier",
        "curriculum_historical": "historical",
        "curriculum_mastered_boundary": "mastered_boundary",
        "curriculum_too_hard_near": "too_hard_near",
    }
    CURRICULUM_SOURCES = CURRICULUM_RESET_SOURCES[1:]

    def __init__(
        self, manager: ReverseCurriculumManager, training_env: VecEnv,
        output: Path, algorithm: str, checkpoint_interval: int,
    ) -> None:
        super().__init__()
        self.manager = manager
        self.curriculum_workers = training_env
        self.output = output
        self.algorithm = algorithm
        self.checkpoint_interval = int(checkpoint_interval)
        self.next_checkpoint_timesteps: int | None = None
        self.source_episode_counts = {
            source: 0 for source in self.RESET_SOURCES
        }
        self.source_transition_counts = {
            source: 0 for source in self.RESET_SOURCES
        }
        self.source_success_counts = {
            source: 0 for source in self.RESET_SOURCES
        }
        self.source_episode_lengths: dict[str, list[float]] = {
            source: [] for source in self.RESET_SOURCES
        }
        self.used_start_distances: dict[str, list[float]] = {
            source: [] for source in self.CURRICULUM_SOURCES
        }
        self.true_start_diagnostics: dict[str, list[float]] = {
            name: [] for name in (
                "best_position_error", "best_pose_metric",
                "position_error_at_best_pose", "rotation_error_at_best_pose",
                "reached_20mm", "reached_10mm", "reached_5mm", "reached_2mm",
            )
        }
        # Window: episodes completed since the preceding curriculum update.
        self.frontier_reset_counts: dict[int, int] = {}
        # Pools are broadcast only at update boundaries, so this snapshot is
        # exactly the reset distribution used throughout the current window.
        if not hasattr(self.manager, "sampling_episode_length_ema"):
            self.manager.sampling_episode_length_ema = {
                name: 1.0 for name in SAMPLING_SOURCE_NAMES
            }
        self.sampling_targets_used = self._sampling_targets()
        self.sampling_effective_resets_used = (
            self._effective_reset_probabilities(self.sampling_targets_used)
        )
        self.sampling_episode_length_ema_used = dict(
            self.manager.sampling_episode_length_ema
        )

    def _broadcast_pool(self, probabilities=None) -> None:
        pools = self.manager.training_reset_pools()
        self.curriculum_workers.env_method(
            "set_curriculum_reset_pools",
            pools["frontier"], pools["historical"],
            pools.get("mastered_boundary", []), pools.get("too_hard_near", []),
        )
        self.curriculum_workers.env_method(
            "set_curriculum_sampling_probabilities",
            (
                self.sampling_effective_resets_used
                if probabilities is None else probabilities
            ),
        )

    def _restore_worker_rngs(self) -> None:
        states = self.manager.worker_rng_states
        if states is None:
            return
        worker_count = len(self.curriculum_workers.get_attr("worker_rank"))
        if len(states) != worker_count:
            warnings.warn(
                "Le nombre d'états RNG curriculum ne correspond pas aux workers; "
                "les seeds de workers seront réutilisés.", RuntimeWarning,
            )
            return
        for index, state in enumerate(states):
            self.curriculum_workers.env_method(
                "set_worker_rng_state", state, indices=index,
            )

    def _worker_rng_states(self) -> list[dict]:
        return self.curriculum_workers.env_method("get_worker_rng_state")

    def _record_distance_distribution(
        self, prefix: str, values: list[float], *, quartiles: bool,
        exclude: str | tuple[str, ...] | None = None,
    ) -> None:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if not array.size:
            return
        statistics = [
            ("min", np.min(array)),
            ("median", np.median(array)),
            ("max", np.max(array)),
        ]
        if quartiles:
            statistics[1:1] = [("q25", np.quantile(array, 0.25))]
            statistics[-1:-1] = [("q75", np.quantile(array, 0.75))]
        for statistic, value in statistics:
            self.logger.record(
                f"curriculum/{prefix}_{statistic}", float(value),
                exclude=exclude,
            )

    def _record_update_metrics(self) -> None:
        revalidation_report = getattr(
            self.manager, "last_revalidation_report", None,
        )
        if revalidation_report is not None:
            for name in (
                "too_hard_revalidated", "too_hard_to_frontier",
                "too_hard_to_mastered", "too_hard_remained_hard",
                "frontier_promoted_to_mastered", "frontier_remained_frontier",
                "frontier_demoted_to_too_hard",
            ):
                value = getattr(revalidation_report, name, None)
                if value is None:
                    continue
                self.logger.record(
                    f"curriculum/{name}", float(value),
                )

            # La revalidation et l'expansion utilisent toutes deux la policy,
            # mais leurs coûts doivent rester distinguables. Ces longues clés
            # sont conservées dans TensorBoard et exclues uniquement de stdout
            # afin d'éviter les collisions de troncature de HumanOutputFormat.
            for metric, attributes in (
                ("revalidation_mastered_rollouts", ("mastered_rollouts",)),
                ("revalidation_too_hard_rollouts", ("too_hard_rollouts",)),
                (
                    "revalidation_wall_time",
                    ("revalidation_wall_time", "wall_time"),
                ),
            ):
                value = next(
                    (
                        getattr(revalidation_report, attribute)
                        for attribute in attributes
                        if hasattr(revalidation_report, attribute)
                    ),
                    None,
                )
                if value is not None:
                    self.logger.record(
                        f"curriculum/{metric}", float(value),
                        exclude="stdout",
                    )

        expansion_report = getattr(
            self.manager, "last_generation_report", None,
        )
        if expansion_report is not None:
            for name in (
                "expansion_candidates", "expansion_hops",
                "expansion_branches", "expansion_rollouts",
                "new_mastered", "new_frontier", "new_too_hard",
                "mean_hops_per_branch", "max_hops_reached",
                "expansion_scale_mean", "expansion_scale_max",
                "frontier_found_per_candidate", "expansion_wall_time",
            ):
                value = getattr(expansion_report, name, None)
                if value is None:
                    continue
                self.logger.record(
                    f"curriculum/{name}", float(value), exclude="stdout",
                )
            # Per-update expansion funnel and exclusive branch stop causes.
            # Missing fields deliberately default to zero for backward-
            # compatible report objects used by older checkpoints/tests.
            for reason in (
                "duplicate", "force", "torque", "force_and_torque",
                "snapshot_invalid", "forbidden_contact", "other_invalid",
                "workspace", "generation_failed",
                "frontier", "too_hard", "max_hops", "attempt_budget",
                "candidate_budget",
            ):
                self.logger.record(
                    f"curriculum/stop_{reason}",
                    float(getattr(expansion_report, "stop_reasons", {}).get(reason, 0)),
                    exclude="stdout",
                )
            funnel_names = (
                "raw_candidates_generated", "valid_candidates",
                "nonduplicate_candidates", "qualified_candidates",
                "expansion_attempts", "attempt_no_candidate", "attempt_duplicate",
                "attempt_candidate_found", "safe_prefix_candidates",
                "full_walk_candidates",
                "proposal_uniform_attempts", "proposal_guided_attempts",
                "proposal_uniform_candidates", "proposal_guided_candidates",
                "proposal_uniform_unique", "proposal_guided_unique",
                "proposal_uniform_safe_prefix", "proposal_guided_safe_prefix",
                "proposal_uniform_attempt_budget_failures",
                "proposal_guided_attempt_budget_failures",
                "persistent_attempts", "independent_attempts",
                "successive_hop_heading_opposition",
                "guided_memory_insertions",
                "guided_memory_rejected_duplicates",
                "new_states_near_ancestor",
            )
            # expansion_hops / expansion_candidates are retained legacy names:
            # hops is every reverse-walk attempt, candidates is every state
            # reaching policy qualification.
            for name in funnel_names:
                self.logger.record(
                    f"curriculum/{name}",
                    float(getattr(expansion_report, name, 0)), exclude="stdout",
                )
            hops = float(getattr(expansion_report, "expansion_hops", 0))
            raw = float(getattr(expansion_report, "raw_candidates_generated", 0))
            qualified = float(getattr(expansion_report, "qualified_candidates", 0))
            self.logger.record(
                "curriculum/raw_candidate_rate", raw / hops if hops else 0.0,
                exclude="stdout",
            )
            self.logger.record(
                "curriculum/qualification_rate", qualified / hops if hops else 0.0,
                exclude="stdout",
            )
            # These are attempt-level observations, not branch stop reasons:
            # a later retry may still make the branch progress.
            self.logger.record(
                "curriculum/reverse_force_limit_exceeded",
                float(len(getattr(expansion_report, "rejected_force_max", []))),
                exclude="stdout",
            )
            self.logger.record(
                "curriculum/reverse_torque_limit_exceeded",
                float(len(getattr(expansion_report, "rejected_torque_max", []))),
                exclude="stdout",
            )
            near_ancestor = float(getattr(
                expansion_report, "new_states_near_ancestor", 0,
            ))
            self.logger.record(
                "curriculum/new_states_near_ancestor_fraction",
                near_ancestor / qualified if qualified else 0.0,
                exclude="stdout",
            )
            for kind in ("uniform", "guided"):
                attempts = float(getattr(
                    expansion_report, f"proposal_{kind}_attempts", 0,
                ))
                candidates = float(getattr(
                    expansion_report, f"proposal_{kind}_candidates", 0,
                ))
                unique = float(getattr(
                    expansion_report, f"proposal_{kind}_unique", 0,
                ))
                self.logger.record(
                    f"curriculum/{kind}_unique_per_attempt",
                    unique / attempts if attempts else 0.0, exclude="stdout",
                )
                self.logger.record(
                    f"curriculum/{kind}_candidate_per_attempt",
                    candidates / attempts if attempts else 0.0, exclude="stdout",
                )
            for prefix, attribute, statistics in (
                ("raw_parent_translation_mm", "raw_parent_translation_mm", ("mean", "min", "max")),
                ("raw_parent_rotation_deg", "raw_parent_rotation_deg", ("mean", "min", "max")),
                ("duplicate_parent_translation_mm", "duplicate_parent_translation_mm", ("mean", "max")),
                ("duplicate_parent_rotation_deg", "duplicate_parent_rotation_deg", ("mean", "max")),
                ("duplicate_nearest_position_mm", "duplicate_nearest_position_mm", ("mean",)),
                ("duplicate_nearest_rotation_deg", "duplicate_nearest_rotation_deg", ("mean",)),
                ("reverse_steps", "reverse_steps", ("mean", "min", "max")),
                ("attempts_per_hop", "attempts_per_hop", ("mean", "max")),
                ("safe_prefix_steps", "safe_prefix_steps", ("mean", "min", "max")),
                ("nearest_ancestor_position_mm", "nearest_ancestor_position_mm", ("mean",)),
                ("nearest_ancestor_rotation_deg", "nearest_ancestor_rotation_deg", ("mean",)),
                ("rejected_force_max", "rejected_force_max", ("mean", "min", "max")),
                ("rejected_torque_max", "rejected_torque_max", ("mean", "min", "max")),
                ("rejected_force_step", "rejected_force_step", ("mean",)),
                ("rejected_torque_step", "rejected_torque_step", ("mean",)),
                ("accepted_reverse_force_max", "accepted_reverse_force_max", ("mean", "max")),
                ("accepted_reverse_torque_max", "accepted_reverse_torque_max", ("mean", "max")),
                ("candidate_final_force", "candidate_final_force", ("mean", "max")),
                ("candidate_final_torque", "candidate_final_torque", ("mean", "max")),
                ("branch_heading_changes", "branch_heading_changes", ("mean", "max")),
                (
                    "attempt_to_heading_deviation",
                    "attempt_to_heading_deviations", ("mean", "max"),
                ),
            ):
                values = np.asarray(getattr(expansion_report, attribute, []), dtype=float)
                values = values[np.isfinite(values)]
                for statistic in statistics:
                    value = 0.0 if not values.size else float(getattr(np, statistic)(values))
                    self.logger.record(
                        f"curriculum/{prefix}_{statistic}", value,
                        exclude="stdout",
                    )
            for prefix, attribute in (
                (
                    "reverse_candidate_parent_delta_position_mm",
                    "raw_parent_translation_mm",
                ),
                (
                    "reverse_candidate_parent_delta_rotation_deg",
                    "raw_parent_rotation_deg",
                ),
            ):
                values = np.asarray(
                    getattr(expansion_report, attribute, []), dtype=float,
                )
                values = values[np.isfinite(values)]
                self.logger.record(
                    f"curriculum/{prefix}_mean",
                    float(np.mean(values)) if values.size else 0.0,
                    exclude="stdout",
                )
                self.logger.record(
                    f"curriculum/{prefix}_max",
                    float(np.max(values)) if values.size else 0.0,
                    exclude="stdout",
                )
            for outcome, attribute in (
                ("rejected", "rejected_contact_counts"),
                ("accepted", "accepted_contact_counts"),
            ):
                counts = getattr(expansion_report, attribute, {})
                for category in (
                    "piece_fixture", "piece_other", "fixture_other", "unknown",
                ):
                    self.logger.record(
                        f"curriculum/{outcome}_reverse_contact_{category}",
                        float(counts.get(category, 0)), exclude="stdout",
                    )
            # One compact update-level summary: reverse trajectory diagnostics
            # never emit one line per MuJoCo substep or candidate.
            stops = getattr(expansion_report, "stop_reasons", {})
            def wrench_summary(attribute: str) -> str:
                samples = np.asarray(
                    getattr(expansion_report, attribute, []), dtype=float,
                )
                samples = samples[np.isfinite(samples)]
                if not samples.size:
                    return "n/a"
                return f"mean={np.mean(samples):.3g} max={np.max(samples):.3g}"

            print(
                "[Curriculum expansion]\n"
                f"hops={int(hops)} raw={int(raw)} "
                f"valid={int(getattr(expansion_report, 'valid_candidates', 0))} "
                f"unique={int(getattr(expansion_report, 'nonduplicate_candidates', 0))} "
                f"qualified={int(qualified)} "
                f"attempts={int(getattr(expansion_report, 'expansion_attempts', 0))}\n"
                "attempts/hop: "
                f"{wrench_summary('attempts_per_hop')}\n"
                "candidates: "
                f"safe_prefix={int(getattr(expansion_report, 'safe_prefix_candidates', 0))} "
                f"full_walk={int(getattr(expansion_report, 'full_walk_candidates', 0))} "
                f"duplicates={int(getattr(expansion_report, 'attempt_duplicate', 0))} "
                f"no_candidate={int(getattr(expansion_report, 'attempt_no_candidate', 0))}\n"
                "stops: " + " ".join(
                    f"{name}={int(stops.get(name, 0))}" for name in (
                        "force", "torque", "force_and_torque", "snapshot_invalid",
                        "workspace", "duplicate", "generation_failed", "frontier",
                        "too_hard", "max_hops", "candidate_budget",
                        "attempt_budget",
                    )
                ) + "\n"
                "reverse rejected: "
                f"Fmax {wrench_summary('rejected_force_max')} "
                f"Tmax {wrench_summary('rejected_torque_max')}\n"
                "reverse accepted: "
                f"Fmax {wrench_summary('accepted_reverse_force_max')} "
                f"Tmax {wrench_summary('accepted_reverse_torque_max')}\n"
                "candidate final: "
                f"F {wrench_summary('candidate_final_force')} "
                f"T {wrench_summary('candidate_final_torque')}",
            )

        distances = np.asarray(
            getattr(self.manager, "last_expansion_seed_distances", []),
            dtype=float,
        )
        distances = distances[np.isfinite(distances)]
        if distances.size:
            for statistic, value in (
                ("min", np.min(distances)),
                ("mean", np.mean(distances)),
                ("max", np.max(distances)),
            ):
                self.logger.record(
                    f"curriculum/expansion_seed_distance_{statistic}",
                    float(value), exclude="stdout",
                )

        depths = np.asarray(
            getattr(self.manager, "last_expansion_seed_depths", []),
            dtype=float,
        )
        depths = depths[np.isfinite(depths)]
        if depths.size:
            for statistic, value in (
                ("mean", np.mean(depths)),
                ("max", np.max(depths)),
            ):
                # Ces séries complètent les distances géométriques sans
                # encombrer ni risquer de tronquer la table console SB3.
                self.logger.record(
                    f"curriculum/expansion_seed_depth_{statistic}",
                    float(value), exclude="stdout",
                )

    def _record_metrics(
        self, *, include_pool_metrics: bool = False,
        include_update_metrics: bool = False,
        sampling_targets_used=None, sampling_targets_next=None,
        sampling_effective_resets_used=None,
        sampling_episode_length_ema_used=None,
    ) -> None:
        # Les listes d'épisodes peuvent contenir des dizaines de milliers de
        # valeurs : ces agrégats sont volontairement calculés à la cadence des
        # updates RCG, jamais à chaque transition d'entraînement.
        if include_pool_metrics:
            sizes = self.manager.pool_sizes()
            reset_pools = self.manager.training_reset_pools()
            self.logger.record(
                "curriculum/mastered_boundary_pool_size",
                float(len(reset_pools.get("mastered_boundary", []))),
            )
            self.logger.record(
                "curriculum/too_hard_near_pool_size",
                float(len(reset_pools.get("too_hard_near", []))),
            )
            all_depths: list[int] = []
            for name, size in sizes.items():
                label = "historical" if name == "mastered" else name
                distances = [
                    state.pose_distance for state in self.manager.pools[name]
                ]
                positions = [
                    float(getattr(state, "position_error", state.pose_distance))
                    for state in self.manager.pools[name]
                ]
                depths = [
                    int(state.generation_depth)
                    for state in self.manager.pools[name]
                ]
                all_depths.extend(depths)
                self.logger.record(f"curriculum/{label}_pool_size", float(size))
                self.logger.record(
                    f"curriculum/{name}_max_depth",
                    float(max(depths, default=0)),
                )
                self._record_distance_distribution(
                    f"{label}_pose_distance", distances, quartiles=True,
                )
                self.logger.record(
                    f"curriculum/{name}_position_max",
                    float(np.max(positions)) if positions else 0.0,
                    exclude="stdout",
                )
                if name == "mastered" and distances:
                    maximum = float(np.max(distances))
                    # Garder les deux libellés explicitement demandés tout en
                    # conservant la série historique existante ci-dessus.
                    self.logger.record(
                        "curriculum/mastered_max_pose_distance", maximum,
                        exclude="stdout",
                    )
                    self.logger.record(
                        "curriculum/mastered_pose_distance_max", maximum,
                        exclude="stdout",
                    )

            self.logger.record(
                "curriculum/max_generation_depth",
                float(max(all_depths, default=0)),
            )
            self.logger.record(
                "curriculum/mastered_boundary_count",
                float(len(self.manager.mastered_boundary_states())),
            )
            lifecycle = getattr(self.manager, "state_lifecycle", {})
            frontier_stats = [
                lifecycle.get(int(getattr(state, "state_id", -1)))
                for state in self.manager.pools["frontier"]
            ]
            frontier_stats = [stats for stats in frontier_stats if stats is not None]
            ages = [
                max(0, int(getattr(self.manager, "update_count", 0))
                    - int(stats.frontier_since_update))
                for stats in frontier_stats
                if stats.frontier_since_update is not None
            ]
            for prefix, values in (
                ("frontier_age_updates", ages),
                ("frontier_revalidation_count", [
                    stats.revalidation_count for stats in frontier_stats
                ]),
                ("frontier_consecutive_updates", [
                    stats.consecutive_frontier_updates for stats in frontier_stats
                ]),
            ):
                self.logger.record(
                    f"curriculum/{prefix}_mean",
                    float(np.mean(values)) if values else 0.0,
                    exclude="stdout",
                )
                self.logger.record(
                    f"curriculum/{prefix}_max",
                    float(np.max(values)) if values else 0.0,
                    exclude="stdout",
                )

        if include_update_metrics:
            self._record_update_metrics()

        targets_used = (
            self.sampling_targets_used
            if sampling_targets_used is None else sampling_targets_used
        )
        targets_next = (
            self._sampling_targets()
            if sampling_targets_next is None else sampling_targets_next
        )
        effective_resets_used = (
            self.sampling_effective_resets_used
            if sampling_effective_resets_used is None
            else sampling_effective_resets_used
        )
        episode_length_ema_used = (
            self.sampling_episode_length_ema_used
            if sampling_episode_length_ema_used is None
            else sampling_episode_length_ema_used
        )
        for name in SAMPLING_SOURCE_NAMES:
            self.logger.record(
                f"curriculum/sampling/target_used/{name}",
                float(getattr(targets_used, name, 0.0)), exclude="stdout",
            )
            self.logger.record(
                f"curriculum/sampling/target_next/{name}",
                float(getattr(targets_next, name, 0.0)), exclude="stdout",
            )
            self.logger.record(
                f"curriculum/sampling/target_transition/{name}",
                float(getattr(targets_used, name, 0.0)), exclude="stdout",
            )
            self.logger.record(
                f"curriculum/sampling/effective_reset/{name}",
                float(getattr(effective_resets_used, name, 0.0)),
                exclude="stdout",
            )
            self.logger.record(
                f"curriculum/sampling/episode_length_ema/{name}",
                float(episode_length_ema_used.get(name, 1.0)),
                exclude="stdout",
            )
        self.logger.record(
            "curriculum/sampling/missing_frontier_budget",
            float(getattr(targets_used, "missing_frontier_budget", 0.0)),
            exclude="stdout",
        )
        self.logger.record(
            "curriculum/sampling/fallback_budget_used",
            float(getattr(targets_used, "fallback_budget_used", 0.0)),
            exclude="stdout",
        )

        total_resets = sum(self.source_episode_counts.values())
        observed_resets = {
            source: (
                self.source_episode_counts[source] / total_resets
                if total_resets else 0.0
            )
            for source in self.RESET_SOURCES
        }
        for source, observed in observed_resets.items():
            self.logger.record(
                "curriculum/sampling/observed/"
                f"{self.SOURCE_LABELS[source]}", observed,
                exclude="stdout",
            )
        if total_resets:
            curriculum_resets = sum(
                self.source_episode_counts[source]
                for source in self.CURRICULUM_SOURCES
            )
            self.logger.record(
                "curriculum/reset_fraction_total",
                curriculum_resets / total_resets,
            )
            for source in self.RESET_SOURCES:
                observed = observed_resets[source]
                self.logger.record(
                    f"curriculum/reset_fraction_{self.SOURCE_LABELS[source]}",
                    observed,
                )

        total_transitions = sum(self.source_transition_counts.values())
        observed_transitions = {
            source: (
                self.source_transition_counts[source] / total_transitions
                if total_transitions else 0.0
            )
            for source in self.RESET_SOURCES
        }
        if total_transitions:
            for source in self.RESET_SOURCES:
                self.logger.record(
                    "curriculum/transition_fraction_"
                    f"{self.SOURCE_LABELS[source]}",
                    observed_transitions[source],
                )
        self.logger.record(
            "curriculum/sampling/transition_target_l1_error",
            float(sum(
                abs(
                    observed_transitions[source]
                    - float(getattr(
                        targets_used, self.SOURCE_LABELS[source], 0.0,
                    ))
                )
                for source in self.RESET_SOURCES
            )),
            exclude="stdout",
        )

        for source in self.RESET_SOURCES:
            label = self.SOURCE_LABELS[source]
            count = self.source_episode_counts[source]
            if count:
                self.logger.record(
                    f"curriculum/success_rate_{label}",
                    self.source_success_counts[source] / count,
                )
            lengths = np.asarray(
                self.source_episode_lengths[source], dtype=float,
            )
            lengths = lengths[np.isfinite(lengths)]
            if lengths.size:
                # HumanOutputFormat tronque les noms a 36 caracteres. Pour
                # mastered_boundary, les suffixes mean/median disparaissent
                # alors tous deux derriere le meme nom tronque et SB3 leve
                # une ValueError. Garder les noms descriptifs dans
                # TensorBoard, mais ne pas les envoyer au tableau stdout.
                self.logger.record(
                    f"curriculum/episode_length_{label}_mean",
                    float(np.mean(lengths)),
                    exclude="stdout",
                )
                self.logger.record(
                    f"curriculum/episode_length_{label}_median",
                    float(np.median(lengths)),
                    exclude="stdout",
                )

        for source in self.CURRICULUM_SOURCES:
            label = self.SOURCE_LABELS[source]
            values = self.used_start_distances[source]
            self._record_distance_distribution(
                f"used_start_distance_{label}", values, quartiles=False,
                exclude="stdout",
            )
            if values:
                self.logger.record(
                    f"curriculum/used_start_distance_{label}_mean",
                    float(np.mean(values)),
                    exclude="stdout",
                )

        for name, samples in self.true_start_diagnostics.items():
            values = np.asarray(samples, dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                self.logger.record(
                    f"true_start/{name}", float(np.mean(values)),
                    exclude="stdout",
                )
        frontier_counts = list(self.frontier_reset_counts.values())
        self.logger.record(
            "curriculum/frontier_resets_per_state_mean",
            float(np.mean(frontier_counts)) if frontier_counts else 0.0,
            exclude="stdout",
        )
        self.logger.record(
            "curriculum/frontier_resets_per_state_max",
            float(np.max(frontier_counts)) if frontier_counts else 0.0,
            exclude="stdout",
        )
        self.logger.record(
            "curriculum/frontier_unique_states_sampled",
            float(len(frontier_counts)), exclude="stdout",
        )

        if include_pool_metrics:
            frontier_mean = self.manager.frontier_success_rate_mean()
            if np.isfinite(frontier_mean):
                self.logger.record(
                    "curriculum/frontier_success_rate_mean", frontier_mean,
                )

    def _reset_cycle_metrics(self) -> None:
        for source in self.RESET_SOURCES:
            self.source_episode_counts[source] = 0
            self.source_transition_counts[source] = 0
            self.source_success_counts[source] = 0
            self.source_episode_lengths[source].clear()
        for source in self.CURRICULUM_SOURCES:
            self.used_start_distances[source].clear()
        for values in self.true_start_diagnostics.values():
            values.clear()
        self.frontier_reset_counts.clear()

    def _save_curriculum(self, path: Path) -> None:
        self.manager.save(
            path, self._worker_rng_states(),
            training_timesteps=int(self.model.num_timesteps),
        )

    def _sampling_targets(self):
        manager_config = getattr(self.manager, "config", {})
        if hasattr(self.manager, "training_reset_pools"):
            pools = self.manager.training_reset_pools()
        else:
            stored = getattr(self.manager, "pools", {})
            pools = {
                "frontier": stored.get("frontier", []),
                "historical": stored.get("mastered", []),
                "mastered_boundary": [],
                "too_hard_near": [],
            }
        return configured_start_sampling_probabilities(
            frontier_pool_size=len(pools.get("frontier", [])),
            historical_pool_size=len(pools.get("historical", [])),
            mastered_boundary_pool_size=len(
                pools.get("mastered_boundary", [])
            ),
            too_hard_near_pool_size=len(pools.get("too_hard_near", [])),
            curriculum_probability=float(
                manager_config.get("curriculum_reset_probability", 0.8)
            ),
            config=manager_config.get("start_sampling", {}),
        )

    def _transition_balance_config(self) -> tuple[str, float, float, int]:
        sampling = getattr(self.manager, "config", {}).get(
            "start_sampling", {}
        )
        settings = sampling.get("transition_balance", {})
        return (
            str(sampling.get("balance_unit", "episodes")),
            float(settings.get("ema_alpha", 0.25)),
            float(settings.get("min_episode_length", 1.0)),
            int(settings.get("min_completed_episodes", 1)),
        )

    def _effective_reset_probabilities(self, targets):
        balance_unit, _, minimum, _ = self._transition_balance_config()
        if balance_unit == "episodes":
            return targets
        return reset_probabilities_for_transition_targets(
            targets, self.manager.sampling_episode_length_ema,
            min_episode_length=minimum,
        )

    def _update_episode_length_ema(self) -> None:
        _, alpha, minimum, minimum_count = self._transition_balance_config()
        completed = {
            self.SOURCE_LABELS[source]: self.source_episode_lengths[source]
            for source in self.RESET_SOURCES
        }
        self.manager.sampling_episode_length_ema = (
            update_sampling_episode_length_ema(
                self.manager.sampling_episode_length_ema, completed,
                ema_alpha=alpha, min_episode_length=minimum,
                min_completed_episodes=minimum_count,
            )
        )

    def _write_curriculum_diagnostics(
        self, targets_used=None, targets_next=None,
        effective_resets_used=None, episode_length_ema_used=None,
    ) -> None:
        targets_used = (
            self.sampling_targets_used if targets_used is None else targets_used
        )
        targets_next = (
            self._sampling_targets() if targets_next is None else targets_next
        )
        effective_resets_used = (
            self.sampling_effective_resets_used
            if effective_resets_used is None else effective_resets_used
        )
        episode_length_ema_used = (
            self.sampling_episode_length_ema_used
            if episode_length_ema_used is None else episode_length_ema_used
        )
        total = sum(self.source_episode_counts.values())
        observed = {
            self.SOURCE_LABELS[source]: (
                self.source_episode_counts[source] / total if total else 0.0
            )
            for source in self.RESET_SOURCES
        }
        transition_total = sum(self.source_transition_counts.values())
        transition_observed = {
            self.SOURCE_LABELS[source]: (
                self.source_transition_counts[source] / transition_total
                if transition_total else 0.0
            )
            for source in self.RESET_SOURCES
        }
        success_rates = {
            self.SOURCE_LABELS[source]: (
                self.source_success_counts[source]
                / self.source_episode_counts[source]
                if self.source_episode_counts[source] else 0.0
            )
            for source in self.RESET_SOURCES
        }
        start_distances = {
            self.SOURCE_LABELS[source]: list(
                self.used_start_distances.get(source, [])
            )
            for source in self.RESET_SOURCES
        }
        timesteps = int(self.model.num_timesteps)
        diagnostics = ExpansionDiagnostics.build(
            self.manager, timesteps, targets_used, observed, targets_next,
            self.frontier_reset_counts,
            sampling_transition_observed=transition_observed,
            sampling_success_rates=success_rates,
            used_start_distances=start_distances,
            sampling_effective_reset=effective_resets_used,
            sampling_episode_length_ema=episode_length_ema_used,
        )
        write_curriculum_diagnostics(
            self.output, diagnostics,
            curriculum_state_rows(self.manager, timesteps),
        )

    def _save_checkpoint(self) -> None:
        steps = int(self.model.num_timesteps)
        checkpoint_dir = self.output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.algorithm}_{steps}_steps"
        self.model.save(checkpoint_dir / stem)
        self.model.save_replay_buffer(
            checkpoint_dir / f"{stem}_replay_buffer.pkl"
        )
        self._save_curriculum(
            checkpoint_dir / f"curriculum_{steps}_steps.pkl"
        )
        self._save_curriculum(self.output / "curriculum_state.pkl")

    def _process_due_work(self) -> None:
        while self.model.num_timesteps >= self.manager.next_update_timesteps:
            targets_used = self.sampling_targets_used
            effective_resets_used = self.sampling_effective_resets_used
            episode_length_ema_used = self.sampling_episode_length_ema_used
            self._update_episode_length_ema()
            self.manager.update(self.model)
            targets_next = self._sampling_targets()
            effective_resets_next = self._effective_reset_probabilities(
                targets_next
            )
            self._broadcast_pool(effective_resets_next)
            self._record_metrics(
                include_pool_metrics=True, include_update_metrics=True,
                sampling_targets_used=targets_used,
                sampling_targets_next=targets_next,
                sampling_effective_resets_used=effective_resets_used,
                sampling_episode_length_ema_used=episode_length_ema_used,
            )
            self._write_curriculum_diagnostics(
                targets_used, targets_next, effective_resets_used,
                episode_length_ema_used,
            )
            self.sampling_targets_used = targets_next
            self.sampling_effective_resets_used = effective_resets_next
            self.sampling_episode_length_ema_used = dict(
                self.manager.sampling_episode_length_ema
            )
            self._reset_cycle_metrics()
        if self.next_checkpoint_timesteps is None:
            self.next_checkpoint_timesteps = (
                (int(self.model.num_timesteps) // self.checkpoint_interval + 1)
                * self.checkpoint_interval
            )
        if self.model.num_timesteps >= self.next_checkpoint_timesteps:
            self._save_checkpoint()
            while self.model.num_timesteps >= self.next_checkpoint_timesteps:
                self.next_checkpoint_timesteps += self.checkpoint_interval

    def _on_training_start(self) -> None:
        self._broadcast_pool()
        self._restore_worker_rngs()
        self.next_checkpoint_timesteps = (
            (int(self.model.num_timesteps) // self.checkpoint_interval + 1)
            * self.checkpoint_interval
        )
        self._record_metrics(include_pool_metrics=True)

    def _on_rollout_start(self) -> None:
        # Ici, la transition et le gradient du rollout précédent sont terminés.
        self._process_due_work()

    def _on_step(self) -> bool:
        for done, info in zip(
            self.locals.get("dones", []), self.locals.get("infos", [])
        ):
            source = info.get("reset_source", "true_start")
            if source not in self.source_episode_counts:
                continue
            self.source_transition_counts[source] += 1
            if not done:
                continue
            self.source_episode_counts[source] += 1
            self.source_success_counts[source] += int(bool(info.get("safe_success")))
            if source == "curriculum_frontier":
                state_id = info.get("curriculum_start_state_id")
                if isinstance(state_id, (int, np.integer)) and int(state_id) >= 0:
                    key = int(state_id)
                    self.frontier_reset_counts[key] = (
                        self.frontier_reset_counts.get(key, 0) + 1
                    )
            if source == "true_start":
                for name, values in self.true_start_diagnostics.items():
                    value = info.get(name)
                    if (isinstance(value, (bool, int, float, np.number))
                            and np.isfinite(float(value))):
                        values.append(float(value))
            episode = info.get("episode")
            episode_length = (
                episode.get("l") if isinstance(episode, dict) else None
            )
            if (isinstance(episode_length, (int, float, np.number))
                    and np.isfinite(float(episode_length))):
                self.source_episode_lengths[source].append(
                    float(episode_length)
                )
            if source in self.used_start_distances:
                distance = info.get("curriculum_start_pose_distance")
                if (isinstance(distance, (int, float, np.number))
                        and np.isfinite(float(distance))):
                    self.used_start_distances[source].append(float(distance))
        return True

    def _on_training_end(self) -> None:
        self._process_due_work()
        self._record_metrics(include_pool_metrics=True)
        # OffPolicyAlgorithm retourne après on_training_end sans dump final :
        # flusher explicitement évite de perdre la dernière fenêtre RCG.
        self.logger.dump(step=int(self.model.num_timesteps))
        self._save_curriculum(self.output / "curriculum_state.pkl")


def make_env(
    config_path: Path, rank: int, base_seed: int, *,
    allow_curriculum_resets: bool = False,
) -> Callable[[], TenonMortaiseEnv]:
    """Retourne une factory picklable créant une simulation MuJoCo indépendante."""
    env_seed = base_seed + rank

    def initialize() -> TenonMortaiseEnv:
        env = TenonMortaiseEnv(
            config_path, allow_curriculum_resets=allow_curriculum_resets,
        )
        # Attributs de diagnostic utiles aux tests, sans effet sur step().
        env.worker_rank = rank
        env.worker_seed = env_seed
        env.worker_pid = os.getpid()
        env.action_space.seed(env_seed)
        return env

    return initialize


def build_vec_env(
    config_path: Path,
    n_envs: int,
    base_seed: int,
    monitor_path: Path,
    *,
    monitor_fields: tuple[str, ...] = MONITOR_FIELDS,
    allow_curriculum_resets: bool = False,
) -> VecMonitor:
    """Construit les workers puis un unique writer VecMonitor dans le parent."""
    if n_envs <= 0:
        raise ValueError("n_envs doit être strictement positif")
    config_path = config_path.resolve()
    factories = [
        make_env(
            config_path, rank, base_seed,
            allow_curriculum_resets=allow_curriculum_resets,
        )
        for rank in range(n_envs)
    ]
    if n_envs == 1:
        vector_env: VecEnv = DummyVecEnv(factories)
    else:
        vector_env = SubprocVecEnv(factories, start_method="spawn")
    vector_env.seed(base_seed)
    return VecMonitor(
        vector_env,
        filename=str(monitor_path),
        info_keywords=monitor_fields,
    )


def scaled_callback_freq(transition_freq: int, n_envs: int) -> int:
    """Convertit une fréquence en transitions vers les appels vectorisés SB3."""
    # Premier appel vectoriel atteignant (jamais précédant) le seuil demandé.
    return max((transition_freq + n_envs - 1) // n_envs, 1)


def resolve_total_timesteps(training: dict, cli_timesteps: int | None) -> int:
    """Resolve CLI override over YAML while retaining the historical default."""
    total_timesteps = (
        cli_timesteps
        if cli_timesteps is not None
        else training.get("total_timesteps", 500_000)
    )
    if (isinstance(total_timesteps, bool)
            or not isinstance(total_timesteps, int)
            or total_timesteps <= 0):
        raise ValueError("timesteps doit être un entier strictement positif")
    return total_timesteps


def learn_model(
    model: BaseAlgorithm, total_timesteps: int, callbacks: CallbackList, *,
    reset_num_timesteps: bool = True,
) -> None:
    """Start SB3 with the already resolved transition budget."""
    kwargs = dict(
        total_timesteps=total_timesteps, callback=callbacks, progress_bar=True,
    )
    if reset_num_timesteps:
        model.learn(**kwargs)
    else:
        model.learn(**kwargs, reset_num_timesteps=False)


def create_sac_model(
    env: VecEnv,
    training: dict,
    *,
    base_seed: int,
    tensorboard_log: Path,
    device: str,
) -> SAC:
    """Construit SAC avec les hyperparamètres effectifs du YAML résolu."""
    return SAC(
        "MlpPolicy", env, seed=base_seed, verbose=1,
        tensorboard_log=str(tensorboard_log), device=device,
        learning_starts=5_000,
        buffer_size=training.get("buffer_size", 50_000),
        learning_rate=training.get("learning_rate", 3e-4),
        batch_size=256,
        train_freq=(1, "step"), gradient_steps=-1,
        gamma=float(training.get("gamma", 0.99)),
        ent_coef=training.get("ent_coef", "auto"),
        target_entropy=training.get("target_entropy", "auto"),
        policy_kwargs={"net_arch": [256, 256]},
    )


def create_td3_model(
    env: VecEnv,
    training: dict,
    *,
    base_seed: int,
    tensorboard_log: Path,
    device: str,
) -> TD3:
    """Construit TD3 avec les paramètres communs et son bruit d'exploration."""
    td3 = training.get("td3", {})
    action_dim = int(env.action_space.shape[-1])
    noise_std = float(td3.get("action_noise_std", 0.1))
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim), sigma=noise_std * np.ones(action_dim),
    )
    return TD3(
        "MlpPolicy", env, seed=base_seed, verbose=1,
        tensorboard_log=str(tensorboard_log), device=device,
        learning_starts=5_000,
        buffer_size=training.get("buffer_size", 50_000),
        learning_rate=training.get("learning_rate", 3e-4),
        batch_size=256,
        train_freq=(1, "step"), gradient_steps=-1,
        gamma=float(training.get("gamma", 0.99)),
        action_noise=action_noise,
        policy_delay=int(td3.get("policy_delay", 2)),
        target_policy_noise=float(td3.get("target_policy_noise", 0.2)),
        target_noise_clip=float(td3.get("target_noise_clip", 0.5)),
        policy_kwargs={"net_arch": [256, 256]},
    )


def create_model(
    env: VecEnv, training: dict, *, base_seed: int,
    tensorboard_log: Path, device: str,
) -> BaseAlgorithm:
    """Sélectionne l'un des deux constructeurs explicites depuis le YAML résolu."""
    common = dict(
        env=env, training=training, base_seed=base_seed,
        tensorboard_log=tensorboard_log, device=device,
    )
    if training["algorithm"] == "sac":
        return create_sac_model(**common)
    if training["algorithm"] == "td3":
        return create_td3_model(**common)
    raise ValueError(
        f"Unsupported RL algorithm: {training['algorithm']}. "
        "Supported algorithms: sac, td3"
    )


def load_training_model(
    path: Path, env: VecEnv, training: dict, *, device: str,
) -> BaseAlgorithm:
    algorithm = training["algorithm"]
    model_class = SAC if algorithm == "sac" else TD3
    return model_class.load(path, env=env, device=device)


def derived_resume_paths(model_path: Path) -> tuple[Path, Path]:
    """Retourne replay et curriculum coordonnés à un checkpoint nommé par step."""
    match = re.search(r"_(\d+)_steps$", model_path.stem)
    if match:
        replay = model_path.with_name(f"{model_path.stem}_replay_buffer.pkl")
        curriculum = model_path.with_name(
            f"curriculum_{match.group(1)}_steps.pkl"
        )
    else:
        replay = model_path.with_name(
            "replay_buffer_interrupted.pkl"
            if model_path.stem == "model_interrupted"
            else "replay_buffer.pkl"
        )
        curriculum = model_path.with_name("curriculum_state.pkl")
    return replay, curriculum


def archive_run_context(
    output: Path, args: argparse.Namespace, total_timesteps: int, algorithm: str,
) -> None:
    """Archive versions, inputs and a source snapshot without invoking Git."""
    def source_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(member.name).parts
        return None if "__pycache__" in parts or member.name.endswith((".pyc", ".pyo")) else member

    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for source in (
            "src", "tests", "configs", "requirements.txt", "Dockerfile",
            "docker-compose.yml", "Makefile", "README.md",
        ):
            path = Path(source)
            if path.exists():
                archive.add(path, arcname=path.as_posix(), filter=source_filter)
    input_hashes = {}
    for root in (
        Path("data/input/cad/tenon-mortaise"),
        Path("data/input/grasp_poses/tenon"),
    ):
        for path in sorted(root.glob("*")):
            if path.is_file():
                input_hashes[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "total_timesteps": total_timesteps,
        "algorithm": algorithm,
        "command": vars(args),
        "input_sha256": input_hashes,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "mujoco": mujoco.__version__, "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test1.yaml")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Remplace training.total_timesteps pour ce run")
    parser.add_argument("--seed", type=int, default=None,
                        help="Remplace training.base_seed pour ce run")
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None,
                        help="Remplace training.checkpoint_freq pour ce run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-model", type=Path, default=None,
                        help="Checkpoint SAC/TD3 à reprendre dans un nouveau run")
    parser.add_argument("--resume-replay-buffer", type=Path, default=None,
                        help="Replay coordonné; dérivé du nom du checkpoint par défaut")
    parser.add_argument("--resume-curriculum", type=Path, default=None,
                        help="État RCG coordonné; bootstrap avec warning s'il manque")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_config = load_config(args.config)
    training = resolved_config["training"]
    algorithm = training["algorithm"]
    total_timesteps = resolve_total_timesteps(training, args.timesteps)
    training["total_timesteps"] = total_timesteps
    if args.seed is not None:
        training["base_seed"] = args.seed
    if args.checkpoint_freq is not None:
        training["checkpoint_freq"] = args.checkpoint_freq
    n_envs = int(training["n_envs"])
    base_seed = int(training["base_seed"])
    checkpoint_freq = int(training["checkpoint_freq"])
    if base_seed < 0 or checkpoint_freq <= 0:
        raise ValueError("base_seed doit être >= 0 et checkpoint_freq > 0")

    # Les valeurs effectives sont archivées à la fois dans le YAML et le manifeste.
    args.n_envs = n_envs
    args.base_seed = base_seed
    args.effective_checkpoint_freq = scaled_callback_freq(checkpoint_freq, n_envs)
    name = args.run or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("data/output") / name
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(resolved_config, output / "config.yaml")
    archive_run_context(output, args, total_timesteps, algorithm)

    curriculum_enabled = bool(resolved_config["curriculum"]["enabled"])
    env = build_vec_env(
        output / "config.yaml", n_envs, base_seed, output / "monitor.csv",
        allow_curriculum_resets=curriculum_enabled,
    )
    evaluation = resolved_config["evaluation"]
    eval_env = None
    eval_callback = None
    if evaluation["enabled"]:
        eval_dir = output / "eval"
        eval_dir.mkdir()
        eval_env = build_vec_env(
            output / "config.yaml", 1, int(evaluation["seed"]),
            eval_dir / "monitor.csv", monitor_fields=EVAL_MONITOR_FIELDS,
            allow_curriculum_resets=False,
        )
        eval_callback = TrainingTimestepEvalCallback(
            eval_env,
            eval_freq=scaled_callback_freq(int(evaluation["eval_freq"]), n_envs),
            n_eval_episodes=int(evaluation["n_eval_episodes"]),
            deterministic=bool(evaluation["deterministic"]),
            best_model_save_path=str(eval_dir),
            log_path=str(eval_dir),
        )
    resume_model = args.resume_model.resolve() if args.resume_model else None
    if resume_model is not None:
        if not resume_model.is_file():
            raise FileNotFoundError(f"Checkpoint de reprise introuvable: {resume_model}")
        model = load_training_model(
            resume_model, env, training, device=args.device,
        )
        model.tensorboard_log = str(output / "tensorboard")
        derived_replay, derived_curriculum = derived_resume_paths(resume_model)
        replay_path = (
            args.resume_replay_buffer.resolve()
            if args.resume_replay_buffer else derived_replay
        )
        if not replay_path.is_file():
            raise FileNotFoundError(
                "Reprise fidèle impossible: replay buffer coordonné introuvable: "
                f"{replay_path}"
            )
        model.load_replay_buffer(replay_path)
        curriculum_resume_path = (
            args.resume_curriculum.resolve()
            if args.resume_curriculum else derived_curriculum
        )
    else:
        model = create_model(
            env, training, base_seed=base_seed,
            tensorboard_log=output / "tensorboard", device=args.device,
        )
        curriculum_resume_path = None

    curriculum_manager = None
    curriculum_env = None
    if curriculum_enabled:
        curriculum_env = TenonMortaiseEnv(
            output / "config.yaml", allow_curriculum_resets=False,
        )
        curriculum_manager = ReverseCurriculumManager(
            curriculum_env, resolved_config["curriculum"], seed=base_seed,
        )
        if curriculum_resume_path is not None and curriculum_resume_path.is_file():
            curriculum_manager.load(curriculum_resume_path)
            print(f"Curriculum restauré: {curriculum_resume_path}")
            saved_curriculum_step = curriculum_manager.loaded_training_timesteps
            if saved_curriculum_step is None:
                warnings.warn(
                    "Cet ancien curriculum_state.pkl ne contient pas le timestep "
                    "SAC coordonné; sa synchronisation avec le modèle repris ne "
                    "peut pas être vérifiée.", RuntimeWarning,
                )
            elif int(saved_curriculum_step) != int(model.num_timesteps):
                warnings.warn(
                    "Curriculum potentiellement stale: sauvegardé à "
                    f"{saved_curriculum_step} transitions, mais le modèle repris "
                    f"est à {model.num_timesteps}.", RuntimeWarning,
                )
        else:
            if curriculum_resume_path is not None:
                warnings.warn(
                    "curriculum_state.pkl absent pour la reprise; bootstrap depuis "
                    "le goal seed.", RuntimeWarning,
                )
            print(
                "Bootstrap RCG: génération physique et qualification "
                "stochastique hors replay...",
                flush=True,
            )
            report = curriculum_manager.bootstrap(model)
            if resume_model is not None:
                curriculum_manager.next_update_timesteps = (
                    int(model.num_timesteps)
                    + int(resolved_config["curriculum"][
                        "update_interval_timesteps"
                    ])
                )
            print(
                "Bootstrap curriculum: "
                f"generated={report.generated}, valid={report.valid}, "
                f"unsafe_rejected={report.unsafe_rejected}, "
                f"successful_excluded={report.successful_excluded}"
            )
    if (resume_model is not None and curriculum_manager is not None
            and curriculum_manager.worker_rng_states is not None):
        saved_worker_rngs = curriculum_manager.worker_rng_states
        if len(saved_worker_rngs) == n_envs:
            # Consommer les seeds VecEnv en attente, puis installer l'état du
            # checkpoint avant le reset effectué par _setup_learn. Le premier
            # épisode repris suit ainsi lui aussi le flux RNG sauvegardé.
            env.reset()
            for index, state in enumerate(saved_worker_rngs):
                env.env_method("set_worker_rng_state", state, indices=index)
            curriculum_manager.worker_rng_states = None
            model._last_obs = None
        else:
            warnings.warn(
                "Le nombre d'états RNG curriculum ne correspond pas à n_envs; "
                "la reprise utilisera les seeds de workers.", RuntimeWarning,
            )
    print(f"RL algorithm: {algorithm.upper()}")
    print(f"total_timesteps: {total_timesteps}")
    print(f"buffer_size: {training['buffer_size']}")
    print(f"learning_rate: {training['learning_rate']}")
    print("network: [256, 256]")
    if algorithm == "sac":
        print(f"ent_coef: {training['ent_coef']}")
        print(f"target_entropy: {training['target_entropy']}")
    else:
        print(f"action_noise_std: {training['td3']['action_noise_std']}")
    callback_items: list[BaseCallback] = []
    curriculum_callback: ReverseCurriculumCallback | None = None
    if curriculum_manager is not None:
        curriculum_callback = ReverseCurriculumCallback(
            curriculum_manager, env, output, algorithm, checkpoint_freq,
        )
        callback_items.append(curriculum_callback)
    else:
        callback_items.append(CheckpointCallback(
            scaled_callback_freq(checkpoint_freq, n_envs),
            str(output / "checkpoints"), name_prefix=algorithm,
        ))
    callback_items.append(EpisodeMetricsCallback())
    if eval_callback is not None:
        callback_items.append(eval_callback)
    callbacks = CallbackList(callback_items)
    if curriculum_callback is not None:
        # SB3 appelle env.reset() avant on_training_start : diffuser maintenant
        # évite de forcer les 16 premiers épisodes au vrai départ.
        curriculum_callback._broadcast_pool()
    learning_timesteps = total_timesteps
    reset_num_timesteps = True
    if resume_model is not None:
        learning_timesteps = total_timesteps - int(model.num_timesteps)
        if learning_timesteps <= 0:
            raise ValueError(
                "Le checkpoint a déjà atteint le budget total demandé: "
                f"{model.num_timesteps} >= {total_timesteps}"
            )
        reset_num_timesteps = False
    try:
        learn_model(
            model, learning_timesteps, callbacks,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(output / "model")
        if curriculum_manager is not None:
            model.save_replay_buffer(output / "replay_buffer.pkl")
        print(f"Essai sauvegardé: {output}")
    except KeyboardInterrupt:
        model.save(output / "model_interrupted")
        if curriculum_manager is not None:
            model.save_replay_buffer(output / "replay_buffer_interrupted.pkl")
            curriculum_manager.save(
                output / "curriculum_state.pkl",
                env.env_method("get_worker_rng_state"),
                training_timesteps=int(model.num_timesteps),
            )
        print(f"Entraînement interrompu; modèle partiel sauvegardé: {output / 'model_interrupted.zip'}")
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()
        if curriculum_env is not None:
            curriculum_env.close()


if __name__ == "__main__":
    main()
