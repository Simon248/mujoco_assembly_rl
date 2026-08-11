"""Reverse Curriculum Generation, indépendant de la boucle d'apprentissage SB3.

Le gestionnaire possède une simulation MuJoCo dédiée. Les marches inverses et
les rollouts de qualification ne passent donc jamais par ``collect_rollouts``
et ne peuvent pas alimenter le replay buffer.

Le curriculum progresse selon le lineage des états physiquement générés par les
marches inverses. La distance géométrique au goal reste un diagnostic : elle
n'est supposée ni monotone le long d'une trajectoire d'assemblage, ni corrélée à
sa difficulté. Cette dernière dépend uniquement du taux de succès de la policy.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import time
import warnings
from typing import Any, Iterator, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch

from src.transforms import quat_to_rotvec, relative

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from src.assembly_env import TenonMortaiseEnv


POOL_NAMES = ("too_hard", "frontier", "mastered")
RESET_SOURCES = (
    "true_start", "curriculum_frontier", "curriculum_historical",
    "curriculum_mastered_boundary", "curriculum_too_hard_near",
)
SAMPLING_SOURCE_NAMES = (
    "true_start", "frontier", "historical", "mastered_boundary",
    "too_hard_near",
)
EXPANSION_DEFAULTS: dict[str, int | float] = {
    "max_hops_per_seed": 4,
    "max_attempts_per_hop": 8,
    "max_candidates_per_update": 24,
    "initial_scale": 1.0,
    "scale_up_factor": 1.25,
    "scale_down_factor": 0.7,
    "min_scale": 0.5,
    "max_scale": 3.0,
}
EXPANSION_STRATEGY_KEYS = frozenset(EXPANSION_DEFAULTS)


class ExpansionStopReason(str, Enum):
    """Exclusive final outcome for one expansion branch in one update."""

    DUPLICATE = "duplicate"
    FORCE = "force"
    TORQUE = "torque"
    FORCE_AND_TORQUE = "force_and_torque"
    SNAPSHOT_INVALID = "snapshot_invalid"
    FORBIDDEN_CONTACT = "forbidden_contact"
    OTHER_INVALID = "other_invalid"
    WORKSPACE = "workspace"
    GENERATION_FAILED = "generation_failed"
    FRONTIER = "frontier"
    TOO_HARD = "too_hard"
    MAX_HOPS = "max_hops"
    CANDIDATE_BUDGET = "candidate_budget"
    ATTEMPT_BUDGET = "attempt_budget"


@dataclass
class CurriculumState:
    """Snapshot physique complet, sans compteur ni statistique d'épisode.

    ``mj_state`` utilise ``mjSTATE_INTEGRATION`` : temps, qpos/qvel, act,
    warm-start, ctrl, forces appliquées, égalités actives, mocap, userdata et
    état des plugins. Les autres champs couvrent les états mutables placés
    hors de ``MjData`` et le contrôleur d'admittance.
    """

    mj_state: np.ndarray
    fixed_body_position: np.ndarray
    fixed_body_quaternion: np.ndarray
    contact_friction: np.ndarray
    friction_scale: float
    admittance_offset: np.ndarray
    admittance_velocity: np.ndarray
    reference_position: np.ndarray | None
    reference_quaternion: np.ndarray | None
    perception_bias_position: np.ndarray
    perception_bias_quaternion: np.ndarray
    environment_rng_state: dict[str, Any] | None
    task_position: np.ndarray
    task_quaternion: np.ndarray
    position_error: float
    rotation_error: float
    pose_distance: float
    success_rate: float = math.nan
    state_id: int = -1
    # Le lineage décrit la topologie d'exploration, jamais la difficulté. Le
    # goal a depth=0/id=-1; ses enfants directs ont parent_id=None/depth=1.
    parent_id: int | None = None
    generation_depth: int = 0


@dataclass(frozen=True)
class CurriculumResetSelection:
    """Résultat d'un tirage de start, indépendant de MuJoCo."""

    source: str
    state: CurriculumState | None
    historical_bin: int | None = None


@dataclass(frozen=True)
class StartSamplingProbabilities:
    true_start: float
    frontier: float
    historical: float
    historical_fraction_effective: float
    mastered_boundary: float = 0.0
    too_hard_near: float = 0.0
    missing_frontier_budget: float = 0.0
    fallback_budget_used: float = 0.0

    @property
    def frontier_fraction_effective(self) -> float:
        return self.frontier

    @property
    def true_start_fraction_effective(self) -> float:
        return self.true_start


def reset_probabilities_for_transition_targets(
    transition_targets: StartSamplingProbabilities,
    episode_lengths: Mapping[str, float], *,
    min_episode_length: float = 1.0,
) -> StartSamplingProbabilities:
    """Convert transition shares into reset probabilities.

    The expected transition mass of a source is proportional to ``p_i * L_i``.
    Therefore ``p_i`` is obtained by normalizing ``q_i / L_i``. Missing,
    non-finite, or non-positive length estimates are rejected explicitly;
    finite positive estimates below ``min_episode_length`` are clamped to it.
    """
    if (not np.isfinite(min_episode_length) or min_episode_length <= 0.0):
        raise ValueError("min_episode_length doit être fini et strictement positif")
    targets: dict[str, float] = {}
    lengths: dict[str, float] = {}
    for name in SAMPLING_SOURCE_NAMES:
        target = float(getattr(transition_targets, name))
        if not np.isfinite(target) or target < 0.0:
            raise ValueError(
                f"La target de transitions {name!r} doit être finie et positive "
                "ou nulle"
            )
        if name not in episode_lengths:
            raise ValueError(f"Longueur d'épisode manquante pour {name!r}")
        length = float(episode_lengths[name])
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(
                f"La longueur d'épisode {name!r} doit être finie et positive"
            )
        targets[name] = target
        lengths[name] = max(float(min_episode_length), length)
    target_total = sum(targets.values())
    if not np.isclose(target_total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Les targets de transitions doivent sommer à 1")
    if len(set(lengths.values())) == 1:
        # Preserve the historical values bit-for-bit during the bootstrap.
        return replace(transition_targets)
    raw = {name: targets[name] / lengths[name] for name in SAMPLING_SOURCE_NAMES}
    raw_total = sum(raw.values())
    if not np.isfinite(raw_total) or raw_total <= 0.0:
        raise ValueError("Impossible de normaliser les probabilités de reset")
    probabilities = {name: raw[name] / raw_total for name in SAMPLING_SOURCE_NAMES}
    result = StartSamplingProbabilities(
        true_start=float(probabilities["true_start"]),
        frontier=float(probabilities["frontier"]),
        historical=float(probabilities["historical"]),
        historical_fraction_effective=float(probabilities["historical"]),
        mastered_boundary=float(probabilities["mastered_boundary"]),
        too_hard_near=float(probabilities["too_hard_near"]),
        missing_frontier_budget=float(
            transition_targets.missing_frontier_budget
        ),
        fallback_budget_used=float(transition_targets.fallback_budget_used),
    )
    values = [getattr(result, name) for name in SAMPLING_SOURCE_NAMES]
    if (any(not np.isfinite(value) or value < 0.0 for value in values)
            or not np.isclose(sum(values), 1.0, rtol=0.0, atol=1e-12)):
        raise AssertionError("Probabilités de reset invalides après conversion")
    return result


def update_sampling_episode_length_ema(
    previous: Mapping[str, float],
    completed_episode_lengths: Mapping[str, Sequence[float]], *,
    ema_alpha: float, min_episode_length: float = 1.0,
    min_completed_episodes: int = 1,
) -> dict[str, float]:
    """Update one episode-length EMA per source from a completed window."""
    if not np.isfinite(ema_alpha) or not 0.0 < ema_alpha <= 1.0:
        raise ValueError("ema_alpha doit être dans ]0, 1]")
    if not np.isfinite(min_episode_length) or min_episode_length <= 0.0:
        raise ValueError("min_episode_length doit être strictement positif")
    if (isinstance(min_completed_episodes, bool)
            or not isinstance(min_completed_episodes, int)
            or min_completed_episodes < 1):
        raise ValueError("min_completed_episodes doit être un entier >= 1")
    updated: dict[str, float] = {}
    for name in SAMPLING_SOURCE_NAMES:
        old = float(previous.get(name, 1.0))
        if not np.isfinite(old) or old <= 0.0:
            old = 1.0
        samples = np.asarray(
            completed_episode_lengths.get(name, ()), dtype=float,
        )
        samples = samples[np.isfinite(samples) & (samples > 0.0)]
        if samples.size < min_completed_episodes:
            updated[name] = old
            continue
        window_mean = max(float(min_episode_length), float(np.mean(samples)))
        updated[name] = float(ema_alpha * window_mean + (1.0 - ema_alpha) * old)
    return updated


def compute_adaptive_three_way_probabilities(
    frontier_pool_size: int, historical_pool_size: int, *,
    frontier_fraction_per_state: float, frontier_fraction_max: float,
    historical_fraction_per_state: float, historical_fraction_max: float,
) -> StartSamplingProbabilities:
    """Direct reset/episode probabilities from available pool diversity."""
    frontier = min(
        frontier_fraction_max,
        frontier_fraction_per_state * max(0, int(frontier_pool_size)),
    )
    historical = min(
        historical_fraction_max,
        historical_fraction_per_state * max(0, int(historical_pool_size)),
    )
    true_start = 1.0 - frontier - historical
    return StartSamplingProbabilities(
        float(true_start), float(frontier), float(historical), float(historical),
    )


def compute_adaptive_diverse_fallback_probabilities(
    frontier_pool_size: int, historical_pool_size: int,
    mastered_boundary_pool_size: int, too_hard_near_pool_size: int, *,
    frontier_fraction_per_state: float, frontier_fraction_max: float,
    historical_fraction_per_state: float, historical_fraction_max: float,
    mastered_boundary_fraction_per_state: float,
    mastered_boundary_fraction_max: float,
    too_hard_near_fraction_per_state: float,
    too_hard_near_fraction_max: float,
    historical_boost_fraction_per_state: float,
    historical_boost_fraction_max: float,
    true_start_fraction_min: float,
) -> StartSamplingProbabilities:
    """Fill only missing frontier capacity with diversity-capped reset pools."""
    frontier = min(
        frontier_fraction_max,
        frontier_fraction_per_state * max(0, int(frontier_pool_size)),
    )
    historical_base = min(
        historical_fraction_max,
        historical_fraction_per_state * max(0, int(historical_pool_size)),
    )
    missing_frontier_budget = max(0.0, frontier_fraction_max - frontier)
    fallback_budget = min(
        missing_frontier_budget,
        max(
            0.0,
            1.0 - true_start_fraction_min - frontier - historical_base,
        ),
    )
    boundary_raw = min(
        mastered_boundary_fraction_max,
        mastered_boundary_fraction_per_state
        * max(0, int(mastered_boundary_pool_size)),
    )
    too_hard_raw = min(
        too_hard_near_fraction_max,
        too_hard_near_fraction_per_state
        * max(0, int(too_hard_near_pool_size)),
    )
    historical_boost_raw = min(
        historical_boost_fraction_max,
        historical_boost_fraction_per_state
        * max(0, int(historical_pool_size)),
    )
    raw_fallback_total = boundary_raw + too_hard_raw + historical_boost_raw
    scale = (
        min(1.0, fallback_budget / raw_fallback_total)
        if raw_fallback_total > 0.0 else 0.0
    )
    mastered_boundary = boundary_raw * scale
    too_hard_near = too_hard_raw * scale
    historical_boost = historical_boost_raw * scale
    historical = historical_base + historical_boost
    fallback_budget_used = mastered_boundary + too_hard_near + historical_boost
    true_start = 1.0 - frontier - historical - mastered_boundary - too_hard_near
    probabilities = StartSamplingProbabilities(
        true_start=float(true_start),
        frontier=float(frontier),
        historical=float(historical),
        historical_fraction_effective=float(historical),
        mastered_boundary=float(mastered_boundary),
        too_hard_near=float(too_hard_near),
        missing_frontier_budget=float(missing_frontier_budget),
        fallback_budget_used=float(fallback_budget_used),
    )
    if probabilities.true_start < true_start_fraction_min - 1e-12:
        raise AssertionError("Le fallback viole le minimum true_start")
    if not np.isclose(
        sum(getattr(probabilities, name) for name in SAMPLING_SOURCE_NAMES),
        1.0, rtol=0.0, atol=1e-12,
    ):
        raise AssertionError("Les probabilités de reset ne somment pas à 1")
    return probabilities


def configured_start_sampling_probabilities(
    *, frontier_pool_size: int, historical_pool_size: int,
    curriculum_probability: float, config: dict[str, Any],
    mastered_boundary_pool_size: int = 0,
    too_hard_near_pool_size: int = 0,
) -> StartSamplingProbabilities:
    """Dispatch one explicit strategy; legacy remains available for A/B runs."""
    strategy = config.get("strategy", "legacy")
    if strategy in {"adaptive_three_way", "adaptive_diverse_fallback"}:
        frontier = config["frontier"]
        historical = config["historical"]
        if strategy == "adaptive_diverse_fallback":
            fallback = config["fallback"]
            boundary = fallback["mastered_boundary"]
            too_hard = fallback["too_hard_near"]
            historical_boost = fallback["historical_boost"]
            return compute_adaptive_diverse_fallback_probabilities(
                frontier_pool_size, historical_pool_size,
                mastered_boundary_pool_size, too_hard_near_pool_size,
                frontier_fraction_per_state=float(frontier["fraction_per_state"]),
                frontier_fraction_max=float(frontier["fraction_max"]),
                historical_fraction_per_state=float(
                    historical["fraction_per_state"]
                ),
                historical_fraction_max=float(historical["fraction_max"]),
                mastered_boundary_fraction_per_state=float(
                    boundary["fraction_per_state"]
                ),
                mastered_boundary_fraction_max=float(boundary["fraction_max"]),
                too_hard_near_fraction_per_state=float(
                    too_hard["fraction_per_state"]
                ),
                too_hard_near_fraction_max=float(too_hard["fraction_max"]),
                historical_boost_fraction_per_state=float(
                    historical_boost["fraction_per_state"]
                ),
                historical_boost_fraction_max=float(
                    historical_boost["fraction_max"]
                ),
                true_start_fraction_min=float(
                    config["true_start"]["fraction_min"]
                ),
            )
        return compute_adaptive_three_way_probabilities(
            frontier_pool_size, historical_pool_size,
            frontier_fraction_per_state=float(frontier["fraction_per_state"]),
            frontier_fraction_max=float(frontier["fraction_max"]),
            historical_fraction_per_state=float(historical["fraction_per_state"]),
            historical_fraction_max=float(historical["fraction_max"]),
        )
    historical_fraction = effective_historical_fraction(
        historical_pool_size,
        adaptive=bool(config.get("adaptive_historical", False)),
        fixed_fraction=float(config.get("historical_fraction", 0.375)),
        fraction_per_state=float(
            config.get("historical_fraction_per_state", 0.01)
        ),
        fraction_max=float(config.get("historical_fraction_max", 0.375)),
    )
    return compute_start_sampling_probabilities(
        curriculum_probability=curriculum_probability,
        historical_fraction=historical_fraction,
        frontier_available=frontier_pool_size > 0,
        historical_available=historical_pool_size > 0,
    )


def effective_historical_fraction(
    historical_pool_size: int, *, adaptive: bool,
    fixed_fraction: float, fraction_per_state: float,
    fraction_max: float,
) -> float:
    """Grow historical replay with absolute learned-memory capacity."""
    if not adaptive:
        return float(fixed_fraction)
    return float(np.clip(
        fraction_per_state * max(0, int(historical_pool_size)),
        0.0, fraction_max,
    ))


def compute_start_sampling_probabilities(
    *, curriculum_probability: float, historical_fraction: float,
    frontier_available: bool, historical_available: bool,
) -> StartSamplingProbabilities:
    """Resolve the three probabilities without cascading empty-pool fallback."""
    if not frontier_available and not historical_available:
        return StartSamplingProbabilities(1.0, 0.0, 0.0, historical_fraction)
    if not historical_available:
        return StartSamplingProbabilities(
            1.0 - curriculum_probability, curriculum_probability, 0.0,
            historical_fraction,
        )
    historical = curriculum_probability * historical_fraction
    frontier = (
        curriculum_probability * (1.0 - historical_fraction)
        if frontier_available else 0.0
    )
    return StartSamplingProbabilities(
        1.0 - historical - frontier, frontier, historical,
        historical_fraction,
    )


def historical_quantile_bins(
    states: list[CurriculumState], bin_count: int,
) -> list[list[CurriculumState]]:
    """Stratifie la mémoire apprise par profondeur, sans hypothèse géométrique."""
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 0:
        raise ValueError("historical_bins doit être un entier strictement positif")
    if not states:
        return []
    ordered = sorted(
        states,
        key=lambda state: (
            int(getattr(state, "generation_depth", 0)),
            int(getattr(state, "state_id", -1)),
        ),
    )
    count = min(bin_count, len(ordered))
    return [
        [ordered[int(index)] for index in indices]
        for indices in np.array_split(np.arange(len(ordered)), count)
        if len(indices)
    ]


def select_training_start(
    rng: np.random.Generator,
    *,
    curriculum_probability: float,
    frontier_fraction: float,
    historical_fraction: float,
    historical_bins: int,
    frontier: list[CurriculumState],
    historical: list[CurriculumState],
    mastered_boundary: list[CurriculumState] | None = None,
    too_hard_near: list[CurriculumState] | None = None,
    requested: str = "auto",
    historical_bin_groups: list[list[CurriculumState]] | None = None,
    probabilities: StartSamplingProbabilities | None = None,
) -> CurriculumResetSelection:
    """Tire la source d'un épisode puis un état dans le pool choisi.

    ``probabilities`` active le tirage direct adaptatif. Sans lui, le
    chemin RNG et les fallbacks du sampler legacy restent inchangés.
    """
    mastered_boundary = [] if mastered_boundary is None else mastered_boundary
    too_hard_near = [] if too_hard_near is None else too_hard_near
    allowed = {"auto", "curriculum", *RESET_SOURCES}
    if requested not in allowed:
        raise ValueError(
            "options.reset_source doit être 'auto', 'curriculum', "
            "'true_start' ou une source curriculum connue"
        )
    fraction_total = frontier_fraction + historical_fraction
    if (not np.isfinite(curriculum_probability)
            or not 0.0 <= curriculum_probability <= 1.0):
        raise ValueError("curriculum_probability doit être dans [0, 1]")
    if (not np.isfinite(frontier_fraction)
            or not np.isfinite(historical_fraction)
            or frontier_fraction < 0.0 or historical_fraction < 0.0
            or not np.isclose(fraction_total, 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("les fractions frontier/historical doivent sommer à 1")
    if requested == "true_start":
        return CurriculumResetSelection("true_start", None)
    if requested in {
        "curriculum_frontier", "curriculum_historical",
        "curriculum_mastered_boundary", "curriculum_too_hard_near",
    }:
        preferred = requested.removeprefix("curriculum_")
        if preferred == "frontier" and not frontier:
            return CurriculumResetSelection("true_start", None)
        if preferred == "historical" and not historical:
            preferred = "frontier" if frontier else "true_start"
        if preferred == "mastered_boundary" and not mastered_boundary:
            preferred = "true_start"
        if preferred == "too_hard_near" and not too_hard_near:
            preferred = "true_start"
    elif requested == "curriculum":
        if frontier and historical:
            preferred = (
                "frontier"
                if rng.random() * fraction_total < frontier_fraction
                else "historical"
            )
        else:
            preferred = "frontier" if frontier else (
                "historical" if historical else "true_start"
            )
    elif probabilities is not None:
        # Adaptive strategies use one direct categorical reset draw.
        draw = float(rng.random())
        if draw < probabilities.frontier:
            preferred = "frontier"
        elif draw < probabilities.frontier + probabilities.historical:
            preferred = "historical"
        elif draw < (
            probabilities.frontier + probabilities.historical
            + probabilities.mastered_boundary
        ):
            preferred = "mastered_boundary"
        elif draw < (
            probabilities.frontier + probabilities.historical
            + probabilities.mastered_boundary + probabilities.too_hard_near
        ):
            preferred = "too_hard_near"
        else:
            preferred = "true_start"
    else:
        # Preserve the historical two-draw RNG path when both pools exist.
        if frontier and historical:
            if rng.random() >= curriculum_probability:
                return CurriculumResetSelection("true_start", None)
            preferred = (
                "frontier"
                if rng.random() * fraction_total < frontier_fraction
                else "historical"
            )
        elif frontier:
            preferred = (
                "frontier" if rng.random() < curriculum_probability
                else "true_start"
            )
        elif historical:
            preferred = (
                "historical"
                if rng.random() < curriculum_probability * historical_fraction
                else "true_start"
            )
        else:
            preferred = "true_start"
    if preferred == "true_start":
        return CurriculumResetSelection("true_start", None)
    if preferred == "frontier" and frontier:
        index = int(rng.integers(len(frontier)))
        return CurriculumResetSelection(
            "curriculum_frontier", frontier[index],
        )
    if preferred == "historical" and historical:
        bins = (
            historical_bin_groups
            if historical_bin_groups is not None
            else historical_quantile_bins(historical, historical_bins)
        )
        if not bins:
            # Cache incohérent : rester sûr et reconstruire depuis le pool.
            bins = historical_quantile_bins(historical, historical_bins)
        bin_index = int(rng.integers(len(bins)))
        state_index = int(rng.integers(len(bins[bin_index])))
        return CurriculumResetSelection(
            "curriculum_historical", bins[bin_index][state_index], bin_index,
        )
    if preferred == "mastered_boundary" and mastered_boundary:
        index = int(rng.integers(len(mastered_boundary)))
        return CurriculumResetSelection(
            "curriculum_mastered_boundary", mastered_boundary[index],
        )
    if preferred == "too_hard_near" and too_hard_near:
        index = int(rng.integers(len(too_hard_near)))
        return CurriculumResetSelection(
            "curriculum_too_hard_near", too_hard_near[index],
        )
    return CurriculumResetSelection("true_start", None)


@dataclass(frozen=True)
class PhysicsStepResult:
    action: np.ndarray
    true_error: np.ndarray
    final_wrench: np.ndarray
    max_force: float
    max_torque: float


@dataclass(frozen=True)
class CurriculumGenerationResult:
    state: CurriculumState
    geometric_success: bool
    success: bool
    unsafe: bool
    unsafe_force: bool
    unsafe_torque: bool
    unsafe_workspace: bool
    position_error: float
    rotation_error: float
    pose_distance: float
    max_force: float = 0.0
    max_torque: float = 0.0
    final_force: float = 0.0
    final_torque: float = 0.0
    contact_categories: tuple[str, ...] = ()


@dataclass
class GenerationReport:
    generated: int = 0
    valid: int = 0
    unsafe_rejected: int = 0
    successful_excluded: int = 0
    not_outward_rejected: int = 0
    deduplicated_rejected: int = 0
    restoration_checks: int = 0
    restoration_failures: int = 0
    invalid_rejected: int = 0
    # Coût et résultat de la dernière expansion multi-hop. ``generated`` reste
    # le nombre historique de pas physiques de reverse walk. Pour compatibilité,
    # ``expansion_hops`` est le nombre de tentatives de reverse walk et
    # ``expansion_candidates`` le nombre de snapshots arrivés à qualification.
    expansion_candidates: int = 0
    expansion_hops: int = 0
    expansion_branches: int = 0
    expansion_rollouts: int = 0
    new_mastered: int = 0
    new_frontier: int = 0
    new_too_hard: int = 0
    mean_hops_per_branch: float = 0.0
    max_hops_reached: int = 0
    expansion_scale_mean: float = 0.0
    expansion_scale_max: float = 0.0
    frontier_found_per_candidate: float = 0.0
    expansion_wall_time: float = 0.0
    stop_reasons: dict[str, int] = field(default_factory=dict)
    # Expansion diagnostics expose the candidate-generation funnel. A hop can
    # fail before policy qualification when reverse generation fails, its
    # snapshot is invalid/outside the workspace, or it is a duplicate.
    # Frontier/too-hard/max-hops/budget are normal branch terminations.
    raw_candidates_generated: int = 0
    valid_candidates: int = 0
    nonduplicate_candidates: int = 0
    qualified_candidates: int = 0
    raw_parent_translation_mm: list[float] = field(default_factory=list)
    raw_parent_rotation_deg: list[float] = field(default_factory=list)
    duplicate_parent_translation_mm: list[float] = field(default_factory=list)
    duplicate_parent_rotation_deg: list[float] = field(default_factory=list)
    duplicate_nearest_position_mm: list[float] = field(default_factory=list)
    duplicate_nearest_rotation_deg: list[float] = field(default_factory=list)
    reverse_steps: list[int] = field(default_factory=list)
    rejected_force_max: list[float] = field(default_factory=list)
    rejected_torque_max: list[float] = field(default_factory=list)
    rejected_force_step: list[int] = field(default_factory=list)
    rejected_torque_step: list[int] = field(default_factory=list)
    accepted_reverse_force_max: list[float] = field(default_factory=list)
    accepted_reverse_torque_max: list[float] = field(default_factory=list)
    candidate_final_force: list[float] = field(default_factory=list)
    candidate_final_torque: list[float] = field(default_factory=list)
    rejected_contact_counts: dict[str, int] = field(default_factory=dict)
    accepted_contact_counts: dict[str, int] = field(default_factory=dict)
    expansion_attempts: int = 0
    attempts_per_hop: list[int] = field(default_factory=list)
    attempt_no_candidate: int = 0
    attempt_duplicate: int = 0
    attempt_candidate_found: int = 0
    safe_prefix_candidates: int = 0
    full_walk_candidates: int = 0
    safe_prefix_steps: list[int] = field(default_factory=list)
    proposal_uniform_attempts: int = 0
    proposal_guided_attempts: int = 0
    proposal_uniform_candidates: int = 0
    proposal_guided_candidates: int = 0
    proposal_uniform_unique: int = 0
    proposal_guided_unique: int = 0
    proposal_uniform_safe_prefix: int = 0
    proposal_guided_safe_prefix: int = 0
    proposal_uniform_attempt_budget_failures: int = 0
    proposal_guided_attempt_budget_failures: int = 0
    persistent_attempts: int = 0
    independent_attempts: int = 0
    branch_heading_changes: list[float] = field(default_factory=list)
    attempt_to_heading_deviations: list[float] = field(default_factory=list)
    successive_hop_heading_opposition: int = 0
    guided_memory_insertions: int = 0
    guided_memory_rejected_duplicates: int = 0
    new_states_near_ancestor: int = 0
    nearest_ancestor_position_mm: list[float] = field(default_factory=list)
    nearest_ancestor_rotation_deg: list[float] = field(default_factory=list)

    def as_dict(self, states: list[CurriculumState]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "generated": self.generated,
            "valid": self.valid,
            "unsafe_rejected": self.unsafe_rejected,
            "successful_excluded": self.successful_excluded,
            "not_outward_rejected": self.not_outward_rejected,
            "deduplicated_rejected": self.deduplicated_rejected,
            "restoration_checks": self.restoration_checks,
            "restoration_failures": self.restoration_failures,
            "invalid_rejected": self.invalid_rejected,
            "expansion_candidates": self.expansion_candidates,
            "expansion_hops": self.expansion_hops,
            "expansion_branches": self.expansion_branches,
            "expansion_rollouts": self.expansion_rollouts,
            "new_mastered": self.new_mastered,
            "new_frontier": self.new_frontier,
            "new_too_hard": self.new_too_hard,
            "mean_hops_per_branch": self.mean_hops_per_branch,
            "max_hops_reached": self.max_hops_reached,
            "expansion_scale_mean": self.expansion_scale_mean,
            "expansion_scale_max": self.expansion_scale_max,
            "frontier_found_per_candidate": (
                self.frontier_found_per_candidate
            ),
            "expansion_wall_time": self.expansion_wall_time,
            "stop_reasons": dict(self.stop_reasons),
            "raw_candidates_generated": self.raw_candidates_generated,
            "valid_candidates": self.valid_candidates,
            "nonduplicate_candidates": self.nonduplicate_candidates,
            "qualified_candidates": self.qualified_candidates,
        }
        for name, values in (
            ("position_error", [state.position_error for state in states]),
            ("rotation_error", [state.rotation_error for state in states]),
            ("pose_distance", [state.pose_distance for state in states]),
        ):
            array = np.asarray(values, dtype=float)
            result[name] = (
                {
                    "min": float(np.min(array)),
                    "median": float(np.median(array)),
                    "max": float(np.max(array)),
                }
                if array.size else {"min": None, "median": None, "max": None}
            )
        return result


@dataclass
class StateLifecycleStats:
    created_update: int
    last_revalidated_update: int = -1
    revalidation_count: int = 0
    frontier_since_update: int | None = None
    consecutive_frontier_updates: int = 0
    nearest_ancestor_position_m: float | None = None
    nearest_ancestor_rotation_deg: float | None = None
    near_ancestor_return: bool = False


@dataclass(frozen=True)
class RevalidationReport:
    """Bilan d'une revalidation, sans effet sur le format des pools."""

    frontier_revalidated: int = 0
    mastered_revalidated: int = 0
    too_hard_revalidated: int = 0
    too_hard_to_frontier: int = 0
    too_hard_to_mastered: int = 0
    too_hard_remained_hard: int = 0
    frontier_promoted_to_mastered: int = 0
    frontier_remained_frontier: int = 0
    frontier_demoted_to_too_hard: int = 0
    frontier_rollouts: int = 0
    mastered_rollouts: int = 0
    too_hard_rollouts: int = 0
    wall_time: float = 0.0

    @property
    def total_revalidated(self) -> int:
        return (
            self.frontier_revalidated
            + self.mastered_revalidated
            + self.too_hard_revalidated
        )


@dataclass
class _ExpansionBranch:
    """État minimal d'une branche remise en queue entre deux hops."""

    current: CurriculumState
    scale: float
    hops: int = 0
    # Ces champs décrivent une hypothèse d'expansion temporaire. Ils ne font
    # volontairement pas partie du CurriculumState ni du state_dict.
    heading: np.ndarray | None = None
    proposal_kind: str | None = None


def mastered_boundary_states(
    mastered: list[CurriculumState],
) -> list[CurriculumState]:
    """Retourne les mastered qui n'ont aucun enfant actuellement mastered.

    Le bord est topologique : ni ``pose_distance`` ni le taux de succès ne
    servent à ordonner les états. La classification est déjà portée par le pool.
    """
    mastered_ids = {
        int(state.state_id) for state in mastered if int(state.state_id) >= 0
    }
    parents_with_mastered_children = {
        int(state.parent_id)
        for state in mastered
        if state.parent_id is not None
        and int(state.parent_id) in mastered_ids
    }
    return [
        state for state in mastered
        if int(state.state_id) not in parents_with_mastered_children
    ]


def mastered_edge_states(
    states: list[CurriculumState], fraction: float | None = None,
) -> list[CurriculumState]:
    """Alias legacy; ``fraction`` est ignorée par le curriculum à lineage."""
    del fraction
    return mastered_boundary_states(states)


def select_too_hard_by_lineage(
    too_hard: list[CurriculumState],
    mastered: list[CurriculumState],
    sample_count: int,
    rng: np.random.Generator,
) -> list[CurriculumState]:
    """Priorise les too_hard enfants d'un mastered, puis complète au hasard."""
    if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count < 0):
        raise ValueError(
            "curriculum.revalidation.too_hard_samples_per_update doit être "
            "un entier positif ou nul"
        )
    if sample_count == 0 or not too_hard:
        return []
    preferred = too_hard_near_states(too_hard, mastered)
    preferred_count = min(sample_count, len(preferred))
    selected: list[CurriculumState] = []
    if preferred_count:
        indices = rng.choice(
            len(preferred), size=preferred_count, replace=False,
        )
        selected.extend(preferred[int(index)] for index in np.atleast_1d(indices))

    remaining_count = min(sample_count - len(selected), len(too_hard) - len(preferred))
    if remaining_count:
        preferred_objects = {id(state) for state in preferred}
        fallback = [
            state for state in too_hard if id(state) not in preferred_objects
        ]
        indices = rng.choice(
            len(fallback), size=remaining_count, replace=False,
        )
        selected.extend(fallback[int(index)] for index in np.atleast_1d(indices))
    return selected


def too_hard_near_states(
    too_hard: list[CurriculumState], mastered: list[CurriculumState],
) -> list[CurriculumState]:
    """Return only too-hard states whose direct parent is mastered now."""
    mastered_ids = {int(state.state_id) for state in mastered}
    return [
        state for state in too_hard
        if state.parent_id is not None and int(state.parent_id) in mastered_ids
    ]


def select_too_hard_near_mastered(
    too_hard: list[CurriculumState],
    mastered: list[CurriculumState],
    sample_count: int,
    rng: np.random.Generator,
) -> list[CurriculumState]:
    """Alias legacy de la sélection désormais fondée uniquement sur le lineage."""
    return select_too_hard_by_lineage(too_hard, mastered, sample_count, rng)


def classify_success_rate(success_rate: float, low: float, high: float) -> str:
    """Classe un taux; les deux seuils appartiennent à la frontier."""
    if not np.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
        raise ValueError("success_rate doit être fini et compris entre 0 et 1")
    if success_rate < low:
        return "too_hard"
    if success_rate <= high:
        return "frontier"
    return "mastered"


class ReverseCurriculumManager:
    """Génère, qualifie, classe et sérialise les starts du curriculum.

    Chaque snapshot possède un ``state_id`` unique. Son ``parent_id`` et sa
    ``generation_depth`` sont immuables, y compris lorsqu'il change de pool.
    Le lineage décrit uniquement la topologie des expansions physiques; la
    difficulté et la classification dépendent du taux de succès courant.
    ``pose_distance`` n'intervient que dans les métriques géométriques de
    diagnostic, jamais dans une décision de progression.
    """

    STATE_VERSION = 4

    def __init__(
        self, env: TenonMortaiseEnv, config: dict[str, Any], *, seed: int,
    ) -> None:
        self.env = env
        self.config = config
        self.walk = config["reverse_random_walk"]
        self.deduplication = config["deduplication"]
        # SeedSequence rend la dérivation explicite sans dépendre du RNG global.
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 21]))
        self.torch_seed = int(seed)
        self.torch_rng_state = self._initial_torch_rng(seed)
        self.torch_cuda_rng_states: list[torch.Tensor] | None = None
        self.pools: dict[str, list[CurriculumState]] = {
            name: [] for name in POOL_NAMES
        }
        self.next_state_id = 1
        self.update_count = 0
        self.next_update_timesteps = int(config["update_interval_timesteps"])
        self.worker_rng_states: list[dict[str, Any]] | None = None
        self.loaded_training_timesteps: int | None = None
        self.goal_seed = self.env.build_goal_seed(seed=seed)
        self.last_generation_report = GenerationReport()
        self.last_revalidation_report = RevalidationReport()
        self.last_expansion_seed_distances: list[float] = []
        self.last_expansion_seed_depths: list[int] = []
        # Guidance is deliberately ephemeral: losing it on resume affects
        # proposal efficiency, never pools, lineage, or physical snapshots.
        self.proposal_memory: dict[int, list[np.ndarray]] = {}
        self._active_proposal_direction: np.ndarray | None = None
        self._active_proposal_kind = "uniform"
        self.state_lifecycle: dict[int, StateLifecycleStats] = {}
        self.sampling_episode_length_ema = {
            name: 1.0 for name in SAMPLING_SOURCE_NAMES
        }

    def _proposal_settings(self) -> dict[str, Any]:
        proposal = self.walk.get("proposal", {})
        return {
            "guided_fraction": float(proposal.get("guided_fraction", 0.0)),
            "guided_noise_std": float(proposal.get("guided_noise_std", 0.20)),
            "memory_size_per_parent": int(
                proposal.get("memory_size_per_parent", 16)
            ),
        }

    def _proposal_mode(self) -> str:
        return str(self.walk.get("proposal_mode", "independent"))

    def _persistent_proposal_settings(self) -> dict[str, float]:
        persistent = self.walk.get("persistent_proposal", {})
        return {
            "attempt_direction_noise_std": float(
                persistent.get("attempt_direction_noise_std", 0.20)
            ),
            "hop_direction_noise_std": float(
                persistent.get("hop_direction_noise_std", 0.15)
            ),
            "step_noise_std": float(persistent.get("step_noise_std", 0.10)),
        }

    def _choose_reverse_proposal(
        self, parent: CurriculumState,
    ) -> tuple[str, np.ndarray | None]:
        settings = self._proposal_settings()
        memory = getattr(self, "proposal_memory", {}).get(int(parent.state_id), [])
        fraction = settings["guided_fraction"]
        # With fraction zero no extra RNG draw is made: A/B uniform runs stay
        # bit-identical to the pre-guidance implementation.
        if not memory or fraction <= 0.0 or self.rng.random() >= fraction:
            return "uniform", None
        index = int(self.rng.integers(len(memory)))
        return "guided", memory[index].copy()

    def _proposal_direction(
        self, parent: CurriculumState, candidate: CurriculumState,
    ) -> np.ndarray:
        """Convert a useful SE(3) displacement into an action-space direction.

        Dividing by one physical action step makes metres and radians
        comparable. Normalizing the largest component then keeps only the
        direction and relative axis mix: a guided attempt is free to extend
        beyond the candidate that originally revealed that direction.
        """
        delta = relative(
            (parent.task_position, parent.task_quaternion),
            (candidate.task_position, candidate.task_quaternion),
        )
        env_config = getattr(getattr(self, "env", None), "cfg", {})
        action = env_config.get("action", {})
        scales = np.r_[
            np.full(3, float(action.get("max_translation_step", 1.0))),
            np.full(3, np.deg2rad(float(
                action.get("max_rotation_step_deg", 57.2957795)
            ))),
        ]
        normalized = np.r_[delta[0], quat_to_rotvec(delta[1])] / scales
        largest_component = float(np.max(np.abs(normalized)))
        if largest_component <= 0.0:
            return np.zeros(6, dtype=float)
        return np.clip(normalized / largest_component, -1.0, 1.0)

    def _remember_proposal(
        self, parent: CurriculumState, candidate: CurriculumState,
    ) -> None:
        """Keep only directions that produced a genuinely new physical state."""
        key = int(parent.state_id)
        if not hasattr(self, "proposal_memory"):
            self.proposal_memory = {}
        memory = self.proposal_memory.setdefault(key, [])
        memory.append(self._proposal_direction(parent, candidate))
        limit = self._proposal_settings()["memory_size_per_parent"]
        while len(memory) > limit:
            memory.pop(0)

    def _initial_branch_heading(
        self, proposal_kind: str, guided_direction: np.ndarray | None,
    ) -> np.ndarray:
        """Initialize one branch, consulting guided memory only at its root."""
        if guided_direction is None:
            return self.rng.uniform(-1.0, 1.0, size=6)
        heading = np.asarray(guided_direction, dtype=float).copy()
        guided_noise = self._proposal_settings()["guided_noise_std"]
        if proposal_kind == "guided" and guided_noise > 0.0:
            heading += self.rng.normal(0.0, guided_noise, size=6)
        return np.clip(heading, -1.0, 1.0)

    def _persistent_attempt_direction(
        self, branch_heading: np.ndarray,
    ) -> np.ndarray:
        """Sample one retry-local variation around a branch heading."""
        direction = np.asarray(branch_heading, dtype=float).copy()
        noise = self._persistent_proposal_settings()[
            "attempt_direction_noise_std"
        ]
        if noise > 0.0:
            direction += self.rng.normal(0.0, noise, size=6)
        return np.clip(direction, -1.0, 1.0)

    def _next_branch_heading(
        self, heading: np.ndarray,
    ) -> np.ndarray:
        """Let a persistent branch curve locally between two mastered hops."""
        next_heading = np.asarray(heading, dtype=float).copy()
        noise = self._persistent_proposal_settings()["hop_direction_noise_std"]
        if noise > 0.0:
            next_heading += self.rng.normal(0.0, noise, size=6)
        return np.clip(next_heading, -1.0, 1.0)

    def _reverse_step_action(
        self, amplitude: float, attempt_direction: np.ndarray | None,
    ) -> np.ndarray:
        """Sample one step while preserving the exact independent RNG path."""
        if self._proposal_mode() == "independent":
            if attempt_direction is None:
                return self.rng.uniform(-amplitude, amplitude, size=6)
            noise = self.rng.normal(
                0.0, self._proposal_settings()["guided_noise_std"], size=6,
            )
            return np.clip(attempt_direction + noise, -1.0, 1.0) * amplitude

        direction = np.asarray(attempt_direction, dtype=float)
        step_noise = self._persistent_proposal_settings()["step_noise_std"]
        if step_noise > 0.0:
            direction = direction + self.rng.normal(0.0, step_noise, size=6)
        return np.clip(direction, -1.0, 1.0) * amplitude

    def _task_config_sha256(
        self, curriculum_config: dict[str, Any] | None = None, *,
        legacy: bool = False,
    ) -> str:
        # Le budget/checkpoint/eval et la stratégie de sampling peuvent changer
        # lors d'une reprise. La physique, la reward et la tâche doivent rester.
        task_config = {
            key: value for key, value in self.env.cfg.items()
            if key not in {"training", "evaluation", "curriculum"}
        }
        if legacy:
            task_config["curriculum"] = (
                self.env.cfg["curriculum"]
                if curriculum_config is None else curriculum_config
            )
        encoded = json.dumps(
            task_config, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _initial_torch_rng(seed: int) -> torch.Tensor:
        training_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            return torch.random.get_rng_state().clone()
        finally:
            torch.random.set_rng_state(training_state)

    @property
    def total_pool_size(self) -> int:
        return sum(len(pool) for pool in self.pools.values())

    def pool_sizes(self) -> dict[str, int]:
        return {name: len(self.pools[name]) for name in POOL_NAMES}

    def all_states(self) -> list[CurriculumState]:
        return [state for name in POOL_NAMES for state in self.pools[name]]

    def training_reset_pools(self) -> dict[str, list[CurriculumState]]:
        """Build all reset views from the current pool classification."""
        return {
            "frontier": list(self.pools["frontier"]),
            "historical": list(self.pools["mastered"]),
            "mastered_boundary": self.mastered_boundary_states(),
            "too_hard_near": too_hard_near_states(
                self.pools["too_hard"], self.pools["mastered"],
            ),
        }

    def training_states(self) -> list[CurriculumState]:
        """Legacy view of the two original training reset pools."""
        pools = self.training_reset_pools()
        return pools["frontier"] + pools["historical"]

    def frontier_success_rate_mean(self) -> float:
        values = [state.success_rate for state in self.pools["frontier"]]
        return float(np.mean(values)) if values else math.nan

    def pool_distance_statistics(self, name: str) -> dict[str, float]:
        if name not in POOL_NAMES:
            raise ValueError(f"Pool curriculum inconnu: {name}")
        values = np.asarray(
            [state.pose_distance for state in self.pools[name]], dtype=float,
        )
        if not values.size:
            return {
                key: math.nan for key in ("min", "q25", "median", "q75", "max")
            }
        return {
            "min": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
        }

    def pool_depth_statistics(self, name: str) -> dict[str, float]:
        """Décrit la progression topologique d'un pool, sans notion de difficulté."""
        if name not in POOL_NAMES:
            raise ValueError(f"Pool curriculum inconnu: {name}")
        values = np.asarray(
            [state.generation_depth for state in self.pools[name]], dtype=float,
        )
        if not values.size:
            return {
                key: math.nan for key in ("min", "q25", "median", "q75", "max")
            }
        return {
            "min": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
        }

    def _build_lineage_index(
        self,
    ) -> tuple[dict[int, CurriculumState], dict[int, list[CurriculumState]]]:
        """Indexe les IDs et enfants sans sérialiser de références Python."""
        states_by_id: dict[int, CurriculumState] = {}
        children_by_parent_id: dict[int, list[CurriculumState]] = {}
        for state in self.all_states():
            state_id = int(state.state_id)
            if state_id < 0 or state_id in states_by_id:
                raise ValueError("Les state_id curriculum doivent être uniques et positifs")
            states_by_id[state_id] = state
            if state.parent_id is not None:
                children_by_parent_id.setdefault(
                    int(state.parent_id), [],
                ).append(state)
        return states_by_id, children_by_parent_id

    def mastered_boundary_states(self) -> list[CurriculumState]:
        """Retourne les feuilles du sous-graphe actuellement classé mastered."""
        return mastered_boundary_states(self.pools["mastered"])

    def mastered_edge_states(self) -> list[CurriculumState]:
        """Alias legacy du bord topologique, sans usage de l'ancienne fraction."""
        return self.mastered_boundary_states()

    def _expansion_settings(self) -> dict[str, int | float]:
        """Retourne la stratégie courante avec defaults pour les anciens YAML."""
        settings = dict(EXPANSION_DEFAULTS)
        configured = self.config.get("expansion", {})
        if isinstance(configured, dict):
            settings.update({
                key: configured[key]
                for key in EXPANSION_STRATEGY_KEYS if key in configured
            })
        return settings

    def _expansion_seeds(self) -> list[CurriculumState]:
        """Mélange toutes les branches éligibles sans classement de profondeur.

        La limite globale est appliquée par ``_expand_branches``. Garder ici
        toutes les feuilles permet à la queue round-robin de donner un premier
        hop au plus grand nombre possible de branches avant leur second hop.
        """
        mastered = self.pools["mastered"]
        preferred = self.mastered_boundary_states() or mastered
        if not preferred:
            preferred = self.pools["frontier"]
        if not preferred:
            selected = [self.goal_seed]
            self.last_expansion_seed_distances = [
                float(self.goal_seed.pose_distance)
            ]
            self.last_expansion_seed_depths = [
                int(self.goal_seed.generation_depth)
            ]
            return selected
        indices = self.rng.permutation(len(preferred))
        selected = [
            preferred[int(index)] for index in np.atleast_1d(indices)
        ]
        self.last_expansion_seed_distances = [
            float(state.pose_distance) for state in selected
        ]
        self.last_expansion_seed_depths = [
            int(state.generation_depth) for state in selected
        ]
        return selected

    def _next_expansion_scale(self, scale: float, category: str) -> float:
        """Adapte uniquement l'amplitude de génération reverse.

        Cette échelle ne mesure ni difficulté ni progression. Dans cette
        version, seul le résultat ``mastered`` est immédiatement ré-expansé;
        la valeur réduite calculable pour ``too_hard`` n'est donc pas réessayée
        dans le même update.
        """
        settings = self._expansion_settings()
        minimum = float(settings["min_scale"])
        maximum = float(settings["max_scale"])
        if category == "mastered":
            scale *= float(settings["scale_up_factor"])
        elif category == "too_hard":
            scale *= float(settings["scale_down_factor"])
        return float(np.clip(scale, minimum, maximum))

    def _is_duplicate(
        self, candidate: CurriculumState, additional: list[CurriculumState],
    ) -> bool:
        """Rejette une pose déjà connue sans créer de graphe multi-parent.

        L'état existant conserve volontairement son lineage d'origine; un
        duplicate ne reçoit donc ni nouvel identifiant ni nouveau parent.
        """
        duplicate, nearest_position, nearest_rotation = self._duplicate_match(
            candidate, additional,
        )
        self._last_duplicate_match = (nearest_position, nearest_rotation)
        return duplicate

    def _duplicate_match(
        self, candidate: CurriculumState, additional: list[CurriculumState],
    ) -> tuple[bool, float | None, float | None]:
        """Return duplicate status and nearest-state deltas in native units."""
        position_tolerance = float(self.deduplication["position_tolerance"])
        rotation_tolerance = np.deg2rad(
            float(self.deduplication["rotation_tolerance_deg"])
        )
        nearest_position: float | None = None
        nearest_rotation: float | None = None
        duplicate = False
        for existing in self.all_states() + additional:
            position_delta = float(np.linalg.norm(
                candidate.task_position - existing.task_position
            ))
            rotation_delta = float(np.linalg.norm(quat_to_rotvec(relative(
                (np.zeros(3), existing.task_quaternion),
                (np.zeros(3), candidate.task_quaternion),
            )[1])))
            # A closest pair is useful only as a diagnostic: duplicate uses
            # the conjunction below, exactly as before.
            if nearest_position is None or position_delta < nearest_position:
                nearest_position, nearest_rotation = position_delta, rotation_delta
            duplicate |= (
                position_delta < position_tolerance
                and rotation_delta < rotation_tolerance
            )
        return duplicate, nearest_position, nearest_rotation

    @staticmethod
    def _parent_candidate_delta(
        parent: CurriculumState, candidate: CurriculumState,
    ) -> tuple[float, float]:
        position = float(np.linalg.norm(
            candidate.task_position - parent.task_position
        ))
        rotation = float(np.linalg.norm(quat_to_rotvec(relative(
            (np.zeros(3), parent.task_quaternion),
            (np.zeros(3), candidate.task_quaternion),
        )[1])))
        return position * 1000.0, float(np.rad2deg(rotation))

    def _ancestor_diagnostics(
        self, candidate: CurriculumState, known: list[CurriculumState],
    ) -> tuple[float | None, float | None, bool]:
        """Measure lineage geometry only; never feed it into acceptance."""
        states = {
            int(state.state_id): state for state in known
            if int(state.state_id) >= 0
        }
        # The direct parent is necessarily local and would make this metric
        # trivially high. Detect returns to earlier lineage states instead.
        parent = states.get(int(candidate.parent_id)) if candidate.parent_id is not None else None
        ancestor_id = None if parent is None else parent.parent_id
        nearest_position: float | None = None
        nearest_rotation: float | None = None
        while ancestor_id is not None and int(ancestor_id) in states:
            ancestor = states[int(ancestor_id)]
            position = float(np.linalg.norm(
                candidate.task_position - ancestor.task_position
            ))
            rotation = float(np.rad2deg(np.linalg.norm(quat_to_rotvec(relative(
                (np.zeros(3), ancestor.task_quaternion),
                (np.zeros(3), candidate.task_quaternion),
            )[1]))))
            if nearest_position is None or position < nearest_position:
                nearest_position, nearest_rotation = position, rotation
            ancestor_id = ancestor.parent_id
        diagnostics = self.config.get("diagnostics", {})
        near = (
            nearest_position is not None
            and nearest_position < float(
                diagnostics.get("near_ancestor_position_m", 0.001)
            )
            and nearest_rotation is not None
            and nearest_rotation < float(
                diagnostics.get("near_ancestor_rotation_deg", 1.0)
            )
        )
        return nearest_position, nearest_rotation, bool(near)

    def _assign_lineage_to_candidate(
        self, candidate: CurriculumState, seed: CurriculumState,
    ) -> CurriculumState:
        """Crée une génération d'expansion, indépendamment des steps du walk."""
        seed_id = int(seed.state_id)
        seed_depth = int(seed.generation_depth)
        if seed_depth < 0:
            raise ValueError("generation_depth doit être positif ou nul")
        return replace(
            candidate,
            state_id=self.next_state_id,
            parent_id=None if seed_id < 0 else seed_id,
            generation_depth=seed_depth + 1,
        )

    @staticmethod
    def _candidate_snapshot_is_valid(candidate: CurriculumState) -> bool:
        """Rejette les snapshots numériques non finis avant tout lineage.

        MuJoCo et les contrôles de workspace traitent déjà les violations
        physiques. Ce garde-fou couvre explicitement les NaN/Inf et les
        quaternions dégénérés sans introduire de critère de difficulté.
        """
        for name in (
            "mj_state", "fixed_body_position", "fixed_body_quaternion",
            "contact_friction", "admittance_offset", "admittance_velocity",
            "reference_position", "reference_quaternion",
            "perception_bias_position", "perception_bias_quaternion",
            "task_position", "task_quaternion",
        ):
            value = getattr(candidate, name, None)
            if value is None:
                continue
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                return False
            if not array.size or not np.all(np.isfinite(array)):
                return False
        for name in (
            "friction_scale", "position_error", "rotation_error",
            "pose_distance",
        ):
            if hasattr(candidate, name):
                try:
                    value = float(getattr(candidate, name))
                except (TypeError, ValueError):
                    return False
                if not np.isfinite(value):
                    return False
        for name in (
            "fixed_body_quaternion", "reference_quaternion",
            "perception_bias_quaternion", "task_quaternion",
        ):
            quaternion = getattr(candidate, name, None)
            if quaternion is not None and float(np.linalg.norm(quaternion)) <= 1e-12:
                return False
        return True

    def _generate_hop_snapshot(
        self, seed: CurriculumState, expansion_scale: float,
        report: GenerationReport,
    ) -> tuple[CurriculumState | None, str | None]:
        """Exécute un reverse walk et retourne au plus son dernier état valide.

        Le snapshot est capturé par l'environnement à l'instant physique exact.
        Une violation unsafe arrête immédiatement la physique, mais le dernier
        prefix safe non-success reste récupérable. Les sous-steps encore
        successful sont traversés sans devenir des candidats. Un hop correspond
        à un seul walk; ``walks_per_seed`` appartient uniquement au générateur
        legacy utilisé par le bootstrap et les diagnostics.
        """
        # Forward safety and reverse state-generation validity are distinct
        # concepts. Reverse walks start from contact-rich assembly states, so
        # transient contact/wrench is measured separately from the final
        # CurriculumState. Existing rejection rules remain unchanged here.
        self.env.restore_curriculum_state(
            seed, reset_episode=False, restore_rng=True,
        )
        # L'action MuJoCo est bornée à [-1, 1]. Borner l'amplitude avant le
        # tirage évite qu'un clipping ultérieur crée artificiellement une masse
        # de probabilité exactement aux deux bornes.
        amplitude = min(
            float(self.walk["action_scale"]) * float(expansion_scale), 1.0,
        )
        proposal_kind = getattr(self, "_active_proposal_kind", "uniform")
        configured_heading = getattr(
            self, "_active_proposal_direction", None,
        )
        if self._proposal_mode() == "persistent":
            report.persistent_attempts += 1
            branch_heading = (
                self._initial_branch_heading(proposal_kind, None)
                if configured_heading is None
                else np.asarray(configured_heading, dtype=float)
            )
            attempt_direction = self._persistent_attempt_direction(branch_heading)
            report.attempt_to_heading_deviations.append(float(np.linalg.norm(
                attempt_direction - branch_heading
            )))
        else:
            report.independent_attempts += 1
            attempt_direction = configured_heading
        last_candidate: CurriculumState | None = None
        walk_max_force = 0.0
        walk_max_torque = 0.0
        final_force = 0.0
        final_torque = 0.0
        walk_contacts: set[str] = set()
        for step_index in range(int(self.walk["max_steps"])):
            action = self._reverse_step_action(
                amplitude, attempt_direction,
            )
            result = self.env.step_for_curriculum_generation(action)
            report.generated += 1
            walk_max_force = max(
                walk_max_force, float(getattr(result, "max_force", 0.0)),
            )
            walk_max_torque = max(
                walk_max_torque, float(getattr(result, "max_torque", 0.0)),
            )
            walk_contacts.update(getattr(result, "contact_categories", ()))
            if result.unsafe:
                report.unsafe_rejected += 1
                if result.unsafe_force:
                    report.rejected_force_max.append(walk_max_force)
                    report.rejected_force_step.append(step_index + 1)
                if result.unsafe_torque:
                    report.rejected_torque_max.append(walk_max_torque)
                    report.rejected_torque_step.append(step_index + 1)
                if result.unsafe_workspace:
                    reason = ExpansionStopReason.WORKSPACE.value
                elif result.unsafe_force and result.unsafe_torque:
                    reason = ExpansionStopReason.FORCE_AND_TORQUE.value
                elif result.unsafe_force:
                    reason = ExpansionStopReason.FORCE.value
                elif result.unsafe_torque:
                    reason = ExpansionStopReason.TORQUE.value
                else:
                    reason = ExpansionStopReason.OTHER_INVALID.value
                for category in walk_contacts:
                    report.rejected_contact_counts[category] = (
                        report.rejected_contact_counts.get(category, 0) + 1
                    )
                # A valid non-success prefix is an admissible curriculum
                # state; stop at the unsafe transition but do not traverse it.
                if last_candidate is not None:
                    report.raw_candidates_generated += 1
                    report.reverse_steps.append(step_index)
                    report.safe_prefix_candidates += 1
                    report.safe_prefix_steps.append(step_index)
                    report.candidate_final_force.append(final_force)
                    report.candidate_final_torque.append(final_torque)
                    return last_candidate, None
                return None, reason
            if not self._candidate_snapshot_is_valid(result.state):
                report.invalid_rejected += 1
                return None, ExpansionStopReason.SNAPSHOT_INVALID.value
            if result.success:
                report.successful_excluded += 1
                continue
            last_candidate = result.state
            final_force = float(getattr(result, "final_force", 0.0))
            final_torque = float(getattr(result, "final_torque", 0.0))
        if last_candidate is None:
            # This is deliberately narrow: the reverse generator completed a
            # walk but never yielded a non-success state usable as a raw hop.
            return None, "generation_failed"
        report.raw_candidates_generated += 1
        report.reverse_steps.append(int(self.walk["max_steps"]))
        report.accepted_reverse_force_max.append(walk_max_force)
        report.accepted_reverse_torque_max.append(walk_max_torque)
        report.candidate_final_force.append(final_force)
        report.candidate_final_torque.append(final_torque)
        report.full_walk_candidates += 1
        for category in walk_contacts:
            report.accepted_contact_counts[category] = (
                report.accepted_contact_counts.get(category, 0) + 1
            )
        return last_candidate, None

    @staticmethod
    def _record_expansion_stop(
        report: GenerationReport, reason: str, count: int = 1,
    ) -> None:
        report.stop_reasons[reason] = report.stop_reasons.get(reason, 0) + count

    def _expand_branches(
        self, model: BaseAlgorithm,
        seeds: list[CurriculumState] | None = None,
    ) -> GenerationReport:
        """Étend plusieurs branches en round-robin avec un budget global.

        A branch is expanded repeatedly within one curriculum update while
        newly generated states are already mastered by the current policy.

        Expansion stops when a frontier or too-hard state is reached, when
        the branch becomes invalid/duplicate, or when a configured budget is
        exhausted. This lets the curriculum cross already-easy regions quickly
        instead of waiting one curriculum update per generation.

        Chaque hop part du snapshot exact du candidat précédent et crée au
        plus un nouvel état. La queue remet une continuation mastered derrière
        toutes les branches en attente : aucune profondeur ou distance n'est
        privilégiée.
        """
        settings = self._expansion_settings()
        max_hops = int(settings["max_hops_per_seed"])
        max_attempts = int(settings["max_attempts_per_hop"])
        candidate_budget = int(settings["max_candidates_per_update"])
        initial_scale = float(np.clip(
            float(settings["initial_scale"]),
            float(settings["min_scale"]),
            float(settings["max_scale"]),
        ))
        selected = self._expansion_seeds() if seeds is None else list(seeds)
        if seeds is not None:
            self.last_expansion_seed_distances = [
                float(state.pose_distance) for state in selected
            ]
            self.last_expansion_seed_depths = [
                int(state.generation_depth) for state in selected
            ]

        branches = []
        for seed in selected:
            branch = _ExpansionBranch(seed, initial_scale)
            if self._proposal_mode() == "persistent":
                proposal_kind, guided_direction = self._choose_reverse_proposal(
                    seed,
                )
                branch.proposal_kind = proposal_kind
                branch.heading = self._initial_branch_heading(
                    proposal_kind, guided_direction,
                )
            branches.append(branch)
        queue = deque(branches)
        report = GenerationReport()
        used_scales: list[float] = []
        attempted_seeds: list[CurriculumState] = []
        # Le pruning peut retirer un état entre deux hops. Garder toutes les
        # poses connues pendant cet update empêche alors sa réintroduction sous
        # un nouvel ID/lineage si une branche la rencontre de nouveau.
        known_during_update = self.all_states()
        rollouts_per_candidate = int(
            self.config["evaluation_rollouts_per_candidate"]
        )
        low = float(self.config["success_rate_low"])
        high = float(self.config["success_rate_high"])
        started_at = time.perf_counter()

        try:
            while queue and report.expansion_candidates < candidate_budget:
                branch = queue.popleft()
                if branch.hops == 0:
                    report.expansion_branches += 1
                    attempted_seeds.append(branch.current)
                # Une tentative est bornée avant toute génération et donc, a
                # fortiori, avant ses rollouts de qualification.
                branch.hops += 1
                report.expansion_hops += 1
                used_scales.append(float(branch.scale))

                candidate: CurriculumState | None = None
                last_proposal_kind = "uniform"
                attempts_before_hop = report.expansion_attempts
                for _ in range(max_attempts):
                    # _generate_hop_snapshot restores branch.current on every
                    # call; retries therefore never inherit a failed walk.
                    report.expansion_attempts += 1
                    if self._proposal_mode() == "persistent":
                        if branch.heading is None or branch.proposal_kind is None:
                            raise AssertionError(
                                "Branche persistante sans heading initialisé"
                            )
                        proposal_kind = branch.proposal_kind
                        direction = branch.heading
                    else:
                        proposal_kind, direction = self._choose_reverse_proposal(
                            branch.current,
                        )
                    last_proposal_kind = proposal_kind
                    setattr(
                        report, f"proposal_{proposal_kind}_attempts",
                        getattr(report, f"proposal_{proposal_kind}_attempts") + 1,
                    )
                    prefix_before = report.safe_prefix_candidates
                    self._active_proposal_kind = proposal_kind
                    self._active_proposal_direction = direction
                    try:
                        snapshot, _ = self._generate_hop_snapshot(
                            branch.current, branch.scale, report,
                        )
                    finally:
                        self._active_proposal_direction = None
                        self._active_proposal_kind = "uniform"
                    if snapshot is None:
                        report.attempt_no_candidate += 1
                        continue

                    setattr(
                        report, f"proposal_{proposal_kind}_candidates",
                        getattr(report, f"proposal_{proposal_kind}_candidates") + 1,
                    )
                    if report.safe_prefix_candidates > prefix_before:
                        setattr(
                            report, f"proposal_{proposal_kind}_safe_prefix",
                            getattr(report, f"proposal_{proposal_kind}_safe_prefix") + 1,
                        )

                    report.valid_candidates += 1
                    translation_mm, rotation_deg = self._parent_candidate_delta(
                        branch.current, snapshot,
                    )
                    report.raw_parent_translation_mm.append(translation_mm)
                    report.raw_parent_rotation_deg.append(rotation_deg)
                    proposal = self._assign_lineage_to_candidate(
                        snapshot, branch.current,
                    )
                    self._last_duplicate_match = (None, None)
                    if self._is_duplicate(proposal, known_during_update):
                        report.deduplicated_rejected += 1
                        report.attempt_duplicate += 1
                        report.duplicate_parent_translation_mm.append(translation_mm)
                        report.duplicate_parent_rotation_deg.append(rotation_deg)
                        nearest_pos, nearest_rot = self._last_duplicate_match
                        if nearest_pos is not None:
                            report.duplicate_nearest_position_mm.append(nearest_pos * 1000.0)
                        if nearest_rot is not None:
                            report.duplicate_nearest_rotation_deg.append(float(np.rad2deg(nearest_rot)))
                        # Duplicate prefixes used to reinforce the very
                        # direction that rediscovered an existing state.
                        report.guided_memory_rejected_duplicates += 1
                        continue
                    candidate = proposal
                    self._remember_proposal(branch.current, proposal)
                    report.guided_memory_insertions += 1
                    setattr(
                        report, f"proposal_{proposal_kind}_unique",
                        getattr(report, f"proposal_{proposal_kind}_unique") + 1,
                    )
                    report.attempt_candidate_found += 1
                    break
                report.attempts_per_hop.append(
                    report.expansion_attempts - attempts_before_hop
                )
                if candidate is None:
                    setattr(
                        report,
                        f"proposal_{last_proposal_kind}_attempt_budget_failures",
                        getattr(
                            report,
                            f"proposal_{last_proposal_kind}_attempt_budget_failures",
                        ) + 1,
                    )
                    self._record_expansion_stop(
                        report, ExpansionStopReason.ATTEMPT_BUDGET.value,
                    )
                    continue

                ancestor_position, ancestor_rotation, near_ancestor = (
                    self._ancestor_diagnostics(candidate, known_during_update)
                )

                # L'identifiant n'est consommé que par un snapshot réellement
                # nouveau. Le budget est encore disponible car il est testé en
                # tête de boucle, avant cet unique appel de qualification.
                self.next_state_id += 1
                report.valid += 1
                report.nonduplicate_candidates += 1
                qualified = self.qualify_candidates(model, [candidate])
                if len(qualified) != 1:
                    raise RuntimeError(
                        "La qualification d'un hop doit retourner exactement un état"
                    )
                state = qualified[0]
                report.qualified_candidates += 1
                category = classify_success_rate(
                    state.success_rate, low, high,
                )
                if not hasattr(self, "state_lifecycle"):
                    self.state_lifecycle = {}
                self.state_lifecycle[int(state.state_id)] = StateLifecycleStats(
                    created_update=int(getattr(self, "update_count", 0)) + 1,
                    frontier_since_update=(
                        int(getattr(self, "update_count", 0)) + 1
                        if category == "frontier" else None
                    ),
                    consecutive_frontier_updates=(
                        1 if category == "frontier" else 0
                    ),
                    nearest_ancestor_position_m=ancestor_position,
                    nearest_ancestor_rotation_deg=ancestor_rotation,
                    near_ancestor_return=near_ancestor,
                )
                if ancestor_position is not None:
                    report.nearest_ancestor_position_mm.append(
                        ancestor_position * 1000.0
                    )
                if ancestor_rotation is not None:
                    report.nearest_ancestor_rotation_deg.append(
                        ancestor_rotation
                    )
                report.new_states_near_ancestor += int(near_ancestor)
                # Le pool peut être élagué ici. La continuation conserve
                # néanmoins l'objet snapshot exact qualifié comme parent du
                # prochain hop; elle ne reconstruit jamais sa pose.
                self._insert([state])
                known_during_update.append(state)
                report.expansion_candidates += 1
                report.expansion_rollouts += rollouts_per_candidate
                if category == "mastered":
                    report.new_mastered += 1
                elif category == "frontier":
                    report.new_frontier += 1
                else:
                    report.new_too_hard += 1

                if category != "mastered":
                    self._record_expansion_stop(report, category)
                    continue
                if branch.hops >= max_hops:
                    self._record_expansion_stop(report, ExpansionStopReason.MAX_HOPS.value)
                    continue
                branch.current = state
                branch.scale = self._next_expansion_scale(
                    branch.scale, category,
                )
                if self._proposal_mode() == "persistent":
                    if branch.heading is None:
                        raise AssertionError(
                            "Branche persistante sans heading à faire évoluer"
                        )
                    previous_heading = branch.heading
                    next_heading = self._next_branch_heading(previous_heading)
                    report.branch_heading_changes.append(float(np.linalg.norm(
                        next_heading - previous_heading
                    )))
                    norm_product = float(
                        np.linalg.norm(previous_heading)
                        * np.linalg.norm(next_heading)
                    )
                    if (
                        norm_product > 0.0
                        and float(np.dot(previous_heading, next_heading))
                        / norm_product < -0.5
                    ):
                        report.successive_hop_heading_opposition += 1
                    branch.heading = next_heading
                queue.append(branch)

            if queue and report.expansion_candidates >= candidate_budget:
                self._record_expansion_stop(
                    report, ExpansionStopReason.CANDIDATE_BUDGET.value, len(queue),
                )
        finally:
            report.expansion_wall_time = time.perf_counter() - started_at

        hop_counts = [branch.hops for branch in branches if branch.hops]
        if hop_counts:
            report.mean_hops_per_branch = float(np.mean(hop_counts))
            report.max_hops_reached = max(hop_counts)
        if used_scales:
            report.expansion_scale_mean = float(np.mean(used_scales))
            report.expansion_scale_max = float(np.max(used_scales))
        if report.expansion_candidates:
            report.frontier_found_per_candidate = (
                report.new_frontier / report.expansion_candidates
            )
        # Every selected branch has exactly one final, exclusive reason.
        if sum(report.stop_reasons.values()) != len(branches):
            raise AssertionError("Une branche d'expansion a disparu sans raison d'arrêt")
        self.last_expansion_seed_distances = [
            float(state.pose_distance) for state in attempted_seeds
        ]
        self.last_expansion_seed_depths = [
            int(state.generation_depth) for state in attempted_seeds
        ]
        self.last_generation_report = report
        return report

    def generate_candidates(
        self, seeds: list[CurriculumState] | None = None,
    ) -> tuple[list[CurriculumState], GenerationReport]:
        """Générateur legacy du bootstrap/diagnostic, hors policy.

        Son plafond ``candidates_per_update`` et ``walks_per_seed`` ne pilotent
        pas un update multi-hop. Celui-ci est borné exclusivement par
        ``expansion.max_candidates_per_update`` dans ``_expand_branches``.
        """
        if not seeds:
            seeds = self._expansion_seeds()
        else:
            seeds = list(seeds)
            self.last_expansion_seed_distances = [
                float(state.pose_distance) for state in seeds
            ]
            self.last_expansion_seed_depths = [
                int(state.generation_depth) for state in seeds
            ]
        candidates: list[CurriculumState] = []
        report = GenerationReport()
        requested = int(self.config["candidates_per_update"])
        walks_per_seed = int(self.walk["walks_per_seed"])
        max_steps = int(self.walk["max_steps"])
        action_scale = float(self.walk["action_scale"])

        for seed_state in seeds:
            for _ in range(walks_per_seed):
                self.env.restore_curriculum_state(
                    seed_state, reset_episode=False, restore_rng=True,
                )
                if self._proposal_mode() == "persistent":
                    report.persistent_attempts += 1
                    branch_heading = self._initial_branch_heading(
                        "uniform", None,
                    )
                    attempt_direction = self._persistent_attempt_direction(
                        branch_heading,
                    )
                    report.attempt_to_heading_deviations.append(float(
                        np.linalg.norm(attempt_direction - branch_heading)
                    ))
                else:
                    report.independent_attempts += 1
                    attempt_direction = None
                for _ in range(max_steps):
                    action = self._reverse_step_action(
                        action_scale, attempt_direction,
                    )
                    result = self.env.step_for_curriculum_generation(action)
                    report.generated += 1
                    if result.unsafe:
                        report.unsafe_rejected += 1
                        break
                    if result.success:
                        report.successful_excluded += 1
                        continue
                    candidate = self._assign_lineage_to_candidate(
                        result.state, seed_state,
                    )
                    if self._is_duplicate(candidate, candidates):
                        report.deduplicated_rejected += 1
                        continue
                    self.next_state_id += 1
                    candidates.append(candidate)
                    report.valid += 1
                    if len(candidates) >= requested:
                        self.last_generation_report = report
                        return candidates, report
        self.last_generation_report = report
        return candidates, report

    @contextmanager
    def _qualification_random_stream(self) -> Iterator[None]:
        """Isole l'aléa SAC de qualification de l'exploration d'entraînement."""
        training_cpu = torch.random.get_rng_state()
        training_cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        torch.random.set_rng_state(self.torch_rng_state)
        if torch.cuda.is_available():
            if self.torch_cuda_rng_states is None:
                torch.cuda.manual_seed_all(self.torch_seed)
            else:
                torch.cuda.set_rng_state_all(self.torch_cuda_rng_states)
        try:
            yield
        finally:
            self.torch_rng_state = torch.random.get_rng_state().clone()
            if torch.cuda.is_available():
                self.torch_cuda_rng_states = [
                    state.clone() for state in torch.cuda.get_rng_state_all()
                ]
            torch.random.set_rng_state(training_cpu)
            if training_cuda is not None:
                torch.cuda.set_rng_state_all(training_cuda)

    @staticmethod
    def _model_training_state(model: BaseAlgorithm) -> tuple[int, int | None]:
        replay = getattr(model, "replay_buffer", None)
        replay_size = replay.size() if replay is not None else None
        return int(model.num_timesteps), replay_size

    def qualify_candidates(
        self, model: BaseAlgorithm, candidates: list[CurriculumState],
    ) -> list[CurriculumState]:
        """Estime P(success) par actions SAC stochastiques, hors replay."""
        before = self._model_training_state(model)
        rollouts = int(self.config["evaluation_rollouts_per_candidate"])
        qualified: list[CurriculumState] = []
        with self._qualification_random_stream():
            for candidate in candidates:
                successes = 0
                for _ in range(rollouts):
                    observation, _ = self.env.restore_curriculum_state(
                        candidate, reset_episode=True, restore_rng=False,
                        reset_source="curriculum",
                    )
                    done = False
                    final_info: dict[str, Any] = {}
                    while not done:
                        action, _ = model.predict(
                            observation, deterministic=False,
                        )
                        observation, _, terminated, truncated, final_info = (
                            self.env.step(action)
                        )
                        done = terminated or truncated
                    successes += int(bool(final_info["safe_success"]))
                qualified.append(replace(
                    candidate, success_rate=successes / rollouts,
                ))
        after = self._model_training_state(model)
        if after != before:
            raise RuntimeError(
                "La qualification curriculum a modifié num_timesteps ou le replay buffer"
            )
        return qualified

    def _insert(self, states: list[CurriculumState]) -> None:
        low = float(self.config["success_rate_low"])
        high = float(self.config["success_rate_high"])
        for state in states:
            category = classify_success_rate(state.success_rate, low, high)
            self.pools[category].append(state)
        self._prune()

    def _remove_by_id(self, state_ids: set[int]) -> None:
        for name in POOL_NAMES:
            self.pools[name] = [
                state for state in self.pools[name]
                if state.state_id not in state_ids
            ]

    @staticmethod
    def _most_redundant_index(
        states: list[CurriculumState],
    ) -> tuple[int, float] | None:
        """Trouve un état redondant par profondeur tout en variant les parents.

        Les profondeurs extrêmes sont protégées. À densité égale, un groupe de
        siblings est élagué avant un état représentant seul sa branche.
        """
        if len(states) <= 2:
            return None
        ordered = sorted(
            enumerate(states),
            key=lambda item: (
                int(item[1].generation_depth), int(item[1].state_id),
            ),
        )
        candidates = []
        depth_range = max(
            int(ordered[-1][1].generation_depth)
            - int(ordered[0][1].generation_depth),
            1,
        )
        lineage_counts: dict[tuple[int, int | None], int] = {}
        parent_ids = {
            int(state.parent_id)
            for state in states if state.parent_id is not None
        }
        for state in states:
            key = (int(state.generation_depth), state.parent_id)
            lineage_counts[key] = lineage_counts.get(key, 0) + 1
        for position in range(1, len(ordered) - 1):
            left = int(ordered[position - 1][1].generation_depth)
            right = int(ordered[position + 1][1].generation_depth)
            original_index, state = ordered[position]
            branch_penalty = int(
                lineage_counts[(int(state.generation_depth), state.parent_id)]
                == 1
            )
            leaf_penalty = 3 * int(int(state.state_id) not in parent_ids)
            candidates.append((
                leaf_penalty + branch_penalty
                + max(0, right - left) / depth_range,
                int(state.state_id), original_index,
            ))
        span, _, index = min(candidates)
        return index, span

    def _prune(self) -> None:
        maximum = int(self.config["max_pool_size"])
        excess = self.total_pool_size - maximum
        while excess > 0:
            dense_candidates: list[tuple[float, int, str, int]] = []
            # En cas d'égalité, élaguer too_hard avant mastered, puis frontier.
            priority = {"too_hard": 0, "mastered": 1, "frontier": 2}
            for name in POOL_NAMES:
                candidate = self._most_redundant_index(self.pools[name])
                if candidate is not None:
                    index, density_span = candidate
                    dense_candidates.append(
                        (density_span, priority[name], name, index)
                    )
            if dense_candidates:
                _, _, name, index = min(dense_candidates)
                del self.pools[name][index]
                excess -= 1
                continue

            # Cas uniquement pathologique (limite plus petite que deux états
            # par pool) : sacrifier too_hard d'abord et garder une mémoire
            # mastered aussi longtemps que possible.
            fallback = next(
                (
                    name for name in (
                        "too_hard", "frontier", "mastered",
                    )
                    if self.pools[name]
                    and not (name == "mastered" and len(self.pools[name]) == 1)
                ),
                None,
            )
            if fallback is None:
                fallback = next(
                    (name for name in POOL_NAMES if self.pools[name]), None,
                )
            if fallback is None:
                break
            self.pools[fallback].pop()
            excess -= 1

    def bootstrap(self, model: BaseAlgorithm) -> GenerationReport:
        if self.total_pool_size:
            return self.last_generation_report
        candidates, report = self.generate_candidates([self.goal_seed])
        qualified = self.qualify_candidates(model, candidates)
        low = float(self.config["success_rate_low"])
        high = float(self.config["success_rate_high"])
        for state in qualified:
            category = classify_success_rate(state.success_rate, low, high)
            self.state_lifecycle[int(state.state_id)] = StateLifecycleStats(
                created_update=self.update_count,
                frontier_since_update=(
                    self.update_count if category == "frontier" else None
                ),
                consecutive_frontier_updates=(1 if category == "frontier" else 0),
            )
        self._insert(qualified)
        return report

    def select_too_hard_for_revalidation(
        self, *, rng: np.random.Generator | None = None,
    ) -> list[CurriculumState]:
        """Priorise la prochaine frontière topologique issue d'un mastered."""
        sample_count = self.config.get("revalidation", {}).get(
            "too_hard_samples_per_update", 12,
        )
        generator = self.rng if rng is None else rng
        return select_too_hard_by_lineage(
            self.pools["too_hard"], self.pools["mastered"],
            sample_count, generator,
        )

    @staticmethod
    def _state_selection_key(state: CurriculumState) -> tuple[str, int]:
        """Identifie un snapshot; les vrais états ont toujours un id positif."""
        state_id = int(state.state_id)
        return (
            ("state_id", state_id)
            if state_id >= 0 else ("object_id", id(state))
        )

    def _replace_revalidated_states(
        self, selected: list[CurriculumState],
        requalified: list[CurriculumState],
    ) -> None:
        """Applique atomiquement un reclassement logique sans duplication."""
        if len(requalified) != len(selected):
            raise RuntimeError(
                "La revalidation curriculum a retourné un nombre d'états "
                "incohérent"
            )
        selected_keys = [self._state_selection_key(state) for state in selected]
        requalified_keys = [
            self._state_selection_key(state) for state in requalified
        ]
        if requalified_keys != selected_keys:
            raise RuntimeError(
                "La revalidation curriculum a modifié l'identité des snapshots"
            )
        selected_lineage = [
            (state.state_id, state.parent_id, state.generation_depth)
            for state in selected
        ]
        requalified_lineage = [
            (state.state_id, state.parent_id, state.generation_depth)
            for state in requalified
        ]
        if requalified_lineage != selected_lineage:
            raise RuntimeError(
                "La revalidation curriculum a modifié le lineage des snapshots"
            )

        low = float(self.config["success_rate_low"])
        high = float(self.config["success_rate_high"])
        categories = [
            classify_success_rate(state.success_rate, low, high)
            for state in requalified
        ]
        key_set = set(selected_keys)
        updated = {
            name: [
                state for state in self.pools[name]
                if self._state_selection_key(state) not in key_set
            ]
            for name in POOL_NAMES
        }
        for state, category in zip(requalified, categories, strict=True):
            updated[category].append(state)
        self.pools = updated
        self._prune()

    def revalidate_existing(self, model: BaseAlgorithm) -> int:
        """Reclasse frontier, mastered sondés et too_hard proches du lineage."""
        started_at = time.perf_counter()
        frontier_selected = list(self.pools["frontier"])
        selected = list(frontier_selected)
        mastered = self.pools["mastered"]
        mastered_count = min(
            int(self.config["revalidation"]["mastered_samples_per_update"]),
            len(mastered),
        )
        if mastered_count:
            indices = self.rng.choice(
                len(mastered), size=mastered_count, replace=False,
            )
            selected.extend(
                mastered[int(index)] for index in np.atleast_1d(indices)
            )
        too_hard_selected = self.select_too_hard_for_revalidation()
        selected.extend(too_hard_selected)

        # Un ancien pickle mal formé ne doit pas multiplier un snapshot lors
        # de sa migration. Le premier exemplaire logique conserve sa place dans
        # l'ordre de qualification, puis tous ses doublons sont retirés.
        unique_selected: list[CurriculumState] = []
        seen_keys: set[tuple[str, int]] = set()
        for state in selected:
            key = self._state_selection_key(state)
            if key not in seen_keys:
                unique_selected.append(state)
                seen_keys.add(key)
        selected = unique_selected

        if selected:
            # Qualifier avant toute mutation : une erreur de rollout laisse les
            # pools intacts, même si les flux RNG de qualification ont avancé.
            requalified = self.qualify_candidates(model, selected)
            old_categories = {
                int(state.state_id): pool
                for pool in POOL_NAMES for state in self.pools[pool]
            }
            new_categories = {
                int(state.state_id): classify_success_rate(
                    state.success_rate,
                    float(self.config["success_rate_low"]),
                    float(self.config["success_rate_high"]),
                )
                for state in requalified
            }
            self._update_lifecycle_after_revalidation(
                old_categories, new_categories,
            )
            too_hard_keys = {
                self._state_selection_key(state)
                for state in too_hard_selected
            }
            too_hard_categories = [
                classify_success_rate(
                    state.success_rate,
                    float(self.config["success_rate_low"]),
                    float(self.config["success_rate_high"]),
                )
                for state in requalified
                if self._state_selection_key(state) in too_hard_keys
            ]
            self._replace_revalidated_states(selected, requalified)
            rollouts = int(self.config.get(
                "evaluation_rollouts_per_candidate", 1,
            ))
            self.last_revalidation_report = RevalidationReport(
                frontier_revalidated=len(frontier_selected),
                mastered_revalidated=mastered_count,
                too_hard_revalidated=len(too_hard_keys),
                too_hard_to_frontier=too_hard_categories.count("frontier"),
                too_hard_to_mastered=too_hard_categories.count("mastered"),
                too_hard_remained_hard=too_hard_categories.count("too_hard"),
                frontier_promoted_to_mastered=sum(
                    old_categories.get(state_id) == "frontier" and category == "mastered"
                    for state_id, category in new_categories.items()
                ),
                frontier_remained_frontier=sum(
                    old_categories.get(state_id) == "frontier" and category == "frontier"
                    for state_id, category in new_categories.items()
                ),
                frontier_demoted_to_too_hard=sum(
                    old_categories.get(state_id) == "frontier" and category == "too_hard"
                    for state_id, category in new_categories.items()
                ),
                frontier_rollouts=len(frontier_selected) * rollouts,
                mastered_rollouts=mastered_count * rollouts,
                too_hard_rollouts=len(too_hard_keys) * rollouts,
                wall_time=time.perf_counter() - started_at,
            )
        else:
            self.last_revalidation_report = RevalidationReport(
                wall_time=time.perf_counter() - started_at,
            )
        return len(selected)

    def _update_lifecycle_after_revalidation(
        self, old_categories: dict[int, str], new_categories: dict[int, str],
    ) -> None:
        if not hasattr(self, "state_lifecycle"):
            self.state_lifecycle = {}
        for state_id, category in new_categories.items():
            stats = self.state_lifecycle.setdefault(
                state_id, StateLifecycleStats(created_update=-1),
            )
            update_count = int(getattr(self, "update_count", 0)) + 1
            stats.last_revalidated_update = update_count
            stats.revalidation_count += 1
            if category == "frontier":
                if old_categories.get(state_id) == "frontier":
                    stats.consecutive_frontier_updates += 1
                else:
                    stats.frontier_since_update = update_count
                    stats.consecutive_frontier_updates = 1
            else:
                stats.frontier_since_update = None
                stats.consecutive_frontier_updates = 0

    def update(self, model: BaseAlgorithm) -> GenerationReport:
        """Revalide à sa cadence puis étend les branches en multi-hop.

        ``update_count`` vaut zéro pour le premier update : celui-ci revalide
        toujours, puis une fréquence N revalide les updates 1, 1+N, 1+2N, ...
        vus par l'utilisateur.
        """
        revalidation = self.config.get("revalidation", {})
        frequency = int(revalidation.get("every_n_curriculum_updates", 1))
        if self.update_count % frequency == 0:
            self.revalidate_existing(model)
        else:
            self.last_revalidation_report = RevalidationReport()
        report = self._expand_branches(model)
        self.update_count += 1
        self.next_update_timesteps += int(self.config["update_interval_timesteps"])
        return report

    @staticmethod
    def _ensure_lineage(
        pools: dict[str, list[CurriculumState]], next_state_id: int,
    ) -> int:
        """Migre les anciens snapshots en racines legacy, sans inférer la géométrie.

        Les IDs entiers valides restent inchangés. Seuls les IDs absents,
        invalides ou dupliqués sont réattribués de façon déterministe dans
        l'ordre des pools. Un état dépourvu de lineage devient une racine de
        depth 0; ses futures expansions construiront naturellement sa branche.
        """
        states = [state for name in POOL_NAMES for state in pools[name]]
        existing_ids = [
            int(vars(state)["state_id"])
            for state in states
            if "state_id" in vars(state)
            and not isinstance(vars(state)["state_id"], bool)
            and isinstance(vars(state)["state_id"], (int, np.integer))
            and int(vars(state)["state_id"]) >= 0
        ]
        cursor = max(
            1, int(next_state_id),
            max(existing_ids, default=0) + 1,
        )
        seen: set[int] = set()
        for state in states:
            fields = vars(state)
            raw_state_id = fields.get("state_id")
            valid_state_id = (
                not isinstance(raw_state_id, bool)
                and isinstance(raw_state_id, (int, np.integer))
                and int(raw_state_id) >= 0
                and int(raw_state_id) not in seen
            )
            if valid_state_id:
                state_id = int(raw_state_id)
            else:
                state_id = cursor
                cursor += 1
            state.state_id = state_id
            seen.add(state_id)

            has_lineage = (
                "parent_id" in fields and "generation_depth" in fields
            )
            if not has_lineage:
                state.parent_id = None
                state.generation_depth = 0
                continue
            parent_id = fields["parent_id"]
            if parent_id is not None:
                if (isinstance(parent_id, bool)
                        or not isinstance(parent_id, (int, np.integer))
                        or int(parent_id) < 0):
                    raise ValueError("parent_id curriculum invalide")
                parent_id = int(parent_id)
                if parent_id == state_id:
                    raise ValueError("Un état curriculum ne peut pas être son parent")
            depth = fields["generation_depth"]
            if (isinstance(depth, bool)
                    or not isinstance(depth, (int, np.integer))
                    or int(depth) < 0):
                raise ValueError("generation_depth curriculum invalide")
            state.parent_id = parent_id
            state.generation_depth = int(depth)
        return max(cursor, max(seen, default=0) + 1)

    @staticmethod
    def _curriculum_configs_compatible(
        saved: dict[str, Any], current: dict[str, Any],
    ) -> bool:
        """Autorise uniquement les paramètres corrigés par cette migration."""
        saved_core = deepcopy(saved)
        current_core = deepcopy(current)
        for config in (saved_core, current_core):
            config.pop("evaluation_rollouts_per_candidate", None)
            config.pop("start_sampling", None)
            config.pop("diagnostics", None)
            config.pop("revalidation", None)
            config.pop("curriculum_reset_probability", None)
            walk = config.get("reverse_random_walk")
            if isinstance(walk, dict):
                walk.pop("min_pose_distance_increase", None)
                walk.pop("proposal", None)
                walk.pop("proposal_mode", None)
                walk.pop("persistent_proposal", None)
                if not walk:
                    config.pop("reverse_random_walk", None)
            # Toute la section expansion décrit une stratégie de découverte,
            # pas la structure des snapshots déjà sérialisés. Elle peut donc
            # évoluer lors d'une reprise (y compris depuis un YAML sans section).
            config.pop("expansion", None)
        return saved_core == current_core

    def state_dict(
        self, worker_rng_states: list[dict[str, Any]] | None = None,
        training_timesteps: int | None = None,
    ) -> dict[str, Any]:
        self.next_state_id = self._ensure_lineage(
            self.pools, self.next_state_id,
        )
        return {
            "version": self.STATE_VERSION,
            "task_config_sha256": self._task_config_sha256(),
            "curriculum_config": deepcopy(self.config),
            "goal_seed": self.goal_seed,
            "pools": self.pools,
            "numpy_rng_state": deepcopy(self.rng.bit_generator.state),
            "qualification_env_rng_state": deepcopy(
                self.env.np_random.bit_generator.state
            ),
            "torch_rng_state": self.torch_rng_state.clone(),
            "torch_seed": self.torch_seed,
            "torch_cuda_rng_states": (
                [state.clone() for state in self.torch_cuda_rng_states]
                if self.torch_cuda_rng_states is not None else None
            ),
            "worker_rng_states": deepcopy(
                worker_rng_states if worker_rng_states is not None
                else self.worker_rng_states
            ),
            # La reprise SAC utilise aussi ces flux globaux (policy/replay).
            "training_python_rng_state": random.getstate(),
            "training_numpy_rng_state": np.random.get_state(),
            "training_torch_rng_state": torch.random.get_rng_state().clone(),
            "training_torch_cuda_rng_states": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None
            ),
            "next_state_id": self.next_state_id,
            "update_count": self.update_count,
            "next_update_timesteps": self.next_update_timesteps,
            "training_timesteps": (
                None if training_timesteps is None else int(training_timesteps)
            ),
            "state_lifecycle": deepcopy(self.state_lifecycle),
            "sampling_episode_length_ema": deepcopy(
                self.sampling_episode_length_ema
            ),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        version = int(payload.get("version", 1))
        if version not in {1, 2, 3, self.STATE_VERSION}:
            raise ValueError(
                "Version de curriculum_state.pkl incompatible: "
                f"{payload.get('version')!r}"
            )
        saved_config = payload.get("curriculum_config")
        saved_hash = payload.get("task_config_sha256")
        if version in {2, 3, self.STATE_VERSION}:
            compatible_task = saved_hash == self._task_config_sha256()
        elif saved_hash is not None and saved_config is not None:
            compatible_task = saved_hash == self._task_config_sha256(
                saved_config, legacy=True,
            )
        else:
            compatible_task = True
            warnings.warn(
                "Migration d'un curriculum_state.pkl V1 ancien sans empreinte "
                "physique vérifiable : les pools et états RNG présents sont "
                "conservés, mais la reprise n'est pas bit-exacte.",
                RuntimeWarning,
            )
        if not compatible_task:
            raise ValueError(
                "curriculum_state.pkl provient d'une tâche/configuration physique "
                "incompatible"
            )
        if (saved_config is not None
                and not self._curriculum_configs_compatible(
                    saved_config, self.config,
                )):
            raise ValueError(
                "curriculum_state.pkl contient des paramètres structurels "
                "incompatibles avec la configuration curriculum courante"
            )
        self.goal_seed = payload["goal_seed"]
        # Le goal est la racine logique réservée : il n'appartient à aucun pool.
        self.goal_seed.state_id = -1
        self.goal_seed.parent_id = None
        self.goal_seed.generation_depth = 0
        pools = payload["pools"]
        if not isinstance(pools, dict) or set(pools) != set(POOL_NAMES):
            raise ValueError("curriculum_state.pkl contient des pools invalides")
        # Le nom sérialisé reste ``mastered``; il devient directement la
        # mémoire historique, sans conversion ni perte d'état.
        self.pools = {name: list(pools[name]) for name in POOL_NAMES}
        self.next_state_id = self._ensure_lineage(
            self.pools, int(payload.get("next_state_id", 1)),
        )
        self.rng.bit_generator.state = payload["numpy_rng_state"]
        if "qualification_env_rng_state" in payload:
            self.env.np_random.bit_generator.state = deepcopy(
                payload["qualification_env_rng_state"]
            )
        self.torch_rng_state = payload["torch_rng_state"].clone()
        self.torch_seed = int(payload.get("torch_seed", self.torch_seed))
        self.torch_cuda_rng_states = payload.get("torch_cuda_rng_states")
        self.worker_rng_states = payload.get("worker_rng_states")
        self.update_count = int(payload["update_count"])
        self.next_update_timesteps = int(payload["next_update_timesteps"])
        self.loaded_training_timesteps = payload.get("training_timesteps")
        saved_episode_length_ema = payload.get(
            "sampling_episode_length_ema", {}
        )
        self.sampling_episode_length_ema = {}
        for name in SAMPLING_SOURCE_NAMES:
            value = float(saved_episode_length_ema.get(name, 1.0))
            self.sampling_episode_length_ema[name] = (
                value if np.isfinite(value) and value > 0.0 else 1.0
            )
        saved_lifecycle = payload.get("state_lifecycle", {})
        self.state_lifecycle = {}
        for state_id, stats in saved_lifecycle.items():
            migrated = (
                stats if isinstance(stats, StateLifecycleStats)
                else StateLifecycleStats(**stats)
            )
            for name, default in (
                ("nearest_ancestor_position_m", None),
                ("nearest_ancestor_rotation_deg", None),
                ("near_ancestor_return", False),
            ):
                if not hasattr(migrated, name):
                    setattr(migrated, name, default)
            self.state_lifecycle[int(state_id)] = migrated
        # Legacy states have unknown creation history. ``-1`` explicitly means
        # unknown; current frontier membership starts its observable age now.
        for pool, states in self.pools.items():
            for state in states:
                state_id = int(state.state_id)
                if state_id not in self.state_lifecycle:
                    self.state_lifecycle[state_id] = StateLifecycleStats(
                        created_update=-1,
                        frontier_since_update=(
                            self.update_count if pool == "frontier" else None
                        ),
                    )
        self.proposal_memory = {}
        self._active_proposal_direction = None
        self._active_proposal_kind = "uniform"
        if "training_python_rng_state" in payload:
            random.setstate(payload["training_python_rng_state"])
        if "training_numpy_rng_state" in payload:
            np.random.set_state(payload["training_numpy_rng_state"])
        if "training_torch_rng_state" in payload:
            torch.random.set_rng_state(payload["training_torch_rng_state"])
        training_cuda = payload.get("training_torch_cuda_rng_states")
        if training_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(training_cuda)

    def save(
        self, path: str | Path,
        worker_rng_states: list[dict[str, Any]] | None = None,
        training_timesteps: int | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(
                self.state_dict(worker_rng_states, training_timesteps), stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary.replace(path)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        self.load_state_dict(payload)
