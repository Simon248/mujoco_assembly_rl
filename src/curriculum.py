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
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import time
import warnings
from typing import Any, Iterator, TYPE_CHECKING

import numpy as np
import torch

from src.transforms import quat_to_rotvec, relative

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from src.assembly_env import TenonMortaiseEnv


POOL_NAMES = ("too_hard", "frontier", "mastered")
RESET_SOURCES = (
    "true_start", "curriculum_frontier", "curriculum_historical",
)
EXPANSION_DEFAULTS: dict[str, int | float] = {
    "max_hops_per_seed": 4,
    "max_candidates_per_update": 24,
    "initial_scale": 1.0,
    "scale_up_factor": 1.25,
    "scale_down_factor": 0.7,
    "min_scale": 0.5,
    "max_scale": 3.0,
}
EXPANSION_STRATEGY_KEYS = frozenset(EXPANSION_DEFAULTS)


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
    requested: str = "auto",
    historical_bin_groups: list[list[CurriculumState]] | None = None,
) -> CurriculumResetSelection:
    """Tire un vrai start, un frontier ou un historique avec fallbacks sûrs.

    Les probabilités globales ne sont jamais codées en dur : avec la config
    V21, elles découlent de ``0.80 * (0.625, 0.375)``.
    """
    allowed = {"auto", "curriculum", *RESET_SOURCES}
    if requested not in allowed:
        raise ValueError(
            "options.reset_source doit être 'auto', 'curriculum', "
            "'true_start', 'curriculum_frontier' ou 'curriculum_historical'"
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
    if requested == "auto" and (
        not frontier and not historical
        or rng.random() >= curriculum_probability
    ):
        return CurriculumResetSelection("true_start", None)

    if requested == "curriculum_frontier":
        preferred = "frontier"
    elif requested == "curriculum_historical":
        preferred = "historical"
    else:
        preferred = (
            "frontier"
            if rng.random() * fraction_total < frontier_fraction
            else "historical"
        )

    # Un pool souhaité vide bascule sur l'autre; aucun pool revient au vrai start.
    if preferred == "frontier" and not frontier:
        preferred = "historical"
    elif preferred == "historical" and not historical:
        preferred = "frontier"
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
    # le nombre historique de pas physiques de reverse walk, tandis que
    # ``expansion_hops`` compte les tentatives de produire au plus un état.
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


@dataclass(frozen=True)
class RevalidationReport:
    """Bilan d'une revalidation, sans effet sur le format des pools."""

    frontier_revalidated: int = 0
    mastered_revalidated: int = 0
    too_hard_revalidated: int = 0
    too_hard_to_frontier: int = 0
    too_hard_to_mastered: int = 0
    too_hard_remained_hard: int = 0
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
    mastered_ids = {int(state.state_id) for state in mastered}
    preferred = [
        state for state in too_hard
        if state.parent_id is not None and int(state.parent_id) in mastered_ids
    ]
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

    STATE_VERSION = 2

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
        """Retourne les deux seules mémoires autorisées comme starts RL."""
        return {
            "frontier": list(self.pools["frontier"]),
            "historical": list(self.pools["mastered"]),
        }

    def training_states(self) -> list[CurriculumState]:
        """Compatibilité API : union des starts, sans aucun ``too_hard``."""
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
        position_tolerance = float(self.deduplication["position_tolerance"])
        rotation_tolerance = np.deg2rad(
            float(self.deduplication["rotation_tolerance_deg"])
        )
        for existing in self.all_states() + additional:
            position_delta = float(np.linalg.norm(
                candidate.task_position - existing.task_position
            ))
            rotation_delta = float(np.linalg.norm(quat_to_rotvec(relative(
                (np.zeros(3), existing.task_quaternion),
                (np.zeros(3), candidate.task_quaternion),
            )[1])))
            if (position_delta < position_tolerance
                    and rotation_delta < rotation_tolerance):
                return True
        return False

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
        Une violation unsafe/invalide annule tout le hop, même si un état
        intermédiaire avait été sûr. Les sous-steps encore successful sont
        traversés sans devenir des candidats. Un hop correspond à un seul walk;
        ``walks_per_seed`` appartient uniquement au générateur legacy utilisé
        par le bootstrap et les diagnostics.
        """
        self.env.restore_curriculum_state(
            seed, reset_episode=False, restore_rng=True,
        )
        # L'action MuJoCo est bornée à [-1, 1]. Borner l'amplitude avant le
        # tirage évite qu'un clipping ultérieur crée artificiellement une masse
        # de probabilité exactement aux deux bornes.
        amplitude = min(
            float(self.walk["action_scale"]) * float(expansion_scale), 1.0,
        )
        last_candidate: CurriculumState | None = None
        for _ in range(int(self.walk["max_steps"])):
            action = self.rng.uniform(-amplitude, amplitude, size=6)
            result = self.env.step_for_curriculum_generation(action)
            report.generated += 1
            if result.unsafe:
                report.unsafe_rejected += 1
                return None, "unsafe"
            if not self._candidate_snapshot_is_valid(result.state):
                report.invalid_rejected += 1
                return None, "invalid"
            if result.success:
                report.successful_excluded += 1
                continue
            last_candidate = result.state
        if last_candidate is None:
            return None, "no_candidate"
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

        branches = [
            _ExpansionBranch(seed, initial_scale) for seed in selected
        ]
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

                snapshot, stop_reason = self._generate_hop_snapshot(
                    branch.current, branch.scale, report,
                )
                if snapshot is None:
                    self._record_expansion_stop(
                        report, stop_reason or "invalid",
                    )
                    continue

                candidate = self._assign_lineage_to_candidate(
                    snapshot, branch.current,
                )
                if self._is_duplicate(candidate, known_during_update):
                    report.deduplicated_rejected += 1
                    self._record_expansion_stop(report, "duplicate")
                    continue

                # L'identifiant n'est consommé que par un snapshot réellement
                # nouveau. Le budget est encore disponible car il est testé en
                # tête de boucle, avant cet unique appel de qualification.
                self.next_state_id += 1
                report.valid += 1
                qualified = self.qualify_candidates(model, [candidate])
                if len(qualified) != 1:
                    raise RuntimeError(
                        "La qualification d'un hop doit retourner exactement un état"
                    )
                state = qualified[0]
                category = classify_success_rate(
                    state.success_rate, low, high,
                )
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
                    self._record_expansion_stop(report, "max_hops")
                    continue
                branch.current = state
                branch.scale = self._next_expansion_scale(
                    branch.scale, category,
                )
                queue.append(branch)

            if queue and report.expansion_candidates >= candidate_budget:
                self._record_expansion_stop(report, "global_budget", len(queue))
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
                for _ in range(max_steps):
                    action = self.rng.uniform(-action_scale, action_scale, size=6)
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
        self._insert(self.qualify_candidates(model, candidates))
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
            config.pop("revalidation", None)
            config.pop("curriculum_reset_probability", None)
            walk = config.get("reverse_random_walk")
            if isinstance(walk, dict):
                walk.pop("min_pose_distance_increase", None)
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
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        version = int(payload.get("version", 1))
        if version not in {1, self.STATE_VERSION}:
            raise ValueError(
                "Version de curriculum_state.pkl incompatible: "
                f"{payload.get('version')!r}"
            )
        saved_config = payload.get("curriculum_config")
        saved_hash = payload.get("task_config_sha256")
        if version == self.STATE_VERSION:
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
