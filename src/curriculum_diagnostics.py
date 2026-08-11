"""Small, append-only diagnostics for periodic curriculum updates."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.curriculum import SAMPLING_SOURCE_NAMES


def _stat(values: Any, statistic: str) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(getattr(np, statistic)(array)) if array.size else 0.0


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _finite(value: Any) -> float:
    number = float(value)
    return number if np.isfinite(number) else 0.0


@dataclass(frozen=True)
class ExpansionDiagnostics:
    values: dict[str, int | float]

    @classmethod
    def build(
        cls, manager: Any, training_timesteps: int,
        sampling_targets_used: Any, sampling_observed: dict[str, float],
        sampling_targets_next: Any,
        frontier_reset_counts: dict[int, int] | None = None,
        sampling_transition_observed: dict[str, float] | None = None,
        sampling_success_rates: dict[str, float] | None = None,
        used_start_distances: dict[str, list[float]] | None = None,
        sampling_effective_reset: Any | None = None,
        sampling_episode_length_ema: dict[str, float] | None = None,
    ) -> "ExpansionDiagnostics":
        report = manager.last_generation_report
        revalidation = manager.last_revalidation_report
        pools = manager.pools
        reset_pools = (
            manager.training_reset_pools()
            if hasattr(manager, "training_reset_pools")
            else {}
        )
        depths = {
            name: [int(state.generation_depth) for state in pools[name]]
            for name in ("mastered", "frontier", "too_hard")
        }
        stops = report.stop_reasons
        lifecycle = getattr(manager, "state_lifecycle", {})
        frontier_lifecycle = [
            lifecycle.get(int(state.state_id)) for state in pools["frontier"]
        ]
        frontier_lifecycle = [item for item in frontier_lifecycle if item is not None]
        frontier_ages = [
            max(0, int(manager.update_count) - int(item.frontier_since_update))
            for item in frontier_lifecycle if item.frontier_since_update is not None
        ]
        attempts = {
            kind: int(getattr(report, f"proposal_{kind}_attempts"))
            for kind in ("uniform", "guided")
        }
        reset_counts = list((frontier_reset_counts or {}).values())
        values: dict[str, int | float] = {
            "training_timesteps": int(training_timesteps),
            "curriculum_update_index": int(manager.update_count),
            "branches": report.expansion_branches,
            "hops": report.expansion_hops,
            "attempts": report.expansion_attempts,
            "raw_candidates": report.raw_candidates_generated,
            "valid_candidates": report.valid_candidates,
            "unique_candidates": report.nonduplicate_candidates,
            "qualified_candidates": report.qualified_candidates,
            "safe_prefix_candidates": report.safe_prefix_candidates,
            "full_walk_candidates": report.full_walk_candidates,
            "duplicates": report.attempt_duplicate,
            "no_candidate": report.attempt_no_candidate,
            "attempts_per_hop_mean": _stat(report.attempts_per_hop, "mean"),
            "attempts_per_hop_max": _stat(report.attempts_per_hop, "max"),
            "safe_prefix_steps_mean": _stat(
                getattr(report, "safe_prefix_steps", []), "mean",
            ),
            "safe_prefix_steps_max": _stat(
                getattr(report, "safe_prefix_steps", []), "max",
            ),
            "reverse_candidate_parent_delta_position_mm_mean": _stat(
                getattr(report, "raw_parent_translation_mm", []), "mean",
            ),
            "reverse_candidate_parent_delta_position_mm_max": _stat(
                getattr(report, "raw_parent_translation_mm", []), "max",
            ),
            "reverse_candidate_parent_delta_rotation_deg_mean": _stat(
                getattr(report, "raw_parent_rotation_deg", []), "mean",
            ),
            "reverse_candidate_parent_delta_rotation_deg_max": _stat(
                getattr(report, "raw_parent_rotation_deg", []), "max",
            ),
            "persistent_attempts": int(getattr(report, "persistent_attempts", 0)),
            "independent_attempts": int(getattr(report, "independent_attempts", 0)),
            "branch_heading_changes_mean": _stat(
                getattr(report, "branch_heading_changes", []), "mean",
            ),
            "branch_heading_changes_max": _stat(
                getattr(report, "branch_heading_changes", []), "max",
            ),
            "attempt_to_heading_deviation_mean": _stat(
                getattr(report, "attempt_to_heading_deviations", []), "mean",
            ),
            "attempt_to_heading_deviation_max": _stat(
                getattr(report, "attempt_to_heading_deviations", []), "max",
            ),
            "successive_hop_heading_opposition": int(getattr(
                report, "successive_hop_heading_opposition", 0,
            )),
            "guided_memory_insertions": int(getattr(
                report, "guided_memory_insertions", 0,
            )),
            "guided_memory_rejected_duplicates": int(getattr(
                report, "guided_memory_rejected_duplicates", 0,
            )),
            **{
                f"stop_{reason}": int(stops.get(reason, 0))
                for reason in (
                    "frontier", "too_hard", "attempt_budget", "max_hops",
                    "candidate_budget", "workspace", "snapshot_invalid",
                    "generation_failed",
                )
            },
            "reverse_rejected_force_mean": _stat(report.rejected_force_max, "mean"),
            "reverse_rejected_force_max": _stat(report.rejected_force_max, "max"),
            "reverse_rejected_torque_mean": _stat(report.rejected_torque_max, "mean"),
            "reverse_rejected_torque_max": _stat(report.rejected_torque_max, "max"),
            "candidate_final_force_mean": _stat(report.candidate_final_force, "mean"),
            "candidate_final_force_max": _stat(report.candidate_final_force, "max"),
            "candidate_final_torque_mean": _stat(report.candidate_final_torque, "mean"),
            "candidate_final_torque_max": _stat(report.candidate_final_torque, "max"),
            "new_mastered": report.new_mastered,
            "new_frontier": report.new_frontier,
            "new_too_hard": report.new_too_hard,
            "mastered_pool_size": len(pools["mastered"]),
            "frontier_pool_size": len(pools["frontier"]),
            "too_hard_pool_size": len(pools["too_hard"]),
            "mastered_boundary_count": len(manager.mastered_boundary_states()),
            "mastered_boundary_pool_size": len(
                reset_pools.get("mastered_boundary", [])
            ),
            "too_hard_near_pool_size": len(
                reset_pools.get("too_hard_near", [])
            ),
            **{
                f"{name}_position_max": _stat(
                    [state.position_error for state in pools[name]], "max",
                )
                for name in ("mastered", "frontier", "too_hard")
            },
            "max_generation_depth": max(
                (depth for values in depths.values() for depth in values), default=0,
            ),
            **{
                f"{name}_max_depth": max(values, default=0)
                for name, values in depths.items()
            },
            "expansion_wall_time": float(report.expansion_wall_time),
            "revalidation_wall_time": float(revalidation.wall_time),
            "frontier_resets_per_state_mean": _stat(reset_counts, "mean"),
            "frontier_resets_per_state_max": _stat(reset_counts, "max"),
            "frontier_unique_states_sampled": len(reset_counts),
            "new_states_near_ancestor": getattr(
                report, "new_states_near_ancestor", 0,
            ),
            "new_states_near_ancestor_fraction": _rate(
                getattr(report, "new_states_near_ancestor", 0),
                report.qualified_candidates,
            ),
            "nearest_ancestor_position_mm_mean": _stat(
                getattr(report, "nearest_ancestor_position_mm", []), "mean",
            ),
            "nearest_ancestor_rotation_deg_mean": _stat(
                getattr(report, "nearest_ancestor_rotation_deg", []), "mean",
            ),
            "frontier_age_updates_mean": _stat(frontier_ages, "mean"),
            "frontier_age_updates_max": _stat(frontier_ages, "max"),
            "frontier_revalidation_count_mean": _stat(
                [item.revalidation_count for item in frontier_lifecycle], "mean",
            ),
            "frontier_revalidation_count_max": _stat(
                [item.revalidation_count for item in frontier_lifecycle], "max",
            ),
            "frontier_consecutive_updates_mean": _stat(
                [item.consecutive_frontier_updates for item in frontier_lifecycle],
                "mean",
            ),
            "frontier_consecutive_updates_max": _stat(
                [item.consecutive_frontier_updates for item in frontier_lifecycle],
                "max",
            ),
        }
        for name in SAMPLING_SOURCE_NAMES:
            transition_target = float(
                getattr(sampling_targets_used, name, 0.0)
            )
            transition_observed = float(
                (sampling_transition_observed or {}).get(name, 0.0)
            )
            values[f"sampling_target_used_{name}"] = float(
                getattr(sampling_targets_used, name, 0.0)
            )
            values[f"sampling_observed_{name}"] = float(
                sampling_observed.get(name, 0.0)
            )
            values[f"sampling_target_next_{name}"] = float(
                getattr(sampling_targets_next, name, 0.0)
            )
            values[f"sampling_transition_observed_{name}"] = float(
                transition_observed
            )
            values[f"sampling_transition_target_{name}"] = transition_target
            values[f"sampling_effective_reset_{name}"] = float(getattr(
                sampling_effective_reset or sampling_targets_used, name, 0.0,
            ))
            values[f"sampling_episode_length_ema_{name}"] = float(
                (sampling_episode_length_ema or {}).get(name, 1.0)
            )
            values[f"success_rate_{name}"] = float(
                (sampling_success_rates or {}).get(name, 0.0)
            )
            distances = (used_start_distances or {}).get(name, [])
            values[f"used_start_distance_{name}_mean"] = _stat(
                distances, "mean",
            )
            values[f"used_start_distance_{name}_max"] = _stat(
                distances, "max",
            )
        values["sampling_transition_target_l1_error"] = float(sum(
            abs(
                float((sampling_transition_observed or {}).get(name, 0.0))
                - float(getattr(sampling_targets_used, name, 0.0))
            )
            for name in SAMPLING_SOURCE_NAMES
        ))
        values["sampling_missing_frontier_budget"] = float(getattr(
            sampling_targets_used, "missing_frontier_budget", 0.0,
        ))
        values["sampling_fallback_budget_used"] = float(getattr(
            sampling_targets_used, "fallback_budget_used", 0.0,
        ))
        for kind in ("uniform", "guided"):
            candidates = int(getattr(report, f"proposal_{kind}_candidates"))
            unique = int(getattr(report, f"proposal_{kind}_unique"))
            for suffix in (
                "attempts", "candidates", "unique", "safe_prefix",
                "attempt_budget_failures",
            ):
                values[f"proposal_{kind}_{suffix}"] = int(
                    getattr(report, f"proposal_{kind}_{suffix}")
                )
            values[f"{kind}_unique_per_attempt"] = _rate(unique, attempts[kind])
            values[f"{kind}_candidate_per_attempt"] = _rate(
                candidates, attempts[kind],
            )
        for name in (
            "frontier_promoted_to_mastered", "frontier_remained_frontier",
            "frontier_demoted_to_too_hard",
        ):
            values[name] = int(getattr(revalidation, name, 0))
        return cls(values)


def curriculum_state_rows(
    manager: Any, training_timesteps: int,
) -> list[dict[str, int | float | str]]:
    rows = []
    lifecycle = getattr(manager, "state_lifecycle", {})
    for pool in ("mastered", "frontier", "too_hard"):
        for state in manager.pools[pool]:
            stats = lifecycle.get(int(state.state_id))
            rows.append({
                "training_timesteps": int(training_timesteps),
                "update_index": int(manager.update_count),
                "state_id": int(state.state_id),
                "parent_id": "" if state.parent_id is None else int(state.parent_id),
                "generation_depth": int(state.generation_depth),
                "pool": pool,
                "success_rate": _finite(state.success_rate),
                "created_update": -1 if stats is None else stats.created_update,
                "last_revalidated_update": -1 if stats is None else stats.last_revalidated_update,
                "revalidation_count": 0 if stats is None else stats.revalidation_count,
                "frontier_since_update": (
                    "" if stats is None or stats.frontier_since_update is None
                    else stats.frontier_since_update
                ),
                "consecutive_frontier_updates": (
                    0 if stats is None else stats.consecutive_frontier_updates
                ),
                "nearest_ancestor_position_m": (
                    "" if stats is None or getattr(
                        stats, "nearest_ancestor_position_m", None,
                    ) is None else stats.nearest_ancestor_position_m
                ),
                "nearest_ancestor_rotation_deg": (
                    "" if stats is None or getattr(
                        stats, "nearest_ancestor_rotation_deg", None,
                    ) is None else stats.nearest_ancestor_rotation_deg
                ),
                "near_ancestor_return": int(
                    False if stats is None else getattr(
                        stats, "near_ancestor_return", False,
                    )
                ),
                "position_error": _finite(state.position_error),
                "rotation_error": _finite(state.rotation_error),
                "pose_distance": _finite(state.pose_distance),
            })
    return rows


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()


def write_curriculum_diagnostics(
    output: Path, diagnostics: ExpansionDiagnostics,
    state_rows: list[dict[str, Any]],
) -> None:
    append_csv(output / "curriculum_expansion.csv", [diagnostics.values])
    append_csv(output / "curriculum_states.csv", state_rows)
