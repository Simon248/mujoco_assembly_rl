"""Smoke test physique du RCG, sans création ni entraînement d'un modèle SAC."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.assembly_env import TenonMortaiseEnv
from src.config import load_config
from src.curriculum import CurriculumState, ReverseCurriculumManager
from src.curriculum import (
    historical_quantile_bins,
    mastered_boundary_states,
    select_training_start,
)


def _physical_signature(env: TenonMortaiseEnv) -> dict[str, np.ndarray]:
    mocap = env.model.body_mocapid[env.target_mocap]
    task = env._pose(env.mobile_body)
    return {
        "mj_state": env._integration_state(),
        "grasp_actual_position": env.data.site_xpos[env.grasp_site].copy(),
        "grasp_actual_quaternion": env._site_quat(),
        "grasp_target_position": env.data.mocap_pos[mocap].copy(),
        "grasp_target_quaternion": env.data.mocap_quat[mocap].copy(),
        "task_position": task[0],
        "task_quaternion": task[1],
        "admittance_offset": env.admittance.offset.copy(),
        "admittance_velocity": env.admittance.velocity.copy(),
    }


def _check_restoration(env: TenonMortaiseEnv, state: CurriculumState) -> bool:
    observation_before, _ = env.restore_curriculum_state(
        state, reset_episode=True, restore_rng=True,
    )
    signature_before = _physical_signature(env)
    env.step_for_curriculum_generation(np.full(6, 0.25, dtype=float))
    observation_after, _ = env.restore_curriculum_state(
        state, reset_episode=True, restore_rng=True,
    )
    signature_after = _physical_signature(env)
    return bool(
        np.allclose(observation_before, observation_after, rtol=0.0, atol=1e-10)
        and all(
            np.allclose(signature_before[key], signature_after[key],
                        rtol=0.0, atol=1e-10)
            for key in signature_before
        )
    )


def _distance_statistics(states: list[CurriculumState]) -> dict[str, float | None]:
    values = np.asarray([state.pose_distance for state in states], dtype=float)
    if not values.size:
        return {
            key: None for key in ("min", "q25", "median", "q75", "max")
        }
    return {
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def _depth_statistics(
    states: list[CurriculumState],
) -> dict[str, int | float | None]:
    values = [int(state.generation_depth) for state in states]
    if not values:
        return {key: None for key in ("min", "median", "max")}
    return {
        "min": min(values),
        "median": float(np.median(values)),
        "max": max(values),
    }


def _state_label(state_id: Any) -> str:
    value = str(state_id)
    return value if value.startswith("state_") else f"state_{value}"


def _deepest_lineages(
    pools: dict[str, list[CurriculumState]], *, limit: int = 5,
) -> list[dict[str, Any]]:
    """Résume quelques feuilles profondes sans construire de graphe persistant."""
    states = [state for pool in pools.values() for state in pool]
    states_by_id = {state.state_id: state for state in states}
    classification_by_id = {
        state.state_id: name for name, pool in pools.items() for state in pool
    }
    parent_ids = {
        state.parent_id for state in states if state.parent_id is not None
    }
    leaves = [state for state in states if state.state_id not in parent_ids]
    leaves.sort(
        key=lambda state: (-int(state.generation_depth), int(state.state_id)),
    )
    result: list[dict[str, Any]] = []
    for leaf in leaves[:limit]:
        path: list[CurriculumState] = []
        current: CurriculumState | None = leaf
        visited: set[Any] = set()
        missing_parent: Any | None = None
        cycle = False
        while current is not None:
            if current.state_id in visited:
                cycle = True
                break
            visited.add(current.state_id)
            path.append(current)
            if current.parent_id is None:
                break
            parent = states_by_id.get(current.parent_id)
            if parent is None:
                missing_parent = current.parent_id
                break
            current = parent
        path.reverse()
        if cycle:
            root_label = "cycle"
        elif missing_parent is not None:
            root_label = f"missing_{_state_label(missing_parent)}"
        elif path and int(path[0].generation_depth) == 0:
            root_label = "legacy_root"
        else:
            root_label = "goal"
        result.append({
            "path": " → ".join([
                root_label,
                *[_state_label(state.state_id) for state in path],
            ]),
            "depth": int(leaf.generation_depth),
            "classification": classification_by_id[leaf.state_id],
            "position_error": float(leaf.position_error),
            "rotation_error": float(leaf.rotation_error),
        })
    return result


def lineage_diagnostic(
    pools: dict[str, list[CurriculumState]],
    *,
    mastered_boundary: list[CurriculumState] | None = None,
) -> dict[str, Any]:
    """Décrit la progression topologique, indépendamment de la géométrie."""
    states = [state for pool in pools.values() for state in pool]
    boundary = (
        mastered_boundary_states(pools["mastered"])
        if mastered_boundary is None else list(mastered_boundary)
    )

    def pool_summary(pool: list[CurriculumState]) -> dict[str, Any]:
        return {
            "count": len(pool),
            "depth": _depth_statistics(pool),
        }

    return {
        "total_states": len(states),
        "max_generation_depth": max(
            (int(state.generation_depth) for state in states), default=0,
        ),
        "pools": {
            name: pool_summary(pools[name])
            for name in ("mastered", "frontier", "too_hard")
        },
        "mastered_boundary": pool_summary(boundary),
        "deepest_lineages": _deepest_lineages(pools),
    }


def virtual_sampling_diagnostic(
    curriculum: dict[str, Any],
    frontier: list[CurriculumState],
    historical: list[CurriculumState],
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Teste le routeur sans reset, step, restauration ni appel MuJoCo."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 23]))
    sampling = curriculum["start_sampling"]
    source_counts = {
        "curriculum_frontier": 0,
        "curriculum_historical": 0,
        "true_start": 0,
    }
    bins = historical_quantile_bins(
        historical, int(sampling["historical_bins"]),
    )
    bin_counts = [0] * len(bins)
    for _ in range(draws):
        selection = select_training_start(
            rng,
            curriculum_probability=float(
                curriculum["curriculum_reset_probability"]
            ),
            frontier_fraction=float(sampling["frontier_fraction"]),
            historical_fraction=float(sampling["historical_fraction"]),
            historical_bins=int(sampling["historical_bins"]),
            frontier=frontier,
            historical=historical,
            historical_bin_groups=bins,
        )
        source_counts[selection.source] += 1
        if selection.historical_bin is not None:
            bin_counts[selection.historical_bin] += 1
    historical_draws = sum(bin_counts)
    return {
        "draws": draws,
        "fractions": {
            "frontier": source_counts["curriculum_frontier"] / draws,
            "historical": source_counts["curriculum_historical"] / draws,
            "true_start": source_counts["true_start"] / draws,
        },
        "historical_bins": [
            {
                "bin": index,
                "size": len(states),
                "min": float(min(state.pose_distance for state in states)),
                "max": float(max(state.pose_distance for state in states)),
                "depth_min": int(min(
                    state.generation_depth for state in states
                )),
                "depth_max": int(max(
                    state.generation_depth for state in states
                )),
                "selection_fraction_within_historical": (
                    count / historical_draws if historical_draws else 0.0
                ),
            }
            for index, (states, count) in enumerate(zip(bins, bin_counts), 1)
        ],
    }


def run_diagnostic(
    config_path: str,
    seed: int | None = None,
    curriculum_state: str | Path | None = None,
) -> dict:
    config = load_config(config_path)
    if not config["curriculum"]["enabled"]:
        raise ValueError("Le diagnostic exige curriculum.enabled=true")
    effective_seed = int(config["training"]["base_seed"] if seed is None else seed)
    env = TenonMortaiseEnv(config_path, allow_curriculum_resets=False)
    try:
        manager = ReverseCurriculumManager(
            env, config["curriculum"], seed=effective_seed,
        )
        if curriculum_state is not None:
            manager.load(curriculum_state)
        candidates, report = manager.generate_candidates([manager.goal_seed])
        for candidate in candidates:
            report.restoration_checks += 1
            if not _check_restoration(env, candidate):
                report.restoration_failures += 1
        if curriculum_state is None:
            # Aucun classement n'est possible sans policy SAC. Pour tester le
            # routeur malgré tout, répartir uniquement les candidats restaurés
            # dans deux pools virtuels clairement identifiés comme tels.
            frontier = candidates[::2]
            historical = candidates[1::2]
            if candidates and not historical:
                historical = candidates[:1]
            sampling_pool_source = "generated_virtual_fixture"
            too_hard: list[CurriculumState] = []
        else:
            frontier = list(manager.pools["frontier"])
            historical = list(manager.pools["mastered"])
            too_hard = list(manager.pools["too_hard"])
            sampling_pool_source = str(curriculum_state)
        result = {
            "config": config_path,
            "seed": effective_seed,
            "curriculum_state": (
                None if curriculum_state is None else str(curriculum_state)
            ),
            "goal_seed": {
                "position_error": manager.goal_seed.position_error,
                "rotation_error": manager.goal_seed.rotation_error,
                "pose_distance": manager.goal_seed.pose_distance,
                "success": True,
                "unsafe": False,
            },
            **report.as_dict(candidates),
            # Les rapports d'efficacité sont volontairement des métriques
            # éphémères de l'update (TensorBoard est leur source historique).
            # Ils ne sont pas ajoutés au pickle curriculum uniquement pour ce
            # diagnostic, afin de garder la reprise et sa migration simples.
            "expansion_efficiency": None,
            "sampling_pool_source": sampling_pool_source,
            "pool_sizes": {
                "frontier": len(frontier),
                "mastered": len(historical),
                "too_hard": len(too_hard),
            },
            "frontier_distance": _distance_statistics(frontier),
            "historical_distance": _distance_statistics(historical),
            "too_hard_distance": _distance_statistics(too_hard),
            "lineage": lineage_diagnostic({
                "frontier": frontier,
                "mastered": historical,
                "too_hard": too_hard,
            }),
            "sampling_test": virtual_sampling_diagnostic(
                config["curriculum"], frontier, historical,
                draws=10_000, seed=effective_seed,
            ),
        }
    finally:
        env.close()
    if not candidates:
        raise RuntimeError("Le RCG n'a généré aucun candidat safe non-successful")
    if report.restoration_failures:
        raise RuntimeError(
            f"{report.restoration_failures} restauration(s) RCG incohérente(s)"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test1V21.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--curriculum-state", type=Path, default=None)
    return parser.parse_args()


def format_diagnostic(result: dict[str, Any]) -> str:
    sizes = result["pool_sizes"]
    frontier = result["frontier_distance"]
    historical = result["historical_distance"]
    too_hard = result["too_hard_distance"]
    lineage = result["lineage"]
    sampling = result["sampling_test"]
    efficiency = result.get("expansion_efficiency")

    def efficiency_value(name: str) -> Any:
        if not isinstance(efficiency, dict):
            return "unavailable (not persisted)"
        value = efficiency.get(name)
        return "unavailable" if value is None else value

    lines = [
        "PHYSICAL GENERATION",
        f"generated: {result['generated']}",
        f"valid: {result['valid']}",
        f"restoration_failures: {result['restoration_failures']}",
        "",
        "EXPANSION EFFICIENCY",
        "last update:",
        f"  branches expanded: {efficiency_value('expansion_branches')}",
        f"  candidates generated: {efficiency_value('expansion_candidates')}",
        f"  hops: {efficiency_value('expansion_hops')}",
        f"  new mastered: {efficiency_value('new_mastered')}",
        f"  new frontier: {efficiency_value('new_frontier')}",
        f"  new too_hard: {efficiency_value('new_too_hard')}",
        "",
        "rollouts:",
        f"  expansion: {efficiency_value('expansion_rollouts')}",
        "  mastered revalidation: "
        f"{efficiency_value('revalidation_mastered_rollouts')}",
        "  too-hard revalidation: "
        f"{efficiency_value('revalidation_too_hard_rollouts')}",
        "",
        "scale:",
        f"  mean: {efficiency_value('expansion_scale_mean')}",
        f"  max: {efficiency_value('expansion_scale_max')}",
        "",
        "efficiency:",
        "  frontier found per candidate: "
        f"{efficiency_value('frontier_found_per_candidate')}",
        "",
        "wall time:",
        f"  expansion: {efficiency_value('expansion_wall_time')}",
        f"  revalidation: {efficiency_value('revalidation_wall_time')}",
        "",
        f"POOL SIZES ({result['sampling_pool_source']})",
        f"frontier: {sizes['frontier']}",
        f"mastered: {sizes['mastered']}",
        f"too_hard: {sizes['too_hard']}",
        "",
        "LINEAGE",
        f"total states: {lineage['total_states']}",
        f"max generation depth: {lineage['max_generation_depth']}",
        "",
    ]
    for name in ("mastered", "frontier", "too_hard"):
        summary = lineage["pools"][name]
        depth = summary["depth"]
        lines.extend([
            f"{name}:",
            f"  count: {summary['count']}",
            f"  depth min: {depth['min']}",
            f"  depth median: {depth['median']}",
            f"  depth max: {depth['max']}",
            "",
        ])
    boundary = lineage["mastered_boundary"]
    boundary_depth = boundary["depth"]
    lines.extend([
        "mastered boundary:",
        f"  count: {boundary['count']}",
        f"  depth min: {boundary_depth['min']}",
        f"  depth median: {boundary_depth['median']}",
        f"  depth max: {boundary_depth['max']}",
        "",
        "DEEPEST LINEAGES",
    ])
    if not lineage["deepest_lineages"]:
        lines.append("none")
    for item in lineage["deepest_lineages"]:
        lines.extend([
            item["path"],
            f"depth={item['depth']}",
            f"classification={item['classification']}",
            f"position_error={item['position_error']}",
            f"rotation_error={item['rotation_error']}",
            "",
        ])
    lines.extend([
        "GEOMETRY DIAGNOSTICS (pose_distance only)",
        "FRONTIER DISTANCE",
        f"min: {frontier['min']}",
        f"median: {frontier['median']}",
        f"max: {frontier['max']}",
        "",
        "MASTERED/HISTORICAL DISTANCE",
        f"min: {historical['min']}",
        f"q25: {historical['q25']}",
        f"median: {historical['median']}",
        f"q75: {historical['q75']}",
        f"max: {historical['max']}",
        "",
        "TOO HARD DISTANCE",
        f"min: {too_hard['min']}",
        f"median: {too_hard['median']}",
        f"max: {too_hard['max']}",
        "",
        f"SAMPLING TEST ({sampling['draws']} virtual draws, no MuJoCo)",
        f"frontier fraction: {sampling['fractions']['frontier']:.4f}",
        f"historical fraction: {sampling['fractions']['historical']:.4f}",
        f"true_start fraction: {sampling['fractions']['true_start']:.4f}",
        "",
        "HISTORICAL DEPTH BINS",
    ])
    for item in sampling["historical_bins"]:
        lines.append(
            "bin {bin}: size={size}, depth={depth_min}-{depth_max}, "
            "pose_distance={min:.9g}-{max:.9g}, "
            "selection_fraction={selection_fraction_within_historical:.4f}".format(
                **item
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = run_diagnostic(
        args.config, args.seed, curriculum_state=args.curriculum_state,
    )
    print(format_diagnostic(result))


if __name__ == "__main__":
    main()
