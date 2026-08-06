"""Rejoue les chemins CAD dans MuJoCo et produit un rapport de contacts.

Modifier SELECTED_SEGMENTS ci-dessous pour ne rejouer qu'un ou plusieurs
segments. Les noms dans les YAML sont ``place``, ``retreat`` et ``approach``.
``retrait`` est accepté comme alias français de ``retreat``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import yaml

from src.mujoco_plugins import load_sdf_plugin


# Sélection à modifier directement dans le code, sans argument de ligne de commande.
# Exemples : ("approach",), ("approach", "place"), ou les trois segments.
SELECTED_SEGMENTS = ("place",)

# True : une pièce est testée contre la table et toutes les autres à leur pose finale.
# False : contre la table et les pièces placées avant elle (ordre part_1, part_2, part_3).
TEST_AGAINST_ALL_OTHER_PARTS = False

MAX_TRANSLATION_STEP_M = 0.002
REPLAY_WITH_VIEWER = "--viewer" in os.sys.argv

# Réglages du diagnostic visuel des collisions dans le viewer.
VIEWER_NORMAL_SPEED = 1.0
VIEWER_COLLISION_SPEED = 0.15
PROMPT_ON_COLLISION = True
CONTACT_MARKER_RADIUS_M = 0.004
WORST_CONTACT_MARKER_RADIUS_M = 0.009

SEGMENT_ALIASES = {"retrait": "retreat"}


def canonical_segments() -> tuple[str, ...]:
    names = tuple(SEGMENT_ALIASES.get(name, name) for name in SELECTED_SEGMENTS)
    unknown = set(names) - {"place", "retreat", "approach"}
    if unknown:
        raise ValueError(f"Segment(s) inconnu(s) : {sorted(unknown)}")
    return names


def pose(value: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    p, q = value["position"], value["orientation"]
    position = np.array([p["x"], p["y"], p["z"]], dtype=float)
    quaternion = np.array([q["w"], q["x"], q["y"], q["z"]], dtype=float)
    return position, quaternion / np.linalg.norm(quaternion)


def load_paths(input_dir: Path) -> dict[str, dict[str, Any]]:
    paths = {}
    for path in sorted(input_dir.glob("*.yaml")):
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = str(content["tracked_frame"])
        if name in paths:
            raise ValueError(f"Deux trajectoires pour {name}")
        paths[name] = content
    if not paths:
        raise FileNotFoundError(f"Aucun YAML dans {input_dir}")
    return paths


def make_model(paths: dict[str, dict[str, Any]], cad_dir: Path) -> tuple[mujoco.MjModel, dict[str, int], dict[str, int]]:
    names = list(paths)
    xml = [
        '<mujoco model="path_collision_test">',
        '<compiler angle="radian" autolimits="true"/>',
        '<option sdf_iterations="10" sdf_initpoints="20"/>',
        '<extension><plugin plugin="mujoco.sdf.sdflib">',
        '<instance name="table_sdf"><config key="aabb" value="0"/></instance>',
    ]
    xml += [f'<instance name="{name}_sdf"><config key="aabb" value="0"/></instance>' for name in names]
    xml += ['</plugin></extension>', '<asset>']
    xml += [f'<mesh name="table_mesh" file="{escape(str(cad_dir / "chandelier_assembly_table_visual.stl"))}"/>']
    xml += [f'<mesh name="{name}_mesh" file="{escape(str(cad_dir / (name + ".stl")))}"/>' for name in names]
    xml += [
        '</asset><worldbody>',
        '<light pos="0 -1 1.5" dir="0 1 -1"/>',
        '<camera name="overview" pos="0.75 -1.10 0.65" xyaxes="0.83 0.56 0 -0.22 0.34 0.91"/>',
        '<geom type="plane" size="0 0 .05" contype="0" conaffinity="0" rgba=".85 .85 .85 1"/>',
        '<geom name="assembly_table_visual" type="mesh" mesh="table_mesh" contype="0" conaffinity="0" rgba=".65 .65 .70 1"/>',
        '<geom name="assembly_table_collision" type="sdf" mesh="table_mesh" rgba=".65 .65 .70 0"><plugin instance="table_sdf"/></geom>',
    ]
    for name in names:
        final_p, final_q = pose(paths[name]["segments"]["place"][-1]["pose"])
        xml += [
            f'<body name="{name}" pos="{" ".join(map(str, final_p))}" quat="{" ".join(map(str, final_q))}">',
            f'<freejoint name="{name}_free"/>',
            f'<geom name="{name}_visual" type="mesh" mesh="{name}_mesh" contype="0" conaffinity="0" rgba=".90 .55 .12 1"/>',
            f'<geom name="{name}_collision" type="sdf" mesh="{name}_mesh" rgba=".90 .55 .12 0"><plugin instance="{name}_sdf"/></geom>',
            '</body>',
        ]
    xml += ['</worldbody></mujoco>']
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as file:
        file.write("\n".join(xml))
        xml_path = Path(file.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    finally:
        xml_path.unlink(missing_ok=True)
    qpos = {name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")]) for name in names}
    geoms = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_collision") for name in names}
    geoms["table"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "assembly_table_collision")
    return model, qpos, geoms


def set_pose(data: mujoco.MjData, address: int, current_pose: tuple[np.ndarray, np.ndarray]) -> None:
    data.qpos[address : address + 3] = current_pose[0]
    data.qpos[address + 3 : address + 7] = current_pose[1]


def slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Interpole deux quaternions wxyz sans dépendre d'une API MuJoCo donnée."""
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    # q et -q représentent la même rotation ; retenir le chemin le plus court.
    if dot < 0.0:
        end = -end
        dot = -dot
    if dot > 0.9995:
        result = start + fraction * (end - start)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    return (np.sin((1.0 - fraction) * angle) / sine) * start + (
        np.sin(fraction * angle) / sine
    ) * end


def samples(points: list[dict[str, Any]]):
    first_p, first_q = pose(points[0]["pose"])
    yield float(points[0]["time_from_start_s"]), first_p, first_q
    for a, b in zip(points, points[1:]):
        ap, aq, bp, bq = *pose(a["pose"]), *pose(b["pose"])
        n = max(1, int(np.ceil(np.linalg.norm(bp - ap) / MAX_TRANSLATION_STEP_M)))
        for i in range(1, n + 1):
            fraction = i / n
            q = slerp(aq, bq, fraction)
            yield (float(a["time_from_start_s"]) + fraction * (float(b["time_from_start_s"]) - float(a["time_from_start_s"])), (1 - fraction) * ap + fraction * bp, q)


def test(paths: dict[str, dict[str, Any]], model: mujoco.MjModel, qpos: dict[str, int], geoms: dict[str, int]) -> list[dict[str, Any]]:
    data, result, names = mujoco.MjData(model), [], list(paths)
    for moving_index, moving in enumerate(names):
        for name in names:
            set_pose(data, qpos[name], pose(paths[name]["segments"]["place"][-1]["pose"]))
        obstacles = names if TEST_AGAINST_ALL_OTHER_PARTS else names[:moving_index]
        obstacle_geoms = {geoms["table"]} | {geoms[name] for name in obstacles if name != moving}
        for segment in canonical_segments():
            points = paths[moving]["segments"].get(segment)
            if not points:
                continue
            events, minimum, count = [], float("inf"), 0
            for index, (time_s, p, q) in enumerate(samples(points)):
                count += 1
                set_pose(data, qpos[moving], (p, q))
                mujoco.mj_forward(model, data)
                for contact in data.contact[: data.ncon]:
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if geoms[moving] not in pair or not (pair & obstacle_geoms):
                        continue
                    other_id = next(item for item in pair if item != geoms[moving])
                    other = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_id)
                    minimum = min(minimum, float(contact.dist))
                    events.append({"sample_index": index, "time_s": time_s, "other_geom": other, "distance_m": float(contact.dist), "position_m": [float(x) for x in contact.pos]})
            counts: dict[str, int] = defaultdict(int)
            for event in events:
                counts[event["other_geom"]] += 1
            result.append({"part": moving, "segment": segment, "samples_tested": count, "collision_detected": bool(events), "collision_samples_by_obstacle": dict(counts), "minimum_contact_distance_m": None if not events else minimum, "events": events})
    return result


def summarize(tests: list[dict[str, Any]]) -> dict[str, Any]:
    """Create the short, human-readable part of the JSON report."""
    results = []
    for test_result in tests:
        events = test_result["events"]
        worst = min(events, key=lambda event: event["distance_m"]) if events else None
        first = events[0] if events else None
        results.append(
            {
                "part": test_result["part"],
                "segment": test_result["segment"],
                "collision_detected": test_result["collision_detected"],
                "obstacles": sorted(test_result["collision_samples_by_obstacle"]),
                "first_contact": None
                if first is None
                else {
                    "time_s": first["time_s"],
                    "obstacle": first["other_geom"],
                    "position_m": first["position_m"],
                },
                "worst_contact": None
                if worst is None
                else {
                    "time_s": worst["time_s"],
                    "obstacle": worst["other_geom"],
                    "penetration_mm": -1000.0 * worst["distance_m"],
                    "position_m": worst["position_m"],
                },
            }
        )
    return {
        "segments_tested": len(results),
        "segments_with_collision": sum(item["collision_detected"] for item in results),
        "results": results,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"Diagnostic : {summary['segments_with_collision']}/"
        f"{summary['segments_tested']} segment(s) avec collision"
    )
    for item in summary["results"]:
        prefix = "COLLISION" if item["collision_detected"] else "OK"
        if not item["collision_detected"]:
            print(f"[{prefix}] {item['part']} / {item['segment']}")
            continue
        worst = item["worst_contact"]
        assert worst is not None
        position = ", ".join(f"{value:.6f}" for value in worst["position_m"])
        obstacles = ", ".join(item["obstacles"])
        print(
            f"[{prefix}] {item['part']} / {item['segment']} | "
            f"obstacles: {obstacles} | pire: {worst['obstacle']} à "
            f"t={worst['time_s']:.3f}s | pénétration={worst['penetration_mm']:.3f} mm | "
            f"position table=({position})"
        )


def set_contact_markers(viewer: Any, events: list[dict[str, Any]], worst_event: dict[str, Any] | None) -> None:
    """Draw red contact points and a larger yellow marker at the worst contact."""
    scene = viewer.user_scn
    scene.ngeom = 0
    identity = np.eye(3).reshape(-1)
    for event in events:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([CONTACT_MARKER_RADIUS_M, 0.0, 0.0]),
            np.asarray(event["position_m"]),
            identity,
            np.array([1.0, 0.0, 0.0, 1.0]),
        )
        scene.ngeom += 1
    if worst_event is not None and scene.ngeom < scene.maxgeom:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([WORST_CONTACT_MARKER_RADIUS_M, 0.0, 0.0]),
            np.asarray(worst_event["position_m"]),
            identity,
            np.array([1.0, 0.85, 0.0, 1.0]),
        )
        scene.ngeom += 1


def view(
    model: mujoco.MjModel,
    paths: dict[str, dict[str, Any]],
    qpos: dict[str, int],
    tests: list[dict[str, Any]],
) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    tests_by_path = {(item["part"], item["segment"]): item for item in tests}
    for name in paths:
        set_pose(data, qpos[name], pose(paths[name]["segments"]["place"][-1]["pose"]))
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        for name, path in paths.items():
            for segment in canonical_segments():
                test_result = tests_by_path[(name, segment)]
                events_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for event in test_result["events"]:
                    events_by_sample[event["sample_index"]].append(event)
                worst = (
                    min(test_result["events"], key=lambda event: event["distance_m"])
                    if test_result["events"]
                    else None
                )
                visible_events: list[dict[str, Any]] = []
                first_contact_seen = False
                active_obstacles: set[str] = set()
                previous_time: float | None = None
                print(f"Viewer : {name} / {segment}")
                for sample_index, (time_s, position, quaternion) in enumerate(
                    samples(path["segments"][segment])
                ):
                    if not viewer.is_running():
                        return
                    set_pose(data, qpos[name], (position, quaternion))
                    mujoco.mj_forward(model, data)
                    new_events = events_by_sample.get(sample_index, [])
                    visible_events.extend(new_events)
                    if new_events:
                        first_contact_seen = True
                    current_obstacles = {event["other_geom"] for event in new_events}
                    entering_obstacles = current_obstacles - active_obstacles
                    at_worst = worst is not None and sample_index == worst["sample_index"]
                    set_contact_markers(
                        viewer,
                        visible_events,
                        worst if at_worst else None,
                    )
                    viewer.sync()
                    if PROMPT_ON_COLLISION and entering_obstacles:
                        obstacles = ", ".join(sorted(entering_obstacles))
                        print(
                            f"Collision détectée : {name} / {segment} avec {obstacles} "
                            f"à t={time_s:.3f}s."
                        )
                        input("Appuyez sur Entrée pour poursuivre la trajectoire... ")
                    active_obstacles = current_obstacles
                    if previous_time is not None:
                        speed = VIEWER_COLLISION_SPEED if first_contact_seen else VIEWER_NORMAL_SPEED
                        time.sleep(max(0.001, (time_s - previous_time) / speed))
                    previous_time = time_s
        print("Relecture terminée : les marqueurs restent affichés. Fermez le viewer pour quitter.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_root = Path(os.environ.get("INPUT_DIR", "/data/input"))
    if not input_root.is_dir():
        input_root = root / "data/input"
    load_sdf_plugin()
    paths = load_paths(input_root / "chemin")
    model, qpos, geoms = make_model(paths, input_root / "cad")
    tests = test(paths, model, qpos, geoms)
    summary = summarize(tests)
    report = {
        "summary": summary,
        "selected_segments": list(canonical_segments()),
        "test_against_all_other_parts": TEST_AGAINST_ALL_OTHER_PARTS,
        "maximum_translation_step_m": MAX_TRANSLATION_STEP_M,
        "paths_metadata": paths,
        "tests": tests,
    }
    output = Path(os.environ.get("OUTPUT_DIR", root / "data/output")) / "collision_report.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Rapport écrit : {output}")
    print_summary(summary)
    if REPLAY_WITH_VIEWER:
        view(model, paths, qpos, tests)


if __name__ == "__main__":
    main()
