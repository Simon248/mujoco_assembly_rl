"""Export a local point-cloud approximation of MuJoCo's table SDF surface.

The SDF itself is held inside MuJoCo's plugin and cannot be exported directly.
This tool probes it with a small sphere through ``mj_geomDistance`` and writes
points close to the zero isosurface as an ASCII PLY file.
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from src.mujoco_plugins import load_sdf_plugin


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _add_probe(xml_path: Path, contact_margin: float) -> str:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "sdf_probe_body"})
    ET.SubElement(
        body,
        "inertial",
        {"pos": "0 0 0", "mass": "0.001", "diaginertia": "1e-9 1e-9 1e-9"},
    )
    ET.SubElement(body, "freejoint", {"name": "sdf_probe_freejoint"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "sdf_probe",
            "type": "sphere",
            "size": "0.0001",
            "margin": f"{contact_margin:.8f}",
            "rgba": "0 1 1 0",
        },
    )
    return ET.tostring(root, encoding="unicode")


def _assets_for_xml(xml_path: Path) -> dict[str, bytes]:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    assets: dict[str, bytes] = {}
    for mesh in root.findall("./asset/mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        relative = Path(meshdir) / filename
        assets[str(relative)] = (xml_path.parent / relative).read_bytes()
    return assets


def _part_bounds(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "part_1_mesh")
    if mesh_id < 0:
        raise ValueError("Mesh 'part_1_mesh' not found")
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[start : start + count].astype(np.float64, copy=True)
    # MuJoCo recenters imported mesh vertices and stores the inverse transform
    # in mesh_pos/mesh_quat. Reapply it so the scan is in the CAD/world frame.
    w, x, y, z = model.mesh_quat[mesh_id]
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    vertices = vertices @ rotation.T + model.mesh_pos[mesh_id]
    return vertices.min(axis=0), vertices.max(axis=0)


def _distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    table_id: int,
    probe_id: int,
    probe_qposadr: int,
    point: np.ndarray,
) -> float:
    data.qpos[probe_qposadr : probe_qposadr + 3] = point
    data.qpos[probe_qposadr + 3 : probe_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    distances = [
        float(data.contact[index].dist)
        for index in range(data.ncon)
        if {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        == {table_id, probe_id}
    ]
    return min(distances) if distances else float("inf")


def _write_ply(path: Path, points: list[tuple[float, float, float, float]]) -> None:
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property float sdf_distance\nend_header\n")
        for x, y, z, distance in points:
            stream.write(f"{x:.7f} {y:.7f} {z:.7f} {distance:.7f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=_path_from_env("MUJOCO_XML_PATH", "/data/input/scene.xml"))
    parser.add_argument("--output-dir", type=Path, default=_path_from_env("SDF_OUTPUT_DIR", "/data/output/sdf"))
    parser.add_argument("--resolution", type=float, default=0.002)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--band", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution <= 0 or args.margin <= 0:
        raise ValueError("--resolution and --margin must be positive")
    xml_path = args.xml.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    band = args.band if args.band is not None else args.resolution * 0.75
    load_sdf_plugin()
    model = mujoco.MjModel.from_xml_string(
        _add_probe(xml_path, band), _assets_for_xml(xml_path)
    )
    data = mujoco.MjData(model)
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "assembly_table_collision")
    probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sdf_probe")
    probe_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "sdf_probe_freejoint")
    if min(table_id, probe_id, probe_joint_id) < 0:
        raise ValueError("Unable to locate SDF table or probe in diagnostic model")
    probe_qposadr = int(model.jnt_qposadr[probe_joint_id])

    lower, upper = _part_bounds(model)
    lower -= args.margin
    upper += args.margin
    axes = [np.arange(lower[i], upper[i] + args.resolution * 0.5, args.resolution) for i in range(3)]
    expected_queries = int(np.prod([len(axis) for axis in axes]))
    print(f"[sdf-export] bounds={lower.tolist()}..{upper.tolist()} resolution={args.resolution:.4f} m")
    print(f"[sdf-export] probing {expected_queries} points; this uses MuJoCo's actual SDF collision.")

    points: list[tuple[float, float, float, float]] = []
    completed = 0
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                point = np.array([x, y, z], dtype=np.float64)
                distance = _distance(model, data, table_id, probe_id, probe_qposadr, point)
                if distance <= band:
                    points.append((float(x), float(y), float(z), distance))
                completed += 1
        print(f"[sdf-export] {completed}/{expected_queries} points, isosurface samples={len(points)}", flush=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "table_isosurface_local.ply"
    _write_ply(ply_path, points)
    metadata = {
        "xml_path": str(xml_path),
        "mujoco_version": mujoco.__version__,
        "bounds_min_m": lower.tolist(),
        "bounds_max_m": upper.tolist(),
        "resolution_m": args.resolution,
        "isosurface_band_m": band,
        "probe_count": expected_queries,
        "point_count": len(points),
        "ply_path": str(ply_path),
    }
    (output_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[sdf-export] wrote {ply_path} ({len(points)} points)")


if __name__ == "__main__":
    main()
