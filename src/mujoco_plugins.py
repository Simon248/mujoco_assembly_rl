"""MuJoCo extension loading used by CAD collision scenes."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco


_SDF_PLUGIN_LOADED = False


def load_sdf_plugin() -> None:
    """Register MuJoCo's bundled SDF library before parsing MJCF.

    The Python wheel ships plugins next to the bindings, but allowing
    ``MUJOCO_PLUGIN_PATH`` also makes a system MuJoCo installation usable.
    """
    global _SDF_PLUGIN_LOADED
    if _SDF_PLUGIN_LOADED:
        return

    search_dirs = [Path(mujoco.__file__).resolve().parent / "plugin"]
    plugin_path = os.environ.get("MUJOCO_PLUGIN_PATH")
    if plugin_path:
        search_dirs.insert(0, Path(plugin_path))

    for directory in search_dirs:
        if not directory.is_dir():
            continue
        candidates = sorted(directory.glob("*sdf*.so"))
        if not candidates:
            continue
        for library in candidates:
            mujoco.mj_loadPluginLibrary(str(library))
        _SDF_PLUGIN_LOADED = True
        return

    searched = ", ".join(str(path) for path in search_dirs)
    raise RuntimeError(
        "MuJoCo SDF plugin library not found. Expected a libsdf shared library in: "
        f"{searched}. Rebuild the Docker image with the pinned mujoco package."
    )
