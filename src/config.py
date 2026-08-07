"""Chargement des configurations d'essais, avec héritage YAML minimal."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]

def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration and return its fully resolved mapping.

    The returned dictionary never contains ``extends`` and is consequently safe
    to archive with a training run.
    """
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"La configuration doit être un mapping YAML: {path}")
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = path.parent / parent
        if not parent_path.is_file():
            raise ValueError(
                f"Configuration archivée non autonome: {path} hérite de "
                f"{parent!r}, introuvable à {parent_path}. "
                "Cet essai utilise l'ancien format et doit être relancé."
            )
        cfg = _merge(load_config(parent_path), cfg)
    required = {"case", "target_pose_fixed_to_mobile", "initial_pose_fixed_to_mobile"}
    missing = required - cfg.keys()
    if missing: raise ValueError(f"Configuration incomplète ({path}): {sorted(missing)}")
    if cfg["case"] not in {"tenon_1", "tenon_2"}: raise ValueError("case doit être tenon_1 ou tenon_2")
    return cfg


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    """Archive one self-contained, human-readable configuration YAML."""
    if "extends" in config:
        raise ValueError("Une configuration résolue ne doit pas contenir 'extends'")
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
