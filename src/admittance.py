"""Contrôleur d'admittance cartésienne indépendant de Gym et de SAC."""
from __future__ import annotations

import numpy as np


class AdmittanceController:
    """Intègre un wrench en un *offset absolu* autour d'une pose de référence.

    L'offset retourné ne doit jamais être ajouté à la pose courante à chaque
    cycle. Il s'applique une fois à la référence cartésienne commandée.
    """

    def __init__(self, config: dict):
        self.mass = np.asarray(config["mass"], dtype=float)
        self.damping = np.asarray(config["damping"], dtype=float)
        self.stiffness = np.asarray(config["stiffness"], dtype=float)
        self.offset_limit = np.asarray(config["max_offset"], dtype=float)
        self.velocity_limit = np.asarray(config["max_velocity"], dtype=float)
        for name, value in {
            "mass": self.mass,
            "damping": self.damping,
            "stiffness": self.stiffness,
            "max_offset": self.offset_limit,
            "max_velocity": self.velocity_limit,
        }.items():
            if value.shape != (6,):
                raise ValueError(f"admittance.{name} doit contenir 6 valeurs")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"admittance.{name} doit contenir des valeurs finies")
        if np.any(self.mass <= 0) or np.any(self.offset_limit <= 0) or np.any(self.velocity_limit <= 0):
            raise ValueError("mass, max_offset et max_velocity doivent être strictement positifs")
        if np.any(self.damping < 0) or np.any(self.stiffness < 0):
            raise ValueError("damping et stiffness doivent être positifs ou nuls")
        self.reset()

    def reset(self) -> None:
        self.offset = np.zeros(6, dtype=float)
        self.velocity = np.zeros(6, dtype=float)

    def step(self, wrench: np.ndarray, dt: float) -> np.ndarray:
        """Retourne l'offset absolu 6D après un pas d'intégration semi-implicite."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt doit être strictement positif")
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,):
            raise ValueError("wrench doit avoir la forme (6,)")
        if not np.all(np.isfinite(wrench)):
            raise ValueError("wrench doit contenir des valeurs finies")

        acceleration = (
            wrench - self.damping * self.velocity - self.stiffness * self.offset
        ) / self.mass
        self.velocity = np.clip(
            self.velocity + acceleration * dt,
            -self.velocity_limit,
            self.velocity_limit,
        )
        candidate = self.offset + self.velocity * dt
        clipped = np.clip(candidate, -self.offset_limit, self.offset_limit)

        # Anti-windup : une vitesse qui pousse plus loin dans une saturation
        # est annulée; une vitesse de retour vers la zone valide est conservée.
        outward = ((candidate > self.offset_limit) & (self.velocity > 0)) | (
            (candidate < -self.offset_limit) & (self.velocity < 0)
        )
        self.velocity[outward] = 0.0
        self.offset = clipped
        return self.offset.copy()
