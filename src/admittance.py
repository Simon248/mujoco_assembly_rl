"""Contrôleur d'admittance indépendant de Gym/SAC."""
from __future__ import annotations
import numpy as np

class AdmittanceController:
    def __init__(self, config: dict):
        self.mass = np.asarray(config["mass"], float)
        self.damping = np.asarray(config["damping"], float)
        self.stiffness = np.asarray(config["stiffness"], float)
        self.limit = np.asarray(config["max_offset"], float)
        if np.any(self.mass <= 0): raise ValueError("Les masses d'admittance doivent être positives")
        self.reset()
    def reset(self): self.offset = np.zeros(6); self.velocity = np.zeros(6)
    def step(self, commanded_delta: np.ndarray, wrench: np.ndarray, dt: float) -> np.ndarray:
        acceleration = (wrench - self.damping*self.velocity - self.stiffness*self.offset) / self.mass
        self.velocity += acceleration * dt
        self.offset = np.clip(self.offset + self.velocity*dt, -self.limit, self.limit)
        return np.asarray(commanded_delta) + self.offset
