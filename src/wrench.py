"""Calcul du wrench de contact au grasp frame, sans confondre capteur et repère."""
from __future__ import annotations
import mujoco
import numpy as np

def contact_wrench_at_site(model: mujoco.MjModel, data: mujoco.MjData, mobile_geom: int, grasp_site: int) -> np.ndarray:
    """Somme les contacts sur le tenon et ramène [F,T] au site grasp, dans son repère.

    `mj_contactForce` est évalué pour chaque contact. Le couple est transporté
    depuis le point de contact : T_grasp = T_contact + (p_contact-p_grasp)xF.
    Le signe est celui de l'effort appliqué au géomètre mobile; cette convention
    est à vérifier visuellement avec `src.debug` avant tout entraînement.
    """
    p_grasp = data.site_xpos[grasp_site]
    rotation = data.site_xmat[grasp_site].reshape(3, 3)
    total_force = np.zeros(3); total_torque = np.zeros(3)
    buffer = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if mobile_geom not in (contact.geom1, contact.geom2): continue
        mujoco.mj_contactForce(model, data, index, buffer)
        # MuJoCo stocke exceptionnellement les axes du contact sur les lignes
        # de frame. La transformation contact -> monde est donc sa transposée.
        contact_rotation = contact.frame.reshape(3, 3)
        force = contact_rotation.T @ buffer[:3]
        torque = contact_rotation.T @ buffer[3:]
        # mj_contactForce convention is force on geom2.
        if contact.geom1 == mobile_geom: force, torque = -force, -torque
        total_force += force
        total_torque += torque + np.cross(contact.pos - p_grasp, force)
    return np.r_[rotation.T @ total_force, rotation.T @ total_torque]
