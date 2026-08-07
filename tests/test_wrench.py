from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from src.wrench import contact_wrench_at_site


class WrenchTransformTest(unittest.TestCase):
    def _data(self, geom1, geom2):
        contact = SimpleNamespace(
            geom1=geom1, geom2=geom2, pos=np.zeros(3),
            # Axes du contact stockés sur les lignes (rotation de 90° autour z).
            frame=np.array([0, 1, 0, -1, 0, 0, 0, 0, 1], dtype=float),
        )
        return SimpleNamespace(
            ncon=1, contact=[contact], site_xpos=np.zeros((1, 3)),
            site_xmat=np.eye(3).reshape(1, 9),
        )

    @patch("src.wrench.mujoco.mj_contactForce")
    def test_contact_axes_are_transposed_from_local_to_world(self, contact_force):
        contact_force.side_effect = lambda model, data, index, output: output.__setitem__(slice(None), [1, 0, 0, 0, 0, 0])
        wrench = contact_wrench_at_site(object(), self._data(3, 7), mobile_geom=7, grasp_site=0)
        np.testing.assert_allclose(wrench, [0, 1, 0, 0, 0, 0])

    @patch("src.wrench.mujoco.mj_contactForce")
    def test_force_sign_is_reversed_when_mobile_is_geom1(self, contact_force):
        contact_force.side_effect = lambda model, data, index, output: output.__setitem__(slice(None), [1, 0, 0, 0, 0, 0])
        wrench = contact_wrench_at_site(object(), self._data(7, 3), mobile_geom=7, grasp_site=0)
        np.testing.assert_allclose(wrench, [0, -1, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
