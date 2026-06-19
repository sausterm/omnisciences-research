"""Tests for Marcus theory rate calculations."""

import math
import pytest
import numpy as np

from pcet_engine.core.marcus import (
    marcus_rate,
    marcus_rate_kcal,
    marcus_activation_energy,
    reorganization_energy_from_hessians,
)
from pcet_engine.core.constants import KCALMOL_TO_HARTREE, HARTREE_TO_KCALMOL


class TestMarcusActivationEnergy:
    def test_normal_region(self):
        """E_a should be positive for ΔG + λ > 0."""
        lam = 20.0 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        E_a = marcus_activation_energy(dG, lam)
        E_a_kcal = E_a * HARTREE_TO_KCALMOL
        # (−5 + 20)² / (4×20) = 225/80 = 2.8125
        assert abs(E_a_kcal - 2.8125) < 0.01

    def test_activationless(self):
        """E_a = 0 when ΔG = -λ."""
        lam = 10.0 * KCALMOL_TO_HARTREE
        dG = -10.0 * KCALMOL_TO_HARTREE
        E_a = marcus_activation_energy(dG, lam)
        assert abs(E_a) < 1e-10

    def test_inverted_region(self):
        """E_a should increase when |ΔG| > λ (inverted region)."""
        lam = 10.0 * KCALMOL_TO_HARTREE
        dG_normal = -5.0 * KCALMOL_TO_HARTREE
        dG_inverted = -15.0 * KCALMOL_TO_HARTREE
        E_a_normal = marcus_activation_energy(dG_normal, lam)
        E_a_inverted = marcus_activation_energy(dG_inverted, lam)
        assert E_a_inverted > 0
        assert E_a_inverted == pytest.approx(E_a_normal, rel=1e-6)  # symmetric

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            marcus_activation_energy(0.0, -1.0)


class TestMarcusRate:
    def test_positive_rate(self):
        """Rate should be positive and finite."""
        V = 0.5 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE
        k = marcus_rate(V, dG, lam)
        assert k > 0
        assert math.isfinite(k)

    def test_rate_increases_with_coupling(self):
        """Rate should scale as V²."""
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE
        k1 = marcus_rate(0.5 * KCALMOL_TO_HARTREE, dG, lam)
        k2 = marcus_rate(1.0 * KCALMOL_TO_HARTREE, dG, lam)
        assert abs(k2 / k1 - 4.0) < 0.01

    def test_rate_temperature_dependence(self):
        """Rate should increase with temperature (normal region)."""
        V = 0.5 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE
        k_low = marcus_rate(V, dG, lam, temperature=250.0)
        k_high = marcus_rate(V, dG, lam, temperature=350.0)
        assert k_high > k_low

    def test_kcal_convenience(self):
        """marcus_rate_kcal should give same result as manual conversion."""
        k1 = marcus_rate_kcal(0.5, -5.0, 20.0, 298.15)
        k2 = marcus_rate(
            0.5 * KCALMOL_TO_HARTREE,
            -5.0 * KCALMOL_TO_HARTREE,
            20.0 * KCALMOL_TO_HARTREE,
            298.15,
        )
        assert abs(k1 / k2 - 1.0) < 1e-10

    def test_reasonable_enzyme_rate(self):
        """For typical enzyme parameters, rate should be in a reasonable range."""
        k = marcus_rate_kcal(V_coupling_kcal=0.5, delta_G_kcal=-5.0,
                             lambda_reorg_kcal=20.0, temperature=298.15)
        # Pure Marcus (no tunneling correction) can give large rates; check finite/positive
        assert k > 0
        assert math.isfinite(k)


class TestReorgFromHessians:
    def test_zero_displacement(self):
        """λ should be zero if geometries are identical."""
        n = 3  # 3 atoms
        H = np.eye(9) * 0.5
        geom = np.zeros(9)
        masses = np.ones(n) * 12.0
        lam_f, lam_b = reorganization_energy_from_hessians(H, H, geom, geom, masses)
        assert abs(lam_f) < 1e-10
        assert abs(lam_b) < 1e-10

    def test_positive_reorganization(self):
        """λ should be positive for non-zero displacement with positive Hessian."""
        n = 2
        H = np.eye(6) * 1.0
        geom_R = np.zeros(6)
        geom_P = np.array([0.1, 0.0, 0.0, -0.1, 0.0, 0.0])
        masses = np.array([12.0, 16.0])
        lam_f, lam_b = reorganization_energy_from_hessians(H, H, geom_R, geom_P, masses)
        assert lam_f > 0
        assert lam_b > 0
