"""Tests for electrochemistry module."""

import numpy as np
import pytest
from pcet_engine.core.electrochemistry import (
    fermi_dirac, electrochemical_rate, tafel_analysis,
)


class TestFermiDirac:
    def test_half_at_fermi_level(self):
        assert abs(fermi_dirac(0.0) - 0.5) < 1e-10

    def test_one_below_fermi(self):
        assert fermi_dirac(-1.0) > 0.999

    def test_zero_above_fermi(self):
        assert fermi_dirac(1.0) < 0.001

    def test_array_input(self):
        eps = np.array([-1, 0, 1])
        f = fermi_dirac(eps)
        assert f.shape == (3,)
        assert f[1] == pytest.approx(0.5, abs=1e-10)

    def test_higher_temp_broader(self):
        f_low = fermi_dirac(0.1, temperature=100)
        f_high = fermi_dirac(0.1, temperature=1000)
        assert f_high > f_low  # More occupation above Fermi at higher T


class TestElectrochemicalRate:
    def test_anodic_rate_positive(self):
        def dummy_rate(dG_eV):
            return 1e6 * np.exp(-abs(dG_eV))
        k, k_eps = electrochemical_rate(dummy_rate, 0.0)
        assert k > 0

    def test_cathodic_rate_positive(self):
        def dummy_rate(dG_eV):
            return 1e6 * np.exp(-abs(dG_eV))
        k, k_eps = electrochemical_rate(dummy_rate, 0.0, direction='cathodic')
        assert k > 0

    def test_overpotential_changes_rate(self):
        def rate_func(dG_eV):
            lam = 1.0
            Ea = (dG_eV + lam)**2 / (4 * lam)
            return 1e6 * np.exp(-Ea / 0.0257)
        k0, _ = electrochemical_rate(rate_func, 0.0, overpotential=0.0)
        k1, _ = electrochemical_rate(rate_func, 0.0, overpotential=0.5, direction='cathodic')
        # Different overpotential should give different rate
        assert k0 != k1


class TestTafelAnalysis:
    def test_known_slope(self):
        RTF = 8.314 * 298 / 96485  # ~0.0257 V
        alpha_true = 0.5
        potentials = np.linspace(-1, 0, 20)
        ln_k = -alpha_true / RTF * potentials + 10.0  # perfect Tafel
        alpha = tafel_analysis(potentials, ln_k)
        assert abs(alpha - alpha_true) < 0.01
