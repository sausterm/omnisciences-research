"""Tests for FGH solver and numerical FC overlaps."""

import numpy as np
import pytest
from pcet_engine.core.fgh_solver import fgh_1d, compute_fc_overlaps
from pcet_engine.core.proton_potential import harmonic_potential, morse_potential
from pcet_engine.core.constants import PROTON_MASS_AMU, DEUTERIUM_MASS_AMU


class TestFGHSolver:
    """Test the Fourier Grid Hamiltonian solver."""

    def test_harmonic_energies_match_analytic(self):
        """FGH energies for harmonic potential match E_n = (n+0.5)hbar*omega."""
        omega_cm = 3000.0
        r = np.linspace(-0.8, 0.8, 256)
        V = harmonic_potential(omega_cm, PROTON_MASS_AMU)

        E, _, _ = fgh_1d(r, V(r), PROTON_MASS_AMU, 5)

        omega_eV = omega_cm / 8065.544
        for n in range(5):
            exact = (n + 0.5) * omega_eV
            assert abs(E[n] - exact) < 1e-6, f"n={n}: {E[n]} vs {exact}"

    def test_isotope_frequency_scaling(self):
        """D frequency should be H frequency / sqrt(2.014/1.008)."""
        r = np.linspace(-0.8, 0.8, 256)
        V = harmonic_potential(3000, PROTON_MASS_AMU)

        E_H, _, _ = fgh_1d(r, V(r), PROTON_MASS_AMU, 2)
        E_D, _, _ = fgh_1d(r, V(r), DEUTERIUM_MASS_AMU, 2)

        ratio = (E_H[1] - E_H[0]) / (E_D[1] - E_D[0])
        expected = np.sqrt(DEUTERIUM_MASS_AMU / PROTON_MASS_AMU)
        assert abs(ratio - expected) < 0.001

    def test_morse_anharmonicity(self):
        """Morse potential should have decreasing level spacings."""
        r = np.linspace(-0.8, 0.8, 256)
        V = morse_potential(De_eV=5.0, beta_inv_angstrom=2.0)

        E, _, _ = fgh_1d(r, V(r), PROTON_MASS_AMU, 5)

        spacings = np.diff(E)
        for i in range(len(spacings) - 1):
            assert spacings[i + 1] < spacings[i], \
                f"Spacing {i+1} ({spacings[i+1]}) should be < spacing {i} ({spacings[i]})"

    def test_wavefunction_normalization(self):
        """Wavefunctions should be normalized: integral |psi|^2 dr = 1."""
        from scipy.integrate import simpson
        r = np.linspace(-0.8, 0.8, 256)
        V = harmonic_potential(3000, PROTON_MASS_AMU)

        _, wfcs, _ = fgh_1d(r, V(r), PROTON_MASS_AMU, 3)

        for i in range(3):
            norm = simpson(wfcs[i]**2, x=r)
            assert abs(norm - 1.0) < 0.01, f"State {i} norm = {norm}"

    def test_wavefunction_orthogonality(self):
        """Different states should be orthogonal."""
        from scipy.integrate import simpson
        r = np.linspace(-0.8, 0.8, 256)
        V = harmonic_potential(3000, PROTON_MASS_AMU)

        _, wfcs, _ = fgh_1d(r, V(r), PROTON_MASS_AMU, 3)

        overlap_01 = simpson(wfcs[0] * wfcs[1], x=r)
        overlap_02 = simpson(wfcs[0] * wfcs[2], x=r)
        assert abs(overlap_01) < 0.01
        assert abs(overlap_02) < 0.01

    def test_small_grid_raises(self):
        """Should raise for too-small grids."""
        with pytest.raises(ValueError):
            fgh_1d(np.linspace(0, 1, 5), np.zeros(5), PROTON_MASS_AMU)


class TestFCOverlaps:
    """Test numerical Franck-Condon overlap computation."""

    def test_identical_potentials_give_identity(self):
        """FC matrix for identical potentials should be ~identity."""
        r = np.linspace(-0.8, 0.8, 256)
        V = harmonic_potential(3000, PROTON_MASS_AMU)

        _, wfcs, _ = fgh_1d(r, V(r), PROTON_MASS_AMU, 3)
        S = compute_fc_overlaps(wfcs, wfcs, r)

        for i in range(3):
            assert abs(S[i, i]) > 0.99, f"Diagonal S[{i},{i}] = {S[i,i]}"

    def test_displacement_reduces_ground_overlap(self):
        """Displacing potentials should reduce |S_00|^2."""
        r = np.linspace(-0.8, 0.8, 256)
        V_small = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=-0.05)
        V_large = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=-0.3)
        V_P = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=0.0)

        _, wfcs_s, _ = fgh_1d(r, V_small(r), PROTON_MASS_AMU, 3)
        _, wfcs_l, _ = fgh_1d(r, V_large(r), PROTON_MASS_AMU, 3)
        _, wfcs_p, _ = fgh_1d(r, V_P(r), PROTON_MASS_AMU, 3)

        S_small = compute_fc_overlaps(wfcs_s, wfcs_p, r)
        S_large = compute_fc_overlaps(wfcs_l, wfcs_p, r)

        assert S_small[0, 0]**2 > S_large[0, 0]**2

    def test_heavier_mass_smaller_overlap(self):
        """D should have smaller FC overlap than H for same displacement."""
        r = np.linspace(-1.0, 1.0, 256)
        V_R = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=-0.3)
        V_P = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=0.3)

        _, wfcs_H_R, _ = fgh_1d(r, V_R(r), PROTON_MASS_AMU, 3)
        _, wfcs_H_P, _ = fgh_1d(r, V_P(r), PROTON_MASS_AMU, 3)
        S_H = compute_fc_overlaps(wfcs_H_R, wfcs_H_P, r)

        _, wfcs_D_R, _ = fgh_1d(r, V_R(r), DEUTERIUM_MASS_AMU, 3)
        _, wfcs_D_P, _ = fgh_1d(r, V_P(r), DEUTERIUM_MASS_AMU, 3)
        S_D = compute_fc_overlaps(wfcs_D_R, wfcs_D_P, r)

        assert S_H[0, 0]**2 > S_D[0, 0]**2


class TestFGHRateEngine:
    """Test end-to-end FGH-based rate calculations."""

    def test_fgh_harmonic_matches_analytic(self):
        """FGH rate with harmonic potentials should match analytic vibronic."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        r = np.linspace(-1.0, 1.0, 256)

        V_R = harmonic_potential(2900, PROTON_MASS_AMU, r_eq=-0.25)
        V_P = harmonic_potential(2900, PROTON_MASS_AMU, r_eq=0.25)

        result_fgh = engine.compute_rate_from_potential(
            r_grid=r, V_reactant=V_R, V_product=V_P,
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
        )
        result_harm = engine.compute_rate(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900, d_DA=2.69, delta_0=0.50,
        )

        # KIE should be within 20% (different tunneling distance treatment)
        ratio = result_fgh.KIE / result_harm.KIE
        assert 0.5 < ratio < 2.0, f"KIE ratio {ratio}"

    def test_fgh_method_label(self):
        """FGH result should have correct method label."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        r = np.linspace(-1.0, 1.0, 128)
        V = harmonic_potential(3000, PROTON_MASS_AMU)

        result = engine.compute_rate_from_potential(
            r_grid=r, V_reactant=V, V_product=V,
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
        )
        assert result.method == "fgh_vibronic"

    def test_fgh_positive_rates(self):
        """FGH should produce positive rates."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        r = np.linspace(-1.0, 1.0, 128)
        V_R = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=-0.2)
        V_P = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=0.2)

        result = engine.compute_rate_from_potential(
            r_grid=r, V_reactant=V_R, V_product=V_P,
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
        )
        assert result.k_H > 0
        assert result.k_D > 0
        assert result.KIE > 1.0
