"""Tests for the FGH (Fourier Grid Hamiltonian) solver."""

import math
import numpy as np
import pytest

from pcet_engine.core.fgh import (
    solve_1d_schrodinger,
    harmonic_potential,
    morse_potential,
    build_proton_potentials,
    fgh_franck_condon_table,
    numerical_fc_overlap,
)
from pcet_engine.core.constants import AMU_TO_AU, CM_TO_HARTREE, ANGSTROM_TO_BOHR, HBAR_AU


class TestFGHSolver:
    """Test the 1D Schrödinger solver against analytic results."""

    def test_harmonic_eigenvalues(self):
        """FGH eigenvalues for harmonic potential should match (n+½)ℏω."""
        omega_cm = 3000.0  # cm⁻¹
        omega_au = omega_cm * CM_TO_HARTREE
        mass_au = 1.00782503207 * AMU_TO_AU  # proton

        grid = np.linspace(-3.0, 3.0, 512)  # bohr
        V = harmonic_potential(grid, omega_au, mass_au)

        result = solve_1d_schrodinger(grid, V, mass_au, n_states=5)

        for n in range(5):
            expected = (n + 0.5) * omega_au
            assert abs(result.energies[n] - expected) / expected < 0.01, (
                f"n={n}: FGH={result.energies[n]:.6f}, exact={(n+0.5)*omega_au:.6f}"
            )

    def test_harmonic_spacing_is_uniform(self):
        """Harmonic oscillator energy levels should be equally spaced."""
        omega_au = 3000.0 * CM_TO_HARTREE
        mass_au = 1.00782503207 * AMU_TO_AU

        grid = np.linspace(-3.0, 3.0, 512)
        V = harmonic_potential(grid, omega_au, mass_au)
        result = solve_1d_schrodinger(grid, V, mass_au, n_states=5)

        spacings = np.diff(result.energies)
        for i in range(len(spacings) - 1):
            assert abs(spacings[i] - spacings[0]) / spacings[0] < 0.01

    def test_morse_anharmonicity(self):
        """Morse energy levels should be closer together at higher n."""
        omega_au = 3000.0 * CM_TO_HARTREE
        mass_au = 1.00782503207 * AMU_TO_AU
        D_e = 20.0 * omega_au  # typical X-H dissociation

        grid = np.linspace(-2.0, 5.0, 512)
        V = morse_potential(grid, omega_au, mass_au, D_e)
        result = solve_1d_schrodinger(grid, V, mass_au, n_states=5)

        spacings = np.diff(result.energies)
        # Morse: spacings decrease with n (anharmonic)
        for i in range(len(spacings) - 1):
            assert spacings[i + 1] < spacings[i], (
                f"Morse spacing should decrease: Δε_{i}={spacings[i]:.6f} vs Δε_{i+1}={spacings[i+1]:.6f}"
            )

    def test_morse_ground_state_matches_harmonic(self):
        """Morse ground state should be close to ½ℏω for deep well."""
        omega_au = 3000.0 * CM_TO_HARTREE
        mass_au = 1.00782503207 * AMU_TO_AU
        D_e = 50.0 * omega_au  # very deep → nearly harmonic

        grid = np.linspace(-2.0, 5.0, 512)
        V = morse_potential(grid, omega_au, mass_au, D_e)
        result = solve_1d_schrodinger(grid, V, mass_au, n_states=1)

        expected = 0.5 * omega_au
        assert abs(result.energies[0] - expected) / expected < 0.02

    def test_wavefunction_normalization(self):
        """Wavefunctions should be normalized."""
        omega_au = 3000.0 * CM_TO_HARTREE
        mass_au = 1.00782503207 * AMU_TO_AU

        grid = np.linspace(-3.0, 3.0, 512)
        V = harmonic_potential(grid, omega_au, mass_au)
        result = solve_1d_schrodinger(grid, V, mass_au, n_states=3)

        dx = grid[1] - grid[0]
        for n in range(3):
            norm = np.sum(result.wavefunctions[:, n] ** 2) * dx
            assert abs(norm - 1.0) < 0.01, f"n={n}: norm={norm}"

    def test_wavefunction_orthogonality(self):
        """Different eigenstates should be orthogonal."""
        omega_au = 3000.0 * CM_TO_HARTREE
        mass_au = 1.00782503207 * AMU_TO_AU

        grid = np.linspace(-3.0, 3.0, 512)
        V = harmonic_potential(grid, omega_au, mass_au)
        result = solve_1d_schrodinger(grid, V, mass_au, n_states=3)

        dx = grid[1] - grid[0]
        for i in range(3):
            for j in range(i + 1, 3):
                overlap = abs(np.sum(result.wavefunctions[:, i] * result.wavefunctions[:, j]) * dx)
                assert overlap < 0.01, f"<{i}|{j}> = {overlap}"


class TestFGHOverlaps:
    """Test FC overlap calculation from FGH wavefunctions."""

    def test_self_overlap_is_one(self):
        """<ψ_n|ψ_n> should be 1 for same potential."""
        omega_au = 3000.0 * CM_TO_HARTREE
        mass_au = 1.00782503207 * AMU_TO_AU

        grid = np.linspace(-3.0, 3.0, 512)
        V = harmonic_potential(grid, omega_au, mass_au)
        result = solve_1d_schrodinger(grid, V, mass_au, n_states=3)

        for n in range(3):
            S_sq = numerical_fc_overlap(result, result, n, n)
            assert abs(S_sq - 1.0) < 0.01, f"|<{n}|{n}>|² = {S_sq}"

    def test_overlap_decreases_with_displacement(self):
        """FC overlap should decrease as tunneling distance increases."""
        omega_cm = 3000.0
        mass_amu = 1.00782503207

        overlaps_at_delta = []
        for delta in [0.2, 0.4, 0.6]:
            table, _, _, _, _ = fgh_franck_condon_table(
                omega_cm, omega_cm, mass_amu, delta,
                n_reactant_states=1, n_product_states=1,
                potential_type="harmonic",
            )
            overlaps_at_delta.append(table[0, 0])

        for i in range(len(overlaps_at_delta) - 1):
            assert overlaps_at_delta[i + 1] < overlaps_at_delta[i]

    def test_fgh_harmonic_matches_analytic(self):
        """FGH with harmonic potential should match analytic FC overlaps."""
        from pcet_engine.core.vibronic import franck_condon_overlap

        omega_cm = 3000.0
        omega_au = omega_cm * CM_TO_HARTREE
        mass_amu = 1.00782503207
        mass_au = mass_amu * AMU_TO_AU
        delta_ang = 0.4
        delta_bohr = delta_ang * ANGSTROM_TO_BOHR

        # Analytic
        S_00_analytic = franck_condon_overlap(omega_au, omega_au, mass_au, delta_bohr, 0, 0)

        # FGH numerical
        table, _, _, _, _ = fgh_franck_condon_table(
            omega_cm, omega_cm, mass_amu, delta_ang,
            n_reactant_states=1, n_product_states=1,
            potential_type="harmonic", n_grid=512,
        )
        S_00_fgh = table[0, 0]

        assert abs(S_00_fgh - S_00_analytic) / (S_00_analytic + 1e-30) < 0.05, (
            f"FGH={S_00_fgh:.6f} vs analytic={S_00_analytic:.6f}"
        )

    def test_morse_overlap_differs_from_harmonic(self):
        """Morse FC overlaps should differ from harmonic at large displacement."""
        omega_cm = 3000.0
        mass_amu = 1.00782503207
        delta_ang = 0.5

        table_harm, _, _, _, _ = fgh_franck_condon_table(
            omega_cm, omega_cm, mass_amu, delta_ang,
            n_reactant_states=1, n_product_states=1,
            potential_type="harmonic", n_grid=512,
        )
        table_morse, _, _, _, _ = fgh_franck_condon_table(
            omega_cm, omega_cm, mass_amu, delta_ang,
            n_reactant_states=1, n_product_states=1,
            potential_type="morse", n_grid=512,
        )

        # They should differ (Morse has longer tails → larger overlap)
        assert table_morse[0, 0] != pytest.approx(table_harm[0, 0], rel=0.01)


class TestFGHRateEngine:
    """Test the FGH-based rate calculation through PCETRateEngine."""

    def test_morse_method_runs(self):
        """vibronic_multi_morse should produce valid rate constants."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.5,
            method="vibronic_multi_morse",
        )

        assert result.k_H > 0
        assert result.k_D > 0
        assert result.KIE > 1.0

    def test_fgh_harmonic_matches_analytic_rate(self):
        """vibronic_multi_fgh should match vibronic_multi for same parameters."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        params = dict(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.5,
        )

        r_analytic = engine.compute_rate(**params, method="vibronic_multi")
        r_fgh = engine.compute_rate(**params, method="vibronic_multi_fgh")

        # KIE should match within 10%
        assert abs(r_fgh.KIE / r_analytic.KIE - 1.0) < 0.10, (
            f"FGH KIE={r_fgh.KIE:.1f} vs analytic KIE={r_analytic.KIE:.1f}"
        )

    def test_morse_kie_for_slo1(self):
        """Morse method should produce reasonable KIE for SLO-1."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.521,
            method="vibronic_multi_morse",
        )

        # Should be in the right ballpark for SLO-1 (exp KIE=81)
        assert 30 < result.KIE < 200, f"KIE={result.KIE}"

    def test_deuterium_has_smaller_kie_morse(self):
        """Morse method: D should always give smaller KIE because heavier."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.5,
            method="vibronic_multi_morse",
        )

        assert result.k_H > result.k_D
        assert result.KIE > 1.0
