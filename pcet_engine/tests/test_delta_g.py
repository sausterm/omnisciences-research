"""Tests for driving force (ΔG) extraction from energies and .fchk files."""

import numpy as np
import pytest

from pcet_engine.core.delta_g import (
    compute_zpe,
    delta_g_from_energies,
    DrivingForceResult,
    DeltaGResult,
)
from pcet_engine.core.constants import HARTREE_TO_KCALMOL, HARTREE_TO_CM


class TestComputeZPE:
    """Tests for ZPE computation from frequencies."""

    def test_single_mode(self):
        """ZPE of a single 3000 cm⁻¹ mode = 0.5 * 3000 / 219474.6 hartree."""
        freqs = np.array([3000.0])
        zpe = compute_zpe(freqs)
        expected = 0.5 * 3000.0 / HARTREE_TO_CM
        assert abs(zpe - expected) < 1e-10

    def test_multiple_modes(self):
        """ZPE sums over all real frequencies."""
        freqs = np.array([1000.0, 2000.0, 3000.0])
        zpe = compute_zpe(freqs)
        expected = 0.5 * 6000.0 / HARTREE_TO_CM
        assert abs(zpe - expected) < 1e-10

    def test_ignores_imaginary_frequencies(self):
        """Imaginary (negative) frequencies should be excluded."""
        freqs = np.array([-500.0, 1000.0, 2000.0])
        zpe = compute_zpe(freqs)
        expected = 0.5 * 3000.0 / HARTREE_TO_CM
        assert abs(zpe - expected) < 1e-10

    def test_ignores_near_zero_modes(self):
        """Near-zero modes (<50 cm⁻¹) should be skipped (trans/rot)."""
        freqs = np.array([0.0, 5.0, 30.0, 1000.0, 2000.0])
        zpe = compute_zpe(freqs)
        expected = 0.5 * 3000.0 / HARTREE_TO_CM
        assert abs(zpe - expected) < 1e-10

    def test_empty_frequencies(self):
        """Empty frequency array gives zero ZPE."""
        zpe = compute_zpe(np.array([]))
        assert zpe == 0.0

    def test_water_zpe(self):
        """Water ZPE from experimental frequencies: ~12.9 kcal/mol."""
        freqs = np.array([1595.0, 3657.0, 3756.0])
        zpe = compute_zpe(freqs)
        zpe_kcal = zpe * HARTREE_TO_KCALMOL
        assert 12.0 < zpe_kcal < 14.0

    def test_exclude_indices(self):
        """Excluding a mode reduces ZPE by that mode's contribution."""
        freqs = np.array([1000.0, 2000.0, 3000.0])
        zpe_full = compute_zpe(freqs)
        zpe_excl = compute_zpe(freqs, exclude_indices=[2])  # Exclude 3000 cm⁻¹
        expected_diff = 0.5 * 3000.0 / HARTREE_TO_CM
        assert abs((zpe_full - zpe_excl) - expected_diff) < 1e-10

    def test_exclude_out_of_range_index(self):
        """Out-of-range exclude index is silently ignored."""
        freqs = np.array([1000.0, 2000.0])
        zpe_full = compute_zpe(freqs)
        zpe_excl = compute_zpe(freqs, exclude_indices=[99])
        assert abs(zpe_full - zpe_excl) < 1e-15

    def test_custom_min_freq(self):
        """Custom min_freq_cm threshold."""
        freqs = np.array([80.0, 200.0, 3000.0])
        zpe_default = compute_zpe(freqs)  # 50 cm⁻¹ cutoff: includes 80
        zpe_strict = compute_zpe(freqs, min_freq_cm=100.0)  # Excludes 80
        assert zpe_default > zpe_strict


class TestDeltaGFromEnergies:
    """Tests for driving force from total energies."""

    def test_electronic_only(self):
        """Without ZPE = simple energy difference."""
        E_R = -100.0
        E_P = -99.99
        result = delta_g_from_energies(E_R, E_P)

        assert isinstance(result, DrivingForceResult)
        assert abs(result.delta_E_zpe_hartree - 0.01) < 1e-10
        assert abs(result.delta_E_zpe_kcal - 0.01 * HARTREE_TO_KCALMOL) < 1e-6
        assert result.zpe_reactant == 0.0
        assert result.zpe_product == 0.0
        assert result.delta_zpe == 0.0

    def test_backward_compatible_alias(self):
        """DeltaGResult is an alias for DrivingForceResult."""
        assert DeltaGResult is DrivingForceResult

    def test_exothermic_reaction(self):
        """ΔE < 0 for exothermic (product lower energy)."""
        E_R = -100.0
        E_P = -100.01
        result = delta_g_from_energies(E_R, E_P)
        assert result.delta_E_zpe_kcal < 0
        assert abs(result.delta_E_zpe_kcal - (-0.01 * HARTREE_TO_KCALMOL)) < 1e-6

    def test_with_zpe_correction(self):
        """ZPE correction shifts result."""
        E_R = -100.0
        E_P = -100.0
        freqs_R = np.array([3000.0])  # Higher ZPE
        freqs_P = np.array([2000.0])  # Lower ZPE

        result = delta_g_from_energies(E_R, E_P, freqs_R, freqs_P)
        assert result.delta_zpe < 0
        assert result.delta_E_zpe_kcal < 0
        assert result.n_freqs_reactant == 1
        assert result.n_freqs_product == 1

    def test_zpe_requires_both(self):
        """Must provide both frequency sets or neither."""
        with pytest.raises(ValueError, match="frequencies_product must also be"):
            delta_g_from_energies(-100.0, -99.9, np.array([1000.0]), None)

    def test_typical_pcet_values(self):
        """Typical PCET ΔE should be in range -30 to +10 kcal/mol."""
        E_R = -382.5000
        E_P = -382.5100
        result = delta_g_from_energies(E_R, E_P)
        assert -10.0 < result.delta_E_zpe_kcal < 0.0

    def test_exclude_proton_mode(self):
        """Excluding proton mode from ZPE reduces the ΔZPE contribution."""
        E_R = -100.0
        E_P = -100.0
        # Mode 2 is the "proton stretch" at 3000 cm⁻¹
        freqs_R = np.array([500.0, 1500.0, 3000.0])
        freqs_P = np.array([500.0, 1500.0, 2800.0])

        result_full = delta_g_from_energies(E_R, E_P, freqs_R, freqs_P)
        result_excl = delta_g_from_energies(
            E_R, E_P, freqs_R, freqs_P,
            exclude_modes_reactant=[2],
            exclude_modes_product=[2],
        )

        # The proton mode difference is (2800-3000)/2 in cm⁻¹ of ZPE
        # Excluding it should change delta_zpe
        assert abs(result_full.delta_zpe - result_excl.delta_zpe) > 0.01

    def test_proton_zpe_tracked(self):
        """Excluded proton ZPE is reported for diagnostics."""
        E_R = -100.0
        E_P = -100.0
        freqs_R = np.array([500.0, 3000.0])
        freqs_P = np.array([500.0, 2800.0])

        result = delta_g_from_energies(
            E_R, E_P, freqs_R, freqs_P,
            exclude_modes_reactant=[1],
            exclude_modes_product=[1],
        )
        # Proton ZPE should be ~4.3 kcal/mol (0.5 * 3000 / 219474.6 * 627.5)
        assert result.proton_zpe_reactant > 4.0
        assert result.proton_zpe_product > 3.5


class TestDeltaGDecomposition:
    """Tests for proper decomposition into components."""

    def test_decomposition_sums(self):
        """delta_E_zpe = delta_E_electronic + delta_zpe."""
        E_R = -100.0
        E_P = -99.98
        freqs_R = np.array([3000.0, 1500.0])
        freqs_P = np.array([2800.0, 1400.0])

        result = delta_g_from_energies(E_R, E_P, freqs_R, freqs_P)
        assert abs(result.delta_E_zpe_kcal - (result.delta_E_electronic + result.delta_zpe)) < 1e-8

    def test_zpe_values_positive(self):
        """ZPE should always be positive for real molecules."""
        freqs = np.array([500.0, 1000.0, 1500.0, 3000.0])
        result = delta_g_from_energies(-100.0, -100.0, freqs, freqs)
        assert result.zpe_reactant > 0
        assert result.zpe_product > 0
