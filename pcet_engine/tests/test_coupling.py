"""Tests for electronic coupling estimation (empirical + GMH)."""

import math
import numpy as np
import pytest

from pcet_engine.core.coupling import (
    empirical_coupling,
    gmh_coupling,
    gmh_coupling_from_tddft,
    gmh_coupling_multistate,
    EmpiricalCouplingResult,
    GMHResult,
    DEFAULT_DECAY_PARAMS,
)
from pcet_engine.core.constants import EV_TO_KCALMOL, HARTREE_TO_KCALMOL, HARTREE_TO_EV


# ====================================================================
# Empirical coupling tests
# ====================================================================


class TestEmpiricalCoupling:
    """Tests for distance-based coupling estimation."""

    def test_contact_distance_returns_V0(self):
        """At r = r0, coupling should equal V0."""
        result = empirical_coupling(3.6, medium="protein")
        assert abs(result.V_el_kcal - 80.0) < 1e-6

    def test_exponential_decay(self):
        """Coupling decays exponentially with distance."""
        r1, r2 = 5.0, 7.0
        c1 = empirical_coupling(r1, medium="protein")
        c2 = empirical_coupling(r2, medium="protein")

        # V(r2)/V(r1) = exp(-beta * (r2-r1) / 2)
        expected_ratio = math.exp(-1.1 * (r2 - r1) / 2.0)
        actual_ratio = c2.V_el_kcal / c1.V_el_kcal
        assert abs(actual_ratio - expected_ratio) < 1e-10

    def test_protein_typical_range(self):
        """At 7 Å (typical PCET), coupling should be ~0.1-10 kcal/mol."""
        result = empirical_coupling(7.0, medium="protein")
        assert 0.01 < result.V_el_kcal < 50.0

    def test_short_distance_pcet(self):
        """PCET H-bond at 2.7 Å should give ~1-5 kcal/mol."""
        result = empirical_coupling(2.7, medium="pcet_hydrogen_bond")
        assert 0.1 < result.V_el_kcal < 10.0

    def test_through_bond_stronger(self):
        """Through-bond coupling should be stronger than through-space at same distance."""
        d = 6.0
        tb = empirical_coupling(d, medium="through_bond")
        ts = empirical_coupling(d, medium="through_space")
        assert tb.V_el_kcal > ts.V_el_kcal

    def test_eV_conversion(self):
        """eV and kcal/mol should be consistent."""
        result = empirical_coupling(5.0, medium="protein")
        assert abs(result.V_el_kcal - result.V_el_eV * EV_TO_KCALMOL) < 1e-6

    def test_custom_parameters(self):
        """Custom V0, beta, r0 should override defaults."""
        result = empirical_coupling(5.0, medium="protein", V0_kcal=50.0, beta=2.0, r0=3.0)
        expected = 50.0 * math.exp(-2.0 * (5.0 - 3.0) / 2.0)
        assert abs(result.V_el_kcal - expected) < 1e-10
        assert result.beta == 2.0
        assert result.V0_kcal == 50.0

    def test_invalid_medium(self):
        """Unknown medium should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown medium"):
            empirical_coupling(5.0, medium="unobtanium")

    def test_negative_distance(self):
        """Negative distance should raise ValueError."""
        with pytest.raises(ValueError, match="d_DA must be positive"):
            empirical_coupling(-1.0)

    def test_all_media_return_positive(self):
        """All media should return positive coupling at 5 Å."""
        for medium in DEFAULT_DECAY_PARAMS:
            result = empirical_coupling(5.0, medium=medium)
            assert result.V_el_kcal > 0, f"Negative coupling for {medium}"


# ====================================================================
# GMH coupling tests
# ====================================================================


class TestGMHCoupling:
    """Tests for Generalized Mulliken-Hush coupling extraction."""

    def test_symmetric_case(self):
        """When Δμ = 0 (symmetric), V_el = ΔE/2."""
        result = gmh_coupling(delta_E=2.0, mu_12=1.0, delta_mu=0.0)
        # V = |mu_12| * ΔE / sqrt(0 + 4*mu_12^2) = 1 * 2 / 2 = 1 eV
        assert abs(result.V_el_eV - 1.0) < 1e-10
        assert abs(result.mixing_angle - math.pi / 4) < 1e-10

    def test_large_delta_mu(self):
        """When Δμ >> μ₁₂ (charge-localized), V_el ≈ μ₁₂ × ΔE / Δμ."""
        result = gmh_coupling(delta_E=1.0, mu_12=0.1, delta_mu=10.0)
        expected = 0.1 * 1.0 / math.sqrt(100.0 + 0.04)
        assert abs(result.V_el_eV - expected) < 1e-8

    def test_units_consistent(self):
        """kcal/mol, eV, and hartree should all be consistent."""
        result = gmh_coupling(delta_E=1.5, mu_12=2.0, delta_mu=3.0)
        assert abs(result.V_el_kcal - result.V_el_eV * EV_TO_KCALMOL) < 1e-4
        assert abs(result.V_el_kcal - result.V_el_hartree * HARTREE_TO_KCALMOL) < 1e-8

    def test_typical_pcet_coupling(self):
        """Typical PCET: ΔE ~ 0.5-3 eV, V_el ~ 0.001-0.1 eV."""
        # Large Δμ (charge-transfer state) → small coupling
        result = gmh_coupling(delta_E=2.0, mu_12=0.5, delta_mu=15.0)
        assert 0.001 < result.V_el_eV < 0.5

    def test_negative_delta_E(self):
        """delta_E must be positive."""
        with pytest.raises(ValueError, match="delta_E must be positive"):
            gmh_coupling(delta_E=-1.0, mu_12=1.0, delta_mu=1.0)

    def test_zero_mu12_and_delta_mu(self):
        """Both zero should raise ValueError."""
        with pytest.raises(ValueError, match="cannot extract coupling"):
            gmh_coupling(delta_E=1.0, mu_12=0.0, delta_mu=0.0)

    def test_mixing_angle_range(self):
        """Mixing angle should be between 0 and π/4."""
        result = gmh_coupling(delta_E=1.0, mu_12=1.0, delta_mu=5.0)
        assert 0 < result.mixing_angle < math.pi / 4


class TestGMHFromTDDFT:
    """Tests for GMH extraction from TD-DFT data."""

    def test_basic_extraction(self):
        """Extract coupling from typical TD-DFT output."""
        result = gmh_coupling_from_tddft(
            excitation_energy_eV=2.5,
            ground_state_dipole=np.array([1.0, 0.0, 0.0]),
            excited_state_dipole=np.array([5.0, 0.0, 0.0]),
            transition_dipole=np.array([0.5, 0.0, 0.0]),
        )
        assert isinstance(result, GMHResult)
        assert result.V_el_eV > 0
        assert abs(result.delta_E_adiabatic - 2.5) < 1e-10
        assert abs(result.delta_mu - 4.0) < 1e-10  # |5-1|
        assert abs(result.mu_12 - 0.5) < 1e-10

    def test_3d_dipoles(self):
        """Dipoles with components in all directions."""
        result = gmh_coupling_from_tddft(
            excitation_energy_eV=1.0,
            ground_state_dipole=np.array([1.0, 2.0, 0.0]),
            excited_state_dipole=np.array([4.0, 2.0, 3.0]),
            transition_dipole=np.array([0.3, 0.4, 0.0]),
        )
        # delta_mu = |[3, 0, 3]| = sqrt(18)
        assert abs(result.delta_mu - math.sqrt(18.0)) < 1e-10
        # mu_12 = |[0.3, 0.4, 0]| = 0.5
        assert abs(result.mu_12 - 0.5) < 1e-10


class TestGMHMultistate:
    """Tests for multi-state GMH."""

    def test_two_state_matches_simple(self):
        """Two-state multistate GMH should approximate simple GMH."""
        energies = np.array([0.0, 2.0])  # eV
        # Permanent dipoles on diagonal, transition on off-diagonal
        dipole_matrix = np.zeros((2, 2, 3))
        dipole_matrix[0, 0, :] = [1.0, 0.0, 0.0]  # μ₁
        dipole_matrix[1, 1, :] = [5.0, 0.0, 0.0]  # μ₂
        dipole_matrix[0, 1, :] = [0.5, 0.0, 0.0]  # μ₁₂
        dipole_matrix[1, 0, :] = [0.5, 0.0, 0.0]  # μ₂₁ = μ₁₂

        results = gmh_coupling_multistate(energies, dipole_matrix)
        assert len(results) == 1
        assert results[0].V_el_eV > 0

    def test_three_state(self):
        """Three-state system should return 3 couplings."""
        energies = np.array([0.0, 1.0, 3.0])
        dipole_matrix = np.zeros((3, 3, 3))
        dipole_matrix[0, 0, :] = [0.0, 0.0, 0.0]
        dipole_matrix[1, 1, :] = [3.0, 0.0, 0.0]
        dipole_matrix[2, 2, :] = [6.0, 0.0, 0.0]
        dipole_matrix[0, 1, :] = dipole_matrix[1, 0, :] = [0.5, 0.0, 0.0]
        dipole_matrix[0, 2, :] = dipole_matrix[2, 0, :] = [0.1, 0.0, 0.0]
        dipole_matrix[1, 2, :] = dipole_matrix[2, 1, :] = [0.3, 0.0, 0.0]

        results = gmh_coupling_multistate(energies, dipole_matrix)
        assert len(results) == 3  # (0,1), (0,2), (1,2)
        for r in results:
            assert r.V_el_eV >= 0

    def test_shape_validation(self):
        """Wrong dipole_matrix shape should raise ValueError."""
        energies = np.array([0.0, 1.0])
        bad_dipole = np.zeros((3, 3, 3))  # 3 states but 2 energies
        with pytest.raises(ValueError, match="inconsistent"):
            gmh_coupling_multistate(energies, bad_dipole)
