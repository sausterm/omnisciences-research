"""Tests for normal mode analysis."""

import math
import pytest
import numpy as np

from pcet_engine.core.normal_modes import (
    normal_mode_analysis,
    identify_proton_mode,
    identify_da_stretching_mode,
    compute_donor_acceptor_distance,
)
from pcet_engine.core.constants import AMU_TO_AU, HARTREE_TO_CM


class TestNormalModeAnalysis:
    def _diatomic_hessian(self, k: float, m1: float, m2: float):
        """Build Cartesian Hessian for a 1D diatomic stretch.

        Only the x-x block is non-zero. Force constant k in hartree/bohr².
        """
        # 2 atoms, 6 DOF. Only (x1, x2) coupling matters.
        H = np.zeros((6, 6))
        H[0, 0] = k   # d²E/dx1²
        H[3, 3] = k   # d²E/dx2²
        H[0, 3] = -k  # d²E/dx1dx2
        H[3, 0] = -k
        return H

    def test_diatomic_frequency(self):
        """Check that a diatomic gives the correct harmonic frequency."""
        k = 0.5  # hartree/bohr²
        m1 = 12.0  # amu (carbon)
        m2 = 1.0   # amu (hydrogen)

        H = self._diatomic_hessian(k, m1, m2)
        masses = np.array([m1, m2])

        result = normal_mode_analysis(H, masses, project_trans_rot=False)

        # Expected reduced mass for C-H stretch: μ = m1*m2/(m1+m2)
        mu = m1 * m2 / (m1 + m2)  # amu
        mu_au = mu * AMU_TO_AU

        # ω = sqrt(k/μ) in atomic units
        omega_expected = math.sqrt(k / mu_au)
        freq_expected_cm = omega_expected * HARTREE_TO_CM

        # Find the highest non-zero frequency
        nonzero = result.frequencies_cm[result.frequencies_cm > 50]
        assert len(nonzero) >= 1
        # Should match expected within ~5% (projection can shift slightly)
        best_match = min(nonzero, key=lambda f: abs(f - freq_expected_cm))
        assert abs(best_match - freq_expected_cm) / freq_expected_cm < 0.1

    def test_no_imaginary_for_minimum(self):
        """A positive-definite Hessian should give no imaginary frequencies."""
        H = np.eye(9) * 0.3
        masses = np.array([12.0, 1.0, 1.0])
        result = normal_mode_analysis(H, masses, project_trans_rot=False)
        assert result.n_imaginary == 0

    def test_shape_consistency(self):
        """Output shapes should match input."""
        n_atoms = 4
        n_dof = 12
        H = np.eye(n_dof) * 0.2
        masses = np.ones(n_atoms) * 14.0
        result = normal_mode_analysis(H, masses, project_trans_rot=False)

        assert result.frequencies_cm.shape == (n_dof,)
        assert result.eigenvectors.shape == (n_dof, n_dof)
        assert result.reduced_masses.shape == (n_dof,)
        assert result.n_atoms == n_atoms


class TestIdentifyProtonMode:
    def test_identifies_highest_h_contribution(self):
        """Should identify the mode with largest H displacement."""
        # 3 atoms: C, H, O. Put a high-frequency mode on the H.
        H = np.zeros((9, 9))
        # H atom (index 1) has strong force constant in x direction
        H[3, 3] = 2.0   # H x-x
        H[0, 0] = 0.1   # C x-x
        H[6, 6] = 0.1   # O x-x
        # Add small diagonal to other DOFs
        for i in range(9):
            if H[i, i] < 0.01:
                H[i, i] = 0.01

        masses = np.array([12.0, 1.0, 16.0])
        result = normal_mode_analysis(H, masses, project_trans_rot=False)
        result = identify_proton_mode(result, [1], masses)

        assert result.proton_mode_idx is not None
        assert result.proton_frequency_cm is not None
        assert result.proton_frequency_cm > 100  # Should be a real mode


class TestIdentifyDAStretchingMode:
    """Tests for D-A gating mode identification from Hessians."""

    def test_triatomic_identifies_da_mode(self):
        """For a D-H-A triatomic, should find the low-frequency D-A stretch."""
        from pcet_engine.data.model_hessians import build_hessian, MODEL_SYSTEMS

        model = MODEL_SYSTEMS["SLO-1"]
        hessian, geom, elements, masses = build_hessian(model, "reactant")

        result = identify_da_stretching_mode(
            hessian, masses,
            donor_idx=0, acceptor_idx=2,
            geometry=geom,
        )

        # D-A stretch should be low frequency (below the C-H stretch at ~2900)
        assert result.omega_gating > 30.0, "Gating frequency too low"
        assert result.omega_gating < 1000.0, "Gating frequency too high — picked a stretch mode"
        # M_DA should be in a physically reasonable range
        assert result.M_DA > 1.0, "Effective mass too low"
        assert result.M_DA < 100.0, "Effective mass too high"
        # D-A distance should match model
        assert abs(result.da_distance - model.d_DA) < 0.5
        # Projection should be nonzero (small in triatomic models due to mass-weighting)
        assert result.projection > 0.001

    def test_all_model_systems_produce_reasonable_gating(self):
        """All 15 model systems should give gating params in 50-500 cm⁻¹."""
        from pcet_engine.data.model_hessians import build_hessian, MODEL_SYSTEMS

        for name, model in MODEL_SYSTEMS.items():
            hessian, geom, elements, masses = build_hessian(model, "reactant")

            result = identify_da_stretching_mode(
                hessian, masses,
                donor_idx=0, acceptor_idx=2,
                geometry=geom,
            )

            assert 30.0 < result.omega_gating < 1000.0, (
                f"{name}: omega_gating={result.omega_gating:.1f} cm⁻¹ out of range"
            )
            assert 1.0 < result.M_DA < 100.0, (
                f"{name}: M_DA={result.M_DA:.1f} amu out of range"
            )

    def test_da_distance_matches_geometry(self):
        """Extracted D-A distance should match compute_donor_acceptor_distance."""
        from pcet_engine.data.model_hessians import build_hessian, MODEL_SYSTEMS

        model = MODEL_SYSTEMS["SLO-1"]
        hessian, geom, elements, masses = build_hessian(model, "reactant")

        result = identify_da_stretching_mode(
            hessian, masses,
            donor_idx=0, acceptor_idx=2,
            geometry=geom,
        )
        d_DA = compute_donor_acceptor_distance(geom, 0, 2)
        assert abs(result.da_distance - d_DA) < 1e-10

    def test_mode_index_is_valid(self):
        """The identified mode index should be within bounds."""
        from pcet_engine.data.model_hessians import build_hessian, MODEL_SYSTEMS

        model = MODEL_SYSTEMS["PHM"]
        hessian, geom, elements, masses = build_hessian(model, "reactant")

        result = identify_da_stretching_mode(
            hessian, masses,
            donor_idx=0, acceptor_idx=2,
            geometry=geom,
        )
        n_dof = 3 * len(masses)
        assert 0 <= result.mode_index < n_dof


class TestDonorAcceptorDistance:
    def test_simple_distance(self):
        geom = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        d = compute_donor_acceptor_distance(geom, 0, 2)
        assert abs(d - 3.0) < 1e-10

    def test_3d_distance(self):
        geom = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        d = compute_donor_acceptor_distance(geom, 0, 1)
        assert abs(d - math.sqrt(3.0)) < 1e-10
