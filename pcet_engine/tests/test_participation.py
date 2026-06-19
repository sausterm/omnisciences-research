"""Tests for participation ratio and geometric tunneling corrections."""

import numpy as np
import pytest

from pcet_engine.core.participation import (
    participation_ratio,
    mode_participation,
    proton_participation,
    effective_tunneling_dimension,
    geometric_tunneling_prefactor,
    tunneling_correction_report,
)


class TestParticipationRatio:
    """Tests for the core participation_ratio function."""

    def test_uniform_eigenvalues(self):
        """Uniform eigenvalues → N_eff = N."""
        vals = np.ones(10)
        assert abs(participation_ratio(vals) - 10.0) < 1e-10

    def test_single_dominant(self):
        """Single dominant eigenvalue → N_eff ≈ 1."""
        vals = np.array([100.0, 0.01, 0.01, 0.01])
        assert participation_ratio(vals) < 1.1

    def test_two_equal(self):
        """Two equal eigenvalues → N_eff = 2."""
        vals = np.array([1.0, 1.0, 0.0, 0.0])
        assert abs(participation_ratio(vals) - 2.0) < 1e-10

    def test_empty_returns_one(self):
        """Zero values → N_eff = 1.0 (fallback)."""
        vals = np.zeros(5)
        assert participation_ratio(vals) == 1.0

    def test_negative_values(self):
        """Negative eigenvalues handled via absolute value."""
        vals = np.array([-4.0, -4.0, -1.0])
        # |vals| = [4, 4, 1], sum = 9, sum_sq = 33
        expected = 81.0 / 33.0
        assert abs(participation_ratio(vals) - expected) < 1e-10

    def test_single_element(self):
        """Single element → N_eff = 1."""
        assert abs(participation_ratio(np.array([5.0])) - 1.0) < 1e-10


class TestModeParticipation:
    """Tests for mode_participation."""

    def test_isolated_atom_mode(self):
        """Mode localized on a single atom gives fraction ≈ 1 for that atom."""
        # 3-atom system: mode concentrated on atom 0
        eigvecs = np.zeros((9, 9))
        eigvecs[0, 0] = 1.0  # mode 0: only atom 0 x-component
        masses = np.array([1.0, 12.0, 16.0])

        fracs = mode_participation(eigvecs, masses, [0])
        assert fracs[0] > 0.99  # atom 0 dominates mode 0

    def test_delocalized_mode(self):
        """Mode shared among all atoms gives fraction proportional to mass."""
        eigvecs = np.zeros((9, 9))
        # mode 1: all atoms contribute equally in x
        for i in range(3):
            eigvecs[3 * i, 1] = 1.0 / np.sqrt(3)
        masses = np.array([1.0, 1.0, 1.0])

        fracs = mode_participation(eigvecs, masses, [0])
        assert abs(fracs[1] - 1.0 / 3.0) < 0.01

    def test_zero_mode(self):
        """Zero eigenvector gives fraction 0."""
        eigvecs = np.zeros((6, 6))
        masses = np.array([1.0, 12.0])
        fracs = mode_participation(eigvecs, masses, [0])
        assert np.all(fracs == 0.0)


class TestProtonParticipation:
    """Tests for proton_participation."""

    def test_basic_system(self):
        """Simple 2-atom system with one proton mode."""
        # 2 atoms: H (idx 0), C (idx 1)
        # Mode 5 (3000 cm⁻¹) is H-C stretch: both atoms contribute
        n_dof = 6
        eigvecs = np.zeros((n_dof, n_dof))
        # High-freq mode: proton (atom 0) and carbon (atom 1) move in x
        eigvecs[0, 5] = 0.95   # proton displacement
        eigvecs[3, 5] = 0.05   # carbon displacement (much smaller)
        # Lower freq modes on carbon
        eigvecs[3, 3] = 1.0
        eigvecs[4, 4] = 1.0
        freqs = np.array([0.0, 0.0, 0.0, 500.0, 1000.0, 3000.0])
        masses = np.array([1.008, 12.0])

        result = proton_participation(eigvecs, freqs, masses, [0], threshold=0.01)

        assert result.n_active_modes > 0
        assert result.n_eff_proton >= 1.0
        assert result.geometric_prefactor > 0

    def test_no_proton_modes(self):
        """System where proton doesn't participate → N_eff = 1."""
        n_dof = 6
        eigvecs = np.zeros((n_dof, n_dof))
        # All modes on atom 1 (not proton)
        for k in range(n_dof):
            eigvecs[3 + (k % 3), k] = 1.0
        freqs = np.array([0.0, 0.0, 0.0, 500.0, 1000.0, 3000.0])
        masses = np.array([1.008, 12.0])

        result = proton_participation(eigvecs, freqs, masses, [0], threshold=0.05)
        assert result.n_active_modes == 0
        assert result.n_eff_proton == 1.0


class TestEffectiveTunnelingDimension:
    """Tests for effective_tunneling_dimension."""

    def test_count_active_modes(self):
        """Counts modes with proton participation above threshold."""
        n_dof = 6
        eigvecs = np.eye(n_dof)
        freqs = np.array([0.0, 0.0, 0.0, 500.0, 1000.0, 3000.0])
        masses = np.array([1.008, 12.0])

        n_dim = effective_tunneling_dimension(eigvecs, freqs, masses, [0])
        # Modes 3,4,5 have freq > 50; proton is atom 0
        # eigvecs = identity: modes 0,1,2 are on atom 0 (freq 0 → excluded)
        # modes 3,4,5 are on atom 1 (not proton) → n_dim should be 0
        # Actually mode 3 is row index 3 → atom 1 x. So proton gets 0.
        assert n_dim == 0


class TestGeometricPrefactor:
    """Tests for geometric_tunneling_prefactor."""

    def test_identity_at_d_ref(self):
        """N_eff = d_ref → prefactor = 1.0."""
        assert abs(geometric_tunneling_prefactor(3.0, 3.0) - 1.0) < 1e-10

    def test_enhancement(self):
        """N_eff > d_ref → prefactor > 1.0."""
        pf = geometric_tunneling_prefactor(12.0, 3.0)
        assert abs(pf - 2.0) < 1e-10  # sqrt(12/3) = 2

    def test_suppression(self):
        """N_eff < d_ref → prefactor < 1.0."""
        pf = geometric_tunneling_prefactor(1.0, 3.0)
        assert pf < 1.0

    def test_zero_neff(self):
        """N_eff = 0 → prefactor = 1.0 (fallback)."""
        assert geometric_tunneling_prefactor(0.0, 3.0) == 1.0


class TestTunnelingCorrectionReport:
    """Tests for tunneling_correction_report."""

    def test_report_has_expected_keys(self):
        """Report dict contains all expected fields."""
        eigvecs = np.eye(6)
        freqs = np.array([0.0, 0.0, 0.0, 500.0, 1000.0, 3000.0])
        masses = np.array([1.008, 12.0])

        report = tunneling_correction_report(eigvecs, freqs, masses, [0])

        expected_keys = [
            "n_eff_overall",
            "n_eff_proton",
            "n_active_modes",
            "geometric_prefactor",
            "active_mode_indices",
            "active_mode_frequencies_cm",
            "active_mode_proton_fractions",
            "dominant_mode_idx",
            "dominant_mode_proton_fraction",
        ]
        for key in expected_keys:
            assert key in report, f"Missing key: {key}"


class TestIntegration:
    """Integration tests combining normal mode analysis with participation."""

    def test_diatomic_hessian(self):
        """Construct a simple Hessian and verify participation ratio pipeline."""
        from pcet_engine.core.normal_modes import normal_mode_analysis
        from pcet_engine.core.constants import AMU_TO_AU

        # Simple diatomic: H-C with spring constant k
        masses = np.array([1.008, 12.0])
        k = 0.5  # hartree/bohr²

        # Hessian for 1D stretch along x, embedded in 3D
        hessian = np.zeros((6, 6))
        hessian[0, 0] = k
        hessian[3, 3] = k
        hessian[0, 3] = -k
        hessian[3, 0] = -k

        nma = normal_mode_analysis(hessian, masses)

        # Now compute participation
        result = proton_participation(
            nma.eigenvectors, nma.frequencies_cm, masses, [0]
        )

        # Should find at least one proton-active mode (the H-C stretch)
        assert result.n_active_modes >= 1
        assert result.n_eff_proton >= 1.0
        assert result.geometric_prefactor > 0

    def test_symmetric_3atom(self):
        """3-atom linear system: participation ratio should reflect delocalization."""
        masses = np.array([1.008, 12.0, 1.008])  # H-C-H
        k = 0.5

        hessian = np.zeros((9, 9))
        # H1-C bond (x direction)
        hessian[0, 0] = k
        hessian[3, 3] = 2 * k  # C coupled to both H
        hessian[6, 6] = k
        hessian[0, 3] = -k
        hessian[3, 0] = -k
        hessian[3, 6] = -k
        hessian[6, 3] = -k

        from pcet_engine.core.normal_modes import normal_mode_analysis

        nma = normal_mode_analysis(hessian, masses)

        # Both H atoms participate
        result = proton_participation(
            nma.eigenvectors, nma.frequencies_cm, masses, [0, 2]
        )

        report = tunneling_correction_report(
            nma.eigenvectors, nma.frequencies_cm, masses, [0, 2]
        )
        assert isinstance(report, dict)
        assert report["n_eff_proton"] >= 1.0


class TestNeffRateEngineIntegration:
    """Tests for N_eff correction wired into the rate engine."""

    def test_neff_correction_enhances_rate(self):
        """Providing n_eff > 3 should increase the rate (shorter tunnel distance)."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        params = dict(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.50,
        )

        result_no_corr = engine.compute_rate(**params)
        result_with_corr = engine.compute_rate(**params, n_eff=12.0)

        # n_eff=12 → shorter tunnel distance → enhanced tunneling → higher rate
        assert result_with_corr.k_H > result_no_corr.k_H
        assert result_with_corr.k_D > result_no_corr.k_D

    def test_neff_3_no_change(self):
        """n_eff=3 (d_ref) should give no change in tunneling distance."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        params = dict(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.50,
        )

        result_no = engine.compute_rate(**params)
        result_3 = engine.compute_rate(**params, n_eff=3.0)

        assert abs(result_3.k_H / result_no.k_H - 1.0) < 1e-6
        assert abs(result_3.KIE / result_no.KIE - 1.0) < 1e-6

    def test_neff_shifts_kie(self):
        """N_eff correction via tunnel distance should shift KIE (mass-dependent).

        When N_eff > 3, the tunnel distance is shortened. Since the FC overlap
        scales as exp(-α δ²) where α ∝ mass, the correction affects H and D
        differently, shifting the KIE.
        """
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        params = dict(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.50,
        )

        result_no = engine.compute_rate(**params)
        result_6 = engine.compute_rate(**params, n_eff=6.0)

        # With n_eff=6, δ_eff = δ₀ × sqrt(3/6) = 0.707 × δ₀
        # Shorter tunnel distance → smaller KIE (less tunneling advantage for H)
        assert result_6.KIE < result_no.KIE

    def test_neff_prefactor_mode_preserves_kie(self):
        """Prefactor mode should preserve KIE (same correction to H and D)."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        params = dict(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.50,
        )

        result_no = engine.compute_rate(**params)
        result_pf = engine.compute_rate(**params, n_eff=12.0, n_eff_mode="prefactor")

        # Prefactor mode: k_corr = k × sqrt(N_eff/3), same for H and D
        assert abs(result_pf.KIE / result_no.KIE - 1.0) < 1e-6
        expected_ratio = np.sqrt(12.0 / 3.0)
        assert abs(result_pf.k_H / result_no.k_H - expected_ratio) < 0.01

    def test_neff_stored_in_result(self):
        """PCETResult should store the n_eff and geometric_prefactor."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, n_eff=5.0,
        )

        assert result.n_eff == 5.0
        assert abs(result.geometric_prefactor - np.sqrt(5.0 / 3.0)) < 1e-10

    def test_default_neff_is_one(self):
        """Without n_eff, PCETResult should have n_eff=1.0, prefactor=1.0."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69,
        )

        assert result.n_eff == 1.0
        assert result.geometric_prefactor == 1.0

    def test_hessian_pipeline_computes_neff(self):
        """compute_rate_from_hessian should compute and apply N_eff."""
        from pcet_engine.core.rate_engine import PCETRateEngine

        engine = PCETRateEngine()

        # Build a simple 3-atom Hessian: D-H-A (donor, proton, acceptor)
        masses = np.array([12.0, 1.008, 16.0])
        k_DH = 0.5  # hartree/bohr²
        k_HA = 0.3

        hessian = np.zeros((9, 9))
        # D-H bond (x direction)
        hessian[0, 0] = k_DH
        hessian[3, 3] = k_DH + k_HA  # proton coupled to both
        hessian[6, 6] = k_HA
        hessian[0, 3] = -k_DH
        hessian[3, 0] = -k_DH
        hessian[3, 6] = -k_HA
        hessian[6, 3] = -k_HA
        # Add small diagonal for other DOFs
        for i in range(9):
            if hessian[i, i] < 0.01:
                hessian[i, i] = 0.01

        geom_R = np.array([
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [2.69, 0.0, 0.0],
        ])
        geom_P = np.array([
            [0.0, 0.0, 0.0],
            [1.73, 0.0, 0.0],
            [2.69, 0.0, 0.0],
        ])

        result = engine.compute_rate_from_hessian(
            hessian_R=hessian, hessian_P=hessian,
            geom_R=geom_R, geom_P=geom_P,
            masses=masses,
            proton_idx=1, donor_idx=0, acceptor_idx=2,
            V_el=0.6, delta_G=-5.4,
            lambda_outer=15.0,
            delta_0=0.50,
        )

        # Should have computed participation
        assert result.participation is not None
        assert result.n_eff >= 1.0
        assert result.geometric_prefactor > 0

    def test_benchmarks_with_neff(self):
        """Benchmark runner should work with N_eff correction."""
        from pcet_engine.benchmarks.systems import run_benchmarks

        results = run_benchmarks(method="vibronic_multi", verbose=False, n_eff=4.0)
        assert len(results) == 28
        for name, res in results.items():
            assert res["result"].k_H > 0
            assert res["n_eff"] == 4.0
