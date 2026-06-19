"""Tests for the Hessian-to-rate pipeline (Phase 2 validation)."""

import math
import os
import pytest
import numpy as np

from pcet_engine.data.model_hessians import (
    MODEL_SYSTEMS,
    build_hessian,
    write_orca_hess,
    generate_all_model_hessians,
    _force_constant_from_frequency,
    _build_geometry,
)
from pcet_engine.core.normal_modes import (
    normal_mode_analysis,
    identify_proton_mode,
    compute_donor_acceptor_distance,
)
from pcet_engine.core.rate_engine import PCETRateEngine
from pcet_engine.core.marcus import reorganization_energy_from_hessians
from pcet_engine.core.constants import ANGSTROM_TO_BOHR, HARTREE_TO_KCALMOL
from pcet_engine.parsers import parse_orca_hess
from pcet_engine.benchmarks.systems import BENCHMARK_SYSTEMS


class TestModelHessianBuilder:
    def test_build_reactant_hessian(self):
        """Reactant Hessian should be symmetric positive-semidefinite."""
        model = MODEL_SYSTEMS["SLO-1"]
        hess, geom, elements, masses = build_hessian(model, "reactant")
        assert hess.shape == (9, 9)
        assert np.allclose(hess, hess.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(hess)
        assert all(ev >= -1e-8 for ev in eigvals)

    def test_build_product_hessian(self):
        """Product Hessian should be symmetric positive-semidefinite."""
        model = MODEL_SYSTEMS["SLO-1"]
        hess, geom, elements, masses = build_hessian(model, "product")
        assert hess.shape == (9, 9)
        assert np.allclose(hess, hess.T, atol=1e-12)

    def test_geometry_donor_acceptor_distance(self):
        """D-A distance should match the model specification."""
        for name, model in MODEL_SYSTEMS.items():
            _, geom, _, _ = build_hessian(model, "reactant")
            d_DA = compute_donor_acceptor_distance(geom, 0, 2)
            assert abs(d_DA - model.d_DA) < 0.02, f"{name}: d_DA={d_DA:.3f} vs {model.d_DA}"

    def test_correct_elements_and_masses(self):
        """Elements and masses should correspond to the model atoms."""
        model = MODEL_SYSTEMS["RNR"]
        _, _, elements, masses = build_hessian(model, "reactant")
        assert elements == ["S", "H", "C"]
        assert abs(masses[0] - 31.972) < 0.01
        assert abs(masses[1] - 1.008) < 0.01
        assert abs(masses[2] - 12.0) < 0.01

    def test_force_constant_reasonable(self):
        """Force constant from 3000 cm⁻¹ C-H stretch should be ~0.3 hartree/bohr²."""
        k = _force_constant_from_frequency(3000.0, 12.0, 1.008)
        assert 0.1 < k < 1.0


class TestFrequencyExtraction:
    """Validate that normal mode analysis extracts correct frequencies from model Hessians."""

    @pytest.mark.parametrize("name", list(MODEL_SYSTEMS.keys()))
    def test_proton_frequency_within_5_percent(self, name):
        """Extracted proton frequency should be within 5% of the target."""
        model = MODEL_SYSTEMS[name]
        hess, geom, _, masses = build_hessian(model, "reactant")
        nma = normal_mode_analysis(hess, masses)
        nma = identify_proton_mode(nma, [1], masses)

        assert nma.proton_frequency_cm is not None
        rel_err = abs(nma.proton_frequency_cm - model.omega_DH) / model.omega_DH
        assert rel_err < 0.05, (
            f"{name}: ω_H={nma.proton_frequency_cm:.0f} vs target {model.omega_DH}"
        )

    @pytest.mark.parametrize("name", list(MODEL_SYSTEMS.keys()))
    def test_three_real_frequencies(self, name):
        """Triatomic should have exactly 3 real vibrational frequencies."""
        model = MODEL_SYSTEMS[name]
        hess, _, _, masses = build_hessian(model, "reactant")
        nma = normal_mode_analysis(hess, masses)
        real_freqs = [f for f in nma.frequencies_cm if f > 50.0]
        assert len(real_freqs) == 3

    def test_proton_mode_is_highest_frequency(self):
        """The identified proton mode should be the highest-frequency mode."""
        model = MODEL_SYSTEMS["SLO-1"]
        hess, _, _, masses = build_hessian(model, "reactant")
        nma = normal_mode_analysis(hess, masses)
        nma = identify_proton_mode(nma, [1], masses)
        real_freqs = sorted([f for f in nma.frequencies_cm if f > 50.0])
        assert nma.proton_frequency_cm == pytest.approx(real_freqs[-1], rel=1e-6)


class TestOrcaHessRoundtrip:
    """Validate write → parse roundtrip for ORCA .hess files."""

    def test_roundtrip_hessian(self, tmp_path):
        """Parsed Hessian should match the original within precision."""
        model = MODEL_SYSTEMS["SLO-1"]
        hess, geom, elements, masses = build_hessian(model, "reactant")
        filepath = str(tmp_path / "test.hess")
        write_orca_hess(filepath, hess, geom, elements, masses)

        data = parse_orca_hess(filepath)
        assert np.allclose(data.hessian, hess, atol=1e-5)
        assert np.allclose(data.geometry, geom, atol=1e-3)
        assert data.n_atoms == 3

    def test_roundtrip_preserves_frequency(self, tmp_path):
        """Frequency from parsed Hessian should match original."""
        model = MODEL_SYSTEMS["AADH"]
        hess, geom, elements, masses = build_hessian(model, "reactant")
        filepath = str(tmp_path / "test.hess")
        write_orca_hess(filepath, hess, geom, elements, masses)

        data = parse_orca_hess(filepath)
        nma = normal_mode_analysis(data.hessian, data.masses)
        nma = identify_proton_mode(nma, [1], data.masses)
        assert abs(nma.proton_frequency_cm - model.omega_DH) / model.omega_DH < 0.05

    def test_generate_all_creates_files(self, tmp_path):
        """generate_all_model_hessians should create files for all systems."""
        output_dir = str(tmp_path / "hess_files")
        results = generate_all_model_hessians(output_dir)
        assert len(results) == 15
        for name, info in results.items():
            assert os.path.exists(info["reactant_file"])
            assert os.path.exists(info["product_file"])


class TestReorganizationEnergy:
    """Test the four-point reorganization energy calculation."""

    def test_zero_for_identical_geometries(self):
        """λ should be zero when R and P geometries are identical."""
        hess = np.eye(9) * 0.5
        geom = np.zeros(9)
        masses = np.ones(3) * 12.0
        lam_f, lam_b = reorganization_energy_from_hessians(hess, hess, geom, geom, masses)
        assert abs(lam_f) < 1e-10
        assert abs(lam_b) < 1e-10

    def test_positive_for_displaced_geometries(self):
        """λ should be positive for displaced heavy atoms."""
        hess = np.eye(6) * 0.5
        geom_R = np.zeros(6)
        geom_P = np.array([0.1, 0.0, 0.0, -0.1, 0.0, 0.0])
        masses = np.array([12.0, 16.0])
        lam_f, lam_b = reorganization_energy_from_hessians(hess, hess, geom_R, geom_P, masses)
        assert lam_f > 0
        assert lam_b > 0

    def test_exclude_atoms(self):
        """Excluding an atom should reduce λ."""
        hess = np.eye(9) * 0.5
        geom_R = np.zeros(9)
        geom_P = np.array([0.1, 0.0, 0.0, 0.5, 0.0, 0.0, -0.1, 0.0, 0.0])
        masses = np.array([12.0, 1.0, 16.0])

        lam_f_full, _ = reorganization_energy_from_hessians(hess, hess, geom_R, geom_P, masses)
        lam_f_excl, _ = reorganization_energy_from_hessians(
            hess, hess, geom_R, geom_P, masses, exclude_atoms=[1]
        )
        assert lam_f_excl < lam_f_full

    def test_model_inner_lambda_small(self):
        """Model triatomic inner-sphere λ should be small (heavy atoms barely move)."""
        model = MODEL_SYSTEMS["SLO-1"]
        hess_R, geom_R, _, masses = build_hessian(model, "reactant")
        hess_P, geom_P, _, _ = build_hessian(model, "product")

        geom_R_flat = (geom_R * ANGSTROM_TO_BOHR).flatten()
        geom_P_flat = (geom_P * ANGSTROM_TO_BOHR).flatten()

        lam_f, lam_b = reorganization_energy_from_hessians(
            hess_R, hess_P, geom_R_flat, geom_P_flat, masses,
            exclude_atoms=[1],
        )
        lambda_inner = 0.5 * (lam_f + lam_b) * HARTREE_TO_KCALMOL
        # Should be < 5 kcal/mol for the triatomic model
        assert lambda_inner < 5.0


class TestEndToEndPipeline:
    """Full Hessian-to-rate pipeline tests."""

    def setup_method(self):
        self.engine = PCETRateEngine()

    @pytest.mark.parametrize("name", list(MODEL_SYSTEMS.keys()))
    def test_pipeline_produces_positive_rates(self, name):
        """Pipeline should produce positive, finite rates for all systems."""
        model = MODEL_SYSTEMS[name]
        bench = BENCHMARK_SYSTEMS[name]

        hess_R, geom_R, _, masses = build_hessian(model, "reactant")
        hess_P, geom_P, _, _ = build_hessian(model, "product")

        result = self.engine.compute_rate_from_hessian(
            hessian_R=hess_R, hessian_P=hess_P,
            geom_R=geom_R, geom_P=geom_P,
            masses=masses,
            proton_idx=1, donor_idx=0, acceptor_idx=2,
            V_el=bench.V_el, delta_G=bench.delta_G,
            lambda_outer=bench.lambda_reorg * 0.6,
            delta_0=bench.delta_0,
        )

        assert result.k_H > 0 and math.isfinite(result.k_H)
        assert result.k_D > 0 and math.isfinite(result.k_D)
        assert result.KIE > 1.0

    @pytest.mark.parametrize("name", list(MODEL_SYSTEMS.keys()))
    def test_kie_within_factor_of_3(self, name):
        """Pipeline KIE should be within a factor of 3 of experiment."""
        model = MODEL_SYSTEMS[name]
        bench = BENCHMARK_SYSTEMS[name]

        hess_R, geom_R, _, masses = build_hessian(model, "reactant")
        hess_P, geom_P, _, _ = build_hessian(model, "product")

        result = self.engine.compute_rate_from_hessian(
            hessian_R=hess_R, hessian_P=hess_P,
            geom_R=geom_R, geom_P=geom_P,
            masses=masses,
            proton_idx=1, donor_idx=0, acceptor_idx=2,
            V_el=bench.V_el, delta_G=bench.delta_G,
            lambda_outer=bench.lambda_reorg * 0.6,
            delta_0=bench.delta_0,
        )

        ratio = result.KIE / bench.KIE_exp
        assert 1 / 3.0 < ratio < 3.0, (
            f"{name}: KIE={result.KIE:.1f} vs exp={bench.KIE_exp:.0f} (ratio={ratio:.2f})"
        )

    def test_pipeline_from_parsed_files(self, tmp_path):
        """Pipeline should work with files written and re-parsed."""
        model = MODEL_SYSTEMS["SLO-1"]
        bench = BENCHMARK_SYSTEMS["SLO-1"]

        hess_R, geom_R, elements, masses = build_hessian(model, "reactant")
        hess_P, geom_P, _, _ = build_hessian(model, "product")

        r_file = str(tmp_path / "reactant.hess")
        p_file = str(tmp_path / "product.hess")
        write_orca_hess(r_file, hess_R, geom_R, elements, masses)
        write_orca_hess(p_file, hess_P, geom_P, elements, masses)

        data_R = parse_orca_hess(r_file)
        data_P = parse_orca_hess(p_file)

        result = self.engine.compute_rate_from_hessian(
            hessian_R=data_R.hessian, hessian_P=data_P.hessian,
            geom_R=data_R.geometry, geom_P=data_P.geometry,
            masses=data_R.masses,
            proton_idx=1, donor_idx=0, acceptor_idx=2,
            V_el=bench.V_el, delta_G=bench.delta_G,
            lambda_outer=bench.lambda_reorg * 0.6,
            delta_0=bench.delta_0,
        )

        assert result.k_H > 0
        assert result.KIE > 1.0

    def test_extracted_bond_lengths(self):
        """Pipeline should extract correct bond lengths from geometry."""
        model = MODEL_SYSTEMS["SLO-1"]
        bench = BENCHMARK_SYSTEMS["SLO-1"]

        hess_R, geom_R, _, masses = build_hessian(model, "reactant")
        hess_P, geom_P, _, _ = build_hessian(model, "product")

        # Check D-H distance in reactant
        r_DH = compute_donor_acceptor_distance(geom_R, 0, 1)
        assert abs(r_DH - model.r_DH) < 0.01

        # Check A-H distance in product
        r_AH = compute_donor_acceptor_distance(geom_P, 2, 1)
        assert abs(r_AH - model.r_AH) < 0.05
