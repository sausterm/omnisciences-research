"""Tests for DFT scan file parsers."""

import numpy as np
import pytest
import tempfile
import os
from pcet_engine.parsers.scan_parser import (
    parse_scan_csv, parse_scan_numpy, parse_scan,
    _convert_energy, _convert_distance,
)
from pcet_engine.core.proton_potential import harmonic_potential
from pcet_engine.core.constants import PROTON_MASS_AMU


class TestScanCSV:
    def test_basic_csv(self, tmp_path):
        f = tmp_path / "scan.csv"
        r = np.linspace(-0.5, 0.5, 20)
        V = 5.0 * r**2
        np.savetxt(f, np.column_stack([r, V]))
        r_out, V_out = parse_scan_csv(str(f))
        np.testing.assert_allclose(r_out, r, atol=1e-10)
        np.testing.assert_allclose(V_out, V, atol=1e-10)

    def test_hartree_units(self, tmp_path):
        f = tmp_path / "scan.dat"
        r = np.array([0.0, 0.1, 0.2])
        V_hartree = np.array([0.0, 0.001, 0.004])
        np.savetxt(f, np.column_stack([r, V_hartree]))
        r_out, V_out = parse_scan_csv(str(f), e_unit="hartree")
        np.testing.assert_allclose(V_out, V_hartree * 27.211386245988, atol=1e-6)

    def test_bohr_distances(self, tmp_path):
        f = tmp_path / "scan.dat"
        r_bohr = np.array([0.0, 1.0, 2.0])
        V = np.array([0.0, 0.5, 2.0])
        np.savetxt(f, np.column_stack([r_bohr, V]))
        r_out, V_out = parse_scan_csv(str(f), r_unit="bohr")
        np.testing.assert_allclose(r_out, r_bohr * 0.529177249, atol=1e-6)

    def test_skip_header(self, tmp_path):
        f = tmp_path / "scan.csv"
        with open(f, "w") as fh:
            fh.write("# distance energy\n")
            fh.write("0.0 0.0\n")
            fh.write("0.1 0.5\n")
        r, V = parse_scan_csv(str(f), skip_header=1)
        assert len(r) == 2

    def test_unsorted_input_gets_sorted(self, tmp_path):
        f = tmp_path / "scan.csv"
        r = np.array([0.5, -0.5, 0.0, 0.3, -0.2])
        V = np.array([1.0, 1.0, 0.0, 0.5, 0.2])
        np.savetxt(f, np.column_stack([r, V]))
        r_out, V_out = parse_scan_csv(str(f))
        assert np.all(np.diff(r_out) >= 0)


class TestScanNumpy:
    def test_npy_2d(self, tmp_path):
        f = tmp_path / "scan.npy"
        r = np.linspace(-0.5, 0.5, 20)
        V = 3.0 * r**2
        np.save(f, np.column_stack([r, V]))
        r_out, V_out = parse_scan_numpy(str(f))
        np.testing.assert_allclose(r_out, r, atol=1e-10)

    def test_npz(self, tmp_path):
        f = tmp_path / "scan.npz"
        r = np.linspace(-0.5, 0.5, 20)
        V = 3.0 * r**2
        np.savez(f, r=r, V=V)
        r_out, V_out = parse_scan_numpy(str(f))
        np.testing.assert_allclose(r_out, r, atol=1e-10)

    def test_npz_alt_keys(self, tmp_path):
        f = tmp_path / "scan.npz"
        r = np.linspace(-0.5, 0.5, 10)
        V = r**2
        np.savez(f, distance=r, energy=V)
        r_out, V_out = parse_scan_numpy(str(f))
        assert len(r_out) == 10


class TestAutoDetect:
    def test_csv_autodetect(self, tmp_path):
        f = tmp_path / "scan.csv"
        np.savetxt(f, np.column_stack([np.linspace(0, 1, 5), np.zeros(5)]))
        r, V = parse_scan(str(f))
        assert len(r) == 5

    def test_npy_autodetect(self, tmp_path):
        f = tmp_path / "scan.npy"
        np.save(f, np.column_stack([np.linspace(0, 1, 5), np.zeros(5)]))
        r, V = parse_scan(str(f))
        assert len(r) == 5


class TestUnitConversion:
    def test_energy_eV(self):
        np.testing.assert_allclose(_convert_energy(np.array([1.0]), "eV"), [1.0])

    def test_energy_hartree(self):
        np.testing.assert_allclose(_convert_energy(np.array([1.0]), "hartree"), [27.211386245988], atol=1e-6)

    def test_energy_kcal(self):
        np.testing.assert_allclose(_convert_energy(np.array([23.0605]), "kcal/mol"), [1.0], atol=0.01)

    def test_distance_bohr(self):
        np.testing.assert_allclose(_convert_distance(np.array([1.0]), "bohr"), [0.529177249], atol=1e-6)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError):
            _convert_energy(np.array([1.0]), "furlongs")


class TestScanToRate:
    """End-to-end: CSV scan → FGH → rate prediction."""

    def test_harmonic_scan_to_rate(self, tmp_path):
        from pcet_engine.core.rate_engine import PCETRateEngine
        from pcet_engine.core.fgh_solver import fgh_1d

        # Generate a harmonic potential scan file
        r = np.linspace(-0.8, 0.8, 100)
        V_func = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=-0.2)
        V = V_func(r)

        f = tmp_path / "reactant_scan.csv"
        np.savetxt(f, np.column_stack([r, V]))

        # Parse it back
        r_parsed, V_parsed = parse_scan(str(f))

        # Run through FGH rate engine
        V_product = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=0.2)
        engine = PCETRateEngine()
        result = engine.compute_rate_from_potential(
            r_grid=r_parsed, V_reactant=V_parsed, V_product=V_product,
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
        )
        assert result.k_H > 0
        assert result.KIE > 1.0
        assert result.method == "fgh_vibronic"
