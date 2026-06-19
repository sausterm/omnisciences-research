"""Tests for quantum chemistry file parsers.

Uses synthetic test data since real .fchk/.hess files are large.
Tests cover format parsing, data extraction, and edge cases.
"""

import math
import pytest
import numpy as np
import tempfile
from pathlib import Path

from pcet_engine.parsers.gaussian_fchk import parse_gaussian_fchk, _lower_triangle_to_full
from pcet_engine.parsers.orca_hess import parse_orca_hess
from pcet_engine.parsers.base import QCData


# =====================================================================
# Fixtures: generate synthetic test files
# =====================================================================

def _make_fchk_content(n_atoms: int = 3) -> str:
    """Generate a minimal valid .fchk file for testing.

    Creates a water molecule (H2O) with a synthetic Hessian.
    """
    # Water: O at origin, H at (0, ±0.757, 0.587) angstrom
    # In bohr: multiply by 1.8897
    coords_bohr = [
        0.0, 0.0, 0.0,       # O
        0.0, 1.431, 1.110,    # H
        0.0, -1.431, 1.110,   # H
    ]

    n_dof = 3 * n_atoms
    n_hess = n_dof * (n_dof + 1) // 2

    # Generate a positive-definite Hessian (lower triangle)
    # Use a simple diagonal-dominant matrix
    hess_full = np.eye(n_dof) * 0.5
    # Add some off-diagonal elements
    for i in range(n_dof - 1):
        hess_full[i, i + 1] = 0.05
        hess_full[i + 1, i] = 0.05

    # Pack to lower triangle
    hess_lt = []
    for i in range(n_dof):
        for j in range(i + 1):
            hess_lt.append(hess_full[i, j])

    lines = []
    lines.append("Water molecule test")
    lines.append("SP        RHF                         STO-3G              0     1")
    lines.append(f"Number of atoms                            I                {n_atoms}")
    lines.append(f"Atomic numbers                             I   N=           {n_atoms}")
    lines.append("           8           1           1")
    lines.append(f"Current cartesian coordinates              R   N=           {3 * n_atoms}")

    # Write coordinates (5 per line)
    for i in range(0, len(coords_bohr), 5):
        chunk = coords_bohr[i : i + 5]
        lines.append("  " + "  ".join(f"{v:16.8E}" for v in chunk))

    lines.append(f"Real atomic weights                        R   N=           {n_atoms}")
    lines.append("  1.59994915E+01  1.00782503E+00  1.00782503E+00")

    lines.append(f"Total Energy                               R     -7.49638920E+01")

    lines.append(f"Cartesian Force Constants                  R   N=          {n_hess}")

    for i in range(0, len(hess_lt), 5):
        chunk = hess_lt[i : i + 5]
        lines.append("  " + "  ".join(f"{v:16.8E}" for v in chunk))

    return "\n".join(lines)


def _make_orca_hess_content(n_atoms: int = 3) -> str:
    """Generate a minimal valid ORCA .hess file for testing."""
    n_dof = 3 * n_atoms

    lines = []

    # Atoms block (coordinates in bohr)
    lines.append("$atoms")
    lines.append(f"{n_atoms}")
    lines.append("O    15.999491  0.000000000  0.000000000  0.000000000")
    lines.append("H     1.007825  0.000000000  1.431000000  1.110000000")
    lines.append("H     1.007825  0.000000000 -1.431000000  1.110000000")

    # Hessian block
    hess = np.eye(n_dof) * 0.5
    for i in range(n_dof - 1):
        hess[i, i + 1] = 0.05
        hess[i + 1, i] = 0.05

    lines.append("")
    lines.append("$hessian")
    lines.append(f"{n_dof}")

    # Write in blocks of 5 columns
    for col_start in range(0, n_dof, 5):
        col_end = min(col_start + 5, n_dof)
        # Column header
        lines.append("         " + "         ".join(f"{c:5d}" for c in range(col_start, col_end)))
        # Data rows
        for row in range(n_dof):
            vals = [f"{hess[row, c]:16.10f}" for c in range(col_start, col_end)]
            lines.append(f"{row:5d} " + " ".join(vals))

    lines.append("")
    lines.append("$end")

    return "\n".join(lines)


# =====================================================================
# Tests: Gaussian .fchk parser
# =====================================================================

class TestGaussianFchk:
    def test_parse_synthetic_water(self, tmp_path):
        """Parse a synthetic water .fchk file."""
        content = _make_fchk_content(3)
        fchk_path = tmp_path / "water.fchk"
        fchk_path.write_text(content)

        data = parse_gaussian_fchk(fchk_path)

        assert data.n_atoms == 3
        assert len(data.atomic_numbers) == 3
        assert data.atomic_numbers[0] == 8  # Oxygen
        assert data.atomic_numbers[1] == 1  # Hydrogen
        assert data.geometry.shape == (3, 3)
        assert data.hessian.shape == (9, 9)
        assert data.source_format == "gaussian_fchk"

    def test_hessian_symmetry(self, tmp_path):
        """Parsed Hessian should be symmetric."""
        content = _make_fchk_content(3)
        fchk_path = tmp_path / "water.fchk"
        fchk_path.write_text(content)

        data = parse_gaussian_fchk(fchk_path)
        asym = np.max(np.abs(data.hessian - data.hessian.T))
        assert asym < 1e-10

    def test_masses_reasonable(self, tmp_path):
        """Masses should be reasonable for water."""
        content = _make_fchk_content(3)
        fchk_path = tmp_path / "water.fchk"
        fchk_path.write_text(content)

        data = parse_gaussian_fchk(fchk_path)
        assert 15.0 < data.masses[0] < 17.0  # Oxygen
        assert 0.9 < data.masses[1] < 1.1    # Hydrogen

    def test_validation_passes(self, tmp_path):
        content = _make_fchk_content(3)
        fchk_path = tmp_path / "water.fchk"
        fchk_path.write_text(content)

        data = parse_gaussian_fchk(fchk_path)
        issues = data.validate()
        assert len(issues) == 0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_gaussian_fchk("/nonexistent/file.fchk")


class TestLowerTriangle:
    def test_identity(self):
        """Lower triangle of identity should reconstruct identity."""
        lt = np.array([1, 0, 1, 0, 0, 1])  # 3x3 identity
        full = _lower_triangle_to_full(lt, 3)
        assert np.allclose(full, np.eye(3))

    def test_symmetric(self):
        """Reconstructed matrix should be symmetric."""
        lt = np.array([1, 2, 3, 4, 5, 6])
        full = _lower_triangle_to_full(lt, 3)
        assert np.allclose(full, full.T)


# =====================================================================
# Tests: ORCA .hess parser
# =====================================================================

class TestOrcaHess:
    def test_parse_synthetic_water(self, tmp_path):
        """Parse a synthetic water .hess file."""
        content = _make_orca_hess_content(3)
        hess_path = tmp_path / "water.hess"
        hess_path.write_text(content)

        data = parse_orca_hess(hess_path)

        assert data.n_atoms == 3
        assert data.atomic_numbers[0] == 8
        assert data.geometry.shape == (3, 3)
        assert data.hessian.shape == (9, 9)
        assert data.source_format == "orca_hess"

    def test_hessian_symmetry(self, tmp_path):
        content = _make_orca_hess_content(3)
        hess_path = tmp_path / "water.hess"
        hess_path.write_text(content)

        data = parse_orca_hess(hess_path)
        asym = np.max(np.abs(data.hessian - data.hessian.T))
        assert asym < 1e-10

    def test_hessian_values_match_input(self, tmp_path):
        """Parsed Hessian should match the synthetic input."""
        content = _make_orca_hess_content(3)
        hess_path = tmp_path / "water.hess"
        hess_path.write_text(content)

        data = parse_orca_hess(hess_path)

        # We put 0.5 on diagonal, 0.05 on first off-diagonals
        assert abs(data.hessian[0, 0] - 0.5) < 1e-6
        assert abs(data.hessian[0, 1] - 0.05) < 1e-6
        assert abs(data.hessian[0, 2] - 0.0) < 1e-6

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_orca_hess("/nonexistent/file.hess")

    def test_validation_passes(self, tmp_path):
        content = _make_orca_hess_content(3)
        hess_path = tmp_path / "water.hess"
        hess_path.write_text(content)

        data = parse_orca_hess(hess_path)
        issues = data.validate()
        assert len(issues) == 0
