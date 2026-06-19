"""
Parser for Gaussian formatted checkpoint (.fchk) files.

Extracts:
- Molecular geometry (Cartesian coordinates)
- Atomic numbers and masses
- Cartesian force constants (Hessian matrix)
- Total energy

The .fchk format stores data as labeled sections with type/length headers:

    Label                          Type  Length
    Atomic numbers                 I       N
    Current cartance coordinates   R      3N
    Cartesian Force Constants      R    3N*(3N+1)/2

Real arrays are stored 5 values per line, integer arrays 6 per line.

References:
    Gaussian documentation: https://gaussian.com/formchk/
"""

import numpy as np
from pathlib import Path

from pcet_engine.parsers.base import QCData, STANDARD_MASSES
from pcet_engine.core.constants import BOHR_TO_ANGSTROM


def parse_gaussian_fchk(filepath: str | Path) -> QCData:
    """Parse a Gaussian .fchk file.

    Args:
        filepath: Path to the .fchk file.

    Returns:
        QCData with geometry, Hessian, masses, etc.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required data is missing or malformed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath) as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"File too short to be a valid .fchk: {filepath}")

    title = lines[0].strip()

    # Parse all sections
    sections = _parse_sections(lines)

    # Extract required data
    n_atoms = _get_scalar_int(sections, "Number of atoms")
    atomic_numbers = _get_int_array(sections, "Atomic numbers", n_atoms)
    energy = _get_scalar_real(sections, "Total Energy")

    # Geometry: stored in bohr, convert to angstrom
    coords_bohr = _get_real_array(sections, "Current cartesian coordinates", 3 * n_atoms)
    geometry = coords_bohr.reshape(n_atoms, 3) * BOHR_TO_ANGSTROM

    # Masses: try to read from file, fallback to standard
    if "Real atomic weights" in sections:
        masses = _get_real_array(sections, "Real atomic weights", n_atoms)
    else:
        masses = np.array([STANDARD_MASSES.get(z, 1.0) for z in atomic_numbers])

    # Hessian: stored as lower triangle, n*(n+1)/2 elements
    n_dof = 3 * n_atoms
    n_hess = n_dof * (n_dof + 1) // 2
    hess_lt = _get_real_array(sections, "Cartesian Force Constants", n_hess)
    hessian = _lower_triangle_to_full(hess_lt, n_dof)

    # Parse charge/multiplicity from the second line
    charge = 0
    multiplicity = 1
    if len(lines) >= 2:
        parts = lines[1].split()
        # Format: "method/basis  type  charge  mult"
        if len(parts) >= 4:
            try:
                charge = int(parts[-2])
                multiplicity = int(parts[-1])
            except ValueError:
                pass

    return QCData(
        n_atoms=n_atoms,
        atomic_numbers=atomic_numbers,
        masses=masses,
        geometry=geometry,
        hessian=hessian,
        energy=energy,
        charge=charge,
        multiplicity=multiplicity,
        title=title,
        source_file=str(filepath),
        source_format="gaussian_fchk",
    )


def _parse_sections(lines: list[str]) -> dict:
    """Parse .fchk file into labeled sections.

    Returns dict mapping label -> (type, length, data) where:
      - type is 'I' (integer), 'R' (real), or 'scalar'
      - length is number of elements (-1 for scalar)
      - data is the raw string values
    """
    sections = {}
    i = 2  # Skip first two header lines

    while i < len(lines):
        line = lines[i]

        # Check for section header: label is left-justified in columns 1-43
        # Type indicator at column 43: I or R
        # N= follows for arrays, or value for scalars
        if len(line) >= 44 and line[43] in ("I", "R"):
            label = line[:43].strip()
            dtype = line[43]

            if "N=" in line[44:]:
                # Array section
                n_str = line[44:].replace("N=", "").strip()
                n = int(n_str)
                sections[label] = {"type": dtype, "length": n, "values": []}

                # Read data lines
                i += 1
                values = []
                while i < len(lines) and len(values) < n:
                    data_line = lines[i].strip()
                    if not data_line:
                        i += 1
                        continue
                    # Check if this line is a new section header
                    if len(lines[i]) >= 44 and lines[i][43] in ("I", "R"):
                        break
                    values.extend(data_line.split())
                    i += 1

                if dtype == "I":
                    sections[label]["values"] = [int(v) for v in values[:n]]
                else:
                    sections[label]["values"] = [float(v.replace("D", "E")) for v in values[:n]]
            else:
                # Scalar section
                val_str = line[44:].strip()
                if dtype == "I":
                    sections[label] = {"type": "scalar_I", "value": int(val_str)}
                else:
                    sections[label] = {"type": "scalar_R", "value": float(val_str.replace("D", "E"))}
                i += 1
        else:
            i += 1

    return sections


def _get_scalar_int(sections: dict, label: str) -> int:
    if label not in sections:
        raise ValueError(f"Missing required section: '{label}'")
    sec = sections[label]
    if sec.get("type") == "scalar_I":
        return sec["value"]
    raise ValueError(f"Section '{label}' is not a scalar integer")


def _get_scalar_real(sections: dict, label: str) -> float:
    if label not in sections:
        return 0.0  # Optional
    sec = sections[label]
    if sec.get("type") == "scalar_R":
        return sec["value"]
    return 0.0


def _get_int_array(sections: dict, label: str, expected_len: int) -> np.ndarray:
    if label not in sections:
        raise ValueError(f"Missing required section: '{label}'")
    sec = sections[label]
    values = sec["values"]
    if len(values) != expected_len:
        raise ValueError(f"Section '{label}': expected {expected_len} values, got {len(values)}")
    return np.array(values, dtype=np.int64)


def _get_real_array(sections: dict, label: str, expected_len: int) -> np.ndarray:
    if label not in sections:
        raise ValueError(f"Missing required section: '{label}'")
    sec = sections[label]
    values = sec["values"]
    if len(values) != expected_len:
        raise ValueError(f"Section '{label}': expected {expected_len} values, got {len(values)}")
    return np.array(values, dtype=np.float64)


def _lower_triangle_to_full(lt: np.ndarray, n: int) -> np.ndarray:
    """Convert lower-triangle packed array to full symmetric matrix.

    Gaussian stores Hessian as: H[0,0], H[1,0], H[1,1], H[2,0], ...
    """
    full = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            full[i, j] = lt[idx]
            full[j, i] = lt[idx]
            idx += 1
    return full
