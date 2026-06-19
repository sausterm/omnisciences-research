"""
Parser for ORCA .hess files.

ORCA writes Hessian data in a plain-text format with labeled blocks:

    $hessian
    N
    v1 v2 v3 v4 v5
    0   h00 h01 h02 h03 h04
    1   h10 h11 h12 h13 h14
    ...

    $atoms
    N
    element  mass  x  y  z
    ...

    $vibrational_frequencies
    N
    0   freq_0
    1   freq_1
    ...

    $normal_modes
    ...

Coordinates in the $atoms block are in bohr.
Hessian elements are in hartree/bohr².

References:
    ORCA manual, Chapter on frequency calculations.
"""

import numpy as np
from pathlib import Path

from pcet_engine.parsers.base import QCData, STANDARD_MASSES
from pcet_engine.core.constants import BOHR_TO_ANGSTROM


# Element symbol to atomic number
ELEMENT_TO_Z = {v: k for k, v in {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 26: "Fe", 29: "Cu",
    30: "Zn", 34: "Se", 42: "Mo", 25: "Mn",
}.items()}


def parse_orca_hess(filepath: str | Path) -> QCData:
    """Parse an ORCA .hess file.

    Args:
        filepath: Path to the .hess file.

    Returns:
        QCData with geometry, Hessian, masses, etc.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required blocks are missing or malformed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath) as f:
        content = f.read()

    lines = content.split("\n")

    # Parse blocks
    atoms_data = _parse_atoms_block(lines)
    hessian = _parse_hessian_block(lines)
    energy = _parse_energy(lines)

    n_atoms = len(atoms_data)
    if n_atoms == 0:
        raise ValueError("No atoms found in $atoms block")

    atomic_numbers = np.array([a["Z"] for a in atoms_data])
    masses = np.array([a["mass"] for a in atoms_data])
    geometry = np.array([[a["x"], a["y"], a["z"]] for a in atoms_data])
    # Convert from bohr to angstrom
    geometry *= BOHR_TO_ANGSTROM

    if hessian.shape[0] != 3 * n_atoms:
        raise ValueError(
            f"Hessian dimension {hessian.shape[0]} inconsistent with "
            f"{n_atoms} atoms (expected {3 * n_atoms})"
        )

    return QCData(
        n_atoms=n_atoms,
        atomic_numbers=atomic_numbers,
        masses=masses,
        geometry=geometry,
        hessian=hessian,
        energy=energy,
        title=f"ORCA calculation from {filepath.name}",
        source_file=str(filepath),
        source_format="orca_hess",
    )


def _parse_atoms_block(lines: list[str]) -> list[dict]:
    """Parse the $atoms block."""
    atoms = []
    in_block = False
    n_atoms = 0
    count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "$atoms":
            in_block = True
            continue

        if in_block and n_atoms == 0 and stripped.isdigit():
            n_atoms = int(stripped)
            continue

        if in_block and count < n_atoms:
            parts = stripped.split()
            if len(parts) >= 5:
                element = parts[0]
                mass = float(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

                Z = ELEMENT_TO_Z.get(element, 0)
                if Z == 0:
                    # Try case-insensitive
                    for sym, num in ELEMENT_TO_Z.items():
                        if sym.lower() == element.lower():
                            Z = num
                            break

                atoms.append({"Z": Z, "mass": mass, "x": x, "y": y, "z": z})
                count += 1

        if stripped.startswith("$") and stripped != "$atoms" and in_block:
            break

    return atoms


def _parse_hessian_block(lines: list[str]) -> np.ndarray:
    """Parse the $hessian block.

    ORCA writes the Hessian in column blocks of 5:
        $hessian
        N
                    0          1          2          3          4
        0     0.123456   0.234567   ...
        1     ...
        ...
                    5          6          7          8          9
        0     ...
    """
    in_block = False
    n_dof = 0
    hessian = None
    current_cols = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "$hessian":
            in_block = True
            continue

        if in_block and n_dof == 0 and stripped.isdigit():
            n_dof = int(stripped)
            hessian = np.zeros((n_dof, n_dof))
            continue

        if not in_block or hessian is None:
            continue

        if stripped.startswith("$") and stripped != "$hessian":
            break

        parts = stripped.split()
        if not parts:
            continue

        # Column header line (all integers)
        try:
            cols = [int(p) for p in parts]
            if all(0 <= c < n_dof for c in cols):
                current_cols = cols
                continue
        except ValueError:
            pass

        # Data line: row_index value1 value2 ...
        if len(parts) >= 2:
            try:
                row = int(parts[0])
                values = [float(v) for v in parts[1:]]
                for j, val in enumerate(values):
                    if j < len(current_cols):
                        col = current_cols[j]
                        if 0 <= row < n_dof and 0 <= col < n_dof:
                            hessian[row, col] = val
            except ValueError:
                continue

    if hessian is None:
        raise ValueError("$hessian block not found or empty")

    # Symmetrize (ORCA writes the full matrix but there can be tiny asymmetries)
    hessian = 0.5 * (hessian + hessian.T)

    return hessian


def _parse_energy(lines: list[str]) -> float:
    """Try to extract total energy from the file."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("$act_energy"):
            # Next non-empty line should have the energy
            idx = lines.index(line)
            for j in range(idx + 1, min(idx + 3, len(lines))):
                try:
                    return float(lines[j].strip())
                except ValueError:
                    continue
    return 0.0
