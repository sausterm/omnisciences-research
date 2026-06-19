"""
Base data structures for parsed quantum chemistry data.
"""

import numpy as np
from dataclasses import dataclass, field


# Atomic number to element symbol
ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 26: "Fe", 29: "Cu",
    30: "Zn", 34: "Se", 42: "Mo", 25: "Mn",
}

# Standard atomic masses (most abundant isotope)
STANDARD_MASSES = {
    1: 1.00782503207, 2: 4.002602, 3: 6.941, 4: 9.012182, 5: 10.811,
    6: 12.0, 7: 14.003074, 8: 15.994915, 9: 18.998403, 10: 20.1797,
    11: 22.989770, 12: 24.3050, 13: 26.981538, 14: 27.976927, 15: 30.973762,
    16: 31.972071, 17: 34.968853, 18: 39.948, 19: 39.0983, 20: 40.078,
    26: 55.845, 29: 63.546, 30: 65.38, 34: 78.971, 42: 95.95, 25: 54.938,
}


@dataclass
class QCData:
    """Parsed quantum chemistry data from a checkpoint/output file.

    Attributes:
        n_atoms: Number of atoms.
        atomic_numbers: Atomic numbers, shape (N,).
        masses: Atomic masses in amu, shape (N,).
        geometry: Cartesian coordinates in angstrom, shape (N, 3).
        hessian: Cartesian force constant matrix in hartree/bohr², shape (3N, 3N).
        energy: Total electronic energy in hartree.
        elements: Element symbols.
        charge: Molecular charge.
        multiplicity: Spin multiplicity.
        title: Title/comment from the file.
        source_file: Path to the source file.
        source_format: Format identifier ('gaussian_fchk', 'orca_hess', etc.).
    """

    n_atoms: int
    atomic_numbers: np.ndarray
    masses: np.ndarray
    geometry: np.ndarray  # angstrom, (N, 3)
    hessian: np.ndarray   # hartree/bohr², (3N, 3N)
    energy: float = 0.0
    elements: list[str] = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    title: str = ""
    source_file: str = ""
    source_format: str = ""

    def __post_init__(self):
        if not self.elements:
            self.elements = [
                ELEMENT_SYMBOLS.get(z, f"X{z}") for z in self.atomic_numbers
            ]

    def validate(self) -> list[str]:
        """Check internal consistency. Returns list of issues (empty = OK)."""
        issues = []
        n = self.n_atoms
        if self.atomic_numbers.shape != (n,):
            issues.append(f"atomic_numbers shape {self.atomic_numbers.shape} != ({n},)")
        if self.masses.shape != (n,):
            issues.append(f"masses shape {self.masses.shape} != ({n},)")
        if self.geometry.shape != (n, 3):
            issues.append(f"geometry shape {self.geometry.shape} != ({n}, 3)")
        if self.hessian.shape != (3 * n, 3 * n):
            issues.append(f"hessian shape {self.hessian.shape} != ({3*n}, {3*n})")

        # Check Hessian symmetry
        asym = np.max(np.abs(self.hessian - self.hessian.T))
        if asym > 1e-6:
            issues.append(f"Hessian asymmetry: max |H - H^T| = {asym:.2e}")

        return issues
