"""
Build physically realistic model Hessian matrices for PCET active-site fragments.

Creates triatomic Donor-H-Acceptor (D-H···A) models where the Hessian is
constructed from internal force constants (stretches + bend) using Wilson's
B-matrix formalism. This ensures normal mode analysis recovers the correct
proton transfer frequency, donor-acceptor distance, and reorganization energy.

Each system has a reactant geometry (H bonded to donor) and product geometry
(H bonded to acceptor), with Hessians that reproduce literature-reported
vibrational frequencies.

Reference:
    Wilson, Decius & Cross, "Molecular Vibrations" (1955) — B-matrix method.
"""

import math
import numpy as np
from dataclasses import dataclass

from pcet_engine.core.constants import (
    AMU_TO_AU,
    ANGSTROM_TO_BOHR,
    CM_TO_HARTREE,
)


# Standard atomic masses for active-site atoms
ATOM_MASSES = {
    "C": 12.000,
    "N": 14.003,
    "O": 15.995,
    "S": 31.972,
    "H": 1.00782503207,
}

ATOM_NUMBERS = {
    "C": 6, "N": 7, "O": 8, "S": 16, "H": 1,
}


@dataclass
class ModelSystem:
    """Parameters defining a D-H-A active-site model.

    Attributes:
        name: System identifier.
        donor_element: Element symbol for the donor heavy atom.
        acceptor_element: Element symbol for the acceptor heavy atom.
        r_DH: D-H bond length in angstrom (reactant).
        r_AH: A-H bond length in angstrom (product).
        d_DA: Donor-acceptor distance in angstrom.
        omega_DH: D-H stretching frequency in cm⁻¹ (reactant).
        omega_AH: A-H stretching frequency in cm⁻¹ (product).
        angle_DHA: D-H-A angle in degrees (180 = linear).
        k_nonbond: Non-bonded stretch force constant in hartree/bohr²
                    (weak H-bond interaction for H···A in reactant).
        k_bend: Bending force constant in hartree/rad².
    """

    name: str
    donor_element: str
    acceptor_element: str
    r_DH: float
    r_AH: float
    d_DA: float
    omega_DH: float
    omega_AH: float
    angle_DHA: float = 170.0
    k_nonbond: float = 0.02
    k_bend: float = 0.05


# =====================================================================
# Model system definitions for each benchmark
# =====================================================================

MODEL_SYSTEMS = {
    "SLO-1": ModelSystem(
        name="SLO-1",
        donor_element="C",
        acceptor_element="O",
        r_DH=1.09,
        r_AH=0.96,
        d_DA=2.69,
        omega_DH=2900.0,
        omega_AH=3600.0,   # O-H stretch
        angle_DHA=170.0,
    ),
    "AADH": ModelSystem(
        name="AADH",
        donor_element="C",
        acceptor_element="N",
        r_DH=1.09,
        r_AH=1.01,
        d_DA=3.05,
        omega_DH=3000.0,
        omega_AH=3300.0,   # N-H stretch
        angle_DHA=168.0,
    ),
    "MADH": ModelSystem(
        name="MADH",
        donor_element="C",
        acceptor_element="N",
        r_DH=1.09,
        r_AH=1.01,
        d_DA=3.10,
        omega_DH=2950.0,
        omega_AH=3300.0,
        angle_DHA=165.0,
    ),
    "PHM": ModelSystem(
        name="PHM",
        donor_element="C",
        acceptor_element="O",
        r_DH=1.09,
        r_AH=0.96,
        d_DA=2.55,
        omega_DH=3100.0,
        omega_AH=3600.0,
        angle_DHA=172.0,
    ),
    "RNR": ModelSystem(
        name="RNR",
        donor_element="S",
        acceptor_element="C",
        r_DH=1.34,
        r_AH=1.09,
        d_DA=2.80,
        omega_DH=2600.0,
        omega_AH=2900.0,
        angle_DHA=167.0,
    ),

    # === SLO-1 mutants (same D-H-A chemistry, different d_DA) ===

    "SLO-1-L546A": ModelSystem(
        name="SLO-1-L546A",
        donor_element="C",
        acceptor_element="O",
        r_DH=1.09,
        r_AH=0.96,
        d_DA=2.88,              # enlarged by mutation
        omega_DH=2900.0,
        omega_AH=3600.0,
        angle_DHA=168.0,
    ),
    "SLO-1-L754A": ModelSystem(
        name="SLO-1-L754A",
        donor_element="C",
        acceptor_element="O",
        r_DH=1.09,
        r_AH=0.96,
        d_DA=2.95,
        omega_DH=2900.0,
        omega_AH=3600.0,
        angle_DHA=166.0,
    ),
    "SLO-1-DM": ModelSystem(
        name="SLO-1-DM",
        donor_element="C",
        acceptor_element="O",
        r_DH=1.09,
        r_AH=0.96,
        d_DA=3.10,              # significantly elongated
        omega_DH=2900.0,
        omega_AH=3600.0,
        angle_DHA=164.0,
    ),

    # === Additional enzymes ===

    "GO": ModelSystem(
        name="GO",
        donor_element="C",
        acceptor_element="O",
        r_DH=1.09,
        r_AH=0.96,
        d_DA=2.72,
        omega_DH=2900.0,
        omega_AH=3600.0,
        angle_DHA=170.0,
    ),
    "LADH": ModelSystem(
        name="LADH",
        donor_element="C",
        acceptor_element="C",         # C-to-C hydride transfer
        r_DH=1.09,
        r_AH=1.09,
        d_DA=2.70,
        omega_DH=3000.0,
        omega_AH=3000.0,
        angle_DHA=172.0,
    ),
    "bc1": ModelSystem(
        name="bc1",
        donor_element="O",           # quinol O-H
        acceptor_element="N",        # His nitrogen
        r_DH=0.96,
        r_AH=1.01,
        d_DA=2.70,
        omega_DH=3300.0,
        omega_AH=3300.0,
        angle_DHA=168.0,
    ),
    "CAO": ModelSystem(
        name="CAO",
        donor_element="C",
        acceptor_element="N",
        r_DH=1.09,
        r_AH=1.01,
        d_DA=3.00,
        omega_DH=2900.0,
        omega_AH=3300.0,
        angle_DHA=166.0,
    ),
    "DHFR": ModelSystem(
        name="DHFR",
        donor_element="C",
        acceptor_element="C",         # hydride C-to-C
        r_DH=1.09,
        r_AH=1.09,
        d_DA=2.65,
        omega_DH=3000.0,
        omega_AH=3000.0,
        angle_DHA=174.0,             # nearly linear in compressed TS
    ),
    "TSase": ModelSystem(
        name="TSase",
        donor_element="C",
        acceptor_element="C",
        r_DH=1.09,
        r_AH=1.09,
        d_DA=2.85,
        omega_DH=2950.0,
        omega_AH=2950.0,
        angle_DHA=168.0,
    ),
    "MAO": ModelSystem(
        name="MAO",
        donor_element="C",
        acceptor_element="N",         # flavin N5
        r_DH=1.09,
        r_AH=1.01,
        d_DA=3.00,
        omega_DH=2950.0,
        omega_AH=3300.0,
        angle_DHA=165.0,
    ),
}


def _force_constant_from_frequency(omega_cm: float, m1_amu: float, m2_amu: float) -> float:
    """Compute stretching force constant from frequency and masses.

    k = μ × ω² in atomic units (hartree/bohr²).

    Args:
        omega_cm: Frequency in cm⁻¹.
        m1_amu: Mass of atom 1 in amu.
        m2_amu: Mass of atom 2 in amu.

    Returns:
        Force constant in hartree/bohr².
    """
    mu_amu = m1_amu * m2_amu / (m1_amu + m2_amu)
    mu_au = mu_amu * AMU_TO_AU
    omega_au = omega_cm * CM_TO_HARTREE
    return mu_au * omega_au**2


def _build_geometry(model: ModelSystem, state: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build 3-atom geometry for reactant or product state.

    Places atoms in a bent D-H-A arrangement:
    - Donor at origin
    - Proton along +x direction
    - Acceptor at angle_DHA from the D-H bond

    Args:
        model: Model system parameters.
        state: "reactant" (H bonded to D) or "product" (H bonded to A).

    Returns:
        Tuple of (geometry_angstrom (3,3), elements, masses_amu).
    """
    elements = [model.donor_element, "H", model.acceptor_element]
    masses = np.array([ATOM_MASSES[e] for e in elements])

    theta = math.radians(model.angle_DHA)

    if state == "reactant":
        # D at origin, H at r_DH along x, A at d_DA with slight bend
        r_DH = model.r_DH
        # H-A distance from geometry
        r_HA = model.d_DA - model.r_DH * math.cos(math.pi - theta)
    else:
        # Product: D at origin, H now bonded to A
        r_DH = model.d_DA - model.r_AH * math.cos(math.pi - theta)

    # Donor at origin
    geom = np.zeros((3, 3))

    # H along +x from D
    if state == "reactant":
        geom[1] = [model.r_DH, 0.0, 0.0]
    else:
        # In product, H is closer to A; D-H distance is longer
        geom[1] = [model.d_DA - model.r_AH, 0.0, 0.0]

    # Acceptor: at d_DA from donor, slightly bent
    bend_angle = math.pi - theta  # deviation from linear
    geom[2] = [
        model.d_DA * math.cos(bend_angle / 2),
        model.d_DA * math.sin(bend_angle / 2),
        0.0,
    ]

    return geom, elements, masses


def _b_matrix_stretch(geom_bohr: np.ndarray, i: int, j: int) -> np.ndarray:
    """Wilson B-matrix row for a stretch between atoms i and j.

    ∂r_ij/∂x = unit vector along bond, sign convention: +j, -i.

    Returns:
        B row of shape (9,) for 3 atoms.
    """
    n = geom_bohr.shape[0]
    b_row = np.zeros(3 * n)
    rij = geom_bohr[j] - geom_bohr[i]
    r = np.linalg.norm(rij)
    if r < 1e-10:
        return b_row
    e = rij / r
    b_row[3 * i: 3 * i + 3] = -e
    b_row[3 * j: 3 * j + 3] = +e
    return b_row


def _b_matrix_bend(geom_bohr: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    """Wilson B-matrix row for a bend angle i-j-k (j is the central atom).

    Returns:
        B row of shape (9,) for 3 atoms.
    """
    n = geom_bohr.shape[0]
    b_row = np.zeros(3 * n)

    rji = geom_bohr[i] - geom_bohr[j]
    rjk = geom_bohr[k] - geom_bohr[j]
    r_ji = np.linalg.norm(rji)
    r_jk = np.linalg.norm(rjk)

    if r_ji < 1e-10 or r_jk < 1e-10:
        return b_row

    e_ji = rji / r_ji
    e_jk = rjk / r_jk

    cos_theta = np.clip(np.dot(e_ji, e_jk), -1.0, 1.0)
    sin_theta = math.sqrt(1.0 - cos_theta**2)
    if sin_theta < 1e-10:
        sin_theta = 1e-10

    # ∂θ/∂r_i, ∂θ/∂r_k, ∂θ/∂r_j
    d_i = (cos_theta * e_ji - e_jk) / (r_ji * sin_theta)
    d_k = (cos_theta * e_jk - e_ji) / (r_jk * sin_theta)
    d_j = -d_i - d_k

    b_row[3 * i: 3 * i + 3] = d_i
    b_row[3 * j: 3 * j + 3] = d_j
    b_row[3 * k: 3 * k + 3] = d_k
    return b_row


def build_hessian(model: ModelSystem, state: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Build Cartesian Hessian for a model D-H-A system.

    Constructs the Hessian from internal force constants via Wilson's B-matrix:
        H_cart = B^T × F_int × B

    Internal coordinates: D-H stretch, H-A stretch, D-H-A bend.

    Args:
        model: Model system parameters.
        state: "reactant" or "product".

    Returns:
        Tuple of (hessian_hartree_bohr2 (9,9), geometry_angstrom (3,3),
                  elements, masses_amu).
    """
    geom_ang, elements, masses = _build_geometry(model, state)
    geom_bohr = geom_ang * ANGSTROM_TO_BOHR

    m_D = ATOM_MASSES[model.donor_element]
    m_H = ATOM_MASSES["H"]
    m_A = ATOM_MASSES[model.acceptor_element]

    if state == "reactant":
        # Strong D-H bond, weak H···A interaction
        k_DH = _force_constant_from_frequency(model.omega_DH, m_D, m_H)
        k_HA = model.k_nonbond  # weak H-bond
    else:
        # Weak D···H interaction, strong H-A bond
        k_DH = model.k_nonbond
        k_HA = _force_constant_from_frequency(model.omega_AH, m_A, m_H)

    k_bend = model.k_bend

    # Build B-matrix (3 internal coords × 9 Cartesian)
    B = np.zeros((3, 9))
    B[0] = _b_matrix_stretch(geom_bohr, 0, 1)  # D-H stretch
    B[1] = _b_matrix_stretch(geom_bohr, 1, 2)  # H-A stretch
    B[2] = _b_matrix_bend(geom_bohr, 0, 1, 2)  # D-H-A bend (H is central for near-linear)

    # Internal force constant matrix
    F_int = np.diag([k_DH, k_HA, k_bend])

    # Small stretch-stretch coupling (off-diagonal)
    coupling = 0.01 * math.sqrt(k_DH * k_HA)
    F_int[0, 1] = coupling
    F_int[1, 0] = coupling

    # Transform to Cartesian: H_cart = B^T F B
    hessian = B.T @ F_int @ B

    # Symmetrize (numerical safety)
    hessian = 0.5 * (hessian + hessian.T)

    return hessian, geom_ang, elements, masses


def write_orca_hess(
    filepath: str,
    hessian: np.ndarray,
    geometry_ang: np.ndarray,
    elements: list[str],
    masses: np.ndarray,
    energy: float = 0.0,
) -> None:
    """Write model Hessian data in ORCA .hess format.

    Args:
        filepath: Output file path.
        hessian: Cartesian Hessian (3N×3N) in hartree/bohr².
        geometry_ang: Geometry (N,3) in angstrom.
        elements: Element symbols.
        masses: Atomic masses in amu.
        energy: Total energy in hartree.
    """
    n_atoms = len(elements)
    n_dof = 3 * n_atoms
    geom_bohr = geometry_ang * ANGSTROM_TO_BOHR

    with open(filepath, "w") as f:
        # Energy block
        f.write("$act_energy\n")
        f.write(f"  {energy:.10f}\n")
        f.write("\n")

        # Atoms block
        f.write("$atoms\n")
        f.write(f"{n_atoms}\n")
        for i, (elem, mass) in enumerate(zip(elements, masses)):
            x, y, z = geom_bohr[i]
            f.write(f"  {elem:>2s}   {mass:12.6f}   {x:16.10f}   {y:16.10f}   {z:16.10f}\n")
        f.write("\n")

        # Hessian block (column blocks of 5)
        f.write("$hessian\n")
        f.write(f"{n_dof}\n")

        n_blocks = (n_dof + 4) // 5
        for block in range(n_blocks):
            col_start = block * 5
            col_end = min(col_start + 5, n_dof)

            # Column header
            header = "         "
            for c in range(col_start, col_end):
                header += f"{c:>12d}"
            f.write(header + "\n")

            # Data rows
            for row in range(n_dof):
                line = f"{row:>4d}  "
                for c in range(col_start, col_end):
                    line += f"{hessian[row, c]:>12.6f}"
                f.write(line + "\n")
        f.write("\n")

        f.write("$end\n")


def generate_all_model_hessians(output_dir: str) -> dict[str, dict]:
    """Generate ORCA .hess files for all benchmark systems.

    Creates reactant and product Hessian files for each of the 5 systems.

    Args:
        output_dir: Directory to write .hess files.

    Returns:
        Dict mapping system name -> {"reactant_file", "product_file",
        "geometry_R", "geometry_P", "masses", "elements", "proton_idx",
        "donor_idx", "acceptor_idx"}.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name, model in MODEL_SYSTEMS.items():
        hess_R, geom_R, elements, masses = build_hessian(model, "reactant")
        hess_P, geom_P, _, _ = build_hessian(model, "product")

        r_file = os.path.join(output_dir, f"{name}_reactant.hess")
        p_file = os.path.join(output_dir, f"{name}_product.hess")

        write_orca_hess(r_file, hess_R, geom_R, elements, masses)
        write_orca_hess(p_file, hess_P, geom_P, elements, masses)

        results[name] = {
            "reactant_file": r_file,
            "product_file": p_file,
            "geometry_R": geom_R,
            "geometry_P": geom_P,
            "hessian_R": hess_R,
            "hessian_P": hess_P,
            "masses": masses,
            "elements": elements,
            "proton_idx": 1,   # H is always atom index 1
            "donor_idx": 0,    # Donor is always atom index 0
            "acceptor_idx": 2, # Acceptor is always atom index 2
        }

    return results
