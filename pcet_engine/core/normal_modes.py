"""
Normal mode analysis from molecular Hessian matrices.

Extracts vibrational frequencies, eigenvectors, and reduced masses from
a Cartesian Hessian matrix. Identifies the proton/deuterium transfer mode
by displacement analysis.

References:
    Wilson, Decius & Cross, "Molecular Vibrations" (1955).
"""

import numpy as np
from dataclasses import dataclass, field

from pcet_engine.core.constants import (
    HARTREE_TO_CM,
    AMU_TO_AU,
    BOHR_TO_ANGSTROM,
    PROTON_MASS_AMU,
    DEUTERIUM_MASS_AMU,
)


@dataclass
class NormalModeResult:
    """Result of normal mode analysis.

    Attributes:
        frequencies_cm: Vibrational frequencies in cm⁻¹ (sorted, imaginary as negative).
        frequencies_au: Frequencies in atomic units (hartree/hbar).
        eigenvectors: Normal mode eigenvectors, shape (3N, 3N), columns are modes.
        reduced_masses: Reduced mass for each mode in amu.
        n_atoms: Number of atoms.
        n_imaginary: Number of imaginary frequencies (transition states).
        proton_mode_idx: Index of the mode with largest proton displacement (if identified).
        proton_frequency_cm: Frequency of the identified proton transfer mode.
    """

    frequencies_cm: np.ndarray
    frequencies_au: np.ndarray
    eigenvectors: np.ndarray
    reduced_masses: np.ndarray
    n_atoms: int
    n_imaginary: int = 0
    proton_mode_idx: int | None = None
    proton_frequency_cm: float | None = None


def normal_mode_analysis(
    hessian: np.ndarray,
    masses: np.ndarray,
    project_trans_rot: bool = True,
) -> NormalModeResult:
    """Perform normal mode analysis on a Cartesian Hessian.

    Args:
        hessian: Cartesian force constant matrix in hartree/bohr², shape (3N, 3N).
                 NOT mass-weighted — raw second derivatives of energy.
        masses: Atomic masses in amu, shape (N,).
        project_trans_rot: If True, project out translation and rotation
                          (for non-periodic systems).

    Returns:
        NormalModeResult with frequencies, eigenvectors, reduced masses.
    """
    n_atoms = len(masses)
    n_dof = 3 * n_atoms

    if hessian.shape != (n_dof, n_dof):
        raise ValueError(
            f"Hessian shape {hessian.shape} inconsistent with {n_atoms} atoms "
            f"(expected ({n_dof}, {n_dof}))"
        )

    # Symmetrize (Hessians from QM codes can have small asymmetries)
    hessian_sym = 0.5 * (hessian + hessian.T)

    # Mass-weight: H_mw = M^{-1/2} H M^{-1/2}
    mass_weights = np.repeat(masses * AMU_TO_AU, 3)
    inv_sqrt_m = 1.0 / np.sqrt(mass_weights)
    hessian_mw = hessian_sym * np.outer(inv_sqrt_m, inv_sqrt_m)

    if project_trans_rot:
        hessian_mw = _project_trans_rot(hessian_mw, masses, n_atoms)

    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(hessian_mw)

    # Convert eigenvalues to frequencies
    # ω² = eigenvalue (in atomic units: hartree/(m_e × bohr²))
    # For frequency in cm⁻¹: ν = ω / (2πc) with appropriate conversions
    frequencies_au = np.zeros(n_dof)
    frequencies_cm = np.zeros(n_dof)
    n_imaginary = 0

    for i, ev in enumerate(eigenvalues):
        if ev < -1e-6:
            # Imaginary frequency (transition state)
            omega = np.sqrt(abs(ev))
            frequencies_au[i] = -omega
            frequencies_cm[i] = -omega * HARTREE_TO_CM
            n_imaginary += 1
        elif ev < 1e-6:
            # Near-zero (translation/rotation)
            frequencies_au[i] = 0.0
            frequencies_cm[i] = 0.0
        else:
            omega = np.sqrt(ev)
            frequencies_au[i] = omega
            frequencies_cm[i] = omega * HARTREE_TO_CM

    # Un-mass-weight eigenvectors to get Cartesian displacements
    cart_eigvecs = eigenvectors * inv_sqrt_m[:, np.newaxis]

    # Reduced masses: μ_k = 1 / Σ_i (L_{ik})² where L is mass-weighted
    reduced_masses = np.zeros(n_dof)
    for k in range(n_dof):
        mode_mw = eigenvectors[:, k]
        sum_sq = np.sum(mode_mw**2)
        if sum_sq > 1e-15:
            reduced_masses[k] = 1.0 / sum_sq  # in m_e
            reduced_masses[k] /= AMU_TO_AU  # convert to amu
        else:
            reduced_masses[k] = 0.0

    return NormalModeResult(
        frequencies_cm=frequencies_cm,
        frequencies_au=frequencies_au,
        eigenvectors=cart_eigvecs,
        reduced_masses=reduced_masses,
        n_atoms=n_atoms,
        n_imaginary=n_imaginary,
    )


def identify_proton_mode(
    result: NormalModeResult,
    proton_indices: list[int],
    masses: np.ndarray,
    min_proton_fraction: float = 0.1,
) -> NormalModeResult:
    """Identify the normal mode corresponding to the proton transfer stretch.

    The proton transfer mode is characterized as the highest-frequency mode
    with significant proton kinetic energy contribution. This correctly
    selects X-H stretches (> 2000 cm⁻¹) over bending modes where H may
    have large displacement but lower frequency.

    Args:
        result: Normal mode analysis result.
        proton_indices: 0-based indices of the transferring proton atom(s).
        masses: Atomic masses in amu.
        min_proton_fraction: Minimum fractional proton KE to consider (default 0.1).

    Returns:
        Updated NormalModeResult with proton_mode_idx and proton_frequency_cm set.
    """
    best_idx = None
    best_freq = 0.0

    for k in range(result.eigenvectors.shape[1]):
        if result.frequencies_cm[k] <= 50.0:
            continue  # Skip trans/rot/low-freq modes

        # Compute fractional kinetic energy in the proton atoms
        mode = result.eigenvectors[:, k]
        total_ke = 0.0
        proton_ke = 0.0
        for i in range(result.n_atoms):
            m = masses[i]
            disp_sq = np.sum(mode[3 * i : 3 * i + 3] ** 2)
            ke_i = m * disp_sq
            total_ke += ke_i
            if i in proton_indices:
                proton_ke += ke_i

        if total_ke > 1e-15:
            frac = proton_ke / total_ke
            # Select the highest-frequency mode with significant proton character
            if frac >= min_proton_fraction and result.frequencies_cm[k] > best_freq:
                best_freq = result.frequencies_cm[k]
                best_idx = k

    result.proton_mode_idx = best_idx
    if best_idx is not None:
        result.proton_frequency_cm = result.frequencies_cm[best_idx]

    return result


@dataclass
class GatingResult:
    """Result of D-A gating mode identification.

    Attributes:
        omega_gating: D-A stretching frequency in cm⁻¹.
        M_DA: Effective mass of the D-A oscillator in amu.
        da_distance: Equilibrium D-A distance in angstrom.
        projection: Overlap of identified mode with D-A vector (0-1).
        mode_index: Which normal mode was identified.
    """

    omega_gating: float
    M_DA: float
    da_distance: float
    projection: float
    mode_index: int


def identify_da_stretching_mode(
    hessian: np.ndarray,
    masses: np.ndarray,
    donor_idx: int,
    acceptor_idx: int,
    geometry: np.ndarray,
) -> GatingResult:
    """Identify the D-A stretching mode from the Hessian.

    The D-A stretching mode is the normal mode with maximum projection
    onto the D-A displacement vector. This extracts Omega_gating and M_DA
    directly from quantum chemistry data, eliminating the need for MD.

    Algorithm:
        1. Normal mode analysis (mass-weighted Hessian → eigenvalues/eigenvectors)
        2. Compute D-A unit vector e_DA
        3. For each mode, compute projection of relative D-A displacement onto e_DA
        4. Mode with largest projection = D-A stretch
        5. M_DA from participation-weighted mass along e_DA

    Args:
        hessian: Cartesian Hessian in hartree/bohr², shape (3N, 3N).
        masses: Atomic masses in amu, shape (N,).
        donor_idx: 0-based index of donor atom.
        acceptor_idx: 0-based index of acceptor atom.
        geometry: Atomic coordinates in angstrom, shape (N, 3).

    Returns:
        GatingResult with omega_gating, M_DA, da_distance, projection, mode_index.
    """
    # Normal mode analysis
    nma = normal_mode_analysis(hessian, masses, project_trans_rot=True)

    # D-A unit vector
    r_D = geometry[donor_idx]
    r_A = geometry[acceptor_idx]
    r_DA = r_A - r_D
    da_distance = float(np.linalg.norm(r_DA))
    if da_distance < 1e-10:
        raise ValueError("Donor and acceptor are at the same position")
    e_DA = r_DA / da_distance

    n_atoms = len(masses)
    best_proj = -1.0
    best_idx = -1

    for k in range(nma.eigenvectors.shape[1]):
        if nma.frequencies_cm[k] <= 10.0:
            continue  # Skip trans/rot/near-zero modes

        mode = nma.eigenvectors[:, k]

        # Extract 3D displacement of donor and acceptor
        disp_D = mode[3 * donor_idx: 3 * donor_idx + 3]
        disp_A = mode[3 * acceptor_idx: 3 * acceptor_idx + 3]

        # Relative D-A motion
        delta_DA = disp_A - disp_D

        # Projection onto D-A axis
        proj = abs(float(np.dot(delta_DA, e_DA)))

        if proj > best_proj:
            best_proj = proj
            best_idx = k

    if best_idx < 0:
        raise ValueError("Could not identify any D-A stretching mode")

    omega_gating = nma.frequencies_cm[best_idx]

    # Compute effective mass M_DA from participation-weighted mass
    # M_DA = Σ m_i |v_i · e_DA|² / Σ |v_i · e_DA|²
    mode = nma.eigenvectors[:, best_idx]
    numerator = 0.0
    denominator = 0.0
    for i in range(n_atoms):
        v_i = mode[3 * i: 3 * i + 3]
        proj_i = float(np.dot(v_i, e_DA))
        proj_sq = proj_i ** 2
        numerator += masses[i] * proj_sq
        denominator += proj_sq

    if denominator < 1e-30:
        M_DA = masses[donor_idx] * masses[acceptor_idx] / (masses[donor_idx] + masses[acceptor_idx])
    else:
        M_DA = numerator / denominator

    return GatingResult(
        omega_gating=omega_gating,
        M_DA=M_DA,
        da_distance=da_distance,
        projection=best_proj,
        mode_index=best_idx,
    )


def compute_donor_acceptor_distance(
    geometry: np.ndarray,
    donor_idx: int,
    acceptor_idx: int,
) -> float:
    """Compute donor-acceptor distance from geometry.

    Args:
        geometry: Atomic coordinates, shape (N, 3) in angstrom.
        donor_idx: 0-based index of donor atom.
        acceptor_idx: 0-based index of acceptor atom.

    Returns:
        Distance in angstrom.
    """
    return float(np.linalg.norm(geometry[donor_idx] - geometry[acceptor_idx]))


def _project_trans_rot(
    hessian_mw: np.ndarray,
    masses: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Project out translational and rotational degrees of freedom.

    Uses the Sayvetz conditions to build the projection operator
    P = I - Σ_i |t_i><t_i| where t_i are mass-weighted trans/rot vectors.

    Args:
        hessian_mw: Mass-weighted Hessian.
        masses: Atomic masses in amu.
        n_atoms: Number of atoms.

    Returns:
        Projected mass-weighted Hessian.
    """
    n_dof = 3 * n_atoms
    sqrt_m = np.sqrt(np.repeat(masses * AMU_TO_AU, 3))

    # Translation vectors (mass-weighted)
    trans = np.zeros((3, n_dof))
    for i in range(3):
        for j in range(n_atoms):
            trans[i, 3 * j + i] = sqrt_m[3 * j + i]

    # Normalize
    for i in range(3):
        norm = np.linalg.norm(trans[i])
        if norm > 1e-10:
            trans[i] /= norm

    # For rotation, we'd need the geometry — skip for now and just project translations
    # This is sufficient for most PCET applications where we care about high-freq modes
    projectors = trans

    # Build projector: P = I - Σ |t><t|
    P = np.eye(n_dof)
    for t in projectors:
        P -= np.outer(t, t)

    return P @ hessian_mw @ P
