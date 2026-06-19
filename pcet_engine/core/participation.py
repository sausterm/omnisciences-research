"""
Participation ratio and effective tunneling dimensionality for PCET.

Computes N_eff from normal mode eigenvalues to quantify how many
degrees of freedom participate in the proton transfer tunneling coordinate.

    N_eff = (Σ|λᵢ|)² / Σλᵢ²

ranges from 1 (single mode dominates) to N (all modes contribute equally).

This quantity originates from the DeWitt metric eigenvalue spectrum on
symmetric spaces G/H, where it determines the effective number of
degrees of freedom contributing to the instanton action.

References:
    DeWitt, B.S. Phys. Rev. 160, 1113 (1967).
    Hay, S. & Scrutton, N.S. Nat. Chem. 4, 161 (2012).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ParticipationResult:
    """Result of participation ratio analysis.

    Attributes:
        n_eff: Participation ratio from eigenvalue spectrum.
        n_eff_proton: Participation ratio restricted to proton-active modes.
        n_active_modes: Number of modes with proton fraction above threshold.
        proton_fractions: Proton kinetic energy fraction for each mode.
        active_mode_indices: Indices of proton-active modes.
        geometric_prefactor: Tunneling prefactor correction sqrt(N_eff / d_ref).
    """

    n_eff: float
    n_eff_proton: float
    n_active_modes: int
    proton_fractions: np.ndarray
    active_mode_indices: np.ndarray
    geometric_prefactor: float


def participation_ratio(values: np.ndarray) -> float:
    """Compute participation ratio from a spectrum of values.

    N_eff = (Σ|v_i|)² / Σv_i²

    Returns 1.0 if a single value dominates, N if all values are equal.

    Args:
        values: Array of values (eigenvalues, fractions, etc.).

    Returns:
        Participation ratio (≥ 1.0).
    """
    abs_vals = np.abs(values)
    sum_abs = np.sum(abs_vals)
    sum_sq = np.sum(abs_vals ** 2)

    if sum_sq < 1e-30:
        return 1.0

    return sum_abs ** 2 / sum_sq


def mode_participation(
    eigenvectors: np.ndarray,
    masses: np.ndarray,
    atom_indices: list[int],
) -> np.ndarray:
    """Compute fractional kinetic energy of specific atoms in each mode.

    For each normal mode k, computes:
        f_k = Σ_{i∈atoms} m_i |L_{k,i}|² / Σ_j m_j |L_{k,j}|²

    Args:
        eigenvectors: Normal mode eigenvectors, shape (3N, 3N).
            Columns are modes, rows are Cartesian displacements.
        masses: Atomic masses in amu, shape (N,).
        atom_indices: 0-based indices of atoms to track.

    Returns:
        Array of fractional kinetic energies, shape (n_modes,).
    """
    n_atoms = len(masses)
    n_modes = eigenvectors.shape[1]
    fractions = np.zeros(n_modes)

    for k in range(n_modes):
        mode = eigenvectors[:, k]
        total_ke = 0.0
        atom_ke = 0.0

        for i in range(n_atoms):
            m = masses[i]
            disp_sq = np.sum(mode[3 * i : 3 * i + 3] ** 2)
            ke_i = m * disp_sq
            total_ke += ke_i
            if i in atom_indices:
                atom_ke += ke_i

        if total_ke > 1e-15:
            fractions[k] = atom_ke / total_ke

    return fractions


def proton_participation(
    eigenvectors: np.ndarray,
    frequencies_cm: np.ndarray,
    masses: np.ndarray,
    proton_indices: list[int],
    threshold: float = 0.05,
    min_freq_cm: float = 50.0,
) -> ParticipationResult:
    """Compute participation ratio for proton tunneling modes.

    Identifies modes with significant proton kinetic energy contribution
    and computes the effective tunneling dimensionality.

    Args:
        eigenvectors: Normal mode eigenvectors, shape (3N, 3N).
        frequencies_cm: Vibrational frequencies in cm⁻¹, shape (3N,).
        masses: Atomic masses in amu, shape (N,).
        proton_indices: 0-based indices of the transferring proton(s).
        threshold: Minimum proton fraction to consider a mode "active".
        min_freq_cm: Minimum frequency (cm⁻¹) to consider (skip trans/rot).

    Returns:
        ParticipationResult with all diagnostics.
    """
    fractions = mode_participation(eigenvectors, masses, proton_indices)

    # Filter: frequency above threshold AND significant proton character
    active_mask = (np.abs(frequencies_cm) > min_freq_cm) & (fractions > threshold)
    active_indices = np.where(active_mask)[0]
    active_fractions = fractions[active_indices]

    n_active = len(active_indices)

    # Overall participation ratio from all eigenvalues
    freq_sq = frequencies_cm[np.abs(frequencies_cm) > min_freq_cm] ** 2
    n_eff_all = participation_ratio(freq_sq) if len(freq_sq) > 0 else 1.0

    # Proton-specific participation ratio
    if n_active > 0:
        n_eff_proton = participation_ratio(active_fractions)
    else:
        n_eff_proton = 1.0

    prefactor = geometric_tunneling_prefactor(n_eff_proton)

    return ParticipationResult(
        n_eff=n_eff_all,
        n_eff_proton=n_eff_proton,
        n_active_modes=n_active,
        proton_fractions=fractions,
        active_mode_indices=active_indices,
        geometric_prefactor=prefactor,
    )


def effective_tunneling_dimension(
    eigenvectors: np.ndarray,
    frequencies_cm: np.ndarray,
    masses: np.ndarray,
    proton_indices: list[int],
    threshold: float = 0.05,
) -> int:
    """Count modes with significant proton participation.

    Simple integer count of modes where the proton kinetic energy
    fraction exceeds the threshold.

    Args:
        eigenvectors: Normal mode eigenvectors, shape (3N, 3N).
        frequencies_cm: Vibrational frequencies in cm⁻¹.
        masses: Atomic masses in amu.
        proton_indices: 0-based indices of the transferring proton(s).
        threshold: Minimum proton fraction.

    Returns:
        Number of proton-active modes.
    """
    fractions = mode_participation(eigenvectors, masses, proton_indices)
    active = (np.abs(frequencies_cm) > 50.0) & (fractions > threshold)
    return int(np.sum(active))


def geometric_tunneling_prefactor(n_eff: float, d_ref: float = 3.0) -> float:
    """Compute geometric correction prefactor for tunneling rate.

    Correction = sqrt(N_eff / d_ref)

    When N_eff = d_ref (a single atom in 3D), the correction is 1.0.
    When N_eff > d_ref, the tunneling rate is enhanced by additional
    participating dimensions.

    Args:
        n_eff: Effective tunneling dimensionality.
        d_ref: Reference dimensionality (default: 3 for a single atom in 3D).

    Returns:
        Multiplicative correction factor (≥ 0).
    """
    if n_eff <= 0 or d_ref <= 0:
        return 1.0

    return float(np.sqrt(n_eff / d_ref))


def tunneling_correction_report(
    eigenvectors: np.ndarray,
    frequencies_cm: np.ndarray,
    masses: np.ndarray,
    proton_indices: list[int],
    threshold: float = 0.05,
) -> dict:
    """Generate a summary report of tunneling dimensionality analysis.

    Args:
        eigenvectors: Normal mode eigenvectors.
        frequencies_cm: Frequencies in cm⁻¹.
        masses: Atomic masses in amu.
        proton_indices: Indices of transferring proton(s).
        threshold: Proton fraction threshold.

    Returns:
        Dictionary with all diagnostics.
    """
    result = proton_participation(
        eigenvectors, frequencies_cm, masses, proton_indices, threshold
    )

    report = {
        "n_eff_overall": result.n_eff,
        "n_eff_proton": result.n_eff_proton,
        "n_active_modes": result.n_active_modes,
        "geometric_prefactor": result.geometric_prefactor,
        "active_mode_indices": result.active_mode_indices.tolist(),
        "active_mode_frequencies_cm": frequencies_cm[result.active_mode_indices].tolist()
        if result.n_active_modes > 0
        else [],
        "active_mode_proton_fractions": result.proton_fractions[result.active_mode_indices].tolist()
        if result.n_active_modes > 0
        else [],
        "dominant_mode_idx": int(np.argmax(result.proton_fractions))
        if np.any(result.proton_fractions > 0)
        else None,
        "dominant_mode_proton_fraction": float(np.max(result.proton_fractions)),
    }

    return report
