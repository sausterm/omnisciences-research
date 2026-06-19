"""
Fourier Grid Hamiltonian (FGH) solver for 1D Schrödinger equation.

Solves for bound-state eigenvalues and eigenfunctions of a particle
in an arbitrary 1D potential on an evenly-spaced grid.

The kinetic energy matrix is constructed in the momentum representation
using the Fourier grid method, then combined with the diagonal potential
matrix and diagonalized.

References:
    Marston, C. C.; Balint-Kurti, G. G.
    J. Chem. Phys. 1989, 91, 3571-3576.

    Balint-Kurti, G. G.; Ward, C. L.; Marston, C. C.
    Computer Physics Communications 1991, 67, 285-292.
"""

import numpy as np
from scipy.integrate import simpson

from pcet_engine.core.constants import (
    AMU_TO_AU,
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    EV_TO_HARTREE,
    PROTON_MASS_AMU,
    DEUTERIUM_MASS_AMU,
    TRITIUM_MASS_AMU,
)


def fgh_1d(
    r_grid: np.ndarray,
    potential: np.ndarray,
    mass_amu: float,
    n_states: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 1D Schrödinger equation on a uniform grid using FGH.

    Args:
        r_grid: Proton coordinate grid in angstrom, shape (N,).
            Must be uniformly spaced.
        potential: Potential energy at each grid point in eV, shape (N,).
        mass_amu: Particle mass in amu (1.008 for H, 2.014 for D).
        n_states: Number of eigenstates to return.

    Returns:
        Tuple of (energies, wavefunctions, r_grid):
            - energies: Eigenvalues in eV, shape (n_states,).
            - wavefunctions: Normalized wavefunctions in Å^(-1/2),
              shape (n_states, N). wavefunctions[i] is the i-th state.
            - r_grid: The input grid (passed through for convenience).
    """
    nx = len(r_grid)
    if nx < 8:
        raise ValueError(f"Grid must have at least 8 points, got {nx}")
    if len(potential) != nx:
        raise ValueError(f"potential length {len(potential)} != grid length {nx}")
    if n_states > nx:
        n_states = nx

    # Convert to atomic units
    r_bohr = r_grid * ANGSTROM_TO_BOHR
    pot_hartree = potential * EV_TO_HARTREE
    mass_au = mass_amu * AMU_TO_AU

    dx = r_bohr[1] - r_bohr[0]
    k_max = np.pi / dx

    # Build Hamiltonian in position representation
    hmat = np.zeros((nx, nx))

    for i in range(nx):
        for j in range(nx):
            if i == j:
                # Diagonal: kinetic + potential
                hmat[i, j] = (k_max**2) / 3.0 / (2.0 * mass_au) + pot_hartree[j]
            else:
                dji = j - i
                # Off-diagonal: kinetic only
                hmat[i, j] = (2.0 * k_max**2) / (np.pi**2) * ((-1)**dji) / (dji**2) / (2.0 * mass_au)

    # Diagonalize (symmetric → use eigh for speed + stability)
    eigvals, eigvecs = np.linalg.eigh(hmat)

    # Extract lowest n_states
    energies_hartree = eigvals[:n_states]
    energies_eV = energies_hartree * HARTREE_TO_EV

    # Normalize wavefunctions: ∫|ψ|² dr = 1 (in angstrom)
    raw_wfcs = eigvecs[:, :n_states].T  # shape (n_states, nx)
    wavefunctions = np.zeros_like(raw_wfcs)
    for i in range(n_states):
        wfc = raw_wfcs[i]
        norm_sq = simpson(wfc * wfc, x=r_grid)
        if norm_sq > 0:
            wavefunctions[i] = wfc / np.sqrt(norm_sq)
        else:
            wavefunctions[i] = wfc

    return energies_eV, wavefunctions, r_grid


def compute_fc_overlaps(
    wfcs_reactant: np.ndarray,
    wfcs_product: np.ndarray,
    r_grid: np.ndarray,
) -> np.ndarray:
    """Compute Franck-Condon overlap matrix from numerical wavefunctions.

    S_μν = ∫ ψ_μ^R(r) · ψ_ν^P(r) dr

    Args:
        wfcs_reactant: Reactant wavefunctions, shape (n_R, N).
        wfcs_product: Product wavefunctions, shape (n_P, N).
        r_grid: Grid points in angstrom, shape (N,).

    Returns:
        Overlap matrix S_μν, shape (n_R, n_P). Access |S|² as S**2.
    """
    n_R = wfcs_reactant.shape[0]
    n_P = wfcs_product.shape[0]
    S = np.zeros((n_R, n_P))

    for mu in range(n_R):
        for nu in range(n_P):
            S[mu, nu] = simpson(
                wfcs_reactant[mu] * wfcs_product[nu], x=r_grid
            )

    return S
