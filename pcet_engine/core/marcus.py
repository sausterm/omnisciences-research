"""
Marcus theory rate constants for electron and proton transfer.

Implements the semiclassical Marcus rate expression:

    k = (2π/ℏ) × |V|² × (1/√(4πλkBT)) × exp(-(ΔG° + λ)²/(4λkBT))

where:
    V    = electronic coupling matrix element
    λ    = total reorganization energy (inner-sphere + outer-sphere)
    ΔG°  = reaction driving force (free energy of reaction)
    T    = temperature

References:
    Marcus, R. A. J. Chem. Phys. 24, 966 (1956).
    Marcus, R. A. & Sutin, N. Biochim. Biophys. Acta 811, 265 (1985).
"""

import math
import numpy as np

from pcet_engine.core.constants import (
    KB_HARTREE,
    HBAR_AU,
    TWO_PI,
    HARTREE_TO_KCALMOL,
    KCALMOL_TO_HARTREE,
    AU_RATE_TO_PER_S,
)


def marcus_activation_energy(
    delta_G: float,
    lambda_reorg: float,
) -> float:
    """Compute Marcus activation energy.

    E_a = (ΔG° + λ)² / (4λ)

    Args:
        delta_G: Reaction free energy in hartree (negative = exothermic).
        lambda_reorg: Total reorganization energy in hartree.

    Returns:
        Activation energy in hartree.
    """
    if lambda_reorg <= 0:
        raise ValueError(f"Reorganization energy must be positive, got {lambda_reorg}")
    return (delta_G + lambda_reorg) ** 2 / (4.0 * lambda_reorg)


def marcus_rate(
    V_coupling: float,
    delta_G: float,
    lambda_reorg: float,
    temperature: float = 298.15,
) -> float:
    """Compute Marcus theory rate constant.

    k = (2π/ℏ) × |V|² / √(4πλkBT) × exp(-(ΔG° + λ)²/(4λkBT))

    Args:
        V_coupling: Electronic coupling in hartree.
        delta_G: Reaction free energy in hartree.
        lambda_reorg: Total reorganization energy in hartree.
        temperature: Temperature in Kelvin.

    Returns:
        Rate constant in s⁻¹.
    """
    kBT = KB_HARTREE * temperature

    E_a = marcus_activation_energy(delta_G, lambda_reorg)

    prefactor = (TWO_PI / HBAR_AU) * V_coupling**2 / math.sqrt(4.0 * math.pi * lambda_reorg * kBT)
    rate_au = prefactor * math.exp(-E_a / kBT)

    return rate_au * AU_RATE_TO_PER_S


def marcus_rate_kcal(
    V_coupling_kcal: float,
    delta_G_kcal: float,
    lambda_reorg_kcal: float,
    temperature: float = 298.15,
) -> float:
    """Convenience wrapper accepting energies in kcal/mol.

    Args:
        V_coupling_kcal: Electronic coupling in kcal/mol.
        delta_G_kcal: Reaction free energy in kcal/mol.
        lambda_reorg_kcal: Reorganization energy in kcal/mol.
        temperature: Temperature in Kelvin.

    Returns:
        Rate constant in s⁻¹.
    """
    return marcus_rate(
        V_coupling=V_coupling_kcal * KCALMOL_TO_HARTREE,
        delta_G=delta_G_kcal * KCALMOL_TO_HARTREE,
        lambda_reorg=lambda_reorg_kcal * KCALMOL_TO_HARTREE,
        temperature=temperature,
    )


def reorganization_energy_from_hessians(
    hessian_reactant: np.ndarray,
    hessian_product: np.ndarray,
    geom_reactant: np.ndarray,
    geom_product: np.ndarray,
    masses: np.ndarray,
    exclude_atoms: list[int] | None = None,
) -> tuple[float, float]:
    """Estimate inner-sphere reorganization energy from reactant/product Hessians.

    Uses the four-point method:
        λ_inner = ½(λ_f + λ_b)
    where:
        λ_f = E_product(R_reactant) - E_product(R_product)
        λ_b = E_reactant(R_product) - E_reactant(R_reactant)

    approximated by harmonic expansion:
        λ_f ≈ ½ ΔR^T · H_product · ΔR
        λ_b ≈ ½ ΔR^T · H_reactant · ΔR

    For PCET, the transferring proton coordinate is handled quantum
    mechanically by the vibronic formalism. Use ``exclude_atoms`` to
    remove the proton displacement from the classical reorganization
    energy calculation.

    Args:
        hessian_reactant: Cartesian Hessian at reactant geometry (3N x 3N), in hartree/bohr².
        hessian_product: Cartesian Hessian at product geometry, in hartree/bohr².
        geom_reactant: Reactant geometry as flat array (3N,) in bohr.
        geom_product: Product geometry as flat array (3N,) in bohr.
        masses: Atomic masses in amu, shape (N,) (unused, kept for API compatibility).
        exclude_atoms: 0-based indices of atoms whose displacement should be
                       excluded (e.g., the transferring proton).

    Returns:
        Tuple of (lambda_forward, lambda_backward) in hartree.
    """
    delta_R = geom_product - geom_reactant

    if exclude_atoms:
        delta_R = delta_R.copy()
        for idx in exclude_atoms:
            delta_R[3 * idx: 3 * idx + 3] = 0.0

    lambda_f = 0.5 * delta_R @ hessian_product @ delta_R
    lambda_b = 0.5 * delta_R @ hessian_reactant @ delta_R

    return lambda_f, lambda_b
