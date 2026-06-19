"""
Electronic coupling (V_el) estimation for PCET reactions.

Two methods:
1. Empirical distance-based: exponential decay with donor-acceptor distance.
   Fast, approximate, useful for screening.
2. Generalized Mulliken-Hush (GMH): extracts V_el from TD-DFT adiabatic
   state energies and transition dipole moments. The standard method for
   computing diabatic couplings.

References:
    Cave & Newton, Chem. Phys. Lett. 249, 15 (1996).    — GMH method
    Cave & Newton, J. Chem. Phys. 106, 9213 (1997).     — GMH details
    Gray & Winkler, PNAS 102, 3534 (2005).               — distance decay
    Moser & Dutton, Biochim. Biophys. Acta 1101, 171 (1992). — Moser-Dutton ruler
    Hammes-Schiffer, Acc. Chem. Res. 34, 273 (2001).    — PCET couplings
"""

import math
import numpy as np
from dataclasses import dataclass

from pcet_engine.core.constants import (
    HARTREE_TO_KCALMOL,
    HARTREE_TO_EV,
    EV_TO_KCALMOL,
)


# ====================================================================
# Empirical distance-based coupling
# ====================================================================

# Default decay parameters from protein electron transfer literature.
# V_el(r) = V0 * exp(-beta * (r - r0) / 2)
# Note: the factor of 2 is because coupling decays as exp(-beta*r/2)
# while the rate (proportional to V_el^2) decays as exp(-beta*r).

# Moser-Dutton ruler (1992): log10(k_ET) = 15 - 0.6*R - 3.1*(ΔG+λ)²/λ
# where R is edge-to-edge distance.  The 0.6/Å in log10 corresponds to
# β_rate ≈ 1.4 Å⁻¹.  Gray-Winkler (2005) find β ≈ 1.1 Å⁻¹ for packed
# protein interiors.  We default to 1.1 (Gray-Winkler).

DEFAULT_DECAY_PARAMS = {
    "protein": {
        "V0_kcal": 80.0,       # kcal/mol at van der Waals contact
        "beta": 1.1,           # Å⁻¹ (rate decay; coupling decays at beta/2)
        "r0": 3.6,             # Å (van der Waals contact distance)
        "description": "Moser-Dutton ruler for protein ET (β ≈ 1.1 Å⁻¹)",
    },
    "through_bond": {
        "V0_kcal": 100.0,
        "beta": 0.7,
        "r0": 3.0,
        "description": "Through-bond coupling in covalent bridges",
    },
    "through_space": {
        "V0_kcal": 50.0,
        "beta": 2.8,
        "r0": 3.6,
        "description": "Through-space (vacuum) coupling",
    },
    "pcet_hydrogen_bond": {
        "V0_kcal": 2.0,
        "beta": 1.5,
        "r0": 2.5,
        "description": "PCET through hydrogen bond interface (shorter range)",
    },
}


@dataclass
class EmpiricalCouplingResult:
    """Result of empirical V_el estimation.

    Attributes:
        V_el_kcal: Electronic coupling in kcal/mol.
        V_el_eV: Electronic coupling in eV.
        d_DA: Donor-acceptor distance in Å.
        beta: Decay constant used (Å⁻¹).
        V0_kcal: Contact coupling used (kcal/mol).
        r0: Contact distance used (Å).
        medium: Medium type used for parameters.
    """

    V_el_kcal: float
    V_el_eV: float
    d_DA: float
    beta: float
    V0_kcal: float
    r0: float
    medium: str


def empirical_coupling(
    d_DA: float,
    medium: str = "protein",
    V0_kcal: float | None = None,
    beta: float | None = None,
    r0: float | None = None,
) -> EmpiricalCouplingResult:
    """Estimate electronic coupling from donor-acceptor distance.

    Uses exponential decay:  V_el = V0 * exp(-β(r - r0) / 2)

    The factor of 2 accounts for the relationship between coupling
    decay and rate decay: k ∝ V_el² ∝ exp(-β(r-r0)).

    Args:
        d_DA: Donor-acceptor distance in angstrom.
        medium: Coupling medium type. Options:
            'protein' — Moser-Dutton ruler (β ≈ 1.1 Å⁻¹)
            'through_bond' — Covalent bridge (β ≈ 0.7 Å⁻¹)
            'through_space' — Vacuum/solvent (β ≈ 2.8 Å⁻¹)
            'pcet_hydrogen_bond' — H-bond interface (β ≈ 1.5 Å⁻¹)
        V0_kcal: Override contact coupling (kcal/mol).
        beta: Override decay constant (Å⁻¹).
        r0: Override contact distance (Å).

    Returns:
        EmpiricalCouplingResult.
    """
    if medium not in DEFAULT_DECAY_PARAMS:
        raise ValueError(
            f"Unknown medium '{medium}'. Options: {list(DEFAULT_DECAY_PARAMS.keys())}"
        )

    params = DEFAULT_DECAY_PARAMS[medium]
    _V0 = V0_kcal if V0_kcal is not None else params["V0_kcal"]
    _beta = beta if beta is not None else params["beta"]
    _r0 = r0 if r0 is not None else params["r0"]

    if d_DA <= 0:
        raise ValueError(f"d_DA must be positive, got {d_DA}")

    # V_el = V0 * exp(-beta * (r - r0) / 2)
    V_el = _V0 * math.exp(-_beta * (d_DA - _r0) / 2.0)

    return EmpiricalCouplingResult(
        V_el_kcal=V_el,
        V_el_eV=V_el / EV_TO_KCALMOL,
        d_DA=d_DA,
        beta=_beta,
        V0_kcal=_V0,
        r0=_r0,
        medium=medium,
    )


# ====================================================================
# Generalized Mulliken-Hush (GMH)
# ====================================================================


@dataclass
class GMHResult:
    """Result of GMH coupling extraction.

    Attributes:
        V_el_kcal: Electronic coupling in kcal/mol.
        V_el_eV: Electronic coupling in eV.
        V_el_hartree: Electronic coupling in hartree.
        delta_E_adiabatic: Adiabatic energy gap in eV.
        mu_12: Transition dipole moment magnitude in Debye.
        delta_mu: Permanent dipole difference |μ₂ - μ₁| in Debye.
        mixing_angle: Rotation angle θ (radians) from adiabatic to diabatic.
    """

    V_el_kcal: float
    V_el_eV: float
    V_el_hartree: float
    delta_E_adiabatic: float
    mu_12: float
    delta_mu: float
    mixing_angle: float


def gmh_coupling(
    delta_E: float,
    mu_12: float,
    delta_mu: float,
) -> GMHResult:
    """Compute electronic coupling via the Generalized Mulliken-Hush method.

    The GMH formula for a two-state system:

        V_el = |μ₁₂| × ΔE / √(Δμ² + 4μ₁₂²)

    where:
        ΔE = E₂ - E₁  (adiabatic energy gap)
        μ₁₂ = transition dipole moment between adiabatic states
        Δμ = |μ₂ - μ₁| (permanent dipole moment difference)

    The mixing angle θ satisfying tan(2θ) = 2μ₁₂/Δμ rotates from
    the adiabatic to diabatic basis.

    Args:
        delta_E: Adiabatic energy gap (E₂ - E₁) in eV. Must be positive.
        mu_12: Transition dipole moment magnitude in Debye.
        delta_mu: Permanent dipole moment difference |μ₂ - μ₁| in Debye.

    Returns:
        GMHResult with V_el in multiple units.
    """
    if delta_E <= 0:
        raise ValueError(f"delta_E must be positive, got {delta_E}")
    if mu_12 < 0:
        raise ValueError(f"mu_12 must be non-negative, got {mu_12}")
    if delta_mu < 0:
        raise ValueError(f"delta_mu must be non-negative, got {delta_mu}")

    # GMH formula
    denominator = math.sqrt(delta_mu**2 + 4.0 * mu_12**2)
    if denominator < 1e-15:
        raise ValueError("Both mu_12 and delta_mu are effectively zero — cannot extract coupling.")

    V_el_eV = abs(mu_12) * delta_E / denominator

    # Mixing angle
    if abs(delta_mu) > 1e-15:
        mixing_angle = 0.5 * math.atan2(2.0 * mu_12, delta_mu)
    else:
        mixing_angle = math.pi / 4.0  # Symmetric case

    V_el_hartree = V_el_eV / HARTREE_TO_EV
    V_el_kcal = V_el_hartree * HARTREE_TO_KCALMOL

    return GMHResult(
        V_el_kcal=V_el_kcal,
        V_el_eV=V_el_eV,
        V_el_hartree=V_el_hartree,
        delta_E_adiabatic=delta_E,
        mu_12=mu_12,
        delta_mu=delta_mu,
        mixing_angle=mixing_angle,
    )


def gmh_coupling_from_tddft(
    excitation_energy_eV: float,
    ground_state_dipole: np.ndarray,
    excited_state_dipole: np.ndarray,
    transition_dipole: np.ndarray,
) -> GMHResult:
    """Extract V_el from TD-DFT output using GMH.

    This is the standard workflow:
    1. Run ground-state DFT → get μ₁ (ground-state permanent dipole)
    2. Run TD-DFT → get excitation energy ΔE, transition dipole μ₁₂
    3. Run excited-state optimization or CIS → get μ₂ (excited-state dipole)

    For CT states in PCET, the two diabatic states are the pre-ET and
    post-ET electronic configurations. The adiabatic states (S₀, S₁)
    from TD-DFT mix these.

    Args:
        excitation_energy_eV: Vertical excitation energy in eV.
        ground_state_dipole: Ground-state permanent dipole (x, y, z) in Debye.
        excited_state_dipole: Excited-state permanent dipole (x, y, z) in Debye.
        transition_dipole: Transition dipole moment (x, y, z) in Debye.

    Returns:
        GMHResult with extracted coupling.
    """
    ground_state_dipole = np.asarray(ground_state_dipole, dtype=float)
    excited_state_dipole = np.asarray(excited_state_dipole, dtype=float)
    transition_dipole = np.asarray(transition_dipole, dtype=float)

    delta_E = excitation_energy_eV
    mu_12 = float(np.linalg.norm(transition_dipole))
    delta_mu = float(np.linalg.norm(excited_state_dipole - ground_state_dipole))

    return gmh_coupling(delta_E, mu_12, delta_mu)


def gmh_coupling_multistate(
    energies_eV: np.ndarray,
    dipole_matrix: np.ndarray,
) -> list[GMHResult]:
    """Multi-state GMH for systems with >2 diabatic states.

    Diagonalizes the dipole moment matrix in the space of adiabatic
    states to find the diabatic basis that maximizes charge localization.

    The diabatic coupling between states i and j is:
        H_ij^d = Σ_a Σ_b U_ai * E_a * δ_ab * U_bj

    where U is the unitary transformation from adiabatic to diabatic
    basis (eigenvectors of the projected dipole matrix).

    Args:
        energies_eV: Adiabatic state energies in eV, shape (n_states,).
        dipole_matrix: Dipole moment matrix in Debye, shape (n_states, n_states, 3).
            Diagonal elements are permanent dipoles, off-diagonal are transition dipoles.

    Returns:
        List of GMHResult for each pair of diabatic states.
    """
    energies_eV = np.asarray(energies_eV, dtype=float)
    dipole_matrix = np.asarray(dipole_matrix, dtype=float)
    n_states = len(energies_eV)

    if dipole_matrix.shape != (n_states, n_states, 3):
        raise ValueError(
            f"dipole_matrix shape {dipole_matrix.shape} inconsistent with "
            f"{n_states} states (expected ({n_states}, {n_states}, 3))"
        )

    # Project dipole matrix onto largest component (charge-transfer direction)
    # Use the ground-to-first-excited permanent dipole difference as CT axis
    if n_states >= 2:
        ct_axis = dipole_matrix[1, 1, :] - dipole_matrix[0, 0, :]
        ct_norm = np.linalg.norm(ct_axis)
        if ct_norm > 1e-10:
            ct_axis /= ct_norm
        else:
            ct_axis = np.array([0.0, 0.0, 1.0])
    else:
        ct_axis = np.array([0.0, 0.0, 1.0])

    # Project dipole matrix onto CT axis
    mu_proj = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            mu_proj[i, j] = np.dot(dipole_matrix[i, j, :], ct_axis)

    # Diagonalize projected dipole matrix → diabatic basis
    mu_eigenvalues, U = np.linalg.eigh(mu_proj)

    # Transform Hamiltonian to diabatic basis: H_d = U^T H_a U
    H_adiabatic = np.diag(energies_eV)
    H_diabatic = U.T @ H_adiabatic @ U

    # Extract couplings between all diabatic state pairs
    results = []
    for i in range(n_states):
        for j in range(i + 1, n_states):
            V_eV = abs(H_diabatic[i, j])
            delta_E = abs(energies_eV[j] - energies_eV[i])

            # Reconstruct effective mu_12 and delta_mu for reporting
            delta_mu_eff = abs(mu_eigenvalues[j] - mu_eigenvalues[i])
            mu_12_eff = abs(mu_proj[i, j]) if i < n_states and j < n_states else 0.0

            V_hartree = V_eV / HARTREE_TO_EV
            V_kcal = V_hartree * HARTREE_TO_KCALMOL

            mixing_angle = 0.0
            if abs(delta_mu_eff) > 1e-15:
                mixing_angle = 0.5 * math.atan2(2.0 * mu_12_eff, delta_mu_eff)

            results.append(GMHResult(
                V_el_kcal=V_kcal,
                V_el_eV=V_eV,
                V_el_hartree=V_hartree,
                delta_E_adiabatic=delta_E,
                mu_12=mu_12_eff,
                delta_mu=delta_mu_eff,
                mixing_angle=mixing_angle,
            ))

    return results
