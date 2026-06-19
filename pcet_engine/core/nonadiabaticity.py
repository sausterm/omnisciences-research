"""
Nonadiabaticity analysis for PCET reactions.

Determines whether a PCET reaction is in the nonadiabatic, adiabatic,
or intermediate regime by computing the adiabaticity parameter p and
the Georgievskii-Stuchebrukhov κ factor.

In the nonadiabatic limit (p << 1): V_eff = V_el × S_μν
In the adiabatic limit (p >> 1):    V_eff = ½ × tunneling splitting
General case:                       V_eff = κ × V_ad

References:
    Georgievskii, Y.; Stuchebrukhov, A. A.
    J. Chem. Phys. 2000, 113, 10438-10450.

    Hammes-Schiffer, S.
    Acc. Chem. Res. 2001, 34, 273-281.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.special import gamma
from dataclasses import dataclass

from pcet_engine.core.constants import (
    KB_EV,
    HARTREE_TO_EV,
    EV_TO_HARTREE,
    AMU_TO_AU,
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    PROTON_MASS_AMU,
)
from pcet_engine.core.fgh_solver import fgh_1d, compute_fc_overlaps


HBAR_EV_S = 6.582119569e-16  # ℏ in eV·s


@dataclass
class NonadiabaticityResult:
    """Result of nonadiabaticity analysis.

    Attributes:
        tau_p: Proton tunneling time in seconds.
        tau_e: Electronic transition time in seconds.
        p: Adiabaticity parameter (tau_p / tau_e).
            p << 1: nonadiabatic. p >> 1: adiabatic.
        kappa: Georgievskii-Stuchebrukhov κ factor.
        V_nonadiabatic: Nonadiabatic vibronic coupling V_el × S_μν in eV.
        V_adiabatic: Adiabatic vibronic coupling (½ tunneling splitting) in eV.
        V_semiclassical: General vibronic coupling κ × V_ad in eV.
        regime: 'nonadiabatic', 'adiabatic', or 'intermediate'.
        r_crossing: Proton coordinate at the diabatic crossing point in Å.
        E_crossing: Energy at the crossing point in eV.
    """
    tau_p: float
    tau_e: float
    p: float
    kappa: float
    V_nonadiabatic: float
    V_adiabatic: float
    V_semiclassical: float
    regime: str
    r_crossing: float
    E_crossing: float


def analyze_nonadiabaticity(
    r_grid: np.ndarray,
    V_reactant: np.ndarray,
    V_product: np.ndarray,
    V_el: float,
    mass_amu: float = PROTON_MASS_AMU,
    mu: int = 0,
    nu: int = 0,
    n_states: int = 10,
) -> NonadiabaticityResult:
    """Analyze the nonadiabaticity of a PCET reaction.

    Given reactant and product proton potentials and the electronic
    coupling, determines the adiabaticity regime and computes the
    effective vibronic coupling.

    Args:
        r_grid: Proton coordinate grid in angstrom, shape (N,).
        V_reactant: Reactant proton potential in eV, shape (N,).
        V_product: Product proton potential in eV, shape (N,).
        V_el: Electronic coupling in eV (constant or at crossing point).
        mass_amu: Particle mass in amu.
        mu: Reactant vibronic state index.
        nu: Product vibronic state index.
        n_states: Number of FGH states to compute.

    Returns:
        NonadiabaticityResult with regime classification and couplings.
    """
    n_states = max(n_states, mu + 1, nu + 1)

    # Solve for vibrational states in both potentials
    E_R, wfcs_R, _ = fgh_1d(r_grid, V_reactant, mass_amu, n_states)
    E_P, wfcs_P, _ = fgh_1d(r_grid, V_product, mass_amu, n_states)

    # Shift potentials so that the μ and ν levels are aligned
    E_mu = E_R[mu]
    E_nu = E_P[nu]
    if E_nu < E_mu:
        shift_R = 0.0
        shift_P = E_mu - E_nu
    else:
        shift_R = E_nu - E_mu
        shift_P = 0.0

    V_R_shifted = V_reactant + shift_R
    V_P_shifted = V_product + shift_P
    E_R_shifted = E_R + shift_R

    # Find crossing point of shifted diabatic potentials
    diff = V_R_shifted - V_P_shifted
    r_cross = None
    E_cross = None
    for i in range(1, len(r_grid)):
        if diff[i] * diff[i - 1] <= 0:
            # Linear interpolation
            frac = abs(diff[i - 1]) / (abs(diff[i - 1]) + abs(diff[i]) + 1e-30)
            r_cross = r_grid[i - 1] + frac * (r_grid[i] - r_grid[i - 1])
            E_cross = 0.25 * (V_R_shifted[i - 1] + V_R_shifted[i] +
                              V_P_shifted[i - 1] + V_P_shifted[i])
            break

    if r_cross is None:
        # No crossing — potentials don't intersect
        # Fall back to nonadiabatic estimate
        S = compute_fc_overlaps(wfcs_R[mu:mu+1], wfcs_P[nu:nu+1], r_grid)
        V_nad = V_el * abs(S[0, 0])
        return NonadiabaticityResult(
            tau_p=float('inf'), tau_e=HBAR_EV_S / V_el,
            p=0.0, kappa=0.0,
            V_nonadiabatic=V_nad, V_adiabatic=0.0, V_semiclassical=V_nad,
            regime='nonadiabatic',
            r_crossing=float('nan'), E_crossing=float('nan'),
        )

    # Slopes at crossing point
    idx = np.argmin(np.abs(r_grid - r_cross))
    idx = max(1, min(idx, len(r_grid) - 2))
    dr = r_grid[idx] - r_grid[idx - 1]
    slope_R = (V_R_shifted[idx] - V_R_shifted[idx - 1]) / dr
    slope_P = (V_P_shifted[idx] - V_P_shifted[idx - 1]) / dr
    delta_slope = abs(slope_R - slope_P)

    # Tunneling energy and velocity
    E0 = E_R_shifted[mu]
    if E0 >= E_cross:
        # Above barrier — classical regime
        return NonadiabaticityResult(
            tau_p=0.0, tau_e=HBAR_EV_S / V_el,
            p=float('inf'), kappa=1.0,
            V_nonadiabatic=0.0, V_adiabatic=V_el, V_semiclassical=V_el,
            regime='adiabatic',
            r_crossing=r_cross, E_crossing=E_cross,
        )

    # Proton tunneling velocity at crossing in Å/s
    mass_au = mass_amu * AMU_TO_AU
    dE_Ha = (E_cross - E0) * EV_TO_HARTREE
    v_tunnel_bohr_per_atu = np.sqrt(2.0 * dE_Ha / mass_au)
    au_time_to_s = 2.4188843265857e-17
    v_tunnel_A_per_s = v_tunnel_bohr_per_atu * BOHR_TO_ANGSTROM / au_time_to_s

    # Characteristic times
    tau_p = V_el / (delta_slope * v_tunnel_A_per_s) if delta_slope > 0 else float('inf')
    tau_e = HBAR_EV_S / V_el

    # Adiabaticity parameter
    p = tau_p / tau_e if tau_e > 0 else float('inf')

    # Georgievskii-Stuchebrukhov κ
    if p > 0 and p < 500:
        kappa = np.sqrt(2.0 * np.pi * p) * np.exp(p * np.log(p) - p) / gamma(p + 1.0)
    elif p >= 500:
        kappa = 1.0  # Stirling limit
    else:
        kappa = np.sqrt(2.0 * np.pi * p)  # Small p limit

    # Nonadiabatic coupling: V_el × S_μν
    S = compute_fc_overlaps(wfcs_R[mu:mu+1], wfcs_P[nu:nu+1], r_grid)
    V_nad = V_el * abs(S[0, 0])

    # Adiabatic coupling: ½ tunneling splitting on adiabatic potential
    V_ad = _compute_tunneling_splitting(
        r_grid, V_R_shifted, V_P_shifted, V_el, mass_amu,
        wfcs_R[mu], wfcs_P[nu], n_states,
    )

    V_sc = kappa * V_ad

    # Classify regime
    if p < 0.1:
        regime = 'nonadiabatic'
    elif p > 10:
        regime = 'adiabatic'
    else:
        regime = 'intermediate'

    return NonadiabaticityResult(
        tau_p=tau_p, tau_e=tau_e, p=p, kappa=kappa,
        V_nonadiabatic=V_nad, V_adiabatic=V_ad, V_semiclassical=V_sc,
        regime=regime, r_crossing=r_cross, E_crossing=E_cross,
    )


def _compute_tunneling_splitting(
    r_grid: np.ndarray,
    V_R: np.ndarray,
    V_P: np.ndarray,
    V_el: float,
    mass_amu: float,
    wfc_R_mu: np.ndarray,
    wfc_P_nu: np.ndarray,
    n_states: int,
) -> float:
    """Compute ½ × tunneling splitting on adiabatic ground-state potential.

    Constructs the adiabatic potential by diagonalizing the 2×2 diabatic
    matrix at each grid point, then solves for vibrational states on the
    adiabatic ground state.

    Returns V_ad in eV.
    """
    # Construct adiabatic potentials
    delta = V_P - V_R
    E_avg = 0.5 * (V_R + V_P)
    E_gs = E_avg - 0.5 * np.sqrt(delta**2 + 4.0 * V_el**2)

    # Solve FGH on adiabatic ground state — need 2× states to find split pairs
    n_adia = 2 * n_states
    E_adia, wfcs_adia, _ = fgh_1d(r_grid, E_gs, mass_amu, n_adia)

    # Build symmetric and antisymmetric combinations of diabatic wavefunctions
    S_sign = simpson(wfc_R_mu * wfc_P_nu, x=r_grid)
    sign = 1.0 if S_sign > 0 else -1.0
    wfc_symm = wfc_R_mu + sign * wfc_P_nu
    wfc_anti = wfc_R_mu - sign * wfc_P_nu

    norm_s = np.sqrt(simpson(wfc_symm**2, x=r_grid))
    norm_a = np.sqrt(simpson(wfc_anti**2, x=r_grid))
    if norm_s > 0:
        wfc_symm /= norm_s
    if norm_a > 0:
        wfc_anti /= norm_a

    # Find adiabatic states with maximum overlap to symmetric/antisymmetric
    overlap_symm = np.array([abs(simpson(w * wfc_symm, x=r_grid)) for w in wfcs_adia])
    overlap_anti = np.array([abs(simpson(w * wfc_anti, x=r_grid)) for w in wfcs_adia])

    idx_symm = np.argmax(overlap_symm)
    # Find best antisymmetric match that's a different state
    overlap_anti_masked = overlap_anti.copy()
    overlap_anti_masked[idx_symm] = 0.0
    idx_anti = np.argmax(overlap_anti_masked)

    # Tunneling splitting = energy difference between symmetric/antisymmetric
    splitting = abs(E_adia[idx_anti] - E_adia[idx_symm])
    return 0.5 * splitting
