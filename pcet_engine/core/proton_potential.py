"""
Proton potential energy surface handling for PCET calculations.

Provides tools for constructing, fitting, and manipulating 1D proton
potential energy curves needed for FGH-based vibronic rate calculations.

Supports:
    - Harmonic oscillator potentials (from frequency + equilibrium position)
    - Morse potentials (anharmonic single-well)
    - Double-well potentials (donor + acceptor Morse wells with coupling)
    - Numerical potentials from DFT/QM scans (polynomial or spline fits)
    - Hessian-derived potentials (from normal mode analysis)
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import BSpline, splrep
from typing import Callable

from pcet_engine.core.constants import (
    HARTREE_TO_EV,
    KCALMOL_TO_HARTREE,
    ANGSTROM_TO_BOHR,
    AMU_TO_AU,
    CM_TO_HARTREE,
)


# =====================================================================
# Analytic potential generators
# =====================================================================

def harmonic_potential(
    omega_cm: float,
    mass_amu: float,
    r_eq: float = 0.0,
    e_offset: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a harmonic oscillator potential.

    V(r) = ½ m ω² (r - r_eq)² + e_offset

    Args:
        omega_cm: Vibrational frequency in cm⁻¹.
        mass_amu: Particle mass in amu.
        r_eq: Equilibrium position in angstrom.
        e_offset: Energy offset in eV.

    Returns:
        Callable V(r) taking r in angstrom, returning V in eV.
    """
    omega_au = omega_cm * CM_TO_HARTREE
    mass_au = mass_amu * AMU_TO_AU
    # Force constant in hartree/bohr²
    k_au = mass_au * omega_au**2

    def V(r: np.ndarray) -> np.ndarray:
        r_bohr = (np.asarray(r) - r_eq) * ANGSTROM_TO_BOHR
        return 0.5 * k_au * r_bohr**2 * HARTREE_TO_EV + e_offset

    return V


def morse_potential(
    De_eV: float,
    beta_inv_angstrom: float,
    r_eq: float = 0.0,
    e_offset: float = 0.0,
    inverted: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a Morse potential.

    V(r) = De × (1 - exp(-β(r - r_eq)))² + e_offset

    For inverted Morse (acceptor well):
    V(r) = De × (1 - exp(+β(r - r_eq)))² + e_offset

    Args:
        De_eV: Well depth in eV.
        beta_inv_angstrom: Morse parameter β in Å⁻¹.
        r_eq: Equilibrium position in angstrom.
        e_offset: Energy offset in eV.
        inverted: If True, use inverted Morse (for acceptor well).

    Returns:
        Callable V(r) in eV.
    """
    sign = 1.0 if not inverted else -1.0

    def V(r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r)
        return De_eV * (1.0 - np.exp(-sign * beta_inv_angstrom * (r_arr - r_eq)))**2 + e_offset

    return V


def double_well_potential(
    De_donor: float,
    De_acceptor: float,
    beta_donor: float,
    beta_acceptor: float,
    R_DA: float,
    delta_E: float = 0.0,
    V_coupling: float = 0.5,
    n_grid: int = 500,
    smooth: str = "bspline",
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a double-well proton potential from two Morse wells.

    Constructs two diabatic Morse potentials (donor and acceptor wells)
    centered at ±R_DA/2, then diagonalizes the 2×2 matrix to get the
    adiabatic ground-state double well.

    Args:
        De_donor: Donor well depth in eV.
        De_acceptor: Acceptor well depth in eV.
        beta_donor: Donor Morse β in Å⁻¹.
        beta_acceptor: Acceptor Morse β in Å⁻¹.
        R_DA: Donor-acceptor separation for proton coordinate in Å.
        delta_E: Energy offset between acceptor and donor minima in eV.
        V_coupling: Coupling strength at the crossing point in eV.
        n_grid: Number of points for internal construction.
        smooth: Smoothing method ('bspline', 'poly6', 'poly8').

    Returns:
        Callable V(r) in eV, with V(r_min) = 0.
    """
    r_int = np.linspace(-1.2 * R_DA, 1.2 * R_DA, n_grid)

    # Donor Morse centered at -R_DA/2, acceptor inverted Morse at +R_DA/2
    E_D = De_donor * (1.0 - np.exp(-beta_donor * (r_int + R_DA / 2)))**2
    E_A = De_acceptor * (1.0 - np.exp(beta_acceptor * (r_int - R_DA / 2)))**2 + delta_E

    # Find crossing region for Gaussian coupling
    diff = E_D - E_A
    crossings = []
    for i in range(1, len(r_int)):
        if diff[i] * diff[i - 1] <= 0:
            crossings.append(r_int[i] if abs(diff[i]) < abs(diff[i - 1]) else r_int[i - 1])

    if len(crossings) < 1:
        # No crossing — use midpoint
        r_cross = 0.0
    else:
        # Use the middle crossing
        r_cross = crossings[len(crossings) // 2]

    # Gaussian coupling centered at crossing
    V_coup = V_coupling * np.exp(-(r_int - r_cross)**2 / (R_DA**2))

    # Diagonalize 2×2 at each grid point
    E_lower = 0.5 * (E_A + E_D - np.sqrt((E_A - E_D)**2 + 4 * V_coup**2))

    # Smooth the result
    E_lower -= np.min(E_lower)

    if smooth == "bspline":
        tck = splrep(r_int, E_lower, s=5e-3)
        def V(r):
            r_arr = np.asarray(r)
            vals = BSpline(*tck)(r_arr)
            return vals - np.min(BSpline(*tck)(r_int))
    elif smooth == "poly8":
        V = _fit_polynomial(r_int, E_lower, degree=8)
    elif smooth == "poly6":
        V = _fit_polynomial(r_int, E_lower, degree=6)
    else:
        raise ValueError(f"smooth must be 'bspline', 'poly6', or 'poly8', got '{smooth}'")

    return V


# =====================================================================
# Numerical potential fitting
# =====================================================================

def fit_potential_from_scan(
    r_points: np.ndarray,
    energies: np.ndarray,
    method: str = "bspline",
    smoothing: float = 5e-3,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a smooth callable potential from discrete scan points.

    Args:
        r_points: Proton positions in angstrom, shape (N,).
        energies: Potential energies in eV, shape (N,).
        method: 'bspline', 'poly6', or 'poly8'.
        smoothing: Smoothing parameter for bspline (ignored for polynomial).

    Returns:
        Callable V(r) in eV, shifted so minimum = 0.
    """
    energies_shifted = energies - np.min(energies)

    if method == "bspline":
        tck = splrep(r_points, energies_shifted, s=smoothing)
        r_fine = np.linspace(np.min(r_points), np.max(r_points), 500)

        def V(r):
            return BSpline(*tck)(np.asarray(r)) - np.min(BSpline(*tck)(r_fine))
        return V

    elif method in ("poly6", "poly8"):
        degree = 6 if method == "poly6" else 8
        return _fit_polynomial(r_points, energies_shifted, degree)

    else:
        raise ValueError(f"method must be 'bspline', 'poly6', or 'poly8', got '{method}'")


def potential_from_hessian(
    omega_cm: float,
    mass_amu: float,
    r_eq_donor: float,
    r_eq_acceptor: float,
    delta_E_eV: float = 0.0,
    anharmonicity_cm: float = 0.0,
) -> tuple[Callable, Callable]:
    """Generate reactant and product proton potentials from Hessian-derived data.

    Creates two displaced potentials: a harmonic (or Morse) well for the
    reactant centered at r_eq_donor and one for the product at r_eq_acceptor.

    If anharmonicity_cm is nonzero, Morse potentials are used with:
        De = ω² / (4 × ωχ)
        β = ω × sqrt(2m / (2 De))

    Args:
        omega_cm: Proton frequency in cm⁻¹.
        mass_amu: Particle mass in amu.
        r_eq_donor: Donor-H equilibrium position in angstrom.
        r_eq_acceptor: Acceptor-H equilibrium position in angstrom.
        delta_E_eV: Product well energy offset in eV (negative = exothermic).
        anharmonicity_cm: Anharmonicity constant ωχ in cm⁻¹ (0 = harmonic).

    Returns:
        Tuple of (V_reactant, V_product), both callable V(r) in eV.
    """
    if anharmonicity_cm > 0:
        omega_eV = omega_cm * CM_TO_HARTREE * HARTREE_TO_EV
        wx_eV = anharmonicity_cm * CM_TO_HARTREE * HARTREE_TO_EV

        De_eV = omega_eV**2 / (4.0 * wx_eV)
        mass_au = mass_amu * AMU_TO_AU
        omega_au = omega_cm * CM_TO_HARTREE
        beta_bohr = omega_au * np.sqrt(mass_au / (2.0 * De_eV / HARTREE_TO_EV))
        beta_angstrom = beta_bohr * ANGSTROM_TO_BOHR

        V_R = morse_potential(De_eV, beta_angstrom, r_eq=r_eq_donor)
        V_P = morse_potential(De_eV, beta_angstrom, r_eq=r_eq_acceptor,
                              e_offset=delta_E_eV, inverted=True)
    else:
        V_R = harmonic_potential(omega_cm, mass_amu, r_eq=r_eq_donor)
        V_P = harmonic_potential(omega_cm, mass_amu, r_eq=r_eq_acceptor,
                                 e_offset=delta_E_eV)

    return V_R, V_P


# =====================================================================
# Internal helpers
# =====================================================================

def _fit_polynomial(
    r_data: np.ndarray,
    e_data: np.ndarray,
    degree: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit data to a polynomial and return a callable."""
    coeffs = np.polyfit(r_data, e_data - np.min(e_data), degree)

    # Refit on a finer grid for stability
    r_fine = np.linspace(np.min(r_data), np.max(r_data), 500)
    e_fine = np.polyval(coeffs, r_fine)
    coeffs = np.polyfit(r_fine, e_fine - np.min(e_fine), degree)

    def V(r):
        return np.polyval(coeffs, np.asarray(r))

    return V
