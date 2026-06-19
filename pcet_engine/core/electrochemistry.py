"""
Electrochemical PCET rate constants.

Extends the thermal vibronic PCET formalism to electrochemical systems where
the driving force depends on the electrode potential and the rate must be
integrated over electrode electronic states weighted by the Fermi-Dirac
distribution.

Supports:
    - Homogeneous electrochemical PCET (constant DOS, simple Fermi integration)
    - Heterogeneous electrochemical PCET (DFT-computed DOS, EDL model, work terms)
    - Tafel analysis (transfer coefficient extraction)

References:
    Hammes-Schiffer, S. et al. J. Am. Chem. Soc. 2015, 137, 8860.
    Hutchison, P. et al. ACS Catal. 2024, 19, 14363-14372.
    Huynh, M. H. V. et al. ACS Cent. Sci. 2017, 3, 372-380.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import fsolve
from dataclasses import dataclass

from pcet_engine.core.constants import (
    KB_HARTREE,
    KB_EV,
    HARTREE_TO_EV,
    EV_TO_HARTREE,
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
)


@dataclass
class ElectrochemicalResult:
    """Result from electrochemical PCET rate calculation.

    Attributes:
        k_H: Rate constant for H transfer in s⁻¹.
        k_D: Rate constant for D transfer in s⁻¹.
        KIE: Kinetic isotope effect k_H / k_D.
        overpotential: Applied overpotential in V.
        direction: 'anodic' or 'cathodic'.
        k_H_per_epsilon: H rate contribution at each electrode energy level.
        k_D_per_epsilon: D rate contribution at each electrode energy level.
        epsilons: Electrode energy levels sampled (in eV).
    """
    k_H: float
    k_D: float
    KIE: float
    overpotential: float
    direction: str
    k_H_per_epsilon: np.ndarray | None = None
    k_D_per_epsilon: np.ndarray | None = None
    epsilons: np.ndarray | None = None


@dataclass
class TafelResult:
    """Result from Tafel analysis.

    Attributes:
        alpha_H: Transfer coefficient for H.
        alpha_D: Transfer coefficient for D.
        potentials: Applied potentials in V.
        ln_k_H: ln(k_H) at each potential.
        ln_k_D: ln(k_D) at each potential.
    """
    alpha_H: float
    alpha_D: float
    potentials: np.ndarray
    ln_k_H: np.ndarray
    ln_k_D: np.ndarray


# =====================================================================
# Fermi-Dirac distribution
# =====================================================================

def fermi_dirac(
    epsilon: np.ndarray | float,
    E_fermi: float = 0.0,
    temperature: float = 298.15,
) -> np.ndarray:
    """Fermi-Dirac distribution.

    f(ε) = 1 / (exp((ε - E_F) / kBT) + 1)

    Args:
        epsilon: Energy level(s) in eV.
        E_fermi: Fermi level in eV (default 0, used as reference).
        temperature: Temperature in K.

    Returns:
        Occupation probability (dimensionless).
    """
    kBT = KB_EV * temperature
    x = (np.asarray(epsilon) - E_fermi) / kBT
    # Clip to avoid overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (np.exp(x) + 1.0)


# =====================================================================
# Electric double layer model
# =====================================================================

def edl_model(
    E_vs_ref: float,
    d_IHL: float,
    d_OHL: float,
    eps_IHL: float,
    eps_static: float,
    eps_optical: float,
    dipole_debye: float | str,
    rho_solvent: float,
    m_solvent: float,
    c_ions: float,
    C_EDL: float,
    PZFC_vs_ref: float,
    temperature: float = 298.15,
) -> callable:
    """Model the electric double layer near an electrode surface.

    Returns a function φ(R) giving the potential drop in volts as a
    function of distance R (in angstrom) from the electrode.

    Three regions:
        1. Inner Helmholtz Layer (0 < R < d_IHL): linear drop
        2. Outer Helmholtz Layer (d_IHL < R < d_IHL + d_OHL): linear with
           field-dependent dielectric (Langevin model)
        3. Diffuse Layer (R > d_IHL + d_OHL): Gouy-Chapman exponential decay

    Args:
        E_vs_ref: Applied potential vs. reference electrode (V).
        d_IHL: Inner Helmholtz layer thickness (Å).
        d_OHL: Outer Helmholtz layer thickness (Å).
        eps_IHL: Relative dielectric constant of IHL.
        eps_static: Static relative dielectric constant of bulk solvent.
        eps_optical: Optical relative dielectric constant of bulk solvent.
        dipole_debye: Solvent dipole moment in Debye, or 'calculate' to
            compute from Onsager-Kirkwood-Fröhlich equation.
        rho_solvent: Solvent density in g/cm³.
        m_solvent: Solvent molar mass in g/mol.
        c_ions: Electrolyte ion concentration in mol/L.
        C_EDL: EDL capacitance in μF/cm².
        PZFC_vs_ref: Potential of zero free charge vs. reference (V).
        temperature: Temperature in K.

    Returns:
        Callable φ(R) taking R in Å, returning potential drop in V.
    """
    NA = 6.02214076e23
    e_charge = 1.602176634e-19

    E_vs_PZFC = E_vs_ref - PZFC_vs_ref
    kBT_eV = KB_EV * temperature

    # Surface charge density in atomic units (e/bohr²)
    A2Bohr = ANGSTROM_TO_BOHR
    cm_to_A = 1e8
    sigma_M = (C_EDL * E_vs_PZFC) / (e_charge * 1e6 * (cm_to_A * A2Bohr)**2)

    # Solvent number density in bohr⁻³
    n_solvent = rho_solvent / m_solvent * NA / (cm_to_A * A2Bohr)**3

    # Solvent dipole in atomic units
    if dipole_debye == 'calculate':
        kBT_Ha = kBT_eV * EV_TO_HARTREE
        dipole_au = (3.0 / (2.0 + eps_optical)) * np.sqrt(
            3.0 * kBT_Ha * (eps_static - eps_optical) / (8.0 * np.pi * n_solvent)
        )
    else:
        Debye_to_au = 1.0 / 2.541746473
        dipole_au = float(dipole_debye) * Debye_to_au

    # Ion number density in bohr⁻³
    n_ions = (c_ions * NA * 1000) / (1e10 * A2Bohr)**3

    d_IHL_bohr = d_IHL * A2Bohr
    d_OHL_bohr = d_OHL * A2Bohr

    # Potential at outer Helmholtz plane (Gouy-Chapman)
    kBT_Ha = kBT_eV * EV_TO_HARTREE
    arg = sigma_M / np.sqrt((2.0 * kBT_Ha * eps_static * n_ions) / np.pi)
    arg = np.clip(arg, -1e10, 1e10)
    phi_OHP_Ha = 2.0 * kBT_Ha * np.arcsinh(arg)
    phi_OHP_V = phi_OHP_Ha * HARTREE_TO_EV

    # OHL dielectric as function of electric field (Langevin model)
    def _langevin(u):
        return 1.0 / np.tanh(u) - 1.0 / u

    def _eps_OHL(E_au):
        u = (2.0 + eps_optical) * dipole_au * E_au / (2.0 * kBT_Ha)
        if abs(u) < 1e-10:
            return eps_static  # Limit: L(u) → u/3
        return eps_optical + (4.0 * np.pi * (2.0 + eps_optical)) / (3.0 * E_au) * n_solvent * dipole_au * _langevin(u)

    def _equation(E_au):
        return E_au * (d_IHL_bohr * _eps_OHL(E_au) / eps_IHL + d_OHL_bohr) - E_vs_PZFC * EV_TO_HARTREE + phi_OHP_Ha

    # Solve for electric field in OHL
    E_solutions = fsolve(_equation, x0=0.1, full_output=False)
    E_OHL_au = E_solutions if np.isscalar(E_solutions) else E_solutions[0]
    E_OHL_V_per_A = E_OHL_au * HARTREE_TO_EV * A2Bohr
    E_IHL_V_per_A = E_OHL_au * _eps_OHL(E_OHL_au) / eps_IHL * HARTREE_TO_EV * A2Bohr

    # Debye screening length
    kappa_inv_A = 1.0 / (np.sqrt((8.0 * np.pi * n_ions) / (eps_static * kBT_Ha)) / A2Bohr)

    def phi(R):
        """Potential drop φ(R) in V as function of distance from electrode in Å."""
        R_arr = np.asarray(R, dtype=float)
        result = np.zeros_like(R_arr)

        mask_IHL = R_arr <= d_IHL
        mask_OHL = (R_arr > d_IHL) & (R_arr <= d_IHL + d_OHL)
        mask_diff = R_arr > d_IHL + d_OHL

        result[mask_IHL] = E_vs_PZFC - R_arr[mask_IHL] * E_IHL_V_per_A
        result[mask_OHL] = E_vs_PZFC - d_IHL * E_IHL_V_per_A - (R_arr[mask_OHL] - d_IHL) * E_OHL_V_per_A

        R_diff = R_arr[mask_diff] - d_IHL - d_OHL
        tanh_arg = phi_OHP_V / (4.0 * kBT_eV)
        tanh_arg = np.clip(tanh_arg, -10, 10)
        result[mask_diff] = 4.0 * kBT_eV * np.arctanh(
            np.tanh(tanh_arg) * np.exp(-R_diff / kappa_inv_A)
        )

        # Handle scalar input
        if np.isscalar(R):
            return float(result)
        return result

    return phi


# =====================================================================
# Electrochemical rate integration
# =====================================================================

def electrochemical_rate(
    rate_func: callable,
    delta_G_base: float,
    epsilons: np.ndarray | None = None,
    rho_DOS: np.ndarray | None = None,
    overpotential: float = 0.0,
    temperature: float = 298.15,
    direction: str = "anodic",
    beta_decay: float = 1.0,
    epsilon_range: tuple[float, float] = (-2.0, 2.0),
    n_epsilon: int = 101,
) -> tuple[float, np.ndarray]:
    """Integrate a PCET rate over electrode electronic states.

    For anodic (oxidation):
        k_anodic = ∫ dε ρ(ε) (1/β') [1 - f(ε)] × k_PCET(ΔG = ΔG⁰ + ε - eη)

    For cathodic (reduction):
        k_cathodic = ∫ dε ρ(ε) (1/β') f(ε) × k_PCET(ΔG = ΔG⁰ - ε + eη)

    Args:
        rate_func: Callable rate_func(delta_G_eV) -> rate in s⁻¹.
            Should compute the vibronic PCET rate for a given driving force.
        delta_G_base: Base reaction free energy in eV (at η=0, ε=0).
        epsilons: Electrode energy levels in eV (relative to Fermi level).
            If None, a uniform grid from epsilon_range is used.
        rho_DOS: Density of states at each epsilon (states/eV). If None,
            constant DOS = 1 is assumed.
        overpotential: Applied overpotential η in V.
        temperature: Temperature in K.
        direction: 'anodic' or 'cathodic'.
        beta_decay: Distance decay parameter β' for electronic coupling (Å⁻¹).
        epsilon_range: (min, max) for epsilon grid if epsilons not provided.
        n_epsilon: Number of epsilon points if epsilons not provided.

    Returns:
        Tuple of (total_rate, rate_per_epsilon).
    """
    if epsilons is None:
        epsilons = np.linspace(epsilon_range[0], epsilon_range[1], n_epsilon)

    if rho_DOS is None:
        rho_DOS = np.ones_like(epsilons)

    f = fermi_dirac(epsilons, temperature=temperature)

    k_per_eps = np.zeros_like(epsilons)
    for i, eps in enumerate(epsilons):
        if direction == "anodic":
            dG = delta_G_base + eps - overpotential
            weight = (1.0 - f[i])
        elif direction == "cathodic":
            dG = delta_G_base - eps + overpotential
            weight = f[i]
        else:
            raise ValueError(f"direction must be 'anodic' or 'cathodic', got '{direction}'")

        k_per_eps[i] = rho_DOS[i] / beta_decay * weight * rate_func(dG)

    total = simpson(k_per_eps, x=epsilons)
    return total, k_per_eps


# =====================================================================
# Tafel analysis
# =====================================================================

def tafel_analysis(
    potentials: np.ndarray,
    ln_rates: np.ndarray,
    temperature: float = 298.15,
) -> float:
    """Extract transfer coefficient α from Tafel plot.

    Fits ln(k) = -αF/(RT) × E + const.

    Args:
        potentials: Applied potentials in V.
        ln_rates: Natural log of rate constants.
        temperature: Temperature in K.

    Returns:
        Transfer coefficient α (dimensionless).
    """
    R_gas = 8.314462618  # J/(mol·K)
    F = 96485.33212  # C/mol
    RTF = R_gas * temperature / F  # V

    # Linear fit: ln(k) = slope × E + intercept
    coeffs = np.polyfit(potentials, ln_rates, 1)
    slope = coeffs[0]

    # slope = -α × F / (RT) = -α / RTF
    alpha = -slope * RTF
    return alpha
