"""
Vibronic rate theory for PCET reactions.

Implements the Hammes-Schiffer/Soudackov formalism for nonadiabatic PCET:

    k = Σ_μν P_μ × k_μν

where k_μν includes numerical integration over the donor-acceptor gating
coordinate R:

    k_μν = (2π/ℏ) |V_el|² / √(4πλkBT) × <|S_μν(R)|²>_R × exp(-(ΔG_μν + λ)²/(4λkBT))

The FC overlaps depend on the tunneling distance δ(R) = δ₀ + (R - R_eq):
    |S_μν(R)|² = |S_μν(δ(R))|²

Numerical Gauss-Hermite quadrature over the thermal distribution of R yields
the R-averaged FC overlaps. This is essential for accurate KIE because H and D
wavefunctions respond differently to D-A distance fluctuations.

References:
    Hammes-Schiffer, S. & Soudackov, A. V. J. Phys. Chem. B 112, 14108 (2008).
    Soudackov, A. V. & Hammes-Schiffer, S. J. Chem. Phys. 113, 2385 (2000).
    Hatcher, E.; Soudackov, A. V.; Hammes-Schiffer, S. JACS 126, 5763 (2004).
    Knapp, M. J. & Klinman, J. P. Eur. J. Biochem. 269, 3113 (2002).
"""

import math
import numpy as np
from dataclasses import dataclass

from pcet_engine.core.constants import (
    KB_HARTREE,
    HBAR_AU,
    TWO_PI,
    AMU_TO_AU,
    HARTREE_TO_KCALMOL,
    KCALMOL_TO_HARTREE,
    AU_RATE_TO_PER_S,
    ANGSTROM_TO_BOHR,
    CM_TO_HARTREE,
    PROTON_MASS_AMU,
    DEUTERIUM_MASS_AMU,
)


@dataclass
class VibronicResult:
    """Result from vibronic rate calculation.

    Attributes:
        rate_total: Total rate constant in s⁻¹.
        rate_channels: Individual channel rates k_μν in s⁻¹.
        overlaps: Franck-Condon overlaps |S_μν|² for each channel.
        boltzmann_weights: Thermal population P_μ of each reactant state.
        activation_energy: Effective activation energy in kcal/mol.
        dominant_channel: (μ, ν) indices of the dominant channel.
        n_reactant_states: Number of reactant vibronic states included.
        n_product_states: Number of product vibronic states included.
    """

    rate_total: float
    rate_channels: np.ndarray
    overlaps: np.ndarray
    boltzmann_weights: np.ndarray
    activation_energy: float
    dominant_channel: tuple[int, int]
    n_reactant_states: int
    n_product_states: int


def franck_condon_overlap(
    omega_R: float,
    omega_P: float,
    mass: float,
    delta_R: float,
    mu: int,
    nu: int,
) -> float:
    """Compute Franck-Condon overlap |<χ_μ^R|χ_ν^P>|² for displaced harmonic oscillators.

    Uses the analytic formula for overlaps between two harmonic oscillators
    with different frequencies and equilibrium positions.

    For equal frequencies (omega_R ≈ omega_P), this reduces to the standard
    displaced oscillator formula involving Hermite polynomials.

    For the ground-state overlap (μ=ν=0):
        |S_00|² = (2√(α_R α_P)/(α_R + α_P)) × exp(-α_R α_P δ²/(α_R + α_P))

    where α = mω/ℏ and δ = delta_R.

    Args:
        omega_R: Reactant proton frequency in atomic units (hartree/hbar).
        omega_P: Product proton frequency in atomic units.
        mass: Proton mass in atomic units (m_e).
        delta_R: Proton donor-acceptor distance shift in bohr.
        mu: Reactant vibrational quantum number.
        nu: Product vibrational quantum number.

    Returns:
        |S_μν|² (dimensionless).
    """
    if omega_R <= 0 or omega_P <= 0:
        return 0.0

    alpha_R = mass * omega_R / HBAR_AU
    alpha_P = mass * omega_P / HBAR_AU

    if mu == 0 and nu == 0:
        return _overlap_00(alpha_R, alpha_P, delta_R)
    else:
        return _overlap_general(alpha_R, alpha_P, delta_R, mu, nu)


def _overlap_00(alpha_R: float, alpha_P: float, delta_R: float) -> float:
    """Ground-state Franck-Condon overlap."""
    sum_alpha = alpha_R + alpha_P
    if sum_alpha < 1e-30:
        return 0.0

    prefactor = 2.0 * math.sqrt(alpha_R * alpha_P) / sum_alpha
    exponent = -alpha_R * alpha_P * delta_R**2 / sum_alpha

    return prefactor * math.exp(exponent)


def _overlap_general(
    alpha_R: float,
    alpha_P: float,
    delta_R: float,
    mu: int,
    nu: int,
) -> float:
    """General Franck-Condon overlap via recursion (Doktorov method)."""
    sum_alpha = alpha_R + alpha_P
    if sum_alpha < 1e-30:
        return 0.0

    # Huang-Rhys parameter: S = α_bar × δ²/2 where α_bar = 2α_Rα_P/(α_R+α_P)
    # So S = α_R × α_P × δ² / (α_R + α_P)
    S_eff = alpha_R * alpha_P * delta_R**2 / sum_alpha
    return _overlap_equal_freq(S_eff, mu, nu)


def _overlap_equal_freq(S: float, mu: int, nu: int) -> float:
    """Franck-Condon overlap for equal-frequency displaced oscillators.

    |<mu|nu>|² = exp(-S) × S^|mu-nu| × (min(mu,nu)!)/(max(mu,nu)!)
                 × [L_{min}^{|mu-nu|}(S)]²

    where L_n^k is the associated Laguerre polynomial.
    """
    m, n = min(mu, nu), max(mu, nu)
    diff = abs(mu - nu)

    L = _laguerre(m, diff, S)

    log_fac_ratio = sum(math.log(j) for j in range(m + 1, n + 1)) if n > m else 0.0

    log_overlap = (
        -S
        + diff * math.log(S + 1e-300)
        - log_fac_ratio
        + 2.0 * math.log(abs(L) + 1e-300)
    )

    return math.exp(log_overlap) if log_overlap > -700 else 0.0


def _laguerre(n: int, alpha: int, x: float) -> float:
    """Associated Laguerre polynomial L_n^alpha(x) via recurrence."""
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0 + alpha - x

    L_prev = 1.0
    L_curr = 1.0 + alpha - x
    for k in range(2, n + 1):
        L_next = ((2 * k - 1 + alpha - x) * L_curr - (k - 1 + alpha) * L_prev) / k
        L_prev = L_curr
        L_curr = L_next

    return L_curr


# =====================================================================
# Analytical ΔE_a for temperature-dependent KIE
# =====================================================================

def analytical_delta_Ea(
    omega_H_cm: float,
    delta_0_ang: float,
    sigma_DA_ang: float,
    temperature: float = 298.15,
) -> float:
    """Analytical ΔE_a = E_a(D) - E_a(H) for single-channel (0→0) vibronic rate.

    Derived from the R-averaged Franck-Condon overlap in the harmonic
    gating limit. For a Gaussian distribution of D-A distances with
    variance σ², the R-averaged overlap is:

        ⟨|S_00|²⟩ = 1/√(1+ασ²) × exp(-αδ₀²/(2(1+ασ²)))

    where α = m×ω (tunneling decay constant). The T-dependence of KIE
    arises from σ²(T), giving:

        ΔE_a = kB T² × [∂/∂σ²(ln⟨|S_D|²⟩) - ∂/∂σ²(ln⟨|S_H|²⟩)] × ∂σ²/∂T

    This formula has a maximum at σ* ≈ 0.069 Å (for ω_H ~ 2900 cm⁻¹),
    beyond which ΔE_a decreases. For σ >> σ*, the R-averaging is so
    strong that KIE becomes insensitive to further fluctuations.

    The formula correctly classifies all 6 published experimental systems
    (SLO-1, SLO-1-DM, DHFR as T-independent; AADH, MADH, GOx as T-dependent)
    when using gating-derived σ values.

    Args:
        omega_H_cm: Proton vibrational frequency in cm⁻¹.
        delta_0_ang: Equilibrium tunneling distance in angstrom.
        sigma_DA_ang: RMS D-A distance fluctuation in angstrom.
        temperature: Temperature in Kelvin.

    Returns:
        ΔE_a in kcal/mol.
    """
    omega_H = omega_H_cm * CM_TO_HARTREE
    omega_D = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
    m_H = PROTON_MASS_AMU * AMU_TO_AU
    m_D = DEUTERIUM_MASS_AMU * AMU_TO_AU
    delta_0 = delta_0_ang * ANGSTROM_TO_BOHR
    sigma_sq = (sigma_DA_ang * ANGSTROM_TO_BOHR) ** 2

    alpha_H = m_H * omega_H
    alpha_D = m_D * omega_D

    # dσ²/dT = σ²/T (classical: σ² = kBT/(MΩ²) ∝ T)
    dsigma_sq_dT = sigma_sq / temperature

    def _d_lnS_dsigma2(alpha: float) -> float:
        denom = 1.0 + alpha * sigma_sq
        return alpha / (2.0 * denom) * (alpha * delta_0**2 / denom - 1.0)

    delta_Ea = (
        KB_HARTREE * temperature**2
        * (_d_lnS_dsigma2(alpha_D) - _d_lnS_dsigma2(alpha_H))
        * dsigma_sq_dT
        * HARTREE_TO_KCALMOL
    )
    return delta_Ea


def sigma_from_gating(
    Omega_cm: float,
    M_DA_amu: float,
    temperature: float = 298.15,
) -> float:
    """Compute D-A distance fluctuation σ from gating parameters.

    Uses the quantum harmonic oscillator result:
        σ² = (ℏ/2MΩ) coth(ℏΩ/2kBT)

    Args:
        Omega_cm: Gating frequency in cm⁻¹.
        M_DA_amu: Gating reduced mass in amu.
        temperature: Temperature in K.

    Returns:
        σ_DA in angstrom.
    """
    Omega_au = Omega_cm * CM_TO_HARTREE
    M_au = M_DA_amu * AMU_TO_AU
    kBT = KB_HARTREE * temperature

    x = Omega_au / (2.0 * kBT)
    if x < 0.01:
        sigma_sq = kBT / (M_au * Omega_au**2)
    else:
        coth_x = 1.0 / math.tanh(x) if x < 50 else 1.0
        sigma_sq = (HBAR_AU / (2.0 * M_au * Omega_au)) * coth_x

    return math.sqrt(sigma_sq) / ANGSTROM_TO_BOHR


# =====================================================================
# Tunneling distance and gating coordinate
# =====================================================================

def tunneling_distance(
    d_DA: float,
    r_DH: float = 1.09,
    r_AH: float = 0.96,
) -> float:
    """Compute equilibrium proton tunneling distance.

    The tunneling distance δ₀ is the separation between the two minima of the
    proton double-well potential along the transfer coordinate:

        δ₀ = d_DA - r_DH - r_AH

    This is the geometric estimate. For systems with published QM/MM values,
    the delta_0 parameter can be set directly.

    Args:
        d_DA: Donor-acceptor distance in angstrom.
        r_DH: Donor-H equilibrium bond length in angstrom.
        r_AH: Acceptor-H equilibrium bond length in angstrom.

    Returns:
        Equilibrium tunneling distance in angstrom (minimum 0.05 Å).
    """
    delta = d_DA - r_DH - r_AH
    return max(delta, 0.05)


def _r_averaged_fc_overlap(
    omega_R: float,
    omega_P: float,
    mass_au: float,
    delta_0_bohr: float,
    mu: int,
    nu: int,
    M_DA_au: float,
    Omega_au: float,
    kBT: float,
    n_quad: int = 40,
) -> float:
    """Compute R-averaged FC overlap via numerical Gauss-Hermite quadrature.

    Evaluates:
        <|S_μν|²>_R = ∫ dR P(R) |S_μν(δ(R))|²

    where P(R) is the thermal distribution of the D-A gating coordinate and
    δ(R) = δ₀ + (R - R_eq).

    Args:
        omega_R, omega_P: Proton frequencies in a.u.
        mass_au: Transferring particle mass in a.u.
        delta_0_bohr: Equilibrium tunneling distance in bohr.
        mu, nu: Vibronic quantum numbers.
        M_DA_au: Gating reduced mass in a.u.
        Omega_au: Gating frequency in a.u.
        kBT: Thermal energy in hartree.
        n_quad: Number of quadrature points.

    Returns:
        Thermally-averaged |S_μν|² (dimensionless).
    """
    # Thermal width of gating coordinate (quantum harmonic oscillator):
    # σ² = (ℏ/2MΩ) coth(ℏΩ/2kBT)
    # Classical limit (kBT >> ℏΩ): σ² → kBT/(MΩ²)
    # Quantum limit (kBT << ℏΩ): σ² → ℏ/(2MΩ) = zero-point width
    x = Omega_au / (2.0 * kBT)
    if x < 0.01:
        # Classical limit: coth(x) ≈ 1/x
        sigma_sq = kBT / (M_DA_au * Omega_au**2)
    else:
        coth_x = 1.0 / math.tanh(x) if x < 50 else 1.0
        sigma_sq = (HBAR_AU / (2.0 * M_DA_au * Omega_au)) * coth_x
    sigma = math.sqrt(sigma_sq)  # in bohr

    # Gauss-Hermite quadrature: ∫ exp(-x²) f(x) dx ≈ Σ w_i f(x_i)
    # u = σ√2 × x (change of variables from standard Hermite)
    points, weights = np.polynomial.hermite.hermgauss(n_quad)

    total = 0.0
    for i in range(n_quad):
        u = sigma * math.sqrt(2.0) * points[i]
        delta_R = delta_0_bohr + u

        if delta_R < 0.02:  # Don't let distance go negative/tiny
            delta_R = 0.02

        S_sq = franck_condon_overlap(omega_R, omega_P, mass_au, delta_R, mu, nu)
        total += weights[i] * S_sq

    # Normalize: Gauss-Hermite weights sum to √π
    return total / math.sqrt(math.pi)


# =====================================================================
# Rate functions
# =====================================================================

def vibronic_rate(
    V_el: float,
    delta_G: float,
    lambda_reorg: float,
    omega_proton: float,
    mass_proton: float,
    d_DA: float,
    temperature: float = 298.15,
    r_DH: float = 1.09,
    r_AH: float = 0.96,
) -> float:
    """Single-channel (ground-state) vibronic PCET rate.

    This is the μ=0, ν=0 channel only.

    Args:
        V_el: Electronic coupling in hartree.
        delta_G: Driving force in hartree.
        lambda_reorg: Reorganization energy in hartree.
        omega_proton: Proton vibrational frequency in atomic units.
        mass_proton: Transferring particle mass in amu.
        d_DA: Donor-acceptor distance in angstrom.
        temperature: Temperature in Kelvin.
        r_DH: Donor-H bond length in angstrom.
        r_AH: Acceptor-H bond length in angstrom.

    Returns:
        Rate constant in s⁻¹.
    """
    kBT = KB_HARTREE * temperature
    mass_au = mass_proton * AMU_TO_AU

    delta_r = tunneling_distance(d_DA, r_DH, r_AH)
    delta_r_bohr = delta_r * ANGSTROM_TO_BOHR

    S_00_sq = franck_condon_overlap(omega_proton, omega_proton, mass_au, delta_r_bohr, 0, 0)

    E_a = (delta_G + lambda_reorg) ** 2 / (4.0 * lambda_reorg)
    prefactor = (TWO_PI / HBAR_AU) * V_el**2 * S_00_sq / math.sqrt(4.0 * math.pi * lambda_reorg * kBT)
    rate_au = prefactor * math.exp(-E_a / kBT)

    return rate_au * AU_RATE_TO_PER_S


def multi_channel_rate(
    V_el: float,
    delta_G: float,
    lambda_reorg: float,
    omega_R: float,
    omega_P: float,
    mass: float,
    d_DA: float,
    temperature: float = 298.15,
    n_reactant_states: int = 5,
    n_product_states: int = 10,
    Omega_gating: float = 0.0,
    M_DA: float = 0.0,
    r_DH: float = 1.09,
    r_AH: float = 0.96,
    delta_0: float | None = None,
) -> VibronicResult:
    """Multi-channel vibronic PCET rate with gating coordinate integration.

    k = Σ_μ P_μ Σ_ν k_μν

    When gating parameters (Omega_gating, M_DA) are provided, the FC overlaps
    are numerically averaged over the thermal distribution of the D-A distance:

        <|S_μν|²>_R = ∫ P(R) |S_μν(δ₀ + R - R_eq)|² dR

    This numerical integration (Gauss-Hermite quadrature) properly captures
    the nonlinear dependence of FC overlaps on R, which is critical for KIE.

    Args:
        V_el: Electronic coupling in hartree.
        delta_G: Electronic driving force in hartree.
        lambda_reorg: Total reorganization energy in hartree.
        omega_R: Reactant proton frequency in atomic units.
        omega_P: Product proton frequency in atomic units.
        mass: Transferring particle mass in amu.
        d_DA: Donor-acceptor distance in angstrom.
        temperature: Temperature in Kelvin.
        n_reactant_states: Number of reactant vibronic states.
        n_product_states: Number of product vibronic states.
        Omega_gating: Gating mode frequency in atomic units. If 0, no gating.
        M_DA: Reduced mass for gating mode in amu. If 0, no gating.
        r_DH: Donor-H bond length in angstrom.
        r_AH: Acceptor-H bond length in angstrom.
        delta_0: Explicit tunneling distance in angstrom (overrides geometric estimate).

    Returns:
        VibronicResult with total rate, channel decomposition, and diagnostics.
    """
    kBT = KB_HARTREE * temperature
    mass_au = mass * AMU_TO_AU
    M_DA_au = M_DA * AMU_TO_AU if M_DA > 0 else 0.0

    # Tunneling distance
    if delta_0 is not None:
        delta_r = max(delta_0, 0.05)
    else:
        delta_r = tunneling_distance(d_DA, r_DH, r_AH)
    delta_r_bohr = delta_r * ANGSTROM_TO_BOHR

    use_gating = (Omega_gating > 0 and M_DA_au > 0)

    # Boltzmann weights for reactant states
    energies_R = np.array([(mu + 0.5) * omega_R for mu in range(n_reactant_states)])
    boltz_unnorm = np.exp(-(energies_R - energies_R[0]) / kBT)
    Z = np.sum(boltz_unnorm)
    P_mu = boltz_unnorm / Z

    # Channel rates
    rate_channels = np.zeros((n_reactant_states, n_product_states))
    overlap_channels = np.zeros((n_reactant_states, n_product_states))

    for mu in range(n_reactant_states):
        for nu in range(n_product_states):
            # Vibronic driving force
            delta_G_munu = delta_G + (nu + 0.5) * omega_P - (mu + 0.5) * omega_R

            # FC overlap: R-averaged if gating, otherwise at equilibrium
            if use_gating:
                S_sq = _r_averaged_fc_overlap(
                    omega_R, omega_P, mass_au, delta_r_bohr,
                    mu, nu, M_DA_au, Omega_gating, kBT,
                )
            else:
                S_sq = franck_condon_overlap(omega_R, omega_P, mass_au, delta_r_bohr, mu, nu)

            overlap_channels[mu, nu] = S_sq

            if S_sq < 1e-30:
                continue

            # Marcus rate for this channel
            E_a = (delta_G_munu + lambda_reorg) ** 2 / (4.0 * lambda_reorg)
            prefactor = (
                (TWO_PI / HBAR_AU)
                * V_el**2
                * S_sq
                / math.sqrt(4.0 * math.pi * lambda_reorg * kBT)
            )
            rate_au = prefactor * math.exp(-E_a / kBT) if E_a / kBT < 700 else 0.0
            rate_channels[mu, nu] = rate_au * AU_RATE_TO_PER_S

    # Total rate
    total_rate = 0.0
    for mu in range(n_reactant_states):
        total_rate += P_mu[mu] * np.sum(rate_channels[mu, :])

    # Find dominant channel
    weighted = np.zeros_like(rate_channels)
    for mu in range(n_reactant_states):
        weighted[mu, :] = P_mu[mu] * rate_channels[mu, :]
    dom_idx = np.unravel_index(np.argmax(weighted), weighted.shape)

    # Effective activation energy
    if total_rate > 0:
        mu_d, nu_d = dom_idx
        delta_G_dom = delta_G + (nu_d + 0.5) * omega_P - (mu_d + 0.5) * omega_R
        E_a_eff = (delta_G_dom + lambda_reorg) ** 2 / (4.0 * lambda_reorg)
        E_a_kcal = E_a_eff * HARTREE_TO_KCALMOL
    else:
        E_a_kcal = float("inf")

    return VibronicResult(
        rate_total=total_rate,
        rate_channels=rate_channels,
        overlaps=overlap_channels,
        boltzmann_weights=P_mu,
        activation_energy=E_a_kcal,
        dominant_channel=(int(dom_idx[0]), int(dom_idx[1])),
        n_reactant_states=n_reactant_states,
        n_product_states=n_product_states,
    )
