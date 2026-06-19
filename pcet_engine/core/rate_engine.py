"""
High-level PCET rate prediction engine.

Orchestrates the full pipeline:
    Hessian data → normal modes → vibronic rates → KIE → benchmarks

This is the main entry point for users who want to compute PCET rates
from quantum chemistry data.
"""

import math
import numpy as np
from dataclasses import dataclass, field

from pcet_engine.core.constants import (
    PROTON_MASS_AMU,
    DEUTERIUM_MASS_AMU,
    HARTREE_TO_KCALMOL,
    KCALMOL_TO_HARTREE,
    CM_TO_HARTREE,
)
from pcet_engine.core.marcus import marcus_rate, marcus_activation_energy
from pcet_engine.core.vibronic import (
    vibronic_rate,
    multi_channel_rate,
    VibronicResult,
)
from pcet_engine.core.normal_modes import (
    normal_mode_analysis,
    identify_proton_mode,
    identify_da_stretching_mode,
    compute_donor_acceptor_distance,
    NormalModeResult,
    GatingResult,
)
from pcet_engine.core.participation import (
    proton_participation,
    geometric_tunneling_prefactor,
    ParticipationResult,
)


def _fgh_multi_channel_rate(
    V_el_au: float,
    delta_G_au: float,
    lambda_au: float,
    omega_R_cm: float,
    omega_P_cm: float,
    mass_amu: float,
    delta_0_angstrom: float,
    temperature: float,
    n_reactant_states: int,
    n_product_states: int,
    potential_type: str = "morse",
) -> "VibronicResult":
    """Multi-channel vibronic rate using FGH numerical wavefunctions.

    Drop-in replacement for multi_channel_rate() but with numerical FC overlaps
    computed from Morse (or harmonic) proton potentials solved via FGH.
    """
    from pcet_engine.core.fgh import fgh_franck_condon_table
    from pcet_engine.core.constants import (
        KB_HARTREE, HBAR_AU, TWO_PI, AU_RATE_TO_PER_S, HARTREE_TO_KCALMOL,
    )

    kBT = KB_HARTREE * temperature

    # Compute FGH overlaps and energy levels
    overlaps, energies_R, energies_P, _, _ = fgh_franck_condon_table(
        omega_R_cm, omega_P_cm, mass_amu, delta_0_angstrom,
        delta_G_hartree=0.0,  # ΔG handled in Marcus factor, not in potential offset
        n_reactant_states=n_reactant_states,
        n_product_states=n_product_states,
        potential_type=potential_type,
    )

    # Boltzmann weights from FGH energy levels (more accurate than harmonic)
    E_R = energies_R - energies_R[0]
    boltz_unnorm = np.exp(-E_R / kBT)
    P_mu = boltz_unnorm / np.sum(boltz_unnorm)

    # Channel rates
    rate_channels = np.zeros((n_reactant_states, n_product_states))

    for mu in range(n_reactant_states):
        for nu in range(n_product_states):
            # Vibronic driving force using FGH energy levels
            dG_munu = delta_G_au + (energies_P[nu] - energies_P[0]) - (energies_R[mu] - energies_R[0])

            S_sq = overlaps[mu, nu]
            if S_sq < 1e-30:
                continue

            E_a = (dG_munu + lambda_au) ** 2 / (4.0 * lambda_au)
            prefactor = (
                (TWO_PI / HBAR_AU)
                * V_el_au**2
                * S_sq
                / math.sqrt(4.0 * math.pi * lambda_au * kBT)
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
        dG_dom = delta_G_au + (energies_P[nu_d] - energies_P[0]) - (energies_R[mu_d] - energies_R[0])
        E_a_eff = (dG_dom + lambda_au) ** 2 / (4.0 * lambda_au)
        E_a_kcal = E_a_eff * HARTREE_TO_KCALMOL
    else:
        E_a_kcal = float("inf")

    return VibronicResult(
        rate_total=total_rate,
        rate_channels=rate_channels,
        overlaps=overlaps,
        boltzmann_weights=P_mu,
        activation_energy=E_a_kcal,
        dominant_channel=(int(dom_idx[0]), int(dom_idx[1])),
        n_reactant_states=n_reactant_states,
        n_product_states=n_product_states,
    )


def _fgh_gating_multi_channel_rate(
    V_el_au: float,
    delta_G_au: float,
    lambda_au: float,
    omega_R_cm: float,
    omega_P_cm: float,
    mass_amu: float,
    delta_0_angstrom: float,
    temperature: float,
    n_reactant_states: int,
    n_product_states: int,
    Omega_gating_cm: float,
    M_DA_amu: float,
    potential_type: str = "morse",
) -> "VibronicResult":
    """Multi-channel vibronic rate with FGH wavefunctions AND R-averaging.

    This combines Morse (or harmonic) FGH overlaps with thermal averaging
    over the D-A gating coordinate. At each Gauss-Hermite quadrature point,
    the product well is shifted and the FGH problem is re-solved, giving
    state-pair-specific distance dependence.

    This is the Soudackov-corrected calculation: α_μν varies by state pair
    because the Morse wavefunction tails differ between ground and excited
    states, leading to different overlap distance dependences.
    """
    from pcet_engine.core.fgh import r_averaged_fgh_fc_table
    from pcet_engine.core.constants import (
        KB_HARTREE, HBAR_AU, TWO_PI, AU_RATE_TO_PER_S, HARTREE_TO_KCALMOL,
    )

    kBT = KB_HARTREE * temperature

    # Compute R-averaged FGH overlaps
    overlaps, energies_R, energies_P = r_averaged_fgh_fc_table(
        omega_R_cm, omega_P_cm, mass_amu, delta_0_angstrom,
        M_DA_amu, Omega_gating_cm, temperature,
        delta_G_hartree=0.0,
        n_reactant_states=n_reactant_states,
        n_product_states=n_product_states,
        potential_type=potential_type,
    )

    # Boltzmann weights from FGH energy levels
    E_R = energies_R - energies_R[0]
    boltz_unnorm = np.exp(-E_R / kBT)
    P_mu = boltz_unnorm / np.sum(boltz_unnorm)

    # Channel rates
    rate_channels = np.zeros((n_reactant_states, n_product_states))

    for mu in range(n_reactant_states):
        for nu in range(n_product_states):
            dG_munu = delta_G_au + (energies_P[nu] - energies_P[0]) - (energies_R[mu] - energies_R[0])

            S_sq = overlaps[mu, nu]
            if S_sq < 1e-30:
                continue

            E_a = (dG_munu + lambda_au) ** 2 / (4.0 * lambda_au)
            prefactor = (
                (TWO_PI / HBAR_AU)
                * V_el_au**2
                * S_sq
                / math.sqrt(4.0 * math.pi * lambda_au * kBT)
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

    if total_rate > 0:
        mu_d, nu_d = dom_idx
        dG_dom = delta_G_au + (energies_P[nu_d] - energies_P[0]) - (energies_R[mu_d] - energies_R[0])
        E_a_eff = (dG_dom + lambda_au) ** 2 / (4.0 * lambda_au)
        E_a_kcal = E_a_eff * HARTREE_TO_KCALMOL
    else:
        E_a_kcal = float("inf")

    return VibronicResult(
        rate_total=total_rate,
        rate_channels=rate_channels,
        overlaps=overlaps,
        boltzmann_weights=P_mu,
        activation_energy=E_a_kcal,
        dominant_channel=(int(dom_idx[0]), int(dom_idx[1])),
        n_reactant_states=n_reactant_states,
        n_product_states=n_product_states,
    )


def _soudackov_corrected_rate(
    V_el_au: float,
    delta_G_au: float,
    lambda_au: float,
    omega_R_cm: float,
    omega_P_cm: float,
    mass_amu: float,
    delta_0_angstrom: float,
    temperature: float,
    n_reactant_states: int,
    n_product_states: int,
    Omega_gating_cm: float,
    M_DA_amu: float,
    potential_type: str = "morse",
) -> "VibronicResult":
    """Multi-channel rate using the Soudackov correction.

    Uses the SHS analytical gating formula (PMC5217758 Eq. 3) with
    state-pair-specific α_μν and γ_μν derived from Morse FGH overlaps.

    This is the physically correct approach:
    1. Compute Morse wavefunctions at R₀
    2. Numerically differentiate log|S_μν(R)|² to get α_μν, γ_μν per pair
    3. Use the analytical gating integral with these parameters
    4. Include the gating reorganization energy correction λ_α,μν

    The analytical formula accounts for the exponential R-dependence
    more accurately than numerical quadrature, especially when α_μν
    is large (deep tunneling). It also naturally captures the correction
    to the Marcus barrier from gating (λ̃ = λ + λ_α).
    """
    from pcet_engine.core.fgh import (
        compute_attenuation_params,
        analytical_gating_rate_table,
        fgh_franck_condon_table,
    )
    from pcet_engine.core.constants import (
        KB_HARTREE, HBAR_AU, TWO_PI, AU_RATE_TO_PER_S, HARTREE_TO_KCALMOL,
    )

    kBT = KB_HARTREE * temperature

    # Step 1: Compute Morse overlaps and state-pair-specific α, γ
    S_sq_0, alpha_mn, gamma_mn = compute_attenuation_params(
        omega_R_cm, omega_P_cm, mass_amu, delta_0_angstrom,
        n_reactant_states, n_product_states, potential_type,
    )

    # Step 2: Apply analytical gating formula (overlap boost, no λ correction)
    S_sq_eff, _ = analytical_gating_rate_table(
        S_sq_0, alpha_mn, gamma_mn,
        M_DA_amu, Omega_gating_cm, temperature,
    )

    # Step 3: Get FGH energy levels for Boltzmann weights and ΔG_μν
    _, energies_R, energies_P, _, _ = fgh_franck_condon_table(
        omega_R_cm, omega_P_cm, mass_amu, delta_0_angstrom,
        delta_G_hartree=0.0,
        n_reactant_states=n_reactant_states,
        n_product_states=n_product_states,
        potential_type=potential_type,
    )

    # Boltzmann weights
    E_R = energies_R - energies_R[0]
    boltz_unnorm = np.exp(-E_R / kBT)
    P_mu = boltz_unnorm / np.sum(boltz_unnorm)

    # Channel rates — Marcus barrier uses original λ (no gating correction)
    rate_channels = np.zeros((n_reactant_states, n_product_states))

    for mu in range(n_reactant_states):
        for nu in range(n_product_states):
            dG_munu = delta_G_au + (energies_P[nu] - energies_P[0]) - (energies_R[mu] - energies_R[0])

            S_sq = S_sq_eff[mu, nu]
            if S_sq < 1e-30:
                continue

            E_a = (dG_munu + lambda_au) ** 2 / (4.0 * lambda_au)
            prefactor = (
                (TWO_PI / HBAR_AU)
                * V_el_au**2
                * S_sq
                / math.sqrt(4.0 * math.pi * lambda_au * kBT)
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

    if total_rate > 0:
        mu_d, nu_d = dom_idx
        dG_dom = delta_G_au + (energies_P[nu_d] - energies_P[0]) - (energies_R[mu_d] - energies_R[0])
        E_a_eff = (dG_dom + lambda_au) ** 2 / (4.0 * lambda_au)
        E_a_kcal = E_a_eff * HARTREE_TO_KCALMOL
    else:
        E_a_kcal = float("inf")

    return VibronicResult(
        rate_total=total_rate,
        rate_channels=rate_channels,
        overlaps=S_sq_eff,
        boltzmann_weights=P_mu,
        activation_energy=E_a_kcal,
        dominant_channel=(int(dom_idx[0]), int(dom_idx[1])),
        n_reactant_states=n_reactant_states,
        n_product_states=n_product_states,
    )


@dataclass
class PCETResult:
    """Complete PCET rate prediction result.

    Attributes:
        k_H: Rate constant for H transfer in s⁻¹.
        k_D: Rate constant for D transfer in s⁻¹.
        KIE: Kinetic isotope effect k_H/k_D.
        E_a: Activation energy in kcal/mol.
        delta_G: Driving force in kcal/mol.
        lambda_reorg: Reorganization energy in kcal/mol.
        V_el: Electronic coupling in kcal/mol.
        omega_H: H vibrational frequency in cm⁻¹.
        omega_D: D vibrational frequency in cm⁻¹ (scaled from H).
        d_DA: Donor-acceptor distance in angstrom.
        vibronic_H: Detailed vibronic result for H (if multi-channel used).
        vibronic_D: Detailed vibronic result for D (if multi-channel used).
        tunneling_contribution: Fraction of rate due to tunneling (vs classical).
        method: Method used ('marcus', 'vibronic_single', 'vibronic_multi').
        n_eff: Effective tunneling dimensionality (participation ratio).
        geometric_prefactor: Multiplicative correction sqrt(N_eff / d_ref).
        participation: Full ParticipationResult (if computed from Hessian).
    """

    k_H: float
    k_D: float
    KIE: float
    E_a: float
    delta_G: float
    lambda_reorg: float
    V_el: float
    omega_H: float
    omega_D: float
    d_DA: float
    vibronic_H: VibronicResult | None = None
    vibronic_D: VibronicResult | None = None
    tunneling_contribution: float = 0.0
    method: str = "vibronic_multi"
    n_eff: float = 1.0
    geometric_prefactor: float = 1.0
    participation: ParticipationResult | None = None
    gating: GatingResult | None = None


class PCETRateEngine:
    """Main PCET rate prediction engine.

    Computes nonadiabatic PCET rates using Marcus theory with vibronic
    tunneling corrections. Supports both single-channel and multi-channel
    (Hammes-Schiffer) formulations.

    Example usage::

        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.5,          # kcal/mol
            delta_G=-5.0,      # kcal/mol
            lambda_reorg=20.0, # kcal/mol
            omega_H=3000.0,    # cm⁻¹
            d_DA=2.7,          # angstrom
        )
        print(f"k_H = {result.k_H:.2e} s⁻¹")
        print(f"KIE = {result.KIE:.1f}")
    """

    def __init__(
        self,
        n_reactant_states: int = 5,
        n_product_states: int = 10,
        temperature: float = 298.15,
    ):
        """Initialize the rate engine.

        Args:
            n_reactant_states: Number of reactant vibronic states for multi-channel.
            n_product_states: Number of product vibronic states for multi-channel.
            temperature: Temperature in Kelvin.
        """
        self.n_reactant_states = n_reactant_states
        self.n_product_states = n_product_states
        self.temperature = temperature

    def compute_rate(
        self,
        V_el: float,
        delta_G: float,
        lambda_reorg: float,
        omega_H: float,
        d_DA: float,
        omega_P: float | None = None,
        method: str = "vibronic_multi",
        Omega_gating: float = 0.0,
        M_DA: float = 0.0,
        r_DH: float = 1.09,
        r_AH: float = 0.96,
        delta_0: float | None = None,
        n_eff: float | None = None,
        n_eff_mode: str = "distance",
    ) -> PCETResult:
        """Compute PCET rate for H and D transfer.

        All energies in kcal/mol, frequency in cm⁻¹, distance in angstrom.

        Args:
            V_el: Electronic coupling in kcal/mol.
            delta_G: Driving force in kcal/mol (negative = exothermic).
            lambda_reorg: Total reorganization energy in kcal/mol.
            omega_H: Proton (H) vibrational frequency in cm⁻¹.
            d_DA: Donor-acceptor distance in angstrom.
            omega_P: Product proton frequency in cm⁻¹ (default: same as omega_H).
            method: 'marcus', 'vibronic_single', 'vibronic_multi',
                'vibronic_multi_morse', or 'vibronic_multi_fgh'.
                The '_morse' variant uses Morse potentials solved via FGH
                for more accurate anharmonic FC overlaps. The '_fgh' variant
                uses harmonic potentials solved numerically (for validation).
            Omega_gating: D-A gating mode frequency in cm⁻¹ (0 = no gating).
            M_DA: Reduced mass for D-A gating mode in amu (0 = no gating).
            r_DH: Donor-H bond length in angstrom.
            r_AH: Acceptor-H bond length in angstrom.
            delta_0: Explicit tunneling distance in Å (overrides geometric estimate).
            n_eff: Effective tunneling dimensionality (participation ratio).
                If provided, applies geometric correction to tunneling distance:
                δ_eff = δ₀ × sqrt(d_ref / N_eff). This shortens the effective
                tunnel distance when multiple modes participate, enhancing tunneling.
                Since the FC overlap depends exponentially on mass × δ², the
                correction affects H and D differently, shifting the KIE.
                Computed automatically in compute_rate_from_hessian().
            n_eff_mode: How to apply N_eff correction. 'distance' (default)
                modifies the tunneling distance (affects KIE). 'prefactor'
                multiplies rates by sqrt(N_eff/3) (KIE-preserving).

        Returns:
            PCETResult with H rate, D rate, KIE, and diagnostics.
        """
        # Input validation
        if lambda_reorg < 0:
            raise ValueError(f"lambda_reorg must be non-negative, got {lambda_reorg}")
        if omega_H <= 0:
            raise ValueError(f"omega_H must be positive, got {omega_H}")
        if d_DA <= 0:
            raise ValueError(f"d_DA must be positive, got {d_DA}")
        if V_el < 0:
            raise ValueError(f"V_el must be non-negative, got {V_el}")

        if omega_P is None:
            omega_P = omega_H

        # Apply N_eff geometric correction to tunneling distance
        geo_pf = 1.0
        n_eff_val = 1.0
        if n_eff is not None and n_eff > 0:
            n_eff_val = n_eff
            geo_pf = geometric_tunneling_prefactor(n_eff)
            if n_eff_mode == "distance":
                # Shorten effective tunneling distance: δ_eff = δ₀ × sqrt(3/N_eff)
                # This enters the FC overlap as exp(-α δ²) where α ∝ mass,
                # so the correction is mass-dependent and shifts KIE.
                d_ref = 3.0
                distance_factor = math.sqrt(d_ref / n_eff) if n_eff > 0 else 1.0
                if delta_0 is not None:
                    delta_0 = delta_0 * distance_factor
                else:
                    # Modify the effective d_DA to achieve the same tunnel distance change
                    # δ = d_DA - r_DH - r_AH, so δ_new = δ_old × factor
                    # d_DA_new = r_DH + r_AH + (d_DA - r_DH - r_AH) × factor
                    delta_geom = max(d_DA - r_DH - r_AH, 0.05)
                    delta_0 = delta_geom * distance_factor

        # Convert to atomic units
        V_au = V_el * KCALMOL_TO_HARTREE
        dG_au = delta_G * KCALMOL_TO_HARTREE
        lam_au = lambda_reorg * KCALMOL_TO_HARTREE
        omega_H_au = omega_H * CM_TO_HARTREE
        omega_P_au = omega_P * CM_TO_HARTREE
        omega_D = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
        omega_D_au = omega_D * CM_TO_HARTREE
        omega_P_D_au = omega_P * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU) * CM_TO_HARTREE
        Omega_au = Omega_gating * CM_TO_HARTREE

        vibronic_H = None
        vibronic_D = None

        if method == "marcus":
            k_H = marcus_rate(V_au, dG_au, lam_au, self.temperature)
            k_D = k_H  # No isotope effect in pure Marcus
            tunneling_frac = 0.0

        elif method == "vibronic_single":
            k_H = vibronic_rate(
                V_au, dG_au, lam_au, omega_H_au, PROTON_MASS_AMU, d_DA,
                self.temperature, r_DH, r_AH,
            )
            k_D = vibronic_rate(
                V_au, dG_au, lam_au, omega_D_au, DEUTERIUM_MASS_AMU, d_DA,
                self.temperature, r_DH, r_AH,
            )
            tunneling_frac = 1.0 - (marcus_rate(V_au, dG_au, lam_au, self.temperature) / k_H if k_H > 0 else 0.0)

        elif method == "vibronic_multi":
            vibronic_H = multi_channel_rate(
                V_au, dG_au, lam_au,
                omega_H_au, omega_P_au, PROTON_MASS_AMU,
                d_DA, self.temperature,
                self.n_reactant_states, self.n_product_states,
                Omega_au, M_DA, r_DH, r_AH, delta_0,
            )
            vibronic_D = multi_channel_rate(
                V_au, dG_au, lam_au,
                omega_D_au, omega_P_D_au, DEUTERIUM_MASS_AMU,
                d_DA, self.temperature,
                self.n_reactant_states, self.n_product_states,
                Omega_au, M_DA, r_DH, r_AH, delta_0,
            )
            k_H = vibronic_H.rate_total
            k_D = vibronic_D.rate_total
            k_classical = marcus_rate(V_au, dG_au, lam_au, self.temperature)
            tunneling_frac = 1.0 - (k_classical / k_H) if k_H > 0 else 0.0

        elif method in ("vibronic_multi_morse", "vibronic_multi_fgh"):
            from pcet_engine.core.fgh import fgh_franck_condon_table
            from pcet_engine.core.vibronic import tunneling_distance

            pot_type = "morse" if method == "vibronic_multi_morse" else "harmonic"
            delta_r = delta_0 if delta_0 is not None else tunneling_distance(d_DA, r_DH, r_AH)

            vibronic_H = _fgh_multi_channel_rate(
                V_au, dG_au, lam_au, omega_H, omega_P, PROTON_MASS_AMU,
                delta_r, self.temperature,
                self.n_reactant_states, self.n_product_states, pot_type,
            )
            omega_D_cm = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
            omega_P_D_cm = omega_P * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
            vibronic_D = _fgh_multi_channel_rate(
                V_au, dG_au, lam_au, omega_D_cm, omega_P_D_cm, DEUTERIUM_MASS_AMU,
                delta_r, self.temperature,
                self.n_reactant_states, self.n_product_states, pot_type,
            )
            k_H = vibronic_H.rate_total
            k_D = vibronic_D.rate_total
            k_classical = marcus_rate(V_au, dG_au, lam_au, self.temperature)
            tunneling_frac = 1.0 - (k_classical / k_H) if k_H > 0 else 0.0

        elif method in ("vibronic_multi_morse_gating", "vibronic_multi_fgh_gating"):
            # R-averaged rate with FGH Morse (or harmonic) wavefunctions.
            # This addresses the Soudackov correction: state-pair-specific
            # overlap distance dependence from true Morse wavefunctions,
            # combined with thermal averaging over the D-A gating coordinate.
            from pcet_engine.core.vibronic import tunneling_distance

            pot_type = "morse" if method == "vibronic_multi_morse_gating" else "harmonic"
            delta_r = delta_0 if delta_0 is not None else tunneling_distance(d_DA, r_DH, r_AH)

            vibronic_H = _fgh_gating_multi_channel_rate(
                V_au, dG_au, lam_au, omega_H, omega_P, PROTON_MASS_AMU,
                delta_r, self.temperature,
                self.n_reactant_states, self.n_product_states,
                Omega_gating, M_DA, pot_type,
            )
            omega_D_cm = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
            omega_P_D_cm = omega_P * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
            vibronic_D = _fgh_gating_multi_channel_rate(
                V_au, dG_au, lam_au, omega_D_cm, omega_P_D_cm, DEUTERIUM_MASS_AMU,
                delta_r, self.temperature,
                self.n_reactant_states, self.n_product_states,
                Omega_gating, M_DA, pot_type,
            )
            k_H = vibronic_H.rate_total
            k_D = vibronic_D.rate_total
            k_classical = marcus_rate(V_au, dG_au, lam_au, self.temperature)
            tunneling_frac = 1.0 - (k_classical / k_H) if k_H > 0 else 0.0

        elif method in ("soudackov_corrected", "soudackov_corrected_harmonic"):
            # Soudackov correction: analytical gating formula with
            # state-pair-specific α_μν and γ_μν from Morse FGH.
            # This is the closest match to PMC5217758 Eq. (3).
            from pcet_engine.core.vibronic import tunneling_distance

            pot_type = "morse" if method == "soudackov_corrected" else "harmonic"
            delta_r = delta_0 if delta_0 is not None else tunneling_distance(d_DA, r_DH, r_AH)

            if Omega_gating <= 0 or M_DA <= 0:
                raise ValueError(
                    "soudackov_corrected requires Omega_gating > 0 and M_DA > 0"
                )

            vibronic_H = _soudackov_corrected_rate(
                V_au, dG_au, lam_au, omega_H, omega_P, PROTON_MASS_AMU,
                delta_r, self.temperature,
                self.n_reactant_states, self.n_product_states,
                Omega_gating, M_DA, pot_type,
            )
            omega_D_cm = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
            omega_P_D_cm = omega_P * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
            vibronic_D = _soudackov_corrected_rate(
                V_au, dG_au, lam_au, omega_D_cm, omega_P_D_cm, DEUTERIUM_MASS_AMU,
                delta_r, self.temperature,
                self.n_reactant_states, self.n_product_states,
                Omega_gating, M_DA, pot_type,
            )
            k_H = vibronic_H.rate_total
            k_D = vibronic_D.rate_total
            k_classical = marcus_rate(V_au, dG_au, lam_au, self.temperature)
            tunneling_frac = 1.0 - (k_classical / k_H) if k_H > 0 else 0.0

        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'marcus', 'vibronic_single', "
                f"'vibronic_multi', 'vibronic_multi_morse', 'vibronic_multi_fgh', "
                f"'vibronic_multi_morse_gating', 'vibronic_multi_fgh_gating', "
                f"or 'soudackov_corrected'."
            )

        # Apply prefactor-mode N_eff correction if requested
        if n_eff is not None and n_eff > 0 and n_eff_mode == "prefactor":
            k_H *= geo_pf
            k_D *= geo_pf

        KIE = k_H / k_D if k_D > 0 else float("inf")
        E_a = marcus_activation_energy(dG_au, lam_au) * HARTREE_TO_KCALMOL

        return PCETResult(
            k_H=k_H,
            k_D=k_D,
            KIE=KIE,
            E_a=E_a,
            delta_G=delta_G,
            lambda_reorg=lambda_reorg,
            V_el=V_el,
            omega_H=omega_H,
            omega_D=omega_D,
            d_DA=d_DA,
            vibronic_H=vibronic_H,
            vibronic_D=vibronic_D,
            tunneling_contribution=max(0.0, min(1.0, tunneling_frac)),
            method=method,
            n_eff=n_eff_val,
            geometric_prefactor=geo_pf,
        )

    def compute_rate_from_hessian(
        self,
        hessian_R: np.ndarray,
        hessian_P: np.ndarray,
        geom_R: np.ndarray,
        geom_P: np.ndarray,
        masses: np.ndarray,
        proton_idx: int,
        donor_idx: int,
        acceptor_idx: int,
        V_el: float,
        delta_G: float,
        lambda_outer: float = 0.0,
        method: str = "vibronic_multi",
        r_DH: float | None = None,
        r_AH: float | None = None,
        delta_0: float | None = None,
        Omega_gating: float = 0.0,
        M_DA: float = 0.0,
        auto_gating: bool = False,
        n_eff: float | None = None,
        n_eff_mode: str = "distance",
    ) -> PCETResult:
        """Compute PCET rate directly from Hessian data.

        Extracts proton frequencies and donor-acceptor distances from normal
        mode analysis, computes inner-sphere reorganization energy from the
        four-point method, and runs the vibronic rate calculation.

        Args:
            hessian_R: Reactant Hessian (3N x 3N) in hartree/bohr².
            hessian_P: Product Hessian (3N x 3N) in hartree/bohr².
            geom_R: Reactant geometry (N, 3) in angstrom.
            geom_P: Product geometry (N, 3) in angstrom.
            masses: Atomic masses in amu (N,).
            proton_idx: 0-based index of the transferring proton.
            donor_idx: 0-based index of the donor heavy atom.
            acceptor_idx: 0-based index of the acceptor heavy atom.
            V_el: Electronic coupling in kcal/mol.
            delta_G: Electronic driving force in kcal/mol.
            lambda_outer: Outer-sphere reorganization energy in kcal/mol.
            method: Rate calculation method.
            r_DH: Donor-H bond length in Å (extracted from geometry if None).
            r_AH: Acceptor-H bond length in Å (extracted from geometry if None).
            delta_0: Explicit tunneling distance in Å (overrides geometric estimate).
            Omega_gating: D-A gating mode frequency in cm⁻¹ (0 = no gating).
            M_DA: Reduced mass for D-A gating mode in amu.
            auto_gating: If True, extract Omega_gating and M_DA from the
                reactant Hessian normal modes (overrides explicit values).
            n_eff: Effective tunneling dimensionality. If None, participation
                ratio is computed and stored but not applied to the rate.
            n_eff_mode: How to apply N_eff ('distance' or 'prefactor').

        Returns:
            PCETResult.
        """
        from pcet_engine.core.marcus import reorganization_energy_from_hessians
        from pcet_engine.core.constants import ANGSTROM_TO_BOHR

        # Normal mode analysis
        nma_R = normal_mode_analysis(hessian_R, masses)
        nma_R = identify_proton_mode(nma_R, [proton_idx], masses)

        nma_P = normal_mode_analysis(hessian_P, masses)
        nma_P = identify_proton_mode(nma_P, [proton_idx], masses)

        # Participation ratio analysis (N_eff)
        part_result = proton_participation(
            nma_R.eigenvectors, nma_R.frequencies_cm, masses, [proton_idx],
        )

        # Proton frequency
        omega_H = nma_R.proton_frequency_cm
        if not omega_H:
            raise ValueError(
                "Could not identify proton vibrational mode in reactant Hessian. "
                "Check that proton_idx is correct and the Hessian is well-converged."
            )
        omega_P_freq = nma_P.proton_frequency_cm or omega_H

        # Donor-acceptor distance (average of R and P)
        d_DA_R = compute_donor_acceptor_distance(geom_R, donor_idx, acceptor_idx)
        d_DA_P = compute_donor_acceptor_distance(geom_P, donor_idx, acceptor_idx)
        d_DA = 0.5 * (d_DA_R + d_DA_P)

        # Extract bond lengths from geometry if not provided
        if r_DH is None:
            r_DH = compute_donor_acceptor_distance(geom_R, donor_idx, proton_idx)
        if r_AH is None:
            r_AH = compute_donor_acceptor_distance(geom_P, acceptor_idx, proton_idx)

        # Auto-gating: extract Omega_gating and M_DA from Hessian
        gating_result = None
        if auto_gating:
            gating_result = identify_da_stretching_mode(
                hessian_R, masses, donor_idx, acceptor_idx, geom_R,
            )
            Omega_gating = gating_result.omega_gating
            M_DA = gating_result.M_DA

        # Inner-sphere reorganization energy from Hessians
        # Exclude the transferring proton — its coordinate is treated
        # quantum mechanically in the vibronic formalism.
        geom_R_flat = (geom_R * ANGSTROM_TO_BOHR).flatten()
        geom_P_flat = (geom_P * ANGSTROM_TO_BOHR).flatten()
        lam_f, lam_b = reorganization_energy_from_hessians(
            hessian_R, hessian_P, geom_R_flat, geom_P_flat, masses,
            exclude_atoms=[proton_idx],
        )
        lambda_inner = 0.5 * (lam_f + lam_b) * HARTREE_TO_KCALMOL
        lambda_total = lambda_inner + lambda_outer

        result = self.compute_rate(
            V_el=V_el,
            delta_G=delta_G,
            lambda_reorg=lambda_total,
            omega_H=omega_H,
            d_DA=d_DA,
            omega_P=omega_P_freq,
            method=method,
            r_DH=r_DH,
            r_AH=r_AH,
            delta_0=delta_0,
            Omega_gating=Omega_gating,
            M_DA=M_DA,
            n_eff=n_eff if n_eff is not None else None,
        )
        # Attach full participation result (always computed, even if not applied)
        result.participation = part_result
        if result.participation is not None:
            result.n_eff = part_result.n_eff_proton
        # Attach gating result
        result.gating = gating_result
        return result

    # =================================================================
    # FGH-based rate from arbitrary proton potentials
    # =================================================================

    def compute_rate_from_potential(
        self,
        r_grid: np.ndarray,
        V_reactant,
        V_product,
        V_el: float,
        delta_G: float,
        lambda_reorg: float,
        n_grid: int = 256,
        r_range: tuple[float, float] | None = None,
    ) -> PCETResult:
        """Compute PCET rate from arbitrary proton potential energy curves.

        Uses the Fourier Grid Hamiltonian (FGH) method to solve for
        exact vibrational wavefunctions in anharmonic potentials, then
        computes numerical Franck-Condon overlaps.

        This method handles anharmonic, asymmetric, and double-well
        potentials — not just the harmonic approximation.

        Args:
            r_grid: Proton coordinate grid in angstrom. If V_reactant/V_product
                are callable, this is used to evaluate them. If arrays, this
                must match their length.
            V_reactant: Reactant proton potential in eV. Either array or
                callable V(r) taking angstrom, returning eV.
            V_product: Product proton potential in eV (same format).
            V_el: Electronic coupling in kcal/mol.
            delta_G: Electronic driving force in kcal/mol (not including
                proton ZPE — that's handled by the vibronic formalism).
            lambda_reorg: Total reorganization energy in kcal/mol.
            n_grid: Number of grid points for FGH (if r_grid not provided).
            r_range: (r_min, r_max) in angstrom for grid generation.

        Returns:
            PCETResult with k_H, k_D, KIE computed from FGH wavefunctions.
        """
        from pcet_engine.core.fgh_solver import fgh_1d, compute_fc_overlaps
        from pcet_engine.core.constants import (
            KCALMOL_TO_HARTREE, HARTREE_TO_KCALMOL,
            HARTREE_TO_EV, EV_TO_HARTREE,
            KB_HARTREE, HBAR_AU, TWO_PI, AU_RATE_TO_PER_S,
        )

        r = np.asarray(r_grid)

        # Evaluate potentials on grid if callable
        if callable(V_reactant):
            pot_R = np.asarray(V_reactant(r))
        else:
            pot_R = np.asarray(V_reactant)

        if callable(V_product):
            pot_P = np.asarray(V_product(r))
        else:
            pot_P = np.asarray(V_product)

        n_R = self.n_reactant_states
        n_P = self.n_product_states
        T = self.temperature

        dG_eV = delta_G * KCALMOL_TO_HARTREE * HARTREE_TO_EV
        lam_eV = lambda_reorg * KCALMOL_TO_HARTREE * HARTREE_TO_EV
        V_el_eV = V_el * KCALMOL_TO_HARTREE * HARTREE_TO_EV

        results = {}
        for label, mass_amu in [("H", PROTON_MASS_AMU), ("D", DEUTERIUM_MASS_AMU)]:
            # Solve FGH for both potentials
            E_R, wfcs_R, _ = fgh_1d(r, pot_R, mass_amu, n_R)
            E_P, wfcs_P, _ = fgh_1d(r, pot_P, mass_amu, n_P)

            # Numerical FC overlaps
            S = compute_fc_overlaps(wfcs_R, wfcs_P, r)

            # Boltzmann weights
            kBT_eV = KB_HARTREE * T * HARTREE_TO_EV
            dE_R = E_R - E_R[0]
            boltz = np.exp(-dE_R / kBT_eV)
            P_mu = boltz / np.sum(boltz)

            # Channel rates
            kBT_Ha = KB_HARTREE * T
            lam_Ha = lambda_reorg * KCALMOL_TO_HARTREE
            V_Ha = V_el * KCALMOL_TO_HARTREE
            k0 = TWO_PI / HBAR_AU * V_Ha**2

            k_total = 0.0
            for mu in range(n_R):
                for nu in range(n_P):
                    # Vibronic driving force
                    dG_munu_eV = dG_eV + (E_P[nu] - E_P[0]) - (E_R[mu] - E_R[0])
                    dG_munu_Ha = dG_munu_eV * EV_TO_HARTREE

                    S_sq = S[mu, nu]**2
                    if S_sq < 1e-30:
                        continue

                    E_a = (dG_munu_Ha + lam_Ha)**2 / (4.0 * lam_Ha)
                    prefactor = k0 * S_sq / math.sqrt(4.0 * math.pi * lam_Ha * kBT_Ha)
                    if E_a / kBT_Ha < 700:
                        rate_au = prefactor * math.exp(-E_a / kBT_Ha)
                    else:
                        rate_au = 0.0

                    k_total += P_mu[mu] * rate_au * AU_RATE_TO_PER_S

            results[label] = k_total

        k_H = results["H"]
        k_D = results["D"]
        KIE = k_H / k_D if k_D > 0 else float("inf")

        dG_Ha = delta_G * KCALMOL_TO_HARTREE
        lam_Ha = lambda_reorg * KCALMOL_TO_HARTREE
        E_a = (dG_Ha + lam_Ha)**2 / (4.0 * lam_Ha) * HARTREE_TO_KCALMOL

        # Get dominant proton frequency from FGH energy levels
        E_R_H, _, _ = fgh_1d(r, pot_R, PROTON_MASS_AMU, 2)
        omega_H_eV = E_R_H[1] - E_R_H[0] if len(E_R_H) > 1 else 0.0
        omega_H_cm = omega_H_eV * EV_TO_HARTREE / CM_TO_HARTREE if omega_H_eV > 0 else 0.0
        omega_D_cm = omega_H_cm * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)

        return PCETResult(
            k_H=k_H,
            k_D=k_D,
            KIE=KIE,
            E_a=E_a,
            delta_G=delta_G,
            lambda_reorg=lambda_reorg,
            V_el=V_el,
            omega_H=omega_H_cm,
            omega_D=omega_D_cm,
            d_DA=0.0,  # Not applicable for potential-based
            method="fgh_vibronic",
        )

    # =================================================================
    # Photochemical PCET (convenience wrapper)
    # =================================================================

    def compute_rate_photochemical(
        self,
        V_el: float,
        E_excited: float,
        E_substrate: float,
        lambda_reorg: float,
        omega_H: float,
        d_DA: float,
        n_electrons: int = 1,
        **kwargs,
    ) -> PCETResult:
        """Compute photochemical PCET rate.

        Photochemical PCET uses the same vibronic rate expression as
        thermal PCET, but the driving force is determined by the
        excited-state redox potential of the photosensitizer:

            ΔG = -n × F × (E°*(photosensitizer) - E°(substrate))

        In kcal/mol: ΔG = -n × 23.06 × (E°* - E°_sub)

        Args:
            V_el: Electronic coupling in kcal/mol.
            E_excited: Excited-state reduction potential E°* in V vs. ref.
            E_substrate: Substrate oxidation potential E° in V vs. ref.
            lambda_reorg: Total reorganization energy in kcal/mol.
            omega_H: Proton vibrational frequency in cm⁻¹.
            d_DA: Donor-acceptor distance in angstrom.
            n_electrons: Number of electrons transferred (usually 1).
            **kwargs: Additional arguments passed to compute_rate().

        Returns:
            PCETResult.
        """
        F_kcal = 23.0605  # F in kcal/(mol·V)
        delta_G = -n_electrons * F_kcal * (E_excited - E_substrate)

        return self.compute_rate(
            V_el=V_el,
            delta_G=delta_G,
            lambda_reorg=lambda_reorg,
            omega_H=omega_H,
            d_DA=d_DA,
            **kwargs,
        )

    # =================================================================
    # Electrochemical PCET
    # =================================================================

    def compute_rate_electrochemical(
        self,
        V_el: float,
        delta_G_base: float,
        lambda_reorg: float,
        omega_H: float,
        d_DA: float,
        overpotential: float = 0.0,
        direction: str = "anodic",
        epsilons: np.ndarray | None = None,
        rho_DOS: np.ndarray | None = None,
        beta_decay: float = 1.0,
        epsilon_range: tuple[float, float] = (-2.0, 2.0),
        n_epsilon: int = 101,
        **kwargs,
    ) -> 'ElectrochemicalResult':
        """Compute electrochemical PCET rate with Fermi-weighted integration.

        Integrates the vibronic rate over electrode electronic states:

            k(η) = ∫ dε ρ(ε)/β' × [1-f(ε)] × k_PCET(ΔG + ε - eη)  [anodic]
            k(η) = ∫ dε ρ(ε)/β' × f(ε) × k_PCET(ΔG - ε + eη)      [cathodic]

        Args:
            V_el: Electronic coupling in kcal/mol.
            delta_G_base: Base reaction free energy at η=0 in kcal/mol.
            lambda_reorg: Total reorganization energy in kcal/mol.
            omega_H: Proton vibrational frequency in cm⁻¹.
            d_DA: Donor-acceptor distance in angstrom.
            overpotential: Applied overpotential η in V.
            direction: 'anodic' or 'cathodic'.
            epsilons: Electrode energy levels in eV (relative to Fermi).
            rho_DOS: Density of states at each epsilon (states/eV).
            beta_decay: Electronic coupling distance decay β' in Å⁻¹.
            epsilon_range: (min, max) for epsilon if not provided.
            n_epsilon: Number of epsilon points if not provided.
            **kwargs: Additional arguments for compute_rate().

        Returns:
            ElectrochemicalResult.
        """
        from pcet_engine.core.electrochemistry import (
            fermi_dirac, electrochemical_rate, ElectrochemicalResult,
        )
        from pcet_engine.core.constants import EV_TO_KCALMOL

        if epsilons is None:
            epsilons = np.linspace(epsilon_range[0], epsilon_range[1], n_epsilon)
        if rho_DOS is None:
            rho_DOS = np.ones_like(epsilons)

        def _rate_at_dG_eV(dG_eV):
            """Compute vibronic rate for a given ΔG in eV."""
            dG_kcal = dG_eV * EV_TO_KCALMOL
            result = self.compute_rate(
                V_el=V_el,
                delta_G=dG_kcal,
                lambda_reorg=lambda_reorg,
                omega_H=omega_H,
                d_DA=d_DA,
                **kwargs,
            )
            return result.k_H

        def _rate_D_at_dG_eV(dG_eV):
            dG_kcal = dG_eV * EV_TO_KCALMOL
            result = self.compute_rate(
                V_el=V_el,
                delta_G=dG_kcal,
                lambda_reorg=lambda_reorg,
                omega_H=omega_H,
                d_DA=d_DA,
                **kwargs,
            )
            return result.k_D

        dG_base_eV = delta_G_base / EV_TO_KCALMOL

        k_H_total, k_H_eps = electrochemical_rate(
            _rate_at_dG_eV, dG_base_eV, epsilons, rho_DOS,
            overpotential, self.temperature, direction, beta_decay,
        )
        k_D_total, k_D_eps = electrochemical_rate(
            _rate_D_at_dG_eV, dG_base_eV, epsilons, rho_DOS,
            overpotential, self.temperature, direction, beta_decay,
        )

        KIE = k_H_total / k_D_total if k_D_total > 0 else float("inf")

        return ElectrochemicalResult(
            k_H=k_H_total,
            k_D=k_D_total,
            KIE=KIE,
            overpotential=overpotential,
            direction=direction,
            k_H_per_epsilon=k_H_eps,
            k_D_per_epsilon=k_D_eps,
            epsilons=epsilons,
        )
