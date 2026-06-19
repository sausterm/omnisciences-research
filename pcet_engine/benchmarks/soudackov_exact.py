"""
Reproduce Soudackov's SLO-1 calculation using his exact Morse parameters.

Morse Parameters (personal communication, 2026-03-26):
  D_CH = 77 kcal/mol,  β_CH = 2.068 Å⁻¹,  r_CH = 1.09 Å
  D_OH = 82 kcal/mol,  β_OH = 2.442 Å⁻¹,  r_OH = 0.96 Å

Attenuation params defined as:
  α_μν = -d(log S_μν)/dR |_{R₀}
  γ_μν = -d²(log S_μν)/dR² |_{R₀}

R_DA = 2.77 Å → δ₀ = R_DA - r_CH - r_OH = 2.77 - 1.09 - 0.96 = 0.72 Å
"""

import math
import numpy as np
from pcet_engine.core.fgh import solve_1d_schrodinger, numerical_fc_overlap, FGHResult
from pcet_engine.core.fgh import analytical_gating_rate_table
from pcet_engine.core.constants import (
    ANGSTROM_TO_BOHR, AMU_TO_AU, KCALMOL_TO_HARTREE,
    KB_HARTREE, TWO_PI, AU_RATE_TO_PER_S, HARTREE_TO_CM,
)

# ======== Soudackov's exact Morse parameters ========
D_CH_KCAL = 77.0    # kcal/mol
D_OH_KCAL = 82.0    # kcal/mol
BETA_CH_AINV = 2.068  # Å⁻¹
BETA_OH_AINV = 2.442  # Å⁻¹
R_CH = 1.09  # Å
R_OH = 0.96  # Å
R_DA = 2.77  # Å

# Convert
D_CH = D_CH_KCAL * KCALMOL_TO_HARTREE
D_OH = D_OH_KCAL * KCALMOL_TO_HARTREE
# β has units [1/length]: β_bohr = β_Å / ANGSTROM_TO_BOHR (DIVIDE, not multiply!)
BETA_CH = BETA_CH_AINV / ANGSTROM_TO_BOHR  # bohr⁻¹
BETA_OH = BETA_OH_AINV / ANGSTROM_TO_BOHR  # bohr⁻¹

MASS_H = 1.00782503207
MASS_D = 2.01410177812

# SLO-1 rate parameters
DELTA_G = -5.4 * KCALMOL_TO_HARTREE
LAMBDA = 13.4 * KCALMOL_TO_HARTREE
M_DA = 100.0  # amu
OMEGA_GATING = 132.8  # cm⁻¹
TEMP = 303.0
K_H_EXP = 297.0
K_D_EXP = 3.7
KIE_EXP = 81.0


def build_morse_potentials(mass_amu, delta_angstrom, n_grid=512, padding=3.0):
    """Build reactant (C-H) and product (O-H) Morse potentials on a shared grid.

    The reactant proton sits at x=0 (C-H minimum).
    The product proton sits at x=δ (O-H minimum), with the O-H potential
    reflected (proton approaches from left).
    """
    mass_au = mass_amu * AMU_TO_AU
    delta_bohr = delta_angstrom * ANGSTROM_TO_BOHR

    x_min = -padding * ANGSTROM_TO_BOHR
    x_max = delta_bohr + padding * ANGSTROM_TO_BOHR
    grid = np.linspace(x_min, x_max, n_grid)

    # Reactant: V_R(x) = D_CH * (1 - exp(-β_CH * x))²
    # Minimum at x=0 (proton at C-H equilibrium)
    V_R = D_CH * (1.0 - np.exp(-BETA_CH * grid)) ** 2

    # Product: V_P(x) = D_OH * (1 - exp(-β_OH * (δ - x)))²
    # Minimum at x=δ (proton at O-H equilibrium, reflected)
    # Note: the proton is measured from C, product well from O
    V_P = D_OH * (1.0 - np.exp(-BETA_OH * (delta_bohr - grid))) ** 2

    return grid, V_R, V_P, mass_au


def compute_overlaps_and_attenuation(mass_amu, n_states=4, h_angstrom=0.01):
    """Compute S²(R₀), α_μν, γ_μν using Soudackov's exact Morse parameters."""
    delta_0 = R_DA - R_CH - R_OH  # tunneling distance

    n_R = n_states
    n_P = n_states
    S_sq_0 = np.zeros((n_R, n_P))
    alpha_mn = np.zeros((n_R, n_P))
    gamma_mn = np.zeros((n_R, n_P))

    h_bohr = h_angstrom * ANGSTROM_TO_BOHR

    for d_offset, label in [(-h_angstrom, 'm'), (0.0, '0'), (h_angstrom, 'p')]:
        delta = delta_0 + d_offset
        if delta < 0.02:
            delta = 0.02

        grid, V_R, V_P, mass_au = build_morse_potentials(mass_amu, delta, n_grid=512)
        fgh_R = solve_1d_schrodinger(grid, V_R, mass_au, n_states)
        fgh_P = solve_1d_schrodinger(grid, V_P, mass_au, n_states)

        for mu in range(n_R):
            for nu in range(n_P):
                S_sq = numerical_fc_overlap(fgh_R, fgh_P, mu, nu)
                if label == 'm':
                    S_sq_0[mu, nu] = S_sq  # temp storage
                elif label == '0':
                    S_m = S_sq_0[mu, nu]  # retrieve previous
                    S_sq_0[mu, nu] = S_sq  # overwrite with R₀ value
                    # Store S_m for later
                    if not hasattr(compute_overlaps_and_attenuation, '_S_m'):
                        compute_overlaps_and_attenuation._S_m = np.zeros((n_R, n_P))
                    compute_overlaps_and_attenuation._S_m[mu, nu] = S_m
                else:  # 'p'
                    S_m = compute_overlaps_and_attenuation._S_m[mu, nu]
                    S_0 = S_sq_0[mu, nu]
                    S_p = S_sq

                    if S_0 < 1e-30:
                        continue

                    # log|S| = 0.5 * log(|S|²)  (Soudackov convention)
                    log_m = 0.5 * math.log(S_m) if S_m > 1e-300 else -350.0
                    log_0 = 0.5 * math.log(S_0) if S_0 > 1e-300 else -350.0
                    log_p = 0.5 * math.log(S_p) if S_p > 1e-300 else -350.0

                    alpha_mn[mu, nu] = -(log_p - log_m) / (2.0 * h_bohr)
                    gamma_mn[mu, nu] = -(log_p - 2.0 * log_0 + log_m) / h_bohr**2

    # Clean up
    if hasattr(compute_overlaps_and_attenuation, '_S_m'):
        del compute_overlaps_and_attenuation._S_m

    return S_sq_0, alpha_mn, gamma_mn


def main():
    delta_0 = R_DA - R_CH - R_OH
    print("=" * 70)
    print("SOUDACKOV EXACT MORSE PARAMETERS — SLO-1 REPRODUCTION")
    print("=" * 70)
    print(f"  D_CH = {D_CH_KCAL} kcal/mol, β_CH = {BETA_CH_AINV} Å⁻¹, r_CH = {R_CH} Å")
    print(f"  D_OH = {D_OH_KCAL} kcal/mol, β_OH = {BETA_OH_AINV} Å⁻¹, r_OH = {R_OH} Å")
    print(f"  R_DA = {R_DA} Å → δ₀ = {delta_0:.2f} Å")
    print()

    # ---- Compute with H mass ----
    print("-" * 70)
    print("Hydrogen overlaps and attenuation parameters")
    print("-" * 70)
    S_sq_H, alpha_H, gamma_H = compute_overlaps_and_attenuation(MASS_H)

    # Soudackov reference
    from pcet_engine.benchmarks.soudackov_correction import soudackov_reference_params
    ref_aH, ref_aD, ref_gH, ref_gD = soudackov_reference_params()

    print(f"\n  {'(μ,ν)':<8} {'|S²(R₀)|':<12} {'α_H ours':<12} {'α_H Soud':<12} {'γ_H ours':<12} {'γ_H Soud':<12}")
    print(f"  {'-'*68}")
    for mu in range(4):
        for nu in range(4):
            print(f"  ({mu},{nu})    {S_sq_H[mu,nu]:<12.3e} {alpha_H[mu,nu]:<12.4f} {ref_aH[mu,nu]:<12.4f} {gamma_H[mu,nu]:<12.4f} {ref_gH[mu,nu]:<12.4f}")

    # ---- Compute with D mass ----
    print()
    print("-" * 70)
    print("Deuterium overlaps and attenuation parameters")
    print("-" * 70)
    OMEGA_D_RATIO = math.sqrt(MASS_H / MASS_D)
    S_sq_D, alpha_D, gamma_D = compute_overlaps_and_attenuation(MASS_D)

    print(f"\n  {'(μ,ν)':<8} {'|S²(R₀)|':<12} {'α_D ours':<12} {'α_D Soud':<12} {'γ_D ours':<12} {'γ_D Soud':<12}")
    print(f"  {'-'*68}")
    for mu in range(4):
        for nu in range(4):
            print(f"  ({mu},{nu})    {S_sq_D[mu,nu]:<12.3e} {alpha_D[mu,nu]:<12.4f} {ref_aD[mu,nu]:<12.4f} {gamma_D[mu,nu]:<12.4f} {ref_gD[mu,nu]:<12.4f}")

    # ---- Compute rates using OUR α,γ (from Soudackov's Morse) ----
    print()
    print("=" * 70)
    print("RATE CALCULATION: Our FGH + Soudackov's Morse params")
    print("=" * 70)

    kBT = KB_HARTREE * TEMP

    # Get energy levels from FGH
    grid_H, V_R_H, V_P_H, mass_au_H = build_morse_potentials(MASS_H, delta_0)
    fgh_R_H = solve_1d_schrodinger(grid_H, V_R_H, mass_au_H, 4)
    fgh_P_H = solve_1d_schrodinger(grid_H, V_P_H, mass_au_H, 4)
    E_R_H = fgh_R_H.energies[:4] - fgh_R_H.energies[0]
    E_P_H = fgh_P_H.energies[:4] - fgh_P_H.energies[0]

    grid_D, V_R_D, V_P_D, mass_au_D = build_morse_potentials(MASS_D, delta_0)
    fgh_R_D = solve_1d_schrodinger(grid_D, V_R_D, mass_au_D, 4)
    fgh_P_D = solve_1d_schrodinger(grid_D, V_P_D, mass_au_D, 4)
    E_R_D = fgh_R_D.energies[:4] - fgh_R_D.energies[0]
    E_P_D = fgh_P_D.energies[:4] - fgh_P_D.energies[0]

    boltz_H = np.exp(-E_R_H / kBT)
    P_mu_H = boltz_H / np.sum(boltz_H)
    boltz_D = np.exp(-E_R_D / kBT)
    P_mu_D = boltz_D / np.sum(boltz_D)

    print(f"  H energy levels (cm⁻¹): {[f'{e/4.5563e-6:.0f}' for e in E_R_H]}")
    print(f"  D energy levels (cm⁻¹): {[f'{e/4.5563e-6:.0f}' for e in E_R_D]}")
    print(f"  H Boltzmann: {P_mu_H}")
    print(f"  D Boltzmann: {P_mu_D}")

    def compute_rate(V_el_kcal, S_sq_eff, E_R, E_P, P_mu):
        V_el_au = V_el_kcal * KCALMOL_TO_HARTREE
        total = 0.0
        n = min(4, len(P_mu))
        for mu in range(n):
            for nu in range(n):
                dG_mn = DELTA_G + (E_P[nu] - E_R[mu])
                S_sq = S_sq_eff[mu, nu]
                if S_sq < 1e-30:
                    continue
                E_a = (dG_mn + LAMBDA)**2 / (4.0 * LAMBDA)
                pf = TWO_PI * V_el_au**2 * S_sq / math.sqrt(4 * math.pi * LAMBDA * kBT)
                rate_au = pf * math.exp(-E_a / kBT) if E_a / kBT < 700 else 0.0
                total += P_mu[mu] * rate_au * AU_RATE_TO_PER_S
        return total

    # Test with both our params and Soudackov's reference params
    for label, aH, aD, gH, gD in [
        ("Our α,γ (Soudackov Morse)", alpha_H, alpha_D, gamma_H, gamma_D),
        ("Soudackov reference α,γ", ref_aH, ref_aD, ref_gH, ref_gD),
    ]:
        print(f"\n  --- {label} ---")

        S_sq_eff_H, _ = analytical_gating_rate_table(
            S_sq_H, aH, gH, M_DA, OMEGA_GATING, TEMP,
        )
        S_sq_eff_D, _ = analytical_gating_rate_table(
            S_sq_D, aD, gD, M_DA, OMEGA_GATING, TEMP,
        )

        # Binary search for V_el
        V_lo, V_hi = 0.0001, 50.0
        for _ in range(80):
            V_mid = (V_lo + V_hi) / 2.0
            k_H = compute_rate(V_mid, S_sq_eff_H, E_R_H, E_P_H, P_mu_H)
            if k_H <= 0:
                V_lo = V_mid
            elif k_H / K_H_EXP > 1.01:
                V_hi = V_mid
            elif k_H / K_H_EXP < 0.99:
                V_lo = V_mid
            else:
                break

        k_D = compute_rate(V_mid, S_sq_eff_D, E_R_D, E_P_D, P_mu_D)
        KIE = k_H / k_D if k_D > 0 else float('inf')
        V_cm = V_mid * KCALMOL_TO_HARTREE * HARTREE_TO_CM

        print(f"    V_el = {V_mid:.4f} kcal/mol = {V_cm:.2f} cm⁻¹  ({V_cm/1.7:.2f}× SHS)")
        print(f"    k_H = {k_H:.1f} s⁻¹ (exp: {K_H_EXP})")
        print(f"    k_D = {k_D:.2f} s⁻¹ (exp: {K_D_EXP})")
        print(f"    KIE = {KIE:.1f} (exp: {KIE_EXP})")

        # Also at V_el = 1.7 cm⁻¹
        V_shs = 1.7 / HARTREE_TO_CM / KCALMOL_TO_HARTREE
        k_H_17 = compute_rate(V_shs, S_sq_eff_H, E_R_H, E_P_H, P_mu_H)
        k_D_17 = compute_rate(V_shs, S_sq_eff_D, E_R_D, E_P_D, P_mu_D)
        KIE_17 = k_H_17 / k_D_17 if k_D_17 > 0 else float('inf')
        print(f"    At V_el=1.7 cm⁻¹: k_H={k_H_17:.2e}, k_D={k_D_17:.2e}, KIE={KIE_17:.1f}, ratio={k_H_17/K_H_EXP:.3f}")


if __name__ == "__main__":
    main()
