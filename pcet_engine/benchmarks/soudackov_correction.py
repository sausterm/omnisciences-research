"""
Test the Soudackov correction: state-pair-specific Morse FC overlaps
with R-averaging over the D-A gating coordinate.

The 6.6x V_el discrepancy (our 11.17 cm⁻¹ vs SHS 1.7 cm⁻¹) should be
explained by the different treatment of FC overlaps:
- Our old approach: harmonic α = mω/ℏ, same for all (μ,ν) pairs
- Soudackov's approach: Morse overlaps with state-pair-specific α_μν

This script compares:
1. vibronic_multi + gating (harmonic FC, numerical R-averaging)
2. vibronic_multi_morse_gating (Morse FGH FC, numerical R-averaging)

If the correction works, the Morse gating method should require a V_el
closer to 1.7 cm⁻¹ to reproduce k_H = 297 s⁻¹.

Reference: PMC5217758 (Soudackov & Hammes-Schiffer)
"""

import math
import numpy as np
from pcet_engine.core.rate_engine import PCETRateEngine
from pcet_engine.core.constants import HARTREE_TO_CM, KCALMOL_TO_HARTREE

# SLO-1 WT parameters from PMC5217758 (Model 1)
SLO_PARAMS = dict(
    delta_G=-5.4,           # kcal/mol
    lambda_reorg=13.4,      # kcal/mol
    omega_H=2900.0,         # cm⁻¹
    d_DA=2.77,              # Å
    Omega_gating=132.8,     # cm⁻¹
    M_DA=100.0,             # amu
    r_DH=1.09,              # Å
    r_AH=0.96,              # Å
)

K_H_EXP = 297.0  # s⁻¹ at 303 K
K_D_EXP = 3.7     # s⁻¹
KIE_EXP = 81.0
TEMPERATURE = 303.0


def find_vel_for_rate(engine, method, target_k_H, params, tol=0.01, max_iter=50):
    """Binary search for V_el that reproduces target k_H."""
    V_lo, V_hi = 0.001, 20.0  # kcal/mol

    for _ in range(max_iter):
        V_mid = (V_lo + V_hi) / 2.0
        result = engine.compute_rate(V_el=V_mid, method=method, **params)
        k_H = result.k_H

        if k_H <= 0:
            V_lo = V_mid
            continue

        ratio = k_H / target_k_H
        if abs(ratio - 1.0) < tol:
            return V_mid, result
        elif ratio > 1.0:
            V_hi = V_mid
        else:
            V_lo = V_mid

    return V_mid, result


def main():
    engine = PCETRateEngine(
        n_reactant_states=3,  # Match SHS: 3 reactant states
        n_product_states=3,   # Match SHS: 3 product states
        temperature=TEMPERATURE,
    )

    print("=" * 70)
    print("Soudackov Correction: State-Pair-Specific Morse FC Overlaps")
    print("=" * 70)
    print(f"\nTarget: k_H = {K_H_EXP} s⁻¹, KIE = {KIE_EXP}")
    print(f"SHS V_el = 1.7 cm⁻¹ (PMC5217758, Model 1)")
    print()

    # ---- Method 1: harmonic + gating (our old approach) ----
    print("-" * 70)
    print("Method 1: vibronic_multi (harmonic FC + numerical R-averaging)")
    print("-" * 70)

    V_el_harm, res_harm = find_vel_for_rate(
        engine, "vibronic_multi", K_H_EXP, SLO_PARAMS,
    )
    V_el_harm_cm = V_el_harm * KCALMOL_TO_HARTREE * HARTREE_TO_CM

    print(f"  V_el fitted = {V_el_harm:.4f} kcal/mol = {V_el_harm_cm:.2f} cm⁻¹")
    print(f"  k_H  = {res_harm.k_H:.1f} s⁻¹ (target: {K_H_EXP})")
    print(f"  k_D  = {res_harm.k_D:.2f} s⁻¹ (exp: {K_D_EXP})")
    print(f"  KIE  = {res_harm.KIE:.1f} (exp: {KIE_EXP})")
    print()

    # ---- Method 2: Morse FGH + numerical R-averaging ----
    print("-" * 70)
    print("Method 2: vibronic_multi_morse_gating (Morse FGH + numerical R-avg)")
    print("-" * 70)

    V_el_morse, res_morse = find_vel_for_rate(
        engine, "vibronic_multi_morse_gating", K_H_EXP, SLO_PARAMS,
    )
    V_el_morse_cm = V_el_morse * KCALMOL_TO_HARTREE * HARTREE_TO_CM

    print(f"  V_el fitted = {V_el_morse:.4f} kcal/mol = {V_el_morse_cm:.2f} cm⁻¹")
    print(f"  k_H  = {res_morse.k_H:.1f} s⁻¹ (target: {K_H_EXP})")
    print(f"  k_D  = {res_morse.k_D:.2f} s⁻¹ (exp: {K_D_EXP})")
    print(f"  KIE  = {res_morse.KIE:.1f} (exp: {KIE_EXP})")
    print()

    # ---- Method 3: Soudackov correction (analytical gating + Morse α_μν, γ_μν) ----
    print("-" * 70)
    print("Method 3: soudackov_corrected (analytical gating + Morse α_μν, γ_μν)")
    print("-" * 70)

    V_el_sc, res_sc = find_vel_for_rate(
        engine, "soudackov_corrected", K_H_EXP, SLO_PARAMS,
    )
    V_el_sc_cm = V_el_sc * KCALMOL_TO_HARTREE * HARTREE_TO_CM

    print(f"  V_el fitted = {V_el_sc:.4f} kcal/mol = {V_el_sc_cm:.2f} cm⁻¹")
    print(f"  k_H  = {res_sc.k_H:.1f} s⁻¹ (target: {K_H_EXP})")
    print(f"  k_D  = {res_sc.k_D:.2f} s⁻¹ (exp: {K_D_EXP})")
    print(f"  KIE  = {res_sc.KIE:.1f} (exp: {KIE_EXP})")
    print()

    # ---- Print α_μν and γ_μν for inspection ----
    print("-" * 70)
    print("State-pair-specific attenuation parameters (Morse)")
    print("-" * 70)

    from pcet_engine.core.fgh import compute_attenuation_params
    from pcet_engine.core.constants import ANGSTROM_TO_BOHR

    delta_0 = SLO_PARAMS["d_DA"] - SLO_PARAMS["r_DH"] - SLO_PARAMS["r_AH"]
    S_sq_0, alpha_mn, gamma_mn = compute_attenuation_params(
        2900.0, 2900.0, 1.00782503207, delta_0,
        n_reactant_states=3, n_product_states=3,
    )

    # Also compute the harmonic α for comparison
    from pcet_engine.core.constants import AMU_TO_AU, HBAR_AU, CM_TO_HARTREE as CM2H
    mass_au = 1.00782503207 * AMU_TO_AU
    omega_au = 2900.0 * CM2H
    alpha_harmonic = mass_au * omega_au / HBAR_AU
    print(f"\n  Harmonic α = mω/ℏ = {alpha_harmonic:.4f} bohr⁻¹")
    print(f"  (same for ALL state pairs in harmonic approx)")
    print()
    print(f"  {'(μ,ν)':<8} {'|S_μν(R₀)|²':<14} {'α_μν (bohr⁻¹)':<16} {'γ_μν (bohr⁻²)':<16}")
    print(f"  {'-'*54}")
    for mu in range(3):
        for nu in range(3):
            print(f"  ({mu},{nu})    {S_sq_0[mu,nu]:<14.6e} {alpha_mn[mu,nu]:<16.4f} {gamma_mn[mu,nu]:<16.4f}")
    print()

    # ---- Direct test with SHS V_el = 1.7 cm⁻¹ ----
    print("-" * 70)
    print("Direct test: V_el = 1.7 cm⁻¹ (SHS value)")
    print("-" * 70)

    V_el_shs_kcal = 1.7 / HARTREE_TO_CM / KCALMOL_TO_HARTREE

    for method_name, method_key in [
        ("Harmonic + num gating", "vibronic_multi"),
        ("Morse FGH + num gating", "vibronic_multi_morse_gating"),
        ("Soudackov corrected", "soudackov_corrected"),
    ]:
        result = engine.compute_rate(V_el=V_el_shs_kcal, method=method_key, **SLO_PARAMS)
        ratio = result.k_H / K_H_EXP if K_H_EXP > 0 else 0
        print(f"  {method_name}:")
        print(f"    k_H = {result.k_H:.2e} s⁻¹ (exp: {K_H_EXP}), ratio = {ratio:.3f}")
        print(f"    k_D = {result.k_D:.2e} s⁻¹, KIE = {result.KIE:.1f}")
        print()

    # ---- Summary ----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  SHS V_el (analytical + Morse α_μν):  1.7 cm⁻¹")
    print(f"  Our V_el (harmonic + num gating):     {V_el_harm_cm:.2f} cm⁻¹  ({V_el_harm_cm/1.7:.1f}× SHS)")
    print(f"  Our V_el (Morse FGH + num gating):    {V_el_morse_cm:.2f} cm⁻¹  ({V_el_morse_cm/1.7:.1f}× SHS)")
    print(f"  Our V_el (Soudackov corrected):       {V_el_sc_cm:.2f} cm⁻¹  ({V_el_sc_cm/1.7:.1f}× SHS)")
    print()
    print(f"  KIE (harmonic):     {res_harm.KIE:.1f} (exp: {KIE_EXP})")
    print(f"  KIE (Morse num):    {res_morse.KIE:.1f}")
    print(f"  KIE (Soudackov):    {res_sc.KIE:.1f}")
    print()
    gap_closed = (V_el_harm_cm - V_el_sc_cm) / (V_el_harm_cm - 1.7) * 100
    print(f"  Gap closed: {gap_closed:.0f}%")

    # ---- D_e sensitivity analysis ----
    print()
    print("=" * 70)
    print("D_e SENSITIVITY: Effect of Morse dissociation energy")
    print("=" * 70)
    print("  Default D_e = 20ℏω = 0.264 hartree (166 kcal/mol)")
    print("  Real C-H D_e ≈ 100 kcal/mol (0.159 hartree)")
    print("  Real O-H D_e ≈ 110 kcal/mol (0.175 hartree)")
    print()

    from pcet_engine.core.fgh import compute_attenuation_params, analytical_gating_rate_table

    D_e_values = {
        "Default (20ℏω)": None,  # 0.264 hartree
        "C-H realistic (100 kcal/mol)": 100.0 * KCALMOL_TO_HARTREE,
        "O-H realistic (110 kcal/mol)": 110.0 * KCALMOL_TO_HARTREE,
        "Lower (80 kcal/mol)": 80.0 * KCALMOL_TO_HARTREE,
        "Higher (150 kcal/mol)": 150.0 * KCALMOL_TO_HARTREE,
    }

    delta_0 = SLO_PARAMS["d_DA"] - SLO_PARAMS["r_DH"] - SLO_PARAMS["r_AH"]

    for label, D_e in D_e_values.items():
        S_sq_0, alpha_mn, gamma_mn = compute_attenuation_params(
            2900.0, 2900.0, 1.00782503207, delta_0,
            n_reactant_states=3, n_product_states=3,
            D_e_hartree=D_e,
        )
        S_eff, _ = analytical_gating_rate_table(
            S_sq_0, alpha_mn, gamma_mn,
            100.0, 132.8, TEMPERATURE,
        )
        print(f"  {label}:")
        print(f"    α₀₀ = {alpha_mn[0,0]:.2f} bohr⁻¹, S²₀₀(R₀) = {S_sq_0[0,0]:.3e}, S²₀₀(eff) = {S_eff[0,0]:.3e}")
        print(f"    α₂₂ = {alpha_mn[2,2]:.2f} bohr⁻¹, S²₂₂(R₀) = {S_sq_0[2,2]:.3e}, S²₂₂(eff) = {S_eff[2,2]:.3e}")
        ratio = S_eff[0,0] / S_sq_0[0,0] if S_sq_0[0,0] > 0 else 0
        print(f"    Gating boost (0,0): {ratio:.1f}×")
        print()

    # ---- Full V_el fit with realistic D_e ----
    print("-" * 70)
    print("V_el fit with realistic C-H D_e = 100 kcal/mol")
    print("-" * 70)

    # We need to thread D_e through the engine. Let's do it manually.
    from pcet_engine.core.fgh import r_averaged_fgh_fc_table, fgh_franck_condon_table
    from pcet_engine.core.constants import KB_HARTREE, TWO_PI, AU_RATE_TO_PER_S

    D_e_real = 100.0 * KCALMOL_TO_HARTREE

    def rate_with_De(V_el_kcal, mass_amu, omega_cm, omega_P_cm):
        """Compute rate with specified D_e."""
        from pcet_engine.core.fgh import compute_attenuation_params, analytical_gating_rate_table, fgh_franck_condon_table
        V_el_au = V_el_kcal * KCALMOL_TO_HARTREE
        dG_au = SLO_PARAMS["delta_G"] * KCALMOL_TO_HARTREE
        lam_au = SLO_PARAMS["lambda_reorg"] * KCALMOL_TO_HARTREE
        kBT = KB_HARTREE * TEMPERATURE

        delta_r = SLO_PARAMS["d_DA"] - SLO_PARAMS["r_DH"] - SLO_PARAMS["r_AH"]

        # Get attenuation params with real D_e
        S_sq_0, alpha_mn, gamma_mn = compute_attenuation_params(
            omega_cm, omega_P_cm, mass_amu, delta_r,
            n_reactant_states=3, n_product_states=3,
            D_e_hartree=D_e_real,
        )

        S_sq_eff, _ = analytical_gating_rate_table(
            S_sq_0, alpha_mn, gamma_mn,
            SLO_PARAMS["M_DA"], SLO_PARAMS["Omega_gating"], TEMPERATURE,
        )

        # Get energy levels
        _, energies_R, energies_P, _, _ = fgh_franck_condon_table(
            omega_cm, omega_P_cm, mass_amu, delta_r,
            n_reactant_states=3, n_product_states=3,
            D_e_hartree=D_e_real,
        )

        E_R = energies_R - energies_R[0]
        boltz = np.exp(-E_R / kBT)
        P_mu = boltz / np.sum(boltz)

        total_rate = 0.0
        for mu in range(3):
            for nu in range(3):
                dG_mn = dG_au + (energies_P[nu] - energies_P[0]) - (energies_R[mu] - energies_R[0])
                S_sq = S_sq_eff[mu, nu]
                if S_sq < 1e-30:
                    continue
                E_a = (dG_mn + lam_au)**2 / (4.0 * lam_au)
                pf = (TWO_PI) * V_el_au**2 * S_sq / math.sqrt(4*math.pi*lam_au*kBT)
                rate_au = pf * math.exp(-E_a / kBT) if E_a / kBT < 700 else 0.0
                total_rate += P_mu[mu] * rate_au * AU_RATE_TO_PER_S

        return total_rate

    # Binary search for V_el
    V_lo, V_hi = 0.001, 20.0
    for _ in range(50):
        V_mid = (V_lo + V_hi) / 2.0
        k_H = rate_with_De(V_mid, 1.00782503207, 2900.0, 2900.0)
        if k_H <= 0:
            V_lo = V_mid
        elif k_H / K_H_EXP > 1.0:
            V_hi = V_mid
        elif k_H / K_H_EXP < 0.99:
            V_lo = V_mid
        else:
            break

    V_el_De_cm = V_mid * KCALMOL_TO_HARTREE * HARTREE_TO_CM

    k_D = rate_with_De(V_mid, 2.01410177812, 2900.0 * math.sqrt(1.00782503207/2.01410177812), 2900.0 * math.sqrt(1.00782503207/2.01410177812))
    KIE_De = k_H / k_D if k_D > 0 else float('inf')

    print(f"  V_el fitted = {V_mid:.4f} kcal/mol = {V_el_De_cm:.2f} cm⁻¹")
    print(f"  k_H  = {k_H:.1f} s⁻¹ (target: {K_H_EXP})")
    print(f"  k_D  = {k_D:.2f} s⁻¹ (exp: {K_D_EXP})")
    print(f"  KIE  = {KIE_De:.1f} (exp: {KIE_EXP})")
    print(f"  Ratio to SHS: {V_el_De_cm/1.7:.1f}×")


def soudackov_reference_params():
    """Load Soudackov's exact attenuation parameters (personal communication, 2026-03-26).

    These are computed from DFT-fitted Morse potentials at R_DA = 2.77 Å.
    Convention: α = -d(log S)/dR, γ = -d²(log S)/dR² (NOT log S²).
    Units: atomic units (bohr⁻¹ for α, bohr⁻² for γ).
    """
    # (mu, nu, alpha_H, alpha_D, gamma_H, gamma_D)
    data = [
        (0, 0, 11.034665, 15.965260, 5.699231, 8.055639),
        (0, 1,  9.923012, 14.872464, 6.265313, 8.608031),
        (0, 2,  8.746970, 13.735947, 6.849143, 9.166544),
        (0, 3,  7.485161, 12.546187, 7.509901, 9.756831),
        (1, 0,  9.896376, 14.841890, 6.261668, 8.598394),
        (1, 1,  8.667494, 13.676831, 7.174968, 9.366207),
        (1, 2,  7.337403, 12.454163, 8.239024, 10.187338),
        (1, 3,  5.844407, 11.154763, 9.698933, 11.126413),
        (2, 0,  8.686988, 13.670702, 6.893588, 9.182247),
        (2, 1,  7.291462, 12.413936, 8.327205, 10.228023),
        (2, 2,  5.738962, 11.081937, 10.167525, 11.397388),
        (2, 3,  3.885230,  9.640960, 13.162857, 12.825504),
        (3, 0,  7.386401, 12.443408, 7.656023, 9.832113),
        (3, 1,  5.730623, 11.065462, 10.003734, 11.262684),
        (3, 2,  3.805784,  9.586861, 13.437247, 12.937917),
        (3, 3,  1.216657,  7.947903, 20.864004, 15.145334),
    ]
    n = 4  # 4x4 state pairs
    alpha_H = np.zeros((n, n))
    alpha_D = np.zeros((n, n))
    gamma_H = np.zeros((n, n))
    gamma_D = np.zeros((n, n))
    for mu, nu, aH, aD, gH, gD in data:
        alpha_H[mu, nu] = aH
        alpha_D[mu, nu] = aD
        gamma_H[mu, nu] = gH
        gamma_D[mu, nu] = gD
    return alpha_H, alpha_D, gamma_H, gamma_D


def test_soudackov_reference():
    """Test rate prediction using Soudackov's exact reference parameters.

    Key fix: compute S²(R₀) separately for H and D mass wavefunctions.
    Soudackov's α^(H), γ^(H) correspond to hydrogen overlaps, while
    α^(D), γ^(D) correspond to deuterium overlaps — each isotope has
    its own wavefunction shape and therefore its own S²(R₀).
    """
    from pcet_engine.core.fgh import (
        analytical_gating_rate_table, fgh_franck_condon_table,
        compute_attenuation_params,
    )
    from pcet_engine.core.constants import KB_HARTREE, TWO_PI, AU_RATE_TO_PER_S

    print()
    print("=" * 70)
    print("Method 4: SOUDACKOV REFERENCE PARAMETERS (exact, 2026-03-26)")
    print("=" * 70)
    print("Using his exact α_μν, γ_μν from DFT Morse potentials at R_DA = 2.77 Å")
    print("Computing S²(R₀) separately for H and D mass wavefunctions")
    print()

    MASS_H = 1.00782503207
    MASS_D = 2.01410177812
    OMEGA_H = 2900.0  # cm⁻¹
    OMEGA_D = OMEGA_H * math.sqrt(MASS_H / MASS_D)  # ~2054 cm⁻¹

    alpha_H, alpha_D, gamma_H, gamma_D = soudackov_reference_params()

    delta_0 = SLO_PARAMS["d_DA"] - SLO_PARAMS["r_DH"] - SLO_PARAMS["r_AH"]
    n_states = 4

    # Try multiple D_e values to see sensitivity
    D_e_values = {
        "D_e = 100 kcal/mol (C-H)": 100.0 * KCALMOL_TO_HARTREE,
        "D_e = 110 kcal/mol (O-H)": 110.0 * KCALMOL_TO_HARTREE,
        "D_e = 20ℏω (default)":     None,
    }

    dG_au = SLO_PARAMS["delta_G"] * KCALMOL_TO_HARTREE
    lam_au = SLO_PARAMS["lambda_reorg"] * KCALMOL_TO_HARTREE
    kBT = KB_HARTREE * TEMPERATURE

    def compute_rate_ref(V_el_kcal, S_sq_eff, energies_R, energies_P):
        V_el_au = V_el_kcal * KCALMOL_TO_HARTREE
        E_R = energies_R - energies_R[0]
        boltz = np.exp(-E_R / kBT)
        P_mu = boltz / np.sum(boltz)
        n_R = min(n_states, len(P_mu))
        n_P = min(n_states, len(energies_P))
        total = 0.0
        for mu in range(n_R):
            for nu in range(n_P):
                dG_mn = dG_au + (energies_P[nu] - energies_P[0]) - (energies_R[mu] - energies_R[0])
                S_sq = S_sq_eff[mu, nu]
                if S_sq < 1e-30:
                    continue
                E_a = (dG_mn + lam_au)**2 / (4.0 * lam_au)
                pf = TWO_PI * V_el_au**2 * S_sq / math.sqrt(4 * math.pi * lam_au * kBT)
                rate_au = pf * math.exp(-E_a / kBT) if E_a / kBT < 700 else 0.0
                total += P_mu[mu] * rate_au * AU_RATE_TO_PER_S
        return total

    for D_e_label, D_e in D_e_values.items():
        print(f"  --- {D_e_label} ---")

        # H wavefunctions → S²_H(R₀)
        S_sq_0_H, our_alpha_H, our_gamma_H = compute_attenuation_params(
            OMEGA_H, OMEGA_H, MASS_H, delta_0,
            n_reactant_states=n_states, n_product_states=n_states,
            D_e_hartree=D_e,
        )
        # D wavefunctions → S²_D(R₀)
        S_sq_0_D, our_alpha_D, our_gamma_D = compute_attenuation_params(
            OMEGA_D, OMEGA_D, MASS_D, delta_0,
            n_reactant_states=n_states, n_product_states=n_states,
            D_e_hartree=D_e,
        )

        print(f"    S²_H(0,0) = {S_sq_0_H[0,0]:.3e},  S²_D(0,0) = {S_sq_0_D[0,0]:.3e}")
        print(f"    Our α_H(0,0) = {our_alpha_H[0,0]:.2f},  Soudackov α_H(0,0) = {alpha_H[0,0]:.2f}")
        print(f"    Our α_D(0,0) = {our_alpha_D[0,0]:.2f},  Soudackov α_D(0,0) = {alpha_D[0,0]:.2f}")

        # Apply Soudackov's α,γ to our S²(R₀)
        S_sq_eff_H, _ = analytical_gating_rate_table(
            S_sq_0_H, alpha_H, gamma_H,
            SLO_PARAMS["M_DA"], SLO_PARAMS["Omega_gating"], TEMPERATURE,
        )
        S_sq_eff_D, _ = analytical_gating_rate_table(
            S_sq_0_D, alpha_D, gamma_D,
            SLO_PARAMS["M_DA"], SLO_PARAMS["Omega_gating"], TEMPERATURE,
        )

        # Get energy levels (use H for Boltzmann, same for both isotopes conceptually)
        _, energies_R_H, energies_P_H, _, _ = fgh_franck_condon_table(
            OMEGA_H, OMEGA_H, MASS_H, delta_0,
            n_reactant_states=n_states, n_product_states=n_states,
            D_e_hartree=D_e,
        )
        _, energies_R_D, energies_P_D, _, _ = fgh_franck_condon_table(
            OMEGA_D, OMEGA_D, MASS_D, delta_0,
            n_reactant_states=n_states, n_product_states=n_states,
            D_e_hartree=D_e,
        )

        # Binary search for V_el using H rate
        V_lo, V_hi = 0.001, 20.0
        for _ in range(50):
            V_mid = (V_lo + V_hi) / 2.0
            k_H = compute_rate_ref(V_mid, S_sq_eff_H, energies_R_H, energies_P_H)
            if k_H <= 0:
                V_lo = V_mid
            elif k_H / K_H_EXP > 1.01:
                V_hi = V_mid
            elif k_H / K_H_EXP < 0.99:
                V_lo = V_mid
            else:
                break

        k_D = compute_rate_ref(V_mid, S_sq_eff_D, energies_R_D, energies_P_D)
        KIE = k_H / k_D if k_D > 0 else float('inf')
        V_el_cm = V_mid * KCALMOL_TO_HARTREE * HARTREE_TO_CM

        print(f"    V_el fitted = {V_mid:.4f} kcal/mol = {V_el_cm:.2f} cm⁻¹  ({V_el_cm/1.7:.1f}× SHS)")
        print(f"    k_H = {k_H:.1f}, k_D = {k_D:.2f}, KIE = {KIE:.1f} (exp: {KIE_EXP})")

        # Also at V_el = 1.7 cm⁻¹
        V_shs = 1.7 / HARTREE_TO_CM / KCALMOL_TO_HARTREE
        k_H_17 = compute_rate_ref(V_shs, S_sq_eff_H, energies_R_H, energies_P_H)
        k_D_17 = compute_rate_ref(V_shs, S_sq_eff_D, energies_R_D, energies_P_D)
        KIE_17 = k_H_17 / k_D_17 if k_D_17 > 0 else float('inf')
        print(f"    At V_el=1.7 cm⁻¹: k_H={k_H_17:.2e}, k_D={k_D_17:.2e}, KIE={KIE_17:.1f}")
        print()


if __name__ == "__main__":
    main()
    test_soudackov_reference()
