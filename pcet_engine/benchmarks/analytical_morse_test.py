"""
Test analytical Morse wavefunctions vs FGH for Soudackov's SLO-1 parameters.

Hypothesis: the ~1.5× α discrepancy comes from FGH not being accurate enough
in the classically forbidden (tail) region. Analytical Morse wavefunctions
are exact everywhere.

Morse wavefunction:
    ψ_n(r) = N_n * z^(λ-n-1/2) * exp(-z/2) * L_n^(2λ-2n-1)(z)

where:
    z = 2λ * exp(-β(r - r_eq))
    λ = sqrt(2 m D_e) / (ℏ β)
    L_n^α(z) = generalized Laguerre polynomial
    N_n = normalization constant

For the reflected product Morse (O-H, proton approaches from left):
    z_P = 2λ_P * exp(+β_OH * (r - r_eq_P))
    (sign flip in exponent)

Reference: PMC5217758, Eq. 15 and Ref. 51 therein.
"""

import math
import numpy as np
from scipy.special import genlaguerre, factorial
from pcet_engine.core.constants import (
    ANGSTROM_TO_BOHR, AMU_TO_AU, KCALMOL_TO_HARTREE,
    KB_HARTREE, TWO_PI, AU_RATE_TO_PER_S, HARTREE_TO_CM,
    CM_TO_HARTREE, HBAR_AU,
)

# Soudackov's exact Morse parameters
D_CH_KCAL = 77.0
D_OH_KCAL = 82.0
BETA_CH_AINV = 2.068  # Å⁻¹
BETA_OH_AINV = 2.442  # Å⁻¹
R_CH = 1.09  # Å
R_OH = 0.96  # Å
R_DA = 2.77  # Å

D_CH = D_CH_KCAL * KCALMOL_TO_HARTREE
D_OH = D_OH_KCAL * KCALMOL_TO_HARTREE
BETA_CH = BETA_CH_AINV / ANGSTROM_TO_BOHR  # bohr⁻¹  (divide: Å⁻¹ → bohr⁻¹ means ×Å/bohr)

# Wait — ANGSTROM_TO_BOHR converts Å → bohr (multiply Å value by this to get bohr).
# So β in bohr⁻¹ = β_Å / ANGSTROM_TO_BOHR ... no.
# β has units of inverse length.  β_bohr = β_Å * Å/bohr = β_Å / (bohr/Å) = β_Å / ANGSTROM_TO_BOHR
# 1 bohr = 0.52918 Å → ANGSTROM_TO_BOHR = 1/0.52918 = 1.8897
# β_Å = 2.068 Å⁻¹ → β_bohr = 2.068 / 1.8897 = 1.094 bohr⁻¹
# Check: β * r should be dimensionless. β_bohr * r_bohr = (β_Å/A2B) * (r_Å * A2B) = β_Å * r_Å ✓

BETA_CH = BETA_CH_AINV / ANGSTROM_TO_BOHR  # bohr⁻¹ — WRONG in soudackov_exact.py!
BETA_OH = BETA_OH_AINV / ANGSTROM_TO_BOHR  # bohr⁻¹

# Let me double check: soudackov_exact.py has BETA_CH = BETA_CH_AINV * ANGSTROM_TO_BOHR
# That would give 2.068 * 1.8897 = 3.908 bohr⁻¹
# My version: 2.068 / 1.8897 = 1.094 bohr⁻¹
# Which is correct?
# β has units [1/length]. To convert β from Å⁻¹ to bohr⁻¹:
# 1 Å⁻¹ = (1/Å) = (1/(bohr * 0.52918)) = (1/0.52918) bohr⁻¹ = 1.8897 bohr⁻¹
# So β_bohr = β_Å * 1.8897 = β_Å * ANGSTROM_TO_BOHR ✓
# soudackov_exact.py is CORRECT.  My reasoning above was wrong.

# β has units [1/length]: β_bohr = β_Å / ANGSTROM_TO_BOHR (DIVIDE, not multiply!)
# Verification: β_CH × r_CH must be dimensionless → 2.068 Å⁻¹ × 1.09 Å = 2.254
# With correct conversion: 1.094 bohr⁻¹ × 2.060 bohr = 2.254 ✓
BETA_CH = BETA_CH_AINV / ANGSTROM_TO_BOHR  # bohr⁻¹
BETA_OH = BETA_OH_AINV / ANGSTROM_TO_BOHR  # bohr⁻¹

MASS_H = 1.00782503207
MASS_D = 2.01410177812


def morse_lambda(mass_amu, D_e_hartree, beta_bohr):
    """Compute the Morse parameter λ = sqrt(2mD) / (ℏβ).

    This determines the number of bound states: n_max = floor(λ - 1/2).
    """
    mass_au = mass_amu * AMU_TO_AU
    return math.sqrt(2.0 * mass_au * D_e_hartree) / (HBAR_AU * beta_bohr)


def analytical_morse_wavefunction(grid_bohr, beta_bohr, D_e_hartree, mass_amu,
                                  r_eq_bohr, n_state=0, reflected=False):
    """Compute the analytical Morse oscillator wavefunction on a grid.

    For the standard Morse: V(r) = D[1 - exp(-β(r-r_eq))]²
        z = 2λ exp(-β(r - r_eq))

    For reflected Morse (product well): V(r) = D[1 - exp(+β(r-r_eq))]²
        z = 2λ exp(+β(r - r_eq))

    Returns:
        psi: Normalized wavefunction on grid, shape (N,).
        energy: Energy eigenvalue in hartree (relative to well minimum).
    """
    mass_au = mass_amu * AMU_TO_AU
    lam = morse_lambda(mass_amu, D_e_hartree, beta_bohr)
    n = n_state

    if n >= lam - 0.5:
        raise ValueError(f"State n={n} exceeds max bound state for λ={lam:.2f} "
                         f"(n_max = {int(lam - 0.5)})")

    # Energy eigenvalue
    omega_e = beta_bohr * math.sqrt(2.0 * D_e_hartree / mass_au) * HBAR_AU
    energy = omega_e * (n + 0.5) - omega_e**2 * (n + 0.5)**2 / (4.0 * D_e_hartree)

    # Morse variable z
    if reflected:
        z = 2.0 * lam * np.exp(beta_bohr * (grid_bohr - r_eq_bohr))
    else:
        z = 2.0 * lam * np.exp(-beta_bohr * (grid_bohr - r_eq_bohr))

    # Wavefunction parameters
    alpha_param = 2.0 * lam - 2.0 * n - 1.0  # Laguerre alpha parameter
    k = lam - n - 0.5  # power of z

    # Compute in log space for numerical stability
    # ψ_n(z) ∝ z^k * exp(-z/2) * L_n^α(z)
    L_n = genlaguerre(n, alpha_param)

    # For large z, exp(-z/2) kills everything. Work in log space.
    log_z = np.log(np.maximum(z, 1e-300))
    log_psi_envelope = k * log_z - z / 2.0

    # Evaluate Laguerre polynomial (can be negative, so handle separately)
    L_values = L_n(z)

    # Combine: psi = exp(log_envelope) * L_values
    # But for very large z, exp(log_envelope) → 0, so we need care
    max_log = np.max(log_psi_envelope[np.isfinite(log_psi_envelope)])
    psi = np.exp(log_psi_envelope - max_log) * L_values

    # Normalize: ∫|ψ|² dr = 1
    dx = grid_bohr[1] - grid_bohr[0]
    norm = np.sqrt(np.sum(psi**2) * dx)
    if norm > 1e-30:
        psi /= norm

    return psi, energy


def compute_overlap_analytical(mass_amu, delta_angstrom, n_grid=2048, padding=5.0,
                               mu=0, nu=0):
    """Compute |<ψ_μ^R | ψ_ν^P>|² using analytical Morse wavefunctions.

    Reactant: C-H Morse centered at x=0
    Product: O-H Morse (reflected) centered at x=δ
    """
    delta_bohr = delta_angstrom * ANGSTROM_TO_BOHR

    x_min = -padding * ANGSTROM_TO_BOHR
    x_max = delta_bohr + padding * ANGSTROM_TO_BOHR
    grid = np.linspace(x_min, x_max, n_grid)
    dx = grid[1] - grid[0]

    # Reactant wavefunction: Morse centered at x=0
    psi_R, E_R = analytical_morse_wavefunction(
        grid, BETA_CH, D_CH, mass_amu, r_eq_bohr=0.0,
        n_state=mu, reflected=False,
    )

    # Product wavefunction: reflected Morse centered at x=δ
    psi_P, E_P = analytical_morse_wavefunction(
        grid, BETA_OH, D_OH, mass_amu, r_eq_bohr=delta_bohr,
        n_state=nu, reflected=True,
    )

    overlap = np.sum(psi_R * psi_P) * dx
    return overlap**2


def compute_attenuation_analytical(mass_amu, n_states=4, h_angstrom=0.01):
    """Compute α_μν, γ_μν using analytical Morse wavefunctions."""
    delta_0 = R_DA - R_CH - R_OH

    n_R = n_states
    n_P = n_states
    S_sq_0 = np.zeros((n_R, n_P))
    alpha_mn = np.zeros((n_R, n_P))
    gamma_mn = np.zeros((n_R, n_P))

    h_bohr = h_angstrom * ANGSTROM_TO_BOHR

    # Compute overlaps at δ-h, δ, δ+h
    S_sq_m = np.zeros((n_R, n_P))
    S_sq_p = np.zeros((n_R, n_P))

    for mu in range(n_R):
        for nu in range(n_P):
            try:
                S_sq_m[mu, nu] = compute_overlap_analytical(mass_amu, delta_0 - h_angstrom, mu=mu, nu=nu)
                S_sq_0[mu, nu] = compute_overlap_analytical(mass_amu, delta_0, mu=mu, nu=nu)
                S_sq_p[mu, nu] = compute_overlap_analytical(mass_amu, delta_0 + h_angstrom, mu=mu, nu=nu)
            except ValueError as e:
                print(f"  Warning: ({mu},{nu}) — {e}")
                continue

            if S_sq_0[mu, nu] < 1e-30:
                continue

            # log|S| = 0.5 * log(|S|²)
            log_m = 0.5 * math.log(S_sq_m[mu, nu]) if S_sq_m[mu, nu] > 1e-300 else -350.0
            log_0 = 0.5 * math.log(S_sq_0[mu, nu]) if S_sq_0[mu, nu] > 1e-300 else -350.0
            log_p = 0.5 * math.log(S_sq_p[mu, nu]) if S_sq_p[mu, nu] > 1e-300 else -350.0

            # Note: increasing δ increases R_DA (donor-acceptor distance)
            # α = -d(log S)/dR > 0 (overlap decreases with distance)
            alpha_mn[mu, nu] = -(log_p - log_m) / (2.0 * h_bohr)
            gamma_mn[mu, nu] = -(log_p - 2.0 * log_0 + log_m) / h_bohr**2

    return S_sq_0, alpha_mn, gamma_mn


def main():
    delta_0 = R_DA - R_CH - R_OH
    print("=" * 70)
    print("ANALYTICAL MORSE WAVEFUNCTIONS vs SOUDACKOV REFERENCE")
    print("=" * 70)
    print(f"  D_CH = {D_CH_KCAL} kcal/mol, β_CH = {BETA_CH_AINV} Å⁻¹ ({BETA_CH:.4f} bohr⁻¹)")
    print(f"  D_OH = {D_OH_KCAL} kcal/mol, β_OH = {BETA_OH_AINV} Å⁻¹ ({BETA_OH:.4f} bohr⁻¹)")
    print(f"  R_DA = {R_DA} Å, δ₀ = {delta_0:.2f} Å = {delta_0 * ANGSTROM_TO_BOHR:.3f} bohr")
    print()

    # Morse parameters
    lam_H_CH = morse_lambda(MASS_H, D_CH, BETA_CH)
    lam_H_OH = morse_lambda(MASS_H, D_OH, BETA_OH)
    lam_D_CH = morse_lambda(MASS_D, D_CH, BETA_CH)
    lam_D_OH = morse_lambda(MASS_D, D_OH, BETA_OH)
    print(f"  λ(H, C-H) = {lam_H_CH:.2f} → n_max = {int(lam_H_CH - 0.5)}")
    print(f"  λ(H, O-H) = {lam_H_OH:.2f} → n_max = {int(lam_H_OH - 0.5)}")
    print(f"  λ(D, C-H) = {lam_D_CH:.2f} → n_max = {int(lam_D_CH - 0.5)}")
    print(f"  λ(D, O-H) = {lam_D_OH:.2f} → n_max = {int(lam_D_OH - 0.5)}")

    # Energy levels
    mass_H_au = MASS_H * AMU_TO_AU
    omega_CH = BETA_CH * math.sqrt(2.0 * D_CH / mass_H_au) * HBAR_AU
    omega_OH = BETA_OH * math.sqrt(2.0 * D_OH / mass_H_au) * HBAR_AU
    print(f"  ω_e(C-H, H) = {omega_CH / (CM_TO_HARTREE):.0f} cm⁻¹")
    print(f"  ω_e(O-H, H) = {omega_OH / (CM_TO_HARTREE):.0f} cm⁻¹")
    print()

    # Load Soudackov's reference
    from pcet_engine.benchmarks.soudackov_correction import soudackov_reference_params
    ref_aH, ref_aD, ref_gH, ref_gD = soudackov_reference_params()

    # --- Hydrogen ---
    print("-" * 70)
    print("HYDROGEN: Analytical Morse overlaps and attenuation")
    print("-" * 70)
    S_sq_H, alpha_H, gamma_H = compute_attenuation_analytical(MASS_H)

    print(f"\n  {'(μ,ν)':<8} {'|S²(R₀)|':<12} {'α_H ours':<12} {'α_H Soud':<12} {'ratio':<8} {'γ_H ours':<12} {'γ_H Soud':<12}")
    print(f"  {'-'*76}")
    for mu in range(4):
        for nu in range(4):
            ratio = alpha_H[mu, nu] / ref_aH[mu, nu] if ref_aH[mu, nu] > 0 else 0
            print(f"  ({mu},{nu})    {S_sq_H[mu,nu]:<12.3e} {alpha_H[mu,nu]:<12.4f} {ref_aH[mu,nu]:<12.4f} {ratio:<8.3f} {gamma_H[mu,nu]:<12.4f} {ref_gH[mu,nu]:<12.4f}")

    # --- Deuterium ---
    print()
    print("-" * 70)
    print("DEUTERIUM: Analytical Morse overlaps and attenuation")
    print("-" * 70)
    S_sq_D, alpha_D, gamma_D = compute_attenuation_analytical(MASS_D)

    print(f"\n  {'(μ,ν)':<8} {'|S²(R₀)|':<12} {'α_D ours':<12} {'α_D Soud':<12} {'ratio':<8} {'γ_D ours':<12} {'γ_D Soud':<12}")
    print(f"  {'-'*76}")
    for mu in range(4):
        for nu in range(4):
            ratio = alpha_D[mu, nu] / ref_aD[mu, nu] if ref_aD[mu, nu] > 0 else 0
            print(f"  ({mu},{nu})    {S_sq_D[mu,nu]:<12.3e} {alpha_D[mu,nu]:<12.4f} {ref_aD[mu,nu]:<12.4f} {ratio:<8.3f} {gamma_D[mu,nu]:<12.4f} {ref_gD[mu,nu]:<12.4f}")

    # --- SLO-1 Rate Calculation ---
    print()
    print("=" * 70)
    print("SLO-1 RATE CALCULATION WITH ANALYTICAL MORSE OVERLAPS")
    print("=" * 70)

    from pcet_engine.core.fgh import analytical_gating_rate_table

    DELTA_G = -5.4 * KCALMOL_TO_HARTREE
    LAMBDA = 13.4 * KCALMOL_TO_HARTREE
    M_DA = 100.0      # amu
    OMEGA_GATING = 132.8  # cm⁻¹
    TEMP = 303.0
    K_H_EXP = 297.0
    K_D_EXP = 3.7
    KIE_EXP = 81.0
    kBT = KB_HARTREE * TEMP

    # Energy levels from analytical Morse
    mass_H_au = MASS_H * AMU_TO_AU
    mass_D_au = MASS_D * AMU_TO_AU
    omega_CH_H = BETA_CH * math.sqrt(2.0 * D_CH / mass_H_au)  # a.u.
    omega_OH_H = BETA_OH * math.sqrt(2.0 * D_OH / mass_H_au)
    omega_CH_D = BETA_CH * math.sqrt(2.0 * D_CH / mass_D_au)
    omega_OH_D = BETA_OH * math.sqrt(2.0 * D_OH / mass_D_au)

    def morse_energies(omega_au, D_e, n_states=4):
        """Morse energy levels: E_n = ω(n+1/2) - ω²(n+1/2)²/(4D)"""
        E = np.zeros(n_states)
        for n in range(n_states):
            E[n] = omega_au * (n + 0.5) - omega_au**2 * (n + 0.5)**2 / (4.0 * D_e)
        return E

    E_R_H = morse_energies(omega_CH_H, D_CH)
    E_P_H = morse_energies(omega_OH_H, D_OH)
    E_R_D = morse_energies(omega_CH_D, D_CH)
    E_P_D = morse_energies(omega_OH_D, D_OH)

    # Relative to ground state
    E_R_H -= E_R_H[0]
    E_P_H -= E_P_H[0]
    E_R_D -= E_R_D[0]
    E_P_D -= E_P_D[0]

    print(f"  H reactant levels (cm⁻¹): {[f'{e*HARTREE_TO_CM:.0f}' for e in E_R_H]}")
    print(f"  H product levels (cm⁻¹):  {[f'{e*HARTREE_TO_CM:.0f}' for e in E_P_H]}")
    print(f"  D reactant levels (cm⁻¹): {[f'{e*HARTREE_TO_CM:.0f}' for e in E_R_D]}")

    boltz_H = np.exp(-E_R_H / kBT)
    P_mu_H = boltz_H / np.sum(boltz_H)
    boltz_D = np.exp(-E_R_D / kBT)
    P_mu_D = boltz_D / np.sum(boltz_D)
    print(f"  H Boltzmann: {[f'{p:.4e}' for p in P_mu_H]}")
    print(f"  D Boltzmann: {[f'{p:.4e}' for p in P_mu_D]}")

    # Gating-averaged effective overlaps
    S_sq_eff_H, _ = analytical_gating_rate_table(
        S_sq_H, alpha_H, gamma_H, M_DA, OMEGA_GATING, TEMP,
    )
    S_sq_eff_D, _ = analytical_gating_rate_table(
        S_sq_D, alpha_D, gamma_D, M_DA, OMEGA_GATING, TEMP,
    )

    print(f"\n  S²_eff_H(0,0) = {S_sq_eff_H[0,0]:.3e} (gating boost: {S_sq_eff_H[0,0]/S_sq_H[0,0]:.1f}×)")
    print(f"  S²_eff_D(0,0) = {S_sq_eff_D[0,0]:.3e} (gating boost: {S_sq_eff_D[0,0]/S_sq_D[0,0]:.1f}×)")

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

    print(f"\n  --- Results (analytical Morse, our α,γ) ---")
    print(f"  V_el = {V_mid:.4f} kcal/mol = {V_cm:.2f} cm⁻¹  (SHS: 1.7 cm⁻¹, ratio: {V_cm/1.7:.2f})")
    print(f"  k_H = {k_H:.1f} s⁻¹ (exp: {K_H_EXP})")
    print(f"  k_D = {k_D:.2f} s⁻¹ (exp: {K_D_EXP})")
    print(f"  KIE = {KIE:.1f} (exp: {KIE_EXP})")

    # Also test at V_el = 1.7 cm⁻¹
    V_shs = 1.7 / HARTREE_TO_CM / KCALMOL_TO_HARTREE
    k_H_17 = compute_rate(V_shs, S_sq_eff_H, E_R_H, E_P_H, P_mu_H)
    k_D_17 = compute_rate(V_shs, S_sq_eff_D, E_R_D, E_P_D, P_mu_D)
    KIE_17 = k_H_17 / k_D_17 if k_D_17 > 0 else float('inf')
    print(f"\n  --- At V_el = 1.7 cm⁻¹ (SHS value) ---")
    print(f"  k_H = {k_H_17:.2e} s⁻¹ (exp: {K_H_EXP}, ratio: {k_H_17/K_H_EXP:.3f})")
    print(f"  k_D = {k_D_17:.2e} s⁻¹ (exp: {K_D_EXP})")
    print(f"  KIE = {KIE_17:.1f} (exp: {KIE_EXP})")


if __name__ == "__main__":
    main()
