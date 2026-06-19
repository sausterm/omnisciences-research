"""
SLO-1 mutant series validation with Soudackov's Model 1 gating parameters.

Key idea: all SLO-1 variants (WT, L546A, L754A, DM) share the same chemistry
(C-H abstraction from linoleic acid by Fe(III)-OH). The mutations enlarge the
active-site cavity, increasing d_DA but leaving V_el, ΔG, λ, Morse potentials
essentially unchanged.

Test 1: Fit V_el to WT → predict mutant KIEs from d_DA alone
Test 2: Fit V_el per mutant → check if V_el stays constant (it should)
Test 3: Temperature dependence for WT and DM

Gating parameters: Soudackov's Model 1 (M_DA=100 amu, Ω=132.8 cm⁻¹)
Morse parameters: Soudackov's exact (D_CH=77, D_OH=82, β_CH=2.068, β_OH=2.442)
"""

import math
import numpy as np
from pcet_engine.core.constants import (
    ANGSTROM_TO_BOHR, AMU_TO_AU, KCALMOL_TO_HARTREE,
    KB_HARTREE, TWO_PI, AU_RATE_TO_PER_S, HARTREE_TO_CM,
    CM_TO_HARTREE, HBAR_AU,
)
from pcet_engine.benchmarks.analytical_morse_test import (
    BETA_CH, BETA_OH, D_CH, D_OH, MASS_H, MASS_D,
    compute_overlap_analytical, compute_attenuation_analytical,
    morse_lambda, analytical_morse_wavefunction,
)
from pcet_engine.core.fgh import analytical_gating_rate_table

# ── Soudackov's Model 1 parameters ──────────────────────────────────
M_DA = 100.0        # amu (Soudackov's value, not the published 14 amu)
OMEGA_GATING = 132.8  # cm⁻¹
DELTA_G = -5.4 * KCALMOL_TO_HARTREE
LAMBDA = 13.4 * KCALMOL_TO_HARTREE

# ── SLO-1 mutant series ─────────────────────────────────────────────
# d_DA values from crystal structures / MD (Hu et al. 2014)
MUTANTS = {
    "WT":    {"d_DA": 2.77, "k_H_exp": 297.0,  "k_D_exp": 3.7,    "KIE_exp": 81,  "E_a_exp": 2.1},
    "L546A": {"d_DA": 2.88, "k_H_exp": 8.2,    "k_D_exp": 0.088,  "KIE_exp": 93,  "E_a_exp": 4.1},
    "L754A": {"d_DA": 2.95, "k_H_exp": 3.0,    "k_D_exp": 0.027,  "KIE_exp": 112, "E_a_exp": 4.5},
    "DM":    {"d_DA": 3.10, "k_H_exp": 0.3,    "k_D_exp": 4.5e-4, "KIE_exp": 661, "E_a_exp": 6.0},
}

R_CH = 1.09  # Å
R_OH = 0.96  # Å


def compute_attenuation_at_dDA(mass_amu, d_DA, n_states=4, h_angstrom=0.01):
    """Compute S², α, γ for a given d_DA (donor-acceptor distance)."""
    delta_0 = d_DA - R_CH - R_OH
    h_bohr = h_angstrom * ANGSTROM_TO_BOHR

    S_sq_0 = np.zeros((n_states, n_states))
    alpha_mn = np.zeros((n_states, n_states))
    gamma_mn = np.zeros((n_states, n_states))

    for mu in range(n_states):
        for nu in range(n_states):
            try:
                S_m = compute_overlap_analytical(mass_amu, delta_0 - h_angstrom, mu=mu, nu=nu)
                S_0 = compute_overlap_analytical(mass_amu, delta_0, mu=mu, nu=nu)
                S_p = compute_overlap_analytical(mass_amu, delta_0 + h_angstrom, mu=mu, nu=nu)
            except ValueError:
                continue

            S_sq_0[mu, nu] = S_0
            if S_0 < 1e-30:
                continue

            log_m = 0.5 * math.log(S_m) if S_m > 1e-300 else -350.0
            log_0 = 0.5 * math.log(S_0) if S_0 > 1e-300 else -350.0
            log_p = 0.5 * math.log(S_p) if S_p > 1e-300 else -350.0

            alpha_mn[mu, nu] = -(log_p - log_m) / (2.0 * h_bohr)
            gamma_mn[mu, nu] = -(log_p - 2.0 * log_0 + log_m) / h_bohr**2

    return S_sq_0, alpha_mn, gamma_mn


def compute_rate(V_el_kcal, S_sq_eff, E_R, E_P, P_mu, temp=303.0):
    """Compute PCET rate for given V_el and pre-computed gating-averaged overlaps."""
    kBT = KB_HARTREE * temp
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


def energy_levels(mass_amu, n_states=4):
    """Get asymmetric Morse energy levels for reactant (C-H) and product (O-H)."""
    mass_au = mass_amu * AMU_TO_AU
    omega_CH = BETA_CH * math.sqrt(2.0 * D_CH / mass_au)
    omega_OH = BETA_OH * math.sqrt(2.0 * D_OH / mass_au)

    def morse_E(omega_au, D_e, ns):
        E = np.zeros(ns)
        for n in range(ns):
            E[n] = omega_au * (n + 0.5) - omega_au**2 * (n + 0.5)**2 / (4.0 * D_e)
        return E

    E_R = morse_E(omega_CH, D_CH, n_states)
    E_P = morse_E(omega_OH, D_OH, n_states)
    E_R -= E_R[0]
    E_P -= E_P[0]
    return E_R, E_P


def compute_system(d_DA, temp=303.0, V_el_kcal=None, target_kH=None,
                    omega_gating=None, m_DA=None):
    """Compute rates for a given d_DA. Either provide V_el or fit to target_kH."""
    kBT = KB_HARTREE * temp
    _omega = omega_gating if omega_gating is not None else OMEGA_GATING
    _m_DA = m_DA if m_DA is not None else M_DA

    # Overlaps and attenuation at this d_DA
    S_sq_H, alpha_H, gamma_H = compute_attenuation_at_dDA(MASS_H, d_DA)
    S_sq_D, alpha_D, gamma_D = compute_attenuation_at_dDA(MASS_D, d_DA)

    # Gating average
    S_sq_eff_H, _ = analytical_gating_rate_table(S_sq_H, alpha_H, gamma_H, _m_DA, _omega, temp)
    S_sq_eff_D, _ = analytical_gating_rate_table(S_sq_D, alpha_D, gamma_D, _m_DA, _omega, temp)

    # Energy levels (same for all mutants — same chemistry)
    E_R_H, E_P_H = energy_levels(MASS_H)
    E_R_D, E_P_D = energy_levels(MASS_D)

    # Boltzmann populations
    boltz_H = np.exp(-E_R_H / kBT)
    P_mu_H = boltz_H / np.sum(boltz_H)
    boltz_D = np.exp(-E_R_D / kBT)
    P_mu_D = boltz_D / np.sum(boltz_D)

    # Fit V_el to target if needed
    if target_kH is not None:
        V_lo, V_hi = 1e-6, 100.0
        for _ in range(100):
            V_mid = (V_lo + V_hi) / 2.0
            k = compute_rate(V_mid, S_sq_eff_H, E_R_H, E_P_H, P_mu_H, temp)
            if k <= 0:
                V_lo = V_mid
            elif k / target_kH > 1.001:
                V_hi = V_mid
            elif k / target_kH < 0.999:
                V_lo = V_mid
            else:
                break
        V_el_kcal = V_mid

    k_H = compute_rate(V_el_kcal, S_sq_eff_H, E_R_H, E_P_H, P_mu_H, temp)
    k_D = compute_rate(V_el_kcal, S_sq_eff_D, E_R_D, E_P_D, P_mu_D, temp)
    KIE = k_H / k_D if k_D > 0 else float('inf')
    V_cm = V_el_kcal * KCALMOL_TO_HARTREE * HARTREE_TO_CM

    return {
        "V_el_kcal": V_el_kcal, "V_el_cm": V_cm,
        "k_H": k_H, "k_D": k_D, "KIE": KIE,
        "S_sq_H_00": S_sq_H[0, 0], "S_sq_eff_H_00": S_sq_eff_H[0, 0],
        "alpha_H_00": alpha_H[0, 0], "gamma_H_00": gamma_H[0, 0],
        "gating_boost_H": S_sq_eff_H[0, 0] / S_sq_H[0, 0] if S_sq_H[0, 0] > 0 else 0,
        "gating_boost_D": S_sq_eff_D[0, 0] / S_sq_D[0, 0] if S_sq_D[0, 0] > 0 else 0,
    }


def test_1_predict_from_wt():
    """Fit V_el to WT, predict mutant KIEs."""
    print("=" * 80)
    print("TEST 1: Fit V_el to WT, predict mutant KIEs (d_DA only changes)")
    print("=" * 80)
    print(f"  Gating: M_DA = {M_DA} amu, Ω = {OMEGA_GATING} cm⁻¹ (Soudackov Model 1)")
    print()

    # Fit to WT
    wt = compute_system(MUTANTS["WT"]["d_DA"], target_kH=MUTANTS["WT"]["k_H_exp"])
    V_el_wt = wt["V_el_kcal"]
    print(f"  WT fit: V_el = {V_el_wt:.4f} kcal/mol = {wt['V_el_cm']:.2f} cm⁻¹")
    print()

    # Predict mutants with same V_el
    print(f"  {'Variant':<10} {'d_DA':<7} {'δ₀(Å)':<8} {'k_H pred':<12} {'k_H exp':<12} "
          f"{'KIE pred':<10} {'KIE exp':<10} {'gating_H':<10} {'gating_D':<10}")
    print(f"  {'-'*97}")

    for name, params in MUTANTS.items():
        d_DA = params["d_DA"]
        delta_0 = d_DA - R_CH - R_OH
        res = compute_system(d_DA, V_el_kcal=V_el_wt)
        print(f"  {name:<10} {d_DA:<7.2f} {delta_0:<8.2f} {res['k_H']:<12.2e} {params['k_H_exp']:<12.2e} "
              f"{res['KIE']:<10.1f} {params['KIE_exp']:<10.0f} {res['gating_boost_H']:<10.1f} {res['gating_boost_D']:<10.1f}")


def test_2_fit_per_mutant():
    """Fit V_el independently for each mutant."""
    print()
    print("=" * 80)
    print("TEST 2: Fit V_el per mutant (should be ~constant if model is correct)")
    print("=" * 80)
    print()

    print(f"  {'Variant':<10} {'d_DA':<7} {'V_el (cm⁻¹)':<14} {'V_el (kcal)':<12} "
          f"{'KIE pred':<10} {'KIE exp':<10} {'k_H pred':<12} {'k_H exp':<12}")
    print(f"  {'-'*95}")

    for name, params in MUTANTS.items():
        res = compute_system(params["d_DA"], target_kH=params["k_H_exp"])
        print(f"  {name:<10} {params['d_DA']:<7.2f} {res['V_el_cm']:<14.2f} {res['V_el_kcal']:<12.4f} "
              f"{res['KIE']:<10.1f} {params['KIE_exp']:<10.0f} {res['k_H']:<12.2e} {params['k_H_exp']:<12.2e}")


def test_3_temperature_dependence():
    """Temperature dependence of KIE for WT and DM."""
    print()
    print("=" * 80)
    print("TEST 3: Temperature dependence (WT and DM)")
    print("=" * 80)

    temps = np.arange(278, 323, 5.0)

    for variant in ["WT", "DM"]:
        params = MUTANTS[variant]
        # Fit V_el at 303 K
        res_303 = compute_system(params["d_DA"], temp=303.0, target_kH=params["k_H_exp"])
        V_el = res_303["V_el_kcal"]

        print(f"\n  {variant}: V_el = {res_303['V_el_cm']:.2f} cm⁻¹ (fitted at 303 K)")
        print(f"  {'T (K)':<8} {'k_H':<12} {'k_D':<12} {'KIE':<10} {'E_a (kcal)':<12}")
        print(f"  {'-'*54}")

        results = []
        for T in temps:
            res = compute_system(params["d_DA"], temp=T, V_el_kcal=V_el)
            results.append(res)
            print(f"  {T:<8.0f} {res['k_H']:<12.2e} {res['k_D']:<12.2e} {res['KIE']:<10.1f}")

        # Arrhenius fit for E_a
        if len(results) >= 5:
            ln_k = [math.log(r["k_H"]) if r["k_H"] > 0 else -999 for r in results]
            inv_T = [1.0 / T for T in temps]
            valid = [(x, y) for x, y in zip(inv_T, ln_k) if y > -900]
            if len(valid) >= 3:
                x, y = zip(*valid)
                p = np.polyfit(x, y, 1)
                E_a = -p[0] * KB_HARTREE * HARTREE_TO_CM * CM_TO_HARTREE / KCALMOL_TO_HARTREE
                # Actually: E_a = -slope * k_B in kcal/mol
                # slope has units of K, E_a = -slope * R (gas constant)
                R_kcal = 1.987e-3  # kcal/(mol·K)
                E_a = -p[0] * R_kcal
                print(f"  → E_a(H) = {E_a:.2f} kcal/mol (exp: {params['E_a_exp']})")


def test_4_overlap_vs_dDA():
    """Show how S²(0,0) and gating boost change across the mutant series."""
    print()
    print("=" * 80)
    print("TEST 4: Overlap and gating evolution across mutant series")
    print("=" * 80)
    print()

    print(f"  {'Variant':<10} {'d_DA':<7} {'δ₀':<7} {'S²_H(R₀)':<12} {'S²_D(R₀)':<12} "
          f"{'S²_H/S²_D':<10} {'α_H(0,0)':<10} {'α_D(0,0)':<10}")
    print(f"  {'-'*78}")

    for name, params in MUTANTS.items():
        d_DA = params["d_DA"]
        delta_0 = d_DA - R_CH - R_OH
        S_sq_H, alpha_H, gamma_H = compute_attenuation_at_dDA(MASS_H, d_DA)
        S_sq_D, alpha_D, gamma_D = compute_attenuation_at_dDA(MASS_D, d_DA)
        ratio = S_sq_H[0, 0] / S_sq_D[0, 0] if S_sq_D[0, 0] > 0 else float('inf')
        print(f"  {name:<10} {d_DA:<7.2f} {delta_0:<7.2f} {S_sq_H[0,0]:<12.3e} {S_sq_D[0,0]:<12.3e} "
              f"{ratio:<10.1f} {alpha_H[0,0]:<10.4f} {alpha_D[0,0]:<10.4f}")


def test_5_varying_omega():
    """Use mutant-specific Ω (from Hu et al.) with Soudackov's M_DA=100."""
    print()
    print("=" * 80)
    print("TEST 5: Mutant-specific Ω (Hu et al.) with M_DA = 100 amu")
    print("=" * 80)
    print()
    print("  Hu et al. found the gating frequency softens with each mutation.")
    print("  Softer Ω → more sampling of shorter d_DA → higher rate & lower KIE.")
    print()

    # Ω from published_gating_params.json (Hu et al. 2014)
    omega_per_mutant = {"WT": 150.0, "L546A": 128.0, "L754A": 117.0, "DM": 96.0}

    # First fit V_el to WT with its own Ω
    wt = compute_system(MUTANTS["WT"]["d_DA"], target_kH=MUTANTS["WT"]["k_H_exp"],
                         omega_gating=omega_per_mutant["WT"])
    V_el_wt = wt["V_el_kcal"]
    print(f"  WT fit: V_el = {V_el_wt:.4f} kcal/mol = {wt['V_el_cm']:.2f} cm⁻¹  (Ω=150 cm⁻¹)")
    print()

    print(f"  {'Variant':<10} {'d_DA':<7} {'Ω':<6} {'k_H pred':<12} {'k_H exp':<12} "
          f"{'KIE pred':<10} {'KIE exp':<10} {'gating_H':<10}")
    print(f"  {'-'*87}")

    for name, params in MUTANTS.items():
        omega = omega_per_mutant[name]
        res = compute_system(params["d_DA"], V_el_kcal=V_el_wt, omega_gating=omega)
        print(f"  {name:<10} {params['d_DA']:<7.2f} {omega:<6.0f} {res['k_H']:<12.2e} {params['k_H_exp']:<12.2e} "
              f"{res['KIE']:<10.1f} {params['KIE_exp']:<10.0f} {res['gating_boost_H']:<10.1f}")

    # Also fit V_el per mutant with their own Ω
    print()
    print(f"  Per-mutant V_el fit:")
    print(f"  {'Variant':<10} {'d_DA':<7} {'Ω':<6} {'V_el (cm⁻¹)':<14} {'KIE pred':<10} {'KIE exp':<10}")
    print(f"  {'-'*57}")
    for name, params in MUTANTS.items():
        omega = omega_per_mutant[name]
        res = compute_system(params["d_DA"], target_kH=params["k_H_exp"], omega_gating=omega)
        print(f"  {name:<10} {params['d_DA']:<7.2f} {omega:<6.0f} {res['V_el_cm']:<14.2f} {res['KIE']:<10.1f} {params['KIE_exp']:<10.0f}")


def test_6_fit_omega_and_dDA():
    """For each mutant, find the Ω that gives the right KIE (with V_el from WT)."""
    print()
    print("=" * 80)
    print("TEST 6: What Ω would reproduce each mutant's KIE? (V_el fixed from WT)")
    print("=" * 80)
    print()

    # Fit V_el to WT with Soudackov params
    wt = compute_system(MUTANTS["WT"]["d_DA"], target_kH=MUTANTS["WT"]["k_H_exp"])
    V_el_wt = wt["V_el_kcal"]

    print(f"  V_el = {wt['V_el_cm']:.2f} cm⁻¹ (from WT fit)")
    print()
    print(f"  {'Variant':<10} {'d_DA':<7} {'Ω needed':<10} {'Ω published':<12} {'KIE pred':<10} {'KIE exp':<10} {'k_H pred':<12} {'k_H exp':<12}")
    print(f"  {'-'*93}")

    for name, params in MUTANTS.items():
        target_KIE = params["KIE_exp"]
        # Binary search for Ω
        omega_lo, omega_hi = 50.0, 500.0
        best_omega = OMEGA_GATING
        for _ in range(60):
            omega_mid = (omega_lo + omega_hi) / 2.0
            res = compute_system(params["d_DA"], V_el_kcal=V_el_wt, omega_gating=omega_mid)
            if res["KIE"] > target_KIE * 1.01:
                omega_hi = omega_mid  # softer gating → lower KIE? No, higher Ω → stiffer → less compensation → higher KIE
                # Actually: higher Ω → stiffer → less gating → KIE closer to static value (which is too high)
                # Lower Ω → softer → more gating → KIE drops
                omega_lo = omega_mid
            elif res["KIE"] < target_KIE * 0.99:
                omega_hi = omega_mid
            else:
                best_omega = omega_mid
                break
        else:
            best_omega = omega_mid

        res = compute_system(params["d_DA"], V_el_kcal=V_el_wt, omega_gating=best_omega)
        omega_pub = {"WT": 150, "L546A": 128, "L754A": 117, "DM": 96}.get(name, 0)
        print(f"  {name:<10} {params['d_DA']:<7.2f} {best_omega:<10.1f} {omega_pub:<12.0f} "
              f"{res['KIE']:<10.1f} {target_KIE:<10.0f} {res['k_H']:<12.2e} {params['k_H_exp']:<12.2e}")


def main():
    test_1_predict_from_wt()
    test_2_fit_per_mutant()
    test_4_overlap_vs_dDA()
    test_5_varying_omega()
    test_6_fit_omega_and_dDA()
    test_3_temperature_dependence()


if __name__ == "__main__":
    main()
