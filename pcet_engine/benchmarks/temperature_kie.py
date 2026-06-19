"""Temperature-dependent KIE for SLO-1 wild type (278K-318K).

Computes k_H, k_D, and KIE across a temperature range to test whether the
model reproduces the key experimental signature of SLO-1: nearly
temperature-INDEPENDENT KIE, indicating tunneling-dominated PCET rather
than classical over-the-barrier transfer.

Experimental data (Knapp et al. JACS 124, 3865, 2002):
    - E_a(H) ~ 2.1 kcal/mol (weak T-dependence)
    - KIE ~ 81 across 278-318K, roughly constant
    - Both k_H and k_D have small Arrhenius slopes

Two model approaches:
    1. Without gating: delta_0=0.50, V_el=0.6, lambda=19.0, delta_G=-5.4,
       omega_H=2900, d_DA=2.69 (our default benchmark params)
    2. With gating (SHS Model 1): V_el=11.17 cm^-1 (converted), lambda=13.4,
       delta_G=-5.4, d_DA=2.77, Omega_gating=132.8, M_DA=100

Usage:
    python3 -m pcet_engine.benchmarks.temperature_kie
"""

import numpy as np

from pcet_engine.core.rate_engine import PCETRateEngine
from pcet_engine.core.constants import (
    CM_TO_HARTREE,
    HARTREE_TO_KCALMOL,
    KB_HARTREE,
)


# ── Experimental reference ──────────────────────────────────────────
E_A_EXP = 2.1  # kcal/mol
KIE_EXP = 81.0

# ── Temperature grid ────────────────────────────────────────────────
TEMPS = np.arange(278, 319, 5, dtype=float)  # 278, 283, ..., 318 K

# ── Model parameters ────────────────────────────────────────────────
# Approach 1: no gating (default benchmark)
PARAMS_NO_GATING = dict(
    V_el=0.6,             # kcal/mol
    delta_G=-5.4,         # kcal/mol
    lambda_reorg=19.0,    # kcal/mol
    omega_H=2900.0,       # cm^-1
    d_DA=2.69,            # angstrom
    delta_0=0.50,         # angstrom
    Omega_gating=0.0,
    M_DA=0.0,
)

# Approach 2: with gating (SHS Model 1, PMC5217758)
V_EL_CM = 11.17  # cm^-1
V_EL_GATING = V_EL_CM * CM_TO_HARTREE * HARTREE_TO_KCALMOL  # kcal/mol

PARAMS_GATING = dict(
    V_el=V_EL_GATING,
    delta_G=-5.4,         # kcal/mol
    lambda_reorg=13.4,    # kcal/mol
    omega_H=2900.0,       # cm^-1
    d_DA=2.77,            # angstrom
    Omega_gating=132.8,   # cm^-1
    M_DA=100.0,           # amu
)


def arrhenius_ea(temps, rates):
    """Compute Arrhenius activation energy from linear fit of ln(k) vs 1/T.

    Returns E_a in kcal/mol.
    """
    inv_T = 1.0 / temps
    ln_k = np.log(rates)
    # ln(k) = ln(A) - E_a / (R * T)  =>  slope = -E_a / R
    # R = KB_HARTREE * HARTREE_TO_KCALMOL * Avogadro... but simpler:
    # R = 1.987e-3 kcal/(mol*K)
    slope, intercept = np.polyfit(inv_T, ln_k, 1)
    R_kcal = 1.9872036e-3  # kcal/(mol*K)
    E_a = -slope * R_kcal
    return E_a


def kie_slope(temps, kies):
    """Slope of KIE vs T (linear fit). Near-zero = T-independent."""
    slope, _ = np.polyfit(temps, kies, 1)
    return slope


def run_temperature_scan(label, params):
    """Run temperature scan and print results."""
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    # Print parameters
    print(f"\n  Parameters:")
    for k, v in params.items():
        print(f"    {k:20s} = {v}")

    k_H_arr = []
    k_D_arr = []
    kie_arr = []

    print(f"\n  {'T (K)':>8s}  {'k_H (s^-1)':>14s}  {'k_D (s^-1)':>14s}  {'KIE':>10s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}")

    for T in TEMPS:
        engine = PCETRateEngine(temperature=T)
        result = engine.compute_rate(**params)
        k_H_arr.append(result.k_H)
        k_D_arr.append(result.k_D)
        kie_arr.append(result.KIE)
        print(f"  {T:8.1f}  {result.k_H:14.4e}  {result.k_D:14.4e}  {result.KIE:10.1f}")

    k_H_arr = np.array(k_H_arr)
    k_D_arr = np.array(k_D_arr)
    kie_arr = np.array(kie_arr)

    # Arrhenius E_a
    E_a_H = arrhenius_ea(TEMPS, k_H_arr)
    E_a_D = arrhenius_ea(TEMPS, k_D_arr)

    # KIE temperature dependence
    dKIE_dT = kie_slope(TEMPS, kie_arr)
    kie_mean = np.mean(kie_arr)
    kie_std = np.std(kie_arr)
    kie_range = np.max(kie_arr) - np.min(kie_arr)

    print(f"\n  Arrhenius analysis:")
    print(f"    E_a(H)  = {E_a_H:.2f} kcal/mol  (exp: {E_A_EXP:.1f} kcal/mol)")
    print(f"    E_a(D)  = {E_a_D:.2f} kcal/mol")
    print(f"    E_a(D) - E_a(H) = {E_a_D - E_a_H:.2f} kcal/mol")

    print(f"\n  KIE temperature dependence:")
    print(f"    KIE mean   = {kie_mean:.1f}  (exp: {KIE_EXP:.0f})")
    print(f"    KIE std    = {kie_std:.1f}")
    print(f"    KIE range  = {kie_range:.1f}  (max - min)")
    print(f"    dKIE/dT    = {dKIE_dT:.3f} per K")

    t_indep = abs(dKIE_dT) < 0.5  # threshold: less than 0.5 per K
    if t_indep:
        print(f"    --> KIE is approximately T-INDEPENDENT (|dKIE/dT| < 0.5)")
    else:
        print(f"    --> KIE is T-DEPENDENT (|dKIE/dT| = {abs(dKIE_dT):.2f} >= 0.5)")

    print(f"\n  Comparison to experiment:")
    ratio_ea = E_a_H / E_A_EXP if E_A_EXP > 0 else float("inf")
    print(f"    E_a(H) / E_a(exp) = {ratio_ea:.2f}")
    print(f"    KIE(298K) / KIE(exp) = {kie_arr[TEMPS.tolist().index(298.0) if 298.0 in TEMPS else len(TEMPS)//2] / KIE_EXP:.2f}" if len(kie_arr) > 0 else "")


def main():
    print("Temperature-Dependent KIE for SLO-1 Wild Type")
    print("Experimental: E_a(H) = 2.1 kcal/mol, KIE ~ 81 (T-independent)")
    print(f"Temperature range: {TEMPS[0]:.0f} - {TEMPS[-1]:.0f} K, step 5 K")

    # Approach 1: without gating
    run_temperature_scan(
        "Approach 1: Without Gating (default benchmark params)",
        PARAMS_NO_GATING,
    )

    # Approach 2: with gating
    run_temperature_scan(
        "Approach 2: With Gating (SHS Model 1)",
        PARAMS_GATING,
    )

    print(f"\n{'='*72}")
    print("  Summary")
    print(f"{'='*72}")
    print("""
  The key experimental signature of SLO-1 is temperature-independent KIE,
  which distinguishes tunneling-dominated PCET from classical over-the-barrier
  transfer. In the Arrhenius plot of ln(k_H) and ln(k_D) vs 1/T:
    - Both k_H and k_D have weak T-dependence (small E_a)
    - The slopes are nearly parallel => KIE ~ constant
    - E_a(H) ~ 2.1 kcal/mol experimentally

  The gating model (Approach 2) incorporates donor-acceptor distance
  fluctuations via a harmonic gating coordinate, which allows shorter
  tunneling distances to be sampled thermally. This can modify both the
  magnitude and temperature dependence of the KIE.
""")


if __name__ == "__main__":
    main()
