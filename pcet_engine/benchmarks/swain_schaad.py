"""Swain-Schaad Exponent Analysis for the 28-System PCET Benchmark.

Computes the Swain-Schaad exponent

    ρ = ln(k_H/k_T) / ln(k_H/k_D)

for all 28 benchmark systems using the calibrated δ₀ values.  The
semiclassical limit is ρ = 3.34 (exact for a symmetric harmonic barrier);
tunneling inflates ρ above this value.

Usage:
    cd pcet_engine/
    python -m pcet_engine.benchmarks.swain_schaad

References:
    Swain, C. G.; Stivers, E. C.; Reuwer, J. F.; Schaad, L. J.
        J. Am. Chem. Soc. 80, 5885 (1958).
    Kohen, A.; Klinman, J. P. Acc. Chem. Res. 31, 397 (1998).
    Scrutton, N. S.; Hay, S. Nat. Chem. 4, 161 (2012).
"""

import math
import sys

from pcet_engine.core.constants import (
    CM_TO_HARTREE,
    KCALMOL_TO_HARTREE,
    PROTON_MASS_AMU,
    DEUTERIUM_MASS_AMU,
    TRITIUM_MASS_AMU,
)
from pcet_engine.core.vibronic import multi_channel_rate
from pcet_engine.benchmarks.systems import BENCHMARK_SYSTEMS

# Semiclassical Swain-Schaad exponent (infinite-parabola limit)
RHO_SEMICLASSICAL = 3.344


def _rate_for_isotope(sys, mass_amu: float, temperature: float = 298.15) -> float:
    """Compute vibronic rate for a given transferring-particle mass.

    The proton/deuteron/triton frequency scales as 1/sqrt(mass), so:
        omega_X = omega_H * sqrt(m_H / m_X)

    The same scaling applies to the product-side frequency (harmonic
    approximation, same force constant, different mass).

    Args:
        sys: BenchmarkSystem instance.
        mass_amu: Transferring-particle mass in amu.
        temperature: Temperature in K.

    Returns:
        Total rate constant in s⁻¹.
    """
    freq_scale = math.sqrt(PROTON_MASS_AMU / mass_amu)

    omega_R_au = sys.omega_H * freq_scale * CM_TO_HARTREE
    # Product frequency: same convention as rate_engine (omega_P = omega_H by default)
    omega_P_au = sys.omega_H * freq_scale * CM_TO_HARTREE

    V_au = sys.V_el * KCALMOL_TO_HARTREE
    dG_au = sys.delta_G * KCALMOL_TO_HARTREE
    lam_au = sys.lambda_reorg * KCALMOL_TO_HARTREE

    # Gating frequency in au (0 → no gating)
    Omega_au = sys.Omega_gating * CM_TO_HARTREE if sys.Omega_gating > 0 else 0.0

    result = multi_channel_rate(
        V_au, dG_au, lam_au,
        omega_R_au, omega_P_au, mass_amu,
        sys.d_DA, temperature,
        Omega_gating=Omega_au,
        M_DA=sys.M_DA,
        r_DH=sys.r_DH,
        r_AH=sys.r_AH,
        delta_0=sys.delta_0,
    )
    return result.rate_total


def compute_swain_schaad(temperature: float = 298.15) -> list[dict]:
    """Compute Swain-Schaad exponents for all benchmark systems.

    Args:
        temperature: Temperature in K.

    Returns:
        List of dicts with keys: name, KIE_HD, KIE_HT, rho, rho_excess,
        delta_0, KIE_exp, category.
    """
    results = []
    for name, sys in BENCHMARK_SYSTEMS.items():
        k_H = _rate_for_isotope(sys, PROTON_MASS_AMU, temperature)
        k_D = _rate_for_isotope(sys, DEUTERIUM_MASS_AMU, temperature)
        k_T = _rate_for_isotope(sys, TRITIUM_MASS_AMU, temperature)

        if k_D <= 0 or k_T <= 0 or k_H <= k_D or k_H <= k_T:
            rho = float("nan")
        else:
            rho = math.log(k_H / k_T) / math.log(k_H / k_D)

        results.append({
            "name": name,
            "k_H": k_H,
            "k_D": k_D,
            "k_T": k_T,
            "KIE_HD": k_H / k_D if k_D > 0 else float("nan"),
            "KIE_HT": k_H / k_T if k_T > 0 else float("nan"),
            "rho": rho,
            "rho_excess": rho - RHO_SEMICLASSICAL if math.isfinite(rho) else float("nan"),
            "delta_0": sys.delta_0,
            "KIE_exp": sys.KIE_exp,
        })
    return results


def print_table(results: list[dict]) -> None:
    """Print formatted Swain-Schaad results table."""
    hdr = (
        f"{'System':<18} {'δ₀(Å)':>6} {'KIE_exp':>8} "
        f"{'KIE_HD':>8} {'KIE_HT':>8} {'ρ':>6} {'ρ−3.34':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        kie_hd = r["KIE_HD"]
        kie_ht = r["KIE_HT"]
        rho = r["rho"]
        excess = r["rho_excess"]

        delta_str = f"{r['delta_0']:.3f}" if r["delta_0"] is not None else "  —  "
        kie_hd_str = f"{kie_hd:8.1f}" if math.isfinite(kie_hd) else "    —   "
        kie_ht_str = f"{kie_ht:8.1f}" if math.isfinite(kie_ht) else "    —   "
        rho_str = f"{rho:6.2f}" if math.isfinite(rho) else "   —  "
        excess_str = f"{excess:+7.2f}" if math.isfinite(excess) else "     —  "

        print(
            f"{r['name']:<18} {delta_str:>6} {r['KIE_exp']:>8.1f} "
            f"{kie_hd_str} {kie_ht_str} {rho_str} {excess_str}"
        )

    print()
    rhos = [r["rho"] for r in results if math.isfinite(r["rho"])]
    if rhos:
        print(f"Semiclassical limit: ρ = {RHO_SEMICLASSICAL:.3f}")
        print(f"Computed range:      ρ = {min(rhos):.2f} – {max(rhos):.2f}")
        n_tunneling = sum(1 for r in rhos if r > RHO_SEMICLASSICAL + 0.5)
        print(f"Systems with ρ > {RHO_SEMICLASSICAL + 0.5:.2f} (tunneling-inflated): {n_tunneling}/{len(rhos)}")

        # Correlation of ρ with calibrated KIE_HD
        import numpy as np
        log_kie = [math.log(r["KIE_HD"]) for r in results if math.isfinite(r["rho"]) and r["KIE_HD"] > 1]
        rho_vals = [r["rho"] for r in results if math.isfinite(r["rho"]) and r["KIE_HD"] > 1]
        if len(log_kie) > 3:
            r_corr = float(np.corrcoef(log_kie, rho_vals)[0, 1])
            print(f"Correlation ln(KIE_HD) vs ρ: r = {r_corr:.3f}")


def main():
    print("Swain-Schaad Exponent Analysis — PCET Benchmark (T = 298.15 K)")
    print(f"Semiclassical limit: ρ = {RHO_SEMICLASSICAL:.3f}")
    print()

    results = compute_swain_schaad(temperature=298.15)
    print_table(results)

    # Flag anomalous systems (rho < 3 or rho > 10)
    anomalous = [r for r in results if math.isfinite(r["rho"]) and (r["rho"] < 3.0 or r["rho"] > 10.0)]
    if anomalous:
        print("\nAnomalous ρ values (outside 3.0–10.0):")
        for r in anomalous:
            print(f"  {r['name']}: ρ = {r['rho']:.2f}  (δ₀ = {r['delta_0']:.3f} Å, KIE_HD = {r['KIE_HD']:.1f})")

    return results


if __name__ == "__main__":
    main()
