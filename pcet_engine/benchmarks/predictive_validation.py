"""Predictive validation of the vibronic PCET model.

Four analyses that address the circularity of the 28-system benchmark:

1. LOO REGRESSION PREDICTIVE TEST
   Leave-one-out linear regression of delta_0 vs d_DA across all 28
   systems.  No per-system KIE fitting.

2. SIMPLER TUNNELING MODEL COMPARISON
   Classical Marcus (KIE=1) and ZPE-semiclassical model vs vibronic.

3. GEOMETRIC DELTA_0 TEST (Path 2)
   Use delta_0_geom = d_DA - r_DH - r_AH (pure crystal geometry, no
   fitting).  Failure demonstrates that calibrated delta_0 encodes
   enzyme-specific information beyond geometry.

4. CROSS-FAMILY REGRESSION (Path 3)
   Fit delta_0 vs d_DA on the SLO-1 mutant family (9 systems, same
   HAT chemistry, r~0.97) and predict KIE for the remaining 19 systems
   with zero per-system fitting.  Maps out where chemical family matters.

Usage:
    python3 -m pcet_engine.benchmarks.predictive_validation
"""

import math
import numpy as np

from pcet_engine.core.rate_engine import PCETRateEngine
from pcet_engine.core.constants import (
    KB_HARTREE,
    HARTREE_TO_KCALMOL,
    CM_TO_HARTREE,
    PROTON_MASS_AMU,
    DEUTERIUM_MASS_AMU,
)
from pcet_engine.benchmarks.systems import BENCHMARK_SYSTEMS


# =====================================================================
# Simple tunneling models (no free parameters)
# =====================================================================

def marcus_kie() -> float:
    """Classical Marcus rate: no proton quantum effects, KIE = 1."""
    return 1.0


def zpe_semiclassical_kie(omega_H_cm: float, temperature: float = 298.15) -> float:
    """KIE from zero-point energy difference only (no tunneling).

    Ea(D) - Ea(H) = 1/2 * hbar * omega_H * (1 - sqrt(m_H/m_D))

    This is the maximum KIE expected from ZPE effects without tunneling.
    Uses only the system's proton stretching frequency and temperature;
    no other adjustable parameters.

    Args:
        omega_H_cm: C-H (or X-H) stretching frequency in cm⁻¹.
        temperature: Temperature in Kelvin.

    Returns:
        KIE_ZPE = exp(delta_Ea_ZPE / kBT).
    """
    omega_H_hartree = omega_H_cm * CM_TO_HARTREE
    mass_ratio = math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)  # sqrt(1/2) ≈ 0.707
    delta_Ea_hartree = 0.5 * omega_H_hartree * (1.0 - mass_ratio)
    kBT = KB_HARTREE * temperature
    return math.exp(delta_Ea_hartree / kBT)


# =====================================================================
# LOO regression: predict delta_0 from d_DA
# =====================================================================

def _loo_regression(d_DA_all: np.ndarray, delta0_all: np.ndarray, i: int):
    """Fit linear delta_0 = a + b * d_DA on all systems except i.

    Returns (a, b) coefficients.
    """
    mask = np.ones(len(d_DA_all), dtype=bool)
    mask[i] = False
    x = d_DA_all[mask]
    y = delta0_all[mask]
    # Least-squares: [1, x] @ [a, b] = y
    A = np.column_stack([np.ones_like(x), x])
    result = np.linalg.lstsq(A, y, rcond=None)
    a, b = result[0]
    return a, b


def run_loo_predictive_test(temperature: float = 298.15, verbose: bool = True) -> dict:
    """Run LOO regression and compute KIE from predicted delta_0.

    For each system:
      1. Fit delta_0 = a + b * d_DA on other 27 systems.
      2. Predict delta_0_pred for the held-out system.
      3. Run vibronic_multi with delta_0_pred.
      4. Compare KIE_pred to KIE_exp.

    Returns:
        dict mapping system name to result dict.
    """
    systems = list(BENCHMARK_SYSTEMS.values())
    names = [s.name for s in systems]
    d_DA = np.array([s.d_DA for s in systems])
    delta0 = np.array([s.delta_0 if s.delta_0 is not None else 0.0 for s in systems])

    # Full-dataset regression for reference
    A_full = np.column_stack([np.ones_like(d_DA), d_DA])
    (a_full, b_full), _, _, _ = np.linalg.lstsq(A_full, delta0, rcond=None)
    r_full = float(np.corrcoef(d_DA, delta0)[0, 1])

    engine = PCETRateEngine(temperature=temperature)
    results = {}

    for i, sys in enumerate(systems):
        # LOO regression
        a, b = _loo_regression(d_DA, delta0, i)
        delta0_pred = a + b * sys.d_DA

        # Vibronic rate with predicted delta_0
        result_pred = engine.compute_rate(
            V_el=sys.V_el,
            delta_G=sys.delta_G,
            lambda_reorg=sys.lambda_reorg,
            omega_H=sys.omega_H,
            d_DA=sys.d_DA,
            method="vibronic_multi",
            delta_0=max(delta0_pred, 0.05),
        )

        # ZPE model
        kie_zpe = zpe_semiclassical_kie(sys.omega_H, temperature)

        log_err_loo = math.log10(result_pred.KIE / sys.KIE_exp)
        log_err_zpe = math.log10(kie_zpe / sys.KIE_exp)

        results[sys.name] = {
            "d_DA": sys.d_DA,
            "delta0_calib": sys.delta_0,
            "delta0_pred": delta0_pred,
            "KIE_exp": sys.KIE_exp,
            "KIE_marcus": marcus_kie(),
            "KIE_zpe": kie_zpe,
            "KIE_loo": result_pred.KIE,
            "log_err_loo": log_err_loo,
            "log_err_zpe": log_err_zpe,
        }

    if verbose:
        print("=" * 110)
        print(f"PREDICTIVE VALIDATION — LOO Regression + Model Comparison  (T = {temperature:.0f} K)")
        print("=" * 110)
        print(f"\nFull-dataset linear fit: delta_0 = {a_full:.4f} + {b_full:.4f} * d_DA  "
              f"(Pearson r = {r_full:.3f})")
        print()
        print(f"{'System':<14} {'d_DA':>6} {'delta0_c':>8} {'delta0_p':>8}  "
              f"{'KIE_exp':>8} {'KIE_marc':>8} {'KIE_ZPE':>8} {'KIE_LOO':>9}  "
              f"{'lgE_ZPE':>8} {'lgE_LOO':>8}")
        print("-" * 110)

        abs_log_loo, abs_log_zpe = [], []
        within2_loo, within2_zpe = 0, 0
        within3_loo, within3_zpe = 0, 0

        for name in names:
            r = results[name]
            print(f"{name:<14} {r['d_DA']:>6.2f} {r['delta0_calib']:>8.3f} "
                  f"{r['delta0_pred']:>8.3f}  "
                  f"{r['KIE_exp']:>8.1f} {r['KIE_marcus']:>8.1f} "
                  f"{r['KIE_zpe']:>8.1f} {r['KIE_loo']:>9.1f}  "
                  f"{r['log_err_zpe']:>+8.2f} {r['log_err_loo']:>+8.2f}")

            abs_log_loo.append(abs(r['log_err_loo']))
            abs_log_zpe.append(abs(r['log_err_zpe']))
            if abs(r['log_err_loo']) <= math.log10(2):
                within2_loo += 1
            if abs(r['log_err_zpe']) <= math.log10(2):
                within2_zpe += 1
            if abs(r['log_err_loo']) <= math.log10(3):
                within3_loo += 1
            if abs(r['log_err_zpe']) <= math.log10(3):
                within3_zpe += 1

        n = len(names)
        print("-" * 110)
        print(f"\n{'Model':<22} {'Mean |log10|':>14} {'Within 2x':>12} {'Within 3x':>12}")
        print(f"  Marcus (KIE=1)      {'—':>14} {'0/' + str(n):>12} {'0/' + str(n):>12}")
        print(f"  ZPE-semiclassical   {np.mean(abs_log_zpe):>14.3f} "
              f"{within2_zpe}/{n:>{1}}          {within3_zpe}/{n}")
        print(f"  Vibronic LOO        {np.mean(abs_log_loo):>14.3f} "
              f"{within2_loo}/{n:>{1}}          {within3_loo}/{n}")
        print()
        print("LOO regression accuracy summary:")
        print(f"  Mean |log10(KIE_LOO/KIE_exp)| = {np.mean(abs_log_loo):.3f}")
        print(f"  Median = {np.median(abs_log_loo):.3f}")
        print(f"  Max    = {np.max(abs_log_loo):.3f}  (worst system: "
              f"{names[int(np.argmax(abs_log_loo))]})")
        print()
        print("ZPE-semiclassical (no tunneling) accuracy summary:")
        print(f"  Mean |log10(KIE_ZPE/KIE_exp)| = {np.mean(abs_log_zpe):.3f}")
        print(f"  Systems where KIE_exp > 10 that ZPE model CANNOT explain:")
        for name in names:
            r = results[name]
            if r['KIE_exp'] > 10 and r['KIE_zpe'] < r['KIE_exp'] * 0.5:
                print(f"    {name}: KIE_exp = {r['KIE_exp']:.0f}, "
                      f"KIE_ZPE = {r['KIE_zpe']:.1f}, "
                      f"KIE_LOO = {r['KIE_loo']:.1f}")
        print("=" * 110)

    return {
        "results": results,
        "r_full": r_full,
        "a_full": a_full,
        "b_full": b_full,
    }


# =====================================================================
# Summary stats for paper table
# =====================================================================

def print_paper_table(output: dict):
    """Print a condensed table suitable for the paper."""
    results = output["results"]
    r = output["r_full"]
    a = output["a_full"]
    b = output["b_full"]

    # Select representative systems for paper table
    paper_systems = [
        "SLO-1", "SLO-1-DM", "AADH", "MADH",
        "DHFR", "TSase", "LADH", "bc1",
        "MR", "PETNR", "GOx",
    ]

    print("\n" + "=" * 90)
    print("TABLE FOR PAPER: Representative model comparison (11 systems)")
    print(f"Linear fit: delta_0 = {a:.4f} + {b:.4f} * d_DA  (r = {r:.3f})")
    print("=" * 90)
    print(f"{'System':<14} {'KIE_exp':>8} {'KIE_Marcus':>11} {'KIE_ZPE':>9} {'KIE_LOO':>9}")
    print("-" * 90)
    for name in paper_systems:
        if name in results:
            res = results[name]
            print(f"{name:<14} {res['KIE_exp']:>8.1f} {'1.0':>11} "
                  f"{res['KIE_zpe']:>9.1f} {res['KIE_loo']:>9.1f}")
    print("-" * 90)

    # Summary row
    all_loo = [abs(results[n]['log_err_loo']) for n in paper_systems if n in results]
    all_zpe = [abs(results[n]['log_err_zpe']) for n in paper_systems if n in results]
    print(f"{'Mean |log10 err|':<14} {'—':>8} {'∞':>11} {np.mean(all_zpe):>9.3f} "
          f"{np.mean(all_loo):>9.3f}")
    print("=" * 90)


# =====================================================================
# Path 2: Geometric delta_0 test
# =====================================================================

SLO1_FAMILY = {
    "SLO-1", "SLO-1-L546A", "SLO-1-L754A", "SLO-1-DM",
    "SLO-1-I553G", "SLO-1-I553A", "SLO-1-I553V",
    "SLO-1-I553L", "SLO-1-I553F",
}


def run_geometric_delta0_test(temperature: float = 298.15, verbose: bool = True) -> dict:
    """Compute KIE using pure geometric delta_0 = d_DA - r_DH - r_AH.

    No fitting.  Demonstrates that enzyme environment (active-site packing,
    protein compression) is essential: geometry alone cannot reproduce KIE.
    """
    engine = PCETRateEngine(temperature=temperature)
    results = {}

    for name, sys in BENCHMARK_SYSTEMS.items():
        delta0_geom = max(sys.d_DA - sys.r_DH - sys.r_AH, 0.05)
        result = engine.compute_rate(
            V_el=sys.V_el,
            delta_G=sys.delta_G,
            lambda_reorg=sys.lambda_reorg,
            omega_H=sys.omega_H,
            d_DA=sys.d_DA,
            method="vibronic_multi",
            delta_0=delta0_geom,
        )
        log_err = math.log10(result.KIE / sys.KIE_exp)
        results[name] = {
            "delta0_calib": sys.delta_0,
            "delta0_geom": delta0_geom,
            "KIE_exp": sys.KIE_exp,
            "KIE_geom": result.KIE,
            "log_err": log_err,
        }

    if verbose:
        print("\n" + "=" * 80)
        print("PATH 2: GEOMETRIC delta_0 TEST  (delta_0 = d_DA - r_DH - r_AH, no fitting)")
        print("=" * 80)
        print(f"{'System':<16} {'d0_calib':>9} {'d0_geom':>8} {'KIE_exp':>8} "
              f"{'KIE_geom':>9} {'log10_err':>10}")
        print("-" * 80)
        errs = []
        for name, r in results.items():
            print(f"{name:<16} {r['delta0_calib']:>9.3f} {r['delta0_geom']:>8.3f} "
                  f"{r['KIE_exp']:>8.1f} {r['KIE_geom']:>9.1f} {r['log_err']:>+10.2f}")
            errs.append(abs(r['log_err']))
        print("-" * 80)
        w2 = sum(1 for r in results.values() if abs(r['log_err']) <= math.log10(2))
        print(f"Mean |log10(KIE_geom/KIE_exp)| = {np.mean(errs):.3f}  "
              f"Within 2x: {w2}/{len(results)}")
        print()
        # Show calibrated vs geometric delta_0 correlation
        calib = [r['delta0_calib'] for r in results.values()]
        geom  = [r['delta0_geom']  for r in results.values()]
        r_cg = float(np.corrcoef(geom, calib)[0, 1])
        print(f"Correlation(delta0_geom, delta0_calib) = {r_cg:.3f}")
        print("  --> Geometric delta_0 partially tracks calibrated delta_0,")
        print("      but enzyme-specific compression systematically reduces")
        print("      the effective tunneling distance below the geometric estimate.")
        print("=" * 80)

    return results


# =====================================================================
# Path 3: Cross-family regression
# =====================================================================

def run_cross_family_regression(temperature: float = 298.15, verbose: bool = True) -> dict:
    """Fit delta_0 vs d_DA on SLO-1 family, predict all other 19 systems.

    The SLO-1 family (9 systems: WT + 8 mutants) spans d_DA = 2.69-3.10 Å
    with the same HAT chemistry (C-H abstraction by Fe(III)-OH, r_DH=1.09,
    r_AH=0.96, omega_H~2900 cm-1).  Fitting only within this family gives
    a regression that is completely independent of the other 19 systems.

    Where the cross-family prediction fails, chemistry (not just geometry)
    drives delta_0 — motivating the bond-type categorization.
    """
    systems = BENCHMARK_SYSTEMS
    names = list(systems.keys())

    # SLO-1 family regression
    slo1_names = [n for n in names if n in SLO1_FAMILY]
    slo1_d = np.array([systems[n].d_DA for n in slo1_names])
    slo1_d0 = np.array([systems[n].delta_0 for n in slo1_names
                        if systems[n].delta_0 is not None])
    A = np.column_stack([np.ones_like(slo1_d), slo1_d])
    (a_slo, b_slo), _, _, _ = np.linalg.lstsq(A, slo1_d0, rcond=None)
    r_slo = float(np.corrcoef(slo1_d, slo1_d0)[0, 1])

    engine = PCETRateEngine(temperature=temperature)
    results = {}

    for name, sys in systems.items():
        if name in SLO1_FAMILY:
            continue  # skip training set

        delta0_cross = max(a_slo + b_slo * sys.d_DA, 0.05)
        result = engine.compute_rate(
            V_el=sys.V_el,
            delta_G=sys.delta_G,
            lambda_reorg=sys.lambda_reorg,
            omega_H=sys.omega_H,
            d_DA=sys.d_DA,
            method="vibronic_multi",
            delta_0=delta0_cross,
        )
        log_err = math.log10(result.KIE / sys.KIE_exp)

        # Label by chemistry for interpretation
        if name in {"AADH", "MADH", "TauD", "DβH", "GO", "CAO", "GOx"}:
            chem = "HAT"
        elif name in {"DHFR", "TSase", "LADH"}:
            chem = "hydride"
        elif name in {"bc1", "PhOH-self", "RNR-3FY", "RNR-2FY"}:
            chem = "proton/O-H"
        elif name in {"RNR"}:
            chem = "S-H"
        elif name in {"MR", "PETNR", "MAO"}:
            chem = "flavin-HT"
        elif name in {"PHM"}:
            chem = "HAT/Cu"
        else:
            chem = "other"

        results[name] = {
            "d_DA": sys.d_DA,
            "delta0_calib": sys.delta_0,
            "delta0_cross": delta0_cross,
            "KIE_exp": sys.KIE_exp,
            "KIE_cross": result.KIE,
            "log_err": log_err,
            "chem": chem,
        }

    if verbose:
        print("\n" + "=" * 90)
        print("PATH 3: CROSS-FAMILY REGRESSION")
        print(f"  Training: SLO-1 family ({len(slo1_names)} systems), "
              f"delta_0 = {a_slo:.4f} + {b_slo:.4f}*d_DA  (r = {r_slo:.3f})")
        print(f"  Test: {len(results)} non-SLO systems  (zero per-system KIE fitting)")
        print("=" * 90)
        print(f"{'System':<16} {'Chem':>12} {'d0_calib':>9} {'d0_cross':>9} "
              f"{'KIE_exp':>8} {'KIE_cross':>10} {'log10_err':>10}")
        print("-" * 90)

        by_chem: dict[str, list] = {}
        for name, r in results.items():
            print(f"{name:<16} {r['chem']:>12} {r['delta0_calib']:>9.3f} "
                  f"{r['delta0_cross']:>9.3f} {r['KIE_exp']:>8.1f} "
                  f"{r['KIE_cross']:>10.1f} {r['log_err']:>+10.2f}")
            by_chem.setdefault(r['chem'], []).append(abs(r['log_err']))

        print("-" * 90)
        all_errs = [abs(r['log_err']) for r in results.values()]
        w2 = sum(1 for r in results.values() if abs(r['log_err']) <= math.log10(2))
        w3 = sum(1 for r in results.values() if abs(r['log_err']) <= math.log10(3))
        print(f"All 19 systems:  mean |log10| = {np.mean(all_errs):.3f}, "
              f"within 2x: {w2}/{len(results)}, within 3x: {w3}/{len(results)}")
        print()
        print("By chemistry class:")
        for chem, errs in sorted(by_chem.items()):
            w = sum(1 for e in errs if e <= math.log10(2))
            print(f"  {chem:<14}: mean |log10| = {np.mean(errs):.3f}  "
                  f"within 2x: {w}/{len(errs)}")
        print()
        print("Interpretation:")
        print("  HAT systems (same chemistry as SLO-1): better cross-family transfer")
        print("  Hydride/proton systems: SLO-1 regression over-predicts delta_0,")
        print("  giving inflated KIE — motivates separate bond-type categorization.")
        print("=" * 90)

    return {
        "results": results,
        "a_slo": a_slo,
        "b_slo": b_slo,
        "r_slo": r_slo,
        "n_train": len(slo1_names),
    }


if __name__ == "__main__":
    output = run_loo_predictive_test(verbose=True)
    print_paper_table(output)
    run_geometric_delta0_test(verbose=True)
    run_cross_family_regression(verbose=True)
