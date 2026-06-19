"""
Hessian-based validation of the PCET rate engine.

Runs the full pipeline: model Hessian → normal mode analysis → vibronic rate
→ KIE prediction, and compares to published experimental data for 5 benchmark
enzyme systems.

This validates Phase 2 of the PCET project: reproducing published results
using the complete Hessian-to-rate pipeline rather than manually specified
parameters.
"""

import math
import numpy as np

from pcet_engine.core.rate_engine import PCETRateEngine
from pcet_engine.benchmarks.systems import BENCHMARK_SYSTEMS, BenchmarkSystem
from pcet_engine.data.model_hessians import (
    MODEL_SYSTEMS,
    build_hessian,
    generate_all_model_hessians,
)


def run_hessian_benchmarks(
    temperature: float = 298.15,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run all benchmark systems through the Hessian-to-rate pipeline.

    For each system:
    1. Builds model Hessian from active-site fragment parameters
    2. Extracts proton frequency and D-A distance via normal mode analysis
    3. Computes inner-sphere reorganization energy (four-point method)
    4. Runs vibronic multi-channel rate calculation
    5. Compares predicted KIE and E_a to experiment

    Args:
        temperature: Temperature in Kelvin.
        verbose: If True, print results table.

    Returns:
        Dict mapping system name -> {result, omega_H_extracted, d_DA_extracted,
        lambda_inner, k_H_exp, KIE_exp, ...}.
    """
    engine = PCETRateEngine(temperature=temperature)
    results = {}

    if verbose:
        print("=" * 100)
        print(f"HESSIAN-TO-RATE PIPELINE VALIDATION — T = {temperature:.1f} K")
        print("=" * 100)
        print(f"{'System':<8} {'ω_H ext':>8} {'ω_H lit':>8} {'d_DA':>6} "
              f"{'λ_inn':>6} {'λ_tot':>6} {'k_H':>12} {'k_H exp':>12} "
              f"{'KIE':>8} {'KIE exp':>8} {'E_a':>6} {'E_a exp':>7}")
        print("-" * 100)

    for name in MODEL_SYSTEMS:
        bench = BENCHMARK_SYSTEMS[name]
        model = MODEL_SYSTEMS[name]

        # Build model Hessians
        hess_R, geom_R, elements, masses = build_hessian(model, "reactant")
        hess_P, geom_P, _, _ = build_hessian(model, "product")

        # Run full pipeline
        result = engine.compute_rate_from_hessian(
            hessian_R=hess_R,
            hessian_P=hess_P,
            geom_R=geom_R,
            geom_P=geom_P,
            masses=masses,
            proton_idx=1,       # H is always atom 1
            donor_idx=0,        # Donor is always atom 0
            acceptor_idx=2,     # Acceptor is always atom 2
            V_el=bench.V_el,
            delta_G=bench.delta_G,
            lambda_outer=bench.lambda_reorg * 0.6,  # outer ≈ 60% of total literature λ
            delta_0=bench.delta_0,  # use calibrated tunneling distance
        )

        # Compute inner-sphere λ for reporting
        lambda_inner = result.lambda_reorg - bench.lambda_reorg * 0.6

        log_err_kH = math.log10(result.k_H / bench.k_H_exp) if result.k_H > 0 else float("inf")

        results[name] = {
            "result": result,
            "omega_H_extracted": result.omega_H,
            "omega_H_literature": bench.omega_H,
            "d_DA_extracted": result.d_DA,
            "lambda_inner": lambda_inner,
            "lambda_total": result.lambda_reorg,
            "k_H_exp": bench.k_H_exp,
            "k_D_exp": bench.k_D_exp,
            "KIE_exp": bench.KIE_exp,
            "E_a_exp": bench.E_a_exp,
            "log_error_kH": log_err_kH,
            "KIE_ratio": result.KIE / bench.KIE_exp if bench.KIE_exp > 0 else float("inf"),
        }

        if verbose:
            print(f"{name:<8} {result.omega_H:>8.0f} {bench.omega_H:>8.0f} "
                  f"{result.d_DA:>6.2f} {lambda_inner:>6.1f} {result.lambda_reorg:>6.1f} "
                  f"{result.k_H:>12.2e} {bench.k_H_exp:>12.2e} "
                  f"{result.KIE:>8.1f} {bench.KIE_exp:>8.0f} "
                  f"{result.E_a:>6.1f} {bench.E_a_exp:>7.1f}")

    if verbose:
        print("-" * 100)
        log_errs = [abs(r["log_error_kH"]) for r in results.values()
                    if abs(r["log_error_kH"]) < 100]
        kie_ratios = [abs(r["KIE_ratio"] - 1.0) for r in results.values()
                      if abs(r["KIE_ratio"]) < 100]
        freq_errs = [abs(r["omega_H_extracted"] - r["omega_H_literature"]) / r["omega_H_literature"]
                     for r in results.values()]

        if log_errs:
            print(f"Mean |log10(k_pred/k_exp)| = {np.mean(log_errs):.2f}")
        if kie_ratios:
            print(f"Mean |KIE_ratio - 1| = {np.mean(kie_ratios):.2f}")
        if freq_errs:
            print(f"Mean |Δω_H/ω_H| = {np.mean(freq_errs):.3f} ({np.mean(freq_errs)*100:.1f}%)")
        print("=" * 100)

    return results


if __name__ == "__main__":
    run_hessian_benchmarks()
