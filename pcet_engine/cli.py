"""
Command-line interface for PCET rate predictions.

Usage:
    pcet rate --V_el 0.5 --delta_G -5.0 --lambda 20.0 --omega 3000 --d_DA 2.7
    pcet hessian reactant.fchk product.fchk --proton 5 --donor 3 --acceptor 8
    pcet benchmark
"""

import argparse
import sys
import json


def cmd_rate(args):
    """Compute PCET rate from explicit parameters."""
    from pcet_engine.core.rate_engine import PCETRateEngine

    engine = PCETRateEngine(temperature=args.temperature)
    result = engine.compute_rate(
        V_el=args.V_el,
        delta_G=args.delta_G,
        lambda_reorg=args.lambda_reorg,
        omega_H=args.omega,
        d_DA=args.d_DA,
        method=args.method,
        delta_0=args.delta_0,
    )

    if args.json:
        print(json.dumps({
            "k_H": result.k_H,
            "k_D": result.k_D,
            "KIE": result.KIE,
            "E_a_kcal": result.E_a,
            "method": result.method,
            "omega_H_cm": result.omega_H,
            "omega_D_cm": result.omega_D,
        }, indent=2))
    else:
        print(f"PCET Rate Prediction ({result.method})")
        print(f"{'='*40}")
        print(f"k_H  = {result.k_H:.3e} s⁻¹")
        print(f"k_D  = {result.k_D:.3e} s⁻¹")
        print(f"KIE  = {result.KIE:.1f}")
        print(f"E_a  = {result.E_a:.2f} kcal/mol")
        print(f"ω_H  = {result.omega_H:.0f} cm⁻¹")
        print(f"ω_D  = {result.omega_D:.0f} cm⁻¹")


def cmd_hessian(args):
    """Compute PCET rate from Hessian files."""
    from pcet_engine.parsers import parse_gaussian_fchk, parse_orca_hess
    from pcet_engine.core.rate_engine import PCETRateEngine

    def load(path):
        if path.endswith(".fchk"):
            return parse_gaussian_fchk(path)
        elif path.endswith(".hess"):
            return parse_orca_hess(path)
        else:
            sys.exit(f"Unknown file format: {path} (expected .fchk or .hess)")

    qc_R = load(args.reactant)
    qc_P = load(args.product)

    engine = PCETRateEngine(temperature=args.temperature)
    result = engine.compute_rate_from_hessian(
        hessian_R=qc_R.hessian,
        hessian_P=qc_P.hessian,
        geom_R=qc_R.geometry,
        geom_P=qc_P.geometry,
        masses=qc_R.masses,
        proton_idx=args.proton,
        donor_idx=args.donor,
        acceptor_idx=args.acceptor,
        V_el=args.V_el,
        delta_G=args.delta_G,
        lambda_outer=args.lambda_outer,
    )

    if args.json:
        print(json.dumps({
            "k_H": result.k_H,
            "k_D": result.k_D,
            "KIE": result.KIE,
            "E_a_kcal": result.E_a,
            "omega_H_cm": result.omega_H,
            "omega_D_cm": result.omega_D,
            "d_DA_A": result.d_DA,
            "method": result.method,
        }, indent=2))
    else:
        print(f"PCET Rate from Hessian ({result.method})")
        print(f"{'='*40}")
        print(f"Reactant: {args.reactant}")
        print(f"Product:  {args.product}")
        print(f"k_H  = {result.k_H:.3e} s⁻¹")
        print(f"k_D  = {result.k_D:.3e} s⁻¹")
        print(f"KIE  = {result.KIE:.1f}")
        print(f"E_a  = {result.E_a:.2f} kcal/mol")
        print(f"ω_H  = {result.omega_H:.0f} cm⁻¹")
        print(f"d_DA = {result.d_DA:.3f} Å")


def cmd_benchmark(args):
    """Run benchmark against published enzyme PCET data."""
    from pcet_engine.benchmarks.systems import BENCHMARK_SYSTEMS
    from pcet_engine.core.rate_engine import PCETRateEngine

    engine = PCETRateEngine()

    print(f"{'System':<8} {'k_H (s⁻¹)':>12} {'KIE calc':>10} {'KIE exp':>10} {'Δ%':>8}")
    print("-" * 52)

    for name, sys in BENCHMARK_SYSTEMS.items():
        result = engine.compute_rate(
            V_el=sys.V_el,
            delta_G=sys.delta_G,
            lambda_reorg=sys.lambda_reorg,
            omega_H=sys.omega_H,
            d_DA=sys.d_DA,
            delta_0=sys.delta_0,
        )
        kie_exp = sys.KIE_exp
        err = (result.KIE - kie_exp) / kie_exp * 100
        print(f"{name:<8} {result.k_H:>12.3e} {result.KIE:>10.1f} {kie_exp:>10.1f} {err:>+7.1f}%")


def main():
    parser = argparse.ArgumentParser(
        prog="pcet",
        description="PCET Rate Theory Engine — predict proton-coupled electron transfer rates",
    )
    sub = parser.add_subparsers(dest="command")

    # rate subcommand
    p_rate = sub.add_parser("rate", help="Compute rate from explicit parameters")
    p_rate.add_argument("--V_el", type=float, required=True, help="Electronic coupling (kcal/mol)")
    p_rate.add_argument("--delta_G", type=float, required=True, help="Driving force (kcal/mol)")
    p_rate.add_argument("--lambda_reorg", type=float, required=True, help="Reorganization energy (kcal/mol)")
    p_rate.add_argument("--omega", type=float, required=True, help="Proton frequency (cm⁻¹)")
    p_rate.add_argument("--d_DA", type=float, required=True, help="Donor-acceptor distance (Å)")
    p_rate.add_argument("--method", default="vibronic_multi", choices=["marcus", "vibronic_single", "vibronic_multi"])
    p_rate.add_argument("--delta_0", type=float, default=None, help="Tunneling distance (Å)")
    p_rate.add_argument("--temperature", type=float, default=298.15, help="Temperature (K)")
    p_rate.add_argument("--json", action="store_true", help="Output as JSON")
    p_rate.set_defaults(func=cmd_rate)

    # hessian subcommand
    p_hess = sub.add_parser("hessian", help="Compute rate from Hessian files (.fchk or .hess)")
    p_hess.add_argument("reactant", help="Reactant Hessian file")
    p_hess.add_argument("product", help="Product Hessian file")
    p_hess.add_argument("--proton", type=int, required=True, help="Proton atom index (0-based)")
    p_hess.add_argument("--donor", type=int, required=True, help="Donor atom index (0-based)")
    p_hess.add_argument("--acceptor", type=int, required=True, help="Acceptor atom index (0-based)")
    p_hess.add_argument("--V_el", type=float, required=True, help="Electronic coupling (kcal/mol)")
    p_hess.add_argument("--delta_G", type=float, required=True, help="Driving force (kcal/mol)")
    p_hess.add_argument("--lambda_outer", type=float, default=0.0, help="Outer-sphere reorganization (kcal/mol)")
    p_hess.add_argument("--temperature", type=float, default=298.15, help="Temperature (K)")
    p_hess.add_argument("--json", action="store_true", help="Output as JSON")
    p_hess.set_defaults(func=cmd_hessian)

    # benchmark subcommand
    p_bench = sub.add_parser("benchmark", help="Run benchmarks against published enzyme data")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
