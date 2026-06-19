"""
Rank enzyme variants by predicted PCET rate.

Given a set of mutations (each with modified molecular parameters),
compute rates for all variants and return a ranked table with
uncertainty quantification and sensitivity analysis.

This is the product-facing module that turns the PCET engine from
an expert tool into a decision-support system for enzyme engineers.

Usage::

    from pcet_engine.core.variant_ranker import VariantRanker

    ranker = VariantRanker(
        V_el=0.6,
        delta_G=-5.4,
        lambda_reorg=19.0,
        omega_H=2900.0,
    )
    results = ranker.rank([
        {"name": "WT",    "d_DA": 2.77},
        {"name": "L546A", "d_DA": 2.88},
        {"name": "L754A", "d_DA": 2.95},
        {"name": "DM",    "d_DA": 3.10},
    ])
    for r in results.ranked:
        print(f"{r.name}: k_H={r.k_H:.2e}, KIE={r.KIE:.1f}, rank={r.rank}")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from pcet_engine.core.rate_engine import PCETRateEngine, PCETResult
from pcet_engine.core.uncertainty import propagate_uncertainty, UncertaintyResult


@dataclass
class VariantResult:
    """Result for a single enzyme variant."""
    name: str
    d_DA: float
    k_H: float
    k_D: float
    KIE: float
    E_a: float
    rate_ratio: float  # vs reference (first variant or WT)
    rank: int
    # UQ fields (populated if run_uq=True)
    k_H_ci: Optional[tuple] = None  # (low, high) 95% CI
    KIE_ci: Optional[tuple] = None
    dominant_sensitivity: Optional[str] = None  # parameter with highest sensitivity


@dataclass
class RankingResult:
    """Ranked list of enzyme variants."""
    ranked: list  # List[VariantResult], sorted by k_H descending
    reference: str  # name of reference variant
    method: str
    temperature: float
    n_variants: int

    def to_dict(self):
        return {
            "reference": self.reference,
            "method": self.method,
            "temperature": self.temperature,
            "n_variants": self.n_variants,
            "variants": [
                {
                    "rank": r.rank,
                    "name": r.name,
                    "d_DA": r.d_DA,
                    "k_H": r.k_H,
                    "k_D": r.k_D,
                    "KIE": r.KIE,
                    "E_a": r.E_a,
                    "rate_ratio": r.rate_ratio,
                    "k_H_ci": list(r.k_H_ci) if r.k_H_ci else None,
                    "KIE_ci": list(r.KIE_ci) if r.KIE_ci else None,
                    "dominant_sensitivity": r.dominant_sensitivity,
                }
                for r in self.ranked
            ],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Variant Ranking ({self.n_variants} variants, "
            f"reference: {self.reference}, method: {self.method})",
            f"{'Rank':<5} {'Name':<12} {'d_DA':>6} {'k_H':>12} "
            f"{'KIE':>8} {'Rate Ratio':>12}",
            "-" * 60,
        ]
        for r in self.ranked:
            ratio_str = f"{r.rate_ratio:.3f}x"
            lines.append(
                f"{r.rank:<5} {r.name:<12} {r.d_DA:>6.3f} {r.k_H:>12.3e} "
                f"{r.KIE:>8.1f} {ratio_str:>12}"
            )
        return "\n".join(lines)


class VariantRanker:
    """Rank enzyme variants by predicted PCET rate.

    Holds the "shared" molecular parameters (V_el, delta_G, lambda_reorg,
    omega_H) constant across variants. Each variant differs in d_DA
    (donor-acceptor distance) and optionally other parameters.

    Parameters
    ----------
    V_el : float
        Electronic coupling in kcal/mol (shared across variants).
    delta_G : float
        Driving force in kcal/mol (shared).
    lambda_reorg : float
        Reorganization energy in kcal/mol (shared).
    omega_H : float
        Proton vibrational frequency in cm⁻¹ (shared).
    method : str
        Rate method ('vibronic_multi' default).
    temperature : float
        Temperature in K (default 298.15).
    """

    def __init__(
        self,
        V_el: float,
        delta_G: float,
        lambda_reorg: float,
        omega_H: float,
        method: str = "vibronic_multi",
        temperature: float = 298.15,
    ):
        self.V_el = V_el
        self.delta_G = delta_G
        self.lambda_reorg = lambda_reorg
        self.omega_H = omega_H
        self.method = method
        self.temperature = temperature
        self.engine = PCETRateEngine(temperature=temperature)

    def rank(
        self,
        variants: list,
        reference: Optional[str] = None,
        run_uq: bool = False,
        n_samples: int = 500,
        param_errors: Optional[dict] = None,
    ) -> RankingResult:
        """Rank variants by predicted H-transfer rate.

        Parameters
        ----------
        variants : list of dict
            Each dict must have 'name' and 'd_DA'. May optionally override
            any shared parameter (V_el, delta_G, lambda_reorg, omega_H).
        reference : str, optional
            Name of reference variant for rate_ratio. Default: first variant.
        run_uq : bool
            If True, run Monte Carlo UQ for each variant.
        n_samples : int
            Number of MC samples (if run_uq=True).
        param_errors : dict, optional
            Parameter uncertainties for UQ. Keys: V_el_err, delta_G_err,
            lambda_reorg_err, omega_H_err, d_DA_err. Defaults provided.

        Returns
        -------
        RankingResult
            Variants ranked by k_H (descending).
        """
        if not variants:
            raise ValueError("Must provide at least one variant")

        if reference is None:
            reference = variants[0]["name"]

        # Default UQ errors
        errors = {
            "V_el_err": self.V_el * 0.1,
            "delta_G_err": 0.5,
            "lambda_reorg_err": 2.0,
            "omega_H_err": 100.0,
            "d_DA_err": 0.05,
        }
        if param_errors:
            errors.update(param_errors)

        # Compute rates for all variants
        results = []
        for v in variants:
            name = v["name"]
            d_DA = v["d_DA"]
            V_el = v.get("V_el", self.V_el)
            delta_G = v.get("delta_G", self.delta_G)
            lambda_reorg = v.get("lambda_reorg", self.lambda_reorg)
            omega_H = v.get("omega_H", self.omega_H)

            rate_result = self.engine.compute_rate(
                V_el=V_el,
                delta_G=delta_G,
                lambda_reorg=lambda_reorg,
                omega_H=omega_H,
                d_DA=d_DA,
                method=self.method,
            )

            uq_data = None
            if run_uq:
                uq_data = propagate_uncertainty(
                    V_el=V_el, V_el_err=errors["V_el_err"],
                    delta_G=delta_G, delta_G_err=errors["delta_G_err"],
                    lambda_reorg=lambda_reorg,
                    lambda_reorg_err=errors["lambda_reorg_err"],
                    omega_H=omega_H, omega_H_err=errors["omega_H_err"],
                    d_DA=d_DA, d_DA_err=errors["d_DA_err"],
                    n_samples=n_samples, seed=42,
                )

            results.append({
                "name": name,
                "d_DA": d_DA,
                "k_H": rate_result.k_H,
                "k_D": rate_result.k_D,
                "KIE": rate_result.KIE,
                "E_a": rate_result.E_a,
                "uq": uq_data,
            })

        # Find reference rate
        ref_k_H = next(
            (r["k_H"] for r in results if r["name"] == reference),
            results[0]["k_H"],
        )

        # Sort by k_H descending (fastest first)
        results.sort(key=lambda r: r["k_H"], reverse=True)

        # Build VariantResult list
        ranked = []
        for i, r in enumerate(results):
            uq = r["uq"]
            ranked.append(VariantResult(
                name=r["name"],
                d_DA=r["d_DA"],
                k_H=r["k_H"],
                k_D=r["k_D"],
                KIE=r["KIE"],
                E_a=r["E_a"],
                rate_ratio=r["k_H"] / ref_k_H if ref_k_H > 0 else float("inf"),
                rank=i + 1,
                k_H_ci=tuple(uq.k_H_ci) if uq else None,
                KIE_ci=tuple(uq.KIE_ci) if uq else None,
                dominant_sensitivity=(
                    max(uq.sensitivities, key=uq.sensitivities.get)
                    if uq and uq.sensitivities else None
                ),
            ))

        return RankingResult(
            ranked=ranked,
            reference=reference,
            method=self.method,
            temperature=self.temperature,
            n_variants=len(ranked),
        )
