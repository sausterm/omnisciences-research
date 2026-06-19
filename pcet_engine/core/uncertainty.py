"""
Uncertainty quantification for PCET rate predictions.

Propagates parameter uncertainties through the rate calculation using
Monte Carlo sampling or analytical sensitivity analysis.

Key question: given uncertainties in V_el, ΔG, λ, ω_H, d_DA, how
uncertain is the predicted k_H, k_D, and KIE?
"""

import numpy as np
from dataclasses import dataclass
from pcet_engine.core.rate_engine import PCETRateEngine, PCETResult


@dataclass
class UncertaintyResult:
    """Result of uncertainty propagation.

    Attributes:
        k_H_mean: Mean H rate in s⁻¹.
        k_H_std: Standard deviation of H rate.
        k_H_ci: 95% confidence interval (low, high).
        k_D_mean: Mean D rate.
        k_D_std: Standard deviation of D rate.
        KIE_mean: Mean KIE.
        KIE_std: Standard deviation of KIE.
        KIE_ci: 95% confidence interval for KIE.
        sensitivities: dict mapping parameter name to d(ln k_H)/d(param).
        n_samples: Number of Monte Carlo samples used.
    """
    k_H_mean: float
    k_H_std: float
    k_H_ci: tuple[float, float]
    k_D_mean: float
    k_D_std: float
    KIE_mean: float
    KIE_std: float
    KIE_ci: tuple[float, float]
    sensitivities: dict[str, float]
    n_samples: int


def propagate_uncertainty(
    V_el: float, V_el_err: float,
    delta_G: float, delta_G_err: float,
    lambda_reorg: float, lambda_reorg_err: float,
    omega_H: float, omega_H_err: float,
    d_DA: float, d_DA_err: float,
    n_samples: int = 1000,
    temperature: float = 298.15,
    delta_0: float | None = None,
    delta_0_err: float = 0.0,
    seed: int | None = None,
) -> UncertaintyResult:
    """Propagate parameter uncertainties via Monte Carlo sampling.

    Each parameter is sampled from a normal distribution with the given
    mean and standard deviation. Negative values are clipped where
    physically required (e.g., lambda > 0).

    Args:
        V_el, V_el_err: Electronic coupling and its uncertainty (kcal/mol).
        delta_G, delta_G_err: Driving force and uncertainty (kcal/mol).
        lambda_reorg, lambda_reorg_err: Reorganization energy and uncertainty (kcal/mol).
        omega_H, omega_H_err: Proton frequency and uncertainty (cm⁻¹).
        d_DA, d_DA_err: Donor-acceptor distance and uncertainty (Å).
        n_samples: Number of Monte Carlo samples.
        temperature: Temperature in K.
        delta_0: Tunneling distance (Å), None for geometric estimate.
        delta_0_err: Uncertainty in delta_0 (Å).
        seed: Random seed for reproducibility.

    Returns:
        UncertaintyResult with statistics and sensitivities.
    """
    rng = np.random.default_rng(seed)
    engine = PCETRateEngine(temperature=temperature)

    # Sample parameters
    V_els = rng.normal(V_el, max(V_el_err, 1e-15), n_samples)
    dGs = rng.normal(delta_G, max(delta_G_err, 1e-15), n_samples)
    lams = rng.normal(lambda_reorg, max(lambda_reorg_err, 1e-15), n_samples)
    omegas = rng.normal(omega_H, max(omega_H_err, 1e-15), n_samples)
    dDAs = rng.normal(d_DA, max(d_DA_err, 1e-15), n_samples)

    # Physical constraints
    V_els = np.clip(V_els, 0.001, None)
    lams = np.clip(lams, 0.1, None)
    omegas = np.clip(omegas, 100, None)
    dDAs = np.clip(dDAs, 1.0, None)

    if delta_0 is not None and delta_0_err > 0:
        d0s = rng.normal(delta_0, delta_0_err, n_samples)
        d0s = np.clip(d0s, 0.01, None)
    else:
        d0s = [delta_0] * n_samples

    k_Hs = np.zeros(n_samples)
    k_Ds = np.zeros(n_samples)
    KIEs = np.zeros(n_samples)

    for i in range(n_samples):
        try:
            result = engine.compute_rate(
                V_el=float(V_els[i]),
                delta_G=float(dGs[i]),
                lambda_reorg=float(lams[i]),
                omega_H=float(omegas[i]),
                d_DA=float(dDAs[i]),
                delta_0=float(d0s[i]) if d0s[i] is not None else None,
            )
            k_Hs[i] = result.k_H
            k_Ds[i] = result.k_D
            KIEs[i] = result.KIE
        except Exception:
            k_Hs[i] = np.nan
            k_Ds[i] = np.nan
            KIEs[i] = np.nan

    # Filter out failures and infinities
    valid = (
        ~np.isnan(k_Hs) & ~np.isnan(k_Ds) &
        ~np.isinf(k_Hs) & ~np.isinf(k_Ds) &
        ~np.isnan(KIEs) & ~np.isinf(KIEs) &
        (k_Hs > 0) & (k_Ds > 0)
    )
    k_Hs = k_Hs[valid]
    k_Ds = k_Ds[valid]
    KIEs = KIEs[valid]
    n_valid = len(k_Hs)

    # Compute sensitivities via finite differences
    sensitivities = _compute_sensitivities(
        engine, V_el, delta_G, lambda_reorg, omega_H, d_DA, delta_0,
    )

    return UncertaintyResult(
        k_H_mean=float(np.mean(k_Hs)),
        k_H_std=float(np.std(k_Hs)),
        k_H_ci=(float(np.percentile(k_Hs, 2.5)), float(np.percentile(k_Hs, 97.5))),
        k_D_mean=float(np.mean(k_Ds)),
        k_D_std=float(np.std(k_Ds)),
        KIE_mean=float(np.mean(KIEs)),
        KIE_std=float(np.std(KIEs)),
        KIE_ci=(float(np.percentile(KIEs, 2.5)), float(np.percentile(KIEs, 97.5))),
        sensitivities=sensitivities,
        n_samples=n_valid,
    )


def _compute_sensitivities(
    engine: PCETRateEngine,
    V_el: float, delta_G: float, lambda_reorg: float,
    omega_H: float, d_DA: float, delta_0: float | None,
) -> dict[str, float]:
    """Compute d(ln k_H)/d(param) via central finite differences."""
    base = engine.compute_rate(
        V_el=V_el, delta_G=delta_G, lambda_reorg=lambda_reorg,
        omega_H=omega_H, d_DA=d_DA, delta_0=delta_0,
    )
    if base.k_H <= 0:
        return {}

    ln_k0 = np.log(base.k_H)
    sensitivities = {}

    params = {
        "V_el": (V_el, max(0.01 * abs(V_el), 0.001)),
        "delta_G": (delta_G, max(0.01 * abs(delta_G), 0.01)),
        "lambda_reorg": (lambda_reorg, 0.01 * lambda_reorg),
        "omega_H": (omega_H, 10.0),
        "d_DA": (d_DA, 0.01),
    }

    for name, (val, step) in params.items():
        kwargs_plus = dict(
            V_el=V_el, delta_G=delta_G, lambda_reorg=lambda_reorg,
            omega_H=omega_H, d_DA=d_DA, delta_0=delta_0,
        )
        kwargs_minus = kwargs_plus.copy()
        kwargs_plus[name] = val + step
        kwargs_minus[name] = val - step

        try:
            r_plus = engine.compute_rate(**kwargs_plus)
            r_minus = engine.compute_rate(**kwargs_minus)
            if r_plus.k_H > 0 and r_minus.k_H > 0:
                sensitivities[name] = (np.log(r_plus.k_H) - np.log(r_minus.k_H)) / (2 * step)
        except Exception:
            pass

    return sensitivities
