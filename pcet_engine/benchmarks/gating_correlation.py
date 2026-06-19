"""
Empirical correlation of D-A gating frequency with structural features.

Analyzes published Omega_gating values against:
- D-A equilibrium distance
- Donor/acceptor atom types
- Metal center type
- Reaction type / enzyme class
- Effective mass M_DA

Goal: predictive model for gating parameters from crystal structure alone.

References:
    Compiled from Hammes-Schiffer, Scrutton/Hay, Klinman, Kohen, Bollinger/Krebs groups.
    See pcet_engine/data/published_gating_params.json for full citations.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reduced masses for common D-A pairs (amu)
REDUCED_MASSES = {
    ("C", "O"): 12.0 * 16.0 / (12.0 + 16.0),   # 6.86
    ("C", "N"): 12.0 * 14.0 / (12.0 + 14.0),   # 6.46
    ("C", "C"): 12.0 * 12.0 / (12.0 + 12.0),   # 6.00
    ("S", "C"): 32.0 * 12.0 / (32.0 + 12.0),   # 8.73
    ("O", "O"): 16.0 * 16.0 / (16.0 + 16.0),   # 8.00
    ("O", "N"): 16.0 * 14.0 / (16.0 + 14.0),   # 7.47
    ("C", "S"): 12.0 * 32.0 / (12.0 + 32.0),   # 8.73
}

# Enzyme class → typical gating frequency range (cm⁻¹)
ENZYME_CLASS_GATING = {
    "lipoxygenase":       {"omega_low": 90, "omega_mid": 150, "omega_high": 200, "M_DA_typical": 14.0},
    "FeIV_oxo":           {"omega_low": 150, "omega_mid": 185, "omega_high": 220, "M_DA_typical": 12.5},
    "radical_enzyme":     {"omega_low": 180, "omega_mid": 225, "omega_high": 280, "M_DA_typical": 8.0},
    "oxidase":            {"omega_low": 250, "omega_mid": 300, "omega_high": 350, "M_DA_typical": 7.0},
    "amine_dehydrogenase": {"omega_low": 300, "omega_mid": 350, "omega_high": 400, "M_DA_typical": 5.85},
    "monooxygenase":      {"omega_low": 350, "omega_mid": 400, "omega_high": 450, "M_DA_typical": 7.0},
    "oxidoreductase":     {"omega_low": 330, "omega_mid": 380, "omega_high": 430, "M_DA_typical": 5.7},
    "dehydrogenase":      {"omega_low": 310, "omega_mid": 360, "omega_high": 410, "M_DA_typical": 5.7},
    "transferase":        {"omega_low": 290, "omega_mid": 340, "omega_high": 390, "M_DA_typical": 5.7},
}


@dataclass
class GatingDataPoint:
    """A single system with published gating parameters."""
    name: str
    omega_gating: float     # cm⁻¹
    M_DA: float             # amu
    d_DA: float             # Å
    donor_atom: str
    acceptor_atom: str
    metal_center: str | None
    reaction_type: str
    enzyme_class: str
    method: str
    reference: str
    k_H_exp: float | None = None
    KIE_exp: float | None = None


def load_published_gating() -> list[GatingDataPoint]:
    """Load published gating parameters from JSON data file.

    Returns:
        List of GatingDataPoint objects.
    """
    data_path = Path(__file__).parent.parent / "data" / "published_gating_params.json"
    with open(data_path) as f:
        data = json.load(f)

    points = []
    for name, sys in data["systems"].items():
        points.append(GatingDataPoint(
            name=name,
            omega_gating=sys["omega_gating"],
            M_DA=sys["M_DA"],
            d_DA=sys["d_DA"],
            donor_atom=sys["donor_atom"],
            acceptor_atom=sys["acceptor_atom"],
            metal_center=sys.get("metal_center"),
            reaction_type=sys["reaction_type"],
            enzyme_class=sys["enzyme_class"],
            method=sys.get("method", "unknown"),
            reference=sys.get("reference", ""),
            k_H_exp=sys.get("k_H_exp"),
            KIE_exp=sys.get("KIE_exp"),
        ))
    return points


def pearson_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson correlation coefficient and p-value (two-tailed)."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    r = np.corrcoef(x, y)[0, 1]
    # t-test for significance
    if abs(r) > 0.9999:
        return float(r), 0.0
    t = r * math.sqrt((n - 2) / (1 - r**2))
    # Approximate p-value from t-distribution (two-tailed)
    from scipy import stats
    p = 2 * stats.t.sf(abs(t), n - 2)
    return float(r), float(p)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation."""
    from scipy import stats
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def linear_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple linear regression y = a + b*x."""
    n = len(x)
    if n < 3:
        return {"slope": 0, "intercept": 0, "r_squared": 0, "rmse": float("inf")}
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = math.sqrt(ss_res / n)
    return {
        "slope": float(coeffs[0]),
        "intercept": float(coeffs[1]),
        "r_squared": float(r_sq),
        "rmse": float(rmse),
    }


def leave_one_out_cv(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out cross-validation RMSE for linear model."""
    n = len(x)
    if n < 3:
        return float("inf")
    errors = []
    for i in range(n):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)
        coeffs = np.polyfit(x_train, y_train, 1)
        y_pred = np.polyval(coeffs, x[i])
        errors.append((y[i] - y_pred) ** 2)
    return math.sqrt(np.mean(errors))


def analyze_gating_correlations(verbose: bool = True) -> dict:
    """Test correlations between Omega_gating and structural features.

    Returns:
        Dict with correlation results for each feature.
    """
    points = load_published_gating()
    n = len(points)

    omegas = np.array([p.omega_gating for p in points])
    d_DAs = np.array([p.d_DA for p in points])
    M_DAs = np.array([p.M_DA for p in points])
    has_metal = np.array([1.0 if p.metal_center else 0.0 for p in points])

    results = {}

    if verbose:
        print("=" * 80)
        print(f"GATING CORRELATION ANALYSIS — {n} systems")
        print("=" * 80)
        print()

    # Feature 1: d_DA vs Omega_gating
    r, p = pearson_r(d_DAs, omegas)
    rho, p_rho = spearman_rho(d_DAs, omegas)
    reg = linear_regression(d_DAs, omegas)
    loo = leave_one_out_cv(d_DAs, omegas)
    results["d_DA"] = {
        "pearson_r": r, "pearson_p": p,
        "spearman_rho": rho, "spearman_p": p_rho,
        "regression": reg, "loo_rmse": loo,
    }
    if verbose:
        print(f"Feature: d_DA (Å) vs Ω_gating (cm⁻¹)")
        print(f"  Pearson r = {r:.3f} (p = {p:.4f})")
        print(f"  Spearman ρ = {rho:.3f} (p = {p_rho:.4f})")
        print(f"  Linear: Ω = {reg['slope']:.1f} × d_DA + {reg['intercept']:.1f}")
        print(f"  R² = {reg['r_squared']:.3f}, RMSE = {reg['rmse']:.1f} cm⁻¹, LOO-RMSE = {loo:.1f} cm⁻¹")
        print()

    # Feature 2: M_DA vs Omega_gating
    r, p = pearson_r(M_DAs, omegas)
    rho, p_rho = spearman_rho(M_DAs, omegas)
    reg = linear_regression(M_DAs, omegas)
    loo = leave_one_out_cv(M_DAs, omegas)
    results["M_DA"] = {
        "pearson_r": r, "pearson_p": p,
        "spearman_rho": rho, "spearman_p": p_rho,
        "regression": reg, "loo_rmse": loo,
    }
    if verbose:
        print(f"Feature: M_DA (amu) vs Ω_gating (cm⁻¹)")
        print(f"  Pearson r = {r:.3f} (p = {p:.4f})")
        print(f"  Spearman ρ = {rho:.3f} (p = {p_rho:.4f})")
        print(f"  Linear: Ω = {reg['slope']:.1f} × M_DA + {reg['intercept']:.1f}")
        print(f"  R² = {reg['r_squared']:.3f}, RMSE = {reg['rmse']:.1f} cm⁻¹, LOO-RMSE = {loo:.1f} cm⁻¹")
        print()

    # Feature 3: has_metal vs Omega_gating
    r, p = pearson_r(has_metal, omegas)
    results["has_metal"] = {"pearson_r": r, "pearson_p": p}
    if verbose:
        metal_omegas = omegas[has_metal > 0.5]
        nometal_omegas = omegas[has_metal < 0.5]
        print(f"Feature: metal center vs Ω_gating")
        print(f"  With metal ({len(metal_omegas)}):    mean = {np.mean(metal_omegas):.0f} cm⁻¹ (range {np.min(metal_omegas):.0f}-{np.max(metal_omegas):.0f})")
        print(f"  Without metal ({len(nometal_omegas)}): mean = {np.mean(nometal_omegas):.0f} cm⁻¹ (range {np.min(nometal_omegas):.0f}-{np.max(nometal_omegas):.0f})")
        print(f"  Pearson r = {r:.3f} (p = {p:.4f})")
        print()

    # Feature 4: Multivariate (d_DA + M_DA)
    X = np.column_stack([d_DAs, M_DAs])
    X_aug = np.column_stack([np.ones(n), d_DAs, M_DAs])
    coeffs, res, rank, sv = np.linalg.lstsq(X_aug, omegas, rcond=None)
    y_pred = X_aug @ coeffs
    ss_res = np.sum((omegas - y_pred) ** 2)
    ss_tot = np.sum((omegas - np.mean(omegas)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = math.sqrt(ss_res / n)

    # LOO for multivariate
    loo_errs = []
    for i in range(n):
        X_train = np.delete(X_aug, i, axis=0)
        y_train = np.delete(omegas, i)
        c, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
        y_p = X_aug[i] @ c
        loo_errs.append((omegas[i] - y_p) ** 2)
    loo_mv = math.sqrt(np.mean(loo_errs))

    results["multivariate_dDA_MDA"] = {
        "intercept": float(coeffs[0]),
        "coeff_dDA": float(coeffs[1]),
        "coeff_MDA": float(coeffs[2]),
        "r_squared": float(r_sq),
        "rmse": float(rmse),
        "loo_rmse": float(loo_mv),
    }
    if verbose:
        print(f"Multivariate: Ω = {coeffs[0]:.1f} + {coeffs[1]:.1f}×d_DA + {coeffs[2]:.1f}×M_DA")
        print(f"  R² = {r_sq:.3f}, RMSE = {rmse:.1f} cm⁻¹, LOO-RMSE = {loo_mv:.1f} cm⁻¹")
        print()

    # Enzyme class grouping
    if verbose:
        print("=" * 80)
        print("ENZYME CLASS GROUPING")
        print("=" * 80)
        classes = {}
        for p in points:
            if p.enzyme_class not in classes:
                classes[p.enzyme_class] = []
            classes[p.enzyme_class].append(p)

        print(f"{'Class':<22} {'N':>3} {'Ω mean':>8} {'Ω range':>14} {'M_DA mean':>10} {'d_DA mean':>10}")
        print("-" * 70)
        for cls, members in sorted(classes.items(), key=lambda x: np.mean([m.omega_gating for m in x[1]])):
            ws = [m.omega_gating for m in members]
            ms = [m.M_DA for m in members]
            ds = [m.d_DA for m in members]
            print(f"{cls:<22} {len(members):>3} {np.mean(ws):>8.0f} "
                  f"{np.min(ws):>6.0f}-{np.max(ws):<6.0f} {np.mean(ms):>10.1f} {np.mean(ds):>10.2f}")

    # Per-system predictions table
    if verbose:
        print()
        print("=" * 80)
        print("PER-SYSTEM PREDICTIONS (multivariate model)")
        print("=" * 80)
        print(f"{'System':<18} {'Ω_pub':>8} {'Ω_pred':>8} {'error':>8} {'M_DA':>8} {'d_DA':>8}")
        print("-" * 60)
        for p in points:
            omega_pred = coeffs[0] + coeffs[1] * p.d_DA + coeffs[2] * p.M_DA
            err = omega_pred - p.omega_gating
            print(f"{p.name:<18} {p.omega_gating:>8.0f} {omega_pred:>8.0f} {err:>+8.0f} {p.M_DA:>8.1f} {p.d_DA:>8.2f}")

    return results


def predict_gating(
    d_DA: float,
    donor_atom: str = "C",
    acceptor_atom: str = "O",
    metal_center: str | None = None,
    enzyme_class: str | None = None,
) -> dict[str, float]:
    """Predict Omega_gating and M_DA from structural features.

    Uses the multivariate linear model fitted on published data.
    Falls back to enzyme-class lookup if class is provided.

    Args:
        d_DA: Donor-acceptor distance in Å.
        donor_atom: Element symbol of donor.
        acceptor_atom: Element symbol of acceptor.
        metal_center: Metal in active site (None if no metal).
        enzyme_class: Functional classification (for lookup table).

    Returns:
        Dict with omega_gating_pred, M_DA_pred, method, confidence.
    """
    # M_DA estimate from atom types
    key = (donor_atom, acceptor_atom)
    if key in REDUCED_MASSES:
        mu_DA = REDUCED_MASSES[key]
    elif (acceptor_atom, donor_atom) in REDUCED_MASSES:
        mu_DA = REDUCED_MASSES[(acceptor_atom, donor_atom)]
    else:
        # Fallback: assume carbon-like
        mu_DA = 6.0

    # If metal center, effective mass is typically larger because the metal
    # and its ligands participate in the D-A oscillation
    if metal_center in ("Fe",):
        mu_DA = 14.0  # Fe-oxo enzymes: M_DA ≈ 12-16 amu (published SLO-1 = 14)
    elif metal_center in ("Cu",):
        mu_DA = 10.0  # Cu enzymes: intermediate
    elif metal_center in ("Zn",):
        mu_DA = 8.0   # Zn enzymes

    # Fit the multivariate model (cached coefficients from analysis)
    results = analyze_gating_correlations(verbose=False)
    mv = results["multivariate_dDA_MDA"]
    omega_pred = mv["intercept"] + mv["coeff_dDA"] * d_DA + mv["coeff_MDA"] * mu_DA
    r_sq = mv["r_squared"]
    method = "multivariate_linear"

    # Override with enzyme class lookup if available and R² is poor
    if enzyme_class and enzyme_class in ENZYME_CLASS_GATING:
        cls = ENZYME_CLASS_GATING[enzyme_class]
        omega_class = cls["omega_mid"]
        mu_class = cls["M_DA_typical"]
        # Use class lookup if multivariate prediction seems unreasonable
        if omega_pred < 50 or omega_pred > 600:
            omega_pred = omega_class
            mu_DA = mu_class
            method = "enzyme_class_lookup"
            r_sq = 0.5  # rough estimate

    return {
        "omega_gating_pred": max(omega_pred, 50.0),
        "M_DA_pred": mu_DA,
        "method": method,
        "r_squared": r_sq,
        "loo_rmse": mv["loo_rmse"],
    }


def apply_to_benchmark_systems(verbose: bool = True) -> dict:
    """Apply predicted gating to all benchmark systems and compute rates.

    Returns:
        Dict mapping system name → rate results with and without predicted gating.
    """
    from pcet_engine.core.rate_engine import PCETRateEngine
    from pcet_engine.benchmarks.systems import BENCHMARK_SYSTEMS

    # Map benchmark systems to their atom types (from model_hessians.py)
    SYSTEM_ATOMS = {
        "SLO-1": ("C", "O", "Fe"), "SLO-1-L546A": ("C", "O", "Fe"),
        "SLO-1-L754A": ("C", "O", "Fe"), "SLO-1-DM": ("C", "O", "Fe"),
        "AADH": ("C", "N", None), "MADH": ("C", "N", None),
        "PHM": ("C", "O", "Cu"), "RNR": ("S", "C", None),
        "GO": ("C", "O", "Cu"), "LADH": ("C", "C", "Zn"),
        "bc1": ("O", "N", None), "CAO": ("C", "N", None),
        "DHFR": ("C", "C", None), "TSase": ("C", "C", None),
        "MAO": ("C", "N", None),
    }

    engine = PCETRateEngine(temperature=298.15)
    results = {}

    if verbose:
        print()
        print("=" * 110)
        print("BENCHMARK RATES WITH PREDICTED GATING")
        print("=" * 110)
        print(f"{'System':<12} {'k_H(no gate)':>13} {'k_H(pred gate)':>14} {'k_H(exp)':>12} "
              f"{'log_no':>8} {'log_pred':>8} {'Ω_pred':>8} {'M_pred':>8}")
        print("-" * 110)

    log_errs_no = []
    log_errs_pred = []

    for name, bm in BENCHMARK_SYSTEMS.items():
        # Get atom types for this system
        donor, acceptor, metal = SYSTEM_ATOMS.get(name, ("C", "O", None))

        # Predict gating
        gating = predict_gating(
            d_DA=bm.d_DA,
            donor_atom=donor,
            acceptor_atom=acceptor,
            metal_center=metal,
        )

        # Without gating
        r_no = engine.compute_rate(
            V_el=bm.V_el, delta_G=bm.delta_G, lambda_reorg=bm.lambda_reorg,
            omega_H=bm.omega_H, d_DA=bm.d_DA,
            r_DH=bm.r_DH, r_AH=bm.r_AH, delta_0=bm.delta_0,
        )

        # With predicted gating (use geometric δ₀)
        delta_geom = max(bm.d_DA - bm.r_DH - bm.r_AH, 0.05)
        r_pred = engine.compute_rate(
            V_el=bm.V_el, delta_G=bm.delta_G, lambda_reorg=bm.lambda_reorg,
            omega_H=bm.omega_H, d_DA=bm.d_DA,
            Omega_gating=gating["omega_gating_pred"],
            M_DA=gating["M_DA_pred"],
            r_DH=bm.r_DH, r_AH=bm.r_AH, delta_0=delta_geom,
        )

        log_no = math.log10(r_no.k_H / bm.k_H_exp) if r_no.k_H > 0 and bm.k_H_exp > 0 else 100
        log_pred = math.log10(r_pred.k_H / bm.k_H_exp) if r_pred.k_H > 0 and bm.k_H_exp > 0 else 100
        log_errs_no.append(abs(log_no))
        log_errs_pred.append(abs(log_pred))

        results[name] = {
            "k_H_no_gate": r_no.k_H,
            "k_H_pred_gate": r_pred.k_H,
            "k_H_exp": bm.k_H_exp,
            "log_err_no": log_no,
            "log_err_pred": log_pred,
            "omega_pred": gating["omega_gating_pred"],
            "M_DA_pred": gating["M_DA_pred"],
        }

        if verbose:
            print(f"{name:<12} {r_no.k_H:>13.2e} {r_pred.k_H:>14.2e} {bm.k_H_exp:>12.2e} "
                  f"{log_no:>+8.2f} {log_pred:>+8.2f} {gating['omega_gating_pred']:>8.0f} "
                  f"{gating['M_DA_pred']:>8.1f}")

    if verbose:
        print("-" * 110)
        print(f"Mean |log10(k_pred/k_exp)|:  no_gating = {np.mean(log_errs_no):.2f},  "
              f"predicted_gating = {np.mean(log_errs_pred):.2f}")
        print("=" * 110)

    return results


if __name__ == "__main__":
    analyze_gating_correlations(verbose=True)
    apply_to_benchmark_systems(verbose=True)
