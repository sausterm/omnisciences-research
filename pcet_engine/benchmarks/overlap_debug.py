"""
Deep dive into the overlap integral and its R-dependence.

Tests:
1. Finite difference convergence (is h=0.01 Å too large?)
2. Grid convergence (are 2048 points enough?)
3. Direct comparison of wavefunctions in the overlap region
4. Analytical S(R) vs numerical S(R) for ground state Morse
"""

import math
import numpy as np
from scipy.special import genlaguerre, gamma as gamma_func
from pcet_engine.core.constants import (
    ANGSTROM_TO_BOHR, AMU_TO_AU, KCALMOL_TO_HARTREE, HBAR_AU,
    CM_TO_HARTREE, HARTREE_TO_CM,
)
from pcet_engine.benchmarks.analytical_morse_test import (
    compute_overlap_analytical, morse_lambda,
    BETA_CH, BETA_OH, D_CH, D_OH, R_CH, R_OH, R_DA, MASS_H,
    analytical_morse_wavefunction,
)
from pcet_engine.benchmarks.soudackov_correction import soudackov_reference_params


def test_finite_difference_convergence():
    """Test if the finite difference step size affects α."""
    print("=" * 70)
    print("TEST 1: Finite difference convergence")
    print("=" * 70)

    delta_0 = R_DA - R_CH - R_OH
    h_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

    ref_aH, _, _, _ = soudackov_reference_params()

    print(f"\n  {'h (Å)':<10} {'α_H(0,0)':<14} {'ratio to Soud':<14}")
    print(f"  {'-'*38}")

    for h in h_values:
        h_bohr = h * ANGSTROM_TO_BOHR
        S_m = compute_overlap_analytical(MASS_H, delta_0 - h, mu=0, nu=0)
        S_0 = compute_overlap_analytical(MASS_H, delta_0, mu=0, nu=0)
        S_p = compute_overlap_analytical(MASS_H, delta_0 + h, mu=0, nu=0)

        log_m = 0.5 * math.log(S_m) if S_m > 1e-300 else -350.0
        log_0 = 0.5 * math.log(S_0) if S_0> 1e-300 else -350.0
        log_p = 0.5 * math.log(S_p) if S_p > 1e-300 else -350.0

        alpha = -(log_p - log_m) / (2.0 * h_bohr)
        gamma = -(log_p - 2.0 * log_0 + log_m) / h_bohr**2
        ratio = alpha / ref_aH[0, 0]
        print(f"  {h:<10.4f} {alpha:<14.4f} {ratio:<14.3f}")


def test_grid_convergence():
    """Test if grid size affects the overlap."""
    print()
    print("=" * 70)
    print("TEST 2: Grid convergence")
    print("=" * 70)

    delta_0 = R_DA - R_CH - R_OH
    n_grids = [512, 1024, 2048, 4096, 8192]
    paddings = [3.0, 5.0, 7.0, 10.0]

    print(f"\n  Grid convergence (padding = 5.0 Å):")
    print(f"  {'n_grid':<10} {'S²(0,0)':<14} {'α_H(0,0)':<14}")
    print(f"  {'-'*38}")

    for n_grid in n_grids:
        from pcet_engine.benchmarks.analytical_morse_test import compute_overlap_analytical as coa
        # Monkey-patch n_grid
        S = coa(MASS_H, delta_0, n_grid=n_grid, mu=0, nu=0)
        print(f"  {n_grid:<10} {S:<14.6e}")

    print(f"\n  Padding convergence (n_grid = 4096):")
    print(f"  {'padding':<10} {'S²(0,0)':<14}")
    print(f"  {'-'*24}")

    for pad in paddings:
        S = coa(MASS_H, delta_0, n_grid=4096, padding=pad, mu=0, nu=0)
        print(f"  {pad:<10.1f} {S:<14.6e}")


def test_wavefunction_tails():
    """Inspect the wavefunctions in the overlap region."""
    print()
    print("=" * 70)
    print("TEST 3: Wavefunction tails in overlap region")
    print("=" * 70)

    delta_0 = R_DA - R_CH - R_OH
    delta_bohr = delta_0 * ANGSTROM_TO_BOHR
    n_grid = 4096
    padding = 5.0

    x_min = -padding * ANGSTROM_TO_BOHR
    x_max = delta_bohr + padding * ANGSTROM_TO_BOHR
    grid = np.linspace(x_min, x_max, n_grid)

    psi_R, E_R = analytical_morse_wavefunction(grid, BETA_CH, D_CH, MASS_H, 0.0, n_state=0)
    psi_P, E_P = analytical_morse_wavefunction(grid, BETA_OH, D_OH, MASS_H, delta_bohr, n_state=0, reflected=True)

    # Find the overlap region (where both wavefunctions are significant)
    product = psi_R * psi_P
    midpoint = delta_bohr / 2.0

    # Sample at midpoint and a few other locations
    mid_idx = np.argmin(np.abs(grid - midpoint))
    print(f"\n  Overlap region (δ₀ = {delta_0:.2f} Å = {delta_bohr:.3f} bohr):")
    print(f"  Midpoint x = {midpoint:.3f} bohr = {midpoint/ANGSTROM_TO_BOHR:.3f} Å")
    print(f"  ψ_R(mid) = {psi_R[mid_idx]:.6e}, ψ_P(mid) = {psi_P[mid_idx]:.6e}")
    print(f"  product = {product[mid_idx]:.6e}")

    dx = grid[1] - grid[0]
    overlap = np.sum(product) * dx
    print(f"  S = {overlap:.6e}, S² = {overlap**2:.6e}")

    # Check Morse variable z at different points
    print(f"\n  Morse variable z values:")
    for x_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = x_frac * delta_bohr
        idx = np.argmin(np.abs(grid - x))
        z_R = 2.0 * morse_lambda(MASS_H, D_CH, BETA_CH) * math.exp(-BETA_CH * x)
        z_P = 2.0 * morse_lambda(MASS_H, D_OH, BETA_OH) * math.exp(BETA_OH * (x - delta_bohr))
        print(f"  x = {x_frac:.2f}δ: z_R = {z_R:.4f}, z_P = {z_P:.4f}, "
              f"ψ_R = {psi_R[idx]:.4e}, ψ_P = {psi_P[idx]:.4e}")


def test_log_overlap_vs_delta():
    """Plot log|S(δ)| vs δ to check linearity (should be roughly linear if α is constant)."""
    print()
    print("=" * 70)
    print("TEST 4: log|S| vs δ (linearity check)")
    print("=" * 70)

    deltas = np.arange(0.50, 1.01, 0.02)
    log_S = []

    ref_aH, _, _, _ = soudackov_reference_params()

    for d in deltas:
        S_sq = compute_overlap_analytical(MASS_H, d, mu=0, nu=0, n_grid=4096)
        log_s = 0.5 * math.log(S_sq) if S_sq > 1e-300 else -350.0
        log_S.append(log_s)

    log_S = np.array(log_S)
    deltas_bohr = deltas * ANGSTROM_TO_BOHR

    print(f"\n  {'δ (Å)':<10} {'δ (bohr)':<12} {'log|S|':<14}")
    print(f"  {'-'*36}")
    for i in range(len(deltas)):
        print(f"  {deltas[i]:<10.2f} {deltas_bohr[i]:<12.3f} {log_S[i]:<14.4f}")

    # Fit a line: log|S| ≈ -α δ + const
    # Use central values for fit
    mask = np.isfinite(log_S)
    if np.sum(mask) > 2:
        p = np.polyfit(deltas_bohr[mask], log_S[mask], 2)
        alpha_fit = -p[1] - 2 * p[0] * (0.72 * ANGSTROM_TO_BOHR)  # slope at δ₀
        gamma_fit = -2 * p[0]  # curvature
        print(f"\n  Quadratic fit: α(δ₀) = {alpha_fit:.4f} bohr⁻¹, γ = {gamma_fit:.4f} bohr⁻²")
        print(f"  Soudackov: α = {ref_aH[0,0]:.4f}, γ = 5.6992")
        print(f"  Ratio: {alpha_fit / ref_aH[0,0]:.3f}")

    # Also try a broader range fit to check if α changes significantly
    print(f"\n  Local slopes (centered differences):")
    for i in range(1, len(deltas) - 1):
        if log_S[i-1] > -300 and log_S[i+1] > -300:
            h = (deltas_bohr[i+1] - deltas_bohr[i-1]) / 2.0
            local_alpha = -(log_S[i+1] - log_S[i-1]) / (2.0 * h)
            print(f"  δ = {deltas[i]:.2f} Å: α_local = {local_alpha:.4f}")


def main():
    test_finite_difference_convergence()
    test_grid_convergence()
    test_wavefunction_tails()
    test_log_overlap_vs_delta()


if __name__ == "__main__":
    main()
