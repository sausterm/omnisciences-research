"""
Publication-Quality Finite-Size Scaling Analysis

Based on GitHub Issue #55

Runs simulations at multiple lattice sizes to:
1. Precisely determine T_c via Binder cumulant crossing
2. Extract critical exponents β/ν, γ/ν
3. Determine if blanket transitions share Ising universality class
"""

import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from dataclasses import dataclass
import time


@dataclass
class ScalingResult:
    """Results for one lattice size."""
    L: int
    temperatures: List[float]
    magnetization: List[float]
    magnetization_err: List[float]
    susceptibility: List[float]
    susceptibility_err: List[float]
    binder: List[float]
    binder_err: List[float]
    individual_blanket: List[float]
    collective_blanket: List[float]


class IsingModelOptimized:
    """
    Optimized 2D Ising model for large-scale simulations.

    Uses vectorized operations where possible.
    """

    def __init__(self, L: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.L = L
        self.N = L * L
        self.spins = np.random.choice([-1, 1], size=(L, L))

        # Precompute neighbor indices for periodic boundaries
        self._setup_neighbors()

        # Precompute Boltzmann factors for Metropolis
        self._boltzmann = {}

    def _setup_neighbors(self):
        """Precompute neighbor index arrays."""
        L = self.L
        i_idx = np.arange(L)

        self.up = np.roll(i_idx, -1)
        self.down = np.roll(i_idx, 1)
        self.left = np.roll(i_idx, 1)
        self.right = np.roll(i_idx, -1)

    def _precompute_boltzmann(self, T: float):
        """Precompute exp(-dE/T) for possible dE values."""
        self._boltzmann = {}
        for dE in [-8, -4, 0, 4, 8]:
            if dE > 0:
                self._boltzmann[dE] = np.exp(-dE / T)
            else:
                self._boltzmann[dE] = 1.0

    def neighbor_sum(self, i: int, j: int) -> int:
        """Sum of neighboring spins."""
        L = self.L
        return (self.spins[(i-1) % L, j] +
                self.spins[(i+1) % L, j] +
                self.spins[i, (j-1) % L] +
                self.spins[i, (j+1) % L])

    def sweep(self, T: float):
        """One Monte Carlo sweep using single-spin flip."""
        L = self.L

        for _ in range(self.N):
            i = np.random.randint(L)
            j = np.random.randint(L)

            s = self.spins[i, j]
            neighbors = self.neighbor_sum(i, j)
            dE = 2 * s * neighbors

            if dE <= 0 or np.random.random() < self._boltzmann.get(dE, np.exp(-dE/T)):
                self.spins[i, j] = -s

    def measure(self) -> Tuple[float, float, float, float]:
        """
        Measure observables.

        Returns: (m, m2, m4, blanket_individual, blanket_collective)
        """
        m = np.mean(self.spins)
        m2 = m * m
        m4 = m2 * m2

        # Blanket metrics
        # Individual: local heterogeneity
        local_var = 0.0
        for i in range(self.L):
            for j in range(self.L):
                neighbors = [
                    self.spins[(i-1) % self.L, j],
                    self.spins[(i+1) % self.L, j],
                    self.spins[i, (j-1) % self.L],
                    self.spins[i, (j+1) % self.L],
                ]
                local = [self.spins[i, j]] + neighbors
                local_var += np.var(local)
        individual = local_var / self.N

        # Collective: global uniformity
        collective = 1.0 - np.var(self.spins.flatten())

        return m, m2, m4, individual, collective


def run_single_size(
    L: int,
    T_min: float = 2.0,
    T_max: float = 2.6,
    n_temps: int = 20,
    n_equilibrate: int = 5000,
    n_measure: int = 20000,
    seed: int = 42,
    verbose: bool = True
) -> ScalingResult:
    """
    Run simulation for a single lattice size.
    """
    temperatures = np.linspace(T_min, T_max, n_temps)

    mag_mean = []
    mag_err = []
    chi_mean = []
    chi_err = []
    binder_mean = []
    binder_err = []
    indiv_blanket = []
    coll_blanket = []

    for ti, T in enumerate(temperatures):
        if verbose:
            print(f"  L={L}, T={T:.3f} ({ti+1}/{n_temps})", end=" ", flush=True)

        model = IsingModelOptimized(L, seed=seed + ti)
        model._precompute_boltzmann(T)

        # Equilibrate
        for _ in range(n_equilibrate):
            model.sweep(T)

        # Measure
        m_samples = []
        m2_samples = []
        m4_samples = []
        indiv_samples = []
        coll_samples = []

        for _ in range(n_measure):
            model.sweep(T)
            m, m2, m4, indiv, coll = model.measure()
            m_samples.append(abs(m))
            m2_samples.append(m2)
            m4_samples.append(m4)
            indiv_samples.append(indiv)
            coll_samples.append(coll)

        m_arr = np.array(m_samples)
        m2_arr = np.array(m2_samples)
        m4_arr = np.array(m4_samples)

        # Magnetization
        mag_mean.append(np.mean(m_arr))
        mag_err.append(np.std(m_arr) / np.sqrt(len(m_arr)))

        # Susceptibility: χ = N * (<m²> - <|m|>²) / T
        chi = L * L * (np.mean(m2_arr) - np.mean(m_arr)**2) / T
        chi_mean.append(chi)
        # Bootstrap error estimate
        chi_bootstrap = []
        for _ in range(100):
            idx = np.random.randint(0, len(m_arr), len(m_arr))
            chi_b = L * L * (np.mean(m2_arr[idx]) - np.mean(m_arr[idx])**2) / T
            chi_bootstrap.append(chi_b)
        chi_err.append(np.std(chi_bootstrap))

        # Binder cumulant: U = 1 - <m⁴> / (3 <m²>²)
        m2_mean = np.mean(m2_arr)
        m4_mean = np.mean(m4_arr)
        if m2_mean > 1e-10:
            U = 1 - m4_mean / (3 * m2_mean**2)
        else:
            U = 0
        binder_mean.append(U)
        # Bootstrap error
        U_bootstrap = []
        for _ in range(100):
            idx = np.random.randint(0, len(m2_arr), len(m2_arr))
            m2_b = np.mean(m2_arr[idx])
            m4_b = np.mean(m4_arr[idx])
            if m2_b > 1e-10:
                U_bootstrap.append(1 - m4_b / (3 * m2_b**2))
        if U_bootstrap:
            binder_err.append(np.std(U_bootstrap))
        else:
            binder_err.append(0)

        # Blanket metrics
        indiv_blanket.append(np.mean(indiv_samples))
        coll_blanket.append(np.mean(coll_samples))

        if verbose:
            print(f"|M|={mag_mean[-1]:.3f}, χ={chi_mean[-1]:.1f}, U={binder_mean[-1]:.3f}")

    return ScalingResult(
        L=L,
        temperatures=list(temperatures),
        magnetization=mag_mean,
        magnetization_err=mag_err,
        susceptibility=chi_mean,
        susceptibility_err=chi_err,
        binder=binder_mean,
        binder_err=binder_err,
        individual_blanket=indiv_blanket,
        collective_blanket=coll_blanket,
    )


def find_binder_crossing(results: List[ScalingResult]) -> Dict:
    """
    Find T_c from Binder cumulant crossing.

    The crossing point of U(T) for different L gives T_c.
    """
    if len(results) < 2:
        return {}

    # Find crossing between consecutive sizes
    crossings = []

    for i in range(len(results) - 1):
        r1, r2 = results[i], results[i+1]

        # Interpolate to find crossing
        temps = np.array(r1.temperatures)
        u1 = np.array(r1.binder)
        u2 = np.array(r2.binder)

        # Find where curves cross
        diff = u1 - u2
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        for idx in sign_changes:
            # Linear interpolation
            t1, t2 = temps[idx], temps[idx + 1]
            d1, d2 = diff[idx], diff[idx + 1]

            if d2 != d1:
                T_cross = t1 - d1 * (t2 - t1) / (d2 - d1)
                crossings.append({
                    "L1": r1.L,
                    "L2": r2.L,
                    "T_c": T_cross,
                })

    if crossings:
        T_c_mean = np.mean([c["T_c"] for c in crossings])
        T_c_std = np.std([c["T_c"] for c in crossings])
    else:
        T_c_mean = 0
        T_c_std = 0

    return {
        "crossings": crossings,
        "T_c_mean": T_c_mean,
        "T_c_std": T_c_std,
        "T_c_theory": 2.0 / np.log(1 + np.sqrt(2)),
    }


def extract_critical_exponents(results: List[ScalingResult], T_c: float) -> Dict:
    """
    Extract critical exponents from finite-size scaling.

    At T_c:
    - |M| ~ L^(-β/ν)
    - χ ~ L^(γ/ν)
    - χ_max ~ L^(γ/ν)
    """
    sizes = []
    mag_at_Tc = []
    chi_max = []

    for r in results:
        sizes.append(r.L)

        # Find value closest to T_c
        temps = np.array(r.temperatures)
        idx = np.argmin(np.abs(temps - T_c))

        mag_at_Tc.append(r.magnetization[idx])
        chi_max.append(max(r.susceptibility))

    sizes = np.array(sizes)
    mag_at_Tc = np.array(mag_at_Tc)
    chi_max = np.array(chi_max)

    # Fit log-log to extract exponents
    # log(M) = -β/ν * log(L) + const
    log_L = np.log(sizes)

    if len(sizes) >= 2:
        # Magnetization scaling
        log_M = np.log(mag_at_Tc + 1e-10)
        coeffs_M = np.polyfit(log_L, log_M, 1)
        beta_nu = -coeffs_M[0]

        # Susceptibility scaling
        log_chi = np.log(chi_max)
        coeffs_chi = np.polyfit(log_L, log_chi, 1)
        gamma_nu = coeffs_chi[0]
    else:
        beta_nu = 0
        gamma_nu = 0

    return {
        "beta_over_nu_measured": beta_nu,
        "gamma_over_nu_measured": gamma_nu,
        "beta_over_nu_theory": 0.125,  # 2D Ising
        "gamma_over_nu_theory": 1.75,   # 2D Ising
        "sizes": list(sizes),
        "mag_at_Tc": list(mag_at_Tc),
        "chi_max": list(chi_max),
    }


def run_finite_size_scaling(
    sizes: List[int] = [8, 16, 32],
    n_temps: int = 25,
    n_equilibrate: int = 5000,
    n_measure: int = 10000,
    seed: int = 42,
    save_path: str = None,
    verbose: bool = True
) -> Dict:
    """
    Run full finite-size scaling analysis.
    """
    print("=" * 60)
    print("FINITE-SIZE SCALING ANALYSIS")
    print("=" * 60)
    print(f"Lattice sizes: {sizes}")
    print(f"Temperatures: {n_temps} points in [2.0, 2.6]")
    print(f"Equilibration: {n_equilibrate} sweeps")
    print(f"Measurement: {n_measure} sweeps")
    print()

    results = []

    for L in sizes:
        print(f"\n{'='*40}")
        print(f"Running L = {L}")
        print(f"{'='*40}")

        start = time.time()
        result = run_single_size(
            L, n_temps=n_temps,
            n_equilibrate=n_equilibrate,
            n_measure=n_measure,
            seed=seed,
            verbose=verbose
        )
        elapsed = time.time() - start

        print(f"  Completed in {elapsed:.1f}s")
        results.append(result)

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    crossing = find_binder_crossing(results)
    print(f"\nBinder cumulant crossing:")
    print(f"  T_c = {crossing['T_c_mean']:.4f} ± {crossing['T_c_std']:.4f}")
    print(f"  Theory: {crossing['T_c_theory']:.4f}")
    print(f"  Deviation: {abs(crossing['T_c_mean'] - crossing['T_c_theory']) / crossing['T_c_theory'] * 100:.2f}%")

    exponents = extract_critical_exponents(results, crossing['T_c_mean'])
    print(f"\nCritical exponents:")
    print(f"  β/ν = {exponents['beta_over_nu_measured']:.3f} (theory: {exponents['beta_over_nu_theory']})")
    print(f"  γ/ν = {exponents['gamma_over_nu_measured']:.3f} (theory: {exponents['gamma_over_nu_theory']})")

    # Compile output
    output = {
        "parameters": {
            "sizes": sizes,
            "n_temps": n_temps,
            "n_equilibrate": n_equilibrate,
            "n_measure": n_measure,
            "seed": seed,
        },
        "results": [
            {
                "L": r.L,
                "temperatures": r.temperatures,
                "magnetization": r.magnetization,
                "magnetization_err": r.magnetization_err,
                "susceptibility": r.susceptibility,
                "susceptibility_err": r.susceptibility_err,
                "binder": r.binder,
                "binder_err": r.binder_err,
                "individual_blanket": r.individual_blanket,
                "collective_blanket": r.collective_blanket,
            }
            for r in results
        ],
        "binder_crossing": crossing,
        "critical_exponents": exponents,
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {save_path}")

    return output


if __name__ == "__main__":
    output_path = Path(__file__).parent / "finite_size_scaling_results.json"

    # Run with moderate settings (can increase for publication)
    run_finite_size_scaling(
        sizes=[8, 16, 32],  # Add 64, 128 for publication quality
        n_temps=25,
        n_equilibrate=3000,
        n_measure=8000,
        seed=42,
        save_path=str(output_path),
        verbose=True
    )
