"""
Blanket Ising Model for Consciousness Phase Transitions

Based on GitHub Issue #54: DISCOVERY: Blanket Phase Transitions and the Combination Problem

This model demonstrates that:
1. Below critical temperature: Individual blankets dissolve, COLLECTIVE blanket dominates
2. Above critical temperature: Individual blankets emerge, agents maintain distinct identity
3. At critical point: Maximum uncertainty about the "correct" level of description

Connection to developmental model:
- "Temperature" maps to inverse connectivity/integration
- Low T (high integration) → collective consciousness
- High T (low integration) → fragmented individual consciousness
- Critical T → consciousness emergence threshold
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class IsingAgent:
    """
    An agent in the Ising model.

    Each agent has:
    - spin: Current belief/state (+1 or -1)
    - confidence: How strongly agent holds its belief (0 to 1)
    - internal_state: Hidden variable affecting dynamics
    """
    spin: int
    confidence: float
    internal_state: float


class BlanketIsingModel:
    """
    2D Ising model with Markov blanket interpretation.

    Each lattice site represents a conscious agent.
    Coupling between sites represents blanket permeability.

    Key insight: The phase transition in Ising model corresponds
    to a transition in blanket structure:
    - Ordered phase (low T): Strong correlations → dissolved boundaries
    - Disordered phase (high T): Weak correlations → individual boundaries
    """

    def __init__(
        self,
        L: int = 16,
        J: float = 1.0,
        seed: int = None
    ):
        """
        Initialize Ising lattice.

        Args:
            L: Linear size (L×L lattice)
            J: Coupling strength
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        self.L = L
        self.N = L * L
        self.J = J

        # Initialize spins randomly
        self.spins = np.random.choice([-1, 1], size=(L, L))

        # Agent confidence (affects dynamics)
        self.confidence = np.random.uniform(0.5, 1.0, size=(L, L))

        # Internal states (hidden variables)
        self.internal = np.random.randn(L, L) * 0.1

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get periodic boundary neighbor indices."""
        L = self.L
        return [
            ((i - 1) % L, j),
            ((i + 1) % L, j),
            (i, (j - 1) % L),
            (i, (j + 1) % L),
        ]

    def local_energy(self, i: int, j: int) -> float:
        """Compute energy contribution from site (i,j)."""
        s = self.spins[i, j]
        neighbor_sum = sum(self.spins[ni, nj] for ni, nj in self.get_neighbors(i, j))
        return -self.J * s * neighbor_sum

    def total_energy(self) -> float:
        """Compute total energy."""
        E = 0.0
        for i in range(self.L):
            for j in range(self.L):
                E += self.local_energy(i, j)
        return E / 2  # Avoid double counting

    def magnetization(self) -> float:
        """Compute magnetization (order parameter)."""
        return np.mean(self.spins)

    def metropolis_step(self, T: float):
        """
        Single Metropolis-Hastings update step.

        Args:
            T: Temperature
        """
        # Pick random site
        i = np.random.randint(self.L)
        j = np.random.randint(self.L)

        # Current energy
        E_old = self.local_energy(i, j)

        # Proposed flip
        self.spins[i, j] *= -1
        E_new = self.local_energy(i, j)

        # Energy change
        dE = E_new - E_old

        # Accept or reject
        if dE > 0:
            if np.random.random() > np.exp(-dE / T):
                # Reject: flip back
                self.spins[i, j] *= -1

    def sweep(self, T: float):
        """One Monte Carlo sweep (N updates)."""
        for _ in range(self.N):
            self.metropolis_step(T)

    def equilibrate(self, T: float, n_sweeps: int = 1000):
        """Equilibrate at temperature T."""
        for _ in range(n_sweeps):
            self.sweep(T)

    def measure(self, T: float, n_sweeps: int = 5000) -> Dict:
        """
        Measure observables after equilibration.

        Returns magnetization, energy, and blanket metrics.
        """
        M_samples = []
        E_samples = []
        blanket_samples = []

        for _ in range(n_sweeps):
            self.sweep(T)

            M_samples.append(self.magnetization())
            E_samples.append(self.total_energy() / self.N)
            blanket_samples.append(self.compute_blanket_index())

        M = np.array(M_samples)
        E = np.array(E_samples)
        B = np.array(blanket_samples)

        return {
            "temperature": T,
            "magnetization_mean": float(np.mean(np.abs(M))),
            "magnetization_std": float(np.std(M)),
            "energy_mean": float(np.mean(E)),
            "energy_std": float(np.std(E)),
            "susceptibility": float(self.N * np.var(M) / T),
            "specific_heat": float(self.N * np.var(E) / (T * T)),
            "individual_blanket": float(np.mean(B[:, 0])),
            "collective_blanket": float(np.mean(B[:, 1])),
            "blanket_ratio": float(np.mean(B[:, 0]) / (np.mean(B[:, 1]) + 1e-10)),
        }

    def compute_blanket_index(self) -> Tuple[float, float]:
        """
        Compute individual and collective blanket indices.

        Individual blanket: How well each agent is shielded from others
        Collective blanket: How well the whole system acts as one unit

        Returns (individual_index, collective_index)
        """
        # Individual blanket strength: variance within local neighborhoods
        # High variance = strong individual boundaries
        individual_index = 0.0

        for i in range(self.L):
            for j in range(self.L):
                neighbors = [self.spins[ni, nj]
                            for ni, nj in self.get_neighbors(i, j)]
                # Local heterogeneity
                local_var = np.var([self.spins[i, j]] + neighbors)
                individual_index += local_var

        individual_index /= self.N

        # Collective blanket strength: global correlation
        # Low variance globally = strong collective boundary
        collective_index = 1.0 - np.var(self.spins.flatten())

        return individual_index, collective_index

    def compute_domain_structure(self) -> Dict:
        """
        Analyze domain structure (clusters of aligned spins).

        Domains represent "cognitive units" - regions acting together.
        """
        from scipy import ndimage

        # Label connected regions of +1 spins
        labeled_up, n_up = ndimage.label(self.spins > 0)

        # Label connected regions of -1 spins
        labeled_down, n_down = ndimage.label(self.spins < 0)

        # Domain sizes
        up_sizes = [np.sum(labeled_up == k) for k in range(1, n_up + 1)]
        down_sizes = [np.sum(labeled_down == k) for k in range(1, n_down + 1)]

        all_sizes = up_sizes + down_sizes

        return {
            "n_domains": n_up + n_down,
            "mean_domain_size": np.mean(all_sizes) if all_sizes else 0,
            "max_domain_size": max(all_sizes) if all_sizes else 0,
            "domain_size_std": np.std(all_sizes) if all_sizes else 0,
        }


class PhaseTransitionAnalysis:
    """
    Analyze phase transitions in the blanket Ising model.
    """

    def __init__(self, L: int = 16, J: float = 1.0, seed: int = 42):
        self.L = L
        self.J = J
        self.seed = seed

        # Theoretical critical temperature for 2D Ising
        self.T_c_theory = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269

    def temperature_sweep(
        self,
        T_min: float = 1.5,
        T_max: float = 3.5,
        n_temps: int = 20,
        n_equilibrate: int = 2000,
        n_measure: int = 5000,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Sweep through temperatures and measure observables.
        """
        temperatures = np.linspace(T_min, T_max, n_temps)
        results = []

        for i, T in enumerate(temperatures):
            if verbose:
                print(f"T = {T:.3f} ({i+1}/{n_temps})")

            model = BlanketIsingModel(self.L, self.J, self.seed + i)
            model.equilibrate(T, n_equilibrate)
            data = model.measure(T, n_measure)

            results.append(data)

            if verbose:
                print(f"  |M| = {data['magnetization_mean']:.3f}")
                print(f"  Individual blanket = {data['individual_blanket']:.3f}")
                print(f"  Collective blanket = {data['collective_blanket']:.3f}")

        return results

    def find_critical_temperature(self, results: List[Dict]) -> Dict:
        """
        Estimate critical temperature from measurements.

        Uses multiple methods:
        1. Peak in susceptibility
        2. Peak in specific heat
        3. Blanket crossover point
        """
        temps = [r["temperature"] for r in results]
        chi = [r["susceptibility"] for r in results]
        cv = [r["specific_heat"] for r in results]
        indiv = [r["individual_blanket"] for r in results]
        coll = [r["collective_blanket"] for r in results]

        # Susceptibility peak
        T_c_chi = temps[np.argmax(chi)]

        # Specific heat peak
        T_c_cv = temps[np.argmax(cv)]

        # Blanket crossover (where individual = collective)
        diff = np.array(indiv) - np.array(coll)
        crossover_idx = np.argmin(np.abs(diff))
        T_c_blanket = temps[crossover_idx]

        return {
            "T_c_susceptibility": T_c_chi,
            "T_c_specific_heat": T_c_cv,
            "T_c_blanket_crossover": T_c_blanket,
            "T_c_theory": self.T_c_theory,
            "T_c_consensus": np.mean([T_c_chi, T_c_cv, T_c_blanket]),
        }

    def binder_cumulant(self, results: List[Dict]) -> List[float]:
        """
        Compute Binder cumulant U = 1 - <m^4>/(3<m^2>^2).

        Crossing point of U(T) for different L gives T_c.
        """
        binder = []
        for r in results:
            m2 = r["magnetization_mean"] ** 2
            # Approximate m4 from variance
            m4 = m2 ** 2 * (1 + 6 * r["magnetization_std"] ** 2 / m2)
            if m2 > 1e-10:
                u = 1 - m4 / (3 * m2 ** 2)
            else:
                u = 0
            binder.append(u)
        return binder


class DevelopmentalConnection:
    """
    Connect Ising model to developmental consciousness model.

    Key mapping:
    - Temperature ↔ Inverse integration/connectivity
    - Low T (ordered) ↔ High connectivity (integrated consciousness)
    - High T (disordered) ↔ Low connectivity (fragmented)
    - Critical T ↔ Consciousness emergence threshold
    """

    def __init__(self, ising_results: List[Dict], dev_results: List[Dict]):
        self.ising = ising_results
        self.dev = dev_results

    def map_temperature_to_development(self) -> Dict:
        """
        Create mapping between Ising temperature and developmental time.
        """
        # In Ising: low T = ordered (collective), high T = disordered (individual)
        # In development: early = fragmented, late = integrated

        # So inverse relationship: T ∝ 1 - developmental_time

        ising_temps = [r["temperature"] for r in self.ising]
        ising_mag = [r["magnetization_mean"] for r in self.ising]

        dev_times = [r["developmental_time"] for r in self.dev]
        dev_sync = [r["global_sync"] for r in self.dev]

        # Find T that gives similar magnetization to developmental sync
        mapping = []
        for dev_time, sync in zip(dev_times, dev_sync):
            # Find closest magnetization
            closest_idx = np.argmin(np.abs(np.array(ising_mag) - sync))
            mapping.append({
                "developmental_time": dev_time,
                "effective_temperature": ising_temps[closest_idx],
                "sync_dev": sync,
                "mag_ising": ising_mag[closest_idx],
            })

        return {
            "mapping": mapping,
            "interpretation": """
            Low effective temperature (high magnetization) corresponds to
            late developmental stages with high synchronization/integration.

            High effective temperature (low magnetization) corresponds to
            early developmental stages with fragmented activity.

            The critical temperature maps to the consciousness emergence
            threshold in development.
            """,
        }

    def compare_phase_transitions(self, ising_critical: Dict, dev_critical: Dict) -> Dict:
        """
        Compare critical points between Ising and developmental models.
        """
        return {
            "ising_T_c": ising_critical.get("T_c_consensus"),
            "dev_t_c": dev_critical.get("consensus_time"),
            "comparison": """
            Both models show phase transition behavior:

            ISING MODEL:
            - Below T_c: Collective order (one cognitive unit)
            - Above T_c: Individual disorder (many cognitive units)
            - At T_c: Critical fluctuations, maximum susceptibility

            DEVELOPMENTAL MODEL:
            - Early development: Fragmented (like high T Ising)
            - Critical point: Consciousness emerges
            - Late development: Integrated but pruned (not extreme low T)

            KEY INSIGHT: The developmental trajectory passes THROUGH the
            critical region, not to the extreme ordered state. This explains
            why consciousness is at intermediate integration, not maximum.
            """,
        }


def run_phase_transition_analysis(
    L: int = 16,
    n_temps: int = 25,
    save_path: str = None,
    verbose: bool = True
) -> Dict:
    """
    Run full phase transition analysis.
    """
    print("=" * 60)
    print("BLANKET ISING MODEL: Phase Transition Analysis")
    print("=" * 60)
    print(f"\nLattice size: {L}×{L}")
    print(f"Theoretical T_c: {2.0 / np.log(1 + np.sqrt(2)):.3f}")
    print()

    analyzer = PhaseTransitionAnalysis(L=L, seed=42)
    results = analyzer.temperature_sweep(
        T_min=1.5, T_max=3.5, n_temps=n_temps,
        verbose=verbose
    )

    critical = analyzer.find_critical_temperature(results)
    binder = analyzer.binder_cumulant(results)

    print("\n" + "=" * 60)
    print("CRITICAL TEMPERATURE ESTIMATES")
    print("=" * 60)
    print(f"From susceptibility peak: T_c = {critical['T_c_susceptibility']:.3f}")
    print(f"From specific heat peak:  T_c = {critical['T_c_specific_heat']:.3f}")
    print(f"From blanket crossover:   T_c = {critical['T_c_blanket_crossover']:.3f}")
    print(f"Theoretical:              T_c = {critical['T_c_theory']:.3f}")
    print(f"Consensus estimate:       T_c = {critical['T_c_consensus']:.3f}")

    print("\n" + "=" * 60)
    print("BLANKET STRUCTURE BY PHASE")
    print("=" * 60)

    # Low T (ordered)
    low_T = [r for r in results if r["temperature"] < critical["T_c_consensus"] - 0.3]
    if low_T:
        print(f"\nLOW T (T < {critical['T_c_consensus'] - 0.3:.2f}) - ORDERED PHASE:")
        print(f"  |M| = {np.mean([r['magnetization_mean'] for r in low_T]):.3f}")
        print(f"  Individual blanket = {np.mean([r['individual_blanket'] for r in low_T]):.3f}")
        print(f"  Collective blanket = {np.mean([r['collective_blanket'] for r in low_T]):.3f}")
        print("  → Collective consciousness dominates")

    # High T (disordered)
    high_T = [r for r in results if r["temperature"] > critical["T_c_consensus"] + 0.3]
    if high_T:
        print(f"\nHIGH T (T > {critical['T_c_consensus'] + 0.3:.2f}) - DISORDERED PHASE:")
        print(f"  |M| = {np.mean([r['magnetization_mean'] for r in high_T]):.3f}")
        print(f"  Individual blanket = {np.mean([r['individual_blanket'] for r in high_T]):.3f}")
        print(f"  Collective blanket = {np.mean([r['collective_blanket'] for r in high_T]):.3f}")
        print("  → Individual blankets dominate")

    # Critical
    crit = [r for r in results
            if abs(r["temperature"] - critical["T_c_consensus"]) < 0.2]
    if crit:
        print(f"\nCRITICAL (T ≈ {critical['T_c_consensus']:.2f}):")
        print(f"  |M| = {np.mean([r['magnetization_mean'] for r in crit]):.3f}")
        print(f"  Susceptibility = {np.mean([r['susceptibility'] for r in crit]):.3f}")
        print("  → Maximum uncertainty, phase coexistence")

    output = {
        "parameters": {"L": L, "n_temps": n_temps},
        "results": results,
        "critical_temperatures": critical,
        "binder_cumulant": binder,
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {save_path}")

    return output


if __name__ == "__main__":
    output_path = Path(__file__).parent / "ising_phase_results.json"
    run_phase_transition_analysis(L=16, n_temps=25, save_path=str(output_path))
