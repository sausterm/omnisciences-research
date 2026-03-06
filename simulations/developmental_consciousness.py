"""
Developmental Consciousness Model

Models the emergence of consciousness during neural development by tracking
Markov blanket formation as network connectivity increases.

Based on GitHub Issue #58: Development Model: The Birth of Consciousness

Key stages:
1. Isolated neural groups (no blankets)
2. Local connections form (local blankets)
3. Long-range connections emerge (hierarchy forms)
4. Global workspace integrates (consciousness emerges)

The model uses coupled oscillators to represent neural populations,
tracking when coherent collective dynamics emerge from initially
isolated components.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path


@dataclass
class DevelopmentalStage:
    """Represents a developmental stage with its characteristics."""
    name: str
    connectivity_range: Tuple[float, float]
    description: str


DEVELOPMENTAL_STAGES = [
    DevelopmentalStage(
        "isolated",
        (0.0, 0.10),
        "Isolated neural groups with minimal connections"
    ),
    DevelopmentalStage(
        "local",
        (0.10, 0.25),
        "Local connections forming within regions"
    ),
    DevelopmentalStage(
        "long_range",
        (0.25, 0.50),
        "Long-range connections emerging between regions"
    ),
    DevelopmentalStage(
        "integrated",
        (0.50, 1.0),
        "Global workspace integration - consciousness emerges"
    ),
]


class CoupledOscillatorNetwork:
    """
    Network of coupled Kuramoto oscillators representing neural populations.

    Each oscillator has a natural frequency and phase.
    Coupling strength determines synchronization tendency.
    """

    def __init__(self, n_nodes: int, n_regions: int = 4, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.n_nodes = n_nodes
        self.n_regions = n_regions
        self.nodes_per_region = n_nodes // n_regions

        # Natural frequencies (distributed around mean)
        self.omega = np.random.normal(1.0, 0.2, n_nodes)

        # Phases (initial random)
        self.theta = np.random.uniform(0, 2 * np.pi, n_nodes)

        # Coupling matrix
        self.K = np.zeros((n_nodes, n_nodes))

        # Region assignments
        self.regions = np.zeros(n_nodes, dtype=int)
        for i in range(n_nodes):
            self.regions[i] = min(i // self.nodes_per_region, n_regions - 1)

    def get_connectivity_density(self) -> float:
        """Calculate connectivity density (fraction of possible edges)."""
        n_possible = self.n_nodes * (self.n_nodes - 1)
        n_actual = np.sum(self.K > 0)
        return n_actual / n_possible if n_possible > 0 else 0

    def set_connectivity(self, local_strength: float, long_range_strength: float):
        """
        Set coupling strengths.

        Args:
            local_strength: Coupling within regions (0-1)
            long_range_strength: Coupling between regions (0-1)
        """
        self.K = np.zeros((self.n_nodes, self.n_nodes))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue

                if self.regions[i] == self.regions[j]:
                    # Local connection
                    if np.random.random() < local_strength:
                        self.K[i, j] = np.random.uniform(0.5, 1.5)
                else:
                    # Long-range connection
                    if np.random.random() < long_range_strength:
                        self.K[i, j] = np.random.uniform(0.3, 1.0)

    def develop_to_stage(self, developmental_time: float):
        """
        Set connectivity appropriate for developmental time (0 to 1).

        Mimics biological development:
        - Early: mostly local connections
        - Later: increasing long-range connections
        """
        # Local connections develop first
        local_strength = min(1.0, developmental_time * 3)

        # Long-range connections develop later
        long_range_strength = max(0, (developmental_time - 0.2) * 1.5)

        self.set_connectivity(local_strength, long_range_strength)

    def step(self, dt: float = 0.1):
        """
        Update oscillator phases using Kuramoto dynamics.

        dθ_i/dt = ω_i + (1/N) Σ_j K_ij sin(θ_j - θ_i)
        """
        # Compute phase differences
        phase_diff = self.theta[np.newaxis, :] - self.theta[:, np.newaxis]

        # Coupling term
        coupling = np.sum(self.K * np.sin(phase_diff), axis=1) / self.n_nodes

        # Update phases
        self.theta += dt * (self.omega + coupling)

        # Keep phases in [0, 2π]
        self.theta = self.theta % (2 * np.pi)

        return self.theta.copy()

    def run(self, n_steps: int, dt: float = 0.1) -> np.ndarray:
        """Run simulation and return phase history."""
        history = np.zeros((n_steps, self.n_nodes))

        for t in range(n_steps):
            history[t] = self.step(dt)

        return history


class ConsciousnessMetrics:
    """
    Compute consciousness-related metrics for oscillator network.

    Key metrics:
    - Global synchronization (order parameter): Collective coherence
    - Metastability: Variability of synchronization over time
    - Integration: How much the system exceeds sum of regional coherence
    - Blanket structure: Boundary vs internal coherence patterns
    """

    def __init__(self, network: CoupledOscillatorNetwork):
        self.network = network

    def compute_order_parameter(self, phases: np.ndarray) -> complex:
        """
        Compute Kuramoto order parameter r*exp(i*ψ).

        r measures synchronization (0 = incoherent, 1 = perfect sync).
        ψ is the mean phase.
        """
        return np.mean(np.exp(1j * phases))

    def compute_global_sync(self, history: np.ndarray) -> float:
        """Compute time-averaged global synchronization."""
        r_values = []
        for t in range(len(history)):
            z = self.compute_order_parameter(history[t])
            r_values.append(np.abs(z))
        return np.mean(r_values)

    def compute_regional_sync(self, history: np.ndarray) -> np.ndarray:
        """Compute synchronization within each region."""
        regional_sync = []

        for region in range(self.network.n_regions):
            region_mask = self.network.regions == region
            region_phases = history[:, region_mask]

            r_values = []
            for t in range(len(history)):
                z = self.compute_order_parameter(region_phases[t])
                r_values.append(np.abs(z))

            regional_sync.append(np.mean(r_values))

        return np.array(regional_sync)

    def compute_metastability(self, history: np.ndarray) -> float:
        """
        Compute metastability (variance of order parameter over time).

        High metastability = rich dynamics, switching between states.
        Low metastability = static (either synced or not).
        """
        r_values = []
        for t in range(len(history)):
            z = self.compute_order_parameter(history[t])
            r_values.append(np.abs(z))

        return np.std(r_values)

    def compute_integration(self, history: np.ndarray) -> float:
        """
        Compute integration: Global sync exceeding sum of regional sync.

        Integration = global_sync - mean(regional_sync)

        Positive = emergent global coherence beyond regional
        Zero or negative = no integration
        """
        global_sync = self.compute_global_sync(history)
        regional_sync = self.compute_regional_sync(history)

        # Integration is the excess of global over regional
        integration = global_sync - np.mean(regional_sync)

        # Normalize to [0, 1]
        return max(0, integration)

    def compute_blanket_strength(self, history: np.ndarray) -> float:
        """
        Compute Markov blanket strength.

        For a conscious system, boundary nodes should mediate
        information flow between internal and external.

        We use coherence patterns:
        - Internal nodes should be more coherent with each other
        - Boundary nodes bridge internal to external
        """
        # Compute pairwise coherence
        n_nodes = self.network.n_nodes
        coherence = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # Phase coherence over time
                phase_diff = history[:, i] - history[:, j]
                coh = np.abs(np.mean(np.exp(1j * phase_diff)))
                coherence[i, j] = coh
                coherence[j, i] = coh

        # Identify boundary vs internal based on connectivity
        degrees = np.sum(self.network.K > 0, axis=1)
        median_degree = np.median(degrees)

        # Nodes with high within-region connections but also some between-region
        # connections are "boundary" nodes
        within_region_degree = np.zeros(n_nodes)
        between_region_degree = np.zeros(n_nodes)

        for i in range(n_nodes):
            for j in range(n_nodes):
                if self.network.K[i, j] > 0:
                    if self.network.regions[i] == self.network.regions[j]:
                        within_region_degree[i] += 1
                    else:
                        between_region_degree[i] += 1

        # Boundary nodes have both types of connections
        boundary_score = np.minimum(within_region_degree, between_region_degree)
        boundary_nodes = np.where(boundary_score > np.median(boundary_score))[0]
        internal_nodes = np.where(boundary_score <= np.median(boundary_score))[0]

        if len(boundary_nodes) < 2 or len(internal_nodes) < 2:
            return 0.0

        # Blanket strength: how much do boundary nodes mediate?
        # Internal-internal coherence should be high
        # Internal-external (across regions) should be lower
        # Boundary should bridge

        internal_coh = np.mean(coherence[np.ix_(internal_nodes, internal_nodes)])
        boundary_coh = np.mean(coherence[np.ix_(boundary_nodes, boundary_nodes)])

        # Good blanket: high internal coherence, boundary mediates
        blanket_strength = internal_coh * (1 + boundary_coh) / 2

        return blanket_strength

    def compute_consciousness_score(self, history: np.ndarray) -> float:
        """
        Compute overall consciousness score.

        Combines:
        - Integration (global coherence exceeds regional)
        - Metastability (rich dynamics)
        - Blanket structure (clear boundary)
        """
        integration = self.compute_integration(history)
        metastability = self.compute_metastability(history)
        blanket = self.compute_blanket_strength(history)
        global_sync = self.compute_global_sync(history)

        # Consciousness requires integration with rich dynamics
        # Not just full synchronization (which would be coma-like)
        # Optimal is intermediate sync with high metastability

        # Penalize both too low and too high sync
        sync_factor = 4 * global_sync * (1 - global_sync)  # Peaks at 0.5

        # Combine factors
        score = (
            0.3 * integration +
            0.3 * metastability +
            0.2 * blanket +
            0.2 * sync_factor
        )

        return score

    def full_analysis(self, n_steps: int = 2000) -> Dict:
        """Run simulation and compute all metrics."""
        # Run simulation
        history = self.network.run(n_steps)

        # Discard transient
        history = history[500:]

        # Compute metrics
        global_sync = self.compute_global_sync(history)
        regional_sync = self.compute_regional_sync(history)
        metastability = self.compute_metastability(history)
        integration = self.compute_integration(history)
        blanket = self.compute_blanket_strength(history)
        consciousness = self.compute_consciousness_score(history)
        connectivity = self.network.get_connectivity_density()

        # Determine stage
        stage = "unknown"
        for s in DEVELOPMENTAL_STAGES:
            if s.connectivity_range[0] <= connectivity < s.connectivity_range[1]:
                stage = s.name
                break
        if connectivity >= DEVELOPMENTAL_STAGES[-1].connectivity_range[0]:
            stage = DEVELOPMENTAL_STAGES[-1].name

        return {
            "connectivity_density": float(connectivity),
            "developmental_stage": stage,
            "global_sync": float(global_sync),
            "regional_sync_mean": float(np.mean(regional_sync)),
            "regional_sync_std": float(np.std(regional_sync)),
            "metastability": float(metastability),
            "integration": float(integration),
            "blanket_strength": float(blanket),
            "consciousness_score": float(consciousness),
        }


class DevelopmentalSimulation:
    """Run a full developmental simulation."""

    def __init__(
        self,
        n_nodes: int = 64,
        n_regions: int = 4,
        n_steps: int = 25,
        seed: int = 42
    ):
        self.n_nodes = n_nodes
        self.n_regions = n_regions
        self.n_steps = n_steps
        self.seed = seed
        self.results = []

    def run(self, verbose: bool = True) -> List[Dict]:
        """Run developmental simulation."""
        developmental_times = np.linspace(0.0, 1.0, self.n_steps)

        for i, dev_time in enumerate(developmental_times):
            if verbose:
                print(f"Developmental step {i+1}/{self.n_steps} (t={dev_time:.2f})")

            # Create network at this stage
            network = CoupledOscillatorNetwork(
                self.n_nodes, self.n_regions, seed=self.seed + i
            )
            network.develop_to_stage(dev_time)

            # Analyze
            metrics = ConsciousnessMetrics(network)
            result = metrics.full_analysis(n_steps=1500)
            result["developmental_time"] = float(dev_time)
            result["step"] = i

            self.results.append(result)

            if verbose:
                print(f"  Stage: {result['developmental_stage']}")
                print(f"  Global sync: {result['global_sync']:.3f}")
                print(f"  Integration: {result['integration']:.3f}")
                print(f"  Metastability: {result['metastability']:.3f}")
                print(f"  Blanket strength: {result['blanket_strength']:.3f}")
                print(f"  Consciousness score: {result['consciousness_score']:.3f}")
                print()

        return self.results

    def find_critical_threshold(self) -> Optional[Dict]:
        """Find critical point where consciousness emerges."""
        if len(self.results) < 3:
            return None

        scores = [r["consciousness_score"] for r in self.results]
        derivatives = np.diff(scores)

        # Find maximum positive derivative
        if np.max(derivatives) <= 0:
            return None

        critical_idx = int(np.argmax(derivatives) + 1)

        return {
            "critical_step": critical_idx,
            "critical_time": float(self.results[critical_idx]["developmental_time"]),
            "critical_connectivity": float(self.results[critical_idx]["connectivity_density"]),
            "critical_stage": self.results[critical_idx]["developmental_stage"],
            "pre_score": float(scores[critical_idx - 1]),
            "post_score": float(scores[critical_idx]),
            "transition_magnitude": float(derivatives[critical_idx - 1]),
        }

    def save_results(self, path: str):
        """Save results to JSON."""
        output = {
            "parameters": {
                "n_nodes": self.n_nodes,
                "n_regions": self.n_regions,
                "n_steps": self.n_steps,
                "seed": self.seed,
            },
            "results": self.results,
            "critical_threshold": self.find_critical_threshold(),
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)


def main():
    """Run developmental consciousness simulation."""
    print("=" * 70)
    print("DEVELOPMENTAL CONSCIOUSNESS MODEL")
    print("Tracking blanket formation and consciousness emergence")
    print("=" * 70)
    print()

    sim = DevelopmentalSimulation(
        n_nodes=64,
        n_regions=4,
        n_steps=25,
        seed=42
    )

    results = sim.run(verbose=True)

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    critical = sim.find_critical_threshold()
    if critical:
        print(f"\nCritical threshold detected:")
        print(f"  Developmental time: {critical['critical_time']:.2f}")
        print(f"  Connectivity: {critical['critical_connectivity']:.3f}")
        print(f"  Stage: {critical['critical_stage']}")
        print(f"  Score jump: {critical['pre_score']:.3f} -> {critical['post_score']:.3f}")

    print("\nBy developmental stage:")
    for stage in DEVELOPMENTAL_STAGES:
        stage_results = [r for r in results if r["developmental_stage"] == stage.name]
        if stage_results:
            avg_score = np.mean([r["consciousness_score"] for r in stage_results])
            avg_integration = np.mean([r["integration"] for r in stage_results])
            avg_sync = np.mean([r["global_sync"] for r in stage_results])
            print(f"\n  {stage.name.upper()}:")
            print(f"    {stage.description}")
            print(f"    Consciousness score: {avg_score:.3f}")
            print(f"    Integration: {avg_integration:.3f}")
            print(f"    Global sync: {avg_sync:.3f}")

    # Save
    output_path = Path(__file__).parent / "developmental_results.json"
    sim.save_results(str(output_path))
    print(f"\nResults saved to: {output_path}")

    return sim


if __name__ == "__main__":
    main()
