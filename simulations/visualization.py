"""
Visualization module for Developmental Consciousness simulations.

Creates publication-quality figures for:
- Consciousness score trajectories
- Phase transition indicators
- Network connectivity evolution
- Metric correlations
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_results(path: str) -> Dict:
    """Load simulation results from JSON."""
    with open(path) as f:
        return json.load(f)


def plot_consciousness_trajectory(
    results: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot consciousness score over developmental time.

    Shows total score and component contributions.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return

    times = [r["developmental_time"] for r in results]

    # Extract metrics
    total = [r["total"] for r in results]
    integration = [r.get("integration", 0) * 5 for r in results]  # Scaled
    complexity = [r.get("complexity", 0) for r in results]
    metastability = [r.get("metastability", 0) * 3 for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top plot: Total consciousness score
    ax1 = axes[0]
    ax1.plot(times, total, 'k-', linewidth=2.5, label='Consciousness Score')
    ax1.fill_between(times, 0, total, alpha=0.3, color='purple')

    # Mark critical point (max derivative)
    deriv = np.diff(total)
    critical_idx = np.argmax(deriv) + 1
    ax1.axvline(times[critical_idx], color='red', linestyle='--',
                alpha=0.7, label=f'Critical point (t={times[critical_idx]:.2f})')

    ax1.set_ylabel('Consciousness Score', fontsize=12)
    ax1.set_title('Consciousness Emergence During Development', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Add stage annotations
    stages = ["Isolated", "Local", "Long-range", "Integrated"]
    stage_boundaries = [0, 0.1, 0.25, 0.5, 1.0]
    colors = ['#ffcccc', '#ffffcc', '#ccffcc', '#ccccff']

    for i, (stage, color) in enumerate(zip(stages, colors)):
        ax1.axvspan(stage_boundaries[i], stage_boundaries[i+1],
                   alpha=0.2, color=color)
        ax1.text((stage_boundaries[i] + stage_boundaries[i+1]) / 2,
                ax1.get_ylim()[1] * 0.95, stage,
                ha='center', fontsize=9, alpha=0.7)

    # Bottom plot: Component breakdown
    ax2 = axes[1]
    ax2.plot(times, integration, '-', linewidth=1.5, label='Integration (×5)')
    ax2.plot(times, complexity, '-', linewidth=1.5, label='Complexity')
    ax2.plot(times, metastability, '-', linewidth=1.5, label='Metastability (×3)')

    ax2.set_xlabel('Developmental Time', fontsize=12)
    ax2.set_ylabel('Component Scores', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_phase_transition(
    results: List[Dict],
    phase_data: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot phase transition indicators.

    Shows susceptibility, Binder cumulant, and critical point detection.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return

    times = [r["developmental_time"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Order parameter (global sync)
    ax1 = axes[0, 0]
    sync = [r["global_sync"] for r in results]
    ax1.plot(times, sync, 'b-', linewidth=2)
    ax1.set_ylabel('Order Parameter (r)', fontsize=11)
    ax1.set_title('Global Synchronization', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Susceptibility
    ax2 = axes[0, 1]
    suscept = phase_data.get("susceptibility", [0] * len(results))
    ax2.plot(times, suscept, 'g-', linewidth=2)
    ax2.set_ylabel('Susceptibility (χ)', fontsize=11)
    ax2.set_title('Susceptibility (Peaks at Phase Transition)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Mark peak
    peak_idx = np.argmax(suscept)
    ax2.axvline(times[peak_idx], color='red', linestyle='--', alpha=0.7)

    # 3. Binder cumulant
    ax3 = axes[1, 0]
    binder = phase_data.get("binder_cumulant", [0] * len(results))
    ax3.plot(times, binder, 'm-', linewidth=2)
    ax3.set_xlabel('Developmental Time', fontsize=11)
    ax3.set_ylabel('Binder Cumulant (U)', fontsize=11)
    ax3.set_title('Binder Cumulant', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. Critical indicators comparison
    ax4 = axes[1, 1]

    indicators = phase_data.get("critical_indices", {})
    if indicators:
        names = list(indicators.keys())
        indices = [indicators[n] for n in names]

        # Convert to times
        indicator_times = [times[min(i, len(times)-1)] for i in indices]

        ax4.barh(range(len(names)), indicator_times, color='steelblue', alpha=0.7)
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels([n.replace('_', ' ').title() for n in names])
        ax4.set_xlabel('Developmental Time', fontsize=11)
        ax4.set_title('Critical Point Indicators', fontsize=12)

        # Consensus line
        consensus = phase_data.get("consensus_time", 0)
        ax4.axvline(consensus, color='red', linestyle='--', linewidth=2,
                   label=f'Consensus: t={consensus:.2f}')
        ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_biological_stages(
    results: List[Dict],
    stages: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot consciousness metrics mapped to biological stages.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get gestational weeks
    weeks = [r.get("gestational_weeks", r["developmental_time"] * 96 + 8)
             for r in results]
    scores = [r["total"] for r in results]

    # Main line
    ax.plot(weeks, scores, 'k-', linewidth=2, label='Consciousness Score')
    ax.fill_between(weeks, 0, scores, alpha=0.3, color='purple')

    # Stage backgrounds
    colors = ['#fee0d2', '#deebf7', '#e5f5e0', '#f2f0f7', '#fff7bc']

    for i, stage in enumerate(stages):
        w1, w2 = stage["weeks"]
        ax.axvspan(w1, w2, alpha=0.3, color=colors[i % len(colors)])

        # Stage label
        mid = (w1 + w2) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.9,
               stage["name"].replace("_", " ").title(),
               ha='center', fontsize=10, fontweight='bold')

        # Key events
        events_text = "\n".join(stage.get("events", [])[:2])
        if events_text:
            ax.text(mid, ax.get_ylim()[1] * 0.1, events_text,
                   ha='center', fontsize=7, alpha=0.7,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Mark key milestones
    ax.axvline(24, color='orange', linestyle=':', alpha=0.7, label='Viability (~24w)')
    ax.axvline(40, color='green', linestyle=':', alpha=0.7, label='Term birth (~40w)')

    ax.set_xlabel('Gestational Weeks', fontsize=12)
    ax.set_ylabel('Consciousness Score', fontsize=12)
    ax.set_title('Consciousness Emergence Across Gestational Development', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(8, 104)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_correlations(
    results: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot correlations between different consciousness metrics.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return

    metrics = ["global_sync", "integration", "complexity", "metastability",
               "phi_proxy", "blanket_strength"]
    labels = ["Global Sync", "Integration", "Complexity", "Metastability",
              "Φ Proxy", "Blanket"]

    # Extract data
    data = {}
    for m in metrics:
        data[m] = [r.get(m, 0) for r in results]

    n = len(metrics)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                ax.hist(data[metrics[i]], bins=15, color='steelblue', alpha=0.7)
                ax.set_ylabel('Count' if j == 0 else '')
            else:
                # Off-diagonal: scatter
                ax.scatter(data[metrics[j]], data[metrics[i]],
                          c=[r["developmental_time"] for r in results],
                          cmap='viridis', alpha=0.7, s=30)

                # Correlation coefficient
                r = np.corrcoef(data[metrics[j]], data[metrics[i]])[0, 1]
                ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes,
                       fontsize=9, va='top')

            # Labels
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=9)

    plt.suptitle('Metric Correlations (color = developmental time)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all(results_path: str, output_dir: str = None, show: bool = False):
    """
    Generate all plots from results file.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting. Install with: pip install matplotlib")
        return

    data = load_results(results_path)
    results = data["results"]
    phase = data.get("phase_transition", {})
    stages = data.get("biological_stages", [])

    if output_dir is None:
        output_dir = Path(results_path).parent / "figures"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    print(f"Generating figures in {output_dir}...")

    plot_consciousness_trajectory(
        results,
        save_path=str(output_dir / "consciousness_trajectory.png"),
        show=show
    )

    if phase:
        plot_phase_transition(
            results, phase,
            save_path=str(output_dir / "phase_transition.png"),
            show=show
        )

    if stages:
        plot_biological_stages(
            results, stages,
            save_path=str(output_dir / "biological_stages.png"),
            show=show
        )

    plot_metric_correlations(
        results,
        save_path=str(output_dir / "metric_correlations.png"),
        show=show
    )

    print(f"All figures saved to {output_dir}")


def create_summary_figure(
    results: List[Dict],
    critical_time: float,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create a single summary figure for publication.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    times = [r["developmental_time"] for r in results]

    # A: Main consciousness trajectory
    ax_a = fig.add_subplot(gs[0, :2])
    scores = [r["total"] for r in results]
    ax_a.plot(times, scores, 'k-', linewidth=2.5)
    ax_a.fill_between(times, 0, scores, alpha=0.3, color='purple')
    ax_a.axvline(critical_time, color='red', linestyle='--', linewidth=2,
                label=f'Critical point (t={critical_time:.2f})')
    ax_a.set_xlabel('Developmental Time')
    ax_a.set_ylabel('Consciousness Score')
    ax_a.set_title('A. Consciousness Emergence', fontsize=12, fontweight='bold')
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)

    # B: Components
    ax_b = fig.add_subplot(gs[0, 2])
    integration = [r.get("integration", 0) for r in results]
    complexity = [r.get("complexity", 0) for r in results]
    ax_b.plot(times, integration, '-', label='Integration')
    ax_b.plot(times, complexity, '-', label='Complexity')
    ax_b.set_xlabel('Developmental Time')
    ax_b.set_ylabel('Score')
    ax_b.set_title('B. Key Components', fontsize=12, fontweight='bold')
    ax_b.legend()
    ax_b.grid(True, alpha=0.3)

    # C: Synchronization
    ax_c = fig.add_subplot(gs[1, 0])
    sync = [r["global_sync"] for r in results]
    ax_c.plot(times, sync, 'b-', linewidth=2)
    ax_c.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax_c.set_xlabel('Developmental Time')
    ax_c.set_ylabel('Global Synchronization')
    ax_c.set_title('C. Neural Coherence', fontsize=12, fontweight='bold')
    ax_c.grid(True, alpha=0.3)

    # D: Metastability
    ax_d = fig.add_subplot(gs[1, 1])
    meta = [r.get("metastability", 0) for r in results]
    ax_d.plot(times, meta, 'g-', linewidth=2)
    ax_d.set_xlabel('Developmental Time')
    ax_d.set_ylabel('Metastability')
    ax_d.set_title('D. Dynamic Richness', fontsize=12, fontweight='bold')
    ax_d.grid(True, alpha=0.3)

    # E: Phase diagram (sync vs metastability colored by time)
    ax_e = fig.add_subplot(gs[1, 2])
    scatter = ax_e.scatter(sync, meta, c=times, cmap='viridis',
                          s=50, alpha=0.8, edgecolors='k', linewidths=0.5)
    ax_e.set_xlabel('Synchronization')
    ax_e.set_ylabel('Metastability')
    ax_e.set_title('E. Phase Space', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax_e, label='Dev. Time')

    # Mark critical region
    crit_idx = np.argmin(np.abs(np.array(times) - critical_time))
    ax_e.scatter([sync[crit_idx]], [meta[crit_idx]],
                s=200, c='red', marker='*', edgecolors='k', linewidths=1,
                label='Critical', zorder=5)
    ax_e.legend()

    plt.suptitle('Developmental Consciousness Model: Summary',
                fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Run visualization on latest results
    results_dir = Path(__file__).parent
    results_file = results_dir / "developmental_results_v2.json"

    if results_file.exists():
        plot_all(str(results_file), show=False)
    else:
        print(f"Results file not found: {results_file}")
        print("Run developmental_consciousness_v2.py first to generate results.")
