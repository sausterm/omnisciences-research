"""
Visualization for Finite-Size Scaling Analysis

Creates publication-quality figures for:
- Binder cumulant crossing
- Magnetization scaling
- Susceptibility scaling
- Scaling collapse
"""

import json
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_scaling_results(path: str) -> dict:
    """Load finite-size scaling results."""
    with open(path) as f:
        return json.load(f)


def plot_binder_crossing(data: dict, save_path: str = None, show: bool = True):
    """
    Plot Binder cumulant vs temperature for multiple sizes.

    Crossing point indicates T_c.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data["results"])))

    for i, r in enumerate(data["results"]):
        L = r["L"]
        temps = r["temperatures"]
        binder = r["binder"]
        binder_err = r.get("binder_err", [0] * len(binder))

        ax.errorbar(temps, binder, yerr=binder_err,
                   marker='o', markersize=4, capsize=2,
                   label=f'L={L}', color=colors[i])

    # Mark theoretical T_c
    T_c_theory = data["binder_crossing"]["T_c_theory"]
    T_c_measured = data["binder_crossing"]["T_c_mean"]

    ax.axvline(T_c_theory, color='gray', linestyle=':', alpha=0.7,
              label=f'T_c (theory) = {T_c_theory:.3f}')
    ax.axvline(T_c_measured, color='red', linestyle='--', alpha=0.7,
              label=f'T_c (measured) = {T_c_measured:.3f}')

    ax.set_xlabel('Temperature T', fontsize=12)
    ax.set_ylabel('Binder Cumulant U', fontsize=12)
    ax.set_title('Binder Cumulant Crossing', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_magnetization_scaling(data: dict, save_path: str = None, show: bool = True):
    """
    Plot magnetization vs temperature and scaling behavior.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data["results"])))

    # Left: M vs T
    ax1 = axes[0]
    for i, r in enumerate(data["results"]):
        ax1.errorbar(r["temperatures"], r["magnetization"],
                    yerr=r.get("magnetization_err", None),
                    marker='o', markersize=4, capsize=2,
                    label=f'L={r["L"]}', color=colors[i])

    T_c = data["binder_crossing"]["T_c_mean"]
    ax1.axvline(T_c, color='red', linestyle='--', alpha=0.7)

    ax1.set_xlabel('Temperature T', fontsize=12)
    ax1.set_ylabel('Magnetization |M|', fontsize=12)
    ax1.set_title('Magnetization vs Temperature', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Log-log scaling at T_c
    ax2 = axes[1]
    exp = data["critical_exponents"]
    sizes = np.array(exp["sizes"])
    mag = np.array(exp["mag_at_Tc"])

    ax2.loglog(sizes, mag, 'ko-', markersize=8, label='Data')

    # Fit line
    log_L = np.log(sizes)
    log_M = np.log(mag + 1e-10)
    coeffs = np.polyfit(log_L, log_M, 1)
    L_fit = np.linspace(sizes.min(), sizes.max(), 100)
    M_fit = np.exp(coeffs[1]) * L_fit ** coeffs[0]
    ax2.loglog(L_fit, M_fit, 'r--',
              label=f'Fit: β/ν = {-coeffs[0]:.3f}\n(Theory: 0.125)')

    ax2.set_xlabel('Lattice Size L', fontsize=12)
    ax2.set_ylabel('|M| at T_c', fontsize=12)
    ax2.set_title('Magnetization Scaling', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_susceptibility_scaling(data: dict, save_path: str = None, show: bool = True):
    """
    Plot susceptibility vs temperature and scaling behavior.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data["results"])))

    # Left: χ vs T
    ax1 = axes[0]
    for i, r in enumerate(data["results"]):
        ax1.errorbar(r["temperatures"], r["susceptibility"],
                    yerr=r.get("susceptibility_err", None),
                    marker='o', markersize=4, capsize=2,
                    label=f'L={r["L"]}', color=colors[i])

    T_c = data["binder_crossing"]["T_c_mean"]
    ax1.axvline(T_c, color='red', linestyle='--', alpha=0.7)

    ax1.set_xlabel('Temperature T', fontsize=12)
    ax1.set_ylabel('Susceptibility χ', fontsize=12)
    ax1.set_title('Susceptibility vs Temperature', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: χ_max scaling
    ax2 = axes[1]
    exp = data["critical_exponents"]
    sizes = np.array(exp["sizes"])
    chi_max = np.array(exp["chi_max"])

    ax2.loglog(sizes, chi_max, 'ko-', markersize=8, label='Data')

    # Fit line
    log_L = np.log(sizes)
    log_chi = np.log(chi_max)
    coeffs = np.polyfit(log_L, log_chi, 1)
    L_fit = np.linspace(sizes.min(), sizes.max(), 100)
    chi_fit = np.exp(coeffs[1]) * L_fit ** coeffs[0]
    ax2.loglog(L_fit, chi_fit, 'r--',
              label=f'Fit: γ/ν = {coeffs[0]:.3f}\n(Theory: 1.75)')

    ax2.set_xlabel('Lattice Size L', fontsize=12)
    ax2.set_ylabel('χ_max', fontsize=12)
    ax2.set_title('Susceptibility Scaling', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_blanket_transition(data: dict, save_path: str = None, show: bool = True):
    """
    Plot blanket metrics showing phase transition.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data["results"])))

    # Left: Individual blanket
    ax1 = axes[0]
    for i, r in enumerate(data["results"]):
        ax1.plot(r["temperatures"], r["individual_blanket"],
                marker='o', markersize=4,
                label=f'L={r["L"]}', color=colors[i])

    T_c = data["binder_crossing"]["T_c_mean"]
    ax1.axvline(T_c, color='red', linestyle='--', alpha=0.7, label='T_c')

    ax1.set_xlabel('Temperature T', fontsize=12)
    ax1.set_ylabel('Individual Blanket Index', fontsize=12)
    ax1.set_title('Individual Blankets (Higher = More Distinct)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Collective blanket
    ax2 = axes[1]
    for i, r in enumerate(data["results"]):
        ax2.plot(r["temperatures"], r["collective_blanket"],
                marker='o', markersize=4,
                label=f'L={r["L"]}', color=colors[i])

    ax2.axvline(T_c, color='red', linestyle='--', alpha=0.7, label='T_c')

    ax2.set_xlabel('Temperature T', fontsize=12)
    ax2.set_ylabel('Collective Blanket Index', fontsize=12)
    ax2.set_title('Collective Blanket (Higher = More Unified)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_summary_figure(data: dict, save_path: str = None, show: bool = True):
    """
    Create a summary figure for publication.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data["results"])))
    T_c = data["binder_crossing"]["T_c_mean"]
    T_c_theory = data["binder_crossing"]["T_c_theory"]

    # A: Binder cumulant
    ax = axes[0, 0]
    for i, r in enumerate(data["results"]):
        ax.errorbar(r["temperatures"], r["binder"],
                   yerr=r.get("binder_err"), marker='o', markersize=4,
                   capsize=2, label=f'L={r["L"]}', color=colors[i])
    ax.axvline(T_c, color='red', linestyle='--', alpha=0.7)
    ax.axvline(T_c_theory, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('T')
    ax.set_ylabel('Binder Cumulant U')
    ax.set_title('A. Binder Cumulant Crossing')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # B: Magnetization
    ax = axes[0, 1]
    for i, r in enumerate(data["results"]):
        ax.plot(r["temperatures"], r["magnetization"],
               marker='o', markersize=4, label=f'L={r["L"]}', color=colors[i])
    ax.axvline(T_c, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('T')
    ax.set_ylabel('|M|')
    ax.set_title('B. Order Parameter')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # C: Susceptibility
    ax = axes[1, 0]
    for i, r in enumerate(data["results"]):
        ax.plot(r["temperatures"], r["susceptibility"],
               marker='o', markersize=4, label=f'L={r["L"]}', color=colors[i])
    ax.axvline(T_c, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('T')
    ax.set_ylabel('χ')
    ax.set_title('C. Susceptibility')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # D: Blanket crossover
    ax = axes[1, 1]
    for i, r in enumerate(data["results"]):
        ax.plot(r["temperatures"], r["individual_blanket"],
               marker='o', markersize=4, linestyle='-',
               label=f'Individual L={r["L"]}', color=colors[i])
        ax.plot(r["temperatures"], r["collective_blanket"],
               marker='s', markersize=4, linestyle='--',
               color=colors[i], alpha=0.6)
    ax.axvline(T_c, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('T')
    ax.set_ylabel('Blanket Index')
    ax.set_title('D. Blanket Phase Transition')
    ax.grid(True, alpha=0.3)
    # Custom legend
    ax.plot([], [], 'k-', label='Individual')
    ax.plot([], [], 'k--', label='Collective')
    ax.legend(fontsize=9)

    plt.suptitle(f'Finite-Size Scaling: T_c = {T_c:.3f} (theory: {T_c_theory:.3f})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_scaling(results_path: str, output_dir: str = None, show: bool = False):
    """Generate all scaling plots."""
    if not HAS_MATPLOTLIB:
        print("matplotlib required")
        return

    data = load_scaling_results(results_path)

    if output_dir is None:
        output_dir = Path(results_path).parent / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Generating scaling figures in {output_dir}...")

    plot_binder_crossing(data, str(output_dir / "binder_crossing.png"), show)
    plot_magnetization_scaling(data, str(output_dir / "magnetization_scaling.png"), show)
    plot_susceptibility_scaling(data, str(output_dir / "susceptibility_scaling.png"), show)
    plot_blanket_transition(data, str(output_dir / "blanket_transition.png"), show)
    plot_summary_figure(data, str(output_dir / "scaling_summary.png"), show)

    print("All scaling figures generated.")


if __name__ == "__main__":
    results_path = Path(__file__).parent / "finite_size_scaling_results.json"

    if results_path.exists():
        plot_all_scaling(str(results_path), show=False)
    else:
        print(f"Results not found: {results_path}")
        print("Run finite_size_scaling.py first.")
