"""
Visualization of D-A gating correlation analysis.

Generates publication-quality figures for the empirical gating model:
  Ω_gating = 808 - 104×d_DA - 27.5×M_DA  (R² = 0.925)

Figures:
  1. Ω vs M_DA scatter (color by enzyme class) — "money plot"
  2. 3D surface: predicted Ω(d_DA, M_DA) with data points
  3. Enzyme class ladder diagram
  4. Rate improvement waterfall (with vs without gating)
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors


# Enzyme class color scheme
CLASS_COLORS = {
    "lipoxygenase":        "#E63946",  # red
    "FeIV_oxo":            "#D62828",  # dark red
    "radical_enzyme":      "#F77F00",  # orange
    "oxidase":             "#FCBF49",  # gold
    "amine_dehydrogenase": "#2A9D8F",  # teal
    "monooxygenase":       "#264653",  # dark teal
    "oxidoreductase":      "#457B9D",  # blue
    "dehydrogenase":       "#1D3557",  # dark blue
    "transferase":         "#6A4C93",  # purple
}

CLASS_MARKERS = {
    "lipoxygenase":        "o",
    "FeIV_oxo":            "D",
    "radical_enzyme":      "s",
    "oxidase":             "^",
    "amine_dehydrogenase": "v",
    "monooxygenase":       "P",
    "oxidoreductase":      "X",
    "dehydrogenase":       "p",
    "transferase":         "*",
}

CLASS_LABELS = {
    "lipoxygenase":        "Lipoxygenase",
    "FeIV_oxo":            "Fe\u1d35\u1d5b=O",
    "radical_enzyme":      "Radical",
    "oxidase":             "Oxidase",
    "amine_dehydrogenase": "Amine DH",
    "monooxygenase":       "Monooxygenase",
    "oxidoreductase":      "Oxidoreductase",
    "dehydrogenase":       "Dehydrogenase",
    "transferase":         "Transferase",
}


def _load_data():
    """Load published gating data."""
    data_path = Path(__file__).parent.parent / "data" / "published_gating_params.json"
    with open(data_path) as f:
        data = json.load(f)
    return data["systems"]


def plot_omega_vs_mda(output_path: str = "gating_omega_vs_mda.png"):
    """Plot 1: Ω_gating vs M_DA scatter colored by enzyme class.

    This is the strongest correlation (r = -0.938, R² = 0.880).
    """
    systems = _load_data()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot regression line first
    m_range = np.linspace(4, 16, 100)
    # Univariate regression: Ω = -27.5 × M_DA + 508.3
    omega_fit = -27.5 * m_range + 508.3
    ax.fill_between(m_range, omega_fit - 33, omega_fit + 33,
                    alpha=0.12, color="#457B9D", label="LOO-RMSE band (±33 cm⁻¹)")
    ax.plot(m_range, omega_fit, "--", color="#457B9D", alpha=0.6, linewidth=1.5)

    # Plot data points by enzyme class
    plotted_classes = set()
    for name, sys in systems.items():
        cls = sys["enzyme_class"]
        color = CLASS_COLORS.get(cls, "#888888")
        marker = CLASS_MARKERS.get(cls, "o")
        label = CLASS_LABELS.get(cls, cls) if cls not in plotted_classes else None
        plotted_classes.add(cls)

        ax.scatter(sys["M_DA"], sys["omega_gating"],
                   c=color, marker=marker, s=100, edgecolors="black",
                   linewidth=0.8, zorder=5, label=label)

        # Label selected points
        offset = (5, 5)
        if name in ("SLO-1", "DHFR-EcDHFR", "TauD", "AADH", "RNR-C439"):
            display_name = name.replace("-EcDHFR", "").replace("-C439", "")
            ax.annotate(display_name, (sys["M_DA"], sys["omega_gating"]),
                       textcoords="offset points", xytext=offset,
                       fontsize=8, color="#333333")

    ax.set_xlabel("Effective D-A mass M$_{DA}$ (amu)", fontsize=12)
    ax.set_ylabel("Gating frequency Ω$_{gating}$ (cm$^{-1}$)", fontsize=12)
    ax.set_title("D-A Gating Frequency vs Effective Mass\n"
                 "16 enzyme systems, r = −0.938, R² = 0.880", fontsize=13)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(3.5, 16)
    ax.set_ylim(50, 450)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_3d_surface(output_path: str = "gating_3d_surface.png"):
    """Plot 2: 3D surface of Ω(d_DA, M_DA) with data points."""
    systems = _load_data()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Multivariate model: Ω = 808 - 104×d_DA - 27.5×M_DA
    d_range = np.linspace(2.4, 3.5, 30)
    m_range = np.linspace(4, 16, 30)
    D, M = np.meshgrid(d_range, m_range)
    Omega = 808.0 - 104.2 * D - 27.5 * M
    Omega = np.clip(Omega, 50, 500)

    # Surface
    surf = ax.plot_surface(D, M, Omega, alpha=0.35, cmap="coolwarm",
                          edgecolor="none", antialiased=True)

    # Data points
    for name, sys in systems.items():
        cls = sys["enzyme_class"]
        color = CLASS_COLORS.get(cls, "#888888")
        ax.scatter(sys["d_DA"], sys["M_DA"], sys["omega_gating"],
                  c=color, s=80, edgecolors="black", linewidth=0.6,
                  marker="o", zorder=10, depthshade=False)

    ax.set_xlabel("\nd$_{DA}$ (Å)", fontsize=11, labelpad=10)
    ax.set_ylabel("\nM$_{DA}$ (amu)", fontsize=11, labelpad=10)
    ax.set_zlabel("\nΩ$_{gating}$ (cm$^{-1}$)", fontsize=11, labelpad=10)
    ax.set_title("Multivariate Gating Model\n"
                 "Ω = 808 − 104·d$_{DA}$ − 27.5·M$_{DA}$   (R² = 0.925)",
                 fontsize=12, pad=15)
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_enzyme_class_ladder(output_path: str = "gating_class_ladder.png"):
    """Plot 3: Enzyme class ladder showing Ω ranges."""
    systems = _load_data()

    # Group by class
    classes = {}
    for name, sys in systems.items():
        cls = sys["enzyme_class"]
        if cls not in classes:
            classes[cls] = []
        classes[cls].append(sys)

    # Sort by mean Ω
    sorted_classes = sorted(classes.items(),
                           key=lambda x: np.mean([s["omega_gating"] for s in x[1]]))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    y_positions = []
    y_labels = []

    for i, (cls, members) in enumerate(sorted_classes):
        omegas = [m["omega_gating"] for m in members]
        omega_min = min(omegas)
        omega_max = max(omegas)
        omega_mean = np.mean(omegas)
        n = len(members)

        color = CLASS_COLORS.get(cls, "#888888")
        y = i

        # Draw range bar
        bar_height = 0.4
        ax.barh(y, omega_max - omega_min, left=omega_min, height=bar_height,
               color=color, alpha=0.7, edgecolor="black", linewidth=0.8)

        # Draw mean marker
        ax.plot(omega_mean, y, "d", color="white", markersize=10,
               markeredgecolor="black", markeredgewidth=1.2, zorder=5)

        # Individual points
        for m in members:
            ax.plot(m["omega_gating"], y, "o", color=color, markersize=6,
                   markeredgecolor="black", markeredgewidth=0.5, zorder=4)

        # Label with count
        label = CLASS_LABELS.get(cls, cls)
        y_positions.append(y)
        y_labels.append(f"{label} (n={n})")

        # Annotate mean
        ax.annotate(f"{omega_mean:.0f}", (omega_mean, y),
                   textcoords="offset points", xytext=(0, -18),
                   fontsize=8, ha="center", color="#333333", fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Ω$_{gating}$ (cm$^{-1}$)", fontsize=12)
    ax.set_title("Gating Frequency by Enzyme Class", fontsize=13)
    ax.set_xlim(50, 450)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Add annotation
    ax.text(0.98, 0.02, "◇ = class mean",
           transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
           style="italic", color="#555555")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_rate_improvement(output_path: str = "gating_rate_improvement.png"):
    """Plot 4: Rate prediction error with vs without predicted gating."""
    from pcet_engine.benchmarks.gating_correlation import apply_to_benchmark_systems

    results = apply_to_benchmark_systems(verbose=False)

    names = list(results.keys())
    log_no = [abs(results[n]["log_err_no"]) for n in names]
    log_pred = [abs(results[n]["log_err_pred"]) for n in names]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, log_no, width, color="#D62828", alpha=0.8,
                  edgecolor="black", linewidth=0.5, label="No gating")
    bars2 = ax.bar(x + width/2, log_pred, width, color="#2A9D8F", alpha=0.8,
                  edgecolor="black", linewidth=0.5, label="Predicted gating")

    # Reference lines
    ax.axhline(y=1.0, color="#333333", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(len(names) - 0.5, 1.1, "1 OOM", fontsize=8, color="#555555", ha="right")
    ax.axhline(y=2.0, color="#333333", linestyle=":", linewidth=1, alpha=0.3)
    ax.text(len(names) - 0.5, 2.1, "2 OOM", fontsize=8, color="#555555", ha="right")

    # Highlight systems where gating brings error < 2 OOM
    for i, n in enumerate(names):
        if log_pred[i] < 2.0:
            ax.annotate("✓", (x[i] + width/2, log_pred[i]),
                       textcoords="offset points", xytext=(0, 5),
                       fontsize=12, ha="center", color="#2A9D8F", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("|log₁₀(k$_{pred}$ / k$_{exp}$)|", fontsize=12)
    ax.set_title("Absolute Rate Error: Without vs With Predicted Gating", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, max(max(log_no), max(log_pred)) * 1.15)

    # Mean error annotations
    mean_no = np.mean(log_no)
    mean_pred = np.mean(log_pred)
    ax.text(0.98, 0.95, f"Mean error: {mean_no:.1f} → {mean_pred:.1f} OOM",
           transform=ax.transAxes, fontsize=11, ha="right", va="top",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_residuals(output_path: str = "gating_residuals.png"):
    """Plot 5: Model residuals — predicted vs published Ω with error bars."""
    systems = _load_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = []
    omega_pub = []
    omega_pred = []
    classes = []

    for name, sys in systems.items():
        names.append(name)
        omega_pub.append(sys["omega_gating"])
        pred = 808.0 - 104.2 * sys["d_DA"] - 27.5 * sys["M_DA"]
        omega_pred.append(pred)
        classes.append(sys["enzyme_class"])

    omega_pub = np.array(omega_pub)
    omega_pred = np.array(omega_pred)
    residuals = omega_pred - omega_pub

    # Left: predicted vs published (1:1 plot)
    for i, name in enumerate(names):
        color = CLASS_COLORS.get(classes[i], "#888888")
        marker = CLASS_MARKERS.get(classes[i], "o")
        ax1.scatter(omega_pub[i], omega_pred[i], c=color, marker=marker,
                   s=80, edgecolors="black", linewidth=0.6, zorder=5)

    # 1:1 line
    lims = [50, 450]
    ax1.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
    ax1.fill_between(lims, [l - 33 for l in lims], [l + 33 for l in lims],
                     alpha=0.1, color="#457B9D")

    ax1.set_xlabel("Published Ω$_{gating}$ (cm$^{-1}$)", fontsize=11)
    ax1.set_ylabel("Predicted Ω$_{gating}$ (cm$^{-1}$)", fontsize=11)
    ax1.set_title("Predicted vs Published", fontsize=12)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    rmse = math.sqrt(np.mean(residuals**2))
    ax1.text(0.05, 0.92, f"RMSE = {rmse:.1f} cm⁻¹\nR² = 0.925",
            transform=ax1.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999"))

    # Right: residual bar chart
    sort_idx = np.argsort(residuals)
    colors = ["#E63946" if r > 0 else "#2A9D8F" for r in residuals[sort_idx]]

    ax2.barh(range(len(names)), residuals[sort_idx], color=colors,
            edgecolor="black", linewidth=0.5, alpha=0.8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([names[i] for i in sort_idx], fontsize=8)
    ax2.set_xlabel("Residual (cm$^{-1}$)", fontsize=11)
    ax2.set_title("Model Residuals", fontsize=12)
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Multivariate Gating Model Performance", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_all_plots(output_dir: str = "pcet_engine/benchmarks/figures"):
    """Generate all gating correlation figures."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    plot_omega_vs_mda(os.path.join(output_dir, "gating_omega_vs_mda.png"))
    plot_3d_surface(os.path.join(output_dir, "gating_3d_surface.png"))
    plot_enzyme_class_ladder(os.path.join(output_dir, "gating_class_ladder.png"))
    plot_residuals(os.path.join(output_dir, "gating_residuals.png"))

    # Rate improvement requires pcet_engine on PYTHONPATH
    try:
        plot_rate_improvement(os.path.join(output_dir, "gating_rate_improvement.png"))
    except Exception as e:
        print(f"Skipped rate improvement plot: {e}")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    generate_all_plots()
