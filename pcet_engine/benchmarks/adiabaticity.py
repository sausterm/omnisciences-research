"""
Adiabaticity analysis for the 28-system PCET benchmark.

Computes the proton-transfer adiabaticity parameter:

    kappa_p = V_el^2 / (hbar * omega_H * delta_0)

where V_el is the electronic coupling (kcal/mol), omega_H is the proton
vibrational frequency (cm^-1), and delta_0 is the tunneling distance (Ang).

When kappa_p << 1: nonadiabatic (SHS vibronic limit, current engine)
When kappa_p >> 1: adiabatic (Marcus-like, proton on single BO surface)
When kappa_p ~ 0.1-1: intermediate (no clean computational treatment exists)

This script:
1. Computes kappa_p for all 28 benchmark systems
2. Plots them on a log axis colored by mechanistic class
3. Checks whether prediction errors (Cat I/II systems) correlate with kappa_p
4. Identifies which systems are in the intermediate regime

References:
    Cukier & Nocera, Annu. Rev. Phys. Chem. 49, 337 (1998)
    Hammes-Schiffer & Soudackov, J. Phys. Chem. B 112, 14108 (2008)
    Georgievskii & Stuchebrukhov, J. Chem. Phys. 113, 10438 (2000)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass

# ── Unit conversion constants ──────────────────────────────────────────────
KCAL_TO_EV = 0.04336          # 1 kcal/mol in eV
CM1_TO_EV  = 1.2398e-4        # 1 cm^-1 in eV  (hbar*omega in eV for omega in cm^-1)
ANG_TO_BOHR = 1.8897           # 1 Ang in Bohr
HBAR_EV_S  = 6.582e-16         # hbar in eV·s


def kappa_p(V_el_kcal, omega_H_cm1, delta_0_ang):
    """
    Proton-transfer adiabaticity parameter.

    kappa_p = V_el^2 / (hbar * omega_H * delta_0)

    All quantities converted to eV / Ang for consistency.
    kappa_p is dimensionless.

    Derivation:
        In the Landau-Zener picture for proton transfer, the proton velocity
        at the crossing point is v_p ~ omega_H * delta_0 (harmonic estimate).
        The non-adiabaticity criterion is V_el << hbar * v_p / delta_0
        = hbar * omega_H, giving kappa_p = V_el^2 / (hbar*omega_H)^2
        as the squared ratio. We use the single-power version
        kappa_p = V_el / (hbar * omega_H) which directly compares
        coupling to vibrational quantum.

    Note: We use kappa_p = V_el / (hbar * omega_H) (ratio of coupling to
    vibrational quantum), which is the cleanest dimensionless diagnostic.
    The delta_0 dependence enters through the FC overlap but not the
    adiabaticity criterion itself. We compute both versions.
    """
    V_el_eV    = V_el_kcal * KCAL_TO_EV
    hbar_om_eV = omega_H_cm1 * CM1_TO_EV   # hbar*omega_H in eV

    # Primary: V_el / (hbar * omega_H) — pure coupling/vibration ratio
    kappa_simple = V_el_eV / hbar_om_eV

    # Extended: includes delta_0 via FC overlap scale factor
    # Rough FC decay: |S_00|^2 ~ exp(-alpha * delta_0^2)
    # alpha_H ~ 87 Ang^-2 (from Note S2 of pcet_benchmark_v6_si)
    alpha_H = 87.0  # Ang^-2
    fc_scale = np.exp(-alpha_H * delta_0_ang**2)

    # Effective coupling including FC: V_eff = V_el * |S_00|
    V_eff_eV = V_el_eV * np.sqrt(fc_scale)
    kappa_fc = V_eff_eV / hbar_om_eV

    return kappa_simple, kappa_fc, fc_scale


# ── System data ────────────────────────────────────────────────────────────
# (name, V_el kcal/mol, omega_H cm^-1, delta_0 Ang, KIE_exp, category, class)
# Category: 1=QM/MM verified, 2=literature, 3=calibrated
# Class: for coloring

SYSTEMS = [
    # name            V_el   omH    d0      KIE_exp  cat  mech_class
    ("SLO-1 WT",      0.60,  2900,  0.500,  81.0,    1,   "SLO-1 family"),
    ("SLO-1 L546A",   0.50,  2900,  0.514,  93.0,    3,   "SLO-1 family"),
    ("SLO-1 L754A",   0.45,  2900,  0.524,  112.0,   3,   "SLO-1 family"),
    ("SLO-1 DM",      0.30,  2900,  0.614,  661.0,   3,   "SLO-1 family"),
    ("SLO-1 I553G",   0.25,  2900,  0.518,  100.0,   3,   "SLO-1 family"),
    ("SLO-1 I553A",   0.35,  2900,  0.510,  87.0,    3,   "SLO-1 family"),
    ("SLO-1 I553V",   0.45,  2900,  0.499,  72.0,    3,   "SLO-1 family"),
    ("SLO-1 I553L",   0.50,  2900,  0.493,  65.0,    3,   "SLO-1 family"),
    ("SLO-1 I553F",   0.55,  2900,  0.499,  71.0,    3,   "SLO-1 family"),
    ("AADH",          0.80,  3000,  0.465,  55.0,    2,   "HAT"),
    ("MADH",          0.50,  2950,  0.432,  30.0,    2,   "HAT"),
    ("PHM",           1.20,  3100,  0.347,  10.0,    1,   "HAT"),
    ("RNR",           0.20,  2600,  0.348,  7.0,     2,   "HAT"),
    ("GO",            0.70,  2900,  0.422,  22.0,    3,   "HAT"),
    ("TauD",          0.50,  2900,  0.436,  27.0,    3,   "HAT"),
    ("DβH",           0.60,  2900,  0.370,  10.8,    3,   "HAT"),
    ("CAO",           0.60,  2900,  0.379,  12.0,    3,   "HAT"),
    ("LADH",          2.00,  3000,  0.264,  3.5,     3,   "Hydride"),
    ("DHFR",          3.00,  3000,  0.247,  3.0,     3,   "Hydride"),
    ("TSase",         0.80,  2950,  0.318,  6.0,     3,   "Hydride"),
    ("MAO",           0.50,  2950,  0.345,  8.0,     3,   "Hydride"),
    ("bc1",           1.50,  3300,  0.250,  3.5,     3,   "Proton/O-H"),
    ("PhOH-self",     1.50,  3000,  0.279,  4.1,     3,   "Proton/O-H"),
    ("RNR-3FY",       0.50,  3200,  0.303,  6.0,     3,   "Proton/O-H"),
    ("RNR-2FY",       0.40,  3200,  0.287,  5.0,     3,   "Proton/O-H"),
    ("MR",            0.80,  2900,  0.440,  25.0,    3,   "Flavin"),
    ("PETNR",         0.80,  2900,  0.315,  5.4,     3,   "Flavin"),
    ("GOx",           0.80,  2900,  0.354,  8.3,     3,   "Flavin"),
]

CLASS_COLORS = {
    "SLO-1 family": "#2166ac",
    "HAT":          "#d73027",
    "Hydride":      "#1a9850",
    "Proton/O-H":   "#f46d43",
    "Flavin":       "#8073ac",
}

CAT_MARKERS = {1: "o", 2: "s", 3: "^"}
CAT_LABELS  = {1: "Cat I (QM/MM)", 2: "Cat II (literature)", 3: "Cat III (calibrated)"}


def main():
    names, kappas_simple, kappas_fc, fc_scales = [], [], [], []
    colors, markers, kie_exps, cats = [], [], [], []

    print(f"\n{'System':<16} {'V_el':>6} {'ω_H':>6} {'δ₀':>6} "
          f"{'κ_simple':>10} {'κ_fc':>10} {'|S₀₀|²':>10} {'KIE_exp':>8} {'Cat':>4}")
    print("-" * 90)

    for (name, V_el, omH, d0, KIE, cat, cls) in SYSTEMS:
        ks, kfc, fc = kappa_p(V_el, omH, d0)
        names.append(name)
        kappas_simple.append(ks)
        kappas_fc.append(kfc)
        fc_scales.append(fc)
        colors.append(CLASS_COLORS[cls])
        markers.append(CAT_MARKERS[cat])
        kie_exps.append(KIE)
        cats.append(cat)

        regime = "nonadiabatic" if ks < 0.05 else ("intermediate" if ks < 0.5 else "adiabatic")
        print(f"{name:<16} {V_el:>6.2f} {omH:>6} {d0:>6.3f} "
              f"{ks:>10.4f} {kfc:>10.4f} {fc:>10.2e} {KIE:>8.1f} {cat:>4}  {regime}")

    kappas_simple = np.array(kappas_simple)
    kappas_fc     = np.array(kappas_fc)
    kie_exps      = np.array(kie_exps)
    cats          = np.array(cats)

    print(f"\n{'─'*60}")
    print(f"κ_simple range: {kappas_simple.min():.4f} – {kappas_simple.max():.4f}")
    print(f"Nonadiabatic  (κ < 0.05): {(kappas_simple < 0.05).sum()} systems")
    print(f"Intermediate  (0.05–0.5): {((kappas_simple >= 0.05) & (kappas_simple < 0.5)).sum()} systems")
    print(f"Near-adiabatic (κ ≥ 0.5): {(kappas_simple >= 0.5).sum()} systems")

    # ── Correlation: kappa vs log(KIE) ────────────────────────────────────
    log_kie = np.log10(kie_exps)
    r = np.corrcoef(np.log10(kappas_simple), log_kie)[0, 1]
    print(f"\nCorrelation log(κ) vs log(KIE): r = {r:.3f}")

    # ── Figure 1: κ_simple per system ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    x = np.arange(len(names))
    for i, (name, ks, col, mk, cat) in enumerate(
            zip(names, kappas_simple, colors, markers, cats)):
        ax.scatter(i, ks, c=col, marker=mk, s=80,
                   zorder=3, edgecolors="k", linewidths=0.5)

    ax.axhline(0.5,  color="gray", lw=1, ls="--", alpha=0.7, label="κ = 0.5 (near-adiabatic)")
    ax.axhline(0.05, color="gray", lw=1, ls=":",  alpha=0.7, label="κ = 0.05 (nonadiabatic)")
    ax.fill_between([-0.5, len(names)-0.5], 0.05, 0.5,
                    alpha=0.08, color="orange", label="Intermediate regime")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"Adiabaticity parameter $\kappa_p = V_\mathrm{el}/\hbar\omega_H$")
    ax.set_title("Proton-transfer adiabaticity across 28 systems")
    ax.legend(fontsize=8, loc="upper right")

    # class color legend
    class_patches = [mpatches.Patch(color=c, label=l)
                     for l, c in CLASS_COLORS.items()]
    ax.legend(handles=class_patches, fontsize=8, loc="lower right")

    # ── Figure 2: κ vs KIE scatter ─────────────────────────────────────────
    ax2 = axes[1]
    for i, (ks, kie, col, mk) in enumerate(
            zip(kappas_simple, kie_exps, colors, markers)):
        ax2.scatter(ks, kie, c=col, marker=mk, s=80,
                    zorder=3, edgecolors="k", linewidths=0.5)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\kappa_p = V_\mathrm{el}/\hbar\omega_H$")
    ax2.set_ylabel(r"KIE$_\mathrm{exp}$")
    ax2.set_title(f"KIE vs adiabaticity  (r = {r:.2f} in log-log)")
    ax2.axvline(0.5,  color="gray", lw=1, ls="--", alpha=0.7)
    ax2.axvline(0.05, color="gray", lw=1, ls=":",  alpha=0.7)
    ax2.axhline(7.7,  color="steelblue", lw=1, ls="-.", alpha=0.7,
                label="ZPE ceiling KIE = 7.7")
    ax2.legend(fontsize=8)

    # fit line in log-log
    lk = np.log10(kappas_simple)
    lKIE = np.log10(kie_exps)
    m, b = np.polyfit(lk, lKIE, 1)
    xfit = np.logspace(np.log10(kappas_simple.min()), np.log10(kappas_simple.max()), 100)
    ax2.plot(xfit, 10**(m * np.log10(xfit) + b), "k--", lw=1, alpha=0.5,
             label=f"slope = {m:.2f}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = "pcet_engine/benchmarks/figures/adiabaticity_map.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nFigures saved: {out}")

    # ── Regime summary table ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("INTERMEDIATE REGIME SYSTEMS (0.05 ≤ κ < 0.5) — theory gap lives here:")
    print(f"{'System':<16} {'κ_simple':>10} {'KIE_exp':>8}")
    for name, ks, kie in zip(names, kappas_simple, kie_exps):
        if 0.05 <= ks < 0.5:
            print(f"  {name:<14} {ks:>10.4f} {kie:>8.1f}")

    print(f"\nNEAR-ADIABATIC SYSTEMS (κ ≥ 0.5) — SHS likely overestimates tunneling:")
    for name, ks, kie in zip(names, kappas_simple, kie_exps):
        if ks >= 0.5:
            print(f"  {name:<14} {ks:>10.4f} {kie:>8.1f}")

    return names, kappas_simple, kappas_fc, kie_exps


if __name__ == "__main__":
    main()
