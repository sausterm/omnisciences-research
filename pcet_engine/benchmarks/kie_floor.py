"""
KIE floor analysis: geometric lower bound on intrinsic KIE.

In the unified LZ framework, as donor-acceptor distance delta_0 -> 0,
KIE_na -> 1 but KIE_uni approaches a nonzero floor:

    KIE_floor = KIE_ad ^ f_bar

because even at zero tunneling distance the adiabatic contribution
(weighted by f_bar) drives KIE above 1.

Key result:
    If KIE_exp < KIE_floor(V_el, omega_H), the enzyme is NECESSARILY
    commitment-limited — no geometric parameters can reconcile the
    observed KIE with the unified intrinsic mechanism.

This provides a hard, parameter-free commitment criterion that doesn't
require isotope partition experiments.

Outputs:
    1. Table: KIE_floor, KIE_exp, and minimum Cf_min for all 28 systems
    2. Figure: KIE_floor vs KIE_exp (log-log), systems below the diagonal
       are provably commitment-limited
    3. Figure: Cf_min vs kappa_p, showing the binding constraint emerges
       only above kappa_p ~ 0.15
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Constants ─────────────────────────────────────────────────────────────────
KCAL_TO_EV = 0.04336
CM1_TO_EV  = 1.2398e-4
SQRT2      = np.sqrt(2.0)
SQRT3      = np.sqrt(3.0)
KT         = 0.02569          # kT at 298 K, eV
ALPHA_H    = 87.0             # Ang^-2, FC decay constant for H

CLASS_COLORS = {
    "SLO-1 family": "#2166ac",
    "HAT":          "#d73027",
    "Hydride":      "#1a9850",
    "Proton/O-H":   "#f46d43",
    "Flavin":       "#8073ac",
}
CAT_MARKERS = {1: "o", 2: "s", 3: "^"}


# ── Core functions ─────────────────────────────────────────────────────────────

def kappa_p(V_el_kcal, omega_H_cm1):
    """Proton adiabaticity parameter (eq. 1)."""
    V_el_eV    = V_el_kcal * KCAL_TO_EV
    hbar_om_eV = omega_H_cm1 * CM1_TO_EV
    return V_el_eV / hbar_om_eV


def lz_weight(kappa):
    """Landau-Zener adiabatic fraction (eq. 2)."""
    return 1.0 - np.exp(-2.0 * np.pi * kappa)


def kie_adiabatic(omega_H_cm1):
    """Bigeleisen semiclassical KIE (adiabatic limit, eq. 4)."""
    hbar_omH = omega_H_cm1 * CM1_TO_EV
    delta_zpe = hbar_omH * (1.0 - 1.0 / SQRT2) / 2.0   # > 0
    return SQRT2 * np.exp(delta_zpe / KT)


def kie_nonadiabatic(delta_0_ang):
    """FC-overlap KIE (nonadiabatic limit, eq. 3)."""
    alpha_D = SQRT2 * ALPHA_H
    return np.exp((alpha_D - ALPHA_H) * delta_0_ang**2)


def kie_floor(V_el_kcal, omega_H_cm1):
    """
    Minimum possible intrinsic KIE as delta_0 -> 0.

    At delta_0 = 0: KIE_na -> 1, so log KIE_na -> 0.
    log KIE_floor = f_bar * log KIE_ad

    where f_bar = (f_H + f_D) / 2.
    """
    kap_H  = kappa_p(V_el_kcal, omega_H_cm1)
    kap_D  = kap_H * SQRT2
    f_H    = lz_weight(kap_H)
    f_D    = lz_weight(kap_D)
    f_bar  = 0.5 * (f_H + f_D)
    KIE_ad = kie_adiabatic(omega_H_cm1)
    KIE_fl = np.exp(f_bar * np.log(KIE_ad))
    return KIE_fl, f_bar, f_H, f_D, KIE_ad


def kie_unified(V_el_kcal, omega_H_cm1, delta_0_ang):
    """Full unified KIE at a given delta_0."""
    kap_H  = kappa_p(V_el_kcal, omega_H_cm1)
    kap_D  = kap_H * SQRT2
    f_H    = lz_weight(kap_H)
    f_D    = lz_weight(kap_D)
    f_bar  = 0.5 * (f_H + f_D)
    KIE_na = kie_nonadiabatic(delta_0_ang)
    KIE_ad = kie_adiabatic(omega_H_cm1)
    log_KIE_uni = (1 - f_bar) * np.log(KIE_na) + f_bar * np.log(KIE_ad)
    return np.exp(log_KIE_uni)


def northrop_cf_min(KIE_exp, KIE_floor_val):
    """
    Minimum commitment factor forced by the KIE floor.

    If KIE_exp < KIE_floor, then even at delta_0 = 0 the intrinsic KIE
    exceeds KIE_exp.  The Northrop equation gives:

        Cf_min = (KIE_floor - KIE_exp) / (KIE_exp - 1)

    Returns NaN if KIE_exp >= KIE_floor (no constraint; Cf could be 0).
    """
    if KIE_exp >= KIE_floor_val or KIE_exp <= 1.0:
        return np.nan
    return (KIE_floor_val - KIE_exp) / (KIE_exp - 1.0)


# ── Benchmark data ─────────────────────────────────────────────────────────────
# (name, V_el kcal/mol, omega_H cm^-1, delta_0 Ang, KIE_exp, cat, class)
SYSTEMS = [
    ("SLO-1 WT",    0.60, 2900, 0.500,  81.0, 1, "SLO-1 family"),
    ("SLO-1 L546A", 0.50, 2900, 0.514,  93.0, 3, "SLO-1 family"),
    ("SLO-1 L754A", 0.45, 2900, 0.524, 112.0, 3, "SLO-1 family"),
    ("SLO-1 DM",    0.30, 2900, 0.614, 661.0, 3, "SLO-1 family"),
    ("SLO-1 I553G", 0.25, 2900, 0.518, 100.0, 3, "SLO-1 family"),
    ("SLO-1 I553A", 0.35, 2900, 0.510,  87.0, 3, "SLO-1 family"),
    ("SLO-1 I553V", 0.45, 2900, 0.499,  72.0, 3, "SLO-1 family"),
    ("SLO-1 I553L", 0.50, 2900, 0.493,  65.0, 3, "SLO-1 family"),
    ("SLO-1 I553F", 0.55, 2900, 0.499,  71.0, 3, "SLO-1 family"),
    ("AADH",        0.80, 3000, 0.465,  55.0, 2, "HAT"),
    ("MADH",        0.50, 2950, 0.432,  30.0, 2, "HAT"),
    ("PHM",         1.20, 3100, 0.347,  10.0, 1, "HAT"),
    ("RNR",         0.20, 2600, 0.348,   7.0, 2, "HAT"),
    ("GO",          0.70, 2900, 0.422,  22.0, 3, "HAT"),
    ("TauD",        0.50, 2900, 0.436,  27.0, 3, "HAT"),
    ("DβH",         0.60, 2900, 0.370,  10.8, 3, "HAT"),
    ("CAO",         0.60, 2900, 0.379,  12.0, 3, "HAT"),
    ("LADH",        2.00, 3000, 0.264,   3.5, 3, "Hydride"),
    ("DHFR",        3.00, 3000, 0.247,   3.0, 3, "Hydride"),
    ("TSase",       0.80, 2950, 0.318,   6.0, 3, "Hydride"),
    ("MAO",         0.50, 2950, 0.345,   8.0, 3, "Hydride"),
    ("bc1",         1.50, 3300, 0.250,   3.5, 3, "Proton/O-H"),
    ("PhOH-self",   1.50, 3000, 0.279,   4.1, 3, "Proton/O-H"),
    ("RNR-3FY",     0.50, 3200, 0.303,   6.0, 3, "Proton/O-H"),
    ("RNR-2FY",     0.40, 3200, 0.287,   5.0, 3, "Proton/O-H"),
    ("MR",          0.80, 2900, 0.440,  25.0, 3, "Flavin"),
    ("PETNR",       0.80, 2900, 0.315,   5.4, 3, "Flavin"),
    ("GOx",         0.80, 2900, 0.354,   8.3, 3, "Flavin"),
]


def main():
    print(f"\n{'System':<16} {'κ_p':>7} {'f̄':>7} {'KIE_ad':>8} "
          f"{'KIE_floor':>10} {'KIE_exp':>9} {'KIE_uni':>9} {'Cf_min':>8} {'Constrained?':>14}")
    print("─" * 102)

    records = []
    n_constrained = 0

    for (name, V_el, omH, d0, KIE_exp, cat, cls) in SYSTEMS:
        kap       = kappa_p(V_el, omH)
        KIE_fl, f_bar, f_H, f_D, KIE_ad = kie_floor(V_el, omH)
        KIE_uni   = kie_unified(V_el, omH, d0)
        Cf_min    = northrop_cf_min(KIE_exp, KIE_fl)
        constrained = not np.isnan(Cf_min)
        if constrained:
            n_constrained += 1
        records.append(dict(
            name=name, kap=kap, f_bar=f_bar, f_H=f_H, f_D=f_D,
            KIE_ad=KIE_ad, KIE_fl=KIE_fl, KIE_exp=KIE_exp,
            KIE_uni=KIE_uni, Cf_min=Cf_min, constrained=constrained,
            cat=cat, cls=cls, d0=d0
        ))
        cf_str = f"{Cf_min:8.2f}" if constrained else "     ---"
        flag   = "  *** FLOOR BIND" if constrained else ""
        print(f"{name:<16} {kap:7.4f} {f_bar:7.4f} {KIE_ad:8.2f} "
              f"{KIE_fl:10.2f} {KIE_exp:9.1f} {KIE_uni:9.1f} {cf_str}{flag}")

    print(f"\n{'─'*70}")
    print(f"Systems where KIE_exp < KIE_floor (necessarily commitment-limited): "
          f"{n_constrained}/28")
    print(f"Systems where KIE_exp > KIE_floor (floor not binding): "
          f"{28-n_constrained}/28")

    print(f"\n{'─'*70}")
    print("FLOOR-BINDING SYSTEMS — minimum commitment factors:")
    print(f"{'System':<16} {'κ_p':>7} {'KIE_floor':>10} {'KIE_exp':>9} {'Cf_min':>8}")
    for r in records:
        if r['constrained']:
            print(f"  {r['name']:<14} {r['kap']:7.4f} {r['KIE_fl']:10.2f} "
                  f"{r['KIE_exp']:9.1f} {r['Cf_min']:8.2f}")

    # ── Figure 1: KIE_floor vs KIE_exp ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    kies_floor = [r['KIE_fl']  for r in records]
    kies_exp   = [r['KIE_exp'] for r in records]
    colors     = [CLASS_COLORS[r['cls']] for r in records]
    markers    = [CAT_MARKERS[r['cat']]  for r in records]

    for r in records:
        col = CLASS_COLORS[r['cls']]
        mk  = CAT_MARKERS[r['cat']]
        fc  = col if r['constrained'] else "white"
        ax.scatter(r['KIE_exp'], r['KIE_fl'], c=fc, marker=mk, s=90,
                   edgecolors=col, linewidths=1.5, zorder=3)
        ax.annotate(r['name'], (r['KIE_exp'], r['KIE_fl']),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points", alpha=0.8)

    # Diagonal: floor = exp line (no floor binding above this)
    xmin, xmax = 1.5, 1000
    ax.plot([xmin, xmax], [xmin, xmax], "k--", lw=1.2, alpha=0.6,
            label="KIE$_{\\rm floor}$ = KIE$_{\\rm exp}$")
    ax.fill_between([xmin, xmax], [xmin, xmax], [xmax, xmax],
                    alpha=0.06, color="red", label="Floor-binding region\n(Cf > 0 required)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"KIE$_{\rm exp}$", fontsize=12)
    ax.set_ylabel(r"KIE$_{\rm floor}$ (minimum intrinsic KIE)", fontsize=12)
    ax.set_title("KIE floor vs experiment\n"
                 "Filled = floor-binding (necessarily commitment-limited)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(xmin, xmax); ax.set_ylim(xmin, xmax)

    # class legend
    class_patches = [mpatches.Patch(color=c, label=l)
                     for l, c in CLASS_COLORS.items()]
    leg = ax.legend(handles=class_patches, fontsize=7.5, loc="upper left",
                    title="Mechanism", title_fontsize=8)
    ax.add_artist(leg)
    ax.plot([xmin, xmax], [xmin, xmax], "k--", lw=1.2, alpha=0.6)

    # ── Figure 2: Cf_min vs kappa_p ─────────────────────────────────────────
    ax2 = axes[1]
    kappas  = [r['kap']    for r in records]
    cf_mins = [r['Cf_min'] if r['constrained'] else 0.0 for r in records]

    for r in records:
        col = CLASS_COLORS[r['cls']]
        mk  = CAT_MARKERS[r['cat']]
        cf_val = r['Cf_min'] if r['constrained'] else 0.0
        ax2.scatter(r['kap'], cf_val, c=col, marker=mk, s=90,
                    edgecolors="k", linewidths=0.5, zorder=3,
                    alpha=1.0 if r['constrained'] else 0.3)
        if r['constrained']:
            ax2.annotate(r['name'], (r['kap'], cf_val),
                         fontsize=5.5, xytext=(3, 2), textcoords="offset points")

    # Theoretical floor curve: Cf_min vs kappa for a range of delta_0 -> 0 scenarios
    kap_range = np.linspace(0.02, 0.40, 200)
    # Use median omega_H = 2950 cm^-1 for the theoretical curve
    for omH_label, omH_val, ls in [
            (r"$\omega_H = 2900$ cm$^{-1}$",  2900, "-"),
            (r"$\omega_H = 3100$ cm$^{-1}$",  3100, "--"),
    ]:
        KIE_ad_curve = kie_adiabatic(omH_val)
        # Example: for KIE_exp = 3.5 (typical Regime 3)
        cf_curve = []
        for kap in kap_range:
            kap_D = kap * SQRT2
            f_bar = 0.5 * (lz_weight(kap) + lz_weight(kap_D))
            KIE_fl = np.exp(f_bar * np.log(KIE_ad_curve))
            kie_exp_ref = 3.5  # reference: Regime 3 typical
            cf_val_c = northrop_cf_min(kie_exp_ref, KIE_fl)
            cf_curve.append(cf_val_c if not np.isnan(cf_val_c) else 0.0)
        ax2.plot(kap_range, cf_curve, color="gray", lw=1.5, ls=ls, alpha=0.7,
                 label=f"{omH_label} (KIE$_{{\\rm exp}}$=3.5)")

    ax2.set_xlabel(r"$\kappa_p = V_{\rm el}/\hbar\omega_H$", fontsize=12)
    ax2.set_ylabel(r"Minimum commitment factor $C_f^{\rm min}$", fontsize=12)
    ax2.set_title("Commitment lower bound from KIE floor\n"
                  "Only systems above floor-threshold require Cf > 0", fontsize=10)
    ax2.axvline(0.07,  color="gray", lw=1, ls=":", alpha=0.6, label="Regime 1|2 boundary")
    ax2.axvline(0.15,  color="gray", lw=1, ls="--", alpha=0.6, label="Regime 2|3 boundary")
    ax2.legend(fontsize=7, loc="upper left")

    class_patches2 = [mpatches.Patch(color=c, label=l)
                      for l, c in CLASS_COLORS.items()]
    ax2.legend(handles=class_patches2, fontsize=7.5, loc="upper right",
               title="Mechanism", title_fontsize=8)

    plt.tight_layout()
    out = "pcet_engine/benchmarks/figures/kie_floor.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nFigures saved: {out}")

    # ── Final summary: the key constraint ───────────────────────────────────
    print(f"\n{'═'*70}")
    print("KEY RESULT: Commitment Necessity Criterion")
    print(f"{'═'*70}")
    print("A system is provably commitment-limited if KIE_exp < KIE_floor,")
    print("where KIE_floor = KIE_ad^f_bar depends only on V_el and omega_H.")
    print()
    print("This requires NO knowledge of delta_0 or gating parameters.")
    print("It is a hard bound, not a fit or approximation.")
    print()

    constrained = [r for r in records if r['constrained']]
    print(f"Floor-binding systems ({len(constrained)}/28):")
    for r in sorted(constrained, key=lambda x: -x['Cf_min']):
        print(f"  {r['name']:<14}  κ={r['kap']:.3f}  floor={r['KIE_fl']:.1f}"
              f"  exp={r['KIE_exp']:.1f}  Cf_min={r['Cf_min']:.2f}")

    print()
    print("Regime 3 (κ > 0.15) floor values:")
    for r in sorted(records, key=lambda x: -x['kap'])[:8]:
        print(f"  {r['name']:<14}  κ={r['kap']:.3f}  floor={r['KIE_fl']:.2f}"
              f"  KIE_ad={r['KIE_ad']:.2f}")

    return records


if __name__ == "__main__":
    main()
