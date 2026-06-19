"""
Self-consistent delta_0 recalibration under the unified LZ framework.

The benchmark delta_0 values were calibrated so that the FULL nonadiabatic
engine (multi-channel Boltzmann sum, SHS formalism) matches KIE_exp.  Here
we find delta_0_uni such that the simplified unified KIE formula matches
KIE_exp directly, for all 24 non-floor-binding systems.

For floor-binding systems (DHFR, LADH, bc1, PhOH-self), KIE_exp < KIE_floor,
so no real-valued delta_0 can satisfy KIE_uni = KIE_exp.  These systems must
be commitment-limited; we calibrate delta_0_uni to match KIE_int = KIE_floor
(the tightest compatible assumption) and report Cf_min separately.

Physical expectation:
    At fixed V_el, omega_H:
    KIE_uni(delta_0) < KIE_na(delta_0) for all delta_0 (adiabatic contribution
    reduces the ratio below the pure FC value).
    To match the same KIE_exp, the unified model requires LARGER delta_0 than
    the nonadiabatic approximation.

    delta_0_uni > delta_0_na  (for Regime 1/2 systems, KIE_exp >> KIE_floor)

    Exceptions: if the current delta_0_na was calibrated to the FULL engine
    (which already includes Boltzmann-weighted multi-channel FC overlaps and
    gives lower KIE than the simplified single-channel formula), then
    delta_0_na > delta_0_simplified_na, and the sign of delta_0_uni - delta_0_na
    can go either way.

Output:
    Table: delta_0_na vs delta_0_uni, fractional change, mechanistic class
    Figure 1: delta_0_na vs delta_0_uni scatter with 1:1 line
    Figure 2: Fractional change (delta_0_uni - delta_0_na)/delta_0_na vs kappa_p
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import brentq

# ── Constants ─────────────────────────────────────────────────────────────────
KCAL_TO_EV = 0.04336
CM1_TO_EV  = 1.2398e-4
SQRT2      = np.sqrt(2.0)
KT         = 0.02569
ALPHA_H    = 87.0

CLASS_COLORS = {
    "SLO-1 family": "#2166ac",
    "HAT":          "#d73027",
    "Hydride":      "#1a9850",
    "Proton/O-H":   "#f46d43",
    "Flavin":       "#8073ac",
}
CAT_MARKERS = {1: "o", 2: "s", 3: "^"}


def kappa_p(V_el_kcal, omega_H_cm1):
    return (V_el_kcal * KCAL_TO_EV) / (omega_H_cm1 * CM1_TO_EV)


def lz_weight(kappa):
    return 1.0 - np.exp(-2.0 * np.pi * kappa)


def kie_adiabatic(omega_H_cm1):
    hbar_omH = omega_H_cm1 * CM1_TO_EV
    delta_zpe = hbar_omH * (1.0 - 1.0 / SQRT2) / 2.0
    return SQRT2 * np.exp(delta_zpe / KT)


def kie_nonadiabatic(delta_0_ang):
    alpha_D = SQRT2 * ALPHA_H
    return np.exp((alpha_D - ALPHA_H) * delta_0_ang**2)


def f_bar(kap_H):
    kap_D = kap_H * SQRT2
    return 0.5 * (lz_weight(kap_H) + lz_weight(kap_D))


def kie_unified(V_el_kcal, omega_H_cm1, delta_0_ang):
    kap_H  = kappa_p(V_el_kcal, omega_H_cm1)
    fb     = f_bar(kap_H)
    KIE_na = kie_nonadiabatic(delta_0_ang)
    KIE_ad = kie_adiabatic(omega_H_cm1)
    return np.exp((1 - fb) * np.log(KIE_na) + fb * np.log(KIE_ad))


def kie_floor(V_el_kcal, omega_H_cm1):
    kap_H  = kappa_p(V_el_kcal, omega_H_cm1)
    fb     = f_bar(kap_H)
    KIE_ad = kie_adiabatic(omega_H_cm1)
    return np.exp(fb * np.log(KIE_ad))


def bisect_delta0_uni(V_el_kcal, omega_H_cm1, KIE_target,
                      d0_lo=0.001, d0_hi=0.90):
    """
    Find delta_0_uni such that KIE_uni(delta_0_uni) = KIE_target.
    Returns NaN if KIE_target < KIE_floor (no solution exists).
    """
    fl = kie_floor(V_el_kcal, omega_H_cm1)
    if KIE_target <= fl:
        return np.nan, fl

    def residual(d0):
        return kie_unified(V_el_kcal, omega_H_cm1, d0) - KIE_target

    if residual(d0_hi) < 0:
        # KIE_uni(d0_hi) < KIE_target — extend upper bound
        d0_hi = 0.999
    if residual(d0_hi) < 0:
        return np.nan, fl  # can't reach target even at 1 Ang

    d0_star = brentq(residual, d0_lo, d0_hi, xtol=1e-6, rtol=1e-6)
    return d0_star, fl


# ── Benchmark data ─────────────────────────────────────────────────────────────
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
    print(f"\n{'System':<16} {'κ_p':>7} {'δ₀_na':>8} {'δ₀_uni':>8} "
          f"{'Δδ₀':>8} {'Δδ₀%':>7} {'Floor?':>8} {'Note'}")
    print("─" * 90)

    records = []
    d0_na_arr, d0_uni_arr = [], []
    kappas, delta_fracs, colors, markers = [], [], [], []

    for (name, V_el, omH, d0, KIE_exp, cat, cls) in SYSTEMS:
        kap = kappa_p(V_el, omH)
        d0_uni, KIE_fl = bisect_delta0_uni(V_el, omH, KIE_exp)
        floor_bound = np.isnan(d0_uni)

        if floor_bound:
            d0_uni_val = np.nan
            delta_d0   = np.nan
            delta_pct  = np.nan
            note = "FLOOR-BINDING (Cf required)"
        else:
            d0_uni_val = d0_uni
            delta_d0   = d0_uni - d0
            delta_pct  = 100.0 * (d0_uni - d0) / d0
            note = ""

        records.append(dict(
            name=name, kap=kap, d0_na=d0, d0_uni=d0_uni_val,
            delta_d0=delta_d0, delta_pct=delta_pct,
            floor_bound=floor_bound, KIE_fl=KIE_fl, KIE_exp=KIE_exp,
            cat=cat, cls=cls,
        ))

        d0_uni_str   = f"{d0_uni_val:8.4f}" if not floor_bound else "     ---"
        delta_str    = f"{delta_d0:+8.4f}"  if not floor_bound else "     ---"
        delta_pct_str = f"{delta_pct:+7.1f}%" if not floor_bound else "     ---"
        print(f"{name:<16} {kap:7.4f} {d0:8.4f} {d0_uni_str} "
              f"{delta_str} {delta_pct_str} {'YES':>8}  {note}" if floor_bound else
              f"{name:<16} {kap:7.4f} {d0:8.4f} {d0_uni_str} "
              f"{delta_str} {delta_pct_str} {'':>8}  {note}")

        if not floor_bound:
            d0_na_arr.append(d0)
            d0_uni_arr.append(d0_uni_val)
            kappas.append(kap)
            delta_fracs.append(delta_pct)
            colors.append(CLASS_COLORS[cls])
            markers.append(CAT_MARKERS[cat])

    # Summary statistics (non-floor-binding only)
    d0_na_arr  = np.array(d0_na_arr)
    d0_uni_arr = np.array(d0_uni_arr)
    delta_pcts = np.array(delta_fracs)

    print(f"\n{'─'*70}")
    print(f"Non-floor-binding systems (24/28):")
    print(f"  Mean δ₀_na  = {d0_na_arr.mean():.4f} Å  (σ = {d0_na_arr.std():.4f})")
    print(f"  Mean δ₀_uni = {d0_uni_arr.mean():.4f} Å  (σ = {d0_uni_arr.std():.4f})")
    print(f"  Mean Δδ₀    = {(d0_uni_arr-d0_na_arr).mean():+.4f} Å  "
          f"({delta_pcts.mean():+.1f}%)")
    print(f"  δ₀_uni > δ₀_na: {(d0_uni_arr > d0_na_arr).sum()}/24 systems")
    print(f"  δ₀_uni < δ₀_na: {(d0_uni_arr < d0_na_arr).sum()}/24 systems")
    print(f"\nPhysical interpretation:")
    print(f"  Systems where δ₀_uni > δ₀_na: nonadiabatic model UNDERESTIMATES")
    print(f"    tunneling distance needed to explain KIE_exp (adiabatic contribution")
    print(f"    reduces KIE, so larger δ₀ required to compensate).")
    print(f"  Systems where δ₀_uni < δ₀_na: nonadiabatic model used an inflated")
    print(f"    δ₀ to compensate for what the adiabatic contribution now provides.")

    # correlation d0_na vs d0_uni
    r_corr = np.corrcoef(d0_na_arr, d0_uni_arr)[0, 1]
    print(f"\n  Correlation δ₀_na vs δ₀_uni: r = {r_corr:.4f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fig 1: delta_0_na vs delta_0_uni scatter
    ax = axes[0]
    for r in records:
        if r['floor_bound']:
            continue
        col = CLASS_COLORS[r['cls']]
        mk  = CAT_MARKERS[r['cat']]
        ax.scatter(r['d0_na'], r['d0_uni'], c=col, marker=mk, s=90,
                   edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(r['name'], (r['d0_na'], r['d0_uni']),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points", alpha=0.8)

    lims = [0.20, 0.70]
    ax.plot(lims, lims, "k--", lw=1.2, alpha=0.6, label="δ₀_uni = δ₀_na")
    ax.set_xlabel(r"$\delta_0^{\rm na}$ (Å) — nonadiabatic calibration", fontsize=11)
    ax.set_ylabel(r"$\delta_0^{\rm uni}$ (Å) — unified calibration", fontsize=11)
    ax.set_title(f"Self-consistent recalibration (r = {r_corr:.3f})\n"
                 f"24 non-floor-binding systems", fontsize=10)
    ax.set_xlim(lims); ax.set_ylim(lims)
    class_patches = [mpatches.Patch(color=c, label=l)
                     for l, c in CLASS_COLORS.items()]
    ax.legend(handles=class_patches, fontsize=7.5, loc="upper left",
              title="Mechanism", title_fontsize=8)

    # Fig 2: fractional shift vs kappa_p
    ax2 = axes[1]
    for r in records:
        if r['floor_bound']:
            continue
        col = CLASS_COLORS[r['cls']]
        mk  = CAT_MARKERS[r['cat']]
        ax2.scatter(r['kap'], r['delta_pct'], c=col, marker=mk, s=90,
                    edgecolors="k", linewidths=0.5, zorder=3)
        if abs(r['delta_pct']) > 8:
            ax2.annotate(r['name'], (r['kap'], r['delta_pct']),
                         fontsize=5.5, xytext=(3, 2), textcoords="offset points")

    ax2.axhline(0, color="k", lw=1, ls="--", alpha=0.5)
    ax2.axvline(0.07,  color="gray", lw=1, ls=":",  alpha=0.6, label="Regime 1|2")
    ax2.axvline(0.15,  color="gray", lw=1, ls="--", alpha=0.6, label="Regime 2|3")
    ax2.fill_between([0.0, 0.07],   -40, 60, alpha=0.04, color="blue")
    ax2.fill_between([0.07, 0.15],  -40, 60, alpha=0.04, color="orange")
    ax2.fill_between([0.15, 0.40],  -40, 60, alpha=0.04, color="red")

    ax2.set_xlabel(r"$\kappa_p = V_{\rm el}/\hbar\omega_H$", fontsize=11)
    ax2.set_ylabel(r"$(\delta_0^{\rm uni} - \delta_0^{\rm na})/\delta_0^{\rm na}$ (%)",
                   fontsize=11)
    ax2.set_title("Fractional shift in donor-acceptor geometry\nupon unified recalibration",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_ylim(-40, 60)

    ax2.text(0.035, 52, "Regime 1\n(tunneling)", ha="center", fontsize=8,
             color="#2166ac", style="italic")
    ax2.text(0.11,  52, "Regime 2\n(intermediate)", ha="center", fontsize=8,
             color="darkorange", style="italic")
    ax2.text(0.22,  52, "Regime 3\n(near-adiabatic)", ha="center", fontsize=8,
             color="firebrick", style="italic")

    plt.tight_layout()
    out = "pcet_engine/benchmarks/figures/unified_recalibration.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nFigures saved: {out}")
    return records


if __name__ == "__main__":
    main()
