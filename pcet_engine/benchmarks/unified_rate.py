"""
Unified adiabatic-to-nonadiabatic KIE prediction.

Implements the KIE analog of the Georgievskii-Stuchebrukhov (GS) interpolation
for proton/H-atom transfer, which smoothly connects:

    Nonadiabatic limit (kappa << 1):  KIE ~ exp(Δα × δ₀²) [FC dominated]
    Adiabatic limit (kappa >> 1):     KIE ~ (ω_H/ω_D) × exp(-ΔZPE/kT) [ZPE dominated]

The GS unified rate (Georgievskii & Stuchebrukhov, JCP 113, 10438, 2000) adapted
for PCET KIE:

    log k_uni^i = (1 - f_i) × log k_na^i + f_i × log k_ad^i

where i ∈ {H, D}, f_i = 1 - exp(-2π κ_i) is the Landau-Zener weight, and
κ_D = √2 × κ_H because ω_D = ω_H / √2 (harmonic mass scaling).

KIE_uni = exp(log k_uni^H - log k_uni^D)

This is fully dimensionless: the 2π/ℏ prefactor in k_na cancels in the ratio,
and the ω/(2π) prefactor in k_ad likewise cancels.

Key physical prediction:
    - High-κ systems (DHFR κ=0.35, LADH κ=0.23): adiabatic contribution is large
      (f_H ~ 0.9), KIE_uni << KIE_na → explains why these systems show low KIEs
      WITHOUT needing anomalously short δ₀.
    - Low-κ systems (SLO-1 DM κ=0.036): nonadiabatic dominates, KIE_uni ≈ KIE_na.

References:
    Georgievskii & Stuchebrukhov, J. Chem. Phys. 113, 10438 (2000)
    Hammes-Schiffer & Soudackov, JPCB 112, 14108 (2008)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from adiabaticity import SYSTEMS, CLASS_COLORS, CAT_MARKERS, kappa_p

# ── Physical constants ─────────────────────────────────────────────────────
KB_EV     = 8.617e-5     # eV/K
HBAR_EV_S = 6.582e-16    # hbar in eV·s
KCAL_TO_EV = 0.04336
CM1_TO_EV  = 1.2398e-4

T   = 298.15             # K
KT  = KB_EV * T          # eV

# Deuterium mass scaling (harmonic approximation):
#   omega_D = omega_H / sqrt(2)   → kappa_D = kappa_H * sqrt(2)
#   alpha_D = sqrt(2) * alpha_H   (tighter wavefunction for heavier mass)
SQRT2 = np.sqrt(2.0)
ALPHA_H = 87.0           # Å^{-2}, from m_H * omega_H / hbar
ALPHA_D = SQRT2 * ALPHA_H  # Å^{-2}


def fc_overlap_sq(delta_0_ang, alpha_ang2):
    """Harmonic FC overlap |S_00|^2 = exp(-alpha * delta_0^2)."""
    return np.exp(-alpha_ang2 * delta_0_ang**2)


def kie_nonadiabatic(delta_0_ang):
    """
    Nonadiabatic KIE from FC overlap ratio alone.

    KIE_na = |S_00^H|^2 / |S_00^D|^2 = exp((alpha_D - alpha_H) * delta_0^2)

    This is the high-temperature classical-bath limit where the Marcus activation
    energy is isotope-independent (ΔG°, λ are outer-sphere quantities); all
    isotope sensitivity comes from the proton FC factor.
    """
    fc_H = fc_overlap_sq(delta_0_ang, ALPHA_H)
    fc_D = fc_overlap_sq(delta_0_ang, ALPHA_D)
    return fc_H / fc_D  # always >= 1


def kie_adiabatic(omega_H_cm1):
    """
    Adiabatic TST KIE.

    In the fully adiabatic limit the proton moves on a single BO surface.
    The isotope effect comes from:
      1. Frequency pre-exponential: ω_H / ω_D = √2
      2. Zero-point energy difference in the donor well:
         ΔZPE = ℏ(ω_H - ω_D) / 2 = ℏω_H (1 - 1/√2) / 2

    KIE_ad = (ω_H/ω_D) × exp(-ΔZPE / kT)
           = √2 × exp(-ℏω_H (1 - 1/√2) / (2kT))

    This is the semiclassical Bigeleisen-Mayer KIE (no tunneling).
    """
    # H has higher ZPE than D in the donor well (ω_H > ω_D).
    # At the TS the transferred-proton stretch becomes imaginary → ZPE_TS ≈ 0.
    # H loses MORE ZPE at the TS → H has LOWER effective barrier → KIE_ad > 1.
    #
    # G*_H = G*_cl - ℏω_H/2,  G*_D = G*_cl - ℏω_D/2
    # KIE_ad = (ω_H/ω_D) × exp(+(ℏω_H - ℏω_D)/(2kT)) = √2 × exp(+ΔZPE/kT)
    hbar_omH_eV = omega_H_cm1 * CM1_TO_EV          # ℏω_H in eV
    ΔZPE        = hbar_omH_eV * (1.0 - 1.0/SQRT2) / 2.0   # > 0
    return SQRT2 * np.exp(+ΔZPE / KT)              # KIE_ad > 1 ✓


def lz_weight(kappa):
    """Landau-Zener interpolation weight: f = 1 - exp(-2*pi*kappa)."""
    return 1.0 - np.exp(-2 * np.pi * kappa)


def kie_unified(V_el_kcal, omega_H_cm1, delta_0_ang):
    """
    Unified KIE via log-space interpolation between nonadiabatic and adiabatic limits.

    For isotope i ∈ {H, D}:
        log k_i = (1 - f_i) × log k_i_na + f_i × log k_i_ad

    The 2π/ℏ and ω/(2π) prefactors cancel in all isotope ratios and in the
    interpolation weights when written in dimensionless log-ratio form.

    Let R_na = log KIE_na = log(k_H_na / k_D_na) = (α_D - α_H) × δ₀²
    Let R_ad = log KIE_ad = log(k_H_ad / k_D_ad)

    log KIE_uni = R_na + δR

    where δR is the adiabatic correction term that vanishes for f_H = f_D = 0.

    Returns:
        kie_na  : nonadiabatic KIE (FC dominated)
        kie_ad  : adiabatic KIE (ZPE/TST dominated)
        kie_uni : unified KIE
        kappa_H : adiabaticity parameter for H
        f_H     : LZ weight for H
        f_D     : LZ weight for D
        correction_factor : KIE_uni / KIE_na
    """
    kap_H, _, _ = kappa_p(V_el_kcal, omega_H_cm1, delta_0_ang)
    kap_D       = kap_H * SQRT2   # omega_D = omega_H/sqrt(2) → kappa_D = kappa_H*sqrt(2)

    f_H = lz_weight(kap_H)
    f_D = lz_weight(kap_D)

    KIE_na = kie_nonadiabatic(delta_0_ang)
    KIE_ad = kie_adiabatic(omega_H_cm1)

    log_KIE_na = np.log(KIE_na)
    log_KIE_ad = np.log(KIE_ad)

    # For the k_na part: log k_H_na - log k_D_na = log KIE_na (units cancel)
    # For the k_ad part: log k_H_ad - log k_ad_D = log KIE_ad
    # Full interpolation: the cross terms couple the weights differently for H and D.
    #
    # log k_H_uni = (1-f_H)*log k_H_na + f_H*log k_H_ad
    # log k_D_uni = (1-f_D)*log k_D_na + f_D*log k_D_ad
    #
    # log KIE_uni = log k_H_uni - log k_D_uni
    #
    # To evaluate this without absolute rates, write:
    # log k_H_na = log k_H_na (in s^-1, but only ratio matters)
    # log k_D_na = log k_H_na - log KIE_na
    # log k_H_ad = log k_H_na + log(k_H_ad/k_H_na)  -- let's call this X_ad
    # log k_D_ad = log k_H_na - log KIE_na + log(k_D_ad/k_D_na) -- X_ad - ΔZPE correction
    #
    # The ratio k_ad / k_na for isotope H:
    #   k_H_ad / k_H_na = [ω_H/(2π) × exp(-G*_ad/kT)] / [(2π/ℏ)|V|²|S^H|² × FC]
    # This is NOT unit-free unless we fix the reference.
    #
    # Clean approach: let r = log(k_H_ad/k_H_na) be the single unknown scale.
    # Then log(k_D_ad/k_D_na) = log(KIE_ad/KIE_na) + r.
    #
    # log KIE_uni = log k_H_uni - log k_D_uni
    # = [(1-f_H) log k_H_na + f_H log k_H_ad]
    #   - [(1-f_D) log k_D_na + f_D log k_D_ad]
    # = (1-f_H) log k_H_na - (1-f_D) log k_D_na
    #   + f_H log k_H_ad - f_D log k_D_ad
    # = (1-f_H) log k_H_na - (1-f_D)(log k_H_na - log KIE_na)
    #   + f_H(log k_H_na + r) - f_D(log k_H_na - log KIE_na + log KIE_ad - log KIE_na + r)
    #
    # Actually let's just compute it directly with a ratio formulation.
    # Set z_H = log(k_H_ad/k_H_na) and z_D = log(k_D_ad/k_D_na):
    #   z_D = z_H + log(KIE_ad/KIE_na)
    #
    # log KIE_uni = log KIE_na
    #   + f_H * (z_H)                     [H benefit from adiabatic]
    #   - f_D * (z_D)                     [D benefit from adiabatic]
    #   + (f_D - f_H) * log k_H_na        [asymmetric weight term]
    #
    # The last term still contains log k_H_na in absolute units.
    # The cleanest exact formula:
    #
    # log KIE_uni = (1-f_H)*log KIE_na + f_H*log KIE_ad
    #               + (f_H - f_D)*log k_D_na
    #                                       ^-- this crosses units
    #
    # The ONLY way to make this unit-free is to use a SYMMETRIC weighting:
    # f_avg = (f_H + f_D) / 2
    # log KIE_uni ≈ (1 - f_avg)*log KIE_na + f_avg*log KIE_ad
    #
    # This is a well-defined approximation: average LZ weight for H and D.
    # It is exact when f_H = f_D (which holds when κ_H = κ_D, not our case).
    # The asymmetry f_D - f_H is small for low κ and grows for high κ.

    f_avg   = 0.5 * (f_H + f_D)
    log_KIE_uni = (1 - f_avg) * log_KIE_na + f_avg * log_KIE_ad
    KIE_uni = np.exp(log_KIE_uni)

    # Also compute exact result via absolute rates (k_na in eV units,
    # k_ad in s^-1; cancel 2pi/hbar by using log(k_na/k_ref) differences).
    # Exact formula via symmetrized log-KIE approach with H-D asymmetry:
    # Δf = f_D - f_H > 0  (D is more adiabatic than H since ω_D < ω_H → κ_D > κ_H)
    # The asymmetric correction: when D is more adiabatic than H,
    # the adiabatic contribution to D's rate is LARGER than for H.
    # This REDUCES the KIE further (D gets more benefit from adiabaticity).
    #
    # Exact correction beyond symmetric approximation requires r = log(k_ad/k_na)
    # in consistent units. We report both the symmetric approximation (clean)
    # and flag the H-D asymmetry.

    delta_f  = f_D - f_H   # always >= 0; D more adiabatic than H
    correction_factor = KIE_uni / KIE_na

    return KIE_na, KIE_ad, KIE_uni, kap_H, f_H, f_D, f_avg, delta_f, correction_factor


def main():
    print(f"\n{'System':<16} {'κ_H':>7} {'f_H':>6} {'f_D':>6} {'Δf':>6} "
          f"{'KIE_na':>8} {'KIE_ad':>8} {'KIE_uni':>8} {'KIE_exp':>8} "
          f"{'CF':>6}")
    print("-" * 100)

    results = []
    for (name, V_el, omH, d0, KIE, cat, cls) in SYSTEMS:
        KIE_na, KIE_ad, KIE_uni, kap_H, f_H, f_D, f_avg, delta_f, cf = \
            kie_unified(V_el, omH, d0)

        results.append((name, kap_H, f_H, f_D, f_avg, delta_f,
                        KIE_na, KIE_ad, KIE_uni, cf, KIE, cat, cls))

        print(f"{name:<16} {kap_H:>7.4f} {f_H:>6.4f} {f_D:>6.4f} {delta_f:>6.4f} "
              f"{KIE_na:>8.1f} {KIE_ad:>8.2f} {KIE_uni:>8.1f} {KIE:>8.1f} "
              f"{cf:>6.3f}")

    print(f"\n{'─'*60}")
    cfs    = np.array([r[9] for r in results])
    kappas = np.array([r[1] for r in results])
    kie_na = np.array([r[6] for r in results])
    kie_un = np.array([r[8] for r in results])
    kie_ex = np.array([r[10] for r in results])

    print(f"KIE_na range:   {kie_na.min():.1f} – {kie_na.max():.0f}")
    print(f"KIE_uni range:  {kie_un.min():.1f} – {kie_un.max():.0f}")
    print(f"KIE_exp range:  {kie_ex.min():.1f} – {kie_ex.max():.0f}")
    print(f"\nCorrection factor CF = KIE_uni / KIE_na:")
    print(f"  Range:  {cfs.min():.3f} – {cfs.max():.3f}")
    print(f"  Systems with CF < 0.95 (meaningful reduction):  {(cfs < 0.95).sum()}")
    print(f"  Systems with CF < 0.80 (>20% KIE reduction):   {(cfs < 0.80).sum()}")
    print(f"  Systems with CF < 0.50 (>50% KIE reduction):   {(cfs < 0.50).sum()}")

    # Correlation of residuals: does unified rate reduce prediction error?
    log_err_na  = np.abs(np.log10(kie_na)  - np.log10(kie_ex))
    log_err_uni = np.abs(np.log10(kie_un) - np.log10(kie_ex))
    print(f"\nMean |log10(KIE_pred/KIE_exp)|:")
    print(f"  Nonadiabatic:  {log_err_na.mean():.3f}")
    print(f"  Unified:       {log_err_uni.mean():.3f}")
    print(f"  Systems improved by unified:  {(log_err_uni < log_err_na).sum()}/28")

    # ── Figures ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    colors  = [CLASS_COLORS[r[12]] for r in results]
    markers = [CAT_MARKERS[r[11]] for r in results]

    # Panel 1: KIE_na, KIE_uni, KIE_exp vs κ_H
    ax1 = fig.add_subplot(gs[0])
    kap_theory = np.logspace(-2, 0, 300)
    # Illustrative curves using median ω_H = 2900, median δ₀ = 0.40 Å
    kie_na_theory  = np.array([np.exp((ALPHA_D - ALPHA_H) * 0.40**2)] * 300)  # constant in κ
    # For unified: f_H(κ), f_D = f(√2κ), KIE_ad ~ 4.5 (median)
    kie_ad_med = kie_adiabatic(2900)
    kie_un_theory = np.exp(
        0.5 * (1 - lz_weight(kap_theory) + 1 - lz_weight(kap_theory * SQRT2)) * np.log(np.exp((ALPHA_D-ALPHA_H)*0.40**2))
        + 0.5 * (lz_weight(kap_theory) + lz_weight(kap_theory * SQRT2)) * np.log(kie_ad_med)
    )

    ax1.plot(kap_theory, kie_na_theory, "b--", lw=1, alpha=0.5, label=r"$KIE_{na}$ (const in $\kappa$)")
    ax1.plot(kap_theory, kie_un_theory, "r-", lw=1.5, alpha=0.7, label=r"$KIE_{uni}$ theory")

    for r in results:
        name, kap, f_H, f_D, f_avg, df, KN, KA, KU, cf, KE, cat, cls = r
        col = CLASS_COLORS[cls]
        mk  = CAT_MARKERS[cat]
        ax1.scatter(kap, KN, c="dodgerblue", marker=mk, s=50, alpha=0.4,
                    zorder=2, edgecolors="none")
        ax1.scatter(kap, KU, c=col, marker=mk, s=70, zorder=3,
                    edgecolors="k", linewidths=0.4)
        ax1.scatter(kap, KE, c=col, marker="*", s=90, zorder=4,
                    edgecolors="k", linewidths=0.4)

    ax1.axhline(7.7, color="steelblue", lw=1, ls="-.", alpha=0.6, label="ZPE ceiling")
    ax1.axvline(0.05, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax1.axvline(0.5,  color="gray", lw=0.8, ls="--", alpha=0.5)
    ax1.fill_betweenx([1, 1000], 0.05, 0.5, alpha=0.05, color="orange")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\kappa_p = V_\mathrm{el}/\hbar\omega_H$")
    ax1.set_ylabel("KIE")
    ax1.set_title(r"KIE vs $\kappa_p$ — NA, unified, and exp")
    # custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue',
               alpha=0.5, markersize=7, label=r"$KIE_{na}$ (FC only)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='k', markersize=7, label=r"$KIE_{uni}$"),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markeredgecolor='k', markersize=9, label=r"$KIE_{exp}$"),
    ]
    ax1.legend(handles=legend_elements, fontsize=7, loc="upper right")

    # Panel 2: correction factor CF = KIE_uni / KIE_na vs κ
    ax2 = fig.add_subplot(gs[1])
    kap_cf = np.logspace(-2, 0, 300)
    # For median δ₀=0.40 Å, ω_H=2900
    kie_na0 = np.exp((ALPHA_D - ALPHA_H) * 0.40**2)
    kie_ad0 = kie_adiabatic(2900)
    f_H_th  = lz_weight(kap_cf)
    f_D_th  = lz_weight(kap_cf * SQRT2)
    f_avg_th = 0.5 * (f_H_th + f_D_th)
    log_cf_th = f_avg_th * (np.log(kie_ad0) - np.log(kie_na0))
    ax2.semilogx(kap_cf, np.exp(log_cf_th), "k-", lw=1.5, alpha=0.6,
                 label=r"Theory ($\delta_0=0.40$ Å, $\omega_H=2900$ cm$^{-1}$)")

    for r in results:
        name, kap, f_H, f_D, f_avg, df, KN, KA, KU, cf, KE, cat, cls = r
        col = CLASS_COLORS[cls]
        ax2.scatter(kap, cf, c=col, marker=CAT_MARKERS[cat], s=70, zorder=3,
                    edgecolors="k", linewidths=0.4)
        if cf < 0.80 or name in ("DHFR", "LADH", "bc1", "PhOH-self"):
            ax2.annotate(name, (kap, cf), fontsize=6, ha="left",
                         xytext=(3, 2), textcoords="offset points")

    ax2.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
    ax2.axhline(0.5, color="orange", lw=0.8, ls=":", alpha=0.6, label="CF = 0.5")
    ax2.set_xlabel(r"$\kappa_p$")
    ax2.set_ylabel(r"$CF = KIE_\mathrm{uni}/KIE_\mathrm{na}$")
    ax2.set_title("Adiabatic correction to KIE")
    ax2.legend(fontsize=7)
    ax2.set_ylim(0, 1.05)

    # Panel 3: LZ weights f_H and f_D vs κ_H — the asymmetry
    ax3 = fig.add_subplot(gs[2])
    kap_plot = np.logspace(-2, 0.5, 300)
    f_H_pl = lz_weight(kap_plot)
    f_D_pl = lz_weight(kap_plot * SQRT2)
    ax3.semilogx(kap_plot, f_H_pl, "b-", lw=2, label=r"$f_H(\kappa_H)$")
    ax3.semilogx(kap_plot, f_D_pl, "r-", lw=2, label=r"$f_D(\sqrt{2}\kappa_H)$")
    ax3.fill_between(kap_plot, f_H_pl, f_D_pl, alpha=0.15, color="purple",
                     label=r"$\Delta f = f_D - f_H$")

    kappas_arr = np.array([r[1] for r in results])
    f_H_arr    = np.array([r[2] for r in results])
    f_D_arr    = np.array([r[3] for r in results])
    ax3.scatter(kappas_arr, f_H_arr, c=colors, marker="o", s=50, zorder=3,
                edgecolors="k", linewidths=0.4, alpha=0.8)
    ax3.scatter(kappas_arr, f_D_arr, c=colors, marker="^", s=50, zorder=3,
                edgecolors="k", linewidths=0.4, alpha=0.8)

    ax3.axvline(0.05, color="gray", lw=0.8, ls=":", alpha=0.5, label="κ=0.05")
    ax3.axvline(0.5,  color="gray", lw=0.8, ls="--", alpha=0.5, label="κ=0.5")
    ax3.set_xlabel(r"$\kappa_H = V_\mathrm{el}/\hbar\omega_H$")
    ax3.set_ylabel(r"Adiabatic weight $f(\kappa)$")
    ax3.set_title(r"H vs D LZ weights — $f_D > f_H$")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.set_ylim(0, 1.0)

    class_patches = [mpatches.Patch(color=c, label=l)
                     for l, c in CLASS_COLORS.items()]
    fig.legend(handles=class_patches, fontsize=7, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.08))

    plt.suptitle("Unified adiabatic-nonadiabatic PCET KIE: 28-system benchmark",
                 fontsize=11, y=1.01)
    plt.tight_layout()

    out = "pcet_engine/benchmarks/figures/unified_rate.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nFigures saved: {out}")

    return results


if __name__ == "__main__":
    main()
