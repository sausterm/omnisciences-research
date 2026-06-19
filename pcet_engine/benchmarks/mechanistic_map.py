"""
Mechanistic map of enzyme PCET: adiabaticity vs commitment factors.

Three analyses unified in one script:

1. SWAIN-SCHAAD EXPONENT (standard definition)
   ρ_std = ln(k_H/k_T) / ln(k_D/k_T)
   Semiclassical: 3.26–3.34.  Tunneling-inflated: > 3.34.
   Pure FC nonadiabatic limit: (√3−1)/(√3−√2) = 2.30  [sub-semiclassical]
   Unified theory: ρ_uni(κ) ∈ (2.30, 3.34), approaching 3.34 for high κ.

2. COMMITMENT FACTOR ANALYSIS (Northrop equation)
   For systems where KIE_exp < KIE_ad (below the ZPE floor, regime 3):
       KIE_obs = (KIE_int + Cf) / (1 + Cf)   →   Cf = (KIE_int − KIE_obs) / (KIE_obs − 1)
   where KIE_int ≈ KIE_ad (Bigeleisen semiclassical) when commitment dominates.
   This quantifies how much of the KIE suppression is kinetic vs. mechanistic.

3. 2D MECHANISTIC MAP
   x-axis: κ_p (adiabaticity parameter)
   y-axis: Cf_implied (commitment factor required to bring KIE_ad to KIE_exp)
   The three regimes cluster naturally:
     - Regime 1 (SLO-1, κ < 0.07): high tunneling, low commitment
     - Regime 2 (HAT/flavin, κ 0.07–0.15): intermediate tunneling, low-mod commitment
     - Regime 3 (hydrides/proton, κ > 0.15): near-adiabatic, high implied commitment

References:
    Northrop, D. B. Biochemistry 14, 2644 (1975); Annu. Rev. Biochem. 50, 103 (1981).
    Kohen, A.; Klinman, J. P. Acc. Chem. Res. 31, 397 (1998).
    Swain, C. G. et al. J. Am. Chem. Soc. 80, 5885 (1958).
    Bigeleisen, J.; Mayer, M. G. J. Chem. Phys. 15, 261 (1947).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── Reuse parameter block from adiabaticity.py ────────────────────────────
# (copied directly to avoid import path issues when run as script)

KCAL_TO_EV = 0.04336
CM1_TO_EV  = 1.2398e-4
KB_EV      = 8.617e-5
T          = 298.15
KT         = KB_EV * T
SQRT2      = np.sqrt(2.0)
SQRT3      = np.sqrt(3.0)
ALPHA_H    = 87.0             # Å^{-2}  (m_H × ω_H / ℏ)
ALPHA_D    = SQRT2 * ALPHA_H  # Å^{-2}  (ω_D = ω_H/√2, m_D = 2 m_H)
ALPHA_T    = SQRT3 * ALPHA_H  # Å^{-2}  (ω_T = ω_H/√3, m_T = 3 m_H)

CLASS_COLORS = {
    "SLO-1 family": "#2166ac",
    "HAT":          "#d73027",
    "Hydride":      "#1a9850",
    "Proton/O-H":   "#f46d43",
    "Flavin":       "#8073ac",
}
CAT_MARKERS = {1: "o", 2: "s", 3: "^"}

# (name, V_el kcal/mol, omega_H cm^-1, delta_0 Å, KIE_exp, cat, mech_class)
SYSTEMS = [
    ("SLO-1 WT",      0.60, 2900, 0.500,  81.0, 1, "SLO-1 family"),
    ("SLO-1 L546A",   0.50, 2900, 0.514,  93.0, 3, "SLO-1 family"),
    ("SLO-1 L754A",   0.45, 2900, 0.524, 112.0, 3, "SLO-1 family"),
    ("SLO-1 DM",      0.30, 2900, 0.614, 661.0, 3, "SLO-1 family"),
    ("SLO-1 I553G",   0.25, 2900, 0.518, 100.0, 3, "SLO-1 family"),
    ("SLO-1 I553A",   0.35, 2900, 0.510,  87.0, 3, "SLO-1 family"),
    ("SLO-1 I553V",   0.45, 2900, 0.499,  72.0, 3, "SLO-1 family"),
    ("SLO-1 I553L",   0.50, 2900, 0.493,  65.0, 3, "SLO-1 family"),
    ("SLO-1 I553F",   0.55, 2900, 0.499,  71.0, 3, "SLO-1 family"),
    ("AADH",          0.80, 3000, 0.465,  55.0, 2, "HAT"),
    ("MADH",          0.50, 2950, 0.432,  30.0, 2, "HAT"),
    ("PHM",           1.20, 3100, 0.347,  10.0, 1, "HAT"),
    ("RNR",           0.20, 2600, 0.348,   7.0, 2, "HAT"),
    ("GO",            0.70, 2900, 0.422,  22.0, 3, "HAT"),
    ("TauD",          0.50, 2900, 0.436,  27.0, 3, "HAT"),
    ("DβH",           0.60, 2900, 0.370,  10.8, 3, "HAT"),
    ("CAO",           0.60, 2900, 0.379,  12.0, 3, "HAT"),
    ("LADH",          2.00, 3000, 0.264,   3.5, 3, "Hydride"),
    ("DHFR",          3.00, 3000, 0.247,   3.0, 3, "Hydride"),
    ("TSase",         0.80, 2950, 0.318,   6.0, 3, "Hydride"),
    ("MAO",           0.50, 2950, 0.345,   8.0, 3, "Hydride"),
    ("bc1",           1.50, 3300, 0.250,   3.5, 3, "Proton/O-H"),
    ("PhOH-self",     1.50, 3000, 0.279,   4.1, 3, "Proton/O-H"),
    ("RNR-3FY",       0.50, 3200, 0.303,   6.0, 3, "Proton/O-H"),
    ("RNR-2FY",       0.40, 3200, 0.287,   5.0, 3, "Proton/O-H"),
    ("MR",            0.80, 2900, 0.440,  25.0, 3, "Flavin"),
    ("PETNR",         0.80, 2900, 0.315,   5.4, 3, "Flavin"),
    ("GOx",           0.80, 2900, 0.354,   8.3, 3, "Flavin"),
]


# ── Physical functions ─────────────────────────────────────────────────────

def kappa_p(V_el_kcal, omega_H_cm1):
    """Proton adiabaticity: κ_H = V_el / (ℏ ω_H)."""
    return (V_el_kcal * KCAL_TO_EV) / (omega_H_cm1 * CM1_TO_EV)


def lz_weight(kappa):
    return 1.0 - np.exp(-2.0 * np.pi * kappa)


def kie_adiabatic_bigeleisen(omega_H_cm1):
    """Bigeleisen semiclassical KIE (H/D), ZPE only.
       KIE_ad = √2 × exp(ℏω_H(1 − 1/√2) / (2kT))
    """
    hbar_om = omega_H_cm1 * CM1_TO_EV
    ΔZPE = hbar_om * (1.0 - 1.0/SQRT2) / 2.0
    return SQRT2 * np.exp(ΔZPE / KT)


def kie_ad_HT(omega_H_cm1):
    """Bigeleisen KIE (H/T)."""
    hbar_om = omega_H_cm1 * CM1_TO_EV
    ΔZPE = hbar_om * (1.0 - 1.0/SQRT3) / 2.0
    return SQRT3 * np.exp(ΔZPE / KT)


def kie_ad_DT(omega_H_cm1):
    """Bigeleisen KIE (D/T)."""
    hbar_om = omega_H_cm1 * CM1_TO_EV
    # ω_D = ω_H/√2, so ℏω_D = ℏω_H/√2
    hbar_om_D = hbar_om / SQRT2
    ΔZPE = hbar_om_D * (1.0 - 1.0/SQRT3) / 2.0   # ω_T = ω_D/√(3/2)
    # Actually: m_T/m_D = 3/2, so ω_T = ω_D/√(3/2), ℏω_T = ℏω_D × √(2/3)
    hbar_om_T = hbar_om_D * np.sqrt(2.0/3.0)
    ΔZPE_DT = (hbar_om_D - hbar_om_T) / 2.0
    return np.sqrt(3.0/2.0) * np.exp(ΔZPE_DT / KT)


def swain_schaad_fc(delta_0_ang):
    """
    Swain-Schaad exponent from pure FC overlap ratios (nonadiabatic limit).
    ρ_std = ln(k_H/k_T) / ln(k_D/k_T)
    From FC: k_XY ∝ exp(-(α_Y - α_X) × δ₀²)
    → ρ_std_na = (α_T - α_H)/(α_T - α_D) = (√3-1)/(√3-√2) ≈ 2.30 [const in δ₀]
    """
    log_kie_HT_na = (ALPHA_T - ALPHA_H) * delta_0_ang**2
    log_kie_DT_na = (ALPHA_T - ALPHA_D) * delta_0_ang**2
    if log_kie_DT_na > 0:
        return log_kie_HT_na / log_kie_DT_na
    return float("nan")


def swain_schaad_ad(omega_H_cm1):
    """Swain-Schaad exponent from Bigeleisen ZPE (adiabatic limit)."""
    KIE_HT = kie_ad_HT(omega_H_cm1)
    KIE_DT = kie_ad_DT(omega_H_cm1)
    KIE_HD = kie_adiabatic_bigeleisen(omega_H_cm1)
    # ρ_std = ln(k_H/k_T) / ln(k_D/k_T)
    # k_D/k_T = k_H/k_T / (k_H/k_D)
    kie_dt = KIE_HT / KIE_HD
    if kie_dt > 1:
        return np.log(KIE_HT) / np.log(kie_dt)
    return float("nan")


def swain_schaad_unified(V_el_kcal, omega_H_cm1, delta_0_ang):
    """
    Unified SS exponent: geometric interpolation between FC-nonadiabatic and adiabatic.
    Uses f_avg = (f_H + f_D + f_T) / 3 as the symmetric LZ weight.
    ρ_uni ≈ ρ_na + f_avg × (ρ_ad − ρ_na)
    """
    kap_H = kappa_p(V_el_kcal, omega_H_cm1)
    kap_D = kap_H * SQRT2   # ω_D = ω_H/√2 → κ_D = √2 κ_H
    kap_T = kap_H * SQRT3   # ω_T = ω_H/√3 → κ_T = √3 κ_H

    f_H = lz_weight(kap_H)
    f_D = lz_weight(kap_D)
    f_T = lz_weight(kap_T)
    f_avg = (f_H + f_D + f_T) / 3.0

    rho_na = swain_schaad_fc(delta_0_ang)
    rho_ad = swain_schaad_ad(omega_H_cm1)

    if np.isfinite(rho_na) and np.isfinite(rho_ad):
        return (1 - f_avg) * rho_na + f_avg * rho_ad, kap_H, f_H, f_D, f_T
    return float("nan"), kap_H, f_H, f_D, f_T


def commitment_factor_northrop(kie_exp, kie_intrinsic):
    """
    Northrop equation (forward commitment only, Cr ≈ 0):
        KIE_obs = (KIE_int + Cf) / (1 + Cf)
        Cf = (KIE_int − KIE_obs) / (KIE_obs − 1)

    If KIE_obs ≥ KIE_int or KIE_obs ≤ 1: returns NaN (model doesn't apply).
    """
    if kie_exp <= 1.0 or kie_exp >= kie_intrinsic:
        return float("nan")
    return (kie_intrinsic - kie_exp) / (kie_exp - 1.0)


def main():
    print("=" * 90)
    print("UNIFIED MECHANISTIC MAP — 28-System PCET Benchmark")
    print("=" * 90)

    # ── Part 1: Swain-Schaad analysis ─────────────────────────────────────
    RHO_SC = (SQRT3 - 1) / (SQRT3 - SQRT2)  # = 2.298, pure FC nonadiabatic limit
    RHO_SC_ZPE = 3.34                         # Bigeleisen semiclassical

    print(f"\nSwain-Schaad Reference Values:")
    print(f"  Pure FC nonadiabatic limit: ρ = {RHO_SC:.3f}")
    print(f"  Bigeleisen semiclassical:   ρ = {RHO_SC_ZPE:.3f}")
    print(f"  Note: experimental 'tunneling-inflated' ρ > 3.34 requires Arrhenius analysis\n")

    print(f"{'System':<16} {'κ_H':>7} {'f_H':>6} {'ρ_na':>7} {'ρ_uni':>7} "
          f"{'KIE_ad':>8} {'Cf_impl':>9} {'KIE_exp':>8}")
    print("-" * 85)

    results = []
    for (name, V_el, omH, d0, KIE_exp, cat, cls) in SYSTEMS:
        rho_uni, kap_H, f_H, f_D, f_T = swain_schaad_unified(V_el, omH, d0)
        rho_na = swain_schaad_fc(d0)
        rho_ad = swain_schaad_ad(omH)
        KIE_ad = kie_adiabatic_bigeleisen(omH)
        Cf = commitment_factor_northrop(KIE_exp, KIE_ad)

        results.append(dict(
            name=name, kap=kap_H, f_H=f_H, f_D=f_D, f_T=f_T,
            rho_na=rho_na, rho_uni=rho_uni, rho_ad=rho_ad,
            KIE_ad=KIE_ad, Cf=Cf, KIE_exp=KIE_exp, cat=cat, cls=cls, d0=d0
        ))

        cf_str  = f"{Cf:9.2f}" if np.isfinite(Cf) else "      —  "
        print(f"{name:<16} {kap_H:>7.4f} {f_H:>6.4f} {rho_na:>7.3f} {rho_uni:>7.3f} "
              f"{KIE_ad:>8.2f} {cf_str} {KIE_exp:>8.1f}")

    kappas  = np.array([r["kap"] for r in results])
    rho_nas = np.array([r["rho_na"] for r in results])
    rho_uns = np.array([r["rho_uni"] for r in results])
    Cfs     = np.array([r["Cf"] for r in results])
    KIE_ads = np.array([r["KIE_ad"] for r in results])
    KIE_exps = np.array([r["KIE_exp"] for r in results])
    f_Hs    = np.array([r["f_H"] for r in results])

    print(f"\n{'─'*60}")
    print(f"Swain-Schaad summary:")
    print(f"  ρ_na range:  {rho_nas.min():.3f} – {rho_nas.max():.3f}  (pure FC, below semiclassical ✓)")
    print(f"  ρ_uni range: {rho_uns.min():.3f} – {rho_uns.max():.3f}  (unified, approaching 3.34)")
    print(f"  Systems with ρ_uni > 3.0: {(rho_uns > 3.0).sum()}/28")
    print(f"  Systems with ρ_uni > 3.34 (supra-SC): {(rho_uns > 3.34).sum()}/28")

    finite_Cf = Cfs[np.isfinite(Cfs)]
    print(f"\nCommitment factor summary (Northrop equation, KIE_int = KIE_ad):")
    print(f"  Systems with finite Cf (KIE_exp < KIE_ad): {np.isfinite(Cfs).sum()}/28")
    if len(finite_Cf) > 0:
        print(f"  Cf range:  {finite_Cf.min():.2f} – {finite_Cf.max():.2f}")
        print(f"  Cf > 1 (commitment-dominated): {(finite_Cf > 1.0).sum()} systems")
        print(f"  Cf > 5 (strongly committed):   {(finite_Cf > 5.0).sum()} systems")

    regime1 = [r for r in results if r["kap"] < 0.07]
    regime2 = [r for r in results if 0.07 <= r["kap"] < 0.15]
    regime3 = [r for r in results if r["kap"] >= 0.15]
    print(f"\nThree-regime breakdown:")
    print(f"  Regime 1 (κ < 0.07,  nonadiabatic-tunneling): {len(regime1)} systems, "
          f"mean KIE_exp = {np.mean([r['KIE_exp'] for r in regime1]):.0f}")
    print(f"  Regime 2 (0.07 ≤ κ < 0.15, intermediate):    {len(regime2)} systems, "
          f"mean KIE_exp = {np.mean([r['KIE_exp'] for r in regime2]):.0f}")
    print(f"  Regime 3 (κ ≥ 0.15,  near-adiabatic):        {len(regime3)} systems, "
          f"mean KIE_exp = {np.mean([r['KIE_exp'] for r in regime3]):.1f}")

    # ── Figures ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    colors  = [CLASS_COLORS[r["cls"]] for r in results]
    markers = [CAT_MARKERS[r["cat"]] for r in results]

    # ── P1: ρ vs κ — Swain-Schaad ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    kap_th = np.logspace(-2, 0, 300)
    # For median δ₀=0.40 Å, ω_H=2900:
    rho_na_th  = np.full_like(kap_th, (SQRT3-1)/(SQRT3-SQRT2))
    rho_ad_th  = np.full_like(kap_th, swain_schaad_ad(2900))
    f_H_th = 1 - np.exp(-2*np.pi*kap_th)
    f_D_th = 1 - np.exp(-2*np.pi*kap_th*SQRT2)
    f_T_th = 1 - np.exp(-2*np.pi*kap_th*SQRT3)
    f_avg_th = (f_H_th + f_D_th + f_T_th) / 3
    rho_uni_th = (1 - f_avg_th) * rho_na_th + f_avg_th * rho_ad_th

    ax1.semilogx(kap_th, rho_uni_th, "k-", lw=2, alpha=0.7, label="ρ_uni theory")
    ax1.semilogx(kap_th, rho_na_th,  "b--", lw=1, alpha=0.5, label=f"ρ_na = {RHO_SC:.2f} (FC)")
    ax1.semilogx(kap_th, rho_ad_th,  "r--", lw=1, alpha=0.5, label="ρ_ad = Bigeleisen")

    for r, col, mk in zip(results, colors, markers):
        ax1.scatter(r["kap"], r["rho_uni"], c=col, marker=mk, s=70,
                    zorder=3, edgecolors="k", linewidths=0.4)

    ax1.axhline(3.34,  color="darkred", lw=1, ls=":", alpha=0.6, label="SC limit 3.34")
    ax1.axhline(RHO_SC, color="blue", lw=1, ls=":", alpha=0.4)
    ax1.axvline(0.07,  color="gray", lw=0.7, ls="--", alpha=0.4)
    ax1.axvline(0.15,  color="gray", lw=0.7, ls="--", alpha=0.4)
    ax1.fill_betweenx([2.2, 3.4], 0.07, 0.15, alpha=0.05, color="orange")
    ax1.set_xlabel(r"$\kappa_p = V_\mathrm{el}/\hbar\omega_H$")
    ax1.set_ylabel(r"$\rho_\mathrm{SS} = \ln(k_H/k_T)/\ln(k_D/k_T)$")
    ax1.set_title("Swain-Schaad exponent\nvs adiabaticity")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_ylim(2.1, 3.5)

    # ── P2: Cf vs κ ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for r, col, mk in zip(results, colors, markers):
        if np.isfinite(r["Cf"]):
            ax2.scatter(r["kap"], r["Cf"], c=col, marker=mk, s=80,
                        zorder=3, edgecolors="k", linewidths=0.4)
            if r["Cf"] > 1.0 or r["kap"] > 0.10:
                ax2.annotate(r["name"], (r["kap"], r["Cf"]),
                             fontsize=6, ha="left",
                             xytext=(3, 2), textcoords="offset points")
        else:
            # Mark systems with no implied Cf (KIE_exp > KIE_ad) at bottom
            ax2.scatter(r["kap"], 0.05, c=col, marker=mk, s=60,
                        zorder=3, edgecolors="k", linewidths=0.4, alpha=0.4)

    ax2.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.5, label="Cf = 1")
    ax2.axvline(0.07, color="gray", lw=0.7, ls="--", alpha=0.4)
    ax2.axvline(0.15, color="gray", lw=0.7, ls="--", alpha=0.4)
    ax2.fill_betweenx([0, 20], 0.07, 0.15, alpha=0.05, color="orange")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\kappa_p$")
    ax2.set_ylabel(r"$C_f$ (implied commitment factor)")
    ax2.set_title("Commitment factor\n(Northrop equation, KIE_int = KIE_ad)")
    ax2.legend(fontsize=8)
    ax2.text(0.03, 0.05, "KIE_exp > KIE_ad\n(faded: no Cf implied)", fontsize=6,
             ha="left", va="bottom", color="gray",
             transform=ax2.get_xaxis_transform())

    # ── P3: 2D mechanistic map (κ vs Cf) ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for r, col, mk in zip(results, colors, markers):
        cf_plot = r["Cf"] if np.isfinite(r["Cf"]) else 0.05
        ax3.scatter(r["kap"], cf_plot, c=col, marker=mk, s=80,
                    zorder=3, edgecolors="k", linewidths=0.4,
                    alpha=1.0 if np.isfinite(r["Cf"]) else 0.35)
        if (np.isfinite(r["Cf"]) and r["Cf"] > 0.5) or r["kap"] > 0.12:
            ax3.annotate(r["name"], (r["kap"], cf_plot), fontsize=5.5,
                         xytext=(3, 2), textcoords="offset points")

    # Regime boundaries
    ax3.axvline(0.07, color="gray", lw=1, ls="--", alpha=0.5)
    ax3.axvline(0.15, color="gray", lw=1, ls="--", alpha=0.5)
    ax3.axhline(1.0,  color="gray", lw=1, ls=":",  alpha=0.5)
    ax3.fill_between([0.02, 0.07],  [0.02, 0.02], [20, 20], alpha=0.06, color="blue",
                     label="Regime 1\n(tunneling)")
    ax3.fill_between([0.07, 0.15],  [0.02, 0.02], [20, 20], alpha=0.06, color="orange",
                     label="Regime 2\n(intermediate)")
    ax3.fill_between([0.15, 0.40],  [0.02, 0.02], [20, 20], alpha=0.06, color="red",
                     label="Regime 3\n(near-adiabatic)")

    ax3.set_xscale("log")
    ax3.set_xlabel(r"$\kappa_p$ (adiabaticity)")
    ax3.set_ylabel(r"$C_f$ (commitment factor)")
    ax3.set_title("2D mechanistic map\n(κ_p vs implied Cf)")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.set_ylim(-0.5, 15)

    # ── P4: KIE_exp vs κ, colored by Cf ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    sc = ax4.scatter(kappas, KIE_exps,
                     c=np.where(np.isfinite(Cfs), Cfs, 0),
                     cmap="RdYlGn_r", vmin=0, vmax=10,
                     s=80, zorder=3, edgecolors="k", linewidths=0.4)
    plt.colorbar(sc, ax=ax4, label=r"$C_f$ (implied)", shrink=0.8)
    ax4.axhline(7.7,  color="steelblue", lw=1, ls="-.", alpha=0.6, label="ZPE ceil 7.7")
    ax4.axhline(np.mean(KIE_ads), color="darkred", lw=1, ls="--",
                alpha=0.6, label=f"KIE_ad ≈ {np.mean(KIE_ads):.1f}")
    ax4.axvline(0.07, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax4.axvline(0.15, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel(r"$\kappa_p$")
    ax4.set_ylabel(r"$KIE_\mathrm{exp}$")
    ax4.set_title(r"KIE vs $\kappa_p$, shaded by $C_f$")
    ax4.legend(fontsize=7)

    # ── P5: ρ_uni vs KIE_exp (testable prediction) ────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    for r, col, mk in zip(results, colors, markers):
        ax5.scatter(r["KIE_exp"], r["rho_uni"], c=col, marker=mk, s=70,
                    zorder=3, edgecolors="k", linewidths=0.4)
        if r["KIE_exp"] < 6 or r["rho_uni"] > 3.2:
            ax5.annotate(r["name"], (r["KIE_exp"], r["rho_uni"]),
                         fontsize=6, ha="left",
                         xytext=(3, 2), textcoords="offset points")

    ax5.axhline(3.34,  color="darkred", lw=1, ls=":", alpha=0.6, label="SC = 3.34")
    ax5.axhline(RHO_SC, color="blue", lw=1, ls=":", alpha=0.4, label=f"FC-NA = {RHO_SC:.2f}")
    ax5.set_xscale("log")
    ax5.set_xlabel(r"$KIE_\mathrm{exp}$")
    ax5.set_ylabel(r"$\rho_\mathrm{SS,uni}$ (unified prediction)")
    ax5.set_title("SS exponent vs experimental KIE\n(testable prediction)")
    ax5.legend(fontsize=7)

    # ── P6: f_H and f_D vs κ — LZ weight asymmetry ────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    kap_th2 = np.logspace(-2, 0, 300)
    f_H_th2 = 1 - np.exp(-2*np.pi*kap_th2)
    f_D_th2 = 1 - np.exp(-2*np.pi*kap_th2*SQRT2)
    f_T_th2 = 1 - np.exp(-2*np.pi*kap_th2*SQRT3)
    Δf_HDT  = f_T_th2 - f_H_th2   # asymmetry: T more adiabatic than H

    ax6.semilogx(kap_th2, f_H_th2, "b-",  lw=2, label=r"$f_H$")
    ax6.semilogx(kap_th2, f_D_th2, "g-",  lw=2, label=r"$f_D$")
    ax6.semilogx(kap_th2, f_T_th2, "r-",  lw=2, label=r"$f_T$")
    ax6.fill_between(kap_th2, f_H_th2, f_T_th2, alpha=0.1, color="purple",
                     label=r"$\Delta f = f_T - f_H$ (mass asymmetry)")

    # Scatter actual f_H values
    ax6.scatter(kappas, f_Hs, c=colors, s=50, zorder=3,
                edgecolors="k", linewidths=0.3, alpha=0.8)

    ax6.axvline(0.07, color="gray", lw=0.7, ls="--", alpha=0.4)
    ax6.axvline(0.15, color="gray", lw=0.7, ls="--", alpha=0.4)
    ax6.set_xlabel(r"$\kappa_H$")
    ax6.set_ylabel(r"LZ weight $f(\kappa)$")
    ax6.set_title("H, D, T Landau-Zener weights\n(mass asymmetry drives SS correction)")
    ax6.legend(fontsize=7, loc="upper left")
    ax6.set_ylim(0, 1.0)

    # Class legend
    class_patches = [mpatches.Patch(color=c, label=l)
                     for l, c in CLASS_COLORS.items()]
    fig.legend(handles=class_patches, fontsize=8, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.04))

    plt.suptitle(
        "Mechanistic Map: Adiabaticity, Swain-Schaad Exponent, and Commitment Factors\n"
        "across 28 Enzyme PCET Systems",
        fontsize=11, y=1.01
    )

    out = "pcet_engine/benchmarks/figures/mechanistic_map.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nFigures saved: {out}")

    return results


if __name__ == "__main__":
    main()
