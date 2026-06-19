"""
Temperature-dependent Swain-Schaad exponent under the unified LZ theory.

For the 9 systems with experimental T-dep KIE data, computes:

1. k_H(T), k_D(T), k_T(T) from pcet_engine WITH gating enabled
2. Adiabatic correction: log k_i_uni(T) = (1-f_i)*log k_na + f_i*log k_ad
3. Arrhenius fit → E_a^H, E_a^D, E_a^T
4. ΔEa = E_a^D - E_a^H  (T-dep criterion)
5. ρ_Arrhenius = (E_a^T - E_a^H) / (E_a^T - E_a^D)

Key prediction:
  - For pure SHS nonadiabatic (f=0): ρ_Arr ≈ 2.30 (inverse anomaly)
  - Unified theory (f>0): ρ_Arr → 3.34 as κ increases
  - For high-κ systems (DHFR, LADH): ρ_Arr_uni closer to experimental values

Experimental ρ_Arrhenius reference values (from Arrhenius pre-factors or ΔEa):
    SLO-1:  ρ ~ 7-10  (strongly inflated, tunneling-dominated)
    SLO-1 DM: ρ ~ 10+ (extreme tunneling)
    AADH:   ρ ~ 4-6
    PHM:    ρ ~ 4-5
    LADH:   ρ ~ 3-5
    DHFR:   ρ ~ 3-4 (near-semiclassical)
    TSase:  ρ ~ 4-5
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pcet_engine.core.rate_engine import PCETRateEngine
from pcet_engine.core.constants import (
    CM_TO_HARTREE, HARTREE_TO_KCALMOL, KCALMOL_TO_HARTREE,
    KB_HARTREE, PROTON_MASS_AMU, DEUTERIUM_MASS_AMU, TRITIUM_MASS_AMU,
)

# ── Constants ─────────────────────────────────────────────────────────────
KB_KCAL   = 1.9872036e-3    # kcal/(mol·K)
CM1_TO_EV = 1.2398e-4
KCAL_TO_EV = 0.04336
KB_EV      = 8.617e-5
SQRT2      = math.sqrt(2.0)
SQRT3      = math.sqrt(3.0)

# Temperature grid: 278 K to 323 K in 5 K steps
TEMPS = np.arange(278, 324, 5, dtype=float)


# ── Gating parameters (from kie_classification.py) ────────────────────────
GATING = {
    "SLO-1":   {"Omega": 150, "M_DA": 14.0},
    "SLO-1-DM":{"Omega":  96, "M_DA": 14.0},
    "AADH":    {"Omega": 315, "M_DA": 5.85},
    "MADH":    {"Omega": 350, "M_DA": 5.85},
    "PHM":     {"Omega": 400, "M_DA": 6.86},
    "RNR":     {"Omega": 200, "M_DA": 8.73},
    "LADH":    {"Omega": 360, "M_DA": 5.74},
    "DHFR":    {"Omega": 380, "M_DA": 5.74},
    "TSase":   {"Omega": 340, "M_DA": 5.74},
}

# Experimental T-dep KIE classification
TDEP_CLASS = {
    "SLO-1":    "T-independent",
    "SLO-1-DM": "T-independent",
    "AADH":     "T-dependent",
    "MADH":     "T-dependent",
    "PHM":      "T-dependent",
    "RNR":      "T-dependent",
    "LADH":     "T-dependent",
    "DHFR":     "T-independent",
    "TSase":    "T-dependent",
}

# Experimental ΔEa values (kcal/mol), E_a^D - E_a^H
DELTA_EA_EXP = {
    "SLO-1":    0.0,    # Knapp et al. 2002
    "SLO-1-DM": 0.0,    # Klinman group
    "AADH":     0.82,   # Scrutton group
    "MADH":     0.60,   # Scrutton group
    "PHM":      0.83,   # Klinman group
    "RNR":      1.10,   # Stubbe group
    "LADH":     0.53,   # Klinman group
    "DHFR":     0.0,    # Hammes-Schiffer / Benkovic
    "TSase":    0.20,   # Kohen group
}

# System parameters from benchmark (V_el, omega_H, delta_0, delta_G, lambda, d_DA, KIE_exp)
SYSTEM_PARAMS = {
    "SLO-1":    (0.60, 2900, 0.500, -5.4, 19.0, 2.690,  81.0),
    "SLO-1-DM": (0.30, 2900, 0.614, -5.4, 23.0, 3.100, 661.0),
    "AADH":     (0.80, 3000, 0.465, -8.0, 35.0, 3.050,  55.0),
    "MADH":     (0.50, 2950, 0.432, -6.5, 40.0, 3.100,  30.0),
    "PHM":      (1.20, 3100, 0.347, -3.0, 15.0, 2.550,  10.0),
    "RNR":      (0.20, 2600, 0.348, -2.0, 12.0, 2.800,   7.0),
    "LADH":     (2.00, 3000, 0.264, -4.0, 25.0, 3.200,   3.5),
    "DHFR":     (3.00, 3000, 0.247, -4.0, 15.0, 3.250,   3.0),
    "TSase":    (0.80, 2950, 0.318, -3.5, 22.0, 3.200,   6.0),
}

CLASS_COLORS = {
    "SLO-1 family": "#2166ac",
    "HAT":          "#d73027",
    "Hydride":      "#1a9850",
    "Proton/O-H":   "#f46d43",
    "Flavin":       "#8073ac",
}
SYS_CLASS = {
    "SLO-1":    "SLO-1 family",
    "SLO-1-DM": "SLO-1 family",
    "AADH":     "HAT",
    "MADH":     "HAT",
    "PHM":      "HAT",
    "RNR":      "HAT",
    "LADH":     "Hydride",
    "DHFR":     "Hydride",
    "TSase":    "Hydride",
}


def kappa_p_val(V_el_kcal, omega_H_cm1):
    return (V_el_kcal * KCAL_TO_EV) / (omega_H_cm1 * CM1_TO_EV)


def lz_weight(kappa):
    return 1.0 - math.exp(-2.0 * math.pi * kappa)


def arrhenius_params(temps, log_rates):
    """Fit ln(k) = ln(A) - E_a/RT → return (E_a_kcal, ln_A)."""
    inv_T = 1.0 / temps
    coeffs = np.polyfit(inv_T, log_rates, 1)  # slope, intercept
    E_a = -coeffs[0] * KB_KCAL   # kcal/mol
    return E_a, coeffs[1]


def compute_engine_rates(name, mass_amu, T_arr):
    """
    Compute pcet_engine nonadiabatic rates for isotope 'mass_amu' at all T.
    Uses published gating parameters.
    """
    V_el, omH, d0, dG, lam, d_DA, _ = SYSTEM_PARAMS[name]
    gate = GATING[name]

    freq_scale = math.sqrt(PROTON_MASS_AMU / mass_amu)
    omega_H_isotope = omH * freq_scale    # cm^-1

    rates = []
    for T in T_arr:
        engine = PCETRateEngine(temperature=T)
        result = engine.compute_rate(
            V_el=V_el,
            delta_G=dG,
            lambda_reorg=lam,
            omega_H=omega_H_isotope,
            d_DA=d_DA,
            delta_0=d0,
            Omega_gating=gate["Omega"],
            M_DA=gate["M_DA"],
        )
        # Use k_H for mass_amu ≈ proton, k_D for deuterium/tritium
        # We pass scaled omega_H already; the engine internally uses omega_H for H and
        # omega_D = omega_H/sqrt(2) for D. We want a single isotope rate at the scaled
        # frequency, so extract k_H (which uses our already-scaled omega_H_isotope).
        rates.append(result.k_H)
    return np.array(rates)


def adiabatic_rate_rel(V_el_kcal, omega_H_cm1, delta_G_kcal, lambda_kcal,
                       mass_amu, T):
    """
    Adiabatic TST rate (relative, same prefactor for all isotopes).
    k_ad = nu_H_isotope × exp(-G*_ad/kT)
    G*_ad = G*_marcus - V_el (resonance stabilisation)
    """
    dG_ev  = delta_G_kcal * KCAL_TO_EV
    lam_ev = lambda_kcal  * KCAL_TO_EV
    V_ev   = V_el_kcal    * KCAL_TO_EV
    kT_ev  = KB_EV * T

    G_marcus = (lam_ev + dG_ev)**2 / (4 * lam_ev)
    G_act_ad = max(G_marcus - V_ev, 0.0)

    freq_scale = math.sqrt(PROTON_MASS_AMU / mass_amu)
    omega_SI   = omega_H_cm1 * freq_scale * 2.0 * math.pi * 3e10   # rad/s
    nu_isotope = omega_SI / (2.0 * math.pi)                          # Hz

    return nu_isotope * math.exp(-G_act_ad / kT_ev)


def compute_unified_rates(name, mass_amu, T_arr):
    """
    Unified rates via LZ interpolation.
    log k_uni = (1-f) × log k_na + f × log k_ad
    """
    V_el, omH, d0, dG, lam, d_DA, _ = SYSTEM_PARAMS[name]
    freq_scale = math.sqrt(PROTON_MASS_AMU / mass_amu)
    omega_H_iso = omH * freq_scale

    kap = kappa_p_val(V_el, omega_H_iso)   # isotope-specific kappa
    f   = lz_weight(kap)

    k_na_arr  = compute_engine_rates(name, mass_amu, T_arr)
    k_ad_arr  = np.array([adiabatic_rate_rel(V_el, omega_H_iso, dG, lam, mass_amu, T)
                          for T in T_arr])

    log_k_na = np.log(k_na_arr)
    log_k_ad = np.log(k_ad_arr)

    # Normalise k_ad to k_na scale at T=298K to avoid absolute unit mismatch.
    # The LZ interpolation should be in dimensionless units; we normalise
    # by computing: log k_na(T) + f × log(k_ad(T)/k_na(T)) × scale
    # where scale brings k_ad to same order as k_na.
    # Equivalently: use the RATIO r(T) = k_ad(T)/k_ad(T_ref) × k_na(T_ref)/k_na(T_ref)
    # to get the adiabatic correction relative to nonadiabatic.
    # Since we want the T-dependence: log k_uni(T) - log k_uni(T_ref)
    # = (1-f) × [log k_na(T) - log k_na(T_ref)] + f × [log k_ad(T) - log k_ad(T_ref)]
    # This is RELATIVE to the T_ref value — independent of absolute normalization.
    # Use this to compute ΔEa cleanly.

    ref_idx   = np.argmin(np.abs(T_arr - 298.0))
    Δlog_k_na = log_k_na - log_k_na[ref_idx]
    Δlog_k_ad = log_k_ad - log_k_ad[ref_idx]
    Δlog_k_un = (1 - f) * Δlog_k_na + f * Δlog_k_ad

    # For absolute unified rate: need to set the reference correctly.
    # The actual k_uni ∝ exp(Δlog_k_un) normalized to k_na at T_ref.
    # For E_a extraction, only the T-dependence matters, not absolute value.
    log_k_uni = log_k_na[ref_idx] + Δlog_k_un  # anchored at k_na(T_ref)

    return log_k_na, log_k_ad, log_k_uni, kap, f


def main():
    print("=" * 90)
    print("TEMPERATURE-DEPENDENT SWAIN-SCHAAD: UNIFIED THEORY vs NONADIABATIC")
    print("=" * 90)
    print(f"\nTemperature range: {TEMPS[0]:.0f}–{TEMPS[-1]:.0f} K ({len(TEMPS)} points)")
    print(f"Semiclassical ρ_Arrhenius = 3.34")
    print(f"Pure FC nonadiabatic ρ_Arrhenius = (√3−1)/(√3−√2) = 2.303")

    print(f"\n{'System':<12} {'κ_H':>7} {'f_H':>6} "
          f"{'ΔEa_na':>8} {'ΔEa_uni':>9} {'ΔEa_exp':>9} "
          f"{'ρ_na':>7} {'ρ_uni':>7} "
          f"{'Class_exp':>14}")
    print("-" * 95)

    results = []
    for name in list(SYSTEM_PARAMS.keys()):
        V_el, omH, d0, dG, lam, d_DA, KIE_exp = SYSTEM_PARAMS[name]
        gate = GATING[name]

        # Compute rates for H, D, T (nonadiabatic and unified)
        lk_H_na, lk_H_ad, lk_H_uni, kap_H, f_H = compute_unified_rates(
            name, PROTON_MASS_AMU,   TEMPS)
        lk_D_na, lk_D_ad, lk_D_uni, kap_D, f_D = compute_unified_rates(
            name, DEUTERIUM_MASS_AMU, TEMPS)
        lk_T_na, lk_T_ad, lk_T_uni, kap_T, f_T = compute_unified_rates(
            name, TRITIUM_MASS_AMU,  TEMPS)

        # Arrhenius fits
        Ea_H_na,  lnA_H_na  = arrhenius_params(TEMPS, lk_H_na)
        Ea_D_na,  lnA_D_na  = arrhenius_params(TEMPS, lk_D_na)
        Ea_T_na,  lnA_T_na  = arrhenius_params(TEMPS, lk_T_na)

        Ea_H_uni, lnA_H_uni = arrhenius_params(TEMPS, lk_H_uni)
        Ea_D_uni, lnA_D_uni = arrhenius_params(TEMPS, lk_D_uni)
        Ea_T_uni, lnA_T_uni = arrhenius_params(TEMPS, lk_T_uni)

        # ΔEa = E_a^D - E_a^H
        delta_Ea_na  = Ea_D_na  - Ea_H_na
        delta_Ea_uni = Ea_D_uni - Ea_H_uni
        delta_Ea_exp = DELTA_EA_EXP.get(name, float("nan"))

        # ρ_Arrhenius = (E_a^T - E_a^H) / (E_a^T - E_a^D)
        # Use ln(A_H/A_T) / ln(A_D/A_T) formulation (Kohen-Klinman):
        # Since E_a and ln(A) are coupled in the Arrhenius fit:
        # Using ΔEa formulation:
        denom_na  = Ea_T_na  - Ea_D_na
        denom_uni = Ea_T_uni - Ea_D_uni

        rho_na  = (Ea_T_na  - Ea_H_na)  / denom_na  if abs(denom_na)  > 1e-6 else float("nan")
        rho_uni = (Ea_T_uni - Ea_H_uni) / denom_uni if abs(denom_uni) > 1e-6 else float("nan")

        t_class = TDEP_CLASS[name]
        results.append(dict(
            name=name, kap_H=kap_H, f_H=f_H,
            delta_Ea_na=delta_Ea_na, delta_Ea_uni=delta_Ea_uni, delta_Ea_exp=delta_Ea_exp,
            rho_na=rho_na, rho_uni=rho_uni,
            KIE_exp=KIE_exp, t_class=t_class, cls=SYS_CLASS[name],
            lk_H_na=lk_H_na, lk_H_uni=lk_H_uni,
            lk_D_na=lk_D_na, lk_D_uni=lk_D_uni,
        ))

        exp_str = f"{delta_Ea_exp:9.2f}" if math.isfinite(delta_Ea_exp) else "      —  "
        rho_na_str  = f"{rho_na:.3f}"  if math.isfinite(rho_na)  else "   —   "
        rho_uni_str = f"{rho_uni:.3f}" if math.isfinite(rho_uni) else "   —   "

        print(f"{name:<12} {kap_H:>7.4f} {f_H:>6.4f} "
              f"{delta_Ea_na:>8.3f} {delta_Ea_uni:>9.3f} {exp_str} "
              f"{rho_na_str:>7} {rho_uni_str:>7} "
              f"{t_class:>14}")

    # ── Summary stats ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    dEa_na_arr  = np.array([r["delta_Ea_na"]  for r in results])
    dEa_uni_arr = np.array([r["delta_Ea_uni"] for r in results])
    dEa_exp_arr = np.array([r["delta_Ea_exp"] for r in results])

    finite = np.isfinite(dEa_exp_arr)
    print(f"ΔEa (D-H) comparison:")
    print(f"  Mean |error| nonadiabatic:   {np.mean(np.abs(dEa_na_arr[finite]  - dEa_exp_arr[finite])):.3f} kcal/mol")
    print(f"  Mean |error| unified:        {np.mean(np.abs(dEa_uni_arr[finite] - dEa_exp_arr[finite])):.3f} kcal/mol")

    # T-dep KIE classification using threshold = 0.2 kcal/mol
    print(f"\nT-dep KIE classification (threshold ΔEa > 0.20 kcal/mol):")
    threshold = 0.20
    correct_na = correct_uni = 0
    for r in results:
        t_class = r["t_class"]
        pred_na  = "T-dependent" if r["delta_Ea_na"]  > threshold else "T-independent"
        pred_uni = "T-dependent" if r["delta_Ea_uni"] > threshold else "T-independent"
        match_na  = (pred_na  == t_class)
        match_uni = (pred_uni == t_class)
        correct_na  += int(match_na)
        correct_uni += int(match_uni)
        print(f"  {r['name']:<12} exp={t_class:<14}  na={pred_na:<14} {'✓' if match_na else '✗'}  "
              f"uni={pred_uni:<14} {'✓' if match_uni else '✗'}")

    n = len(results)
    print(f"\n  Nonadiabatic: {correct_na}/{n} correct")
    print(f"  Unified:      {correct_uni}/{n} correct")

    # ── Figures ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    colors = [CLASS_COLORS[r["cls"]] for r in results]
    kappas = np.array([r["kap_H"] for r in results])
    dEa_na = np.array([r["delta_Ea_na"]  for r in results])
    dEa_un = np.array([r["delta_Ea_uni"] for r in results])
    dEa_ex = np.array([r["delta_Ea_exp"] for r in results])
    rho_na = np.array([r["rho_na"]  for r in results])
    rho_un = np.array([r["rho_uni"] for r in results])

    # P1: ΔEa vs κ — nonadiabatic and unified
    ax = axes[0, 0]
    ax.scatter(kappas, dEa_na, c=colors, marker="o", s=80, zorder=3,
               edgecolors="k", linewidths=0.4, alpha=0.7, label=r"$\Delta E_a$ (NA)")
    ax.scatter(kappas, dEa_un, c=colors, marker="^", s=80, zorder=3,
               edgecolors="k", linewidths=0.4, label=r"$\Delta E_a$ (unified)")
    for r, col in zip(results, colors):
        ax.annotate(r["name"], (r["kap_H"], r["delta_Ea_uni"]),
                    fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0.20, color="gray", lw=1, ls="--", alpha=0.6, label="threshold 0.20 kcal/mol")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_p$")
    ax.set_ylabel(r"$\Delta E_a = E_a^D - E_a^H$ (kcal/mol)")
    ax.set_title(r"Isotope $\Delta E_a$ vs adiabaticity")
    ax.legend(fontsize=7)

    # P2: Predicted ΔEa vs experimental ΔEa
    ax2 = axes[0, 1]
    fin = np.isfinite(dEa_ex)
    ax2.scatter(dEa_ex[fin], dEa_na[fin],  c=np.array(colors)[fin], marker="o",
                s=80, alpha=0.7, edgecolors="k", linewidths=0.4, label="NA prediction")
    ax2.scatter(dEa_ex[fin], dEa_un[fin], c=np.array(colors)[fin], marker="^",
                s=80, edgecolors="k", linewidths=0.4, label="Unified prediction")
    for i, r in enumerate(results):
        if fin[i]:
            ax2.annotate(r["name"], (r["delta_Ea_exp"], r["delta_Ea_uni"]),
                         fontsize=6, xytext=(3, 2), textcoords="offset points")
    diag = np.linspace(0, max(dEa_ex[fin])*1.1, 50)
    ax2.plot(diag, diag, "k--", lw=1, alpha=0.5, label="y = x")
    ax2.set_xlabel(r"$\Delta E_a$ experimental (kcal/mol)")
    ax2.set_ylabel(r"$\Delta E_a$ predicted (kcal/mol)")
    ax2.set_title("Predicted vs experimental ΔEa")
    ax2.legend(fontsize=7)

    # P3: ρ_Arrhenius vs κ
    ax3 = axes[1, 0]
    fin_rho = np.isfinite(rho_na) & np.isfinite(rho_un)
    ax3.scatter(kappas[fin_rho], rho_na[fin_rho],  c=np.array(colors)[fin_rho],
                marker="o", s=80, alpha=0.7, edgecolors="k", linewidths=0.4,
                label=r"$\rho$ (NA)")
    ax3.scatter(kappas[fin_rho], rho_un[fin_rho], c=np.array(colors)[fin_rho],
                marker="^", s=80, edgecolors="k", linewidths=0.4,
                label=r"$\rho$ (unified)")
    for i, r in enumerate(results):
        if fin_rho[i]:
            ax3.annotate(r["name"], (r["kap_H"], r["rho_uni"]),
                         fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax3.axhline(3.34,  color="darkred", lw=1, ls=":", alpha=0.6, label="SC limit 3.34")
    ax3.axhline(2.303, color="blue",    lw=1, ls=":", alpha=0.4, label="FC-NA limit 2.30")
    ax3.set_xscale("log")
    ax3.set_xlabel(r"$\kappa_p$")
    ax3.set_ylabel(r"$\rho_\mathrm{Arrhenius}$")
    ax3.set_title("Arrhenius Swain-Schaad exponent\nvs adiabaticity")
    ax3.legend(fontsize=7)

    # P4: Arrhenius plots for SLO-1 vs DHFR (comparison case)
    ax4 = axes[1, 1]
    inv_T = 1000.0 / TEMPS   # 10^3/T for x-axis
    for sname, ls, lw in [("SLO-1", "-", 2), ("DHFR", "--", 2)]:
        r = next(x for x in results if x["name"] == sname)
        col = CLASS_COLORS[r["cls"]]
        ax4.plot(inv_T, r["lk_H_na"],  color=col, ls=ls, lw=lw, alpha=0.5)
        ax4.plot(inv_T, r["lk_H_uni"], color=col, ls=ls, lw=lw*1.3,
                 label=f"{sname} H (uni)")
        ax4.plot(inv_T, r["lk_D_na"],  color=col, ls=ls, lw=lw, alpha=0.3)
        ax4.plot(inv_T, r["lk_D_uni"], color=col, ls=ls, lw=lw*1.3, alpha=0.7,
                 label=f"{sname} D (uni)")

    ax4.set_xlabel(r"$10^3/T$ (K$^{-1}$)")
    ax4.set_ylabel(r"$\ln k$ (relative)")
    ax4.set_title("Arrhenius plots: SLO-1 vs DHFR\n(solid=H, dashed lighter=D)")
    ax4.legend(fontsize=7, ncol=2)

    class_patches = [mpatches.Patch(color=c, label=l)
                     for l, c in CLASS_COLORS.items()]
    fig.legend(handles=class_patches, fontsize=8, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.04))

    plt.suptitle(
        "Temperature-Dependent Swain-Schaad: Unified LZ Theory vs Nonadiabatic\n"
        "9 systems with experimental T-dep KIE data",
        fontsize=11, y=1.01
    )

    out = "pcet_engine/benchmarks/figures/tdep_swain_schaad.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nFigures saved: {out}")

    return results


if __name__ == "__main__":
    main()
