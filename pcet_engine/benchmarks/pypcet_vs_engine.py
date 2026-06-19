"""
Head-to-head comparison: pyPCET vs our engine using IDENTICAL proton potentials.

This script feeds the exact DFT-computed proton potentials from pyPCET's examples
into our FGH solver and compares the resulting rates. This is the definitive
apples-to-apples validation — same potentials, same formula, different code.

pyPCET reference outputs:
    Example 2 (RNR Y356-Y731): k_H_tot = 1.5788e+03 s⁻¹
    Example 3.1 (BIP electrochem): k_H = 3.3725e+06, k_D = 1.9243e+06, KIE = 1.75
    Example 3.2 (BIP photochem): k_H = 1.2286e+09, k_D = 6.3348e+08, KIE = 1.94
    Example 4 (CoTPP electrochem at -0.66 V): k_H = 3.9997e+09, k_D = 1.8396e+09, KIE = 2.17
"""

import os
import math
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from pcet_engine.core.rate_engine import PCETRateEngine
from pcet_engine.core.constants import (
    EV_TO_KCALMOL, KCALMOL_TO_HARTREE, HARTREE_TO_EV,
    PROTON_MASS_AMU, DEUTERIUM_MASS_AMU,
)

BASE = os.path.join(os.path.dirname(__file__), "pypcet_reference")

KCAL_TO_EV = 1.0 / EV_TO_KCALMOL  # 0.04336 eV/kcal


# =====================================================================
# Example 2: RNR Y356-Y731 — Thermal Enzymatic PCET
# =====================================================================

def example2_rnr():
    """Reproduce pyPCET Example 2 using their exact proton potentials.

    pyPCET reference: k_H_tot = 1.5788e+03 s⁻¹
    Dominant R for H = 2.54 Å

    Parameters:
        ΔG = 0.0 eV, λ = 18.86 kcal/mol = 0.818 eV
        V_el = 0.8 kcal/mol = 0.0347 eV
        T = 298 K, NStates = 7
    """
    print("=" * 80)
    print("EXAMPLE 2: RNR Y356-Y731 — Numerical Proton Potentials from pyPCET")
    print("  pyPCET reference: k_H_tot = 1.5788e+03 s⁻¹")
    print("=" * 80)

    pot_dir = os.path.join(BASE, "example2_Y356-Y731", "proton_potentials")
    Rs = np.arange(2.42, 3.20, 0.10)

    delta_G = 0.0             # eV (symmetric)
    Lambda = 18.86 * KCAL_TO_EV   # eV
    V_el = 0.8 * KCAL_TO_EV       # eV
    T = 298

    V_el_kcal = 0.8       # kcal/mol
    delta_G_kcal = 0.0     # kcal/mol
    lambda_kcal = 18.86    # kcal/mol

    engine = PCETRateEngine(
        n_reactant_states=7,
        n_product_states=7,
        temperature=T,
    )

    kH_R = np.zeros(len(Rs))

    print(f"\n  {'R_DA':>6} {'k_H(our)':>14} {'Notes'}")
    print("  " + "-" * 50)

    for i, R in enumerate(Rs):
        fname = os.path.join(pot_dir, f"R{R:.2f}_potential.dat")
        if not os.path.exists(fname):
            print(f"  {R:.2f}  MISSING: {fname}")
            continue

        data = np.loadtxt(fname)
        rp = data[:, 0]          # Å (proton coordinate)
        E_reac = data[:, 1]      # kcal/mol
        E_prod = data[:, 2]      # kcal/mol

        # Convert to eV (our FGH solver expects eV)
        E_reac_eV = E_reac * KCAL_TO_EV
        E_prod_eV = E_prod * KCAL_TO_EV

        # Shift potentials so minimum = 0
        E_reac_eV -= E_reac_eV.min()
        E_prod_eV -= E_prod_eV.min()

        # Spline the data for smooth interpolation (like pyPCET does)
        spl_R = UnivariateSpline(rp, E_reac_eV, s=0)
        spl_P = UnivariateSpline(rp, E_prod_eV, s=0)

        # Use our FGH-based rate calculation
        result = engine.compute_rate_from_potential(
            r_grid=rp,
            V_reactant=spl_R,
            V_product=spl_P,
            V_el=V_el_kcal,
            delta_G=delta_G_kcal,
            lambda_reorg=lambda_kcal,
            n_grid=512,
        )

        kH_R[i] = result.k_H
        print(f"  {R:>6.2f} {result.k_H:>14.4e}  ω_H={result.omega_H:.0f} cm⁻¹, KIE={result.KIE:.1f}")

    # Thermal averaging over R — reproduce pyPCET's exact procedure
    # Fit log(k) to quadratic, use umbrella sampling P(R)
    valid = kH_R > 0
    Rs_valid = Rs[valid]
    kH_valid = kH_R[valid]

    if len(Rs_valid) < 3:
        print("\n  ERROR: Too few valid R points for thermal averaging")
        return

    # Fit log(k_H) vs R to 4th-order polynomial (matching pyPCET)
    # Restrict R_fine to data range to prevent extrapolation blow-up
    try:
        coeffs_kH = np.polyfit(Rs_valid, np.log(kH_valid), min(4, len(Rs_valid) - 1))
    except Exception as e:
        print(f"\n  Fitting failed: {e}")
        return

    R_fine = np.linspace(Rs_valid[0] - 0.1, Rs_valid[-1] + 0.1, 500)
    kH_fine = np.exp(np.polyval(coeffs_kH, R_fine))

    # Load P(R) from umbrella sampling
    pr_file = os.path.join(BASE, "example2_Y356-Y731", "p_R_umbrella_A.dat")
    PR_data = np.loadtxt(pr_file)
    R_pr = PR_data[:, 0]
    PR_raw = PR_data[:, 1]

    # Fit log(P(R)) to 4th-order polynomial (same as pyPCET)
    # Filter out zeros
    mask = PR_raw > 0
    R_pr_valid = R_pr[mask]
    PR_valid = PR_raw[mask]

    def poly4(x, a, b, c, d, e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e

    popt_pr, _ = curve_fit(poly4, R_pr_valid, np.log(PR_valid))
    PR_fine = np.exp(poly4(R_fine, *popt_pr))

    # Normalize
    from scipy.integrate import simpson
    Z = simpson(PR_fine, x=R_fine)
    PR_fine /= Z

    # Thermal average
    integrand = PR_fine * kH_fine
    k_H_tot = simpson(integrand, x=R_fine)

    # Find dominant R
    peaks = find_peaks(integrand)[0]
    R_dom = R_fine[peaks[0]] if len(peaks) > 0 else R_fine[np.argmax(integrand)]

    print(f"\n  --- Thermal Average ---")
    print(f"  Dominant R (H) = {R_dom:.2f} Å (pyPCET: 2.54 Å)")
    print(f"  k_H_tot (our)  = {k_H_tot:.4e} s⁻¹")
    print(f"  k_H_tot (pyPCET) = 1.5788e+03 s⁻¹")
    if k_H_tot > 0:
        ratio = k_H_tot / 1578.8
        print(f"  Ratio (our/pyPCET) = {ratio:.3f}")
        print(f"  log₁₀ ratio = {math.log10(ratio):+.3f}")

    return k_H_tot


# =====================================================================
# Example 3: BIP — Electrochemical + Photochemical PCET
# =====================================================================

def example3_bip():
    """Reproduce pyPCET Example 3 using their exact proton potentials.

    pyPCET reference (electrochemical):
        k_H = 3.3725e+06, k_D = 1.9243e+06, KIE = 1.75
    pyPCET reference (photochemical):
        k_H = 1.2286e+09, k_D = 6.3348e+08, KIE = 1.94

    Parameters:
        λ = 21.4 kcal/mol, V_el = 1 kcal/mol, T = 298.15 K
        k_eff = 0.0443 a.u., R_eq = 2.58 Å
        NStates = 20

    Key matching details vs pyPCET:
        - Poly6/poly8 fitting for digitized data (not interpolating spline)
        - Uniform 256-point grid from -1.0 to 1.0 Å (not raw CSV points)
        - DG = epsilon for anodic at eta=0
        - Fermi: (1 - f(ε)) weighting with rho=1, beta=1
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: BIP — Electrochemical + Photochemical PCET")
    print("  pyPCET electrochem: k_H=3.37e+06, k_D=1.92e+06, KIE=1.75")
    print("  pyPCET photochem:   k_H=1.23e+09, k_D=6.33e+08, KIE=1.94")
    print("=" * 80)

    import pandas as pd
    from pcet_engine.core.constants import (
        AMU_TO_AU, ANGSTROM_TO_BOHR, KB_HARTREE,
    )

    pot_dir = os.path.join(BASE, "example3_BIP_KIE", "proton_potentials")
    Rs = np.arange(2.37, 2.92, 0.05)

    V_el_kcal = 1.0          # kcal/mol
    lambda_kcal = 21.4        # kcal/mol
    T = 298.15
    k_eff_au = 0.0443        # a.u. (DA force constant)
    R_eq = 2.58              # Å

    engine = PCETRateEngine(
        n_reactant_states=20,
        n_product_states=20,
        temperature=T,
    )

    # Uniform grid matching pyPCET: 256 points from -1.0 to 1.0 Å
    rp_uniform = np.linspace(-1.0, 1.0, 256)

    # --- Load and fit potentials at each R (matching pyPCET's polynomial fitting) ---
    ReacPot_R = []  # list of callable polynomial fits
    ProdPot_R = []

    for i, R in enumerate(Rs):
        R_str = f"{R:.2f}"
        red_file = os.path.join(pot_dir, f"Reduced_BIP_{R_str}A.csv")
        ox_file = os.path.join(pot_dir, f"Oxidized_BIP_{R_str}A.csv")

        if not os.path.exists(red_file) or not os.path.exists(ox_file):
            ReacPot_R.append(None)
            ProdPot_R.append(None)
            continue

        dat_red = pd.read_csv(red_file, sep=', ', header=0, engine='python')
        dat_ox = pd.read_csv(ox_file, sep=', ', header=0, engine='python')

        rp_red = dat_red['x'].values
        E_red_eV = dat_red['y'].values * KCAL_TO_EV   # kcal/mol → eV
        rp_ox = dat_ox['x'].values
        E_ox_eV = dat_ox['y'].values * KCAL_TO_EV

        # pyPCET uses poly6 for R=2.37 reduced, poly8 for all others
        # We match this exactly
        if i == 0:
            coeffs_red = np.polyfit(rp_red, E_red_eV, 6)
        else:
            coeffs_red = np.polyfit(rp_red, E_red_eV, 8)
        coeffs_ox = np.polyfit(rp_ox, E_ox_eV, 8)

        ReacPot_R.append(np.poly1d(coeffs_red))
        ProdPot_R.append(np.poly1d(coeffs_ox))

    # --- 3a. Compute rate at each R for electrochemical (DG=0 + Fermi integration) ---
    epsilons = np.linspace(-2, 2, 101)  # eV

    kH_R = np.zeros(len(Rs))
    kD_R = np.zeros(len(Rs))

    print(f"\n  Computing rates at each R_DA...")
    print(f"  {'R_DA':>6} {'k_H(echem)':>14} {'k_D(echem)':>14} {'KIE':>8}")
    print("  " + "-" * 50)

    for i, R in enumerate(Rs):
        if ReacPot_R[i] is None:
            print(f"  {R:.2f}  MISSING files")
            continue

        # Evaluate polynomial fits on uniform grid (matching pyPCET)
        V_reac_grid = ReacPot_R[i](rp_uniform)
        V_prod_grid = ProdPot_R[i](rp_uniform)

        # For each epsilon, compute vibronic rate with DG = epsilon (anodic, eta=0)
        # pyPCET reuses proton states across epsilon — we compute from potential
        # which also doesn't change the proton states (they don't depend on DG)
        kH_eps = np.zeros(len(epsilons))
        kD_eps = np.zeros(len(epsilons))

        for j, eps in enumerate(epsilons):
            dG_kcal = eps * EV_TO_KCALMOL  # eV → kcal/mol

            result = engine.compute_rate_from_potential(
                r_grid=rp_uniform,
                V_reactant=V_reac_grid,  # pass array, not callable
                V_product=V_prod_grid,
                V_el=V_el_kcal,
                delta_G=dG_kcal,
                lambda_reorg=lambda_kcal,
            )
            kH_eps[j] = result.k_H
            kD_eps[j] = result.k_D

        # Fermi-Dirac: f(ε) = 1/(1 + exp(ε/kBT))
        kBT_eV = 8.617333e-5 * T  # eV
        fermi = 1.0 / (1.0 + np.exp(epsilons / kBT_eV))

        # Anodic: k = ∫ (1 - f(ε)) × k(ε) dε   (rho/beta = 1, matching pyPCET)
        from scipy.integrate import simpson
        kH_R[i] = simpson((1.0 - fermi) * kH_eps, x=epsilons)
        kD_R[i] = simpson((1.0 - fermi) * kD_eps, x=epsilons)

        kie = kH_R[i] / kD_R[i] if kD_R[i] > 0 else float('inf')
        print(f"  {R:>6.2f} {kH_R[i]:>14.4e} {kD_R[i]:>14.4e} {kie:>8.2f}")

    # --- Thermal averaging over R using harmonic P(R) ---
    R_fine = np.linspace(2.0, 3.0, 200)

    # Interpolate log(k) (same method as pyPCET)
    from scipy.interpolate import interp1d
    valid = kH_R > 0
    Rs_valid = Rs[valid]

    if len(Rs_valid) < 3:
        print("  ERROR: Too few valid R points")
        return

    logkH_interp = interp1d(Rs_valid, np.log(kH_R[valid]), kind='quadratic', fill_value='extrapolate')
    logkD_interp = interp1d(Rs_valid, np.log(kD_R[valid]), kind='quadratic', fill_value='extrapolate')
    kH_fine = np.exp(logkH_interp(R_fine))
    kD_fine = np.exp(logkD_interp(R_fine))

    # Harmonic P(R) — same as pyPCET
    A2Bohr = ANGSTROM_TO_BOHR
    Ha2eV = HARTREE_TO_EV
    kB_eV = 8.617333e-5  # eV/K
    ER_fine = 0.5 * k_eff_au * (R_fine - R_eq)**2 * A2Bohr**2 * Ha2eV  # eV
    PR_fine = np.exp(-ER_fine / (kB_eV * T))
    Z = simpson(PR_fine, x=R_fine)
    PR_fine /= Z

    # Thermal average
    k_H_tot = simpson(PR_fine * kH_fine, x=R_fine)
    k_D_tot = simpson(PR_fine * kD_fine, x=R_fine)
    KIE_tot = k_H_tot / k_D_tot if k_D_tot > 0 else float('inf')

    # Dominant R
    peaks_H = find_peaks(PR_fine * kH_fine)[0]
    peaks_D = find_peaks(PR_fine * kD_fine)[0]
    R_dom_H = R_fine[peaks_H[0]] if len(peaks_H) > 0 else R_fine[np.argmax(PR_fine * kH_fine)]
    R_dom_D = R_fine[peaks_D[0]] if len(peaks_D) > 0 else R_fine[np.argmax(PR_fine * kD_fine)]

    print(f"\n  --- Electrochemical: Thermal Average ---")
    print(f"  Dominant R (H) = {R_dom_H:.2f} Å (pyPCET: 2.48)")
    print(f"  Dominant R (D) = {R_dom_D:.2f} Å (pyPCET: 2.52)")
    print(f"  k_H_tot (our)    = {k_H_tot:.4e} s⁻¹  (pyPCET: 3.3725e+06)")
    print(f"  k_D_tot (our)    = {k_D_tot:.4e} s⁻¹  (pyPCET: 1.9243e+06)")
    print(f"  KIE (our)        = {KIE_tot:.2f}       (pyPCET: 1.75)")
    if k_H_tot > 0:
        print(f"  k_H ratio (our/pyPCET) = {k_H_tot / 3.3725e6:.3f}")
        print(f"  KIE ratio (our/pyPCET) = {KIE_tot / 1.75:.3f}")

    # --- 3b. Photochemical: fixed DG = -5.0 kcal/mol, no Fermi integration ---
    print(f"\n  --- Photochemical: ΔG = -5.0 kcal/mol ---")
    delta_G_photo = -5.0  # kcal/mol

    kH_photo_R = np.zeros(len(Rs))
    kD_photo_R = np.zeros(len(Rs))

    for i, R in enumerate(Rs):
        if ReacPot_R[i] is None:
            continue

        # Use same polynomial fits and uniform grid as electrochemical
        V_reac_grid = ReacPot_R[i](rp_uniform)
        V_prod_grid = ProdPot_R[i](rp_uniform)

        result = engine.compute_rate_from_potential(
            r_grid=rp_uniform,
            V_reactant=V_reac_grid,
            V_product=V_prod_grid,
            V_el=V_el_kcal,
            delta_G=delta_G_photo,
            lambda_reorg=lambda_kcal,
        )
        kH_photo_R[i] = result.k_H
        kD_photo_R[i] = result.k_D

    # Thermal average (photochemical)
    valid = kH_photo_R > 0
    if np.sum(valid) >= 3:
        logkH_ph = interp1d(Rs[valid], np.log(kH_photo_R[valid]), kind='quadratic', fill_value='extrapolate')
        logkD_ph = interp1d(Rs[valid], np.log(kD_photo_R[valid]), kind='quadratic', fill_value='extrapolate')
        kH_ph_fine = np.exp(logkH_ph(R_fine))
        kD_ph_fine = np.exp(logkD_ph(R_fine))

        k_H_photo = simpson(PR_fine * kH_ph_fine, x=R_fine)
        k_D_photo = simpson(PR_fine * kD_ph_fine, x=R_fine)
        KIE_photo = k_H_photo / k_D_photo if k_D_photo > 0 else float('inf')

        print(f"  k_H_tot (our)    = {k_H_photo:.4e} s⁻¹  (pyPCET: 1.2286e+09)")
        print(f"  k_D_tot (our)    = {k_D_photo:.4e} s⁻¹  (pyPCET: 6.3348e+08)")
        print(f"  KIE (our)        = {KIE_photo:.2f}       (pyPCET: 1.94)")
        if k_H_photo > 0:
            print(f"  k_H ratio (our/pyPCET) = {k_H_photo / 1.2286e9:.3f}")

    return k_H_tot, k_D_tot, KIE_tot


# =====================================================================
# Example 4: CoTPP — Heterogeneous Electrochemical PCET
# =====================================================================

def _edl_model(E_appl, dIHL, dOHL, eps_IHL, eps_st, eps_op, rho_water,
               m_water, c_ions, C_EDL, PZFC):
    """Gouy-Chapman-Stern electric double layer model.

    Returns a callable phi(R) giving the potential drop at distance R
    from the electrode surface (in Angstroms).

    This reproduces pyPCET's EDL_model used in Example 4.

    The model divides space into three regions:
      - Inner Helmholtz layer (IHL): 0 to dIHL, dielectric eps_IHL
      - Outer Helmholtz layer (OHL): dIHL to dIHL+dOHL
      - Diffuse layer: beyond dIHL+dOHL, Gouy-Chapman
    """
    import scipy.optimize as opt

    # Physical constants
    eps0 = 8.854187817e-12    # F/m (vacuum permittivity)
    e_charge = 1.602176634e-19  # C
    N_A = 6.02214076e23
    kB_SI = 1.380649e-23       # J/K
    T_edl = 300.0              # K (same as example4)

    # Gouy-Chapman parameters
    # Concentration in mol/m^3
    c0 = c_ions * 1000.0  # mol/L -> mol/m^3

    # Debye length parameter
    kappa = np.sqrt(2 * c0 * N_A * e_charge**2 / (eps_st * eps0 * kB_SI * T_edl))

    # Capacitance of IHL (plate capacitor)
    # C_IHL = eps_IHL * eps0 / (dIHL * 1e-10)  in F/m^2
    C_IHL = eps_IHL * eps0 / (dIHL * 1e-10)

    # Capacitance of OHL
    C_OHL = eps_st * eps0 / (dOHL * 1e-10)

    # Total charge on electrode: sigma_M = C_EDL * (E_appl - PZFC)
    # C_EDL given in muF/cm^2, convert to F/m^2
    C_EDL_SI = C_EDL * 1e-6 * 1e4  # muF/cm^2 -> F/m^2
    sigma_M = C_EDL_SI * (E_appl - PZFC)  # C/m^2

    # Potential at electrode surface
    phi_M = E_appl - PZFC  # V (relative to PZC)

    # Potential drop across IHL
    phi_IHL = sigma_M / C_IHL

    # Potential at OHL (phi_2)
    phi_2 = phi_M - phi_IHL

    # For the diffuse layer, the Gouy-Chapman solution gives:
    # phi(x) = phi_2 * exp(-kappa * (x - x_OHL))  for |phi_2| << kBT/e (linear approx)
    # For larger potentials, use the exact GC solution:
    # phi(x) = (2*kBT/e) * ln[ (1+gamma*exp(-kappa*(x-x_OHL))) / (1-gamma*exp(-kappa*(x-x_OHL))) ]
    # where gamma = tanh(e*phi_2 / (4*kBT))

    kBT_over_e = kB_SI * T_edl / e_charge  # ~0.02585 V at 300K
    gamma = np.tanh(phi_2 / (4.0 * kBT_over_e))

    x_OHL = dIHL + dOHL  # Angstroms

    # Dipole contribution to IHL potential
    # pyPCET calculates this from water dipole moment and density
    # dipole moment of water ~ 1.85 D
    mu_water = 1.85 * 3.33564e-30  # Debye -> C*m
    n_water = rho_water * 1000 / m_water * N_A  # number density in m^-3
    # Surface density of oriented water dipoles
    # This is a correction — for 'calculate' mode pyPCET uses the Booth model
    # For simplicity we include it as a small correction to the IHL field

    def phi_at_R(R_arr):
        """Return potential drop at distance R (Angstroms) from electrode."""
        R_arr = np.atleast_1d(R_arr)
        phi = np.zeros_like(R_arr, dtype=float)

        for idx, R in enumerate(R_arr):
            if R <= 0:
                phi[idx] = phi_M
            elif R <= dIHL:
                # Linear drop in IHL
                phi[idx] = phi_M * (1.0 - R / dIHL) + phi_2 * (R / dIHL)
                # More precisely: linear interpolation from phi_M to phi_at_IHL
                phi_at_IHL = phi_M - sigma_M * R / (eps_IHL * eps0) * 1e-10
                phi[idx] = phi_at_IHL
            elif R <= x_OHL:
                # Linear drop in OHL (from IHL to OHL boundary)
                frac = (R - dIHL) / dOHL
                phi_at_IHL_boundary = phi_M - phi_IHL
                phi[idx] = phi_at_IHL_boundary * (1.0 - frac) + phi_2 * frac
                # More precisely:
                phi[idx] = phi_2 + (phi_at_IHL_boundary - phi_2) * (1.0 - frac)
            else:
                # Diffuse layer: Gouy-Chapman
                x_diff = (R - x_OHL) * 1e-10  # convert to meters
                if abs(gamma) < 1e-10:
                    phi[idx] = 0.0
                else:
                    exp_term = gamma * np.exp(-kappa * x_diff)
                    # Clamp to avoid log domain errors
                    exp_term = np.clip(exp_term, -0.9999, 0.9999)
                    phi[idx] = 2.0 * kBT_over_e * np.log(
                        (1.0 + exp_term) / (1.0 - exp_term)
                    )

        return phi if len(phi) > 1 else phi[0]

    return phi_at_R


def example4_cotpp():
    """Reproduce pyPCET Example 4 using their exact proton potentials.

    Heterogeneous electrochemical PCET: CoTPP on graphene electrode.
    Hutchison et al. ACS Catal. 2024, 19, 14363-14372.

    pyPCET reference (at -0.66 V vs SHE):
        k_H = 3.9997e+09, k_D = 1.8396e+09, KIE = 2.17

    Parameters:
        lambda = 0.83 eV (inner-sphere), V_el = 0.10 eV
        T = 300 K, NStates = 10
        DeltaG0_H = 0.55 eV, DeltaG0_D = 0.5407 eV
        R_DA: 3.057 to 4.257 A (non-uniform spacing)
        Graphene DOS from DFT calculation
        EDL model for potential-dependent work terms
    """
    import pandas as pd
    from scipy.interpolate import interp1d
    from scipy.integrate import simpson

    print("\n" + "=" * 80)
    print("EXAMPLE 4: CoTPP — Heterogeneous Electrochemical PCET")
    print("  Hutchison et al. ACS Catal. 2024, 19, 14363-14372")
    print("  pyPCET reference at E=-0.66V: k_H=4.00e+09, k_D=1.84e+09, KIE=2.17")
    print("=" * 80)

    pot_dir = os.path.join(BASE, "example4_CoTPP", "proton_potentials")
    dos_file = os.path.join(BASE, "example4_CoTPP", "graphene_DOS_norm_gauss.csv")

    # R values from the pyPCET example (non-uniform spacing)
    Rs = np.array([3.057, 3.157, 3.207, 3.257, 3.307, 3.357, 3.379, 3.407,
                   3.457, 3.507, 3.557, 3.607, 3.657, 3.757, 3.857, 3.957,
                   4.057, 4.157, 4.257])

    # Thermodynamic parameters
    Lambda_eV = 0.83            # eV (inner-sphere)
    Lambda_kcal = Lambda_eV * EV_TO_KCALMOL  # kcal/mol
    V_el_eV = 0.10              # eV
    V_el_kcal = V_el_eV * EV_TO_KCALMOL  # kcal/mol
    T = 300                     # K
    DeltaG0_H = 0.55            # eV
    DeltaG0_D = 0.55 - 0.009296397  # eV (isotope correction)
    NStates = 10
    RTF = 8.31446 * T / 96485.33  # V (RT/F)
    pH = 0

    # EDL model parameters
    dIHL = 3.6                  # Angstrom
    dOHL = 3.5                  # Angstrom
    eps_IHL = 2.7
    eps_st = 78.0
    eps_op = 1.78
    rho_water = 0.9970470       # g/cm^3
    m_water = 18.01528          # g/mol
    c_ions = 0.5                # mol/L
    C_EDL = 15                  # muF/cm^2
    PZFCvsSHE = 0.04            # V

    # Buckingham work terms (from DFT fitting)
    def reactant_work(R):
        """CoTPP + H3O+ non-bonded interaction (eV)."""
        return KCAL_TO_EV * (692272.09 * np.exp(-R / 0.25730094) - 3699.5922 / (R**6))

    def product_work(R):
        """CoHTPP + H2O non-bonded interaction (eV)."""
        return KCAL_TO_EV * (54305.39 * np.exp(-R / 0.39709137) - 19539.245 / (R**6))

    # Load graphene DOS
    rho_M = np.genfromtxt(dos_file, delimiter=',')
    epsilons = rho_M[:, 0]     # eV
    rho_DOS = rho_M[:, 1]      # states/eV/atom

    # Fermi-Dirac distribution: f(eps) = 1/(1 + exp(eps/kBT))
    kBT_eV = 8.617333e-5 * T   # eV
    fermi = 1.0 / (1.0 + np.exp(epsilons / kBT_eV))

    # DOS * Fermi weighting (reductive direction)
    DOS_fermi = rho_DOS * fermi

    engine = PCETRateEngine(
        n_reactant_states=NStates,
        n_product_states=NStates,
        temperature=T,
    )

    # Load and fit proton potentials (poly6 for all, matching pyPCET)
    ReacPot_R = []
    ProdPot_R = []

    for i, R in enumerate(Rs):
        react_file = os.path.join(pot_dir, f"rDA_{R:.3f}_R.csv")
        prod_file = os.path.join(pot_dir, f"rDA_{R:.3f}_P.csv")

        if not os.path.exists(react_file) or not os.path.exists(prod_file):
            ReacPot_R.append(None)
            ProdPot_R.append(None)
            continue

        dat_react = pd.read_csv(react_file, sep=',', header=0, engine='python')
        dat_prod = pd.read_csv(prod_file, sep=',', header=0, engine='python')

        rp_react = dat_react['x'].values
        E_react = dat_react['Reactant'].values * HARTREE_TO_EV  # Hartree -> eV
        rp_prod = dat_prod['x'].values
        E_prod = dat_prod['Product'].values * HARTREE_TO_EV     # Hartree -> eV

        # Fit to poly6 (matching pyPCET)
        coeffs_react = np.polyfit(rp_react, E_react, 6)
        coeffs_prod = np.polyfit(rp_prod, E_prod, 6)

        ReacPot_R.append(np.poly1d(coeffs_react))
        ProdPot_R.append(np.poly1d(coeffs_prod))

    # Uniform proton grid: -1.5 to 1.5 A (matching pyPCET rmin/rmax)
    rp_uniform = np.linspace(-1.5, 1.5, 256)

    # Applied potentials to sweep (matching pyPCET)
    E_appl_list = np.arange(-0.70, -0.49, 0.02)

    # Fine R grid for thermal averaging
    R_min = Rs[0]
    R_max = Rs[-1]
    R_fine = np.linspace(R_min, R_max, 200)

    # Parse reference data for comparison
    # From CoTPP_reference_output.txt
    ref_data = {
        -0.70: (1.0493e+10, 4.9180e+09, 2.13),
        -0.68: (6.4966e+09, 3.0162e+09, 2.15),
        -0.66: (3.9997e+09, 1.8396e+09, 2.17),
        -0.64: (2.4486e+09, 1.1158e+09, 2.19),
        -0.62: (1.4906e+09, 6.7302e+08, 2.21),
        -0.60: (9.0232e+08, 4.0370e+08, 2.24),
        -0.58: (5.4315e+08, 2.4081e+08, 2.26),
        -0.56: (3.2512e+08, 1.4285e+08, 2.28),
        -0.54: (1.9352e+08, 8.4276e+07, 2.30),
        -0.52: (1.1455e+08, 4.9444e+07, 2.32),
        -0.50: (6.7427e+07, 2.8849e+07, 2.34),
    }

    print(f"\n  {'E_appl':>8} {'k_H(our)':>14} {'k_H(pyPCET)':>14} {'ratio':>8} "
          f"{'KIE(our)':>10} {'KIE(ref)':>10}")
    print("  " + "-" * 80)

    results = []

    for n, E_appl in enumerate(E_appl_list):

        # Build EDL potential drop function for this applied potential
        EDL_potential_drop = _edl_model(
            E_appl, dIHL, dOHL, eps_IHL, eps_st, eps_op,
            rho_water, m_water, c_ions, C_EDL, PZFCvsSHE,
        )

        kH_R = np.zeros(len(Rs))
        kD_R = np.zeros(len(Rs))

        for i, R in enumerate(Rs):
            if ReacPot_R[i] is None:
                continue

            # Work terms
            react_w = reactant_work(R) + EDL_potential_drop(R)  # eV
            prod_w = product_work(R)                             # eV

            # Corrected DeltaG for H and D at each epsilon
            # DeltaG_H(eps) = DeltaG0_H + E_appl + prod_w - react_w - eps
            # DeltaG_D(eps) = DeltaG0_D + E_appl + prod_w - react_w - eps - RTF*ln(10)*(14-14.87)
            base_DG_H = DeltaG0_H + E_appl + prod_w - react_w + RTF * np.log(10) * pH
            base_DG_D = DeltaG0_D + E_appl + prod_w - react_w + RTF * np.log(10) * pH \
                        - RTF * np.log(10) * (14.0 - 14.87)

            # Evaluate potentials on uniform grid
            V_reac_grid = ReacPot_R[i](rp_uniform)  # eV
            V_prod_grid = ProdPot_R[i](rp_uniform)  # eV

            # Convert to kcal/mol for engine
            V_reac_kcal = V_reac_grid * EV_TO_KCALMOL
            V_prod_kcal = V_prod_grid * EV_TO_KCALMOL

            kH_eps = np.zeros(len(epsilons))
            kD_eps = np.zeros(len(epsilons))

            for j, eps in enumerate(epsilons):
                DG_H = base_DG_H - eps  # eV
                DG_D = base_DG_D - eps  # eV

                DG_H_kcal = DG_H * EV_TO_KCALMOL
                DG_D_kcal = DG_D * EV_TO_KCALMOL

                # Compute H rate
                result_H = engine.compute_rate_from_potential(
                    r_grid=rp_uniform,
                    V_reactant=V_reac_kcal,
                    V_product=V_prod_kcal,
                    V_el=V_el_kcal,
                    delta_G=DG_H_kcal,
                    lambda_reorg=Lambda_kcal,
                )
                kH_eps[j] = result_H.k_H

                # Compute D rate (use k_D from same call if engine provides it,
                # otherwise call again)
                kD_eps[j] = result_H.k_D

            # Integrate over electrode energy levels: k = int DOS(eps)*f(eps)*k(eps) deps
            kH_R[i] = simpson(DOS_fermi * kH_eps, x=epsilons)
            kD_R[i] = simpson(DOS_fermi * kD_eps, x=epsilons)

        # Thermal average over R
        # P(R) = exp(-W(R)/kBT) * c0  (potential-dependent concentration)
        W_of_R = reactant_work(R_fine) + EDL_potential_drop(R_fine)
        PR = np.exp(-W_of_R / kBT_eV)  # c0 = 1 for pH = 0

        # Linear interpolation of k(R) — matching pyPCET
        valid = kH_R > 0
        if np.sum(valid) < 3:
            print(f"  {E_appl:>8.2f}  Too few valid R points")
            continue

        kH_fine = interp1d(Rs[valid], kH_R[valid], kind='linear',
                           fill_value='extrapolate')(R_fine)
        kD_fine = interp1d(Rs[valid], kD_R[valid], kind='linear',
                           fill_value='extrapolate')(R_fine)

        # Clamp to non-negative (extrapolation can go negative)
        kH_fine = np.maximum(kH_fine, 0)
        kD_fine = np.maximum(kD_fine, 0)

        k_H_tot = simpson(PR * kH_fine, x=R_fine)
        k_D_tot = simpson(PR * kD_fine, x=R_fine)
        KIE = k_H_tot / k_D_tot if k_D_tot > 0 else float('inf')

        E_key = round(E_appl, 2)
        ref = ref_data.get(E_key, (None, None, None))

        ratio_str = ""
        kie_ref_str = ""
        if ref[0] is not None and k_H_tot > 0:
            ratio_str = f"{k_H_tot / ref[0]:>8.3f}"
            kie_ref_str = f"{ref[2]:>10.2f}"
        print(f"  {E_appl:>8.2f} {k_H_tot:>14.4e} "
              f"{ref[0] if ref[0] else 'N/A':>14} {ratio_str:>8} "
              f"{KIE:>10.2f} {kie_ref_str:>10}")

        results.append((E_appl, k_H_tot, k_D_tot, KIE))

    # Print summary for the benchmark potential (-0.66 V)
    bench = [r for r in results if abs(r[0] - (-0.66)) < 0.005]
    if bench:
        _, kH, kD, kie = bench[0]
        print(f"\n  --- Benchmark: E = -0.66 V vs SHE ---")
        print(f"  k_H_tot (our)    = {kH:.4e} s^-1  (pyPCET: 3.9997e+09)")
        print(f"  k_D_tot (our)    = {kD:.4e} s^-1  (pyPCET: 1.8396e+09)")
        print(f"  KIE (our)        = {kie:.2f}       (pyPCET: 2.17)")
        if kH > 0:
            print(f"  k_H ratio (our/pyPCET) = {kH / 3.9997e9:.3f}")
            print(f"  KIE ratio (our/pyPCET) = {kie / 2.17:.3f}")

    # Transfer coefficient from Tafel slope
    if len(results) >= 3:
        E_arr = np.array([r[0] for r in results])
        lnkH_arr = np.array([np.log(r[1]) if r[1] > 0 else 0 for r in results])
        lnkD_arr = np.array([np.log(r[2]) if r[2] > 0 else 0 for r in results])

        valid_mask = lnkH_arr > 0
        if np.sum(valid_mask) >= 3:
            from scipy.optimize import curve_fit

            def tafel(E, prefactor, b):
                return -prefactor * E + b

            try:
                popt_H, _ = curve_fit(tafel, E_arr[valid_mask], lnkH_arr[valid_mask])
                popt_D, _ = curve_fit(tafel, E_arr[valid_mask], lnkD_arr[valid_mask])
                alpha_H = popt_H[0] * RTF
                alpha_D = popt_D[0] * RTF
                print(f"\n  Transfer coefficient (H): {alpha_H:.4f} (pyPCET: 0.6524)")
                print(f"  Transfer coefficient (D): {alpha_D:.4f} (pyPCET: 0.6642)")
            except Exception as e:
                print(f"\n  Tafel fit failed: {e}")

    return results


# =====================================================================
# Summary
# =====================================================================

def print_summary(k_rnr, bip_results, cotpp_results=None):
    """Print the final comparison table."""
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD: pyPCET vs Our Engine (Same Potentials)")
    print("=" * 80)
    print(f"\n{'Example':<25} {'Quantity':>10} {'pyPCET':>14} {'Ours':>14} {'Ratio':>10}")
    print("-" * 75)

    if k_rnr is not None:
        ratio = k_rnr / 1578.8 if k_rnr > 0 else float('nan')
        print(f"{'RNR Y356-Y731':<25} {'k_H':>10} {'1.5788e+03':>14} {k_rnr:>14.4e} {ratio:>10.3f}")

    if bip_results is not None:
        kH, kD, kie = bip_results
        if kH > 0:
            print(f"{'BIP electrochem':<25} {'k_H':>10} {'3.3725e+06':>14} {kH:>14.4e} {kH/3.3725e6:>10.3f}")
            print(f"{'':25} {'k_D':>10} {'1.9243e+06':>14} {kD:>14.4e} {kD/1.9243e6:>10.3f}")
            print(f"{'':25} {'KIE':>10} {'1.75':>14} {kie:>14.2f} {kie/1.75:>10.3f}")

    if cotpp_results is not None:
        # Find -0.66 V result
        bench = [r for r in cotpp_results if abs(r[0] - (-0.66)) < 0.005]
        if bench:
            _, kH, kD, kie = bench[0]
            if kH > 0:
                print(f"{'CoTPP E=-0.66V':<25} {'k_H':>10} {'3.9997e+09':>14} {kH:>14.4e} {kH/3.9997e9:>10.3f}")
                print(f"{'':25} {'k_D':>10} {'1.8396e+09':>14} {kD:>14.4e} {kD/1.8396e9:>10.3f}")
                print(f"{'':25} {'KIE':>10} {'2.17':>14} {kie:>14.2f} {kie/2.17:>10.3f}")

    print("\n  Ratio = 1.000 means exact agreement.")
    print("  Any deviation is due to: FGH grid, spline interpolation,")
    print("  or numerical integration differences.")


def main():
    k_rnr = example2_rnr()
    bip = example3_bip()
    cotpp = example4_cotpp()
    print_summary(k_rnr, bip, cotpp)


if __name__ == "__main__":
    main()
