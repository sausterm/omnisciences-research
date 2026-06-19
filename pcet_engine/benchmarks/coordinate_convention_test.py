"""
Test whether the α discrepancy comes from the coordinate convention.

Key question: When R_DA changes by dR, does δ (tunneling distance) change by dR?

In Soudackov's formulation:
  α = -d(log S)/dR  where R is the D-A distance

In our formulation:
  α_ours = -d(log S)/dδ  where δ = R - r_CH - r_OH

If r_CH and r_OH are fixed, then dδ = dR, and α_ours = α_Soudackov.
BUT if r_CH and r_OH depend on R (e.g., the proton equilibrium positions
shift as the donor-acceptor distance changes), then:
  α_Soud = α_ours * (dδ/dR)
  and dδ/dR < 1 would explain α_Soud < α_ours

HOWEVER: The ratio α_ours/α_Soud ≈ 1.57 consistently.
If dδ/dR = 1/1.57 = 0.637, then dR of 1 Å would only increase δ by 0.637 Å.
That means r_CH + r_OH shift outward by 0.363 Å per 1 Å of R increase.
Doesn't make physical sense for fixed Morse wells.

Alternative: maybe the issue is that the Morse wavefunctions should be
parameterized by β*(r - r_eq) where r is measured from the respective
atom, not from a common origin.

Let me test: what if we keep the wavefunctions fixed in their own frames
and vary only the inter-well distance?

Actually, the most likely explanation: the ASYMMETRIC Morse wells.
The C-H and O-H Morse parameters differ (β_CH ≠ β_OH, D_CH ≠ D_OH).
Soudackov might define the overlap differently w.r.t. coordinate frame.

Key test: compute α analytically from the Morse wavefunction form.
For the ground state Morse: ψ₀(r) ∝ z^(λ-1/2) exp(-z/2) where z = 2λ exp(-β r)
In the tail (large r → z → 0): ψ₀ ~ exp(-(λ-1/2) β r)
So the tail decays as exp(-κ r) with κ = (λ-1/2)β

For the overlap integral, the dominant contribution is where both
wavefunctions have significant amplitude — the classically forbidden region.
The overlap should decay as exp(-(κ_R + κ_P) δ) approximately, giving
α ≈ κ_R + κ_P = (λ_R - 1/2)β_R + (λ_P - 1/2)β_P

Let's check this!
"""

import math
import numpy as np
from pcet_engine.core.constants import (
    ANGSTROM_TO_BOHR, AMU_TO_AU, KCALMOL_TO_HARTREE, HBAR_AU,
)

# Soudackov's Morse parameters
D_CH_KCAL = 77.0
D_OH_KCAL = 82.0
BETA_CH_AINV = 2.068
BETA_OH_AINV = 2.442
R_CH = 1.09
R_OH = 0.96

D_CH = D_CH_KCAL * KCALMOL_TO_HARTREE
D_OH = D_OH_KCAL * KCALMOL_TO_HARTREE
BETA_CH = BETA_CH_AINV * ANGSTROM_TO_BOHR  # bohr⁻¹
BETA_OH = BETA_OH_AINV * ANGSTROM_TO_BOHR  # bohr⁻¹

MASS_H = 1.00782503207
MASS_D = 2.01410177812


def morse_lambda(mass_amu, D_e, beta):
    mass_au = mass_amu * AMU_TO_AU
    return math.sqrt(2.0 * mass_au * D_e) / (HBAR_AU * beta)


def tail_decay_rate(mass_amu, D_e, beta, n_state=0):
    """Decay rate κ_n = (λ - n - 1/2) × β for the Morse wavefunction tail."""
    lam = morse_lambda(mass_amu, D_e, beta)
    return (lam - n_state - 0.5) * beta


def predicted_alpha(mass_amu, mu, nu):
    """Predicted α from sum of tail decay rates: α ≈ κ_R(μ) + κ_P(ν)."""
    kappa_R = tail_decay_rate(mass_amu, D_CH, BETA_CH, mu)
    kappa_P = tail_decay_rate(mass_amu, D_OH, BETA_OH, nu)
    return kappa_R + kappa_P


def main():
    print("=" * 70)
    print("COORDINATE CONVENTION TEST: α from tail decay rates")
    print("=" * 70)
    print()
    print("If α ≈ κ_R(μ) + κ_P(ν) where κ_n = (λ-n-1/2)β,")
    print("then α is determined by the Morse parameter λ and β.")
    print()

    # Reference
    from pcet_engine.benchmarks.soudackov_correction import soudackov_reference_params
    ref_aH, ref_aD, ref_gH, ref_gD = soudackov_reference_params()

    for mass_label, mass_amu, ref_a in [("H", MASS_H, ref_aH), ("D", MASS_D, ref_aD)]:
        lam_CH = morse_lambda(mass_amu, D_CH, BETA_CH)
        lam_OH = morse_lambda(mass_amu, D_OH, BETA_OH)
        print(f"--- {mass_label} ---")
        print(f"  λ(C-H) = {lam_CH:.4f}, β_CH = {BETA_CH:.4f} bohr⁻¹")
        print(f"  λ(O-H) = {lam_OH:.4f}, β_OH = {BETA_OH:.4f} bohr⁻¹")
        print()

        for n in range(4):
            kappa_CH = tail_decay_rate(mass_amu, D_CH, BETA_CH, n)
            kappa_OH = tail_decay_rate(mass_amu, D_OH, BETA_OH, n)
            print(f"  κ_CH(n={n}) = {kappa_CH:.4f}, κ_OH(n={n}) = {kappa_OH:.4f}")

        print()
        print(f"  {'(μ,ν)':<8} {'α_pred':<12} {'α_Soud':<12} {'ratio':<8} {'α_num':<12}")
        print(f"  {'-'*52}")

        # Also load our numerical analytical results
        from pcet_engine.benchmarks.analytical_morse_test import compute_attenuation_analytical
        _, alpha_num, _ = compute_attenuation_analytical(mass_amu)

        for mu in range(4):
            for nu in range(4):
                a_pred = predicted_alpha(mass_amu, mu, nu)
                a_soud = ref_a[mu, nu]
                ratio = a_pred / a_soud if a_soud > 0 else 0
                a_n = alpha_num[mu, nu]
                print(f"  ({mu},{nu})    {a_pred:<12.4f} {a_soud:<12.4f} {ratio:<8.3f} {a_n:<12.4f}")
        print()


if __name__ == "__main__":
    main()
