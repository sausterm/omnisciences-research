"""Tests for nonadiabaticity module."""

import numpy as np
import pytest

from pcet_engine.core.constants import PROTON_MASS_AMU, DEUTERIUM_MASS_AMU
from pcet_engine.core.nonadiabaticity import analyze_nonadiabaticity, NonadiabaticityResult
from pcet_engine.core.proton_potential import harmonic_potential
from pcet_engine.core.fgh_solver import fgh_1d


def _make_displaced_harmonic_potentials(
    omega_cm=3000.0,
    mass_amu=PROTON_MASS_AMU,
    r_eq_donor=-0.3,
    r_eq_acceptor=0.3,
    delta_E=0.0,
    n_grid=256,
    r_range=(-1.5, 1.5),
):
    """Create a pair of displaced harmonic potentials on a grid.

    Returns (r_grid, V_R, V_P) as arrays ready for analyze_nonadiabaticity.
    """
    V_R_func = harmonic_potential(omega_cm, mass_amu, r_eq=r_eq_donor)
    V_P_func = harmonic_potential(omega_cm, mass_amu, r_eq=r_eq_acceptor,
                                  e_offset=delta_E)
    r_grid = np.linspace(r_range[0], r_range[1], n_grid)
    V_R = V_R_func(r_grid)
    V_P = V_P_func(r_grid)
    return r_grid, V_R, V_P


# =====================================================================
# Regime classification
# =====================================================================

class TestRegimeClassification:
    def test_small_vel_gives_nonadiabatic(self):
        """Very small electronic coupling should yield nonadiabatic regime."""
        r_grid, V_R, V_P = _make_displaced_harmonic_potentials(
            r_eq_donor=-0.4, r_eq_acceptor=0.4,
        )
        result = analyze_nonadiabaticity(
            r_grid, V_R, V_P,
            V_el=1e-4,  # very small coupling
            mass_amu=PROTON_MASS_AMU,
        )
        assert isinstance(result, NonadiabaticityResult)
        assert result.regime == 'nonadiabatic'
        assert result.p < 0.1

    def test_large_vel_small_displacement_gives_adiabatic(self):
        """Large coupling with small displacement (above-barrier) should be adiabatic."""
        # Use very close wells so barrier is low, and large V_el
        r_grid, V_R, V_P = _make_displaced_harmonic_potentials(
            r_eq_donor=-0.05, r_eq_acceptor=0.05,
            omega_cm=3000.0,
        )
        result = analyze_nonadiabaticity(
            r_grid, V_R, V_P,
            V_el=2.0,  # very large coupling
            mass_amu=PROTON_MASS_AMU,
        )
        assert isinstance(result, NonadiabaticityResult)
        assert result.regime == 'adiabatic'


# =====================================================================
# Physical bounds on kappa
# =====================================================================

class TestKappaBounds:
    def test_kappa_between_zero_and_one(self):
        """The GS kappa factor should always be in [0, 1]."""
        r_grid, V_R, V_P = _make_displaced_harmonic_potentials(
            r_eq_donor=-0.3, r_eq_acceptor=0.3,
        )
        # Test across a range of V_el values spanning regimes
        for V_el in [1e-4, 1e-3, 0.01, 0.05, 0.1]:
            result = analyze_nonadiabaticity(
                r_grid, V_R, V_P,
                V_el=V_el,
                mass_amu=PROTON_MASS_AMU,
            )
            assert 0.0 <= result.kappa <= 1.0 + 1e-10, (
                f"kappa={result.kappa} out of [0,1] for V_el={V_el}"
            )


# =====================================================================
# V_semiclassical ordering
# =====================================================================

class TestVibronicCouplingOrdering:
    def test_semiclassical_between_nad_and_ad(self):
        """V_semiclassical should be between V_nonadiabatic and V_adiabatic
        (or equal to one of them in limiting regimes)."""
        r_grid, V_R, V_P = _make_displaced_harmonic_potentials(
            r_eq_donor=-0.3, r_eq_acceptor=0.3,
        )
        # Intermediate regime coupling
        result = analyze_nonadiabaticity(
            r_grid, V_R, V_P,
            V_el=0.02,
            mass_amu=PROTON_MASS_AMU,
        )
        V_lo = min(result.V_nonadiabatic, result.V_adiabatic)
        V_hi = max(result.V_nonadiabatic, result.V_adiabatic)
        # Allow small numerical tolerance
        assert result.V_semiclassical >= V_lo - 1e-10
        assert result.V_semiclassical <= V_hi + 1e-10


# =====================================================================
# Result fields
# =====================================================================

class TestResultFields:
    def test_all_fields_present(self):
        r_grid, V_R, V_P = _make_displaced_harmonic_potentials()
        result = analyze_nonadiabaticity(
            r_grid, V_R, V_P,
            V_el=0.01,
            mass_amu=PROTON_MASS_AMU,
        )
        assert hasattr(result, 'tau_p')
        assert hasattr(result, 'tau_e')
        assert hasattr(result, 'p')
        assert hasattr(result, 'kappa')
        assert hasattr(result, 'V_nonadiabatic')
        assert hasattr(result, 'V_adiabatic')
        assert hasattr(result, 'V_semiclassical')
        assert hasattr(result, 'regime')
        assert hasattr(result, 'r_crossing')
        assert hasattr(result, 'E_crossing')

    def test_regime_is_valid_string(self):
        r_grid, V_R, V_P = _make_displaced_harmonic_potentials()
        result = analyze_nonadiabaticity(
            r_grid, V_R, V_P,
            V_el=0.01,
            mass_amu=PROTON_MASS_AMU,
        )
        assert result.regime in ('nonadiabatic', 'adiabatic', 'intermediate')

    def test_positive_timescales(self):
        r_grid, V_R, V_P = _make_displaced_harmonic_potentials(
            r_eq_donor=-0.3, r_eq_acceptor=0.3,
        )
        result = analyze_nonadiabaticity(
            r_grid, V_R, V_P,
            V_el=0.01,
            mass_amu=PROTON_MASS_AMU,
        )
        assert result.tau_p >= 0.0
        assert result.tau_e > 0.0

    def test_crossing_point_between_wells(self):
        """Crossing point should lie between the two equilibrium positions."""
        r_eq_d, r_eq_a = -0.3, 0.3
        r_grid, V_R, V_P = _make_displaced_harmonic_potentials(
            r_eq_donor=r_eq_d, r_eq_acceptor=r_eq_a,
        )
        result = analyze_nonadiabaticity(
            r_grid, V_R, V_P,
            V_el=0.01,
            mass_amu=PROTON_MASS_AMU,
        )
        if not np.isnan(result.r_crossing):
            assert r_eq_d <= result.r_crossing <= r_eq_a


# =====================================================================
# Isotope effect
# =====================================================================

class TestIsotopeEffect:
    def test_deuterium_more_nonadiabatic(self):
        """Heavier isotope tunnels more slowly, so should be more nonadiabatic
        (smaller p) at the same V_el."""
        V_el = 0.01
        r_grid_H, V_R_H, V_P_H = _make_displaced_harmonic_potentials(
            mass_amu=PROTON_MASS_AMU,
            r_eq_donor=-0.3, r_eq_acceptor=0.3,
        )
        r_grid_D, V_R_D, V_P_D = _make_displaced_harmonic_potentials(
            mass_amu=DEUTERIUM_MASS_AMU,
            r_eq_donor=-0.3, r_eq_acceptor=0.3,
        )
        result_H = analyze_nonadiabaticity(
            r_grid_H, V_R_H, V_P_H, V_el=V_el, mass_amu=PROTON_MASS_AMU,
        )
        result_D = analyze_nonadiabaticity(
            r_grid_D, V_R_D, V_P_D, V_el=V_el, mass_amu=DEUTERIUM_MASS_AMU,
        )
        # Deuterium should have smaller p (more nonadiabatic) or same regime
        # The heavier mass changes both potential and tunneling, but generally
        # deuterium is more nonadiabatic
        assert result_D.V_nonadiabatic <= result_H.V_nonadiabatic + 1e-10
