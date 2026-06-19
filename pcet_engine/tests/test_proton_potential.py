"""Tests for proton_potential module."""

import numpy as np
import pytest

from pcet_engine.core.constants import PROTON_MASS_AMU, DEUTERIUM_MASS_AMU
from pcet_engine.core.proton_potential import (
    harmonic_potential,
    morse_potential,
    double_well_potential,
    fit_potential_from_scan,
    potential_from_hessian,
)
from pcet_engine.core.fgh_solver import fgh_1d


# =====================================================================
# harmonic_potential
# =====================================================================

class TestHarmonicPotential:
    def test_returns_callable(self):
        V = harmonic_potential(omega_cm=3000.0, mass_amu=PROTON_MASS_AMU)
        assert callable(V)

    def test_minimum_at_r_eq(self):
        r_eq = 1.0
        V = harmonic_potential(omega_cm=3000.0, mass_amu=PROTON_MASS_AMU, r_eq=r_eq)
        r = np.linspace(-1.0, 3.0, 500)
        vals = V(r)
        r_min = r[np.argmin(vals)]
        assert abs(r_min - r_eq) < (r[1] - r[0])

    def test_minimum_value_equals_offset(self):
        offset = 0.5
        V = harmonic_potential(omega_cm=3000.0, mass_amu=PROTON_MASS_AMU,
                               r_eq=0.0, e_offset=offset)
        assert abs(V(0.0) - offset) < 1e-12

    def test_symmetry_about_r_eq(self):
        r_eq = 0.5
        V = harmonic_potential(omega_cm=3000.0, mass_amu=PROTON_MASS_AMU, r_eq=r_eq)
        delta = 0.3
        assert abs(V(r_eq + delta) - V(r_eq - delta)) < 1e-12

    def test_accepts_array_input(self):
        V = harmonic_potential(omega_cm=3000.0, mass_amu=PROTON_MASS_AMU)
        r = np.array([0.0, 0.1, 0.2])
        vals = V(r)
        assert vals.shape == (3,)

    def test_heavier_mass_narrower_well(self):
        """Heavier mass at same frequency gives a stiffer (larger force constant) well."""
        V_H = harmonic_potential(omega_cm=3000.0, mass_amu=PROTON_MASS_AMU)
        V_D = harmonic_potential(omega_cm=3000.0, mass_amu=DEUTERIUM_MASS_AMU)
        # At same displacement, heavier mass should give higher energy (k = m*omega^2)
        r_test = 0.2  # angstrom from equilibrium
        assert V_D(r_test) > V_H(r_test)


# =====================================================================
# morse_potential
# =====================================================================

class TestMorsePotential:
    def test_well_depth(self):
        De = 4.5
        V = morse_potential(De_eV=De, beta_inv_angstrom=2.0, r_eq=1.0)
        # At r -> inf, V -> De; at r_eq, V = 0
        r_far = np.array([50.0])
        assert abs(V(r_far)[0] - De) < 0.01

    def test_minimum_at_r_eq(self):
        r_eq = 1.0
        V = morse_potential(De_eV=4.5, beta_inv_angstrom=2.0, r_eq=r_eq)
        r = np.linspace(0.0, 3.0, 1000)
        vals = V(r)
        r_min = r[np.argmin(vals)]
        assert abs(r_min - r_eq) < (r[1] - r[0])

    def test_minimum_value_with_offset(self):
        offset = 1.0
        V = morse_potential(De_eV=4.5, beta_inv_angstrom=2.0, r_eq=1.0,
                            e_offset=offset)
        assert abs(V(1.0) - offset) < 1e-12

    def test_inverted_form(self):
        """Inverted Morse should have minimum at r_eq from the other side."""
        r_eq = 1.0
        V_normal = morse_potential(De_eV=4.5, beta_inv_angstrom=2.0, r_eq=r_eq,
                                   inverted=False)
        V_inv = morse_potential(De_eV=4.5, beta_inv_angstrom=2.0, r_eq=r_eq,
                                inverted=True)
        # Both should have minimum at r_eq
        assert abs(V_normal(r_eq)) < 1e-12
        assert abs(V_inv(r_eq)) < 1e-12

        # Normal Morse rises for r > r_eq approaching De
        # Inverted Morse rises for r < r_eq approaching De
        assert V_normal(r_eq + 5.0) > V_normal(r_eq + 0.1)
        assert V_inv(r_eq - 5.0) > V_inv(r_eq - 0.1)

    def test_inverted_asymptote(self):
        De = 4.5
        V_inv = morse_potential(De_eV=De, beta_inv_angstrom=2.0, r_eq=1.0,
                                inverted=True)
        # At r -> -inf, V_inv -> De
        r_far_neg = np.array([-50.0])
        assert abs(V_inv(r_far_neg)[0] - De) < 0.01


# =====================================================================
# fit_potential_from_scan
# =====================================================================

class TestFitPotentialFromScan:
    @pytest.fixture
    def harmonic_scan_data(self):
        """Generate synthetic harmonic scan data."""
        r = np.linspace(-1.0, 1.0, 50)
        E = 2.0 * r**2 + 0.5  # parabola with offset
        return r, E

    def test_bspline_method(self, harmonic_scan_data):
        r, E = harmonic_scan_data
        V = fit_potential_from_scan(r, E, method="bspline")
        assert callable(V)
        # Minimum should be near zero (shifted)
        vals = V(r)
        assert np.min(vals) < 0.05

    def test_poly6_method(self, harmonic_scan_data):
        r, E = harmonic_scan_data
        V = fit_potential_from_scan(r, E, method="poly6")
        assert callable(V)
        vals = V(r)
        # Should reproduce parabolic shape: min near r=0
        r_min = r[np.argmin(vals)]
        assert abs(r_min) < 0.1

    def test_poly8_method(self, harmonic_scan_data):
        r, E = harmonic_scan_data
        V = fit_potential_from_scan(r, E, method="poly8")
        assert callable(V)
        vals = V(r)
        r_min = r[np.argmin(vals)]
        assert abs(r_min) < 0.1

    def test_invalid_method_raises(self, harmonic_scan_data):
        r, E = harmonic_scan_data
        with pytest.raises(ValueError, match="method must be"):
            fit_potential_from_scan(r, E, method="cubic")

    def test_fit_preserves_shape(self, harmonic_scan_data):
        r, E = harmonic_scan_data
        V = fit_potential_from_scan(r, E, method="poly6")
        # Should be roughly parabolic: edge values higher than center
        assert V(r[0]) > V(r[len(r) // 2])
        assert V(r[-1]) > V(r[len(r) // 2])


# =====================================================================
# potential_from_hessian
# =====================================================================

class TestPotentialFromHessian:
    def test_returns_two_callables(self):
        V_R, V_P = potential_from_hessian(
            omega_cm=3000.0,
            mass_amu=PROTON_MASS_AMU,
            r_eq_donor=0.0,
            r_eq_acceptor=1.5,
        )
        assert callable(V_R)
        assert callable(V_P)

    def test_harmonic_case_minima(self):
        """Without anharmonicity, should produce harmonic wells at correct positions."""
        V_R, V_P = potential_from_hessian(
            omega_cm=3000.0,
            mass_amu=PROTON_MASS_AMU,
            r_eq_donor=0.0,
            r_eq_acceptor=1.5,
        )
        r = np.linspace(-1.0, 2.5, 500)
        r_min_R = r[np.argmin(V_R(r))]
        r_min_P = r[np.argmin(V_P(r))]
        dr = r[1] - r[0]
        assert abs(r_min_R - 0.0) < dr
        assert abs(r_min_P - 1.5) < dr

    def test_anharmonic_spacing_less_than_harmonic(self):
        """Morse anharmonicity should give smaller level spacing than harmonic."""
        omega = 3000.0
        mass = PROTON_MASS_AMU
        r_eq_d = 0.0
        r_eq_a = 2.5

        # Harmonic case
        V_R_harm, _ = potential_from_hessian(
            omega_cm=omega, mass_amu=mass,
            r_eq_donor=r_eq_d, r_eq_acceptor=r_eq_a,
            anharmonicity_cm=0.0,
        )

        # Anharmonic case (typical O-H anharmonicity ~80 cm^-1)
        V_R_anh, _ = potential_from_hessian(
            omega_cm=omega, mass_amu=mass,
            r_eq_donor=r_eq_d, r_eq_acceptor=r_eq_a,
            anharmonicity_cm=80.0,
        )

        # Solve FGH on both
        r_grid = np.linspace(-1.5, 1.5, 256)
        E_harm, _, _ = fgh_1d(r_grid, V_R_harm(r_grid), mass, n_states=5)
        E_anh, _, _ = fgh_1d(r_grid, V_R_anh(r_grid), mass, n_states=5)

        # Morse spacing (E1-E0) should be less than harmonic spacing
        spacing_harm = E_harm[1] - E_harm[0]
        spacing_anh = E_anh[1] - E_anh[0]
        assert spacing_anh < spacing_harm

    def test_delta_E_offset(self):
        """Product well should be offset by delta_E."""
        delta_E = -0.3
        V_R, V_P = potential_from_hessian(
            omega_cm=3000.0,
            mass_amu=PROTON_MASS_AMU,
            r_eq_donor=0.0,
            r_eq_acceptor=1.5,
            delta_E_eV=delta_E,
        )
        # Product minimum should be at delta_E relative to reactant minimum
        assert abs(V_P(1.5) - (V_R(0.0) + delta_E)) < 1e-10

    def test_anharmonic_returns_morse(self):
        """With anharmonicity, result should be Morse (finite dissociation energy)."""
        V_R, _ = potential_from_hessian(
            omega_cm=3000.0,
            mass_amu=PROTON_MASS_AMU,
            r_eq_donor=0.0,
            r_eq_acceptor=2.0,
            anharmonicity_cm=80.0,
        )
        # Far from equilibrium, Morse approaches De (finite), not infinity
        V_near = V_R(2.0)
        V_far = V_R(10.0)
        V_farther = V_R(20.0)
        # Should plateau — difference between far and farther should be small
        assert abs(V_farther - V_far) < abs(V_far - V_near)
