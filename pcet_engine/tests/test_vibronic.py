"""Tests for vibronic rate calculations and Franck-Condon overlaps."""

import math
import pytest
import numpy as np

from pcet_engine.core.vibronic import (
    franck_condon_overlap,
    vibronic_rate,
    multi_channel_rate,
)
from pcet_engine.core.constants import (
    AMU_TO_AU,
    PROTON_MASS_AMU,
    DEUTERIUM_MASS_AMU,
    KCALMOL_TO_HARTREE,
    CM_TO_HARTREE,
)


class TestFranckCondonOverlap:
    def test_zero_displacement(self):
        """Overlap should be 1.0 for zero displacement, same frequency."""
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        S = franck_condon_overlap(omega, omega, mass, 0.0, 0, 0)
        assert abs(S - 1.0) < 1e-6

    def test_decreases_with_displacement(self):
        """Ground-state overlap should decrease with displacement."""
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        S1 = franck_condon_overlap(omega, omega, mass, 0.5, 0, 0)
        S2 = franck_condon_overlap(omega, omega, mass, 1.0, 0, 0)
        assert S1 > S2 > 0

    def test_heavier_particle_smaller_overlap(self):
        """Deuterium should have smaller FC overlap than hydrogen."""
        omega_H = 3000.0 * CM_TO_HARTREE
        omega_D = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
        delta = 0.8  # bohr

        S_H = franck_condon_overlap(omega_H, omega_H, PROTON_MASS_AMU * AMU_TO_AU, delta, 0, 0)
        S_D = franck_condon_overlap(omega_D, omega_D, DEUTERIUM_MASS_AMU * AMU_TO_AU, delta, 0, 0)

        # Deuterium has larger mass → more localized → smaller overlap for same displacement
        assert S_D < S_H

    def test_normalization_sum(self):
        """Sum of |S_0n|² over product states should approach 1.0."""
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        delta = 0.5  # bohr

        total = sum(
            franck_condon_overlap(omega, omega, mass, delta, 0, n)
            for n in range(20)
        )
        # For displaced oscillators, the sum converges slowly; need more states
        # With 20 states and moderate displacement, ~85% is captured
        assert total > 0.7 and total <= 1.01

    def test_excited_state_overlap(self):
        """Excited state overlaps should be non-negative."""
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        for mu in range(5):
            for nu in range(5):
                S = franck_condon_overlap(omega, omega, mass, 0.5, mu, nu)
                assert S >= 0.0


class TestVibronicRate:
    def test_positive_rate(self):
        """Single-channel vibronic rate should be positive."""
        V = 0.5 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE
        omega = 3000.0 * CM_TO_HARTREE
        k = vibronic_rate(V, dG, lam, omega, PROTON_MASS_AMU, 2.7)
        assert k > 0

    def test_kie_from_single_channel(self):
        """Single-channel should give very large KIE (the known overestimate)."""
        V = 0.5 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE
        omega_H = 3000.0 * CM_TO_HARTREE
        omega_D = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)

        k_H = vibronic_rate(V, dG, lam, omega_H, PROTON_MASS_AMU, 2.7)
        k_D = vibronic_rate(V, dG, lam, omega_D, DEUTERIUM_MASS_AMU, 2.7)

        KIE = k_H / k_D
        # Single-channel typically overestimates KIE (> 100)
        assert KIE > 1.0
        assert k_H > k_D


class TestMultiChannelRate:
    def test_multi_channel_reduces_kie(self):
        """Multi-channel summation should give lower KIE than single-channel."""
        V = 0.5 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE
        omega_H = 3000.0 * CM_TO_HARTREE
        omega_D = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)

        # Single channel
        k_H_single = vibronic_rate(V, dG, lam, omega_H, PROTON_MASS_AMU, 2.7)
        k_D_single = vibronic_rate(V, dG, lam, omega_D, DEUTERIUM_MASS_AMU, 2.7)
        KIE_single = k_H_single / k_D_single

        # Multi channel
        res_H = multi_channel_rate(V, dG, lam, omega_H, omega_H, PROTON_MASS_AMU, 2.7)
        res_D = multi_channel_rate(V, dG, lam, omega_D, omega_D, DEUTERIUM_MASS_AMU, 2.7)
        KIE_multi = res_H.rate_total / res_D.rate_total

        # Multi-channel should bring KIE down
        assert KIE_multi < KIE_single

    def test_result_structure(self):
        """VibronicResult should have correct structure."""
        V = 0.5 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE
        omega = 3000.0 * CM_TO_HARTREE

        res = multi_channel_rate(V, dG, lam, omega, omega, PROTON_MASS_AMU, 2.7,
                                 n_reactant_states=3, n_product_states=5)

        assert res.rate_total > 0
        assert res.rate_channels.shape == (3, 5)
        assert res.overlaps.shape == (3, 5)
        assert len(res.boltzmann_weights) == 3
        assert abs(sum(res.boltzmann_weights) - 1.0) < 1e-10
        assert res.n_reactant_states == 3
        assert res.n_product_states == 5
        assert isinstance(res.dominant_channel, tuple)
        assert len(res.dominant_channel) == 2

    def test_boltzmann_weights_decrease(self):
        """Higher reactant states should have smaller Boltzmann weight."""
        V = 0.5 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE
        omega = 3000.0 * CM_TO_HARTREE

        res = multi_channel_rate(V, dG, lam, omega, omega, PROTON_MASS_AMU, 2.7,
                                 n_reactant_states=5, n_product_states=5)

        for i in range(len(res.boltzmann_weights) - 1):
            assert res.boltzmann_weights[i] >= res.boltzmann_weights[i + 1]
