"""Tests for uncertainty quantification module."""

import numpy as np
import pytest
from pcet_engine.core.uncertainty import propagate_uncertainty


class TestUncertaintyPropagation:
    def test_zero_uncertainty_gives_small_spread(self):
        result = propagate_uncertainty(
            V_el=0.5, V_el_err=0.0,
            delta_G=-5.0, delta_G_err=0.0,
            lambda_reorg=20.0, lambda_reorg_err=0.0,
            omega_H=3000, omega_H_err=0.0,
            d_DA=2.7, d_DA_err=0.0,
            n_samples=50, seed=42,
        )
        # With zero error, std should be essentially zero
        assert result.k_H_std / result.k_H_mean < 0.01
        assert result.KIE_std / result.KIE_mean < 0.01

    def test_large_uncertainty_gives_large_spread(self):
        result = propagate_uncertainty(
            V_el=0.5, V_el_err=0.2,
            delta_G=-5.0, delta_G_err=2.0,
            lambda_reorg=20.0, lambda_reorg_err=5.0,
            omega_H=3000, omega_H_err=500,
            d_DA=2.7, d_DA_err=0.3,
            n_samples=200, seed=42,
        )
        # Large errors should produce significant spread
        assert result.k_H_std > 0
        assert result.KIE_std > 0
        assert result.k_H_ci[0] < result.k_H_mean < result.k_H_ci[1]

    def test_positive_rates(self):
        result = propagate_uncertainty(
            V_el=0.5, V_el_err=0.05,
            delta_G=-5.0, delta_G_err=0.5,
            lambda_reorg=20.0, lambda_reorg_err=2.0,
            omega_H=3000, omega_H_err=100,
            d_DA=2.7, d_DA_err=0.05,
            n_samples=100, seed=42,
        )
        assert result.k_H_mean > 0
        assert result.k_D_mean > 0
        assert result.KIE_mean > 0

    def test_sensitivities_computed(self):
        result = propagate_uncertainty(
            V_el=0.5, V_el_err=0.05,
            delta_G=-5.0, delta_G_err=0.5,
            lambda_reorg=20.0, lambda_reorg_err=2.0,
            omega_H=3000, omega_H_err=100,
            d_DA=2.7, d_DA_err=0.05,
            n_samples=50, seed=42,
        )
        assert "V_el" in result.sensitivities
        assert "delta_G" in result.sensitivities
        assert "lambda_reorg" in result.sensitivities

    def test_vel_sensitivity_positive(self):
        """Rate should increase with V_el (V² dependence)."""
        result = propagate_uncertainty(
            V_el=0.5, V_el_err=0.01,
            delta_G=-5.0, delta_G_err=0.01,
            lambda_reorg=20.0, lambda_reorg_err=0.01,
            omega_H=3000, omega_H_err=1,
            d_DA=2.7, d_DA_err=0.001,
            n_samples=50, seed=42,
        )
        # d(ln k)/d(V_el) should be positive (higher coupling = higher rate)
        assert result.sensitivities["V_el"] > 0

    def test_reproducibility_with_seed(self):
        kwargs = dict(
            V_el=0.5, V_el_err=0.05,
            delta_G=-5.0, delta_G_err=0.5,
            lambda_reorg=20.0, lambda_reorg_err=2.0,
            omega_H=3000, omega_H_err=100,
            d_DA=2.7, d_DA_err=0.05,
            n_samples=50, seed=123,
        )
        r1 = propagate_uncertainty(**kwargs)
        r2 = propagate_uncertainty(**kwargs)
        assert abs(r1.k_H_mean - r2.k_H_mean) < 1e-10
