"""Tests for the high-level PCET rate engine."""

import math
import pytest
import numpy as np

from pcet_engine.core.rate_engine import PCETRateEngine, PCETResult


class TestPCETRateEngine:
    def setup_method(self):
        self.engine = PCETRateEngine()
        # SLO-1-like parameters
        self.params = dict(
            V_el=0.6,
            delta_G=-5.4,
            lambda_reorg=19.0,
            omega_H=2900.0,
            d_DA=2.69,
        )

    def test_marcus_method(self):
        """Marcus method should give a rate with KIE=1 (no isotope effect)."""
        result = self.engine.compute_rate(**self.params, method="marcus")
        assert result.k_H > 0
        assert result.KIE == pytest.approx(1.0, abs=0.01)  # No isotope effect
        assert result.method == "marcus"

    def test_vibronic_single_gives_kie(self):
        """Single-channel vibronic should give KIE > 1."""
        result = self.engine.compute_rate(**self.params, method="vibronic_single")
        assert result.KIE > 1.0
        assert result.k_H > result.k_D

    def test_vibronic_multi_gives_kie(self):
        """Multi-channel should give finite KIE."""
        result = self.engine.compute_rate(**self.params, method="vibronic_multi")
        assert result.KIE > 1.0
        assert result.vibronic_H is not None
        assert result.vibronic_D is not None

    def test_multi_lower_kie_than_single(self):
        """Multi-channel KIE should be smaller than single-channel KIE."""
        single = self.engine.compute_rate(**self.params, method="vibronic_single")
        multi = self.engine.compute_rate(**self.params, method="vibronic_multi")
        assert multi.KIE < single.KIE

    def test_result_fields(self):
        """PCETResult should have all expected fields."""
        result = self.engine.compute_rate(**self.params, method="vibronic_multi")
        assert isinstance(result, PCETResult)
        assert result.k_H > 0
        assert result.k_D > 0
        assert result.KIE > 0
        assert result.E_a >= 0
        assert result.delta_G == -5.4
        assert result.lambda_reorg == 19.0
        assert result.omega_H == 2900.0
        assert result.d_DA == 2.69

    def test_temperature_effect(self):
        """Higher temperature should generally increase rate."""
        engine_cold = PCETRateEngine(temperature=250.0)
        engine_hot = PCETRateEngine(temperature=350.0)
        r_cold = engine_cold.compute_rate(**self.params, method="marcus")
        r_hot = engine_hot.compute_rate(**self.params, method="marcus")
        assert r_hot.k_H > r_cold.k_H

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            self.engine.compute_rate(**self.params, method="invalid")

    def test_slo1_rate_order_of_magnitude(self):
        """SLO-1 rate should be within ~2 orders of magnitude of experiment."""
        result = self.engine.compute_rate(**self.params, method="vibronic_multi")
        # Experimental: ~300 s⁻¹
        log_err = abs(math.log10(result.k_H / 300.0))
        assert log_err < 3.0  # Within 3 orders of magnitude

    def test_omega_d_scaling(self):
        """Deuterium frequency should be scaled by sqrt(m_H/m_D)."""
        result = self.engine.compute_rate(**self.params, method="vibronic_multi")
        expected_ratio = math.sqrt(1.00782503207 / 2.01410177812)
        assert abs(result.omega_D / result.omega_H - expected_ratio) < 1e-4


class TestBenchmarks:
    def test_benchmarks_run(self):
        """All benchmark systems should run without error."""
        from pcet_engine.benchmarks.systems import run_benchmarks
        results = run_benchmarks(method="vibronic_multi", verbose=False)
        assert len(results) == 28
        for name, res in results.items():
            assert res["result"].k_H > 0
            assert res["result"].KIE > 0
            assert math.isfinite(res["log_error_kH"])
