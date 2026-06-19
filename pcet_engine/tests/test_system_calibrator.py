"""Tests for system calibrator."""

import pytest
from pcet_engine.core.system_calibrator import (
    SystemCalibrator, EmpiricalCalibration,
)

SLO1_KNOWN = [
    {"name": "WT",    "d_DA": 2.77, "k_H_exp": 297.0},
    {"name": "L546A", "d_DA": 2.88, "k_H_exp": 8.2},
    {"name": "L754A", "d_DA": 2.95, "k_H_exp": 3.0},
]

SLO1_ALL = SLO1_KNOWN + [
    {"name": "DM", "d_DA": 3.10, "k_H_exp": 0.3},
]


@pytest.fixture
def cal():
    return SystemCalibrator(temperature=303.0)


class TestEmpiricalCalibration:
    def test_returns_empirical_calibration(self, cal):
        result = cal.calibrate_empirical(SLO1_KNOWN)
        assert isinstance(result, EmpiricalCalibration)

    def test_r_squared_high(self, cal):
        result = cal.calibrate_empirical(SLO1_KNOWN)
        assert result.r_squared > 0.90

    def test_slope_negative(self, cal):
        """Rate should decrease with distance → slope < 0."""
        result = cal.calibrate_empirical(SLO1_KNOWN)
        assert result.slope < 0

    def test_predict_rate_at_d0(self, cal):
        result = cal.calibrate_empirical(SLO1_KNOWN)
        k_d0 = result.predict_rate(result.d0)
        # Should be close to WT rate (297)
        assert 50 < k_d0 < 1000

    def test_predict_rate_decreases(self, cal):
        result = cal.calibrate_empirical(SLO1_KNOWN)
        k_close = result.predict_rate(2.77)
        k_far = result.predict_rate(3.10)
        assert k_close > k_far

    def test_ranking_correct(self, cal):
        result = cal.calibrate_empirical(SLO1_KNOWN)
        preds = result.predict_variants([
            {"name": "WT",    "d_DA": 2.77},
            {"name": "L546A", "d_DA": 2.88},
            {"name": "L754A", "d_DA": 2.95},
            {"name": "DM",    "d_DA": 3.10},
        ])
        names = [p["name"] for p in preds]
        assert names == ["WT", "L546A", "L754A", "DM"]

    def test_held_out_dm_within_order_of_magnitude(self, cal):
        """DM predicted from 3 variants should be within 10x of experimental."""
        result = cal.calibrate_empirical(SLO1_KNOWN)
        k_dm = result.predict_rate(3.10)
        assert 0.03 < k_dm < 3.0  # within ~10x of 0.3

    def test_four_variant_fit(self, cal):
        result = cal.calibrate_empirical(SLO1_ALL)
        assert result.r_squared > 0.90
        assert result.n_variants == 4

    def test_two_variants_minimum(self, cal):
        result = cal.calibrate_empirical(SLO1_KNOWN[:2])
        assert result.r_squared == pytest.approx(1.0)  # perfect fit with 2 points

    def test_one_variant_raises(self, cal):
        with pytest.raises(ValueError):
            cal.calibrate_empirical(SLO1_KNOWN[:1])

    def test_summary_string(self, cal):
        result = cal.calibrate_empirical(SLO1_KNOWN)
        s = result.summary()
        assert "R²" in s
        assert "WT" in s

    def test_to_dict(self, cal):
        result = cal.calibrate_empirical(SLO1_KNOWN)
        d = result.to_dict()
        assert "slope" in d
        assert "r_squared" in d
        assert d["r_squared"] > 0.90

    def test_rate_halving_distance(self, cal):
        result = cal.calibrate_empirical(SLO1_KNOWN)
        d = result.to_dict()
        # Rate halving distance should be small (0.02-0.05 Å)
        assert 0.01 < d["rate_halving_distance"] < 0.1
