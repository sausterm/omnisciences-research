"""Tests for volume-based enzyme variant screening."""

import math
import pytest
from pcet_engine.core.volume_calibrator import VolumeCalibrator, VolumeCalibrationResult


SLO1 = [
    {"name": "WT",    "mutation": None,    "k_exp": 297.0},
    {"name": "L546A", "mutation": "L546A", "k_exp": 8.2},
    {"name": "L754A", "mutation": "L754A", "k_exp": 3.0},
    {"name": "DM",    "mutation": "L546A/L754A", "k_exp": 0.3},
]

DHFR = [
    {"name": "WT",   "mutation": None,   "k_exp": 950.0},
    {"name": "I14V", "mutation": "I14V", "k_exp": 135.0},
    {"name": "I14A", "mutation": "I14A", "k_exp": 30.0},
    {"name": "I14G", "mutation": "I14G", "k_exp": 0.95},
]


@pytest.fixture
def cal():
    return VolumeCalibrator()


class TestCalibration:
    def test_slo1_r_squared(self, cal):
        result = cal.calibrate(SLO1)
        assert result.r_squared > 0.90

    def test_dhfr_r_squared(self, cal):
        result = cal.calibrate(DHFR)
        assert result.r_squared > 0.85

    def test_positive_slope_for_volume_reducing_mutations(self, cal):
        """For SLO-1/DHFR, removing volume (ΔV<0) reduces rate → slope > 0."""
        result = cal.calibrate(SLO1)
        assert result.slope > 0

    def test_returns_calibration_result(self, cal):
        result = cal.calibrate(SLO1)
        assert isinstance(result, VolumeCalibrationResult)

    def test_wt_rate_prediction(self, cal):
        result = cal.calibrate(SLO1)
        k_wt = result.predict_rate(None)
        assert 50 < k_wt < 1000  # should be near 297

    def test_two_variants_minimum(self, cal):
        result = cal.calibrate(SLO1[:2])
        assert result.r_squared == pytest.approx(1.0)

    def test_one_variant_raises(self, cal):
        with pytest.raises(ValueError):
            cal.calibrate(SLO1[:1])


class TestScreening:
    def test_screen_returns_ranked_list(self, cal):
        result = cal.calibrate(SLO1[:3])
        preds = result.screen(["L546V", "L546W", "L546G"])
        assert len(preds) == 3
        assert all("rank" in p for p in preds)
        assert preds[0]["rank"] == 1

    def test_screen_ranking_order(self, cal):
        """Trp (biggest) should be fastest, Gly (smallest) slowest."""
        result = cal.calibrate(SLO1[:3])
        preds = result.screen(["L546W", "L546A", "L546G"])
        names = [p["name"] for p in preds]
        assert names[0] == "L546W"   # biggest residue → fastest
        assert names[-1] == "L546G"  # smallest residue → slowest

    def test_wt_in_screen(self, cal):
        result = cal.calibrate(SLO1[:3])
        preds = result.screen(["WT", "L546A"])
        wt = next(p for p in preds if p["name"] == "WT")
        assert wt["rate_ratio"] == pytest.approx(1.0, rel=0.01)

    def test_double_mutant(self, cal):
        result = cal.calibrate(SLO1[:3])
        preds = result.screen(["L546A/L754A"])
        assert len(preds) == 1
        assert preds[0]["k_pred"] < result.predict_rate(None)  # DM slower than WT

    def test_held_out_dm_prediction(self, cal):
        """Calibrate on WT + 2 singles, predict DM."""
        result = cal.calibrate(SLO1[:3])  # WT, L546A, L754A
        preds = result.screen(["L546A/L754A"])
        k_dm_pred = preds[0]["k_pred"]
        # Should be within 1 order of magnitude of 0.3
        assert 0.01 < k_dm_pred < 10.0


class TestSaturationScan:
    def test_returns_19_mutations(self, cal):
        result = cal.calibrate(SLO1[:3])
        scan = result.saturation_scan(546, "L")
        assert len(scan) == 19

    def test_sorted_by_rate(self, cal):
        result = cal.calibrate(SLO1[:3])
        scan = result.saturation_scan(546, "L")
        rates = [s["k_pred"] for s in scan]
        assert rates == sorted(rates, reverse=True)

    def test_trp_is_fastest(self, cal):
        result = cal.calibrate(SLO1[:3])
        scan = result.saturation_scan(546, "L")
        assert scan[0]["name"] == "L546W"

    def test_gly_is_slowest(self, cal):
        result = cal.calibrate(SLO1[:3])
        scan = result.saturation_scan(546, "L")
        assert scan[-1]["name"] == "L546G"


class TestCrossSystem:
    def test_dhfr_screening(self, cal):
        """DHFR should also rank correctly via ΔV."""
        result = cal.calibrate(DHFR[:3])  # WT, I14V, I14A
        preds = result.screen(["I14G"])
        # I14G (Gly, smallest) should be predicted slower than WT
        assert preds[0]["rate_ratio"] < 0.1

    def test_different_slopes(self, cal):
        """SLO-1 and DHFR should have different slopes."""
        slo_result = cal.calibrate(SLO1)
        dhfr_result = cal.calibrate(DHFR)
        # Both positive (volume reduction → rate reduction)
        assert slo_result.slope > 0
        assert dhfr_result.slope > 0
        # But different magnitudes
        assert abs(slo_result.slope - dhfr_result.slope) > 0.001


class TestOutput:
    def test_summary_string(self, cal):
        result = cal.calibrate(SLO1)
        s = result.summary()
        assert "R²" in s
        assert "WT" in s

    def test_to_dict(self, cal):
        result = cal.calibrate(SLO1)
        d = result.to_dict()
        assert "slope" in d
        assert "r_squared" in d
        assert "wt_rate" in d
