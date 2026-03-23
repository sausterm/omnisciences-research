"""Tests for solar forcing analysis on SPD manifold."""
import numpy as np
import pytest

from omni_toolkit.applications.solar_forcing import (
    SolarDataLoader,
    SolarClimateAnalyzer,
    SolarClimateResults,
    forcing_response_geodesic,
    SOLAR_REGISTRY,
    SOLAR_INDICES,
    INDICES_16,
    SOLAR_CYCLES,
    run_solar_demo,
)
from omni_toolkit.applications.climate_analysis import ClimateData


class TestSolarRegistry:
    """Test solar index registry and constants."""

    def test_registry_has_4_indices(self):
        assert len(SOLAR_REGISTRY) == 4

    def test_indices_16_length(self):
        assert len(INDICES_16) == 16

    def test_solar_indices_in_16(self):
        for idx in SOLAR_INDICES:
            assert idx in INDICES_16

    def test_registry_fields(self):
        for name, info in SOLAR_REGISTRY.items():
            assert "name" in info
            assert "group" in info
            assert info["group"] == "solar"
            assert "source" in info
            assert "unit" in info

    def test_solar_cycles_ordered(self):
        for i in range(len(SOLAR_CYCLES) - 1):
            assert SOLAR_CYCLES[i]["cycle"] < SOLAR_CYCLES[i + 1]["cycle"]


class TestSolarDataLoader:
    """Test synthetic solar+climate data generation."""

    def test_generate_synthetic_shape(self):
        data = SolarDataLoader.generate_synthetic(T=240, seed=42)
        assert data.values.shape == (240, 16)
        assert len(data.dates) == 240
        assert len(data.index_names) == 16

    def test_synthetic_has_solar_indices(self):
        data = SolarDataLoader.generate_synthetic(T=120)
        for idx in SOLAR_INDICES:
            assert idx in data.index_names

    def test_synthetic_solar_values_reasonable(self):
        data = SolarDataLoader.generate_synthetic(T=816, seed=42)
        # TSI anomaly should be ~[-2, 2] range
        tsi_col = data.index_names.index("tsi")
        tsi = data.values[:, tsi_col]
        assert abs(tsi.mean()) < 1.0
        assert tsi.std() > 0.2

        # Sunspot should have visible cycle
        ssn_col = data.index_names.index("sunspot")
        ssn = data.values[:, ssn_col]
        assert ssn.std() > 0.3

    def test_synthetic_has_11yr_cycle(self):
        """Verify TSI has ~11-year periodicity."""
        data = SolarDataLoader.generate_synthetic(T=816, seed=42)
        tsi_col = data.index_names.index("tsi")
        tsi = data.values[:, tsi_col]

        # FFT to find dominant period
        fft = np.abs(np.fft.rfft(tsi - tsi.mean()))
        freqs = np.fft.rfftfreq(len(tsi), d=1)  # monthly
        # Exclude DC
        fft[0] = 0
        peak_idx = np.argmax(fft)
        peak_period_months = 1 / freqs[peak_idx] if freqs[peak_idx] > 0 else 0
        peak_period_years = peak_period_months / 12

        # Should be near 11 years (10-12 range acceptable)
        assert 9.0 < peak_period_years < 13.0

    def test_synthetic_metadata(self):
        data = SolarDataLoader.generate_synthetic()
        assert data.metadata["synthetic"] is True
        assert data.metadata["solar_coupling"] is True

    def test_from_arrays(self):
        from omni_toolkit.applications.climate_analysis import ClimateDataLoader
        climate = ClimateDataLoader.generate_synthetic(T=120, d=12)
        solar = np.random.randn(120, 4)
        merged = SolarDataLoader.from_arrays(climate, solar)
        assert merged.d == 16
        assert merged.T == 120

    def test_from_arrays_length_mismatch(self):
        from omni_toolkit.applications.climate_analysis import ClimateDataLoader
        climate = ClimateDataLoader.generate_synthetic(T=120, d=12)
        solar = np.random.randn(100, 4)  # wrong length
        with pytest.raises(AssertionError):
            SolarDataLoader.from_arrays(climate, solar)


class TestSolarClimateAnalyzer:
    """Test the main solar-climate analysis."""

    @pytest.fixture
    def data(self):
        return SolarDataLoader.generate_synthetic(T=480, seed=42)

    @pytest.fixture
    def analyzer(self):
        return SolarClimateAnalyzer(window=48, threshold=2.0, shrinkage=0.15)

    def test_analyse_returns_results(self, data, analyzer):
        results = analyzer.analyse(data)
        assert isinstance(results, SolarClimateResults)

    def test_full_results_16d(self, data, analyzer):
        results = analyzer.analyse(data)
        assert len(results.full_results.index_names) == 16

    def test_climate_only_results_12d(self, data, analyzer):
        results = analyzer.analyse(data)
        assert len(results.climate_only_results.index_names) == 12

    def test_solar_geodesic_positive(self, data, analyzer):
        results = analyzer.analyse(data)
        assert results.solar_geodesic > 0

    def test_v_plus_v_minus_positive(self, data, analyzer):
        results = analyzer.analyse(data)
        assert results.solar_v_plus >= 0
        assert results.solar_v_minus >= 0

    def test_solar_max_min_covs_spd(self, data, analyzer):
        results = analyzer.analyse(data)
        # Both covariance matrices should be SPD
        eigvals_max = np.linalg.eigvalsh(results.solar_max_cov)
        eigvals_min = np.linalg.eigvalsh(results.solar_min_cov)
        assert eigvals_max[0] > 0
        assert eigvals_min[0] > 0

    def test_cycle_stats(self, data, analyzer):
        results = analyzer.analyse(data)
        assert len(results.cycle_stats) > 0
        for cs in results.cycle_stats:
            assert "cycle" in cs
            assert "transitions_near_max" in cs
            assert "transitions_near_min" in cs

    def test_no_solar_indices_raises(self, analyzer):
        from omni_toolkit.applications.climate_analysis import ClimateDataLoader
        climate_only = ClimateDataLoader.generate_synthetic(T=240, d=12)
        with pytest.raises(ValueError, match="No solar indices"):
            analyzer.analyse(climate_only)


class TestForcingResponseGeodesic:
    """Test the forcing → response geodesic computation."""

    @pytest.fixture
    def data(self):
        return SolarDataLoader.generate_synthetic(T=480, seed=42)

    def test_basic_computation(self, data):
        climate_names = [n for n in data.index_names if n not in SOLAR_INDICES]
        result = forcing_response_geodesic(
            data, ["tsi"], climate_names, lag_months=0
        )
        assert "geodesic_distance" in result
        assert result["geodesic_distance"] >= 0

    def test_v_plus_v_minus(self, data):
        climate_names = [n for n in data.index_names if n not in SOLAR_INDICES]
        result = forcing_response_geodesic(data, ["tsi"], climate_names)
        assert result["v_plus"] >= 0
        assert result["v_minus"] >= 0

    def test_v_ratio_bounded(self, data):
        climate_names = [n for n in data.index_names if n not in SOLAR_INDICES]
        result = forcing_response_geodesic(data, ["tsi"], climate_names)
        assert 0 <= result["v_ratio"] <= 1

    def test_with_lag(self, data):
        climate_names = [n for n in data.index_names if n not in SOLAR_INDICES]
        r0 = forcing_response_geodesic(data, ["tsi"], climate_names, lag_months=0)
        r6 = forcing_response_geodesic(data, ["tsi"], climate_names, lag_months=6)
        # Both should produce valid results
        assert r0["geodesic_distance"] >= 0
        assert r6["geodesic_distance"] >= 0

    def test_multiple_forcings(self, data):
        climate_names = [n for n in data.index_names if n not in SOLAR_INDICES]
        result = forcing_response_geodesic(
            data, ["tsi", "f107"], climate_names
        )
        assert result["geodesic_distance"] >= 0

    def test_interpretation_present(self, data):
        climate_names = [n for n in data.index_names if n not in SOLAR_INDICES]
        result = forcing_response_geodesic(data, ["tsi"], climate_names)
        assert "interpretation" in result
        assert len(result["interpretation"]) > 0

    def test_partition_sizes(self, data):
        climate_names = [n for n in data.index_names if n not in SOLAR_INDICES]
        result = forcing_response_geodesic(data, ["tsi"], climate_names)
        assert result["n_high"] > 0
        assert result["n_low"] > 0


class TestSolarDemo:
    """Test the demo runner."""

    def test_demo_runs(self):
        results = run_solar_demo(verbose=False)
        assert isinstance(results, SolarClimateResults)
        assert results.solar_geodesic > 0
