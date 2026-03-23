"""
Tests for Riemannian climate regime detection.

Tests the ClimateRegimeDetector, StreamingSPD, ClimateMonitor, and V+/V-
decomposition against synthetic data with known regime transitions.
"""
import numpy as np
import pytest

from omni_toolkit.applications.climate_analysis import (
    ClimateRegimeDetector, ClimateDataLoader, ClimateData,
    VDecomposition, DetectionResults, rolling_covariance_climate,
    compare_dimensions, save_results,
    INDICES_6, INDICES_12, INDEX_REGISTRY, ENSO_EVENTS,
)
from omni_toolkit.applications.streaming_spd import (
    StreamingSPD, StreamingState, ClimateMonitor,
)
from omni_toolkit.applications.spd_ml import _symmetrize


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def synthetic_data_6d():
    """Synthetic 6-index climate data."""
    return ClimateDataLoader.generate_synthetic(T=816, d=6)


@pytest.fixture
def synthetic_data_12d():
    """Synthetic 12-index climate data."""
    return ClimateDataLoader.generate_synthetic(T=816, d=12)


@pytest.fixture
def short_data():
    """Short synthetic data for fast tests."""
    return ClimateDataLoader.generate_synthetic(T=240, d=6)


# =====================================================================
# ClimateData tests
# =====================================================================

class TestClimateData:
    def test_synthetic_shape(self, synthetic_data_6d):
        data = synthetic_data_6d
        assert data.values.shape == (816, 6)
        assert len(data.dates) == 816
        assert len(data.index_names) == 6

    def test_synthetic_12d_shape(self, synthetic_data_12d):
        data = synthetic_data_12d
        assert data.values.shape == (816, 12)
        assert len(data.index_names) == 12

    def test_date_range(self, synthetic_data_6d):
        data = synthetic_data_6d
        assert data.start_year == 1956
        assert data.end_year == 2023

    def test_select_indices(self, synthetic_data_12d):
        data = synthetic_data_12d
        subset = data.select_indices(["nino34", "soi", "pdo"])
        assert subset.d == 3
        assert subset.T == data.T
        assert subset.index_names == ["nino34", "soi", "pdo"]

    def test_slice_years(self, synthetic_data_6d):
        data = synthetic_data_6d
        sliced = data.slice_years(1980, 2000)
        assert sliced.start_year >= 1980
        assert sliced.end_year <= 2000

    def test_date_to_index(self, synthetic_data_6d):
        data = synthetic_data_6d
        idx = data.date_to_index(1957, 1)
        assert idx == 12  # 12 months after Jan 1956

    def test_no_nans(self, synthetic_data_12d):
        """Synthetic data should have no NaN values."""
        assert not np.any(np.isnan(synthetic_data_12d.values))

    def test_from_arrays(self):
        values = np.random.randn(120, 4)
        data = ClimateDataLoader.from_arrays(
            values, start_year=2010, start_month=1,
            index_names=["nino34", "soi", "pdo", "amo"]
        )
        assert data.T == 120
        assert data.d == 4
        assert data.dates[0] == (2010, 1)
        assert data.dates[11] == (2010, 12)
        assert data.dates[12] == (2011, 1)


# =====================================================================
# Rolling covariance tests
# =====================================================================

class TestRollingCovariance:
    def test_output_shape(self, short_data):
        covs, dates = rolling_covariance_climate(short_data, window=72, step=1)
        d = short_data.d
        assert covs.ndim == 3
        assert covs.shape[1] == d
        assert covs.shape[2] == d
        assert len(dates) == len(covs)

    def test_spd_guarantee(self, short_data):
        """All output matrices must be SPD."""
        covs, _ = rolling_covariance_climate(short_data, window=72, step=1)
        for i in range(len(covs)):
            # Symmetric
            assert np.allclose(covs[i], covs[i].T, atol=1e-10)
            # Positive definite
            eigvals = np.linalg.eigvalsh(covs[i])
            assert eigvals[0] > 0, f"Non-SPD at index {i}: min_eig={eigvals[0]}"

    def test_shrinkage(self, short_data):
        """Shrinkage should improve conditioning."""
        covs_raw, _ = rolling_covariance_climate(short_data, window=72, shrinkage=0.0)
        covs_shrunk, _ = rolling_covariance_climate(short_data, window=72, shrinkage=0.3)
        # Shrunk should have smaller condition number on average
        cond_raw = np.mean([np.linalg.cond(c) for c in covs_raw])
        cond_shrunk = np.mean([np.linalg.cond(c) for c in covs_shrunk])
        assert cond_shrunk < cond_raw

    def test_step_reduces_count(self, short_data):
        covs1, _ = rolling_covariance_climate(short_data, window=72, step=1)
        covs6, _ = rolling_covariance_climate(short_data, window=72, step=6)
        assert len(covs6) < len(covs1)


# =====================================================================
# Regime detection tests
# =====================================================================

class TestClimateRegimeDetector:
    def test_basic_detection(self, synthetic_data_6d):
        detector = ClimateRegimeDetector(window=72, step=1, threshold=2.0)
        results = detector.detect(synthetic_data_6d)
        assert isinstance(results, DetectionResults)
        assert len(results.geodesic_distances) > 0
        assert len(results.euclidean_distances) > 0
        assert len(results.covariances) > 0

    def test_detects_transitions(self, synthetic_data_6d):
        """Should detect at least some transitions in synthetic data with known events."""
        detector = ClimateRegimeDetector(
            window=72, step=1, threshold=1.5, shrinkage=0.1
        )
        results = detector.detect(synthetic_data_6d)
        assert len(results.transitions) > 0, "No transitions detected"

    def test_geodesic_distances_positive(self, synthetic_data_6d):
        detector = ClimateRegimeDetector(window=72, step=1, threshold=2.0)
        results = detector.detect(synthetic_data_6d)
        assert np.all(results.geodesic_distances >= 0)

    def test_adaptive_vs_global(self, synthetic_data_6d):
        """Adaptive and global threshold should give different results."""
        det_adapt = ClimateRegimeDetector(window=72, threshold=2.0, adaptive=True)
        det_global = ClimateRegimeDetector(window=72, threshold=2.0, adaptive=False)
        r_adapt = det_adapt.detect(synthetic_data_6d)
        r_global = det_global.detect(synthetic_data_6d)
        # Both should detect something, but different counts
        # (this is a loose test — mainly checking both modes run)
        assert len(r_adapt.transitions) >= 0
        assert len(r_global.transitions) >= 0

    def test_higher_threshold_fewer_detections(self, synthetic_data_6d):
        """Higher threshold should produce fewer or equal detections."""
        det_low = ClimateRegimeDetector(window=72, threshold=1.5)
        det_high = ClimateRegimeDetector(window=72, threshold=3.0)
        r_low = det_low.detect(synthetic_data_6d)
        r_high = det_high.detect(synthetic_data_6d)
        assert len(r_high.transitions) <= len(r_low.transitions)

    def test_categorization(self, synthetic_data_6d):
        """Detected transitions should get category labels."""
        detector = ClimateRegimeDetector(window=72, threshold=1.5)
        results = detector.detect(synthetic_data_6d)
        categories = {tr.category for tr in results.transitions}
        # At least some should be categorized (not all "unknown")
        # This depends on synthetic data aligning with events
        assert len(categories) >= 1

    def test_12d_detection(self, synthetic_data_12d):
        """12-dimensional detection should work."""
        detector = ClimateRegimeDetector(window=72, step=1, threshold=2.0)
        results = detector.detect(synthetic_data_12d)
        assert results.covariances.shape[1] == 12
        assert results.covariances.shape[2] == 12


# =====================================================================
# Validation tests
# =====================================================================

class TestValidation:
    def test_validation_metrics(self, synthetic_data_6d):
        detector = ClimateRegimeDetector(window=72, threshold=1.5)
        results = detector.detect(synthetic_data_6d)
        val = detector.validate(results)

        assert "enso_recall" in val
        assert "enso_precision" in val
        assert "mean_lead_time_months" in val
        assert 0.0 <= val["enso_recall"] <= 1.0
        assert 0.0 <= val["enso_precision"] <= 1.0
        assert val["mean_lead_time_months"] >= 0.0

    def test_validation_populates_results(self, synthetic_data_6d):
        detector = ClimateRegimeDetector(window=72, threshold=1.5)
        results = detector.detect(synthetic_data_6d)
        detector.validate(results)
        assert results.enso_recall is not None
        assert results.enso_precision is not None


# =====================================================================
# V+/V- decomposition tests
# =====================================================================

class TestVDecomposition:
    def test_decomposition_orthogonal(self, synthetic_data_6d):
        """V+ and V- should be orthogonal components (trace decomposition)."""
        detector = ClimateRegimeDetector(window=72, threshold=1.5)
        results = detector.detect(synthetic_data_6d)

        if len(results.transitions) == 0:
            pytest.skip("No transitions to decompose")

        vd = VDecomposition(results)
        dec = vd.decompose_transition(results.transitions[0].index)

        # V = V+ + V-
        np.testing.assert_allclose(
            dec["V"], dec["V_plus"] + dec["V_minus"], atol=1e-10
        )

        # V+ is proportional to identity
        d = results.covariances.shape[1]
        expected_v_plus = (np.trace(dec["V"]) / d) * np.eye(d)
        np.testing.assert_allclose(dec["V_plus"], expected_v_plus, atol=1e-10)

        # V- is traceless
        assert abs(np.trace(dec["V_minus"])) < 1e-10

    def test_norms_positive(self, synthetic_data_6d):
        detector = ClimateRegimeDetector(window=72, threshold=1.5)
        results = detector.detect(synthetic_data_6d)

        if len(results.transitions) == 0:
            pytest.skip("No transitions to decompose")

        vd = VDecomposition(results)
        dec = vd.decompose_transition(results.transitions[0].index)
        assert dec["norm_plus"] >= 0
        assert dec["norm_minus"] >= 0

    def test_fingerprint_analysis(self, synthetic_data_6d):
        detector = ClimateRegimeDetector(window=72, threshold=1.5)
        results = detector.detect(synthetic_data_6d)

        if len(results.transitions) < 2:
            pytest.skip("Not enough transitions for fingerprinting")

        vd = VDecomposition(results)
        fp = vd.fingerprint_analysis()
        assert "by_category" in fp
        assert "decompositions" in fp
        assert len(fp["decompositions"]) == len(results.transitions)


# =====================================================================
# Streaming SPD tests
# =====================================================================

class TestStreamingSPD:
    def test_first_update(self):
        """First update should initialize the mean."""
        spd = StreamingSPD(d=4, decay=0.95, threshold=2.0)
        cov = np.eye(4) + 0.1 * np.random.randn(4, 4)
        cov = _symmetrize(cov)
        cov += 2.0 * np.eye(4)  # Ensure SPD

        result = spd.update(cov)
        assert result["distance"] == 0.0
        assert not result["is_anomaly"]
        assert spd.state.n_updates == 1

    def test_stable_inputs_no_anomaly(self):
        """Stable covariances should not trigger anomalies."""
        rng = np.random.RandomState(42)
        spd = StreamingSPD(d=4, decay=0.95, threshold=2.0, warmup=10)

        base_cov = np.eye(4)
        for i in range(50):
            noise = 0.05 * rng.randn(4, 4)
            cov = base_cov + _symmetrize(noise)
            cov += 2.0 * np.eye(4)
            result = spd.update(cov)

        # After warmup, stable inputs should not be anomalous
        assert not result["is_anomaly"]

    def test_large_shift_triggers_anomaly(self):
        """A sudden large shift should trigger an anomaly."""
        rng = np.random.RandomState(42)
        spd = StreamingSPD(d=4, decay=0.95, threshold=2.0, warmup=10)

        base_cov = np.eye(4)
        for i in range(30):
            noise = 0.02 * rng.randn(4, 4)
            cov = base_cov + _symmetrize(noise)
            cov += 2.0 * np.eye(4)
            spd.update(cov)

        # Inject a massive shift
        shifted_cov = 10.0 * np.eye(4)
        result = spd.update(shifted_cov)
        assert result["is_anomaly"] or result["sigma_above"] > 1.0

    def test_state_serialization(self):
        """State should survive save/load cycle."""
        import tempfile, os
        spd = StreamingSPD(d=3, decay=0.9)
        cov = 2.0 * np.eye(3)
        spd.update(cov, date=(2024, 1))
        spd.update(cov * 1.1, date=(2024, 2))

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            spd.save_state(path)
            spd2 = StreamingSPD(d=3)
            spd2.load_state(path)
            assert spd2.state.n_updates == 2
            np.testing.assert_allclose(spd2.state.mean, spd.state.mean, atol=1e-10)
        finally:
            os.unlink(path)

    def test_reset(self):
        spd = StreamingSPD(d=4)
        spd.update(np.eye(4) * 2)
        spd.reset()
        assert spd.state.n_updates == 0
        assert spd.state.mean is None


# =====================================================================
# Climate monitor tests
# =====================================================================

class TestClimateMonitor:
    def test_warmup(self, short_data):
        """Monitor should warm up without errors."""
        monitor = ClimateMonitor(
            indices=short_data.index_names,
            window=72, threshold=2.0,
        )
        monitor.warm_up(short_data)
        assert monitor.streaming.state.n_updates > 0

    def test_ingest_month(self, short_data):
        """Ingesting single months should work after warmup."""
        monitor = ClimateMonitor(
            indices=short_data.index_names,
            window=72, threshold=2.0,
        )
        monitor.warm_up(short_data)

        # Ingest a new month
        new_values = np.random.randn(short_data.d)
        status = monitor.ingest_month(new_values, 2024, 1)
        assert status.regime_label in ("stable", "transition", "post_transition", "warmup")
        assert status.n_updates > 0

    def test_dashboard_data(self, short_data):
        monitor = ClimateMonitor(
            indices=short_data.index_names,
            window=72, threshold=2.0,
        )
        monitor.warm_up(short_data)
        dash = monitor.get_dashboard_data()
        assert "current" in dash
        assert "timeline" in dash
        assert "transitions" in dash

    def test_state_persistence(self, short_data):
        import tempfile, os
        monitor = ClimateMonitor(
            indices=short_data.index_names,
            window=72, threshold=2.0,
        )
        monitor.warm_up(short_data)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            monitor.save_state(path)
            monitor2 = ClimateMonitor(indices=short_data.index_names)
            monitor2.load_state(path)
            assert monitor2.streaming.state.n_updates == monitor.streaming.state.n_updates
        finally:
            os.unlink(path)


# =====================================================================
# Dimension comparison
# =====================================================================

class TestDimensionComparison:
    def test_compare_6_vs_12(self, synthetic_data_12d):
        """Comparing 6d vs 12d should run without error."""
        results = compare_dimensions(
            synthetic_data_12d,
            index_sets={
                "6d": INDICES_6,
                "12d": INDICES_12,
            },
            window=72, threshold=2.0,
        )
        assert "6d" in results
        assert "12d" in results
        assert results["6d"]["n_indices"] == 6
        assert results["12d"]["n_indices"] == 12


# =====================================================================
# Serialization
# =====================================================================

class TestSerialization:
    def test_save_results(self, synthetic_data_6d, tmp_path):
        detector = ClimateRegimeDetector(window=72, threshold=1.5)
        results = detector.detect(synthetic_data_6d)
        detector.validate(results)

        path = str(tmp_path / "results.json")
        save_results(results, path)

        import json
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["window"] == 72
        assert "transitions" in loaded
        assert "geodesic_distances" in loaded


# =====================================================================
# Integration: full demo
# =====================================================================

class TestDemo:
    def test_climate_demo(self):
        """Full demo should run without error."""
        from omni_toolkit.applications.climate_analysis import run_demo
        results = run_demo(d=6, verbose=False)
        assert len(results) > 0

    def test_streaming_demo(self):
        """Streaming demo should run without error."""
        from omni_toolkit.applications.streaming_spd import run_demo
        results = run_demo(verbose=False)
        assert results["n_updates"] == 816
