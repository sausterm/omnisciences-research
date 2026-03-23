"""Tests for climate forcing attribution on SPD manifold."""
import numpy as np
import pytest

from omni_toolkit.applications.climate_attribution import (
    ClimateAttribution,
    AttributionResults,
    ForcingFingerprint,
    TangentSpaceRegression,
    TangentRegressionResults,
    RegressionCoefficient,
    generate_anthropogenic_series,
    generate_attribution_data,
    run_attribution_demo,
    run_tangent_regression_demo,
    _sym_to_vec,
    _vec_to_sym,
    phase_randomise,
    ANTHROPOGENIC_INDICES,
    ANTHRO_REGISTRY,
)
from omni_toolkit.applications.solar_forcing import SOLAR_INDICES


class TestAnthroRegistry:
    """Test anthropogenic forcing registry and constants."""

    def test_registry_has_3_indices(self):
        assert len(ANTHRO_REGISTRY) == 3

    def test_anthropogenic_indices_in_registry(self):
        for idx in ANTHROPOGENIC_INDICES:
            assert idx in ANTHRO_REGISTRY

    def test_registry_fields(self):
        for name, info in ANTHRO_REGISTRY.items():
            assert "name" in info
            assert "group" in info
            assert "source" in info
            assert "unit" in info


class TestAnthropogenicSeries:
    """Test synthetic anthropogenic forcing generation."""

    def test_series_lengths(self):
        series = generate_anthropogenic_series(T=240)
        for key in ANTHROPOGENIC_INDICES:
            assert len(series[key]) == 240

    def test_co2_increasing(self):
        series = generate_anthropogenic_series(T=816, seed=42)
        co2 = series["co2"]
        # CO2 should trend upward: late mean > early mean
        assert co2[600:].mean() > co2[:200].mean()

    def test_aod_has_pinatubo_spike(self):
        series = generate_anthropogenic_series(T=816, start_year=1956, seed=42)
        aod = series["aod"]
        # Pinatubo is 1991 → index ~(1991-1956)*12 = 420
        pinatubo_idx = (1991 - 1956) * 12 + 5  # June 1991
        if pinatubo_idx < len(aod):
            # Should have elevated AOD near Pinatubo
            assert aod[pinatubo_idx:pinatubo_idx + 12].max() > aod[:100].mean() + 1.0

    def test_ch4_plateau(self):
        series = generate_anthropogenic_series(T=816, start_year=1956, seed=42)
        ch4 = series["ch4"]
        # CH4 should have a plateau period (1998-2007)
        idx_1998 = (1998 - 1956) * 12
        idx_2007 = (2007 - 1956) * 12
        if idx_2007 < len(ch4):
            plateau = ch4[idx_1998:idx_2007]
            assert plateau.std() < ch4[:idx_1998].std() * 2  # relatively flat


class TestAttributionData:
    """Test combined attribution dataset generation."""

    def test_shape(self):
        data = generate_attribution_data(T=240)
        # 12 climate + 4 solar + 3 anthropogenic = 19
        assert data.values.shape[0] == 240
        assert data.d >= 17  # at least 12+4+1

    def test_has_all_indices(self):
        data = generate_attribution_data(T=240)
        for idx in ANTHROPOGENIC_INDICES:
            assert idx in data.index_names

    def test_has_solar_indices(self):
        data = generate_attribution_data(T=240)
        for idx in SOLAR_INDICES:
            assert idx in data.index_names


class TestForcingFingerprint:
    """Test individual forcing fingerprint computation."""

    @pytest.fixture
    def data(self):
        return generate_attribution_data(T=480, seed=42)

    @pytest.fixture
    def engine(self):
        return ClimateAttribution(window=60, shrinkage=0.15, n_permutations=10)

    def test_solar_fingerprint(self, data, engine):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        fp = engine.compute_fingerprint(
            data, ["sunspot", "f107"], response, label="solar"
        )
        assert isinstance(fp, ForcingFingerprint)
        assert fp.geodesic_distance >= 0
        assert fp.name == "solar"

    def test_anthropogenic_fingerprint(self, data, engine):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        fp = engine.compute_fingerprint(
            data, ["co2"], response, label="anthropogenic"
        )
        assert fp.geodesic_distance >= 0
        assert fp.v_plus >= 0
        assert fp.v_minus >= 0

    def test_volcanic_fingerprint(self, data, engine):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        fp = engine.compute_fingerprint(
            data, ["aod"], response, label="volcanic"
        )
        assert fp.geodesic_distance >= 0

    def test_v_ratio_bounded(self, data, engine):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        fp = engine.compute_fingerprint(
            data, ["co2"], response, label="co2"
        )
        assert 0 <= fp.v_ratio <= 1

    def test_tangent_vector_symmetric(self, data, engine):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        fp = engine.compute_fingerprint(
            data, ["co2"], response, label="co2"
        )
        d = fp.tangent_vector.shape[0]
        assert fp.tangent_vector.shape == (d, d)
        np.testing.assert_allclose(
            fp.tangent_vector, fp.tangent_vector.T, atol=1e-10
        )

    def test_with_lag(self, data, engine):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        fp0 = engine.compute_fingerprint(
            data, ["co2"], response, lag_months=0
        )
        fp12 = engine.compute_fingerprint(
            data, ["co2"], response, lag_months=12
        )
        assert fp0.geodesic_distance >= 0
        assert fp12.geodesic_distance >= 0


class TestObservedTrend:
    """Test observed covariance trend computation."""

    @pytest.fixture
    def data(self):
        return generate_attribution_data(T=480, seed=42)

    @pytest.fixture
    def engine(self):
        return ClimateAttribution(window=60, shrinkage=0.15, n_permutations=10)

    def test_trend_tangent_shape(self, data, engine):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        tangent, dist = engine.compute_observed_trend(data, response)
        d = len(response)
        assert tangent.shape == (d, d)
        assert dist >= 0

    def test_trend_tangent_symmetric(self, data, engine):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        tangent, _ = engine.compute_observed_trend(data, response)
        np.testing.assert_allclose(tangent, tangent.T, atol=1e-10)

    def test_with_split_year(self, data, engine):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        _, d1 = engine.compute_observed_trend(data, response, split_year=1980)
        _, d2 = engine.compute_observed_trend(data, response, split_year=1990)
        assert d1 >= 0
        assert d2 >= 0


class TestProjection:
    """Test tangent vector projection."""

    def test_identical_vectors(self):
        engine = ClimateAttribution()
        v = np.eye(5) * 0.1
        assert abs(engine.project_onto_fingerprint(v, v) - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        engine = ClimateAttribution()
        # Diagonal vs off-diagonal are not necessarily orthogonal
        # Use truly orthogonal symmetric matrices
        v1 = np.diag([1, 0, 0, 0, 0]).astype(float)
        v2 = np.diag([0, 1, 0, 0, 0]).astype(float)
        assert abs(engine.project_onto_fingerprint(v1, v2)) < 1e-10

    def test_projection_bounded(self):
        engine = ClimateAttribution()
        rng = np.random.RandomState(42)
        for _ in range(10):
            v1 = rng.randn(5, 5)
            v1 = (v1 + v1.T) / 2
            v2 = rng.randn(5, 5)
            v2 = (v2 + v2.T) / 2
            proj = engine.project_onto_fingerprint(v1, v2)
            assert -1.0 - 1e-10 <= proj <= 1.0 + 1e-10


class TestFullAttribution:
    """Test the complete attribution pipeline."""

    @pytest.fixture
    def data(self):
        return generate_attribution_data(T=480, seed=42)

    def test_attribution_runs(self, data):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        forcing_groups = {
            "solar": ["sunspot", "f107"],
            "anthropogenic": ["co2"],
        }
        engine = ClimateAttribution(window=60, shrinkage=0.15, n_permutations=10)
        results = engine.attribute(
            data, forcing_groups, response,
            test_significance=False,
        )
        assert isinstance(results, AttributionResults)

    def test_fractions_sum_to_one(self, data):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        forcing_groups = {
            "solar": ["sunspot", "f107"],
            "anthropogenic": ["co2"],
            "volcanic": ["aod"],
        }
        engine = ClimateAttribution(window=60, shrinkage=0.15, n_permutations=10)
        results = engine.attribute(
            data, forcing_groups, response,
            test_significance=False,
        )
        total = sum(results.attribution_fractions.values())
        assert abs(total - 1.0) < 1e-10

    def test_has_dominant_forcing(self, data):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        forcing_groups = {
            "solar": ["sunspot"],
            "anthropogenic": ["co2"],
        }
        engine = ClimateAttribution(window=60, shrinkage=0.15, n_permutations=10)
        results = engine.attribute(
            data, forcing_groups, response,
            test_significance=False,
        )
        assert results.dominant_forcing in forcing_groups

    def test_interpretation_populated(self, data):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        forcing_groups = {
            "solar": ["sunspot"],
            "anthropogenic": ["co2"],
        }
        engine = ClimateAttribution(window=60, shrinkage=0.15, n_permutations=10)
        results = engine.attribute(
            data, forcing_groups, response,
            test_significance=False,
        )
        assert len(results.interpretation) > 0

    def test_with_lags(self, data):
        response = [n for n in data.index_names
                     if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        forcing_groups = {
            "solar": ["sunspot"],
            "anthropogenic": ["co2"],
        }
        engine = ClimateAttribution(window=60, shrinkage=0.15, n_permutations=10)
        results = engine.attribute(
            data, forcing_groups, response,
            lag_months={"solar": 6, "anthropogenic": 12},
            test_significance=False,
        )
        assert results.fingerprints["solar"].lag_months == 6
        assert results.fingerprints["anthropogenic"].lag_months == 12


class TestVechRoundTrip:
    """Test symmetric matrix vectorisation."""

    def test_roundtrip(self):
        d = 5
        rng = np.random.RandomState(42)
        S = rng.randn(d, d)
        S = (S + S.T) / 2
        v = _sym_to_vec(S)
        assert len(v) == d * (d + 1) // 2
        S2 = _vec_to_sym(v, d)
        np.testing.assert_allclose(S, S2, atol=1e-12)

    def test_norm_preservation(self):
        """vech with √2 scaling preserves Frobenius norm."""
        d = 4
        rng = np.random.RandomState(42)
        S = rng.randn(d, d)
        S = (S + S.T) / 2
        v = _sym_to_vec(S)
        np.testing.assert_allclose(
            np.linalg.norm(S, 'fro'),
            np.linalg.norm(v),
            atol=1e-12,
        )


class TestPhaseRandomise:
    """Test phase-randomisation surrogates."""

    def test_preserves_spectrum(self):
        rng = np.random.RandomState(42)
        x = np.sin(np.linspace(0, 10 * np.pi, 200)) + rng.randn(200) * 0.3
        surr = phase_randomise(x, rng)
        # Power spectra should be identical
        psd_orig = np.abs(np.fft.rfft(x)) ** 2
        psd_surr = np.abs(np.fft.rfft(surr)) ** 2
        np.testing.assert_allclose(psd_orig, psd_surr, atol=1e-8)

    def test_destroys_phase(self):
        rng = np.random.RandomState(42)
        # Use broadband signal (many frequencies) so phase randomisation
        # actually destroys correlation — a pure sinusoid has only one
        # frequency so phase shift just translates it
        t = np.linspace(0, 20 * np.pi, 500)
        x = np.sin(t) + 0.5 * np.sin(3.7 * t) + 0.3 * np.cos(7.1 * t) + rng.randn(500) * 0.2
        surr = phase_randomise(x, rng)
        # Run multiple surrogates to get average decorrelation
        corrs = []
        for _ in range(10):
            surr = phase_randomise(x, rng)
            corrs.append(abs(np.corrcoef(x, surr)[0, 1]))
        assert np.mean(corrs) < 0.6

    def test_preserves_length(self):
        rng = np.random.RandomState(42)
        x = rng.randn(150)
        surr = phase_randomise(x, rng)
        assert len(surr) == len(x)

    def test_real_valued(self):
        rng = np.random.RandomState(42)
        x = rng.randn(200)
        surr = phase_randomise(x, rng)
        assert np.isreal(surr).all()


class TestTangentSpaceRegression:
    """Test the tangent-space regression engine."""

    @pytest.fixture
    def data(self):
        return generate_attribution_data(T=480, seed=42)

    @pytest.fixture
    def response_names(self, data):
        return [n for n in data.index_names
                if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]

    @pytest.fixture
    def forcing_groups(self, data):
        groups = {
            "solar": ["sunspot", "f107"],
            "anthropogenic": ["co2"],
        }
        return {k: [n for n in v if n in data.index_names]
                for k, v in groups.items()}

    def test_regress_returns_results(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=False,
        )
        assert isinstance(results, TangentRegressionResults)

    def test_r_squared_bounded(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=False,
        )
        assert 0 <= results.r_squared <= 1

    def test_fractions_sum_to_one(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=False,
        )
        total = sum(results.attribution_fractions.values())
        assert abs(total - 1.0) < 1e-10

    def test_coefficients_present(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=False,
        )
        for label in forcing_groups:
            assert label in results.coefficients
            c = results.coefficients[label]
            assert isinstance(c, RegressionCoefficient)
            assert c.beta_matrix.shape == (len(response_names), len(response_names))

    def test_beta_matrix_symmetric(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=False,
        )
        for label, c in results.coefficients.items():
            np.testing.assert_allclose(
                c.beta_matrix, c.beta_matrix.T, atol=1e-10,
            )

    def test_v_ratio_bounded(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=False,
        )
        for c in results.coefficients.values():
            assert 0 <= c.v_ratio <= 1

    def test_frechet_mean_spd(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=False,
        )
        eigvals = np.linalg.eigvalsh(results.frechet_mean)
        assert eigvals[0] > 0

    def test_with_lags(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            lag_months={"solar": 6, "anthropogenic": 12},
            test_surrogates=False,
        )
        assert isinstance(results, TangentRegressionResults)

    def test_surrogate_p_values(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=20,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=True,
        )
        for c in results.coefficients.values():
            assert c.p_value_surrogate is not None
            assert 0 < c.p_value_surrogate <= 1

    def test_interpretation_populated(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=False,
        )
        assert len(results.interpretation) > 0

    def test_dimensions_consistent(self, data, forcing_groups, response_names):
        engine = TangentSpaceRegression(
            window=60, step=6, shrinkage=0.15, n_surrogates=0,
        )
        results = engine.regress(
            data, forcing_groups, response_names,
            test_surrogates=False,
        )
        d = len(response_names)
        assert results.d_climate == d
        assert results.d_tangent == d * (d + 1) // 2
        assert results.n_forcings == len(forcing_groups)


class TestDemo:
    """Test the demo runners."""

    def test_partition_demo_runs(self):
        results = run_attribution_demo(verbose=False)
        assert isinstance(results, AttributionResults)
        assert results.observed_geodesic > 0
        assert len(results.fingerprints) >= 2

    def test_tangent_demo_runs(self):
        results = run_tangent_regression_demo(verbose=False)
        assert isinstance(results, TangentRegressionResults)
        assert results.r_squared >= 0
        assert len(results.coefficients) >= 2
