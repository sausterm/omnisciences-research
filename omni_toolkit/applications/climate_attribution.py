"""
Climate forcing attribution on the SPD covariance manifold.

Separates natural (solar, volcanic) from anthropogenic (GHG, aerosol)
contributions to observed climate covariance changes using Riemannian
geometry on SPD(d) = GL+(d)/SO(d).

Two attribution methods:

1. **Partition-based** (ClimateAttribution):
   Split time into high/low forcing, compare covariances via geodesic.
   Simple but susceptible to confounding when forcings are correlated.

2. **Tangent-space regression** (TangentSpaceRegression):
   Map rolling covariances to the tangent space at the Fréchet mean,
   then simultaneously regress against all forcings. This is optimal
   fingerprinting transported to the SPD manifold — it handles
   confounding, provides proper β coefficients, and supports
   phase-randomisation surrogates for significance.

   Method:
       a. Rolling covariance → time series of SPD matrices {C(t)}
       b. Fréchet mean C̄ = argmin Σ d²(C̄, C(t))
       c. Log map: V(t) = log_{C̄}(C(t)) ∈ T_{C̄} SPD(d)
       d. Vectorise: v(t) = vech(V(t)) ∈ ℝ^{d(d+1)/2}
       e. Regression: v(t) = Σ_i β_i F_i(t) + ε(t)
       f. Each β_i reshaped to symmetric matrix → V+/V- decomposition
       g. Significance via phase-randomisation (preserves spectrum,
          destroys coupling) or Wishart-based F-test

Key insight: Different forcings reshape climate covariance in
geometrically distinct ways:
    - Anthropogenic (CO2): Primarily V+ (trace) — increases total
      variance / energy budget. Slow, monotonic trend.
    - Solar cycle: Primarily V- (traceless) — reorganises teleconnection
      correlations without much total variance change. Periodic ~11yr.
    - Volcanic: Mixed V+/V- — sudden global cooling (V+) plus
      disrupted ENSO/NAO coupling (V-). Episodic, decays in ~2yr.

References:
    - Hasselmann (1979). On the signal-to-noise problem in atmospheric
      response studies. In Meteorology of Tropical Oceans.
    - Allen & Stott (2003). Estimating signal amplitudes in optimal
      fingerprinting. Climate Dynamics, 21, 477-491.
    - Hegerl et al. (2007). Understanding and attributing climate change.
      IPCC AR4 WG1 Ch. 9.
    - Pennec et al. (2006). A Riemannian framework for tensor computing.
      IJCV, 66(1), 41-66.
    - Said et al. (2017). Riemannian Gaussian distributions on the space
      of SPD matrices. IEEE Trans. Info. Theory.
    - Austermann (2026). Riemannian methods for climate attribution.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .climate_analysis import ClimateData, ClimateDataLoader
from .solar_forcing import (
    SOLAR_INDICES, SolarDataLoader, forcing_response_geodesic,
)
from .spd_ml import (
    SPDLayer, _matrix_log, _matrix_sqrt, _matrix_invsqrt,
    _matrix_sqrt_pair, _symmetrize, _matrix_exp,
    geodesic_shrinkage, tyler_m_estimator,
    power_euclidean_mean, power_euclidean_log_map,
)


# =====================================================================
# Anthropogenic forcing registry
# =====================================================================

ANTHROPOGENIC_INDICES = ["co2", "ch4", "aod"]

ANTHRO_REGISTRY = {
    "co2": {
        "name": "CO₂ Concentration",
        "group": "anthropogenic",
        "source": "NOAA/GML Mauna Loa",
        "unit": "ppm",
        "description": "Monthly mean atmospheric CO₂ from Mauna Loa. "
                        "Primary long-lived greenhouse gas. ~280 ppm pre-industrial, "
                        "~420 ppm in 2024. Near-monotonic increase since 1958.",
    },
    "ch4": {
        "name": "CH₄ Concentration",
        "group": "anthropogenic",
        "source": "NOAA/GML",
        "unit": "ppb",
        "description": "Monthly mean atmospheric methane. Second most important "
                        "long-lived GHG. Plateau in late 1990s, resumed growth ~2007.",
    },
    "aod": {
        "name": "Aerosol Optical Depth",
        "group": "volcanic",
        "source": "NASA GISS / Sato et al.",
        "unit": "dimensionless",
        "description": "Stratospheric aerosol optical depth at 550nm. Spikes during "
                        "major volcanic eruptions (El Chichón 1982, Pinatubo 1991). "
                        "Affects both SW and LW radiation.",
    },
}

# Known volcanic eruption AOD peaks for synthetic data
VOLCANIC_AOD_EVENTS = [
    {"name": "Agung", "year": 1963, "month": 3, "peak_aod": 0.10, "decay_months": 24},
    {"name": "El Chichón", "year": 1982, "month": 4, "peak_aod": 0.08, "decay_months": 18},
    {"name": "Pinatubo", "year": 1991, "month": 6, "peak_aod": 0.15, "decay_months": 30},
]


# =====================================================================
# Forcing fingerprint
# =====================================================================

@dataclass
class ForcingFingerprint:
    """Geometric fingerprint of a single forcing's effect on climate."""
    name: str
    geodesic_distance: float       # d(C_high, C_low)
    v_plus: float                  # trace component magnitude
    v_minus: float                 # traceless component magnitude
    v_ratio: float                 # v+ / (v+ + v-)
    tangent_vector: np.ndarray     # log(C_high^{-1/2} C_low C_high^{-1/2})
    n_high: int                    # samples in high-forcing partition
    n_low: int                     # samples in low-forcing partition
    lag_months: int = 0
    mechanism: str = ""            # human-readable interpretation


@dataclass
class AttributionResults:
    """Results from multi-forcing attribution analysis."""
    fingerprints: Dict[str, ForcingFingerprint]
    # Fractional attribution (sums to ~1)
    attribution_fractions: Dict[str, float]
    # Observed trend fingerprint for comparison
    observed_tangent: Optional[np.ndarray] = None
    observed_geodesic: Optional[float] = None
    # Projection coefficients (how much each forcing explains observed)
    projections: Dict[str, float] = field(default_factory=dict)
    # Statistical significance
    p_values: Dict[str, float] = field(default_factory=dict)
    # Residual (unexplained fraction)
    residual_fraction: float = 0.0
    # Summary
    dominant_forcing: str = ""
    interpretation: List[str] = field(default_factory=list)


# =====================================================================
# Synthetic anthropogenic data
# =====================================================================

def generate_anthropogenic_series(T: int, start_year: int = 1956,
                                  seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate realistic synthetic anthropogenic forcing time series.

    Returns normalised anomaly series suitable for merging with climate data.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    years = start_year + t / 12
    fractional = years

    # --- CO2: Keeling curve (exponential + seasonal) ---
    # Pre-industrial ~280ppm, 1958 ~315ppm, 2024 ~425ppm
    co2_trend = 315 + 1.5 * (fractional - 1958) + 0.012 * (fractional - 1958) ** 2
    co2_seasonal = 3.0 * np.sin(2 * np.pi * (fractional - 1958.3))  # NH growing season
    co2_raw = co2_trend + co2_seasonal + rng.normal(0, 0.3, T)
    # Normalise as anomaly from 1980 mean
    co2_1980 = 315 + 1.5 * 22 + 0.012 * 22**2  # ~343
    co2 = (co2_raw - co2_1980) / 20.0  # ~[-2, 4] range

    # --- CH4: plateau then resumed growth ---
    ch4_base = 1600 + 12 * (fractional - 1980)
    # Plateau 1998-2007
    plateau_mask = (fractional > 1998) & (fractional < 2007)
    ch4_base[plateau_mask] = 1600 + 12 * (1998 - 1980)
    # Resumed growth
    resumed = fractional > 2007
    ch4_base[resumed] = 1600 + 12 * (1998 - 1980) + 8 * (fractional[resumed] - 2007)
    ch4_raw = ch4_base + rng.normal(0, 5, T)
    ch4 = (ch4_raw - 1750) / 100.0  # normalise

    # --- AOD: volcanic spikes on clean background ---
    aod_raw = 0.002 * np.ones(T) + rng.exponential(0.001, T)
    for event in VOLCANIC_AOD_EVENTS:
        onset = (event["year"] - start_year) * 12 + event["month"] - 1
        if 0 <= onset < T:
            for dt in range(min(event["decay_months"] * 2, T - onset)):
                aod_raw[onset + dt] += event["peak_aod"] * np.exp(
                    -dt / event["decay_months"]
                )
    aod = (aod_raw - 0.003) / 0.03  # normalise

    return {"co2": co2, "ch4": ch4, "aod": aod}


# =====================================================================
# Attribution engine
# =====================================================================

class ClimateAttribution:
    """Attribute observed climate covariance changes to multiple forcings.

    Uses forcing-conditioned geodesics on SPD(d) to compute each forcing's
    geometric fingerprint, then projects observed covariance trends onto
    the forcing fingerprint space to estimate fractional attribution.
    """

    def __init__(self, window: int = 60, shrinkage: float = 0.15,
                 n_permutations: int = 200):
        """
        Args:
            window: rolling covariance window in months.
            shrinkage: Ledoit-Wolf shrinkage parameter.
            n_permutations: number of permutations for significance testing.
        """
        self.window = window
        self.shrinkage = shrinkage
        self.n_permutations = n_permutations

    def compute_fingerprint(self, data: ClimateData,
                            forcing_names: List[str],
                            response_names: List[str],
                            lag_months: int = 0,
                            label: str = "") -> ForcingFingerprint:
        """Compute the geometric fingerprint of a single forcing.

        Partitions time into high/low forcing periods and measures the
        geodesic distance between their response covariance matrices.
        """
        forcing_idx = [data.index_names.index(n) for n in forcing_names]
        response_idx = [data.index_names.index(n) for n in response_names]

        forcing = data.values[:, forcing_idx]
        response = data.values[:, response_idx]
        d_resp = len(response_idx)

        # Apply lag
        if lag_months > 0:
            forcing = forcing[:-lag_months]
            response = response[lag_months:]
        elif lag_months < 0:
            forcing = forcing[-lag_months:]
            response = response[:lag_months]

        # Composite forcing (first PC if multiple)
        if forcing.shape[1] > 1:
            fc = forcing - forcing.mean(axis=0)
            _, _, Vt = np.linalg.svd(fc, full_matrices=False)
            composite = fc @ Vt[0]
        else:
            composite = forcing[:, 0]

        # Partition into high/low
        median = np.median(composite)
        std = np.std(composite)
        high_mask = composite > median + 0.3 * std
        low_mask = composite < median - 0.3 * std

        resp_high = response[high_mask]
        resp_low = response[low_mask]

        if len(resp_high) < d_resp + 5 or len(resp_low) < d_resp + 5:
            return ForcingFingerprint(
                name=label or "+".join(forcing_names),
                geodesic_distance=0.0, v_plus=0.0, v_minus=0.0,
                v_ratio=0.0, tangent_vector=np.zeros((d_resp, d_resp)),
                n_high=int(high_mask.sum()), n_low=int(low_mask.sum()),
                lag_months=lag_months,
                mechanism="Insufficient data for partitioning",
            )

        C_high = np.cov(resp_high, rowvar=False)
        C_low = np.cov(resp_low, rowvar=False)

        # Shrinkage
        if self.shrinkage > 0:
            mu_h = np.trace(C_high) / d_resp
            mu_l = np.trace(C_low) / d_resp
            C_high = (1 - self.shrinkage) * C_high + self.shrinkage * mu_h * np.eye(d_resp)
            C_low = (1 - self.shrinkage) * C_low + self.shrinkage * mu_l * np.eye(d_resp)

        # Ensure SPD
        for C in (C_high, C_low):
            C[:] = _symmetrize(C)
            eigvals = np.linalg.eigvalsh(C)
            if eigvals[0] <= 0:
                C += (abs(eigvals[0]) + 1e-8) * np.eye(d_resp)

        # Geodesic and tangent vector
        A_invsqrt = _matrix_invsqrt(C_high)
        M = A_invsqrt @ C_low @ A_invsqrt
        M = _symmetrize(M)
        tangent = _matrix_log(M)

        geodesic = np.linalg.norm(tangent, 'fro')

        # V+/V- decomposition
        trace_part = np.trace(tangent) / d_resp
        v_plus = abs(trace_part) * np.sqrt(d_resp)
        traceless = tangent - trace_part * np.eye(d_resp)
        v_minus = np.linalg.norm(traceless, 'fro')
        v_ratio = v_plus / (v_plus + v_minus) if (v_plus + v_minus) > 1e-10 else 0.5

        # Interpret mechanism
        if v_ratio > 0.6:
            mechanism = "Energy budget shift (V+ dominated)"
        elif v_ratio < 0.25:
            mechanism = "Teleconnection reorganisation (V- dominated)"
        else:
            mechanism = "Mixed energy + coupling change"

        return ForcingFingerprint(
            name=label or "+".join(forcing_names),
            geodesic_distance=geodesic,
            v_plus=v_plus,
            v_minus=v_minus,
            v_ratio=v_ratio,
            tangent_vector=tangent,
            n_high=int(high_mask.sum()),
            n_low=int(low_mask.sum()),
            lag_months=lag_months,
            mechanism=mechanism,
        )

    def compute_observed_trend(self, data: ClimateData,
                                response_names: List[str],
                                split_year: Optional[int] = None
                                ) -> Tuple[np.ndarray, float]:
        """Compute the observed covariance trend as a tangent vector.

        Splits data into early and late halves (or at split_year)
        and computes the geodesic between their covariance matrices.

        Returns:
            (tangent_vector, geodesic_distance)
        """
        response_idx = [data.index_names.index(n) for n in response_names]
        response = data.values[:, response_idx]
        d_resp = len(response_idx)

        if split_year is None:
            mid = len(response) // 2
        else:
            mid = data.date_to_index(split_year, 1)
            mid = max(d_resp + 5, min(mid, len(response) - d_resp - 5))

        early = response[:mid]
        late = response[mid:]

        C_early = np.cov(early, rowvar=False)
        C_late = np.cov(late, rowvar=False)

        # Shrinkage
        if self.shrinkage > 0:
            mu_e = np.trace(C_early) / d_resp
            mu_l = np.trace(C_late) / d_resp
            C_early = (1 - self.shrinkage) * C_early + self.shrinkage * mu_e * np.eye(d_resp)
            C_late = (1 - self.shrinkage) * C_late + self.shrinkage * mu_l * np.eye(d_resp)

        # Ensure SPD
        for C in (C_early, C_late):
            C[:] = _symmetrize(C)
            eigvals = np.linalg.eigvalsh(C)
            if eigvals[0] <= 0:
                C += (abs(eigvals[0]) + 1e-8) * np.eye(d_resp)

        A_invsqrt = _matrix_invsqrt(C_early)
        M = A_invsqrt @ C_late @ A_invsqrt
        M = _symmetrize(M)
        tangent = _matrix_log(M)

        geodesic = np.linalg.norm(tangent, 'fro')
        return tangent, geodesic

    def project_onto_fingerprint(self, observed: np.ndarray,
                                  fingerprint: np.ndarray) -> float:
        """Project observed tangent vector onto a forcing fingerprint.

        Uses the Frobenius inner product on symmetric matrices (which is
        the Riemannian inner product at the identity on SPD).

        Returns the cosine similarity (projection coefficient).
        """
        norm_o = np.linalg.norm(observed, 'fro')
        norm_f = np.linalg.norm(fingerprint, 'fro')
        if norm_o < 1e-12 or norm_f < 1e-12:
            return 0.0
        return float(np.sum(observed * fingerprint) / (norm_o * norm_f))

    def permutation_test(self, data: ClimateData,
                         forcing_names: List[str],
                         response_names: List[str],
                         observed_distance: float,
                         lag_months: int = 0,
                         seed: int = 42) -> float:
        """Test significance of a forcing-response geodesic via permutation.

        Shuffles forcing time series to destroy temporal correlation with
        response, recomputes geodesic distance, and estimates p-value as
        the fraction of permuted distances >= observed.
        """
        rng = np.random.RandomState(seed)
        forcing_idx = [data.index_names.index(n) for n in forcing_names]

        count_ge = 0
        for _ in range(self.n_permutations):
            # Shuffle forcing in time (block shuffle to preserve autocorrelation)
            perm_data = ClimateData(
                values=data.values.copy(),
                dates=data.dates,
                index_names=data.index_names,
                metadata=data.metadata,
            )
            block_size = 12  # annual blocks
            n_blocks = len(perm_data.values) // block_size
            perm_order = rng.permutation(n_blocks)
            shuffled = np.concatenate([
                perm_data.values[b * block_size:(b + 1) * block_size, forcing_idx]
                for b in perm_order
            ], axis=0)[:len(perm_data.values)]
            perm_data.values[:len(shuffled), forcing_idx] = shuffled

            result = forcing_response_geodesic(
                perm_data, forcing_names, response_names,
                window=self.window, lag_months=lag_months,
                shrinkage=self.shrinkage,
            )
            if result["geodesic_distance"] >= observed_distance:
                count_ge += 1

        return (count_ge + 1) / (self.n_permutations + 1)

    def attribute(self, data: ClimateData,
                  forcing_groups: Dict[str, List[str]],
                  response_names: List[str],
                  lag_months: Dict[str, int] = None,
                  split_year: Optional[int] = None,
                  test_significance: bool = True) -> AttributionResults:
        """Run full multi-forcing attribution.

        Args:
            data: ClimateData containing both forcing and response indices.
            forcing_groups: Dict mapping label → list of forcing index names.
                Example: {"solar": ["sunspot", "f107"],
                          "anthropogenic": ["co2"],
                          "volcanic": ["aod"]}
            response_names: list of climate response index names.
            lag_months: Dict mapping label → lag in months.
                Default: 0 for all.
            split_year: year to split early/late for observed trend.
            test_significance: whether to run permutation tests.

        Returns:
            AttributionResults with fingerprints, projections, and fractions.
        """
        if lag_months is None:
            lag_months = {k: 0 for k in forcing_groups}

        # 1. Compute fingerprint for each forcing
        fingerprints = {}
        for label, forcing_names_list in forcing_groups.items():
            lag = lag_months.get(label, 0)
            fp = self.compute_fingerprint(
                data, forcing_names_list, response_names,
                lag_months=lag, label=label,
            )
            fingerprints[label] = fp

        # 2. Compute observed covariance trend
        observed_tangent, observed_geodesic = self.compute_observed_trend(
            data, response_names, split_year=split_year
        )

        # 3. Project observed trend onto each fingerprint
        projections = {}
        for label, fp in fingerprints.items():
            proj = self.project_onto_fingerprint(
                observed_tangent, fp.tangent_vector
            )
            projections[label] = proj

        # 4. Compute attribution fractions from absolute projections
        abs_projections = {k: abs(v) for k, v in projections.items()}
        total_proj = sum(abs_projections.values())
        if total_proj > 1e-10:
            attribution = {k: v / total_proj for k, v in abs_projections.items()}
        else:
            attribution = {k: 1.0 / len(forcing_groups) for k in forcing_groups}

        # Residual: how much of observed is NOT explained by any forcing
        # Use sum of squared projections vs observed norm
        explained_var = sum(p ** 2 for p in projections.values())
        residual = max(0.0, 1.0 - explained_var)

        # 5. Permutation tests for significance
        p_values = {}
        if test_significance:
            for label, forcing_names_list in forcing_groups.items():
                lag = lag_months.get(label, 0)
                fp = fingerprints[label]
                if fp.geodesic_distance > 0:
                    p = self.permutation_test(
                        data, forcing_names_list, response_names,
                        fp.geodesic_distance, lag_months=lag,
                    )
                    p_values[label] = p
                else:
                    p_values[label] = 1.0

        # 6. Determine dominant forcing
        dominant = max(attribution, key=attribution.get)

        # 7. Build interpretation
        interpretation = []

        # Rank forcings by attribution
        ranked = sorted(attribution.items(), key=lambda x: -x[1])
        for label, frac in ranked:
            fp = fingerprints[label]
            sig_str = ""
            if label in p_values:
                if p_values[label] < 0.01:
                    sig_str = " (p < 0.01, highly significant)"
                elif p_values[label] < 0.05:
                    sig_str = f" (p = {p_values[label]:.3f}, significant)"
                else:
                    sig_str = f" (p = {p_values[label]:.3f}, not significant)"
            interpretation.append(
                f"{label}: {frac:.1%} of observed change, "
                f"d = {fp.geodesic_distance:.3f}, "
                f"V+/V- ratio = {fp.v_ratio:.3f} "
                f"({fp.mechanism}){sig_str}"
            )

        # Compare V+/V- signatures
        if len(fingerprints) >= 2:
            vr = {k: fp.v_ratio for k, fp in fingerprints.items()}
            most_trace = max(vr, key=vr.get)
            most_shape = min(vr, key=vr.get)
            if vr[most_trace] - vr[most_shape] > 0.15:
                interpretation.append(
                    f"Distinct mechanisms: {most_trace} acts via energy budget "
                    f"(V+ ratio {vr[most_trace]:.2f}), "
                    f"{most_shape} acts via teleconnections "
                    f"(V+ ratio {vr[most_shape]:.2f})"
                )

        interpretation.append(
            f"Residual (unexplained): {residual:.1%}"
        )

        return AttributionResults(
            fingerprints=fingerprints,
            attribution_fractions=attribution,
            observed_tangent=observed_tangent,
            observed_geodesic=observed_geodesic,
            projections=projections,
            p_values=p_values,
            residual_fraction=residual,
            dominant_forcing=dominant,
            interpretation=interpretation,
        )


# =====================================================================
# Tangent-space regression (publication-grade attribution)
# =====================================================================

def _sym_to_vec(S: np.ndarray) -> np.ndarray:
    """Vectorise symmetric matrix using vech (upper triangle, row-major).

    For d×d symmetric S, returns d(d+1)/2 vector with diagonal elements
    scaled by 1.0 and off-diagonal by √2 (so that ||vec||² = ||S||²_F).
    """
    d = S.shape[0]
    v = []
    for i in range(d):
        for j in range(i, d):
            if i == j:
                v.append(S[i, j])
            else:
                v.append(S[i, j] * np.sqrt(2))
    return np.array(v)


def _vec_to_sym(v: np.ndarray, d: int) -> np.ndarray:
    """Reconstruct symmetric matrix from vech vector."""
    S = np.zeros((d, d))
    k = 0
    for i in range(d):
        for j in range(i, d):
            if i == j:
                S[i, j] = v[k]
            else:
                S[i, j] = v[k] / np.sqrt(2)
                S[j, i] = S[i, j]
            k += 1
    return S


def phase_randomise(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Phase-randomisation surrogate for a 1D time series.

    Preserves the power spectrum (autocorrelation structure) but
    destroys phase relationships with other series. This is the
    standard surrogate method from Theiler et al. (1992).

    For a real-valued signal, we randomise phases while maintaining
    conjugate symmetry so the result is real.
    """
    n = len(x)
    ft = np.fft.rfft(x)
    magnitudes = np.abs(ft)

    # Random phases (keep DC and Nyquist real)
    random_phases = rng.uniform(0, 2 * np.pi, len(ft))
    random_phases[0] = 0  # DC
    if n % 2 == 0:
        random_phases[-1] = 0  # Nyquist

    ft_surrogate = magnitudes * np.exp(1j * random_phases)
    surrogate = np.fft.irfft(ft_surrogate, n=n)
    return surrogate


@dataclass
class RegressionCoefficient:
    """Single forcing's regression coefficient in tangent space."""
    name: str
    beta_vector: np.ndarray        # vech vector, length d(d+1)/2
    beta_matrix: np.ndarray        # d×d symmetric matrix
    v_plus: float                  # ||trace component||
    v_minus: float                 # ||traceless component||
    v_ratio: float                 # v+ / (v+ + v-)
    norm: float                    # ||β||_F
    t_statistic: float             # max |t| across components
    f_statistic: float             # multivariate F statistic
    p_value_parametric: float      # from F distribution
    p_value_surrogate: Optional[float] = None  # from phase randomisation
    explained_variance: float = 0.0  # partial R² for this forcing
    mechanism: str = ""


@dataclass
class TangentRegressionResults:
    """Results from tangent-space regression attribution."""
    coefficients: Dict[str, RegressionCoefficient]
    # Model fit
    r_squared: float               # overall R²
    adjusted_r_squared: float      # adjusted for number of predictors
    residual_norm: float           # ||ε||_F
    # Attribution fractions (from partial R²)
    attribution_fractions: Dict[str, float]
    # Observed quantities
    n_timepoints: int
    d_climate: int
    d_tangent: int                 # d(d+1)/2
    n_forcings: int
    # Fréchet mean
    frechet_mean: np.ndarray       # d×d SPD matrix
    # Interpretation
    dominant_forcing: str
    interpretation: List[str] = field(default_factory=list)
    # Ridge / PCA regularisation
    ridge_alpha: float = 0.0
    pca_variance_explained: Optional[np.ndarray] = None


@dataclass
class BootstrapResults:
    """Results from bootstrap stability analysis."""
    # Distributions of attribution fractions [n_boot] per forcing
    attribution_distributions: Dict[str, np.ndarray]
    # 95% confidence intervals
    ci_95: Dict[str, Tuple[float, float]]
    ci_median: Dict[str, float]
    # Model fit distribution
    r_squared_distribution: np.ndarray
    # Metadata
    n_bootstrap: int
    periods_used: List[Tuple[int, int]]
    n_failed: int = 0


class BootstrapStability:
    """Bootstrap over time windows to assess attribution fraction stability.

    Repeatedly subsamples different time periods and window sizes, runs
    the tangent-space regression (without surrogates for speed), and
    collects the distribution of attribution fractions.

    This answers: "How sensitive are the attribution percentages to the
    choice of analysis period?"
    """

    def __init__(self, engine: "TangentSpaceRegression",
                 n_bootstrap: int = 200,
                 min_years: int = 15,
                 seed: int = 42):
        self.engine = engine
        self.n_bootstrap = n_bootstrap
        self.min_years = min_years
        self.seed = seed

    def run(self, data: "ClimateData",
            forcing_groups: Dict[str, List[str]],
            response_names: List[str],
            lag_months: Optional[Dict[str, int]] = None,
            verbose: bool = True) -> BootstrapResults:
        """Run bootstrap stability analysis.

        Args:
            data: Full ClimateData to subsample from.
            forcing_groups: Forcing group definitions.
            response_names: Response variable names.
            lag_months: Lag configuration.
            verbose: Print progress.

        Returns:
            BootstrapResults with distributions and CIs.
        """
        from .climate_analysis import ClimateData

        rng = np.random.RandomState(self.seed)
        labels = list(forcing_groups.keys())
        fracs = {label: [] for label in labels}
        r2s = []
        periods = []
        n_failed = 0

        start_yr = data.start_year
        end_yr = data.end_year
        total_years = end_yr - start_yr

        # Suppress surrogates for speed
        orig_surrogates = self.engine.n_surrogates
        self.engine.n_surrogates = 0

        for b in range(self.n_bootstrap):
            # Random subperiod
            duration = rng.randint(self.min_years, total_years + 1)
            yr_start = rng.randint(start_yr, end_yr - duration + 1)
            yr_end = yr_start + duration

            try:
                sub = data.slice_years(yr_start, yr_end)
                if sub.T < self.engine.window + 20:
                    n_failed += 1
                    continue

                results = self.engine.regress(
                    sub, forcing_groups, response_names,
                    lag_months=lag_months, test_surrogates=False,
                )

                for label in labels:
                    fracs[label].append(
                        results.attribution_fractions.get(label, 0.0)
                    )
                r2s.append(results.r_squared)
                periods.append((yr_start, yr_end))

            except Exception:
                n_failed += 1
                continue

            if verbose and (b + 1) % 50 == 0:
                print(f"  Bootstrap {b + 1}/{self.n_bootstrap} "
                      f"({n_failed} failed)", flush=True)

        # Restore surrogates
        self.engine.n_surrogates = orig_surrogates

        # Convert to arrays
        distributions = {k: np.array(v) for k, v in fracs.items()}
        ci_95 = {}
        ci_median = {}
        for label in labels:
            arr = distributions[label]
            if len(arr) > 0:
                ci_95[label] = (
                    float(np.percentile(arr, 2.5)),
                    float(np.percentile(arr, 97.5)),
                )
                ci_median[label] = float(np.median(arr))
            else:
                ci_95[label] = (0.0, 0.0)
                ci_median[label] = 0.0

        return BootstrapResults(
            attribution_distributions=distributions,
            ci_95=ci_95,
            ci_median=ci_median,
            r_squared_distribution=np.array(r2s),
            n_bootstrap=self.n_bootstrap,
            periods_used=periods,
            n_failed=n_failed,
        )


class TangentSpaceRegression:
    """Riemannian optimal fingerprinting on SPD(d).

    Maps rolling climate covariance matrices to the tangent space at
    the Fréchet mean, then performs simultaneous multivariate OLS
    regression against forcing time series. This handles confounding
    between forcings (unlike partition-based methods) and produces
    proper regression coefficients with F-test significance.

    The V+/V- decomposition of each β coefficient reveals the mechanism:
    does the forcing change the energy budget (V+) or reorganise
    teleconnections (V-)?

    Phase-randomisation surrogates provide non-parametric significance
    tests that account for autocorrelation in both forcing and response.
    """

    def __init__(self, window: int = 60, step: int = 1,
                 shrinkage: float = 0.15,
                 n_surrogates: int = 500,
                 frechet_tol: float = 1e-8,
                 frechet_max_iter: int = 50,
                 alpha: float = 0.0,
                 n_components: Optional[int] = None,
                 metric: str = "affine_invariant",
                 power_alpha: float = 0.5,
                 covariance_method: str = "sample",
                 shrinkage_method: str = "euclidean"):
        """
        Args:
            window: rolling covariance window in months.
            step: step between covariance windows.
            shrinkage: Ledoit-Wolf shrinkage for covariance estimation.
            n_surrogates: number of phase-randomisation surrogates.
            frechet_tol: convergence tolerance for Fréchet mean.
            frechet_max_iter: max iterations for Fréchet mean.
            alpha: Ridge regression penalty (0.0 = OLS). Does not penalise
                the intercept column. Higher values shrink beta towards zero,
                useful when d_tangent >> n_timepoints.
            n_components: Number of PCA components to retain in tangent space
                before regression. If None or >= d_tangent, no reduction is
                applied. When set, Y is projected to the top-n_components
                principal components, regression is run in that subspace,
                and betas are projected back to the full tangent space.
            metric: Riemannian metric to use for Frechet mean and log map.
                "affine_invariant" (default, Paper 1) or "power_euclidean".
            power_alpha: power parameter for power-Euclidean metric
                (default 0.5). Only used when metric="power_euclidean".
            covariance_method: method for estimating rolling covariances.
                "sample" (default, Paper 1) or "tyler" (robust M-estimator).
            shrinkage_method: how to apply shrinkage regularisation.
                "euclidean" (default, Paper 1) or "geodesic" (along SPD
                geodesic, preserving affine invariance).
        """
        self.window = window
        self.step = step
        self.shrinkage = shrinkage
        self.n_surrogates = n_surrogates
        self.frechet_tol = frechet_tol
        self.frechet_max_iter = frechet_max_iter
        self.alpha = alpha
        self.n_components = n_components
        self.metric = metric
        self.power_alpha = power_alpha
        self.covariance_method = covariance_method
        self.shrinkage_method = shrinkage_method

    def _rolling_covariances(self, data: ClimateData,
                              response_names: List[str]
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rolling covariance matrices for response indices.

        Supports different covariance estimation methods (sample or Tyler's
        M-estimator) and shrinkage strategies (Euclidean or geodesic).

        Returns:
            covs: [N, d, d] array of SPD covariance matrices.
            center_times: [N] array of fractional years at window centers.
        """
        response_idx = [data.index_names.index(n) for n in response_names]
        response = data.values[:, response_idx]
        d = len(response_idx)
        T = len(response)

        covs = []
        center_times = []

        for t in range(self.window, T + 1, self.step):
            chunk = response[t - self.window:t]

            # Covariance estimation
            if self.covariance_method == "tyler":
                C = tyler_m_estimator(chunk)
                # Tyler returns det=1; rescale to match sample trace
                sample_trace = np.trace(np.cov(chunk, rowvar=False))
                C = C * (sample_trace / max(np.trace(C), 1e-15))
            else:
                C = np.cov(chunk, rowvar=False)

            # Shrinkage regularisation
            if self.shrinkage > 0:
                if self.shrinkage_method == "geodesic":
                    # Ensure SPD before geodesic shrinkage
                    C = _symmetrize(C)
                    eigvals = np.linalg.eigvalsh(C)
                    if eigvals[0] <= 0:
                        C += (abs(eigvals[0]) + 1e-8) * np.eye(d)
                    C = geodesic_shrinkage(C, self.shrinkage)
                else:
                    # Euclidean (default, Paper 1 behavior)
                    mu = np.trace(C) / d
                    C = (1 - self.shrinkage) * C + self.shrinkage * mu * np.eye(d)

            C = _symmetrize(C)
            eigvals = np.linalg.eigvalsh(C)
            if eigvals[0] <= 0:
                C += (abs(eigvals[0]) + 1e-8) * np.eye(d)

            covs.append(C)
            center_idx = t - self.window // 2
            yr, mo = data.dates[min(center_idx, T - 1)]
            center_times.append(yr + (mo - 1) / 12)

        return np.array(covs), np.array(center_times)

    def _frechet_mean(self, covs: np.ndarray) -> np.ndarray:
        """Compute Frechet mean of SPD matrices.

        Dispatches on self.metric:
        - "affine_invariant": iterative Riemannian gradient descent
          with log-Euclidean initialisation (Paper 1 default).
        - "power_euclidean": closed-form power-Euclidean mean.
        """
        if self.metric == "power_euclidean":
            return power_euclidean_mean(covs, alpha=self.power_alpha)

        # Affine-invariant Frechet mean (default)
        N, d, _ = covs.shape

        # Log-Euclidean init: exp(mean of logs)
        logs = np.array([_matrix_log(C) for C in covs])
        mean_log = logs.mean(axis=0)
        mu = _matrix_exp(_symmetrize(mean_log))

        # Riemannian gradient descent
        for iteration in range(self.frechet_max_iter):
            mu_invsqrt = _matrix_invsqrt(mu)
            mu_sqrt = _matrix_sqrt(mu)

            # Gradient = mean of log maps
            grad = np.zeros((d, d))
            for i in range(N):
                M = mu_invsqrt @ covs[i] @ mu_invsqrt
                M = _symmetrize(M)
                grad += _matrix_log(M)
            grad /= N

            # Check convergence
            grad_norm = np.linalg.norm(grad, 'fro')
            if grad_norm < self.frechet_tol:
                break

            # Step along geodesic: mu' = mu^{1/2} exp(grad) mu^{1/2}
            mu = mu_sqrt @ _matrix_exp(grad) @ mu_sqrt
            mu = _symmetrize(mu)

        return mu

    def _log_map_all(self, covs: np.ndarray,
                      mu: np.ndarray) -> np.ndarray:
        """Map all covariance matrices to tangent space at mu.

        Dispatches on self.metric:
        - "affine_invariant": standard Riemannian log map on SPD(d).
        - "power_euclidean": power-Euclidean log map.

        Returns [N, d(d+1)/2] array of vectorised tangent vectors.
        """
        N, d, _ = covs.shape
        p = d * (d + 1) // 2

        if self.metric == "power_euclidean":
            tangent_mats = power_euclidean_log_map(
                covs, mu, alpha=self.power_alpha
            )  # [N, d, d]
            tangent_vecs = np.zeros((N, p))
            for i in range(N):
                tangent_vecs[i] = _sym_to_vec(tangent_mats[i])
            return tangent_vecs

        # Affine-invariant log map (default)
        mu_invsqrt = _matrix_invsqrt(mu)

        tangent_vecs = np.zeros((N, p))
        for i in range(N):
            M = mu_invsqrt @ covs[i] @ mu_invsqrt
            M = _symmetrize(M)
            logM = _matrix_log(M)
            tangent_vecs[i] = _sym_to_vec(logM)

        return tangent_vecs

    def _align_forcing(self, data: ClimateData,
                        forcing_names: List[str],
                        center_times: np.ndarray,
                        lag_months: int = 0) -> np.ndarray:
        """Align forcing time series to covariance window centers.

        Averages forcing over each covariance window, with optional lag.
        """
        forcing_idx = [data.index_names.index(n) for n in forcing_names]
        T = len(data.values)

        # If multiple forcings, use first PC
        forcing_raw = data.values[:, forcing_idx]
        if forcing_raw.shape[1] > 1:
            fc = forcing_raw - forcing_raw.mean(axis=0)
            _, _, Vt = np.linalg.svd(fc, full_matrices=False)
            forcing_1d = fc @ Vt[0]
        else:
            forcing_1d = forcing_raw[:, 0]

        # Apply lag
        if lag_months != 0:
            forcing_1d = np.roll(forcing_1d, lag_months)
            if lag_months > 0:
                forcing_1d[:lag_months] = forcing_1d[lag_months]
            else:
                forcing_1d[lag_months:] = forcing_1d[lag_months - 1]

        # Average over each window
        aligned = np.zeros(len(center_times))
        for i, ct in enumerate(center_times):
            # Find the window center index in the original data
            t_center = int(round((ct - data.dates[0][0]) * 12))
            t_start = max(0, t_center - self.window // 2)
            t_end = min(T, t_start + self.window)
            aligned[i] = forcing_1d[t_start:t_end].mean()

        return aligned

    def _decompose_beta(self, beta_vec: np.ndarray, d: int,
                         label: str) -> RegressionCoefficient:
        """Decompose a regression coefficient into V+/V-."""
        beta_mat = _vec_to_sym(beta_vec, d)
        norm = np.linalg.norm(beta_mat, 'fro')

        # V+/V-
        trace_part = np.trace(beta_mat) / d
        v_plus = abs(trace_part) * np.sqrt(d)
        traceless = beta_mat - trace_part * np.eye(d)
        v_minus = np.linalg.norm(traceless, 'fro')
        v_ratio = v_plus / (v_plus + v_minus) if (v_plus + v_minus) > 1e-10 else 0.5

        if v_ratio > 0.6:
            mechanism = "Energy budget shift (V+ dominated)"
        elif v_ratio < 0.25:
            mechanism = "Teleconnection reorganisation (V- dominated)"
        else:
            mechanism = "Mixed energy + coupling change"

        return RegressionCoefficient(
            name=label,
            beta_vector=beta_vec,
            beta_matrix=beta_mat,
            v_plus=v_plus,
            v_minus=v_minus,
            v_ratio=v_ratio,
            norm=norm,
            t_statistic=0.0,  # filled in by regression
            f_statistic=0.0,
            p_value_parametric=1.0,
            mechanism=mechanism,
        )

    def regress(self, data: ClimateData,
                forcing_groups: Dict[str, List[str]],
                response_names: List[str],
                lag_months: Optional[Dict[str, int]] = None,
                test_surrogates: bool = True) -> TangentRegressionResults:
        """Run tangent-space regression attribution.

        This is the core method — Riemannian optimal fingerprinting.

        Args:
            data: ClimateData with forcing and response indices.
            forcing_groups: Dict mapping label → forcing index names.
            response_names: climate response index names.
            lag_months: Dict mapping label → lag (positive = forcing leads).
            test_surrogates: whether to run phase-randomisation tests.

        Returns:
            TangentRegressionResults with coefficients and attribution.
        """
        if lag_months is None:
            lag_months = {k: 0 for k in forcing_groups}

        d = len(response_names)
        p = d * (d + 1) // 2
        forcing_labels = list(forcing_groups.keys())
        k = len(forcing_labels)

        # 1. Rolling covariances
        covs, center_times = self._rolling_covariances(data, response_names)
        N = len(covs)

        # 2. Fréchet mean
        mu = self._frechet_mean(covs)

        # 3. Log map to tangent space
        Y = self._log_map_all(covs, mu)  # [N, p]

        # 4. Build design matrix [N, k+1] (forcings + intercept)
        X = np.ones((N, k + 1))
        for j, label in enumerate(forcing_labels):
            lag = lag_months.get(label, 0)
            X[:, j + 1] = self._align_forcing(
                data, forcing_groups[label], center_times, lag
            )

        # Standardise forcing columns (not intercept)
        forcing_means = X[:, 1:].mean(axis=0)
        forcing_stds = X[:, 1:].std(axis=0)
        forcing_stds[forcing_stds < 1e-10] = 1.0
        X[:, 1:] = (X[:, 1:] - forcing_means) / forcing_stds

        # 5. Optional PCA pre-reduction of tangent space
        pca_proj = None  # [p, n_comp] projection matrix
        pca_var_explained = None
        p_eff = p  # effective tangent dimension for regression

        if self.n_components is not None and self.n_components < p:
            # Centre Y (column-wise) for PCA
            Y_mean = Y.mean(axis=0)
            Y_c = Y - Y_mean
            # Econ SVD: U [N, n_comp], S [n_comp], Vt [n_comp, p]
            U_pca, S_pca, Vt_pca = np.linalg.svd(Y_c, full_matrices=False)
            # Keep top n_components
            nc = self.n_components
            pca_proj = Vt_pca[:nc].T  # [p, nc]
            total_var = np.sum(S_pca ** 2)
            pca_var_explained = (S_pca[:nc] ** 2) / total_var if total_var > 0 else S_pca[:nc] * 0
            # Project Y into PCA subspace
            Y_reg = Y_c @ pca_proj  # [N, nc]
            p_eff = nc
        else:
            Y_reg = Y

        # 6. Regression: Y_reg = X @ B_reg + E
        #    If alpha > 0: Ridge (don't penalise intercept column 0)
        #    If alpha == 0: OLS via lstsq
        if self.alpha > 0:
            XtX = X.T @ X
            # Penalty matrix: penalise all columns except intercept
            penalty = self.alpha * np.eye(k + 1)
            penalty[0, 0] = 0.0  # no penalty on intercept
            B_reg = np.linalg.solve(XtX + penalty, X.T @ Y_reg)
        else:
            B_reg, _, _, _ = np.linalg.lstsq(X, Y_reg, rcond=None)

        Y_hat_reg = X @ B_reg
        E_reg = Y_reg - Y_hat_reg

        # If PCA was used, project betas back to full tangent space
        if pca_proj is not None:
            B = B_reg @ pca_proj.T  # [k+1, p]
            Y_hat = X @ B  # predictions in full space (centred)
            # Residuals in full space (relative to centred Y)
            Y_c_full = Y - Y.mean(axis=0)
            E = Y_c_full - Y_hat
        else:
            B = B_reg
            Y_hat = X @ B
            E = Y - Y_hat

        # 7. Model fit statistics
        SS_total = np.sum((Y - Y.mean(axis=0)) ** 2)
        SS_residual = np.sum(E ** 2)
        r_squared = 1 - SS_residual / SS_total if SS_total > 0 else 0.0
        adj_r_squared = 1 - (1 - r_squared) * (N - 1) / max(N - k - 1, 1)

        # 8. Coefficient analysis
        #    Residual covariance for t/F statistics
        dof = max(N - k - 1, 1)
        sigma2 = SS_residual / (dof * p)  # mean squared error per component

        # (X'X + penalty)^{-1} for standard errors (Ridge-aware)
        if self.alpha > 0:
            penalty = self.alpha * np.eye(k + 1)
            penalty[0, 0] = 0.0
            XtX_reg = X.T @ X + penalty
            XtX_inv = np.linalg.pinv(XtX_reg)
        else:
            XtX_inv = np.linalg.pinv(X.T @ X)

        coefficients = {}
        partial_r2 = {}

        for j, label in enumerate(forcing_labels):
            beta_vec = B[j + 1]  # skip intercept row (full tangent space)
            coeff = self._decompose_beta(beta_vec, d, label)

            # Standard errors and t-statistics
            se_scale = np.sqrt(sigma2 * XtX_inv[j + 1, j + 1])
            if se_scale > 0:
                t_stats = beta_vec / (se_scale + 1e-15)
                coeff.t_statistic = float(np.max(np.abs(t_stats)))
            else:
                coeff.t_statistic = 0.0

            # Multivariate F-test for this forcing
            # Compare full model vs model without this forcing
            # Both models use the same regression method (Ridge/PCA)
            X_reduced = np.delete(X, j + 1, axis=1)
            if self.alpha > 0:
                XtX_r = X_reduced.T @ X_reduced
                pen_r = self.alpha * np.eye(X_reduced.shape[1])
                pen_r[0, 0] = 0.0
                B_reduced_reg = np.linalg.solve(XtX_r + pen_r, X_reduced.T @ Y_reg)
            else:
                B_reduced_reg, _, _, _ = np.linalg.lstsq(X_reduced, Y_reg, rcond=None)

            if pca_proj is not None:
                B_reduced_full = B_reduced_reg @ pca_proj.T
                Y_hat_reduced = X_reduced @ B_reduced_full
                Y_c_full = Y - Y.mean(axis=0)
                SS_reduced = np.sum((Y_c_full - Y_hat_reduced) ** 2)
            else:
                Y_hat_reduced = X_reduced @ B_reduced_reg
                SS_reduced = np.sum((Y - Y_hat_reduced) ** 2)

            # F = ((SS_reduced - SS_residual) / p_diff) / (SS_residual / dof)
            SS_diff = SS_reduced - SS_residual
            if SS_residual > 0:
                f_stat = (SS_diff / p) / (SS_residual / (dof * p))
                coeff.f_statistic = float(f_stat)
            else:
                coeff.f_statistic = 0.0

            # Parametric p-value from F distribution
            from scipy.stats import f as f_dist
            if coeff.f_statistic > 0 and dof > 0:
                coeff.p_value_parametric = float(
                    1 - f_dist.cdf(coeff.f_statistic, p, dof * p)
                )
            else:
                coeff.p_value_parametric = 1.0

            # Partial R²
            partial = SS_diff / SS_reduced if SS_reduced > 0 else 0.0
            coeff.explained_variance = float(max(0, partial))
            partial_r2[label] = coeff.explained_variance

            coefficients[label] = coeff

        # 9. Phase-randomisation surrogate testing
        if test_surrogates and self.n_surrogates > 0:
            self._surrogate_test(
                data, forcing_groups, response_names,
                lag_months, coefficients, Y, X, mu, center_times,
                pca_proj=pca_proj,
            )

        # 10. Attribution fractions from partial R²
        total_partial = sum(partial_r2.values())
        if total_partial > 1e-10:
            attribution = {k: v / total_partial for k, v in partial_r2.items()}
        else:
            attribution = {k: 1.0 / len(forcing_groups) for k in forcing_groups}

        # 11. Dominant forcing
        dominant = max(attribution, key=attribution.get)

        # 12. Interpretation
        interpretation = []
        ranked = sorted(attribution.items(), key=lambda x: -x[1])
        for label, frac in ranked:
            c = coefficients[label]
            sig = ""
            if c.p_value_surrogate is not None:
                pv = c.p_value_surrogate
                if pv < 0.01:
                    sig = f" (p_surr < 0.01 ***)"
                elif pv < 0.05:
                    sig = f" (p_surr = {pv:.3f} **)"
                elif pv < 0.10:
                    sig = f" (p_surr = {pv:.3f} *)"
                else:
                    sig = f" (p_surr = {pv:.3f})"
            elif c.p_value_parametric < 1.0:
                pv = c.p_value_parametric
                if pv < 0.01:
                    sig = f" (p_F < 0.01 ***)"
                elif pv < 0.05:
                    sig = f" (p_F = {pv:.3f} **)"
                else:
                    sig = f" (p_F = {pv:.3f})"

            interpretation.append(
                f"{label}: {frac:.1%} of explained covariance change, "
                f"||β|| = {c.norm:.4f}, "
                f"V+/V- = {c.v_ratio:.3f} ({c.mechanism}){sig}"
            )

        # V+/V- comparison
        if len(coefficients) >= 2:
            vr = {k: c.v_ratio for k, c in coefficients.items()}
            most_trace = max(vr, key=vr.get)
            most_shape = min(vr, key=vr.get)
            if vr[most_trace] - vr[most_shape] > 0.1:
                interpretation.append(
                    f"Mechanistic separation: {most_trace} → energy budget "
                    f"(V+ = {vr[most_trace]:.2f}), "
                    f"{most_shape} → teleconnections "
                    f"(V+ = {vr[most_shape]:.2f})"
                )

        interpretation.append(f"Model R² = {r_squared:.4f} "
                              f"(adjusted = {adj_r_squared:.4f})")

        return TangentRegressionResults(
            coefficients=coefficients,
            r_squared=float(r_squared),
            adjusted_r_squared=float(adj_r_squared),
            residual_norm=float(np.sqrt(SS_residual)),
            attribution_fractions=attribution,
            n_timepoints=N,
            d_climate=d,
            d_tangent=p,
            n_forcings=k,
            frechet_mean=mu,
            dominant_forcing=dominant,
            interpretation=interpretation,
            ridge_alpha=self.alpha,
            pca_variance_explained=pca_var_explained,
        )

    def _surrogate_test(self, data, forcing_groups, response_names,
                         lag_months, coefficients, Y, X, mu, center_times,
                         pca_proj=None):
        """Block-permutation surrogate test for each forcing.

        For trend-like forcings (CO2, CH4), phase-randomisation preserves
        the power spectrum — so surrogates of a monotonic trend are also
        monotonic, making rejection impossible. Block permutation is the
        correct null: it shuffles blocks of the forcing time series,
        destroying the forcing-response temporal alignment while preserving
        short-range autocorrelation within blocks.

        Block size is set to the covariance window size (self.window // step)
        to match the effective temporal resolution of the regression.
        """
        rng = np.random.RandomState(42)
        forcing_labels = list(forcing_groups.keys())
        k = len(forcing_labels)
        N, p = Y.shape

        # Prepare the regression target (PCA-reduced if applicable)
        if pca_proj is not None:
            Y_mean = Y.mean(axis=0)
            Y_reg = (Y - Y_mean) @ pca_proj
        else:
            Y_reg = Y

        # Block size: covariance window in units of regression samples
        block_size = max(4, self.window // max(self.step, 1) // 4)

        for j, label in enumerate(forcing_labels):
            coeff = coefficients[label]
            observed_norm = coeff.norm

            null_norms = np.zeros(self.n_surrogates)
            original_col = X[:, j + 1].copy()

            for s in range(self.n_surrogates):
                X_surr = X.copy()

                # Block-shuffle: divide into blocks, permute block order
                n_blocks = N // block_size
                remainder = N - n_blocks * block_size

                blocks = []
                for b in range(n_blocks):
                    blocks.append(original_col[b * block_size:(b + 1) * block_size])
                if remainder > 0:
                    blocks.append(original_col[n_blocks * block_size:])

                perm = rng.permutation(len(blocks))
                shuffled = np.concatenate([blocks[p_idx] for p_idx in perm])[:N]

                # Re-standardise
                sm, ss = shuffled.mean(), shuffled.std()
                if ss > 1e-10:
                    X_surr[:, j + 1] = (shuffled - sm) / ss
                else:
                    X_surr[:, j + 1] = 0.0

                # Refit using same method (Ridge/OLS) and PCA subspace
                if self.alpha > 0:
                    XtX_s = X_surr.T @ X_surr
                    pen_s = self.alpha * np.eye(k + 1)
                    pen_s[0, 0] = 0.0
                    B_surr_reg = np.linalg.solve(XtX_s + pen_s, X_surr.T @ Y_reg)
                else:
                    B_surr_reg, _, _, _ = np.linalg.lstsq(X_surr, Y_reg, rcond=None)

                # Project back to full tangent space if PCA was used
                if pca_proj is not None:
                    beta_surr = B_surr_reg[j + 1] @ pca_proj.T
                else:
                    beta_surr = B_surr_reg[j + 1]

                null_norms[s] = np.linalg.norm(
                    _vec_to_sym(beta_surr, coefficients[label].beta_matrix.shape[0]),
                    'fro'
                )

            # p-value: fraction of surrogates with ||β|| >= observed
            p_value = (np.sum(null_norms >= observed_norm) + 1) / (self.n_surrogates + 1)
            coeff.p_value_surrogate = float(p_value)

    @classmethod
    def cv_alpha(cls, data: ClimateData,
                 forcing_groups: Dict[str, List[str]],
                 response_names: List[str],
                 alphas: Optional[List[float]] = None,
                 n_folds: int = 5,
                 lag_months: Optional[Dict[str, int]] = None,
                 n_components: Optional[int] = None,
                 window: int = 60, step: int = 1,
                 shrinkage: float = 0.15,
                 seed: int = 42) -> Tuple[float, Dict[float, float]]:
        """K-fold cross-validation to select optimal Ridge alpha.

        Splits the N covariance time points into k folds (contiguous
        blocks to respect temporal ordering), fits Ridge regression on
        k-1 folds, and evaluates prediction error on the held-out fold.

        Args:
            data: ClimateData with forcing and response indices.
            forcing_groups: Dict mapping label to forcing index names.
            response_names: climate response index names.
            alphas: list of alpha values to try.
                Default: np.logspace(-4, 4, 17).
            n_folds: number of CV folds (contiguous temporal blocks).
            lag_months: Dict mapping label to lag.
            n_components: PCA components (applied inside each fold).
            window: rolling covariance window.
            step: step between windows.
            shrinkage: Ledoit-Wolf shrinkage.
            seed: random seed (unused, folds are deterministic).

        Returns:
            (best_alpha, scores) where scores maps alpha -> mean CV error.
        """
        if alphas is None:
            alphas = list(np.logspace(-4, 4, 17))

        if lag_months is None:
            lag_months = {k_: 0 for k_ in forcing_groups}

        # Build a temporary engine to get rolling covariances and design matrix
        engine = cls(window=window, step=step, shrinkage=shrinkage,
                     n_surrogates=0, alpha=0.0, n_components=None)

        d = len(response_names)
        p = d * (d + 1) // 2
        forcing_labels = list(forcing_groups.keys())
        k = len(forcing_labels)

        covs, center_times = engine._rolling_covariances(data, response_names)
        N = len(covs)
        mu = engine._frechet_mean(covs)
        Y = engine._log_map_all(covs, mu)  # [N, p]

        # Design matrix
        X = np.ones((N, k + 1))
        for j, label in enumerate(forcing_labels):
            lag = lag_months.get(label, 0)
            X[:, j + 1] = engine._align_forcing(
                data, forcing_groups[label], center_times, lag
            )
        forcing_stds = X[:, 1:].std(axis=0)
        forcing_stds[forcing_stds < 1e-10] = 1.0
        X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / forcing_stds

        # Contiguous temporal folds
        fold_size = N // n_folds
        folds = []
        for f in range(n_folds):
            start = f * fold_size
            end = start + fold_size if f < n_folds - 1 else N
            folds.append(np.arange(start, end))

        scores: Dict[float, float] = {}
        for alpha_val in alphas:
            fold_errors = []
            for f_idx in range(n_folds):
                test_idx = folds[f_idx]
                train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != f_idx])

                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]

                # Optional PCA on training data
                if n_components is not None and n_components < p:
                    Y_mean_tr = Y_train.mean(axis=0)
                    Y_c_tr = Y_train - Y_mean_tr
                    _, S_tr, Vt_tr = np.linalg.svd(Y_c_tr, full_matrices=False)
                    nc = n_components
                    proj_tr = Vt_tr[:nc].T  # [p, nc]
                    Y_train_r = Y_c_tr @ proj_tr
                    Y_test_r = (Y_test - Y_mean_tr) @ proj_tr
                else:
                    Y_train_r = Y_train
                    Y_test_r = Y_test
                    proj_tr = None

                # Ridge fit
                if alpha_val > 0:
                    XtX = X_train.T @ X_train
                    pen = alpha_val * np.eye(k + 1)
                    pen[0, 0] = 0.0
                    B_cv = np.linalg.solve(XtX + pen, X_train.T @ Y_train_r)
                else:
                    B_cv, _, _, _ = np.linalg.lstsq(X_train, Y_train_r, rcond=None)

                Y_pred = X_test @ B_cv

                # Error in reduced or full space
                fold_errors.append(np.mean((Y_test_r - Y_pred) ** 2))

            scores[alpha_val] = float(np.mean(fold_errors))

        best_alpha = min(scores, key=scores.get)
        return best_alpha, scores


def run_tangent_regression_demo(verbose: bool = True) -> TangentRegressionResults:
    """Run tangent-space regression on synthetic data.

    Demonstrates simultaneous multivariate attribution with proper
    confounding control and phase-randomisation significance testing.
    """
    if verbose:
        print("=" * 70)
        print("TANGENT-SPACE REGRESSION — Riemannian Optimal Fingerprinting")
        print("=" * 70)

    data = generate_attribution_data(T=816, seed=42)
    response_names = [n for n in data.index_names
                      if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
    forcing_groups = {
        "solar": ["tsi", "f107", "sunspot", "aa_index"],
        "anthropogenic": ["co2", "ch4"],
        "volcanic": ["aod"],
    }
    forcing_groups = {
        k: [n for n in v if n in data.index_names]
        for k, v in forcing_groups.items()
    }
    forcing_groups = {k: v for k, v in forcing_groups.items() if v}

    if verbose:
        print(f"\nData: {data.T} months × {data.d} indices")
        print(f"Response ({len(response_names)}): {response_names}")
        print(f"Forcings: {forcing_groups}")

    engine = TangentSpaceRegression(
        window=60, step=3, shrinkage=0.15,
        n_surrogates=200,
    )
    results = engine.regress(
        data, forcing_groups, response_names,
        lag_months={"solar": 6, "anthropogenic": 12, "volcanic": 3},
        test_surrogates=verbose,
    )

    if verbose:
        print(f"\n{'=' * 70}")
        print("RESULTS")
        print(f"{'=' * 70}")
        print(f"\nModel: {results.n_timepoints} covariance matrices, "
              f"{results.d_climate}D climate → {results.d_tangent}D tangent space")
        print(f"R² = {results.r_squared:.4f} (adjusted = {results.adjusted_r_squared:.4f})")

        print(f"\nRegression coefficients (β):")
        for label, c in results.coefficients.items():
            p_str = ""
            if c.p_value_surrogate is not None:
                p_str = f"p_surr={c.p_value_surrogate:.3f}"
            else:
                p_str = f"p_F={c.p_value_parametric:.3f}"
            print(f"\n  {label}:")
            print(f"    ||β|| = {c.norm:.4f}  F = {c.f_statistic:.2f}  {p_str}")
            print(f"    V+ = {c.v_plus:.4f}  V- = {c.v_minus:.4f}  "
                  f"ratio = {c.v_ratio:.3f}")
            print(f"    Partial R² = {c.explained_variance:.4f}")
            print(f"    {c.mechanism}")

        print(f"\nAttribution:")
        for label, frac in sorted(results.attribution_fractions.items(),
                                   key=lambda x: -x[1]):
            print(f"  {label:>15s}: {frac:>6.1%}")

        print(f"\nInterpretation:")
        for line in results.interpretation:
            print(f"  • {line}")

    return results


# =====================================================================
# Demo with synthetic + real data
# =====================================================================

def generate_attribution_data(T: int = 816, seed: int = 42) -> ClimateData:
    """Generate synthetic data with solar + anthropogenic + volcanic forcing.

    Returns a ClimateData with 12 climate + 2 solar + 3 anthropogenic = 17D.
    """
    # Start with solar+climate data (16D)
    solar_data = SolarDataLoader.generate_synthetic(T=T, seed=seed)

    # Generate anthropogenic series
    anthro = generate_anthropogenic_series(T, start_year=1956, seed=seed + 1)

    # Add anthropogenic coupling to climate
    climate_cols = [i for i, n in enumerate(solar_data.index_names)
                    if n not in SOLAR_INDICES]
    values = solar_data.values.copy()

    t = np.arange(T)
    years = 1956 + t / 12

    for ti in range(T):
        # CO2 → global warming trend (V+ effect)
        # Anthropogenic forcing dominates real-world attribution:
        # ~2.5 W/m² vs ~0.1 W/m² volcanic (average), ~0.1 W/m² solar
        co2_effect = 0.08 * anthro["co2"][ti]
        for ci in climate_cols:
            values[ti, ci] += co2_effect

        # CH4 → additional warming (correlated with CO2 but distinct)
        ch4_effect = 0.03 * anthro["ch4"][ti]
        for ci in climate_cols:
            values[ti, ci] += ch4_effect

        # CO2 → enhanced ENSO variability (V- effect, post-1990)
        # Earlier onset than previous 2000 cutoff — warming-driven
        # teleconnection changes are detectable from ~1990
        if years[ti] > 1990:
            for ci in climate_cols[:5]:  # ENSO indices
                values[ti, ci] *= 1.0 + 0.04 * max(0, anthro["co2"][ti])

        # CO2 → PDO/AMO amplitude increase (V- effect)
        if years[ti] > 1985:
            for ci in climate_cols[5:8]:  # PDO, AMO, DMI
                values[ti, ci] += 0.02 * anthro["co2"][ti]

        # AOD → global cooling spikes (V+ effect)
        aod_effect = -0.3 * max(0, anthro["aod"][ti])
        for ci in climate_cols:
            values[ti, ci] += aod_effect

        # AOD → disrupted ENSO (V- effect during volcanic events)
        if anthro["aod"][ti] > 1.0:
            for ci in climate_cols[:5]:
                values[ti, ci] *= 0.7

    # Combine: 12 climate + 4 solar + 3 anthropogenic
    anthro_values = np.column_stack([anthro[k] for k in ANTHROPOGENIC_INDICES])
    combined = np.hstack([values, anthro_values])
    combined_names = solar_data.index_names + ANTHROPOGENIC_INDICES

    return ClimateData(
        values=combined,
        dates=solar_data.dates,
        index_names=combined_names,
        metadata={
            "synthetic": True,
            "seed": seed,
            "anthropogenic_coupling": True,
            "description": "17D climate+solar+anthropogenic synthetic data",
        },
    )


def run_attribution_demo(verbose: bool = True) -> AttributionResults:
    """Run full attribution analysis on synthetic data.

    Demonstrates separating solar, anthropogenic, and volcanic forcing
    signatures using Riemannian geometry on SPD(d).
    """
    if verbose:
        print("=" * 70)
        print("CLIMATE FORCING ATTRIBUTION — Riemannian SPD Analysis")
        print("=" * 70)

    # Generate data
    data = generate_attribution_data(T=816, seed=42)
    if verbose:
        print(f"\nData: {data.T} months × {data.d} indices "
              f"({data.start_year}–{data.end_year})")
        solar_in = [n for n in data.index_names if n in SOLAR_INDICES]
        anthro_in = [n for n in data.index_names if n in ANTHROPOGENIC_INDICES]
        climate_in = [n for n in data.index_names
                      if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
        print(f"  Climate ({len(climate_in)}): {climate_in}")
        print(f"  Solar ({len(solar_in)}):   {solar_in}")
        print(f"  Anthro ({len(anthro_in)}):  {anthro_in}")

    # Define forcing groups and response
    response_names = [n for n in data.index_names
                      if n not in SOLAR_INDICES and n not in ANTHROPOGENIC_INDICES]
    forcing_groups = {
        "solar": [n for n in data.index_names if n in SOLAR_INDICES],
        "anthropogenic": ["co2", "ch4"],
        "volcanic": ["aod"],
    }

    # Only include groups that exist in data
    forcing_groups = {
        k: [n for n in v if n in data.index_names]
        for k, v in forcing_groups.items()
    }
    forcing_groups = {k: v for k, v in forcing_groups.items() if v}

    if verbose:
        print(f"\nForcing groups: {forcing_groups}")
        print(f"Response indices: {response_names}")

    # Run attribution
    engine = ClimateAttribution(
        window=60, shrinkage=0.15,
        n_permutations=100,  # fewer for demo speed
    )
    results = engine.attribute(
        data, forcing_groups, response_names,
        lag_months={"solar": 6, "anthropogenic": 12, "volcanic": 3},
        split_year=1998,
        test_significance=verbose,  # skip permutations if not verbose
    )

    if verbose:
        print(f"\n{'=' * 70}")
        print("RESULTS")
        print(f"{'=' * 70}")

        print(f"\nObserved covariance trend (early vs late):")
        print(f"  Geodesic distance: {results.observed_geodesic:.4f}")

        print(f"\nForcing fingerprints:")
        for label, fp in results.fingerprints.items():
            print(f"\n  {label}:")
            print(f"    Geodesic distance:  {fp.geodesic_distance:.4f}")
            print(f"    V+ (energy budget): {fp.v_plus:.4f}")
            print(f"    V- (teleconnect.):  {fp.v_minus:.4f}")
            print(f"    V+ ratio:           {fp.v_ratio:.3f}")
            print(f"    Mechanism:          {fp.mechanism}")
            print(f"    Partition:          {fp.n_high} high / {fp.n_low} low")

        print(f"\nAttribution fractions:")
        for label, frac in sorted(results.attribution_fractions.items(),
                                   key=lambda x: -x[1]):
            sig = ""
            if label in results.p_values:
                sig = f"  (p = {results.p_values[label]:.3f})"
            print(f"  {label:>15s}: {frac:>6.1%}  "
                  f"(projection = {results.projections[label]:+.3f}){sig}")
        print(f"  {'residual':>15s}: {results.residual_fraction:>6.1%}")

        print(f"\nDominant forcing: {results.dominant_forcing}")

        print(f"\nInterpretation:")
        for line in results.interpretation:
            print(f"  • {line}")

    return results
