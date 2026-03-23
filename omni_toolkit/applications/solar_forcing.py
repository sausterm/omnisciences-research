"""
Solar & orbital forcing analysis on the SPD covariance manifold.

Extends climate regime detection to include external forcing factors:
solar irradiance (TSI), solar UV proxy (F10.7), sunspot number, and
geomagnetic activity (aa index). Tracks how solar activity reshapes
climate teleconnections using geodesic distance on SPD(d+k).

Key insight: Solar forcing doesn't just shift mean temperatures — it
modulates the *covariance structure* of climate indices. During solar
minima, NAO-AO coupling weakens and ENSO statistics shift. V+/V-
decomposition separates radiative forcing (trace = energy budget)
from teleconnection reorganisation (traceless = coupling changes).

Phase 1: Solar cycle forcing (11-yr, monthly resolution)
    - TSI (Total Solar Irradiance) from SORCE/TSIS-1
    - F10.7 solar radio flux (UV proxy)
    - Sunspot number (SILSO)
    - Geomagnetic aa index (solar wind)

Phase 2 (future): Milankovitch orbital forcing (paleoclimate)
    - Obliquity, eccentricity, precession
    - Requires ~1kyr rolling windows on proxy records

References:
    - Gray et al. (2010). Solar influences on climate. Rev. Geophys.
    - Lean & Rind (2008). How natural and anthropogenic influences
      alter global and regional surface temperatures. Geophys. Res. Lett.
    - Meehl et al. (2009). Amplifying the Pacific climate system response
      to a small 11-year solar cycle forcing. Science.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .climate_analysis import (
    ClimateData, ClimateDataLoader, ClimateRegimeDetector,
    DetectionResults, INDEX_REGISTRY, INDICES_12,
    rolling_covariance_climate,
)
from .spd_ml import _matrix_log, _matrix_sqrt, _matrix_invsqrt, _symmetrize


# =====================================================================
# Solar forcing index registry
# =====================================================================

SOLAR_REGISTRY = {
    "tsi": {
        "name": "Total Solar Irradiance",
        "group": "solar",
        "source": "LASP/SORCE/TSIS-1",
        "unit": "W/m²",
        "description": "Disk-integrated solar radiative output. Varies ~1.4 W/m² "
                        "over 11-yr cycle. Primary driver of radiative forcing.",
    },
    "f107": {
        "name": "F10.7 Solar Radio Flux",
        "group": "solar",
        "source": "NRC Canada / NOAA SWPC",
        "unit": "SFU (10⁻²² W/m²/Hz)",
        "description": "10.7 cm radio emission from solar corona. Proxy for solar "
                        "UV and EUV that modulates stratospheric ozone. Strong 11-yr cycle.",
    },
    "sunspot": {
        "name": "Sunspot Number",
        "group": "solar",
        "source": "SILSO/Royal Observatory Belgium",
        "unit": "count",
        "description": "International sunspot number v2.0. Most direct measure "
                        "of solar magnetic activity. 11-yr Schwabe cycle.",
    },
    "aa_index": {
        "name": "Geomagnetic aa Index",
        "group": "solar",
        "source": "ISGI/BGS",
        "unit": "nT",
        "description": "Geomagnetic disturbance index measuring solar wind–"
                        "magnetosphere coupling. Modulates cosmic ray flux "
                        "(Forbush decreases).",
    },
}

# Combined registry
FULL_REGISTRY = {**INDEX_REGISTRY, **SOLAR_REGISTRY}

# Index sets
SOLAR_INDICES = ["tsi", "f107", "sunspot", "aa_index"]
INDICES_16 = INDICES_12 + SOLAR_INDICES

# Solar cycle events (approximate dates)
SOLAR_CYCLES = [
    {"cycle": 19, "min": (1954, 4), "max": (1958, 3), "strength": "very_strong"},
    {"cycle": 20, "min": (1964, 10), "max": (1968, 11), "strength": "moderate"},
    {"cycle": 21, "min": (1976, 3), "max": (1979, 12), "strength": "strong"},
    {"cycle": 22, "min": (1986, 9), "max": (1989, 7), "strength": "strong"},
    {"cycle": 23, "min": (1996, 5), "max": (2001, 11), "strength": "moderate"},
    {"cycle": 24, "min": (2008, 12), "max": (2014, 4), "strength": "weak"},
    {"cycle": 25, "min": (2019, 12), "max": (2024, 6), "strength": "moderate"},
]

GRAND_MINIMA = [
    # For reference — not in our data window but important context
    {"name": "Maunder Minimum", "start": 1645, "end": 1715},
    {"name": "Dalton Minimum", "start": 1790, "end": 1830},
    {"name": "Modern Maximum", "start": 1950, "end": 2000},
]

# NOAA/NASA data endpoints for solar indices
SOLAR_ENDPOINTS = {
    "f107": "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json",
    "sunspot": "https://www.sidc.be/SILSO/INFO/snmtotcsv.php",
    "tsi": "https://lasp.colorado.edu/lisird/latis/dap/sorce_tsi_24hr_l3.csv",
}


# =====================================================================
# Solar data loading
# =====================================================================

class SolarDataLoader:
    """Load solar forcing data and merge with climate indices.

    Handles fetching TSI, F10.7, sunspot number, and aa index from
    their respective data sources, normalizing to monthly anomalies,
    and aligning with the climate index time series.
    """

    @staticmethod
    def generate_synthetic(T: int = 816, seed: int = 42) -> ClimateData:
        """Generate synthetic solar+climate data with realistic coupling.

        Creates 68 years (1956-2023) of 16-dimensional data:
        - 12 climate indices (from ClimateDataLoader.generate_synthetic)
        - 4 solar forcing indices with realistic 11-yr cycle

        Solar-climate coupling implemented:
        - TSI modulates global mean (V+ component)
        - F10.7/UV modulates QBO period and NAO
        - Solar wind (aa) modulates AO coupling strength
        - Sunspot-ENSO correlation (Meehl 2009 pathway)
        """
        rng = np.random.RandomState(seed)

        # Get base climate data
        climate_data = ClimateDataLoader.generate_synthetic(T=T, d=12, seed=seed)

        # Generate solar forcing indices
        t = np.arange(T)
        years = 1956 + t // 12
        months = 1 + t % 12
        fractional_years = years + (months - 1) / 12

        # --- TSI ---
        # 11-yr cycle with ~1.4 W/m² amplitude, plus noise
        # Mean TSI ≈ 1361 W/m², but we store as anomaly
        solar_phase = 2 * np.pi * (fractional_years - 1954.3) / 10.8
        tsi_cycle = 0.7 * np.sin(solar_phase)  # ±0.7 W/m² (half amplitude)
        # Add longer-period modulation (Gleissberg cycle ~88yr)
        gleissberg = 0.2 * np.sin(2 * np.pi * fractional_years / 88)
        tsi = tsi_cycle + gleissberg + rng.normal(0, 0.1, T)

        # --- F10.7 ---
        # Correlates with TSI but more variable, range ~70-250 SFU
        # Store as normalised anomaly
        f107_raw = 120 + 60 * np.sin(solar_phase) + 20 * gleissberg
        f107_raw += rng.normal(0, 12, T)
        f107_raw = np.clip(f107_raw, 60, 300)
        f107 = (f107_raw - 120) / 40  # normalise to ~[-1.5, 4.5]

        # --- Sunspot number ---
        # Asymmetric cycle: fast rise, slow decay
        # Approximate with rectified sinusoid
        ssn_phase = np.sin(solar_phase)
        ssn_asymmetry = np.where(ssn_phase > 0,
                                  np.abs(ssn_phase) ** 0.7,
                                  -np.abs(ssn_phase) ** 1.5)
        ssn_raw = 80 + 70 * ssn_asymmetry + 15 * gleissberg
        ssn_raw += rng.normal(0, 15, T)
        ssn_raw = np.clip(ssn_raw, 0, 300)
        sunspot = (ssn_raw - 80) / 50  # normalise

        # --- Geomagnetic aa index ---
        # Peaks at solar declining phase (lag ~2-3 yrs after solar max)
        aa_phase = 2 * np.pi * (fractional_years - 1954.3 - 2.5) / 10.8
        aa_raw = 20 + 10 * np.sin(aa_phase) + rng.normal(0, 4, T)
        aa_raw = np.clip(aa_raw, 5, 60)
        aa_index = (aa_raw - 20) / 10  # normalise

        solar_values = np.column_stack([tsi, f107, sunspot, aa_index])

        # --- Solar-climate coupling ---
        # Modulate climate indices based on solar state
        climate_values = climate_data.values.copy()

        for ti in range(T):
            solar_state = tsi[ti]  # positive = solar max

            # 1. TSI → global mean temp (weak: ~0.1°C over cycle)
            if climate_values.shape[1] > 12:
                climate_values[ti, 12] += 0.05 * solar_state

            # 2. F10.7/UV → QBO modulation (stratospheric pathway)
            #    During solar max, QBO period slightly shorter
            if 11 < climate_values.shape[1]:
                climate_values[ti, 11] += 2.0 * f107[ti] * 0.05

            # 3. Solar → NAO (via stratosphere)
            #    Solar max favours positive NAO in winter
            if 8 < climate_values.shape[1]:
                mo = months[ti]
                if mo in (12, 1, 2):  # winter only
                    climate_values[ti, 8] += 0.15 * solar_state

            # 4. Solar wind → AO (cosmic ray / ozone pathway)
            if 10 < climate_values.shape[1]:
                climate_values[ti, 10] += 0.1 * aa_index[ti]

            # 5. Solar → ENSO modulation (Meehl et al. 2009 pathway)
            #    Solar max enhances La Niña-like cooling in tropical Pacific
            #    via "top-down" stratospheric and "bottom-up" ocean coupling
            for ci in range(min(5, climate_values.shape[1])):
                climate_values[ti, ci] -= 0.08 * solar_state

            # 6. Solar → PDO (decadal modulation via ENSO integration)
            if 5 < climate_values.shape[1]:
                climate_values[ti, 5] -= 0.06 * solar_state

        # Combine climate + solar
        combined = np.hstack([climate_values, solar_values])
        combined_names = climate_data.index_names + SOLAR_INDICES

        return ClimateData(
            values=combined,
            dates=climate_data.dates,
            index_names=combined_names,
            metadata={
                "synthetic": True,
                "seed": seed,
                "solar_coupling": True,
                "description": "16D climate+solar synthetic data with "
                               "realistic 11-yr cycle and solar-climate coupling",
            },
        )

    @staticmethod
    def from_arrays(climate_data: ClimateData,
                    solar_values: np.ndarray,
                    solar_names: Optional[List[str]] = None) -> ClimateData:
        """Merge climate and solar data arrays.

        Args:
            climate_data: existing ClimateData (12D).
            solar_values: [T, k] array of solar indices.
            solar_names: list of k solar index keys.
        """
        if solar_names is None:
            solar_names = SOLAR_INDICES[:solar_values.shape[1]]

        assert solar_values.shape[0] == climate_data.T, \
            f"Length mismatch: climate={climate_data.T}, solar={solar_values.shape[0]}"

        combined = np.hstack([climate_data.values, solar_values])
        return ClimateData(
            values=combined,
            dates=climate_data.dates,
            index_names=climate_data.index_names + solar_names,
            metadata={**climate_data.metadata, "solar_merged": True},
        )


# =====================================================================
# Solar-conditioned analysis
# =====================================================================

@dataclass
class SolarClimateResults:
    """Results from solar-conditioned regime detection."""
    # Full 16D detection
    full_results: DetectionResults
    # Climate-only 12D detection (for comparison)
    climate_only_results: DetectionResults
    # Solar max vs min comparison
    solar_max_cov: np.ndarray       # mean covariance during solar max
    solar_min_cov: np.ndarray       # mean covariance during solar min
    solar_geodesic: float           # geodesic distance between max/min
    # V+/V- decomposition of solar effect
    solar_v_plus: float             # trace component (energy budget)
    solar_v_minus: float            # traceless component (coupling changes)
    # Per-cycle statistics
    cycle_stats: List[Dict]
    # Metadata
    solar_indices_used: List[str] = field(default_factory=list)


class SolarClimateAnalyzer:
    """Analyse solar forcing effects on climate covariance structure.

    Compares SPD geometry during solar max vs. solar min periods to
    measure how much solar activity reshapes climate teleconnections.
    Uses V+/V- decomposition to separate radiative forcing (V+) from
    teleconnection reorganisation (V-).
    """

    def __init__(self, window: int = 72, threshold: float = 2.0,
                 shrinkage: float = 0.1):
        self.window = window
        self.threshold = threshold
        self.shrinkage = shrinkage
        self.detector = ClimateRegimeDetector(
            window=window, threshold=threshold, shrinkage=shrinkage
        )

    def analyse(self, data: ClimateData) -> SolarClimateResults:
        """Run full solar-climate analysis.

        Args:
            data: 16D ClimateData with both climate and solar indices.

        Returns:
            SolarClimateResults with detection, conditioning, and V+/V-.
        """
        # Identify solar indices in the data
        solar_idx = []
        solar_names = []
        climate_names = []
        for i, name in enumerate(data.index_names):
            if name in SOLAR_REGISTRY:
                solar_idx.append(i)
                solar_names.append(name)
            else:
                climate_names.append(name)

        if not solar_idx:
            raise ValueError("No solar indices found in data. "
                             f"Expected some of: {SOLAR_INDICES}")

        # 1. Full 16D detection
        full_results = self.detector.detect(data)

        # 2. Climate-only detection (for comparison)
        climate_data = data.select_indices(climate_names)
        climate_only = self.detector.detect(climate_data)

        # 3. Solar-conditioned covariance comparison
        # Classify each month as solar max or min using sunspot/TSI
        solar_classifier_idx = None
        for name in ("sunspot", "tsi", "f107"):
            if name in data.index_names:
                solar_classifier_idx = data.index_names.index(name)
                break

        if solar_classifier_idx is None:
            # Fallback: use first solar index
            solar_classifier_idx = solar_idx[0]

        solar_series = data.values[:, solar_classifier_idx]
        solar_median = np.median(solar_series)

        # Compute covariance matrices for solar max and min periods
        # (using climate indices only, to see how solar affects climate)
        max_mask = solar_series > solar_median + 0.3 * np.std(solar_series)
        min_mask = solar_series < solar_median - 0.3 * np.std(solar_series)

        climate_vals = data.select_indices(climate_names).values
        d_clim = climate_vals.shape[1]

        # Covariance during solar max
        max_data = climate_vals[max_mask]
        if len(max_data) > d_clim + 5:
            C_max = np.cov(max_data, rowvar=False)
            C_max = _symmetrize(C_max)
            eigvals = np.linalg.eigvalsh(C_max)
            if eigvals[0] <= 0:
                C_max += (abs(eigvals[0]) + 1e-8) * np.eye(d_clim)
        else:
            C_max = np.eye(d_clim)

        # Covariance during solar min
        min_data = climate_vals[min_mask]
        if len(min_data) > d_clim + 5:
            C_min = np.cov(min_data, rowvar=False)
            C_min = _symmetrize(C_min)
            eigvals = np.linalg.eigvalsh(C_min)
            if eigvals[0] <= 0:
                C_min += (abs(eigvals[0]) + 1e-8) * np.eye(d_clim)
        else:
            C_min = np.eye(d_clim)

        # Geodesic distance between solar max and min covariance
        A_invsqrt = _matrix_invsqrt(C_max)
        M = A_invsqrt @ C_min @ A_invsqrt
        M = _symmetrize(M)
        logM = _matrix_log(M)
        solar_geodesic = np.linalg.norm(logM, 'fro')

        # V+/V- decomposition of the solar effect
        # V+ (trace) = volume change = energy budget shift
        # V- (traceless) = shape change = coupling reorganisation
        tangent = logM
        trace_part = np.trace(tangent) / d_clim
        v_plus = abs(trace_part) * np.sqrt(d_clim)  # normalised
        tangent_traceless = tangent - trace_part * np.eye(d_clim)
        v_minus = np.linalg.norm(tangent_traceless, 'fro')

        # 4. Per-cycle statistics
        cycle_stats = []
        for cycle in SOLAR_CYCLES:
            min_yr, min_mo = cycle["min"]
            max_yr, max_mo = cycle["max"]

            # Check if this cycle overlaps our data
            if max_yr < data.start_year or min_yr > data.end_year:
                continue

            # Count transitions near solar max/min
            transitions_near_max = 0
            transitions_near_min = 0

            for tr in full_results.transitions:
                tr_yr, tr_mo = tr.date
                # Within 1 year of solar max
                dt_max = abs((tr_yr - max_yr) + (tr_mo - max_mo) / 12)
                dt_min = abs((tr_yr - min_yr) + (tr_mo - min_mo) / 12)

                if dt_max < 1.5:
                    transitions_near_max += 1
                if dt_min < 1.5:
                    transitions_near_min += 1

            cycle_stats.append({
                "cycle": cycle["cycle"],
                "strength": cycle["strength"],
                "solar_min": cycle["min"],
                "solar_max": cycle["max"],
                "transitions_near_max": transitions_near_max,
                "transitions_near_min": transitions_near_min,
            })

        return SolarClimateResults(
            full_results=full_results,
            climate_only_results=climate_only,
            solar_max_cov=C_max,
            solar_min_cov=C_min,
            solar_geodesic=solar_geodesic,
            solar_v_plus=v_plus,
            solar_v_minus=v_minus,
            cycle_stats=cycle_stats,
            solar_indices_used=solar_names,
        )


# =====================================================================
# Causal geometry: forcing → response
# =====================================================================

def forcing_response_geodesic(data: ClimateData,
                              forcing_names: List[str],
                              response_names: List[str],
                              window: int = 72,
                              lag_months: int = 0,
                              shrinkage: float = 0.1) -> Dict:
    """Compute geodesic distance between forcing-conditioned response covariances.

    This measures how much a forcing factor reshapes the covariance structure
    of the response variables. Coordinate-free, no regression assumptions.

    Method:
        1. Split data into high/low forcing periods (above/below median)
        2. Compute response covariance C_high and C_low
        3. Geodesic distance d(C_high, C_low) = ||log(C_high^{-1/2} C_low C_high^{-1/2})||_F
        4. V+/V- decomposition:
           - V+ = trace change = forcing shifts overall variance (energy budget)
           - V- = traceless change = forcing reshapes correlations (teleconnections)

    Args:
        data: ClimateData with forcing and response indices.
        forcing_names: list of forcing index keys.
        response_names: list of response index keys.
        window: rolling window (unused for static version, kept for API compat).
        lag_months: lag between forcing and response (positive = forcing leads).
        shrinkage: Ledoit-Wolf shrinkage.

    Returns:
        dict with geodesic_distance, v_plus, v_minus, v_ratio,
        C_high, C_low, and interpretation.
    """
    # Get forcing and response series
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

    # Composite forcing index (first PC if multiple)
    if forcing.shape[1] > 1:
        # Use first PC
        forcing_centered = forcing - forcing.mean(axis=0)
        _, _, Vt = np.linalg.svd(forcing_centered, full_matrices=False)
        composite = forcing_centered @ Vt[0]
    else:
        composite = forcing[:, 0]

    # Split into high/low forcing
    median = np.median(composite)
    std = np.std(composite)
    high_mask = composite > median + 0.3 * std
    low_mask = composite < median - 0.3 * std

    # Covariance during high forcing
    resp_high = response[high_mask]
    resp_low = response[low_mask]

    if len(resp_high) < d_resp + 5 or len(resp_low) < d_resp + 5:
        return {
            "geodesic_distance": 0.0,
            "v_plus": 0.0,
            "v_minus": 0.0,
            "v_ratio": 0.0,
            "error": "Insufficient data in high/low partitions",
        }

    C_high = np.cov(resp_high, rowvar=False)
    C_low = np.cov(resp_low, rowvar=False)

    # Shrinkage
    if shrinkage > 0:
        mu_h = np.trace(C_high) / d_resp
        mu_l = np.trace(C_low) / d_resp
        C_high = (1 - shrinkage) * C_high + shrinkage * mu_h * np.eye(d_resp)
        C_low = (1 - shrinkage) * C_low + shrinkage * mu_l * np.eye(d_resp)

    # Ensure SPD
    for C in (C_high, C_low):
        C[:] = _symmetrize(C)
        eigvals = np.linalg.eigvalsh(C)
        if eigvals[0] <= 0:
            C += (abs(eigvals[0]) + 1e-8) * np.eye(d_resp)

    # Geodesic distance
    A_invsqrt = _matrix_invsqrt(C_high)
    M = A_invsqrt @ C_low @ A_invsqrt
    M = _symmetrize(M)
    logM = _matrix_log(M)
    geodesic = np.linalg.norm(logM, 'fro')

    # V+/V- decomposition
    trace_part = np.trace(logM) / d_resp
    v_plus = abs(trace_part) * np.sqrt(d_resp)
    traceless = logM - trace_part * np.eye(d_resp)
    v_minus = np.linalg.norm(traceless, 'fro')

    v_ratio = v_plus / (v_plus + v_minus) if (v_plus + v_minus) > 1e-10 else 0.5

    interpretation = []
    if v_ratio > 0.6:
        interpretation.append("Forcing primarily shifts variance (energy budget)")
    elif v_ratio < 0.3:
        interpretation.append("Forcing primarily reshapes correlations (teleconnections)")
    else:
        interpretation.append("Forcing affects both variance and correlations")

    if geodesic > 1.0:
        interpretation.append("Strong forcing effect on covariance structure")
    elif geodesic > 0.3:
        interpretation.append("Moderate forcing effect")
    else:
        interpretation.append("Weak forcing effect")

    return {
        "geodesic_distance": float(geodesic),
        "v_plus": float(v_plus),
        "v_minus": float(v_minus),
        "v_ratio": float(v_ratio),
        "C_high": C_high,
        "C_low": C_low,
        "n_high": int(high_mask.sum()),
        "n_low": int(low_mask.sum()),
        "lag_months": lag_months,
        "forcing_names": forcing_names,
        "response_names": response_names,
        "interpretation": interpretation,
    }


# =====================================================================
# Demo / validation
# =====================================================================

def run_solar_demo(verbose: bool = True) -> SolarClimateResults:
    """Run the full solar-climate analysis demo.

    Generates synthetic 16D data, runs both full and climate-only detection,
    computes solar-conditioned covariances, and reports results.
    """
    if verbose:
        print("=" * 70)
        print("Solar-Climate Forcing Analysis — Riemannian Regime Detection")
        print("=" * 70)

    # Generate synthetic data
    data = SolarDataLoader.generate_synthetic(T=816, seed=42)
    if verbose:
        print(f"\nData: {data.T} months × {data.d} indices "
              f"({data.start_year}–{data.end_year})")
        print(f"  Climate: {[n for n in data.index_names if n not in SOLAR_INDICES]}")
        print(f"  Solar:   {[n for n in data.index_names if n in SOLAR_INDICES]}")

    # Run analysis
    analyzer = SolarClimateAnalyzer(window=72, threshold=2.0, shrinkage=0.1)
    results = analyzer.analyse(data)

    if verbose:
        print(f"\n--- Full 16D Detection (climate + solar) ---")
        print(f"  Transitions detected: {len(results.full_results.transitions)}")
        print(f"  Mean geodesic dist:   {results.full_results.geodesic_distances.mean():.4f}")

        print(f"\n--- Climate-only 12D Detection ---")
        print(f"  Transitions detected: {len(results.climate_only_results.transitions)}")
        print(f"  Mean geodesic dist:   {results.climate_only_results.geodesic_distances.mean():.4f}")

        print(f"\n--- Solar Max vs Min Comparison ---")
        print(f"  Geodesic distance:    {results.solar_geodesic:.4f}")
        print(f"  V+ (energy budget):   {results.solar_v_plus:.4f}")
        print(f"  V- (teleconnections): {results.solar_v_minus:.4f}")
        ratio = results.solar_v_plus / (results.solar_v_plus + results.solar_v_minus) \
            if (results.solar_v_plus + results.solar_v_minus) > 1e-10 else 0.5
        print(f"  V+/(V++V-) ratio:     {ratio:.3f}")
        if ratio > 0.5:
            print(f"  → Solar forcing primarily shifts overall variance (radiative)")
        else:
            print(f"  → Solar forcing primarily reshapes correlations (teleconnections)")

        # Per-cycle stats
        if results.cycle_stats:
            print(f"\n--- Per Solar Cycle ---")
            for cs in results.cycle_stats:
                print(f"  Cycle {cs['cycle']} ({cs['strength']}): "
                      f"{cs['transitions_near_max']} transitions near max, "
                      f"{cs['transitions_near_min']} near min")

        # Forcing-response analysis
        print(f"\n--- Forcing → Response Geodesics ---")

        # TSI → all climate indices
        climate_names = [n for n in data.index_names if n not in SOLAR_INDICES]
        for forcing in ["tsi", "f107", "sunspot"]:
            if forcing in data.index_names:
                fr = forcing_response_geodesic(
                    data, [forcing], climate_names, lag_months=3
                )
                print(f"  {forcing:8s} → climate (lag=3mo): "
                      f"d={fr['geodesic_distance']:.4f}  "
                      f"V+={fr['v_plus']:.4f}  V-={fr['v_minus']:.4f}  "
                      f"({fr['interpretation'][0]})")

    return results
