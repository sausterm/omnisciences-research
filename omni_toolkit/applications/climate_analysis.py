"""
Riemannian climate regime detection on the SPD covariance manifold.

Monitors geodesic distance between rolling covariance matrices of climate
indices to detect regime transitions (ENSO, PDO, AMO, volcanic events, etc.).

Key insight: Climate indices are correlated, and their *correlation structure*
shifts during regime transitions. By tracking covariance matrices on the SPD
manifold GL+(d)/SO(d), we detect these structural shifts with affine-invariant
sensitivity — a 2x variance change is equally detectable at any base level.

Supported indices (12-15 dimensional):
    Core ENSO:     Nino3.4, SOI, Nino1+2, Nino3, Nino4
    Oceanic:       PDO, AMO, DMI (Indian Ocean Dipole)
    Atmospheric:   NAO, PNA, AO (Arctic Oscillation), QBO
    Global:        Global Mean Temperature anomaly

Usage:
    from omni_toolkit.applications.climate_analysis import (
        ClimateRegimeDetector, ClimateDataLoader, VDecomposition
    )

    loader = ClimateDataLoader()
    data = loader.load_all()  # or loader.from_csv("path/to/data.csv")

    detector = ClimateRegimeDetector(window=72, step=1, threshold=2.0)
    results = detector.detect(data)

    # V+/V- decomposition for event fingerprinting
    vd = VDecomposition(results)
    fingerprints = vd.classify_events()

References:
    - Austermann (2026). Riemannian Regime Detection for Climate Science.
    - Pennec, Fillard, Ayache (2006). A Riemannian Framework for Tensor Computing.
    - Said et al. (2017). Riemannian Gaussian distributions on SPD matrices.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

from .spd_ml import (
    SPDLayer, _matrix_log, _matrix_sqrt, _matrix_invsqrt,
    _matrix_sqrt_pair, _symmetrize,
)


# =====================================================================
# Climate index registry
# =====================================================================

# Canonical ordering of indices. Subset selection uses this order.
INDEX_REGISTRY = {
    # Core ENSO
    "nino34":   {"name": "Niño 3.4",           "group": "enso",  "source": "NOAA/CPC"},
    "soi":      {"name": "SOI",                 "group": "enso",  "source": "NOAA/CPC"},
    "nino12":   {"name": "Niño 1+2",            "group": "enso",  "source": "NOAA/CPC"},
    "nino3":    {"name": "Niño 3",              "group": "enso",  "source": "NOAA/CPC"},
    "nino4":    {"name": "Niño 4",              "group": "enso",  "source": "NOAA/CPC"},
    # Oceanic
    "pdo":      {"name": "PDO",                 "group": "oceanic", "source": "NOAA/NCEI"},
    "amo":      {"name": "AMO",                 "group": "oceanic", "source": "NOAA/ESRL"},
    "dmi":      {"name": "DMI (IOD)",           "group": "oceanic", "source": "NOAA/PSL"},
    # Atmospheric
    "nao":      {"name": "NAO",                 "group": "atmos",   "source": "NOAA/CPC"},
    "pna":      {"name": "PNA",                 "group": "atmos",   "source": "NOAA/CPC"},
    "ao":       {"name": "AO",                  "group": "atmos",   "source": "NOAA/CPC"},
    "qbo":      {"name": "QBO",                 "group": "atmos",   "source": "FUB/NOAA"},
    # Global
    "gmta":     {"name": "Global Mean Temp",    "group": "global",  "source": "NOAA/NCEI"},
}

# Default 6-index set (original paper)
INDICES_6 = ["nino34", "soi", "pdo", "amo", "nao", "pna"]

# Expanded 12-index set
INDICES_12 = [
    "nino34", "soi", "nino12", "nino3", "nino4",
    "pdo", "amo", "dmi",
    "nao", "pna", "ao", "qbo",
]

# Full 13-index set (with global temperature)
INDICES_13 = INDICES_12 + ["gmta"]


# =====================================================================
# Known climate events for validation
# =====================================================================

# Major ENSO events (year, month of onset, type, strength)
ENSO_EVENTS = [
    (1957, 4, "nino", "strong"),
    (1965, 5, "nino", "moderate"),
    (1968, 10, "nino", "moderate"),
    (1972, 5, "nino", "strong"),
    (1976, 9, "nina", "moderate"),
    (1982, 4, "nino", "very_strong"),
    (1986, 9, "nino", "moderate"),
    (1988, 5, "nina", "strong"),
    (1991, 5, "nino", "moderate"),
    (1994, 9, "nino", "moderate"),
    (1997, 5, "nino", "very_strong"),
    (1998, 7, "nina", "strong"),
    (1999, 5, "nina", "moderate"),
    (2002, 5, "nino", "moderate"),
    (2004, 7, "nino", "weak"),
    (2006, 9, "nino", "weak"),
    (2007, 8, "nina", "moderate"),
    (2009, 7, "nino", "moderate"),
    (2010, 7, "nina", "strong"),
    (2014, 10, "nino", "weak"),
    (2015, 3, "nino", "very_strong"),
    (2017, 10, "nina", "weak"),
    (2020, 8, "nina", "moderate"),
    (2023, 5, "nino", "strong"),
]

# Major volcanic eruptions affecting climate
VOLCANIC_EVENTS = [
    (1963, 3, "Agung"),
    (1982, 4, "El Chichón"),
    (1991, 6, "Pinatubo"),
]

# PDO phase shifts
PDO_SHIFTS = [
    (1977, 1, "positive"),   # Great Pacific Climate Shift
    (1998, 1, "negative"),
    (2014, 1, "positive"),
]

# AMO phase shifts
AMO_SHIFTS = [
    (1965, 1, "negative"),
    (1995, 1, "positive"),
]


# =====================================================================
# Data loading
# =====================================================================

@dataclass
class ClimateData:
    """Container for multi-index climate time series.

    Attributes:
        values: [T, d] array of monthly index values (anomalies).
        dates: [T] array of (year, month) tuples.
        index_names: list of d index keys (from INDEX_REGISTRY).
        metadata: dict with source info, date range, etc.
    """
    values: np.ndarray
    dates: List[Tuple[int, int]]
    index_names: List[str]
    metadata: Dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.values.shape[0]

    @property
    def d(self) -> int:
        return self.values.shape[1]

    @property
    def start_year(self) -> int:
        return self.dates[0][0]

    @property
    def end_year(self) -> int:
        return self.dates[-1][0]

    def select_indices(self, keys: List[str]) -> "ClimateData":
        """Return a ClimateData with only the specified indices."""
        idx = [self.index_names.index(k) for k in keys]
        return ClimateData(
            values=self.values[:, idx],
            dates=self.dates,
            index_names=keys,
            metadata={**self.metadata, "selected_from": self.index_names},
        )

    def date_to_index(self, year: int, month: int) -> int:
        """Convert (year, month) to array index."""
        y0, m0 = self.dates[0]
        return (year - y0) * 12 + (month - m0)

    def slice_years(self, start: int, end: int) -> "ClimateData":
        """Return data for [start_year, end_year]."""
        i0 = self.date_to_index(start, 1)
        i1 = self.date_to_index(end, 12) + 1
        i0 = max(0, i0)
        i1 = min(self.T, i1)
        return ClimateData(
            values=self.values[i0:i1],
            dates=self.dates[i0:i1],
            index_names=self.index_names,
            metadata={**self.metadata, "sliced": (start, end)},
        )


class ClimateDataLoader:
    """Load climate index data from CSV or generate synthetic data for testing."""

    @staticmethod
    def from_csv(path: str, index_columns: Optional[Dict[str, str]] = None,
                 date_columns: Tuple[str, str] = ("year", "month")) -> ClimateData:
        """Load climate data from a CSV file.

        Args:
            path: Path to CSV file.
            index_columns: Mapping from INDEX_REGISTRY keys to CSV column names.
                           If None, assumes columns match registry keys.
            date_columns: (year_col, month_col) names in CSV.
        """
        import csv
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"Empty CSV: {path}")

        year_col, month_col = date_columns
        dates = [(int(r[year_col]), int(r[month_col])) for r in rows]

        if index_columns is None:
            # Auto-detect: use columns that match registry keys
            available = [k for k in INDEX_REGISTRY if k in rows[0]]
            index_columns = {k: k for k in available}

        index_names = list(index_columns.keys())
        values = np.array([
            [float(r.get(index_columns[k], np.nan)) for k in index_names]
            for r in rows
        ])

        # Interpolate NaNs (linear)
        for col in range(values.shape[1]):
            mask = np.isnan(values[:, col])
            if mask.any() and not mask.all():
                good = np.where(~mask)[0]
                values[mask, col] = np.interp(
                    np.where(mask)[0], good, values[good, col]
                )

        return ClimateData(
            values=values, dates=dates, index_names=index_names,
            metadata={"source": str(path), "loaded_columns": index_columns},
        )

    @staticmethod
    def from_arrays(values: np.ndarray, start_year: int, start_month: int,
                    index_names: List[str]) -> ClimateData:
        """Create ClimateData from numpy arrays.

        Args:
            values: [T, d] array.
            start_year, start_month: date of first row.
            index_names: list of d index keys.
        """
        T = values.shape[0]
        dates = []
        y, m = start_year, start_month
        for _ in range(T):
            dates.append((y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1
        return ClimateData(values=values, dates=dates, index_names=index_names)

    @staticmethod
    def generate_synthetic(T: int = 816, d: int = 12,
                           seed: int = 42) -> ClimateData:
        """Generate synthetic climate data with known regime transitions.

        Creates 68 years (1956-2023) of monthly data for d indices with:
        - Baseline correlated noise
        - ENSO-like events (periodic warm/cool phases)
        - A volcanic signal (1991)
        - A PDO shift (1977)
        - Seasonal cycle in tropical indices

        This synthetic data is designed to test the regime detector, not to
        reproduce realistic climate statistics.
        """
        rng = np.random.RandomState(seed)
        index_names = INDICES_12[:d] if d <= 12 else INDICES_13[:d]

        # Time axis: monthly from Jan 1956
        start_year, start_month = 1956, 1
        t = np.arange(T)
        years = start_year + t // 12
        months = 1 + t % 12

        # Baseline correlation structure
        # ENSO indices (0-4) are highly correlated
        # Oceanic (5-7) moderate correlation with ENSO
        # Atmospheric (8-11) weak correlation with ENSO
        base_corr = np.eye(d)
        for i in range(min(d, 5)):
            for j in range(i + 1, min(d, 5)):
                c = 0.7 + 0.2 * rng.rand()
                base_corr[i, j] = c
                base_corr[j, i] = c
        for i in range(min(d, 5)):
            for j in range(5, min(d, 8)):
                c = 0.3 + 0.2 * rng.rand()
                base_corr[i, j] = c
                base_corr[j, i] = c

        # Base volatilities
        base_vol = 0.8 * np.ones(d)
        base_vol[:5] = 1.2  # ENSO indices more volatile

        # Generate data with regime modulation
        values = np.zeros((T, d))

        for ti in range(T):
            yr, mo = years[ti], months[ti]

            # Seasonal modulation for tropical indices
            seasonal = 0.3 * np.sin(2 * np.pi * (mo - 1) / 12)

            # ENSO signal: quasi-periodic with ~4-year period
            enso_phase = np.sin(2 * np.pi * (yr + mo / 12 - 1956) / 4.2)

            # Major El Niño events — boost ENSO indices
            enso_boost = 0.0
            for (ey, em, etype, estr) in ENSO_EVENTS:
                dt = (yr - ey) + (mo - em) / 12
                if -0.2 < dt < 1.5:
                    amp = {"weak": 0.8, "moderate": 1.5, "strong": 2.5,
                           "very_strong": 3.5}.get(estr, 1.0)
                    sign = 1.0 if etype == "nino" else -1.0
                    # Gaussian envelope peaking ~6 months after onset
                    enso_boost += sign * amp * np.exp(-0.5 * ((dt - 0.5) / 0.4) ** 2)

            # Volcanic cooling (global)
            volcanic = 0.0
            for (vy, vm, _) in VOLCANIC_EVENTS:
                dt = (yr - vy) + (mo - vm) / 12
                if 0 < dt < 3:
                    volcanic -= 1.5 * np.exp(-dt / 0.8)

            # PDO shift
            pdo_shift = 0.0
            if yr >= 1977 and yr < 1998:
                pdo_shift = 0.5
            elif yr >= 2014:
                pdo_shift = 0.3

            # AMO shift
            amo_shift = 0.0
            if yr >= 1995:
                amo_shift = 0.3

            # Modulate correlation matrix during events
            mod_corr = base_corr.copy()
            if abs(enso_boost) > 1.0:
                # During strong ENSO: correlations increase
                for i in range(min(d, 5)):
                    for j in range(i + 1, min(d, 8)):
                        mod_corr[i, j] = min(0.95, mod_corr[i, j] + 0.3)
                        mod_corr[j, i] = mod_corr[i, j]

            if abs(volcanic) > 0.5:
                # During volcanic events: global correlations increase
                for i in range(d):
                    for j in range(i + 1, d):
                        mod_corr[i, j] = min(0.95, mod_corr[i, j] + 0.2)
                        mod_corr[j, i] = mod_corr[i, j]

            # Ensure SPD
            eigvals = np.linalg.eigvalsh(mod_corr)
            if eigvals[0] <= 0:
                mod_corr += (abs(eigvals[0]) + 0.01) * np.eye(d)
                # Re-normalize to correlation
                diag = np.sqrt(np.diag(mod_corr))
                mod_corr = mod_corr / np.outer(diag, diag)

            Sigma = np.diag(base_vol) @ mod_corr @ np.diag(base_vol)

            # Sample
            noise = rng.multivariate_normal(np.zeros(d), Sigma)

            # Add deterministic signals
            for idx in range(min(d, 5)):
                noise[idx] += enso_boost + seasonal * (0.5 if idx < 5 else 0.1)
            if 1 in range(d):  # SOI is anti-correlated with Niño
                noise[1] *= -0.8
            if d > 5:
                noise[5] += pdo_shift
            if d > 6:
                noise[6] += amo_shift
            # Volcanic affects all indices
            for idx in range(d):
                noise[idx] += volcanic * (0.5 if idx >= 8 else 0.3)
            # Global temperature includes volcanic + slow trend
            if d > 12:
                noise[12] += 0.01 * (yr - 1956) + volcanic

            values[ti] = noise

        dates = [(int(years[ti]), int(months[ti])) for ti in range(T)]
        return ClimateData(
            values=values, dates=dates, index_names=index_names,
            metadata={"synthetic": True, "seed": seed},
        )


# =====================================================================
# Rolling covariance for climate data
# =====================================================================

def rolling_covariance_climate(data: ClimateData, window: int = 72,
                               step: int = 1,
                               shrinkage: float = 0.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Compute rolling covariance matrices from climate data.

    Args:
        data: ClimateData with [T, d] values.
        window: rolling window in months (default 72 = 6 years).
        step: step size in months.
        shrinkage: Ledoit-Wolf shrinkage toward identity (0-1).

    Returns:
        covs: [N, d, d] SPD covariance matrices.
        center_dates: [N] list of (year, month) at center of each window.
    """
    T, d = data.values.shape
    covs = []
    center_dates = []

    for t in range(window, T + 1, step):
        chunk = data.values[t - window:t]
        C = np.cov(chunk, rowvar=False)

        if shrinkage > 0:
            mu = np.trace(C) / d
            C = (1 - shrinkage) * C + shrinkage * mu * np.eye(d)

        C = _symmetrize(C)
        eigvals = np.linalg.eigvalsh(C)
        if eigvals[0] <= 0:
            C += (abs(eigvals[0]) + 1e-8) * np.eye(d)

        covs.append(C)
        # Center of window
        center_idx = t - window // 2
        if center_idx < len(data.dates):
            center_dates.append(data.dates[center_idx])
        else:
            center_dates.append(data.dates[-1])

    return np.array(covs), center_dates


# =====================================================================
# Climate regime detector
# =====================================================================

@dataclass
class RegimeTransition:
    """A detected regime transition."""
    index: int                     # index in the covariance sequence
    date: Tuple[int, int]          # (year, month)
    geodesic_distance: float       # distance that triggered detection
    baseline_mean: float           # running mean distance
    baseline_std: float            # running std distance
    sigma_above: float             # how many sigma above mean
    category: str = "unknown"      # assigned category (enso, volcanic, pdo, etc.)
    matched_event: Optional[str] = None  # matched known event


@dataclass
class DetectionResults:
    """Full results from regime detection."""
    geodesic_distances: np.ndarray
    euclidean_distances: np.ndarray
    transitions: List[RegimeTransition]
    covariances: np.ndarray
    center_dates: List[Tuple[int, int]]
    window: int
    step: int
    threshold: float
    index_names: List[str]

    # Validation metrics (populated by validate())
    enso_recall: Optional[float] = None
    enso_precision: Optional[float] = None
    mean_lead_time: Optional[float] = None
    false_positive_rate: Optional[float] = None


class ClimateRegimeDetector:
    """Detect climate regime transitions using geodesic distance on SPD(d).

    Monitors the geodesic distance between successive rolling covariance
    matrices of climate indices. A regime transition is flagged when the
    distance exceeds threshold * sigma above the running mean.

    The geodesic distance on SPD(d) is:
        d(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F

    This is affine-invariant: equally sensitive to proportional changes
    regardless of base volatility level.
    """

    def __init__(self, window: int = 72, step: int = 1,
                 threshold: float = 2.0, shrinkage: float = 0.1,
                 adaptive: bool = True, adaptive_window: int = 120):
        """
        Args:
            window: rolling covariance window in months.
            step: step between windows.
            threshold: sigma threshold for transition detection.
            shrinkage: Ledoit-Wolf shrinkage for covariance estimation.
            adaptive: if True, use a rolling baseline for the threshold
                      (adapts to changing background variability).
            adaptive_window: lookback for adaptive baseline (months).
        """
        self.window = window
        self.step = step
        self.threshold = threshold
        self.shrinkage = shrinkage
        self.adaptive = adaptive
        self.adaptive_window = adaptive_window

    def detect(self, data: ClimateData) -> DetectionResults:
        """Run regime detection on climate data.

        Args:
            data: ClimateData with [T, d] values.

        Returns:
            DetectionResults with distances, transitions, covariances, dates.
        """
        covs, center_dates = rolling_covariance_climate(
            data, self.window, self.step, self.shrinkage
        )
        d = covs.shape[1]
        layer = SPDLayer(d)

        # Compute pairwise geodesic and Euclidean distances
        N = len(covs)
        geo_dists = np.zeros(N - 1)
        euc_dists = np.zeros(N - 1)

        for i in range(1, N):
            geo_dists[i - 1] = layer.geodesic_distance(covs[i - 1], covs[i])
            euc_dists[i - 1] = float(np.linalg.norm(covs[i] - covs[i - 1], 'fro'))

        # Detect transitions
        transitions = []

        if self.adaptive:
            # Rolling adaptive threshold
            hw = self.adaptive_window // (2 * self.step)
            for i in range(len(geo_dists)):
                lo = max(0, i - hw)
                hi = min(len(geo_dists), i + hw)
                window_dists = geo_dists[lo:hi]
                mu = np.mean(window_dists)
                sigma = max(np.std(window_dists), 1e-10)
                n_sigma = (geo_dists[i] - mu) / sigma

                if n_sigma > self.threshold:
                    transitions.append(RegimeTransition(
                        index=i,
                        date=center_dates[i + 1] if i + 1 < len(center_dates) else center_dates[-1],
                        geodesic_distance=float(geo_dists[i]),
                        baseline_mean=float(mu),
                        baseline_std=float(sigma),
                        sigma_above=float(n_sigma),
                    ))
        else:
            # Global threshold
            mu = np.mean(geo_dists)
            sigma = np.std(geo_dists)
            for i in range(len(geo_dists)):
                n_sigma = (geo_dists[i] - mu) / sigma
                if n_sigma > self.threshold:
                    transitions.append(RegimeTransition(
                        index=i,
                        date=center_dates[i + 1] if i + 1 < len(center_dates) else center_dates[-1],
                        geodesic_distance=float(geo_dists[i]),
                        baseline_mean=float(mu),
                        baseline_std=float(sigma),
                        sigma_above=float(n_sigma),
                    ))

        # Categorize transitions by matching to known events
        self._categorize_transitions(transitions)

        return DetectionResults(
            geodesic_distances=geo_dists,
            euclidean_distances=euc_dists,
            transitions=transitions,
            covariances=covs,
            center_dates=center_dates,
            window=self.window,
            step=self.step,
            threshold=self.threshold,
            index_names=data.index_names,
        )

    def _categorize_transitions(self, transitions: List[RegimeTransition]):
        """Assign categories to detected transitions based on known events."""
        for tr in transitions:
            yr, mo = tr.date
            best_match = None
            best_dist = 24  # max months to match

            # Check ENSO events (within 12 months)
            for (ey, em, etype, estr) in ENSO_EVENTS:
                dt_months = (yr - ey) * 12 + (mo - em)
                if -6 <= dt_months <= 12 and abs(dt_months) < best_dist:
                    best_dist = abs(dt_months)
                    best_match = ("enso", f"{etype} {estr} ({ey})")
                    tr.category = "enso"
                    tr.matched_event = best_match[1]

            # Check volcanic events (within 6 months after eruption)
            for (vy, vm, vname) in VOLCANIC_EVENTS:
                dt_months = (yr - vy) * 12 + (mo - vm)
                if -2 <= dt_months <= 18 and abs(dt_months) < best_dist:
                    best_dist = abs(dt_months)
                    best_match = ("volcanic", f"{vname} ({vy})")
                    tr.category = "volcanic"
                    tr.matched_event = best_match[1]

            # Check PDO shifts (within 12 months)
            for (py, pm, ptype) in PDO_SHIFTS:
                dt_months = (yr - py) * 12 + (mo - pm)
                if -6 <= dt_months <= 12 and abs(dt_months) < best_dist:
                    best_dist = abs(dt_months)
                    best_match = ("pdo", f"PDO {ptype} ({py})")
                    tr.category = "pdo"
                    tr.matched_event = best_match[1]

            # Check AMO shifts
            for (ay, am, atype) in AMO_SHIFTS:
                dt_months = (yr - ay) * 12 + (mo - am)
                if -6 <= dt_months <= 12 and abs(dt_months) < best_dist:
                    best_dist = abs(dt_months)
                    best_match = ("amo", f"AMO {atype} ({ay})")
                    tr.category = "amo"
                    tr.matched_event = best_match[1]

    def validate(self, results: DetectionResults,
                 max_lead_months: int = 12) -> Dict:
        """Validate detection results against known ENSO events.

        Returns:
            Dict with recall, precision, mean_lead_time, false_positive analysis.
        """
        detected_enso = [tr for tr in results.transitions if tr.category == "enso"]
        detected_other = [tr for tr in results.transitions if tr.category != "enso" and tr.category != "unknown"]
        detected_unknown = [tr for tr in results.transitions if tr.category == "unknown"]

        # ENSO recall: fraction of known events detected
        events_in_range = [
            (ey, em) for (ey, em, _, _) in ENSO_EVENTS
            if results.center_dates[0][0] <= ey <= results.center_dates[-1][0]
        ]
        matched_events = set()
        for tr in detected_enso:
            for (ey, em, _, _) in ENSO_EVENTS:
                dt = abs((tr.date[0] - ey) * 12 + (tr.date[1] - em))
                if dt <= max_lead_months:
                    matched_events.add((ey, em))

        recall = len(matched_events) / max(len(events_in_range), 1)

        # Lead time for matched events
        lead_times = []
        for (ey, em) in matched_events:
            # Find earliest detection
            earliest = None
            for tr in detected_enso:
                dt = (ey - tr.date[0]) * 12 + (em - tr.date[1])
                if 0 <= dt <= max_lead_months:
                    if earliest is None or dt > earliest:
                        earliest = dt
            if earliest is not None:
                lead_times.append(earliest)

        mean_lead = np.mean(lead_times) if lead_times else 0.0

        # Precision: fraction of detections that match known events
        total_det = len(results.transitions)
        matched_det = len(detected_enso) + len(detected_other)
        precision = matched_det / max(total_det, 1)

        # "Corrected precision" — unknown detections that are near non-ENSO events
        # (volcanoes, PDO shifts) should count as true positives

        results.enso_recall = recall
        results.enso_precision = precision
        results.mean_lead_time = mean_lead
        results.false_positive_rate = len(detected_unknown) / max(total_det, 1)

        return {
            "enso_recall": recall,
            "enso_precision": precision,
            "corrected_precision": matched_det / max(total_det, 1),
            "mean_lead_time_months": mean_lead,
            "n_detected": total_det,
            "n_enso_matched": len(detected_enso),
            "n_other_matched": len(detected_other),
            "n_unknown": len(detected_unknown),
            "events_in_range": len(events_in_range),
            "events_matched": len(matched_events),
            "unknown_dates": [(tr.date, tr.sigma_above) for tr in detected_unknown],
        }


# =====================================================================
# V+/V- decomposition for event fingerprinting
# =====================================================================

class VDecomposition:
    """Decompose covariance changes into volume (V+) and shape (V-) components.

    At each transition, the tangent vector log_A(B) decomposes as:
        V = V+ + V-
    where V+ = (tr(V)/d) * I  (isotropic volume change)
          V- = V - V+          (anisotropic shape change)

    ENSO transitions are expected to be V- dominated (correlation structure
    changes more than overall variance). Volcanic events may be V+ dominated
    (global variance increase).
    """

    def __init__(self, results: DetectionResults):
        self.results = results
        self.d = results.covariances.shape[1]
        self.layer = SPDLayer(self.d)

    def decompose_transition(self, idx: int) -> Dict:
        """Decompose the tangent vector at transition index idx.

        Returns:
            Dict with V_plus (trace/volume), V_minus (shape),
            ratio (|V-|/|V+|), and the full tangent vector.
        """
        A = self.results.covariances[idx]
        B = self.results.covariances[idx + 1]

        # Tangent vector: log_A(B)
        V = self.layer.log_map(B, A)

        # Decompose
        trace_V = np.trace(V)
        V_plus = (trace_V / self.d) * np.eye(self.d)
        V_minus = V - V_plus

        norm_plus = np.linalg.norm(V_plus, 'fro')
        norm_minus = np.linalg.norm(V_minus, 'fro')

        return {
            "V": V,
            "V_plus": V_plus,
            "V_minus": V_minus,
            "norm_plus": float(norm_plus),
            "norm_minus": float(norm_minus),
            "ratio": float(norm_minus / max(norm_plus, 1e-12)),
            "trace": float(trace_V),
            "volume_fraction": float(norm_plus / max(norm_plus + norm_minus, 1e-12)),
        }

    def decompose_all_transitions(self) -> List[Dict]:
        """Decompose all detected transitions."""
        decomps = []
        for tr in self.results.transitions:
            dec = self.decompose_transition(tr.index)
            dec["date"] = tr.date
            dec["category"] = tr.category
            dec["matched_event"] = tr.matched_event
            dec["sigma_above"] = tr.sigma_above
            decomps.append(dec)
        return decomps

    def fingerprint_analysis(self) -> Dict:
        """Statistical analysis of V+/V- by event category.

        Tests whether ENSO vs volcanic events have different V+/V- signatures.
        """
        decomps = self.decompose_all_transitions()

        by_category = {}
        for dec in decomps:
            cat = dec["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(dec)

        stats = {}
        for cat, decs in by_category.items():
            ratios = [d["ratio"] for d in decs]
            vol_fracs = [d["volume_fraction"] for d in decs]
            stats[cat] = {
                "count": len(decs),
                "mean_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios)),
                "mean_vol_fraction": float(np.mean(vol_fracs)),
                "std_vol_fraction": float(np.std(vol_fracs)),
            }

        # Statistical test: ENSO vs volcanic (if both present)
        p_value = None
        if "enso" in by_category and "volcanic" in by_category:
            enso_ratios = [d["ratio"] for d in by_category["enso"]]
            volc_ratios = [d["ratio"] for d in by_category["volcanic"]]
            if len(enso_ratios) >= 2 and len(volc_ratios) >= 2:
                from scipy.stats import mannwhitneyu
                try:
                    _, p_value = mannwhitneyu(enso_ratios, volc_ratios,
                                              alternative='two-sided')
                except ValueError:
                    p_value = 1.0

        return {
            "by_category": stats,
            "enso_vs_volcanic_p": float(p_value) if p_value is not None else None,
            "decompositions": decomps,
        }


# =====================================================================
# Dimension comparison analysis
# =====================================================================

def compare_dimensions(data: ClimateData,
                       index_sets: Optional[Dict[str, List[str]]] = None,
                       **detector_kwargs) -> Dict:
    """Compare regime detection across different index set sizes.

    Tests whether expanding from 6 to 12+ indices improves detection.

    Args:
        data: ClimateData with all available indices.
        index_sets: Dict mapping name → list of index keys.
                    Default: {"6d": INDICES_6, "12d": INDICES_12}.
        **detector_kwargs: passed to ClimateRegimeDetector.

    Returns:
        Dict with detection results and validation for each index set.
    """
    if index_sets is None:
        # Build index sets from what's available in the data
        available = set(data.index_names)
        index_sets = {}
        for name, indices in [("6d", INDICES_6), ("12d", INDICES_12), ("13d", INDICES_13)]:
            subset = [idx for idx in indices if idx in available]
            if len(subset) >= 4:
                index_sets[f"{len(subset)}d"] = subset

    results = {}
    for name, indices in index_sets.items():
        subset = data.select_indices(indices)
        detector = ClimateRegimeDetector(**detector_kwargs)
        det_results = detector.detect(subset)
        validation = detector.validate(det_results)

        # V+/V- analysis
        vd = VDecomposition(det_results)
        fingerprints = vd.fingerprint_analysis()

        results[name] = {
            "n_indices": len(indices),
            "indices": indices,
            "n_transitions": len(det_results.transitions),
            "validation": validation,
            "fingerprints": fingerprints,
            "enso_vs_volcanic_p": fingerprints["enso_vs_volcanic_p"],
        }

    return results


# =====================================================================
# Serialization
# =====================================================================

def save_results(results: DetectionResults, path: str):
    """Save detection results to JSON."""
    out = {
        "window": results.window,
        "step": results.step,
        "threshold": results.threshold,
        "index_names": results.index_names,
        "n_covariances": len(results.covariances),
        "n_transitions": len(results.transitions),
        "transitions": [
            {
                "index": tr.index,
                "date": list(tr.date),
                "geodesic_distance": tr.geodesic_distance,
                "sigma_above": tr.sigma_above,
                "category": tr.category,
                "matched_event": tr.matched_event,
            }
            for tr in results.transitions
        ],
        "geodesic_distances": results.geodesic_distances.tolist(),
        "euclidean_distances": results.euclidean_distances.tolist(),
        "center_dates": [list(d) for d in results.center_dates],
        "validation": {
            "enso_recall": results.enso_recall,
            "enso_precision": results.enso_precision,
            "mean_lead_time": results.mean_lead_time,
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


# =====================================================================
# Demo runner
# =====================================================================

def run_demo(d: int = 12, verbose: bool = True) -> Dict:
    """Run the full climate regime detection demo.

    Generates synthetic 68-year climate data, runs detection at 6d and 12d,
    validates against known ENSO events, and performs V+/V- fingerprinting.
    """
    if verbose:
        print("=" * 90)
        print("RIEMANNIAN CLIMATE REGIME DETECTION")
        print("=" * 90)

    # Generate synthetic data
    data = ClimateDataLoader.generate_synthetic(T=816, d=d)
    if verbose:
        print(f"\nData: {data.T} months ({data.start_year}-{data.end_year}), "
              f"{data.d} indices")
        print(f"Indices: {', '.join(data.index_names)}")

    # Run detection at multiple dimensions
    all_results = {}

    for dim_label, indices in [("6d", INDICES_6), ("12d", INDICES_12[:d])]:
        available = [idx for idx in indices if idx in data.index_names]
        if len(available) < 4:
            continue
        subset = data.select_indices(available)

        if verbose:
            print(f"\n{'─' * 90}")
            print(f"Detection at {len(available)}D: {', '.join(available)}")
            print(f"{'─' * 90}")

        detector = ClimateRegimeDetector(
            window=72, step=1, threshold=2.0, shrinkage=0.1
        )
        results = detector.detect(subset)

        if verbose:
            print(f"  Covariance matrices: {len(results.covariances)}")
            print(f"  Transitions detected: {len(results.transitions)}")
            print(f"  Geodesic distance: mean={np.mean(results.geodesic_distances):.4f}, "
                  f"max={np.max(results.geodesic_distances):.4f}")

        # Validate
        val = detector.validate(results)
        if verbose:
            print(f"\n  ENSO Validation:")
            print(f"    Recall:     {val['enso_recall']:.1%} "
                  f"({val['events_matched']}/{val['events_in_range']} events)")
            print(f"    Precision:  {val['enso_precision']:.1%}")
            print(f"    Lead time:  {val['mean_lead_time_months']:.1f} months")
            print(f"    Unknown:    {val['n_unknown']} detections")

        # V+/V- analysis
        vd = VDecomposition(results)
        fp = vd.fingerprint_analysis()

        if verbose:
            print(f"\n  V+/V- Fingerprinting:")
            for cat, stats in fp["by_category"].items():
                print(f"    {cat:<10s}: n={stats['count']}, "
                      f"V-/V+ ratio={stats['mean_ratio']:.2f}±{stats['std_ratio']:.2f}, "
                      f"vol_frac={stats['mean_vol_fraction']:.2f}")
            if fp["enso_vs_volcanic_p"] is not None:
                print(f"    ENSO vs Volcanic p-value: {fp['enso_vs_volcanic_p']:.4f}")
            else:
                print(f"    (not enough volcanic events for statistical test)")

        # Show transitions
        if verbose and results.transitions:
            print(f"\n  Detected transitions:")
            for tr in results.transitions:
                event_str = f" → {tr.matched_event}" if tr.matched_event else ""
                print(f"    {tr.date[0]}-{tr.date[1]:02d}  "
                      f"d={tr.geodesic_distance:.4f}  "
                      f"({tr.sigma_above:.1f}σ)  "
                      f"[{tr.category}]{event_str}")

        all_results[dim_label] = {
            "results": results,
            "validation": val,
            "fingerprints": fp,
        }

    # Summary comparison
    if verbose and len(all_results) > 1:
        print(f"\n{'=' * 90}")
        print("DIMENSION COMPARISON")
        print(f"{'=' * 90}")
        for label, data_dict in all_results.items():
            v = data_dict["validation"]
            fp = data_dict["fingerprints"]
            p_str = (f"{fp['enso_vs_volcanic_p']:.4f}"
                     if fp["enso_vs_volcanic_p"] is not None else "N/A")
            print(f"  {label}: recall={v['enso_recall']:.1%}, "
                  f"precision={v['enso_precision']:.1%}, "
                  f"lead={v['mean_lead_time_months']:.1f}mo, "
                  f"V+/V- p={p_str}")

    return all_results
