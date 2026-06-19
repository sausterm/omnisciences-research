"""
Per-system calibration of PCET coupling parameters.

Given 3-5 known variants of an enzyme (with measured rates or KIEs
and known d_DA values), fit the coupling decay constant β and
prefactor V0 to best reproduce the experimental data. Then use the
calibrated parameters to predict rates for novel variants.

This is the consulting-enabling feature: "Give us your known data,
we calibrate to your system, then predict the rest."

Usage::

    from pcet_engine.core.system_calibrator import SystemCalibrator

    cal = SystemCalibrator()
    result = cal.calibrate(
        known_variants=[
            {"name": "WT",    "d_DA": 2.77, "k_H_exp": 297.0},
            {"name": "L546A", "d_DA": 2.88, "k_H_exp": 8.2},
            {"name": "L754A", "d_DA": 2.95, "k_H_exp": 3.0},
        ],
        delta_G=-5.4,
        lambda_reorg=19.0,
        omega_H=2900.0,
    )
    print(f"Calibrated: β={result.beta:.1f}, V0={result.V0:.4f}")
    print(f"R²={result.r_squared:.4f}")

    # Now predict novel variants
    predictions = result.predict([
        {"name": "new_mut1", "d_DA": 2.83},
        {"name": "new_mut2", "d_DA": 3.05},
    ])
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import minimize_scalar, minimize

from pcet_engine.core.rate_engine import PCETRateEngine


@dataclass
class CalibrationResult:
    """Result of per-system calibration."""
    beta: float           # calibrated coupling decay constant (Å⁻¹)
    V0: float             # calibrated coupling prefactor (kcal/mol)
    d0: float             # reference distance (Å)
    r_squared: float      # R² of log(k) fit
    rmse_log_k: float     # RMSE of log10(k_H) predictions
    n_variants: int       # number of variants used for calibration
    residuals: dict       # per-variant residuals

    # Fixed parameters used during calibration
    delta_G: float
    lambda_reorg: float
    omega_H: float
    method: str
    temperature: float

    def coupling_at(self, d_DA: float) -> float:
        """Compute V_el at a given d_DA using calibrated parameters."""
        return self.V0 * math.exp(-self.beta * (d_DA - self.d0))

    def predict(
        self,
        variants: list,
        run_uq: bool = False,
        n_samples: int = 500,
    ):
        """Predict rates for novel variants using calibrated coupling.

        Parameters
        ----------
        variants : list of dict
            Each dict must have 'name' and 'd_DA'.

        Returns
        -------
        list of dict
            Predicted rates for each variant.
        """
        from pcet_engine.core.variant_ranker import VariantRanker

        # Build variant list with calibrated V_el per variant
        ranked_variants = []
        for v in variants:
            V_el = self.coupling_at(v["d_DA"])
            ranked_variants.append({
                "name": v["name"],
                "d_DA": v["d_DA"],
                "V_el": V_el,
            })

        ranker = VariantRanker(
            V_el=self.V0,  # reference V_el (overridden per-variant)
            delta_G=self.delta_G,
            lambda_reorg=self.lambda_reorg,
            omega_H=self.omega_H,
            method=self.method,
            temperature=self.temperature,
        )
        return ranker.rank(ranked_variants, run_uq=run_uq, n_samples=n_samples)

    def summary(self) -> str:
        lines = [
            f"System Calibration ({self.n_variants} variants)",
            f"  β  = {self.beta:.2f} Å⁻¹",
            f"  V0 = {self.V0:.4f} kcal/mol (at d0 = {self.d0:.3f} Å)",
            f"  R² = {self.r_squared:.4f}",
            f"  RMSE(log10 k_H) = {self.rmse_log_k:.3f}",
            f"",
            f"  Per-variant residuals:",
        ]
        for name, res in self.residuals.items():
            lines.append(
                f"    {name}: log10(k_pred/k_exp) = {res['log_ratio']:.3f}"
                f"  (k_pred={res['k_pred']:.2e}, k_exp={res['k_exp']:.2e})"
            )
        return "\n".join(lines)

    def to_dict(self):
        return {
            "beta": self.beta,
            "V0": self.V0,
            "d0": self.d0,
            "r_squared": self.r_squared,
            "rmse_log_k": self.rmse_log_k,
            "n_variants": self.n_variants,
            "residuals": self.residuals,
        }


@dataclass
class EmpiricalCalibration:
    """Empirical log-linear fit: log10(k_H) = intercept + slope * (d_DA - d0)."""
    slope: float          # Δlog10(k) per Å (negative = rate decreases with distance)
    intercept: float      # log10(k) at d0
    d0: float             # reference distance
    r_squared: float
    rmse_log_k: float
    n_variants: int
    residuals: dict

    def predict_rate(self, d_DA: float) -> float:
        """Predict k_H at a given d_DA."""
        log_k = self.intercept + self.slope * (d_DA - self.d0)
        return 10**log_k

    def predict_variants(self, variants: list) -> list:
        """Predict rates for a list of variants."""
        results = []
        k_ref = self.predict_rate(self.d0)
        preds = []
        for v in variants:
            k = self.predict_rate(v["d_DA"])
            preds.append((v["name"], v["d_DA"], k))
        preds.sort(key=lambda x: x[2], reverse=True)
        for rank, (name, d_DA, k) in enumerate(preds, 1):
            results.append({
                "rank": rank,
                "name": name,
                "d_DA": d_DA,
                "k_H_pred": k,
                "rate_ratio": k / k_ref if k_ref > 0 else 0,
            })
        return results

    def summary(self) -> str:
        lines = [
            f"Empirical Calibration ({self.n_variants} variants)",
            f"  log10(k_H) = {self.intercept:.3f} + ({self.slope:.3f}) × (d_DA - {self.d0:.3f})",
            f"  Rate halving distance: {-math.log10(2) / self.slope:.3f} Å"
            if self.slope < 0 else "",
            f"  R² = {self.r_squared:.4f}",
            f"  RMSE(log10 k_H) = {self.rmse_log_k:.3f}",
            "",
            "  Per-variant fit:",
        ]
        for name, res in self.residuals.items():
            lines.append(
                f"    {name}: k_pred={res['k_pred']:.2e}, "
                f"k_exp={res['k_exp']:.2e}, "
                f"error={res['log_ratio']:.3f} dex"
            )
        return "\n".join(lines)

    def to_dict(self):
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "d0": self.d0,
            "r_squared": self.r_squared,
            "rmse_log_k": self.rmse_log_k,
            "n_variants": self.n_variants,
            "residuals": self.residuals,
            "rate_halving_distance": -math.log10(2) / self.slope if self.slope < 0 else None,
        }


class SystemCalibrator:
    """Calibrate PCET coupling to a specific enzyme system.

    Given known variants with measured rates, find the β and V0
    that best reproduce the experimental data.

    Parameters
    ----------
    method : str
        Rate method for the engine.
    temperature : float
        Temperature in K.
    """

    def __init__(
        self,
        method: str = "vibronic_multi",
        temperature: float = 298.15,
    ):
        self.method = method
        self.temperature = temperature
        self.engine = PCETRateEngine(temperature=temperature)

    def calibrate_empirical(
        self,
        known_variants: list,
        d0: Optional[float] = None,
    ) -> "EmpiricalCalibration":
        """Purely empirical calibration: fit log10(k_H) vs d_DA.

        No physical model — just linear regression on log-rate vs distance.
        This is the most robust approach for within-system variant ranking.

        Parameters
        ----------
        known_variants : list of dict
            Each dict must have 'name', 'd_DA', and 'k_H_exp'.
        d0 : float, optional
            Reference distance for centering.

        Returns
        -------
        EmpiricalCalibration
        """
        if len(known_variants) < 2:
            raise ValueError("Need at least 2 variants")

        if d0 is None:
            d0 = known_variants[0]["d_DA"]

        d_arr = np.array([v["d_DA"] for v in known_variants])
        log_k_arr = np.log10([v["k_H_exp"] for v in known_variants])

        # Fit log10(k) = intercept + slope * (d - d0)
        x = d_arr - d0
        n = len(x)
        x_mean = x.mean()
        y_mean = log_k_arr.mean()
        ss_xx = float(np.sum((x - x_mean)**2))
        ss_xy = float(np.sum((x - x_mean) * (log_k_arr - y_mean)))

        slope = ss_xy / ss_xx if ss_xx > 0 else 0
        intercept = y_mean - slope * x_mean

        # Predictions and R²
        log_k_pred = intercept + slope * x
        ss_res = float(np.sum((log_k_pred - log_k_arr)**2))
        ss_tot = float(np.sum((log_k_arr - y_mean)**2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        residuals = {}
        for i, v in enumerate(known_variants):
            k_pred = 10**float(log_k_pred[i])
            residuals[v["name"]] = {
                "k_pred": k_pred,
                "k_exp": v["k_H_exp"],
                "log_ratio": float(log_k_pred[i] - log_k_arr[i]),
            }

        return EmpiricalCalibration(
            slope=slope,
            intercept=intercept,
            d0=d0,
            r_squared=r_squared,
            rmse_log_k=float(np.sqrt(ss_res / n)),
            n_variants=n,
            residuals=residuals,
        )

    def _find_vel_for_rate(
        self,
        k_target: float,
        delta_G: float,
        lambda_reorg: float,
        omega_H: float,
        d_DA: float,
    ) -> float:
        """Find V_el that reproduces a target rate (bisection on log scale).

        Since k ∝ V_el², this is monotonic and bisection works reliably.
        """
        # k ∝ V_el² → log(k) ∝ 2*log(V_el) → monotonically increasing
        log_k_target = math.log10(k_target)

        def rate_at_vel(log_vel):
            V_el = 10**log_vel
            result = self.engine.compute_rate(
                V_el=V_el, delta_G=delta_G, lambda_reorg=lambda_reorg,
                omega_H=omega_H, d_DA=d_DA, method=self.method,
            )
            if result.k_H <= 0:
                return -100
            return math.log10(result.k_H)

        # Bisect on log10(V_el) in range [1e-4, 1e3] kcal/mol
        lo, hi = -4.0, 3.0
        for _ in range(100):
            mid = (lo + hi) / 2
            log_k = rate_at_vel(mid)
            if log_k < log_k_target:
                lo = mid
            else:
                hi = mid
            if abs(log_k - log_k_target) < 0.001:
                break

        return 10**((lo + hi) / 2)

    def calibrate(
        self,
        known_variants: list,
        delta_G: float,
        lambda_reorg: float,
        omega_H: float,
        d0: Optional[float] = None,
    ) -> CalibrationResult:
        """Fit coupling parameters to known variant data.

        Two-step calibration:
        1. For each variant, find V_el that reproduces k_H_exp exactly
        2. Fit β to the log(V_el) vs d_DA relationship (linear regression)

        This is more robust than joint optimization because step 1 is
        exact and step 2 is a simple linear fit.

        Parameters
        ----------
        known_variants : list of dict
            Each dict must have 'name', 'd_DA', and 'k_H_exp'.
        delta_G : float
            Driving force in kcal/mol.
        lambda_reorg : float
            Reorganization energy in kcal/mol.
        omega_H : float
            Proton frequency in cm⁻¹.
        d0 : float, optional
            Reference distance. Default: d_DA of first variant.

        Returns
        -------
        CalibrationResult
        """
        if len(known_variants) < 2:
            raise ValueError("Need at least 2 variants for calibration")

        has_rates = all("k_H_exp" in v for v in known_variants)
        if not has_rates:
            raise ValueError("All variants must have 'k_H_exp' for calibration")

        if d0 is None:
            d0 = known_variants[0]["d_DA"]

        # Step 1: Find V_el for each variant independently
        vel_values = []
        for v in known_variants:
            vel = self._find_vel_for_rate(
                v["k_H_exp"], delta_G, lambda_reorg, omega_H, v["d_DA"],
            )
            vel_values.append(vel)

        # Step 2: Fit log(V_el) = log(V0) - β * (d_DA - d0)
        # This is linear regression: y = a + b*x
        # where y = log(V_el), x = (d_DA - d0), a = log(V0), b = -β
        d_DA_arr = np.array([v["d_DA"] for v in known_variants])
        log_vel_arr = np.log(np.array(vel_values))  # natural log
        x = d_DA_arr - d0

        # Linear regression
        n = len(x)
        x_mean = x.mean()
        y_mean = log_vel_arr.mean()
        ss_xx = float(np.sum((x - x_mean)**2))
        ss_xy = float(np.sum((x - x_mean) * (log_vel_arr - y_mean)))

        if ss_xx < 1e-15:
            # All d_DA values are the same — can't fit β
            beta_opt = 16.3  # default
            V0_opt = vel_values[0]
        else:
            slope = ss_xy / ss_xx       # = -β
            intercept = y_mean - slope * x_mean  # = log(V0)
            beta_opt = max(-slope, 0.1)  # β must be positive
            V0_opt = math.exp(intercept)

        # Compute final predictions using calibrated β, V0
        residuals = {}
        k_exp_arr = np.array([v["k_H_exp"] for v in known_variants])
        log_k_exp = np.log10(k_exp_arr)
        log_k_pred = np.zeros(n)

        for i, v in enumerate(known_variants):
            V_el = V0_opt * math.exp(-beta_opt * (v["d_DA"] - d0))
            result = self.engine.compute_rate(
                V_el=V_el, delta_G=delta_G, lambda_reorg=lambda_reorg,
                omega_H=omega_H, d_DA=v["d_DA"], method=self.method,
            )
            log_k_pred[i] = math.log10(result.k_H) if result.k_H > 0 else -30
            log_ratio = log_k_pred[i] - log_k_exp[i]
            residuals[v["name"]] = {
                "k_pred": result.k_H,
                "k_exp": v["k_H_exp"],
                "log_ratio": float(log_ratio),
                "V_el_fit": float(vel_values[i]),
                "V_el_model": float(V0_opt * math.exp(-beta_opt * (v["d_DA"] - d0))),
            }

        ss_res = float(np.sum((log_k_pred - log_k_exp)**2))
        ss_tot = float(np.sum((log_k_exp - log_k_exp.mean())**2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / n))

        return CalibrationResult(
            beta=float(beta_opt),
            V0=float(V0_opt),
            d0=d0,
            r_squared=r_squared,
            rmse_log_k=rmse,
            n_variants=n,
            residuals=residuals,
            delta_G=delta_G,
            lambda_reorg=lambda_reorg,
            omega_H=omega_H,
            method=self.method,
            temperature=self.temperature,
        )
