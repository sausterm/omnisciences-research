"""
Volume-based enzyme variant screening.

The simplest and most general approach: fit log10(k) vs ΔV (side-chain
volume change) on known variants, then predict rates for novel mutations.

NO crystal structures needed. NO AlphaFold. NO QM. Just the mutation
name and a table of amino acid volumes (Zamyatnin 1972).

This works because side-chain volume change captures the dominant effect
of active-site mutations: cavity expansion/contraction affects the
donor-acceptor distance, dynamics, and tunneling probability. The exact
mechanism varies by enzyme, but the log-linear ΔV-rate correlation is
robust across enzyme families:

    SLO-1 (HAT):      R² = 0.96, slope = 0.019
    DHFR (hydride):   R² = 0.93, slope = 0.025
    HLADH (hydride):  slope = 0.139 (2 points only)

The slope is system-specific (varies ~7x), so it must be calibrated
per enzyme from known variants. But within a system, the correlation
is strong enough for screening.

Usage::

    from pcet_engine.core.volume_calibrator import VolumeCalibrator

    cal = VolumeCalibrator()
    result = cal.calibrate([
        {"name": "WT",    "mutation": None,    "k_exp": 297.0},
        {"name": "L546A", "mutation": "L546A", "k_exp": 8.2},
        {"name": "L754A", "mutation": "L754A", "k_exp": 3.0},
    ])
    print(result.summary())

    # Screen 200 mutations — just names, nothing else needed
    predictions = result.screen(["L546V", "L546F", "L546W", "L546G",
                                  "L754V", "L754F", "L754W", "L754G"])
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from pcet_engine.core.mutation_modeler import SIDECHAIN_VOLUMES


@dataclass
class VolumeCalibrationResult:
    """Result of volume-based calibration."""
    slope: float            # Δlog10(k) per ų of volume change
    intercept: float        # log10(k) at ΔV = 0 (wild-type rate)
    r_squared: float
    rmse_log_k: float
    n_variants: int
    residuals: dict         # per-variant fit quality

    def predict_rate(self, mutation: Optional[str]) -> float:
        """Predict k from a mutation string (e.g., 'L546A')."""
        dv = _compute_dv(mutation) if mutation else 0.0
        return 10**(self.intercept + self.slope * dv)

    def screen(self, mutations: list) -> list:
        """Screen a list of mutations and rank by predicted rate.

        Parameters
        ----------
        mutations : list of str
            Mutation strings, e.g., ["L546A", "L546W", "L754G"].
            Use "WT" or None for wild-type.

        Returns
        -------
        list of dict, sorted by predicted rate (fastest first).
        """
        results = []
        k_wt = 10**self.intercept

        for mut in mutations:
            if mut is None or mut.upper() == "WT":
                dv = 0.0
                name = "WT"
            elif "/" in mut:
                # Double/triple mutant
                parts = mut.split("/")
                dv = sum(_compute_single_dv(p.strip()) for p in parts)
                name = mut
            else:
                dv = _compute_single_dv(mut)
                name = mut

            k_pred = 10**(self.intercept + self.slope * dv)
            results.append({
                "rank": 0,
                "name": name,
                "mutation": mut,
                "dV": dv,
                "k_pred": k_pred,
                "rate_ratio": k_pred / k_wt if k_wt > 0 else 0,
                "log10_k": self.intercept + self.slope * dv,
            })

        # Sort by rate (fastest first) and assign ranks
        results.sort(key=lambda x: x["k_pred"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    def saturation_scan(self, position: int, wt_residue: str) -> list:
        """Screen all 19 possible substitutions at one position.

        Parameters
        ----------
        position : int
            Residue position number.
        wt_residue : str
            One-letter code of the wild-type residue.

        Returns
        -------
        list of dict, sorted by predicted rate.
        """
        mutations = [
            f"{wt_residue}{position}{aa}"
            for aa in SIDECHAIN_VOLUMES
            if aa != wt_residue
        ]
        return self.screen(mutations)

    def summary(self) -> str:
        lines = [
            f"Volume Calibration ({self.n_variants} variants)",
            f"  log10(k) = {self.intercept:.3f} + ({self.slope:.5f}) × ΔV",
            f"  R² = {self.r_squared:.4f}",
            f"  RMSE(log10 k) = {self.rmse_log_k:.3f}",
            f"",
            f"  Interpretation:",
            f"    WT predicted rate: {10**self.intercept:.1f} s⁻¹",
        ]
        if self.slope > 0:
            lines.append(f"    Larger residues → FASTER (slope > 0)")
            lines.append(f"    Each +10 ų → {10**(self.slope*10):.2f}x rate increase")
        else:
            lines.append(f"    Larger residues → SLOWER (slope < 0)")
            lines.append(f"    Each -10 ų → {10**(self.slope*-10):.2f}x rate increase")
        lines.append(f"")
        lines.append(f"  Per-variant fit:")
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
            "r_squared": self.r_squared,
            "rmse_log_k": self.rmse_log_k,
            "n_variants": self.n_variants,
            "wt_rate": 10**self.intercept,
            "residuals": self.residuals,
        }


class VolumeCalibrator:
    """Calibrate log(k) vs ΔV for an enzyme system.

    The simplest possible product: customer provides mutation names
    and measured rates for 2+ variants. We fit, they screen.
    """

    def calibrate(self, known_variants: list) -> VolumeCalibrationResult:
        """Fit log10(k) vs ΔV on known variants.

        Parameters
        ----------
        known_variants : list of dict
            Each dict must have:
            - 'name': variant identifier
            - 'mutation': mutation string (e.g., "L546A") or None for WT
            - 'k_exp': experimental rate constant

        Returns
        -------
        VolumeCalibrationResult
        """
        if len(known_variants) < 2:
            raise ValueError("Need at least 2 variants for calibration")

        dv_arr = []
        log_k_arr = []
        for v in known_variants:
            mut = v.get("mutation")
            if mut is None or (isinstance(mut, str) and mut.upper() == "WT"):
                dv = 0.0
            elif "/" in str(mut):
                parts = str(mut).split("/")
                dv = sum(_compute_single_dv(p.strip()) for p in parts)
            else:
                dv = _compute_single_dv(str(mut))
            dv_arr.append(dv)
            log_k_arr.append(math.log10(v["k_exp"]))

        dv_arr = np.array(dv_arr)
        log_k_arr = np.array(log_k_arr)

        # Linear regression
        n = len(dv_arr)
        x_mean = dv_arr.mean()
        y_mean = log_k_arr.mean()
        ss_xx = float(np.sum((dv_arr - x_mean)**2))
        ss_xy = float(np.sum((dv_arr - x_mean) * (log_k_arr - y_mean)))

        slope = ss_xy / ss_xx if ss_xx > 1e-15 else 0.0
        intercept = y_mean - slope * x_mean

        # R² and RMSE
        log_k_pred = intercept + slope * dv_arr
        ss_res = float(np.sum((log_k_pred - log_k_arr)**2))
        ss_tot = float(np.sum((log_k_arr - y_mean)**2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / n))

        # Per-variant residuals
        residuals = {}
        for i, v in enumerate(known_variants):
            k_pred = 10**float(log_k_pred[i])
            residuals[v["name"]] = {
                "k_pred": k_pred,
                "k_exp": v["k_exp"],
                "log_ratio": float(log_k_pred[i] - log_k_arr[i]),
                "dV": float(dv_arr[i]),
            }

        return VolumeCalibrationResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            rmse_log_k=rmse,
            n_variants=n,
            residuals=residuals,
        )


def _compute_single_dv(mutation: str) -> float:
    """Compute ΔV for a single point mutation (e.g., 'L546A')."""
    if not mutation or len(mutation) < 3:
        return 0.0
    wt_aa = mutation[0].upper()
    new_aa = mutation[-1].upper()
    vol_wt = SIDECHAIN_VOLUMES.get(wt_aa, 130.0)
    vol_new = SIDECHAIN_VOLUMES.get(new_aa, 130.0)
    return vol_new - vol_wt


def _compute_dv(mutation: Optional[str]) -> float:
    """Compute ΔV for a mutation string, handling double mutants."""
    if not mutation or mutation.upper() == "WT":
        return 0.0
    if "/" in mutation:
        return sum(_compute_single_dv(p.strip()) for p in mutation.split("/"))
    return _compute_single_dv(mutation)
