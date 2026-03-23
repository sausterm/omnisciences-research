"""
Standard Euclidean optimal fingerprinting (Allen & Stott 2003) baseline.

Implements the classical detection/attribution method that works on
temperature MEAN vectors rather than covariance structure. Provided as
a comparison baseline for the Riemannian tangent-space regression in
climate_attribution.py.

Key differences from the Riemannian method:
    - Euclidean works on temperature MEANS; Riemannian on COVARIANCE STRUCTURE.
    - Euclidean cannot distinguish V+ from V- (energy budget vs teleconnection).
    - Riemannian is affine-invariant (rescaling indices does not change results).
    - Euclidean GLS accounts for internal variability covariance; Riemannian
      operates on the manifold where that structure is the *data*.

References:
    - Allen & Stott (2003). Estimating signal amplitudes in optimal
      fingerprinting, Part I. Climate Dynamics, 21, 477-491.
    - Hasselmann (1979). On the signal-to-noise problem in atmospheric
      response studies.
    - Hegerl et al. (2007). IPCC AR4 WG1 Chapter 9.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .climate_analysis import ClimateData


# =====================================================================
# Results dataclass
# =====================================================================

@dataclass
class EuclideanCoefficient:
    """GLS regression coefficient for a single forcing."""
    name: str
    beta: np.ndarray            # [d_response] scaling factors
    beta_se: np.ndarray         # standard errors
    f_statistic: float          # partial F-statistic
    p_value_parametric: float   # from F distribution
    p_value_surrogate: Optional[float] = None
    attribution_fraction: float = 0.0


@dataclass
class EuclideanFingerprintResults:
    """Results from Euclidean optimal fingerprinting."""
    coefficients: Dict[str, EuclideanCoefficient]
    attribution_fractions: Dict[str, float]
    r_squared: float
    residual_norm: float
    n_windows: int
    d_response: int
    n_forcings: int
    dominant_forcing: str = ""
    interpretation: List[str] = field(default_factory=list)


# =====================================================================
# Optimal fingerprinting engine
# =====================================================================

class OptimalFingerprinting:
    """Standard Euclidean optimal fingerprinting (Allen & Stott 2003).

    Operates on sliding-window mean response vectors (not covariance).
    Uses generalised least squares (GLS) with internal-variability
    covariance estimated from regression residuals.

    Steps:
        1. Compute sliding-window mean response vectors y(t).
        2. Build forcing design matrix X from window-averaged forcings.
        3. Estimate internal variability covariance C_n from OLS residuals.
        4. GLS: beta = (X' C_n^{-1} X)^{-1} X' C_n^{-1} y.
        5. F-statistics and p-values for each forcing.
        6. Optional block-permutation surrogates for non-parametric p-values.
    """

    def __init__(self, window: int = 120, step: int = 12,
                 n_surrogates: int = 500, block_size: int = 24):
        """
        Args:
            window: sliding window in months for computing means.
            step: step between windows in months.
            n_surrogates: number of block-permutation surrogates.
            block_size: block size in months for block permutation.
        """
        self.window = window
        self.step = step
        self.n_surrogates = n_surrogates
        self.block_size = block_size

    def _sliding_means(self, data: ClimateData,
                       index_names: List[str]
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sliding-window mean vectors.

        Returns:
            means: [N, d] array of window-averaged response vectors.
            center_times: [N] fractional years at window centers.
        """
        idx = [data.index_names.index(n) for n in index_names]
        vals = data.values[:, idx]
        T = len(vals)

        means = []
        center_times = []
        for t in range(self.window, T + 1, self.step):
            chunk = vals[t - self.window:t]
            means.append(chunk.mean(axis=0))
            center_idx = t - self.window // 2
            yr, mo = data.dates[min(center_idx, T - 1)]
            center_times.append(yr + (mo - 1) / 12.0)

        return np.array(means), np.array(center_times)

    def _align_forcing(self, data: ClimateData,
                       forcing_names: List[str],
                       center_times: np.ndarray,
                       lag_months: int = 0) -> np.ndarray:
        """Average forcing over each window, with optional lag."""
        fidx = [data.index_names.index(n) for n in forcing_names]
        T = len(data.values)
        forcing_raw = data.values[:, fidx]

        # Collapse to 1D via first PC if multiple
        if forcing_raw.shape[1] > 1:
            fc = forcing_raw - forcing_raw.mean(axis=0)
            _, _, Vt = np.linalg.svd(fc, full_matrices=False)
            f1d = fc @ Vt[0]
        else:
            f1d = forcing_raw[:, 0]

        # Lag
        if lag_months != 0:
            f1d = np.roll(f1d, lag_months)
            if lag_months > 0:
                f1d[:lag_months] = f1d[lag_months]
            else:
                f1d[lag_months:] = f1d[lag_months - 1]

        # Window average
        aligned = np.zeros(len(center_times))
        for i, ct in enumerate(center_times):
            t_center = int(round((ct - data.dates[0][0]) * 12))
            t_start = max(0, t_center - self.window // 2)
            t_end = min(T, t_start + self.window)
            aligned[i] = f1d[t_start:t_end].mean()

        return aligned

    def _block_permute(self, x: np.ndarray,
                       rng: np.random.RandomState) -> np.ndarray:
        """Block permutation of a 1D time series.

        Preserves within-block autocorrelation structure while
        destroying long-range phase relationships with response.
        """
        n = len(x)
        bs = self.block_size
        n_blocks = max(1, (n + bs - 1) // bs)  # ceil division
        # Pad to full blocks
        padded = np.zeros(n_blocks * bs)
        padded[:n] = x
        if n < n_blocks * bs:
            padded[n:] = x[-1]  # pad with last value
        blocks = padded.reshape(n_blocks, bs)
        perm = rng.permutation(n_blocks)
        shuffled = blocks[perm].ravel()
        return shuffled[:n]

    def regress(self, data: ClimateData,
                forcing_groups: Dict[str, List[str]],
                response_names: List[str],
                lag_months: Optional[Dict[str, int]] = None,
                test_surrogates: bool = True
                ) -> EuclideanFingerprintResults:
        """Run Euclidean optimal fingerprinting.

        Same signature as TangentSpaceRegression.regress() for easy
        comparison.

        Args:
            data: ClimateData with forcing and response indices.
            forcing_groups: dict mapping label -> forcing index names.
            response_names: climate response index names.
            lag_months: dict mapping label -> lag in months.
            test_surrogates: whether to run block-permutation tests.

        Returns:
            EuclideanFingerprintResults with scaling factors and p-values.
        """
        from scipy import stats as sp_stats

        if lag_months is None:
            lag_months = {k: 0 for k in forcing_groups}

        d = len(response_names)
        labels = list(forcing_groups.keys())
        k = len(labels)

        # 1. Sliding-window means
        Y, center_times = self._sliding_means(data, response_names)
        N = len(Y)

        # 2. Design matrix [N, k+1] with intercept
        X = np.ones((N, k + 1))
        for j, label in enumerate(labels):
            lag = lag_months.get(label, 0)
            X[:, j + 1] = self._align_forcing(
                data, forcing_groups[label], center_times, lag
            )

        # Standardise forcing columns
        f_means = X[:, 1:].mean(axis=0)
        f_stds = X[:, 1:].std(axis=0)
        f_stds[f_stds < 1e-10] = 1.0
        X[:, 1:] = (X[:, 1:] - f_means) / f_stds

        # 3. OLS to estimate residuals for C_n
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(k + 1))
        B_ols = XtX_inv @ X.T @ Y  # [k+1, d]
        E = Y - X @ B_ols  # [N, d] residuals

        # Internal variability covariance from residuals
        C_n = (E.T @ E) / max(1, N - k - 1)  # [d, d]
        # Regularise
        mu_cn = np.trace(C_n) / d
        C_n = 0.9 * C_n + 0.1 * mu_cn * np.eye(d)

        C_n_inv = np.linalg.inv(C_n)

        # 4. GLS regression
        # Transform: X* = X, Y* = Y @ C_n^{-1/2}
        # Equivalent: beta = (X' X)^{-1} X' Y then scale, but proper GLS
        # uses the full formula per-column simultaneously.
        # For multivariate: vec(B) = (I_d kron (X'X)^-1 X') vec(Y)
        # when errors have covariance C_n kron I_N.
        # With correlated responses, use:
        #   B_gls = (X' (C_n^{-1} kron I_N) X)^{-1} X' (C_n^{-1} kron I_N) Y
        # For our shape [N,d] this simplifies column-by-column with C_n^{-1}
        # weighting across columns. The standard approach:
        # Y_w = Y @ L^{-T} where C_n = L L^T, then OLS on Y_w.
        L = np.linalg.cholesky(C_n)
        L_inv = np.linalg.inv(L)
        Y_w = Y @ L_inv.T   # [N, d] whitened
        B_gls_w = XtX_inv @ X.T @ Y_w  # [k+1, d] in whitened space
        B_gls = B_gls_w @ L.T  # back to original space [k+1, d]

        # Predicted and residual
        Y_hat = X @ B_gls
        E_gls = Y - Y_hat

        # 5. Goodness of fit
        SS_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        SS_res = np.sum(E_gls ** 2)
        r_squared = 1.0 - SS_res / SS_tot if SS_tot > 0 else 0.0
        residual_norm = np.sqrt(SS_res)

        # 6. F-statistics and p-values per forcing
        # Residual covariance
        sigma2 = SS_res / max(1, N * d - (k + 1) * d)
        coefficients = {}
        partial_ss = {}

        for j, label in enumerate(labels):
            beta_j = B_gls[j + 1, :]  # [d]

            # Standard error: diagonal of (X'X)^{-1} * sigma2
            se_j = np.sqrt(np.abs(XtX_inv[j + 1, j + 1]) * sigma2) * np.ones(d)

            # Partial F-stat: drop this forcing, measure increase in SS_res
            mask = list(range(k + 1))
            mask.pop(j + 1)
            X_red = X[:, mask]
            XtX_red = X_red.T @ X_red
            XtX_red_inv = np.linalg.inv(XtX_red + 1e-10 * np.eye(len(mask)))
            B_red = XtX_red_inv @ X_red.T @ Y
            SS_res_red = np.sum((Y - X_red @ B_red) ** 2)
            partial_ss[label] = SS_res_red - SS_res

            df_num = d
            df_den = max(1, N * d - (k + 1) * d)
            f_stat = ((SS_res_red - SS_res) / df_num) / (SS_res / df_den)
            f_stat = max(0.0, f_stat)

            p_val = 1.0 - sp_stats.f.cdf(f_stat, df_num, df_den)

            coefficients[label] = EuclideanCoefficient(
                name=label,
                beta=beta_j,
                beta_se=se_j,
                f_statistic=f_stat,
                p_value_parametric=p_val,
            )

        # 7. Attribution fractions from partial SS
        total_partial = sum(max(0, v) for v in partial_ss.values())
        for label in labels:
            frac = max(0, partial_ss[label]) / total_partial if total_partial > 0 else 0.0
            coefficients[label].attribution_fraction = frac

        # 8. Block-permutation surrogates
        if test_surrogates and self.n_surrogates > 0:
            rng = np.random.RandomState(42)
            null_f = {label: [] for label in labels}

            for _ in range(self.n_surrogates):
                X_surr = X.copy()
                for j, label in enumerate(labels):
                    X_surr[:, j + 1] = self._block_permute(X[:, j + 1], rng)

                XtX_s = X_surr.T @ X_surr
                XtX_s_inv = np.linalg.inv(XtX_s + 1e-10 * np.eye(k + 1))
                B_s = XtX_s_inv @ X_surr.T @ Y
                SS_res_s = np.sum((Y - X_surr @ B_s) ** 2)

                for j, label in enumerate(labels):
                    mask_s = list(range(k + 1))
                    mask_s.pop(j + 1)
                    X_red_s = X_surr[:, mask_s]
                    XtX_red_s = X_red_s.T @ X_red_s
                    XtX_red_s_inv = np.linalg.inv(
                        XtX_red_s + 1e-10 * np.eye(len(mask_s))
                    )
                    B_red_s = XtX_red_s_inv @ X_red_s.T @ Y
                    SS_red_s = np.sum((Y - X_red_s @ B_red_s) ** 2)
                    df_num = d
                    df_den = max(1, N * d - (k + 1) * d)
                    f_s = ((SS_red_s - SS_res_s) / df_num) / (
                        SS_res_s / df_den
                    )
                    null_f[label].append(max(0.0, f_s))

            for label in labels:
                obs_f = coefficients[label].f_statistic
                null_arr = np.array(null_f[label])
                p_surr = np.mean(null_arr >= obs_f)
                coefficients[label].p_value_surrogate = float(p_surr)

        # 9. Build results
        attr_fracs = {
            label: coefficients[label].attribution_fraction
            for label in labels
        }
        dominant = max(attr_fracs, key=attr_fracs.get) if attr_fracs else ""

        interp = []
        for label in labels:
            c = coefficients[label]
            sig = "significant" if c.p_value_parametric < 0.05 else "not significant"
            interp.append(
                f"{label}: {c.attribution_fraction:.1%} attribution, "
                f"F={c.f_statistic:.2f}, p={c.p_value_parametric:.4f} ({sig})"
            )

        return EuclideanFingerprintResults(
            coefficients=coefficients,
            attribution_fractions=attr_fracs,
            r_squared=r_squared,
            residual_norm=residual_norm,
            n_windows=N,
            d_response=d,
            n_forcings=k,
            dominant_forcing=dominant,
            interpretation=interp,
        )


# =====================================================================
# Method comparison
# =====================================================================

class MethodComparison:
    """Compare Euclidean and Riemannian attribution on the same data.

    Runs both OptimalFingerprinting (Euclidean, Allen & Stott 2003) and
    TangentSpaceRegression (Riemannian) on identical inputs and produces
    a side-by-side comparison table.

    Key differences:
        - Euclidean works on temperature MEANS; Riemannian on COVARIANCE
          STRUCTURE (SPD matrices on the GL+(d)/SO(d) manifold).
        - Euclidean cannot distinguish V+ from V- (energy budget vs
          teleconnection reorganisation).
        - Riemannian is affine-invariant: rescaling indices does not
          change results. Euclidean results depend on units/normalisation.
    """

    def __init__(self, euclidean_kwargs: Optional[Dict] = None,
                 riemannian_kwargs: Optional[Dict] = None):
        """
        Args:
            euclidean_kwargs: kwargs for OptimalFingerprinting().
            riemannian_kwargs: kwargs for TangentSpaceRegression().
        """
        self.euclidean_kwargs = euclidean_kwargs or {}
        self.riemannian_kwargs = riemannian_kwargs or {}

    def compare(self, data: ClimateData,
                forcing_groups: Dict[str, List[str]],
                response_names: List[str],
                lag_months: Optional[Dict[str, int]] = None,
                verbose: bool = True,
                ) -> Dict[str, Dict]:
        """Run both methods and return comparison.

        Args:
            data: ClimateData with forcing and response indices.
            forcing_groups: dict mapping label -> forcing index names.
            response_names: response variable names.
            lag_months: dict mapping label -> lag in months.
            verbose: print formatted comparison table.

        Returns:
            Dict with keys 'euclidean', 'riemannian', 'table' containing
            results objects and a list-of-dicts comparison table.
        """
        from .climate_attribution import TangentSpaceRegression

        # Run Euclidean
        euc = OptimalFingerprinting(**self.euclidean_kwargs)
        euc_results = euc.regress(
            data, forcing_groups, response_names,
            lag_months=lag_months, test_surrogates=True,
        )

        # Run Riemannian
        riem = TangentSpaceRegression(**self.riemannian_kwargs)
        riem_results = riem.regress(
            data, forcing_groups, response_names,
            lag_months=lag_months, test_surrogates=True,
        )

        # Build comparison table
        labels = list(forcing_groups.keys())
        table = []
        for label in labels:
            ec = euc_results.coefficients[label]
            rc = riem_results.coefficients[label]
            row = {
                "forcing": label,
                "euc_attribution": ec.attribution_fraction,
                "euc_f_stat": ec.f_statistic,
                "euc_p_param": ec.p_value_parametric,
                "euc_p_surr": ec.p_value_surrogate,
                "riem_attribution": riem_results.attribution_fractions.get(label, 0.0),
                "riem_f_stat": rc.f_statistic,
                "riem_p_param": rc.p_value_parametric,
                "riem_p_surr": rc.p_value_surrogate,
                "riem_v_ratio": rc.v_ratio,
                "riem_mechanism": rc.mechanism,
            }
            table.append(row)

        if verbose:
            self._print_table(table, euc_results, riem_results)

        return {
            "euclidean": euc_results,
            "riemannian": riem_results,
            "table": table,
        }

    @staticmethod
    def _print_table(table: List[Dict],
                     euc: EuclideanFingerprintResults,
                     riem) -> None:
        """Print formatted comparison table."""
        print("\n" + "=" * 78)
        print("  ATTRIBUTION COMPARISON: Euclidean vs Riemannian")
        print("=" * 78)
        print(f"  Euclidean: {euc.n_windows} windows, "
              f"R^2 = {euc.r_squared:.3f}")
        print(f"  Riemannian: R^2 = {riem.r_squared:.3f}")
        print("-" * 78)
        print(f"  {'Forcing':<14} | {'Euclidean':^26} | {'Riemannian':^30}")
        print(f"  {'':14} | {'Attrib':>7} {'F':>6} {'p':>7} | "
              f"{'Attrib':>7} {'F':>6} {'p':>7} {'V+/V-':>6}")
        print("-" * 78)

        for row in table:
            euc_sig = "*" if row["euc_p_param"] < 0.05 else " "
            riem_sig = "*" if row["riem_p_param"] < 0.05 else " "
            # Use surrogate p if available
            ep = row["euc_p_surr"] if row["euc_p_surr"] is not None else row["euc_p_param"]
            rp = row["riem_p_surr"] if row["riem_p_surr"] is not None else row["riem_p_param"]

            print(
                f"  {row['forcing']:<14} | "
                f"{row['euc_attribution']:6.1%}{euc_sig} "
                f"{row['euc_f_stat']:6.1f} "
                f"{ep:7.4f} | "
                f"{row['riem_attribution']:6.1%}{riem_sig} "
                f"{row['riem_f_stat']:6.1f} "
                f"{rp:7.4f} "
                f"{row['riem_v_ratio']:5.2f}"
            )

        print("-" * 78)
        print("  * = significant at p < 0.05")
        print("  V+/V- ratio: >0.6 energy budget, <0.25 teleconnection, "
              "else mixed")
        print("  Euclidean cannot compute V+/V- (works on means, not covariance)")
        print("=" * 78 + "\n")
