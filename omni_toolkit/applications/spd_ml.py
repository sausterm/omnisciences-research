"""
Differentiable geometric primitives for machine learning on SPD manifolds.

The space of symmetric positive-definite (SPD) d×d matrices is the symmetric
space GL+(d)/SO(d), equipped with the affine-invariant (DeWitt) metric. This
module provides ML-ready building blocks that reuse the omni_toolkit
SymmetricSpace and DeWittMetric infrastructure:

  - SPDLayer: log/exp maps, Fréchet mean, geodesic distance, parallel transport
  - RiemannianBatchNorm: batch normalization on SPD manifolds
  - SPDKernel: kernel functions (affine-invariant Gaussian, log-Euclidean, Stein)
  - CovarianceDescriptor: extract SPD features from raw data

All operations work for arbitrary dimension d and accept batches of SPD matrices
as numpy arrays of shape (N, d, d).

References
----------
- Pennec, Fillard, Ayache (2006). A Riemannian Framework for Tensor Computing.
- Huang & Van Gool (2017). A Riemannian Network for SPD Matrix Learning.
- Barachant et al. (2012). Multiclass Brain-Computer Interface Classification
  by Riemannian Geometry.
- Tuzel, Porikli, Meer (2006). Region Covariance: A Fast Descriptor for
  Detection and Classification.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from scipy.linalg import logm, expm, sqrtm

from ..core.symmetric_space import SymmetricSpace, DeWittMetric


# =====================================================================
# Helpers
# =====================================================================

def _validate_spd(P: np.ndarray, name: str = "P") -> np.ndarray:
    """Validate that P is a d×d symmetric positive-definite matrix."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"{name} must be square, got shape {P.shape}")
    if np.max(np.abs(P - P.T)) > 1e-10:
        raise ValueError(f"{name} is not symmetric (max asymmetry: "
                         f"{np.max(np.abs(P - P.T)):.2e})")
    eigvals = np.linalg.eigvalsh(P)
    if eigvals[0] <= 0:
        raise ValueError(f"{name} is not positive-definite "
                         f"(min eigenvalue: {eigvals[0]:.2e})")
    return P


def _validate_spd_batch(matrices: np.ndarray, name: str = "matrices") -> np.ndarray:
    """Validate a batch of SPD matrices with shape (N, d, d)."""
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f"{name} must have shape (N, d, d), got {matrices.shape}")
    return matrices


def _matrix_sqrt(P: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, 1e-12)
    return (eigvecs * np.sqrt(eigvals)[np.newaxis, :]) @ eigvecs.T


def _matrix_invsqrt(P: np.ndarray) -> np.ndarray:
    """Inverse square root of an SPD matrix via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, 1e-12)
    return (eigvecs * (1.0 / np.sqrt(eigvals))[np.newaxis, :]) @ eigvecs.T


def _matrix_sqrt_pair(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sqrt, invsqrt) from a single eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, 1e-12)
    sqrt_ev = np.sqrt(eigvals)
    S = (eigvecs * sqrt_ev[np.newaxis, :]) @ eigvecs.T
    Si = (eigvecs * (1.0 / sqrt_ev)[np.newaxis, :]) @ eigvecs.T
    return S, Si


def _matrix_log(P: np.ndarray) -> np.ndarray:
    """Symmetric matrix logarithm via eigendecomposition.

    Works for 2D (d, d) or batched 3D (N, d, d) input.
    """
    if P.ndim == 3:
        return _matrix_log_batch(P)
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, 1e-12)
    return (eigvecs * np.log(eigvals)[np.newaxis, :]) @ eigvecs.T


def _matrix_log_batch(P: np.ndarray) -> np.ndarray:
    """Batched symmetric matrix logarithm for (N, d, d) arrays."""
    eigvals, eigvecs = np.linalg.eigh(P)  # (N, d), (N, d, d)
    eigvals = np.maximum(eigvals, 1e-12)
    log_ev = np.log(eigvals)  # (N, d)
    # (N, d, d) * (N, 1, d) -> broadcast, then batch matmul
    return (eigvecs * log_ev[:, np.newaxis, :]) @ np.swapaxes(eigvecs, -2, -1)


def _matrix_exp(S: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, -500.0, 500.0)
    return (eigvecs * np.exp(eigvals)[np.newaxis, :]) @ eigvecs.T


def _symmetrize(M: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix (2D) or batch of matrices (3D)."""
    return 0.5 * (M + np.swapaxes(M, -2, -1))


# =====================================================================
# SPDLayer
# =====================================================================

class SPDLayer:
    """Neural-network-compatible operations on SPD(d) matrices.

    Provides log/exp maps, Fréchet mean, geodesic distance, and parallel
    transport on the manifold of d×d symmetric positive-definite matrices
    equipped with the affine-invariant metric.

    Reuses the omni_toolkit SymmetricSpace / DeWittMetric infrastructure
    for the underlying GL+(d)/SO(d) geometry.

    Parameters
    ----------
    d : int
        Matrix dimension.
    lam : float
        DeWitt trace coupling (default 0.5).
    """

    def __init__(self, d: int, lam: float = 0.5):
        self.d = d
        self.lam = lam
        self.eta = np.eye(d)
        # Lazy-init: SymmetricSpace/DeWittMetric are expensive to construct
        # (O(d^4) metric tensor) but never actually used by our SPD operations.
        # Only create if explicitly accessed.
        self._space = None
        self._dewitt = None

    @property
    def space(self):
        if self._space is None:
            self._space = SymmetricSpace(self.eta, lam=self.lam)
        return self._space

    @property
    def dewitt(self):
        if self._dewitt is None:
            self._dewitt = self.space.dewitt
        return self._dewitt

    def log_map(self, P: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Riemannian logarithmic map: SPD(d) → T_{base} SPD(d).

        Maps P to a tangent vector at base along the geodesic connecting them.

        log_base(P) = base^{1/2} log(base^{-1/2} P base^{-1/2}) base^{1/2}

        Parameters
        ----------
        P : ndarray, shape (d, d) or (N, d, d)
            Point(s) on SPD(d).
        base : ndarray, shape (d, d)
            Base point for the tangent space.

        Returns
        -------
        ndarray, same shape as P
            Tangent vector(s) at base.
        """
        base = _validate_spd(base, "base")
        P = np.asarray(P, dtype=float)
        base_sqrt = _matrix_sqrt(base)
        base_invsqrt = _matrix_invsqrt(base)

        if P.ndim == 2:
            P = _validate_spd(P, "P")
            M = base_invsqrt @ P @ base_invsqrt
            return _symmetrize(base_sqrt @ _matrix_log(M) @ base_sqrt)

        # Batch: (N, d, d)
        P = _validate_spd_batch(P, "P")
        results = np.empty_like(P)
        for i in range(P.shape[0]):
            M = base_invsqrt @ P[i] @ base_invsqrt
            results[i] = _symmetrize(base_sqrt @ _matrix_log(M) @ base_sqrt)
        return results

    def exp_map(self, V: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Riemannian exponential map: T_{base} SPD(d) → SPD(d).

        Maps a tangent vector V at base to a point on the manifold.

        exp_base(V) = base^{1/2} exp(base^{-1/2} V base^{-1/2}) base^{1/2}

        Parameters
        ----------
        V : ndarray, shape (d, d) or (N, d, d)
            Tangent vector(s) at base.
        base : ndarray, shape (d, d)
            Base point on SPD(d).

        Returns
        -------
        ndarray, same shape as V
            Point(s) on SPD(d).
        """
        base = _validate_spd(base, "base")
        V = np.asarray(V, dtype=float)
        base_sqrt = _matrix_sqrt(base)
        base_invsqrt = _matrix_invsqrt(base)

        if V.ndim == 2:
            M = base_invsqrt @ V @ base_invsqrt
            return _symmetrize(base_sqrt @ _matrix_exp(M) @ base_sqrt)

        # Batch: (N, d, d)
        results = np.empty_like(V)
        for i in range(V.shape[0]):
            M = base_invsqrt @ V[i] @ base_invsqrt
            results[i] = _symmetrize(base_sqrt @ _matrix_exp(M) @ base_sqrt)
        return results

    def geodesic_distance(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Affine-invariant geodesic distance on SPD(d).

        d(P, Q) = ||log(P^{-1/2} Q P^{-1/2})||_F

        Parameters
        ----------
        P : ndarray, shape (d, d) or (N, d, d)
            First SPD matrix or batch.
        Q : ndarray, shape (d, d) or (N, d, d)
            Second SPD matrix or batch.

        Returns
        -------
        float or ndarray of shape (N,)
            Geodesic distance(s).
        """
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)

        if P.ndim == 2 and Q.ndim == 2:
            P = _validate_spd(P, "P")
            Q = _validate_spd(Q, "Q")
            P_invsqrt = _matrix_invsqrt(P)
            M = P_invsqrt @ Q @ P_invsqrt
            eigvals = np.linalg.eigvalsh(M)
            return float(np.sqrt(np.sum(np.log(np.maximum(eigvals, 1e-12)) ** 2)))

        # Batch
        if P.ndim == 2:
            P = np.broadcast_to(P, Q.shape)
        if Q.ndim == 2:
            Q = np.broadcast_to(Q, P.shape)
        N = P.shape[0]
        dists = np.empty(N)
        for i in range(N):
            P_invsqrt = _matrix_invsqrt(P[i])
            M = P_invsqrt @ Q[i] @ P_invsqrt
            eigvals = np.linalg.eigvalsh(M)
            dists[i] = np.sqrt(np.sum(np.log(np.maximum(eigvals, 1e-12)) ** 2))
        return dists

    def frechet_mean(self, matrices: np.ndarray,
                     weights: Optional[np.ndarray] = None,
                     max_iter: int = 15,
                     tol: float = 1e-6) -> np.ndarray:
        """Iterative Karcher/Fréchet mean on SPD(d).

        Optimized implementation:
          - Log-Euclidean initialization (much closer to true mean → fewer iters)
          - Batched matrix operations (single eigendecomp for N matrices)
          - Shared sqrt/invsqrt eigendecomposition

        Parameters
        ----------
        matrices : ndarray, shape (N, d, d)
            Collection of SPD matrices.
        weights : ndarray, shape (N,), optional
            Non-negative weights summing to 1. Uniform if None.
        max_iter : int
            Maximum number of iterations (default 15).
        tol : float
            Convergence tolerance on tangent vector norm (default 1e-6).

        Returns
        -------
        ndarray, shape (d, d)
            The Fréchet mean (SPD matrix).
        """
        matrices = _validate_spd_batch(matrices, "matrices")
        N = matrices.shape[0]
        d = matrices.shape[1]

        if weights is None:
            weights = np.ones(N) / N
        else:
            weights = np.asarray(weights, dtype=float)
            weights = weights / weights.sum()

        # ── Initialize with log-Euclidean mean (much better than arithmetic) ──
        # log-Euclidean mean = exp(Σ w_i log(S_i)) — one batched eigendecomp
        log_matrices = _matrix_log_batch(matrices)  # (N, d, d)
        weighted_log = np.tensordot(weights, log_matrices, axes=([0], [0]))  # (d, d)
        mean = _matrix_exp(_symmetrize(weighted_log))

        for iteration in range(max_iter):
            mean = _symmetrize(mean)
            # Ensure SPD
            eigvals_m = np.linalg.eigvalsh(mean)
            if eigvals_m[0] <= 0:
                mean += (abs(eigvals_m[0]) + 1e-10) * np.eye(d)

            # Single eigendecomp for both sqrt and invsqrt
            mean_sqrt, mean_invsqrt = _matrix_sqrt_pair(mean)

            # ── Batched log-map: compute all N transformed matrices at once ──
            # M_i = mean^{-1/2} @ S_i @ mean^{-1/2}  for all i
            # Shape: (N, d, d)
            Si = mean_invsqrt[np.newaxis]  # (1, d, d) for broadcasting
            transformed = Si @ matrices @ Si  # (N, d, d)
            transformed = _symmetrize(transformed)  # (N, d, d) — batched symmetrize works

            # Batched matrix log — single call to eigh on (N, d, d)
            log_transformed = _matrix_log_batch(transformed)  # (N, d, d)

            # Weighted tangent vector via tensordot
            tangent = np.tensordot(weights, log_transformed, axes=([0], [0]))  # (d, d)
            tangent = _symmetrize(tangent)

            # Check convergence
            norm = np.linalg.norm(tangent, 'fro')
            if norm < tol:
                break

            # Dampen step if tangent is large
            step = tangent if norm <= 1.0 else tangent / norm

            # Step along geodesic
            mean = _symmetrize(mean_sqrt @ _matrix_exp(step) @ mean_sqrt)

        return mean

    def parallel_transport(self, V: np.ndarray, P: np.ndarray,
                           Q: np.ndarray) -> np.ndarray:
        """Parallel transport of tangent vector V from T_P to T_Q along geodesic.

        Uses the Schild's ladder closed-form for the affine-invariant metric:
          Γ_{P→Q}(V) = E V E^T,   where E = (QP^{-1})^{1/2}

        Parameters
        ----------
        V : ndarray, shape (d, d) or (N, d, d)
            Tangent vector(s) at P.
        P : ndarray, shape (d, d)
            Source point on SPD(d).
        Q : ndarray, shape (d, d)
            Target point on SPD(d).

        Returns
        -------
        ndarray, same shape as V
            Transported tangent vector(s) at Q.
        """
        P = _validate_spd(P, "P")
        Q = _validate_spd(Q, "Q")
        V = np.asarray(V, dtype=float)

        # E = (Q P^{-1})^{1/2}
        P_inv = np.linalg.inv(P)
        QP_inv = Q @ P_inv
        # Use real part of sqrtm for numerical stability
        E = np.real(sqrtm(QP_inv))

        if V.ndim == 2:
            return _symmetrize(E @ V @ E.T)

        # Batch
        results = np.empty_like(V)
        for i in range(V.shape[0]):
            results[i] = _symmetrize(E @ V[i] @ E.T)
        return results


# =====================================================================
# RiemannianBatchNorm
# =====================================================================

class RiemannianBatchNorm:
    """Batch normalization on SPD manifolds.

    Normalizes a batch of SPD matrices by:
      1. Computing the running Fréchet mean
      2. Transporting all matrices to the identity via the mean
      3. Scaling by geodesic variance
      4. Applying learnable bias (SPD) and scale (positive scalar)

    Parameters
    ----------
    d : int
        Matrix dimension.
    momentum : float
        Momentum for running statistics update (default 0.1).
    epsilon : float
        Small constant for numerical stability in variance (default 1e-5).
    """

    def __init__(self, d: int, momentum: float = 0.1, epsilon: float = 1e-5):
        self.d = d
        self.momentum = momentum
        self.epsilon = epsilon
        self.layer = SPDLayer(d)

        # Running statistics
        self.running_mean: np.ndarray = np.eye(d)
        self.running_var: float = 1.0

        # Learnable parameters
        self.bias: np.ndarray = np.eye(d)  # SPD bias
        self.scale: float = 1.0  # positive scalar

    def compute_stats(self, matrices: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute batch Fréchet mean and geodesic variance.

        Parameters
        ----------
        matrices : ndarray, shape (N, d, d)
            Batch of SPD matrices.

        Returns
        -------
        mean : ndarray, shape (d, d)
            Fréchet mean of the batch.
        var : float
            Mean squared geodesic distance to the mean.
        """
        matrices = _validate_spd_batch(matrices, "matrices")
        mean = self.layer.frechet_mean(matrices)

        # Geodesic variance = mean of d²(X_i, mean)
        dists = self.layer.geodesic_distance(matrices, mean)
        var = float(np.mean(dists ** 2))

        return mean, var

    def forward(self, matrices: np.ndarray,
                training: bool = True) -> np.ndarray:
        """Apply Riemannian batch normalization.

        Parameters
        ----------
        matrices : ndarray, shape (N, d, d)
            Batch of SPD matrices.
        training : bool
            If True, compute and update running statistics.
            If False, use running statistics.

        Returns
        -------
        ndarray, shape (N, d, d)
            Normalized SPD matrices.
        """
        matrices = _validate_spd_batch(matrices, "matrices")
        N = matrices.shape[0]

        if training:
            mean, var = self.compute_stats(matrices)
            # Update running statistics
            self.running_mean = self.layer.exp_map(
                (1 - self.momentum) * self.layer.log_map(self.running_mean, mean)
                + self.momentum * np.zeros((self.d, self.d)),
                mean
            )
            # Simpler running mean update: geometric interpolation
            self.running_mean = self.layer.frechet_mean(
                np.stack([self.running_mean, mean]),
                weights=np.array([1 - self.momentum, self.momentum])
            )
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize: transport to identity and scale
        mean_invsqrt = _matrix_invsqrt(mean)
        mean_sqrt = _matrix_sqrt(mean)
        std_inv = 1.0 / np.sqrt(var + self.epsilon)

        # Bias square root for applying bias
        bias_sqrt = _matrix_sqrt(self.bias)

        result = np.empty_like(matrices)
        for i in range(N):
            # Transport to identity: M = mean^{-1/2} X mean^{-1/2}
            M = mean_invsqrt @ matrices[i] @ mean_invsqrt
            # Scale in log space: log(M) * scale * std_inv
            logM = _matrix_log(M)
            scaled = logM * self.scale * std_inv
            # Map back and apply bias
            normalized = _matrix_exp(scaled)
            # Apply bias: bias^{1/2} * normalized * bias^{1/2}
            result[i] = _symmetrize(bias_sqrt @ normalized @ bias_sqrt)

        return result


# =====================================================================
# SPDKernel
# =====================================================================

class SPDKernel:
    """Kernel functions for SPD matrix data.

    Provides three kernel functions on SPD(d):
      - Affine-invariant Gaussian: k(P,Q) = exp(-γ d²_{AI}(P,Q))
      - Log-Euclidean Gaussian: k(P,Q) = exp(-γ ||logm(P) - logm(Q)||²_F)
      - Stein divergence: k(P,Q) = exp(-γ S(P,Q))

    Parameters
    ----------
    gamma : float
        Kernel bandwidth parameter (default 1.0).
    """

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        self._layer_cache = {}

    def _get_layer(self, d: int) -> SPDLayer:
        """Get or create an SPDLayer for dimension d."""
        if d not in self._layer_cache:
            self._layer_cache[d] = SPDLayer(d)
        return self._layer_cache[d]

    def affine_invariant(self, P: np.ndarray, Q: np.ndarray) -> float:
        """Affine-invariant Gaussian kernel.

        k(P, Q) = exp(-γ · d²_{AI}(P, Q))

        where d_{AI} is the affine-invariant geodesic distance.

        Parameters
        ----------
        P, Q : ndarray, shape (d, d)
            SPD matrices.

        Returns
        -------
        float
            Kernel value in (0, 1].
        """
        P = _validate_spd(P, "P")
        Q = _validate_spd(Q, "Q")
        layer = self._get_layer(P.shape[0])
        dist_sq = layer.geodesic_distance(P, Q) ** 2
        return float(np.exp(-self.gamma * dist_sq))

    def log_euclidean(self, P: np.ndarray, Q: np.ndarray) -> float:
        """Log-Euclidean Gaussian kernel.

        k(P, Q) = exp(-γ · ||logm(P) - logm(Q)||²_F)

        Maps SPD matrices to the Euclidean space of symmetric matrices
        via the matrix logarithm, then uses the Frobenius norm.

        Parameters
        ----------
        P, Q : ndarray, shape (d, d)
            SPD matrices.

        Returns
        -------
        float
            Kernel value in (0, 1].
        """
        P = _validate_spd(P, "P")
        Q = _validate_spd(Q, "Q")
        logP = _matrix_log(P)
        logQ = _matrix_log(Q)
        dist_sq = np.sum((logP - logQ) ** 2)
        return float(np.exp(-self.gamma * dist_sq))

    def stein_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:
        """Stein divergence kernel.

        k(P, Q) = exp(-γ · S(P, Q))

        where S(P, Q) = ln det((P+Q)/2) - 0.5 ln det(PQ)
                       = ln det((P+Q)/2) - 0.5 (ln det P + ln det Q)

        The Stein divergence is a symmetric, non-negative Bregman divergence
        on SPD matrices that is also affine-invariant.

        Parameters
        ----------
        P, Q : ndarray, shape (d, d)
            SPD matrices.

        Returns
        -------
        float
            Kernel value in (0, 1].
        """
        P = _validate_spd(P, "P")
        Q = _validate_spd(Q, "Q")

        avg = 0.5 * (P + Q)
        _, logdet_avg = np.linalg.slogdet(avg)
        _, logdet_P = np.linalg.slogdet(P)
        _, logdet_Q = np.linalg.slogdet(Q)

        S = logdet_avg - 0.5 * (logdet_P + logdet_Q)
        return float(np.exp(-self.gamma * max(S, 0.0)))

    def gram_matrix(self, matrices: np.ndarray,
                    kernel: str = "affine_invariant") -> np.ndarray:
        """Compute the Gram (kernel) matrix for a collection of SPD matrices.

        Parameters
        ----------
        matrices : ndarray, shape (N, d, d)
            Collection of N SPD matrices.
        kernel : str
            Kernel type: 'affine_invariant', 'log_euclidean', or 'stein'.

        Returns
        -------
        ndarray, shape (N, N)
            Symmetric positive-semidefinite Gram matrix.
        """
        matrices = _validate_spd_batch(matrices, "matrices")
        N = matrices.shape[0]

        kernel_fn = {
            'affine_invariant': self.affine_invariant,
            'log_euclidean': self.log_euclidean,
            'stein': self.stein_divergence,
        }
        if kernel not in kernel_fn:
            raise ValueError(f"Unknown kernel '{kernel}'. "
                             f"Choose from: {list(kernel_fn.keys())}")
        fn = kernel_fn[kernel]

        G = np.empty((N, N))
        for i in range(N):
            G[i, i] = 1.0  # k(P, P) = exp(0) = 1 for all kernels
            for j in range(i + 1, N):
                val = fn(matrices[i], matrices[j])
                G[i, j] = val
                G[j, i] = val
        return G


# =====================================================================
# CovarianceDescriptor
# =====================================================================

class CovarianceDescriptor:
    """Extract SPD covariance descriptors from raw data.

    Provides methods to compute covariance matrices from feature vectors
    and time-series data, producing SPD matrices suitable for Riemannian
    ML pipelines.

    Parameters
    ----------
    regularization : float
        Regularization parameter for shrinkage estimator (default 1e-6).
    """

    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization

    def region_covariance(self, features: np.ndarray) -> np.ndarray:
        """Region covariance descriptor (Tuzel et al., 2006).

        Computes the sample covariance of a set of feature vectors,
        producing an SPD matrix that summarizes the region statistics.

        Parameters
        ----------
        features : ndarray, shape (N, d)
            N feature vectors of dimension d. Each row is a feature vector
            extracted from a region (e.g., pixel features in an image patch).

        Returns
        -------
        ndarray, shape (d, d)
            SPD covariance descriptor.
        """
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D (N, d), got shape {features.shape}")
        N, d = features.shape
        if N < 2:
            raise ValueError(f"Need at least 2 feature vectors, got {N}")

        # Mean-centered features
        centered = features - features.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / (N - 1)

        # Regularize for guaranteed positive-definiteness
        cov += self.regularization * np.eye(d)

        return _symmetrize(cov)

    def temporal_covariance(self, data: np.ndarray,
                            window_size: int,
                            stride: int = 1) -> np.ndarray:
        """Temporal covariance from time-series windows.

        Slides a window across a multi-channel time series and computes
        the channel-channel covariance matrix for each window, producing
        a sequence of SPD matrices.

        Parameters
        ----------
        data : ndarray, shape (T, C)
            Time series with T time points and C channels
            (e.g., EEG electrodes, sensor array channels).
        window_size : int
            Number of time points per window. Must be >= 2.
        stride : int
            Step size between consecutive windows (default 1).

        Returns
        -------
        ndarray, shape (n_windows, C, C)
            Stack of SPD covariance descriptors, one per window.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"data must be 2D (T, C), got shape {data.shape}")
        T, C = data.shape
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if window_size > T:
            raise ValueError(f"window_size ({window_size}) > time points ({T})")

        n_windows = (T - window_size) // stride + 1
        covariances = np.empty((n_windows, C, C))

        for i in range(n_windows):
            start = i * stride
            window = data[start:start + window_size]
            covariances[i] = self.region_covariance(window)

        return covariances

    def shrinkage_estimator(self, features: np.ndarray,
                            alpha: Optional[float] = None) -> np.ndarray:
        """Ledoit-Wolf-style shrinkage covariance estimator.

        Computes a regularized covariance by shrinking the sample covariance
        toward a structured target (scaled identity):

          Σ_shrunk = (1 - α) Σ_sample + α · (tr(Σ)/d) · I

        This is well-conditioned even when N < d (more features than samples).

        Parameters
        ----------
        features : ndarray, shape (N, d)
            N feature vectors of dimension d.
        alpha : float, optional
            Shrinkage intensity in [0, 1]. If None, uses the Oracle
            Approximating Shrinkage (OAS) estimator.

        Returns
        -------
        ndarray, shape (d, d)
            Shrinkage-regularized SPD covariance matrix.
        """
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D (N, d), got shape {features.shape}")
        N, d = features.shape
        if N < 2:
            raise ValueError(f"Need at least 2 feature vectors, got {N}")

        centered = features - features.mean(axis=0, keepdims=True)
        sample_cov = (centered.T @ centered) / (N - 1)

        # Target: scaled identity
        mu = np.trace(sample_cov) / d
        target = mu * np.eye(d)

        if alpha is None:
            # OAS estimator (Chen, Wiesel, Eldar, Hero 2010)
            rho_num = (1 - 2.0 / d) * np.sum(sample_cov ** 2) + mu ** 2 * d
            rho_den = (N + 1 - 2.0 / d) * (np.sum(sample_cov ** 2) - mu ** 2 * d)
            if abs(rho_den) < 1e-30:
                alpha = 1.0
            else:
                alpha = float(np.clip(rho_num / rho_den, 0.0, 1.0))

        result = (1 - alpha) * sample_cov + alpha * target
        result += self.regularization * np.eye(d)

        return _symmetrize(result)


# =====================================================================
# Geodesic shrinkage
# =====================================================================

def geodesic_shrinkage(C: np.ndarray, t: float,
                       target: np.ndarray = None) -> np.ndarray:
    """Shrink C toward target along the geodesic on SPD(d).

    Euclidean: C_s = (1-t)*C + t*T  (breaks affine invariance)
    Geodesic:  C_s = C^{1/2} (C^{-1/2} T C^{-1/2})^t C^{1/2}

    For target = mu*I (scaled identity), simplifies to: mu^t * C^{1-t}

    Args:
        C: d x d SPD matrix
        t: shrinkage intensity in [0,1]. t=0 -> C, t=1 -> target
        target: d x d SPD target matrix. Default: (tr(C)/d)*I
    Returns:
        d x d SPD shrunk matrix
    """
    C = _validate_spd(C, "C")
    d = C.shape[0]

    if target is None:
        # Default target: scaled identity mu*I where mu = tr(C)/d
        mu = np.trace(C) / d
        # Simplified formula: mu^t * C^{1-t}
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        shrunk_eigvals = (mu ** t) * (eigvals ** (1.0 - t))
        result = (eigvecs * shrunk_eigvals[np.newaxis, :]) @ eigvecs.T
        return _symmetrize(result)

    # General case: full geodesic formula
    target = _validate_spd(target, "target")
    C_sqrt, C_invsqrt = _matrix_sqrt_pair(C)

    # M = C^{-1/2} T C^{-1/2}
    M = C_invsqrt @ target @ C_invsqrt
    M = _symmetrize(M)

    # M^t via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    Mt = (eigvecs * (eigvals ** t)[np.newaxis, :]) @ eigvecs.T

    # C_s = C^{1/2} M^t C^{1/2}
    result = C_sqrt @ Mt @ C_sqrt
    return _symmetrize(result)


# =====================================================================
# Tyler's M-estimator
# =====================================================================

def tyler_m_estimator(data: np.ndarray, max_iter: int = 100,
                      tol: float = 1e-8) -> np.ndarray:
    """Tyler's robust M-estimator of scatter.

    Iterates: C_{k+1} = (d/n) * sum_i (x_i x_i^T) / (x_i^T C_k^{-1} x_i)
    Converges to the MLE under elliptical distributions.
    Result has det(C) = 1.

    Args:
        data: [n, d] array of observations
        max_iter: maximum iterations
        tol: convergence tolerance on relative change
    Returns:
        d x d SPD matrix with det=1
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n, d), got shape {data.shape}")
    n, d = data.shape
    if n < d + 1:
        raise ValueError(f"Need n > d observations, got n={n}, d={d}")

    # Initialize with sample covariance
    centered = data - data.mean(axis=0, keepdims=True)
    C = (centered.T @ centered) / (n - 1)
    C = _symmetrize(C)
    # Ensure SPD
    eigvals = np.linalg.eigvalsh(C)
    if eigvals[0] <= 0:
        C += (abs(eigvals[0]) + 1e-8) * np.eye(d)

    # Normalize to det=1
    det_C = np.linalg.det(C)
    if det_C > 0:
        C = C / (det_C ** (1.0 / d))

    for iteration in range(max_iter):
        C_inv = np.linalg.inv(C)

        # Mahalanobis distances: x_i^T C^{-1} x_i
        # Efficient: (data @ C_inv) * data summed over columns
        mahal = np.sum((data @ C_inv) * data, axis=1)  # [n]
        mahal = np.maximum(mahal, 1e-15)

        # Weighted scatter: (d/n) * sum_i (x_i x_i^T) / mahal_i
        weights = 1.0 / mahal  # [n]
        C_new = (d / n) * (data.T * weights[np.newaxis, :]) @ data
        C_new = _symmetrize(C_new)

        # Normalize det=1
        det_new = np.linalg.det(C_new)
        if det_new > 0:
            C_new = C_new / (det_new ** (1.0 / d))

        # Check convergence: relative change in Frobenius norm
        rel_change = np.linalg.norm(C_new - C, 'fro') / max(
            np.linalg.norm(C, 'fro'), 1e-15)
        C = C_new

        if rel_change < tol:
            break
    else:
        # Did not converge -- fall back to normalized sample covariance
        centered = data - data.mean(axis=0, keepdims=True)
        C = (centered.T @ centered) / (n - 1)
        C = _symmetrize(C)
        eigvals = np.linalg.eigvalsh(C)
        if eigvals[0] <= 0:
            C += (abs(eigvals[0]) + 1e-8) * np.eye(d)
        det_C = np.linalg.det(C)
        if det_C > 0:
            C = C / (det_C ** (1.0 / d))

    return C


# =====================================================================
# Power-Euclidean operations
# =====================================================================

def _matrix_power(P: np.ndarray, alpha: float) -> np.ndarray:
    """Raise an SPD matrix to a real power via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, 1e-12)
    return (eigvecs * (eigvals ** alpha)[np.newaxis, :]) @ eigvecs.T


def power_euclidean_distance(C1: np.ndarray, C2: np.ndarray,
                              alpha: float = 0.5) -> float:
    """Power-Euclidean distance: (1/alpha) * ||C1^alpha - C2^alpha||_F.

    Interpolates between log-Euclidean (alpha -> 0) and Euclidean (alpha = 1).

    Args:
        C1: d x d SPD matrix
        C2: d x d SPD matrix
        alpha: power parameter (default 0.5)
    Returns:
        Non-negative distance.
    """
    C1 = _validate_spd(C1, "C1")
    C2 = _validate_spd(C2, "C2")
    C1a = _matrix_power(C1, alpha)
    C2a = _matrix_power(C2, alpha)
    return float(np.linalg.norm(C1a - C2a, 'fro') / alpha)


def power_euclidean_mean(matrices: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    """Closed-form power-Euclidean mean: ((1/N) sum C_i^alpha)^{1/alpha}.

    Args:
        matrices: [N, d, d] array of SPD matrices
        alpha: power parameter (default 0.5)
    Returns:
        d x d SPD mean matrix
    """
    matrices = _validate_spd_batch(matrices, "matrices")
    N = matrices.shape[0]
    # Compute C_i^alpha for each matrix
    powered = np.array([_matrix_power(matrices[i], alpha) for i in range(N)])
    # Arithmetic mean in the powered space
    mean_powered = powered.mean(axis=0)
    mean_powered = _symmetrize(mean_powered)
    # Invert the power: raise to 1/alpha
    result = _matrix_power(mean_powered, 1.0 / alpha)
    return _symmetrize(result)


def power_euclidean_log_map(C: np.ndarray, base: np.ndarray,
                             alpha: float = 0.5) -> np.ndarray:
    """Power-Euclidean tangent vector: (1/alpha) * (C^alpha - base^alpha).

    Maps C to a tangent vector at base in the power-Euclidean geometry.

    Args:
        C: d x d SPD matrix (or [N, d, d] batch)
        base: d x d SPD base point
        alpha: power parameter (default 0.5)
    Returns:
        d x d symmetric matrix (tangent vector), or [N, d, d] batch
    """
    base = _validate_spd(base, "base")
    C = np.asarray(C, dtype=float)
    base_a = _matrix_power(base, alpha)

    if C.ndim == 2:
        C = _validate_spd(C, "C")
        Ca = _matrix_power(C, alpha)
        return _symmetrize((Ca - base_a) / alpha)

    # Batch: [N, d, d]
    C = _validate_spd_batch(C, "C")
    N = C.shape[0]
    results = np.empty_like(C)
    for i in range(N):
        Ca = _matrix_power(C[i], alpha)
        results[i] = _symmetrize((Ca - base_a) / alpha)
    return results
