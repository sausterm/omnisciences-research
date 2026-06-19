"""
Fourier Grid Hamiltonian (FGH) solver for 1D proton vibrational states.

Solves the 1D Schrödinger equation numerically on a grid, replacing the
analytic harmonic oscillator approximation with exact eigenstates of
arbitrary proton potentials (Morse, double-well, etc.).

This closes the accuracy gap with pyPCET while keeping our Hessian-to-rate
pipeline intact: we construct the 1D proton potential from the reactant
and product Hessian data, then solve numerically.

The FGH method:
    1. Define V(x) on a uniform grid of N points
    2. Build kinetic energy matrix T via FFT (spectral representation)
    3. Diagonalize H = T + V → eigenvalues ε_n, eigenvectors ψ_n(x)
    4. Compute overlaps S_μν = ∫ ψ_μ^R(x) ψ_ν^P(x) dx numerically

Reference:
    Marston, C. C. & Balint-Kurti, G. G. J. Chem. Phys. 91, 3571 (1989).
    Colbert, D. T. & Miller, W. H. J. Chem. Phys. 96, 1982 (1992).
"""

import math
import numpy as np
from dataclasses import dataclass

from pcet_engine.core.constants import (
    HBAR_AU,
    AMU_TO_AU,
    ANGSTROM_TO_BOHR,
    CM_TO_HARTREE,
    KB_HARTREE,
)


@dataclass
class FGHResult:
    """Result from FGH eigenvalue calculation.

    Attributes:
        energies: Eigenvalues in hartree, shape (n_states,).
        wavefunctions: Eigenvectors on grid, shape (n_grid, n_states).
        grid: Position grid in bohr, shape (n_grid,).
        potential: Potential on grid in hartree, shape (n_grid,).
    """

    energies: np.ndarray
    wavefunctions: np.ndarray
    grid: np.ndarray
    potential: np.ndarray


def harmonic_potential(
    grid: np.ndarray,
    omega_au: float,
    mass_au: float,
    x_eq: float = 0.0,
    energy_offset: float = 0.0,
) -> np.ndarray:
    """Harmonic oscillator potential on a grid.

    V(x) = ½ m ω² (x - x_eq)² + offset

    Args:
        grid: Position grid in bohr.
        omega_au: Frequency in atomic units.
        mass_au: Mass in atomic units (electron masses).
        x_eq: Equilibrium position in bohr.
        energy_offset: Energy offset in hartree.

    Returns:
        Potential array in hartree.
    """
    k = mass_au * omega_au**2
    return 0.5 * k * (grid - x_eq) ** 2 + energy_offset


def morse_potential(
    grid: np.ndarray,
    omega_au: float,
    mass_au: float,
    D_e: float,
    x_eq: float = 0.0,
    energy_offset: float = 0.0,
) -> np.ndarray:
    """Morse potential on a grid.

    V(x) = D_e [1 - exp(-a(x - x_eq))]² + offset

    where a = omega × sqrt(mass / (2 D_e)).

    The Morse potential matches the harmonic oscillator near x_eq but
    has the correct asymptotic behavior (finite dissociation energy).
    This gives vibrational wavefunctions with longer tails compared to
    harmonic oscillators, leading to larger Franck-Condon overlaps at
    large tunneling distances.

    Args:
        grid: Position grid in bohr.
        omega_au: Frequency in atomic units.
        mass_au: Mass in atomic units.
        D_e: Dissociation energy in hartree.
        x_eq: Equilibrium position in bohr.
        energy_offset: Energy offset in hartree.

    Returns:
        Potential array in hartree.
    """
    a = omega_au * math.sqrt(mass_au / (2.0 * D_e))
    return D_e * (1.0 - np.exp(-a * (grid - x_eq))) ** 2 + energy_offset


def solve_1d_schrodinger(
    grid: np.ndarray,
    potential: np.ndarray,
    mass_au: float,
    n_states: int = 10,
) -> FGHResult:
    """Solve the 1D Schrödinger equation using the FGH method.

    Constructs the Hamiltonian matrix on a uniform grid:
        H_{ij} = T_{ij} + V(x_i) δ_{ij}

    where T is the kinetic energy matrix computed via the Colbert-Miller
    DVR formula (sinc-function basis):

        T_{ii} = (ℏ² π²) / (6 m Δx²)
        T_{ij} = (ℏ² (-1)^{i-j}) / (m Δx² (i-j)²)    for i ≠ j

    Args:
        grid: Uniform position grid in bohr, shape (N,).
        potential: Potential energy on grid in hartree, shape (N,).
        mass_au: Particle mass in atomic units (electron masses).
        n_states: Number of eigenstates to return.

    Returns:
        FGHResult with energies, wavefunctions, grid, and potential.
    """
    N = len(grid)
    dx = grid[1] - grid[0]
    n_states = min(n_states, N)

    # Build kinetic energy matrix (Colbert-Miller DVR)
    T = np.zeros((N, N))
    prefactor = HBAR_AU**2 / (2.0 * mass_au * dx**2)

    for i in range(N):
        T[i, i] = prefactor * math.pi**2 / 3.0
        for j in range(i + 1, N):
            diff = i - j
            T[i, j] = prefactor * 2.0 * (-1) ** diff / diff**2
            T[j, i] = T[i, j]

    # Full Hamiltonian
    H = T + np.diag(potential)

    # Diagonalize (only need lowest n_states)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Normalize wavefunctions: ∫|ψ|² dx = 1
    for k in range(min(n_states, N)):
        norm = np.sqrt(np.sum(eigenvectors[:, k] ** 2) * dx)
        if norm > 1e-15:
            eigenvectors[:, k] /= norm

    return FGHResult(
        energies=eigenvalues[:n_states],
        wavefunctions=eigenvectors[:, :n_states],
        grid=grid,
        potential=potential,
    )


def numerical_fc_overlap(
    result_R: FGHResult,
    result_P: FGHResult,
    mu: int,
    nu: int,
) -> float:
    """Compute |<ψ_μ^R | ψ_ν^P>|² numerically on a shared grid.

    The reactant and product wavefunctions must be defined on the same
    grid (same grid points and spacing).

    Args:
        result_R: FGH solution for reactant potential.
        result_P: FGH solution for product potential.
        mu: Reactant vibrational quantum number.
        nu: Product vibrational quantum number.

    Returns:
        |S_μν|² (dimensionless).
    """
    if mu >= result_R.wavefunctions.shape[1]:
        return 0.0
    if nu >= result_P.wavefunctions.shape[1]:
        return 0.0

    dx = result_R.grid[1] - result_R.grid[0]
    psi_R = result_R.wavefunctions[:, mu]
    psi_P = result_P.wavefunctions[:, nu]

    overlap = np.sum(psi_R * psi_P) * dx
    return overlap**2


def build_proton_potentials(
    omega_R_cm: float,
    omega_P_cm: float,
    mass_amu: float,
    delta_0_angstrom: float,
    delta_G_hartree: float = 0.0,
    potential_type: str = "morse",
    D_e_hartree: float | None = None,
    n_grid: int = 256,
    grid_padding: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build reactant and product proton potentials from Hessian-derived parameters.

    Constructs two 1D potentials along the proton transfer coordinate:
    - Reactant: centered at x = 0 (proton on donor)
    - Product: centered at x = δ₀ (proton on acceptor), offset by ΔG

    For "morse" type, uses Morse potential which captures anharmonicity
    and gives more realistic wavefunction tails. For "harmonic", uses
    simple parabolas (equivalent to the analytic FC overlap approach).

    Args:
        omega_R_cm: Reactant proton frequency in cm⁻¹.
        omega_P_cm: Product proton frequency in cm⁻¹.
        mass_amu: Transferring particle mass in amu (1.008 for H, 2.014 for D).
        delta_0_angstrom: Tunneling distance in angstrom.
        delta_G_hartree: Vibronic driving force in hartree (negative = exothermic).
        potential_type: "morse" or "harmonic".
        D_e_hartree: Dissociation energy in hartree for Morse potential.
            If None, estimated as 20 × ℏω (reasonable for O-H, N-H, C-H bonds).
        n_grid: Number of grid points.
        grid_padding: Extra grid range beyond the two minima in angstrom.

    Returns:
        Tuple of (grid_bohr, V_reactant_hartree, V_product_hartree).
    """
    mass_au = mass_amu * AMU_TO_AU
    omega_R_au = omega_R_cm * CM_TO_HARTREE
    omega_P_au = omega_P_cm * CM_TO_HARTREE
    delta_0_bohr = delta_0_angstrom * ANGSTROM_TO_BOHR

    # Grid: covers both wells with padding
    x_min = -grid_padding * ANGSTROM_TO_BOHR
    x_max = delta_0_bohr + grid_padding * ANGSTROM_TO_BOHR
    grid = np.linspace(x_min, x_max, n_grid)

    if D_e_hartree is None:
        # Estimate: D_e ≈ 20 × ℏω for typical X-H bonds
        # C-H: ~100 kcal/mol ≈ 0.16 hartree, ω ≈ 3000 cm⁻¹ ≈ 0.014 hartree
        # ratio ~11, but 20 is conservative and works well
        D_e_hartree = 20.0 * omega_R_au

    if potential_type == "morse":
        V_R = morse_potential(grid, omega_R_au, mass_au, D_e_hartree, x_eq=0.0)
        V_P = morse_potential(
            grid, omega_P_au, mass_au, D_e_hartree,
            x_eq=delta_0_bohr, energy_offset=delta_G_hartree,
        )
    elif potential_type == "harmonic":
        V_R = harmonic_potential(grid, omega_R_au, mass_au, x_eq=0.0)
        V_P = harmonic_potential(
            grid, omega_P_au, mass_au,
            x_eq=delta_0_bohr, energy_offset=delta_G_hartree,
        )
    else:
        raise ValueError(f"Unknown potential_type: {potential_type!r}")

    return grid, V_R, V_P


def fgh_franck_condon_table(
    omega_R_cm: float,
    omega_P_cm: float,
    mass_amu: float,
    delta_0_angstrom: float,
    delta_G_hartree: float = 0.0,
    n_reactant_states: int = 5,
    n_product_states: int = 10,
    potential_type: str = "morse",
    D_e_hartree: float | None = None,
    n_grid: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FGHResult, FGHResult]:
    """Compute full table of FC overlaps using numerical FGH wavefunctions.

    This is the drop-in replacement for the analytic FC overlap calculation.
    Returns the same (n_R, n_P) overlap table but computed from numerical
    wavefunctions on Morse (or harmonic) potentials.

    Args:
        omega_R_cm: Reactant frequency in cm⁻¹.
        omega_P_cm: Product frequency in cm⁻¹.
        mass_amu: Particle mass in amu.
        delta_0_angstrom: Tunneling distance in angstrom.
        delta_G_hartree: Driving force in hartree.
        n_reactant_states: Number of reactant states.
        n_product_states: Number of product states.
        potential_type: "morse" or "harmonic".
        D_e_hartree: Morse dissociation energy (None = auto-estimate).
        n_grid: FGH grid points.

    Returns:
        Tuple of:
            overlaps: |S_μν|² array, shape (n_R, n_P).
            energies_R: Reactant eigenvalues in hartree, shape (n_R,).
            energies_P: Product eigenvalues in hartree, shape (n_P,).
            fgh_R: Full FGH result for reactant.
            fgh_P: Full FGH result for product.
    """
    grid, V_R, V_P = build_proton_potentials(
        omega_R_cm, omega_P_cm, mass_amu, delta_0_angstrom,
        delta_G_hartree, potential_type, D_e_hartree, n_grid,
    )

    mass_au = mass_amu * AMU_TO_AU
    n_max = max(n_reactant_states, n_product_states)

    fgh_R = solve_1d_schrodinger(grid, V_R, mass_au, n_max)
    fgh_P = solve_1d_schrodinger(grid, V_P, mass_au, n_max)

    overlaps = np.zeros((n_reactant_states, n_product_states))
    for mu in range(n_reactant_states):
        for nu in range(n_product_states):
            overlaps[mu, nu] = numerical_fc_overlap(fgh_R, fgh_P, mu, nu)

    return (
        overlaps,
        fgh_R.energies[:n_reactant_states],
        fgh_P.energies[:n_product_states],
        fgh_R,
        fgh_P,
    )


def morse_overlap_at_distance(
    omega_R_cm: float,
    omega_P_cm: float,
    mass_amu: float,
    delta_angstrom: float,
    mu: int,
    nu: int,
    potential_type: str = "morse",
    D_e_hartree: float | None = None,
    n_grid: int = 256,
    grid_padding: float = 2.0,
) -> float:
    """Compute |S_μν|² for Morse wavefunctions at a given tunneling distance.

    This is used internally for computing state-pair-specific attenuation
    parameters α_μν and γ_μν via numerical differentiation.

    Args:
        omega_R_cm: Reactant frequency in cm⁻¹.
        omega_P_cm: Product frequency in cm⁻¹.
        mass_amu: Particle mass in amu.
        delta_angstrom: Tunneling distance in angstrom.
        mu, nu: Vibrational quantum numbers.
        potential_type: "morse" or "harmonic".
        D_e_hartree: Morse dissociation energy.
        n_grid: Grid points.
        grid_padding: Grid padding in angstrom.

    Returns:
        |S_μν|² (dimensionless).
    """
    mass_au = mass_amu * AMU_TO_AU
    omega_R_au = omega_R_cm * CM_TO_HARTREE
    omega_P_au = omega_P_cm * CM_TO_HARTREE
    delta_bohr = delta_angstrom * ANGSTROM_TO_BOHR

    if D_e_hartree is None:
        D_e_hartree = 20.0 * omega_R_au

    x_min = -grid_padding * ANGSTROM_TO_BOHR
    x_max = delta_bohr + grid_padding * ANGSTROM_TO_BOHR
    grid = np.linspace(x_min, x_max, n_grid)

    n_states = max(mu, nu) + 1

    if potential_type == "morse":
        a_R = omega_R_au * math.sqrt(mass_au / (2.0 * D_e_hartree))
        a_P = omega_P_au * math.sqrt(mass_au / (2.0 * D_e_hartree))
        V_R = D_e_hartree * (1.0 - np.exp(-a_R * grid)) ** 2
        V_P = D_e_hartree * (1.0 - np.exp(-a_P * (grid - delta_bohr))) ** 2
    else:
        k_R = mass_au * omega_R_au**2
        k_P = mass_au * omega_P_au**2
        V_R = 0.5 * k_R * grid**2
        V_P = 0.5 * k_P * (grid - delta_bohr) ** 2

    fgh_R = solve_1d_schrodinger(grid, V_R, mass_au, n_states)
    fgh_P = solve_1d_schrodinger(grid, V_P, mass_au, n_states)

    return numerical_fc_overlap(fgh_R, fgh_P, mu, nu)


def compute_attenuation_params(
    omega_R_cm: float,
    omega_P_cm: float,
    mass_amu: float,
    delta_0_angstrom: float,
    n_reactant_states: int = 3,
    n_product_states: int = 3,
    potential_type: str = "morse",
    D_e_hartree: float | None = None,
    n_grid: int = 256,
    h_angstrom: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute state-pair-specific attenuation parameters from Morse overlaps.

    For each pair (μ,ν), computes:
        α_μν = -d(log S_μν)/dR |_{R₀}     (first derivative)
        γ_μν = -d²(log S_μν)/dR² |_{R₀}   (second derivative)

    These are the Soudackov correction parameters (PMC5217758): the
    analytical gating formula uses state-pair-specific α_μν and γ_μν
    instead of a single α = mω/ℏ for all pairs.

    Args:
        omega_R_cm, omega_P_cm: Frequencies in cm⁻¹.
        mass_amu: Particle mass in amu.
        delta_0_angstrom: Equilibrium tunneling distance in Å.
        n_reactant_states, n_product_states: Number of states.
        potential_type: "morse" or "harmonic".
        D_e_hartree: Morse dissociation energy.
        n_grid: FGH grid points.
        h_angstrom: Finite difference step size in Å.

    Returns:
        Tuple of:
            S_sq_0: |S_μν(R₀)|², shape (n_R, n_P).
            alpha_mn: α_μν in bohr⁻¹, shape (n_R, n_P).
            gamma_mn: γ_μν in bohr⁻², shape (n_R, n_P).
    """
    n_R = n_reactant_states
    n_P = n_product_states

    S_sq_0 = np.zeros((n_R, n_P))
    alpha_mn = np.zeros((n_R, n_P))
    gamma_mn = np.zeros((n_R, n_P))

    # Evaluate overlaps at R₀ - h, R₀, R₀ + h for finite differences
    deltas = [
        delta_0_angstrom - h_angstrom,
        delta_0_angstrom,
        delta_0_angstrom + h_angstrom,
    ]

    h_bohr = h_angstrom * ANGSTROM_TO_BOHR

    for mu in range(n_R):
        for nu in range(n_P):
            S_vals = []
            for d in deltas:
                if d < 0.02:
                    d = 0.02
                S_sq = morse_overlap_at_distance(
                    omega_R_cm, omega_P_cm, mass_amu, d,
                    mu, nu, potential_type, D_e_hartree, n_grid,
                )
                S_vals.append(S_sq)

            S_m, S_0, S_p = S_vals
            S_sq_0[mu, nu] = S_0

            if S_0 < 1e-30:
                continue

            # log |S_μν| at three points
            # morse_overlap_at_distance returns |S|², so use
            # 0.5*log(|S|²) = log(|S|) to match Soudackov's convention
            # where α = -d(log S)/dR, NOT -d(log S²)/dR.
            log_m = 0.5 * math.log(S_m) if S_m > 1e-300 else -350.0
            log_0 = 0.5 * math.log(S_0) if S_0 > 1e-300 else -350.0
            log_p = 0.5 * math.log(S_p) if S_p > 1e-300 else -350.0

            # α_μν = -d(log S)/dR ≈ -(log_p - log_m) / (2h)
            # Note: S decreases with R, so -dlog/dR is positive
            alpha_mn[mu, nu] = -(log_p - log_m) / (2.0 * h_bohr)

            # γ_μν = -d²(log S)/dR² ≈ -(log_p - 2*log_0 + log_m) / h²
            gamma_mn[mu, nu] = -(log_p - 2.0 * log_0 + log_m) / h_bohr**2

    return S_sq_0, alpha_mn, gamma_mn


def analytical_gating_rate_table(
    S_sq_0: np.ndarray,
    alpha_mn: np.ndarray,
    gamma_mn: np.ndarray,
    M_DA_amu: float,
    Omega_gating_cm: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the SHS analytical gating formula with state-pair-specific params.

    From Soudackov & Hammes-Schiffer (PMC5217758), the R-averaged rate is:

        k_μν ∝ |S_μν(R₀)|² × √(MΩ²/(MΩ² + γ_μν kBT))
                             × exp(α_μν² kBT / (2(MΩ² + γ_μν kBT)))

    where α_μν = -d(log |S_μν|)/dR and γ_μν = -d²(log |S_μν|)/dR²
    are evaluated at R₀ (Soudackov convention: log of overlap, not log of
    squared overlap).

    The key insight: γ_μν > 0 (overlap is concave on log scale), so the
    denominator is MΩ² + γkBT (NOT minus). The gating mode's effective
    spring constant is stiffened by the curvature of the overlap.

    The Marcus barrier uses the ORIGINAL λ (no λ_α correction), but
    the overlap gets an exponential boost from sampling shorter R values.

    Args:
        S_sq_0: |S_μν(R₀)|², shape (n_R, n_P).
        alpha_mn: α_μν = -d(log S)/dR in bohr⁻¹, shape (n_R, n_P).
        gamma_mn: γ_μν = -d²(log S)/dR² in bohr⁻², shape (n_R, n_P).
        M_DA_amu: D-A gating mass in amu.
        Omega_gating_cm: Gating frequency in cm⁻¹.
        temperature: Temperature in K.

    Returns:
        Tuple of:
            S_sq_eff: Effective (R-averaged) overlaps, shape (n_R, n_P).
            prefactor: √(MΩ²/(MΩ² + γkBT)) normalization, shape (n_R, n_P).
    """
    kBT = KB_HARTREE * temperature
    M_au = M_DA_amu * AMU_TO_AU
    Omega_au = Omega_gating_cm * CM_TO_HARTREE

    MO2 = M_au * Omega_au**2  # MΩ² in hartree/bohr²

    n_R, n_P = S_sq_0.shape
    S_sq_eff = np.zeros((n_R, n_P))
    gating_prefactor = np.zeros((n_R, n_P))

    for mu in range(n_R):
        for nu in range(n_P):
            a = alpha_mn[mu, nu]
            g = gamma_mn[mu, nu]

            # The gating integral averages S²(R) over the thermal distribution
            # of R. With α = -d(log S)/dR and γ = -d²(log S)/dR² (Soudackov
            # convention), the expansion is:
            #   S²(R) = S²(R₀) exp(-2α δR - γ δR²)
            # Completing the square with P(R) ∝ exp(-MΩ²δR²/(2kBT)):
            #   denom = MΩ² + 2γkBT
            #   zeta = 2α²kBT / denom
            denom = MO2 + 2.0 * g * kBT
            if denom <= 0:
                S_sq_eff[mu, nu] = S_sq_0[mu, nu]
                gating_prefactor[mu, nu] = 1.0
                continue

            # Gating boost: exp(2α²kBT / (MΩ² + 2γkBT))
            zeta = 2.0 * a**2 * kBT / denom

            # Normalization prefactor from Gaussian width change
            pf = math.sqrt(MO2 / denom)

            S_sq_eff[mu, nu] = S_sq_0[mu, nu] * pf * math.exp(zeta)
            gating_prefactor[mu, nu] = pf

    return S_sq_eff, gating_prefactor


def r_averaged_fgh_fc_table(
    omega_R_cm: float,
    omega_P_cm: float,
    mass_amu: float,
    delta_0_angstrom: float,
    M_DA_amu: float,
    Omega_gating_cm: float,
    temperature: float,
    delta_G_hartree: float = 0.0,
    n_reactant_states: int = 5,
    n_product_states: int = 10,
    potential_type: str = "morse",
    D_e_hartree: float | None = None,
    n_grid: int = 256,
    n_quad: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute R-averaged FC overlaps using FGH Morse wavefunctions.

    This is the key function that addresses the Soudackov correction:
    instead of using a single set of harmonic α parameters for all state
    pairs, we compute the actual Morse wavefunction overlaps at each
    D-A distance R sampled by Gauss-Hermite quadrature.

    For each quadrature point R_i:
      1. Compute δ(R_i) = δ₀ + (R_i - R_eq)
      2. Build Morse potentials with this tunneling distance
      3. Solve FGH for reactant and product wavefunctions
      4. Compute S_μν(R_i) numerically

    Then average: <|S_μν|²>_R = Σ_i w_i |S_μν(R_i)|²

    This captures the state-pair-specific distance dependence that the
    harmonic approximation misses. For excited Morse states, the overlap
    falls off more slowly with R than the harmonic formula predicts,
    because Morse wavefunctions have longer tails.

    Args:
        omega_R_cm: Reactant proton frequency in cm⁻¹.
        omega_P_cm: Product proton frequency in cm⁻¹.
        mass_amu: Transferring particle mass in amu.
        delta_0_angstrom: Equilibrium tunneling distance in angstrom.
        M_DA_amu: D-A gating reduced mass in amu.
        Omega_gating_cm: D-A gating frequency in cm⁻¹.
        temperature: Temperature in Kelvin.
        delta_G_hartree: Driving force in hartree.
        n_reactant_states: Number of reactant vibronic states.
        n_product_states: Number of product vibronic states.
        potential_type: "morse" or "harmonic".
        D_e_hartree: Morse dissociation energy (None = auto-estimate).
        n_grid: FGH grid points.
        n_quad: Number of Gauss-Hermite quadrature points.

    Returns:
        Tuple of:
            overlaps: R-averaged |S_μν|² array, shape (n_R, n_P).
            energies_R: Reactant eigenvalues in hartree (at equilibrium δ₀).
            energies_P: Product eigenvalues in hartree (at equilibrium δ₀).
    """
    kBT = KB_HARTREE * temperature
    M_DA_au = M_DA_amu * AMU_TO_AU
    Omega_au = Omega_gating_cm * CM_TO_HARTREE
    mass_au = mass_amu * AMU_TO_AU

    # Thermal width of gating coordinate (quantum harmonic oscillator):
    # σ² = (ℏ/2MΩ) coth(ℏΩ/2kBT)
    x = Omega_au / (2.0 * kBT)
    if x < 0.01:
        sigma_sq = kBT / (M_DA_au * Omega_au**2)
    else:
        coth_x = 1.0 / math.tanh(x) if x < 50 else 1.0
        sigma_sq = (HBAR_AU / (2.0 * M_DA_au * Omega_au)) * coth_x
    sigma_bohr = math.sqrt(sigma_sq)  # in bohr

    # Gauss-Hermite quadrature points and weights
    points, weights = np.polynomial.hermite.hermgauss(n_quad)

    n_R = n_reactant_states
    n_P = n_product_states
    n_max = max(n_R, n_P)

    overlaps_avg = np.zeros((n_R, n_P))

    # Auto-estimate D_e once (same for all R)
    if D_e_hartree is None:
        omega_R_au = omega_R_cm * CM_TO_HARTREE
        D_e_hartree = 20.0 * omega_R_au

    # Precompute grid bounds (wide enough for all R displacements)
    delta_0_bohr = delta_0_angstrom * ANGSTROM_TO_BOHR
    max_displacement = sigma_bohr * math.sqrt(2.0) * abs(points[-1])
    grid_padding_bohr = 2.0 * ANGSTROM_TO_BOHR  # 2 Å padding

    x_min = -grid_padding_bohr
    x_max = (delta_0_bohr + max_displacement) + grid_padding_bohr
    grid = np.linspace(x_min, x_max, n_grid)

    # Reactant potential doesn't change with R (centered at x=0 always)
    omega_R_au = omega_R_cm * CM_TO_HARTREE
    omega_P_au = omega_P_cm * CM_TO_HARTREE
    a_R = omega_R_au * math.sqrt(mass_au / (2.0 * D_e_hartree))

    if potential_type == "morse":
        V_R = D_e_hartree * (1.0 - np.exp(-a_R * grid)) ** 2
    else:
        k_R = mass_au * omega_R_au**2
        V_R = 0.5 * k_R * grid**2

    fgh_R = solve_1d_schrodinger(grid, V_R, mass_au, n_max)

    # Also solve at equilibrium for energy levels (returned to caller)
    if potential_type == "morse":
        a_P = omega_P_au * math.sqrt(mass_au / (2.0 * D_e_hartree))
        V_P_eq = D_e_hartree * (1.0 - np.exp(-a_P * (grid - delta_0_bohr))) ** 2 + delta_G_hartree
    else:
        k_P = mass_au * omega_P_au**2
        V_P_eq = 0.5 * k_P * (grid - delta_0_bohr) ** 2 + delta_G_hartree

    fgh_P_eq = solve_1d_schrodinger(grid, V_P_eq, mass_au, n_max)

    for i in range(n_quad):
        # Displacement from equilibrium in bohr
        u = sigma_bohr * math.sqrt(2.0) * points[i]
        delta_R_bohr = delta_0_bohr + u

        if delta_R_bohr < 0.02 * ANGSTROM_TO_BOHR:
            delta_R_bohr = 0.02 * ANGSTROM_TO_BOHR

        # Build product potential at this R
        if potential_type == "morse":
            V_P = D_e_hartree * (1.0 - np.exp(-a_P * (grid - delta_R_bohr))) ** 2 + delta_G_hartree
        else:
            V_P = 0.5 * k_P * (grid - delta_R_bohr) ** 2 + delta_G_hartree

        fgh_P = solve_1d_schrodinger(grid, V_P, mass_au, n_max)

        # Compute overlaps at this R
        for mu in range(n_R):
            for nu in range(n_P):
                S_sq = numerical_fc_overlap(fgh_R, fgh_P, mu, nu)
                overlaps_avg[mu, nu] += weights[i] * S_sq

    # Normalize: Gauss-Hermite weights sum to √π
    overlaps_avg /= math.sqrt(math.pi)

    return (
        overlaps_avg,
        fgh_R.energies[:n_R],
        fgh_P_eq.energies[:n_P],
    )
