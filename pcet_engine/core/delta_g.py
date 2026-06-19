"""
Driving force (ΔG) extraction from quantum chemistry output files.

Computes the reaction energy difference from reactant and product
total energies, with optional zero-point energy (ZPE) correction.

IMPORTANT — naming and scope:
    This module computes ΔE_elec + ΔZPE, which approximates ΔH at 0 K.
    A true Gibbs free energy (ΔG at finite T) additionally requires
    thermal corrections and entropy from partition functions.  For most
    PCET applications the 0 K approximation is acceptable; for precise
    thermochemistry, use Gaussian's thermochemistry output directly.

IMPORTANT — PCET vibronic rate theory:
    In the Hammes-Schiffer vibronic formalism, the proton coordinate is
    treated quantum mechanically.  The driving force fed into the Marcus
    exponential must be for the HEAVY-ATOM subsystem only.  If you include
    ZPE, you MUST exclude the proton transfer mode to avoid double-counting
    (the vibronic channel sum already accounts for proton vibrational
    levels).  Use ``exclude_modes`` or ``delta_g_for_pcet`` to handle this.

References:
    Hammes-Schiffer & Stuchebrukhov, Chem. Rev. 110, 6939 (2010).
"""

import numpy as np
from dataclasses import dataclass

from pcet_engine.core.constants import HARTREE_TO_KCALMOL, HARTREE_TO_CM


@dataclass
class DrivingForceResult:
    """Result of driving force extraction.

    Attributes:
        delta_E_zpe_kcal: ΔE + ΔZPE in kcal/mol (0 K enthalpy approximation).
        delta_E_zpe_hartree: Same in hartree.
        delta_E_electronic: Pure electronic energy difference in kcal/mol.
        zpe_reactant: Reactant ZPE in kcal/mol (0 if not computed).
        zpe_product: Product ZPE in kcal/mol (0 if not computed).
        delta_zpe: ZPE correction (ZPE_product - ZPE_reactant) in kcal/mol.
        n_freqs_reactant: Number of real frequencies in reactant.
        n_freqs_product: Number of real frequencies in product.
        proton_zpe_reactant: ZPE of excluded proton mode(s) in reactant (kcal/mol).
        proton_zpe_product: ZPE of excluded proton mode(s) in product (kcal/mol).
    """

    delta_E_zpe_kcal: float
    delta_E_zpe_hartree: float
    delta_E_electronic: float
    zpe_reactant: float
    zpe_product: float
    delta_zpe: float
    n_freqs_reactant: int
    n_freqs_product: int
    proton_zpe_reactant: float = 0.0
    proton_zpe_product: float = 0.0


# Keep backward-compatible alias
DeltaGResult = DrivingForceResult


def compute_zpe(
    frequencies_cm: np.ndarray,
    exclude_indices: list[int] | None = None,
    min_freq_cm: float = 50.0,
) -> float:
    """Compute zero-point energy from vibrational frequencies.

    ZPE = (1/2) Σ_i ħω_i  for all real frequencies above the cutoff.

    Note: Unscaled harmonic frequencies are used.  For higher accuracy,
    apply a method-specific ZPE scaling factor (e.g., 0.9804 for
    B3LYP/6-31G*) before calling this function.

    Args:
        frequencies_cm: Vibrational frequencies in cm⁻¹. Imaginary
            frequencies (negative values) are excluded automatically.
        exclude_indices: Mode indices to exclude (e.g., proton transfer
            mode for PCET).  These modes are skipped in the ZPE sum.
        min_freq_cm: Minimum frequency to include (cm⁻¹).  Modes below
            this are treated as translations/rotations and excluded.
            Default 50 cm⁻¹ compensates for incomplete rotation projection
            in our NMA code.

    Returns:
        ZPE in hartree.
    """
    mask = frequencies_cm > min_freq_cm
    if exclude_indices is not None:
        for idx in exclude_indices:
            if 0 <= idx < len(mask):
                mask[idx] = False
    real_freqs = frequencies_cm[mask]
    zpe_hartree = 0.5 * np.sum(real_freqs) / HARTREE_TO_CM
    return zpe_hartree


def delta_g_from_energies(
    energy_reactant: float,
    energy_product: float,
    frequencies_reactant: np.ndarray | None = None,
    frequencies_product: np.ndarray | None = None,
    exclude_modes_reactant: list[int] | None = None,
    exclude_modes_product: list[int] | None = None,
) -> DrivingForceResult:
    """Compute driving force from total energies and (optionally) vibrational frequencies.

    Returns ΔE_electronic + ΔZPE, which approximates ΔH at 0 K.
    This is NOT the full Gibbs free energy at finite temperature.

    For PCET vibronic calculations, pass ``exclude_modes_reactant`` and
    ``exclude_modes_product`` to exclude the proton transfer stretch from
    the ZPE.  This avoids double-counting with the vibronic channel sum.

    Args:
        energy_reactant: Total electronic energy of reactant in hartree.
        energy_product: Total electronic energy of product in hartree.
        frequencies_reactant: Reactant vibrational frequencies in cm⁻¹.
            If provided, ZPE correction is included.
        frequencies_product: Product vibrational frequencies in cm⁻¹.
            Required if frequencies_reactant is provided.
        exclude_modes_reactant: Mode indices to exclude from reactant ZPE.
        exclude_modes_product: Mode indices to exclude from product ZPE.

    Returns:
        DrivingForceResult with energy decomposition.
    """
    delta_E = energy_product - energy_reactant
    delta_E_kcal = delta_E * HARTREE_TO_KCALMOL

    zpe_R = 0.0
    zpe_P = 0.0
    n_freqs_R = 0
    n_freqs_P = 0
    proton_zpe_R = 0.0
    proton_zpe_P = 0.0

    if frequencies_reactant is not None:
        if frequencies_product is None:
            raise ValueError(
                "If frequencies_reactant is provided, frequencies_product must also be provided."
            )
        zpe_R = compute_zpe(frequencies_reactant, exclude_indices=exclude_modes_reactant)
        zpe_P = compute_zpe(frequencies_product, exclude_indices=exclude_modes_product)
        n_freqs_R = int(np.sum(frequencies_reactant > 50.0))
        n_freqs_P = int(np.sum(frequencies_product > 50.0))

        # Track excluded proton ZPE for diagnostics
        if exclude_modes_reactant:
            zpe_R_full = compute_zpe(frequencies_reactant)
            proton_zpe_R = (zpe_R_full - zpe_R) * HARTREE_TO_KCALMOL
        if exclude_modes_product:
            zpe_P_full = compute_zpe(frequencies_product)
            proton_zpe_P = (zpe_P_full - zpe_P) * HARTREE_TO_KCALMOL

    delta_zpe = zpe_P - zpe_R
    delta_E_zpe_hartree = delta_E + delta_zpe
    delta_E_zpe_kcal = delta_E_zpe_hartree * HARTREE_TO_KCALMOL

    return DrivingForceResult(
        delta_E_zpe_kcal=delta_E_zpe_kcal,
        delta_E_zpe_hartree=delta_E_zpe_hartree,
        delta_E_electronic=delta_E_kcal,
        zpe_reactant=zpe_R * HARTREE_TO_KCALMOL,
        zpe_product=zpe_P * HARTREE_TO_KCALMOL,
        delta_zpe=delta_zpe * HARTREE_TO_KCALMOL,
        n_freqs_reactant=n_freqs_R,
        n_freqs_product=n_freqs_P,
        proton_zpe_reactant=proton_zpe_R,
        proton_zpe_product=proton_zpe_P,
    )


def delta_g_from_fchk(
    fchk_reactant: str,
    fchk_product: str,
    include_zpe: bool = True,
    exclude_proton: bool = False,
    proton_idx: int | None = None,
) -> DrivingForceResult:
    """Compute driving force from two Gaussian .fchk files.

    Parses total energies from both files. If include_zpe=True, also
    performs normal mode analysis to compute ZPE corrections.

    Args:
        fchk_reactant: Path to reactant .fchk file.
        fchk_product: Path to product .fchk file.
        include_zpe: Whether to include ZPE correction (requires Hessian
            data in both .fchk files).
        exclude_proton: If True, exclude the proton transfer mode from
            ZPE to avoid double-counting in vibronic rate calculations.
            Requires proton_idx.
        proton_idx: 0-based atom index of the transferring proton.
            Required if exclude_proton=True.

    Returns:
        DrivingForceResult with driving force in kcal/mol and hartree.
    """
    from pcet_engine.parsers import parse_gaussian_fchk
    from pcet_engine.core.normal_modes import normal_mode_analysis, identify_proton_mode

    if exclude_proton and proton_idx is None:
        raise ValueError("proton_idx is required when exclude_proton=True")

    qc_R = parse_gaussian_fchk(fchk_reactant)
    qc_P = parse_gaussian_fchk(fchk_product)

    freqs_R = None
    freqs_P = None
    exclude_R = None
    exclude_P = None

    if include_zpe:
        nma_R = normal_mode_analysis(qc_R.hessian, qc_R.masses)
        nma_P = normal_mode_analysis(qc_P.hessian, qc_P.masses)

        if exclude_proton and proton_idx is not None:
            nma_R = identify_proton_mode(nma_R, [proton_idx], qc_R.masses)
            nma_P = identify_proton_mode(nma_P, [proton_idx], qc_P.masses)
            if nma_R.proton_mode_idx is not None:
                exclude_R = [nma_R.proton_mode_idx]
            if nma_P.proton_mode_idx is not None:
                exclude_P = [nma_P.proton_mode_idx]

        freqs_R = nma_R.frequencies_cm
        freqs_P = nma_P.frequencies_cm

    return delta_g_from_energies(
        energy_reactant=qc_R.energy,
        energy_product=qc_P.energy,
        frequencies_reactant=freqs_R,
        frequencies_product=freqs_P,
        exclude_modes_reactant=exclude_R,
        exclude_modes_product=exclude_P,
    )


def delta_g_for_pcet(
    fchk_reactant: str,
    fchk_product: str,
    proton_idx: int,
) -> DrivingForceResult:
    """Convenience wrapper: driving force for PCET vibronic rate calculations.

    Computes ΔE + ΔZPE with the proton transfer mode automatically
    excluded from the ZPE sum.  This is the correct quantity to feed
    into ``PCETRateEngine.compute_rate()`` or ``compute_rate_from_hessian()``,
    which handle proton vibrational levels via the vibronic channel sum.

    Args:
        fchk_reactant: Path to reactant .fchk file.
        fchk_product: Path to product .fchk file.
        proton_idx: 0-based atom index of the transferring proton.

    Returns:
        DrivingForceResult with proton-mode-excluded ZPE correction.
    """
    return delta_g_from_fchk(
        fchk_reactant,
        fchk_product,
        include_zpe=True,
        exclude_proton=True,
        proton_idx=proton_idx,
    )
