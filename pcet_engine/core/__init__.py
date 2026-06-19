"""Core rate theory engine: Marcus theory, vibronic coupling, tunneling."""

from pcet_engine.core.constants import *  # noqa: F401,F403
from pcet_engine.core.marcus import marcus_rate, marcus_activation_energy
from pcet_engine.core.vibronic import (
    vibronic_rate,
    multi_channel_rate,
    tunneling_distance,
)
from pcet_engine.core.normal_modes import (
    normal_mode_analysis,
    identify_da_stretching_mode,
    NormalModeResult,
    GatingResult,
)
from pcet_engine.core.rate_engine import PCETRateEngine, PCETResult
from pcet_engine.core.fgh_solver import fgh_1d, compute_fc_overlaps
from pcet_engine.core.proton_potential import (
    harmonic_potential,
    morse_potential,
    double_well_potential,
    fit_potential_from_scan,
    potential_from_hessian,
)
from pcet_engine.core.electrochemistry import (
    fermi_dirac,
    edl_model,
    electrochemical_rate,
    ElectrochemicalResult,
)
from pcet_engine.core.nonadiabaticity import (
    analyze_nonadiabaticity,
    NonadiabaticityResult,
)
from pcet_engine.core.uncertainty import (
    propagate_uncertainty,
    UncertaintyResult,
)
from pcet_engine.core.participation import (
    participation_ratio,
    mode_participation,
    proton_participation,
    effective_tunneling_dimension,
    geometric_tunneling_prefactor,
    tunneling_correction_report,
    ParticipationResult,
)
from pcet_engine.core.delta_g import (
    delta_g_from_energies,
    delta_g_from_fchk,
    delta_g_for_pcet,
    compute_zpe,
    DrivingForceResult,
    DeltaGResult,
)
from pcet_engine.core.coupling import (
    empirical_coupling,
    gmh_coupling,
    gmh_coupling_from_tddft,
    gmh_coupling_multistate,
    EmpiricalCouplingResult,
    GMHResult,
)

__all__ = [
    "marcus_rate",
    "marcus_activation_energy",
    "vibronic_rate",
    "multi_channel_rate",
    "normal_mode_analysis",
    "NormalModeResult",
    "identify_da_stretching_mode",
    "GatingResult",
    "PCETRateEngine",
    "PCETResult",
    "fgh_1d",
    "compute_fc_overlaps",
    "harmonic_potential",
    "morse_potential",
    "double_well_potential",
    "fit_potential_from_scan",
    "potential_from_hessian",
    "fermi_dirac",
    "edl_model",
    "electrochemical_rate",
    "ElectrochemicalResult",
    "analyze_nonadiabaticity",
    "NonadiabaticityResult",
    "propagate_uncertainty",
    "UncertaintyResult",
    "participation_ratio",
    "mode_participation",
    "proton_participation",
    "effective_tunneling_dimension",
    "geometric_tunneling_prefactor",
    "tunneling_correction_report",
    "ParticipationResult",
    # ΔG extraction
    "delta_g_from_energies",
    "delta_g_from_fchk",
    "compute_zpe",
    "DrivingForceResult",
    "DeltaGResult",
    "delta_g_for_pcet",
    # Electronic coupling
    "empirical_coupling",
    "gmh_coupling",
    "gmh_coupling_from_tddft",
    "gmh_coupling_multistate",
    "EmpiricalCouplingResult",
    "GMHResult",
]
