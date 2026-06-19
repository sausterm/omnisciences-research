"""
Extract PCET parameters from protein structure (PDB coordinates).

Bridges the gap between protein structures (PDB/AlphaFold) and the
5 molecular parameters the PCET engine needs. Uses empirical correlations
from the biophysics literature — not QM calculations.

This is the #1 product unlock: it transforms the PCET engine from a tool
for 500 QM specialists into a product for 20,000+ enzyme engineers.

Correlations used:
    V_el:       Moser-Dutton ruler (exponential distance decay)
    lambda_out: Marcus continuum model (protein dielectric)
    omega_H:    Bond-type lookup table (IR spectroscopy data)
    d_DA:       Direct from atomic coordinates
    delta_G:    Redox potential lookup (tabulated cofactors)

Accuracy: These empirical estimates give ~2-5x error in absolute rates
(because V_el enters quadratically). But for RANKING variants — which is
what enzyme engineers want — the errors largely cancel because you're
comparing Δrate between mutations, not predicting absolute rates.

References
----------
- Moser, Keske, Warncke, Farid, Dutton (1992). Nature 355, 796-802.
- Page, Moser, Chen, Dutton (1999). Nature 402, 47-52.
- Marcus & Sutin (1985). BBA 811, 265-322.
- Hay, Scrutton (2012). Nature Chemistry 4, 161-168.

Usage::

    from pcet_engine.core.structure_to_params import StructureToParams

    s2p = StructureToParams()
    params = s2p.from_coordinates(
        donor_xyz=(10.0, 20.0, 30.0),
        acceptor_xyz=(12.5, 20.0, 30.0),
        donor_element="Fe",
        acceptor_element="C",
        bond_type="C-H",
    )
    print(f"V_el = {params.V_el:.3f} kcal/mol")
    print(f"d_DA = {params.d_DA:.2f} Å")
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# =====================================================================
# Moser-Dutton ruler: V_el from distance
# =====================================================================

# Standard Moser-Dutton parameters (Page et al. 1999 Nature)
# For pure electron transfer: V_el = V0 * exp(-beta * (R_edge - R0))
# β = 1.1 Å⁻¹ (consensus, Moser & Dutton 1992)
MOSER_DUTTON_BETA = 1.1  # Å⁻¹ (coupling decay constant)
MOSER_DUTTON_V0 = 73.0   # kcal/mol (coupling at van der Waals contact)
MOSER_DUTTON_R0 = 3.6     # Å (van der Waals contact distance)

# For PCET: the proton tunneling distance matters more than ET distance.
# The effective coupling for vibronic PCET depends on d_DA (proton
# donor-to-acceptor distance), not the electron donor-to-acceptor edge
# distance. SHS parameterize V_el directly (typically 0.5-2 kcal/mol
# for PCET). When estimating from structure, we use d_DA with a
# calibrated prefactor.
#
# Calibration: SLO-1 WT has d_DA = 2.77 Å and V_el ≈ 0.6 kcal/mol
# (from fitting to experimental rate). This gives:
# V_el_PCET = V0_PCET * exp(-beta_PCET * (d_DA - d0_PCET))
# Calibration from SLO-1 series:
# WT (d=2.77, k=297), L546A (d=2.88, k=8.2): ratio = 0.028
# k ∝ V_el², V_el ∝ exp(-β*Δd) → exp(-2β*0.11) = 0.028 → β ≈ 16.3
# Cross-check: L754A (d=2.95): predicted ratio = exp(-2*16.3*0.18) = 0.003
#              experimental ratio = 3.0/297 = 0.010. Reasonable (factor of 3).
PCET_BETA = 16.3      # Å⁻¹ (calibrated to SLO-1 WT→L546A rate ratio)
PCET_V0 = 0.6         # kcal/mol (calibrated to SLO-1 WT at d_DA = 2.77)
PCET_D0 = 2.77        # Å (reference distance = SLO-1 WT)

# Alternative: Hay & Scrutton (2012) for PCET specifically
HAY_SCRUTTON_BETA = 1.0   # Å⁻¹ (for ET component only)


def moser_dutton_coupling(
    R_edge: float,
    beta: float = MOSER_DUTTON_BETA,
    V0: float = MOSER_DUTTON_V0,
    R0: float = MOSER_DUTTON_R0,
) -> float:
    """Electronic coupling from Moser-Dutton ruler.

    Parameters
    ----------
    R_edge : float
        Edge-to-edge distance between donor and acceptor in Å.
        This is d_DA minus the van der Waals radii of the terminal atoms.
    beta : float
        Distance decay constant in Å⁻¹.
    V0 : float
        Coupling at contact distance in kcal/mol.
    R0 : float
        Contact distance in Å.

    Returns
    -------
    float
        Electronic coupling V_el in kcal/mol.
    """
    return V0 * math.exp(-beta * (R_edge - R0))


def pcet_coupling(
    d_DA: float,
    beta: float = PCET_BETA,
    V0: float = PCET_V0,
    d0: float = PCET_D0,
) -> float:
    """Electronic coupling for PCET from donor-acceptor distance.

    Uses a steeper distance dependence than pure ET because proton
    tunneling coupling decays much faster with distance.

    Calibrated to SLO-1 WT: V_el = 0.6 kcal/mol at d_DA = 2.77 Å.

    Parameters
    ----------
    d_DA : float
        Proton donor-to-acceptor distance in Å.
    beta : float
        Distance decay constant in Å⁻¹. Default 25.0 (proton tunneling).
    V0 : float
        Coupling at reference distance in kcal/mol.
    d0 : float
        Reference distance in Å.

    Returns
    -------
    float
        Electronic coupling V_el in kcal/mol.
    """
    return V0 * math.exp(-beta * (d_DA - d0))


# =====================================================================
# Marcus continuum: outer-sphere reorganization energy
# =====================================================================

# Protein dielectric constants
EPSILON_OPT_PROTEIN = 2.0    # optical (electronic) dielectric
EPSILON_S_PROTEIN = 4.0      # static dielectric (protein interior)
EPSILON_S_WATER = 78.4       # static dielectric (water/surface)

# Common cofactor radii (Å) — effective spherical radii for Marcus model
COFACTOR_RADII = {
    "Fe": 1.5,      # iron center (heme, non-heme)
    "Cu": 1.4,      # copper center
    "Mn": 1.5,      # manganese center
    "flavin": 3.5,  # isoalloxazine ring
    "NAD": 3.5,     # nicotinamide ring
    "quinone": 3.0, # benzoquinone ring
    "Tyr": 2.5,     # tyrosyl radical
    "Trp": 2.8,     # tryptophan radical
    "Cys": 1.8,     # cysteine sulfur
    "default": 2.0, # fallback
}

# Conversion: e²/(4πε₀) in kcal·mol⁻¹·Å
E2_4PI_EPS0 = 332.06  # kcal·mol⁻¹·Å (Coulomb constant in chem units)


def marcus_reorganization_outer(
    R_DA: float,
    r_D: float = 2.0,
    r_A: float = 2.0,
    epsilon_opt: float = EPSILON_OPT_PROTEIN,
    epsilon_s: float = EPSILON_S_PROTEIN,
    delta_q: float = 1.0,
) -> float:
    """Outer-sphere reorganization energy from Marcus continuum model.

    Parameters
    ----------
    R_DA : float
        Donor-acceptor center-to-center distance in Å.
    r_D, r_A : float
        Effective radii of donor and acceptor in Å.
    epsilon_opt : float
        Optical (high-frequency) dielectric constant.
    epsilon_s : float
        Static dielectric constant.
    delta_q : float
        Charge transferred (elementary charges).

    Returns
    -------
    float
        Outer-sphere reorganization energy in kcal/mol.
    """
    if R_DA <= r_D + r_A:
        # Donor-acceptor overlap — use minimum distance
        R_DA = r_D + r_A + 0.1

    pekar = (1.0 / epsilon_opt) - (1.0 / epsilon_s)
    geometry = (1.0 / (2.0 * r_D)) + (1.0 / (2.0 * r_A)) - (1.0 / R_DA)

    return E2_4PI_EPS0 * delta_q**2 * geometry * pekar


# Inner-sphere (vibrational) reorganization — typical ranges
LAMBDA_INNER_TYPICAL = {
    "small_molecule": 10.0,    # kcal/mol — small reorganization
    "protein": 5.0,            # kcal/mol — protein environment constrains
    "heme": 3.0,               # kcal/mol — rigid heme environment
    "non_heme_iron": 8.0,      # kcal/mol
    "copper": 6.0,             # kcal/mol
    "flavin": 15.0,            # kcal/mol — flexible
    "default": 7.0,            # kcal/mol
}


# =====================================================================
# Proton frequency from bond type
# =====================================================================

# IR spectroscopy data for X-H stretching frequencies (cm⁻¹)
BOND_FREQUENCIES = {
    "C-H": {"mean": 2950.0, "range": (2850, 3100), "note": "sp3 C-H"},
    "C-H_sp2": {"mean": 3050.0, "range": (3000, 3100), "note": "sp2 C-H"},
    "C-H_sp": {"mean": 3300.0, "range": (3200, 3400), "note": "sp C-H"},
    "O-H": {"mean": 3400.0, "range": (3200, 3600), "note": "alcohol/phenol O-H"},
    "O-H_acid": {"mean": 2900.0, "range": (2500, 3300), "note": "carboxylic acid O-H"},
    "N-H": {"mean": 3350.0, "range": (3200, 3500), "note": "amine N-H"},
    "N-H_amide": {"mean": 3300.0, "range": (3100, 3500), "note": "amide N-H"},
    "S-H": {"mean": 2550.0, "range": (2500, 2600), "note": "thiol S-H"},
}


def proton_frequency(bond_type: str) -> float:
    """Proton vibrational frequency from bond type.

    Parameters
    ----------
    bond_type : str
        Bond type string, e.g. "C-H", "O-H", "N-H", "S-H".

    Returns
    -------
    float
        Proton frequency in cm⁻¹.
    """
    if bond_type in BOND_FREQUENCIES:
        return BOND_FREQUENCIES[bond_type]["mean"]

    # Try partial match
    for key, data in BOND_FREQUENCIES.items():
        if bond_type.upper().startswith(key.split("_")[0].upper()):
            return data["mean"]

    raise ValueError(
        f"Unknown bond type '{bond_type}'. "
        f"Known types: {list(BOND_FREQUENCIES.keys())}"
    )


# =====================================================================
# Driving force from redox potentials
# =====================================================================

# Standard reduction potentials (V vs SHE) at pH 7
# E°' values for common biological redox couples
REDOX_POTENTIALS = {
    # Metal centers
    "Fe3+/Fe2+_heme": 0.27,           # cytochrome c
    "Fe3+/Fe2+_non_heme": -0.05,      # typical non-heme iron
    "Fe3+/Fe2+_rubredoxin": -0.06,
    "Fe4+=O/Fe3+-OH": 1.0,            # compound I → compound II (P450)
    "Fe3+-OH/Fe3+-OH2": 0.9,          # SLO-1 type
    "Cu2+/Cu+_blue": 0.30,            # blue copper (plastocyanin)
    "Cu2+/Cu+_type2": 0.26,           # type 2 copper
    "Mn4+/Mn3+": 0.80,               # manganese catalase

    # Organic cofactors
    "NAD+/NADH": -0.32,
    "NADP+/NADPH": -0.32,
    "FAD/FADH2": -0.22,              # free flavin
    "FMN/FMNH2": -0.22,
    "ubiquinone/ubiquinol": 0.045,
    "plastoquinone/plastoquinol": 0.08,

    # Amino acid radicals
    "Tyr-O•/Tyr-OH": 0.93,           # tyrosyl radical
    "Trp•+/Trp": 1.01,               # tryptophan radical
    "Cys-S•/Cys-SH": 0.92,           # cysteinyl radical
    "Gly•/Gly-H": 0.58,              # glycyl radical

    # Substrate bonds
    "C-H_linoleic": 0.76,            # SLO-1 substrate (bisallylic C-H)
    "C-H_alkyl": 0.50,               # generic alkyl C-H
    "C-H_benzylic": 0.73,            # benzylic C-H
}


def driving_force(
    E_acceptor: float,
    E_donor: float,
    n_electrons: int = 1,
) -> float:
    """Driving force ΔG from reduction potentials.

    ΔG = -n * F * (E_acceptor - E_donor) in kcal/mol

    Parameters
    ----------
    E_acceptor : float
        Reduction potential of the acceptor (V vs SHE).
    E_donor : float
        Reduction potential of the donor (V vs SHE).
    n_electrons : int
        Number of electrons transferred.

    Returns
    -------
    float
        Driving force in kcal/mol (negative = exothermic).
    """
    FARADAY_KCAL = 23.061  # kcal/(mol·V)
    return -n_electrons * FARADAY_KCAL * (E_acceptor - E_donor)


def lookup_redox_potential(name: str) -> float:
    """Look up a standard reduction potential by name.

    Parameters
    ----------
    name : str
        Redox couple name, e.g. "Fe3+/Fe2+_heme" or "NAD+/NADH".

    Returns
    -------
    float
        Standard reduction potential in V vs SHE at pH 7.
    """
    if name in REDOX_POTENTIALS:
        return REDOX_POTENTIALS[name]

    # Try case-insensitive partial match
    name_lower = name.lower()
    for key, val in REDOX_POTENTIALS.items():
        if name_lower in key.lower() or key.lower() in name_lower:
            return val

    raise ValueError(
        f"Unknown redox couple '{name}'. "
        f"Known couples: {list(REDOX_POTENTIALS.keys())}"
    )


# =====================================================================
# Van der Waals radii for edge-to-edge distance
# =====================================================================

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
    "S": 1.80, "Fe": 1.95, "Cu": 1.96, "Mn": 1.97,
    "Zn": 2.01, "Co": 1.92, "Ni": 1.91, "Mo": 2.17,
}


# =====================================================================
# Main class: StructureToParams
# =====================================================================

@dataclass
class PCETParams:
    """Extracted PCET parameters from structure."""
    V_el: float           # kcal/mol
    delta_G: float        # kcal/mol
    lambda_reorg: float   # kcal/mol (outer + inner)
    omega_H: float        # cm⁻¹
    d_DA: float           # Å

    # Diagnostic fields
    R_edge: float         # Å (edge-to-edge distance for V_el)
    lambda_outer: float   # kcal/mol
    lambda_inner: float   # kcal/mol
    method: str = "empirical"

    def to_dict(self):
        return {
            "V_el": self.V_el,
            "delta_G": self.delta_G,
            "lambda_reorg": self.lambda_reorg,
            "omega_H": self.omega_H,
            "d_DA": self.d_DA,
            "R_edge": self.R_edge,
            "lambda_outer": self.lambda_outer,
            "lambda_inner": self.lambda_inner,
            "method": self.method,
        }

    def summary(self) -> str:
        return (
            f"PCET Parameters (empirical)\n"
            f"  V_el       = {self.V_el:.4f} kcal/mol\n"
            f"  delta_G    = {self.delta_G:.2f} kcal/mol\n"
            f"  lambda     = {self.lambda_reorg:.2f} kcal/mol "
            f"(outer={self.lambda_outer:.2f} + inner={self.lambda_inner:.2f})\n"
            f"  omega_H    = {self.omega_H:.0f} cm⁻¹\n"
            f"  d_DA       = {self.d_DA:.3f} Å\n"
            f"  R_edge     = {self.R_edge:.3f} Å"
        )


class StructureToParams:
    """Extract PCET parameters from protein structure coordinates.

    Parameters
    ----------
    coupling_beta : float
        Moser-Dutton decay constant (Å⁻¹). Default 1.1.
    epsilon_s : float
        Static dielectric of protein environment. Default 4.0.
    lambda_inner : float or str
        Inner-sphere reorganization energy in kcal/mol,
        or a key from LAMBDA_INNER_TYPICAL. Default "protein".
    """

    def __init__(
        self,
        coupling_beta: float = PCET_BETA,
        epsilon_s: float = EPSILON_S_PROTEIN,
        lambda_inner: float | str = "protein",
    ):
        self.coupling_beta = coupling_beta
        self.epsilon_s = epsilon_s
        if isinstance(lambda_inner, str):
            self.lambda_inner = LAMBDA_INNER_TYPICAL.get(
                lambda_inner, LAMBDA_INNER_TYPICAL["default"]
            )
        else:
            self.lambda_inner = lambda_inner

    def from_coordinates(
        self,
        donor_xyz: Tuple[float, float, float],
        acceptor_xyz: Tuple[float, float, float],
        donor_element: str = "Fe",
        acceptor_element: str = "C",
        bond_type: str = "C-H",
        E_donor: Optional[float] = None,
        E_acceptor: Optional[float] = None,
        donor_redox: Optional[str] = None,
        acceptor_redox: Optional[str] = None,
        delta_G_override: Optional[float] = None,
        cofactor_type: Optional[str] = None,
    ) -> PCETParams:
        """Extract PCET parameters from donor/acceptor coordinates.

        Parameters
        ----------
        donor_xyz, acceptor_xyz : tuple of 3 floats
            3D coordinates of donor and acceptor atoms in Å.
        donor_element, acceptor_element : str
            Element symbols (for van der Waals radii).
        bond_type : str
            Type of X-H bond being broken (e.g. "C-H", "O-H", "S-H").
        E_donor, E_acceptor : float, optional
            Reduction potentials (V vs SHE). If not provided, use lookups.
        donor_redox, acceptor_redox : str, optional
            Names for redox potential lookup (e.g. "Fe3+-OH/Fe3+-OH2").
        delta_G_override : float, optional
            Explicit driving force in kcal/mol (overrides redox calculation).
        cofactor_type : str, optional
            Cofactor type for inner-sphere lambda (e.g. "heme", "copper").

        Returns
        -------
        PCETParams
            Extracted parameters ready for PCETRateEngine.compute_rate().
        """
        # 1. Donor-acceptor distance
        d_DA = _distance(donor_xyz, acceptor_xyz)

        # 2. Edge-to-edge distance (for ET coupling, if needed)
        r_D = VDW_RADII.get(donor_element, 1.7)
        r_A = VDW_RADII.get(acceptor_element, 1.7)
        R_edge = max(d_DA - r_D - r_A, 0.1)  # floor at 0.1 Å

        # 3. Electronic coupling
        # For PCET, use d_DA directly (proton tunneling distance-dependent)
        # For pure ET, use Moser-Dutton on edge-to-edge distance
        V_el = pcet_coupling(d_DA, beta=self.coupling_beta)

        # 4. Reorganization energy
        r_cofactor_D = COFACTOR_RADII.get(donor_element,
                       COFACTOR_RADII.get(cofactor_type or "default",
                       COFACTOR_RADII["default"]))
        r_cofactor_A = COFACTOR_RADII.get(acceptor_element,
                       COFACTOR_RADII["default"])

        lambda_outer = marcus_reorganization_outer(
            R_DA=d_DA,
            r_D=r_cofactor_D,
            r_A=r_cofactor_A,
            epsilon_s=self.epsilon_s,
        )
        lambda_inner = self.lambda_inner
        if cofactor_type and cofactor_type in LAMBDA_INNER_TYPICAL:
            lambda_inner = LAMBDA_INNER_TYPICAL[cofactor_type]
        lambda_reorg = lambda_outer + lambda_inner

        # 5. Proton frequency
        omega_H = proton_frequency(bond_type)

        # 6. Driving force
        if delta_G_override is not None:
            delta_G = delta_G_override
        elif E_donor is not None and E_acceptor is not None:
            delta_G = driving_force(E_acceptor, E_donor)
        elif donor_redox and acceptor_redox:
            E_d = lookup_redox_potential(donor_redox)
            E_a = lookup_redox_potential(acceptor_redox)
            delta_G = driving_force(E_a, E_d)
        else:
            # Default: mildly exothermic
            delta_G = -5.0

        return PCETParams(
            V_el=V_el,
            delta_G=delta_G,
            lambda_reorg=lambda_reorg,
            omega_H=omega_H,
            d_DA=d_DA,
            R_edge=R_edge,
            lambda_outer=lambda_outer,
            lambda_inner=lambda_inner,
        )

    def from_pdb_atoms(
        self,
        donor_coords: np.ndarray,
        acceptor_coords: np.ndarray,
        donor_element: str = "Fe",
        acceptor_element: str = "C",
        **kwargs,
    ) -> PCETParams:
        """Extract parameters from numpy coordinate arrays.

        Convenience wrapper for from_coordinates() that accepts numpy arrays.
        """
        return self.from_coordinates(
            donor_xyz=tuple(donor_coords[:3]),
            acceptor_xyz=tuple(acceptor_coords[:3]),
            donor_element=donor_element,
            acceptor_element=acceptor_element,
            **kwargs,
        )


def _distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))
