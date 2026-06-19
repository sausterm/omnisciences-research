"""Physical constants and unit conversion factors for PCET rate calculations.

All internal calculations use atomic units (hartree, bohr, amu).
Conversion factors provided for common chemistry units.
"""

import math

# Fundamental constants (CODATA 2018)
KB_HARTREE = 3.1668115634556e-6   # Boltzmann constant in hartree/K
HBAR_AU = 1.0                      # hbar = 1 in atomic units
KB_EV = 8.617333262e-5             # Boltzmann constant in eV/K
HBAR_SI = 1.054571817e-34          # J·s
KB_SI = 1.380649e-23               # J/K
AVOGADRO = 6.02214076e23
PLANCK_SI = 6.62607015e-34         # J·s

# Mass constants
AMU_TO_KG = 1.66053906660e-27
AMU_TO_AU = 1822.888486209         # 1 amu in atomic units of mass (m_e)
PROTON_MASS_AMU = 1.00782503207
DEUTERIUM_MASS_AMU = 2.01410177812
TRITIUM_MASS_AMU = 3.0160492777

# Energy conversion
HARTREE_TO_KCALMOL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KJ = 2625.4996394799
HARTREE_TO_CM = 219474.63136320   # cm^-1
EV_TO_KCALMOL = 23.060541945
KCALMOL_TO_HARTREE = 1.0 / HARTREE_TO_KCALMOL
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
CM_TO_HARTREE = 1.0 / HARTREE_TO_CM

# Length conversion
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Time conversion
AU_TIME_TO_S = 2.4188843265857e-17  # atomic time unit in seconds

# Rate conversion
AU_RATE_TO_PER_S = 1.0 / AU_TIME_TO_S

# Temperature
ROOM_TEMP = 298.15  # K

# Useful derived constants
BETA_ROOM = 1.0 / (KB_HARTREE * ROOM_TEMP)  # 1/(kBT) at 298.15 K in 1/hartree
TWO_PI = 2.0 * math.pi
