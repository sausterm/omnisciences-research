"""
PCET Rate Theory Engine
=======================

Proton-Coupled Electron Transfer rate prediction from molecular Hessian data.
Built on Marcus theory + vibronic tunneling overlap with multi-channel summation.

Key capabilities:
- Parse Gaussian .fchk and ORCA .hess files to extract Hessian matrices
- Normal mode analysis: frequencies, eigenvectors, reduced masses
- Marcus theory rate constants with nuclear tunneling corrections
- Multi-channel vibronic rate summation (Hammes-Schiffer formalism)
- Kinetic isotope effect (KIE) prediction
- Benchmark against 5 published enzyme PCET systems
"""

__version__ = "0.1.0"
