# PCET Rate Theory Engine

**Patent Pending** — U.S. Provisional Application No. 64/003,092 (Filed March 11, 2026)

Proton-Coupled Electron Transfer (PCET) rate prediction from molecular Hessian data.

Built on Marcus theory + vibronic tunneling overlap with multi-channel summation
(Hammes-Schiffer formalism).

## Installation

```bash
cd pcet_engine
pip install -e .
```

## Architecture

```
pcet_engine/
├── core/                       # Rate theory engine
│   ├── constants.py            # Physical constants and unit conversions
│   ├── marcus.py               # Marcus theory rates, reorganization energy
│   ├── vibronic.py             # Franck-Condon overlaps, multi-channel vibronic rates
│   ├── normal_modes.py         # Normal mode analysis from Hessian matrices
│   ├── rate_engine.py          # High-level PCETRateEngine orchestrator
│   ├── fgh_solver.py           # Fourier Grid Hamiltonian 1D Schrödinger solver
│   ├── proton_potential.py     # Harmonic, Morse, double-well potential builders
│   ├── electrochemistry.py     # Fermi integration, EDL model, Tafel analysis
│   ├── nonadiabaticity.py      # Georgievskii-Stuchebrukhov regime classification
│   └── uncertainty.py          # Monte Carlo uncertainty propagation
├── parsers/                    # Quantum chemistry file parsers
│   ├── base.py                 # QCData structure
│   ├── gaussian_fchk.py        # Gaussian .fchk parser
│   ├── orca_hess.py            # ORCA .hess parser
│   └── scan_parser.py          # DFT scan parser (Gaussian .log, ORCA .out, CSV, NumPy)
├── data/                       # Model Hessian data for validation
│   └── model_hessians.py       # D-H-A active-site model builder
├── benchmarks/                 # Validation against published enzyme data
│   ├── systems.py              # 5 benchmark systems (SLO-1, AADH, MADH, PHM, RNR)
│   └── hessian_validation.py   # Hessian-to-rate pipeline validation
├── cli.py                      # Command-line interface
├── pyproject.toml              # Package metadata (pip-installable)
└── tests/                      # Test suite (157 tests)
    ├── test_marcus.py
    ├── test_vibronic.py
    ├── test_parsers.py
    ├── test_normal_modes.py
    ├── test_rate_engine.py
    ├── test_hessian_validation.py
    ├── test_fgh_solver.py
    ├── test_electrochemistry.py
    ├── test_proton_potential.py
    ├── test_nonadiabaticity.py
    ├── test_scan_parser.py
    └── test_uncertainty.py
```

## Quick Start

```python
from pcet_engine.core import PCETRateEngine

engine = PCETRateEngine()
result = engine.compute_rate(
    V_el=0.6,           # electronic coupling (kcal/mol)
    delta_G=-5.4,        # driving force (kcal/mol)
    lambda_reorg=19.0,   # reorganization energy (kcal/mol)
    omega_H=2900.0,      # proton frequency (cm-1)
    d_DA=2.69,           # donor-acceptor distance (angstrom)
)
print(f"k_H = {result.k_H:.2e} s-1")
print(f"k_D = {result.k_D:.2e} s-1")
print(f"KIE = {result.KIE:.1f}")
print(f"E_a = {result.E_a:.1f} kcal/mol")
```

## CLI

```bash
# Compute rate from parameters
pcet rate --V_el 0.5 --delta_G -5.0 --lambda_reorg 20.0 --omega 3000 --d_DA 2.7

# Run benchmarks
pcet benchmark

# From Hessian files
pcet hessian reactant.fchk product.fchk --proton 5 --donor 3 --acceptor 8 --V_el 0.5 --delta_G -5.0

# JSON output
pcet rate --V_el 0.5 --delta_G -5.0 --lambda_reorg 20.0 --omega 3000 --d_DA 2.7 --json
```

## From Hessian Files

```python
from pcet_engine.parsers import parse_orca_hess
from pcet_engine.core import PCETRateEngine

data_R = parse_orca_hess("reactant.hess")
data_P = parse_orca_hess("product.hess")

engine = PCETRateEngine()
result = engine.compute_rate_from_hessian(
    hessian_R=data_R.hessian, hessian_P=data_P.hessian,
    geom_R=data_R.geometry, geom_P=data_P.geometry,
    masses=data_R.masses,
    proton_idx=5, donor_idx=3, acceptor_idx=8,
    V_el=0.5, delta_G=-5.0, lambda_outer=10.0,
)
```

## FGH Rate from DFT Scans

```python
from pcet_engine.parsers import parse_scan
from pcet_engine.core import PCETRateEngine
from pcet_engine.core.proton_potential import harmonic_potential
from pcet_engine.core.constants import PROTON_MASS_AMU

# Load a DFT scan (auto-detects format: .log, .out, .csv, .npy)
r, V_R = parse_scan("reactant_scan.csv")

# Or build potentials analytically
V_P = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=0.25)

engine = PCETRateEngine()
result = engine.compute_rate_from_potential(
    r_grid=r, V_reactant=V_R, V_product=V_P,
    V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
)
```

## Electrochemical PCET

```python
engine = PCETRateEngine()
result = engine.compute_rate_electrochemical(
    V_el=0.5, delta_G_base=-3.0, lambda_reorg=20.0,
    omega_H=3000, d_DA=2.7, overpotential=0.3,
    direction='anodic',
)
print(f"k_H(eta=0.3V) = {result.k_H:.3e} s-1")
```

## Uncertainty Quantification

```python
from pcet_engine.core.uncertainty import propagate_uncertainty

uq = propagate_uncertainty(
    V_el=0.5, V_el_err=0.05,
    delta_G=-5.0, delta_G_err=0.5,
    lambda_reorg=20.0, lambda_reorg_err=2.0,
    omega_H=3000, omega_H_err=100,
    d_DA=2.7, d_DA_err=0.05,
    n_samples=1000, seed=42,
)
print(f"k_H = {uq.k_H_mean:.2e} +/- {uq.k_H_std:.2e}")
print(f"KIE = {uq.KIE_mean:.1f} (95% CI: {uq.KIE_ci[0]:.1f}-{uq.KIE_ci[1]:.1f})")
print(f"Sensitivities: {uq.sensitivities}")
```

## Nonadiabaticity Analysis

```python
from pcet_engine.core import analyze_nonadiabaticity
from pcet_engine.core.proton_potential import harmonic_potential
from pcet_engine.core.constants import PROTON_MASS_AMU
import numpy as np

r = np.linspace(-1.0, 1.0, 256)
V_R = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=-0.25)(r)
V_P = harmonic_potential(3000, PROTON_MASS_AMU, r_eq=0.25)(r)

result = analyze_nonadiabaticity(r, V_R, V_P, V_el=0.01)
print(f"Regime: {result.regime}")
print(f"p = {result.p:.3f}, kappa = {result.kappa:.3f}")
```

## Methods

- **`marcus`**: Pure Marcus theory (no tunneling, KIE = 1)
- **`vibronic_single`**: Ground-state vibronic channel only (overestimates KIE)
- **`vibronic_multi`**: Full multi-channel summation (default, best for KIE)
- **`fgh_vibronic`**: FGH solver for arbitrary anharmonic potentials

## Validation Status

### Parameter-based (calibrated delta_0, Phase 3)

| System | KIE pred | KIE exp | E_a pred | E_a exp | Notes |
|--------|----------|---------|----------|---------|-------|
| SLO-1  | 72.8     | 81      | 2.4      | 2.1     | delta_0=0.50 A |
| AADH   | 46.8     | 55      | 3.4      | 11.4    | delta_0=0.465 A |
| MADH   | 26.5     | 30      | 4.6      | 14.1    | delta_0=0.432 A |
| PHM    | 9.6      | 10      | 2.3      | 4.0     | delta_0=0.347 A |
| RNR    | 6.5      | 7.0     | 1.9      | 3.5     | delta_0=0.348 A |

### pyPCET head-to-head (identical DFT proton potentials)

| Example | Quantity | pyPCET | Our Engine | Ratio |
|---------|----------|--------|-----------|-------|
| BIP electrochem | k_H | 3.37e+06 | 3.37e+06 | **1.000** |
| BIP electrochem | KIE | 1.75 | 1.75 | **1.001** |
| BIP photochem | k_H | 1.23e+09 | 1.23e+09 | **1.000** |
| RNR Y356-Y731 | k_H | 1.58e+03 | 1.37e+03 | 0.871 |

### Hessian pipeline (Phase 2)

| System | omega_H extracted | omega_H lit | KIE pred | KIE exp | Notes |
|--------|-------------------|-------------|----------|---------|-------|
| SLO-1  | 2975              | 2900        | 129.5    | 81      | Model triatomic C-H-O |
| AADH   | 3072              | 3000        | 59.3     | 55      | Model triatomic C-H-N |
| MADH   | 3022              | 2950        | 33.1     | 30      | Model triatomic C-H-N |
| PHM    | 3170              | 3100        | 11.0     | 10      | Model triatomic C-H-O |
| RNR    | 2687              | 2600        | 7.5      | 7       | Model triatomic S-H-C |

Mean frequency extraction error: 2.6%. Mean |KIE_ratio - 1| = 0.19.

## Performance

Measured on Apple M-series, single-threaded (March 2026):

| Operation | Time |
|-----------|------|
| Marcus rate | 2 µs |
| Vibronic multi-channel | 200 µs |
| Vibronic + D-A gating | 39 ms |
| FGH from numerical potential | 160 ms |
| Batch throughput | 3,300 vibronic rates/sec |

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest pcet_engine/tests/ -q
```

163 tests, all passing.
