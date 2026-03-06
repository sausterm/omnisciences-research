# Omniscience Research

A mathematical research programme connecting consciousness theory with fundamental physics.

## Overview

This repository contains the complete research output of the **Structural Idealism** programme, which establishes formal connections between two domains:

1. **Consciousness theory** — A functorial correspondence **Θ: Con → MB** mapping Hoffman's Conscious Agent networks to Friston's Markov blanket systems, proving structural equivalence between the two leading mathematical frameworks for consciousness.

2. **Fundamental physics** — The **Metric Bundle Programme**, showing that the geometry of Met(X⁴) (the bundle of Lorentzian metrics over spacetime) yields the Pati-Salam gauge group, coupling unification, and quantum mechanics from finite observation.

The unifying thesis: *consciousness is the intrinsic nature of physical structure* — what physics describes extrinsically as gauge fields and spacetime curvature is, from the inside, the dynamics of conscious agents.

## Papers

### Consciousness Programme (Θ: Con → MB)

| Paper | Title | Zenodo DOI |
|-------|-------|------------|
| **A** | Structural Idealism and the Formal Foundations of Mind | [10.5281/zenodo.18521824](https://doi.org/10.5281/zenodo.18521824) |
| **B** | Formal Correspondences Between Conscious Agents and Markov Blankets | [10.5281/zenodo.18521824](https://doi.org/10.5281/zenodo.18521824) |
| **Technical** | Conscious Agents and Markov Blankets: A Categorical Correspondence | [10.5281/zenodo.18521824](https://doi.org/10.5281/zenodo.18521824) |

### Metric Bundle Programme (Gauge Unification)

| Paper | Title | Zenodo DOI |
|-------|-------|------------|
| **1** | Gauge Structure from the Metric Bundle | [10.5281/zenodo.18860687](https://doi.org/10.5281/zenodo.18860687) |
| **2** | Torsion, Free Energy, and the Conscious Observer | [10.5281/zenodo.18860689](https://doi.org/10.5281/zenodo.18860689) |
| **3** | Gauge Dynamics from the Gauss Equation | [10.5281/zenodo.18860691](https://doi.org/10.5281/zenodo.18860691) |
| **4** | Anomaly Cancellation in the Metric Bundle | [10.5281/zenodo.18860693](https://doi.org/10.5281/zenodo.18860693) |
| **5** | Three Generations from Quaternionic Structure | [10.5281/zenodo.18860697](https://doi.org/10.5281/zenodo.18860697) |
| **6** | Quantum Mechanics from Finite Observation | — |

## Repository Structure

```
omniscience-research/
├── consciousness/           # Θ: Con → MB programme
│   ├── paper-a/             # Philosophical foundations
│   ├── paper-b/             # Formal correspondences
│   └── technical-paper/     # Categorical proofs
│
├── metric-bundle/           # Met(X⁴) gauge unification
│   ├── paper-{1..6}/        # Six papers (TeX + PDF)
│   ├── computations/        # Python verification scripts
│   └── handoff.md           # Master research document
│
├── simulations/             # Consciousness emergence models
│   ├── *.py                 # Ising, developmental, scaling
│   ├── results/             # JSON output data
│   └── figures/             # Generated plots
│
├── cit/                     # Constitutive Interface Theory
│   └── core/                # Core philosophical argument
│
└── website/                 # Interactive research visualizations
```

## Key Results

### Consciousness

- **Theorem (Θ: Con → MB):** Every conscious agent network maps functorially to a Markov blanket system preserving compositional structure under BMIC (Blanket-Mediated Interaction Condition).
- **Theorem (Inverse):** A Markov blanket system is representable as a conscious agent iff it satisfies the Agenthood Tetrad (N1–N4).
- **Result (Measure Zero):** Agent structure is rare — generic dynamical systems do not satisfy the factorisation conditions.

### Physics

- **Theorem:** The DeWitt metric on Met(X⁴) with Lorentzian background has signature **(6,4)**.
- **Corollary:** Normal bundle structure group SO(6,4) → max compact SO(6)×SO(4) ≅ SU(4)×SU(2)_L×SU(2)_R = **Pati-Salam**.
- **Result:** Dynkin index equality gives sin²θ_W = 3/8 at unification → **0.222 at M_Z** (observed: 0.231).
- **Theorem (Born Rule Uniqueness):** Among α-rules p_i ∝ |⟨e_i|ψ⟩|^α, only α = 2 gives state-independent total Fisher trace.

## Compilation

All LaTeX files compile with `pdflatex`. Metric bundle papers require BibTeX:

```bash
cd metric-bundle/paper-1
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Python scripts require NumPy and SciPy:

```bash
pip install numpy scipy matplotlib
python metric-bundle/computations/verification_suite.py
```

## Licenses

- **Papers** (`.tex`, `.pdf`): [CC BY 4.0](LICENSE-CC-BY-4.0)
- **Code** (`.py`, `.js`, `.html`, `.css`): [MIT](LICENSE-MIT)

## Citation

If you use this work, please cite the relevant paper(s). See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## Author

**Sloan Austermann**

## Acknowledgements

Computational verification assisted by Claude (Anthropic). All mathematical results independently verifiable via the included Python scripts.
