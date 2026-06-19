# Cross-System Validation of Mutation Modeler

**Date:** April 7, 2026
**Author:** Session B (Claude Code)

## Summary

The cavity volume heuristic (ΔV → Δd_DA → Δrate) was tested against crystal structure data from 3 enzyme families. **It works for SLO-1 but does NOT generalize with a universal coefficient.**

The per-system calibration method (fit log(k) vs d_DA on known variants) works by construction and is the correct commercial approach.

## Validation Data

### System 1: SLO-1 (C-H abstraction by Fe(III)-OH) — CALIBRATION SET

| Variant | PDB | d_DA (Å) | k_H (s⁻¹) | KIE | ΔV (ų) |
|---------|-----|:--:|:--:|:--:|:--:|
| WT | 3PZW | 2.77 | 297 | 81 | — |
| L546A | 5TQN | 2.88 | 8.2 | 93 | -78.1 |
| L754A | 5TR0 | 2.95 | 3.0 | 112 | -78.1 |
| DM | 4WHA | 3.10 | 0.3 | 661 | -156.2 |

**Volume-to-d_DA coefficient:** 0.002 Å/ų
**Calibrator R²:** 0.97
**Mean d_DA error (heuristic):** 0.029 Å

### System 2: ecDHFR (hydride transfer C4N→C6) — VALIDATION

| Variant | PDB | d_DA (Å) | k_hyd (s⁻¹) | Fold reduction | ΔV (ų) |
|---------|-----|:--:|:--:|:--:|:--:|
| WT | 1RX2 | 3.338 | 950 | 1x | — |
| I14V | 4QLG | 3.195 | ~135 | ~7x | -26.7 |
| I14A | 4QLE | 3.207 | ~30 | ~40x | -78.1 |
| I14G | 4QLF | 13.72* | ~0.95 | ~1000x | -106.6 |

*I14G d_DA = 13.7 Å indicates folate is in a non-reactive conformation in the crystal.

**RESULT: HEURISTIC FAILS.**
- I14V DECREASES d_DA by 0.14 Å despite removing volume — opposite to prediction
- I14A barely changes d_DA despite removing 78 ų
- I14G: folate is completely mispositioned
- The I14 mutations affect Met20 loop dynamics and cofactor repositioning, not simple cavity expansion

**Note:** DHFR is hydride (H⁻) transfer, not PCET (H-atom). KIEs are 3-8 (vs 80+ for SLO-1). The vibronic framework applies but the physics is different — more adiabatic, stronger electronic coupling.

**Calibrator R² (log k vs d_DA):** Cannot compute — d_DA does not monotonically correlate with rate for this system. The rate drops 1000x while d_DA stays roughly constant (3.2-3.3 Å for I14V and I14A). The rate change is driven by DAD distribution broadening and dynamics, not average d_DA.

### System 3: HLADH (hydride transfer C4N→C7) — PARTIAL VALIDATION

| Variant | PDB | d_DA (Å) | Substrate | Rate Effect |
|---------|-----|:--:|----------|-------------|
| WT | 4DWV | 3.358 | PFB (pentafluorobenzyl alcohol) | — |
| WT | 1HLD | 3.357 | PFB | — (consistent) |
| F93W/V203A DM | 1A71 | 3.663 | ETF (trifluoroethanol) | 75x reduction |

**Δd_DA:** +0.31 Å for the double mutant
**Net ΔV:** -13.5 ų (V203A: -51.4, F93W: +37.9)
**Coefficient:** 0.023 Å/ų — **11x larger than SLO-1**

**Caveat:** Different substrates between WT and DM structures make this comparison uncertain.

## Cross-System Coefficient Comparison

| System | Mechanism | Coefficient (Å/ų) | Relative to SLO-1 |
|--------|-----------|:--:|:--:|
| SLO-1 | HAT (C-H → Fe-OH) | 0.002 | 1.0x |
| HLADH | Hydride (C-H → NAD+) | 0.023 | 11x |
| DHFR | Hydride (NADPH → folate) | **wrong sign** | N/A |

## Conclusions

1. **A universal volume-to-d_DA coefficient does NOT exist.** The coefficient varies by >10x between enzyme families and can even have the wrong sign.

2. **Why it fails:** Different enzymes respond to mutations differently:
   - SLO-1: mutations open the substrate cavity, directly increasing d_DA
   - DHFR: mutations destabilize the closed conformation, changing dynamics without systematically changing average d_DA
   - HLADH: mutations allow cofactor repositioning, with a much larger d_DA response per unit volume

3. **What DOES work:** Per-system calibration of log(k) vs d_DA on 2-3 known variants. R² = 0.97 for SLO-1. This works by construction because you're fitting to the customer's own data.

4. **The volume heuristic is still useful** as a first-pass estimate within a single enzyme family (especially for cavity-lining residues in non-heme iron enzymes like SLO-1). But it should not be marketed as a general tool.

5. **For DHFR-type systems:** The rate change is driven by DAD distribution dynamics (broader sampling at longer average d_DA), not by a simple shift in equilibrium d_DA. The Kohen/Marcus-like tunneling model with a gating coordinate is needed — which is exactly what the PCET engine already implements (D-A gating integration).

## Data Files

Crystal structures downloaded to: `pcet_engine/data/pdb/`
- ecDHFR: 1RX2.pdb, 4QLG.pdb, 4QLE.pdb, 4QLF.pdb
- HLADH: 4DWV.pdb, 1HLD.pdb, 1A71.pdb

## References

- Hu et al. JACS 136, 8157 (2014) — SLO-1 mutant crystal structures
- Stojkovic et al. JACS 134, 1738 (2012) — DHFR I14 series kinetics
- Bahnson et al. PNAS 94, 12797 (1997) — HLADH tunneling-structure link
- Klinman & Kohen, Annu. Rev. Biochem. 82, 471 (2013) — Review
