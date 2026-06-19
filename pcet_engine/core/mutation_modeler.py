"""
Model structural effects of point mutations on PCET active sites.

Instead of re-predicting the entire structure (AlphaFold can't capture
mutation effects), we estimate the change in donor-acceptor distance
(Δd_DA) from the mutation using geometric heuristics.

Three approaches (simplest to most accurate):

1. **Cavity volume heuristic**: Mutations that change residue size
   (e.g., Leu→Ala) enlarge or shrink the active site cavity,
   shifting d_DA proportionally.

2. **FoldX integration**: Use FoldX (external tool) to model the
   mutation and compute the new d_DA from the repaired structure.

3. **Literature d_DA values**: For well-studied systems, use
   published d_DA values from crystal structures or MD simulations.

For the commercial product, approach 1 gives fast screening (~1ms
per mutation), approach 2 gives accurate modeling (~1s per mutation),
and approach 3 gives ground truth when available.

Usage::

    from pcet_engine.core.mutation_modeler import MutationModeler

    modeler = MutationModeler(d_DA_wt=2.77)
    results = modeler.estimate_mutations([
        "L546A",   # large → small: d_DA increases
        "L754A",   # large → small: d_DA increases
        "A546L",   # small → large: d_DA decreases (hypothetical)
    ])
    for name, d_DA in results:
        print(f"{name}: d_DA = {d_DA:.3f} Å")
"""

from dataclasses import dataclass
from typing import Optional


# Amino acid side-chain volumes (Å³) from Zamyatnin (1972)
# Used to estimate cavity size changes from mutations
SIDECHAIN_VOLUMES = {
    "G": 60.1,   "A": 88.6,   "V": 140.0,  "L": 166.7,
    "I": 166.7,  "P": 122.7,  "F": 189.9,  "W": 227.8,
    "M": 162.9,  "S": 89.0,   "T": 116.1,  "C": 108.5,
    "Y": 193.6,  "H": 153.2,  "D": 111.1,  "E": 138.4,
    "N": 114.1,  "Q": 143.8,  "K": 168.6,  "R": 173.4,
}

# Empirical relationship between cavity volume change and d_DA shift.
# Calibrated from SLO-1 mutant series:
#   L546A: ΔV = 88.6 - 166.7 = -78.1 Å³, Δd_DA = +0.11 Å → coeff = -0.0014
#   L754A: ΔV = 88.6 - 166.7 = -78.1 Å³, Δd_DA = +0.18 Å → coeff = -0.0023
#   DM:    ΔV = -156.2 Å³, Δd_DA = +0.33 Å → coeff = -0.0021
# Average: coeff ≈ -0.0019 Å/Å³ (negative because volume decrease → d_DA increase)
VOLUME_TO_DDA_COEFF = -0.0020  # Å per Å³ of volume change


@dataclass
class MutationEffect:
    """Predicted structural effect of a mutation."""
    mutation: str
    wt_residue: str
    new_residue: str
    position: int
    volume_change: float       # Å³ (negative = cavity opens)
    d_DA_shift: float          # Å (positive = donor-acceptor moves apart)
    d_DA_predicted: float      # Å
    confidence: str            # "high" if near active site, "low" if distant


class MutationModeler:
    """Estimate d_DA changes from point mutations.

    Parameters
    ----------
    d_DA_wt : float
        Wild-type donor-acceptor distance in Å.
    active_site_residues : list of int, optional
        Residue positions that are in/near the active site.
        Mutations at these positions get "high" confidence;
        distant mutations get "low" confidence.
    volume_coeff : float
        Conversion factor: Å of d_DA shift per Å³ of volume change.
    """

    def __init__(
        self,
        d_DA_wt: float,
        active_site_residues: Optional[list] = None,
        volume_coeff: float = VOLUME_TO_DDA_COEFF,
    ):
        self.d_DA_wt = d_DA_wt
        self.active_site_residues = set(active_site_residues or [])
        self.volume_coeff = volume_coeff

    def estimate_mutation(self, mutation: str) -> MutationEffect:
        """Estimate d_DA change from a single point mutation.

        Parameters
        ----------
        mutation : str
            e.g., "L546A" (Leu at position 546 → Ala)

        Returns
        -------
        MutationEffect
        """
        wt_aa = mutation[0]
        new_aa = mutation[-1]
        position = int(mutation[1:-1])

        vol_wt = SIDECHAIN_VOLUMES.get(wt_aa, 130.0)
        vol_new = SIDECHAIN_VOLUMES.get(new_aa, 130.0)
        volume_change = vol_new - vol_wt  # negative if new is smaller

        d_DA_shift = self.volume_coeff * volume_change
        d_DA_predicted = self.d_DA_wt + d_DA_shift

        confidence = "high" if (
            not self.active_site_residues or
            position in self.active_site_residues
        ) else "low"

        return MutationEffect(
            mutation=mutation,
            wt_residue=wt_aa,
            new_residue=new_aa,
            position=position,
            volume_change=volume_change,
            d_DA_shift=d_DA_shift,
            d_DA_predicted=d_DA_predicted,
            confidence=confidence,
        )

    def estimate_mutations(self, mutations: list) -> list:
        """Estimate d_DA for multiple mutations.

        Parameters
        ----------
        mutations : list of str
            e.g., ["L546A", "L754A", "L546A/L754A"]

        Returns
        -------
        list of MutationEffect
        """
        results = []
        for mut in mutations:
            if "/" in mut:
                # Double/multiple mutant: sum the individual effects
                total_shift = 0.0
                total_vol = 0.0
                parts = mut.split("/")
                for part in parts:
                    effect = self.estimate_mutation(part.strip())
                    total_shift += effect.d_DA_shift
                    total_vol += effect.volume_change
                results.append(MutationEffect(
                    mutation=mut,
                    wt_residue="/".join(p[0] for p in parts),
                    new_residue="/".join(p[-1] for p in parts),
                    position=0,  # multiple positions
                    volume_change=total_vol,
                    d_DA_shift=total_shift,
                    d_DA_predicted=self.d_DA_wt + total_shift,
                    confidence="high" if not self.active_site_residues else (
                        "high" if all(
                            int(p.strip()[1:-1]) in self.active_site_residues
                            for p in parts
                        ) else "low"
                    ),
                ))
            else:
                results.append(self.estimate_mutation(mut))
        return results

    def screen_all_single_mutations(
        self,
        position: int,
        wt_residue: str,
    ) -> list:
        """Screen all 19 possible single mutations at one position.

        Parameters
        ----------
        position : int
            Residue position.
        wt_residue : str
            One-letter code of the wild-type residue at this position.

        Returns
        -------
        list of MutationEffect, sorted by predicted rate (fastest first,
        i.e., smallest d_DA first).
        """
        results = []
        for aa in SIDECHAIN_VOLUMES:
            if aa == wt_residue:
                continue
            mut_str = f"{wt_residue}{position}{aa}"
            results.append(self.estimate_mutation(mut_str))

        # Sort by d_DA (smallest = fastest predicted rate)
        results.sort(key=lambda x: x.d_DA_predicted)
        return results
