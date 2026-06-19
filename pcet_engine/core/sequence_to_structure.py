"""
Predict protein structures from amino acid sequences via ESMFold or AlphaFold DB.

This is the final link in the chain:
    Sequence → Structure → Parameters → Rate → Ranking

Three backends:
    1. AlphaFold DB: Pre-computed structures for known proteins (by UniProt ID)
    2. ESMFold API: Fast structure prediction for any sequence (including mutants)
    3. Local: Use a pre-existing PDB file

Usage::

    from pcet_engine.core.sequence_to_structure import StructurePredictor

    predictor = StructurePredictor()

    # From UniProt ID (AlphaFold DB)
    pdb_path = predictor.from_uniprot("P08170")  # SLO-1

    # From sequence (ESMFold)
    pdb_path = predictor.from_sequence(
        "MVLSPADKTNVKAAWGKVGAHAGEYGAE...",
        name="SLO1_L546A",
    )

    # Generate mutant structures
    structures = predictor.mutant_scan(
        wild_type_sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAE...",
        mutations=["L546A", "L754A", "L546A/L754A"],
    )
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import urllib.request
    import urllib.error
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False


ALPHAFOLD_DB_URL = "https://alphafold.ebi.ac.uk/api/prediction"
# NOTE: ESMFold API was shut down by Meta in mid-2024.
# Self-hosting requires a GPU with 16+ GB VRAM.
# For mutant structures, FoldX or Rosetta are more appropriate than
# re-predicting with ESMFold (AlphaFold doesn't capture mutation effects).
ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"  # DEFUNCT

# Standard amino acid 3-letter to 1-letter code
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


@dataclass
class StructurePrediction:
    """Result of a structure prediction."""
    pdb_path: str          # path to PDB file
    source: str            # "alphafold_db", "esmfold", "local"
    name: str              # identifier
    sequence_length: int
    confidence: Optional[float] = None  # pLDDT score (0-100)


def apply_mutation(sequence: str, mutation: str) -> str:
    """Apply a point mutation to a sequence.

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence.
    mutation : str
        Mutation string, e.g., "L546A" (Leu at position 546 → Ala).
        For multiple mutations: "L546A/L754A".

    Returns
    -------
    str
        Mutated sequence.
    """
    seq = list(sequence)
    for mut in mutation.split("/"):
        mut = mut.strip()
        if len(mut) < 3:
            raise ValueError(f"Invalid mutation format: {mut}")
        wt_aa = mut[0]
        new_aa = mut[-1]
        pos = int(mut[1:-1]) - 1  # convert to 0-indexed

        if pos < 0 or pos >= len(seq):
            raise ValueError(
                f"Position {pos + 1} out of range for sequence length {len(seq)}"
            )
        if seq[pos] != wt_aa:
            raise ValueError(
                f"Expected {wt_aa} at position {pos + 1}, found {seq[pos]}"
            )
        seq[pos] = new_aa

    return "".join(seq)


class StructurePredictor:
    """Predict protein structures from sequences.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to cache downloaded/predicted structures.
        Default: temp directory.
    backend : str
        Default backend: "esmfold" or "alphafold_db".
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        backend: str = "esmfold",
        timeout: int = 120,
    ):
        if cache_dir is None:
            self._cache_dir = Path(tempfile.mkdtemp(prefix="pcet_structures_"))
        else:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.timeout = timeout

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def from_uniprot(self, uniprot_id: str) -> StructurePrediction:
        """Fetch pre-computed AlphaFold structure by UniProt ID.

        Parameters
        ----------
        uniprot_id : str
            UniProt accession (e.g., "P08170" for SLO-1).

        Returns
        -------
        StructurePrediction
        """
        if not _HAS_URLLIB:
            raise RuntimeError("urllib not available")

        cache_path = self._cache_dir / f"AF-{uniprot_id}.pdb"
        if cache_path.exists():
            seq_len = _count_residues(str(cache_path))
            return StructurePrediction(
                pdb_path=str(cache_path),
                source="alphafold_db",
                name=f"AF-{uniprot_id}",
                sequence_length=seq_len,
            )

        # Fetch from AlphaFold DB
        url = f"{ALPHAFOLD_DB_URL}/{uniprot_id}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise ValueError(
                f"AlphaFold DB lookup failed for {uniprot_id}: {e.code}"
            )

        if not data or not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"No AlphaFold structure found for {uniprot_id}")

        pdb_url = data[0].get("pdbUrl")
        if not pdb_url:
            raise ValueError(f"No PDB URL in AlphaFold response for {uniprot_id}")

        # Download PDB
        urllib.request.urlretrieve(pdb_url, str(cache_path))

        seq_len = _count_residues(str(cache_path))
        confidence = data[0].get("globalMetricValue")

        return StructurePrediction(
            pdb_path=str(cache_path),
            source="alphafold_db",
            name=f"AF-{uniprot_id}",
            sequence_length=seq_len,
            confidence=confidence,
        )

    def from_sequence(
        self,
        sequence: str,
        name: str = "prediction",
    ) -> StructurePrediction:
        """Predict structure from amino acid sequence via ESMFold.

        Parameters
        ----------
        sequence : str
            One-letter amino acid sequence.
        name : str
            Identifier for the prediction.

        Returns
        -------
        StructurePrediction
        """
        if not _HAS_URLLIB:
            raise RuntimeError("urllib not available")

        # Check cache
        cache_path = self._cache_dir / f"{name}.pdb"
        if cache_path.exists():
            return StructurePrediction(
                pdb_path=str(cache_path),
                source="esmfold",
                name=name,
                sequence_length=len(sequence),
            )

        # Clean sequence
        sequence = sequence.strip().upper()
        sequence = "".join(c for c in sequence if c in AA_3TO1.values())

        # Submit to ESMFold
        req = urllib.request.Request(
            ESMFOLD_API_URL,
            data=sequence.encode(),
            headers={"Content-Type": "text/plain"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                pdb_text = resp.read().decode()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ESMFold prediction failed: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"ESMFold API unreachable: {e.reason}")

        cache_path.write_text(pdb_text)

        return StructurePrediction(
            pdb_path=str(cache_path),
            source="esmfold",
            name=name,
            sequence_length=len(sequence),
        )

    def from_local(self, pdb_path: str | Path, name: str = "local") -> StructurePrediction:
        """Use an existing local PDB file."""
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        seq_len = _count_residues(str(pdb_path))
        return StructurePrediction(
            pdb_path=str(pdb_path),
            source="local",
            name=name,
            sequence_length=seq_len,
        )

    def mutant_scan(
        self,
        wild_type_sequence: str,
        mutations: list,
        wt_name: str = "WT",
    ) -> list:
        """Predict structures for wild-type and a list of mutants.

        Parameters
        ----------
        wild_type_sequence : str
            Wild-type one-letter amino acid sequence.
        mutations : list of str
            Mutation strings, e.g., ["L546A", "L754A", "L546A/L754A"].
        wt_name : str
            Name for the wild-type prediction.

        Returns
        -------
        list of StructurePrediction
            One per variant (WT + all mutants).
        """
        results = []

        # Wild-type
        wt_pred = self.from_sequence(wild_type_sequence, name=wt_name)
        results.append(wt_pred)

        # Mutants
        for mut in mutations:
            mut_seq = apply_mutation(wild_type_sequence, mut)
            mut_name = mut.replace("/", "_")
            pred = self.from_sequence(mut_seq, name=mut_name)
            results.append(pred)

        return results


def _count_residues(pdb_path: str) -> int:
    """Count unique residues in a PDB file."""
    residues = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21]
                resid = line[22:26].strip()
                residues.add((chain, resid))
    return len(residues)
