"""
Parse PDB files to extract PCET-relevant atom coordinates.

Extracts donor and acceptor atom positions for computing d_DA and
feeding into the Structure-to-Parameters module.

Handles standard PDB format (ATOM/HETATM records) and common
cofactor/residue naming conventions.

Usage::

    from pcet_engine.parsers.pdb_parser import parse_pdb, find_pcet_site

    atoms = parse_pdb("1YGE.pdb")
    site = find_pcet_site(atoms, donor=("FE2", "FE"), acceptor=("LNO", "C11"))
    print(f"d_DA = {site.d_DA:.2f} Å")
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class PDBAtom:
    """A single atom from a PDB file."""
    serial: int
    name: str        # atom name (e.g., "CA", "FE", "C11")
    resname: str     # residue name (e.g., "CYS", "HEM", "LNO")
    chain: str       # chain ID
    resid: int       # residue sequence number
    x: float
    y: float
    z: float
    element: str     # element symbol
    is_hetatm: bool  # True if HETATM, False if ATOM

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class PCETSite:
    """Identified PCET donor-acceptor site."""
    donor_atom: PDBAtom
    acceptor_atom: PDBAtom
    d_DA: float                    # donor-acceptor distance (Å)
    donor_element: str
    acceptor_element: str
    bond_type: Optional[str] = None  # e.g., "C-H", "O-H", "S-H"

    def to_dict(self):
        return {
            "d_DA": self.d_DA,
            "donor": f"{self.donor_atom.resname}.{self.donor_atom.name}",
            "acceptor": f"{self.acceptor_atom.resname}.{self.acceptor_atom.name}",
            "donor_element": self.donor_element,
            "acceptor_element": self.acceptor_element,
            "bond_type": self.bond_type,
            "donor_coords": [self.donor_atom.x, self.donor_atom.y, self.donor_atom.z],
            "acceptor_coords": [self.acceptor_atom.x, self.acceptor_atom.y, self.acceptor_atom.z],
        }


def parse_pdb(filepath: str | Path) -> List[PDBAtom]:
    """Parse a PDB file and return all atoms.

    Parameters
    ----------
    filepath : str or Path
        Path to PDB file.

    Returns
    -------
    list of PDBAtom
        All ATOM and HETATM records.
    """
    atoms = []
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDB file not found: {filepath}")

    with open(filepath) as f:
        for line in f:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue

            try:
                serial = int(line[6:11])
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21:22].strip() or "A"
                resid = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                # Element is in columns 77-78, or infer from atom name
                element = line[76:78].strip() if len(line) > 77 else ""
                if element:
                    element = element.capitalize()  # "FE" → "Fe", "C" → "C"
                if not element:
                    element = _infer_element(name)

                atoms.append(PDBAtom(
                    serial=serial, name=name, resname=resname,
                    chain=chain, resid=resid, x=x, y=y, z=z,
                    element=element, is_hetatm=(record == "HETATM"),
                ))
            except (ValueError, IndexError):
                continue  # skip malformed lines

    return atoms


def find_atom(
    atoms: List[PDBAtom],
    resname: str,
    atom_name: str,
    chain: str = "",
    resid: Optional[int] = None,
) -> PDBAtom:
    """Find a specific atom by residue name and atom name.

    Parameters
    ----------
    atoms : list of PDBAtom
    resname : str
        Residue name (e.g., "CYS", "HEM", "FE2").
    atom_name : str
        Atom name (e.g., "SG", "FE", "C11").
    chain : str, optional
        Chain ID filter.
    resid : int, optional
        Residue number filter.

    Returns
    -------
    PDBAtom

    Raises
    ------
    ValueError
        If atom not found or multiple matches without resid.
    """
    matches = [
        a for a in atoms
        if a.resname == resname and a.name == atom_name
        and (not chain or a.chain == chain)
        and (resid is None or a.resid == resid)
    ]
    if len(matches) == 0:
        raise ValueError(
            f"Atom not found: {resname}.{atom_name}"
            + (f" chain={chain}" if chain else "")
            + (f" resid={resid}" if resid else "")
        )
    if len(matches) > 1 and resid is None:
        # Return first match but warn
        pass
    return matches[0]


def find_pcet_site(
    atoms: List[PDBAtom],
    donor: Tuple[str, str],
    acceptor: Tuple[str, str],
    donor_chain: str = "",
    acceptor_chain: str = "",
    donor_resid: Optional[int] = None,
    acceptor_resid: Optional[int] = None,
    bond_type: Optional[str] = None,
) -> PCETSite:
    """Identify a PCET donor-acceptor site from atom specifications.

    Parameters
    ----------
    atoms : list of PDBAtom
        Parsed PDB atoms.
    donor : tuple of (resname, atom_name)
        Donor specification, e.g., ("FE2", "FE") or ("HEM", "FE").
    acceptor : tuple of (resname, atom_name)
        Acceptor specification, e.g., ("LNO", "C11") or ("CYS", "SG").
    bond_type : str, optional
        X-H bond type. If not provided, inferred from acceptor element.

    Returns
    -------
    PCETSite
    """
    d_atom = find_atom(atoms, donor[0], donor[1], donor_chain, donor_resid)
    a_atom = find_atom(atoms, acceptor[0], acceptor[1], acceptor_chain, acceptor_resid)

    d_DA = float(np.linalg.norm(d_atom.coords - a_atom.coords))

    if bond_type is None:
        bond_type = _infer_bond_type(a_atom.element)

    return PCETSite(
        donor_atom=d_atom,
        acceptor_atom=a_atom,
        d_DA=d_DA,
        donor_element=d_atom.element,
        acceptor_element=a_atom.element,
        bond_type=bond_type,
    )


def _infer_element(atom_name: str) -> str:
    """Infer element from PDB atom name."""
    # PDB convention: element right-justified in columns 13-14
    # For common cases: CA→C, FE→Fe, etc.
    name = atom_name.strip()
    if len(name) >= 2 and name[:2] in ("FE", "CU", "MN", "ZN", "CO", "NI", "MO"):
        return name[:2].capitalize()
    if name[0] in "CNOSH":
        return name[0]
    return name[0]


def _infer_bond_type(acceptor_element: str) -> str:
    """Infer X-H bond type from the acceptor heavy atom element."""
    mapping = {
        "C": "C-H",
        "O": "O-H",
        "N": "N-H",
        "S": "S-H",
    }
    return mapping.get(acceptor_element, "C-H")


# =====================================================================
# Known PCET systems — auto-detect donor/acceptor from residue names
# =====================================================================

KNOWN_SYSTEMS = {
    "SLO1": {
        "description": "Soybean lipoxygenase-1: Fe(III)-OH abstracts H from linoleic acid",
        "donor": ("FE2", "FE"),       # or ("HEM", "FE") depending on PDB
        "acceptor": ("LNO", "C11"),    # linoleic acid C11
        "bond_type": "C-H",
        "delta_G": -5.4,              # kcal/mol
        "donor_redox": "Fe3+-OH/Fe3+-OH2",
    },
    "RNR": {
        "description": "Ribonucleotide reductase: Cys thiyl radical abstracts H from substrate",
        "donor": ("CYS", "SG"),
        "acceptor": ("SUB", "C3'"),
        "bond_type": "C-H",
    },
    "P450": {
        "description": "Cytochrome P450 Compound I: Fe=O abstracts H from substrate",
        "donor": ("HEM", "O"),         # Compound I oxo
        "acceptor": None,              # substrate-dependent
        "bond_type": "C-H",
    },
    "AADH": {
        "description": "Aromatic amine dehydrogenase: TTQ cofactor",
        "donor": ("TTQ", "C6"),
        "acceptor": None,
        "bond_type": "C-H",
    },
}
