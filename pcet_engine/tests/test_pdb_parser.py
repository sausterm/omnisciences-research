"""Tests for PDB parser."""

import math
import tempfile
import pytest
from pcet_engine.parsers.pdb_parser import (
    parse_pdb, find_atom, find_pcet_site, PDBAtom, PCETSite,
)

# Minimal PDB content simulating SLO-1 active site
MOCK_PDB = """\
HEADER    OXIDOREDUCTASE                          01-JAN-00   1YGE
ATOM      1  FE  FE2 A   1       0.000   0.000   0.000  1.00 20.00          FE
ATOM      2  O   FE2 A   1       1.800   0.000   0.000  1.00 20.00           O
HETATM    3  C11 LNO A 100       2.770   0.000   0.000  1.00 20.00           C
HETATM    4  C10 LNO A 100       3.500   1.200   0.000  1.00 20.00           C
ATOM      5  SG  CYS A  50       5.000   3.000   1.000  1.00 20.00           S
ATOM      6  CA  ALA A  51       6.000   3.000   1.000  1.00 20.00           C
END
"""


@pytest.fixture
def pdb_file(tmp_path):
    path = tmp_path / "test.pdb"
    path.write_text(MOCK_PDB)
    return str(path)


@pytest.fixture
def atoms(pdb_file):
    return parse_pdb(pdb_file)


class TestParsePDB:
    def test_reads_all_atoms(self, atoms):
        assert len(atoms) == 6

    def test_atom_fields(self, atoms):
        fe = atoms[0]
        assert fe.name == "FE"
        assert fe.resname == "FE2"
        assert fe.chain == "A"
        assert fe.resid == 1
        assert fe.x == pytest.approx(0.0)
        assert fe.element == "Fe"

    def test_hetatm_flag(self, atoms):
        assert atoms[0].is_hetatm is False  # ATOM
        assert atoms[2].is_hetatm is True   # HETATM

    def test_coords_property(self, atoms):
        coords = atoms[2].coords
        assert coords[0] == pytest.approx(2.77)
        assert coords[1] == pytest.approx(0.0)
        assert coords[2] == pytest.approx(0.0)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_pdb("/nonexistent/path.pdb")


class TestFindAtom:
    def test_find_by_resname_and_name(self, atoms):
        fe = find_atom(atoms, "FE2", "FE")
        assert fe.name == "FE"
        assert fe.x == pytest.approx(0.0)

    def test_find_hetatm(self, atoms):
        c11 = find_atom(atoms, "LNO", "C11")
        assert c11.x == pytest.approx(2.77)

    def test_not_found_raises(self, atoms):
        with pytest.raises(ValueError, match="Atom not found"):
            find_atom(atoms, "XXX", "YY")

    def test_filter_by_resid(self, atoms):
        c11 = find_atom(atoms, "LNO", "C11", resid=100)
        assert c11.resid == 100


class TestFindPCETSite:
    def test_slo1_site(self, atoms):
        site = find_pcet_site(
            atoms,
            donor=("FE2", "FE"),
            acceptor=("LNO", "C11"),
        )
        assert isinstance(site, PCETSite)
        assert site.d_DA == pytest.approx(2.77, rel=0.01)
        assert site.donor_element == "Fe"
        assert site.acceptor_element == "C"
        assert site.bond_type == "C-H"

    def test_cys_to_substrate(self, atoms):
        """Cys SG at (5,3,1) to LNO C11 at (2.77,0,0)."""
        site = find_pcet_site(
            atoms,
            donor=("CYS", "SG"),
            acceptor=("LNO", "C11"),
        )
        expected = math.sqrt((5-2.77)**2 + 3**2 + 1**2)
        assert site.d_DA == pytest.approx(expected, rel=0.01)

    def test_inferred_bond_type(self, atoms):
        site = find_pcet_site(
            atoms,
            donor=("FE2", "FE"),
            acceptor=("CYS", "SG"),
        )
        assert site.bond_type == "S-H"

    def test_explicit_bond_type(self, atoms):
        site = find_pcet_site(
            atoms,
            donor=("FE2", "FE"),
            acceptor=("LNO", "C11"),
            bond_type="C-H_sp2",
        )
        assert site.bond_type == "C-H_sp2"

    def test_to_dict(self, atoms):
        site = find_pcet_site(
            atoms,
            donor=("FE2", "FE"),
            acceptor=("LNO", "C11"),
        )
        d = site.to_dict()
        assert "d_DA" in d
        assert "donor_coords" in d
        assert d["bond_type"] == "C-H"


class TestEndToEnd:
    def test_pdb_to_params(self, atoms):
        """Full pipeline: PDB → PCETSite → StructureToParams → PCETParams."""
        from pcet_engine.core.structure_to_params import StructureToParams

        site = find_pcet_site(
            atoms,
            donor=("FE2", "FE"),
            acceptor=("LNO", "C11"),
        )

        s2p = StructureToParams(lambda_inner="non_heme_iron")
        params = s2p.from_pdb_atoms(
            donor_coords=site.donor_atom.coords,
            acceptor_coords=site.acceptor_atom.coords,
            donor_element=site.donor_element,
            acceptor_element=site.acceptor_element,
            bond_type=site.bond_type,
            delta_G_override=-5.4,
        )

        assert params.d_DA == pytest.approx(2.77, rel=0.01)
        assert params.V_el == pytest.approx(0.6, rel=0.01)
        assert 2800 < params.omega_H < 3200
