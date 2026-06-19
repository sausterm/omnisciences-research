"""Tests for Structure-to-Parameters module."""

import math
import pytest
from pcet_engine.core.structure_to_params import (
    StructureToParams,
    PCETParams,
    moser_dutton_coupling,
    pcet_coupling,
    marcus_reorganization_outer,
    proton_frequency,
    driving_force,
    lookup_redox_potential,
    PCET_V0,
    PCET_D0,
)


# ── Moser-Dutton coupling ──────────────────────────────────────────

class TestMoserDutton:
    def test_contact_distance(self):
        """At contact distance, coupling = V0."""
        V = moser_dutton_coupling(3.6)
        assert V == pytest.approx(73.0, rel=1e-3)

    def test_decays_with_distance(self):
        """Coupling decreases with distance."""
        V_close = moser_dutton_coupling(4.0)
        V_far = moser_dutton_coupling(8.0)
        assert V_close > V_far

    def test_exponential_form(self):
        """V(R) = V0 * exp(-beta * (R - R0))."""
        R = 5.0
        V = moser_dutton_coupling(R, beta=1.1, V0=73.0, R0=3.6)
        expected = 73.0 * math.exp(-1.1 * (5.0 - 3.6))
        assert V == pytest.approx(expected, rel=1e-10)


# ── PCET coupling ──────────────────────────────────────────────────

class TestPCETCoupling:
    def test_reference_distance(self):
        """At reference distance, coupling = V0."""
        V = pcet_coupling(PCET_D0)
        assert V == pytest.approx(PCET_V0, rel=1e-10)

    def test_decays_with_distance(self):
        V_close = pcet_coupling(2.70)
        V_far = pcet_coupling(3.10)
        assert V_close > V_far

    def test_slo1_wt_coupling(self):
        """SLO-1 WT at d_DA = 2.77 should give V_el ≈ 0.6 kcal/mol."""
        V = pcet_coupling(2.77)
        assert V == pytest.approx(0.6, rel=0.01)

    def test_slo1_mutant_ordering(self):
        """V_el should decrease: WT > L546A > L754A > DM."""
        couplings = [pcet_coupling(d) for d in [2.77, 2.88, 2.95, 3.10]]
        for i in range(len(couplings) - 1):
            assert couplings[i] > couplings[i + 1]


# ── Marcus reorganization ──────────────────────────────────────────

class TestMarcusReorg:
    def test_positive(self):
        lam = marcus_reorganization_outer(5.0, r_D=1.5, r_A=1.5)
        assert lam > 0

    def test_pekar_factor(self):
        """Water (high epsilon_s) has smaller Pekar factor but geometry
        term is the same → larger lambda because 1/eps_opt dominates."""
        lam_protein = marcus_reorganization_outer(5.0, epsilon_s=4.0)
        lam_water = marcus_reorganization_outer(5.0, epsilon_s=78.4)
        # Water has larger lambda because (1/eps_opt - 1/eps_s) is larger
        # when eps_s is large (1/78 ≈ 0 vs 1/4 = 0.25)
        assert lam_water > lam_protein

    def test_increases_with_distance(self):
        """Lambda increases as R_DA grows: 1/R_DA shrinks → geometry term grows."""
        lam_close = marcus_reorganization_outer(4.0)
        lam_far = marcus_reorganization_outer(10.0)
        assert lam_far > lam_close


# ── Proton frequency ──────────────────────────────────────────────

class TestProtonFrequency:
    def test_ch_frequency(self):
        omega = proton_frequency("C-H")
        assert 2800 < omega < 3200

    def test_oh_frequency(self):
        omega = proton_frequency("O-H")
        assert 3100 < omega < 3700

    def test_sh_frequency(self):
        omega = proton_frequency("S-H")
        assert 2400 < omega < 2700

    def test_nh_frequency(self):
        omega = proton_frequency("N-H")
        assert 3100 < omega < 3600

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            proton_frequency("X-Y")


# ── Driving force ──────────────────────────────────────────────────

class TestDrivingForce:
    def test_exothermic(self):
        """Higher E_acceptor → negative ΔG (exothermic)."""
        dG = driving_force(E_acceptor=0.5, E_donor=0.0)
        assert dG < 0

    def test_endothermic(self):
        dG = driving_force(E_acceptor=0.0, E_donor=0.5)
        assert dG > 0

    def test_zero(self):
        dG = driving_force(E_acceptor=0.3, E_donor=0.3)
        assert dG == pytest.approx(0.0, abs=1e-10)

    def test_lookup(self):
        E = lookup_redox_potential("NAD+/NADH")
        assert E == pytest.approx(-0.32, rel=0.01)

    def test_lookup_unknown_raises(self):
        with pytest.raises(ValueError):
            lookup_redox_potential("nonexistent_couple")


# ── Full pipeline ──────────────────────────────────────────────────

class TestStructureToParams:
    @pytest.fixture
    def s2p(self):
        return StructureToParams(lambda_inner="non_heme_iron")

    def test_returns_pcet_params(self, s2p):
        params = s2p.from_coordinates(
            donor_xyz=(0, 0, 0),
            acceptor_xyz=(2.77, 0, 0),
            donor_element="Fe",
            acceptor_element="C",
            bond_type="C-H",
            delta_G_override=-5.4,
        )
        assert isinstance(params, PCETParams)

    def test_slo1_wt_params(self, s2p):
        """SLO-1 WT should give reasonable parameters."""
        params = s2p.from_coordinates(
            donor_xyz=(0, 0, 0),
            acceptor_xyz=(2.77, 0, 0),
            donor_element="Fe",
            acceptor_element="C",
            bond_type="C-H",
            delta_G_override=-5.4,
        )
        assert params.V_el == pytest.approx(0.6, rel=0.01)
        assert params.d_DA == pytest.approx(2.77, rel=0.01)
        assert params.delta_G == pytest.approx(-5.4, rel=0.01)
        assert 2800 < params.omega_H < 3200
        assert params.lambda_reorg > 10  # should be substantial

    def test_mutant_ranking_preserved(self, s2p):
        """Extracted V_el should decrease with increasing d_DA."""
        params_list = []
        for d_DA in [2.77, 2.88, 2.95, 3.10]:
            p = s2p.from_coordinates(
                donor_xyz=(0, 0, 0),
                acceptor_xyz=(d_DA, 0, 0),
                donor_element="Fe",
                acceptor_element="C",
                bond_type="C-H",
                delta_G_override=-5.4,
            )
            params_list.append(p)
        for i in range(len(params_list) - 1):
            assert params_list[i].V_el > params_list[i + 1].V_el

    def test_redox_lookup(self, s2p):
        params = s2p.from_coordinates(
            donor_xyz=(0, 0, 0),
            acceptor_xyz=(5.0, 0, 0),
            donor_element="Fe",
            acceptor_element="C",
            bond_type="C-H",
            donor_redox="Fe3+-OH/Fe3+-OH2",
            acceptor_redox="C-H_linoleic",
        )
        assert params.delta_G != -5.0  # should use looked-up values

    def test_3d_distance(self, s2p):
        """Distance should work in 3D, not just along x-axis."""
        params = s2p.from_coordinates(
            donor_xyz=(1, 2, 3),
            acceptor_xyz=(3, 4, 5),
            delta_G_override=-5.0,
        )
        expected_d = math.sqrt(4 + 4 + 4)
        assert params.d_DA == pytest.approx(expected_d, rel=1e-10)

    def test_to_dict(self, s2p):
        params = s2p.from_coordinates(
            donor_xyz=(0, 0, 0),
            acceptor_xyz=(2.77, 0, 0),
            delta_G_override=-5.0,
        )
        d = params.to_dict()
        assert "V_el" in d
        assert "d_DA" in d
        assert "lambda_reorg" in d

    def test_summary_string(self, s2p):
        params = s2p.from_coordinates(
            donor_xyz=(0, 0, 0),
            acceptor_xyz=(2.77, 0, 0),
            delta_G_override=-5.0,
        )
        s = params.summary()
        assert "V_el" in s
        assert "d_DA" in s
