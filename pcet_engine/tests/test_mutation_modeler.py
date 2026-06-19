"""Tests for mutation modeler."""

import pytest
from pcet_engine.core.mutation_modeler import (
    MutationModeler, MutationEffect, SIDECHAIN_VOLUMES,
)


@pytest.fixture
def modeler():
    return MutationModeler(d_DA_wt=2.77, active_site_residues=[546, 754])


class TestMutationEffect:
    def test_large_to_small_increases_dda(self, modeler):
        """Leu→Ala (large→small) should increase d_DA."""
        effect = modeler.estimate_mutation("L546A")
        assert effect.d_DA_predicted > 2.77

    def test_small_to_large_decreases_dda(self, modeler):
        """Ala→Leu (small→large) should decrease d_DA."""
        m = MutationModeler(d_DA_wt=2.90)
        effect = m.estimate_mutation("A546L")
        assert effect.d_DA_predicted < 2.90

    def test_same_size_no_change(self, modeler):
        """Leu→Ile (same volume) should barely change d_DA."""
        effect = modeler.estimate_mutation("L546I")
        assert abs(effect.d_DA_predicted - 2.77) < 0.01

    def test_slo1_l546a_accuracy(self, modeler):
        """L546A should predict d_DA ≈ 2.88 (exp) within 0.05 Å."""
        effect = modeler.estimate_mutation("L546A")
        assert abs(effect.d_DA_predicted - 2.88) < 0.05

    def test_slo1_l754a_accuracy(self, modeler):
        """L754A should predict d_DA ≈ 2.95 (exp) within 0.05 Å."""
        effect = modeler.estimate_mutation("L754A")
        assert abs(effect.d_DA_predicted - 2.95) < 0.05

    def test_double_mutant(self, modeler):
        """DM (L546A/L754A) should predict d_DA ≈ 3.10 within 0.05 Å."""
        results = modeler.estimate_mutations(["L546A/L754A"])
        assert len(results) == 1
        assert abs(results[0].d_DA_predicted - 3.10) < 0.05

    def test_confidence_active_site(self, modeler):
        effect = modeler.estimate_mutation("L546A")
        assert effect.confidence == "high"

    def test_confidence_distant(self, modeler):
        effect = modeler.estimate_mutation("L100A")
        assert effect.confidence == "low"


class TestSaturationScan:
    def test_scan_returns_19(self, modeler):
        scan = modeler.screen_all_single_mutations(546, "L")
        assert len(scan) == 19  # 20 AAs minus wild-type

    def test_scan_sorted_by_dda(self, modeler):
        scan = modeler.screen_all_single_mutations(546, "L")
        d_das = [e.d_DA_predicted for e in scan]
        assert d_das == sorted(d_das)

    def test_trp_compresses(self, modeler):
        """Trp is largest AA — should give smallest d_DA."""
        scan = modeler.screen_all_single_mutations(546, "L")
        assert scan[0].new_residue == "W"
        assert scan[0].d_DA_predicted < 2.77

    def test_gly_opens(self, modeler):
        """Gly is smallest AA — should give largest d_DA."""
        scan = modeler.screen_all_single_mutations(546, "L")
        assert scan[-1].new_residue == "G"
        assert scan[-1].d_DA_predicted > 2.77


class TestBatchMutations:
    def test_multiple_mutations(self, modeler):
        results = modeler.estimate_mutations(["L546A", "L754A"])
        assert len(results) == 2

    def test_ranking_preserved(self, modeler):
        """L546A and L754A both open the cavity — L546A+L754A opens more."""
        results = modeler.estimate_mutations(["L546A", "L754A", "L546A/L754A"])
        d_das = [r.d_DA_predicted for r in results]
        # DM should have largest d_DA
        assert results[2].d_DA_predicted > results[0].d_DA_predicted
        assert results[2].d_DA_predicted > results[1].d_DA_predicted
