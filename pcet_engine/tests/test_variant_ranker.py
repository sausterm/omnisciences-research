"""Tests for the variant ranker module."""

import pytest
from pcet_engine.core.variant_ranker import VariantRanker, RankingResult


SLO1_VARIANTS = [
    {"name": "WT",    "d_DA": 2.77},
    {"name": "L546A", "d_DA": 2.88},
    {"name": "L754A", "d_DA": 2.95},
    {"name": "DM",    "d_DA": 3.10},
]


@pytest.fixture
def ranker():
    return VariantRanker(
        V_el=0.6, delta_G=-5.4, lambda_reorg=19.0, omega_H=2900.0,
    )


def test_rank_returns_ranking_result(ranker):
    result = ranker.rank(SLO1_VARIANTS)
    assert isinstance(result, RankingResult)
    assert result.n_variants == 4
    assert result.reference == "WT"


def test_rank_order_matches_experiment(ranker):
    """WT should be fastest, DM slowest (larger d_DA = slower rate)."""
    result = ranker.rank(SLO1_VARIANTS)
    names = [r.name for r in result.ranked]
    assert names == ["WT", "L546A", "L754A", "DM"]


def test_rate_decreases_with_distance(ranker):
    """Rate should decrease monotonically with increasing d_DA."""
    result = ranker.rank(SLO1_VARIANTS)
    rates = [r.k_H for r in result.ranked]
    for i in range(len(rates) - 1):
        assert rates[i] > rates[i + 1]


def test_kie_increases_with_distance(ranker):
    """KIE should increase with d_DA (more tunneling at longer distance)."""
    result = ranker.rank(SLO1_VARIANTS)
    kies = [r.KIE for r in result.ranked]
    for i in range(len(kies) - 1):
        assert kies[i] < kies[i + 1]


def test_rate_ratio_reference(ranker):
    result = ranker.rank(SLO1_VARIANTS, reference="WT")
    wt = next(r for r in result.ranked if r.name == "WT")
    assert wt.rate_ratio == pytest.approx(1.0)
    # All mutants should be slower than WT
    for r in result.ranked:
        if r.name != "WT":
            assert r.rate_ratio < 1.0


def test_custom_reference(ranker):
    result = ranker.rank(SLO1_VARIANTS, reference="L546A")
    assert result.reference == "L546A"
    l546a = next(r for r in result.ranked if r.name == "L546A")
    assert l546a.rate_ratio == pytest.approx(1.0)


def test_single_variant(ranker):
    result = ranker.rank([{"name": "WT", "d_DA": 2.77}])
    assert result.n_variants == 1
    assert result.ranked[0].rank == 1
    assert result.ranked[0].rate_ratio == pytest.approx(1.0)


def test_per_variant_override(ranker):
    """Variants can override shared parameters."""
    result = ranker.rank([
        {"name": "normal", "d_DA": 2.77},
        {"name": "modified", "d_DA": 2.77, "delta_G": -10.0},
    ])
    # More exothermic should be faster
    normal = next(r for r in result.ranked if r.name == "normal")
    modified = next(r for r in result.ranked if r.name == "modified")
    assert modified.k_H > normal.k_H


def test_to_dict(ranker):
    result = ranker.rank(SLO1_VARIANTS)
    d = result.to_dict()
    assert "variants" in d
    assert len(d["variants"]) == 4
    assert d["variants"][0]["rank"] == 1


def test_summary_string(ranker):
    result = ranker.rank(SLO1_VARIANTS)
    s = result.summary()
    assert "WT" in s
    assert "DM" in s
    assert "Rank" in s


def test_uq_mode(ranker):
    result = ranker.rank(SLO1_VARIANTS[:2], run_uq=True, n_samples=50)
    for r in result.ranked:
        assert r.k_H_ci is not None
        assert r.KIE_ci is not None
        assert r.dominant_sensitivity is not None
        assert r.k_H_ci[0] < r.k_H_ci[1]


def test_empty_variants_raises(ranker):
    with pytest.raises(ValueError):
        ranker.rank([])
