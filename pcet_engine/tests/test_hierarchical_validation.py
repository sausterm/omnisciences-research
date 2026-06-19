"""Lock the hierarchical held-out predictive result reported in the paper.

The hierarchical model (per-class intercept + shared slope, strict
leave-one-out) must beat both the pooled regression and the ZPE ceiling, and
must reproduce the published headline numbers within tolerance.  These are
regression tests on a deterministic computation.
"""

import math

from pcet_engine.benchmarks.hierarchical_validation import (
    CLASSES,
    NAME2CLASS,
    _fit,
    run,
)
from pcet_engine.benchmarks.systems import BENCHMARK_SYSTEMS


def test_taxonomy_covers_all_28_systems_disjointly():
    members = [n for ns in CLASSES.values() for n in ns]
    assert len(members) == 28
    assert set(members) == set(BENCHMARK_SYSTEMS)
    assert len(set(members)) == 28  # disjoint
    assert min(len(ns) for ns in CLASSES.values()) >= 2  # no singletons


def test_full_data_fit_structure():
    a, b = _fit(list(BENCHMARK_SYSTEMS))
    # single positive shared slope, physically sensible magnitude
    assert 0.10 < b < 0.30
    # class offsets ordered: hydride shortest, iron-oxo longest
    assert a["hydride"] < a["tyrosyl/phenoxyl"] < a["iron-oxo HAT"]


def test_hierarchical_beats_baselines_held_out():
    res = run(verbose=False)
    s = res["summary"]
    hier = s["all28"]["le_hier"]
    pool = s["all28"]["le_pool"]
    zpe = s["all28"]["le_zpe"]
    # headline numbers (deterministic) within tolerance
    assert abs(hier - 0.215) < 0.02
    assert abs(pool - 0.499) < 0.02
    assert abs(zpe - 0.569) < 0.02
    # the central claim: hierarchical beats both baselines on the full set
    assert hier < pool
    assert hier < zpe
    # and on cross-chemistry (non-SLO) it beats the pooled regression
    assert s["nonslo19"]["le_hier"] < s["nonslo19"]["le_pool"]


def test_within_2x_counts():
    res = run(verbose=False)
    rows = res["rows"]
    thr = math.log10(2)
    hier2 = sum(abs(r["le_hier"]) < thr for r in rows)
    pool2 = sum(abs(r["le_pool"]) < thr for r in rows)
    assert hier2 >= 20  # published: 21/28
    assert hier2 > pool2  # published: 21 vs 8
