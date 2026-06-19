"""Hierarchical (per-class intercept + shared slope) held-out predictive test.

Motivation
----------
The pooled LOO regression delta_0 = a + b*d_DA fails across chemistry
(4.1x mean error on the 19 non-SLO-1 systems, worse than the 2.0x ZPE
ceiling) because each mechanistic class sits at a different delta_0 offset
at a given d_DA: the proton-potential shape (single- vs double-well, donor/
acceptor atom identity) differs by class.  The pooled fit mixes these and
cannot transfer.

This script tests whether delta_0 carries *transferable within-class
structure* by fitting a hierarchical (ANCOVA / fixed-effects) model

        delta_0_ij = a_class(i) + b * d_DA_ij

with ONE intercept per mechanistic class and a SINGLE shared slope b
borrowed across all classes.  Prediction uses (class label, d_DA), both
known a priori from structure; no per-system KIE fitting.  Parameters:
5 class intercepts + 1 slope = 6, versus 28 per-system delta_0 values.

Evaluation is strict leave-one-out: each system is held out, the model is
refit on the other 27, delta_0 is predicted for the held-out system, and the
vibronic KIE is computed with that predicted delta_0 and compared to
experiment.  Baselines: classical Marcus (KIE=1), the parameter-free ZPE
ceiling, and the pooled (single-intercept) LOO regression.
"""

from __future__ import annotations

import math

import numpy as np

from pcet_engine.benchmarks.systems import BENCHMARK_SYSTEMS
from pcet_engine.core import PCETRateEngine

# Paper Table 1 mechanistic classes.  The Zn-alcohol hydride singleton (LADH)
# is merged with the C-C hydride class (DHFR, TSase) into one "hydride
# transfer" class to avoid an undefined intercept when held out; this is
# chemically natural (both are hydride transfers) and is the only change from
# the six Table 1 classes.
CLASSES = {
    "iron-oxo HAT": [
        "SLO-1", "SLO-1-L546A", "SLO-1-L754A", "SLO-1-DM",
        "SLO-1-I553G", "SLO-1-I553A", "SLO-1-I553V", "SLO-1-I553L",
        "SLO-1-I553F", "TauD",
    ],
    "Cu HAT": ["PHM", "DβH"],
    "flavoenzyme": ["AADH", "MADH", "CAO", "MAO", "GOx", "MR", "PETNR"],
    "hydride": ["LADH", "DHFR", "TSase"],
    "tyrosyl/phenoxyl": ["RNR", "RNR-3FY", "RNR-2FY", "PhOH-self", "GO", "bc1"],
}
NAME2CLASS = {n: c for c, ns in CLASSES.items() for n in ns}


def _kie_zpe(omega_H_cm: float, T: float = 298.15) -> float:
    """Parameter-free zero-point-energy ceiling KIE (no tunneling)."""
    h = 6.62607015e-34
    c = 2.99792458e10
    kB = 1.380649e-23
    mH, mD = 1.00783, 2.01410
    dzpe = 0.5 * h * c * omega_H_cm * (1.0 - math.sqrt(mH / mD))
    return math.exp(dzpe / (kB * T))


def _fit(train_names, single_intercept=False):
    """Fit delta_0 = a_class + b*d_DA (ANCOVA) on the training names.

    If single_intercept, fit the pooled delta_0 = a + b*d_DA (one intercept)
    -- the published baseline.  Returns (intercepts dict, slope b).
    """
    if single_intercept:
        d = np.array([BENCHMARK_SYSTEMS[n].d_DA for n in train_names])
        y = np.array([BENCHMARK_SYSTEMS[n].delta_0 for n in train_names])
        A = np.column_stack([np.ones_like(d), d])
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        return {"_pooled": float(a)}, float(b)

    classes = sorted({NAME2CLASS[n] for n in train_names})
    cidx = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    for n in train_names:
        row = [0.0] * len(classes) + [BENCHMARK_SYSTEMS[n].d_DA]
        row[cidx[NAME2CLASS[n]]] = 1.0
        X.append(row)
        y.append(BENCHMARK_SYSTEMS[n].delta_0)
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    intercepts = {c: float(coef[cidx[c]]) for c in classes}
    return intercepts, float(coef[-1])


def _predict_d0(name, intercepts, b, single_intercept=False):
    d_DA = BENCHMARK_SYSTEMS[name].d_DA
    a = intercepts["_pooled"] if single_intercept else intercepts[NAME2CLASS[name]]
    return max(a + b * d_DA, 0.05)


def run(temperature: float = 298.15, verbose: bool = True) -> dict:
    engine = PCETRateEngine(temperature=temperature)
    names = list(BENCHMARK_SYSTEMS)
    rows = []
    for held in names:
        train = [n for n in names if n != held]
        sysd = BENCHMARK_SYSTEMS[held]

        a_h, b_h = _fit(train, single_intercept=False)
        d0_hier = _predict_d0(held, a_h, b_h, single_intercept=False)
        kie_hier = engine.compute_rate(
            V_el=sysd.V_el, delta_G=sysd.delta_G, lambda_reorg=sysd.lambda_reorg,
            omega_H=sysd.omega_H, d_DA=sysd.d_DA, method="vibronic_multi",
            delta_0=d0_hier,
        ).KIE

        a_p, b_p = _fit(train, single_intercept=True)
        d0_pool = _predict_d0(held, a_p, b_p, single_intercept=True)
        kie_pool = engine.compute_rate(
            V_el=sysd.V_el, delta_G=sysd.delta_G, lambda_reorg=sysd.lambda_reorg,
            omega_H=sysd.omega_H, d_DA=sysd.d_DA, method="vibronic_multi",
            delta_0=d0_pool,
        ).KIE

        kie_zpe = _kie_zpe(sysd.omega_H, temperature)
        rows.append({
            "name": held, "klass": NAME2CLASS[held], "d_DA": sysd.d_DA,
            "d0_calib": sysd.delta_0, "d0_hier": d0_hier,
            "KIE_exp": sysd.KIE_exp, "KIE_hier": kie_hier,
            "KIE_pool": kie_pool, "KIE_zpe": kie_zpe,
            "le_hier": math.log10(kie_hier / sysd.KIE_exp),
            "le_pool": math.log10(kie_pool / sysd.KIE_exp),
            "le_zpe": math.log10(kie_zpe / sysd.KIE_exp),
        })

    def mae(rs, key):
        return float(np.mean([abs(r[key]) for r in rs]))

    nonslo = [r for r in rows if not r["name"].startswith("SLO-1")]
    summary = {
        "all28": {k: mae(rows, k) for k in ("le_hier", "le_pool", "le_zpe")},
        "nonslo19": {k: mae(nonslo, k) for k in ("le_hier", "le_pool", "le_zpe")},
    }

    if verbose:
        print("=" * 92)
        print("HIERARCHICAL HELD-OUT PREDICTION  (per-class intercept + shared slope)")
        print("  delta_0 = a_class + b*d_DA ; strict leave-one-out ; no per-system KIE fitting")
        print("=" * 92)
        hdr = f"{'System':16s}{'class':18s}{'d_DA':>6s}{'d0cal':>7s}{'d0hier':>7s}" \
              f"{'KIEexp':>8s}{'KIEhier':>9s}{'KIEpool':>9s}{'KIEzpe':>8s}{'|lg|hier':>9s}"
        print(hdr)
        print("-" * 92)
        for r in rows:
            print(f"{r['name']:16s}{r['klass']:18s}{r['d_DA']:6.2f}{r['d0_calib']:7.3f}"
                  f"{r['d0_hier']:7.3f}{r['KIE_exp']:8.1f}{r['KIE_hier']:9.1f}"
                  f"{r['KIE_pool']:9.1f}{r['KIE_zpe']:8.1f}{abs(r['le_hier']):9.2f}")
        print("-" * 92)
        s = summary
        print(f"{'ALL 28':34s} mean|log10|: "
              f"hier={s['all28']['le_hier']:.3f}  pool={s['all28']['le_pool']:.3f}  "
              f"ZPE={s['all28']['le_zpe']:.3f}")
        print(f"{'19 non-SLO-1 (cross-chemistry)':34s} mean|log10|: "
              f"hier={s['nonslo19']['le_hier']:.3f}  pool={s['nonslo19']['le_pool']:.3f}  "
              f"ZPE={s['nonslo19']['le_zpe']:.3f}")
        print(f"  (hier factor = {10**s['nonslo19']['le_hier']:.2f}x ; "
              f"pool = {10**s['nonslo19']['le_pool']:.2f}x ; "
              f"ZPE = {10**s['nonslo19']['le_zpe']:.2f}x)")
        # paired one-sided tests
        from scipy import stats
        ah = np.array([abs(r["le_hier"]) for r in nonslo])
        az = np.array([abs(r["le_zpe"]) for r in nonslo])
        ap = np.array([abs(r["le_pool"]) for r in nonslo])
        t_hz, p_hz = stats.ttest_rel(ah, az)
        t_hp, p_hp = stats.ttest_rel(ah, ap)
        print(f"  paired t (non-SLO): hier vs ZPE  t={t_hz:+.2f} p={p_hz:.3f} (two-sided)")
        print(f"                      hier vs pool t={t_hp:+.2f} p={p_hp:.3f} (two-sided)")
        print("  per-class mean|log10| (hier):")
        for c in CLASSES:
            cr = [r for r in rows if r["klass"] == c]
            print(f"    {c:18s} n={len(cr):2d}  {mae(cr,'le_hier'):.3f} "
                  f"(within 2x: {sum(abs(r['le_hier'])<math.log10(2) for r in cr)}/{len(cr)})")
        within2 = sum(abs(r["le_hier"]) < math.log10(2) for r in rows)
        print(f"  ALL 28 within 2x (hier): {within2}/28 ; "
              f"pool: {sum(abs(r['le_pool'])<math.log10(2) for r in rows)}/28 ; "
              f"ZPE: {sum(abs(r['le_zpe'])<math.log10(2) for r in rows)}/28")
        print("=" * 92)

    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    run(verbose=True)
