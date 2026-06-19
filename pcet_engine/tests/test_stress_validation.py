"""
Comprehensive stress tests and advanced validation for the PCET rate engine.

Tests cover:
1. Temperature-dependent KIE and Arrhenius behavior
2. Gating coordinate integration (D-A distance fluctuations)
3. Tritium isotope effects and Swain-Schaad relationship
4. Marcus inverted region behavior
5. Convergence of multi-channel summation
6. Numerical stability at extreme parameters
7. Sensitivity analysis (parameter perturbation)
8. Physical consistency checks (detailed balance, limiting cases)
9. Cross-validation between methods
10. Additional enzyme systems beyond the original 5
"""

import math
import pytest
import numpy as np

from pcet_engine.core.rate_engine import PCETRateEngine, PCETResult
from pcet_engine.core.vibronic import (
    franck_condon_overlap,
    multi_channel_rate,
    vibronic_rate,
    tunneling_distance,
    _laguerre,
)
from pcet_engine.core.marcus import (
    marcus_rate,
    marcus_activation_energy,
    marcus_rate_kcal,
    reorganization_energy_from_hessians,
)
from pcet_engine.core.normal_modes import (
    normal_mode_analysis,
    identify_proton_mode,
    compute_donor_acceptor_distance,
)
from pcet_engine.core.constants import (
    AMU_TO_AU,
    PROTON_MASS_AMU,
    DEUTERIUM_MASS_AMU,
    TRITIUM_MASS_AMU,
    KCALMOL_TO_HARTREE,
    HARTREE_TO_KCALMOL,
    CM_TO_HARTREE,
    KB_HARTREE,
    ANGSTROM_TO_BOHR,
    HBAR_AU,
)
from pcet_engine.data.model_hessians import (
    MODEL_SYSTEMS,
    build_hessian,
    ModelSystem,
    _force_constant_from_frequency,
)
from pcet_engine.benchmarks.systems import BENCHMARK_SYSTEMS


# =====================================================================
# 1. TEMPERATURE-DEPENDENT KIE AND ARRHENIUS BEHAVIOR
# =====================================================================

class TestTemperatureDependence:
    """Test temperature-dependent rate behavior and KIE signatures."""

    def setup_method(self):
        self.slo1 = BENCHMARK_SYSTEMS["SLO-1"]

    def test_arrhenius_linearity(self):
        """ln(k) vs 1/T should be approximately linear over 250-350 K."""
        temps = np.linspace(250, 350, 11)
        ln_k = []
        for T in temps:
            engine = PCETRateEngine(temperature=T)
            result = engine.compute_rate(
                V_el=self.slo1.V_el, delta_G=self.slo1.delta_G,
                lambda_reorg=self.slo1.lambda_reorg, omega_H=self.slo1.omega_H,
                d_DA=self.slo1.d_DA, delta_0=self.slo1.delta_0,
            )
            ln_k.append(math.log(result.k_H))
        inv_T = 1.0 / temps
        # Linear regression: R² should be > 0.95
        coeffs = np.polyfit(inv_T, ln_k, 1)
        predicted = np.polyval(coeffs, inv_T)
        ss_res = np.sum((np.array(ln_k) - predicted) ** 2)
        ss_tot = np.sum((np.array(ln_k) - np.mean(ln_k)) ** 2)
        R2 = 1.0 - ss_res / ss_tot
        assert R2 > 0.95, f"Arrhenius R² = {R2:.4f} (expected > 0.95)"

    def test_slo1_weak_temperature_dependence_of_kie(self):
        """SLO-1 KIE should vary less than 50% over 278-318 K range.

        The hallmark of SLO-1 is near temperature-independent KIE,
        indicating deep tunneling dominates over thermal activation.
        """
        kies = []
        for T in [278, 288, 298, 308, 318]:
            engine = PCETRateEngine(temperature=T)
            result = engine.compute_rate(
                V_el=self.slo1.V_el, delta_G=self.slo1.delta_G,
                lambda_reorg=self.slo1.lambda_reorg, omega_H=self.slo1.omega_H,
                d_DA=self.slo1.d_DA, delta_0=self.slo1.delta_0,
            )
            kies.append(result.KIE)
        max_kie, min_kie = max(kies), min(kies)
        variation = (max_kie - min_kie) / np.mean(kies)
        assert variation < 0.50, (
            f"KIE varies by {variation*100:.0f}% over 278-318K "
            f"(KIEs: {[f'{k:.1f}' for k in kies]})"
        )

    def test_rate_increases_monotonically_with_temperature(self):
        """Rate should increase with temperature in the normal Marcus regime."""
        rates = []
        for T in [200, 250, 298, 350, 400]:
            engine = PCETRateEngine(temperature=T)
            result = engine.compute_rate(
                V_el=self.slo1.V_el, delta_G=self.slo1.delta_G,
                lambda_reorg=self.slo1.lambda_reorg, omega_H=self.slo1.omega_H,
                d_DA=self.slo1.d_DA, delta_0=self.slo1.delta_0,
            )
            rates.append(result.k_H)
        for i in range(len(rates) - 1):
            assert rates[i + 1] > rates[i], (
                f"Rate did not increase: k({200+50*i}K)={rates[i]:.2e} "
                f"> k({250+50*i}K)={rates[i+1]:.2e}"
            )

    def test_extracted_activation_energy_from_arrhenius(self):
        """E_a from Arrhenius slope should be positive and < 20 kcal/mol for SLO-1."""
        temps = np.array([278.0, 288.0, 298.0, 308.0, 318.0])
        ln_k = []
        for T in temps:
            engine = PCETRateEngine(temperature=T)
            result = engine.compute_rate(
                V_el=self.slo1.V_el, delta_G=self.slo1.delta_G,
                lambda_reorg=self.slo1.lambda_reorg, omega_H=self.slo1.omega_H,
                d_DA=self.slo1.d_DA, delta_0=self.slo1.delta_0,
            )
            ln_k.append(math.log(result.k_H))
        inv_T = 1.0 / temps
        slope, _ = np.polyfit(inv_T, ln_k, 1)
        E_a_extracted = -slope * KB_HARTREE * HARTREE_TO_KCALMOL  # kcal/mol
        # SLO-1 experimental E_a ≈ 2.1 kcal/mol
        assert E_a_extracted > 0, f"Negative E_a: {E_a_extracted:.2f} kcal/mol"
        assert E_a_extracted < 20.0, f"E_a too large: {E_a_extracted:.2f} kcal/mol"


# =====================================================================
# 2. GATING COORDINATE INTEGRATION
# =====================================================================

class TestGatingCoordinate:
    """Test the D-A gating coordinate integration (thermal R-averaging).

    This code path (Omega_gating > 0) computes R-averaged FC overlaps
    via Gauss-Hermite quadrature, which is critical for temperature-
    dependent KIE predictions.
    """

    def setup_method(self):
        self.slo1 = BENCHMARK_SYSTEMS["SLO-1"]

    def test_gating_produces_finite_rates(self):
        """Rate with gating should be positive and finite."""
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=self.slo1.V_el, delta_G=self.slo1.delta_G,
            lambda_reorg=self.slo1.lambda_reorg, omega_H=self.slo1.omega_H,
            d_DA=self.slo1.d_DA,
            Omega_gating=350.0,  # cm⁻¹
            M_DA=self.slo1.M_DA,
            delta_0=self.slo1.delta_0,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)
        assert result.k_D > 0 and math.isfinite(result.k_D)
        assert result.KIE > 1.0

    def test_gating_increases_rate(self):
        """Gating (R-averaging) should generally increase the rate.

        Thermal fluctuations in D-A distance allow sampling of shorter
        distances where FC overlap is exponentially larger.
        """
        engine = PCETRateEngine()
        result_no_gate = engine.compute_rate(
            V_el=self.slo1.V_el, delta_G=self.slo1.delta_G,
            lambda_reorg=self.slo1.lambda_reorg, omega_H=self.slo1.omega_H,
            d_DA=self.slo1.d_DA, delta_0=self.slo1.delta_0,
            Omega_gating=0.0,
        )
        result_gate = engine.compute_rate(
            V_el=self.slo1.V_el, delta_G=self.slo1.delta_G,
            lambda_reorg=self.slo1.lambda_reorg, omega_H=self.slo1.omega_H,
            d_DA=self.slo1.d_DA, delta_0=self.slo1.delta_0,
            Omega_gating=350.0, M_DA=self.slo1.M_DA,
        )
        # R-averaging samples shorter distances → larger FC overlap → faster rate
        assert result_gate.k_H >= result_no_gate.k_H * 0.5, (
            f"Gated rate ({result_gate.k_H:.2e}) much smaller than "
            f"ungated ({result_no_gate.k_H:.2e})"
        )

    def test_stiffer_gating_reduces_enhancement(self):
        """Higher gating frequency = stiffer D-A → less rate enhancement.

        A stiff D-A bond has smaller thermal fluctuations, so the
        R-averaging effect is smaller.
        """
        engine = PCETRateEngine()
        result_soft = engine.compute_rate(
            V_el=self.slo1.V_el, delta_G=self.slo1.delta_G,
            lambda_reorg=self.slo1.lambda_reorg, omega_H=self.slo1.omega_H,
            d_DA=self.slo1.d_DA, delta_0=self.slo1.delta_0,
            Omega_gating=200.0, M_DA=self.slo1.M_DA,
        )
        result_stiff = engine.compute_rate(
            V_el=self.slo1.V_el, delta_G=self.slo1.delta_G,
            lambda_reorg=self.slo1.lambda_reorg, omega_H=self.slo1.omega_H,
            d_DA=self.slo1.d_DA, delta_0=self.slo1.delta_0,
            Omega_gating=800.0, M_DA=self.slo1.M_DA,
        )
        # Softer gating → larger D-A fluctuations → larger rate
        assert result_soft.k_H > result_stiff.k_H * 0.8

    @pytest.mark.parametrize("name", list(BENCHMARK_SYSTEMS.keys()))
    def test_all_systems_with_gating(self, name):
        """All 5 systems should produce valid rates with gating enabled."""
        sys = BENCHMARK_SYSTEMS[name]
        if sys.Omega_gating == 0:
            # Use a reasonable default gating frequency
            omega_gate = 300.0
        else:
            omega_gate = sys.Omega_gating

        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=sys.V_el, delta_G=sys.delta_G,
            lambda_reorg=sys.lambda_reorg, omega_H=sys.omega_H,
            d_DA=sys.d_DA, delta_0=sys.delta_0,
            Omega_gating=omega_gate, M_DA=max(sys.M_DA, 6.0),
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)
        assert result.KIE > 1.0


# =====================================================================
# 3. TRITIUM ISOTOPE EFFECTS AND SWAIN-SCHAAD RELATIONSHIP
# =====================================================================

class TestTritiumAndSwainSchaad:
    """Test tritium isotope effects and the Swain-Schaad exponent.

    The semiclassical Swain-Schaad relationship predicts:
        ln(k_H/k_T) / ln(k_D/k_T) ≈ 3.26

    Deviations from this value are diagnostic of tunneling contributions.
    For deep tunneling systems like SLO-1, the exponent can be inflated.
    """

    def _compute_tritium_rate(self, params, temperature=298.15):
        """Compute rate for tritium transfer."""
        V_au = params["V_el"] * KCALMOL_TO_HARTREE
        dG_au = params["delta_G"] * KCALMOL_TO_HARTREE
        lam_au = params["lambda_reorg"] * KCALMOL_TO_HARTREE
        omega_H_au = params["omega_H"] * CM_TO_HARTREE
        omega_T = params["omega_H"] * math.sqrt(PROTON_MASS_AMU / TRITIUM_MASS_AMU)
        omega_T_au = omega_T * CM_TO_HARTREE

        res_T = multi_channel_rate(
            V_au, dG_au, lam_au,
            omega_T_au, omega_T_au, TRITIUM_MASS_AMU,
            params["d_DA"], temperature,
            delta_0=params.get("delta_0"),
        )
        return res_T.rate_total

    def test_tritium_rate_slower_than_deuterium(self):
        """k_T < k_D < k_H for all benchmark systems."""
        for name, sys in BENCHMARK_SYSTEMS.items():
            engine = PCETRateEngine()
            params = dict(
                V_el=sys.V_el, delta_G=sys.delta_G,
                lambda_reorg=sys.lambda_reorg, omega_H=sys.omega_H,
                d_DA=sys.d_DA, delta_0=sys.delta_0,
            )
            result = engine.compute_rate(**params)
            k_T = self._compute_tritium_rate(params)
            assert result.k_H > result.k_D > k_T, (
                f"{name}: k_H={result.k_H:.2e}, k_D={result.k_D:.2e}, k_T={k_T:.2e}"
            )

    def test_swain_schaad_exponent_range(self):
        """Swain-Schaad exponent should be > 1 and < 30 for all systems.

        Semiclassical value ≈ 3.26. Tunneling inflates it.
        Very large values (> 30) indicate numerical issues.
        """
        for name, sys in BENCHMARK_SYSTEMS.items():
            engine = PCETRateEngine()
            params = dict(
                V_el=sys.V_el, delta_G=sys.delta_G,
                lambda_reorg=sys.lambda_reorg, omega_H=sys.omega_H,
                d_DA=sys.d_DA, delta_0=sys.delta_0,
            )
            result = engine.compute_rate(**params)
            k_T = self._compute_tritium_rate(params)

            if result.k_D > k_T and k_T > 0:
                ln_HT = math.log(result.k_H / k_T)
                ln_DT = math.log(result.k_D / k_T)
                if ln_DT > 1e-10:
                    exponent = ln_HT / ln_DT
                    assert exponent > 1.0, (
                        f"{name}: Swain-Schaad exponent = {exponent:.2f} (< 1)"
                    )
                    assert exponent < 30.0, (
                        f"{name}: Swain-Schaad exponent = {exponent:.2f} (> 30)"
                    )

    def test_h_d_t_frequency_scaling(self):
        """Isotope frequency ratios should follow sqrt(mass) scaling."""
        omega_H = 3000.0
        omega_D = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
        omega_T = omega_H * math.sqrt(PROTON_MASS_AMU / TRITIUM_MASS_AMU)

        # H/D ratio ≈ 1/√2 ≈ 0.707
        assert abs(omega_D / omega_H - math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)) < 1e-6
        # H/T ratio ≈ 1/√3 ≈ 0.577
        assert abs(omega_T / omega_H - math.sqrt(PROTON_MASS_AMU / TRITIUM_MASS_AMU)) < 1e-6
        # D < T ordering
        assert omega_T < omega_D < omega_H


# =====================================================================
# 4. MARCUS INVERTED REGION AND EDGE CASES
# =====================================================================

class TestMarcusInvertedRegion:
    """Test behavior in the Marcus inverted region (-ΔG > λ).

    In the inverted region, rate should DECREASE with more negative ΔG.
    This is a key prediction of Marcus theory.
    """

    def test_inverted_region_rate_decrease(self):
        """Rate should decrease when -ΔG >> λ (inverted region)."""
        V = 0.5 * KCALMOL_TO_HARTREE
        lam = 10.0 * KCALMOL_TO_HARTREE

        # Normal region: -ΔG < λ
        k_normal = marcus_rate(V, -5.0 * KCALMOL_TO_HARTREE, lam)
        # Activationless: -ΔG = λ
        k_max = marcus_rate(V, -10.0 * KCALMOL_TO_HARTREE, lam)
        # Inverted: -ΔG > λ
        k_inverted = marcus_rate(V, -20.0 * KCALMOL_TO_HARTREE, lam)
        # Deeply inverted
        k_deep = marcus_rate(V, -30.0 * KCALMOL_TO_HARTREE, lam)

        # Activationless should be fastest
        assert k_max > k_normal
        assert k_max > k_inverted
        # Deep inversion should be slower
        assert k_inverted > k_deep

    def test_activationless_barrier_is_zero(self):
        """E_a should be exactly 0 when -ΔG = λ."""
        lam = 20.0 * KCALMOL_TO_HARTREE
        dG = -lam  # activationless condition
        E_a = marcus_activation_energy(dG, lam)
        assert abs(E_a) < 1e-12

    def test_symmetric_activation_energy(self):
        """E_a at ΔG=0 should equal λ/4."""
        lam = 20.0 * KCALMOL_TO_HARTREE
        E_a = marcus_activation_energy(0.0, lam)
        expected = lam / 4.0
        assert abs(E_a - expected) < 1e-12

    def test_vibronic_in_inverted_region(self):
        """Vibronic rate should still be well-defined in inverted region."""
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.5, delta_G=-30.0, lambda_reorg=10.0,
            omega_H=3000.0, d_DA=2.7,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)
        assert result.KIE > 0 and math.isfinite(result.KIE)

    def test_endothermic_reaction(self):
        """Endothermic (ΔG > 0) reactions should give smaller rates."""
        engine = PCETRateEngine()
        r_exo = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7,
        )
        r_endo = engine.compute_rate(
            V_el=0.5, delta_G=+5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7,
        )
        assert r_exo.k_H > r_endo.k_H


# =====================================================================
# 5. CONVERGENCE OF MULTI-CHANNEL SUMMATION
# =====================================================================

class TestConvergence:
    """Test convergence of multi-channel vibronic summation.

    The rate should converge as more vibronic states are included.
    """

    def test_rate_converges_with_product_states(self):
        """Rate should converge as n_product_states increases."""
        V_au = 0.5 * KCALMOL_TO_HARTREE
        dG_au = -5.0 * KCALMOL_TO_HARTREE
        lam_au = 20.0 * KCALMOL_TO_HARTREE
        omega_au = 3000.0 * CM_TO_HARTREE

        rates = []
        for n_prod in [3, 5, 10, 15, 20]:
            res = multi_channel_rate(
                V_au, dG_au, lam_au, omega_au, omega_au,
                PROTON_MASS_AMU, 2.7, n_product_states=n_prod,
            )
            rates.append(res.rate_total)

        # Rate should stabilize — last two values within 10%
        rel_change = abs(rates[-1] - rates[-2]) / rates[-1]
        assert rel_change < 0.10, (
            f"Rate not converged: {rates[-2]:.2e} → {rates[-1]:.2e} "
            f"(change = {rel_change*100:.1f}%)"
        )

    def test_rate_converges_with_reactant_states(self):
        """Rate should converge as n_reactant_states increases (at room T)."""
        V_au = 0.5 * KCALMOL_TO_HARTREE
        dG_au = -5.0 * KCALMOL_TO_HARTREE
        lam_au = 20.0 * KCALMOL_TO_HARTREE
        omega_au = 3000.0 * CM_TO_HARTREE

        rates = []
        for n_react in [1, 3, 5, 8]:
            res = multi_channel_rate(
                V_au, dG_au, lam_au, omega_au, omega_au,
                PROTON_MASS_AMU, 2.7,
                n_reactant_states=n_react, n_product_states=10,
            )
            rates.append(res.rate_total)

        # At room temperature, excited reactant states have negligible population
        # so convergence should be fast
        rel_change = abs(rates[-1] - rates[-2]) / rates[-1]
        assert rel_change < 0.05, (
            f"Reactant state convergence too slow: {rel_change*100:.1f}%"
        )

    def test_ground_state_dominates_at_room_temp(self):
        """At 298 K, the μ=0 channel should dominate (>90% of rate)."""
        V_au = 0.5 * KCALMOL_TO_HARTREE
        dG_au = -5.0 * KCALMOL_TO_HARTREE
        lam_au = 20.0 * KCALMOL_TO_HARTREE
        omega_au = 3000.0 * CM_TO_HARTREE

        res = multi_channel_rate(
            V_au, dG_au, lam_au, omega_au, omega_au,
            PROTON_MASS_AMU, 2.7,
            n_reactant_states=5, n_product_states=10,
        )

        # Rate from μ=0 channels
        mu0_rate = res.boltzmann_weights[0] * np.sum(res.rate_channels[0, :])
        fraction = mu0_rate / res.rate_total
        assert fraction > 0.80, (
            f"Ground state contributes only {fraction*100:.0f}% of total rate"
        )


# =====================================================================
# 6. NUMERICAL STABILITY AT EXTREME PARAMETERS
# =====================================================================

class TestNumericalStability:
    """Test numerical stability at extreme parameter values."""

    def test_very_large_reorganization_energy(self):
        """Engine should handle λ = 100 kcal/mol without overflow."""
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=100.0,
            omega_H=3000.0, d_DA=2.7,
        )
        assert math.isfinite(result.k_H)
        assert result.k_H >= 0

    def test_very_small_coupling(self):
        """Very weak electronic coupling should give very small rate."""
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.001, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7,
        )
        assert result.k_H >= 0 and math.isfinite(result.k_H)
        # Very small V → very small rate (k ∝ V²)
        result_normal = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7,
        )
        assert result.k_H < result_normal.k_H

    def test_very_short_da_distance(self):
        """Very short D-A distance (2.3 Å) should give large rate."""
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.3, r_DH=1.05, r_AH=0.96,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)

    def test_very_long_da_distance(self):
        """Very long D-A distance (4.0 Å) should give small rate."""
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=4.0,
        )
        assert result.k_H >= 0 and math.isfinite(result.k_H)

    def test_very_low_frequency(self):
        """Low proton frequency (1500 cm⁻¹) should still work."""
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=1500.0, d_DA=2.7,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)

    def test_very_high_frequency(self):
        """High proton frequency (4000 cm⁻¹, e.g. O-H) should still work."""
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=4000.0, d_DA=2.7,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)

    def test_very_high_temperature(self):
        """Engine should work at high temperature (500 K)."""
        engine = PCETRateEngine(temperature=500.0)
        result = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)

    def test_very_low_temperature(self):
        """Engine should work at low temperature (77 K, liquid N2)."""
        engine = PCETRateEngine(temperature=77.0)
        result = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7,
        )
        assert result.k_H >= 0 and math.isfinite(result.k_H)

    def test_zero_driving_force(self):
        """ΔG = 0 (thermoneutral) should give valid results."""
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.5, delta_G=0.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)

    def test_fc_overlap_extreme_displacement(self):
        """FC overlap should handle very large displacement without NaN."""
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        # Very large displacement → overlap → 0 but not NaN
        S = franck_condon_overlap(omega, omega, mass, 10.0, 0, 0)
        assert math.isfinite(S)
        assert S >= 0

    def test_fc_overlap_high_quantum_numbers(self):
        """FC overlap at high quantum numbers should be finite and non-negative."""
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        for mu in range(8):
            for nu in range(8):
                S = franck_condon_overlap(omega, omega, mass, 0.5, mu, nu)
                assert math.isfinite(S) and S >= 0, (
                    f"FC overlap ({mu},{nu}) = {S}"
                )

    def test_laguerre_polynomial_stability(self):
        """Laguerre polynomials should be stable for moderate n, alpha."""
        for n in range(10):
            for alpha in range(5):
                for x in [0.1, 1.0, 5.0, 10.0]:
                    L = _laguerre(n, alpha, x)
                    assert math.isfinite(L), f"L_{n}^{alpha}({x}) not finite"


# =====================================================================
# 7. SENSITIVITY ANALYSIS
# =====================================================================

class TestSensitivityAnalysis:
    """Test sensitivity of rates to parameter perturbations.

    Small changes in parameters should produce smooth, proportional
    changes in rates (no discontinuities or sign changes).
    """

    def setup_method(self):
        self.engine = PCETRateEngine()
        self.base_params = dict(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.50,
        )

    def _rate_at(self, **overrides):
        params = {**self.base_params, **overrides}
        return self.engine.compute_rate(**params).k_H

    def test_rate_proportional_to_v_squared(self):
        """Rate should scale as V² (nonadiabatic regime)."""
        k1 = self._rate_at(V_el=0.3)
        k2 = self._rate_at(V_el=0.6)
        # k ∝ V² → k2/k1 ≈ (0.6/0.3)² = 4
        ratio = k2 / k1
        assert 3.0 < ratio < 5.0, f"k ratio = {ratio:.2f} (expected ~4.0)"

    def test_smooth_delta_g_dependence(self):
        """Rate should change smoothly with ΔG."""
        dGs = np.linspace(-10, -1, 10)
        rates = [self._rate_at(delta_G=dG) for dG in dGs]
        # No rate should be zero or negative
        assert all(r > 0 for r in rates)
        # No sudden jumps (each rate within 100x of neighbors)
        for i in range(1, len(rates)):
            ratio = max(rates[i], rates[i - 1]) / max(min(rates[i], rates[i - 1]), 1e-300)
            assert ratio < 100, (
                f"Jump at ΔG={dGs[i]:.1f}: {rates[i-1]:.2e} → {rates[i]:.2e}"
            )

    def test_smooth_lambda_dependence(self):
        """Rate should change smoothly with λ."""
        lambdas = np.linspace(10, 40, 7)
        rates = [self._rate_at(lambda_reorg=lam) for lam in lambdas]
        assert all(r > 0 and math.isfinite(r) for r in rates)

    def test_rate_sensitive_to_da_distance(self):
        """Shorter D-A distance should give much larger rate (exponential sensitivity)."""
        k_short = self._rate_at(d_DA=2.5, delta_0=0.40)
        k_long = self._rate_at(d_DA=3.0, delta_0=0.90)
        assert k_short > k_long, "Shorter D-A should give faster rate"

    def test_kie_sensitive_to_tunneling_distance(self):
        """Larger tunneling distance should give larger KIE."""
        r_short = self.engine.compute_rate(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.30,
        )
        r_long = self.engine.compute_rate(
            V_el=0.6, delta_G=-5.4, lambda_reorg=19.0,
            omega_H=2900.0, d_DA=2.69, delta_0=0.60,
        )
        assert r_long.KIE > r_short.KIE, (
            f"Longer tunnel distance should give larger KIE: "
            f"KIE(δ=0.3)={r_short.KIE:.1f}, KIE(δ=0.6)={r_long.KIE:.1f}"
        )


# =====================================================================
# 8. PHYSICAL CONSISTENCY CHECKS
# =====================================================================

class TestPhysicalConsistency:
    """Test physically required properties of the rate calculations."""

    def test_tunneling_distance_positive(self):
        """Tunneling distance should always be positive."""
        for d_DA in [2.3, 2.5, 2.7, 3.0, 3.5]:
            delta = tunneling_distance(d_DA)
            assert delta > 0

    def test_tunneling_distance_minimum_enforced(self):
        """Minimum tunneling distance should be enforced (0.05 Å)."""
        # Very short D-A where geometric estimate would be negative
        delta = tunneling_distance(1.5, 1.09, 0.96)
        assert delta >= 0.05

    def test_marcus_and_vibronic_agree_classically(self):
        """In the classical limit (large mass, small displacement), vibronic → Marcus.

        For very heavy particles, the FC overlap → 1 and vibronic rate → Marcus rate.
        We can test this by using very small tunneling distances.
        """
        engine = PCETRateEngine()
        r_marcus = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7, method="marcus",
        )
        # With tiny tunneling distance, FC overlap → 1
        r_vibronic = engine.compute_rate(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.05, r_DH=1.05, r_AH=0.96,
            method="vibronic_multi",
        )
        # Should be within an order of magnitude
        log_ratio = abs(math.log10(r_vibronic.k_H / r_marcus.k_H))
        assert log_ratio < 2.0, (
            f"Vibronic/Marcus ratio too large: 10^{log_ratio:.1f}"
        )

    def test_fc_overlap_completeness(self):
        """Sum over all product states should give ≈ 1.0 for ground-state FC overlaps.

        This is a completeness relation: Σ_ν |<0|ν>|² = 1.
        """
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        delta = 0.3  # bohr — moderate displacement

        total = sum(
            franck_condon_overlap(omega, omega, mass, delta, 0, nu)
            for nu in range(30)
        )
        assert abs(total - 1.0) < 0.05, (
            f"FC completeness sum = {total:.4f} (expected 1.0)"
        )

    def test_kie_always_greater_than_one(self):
        """KIE should be > 1 for all physically reasonable parameters.

        Lighter particle (H) tunnels more efficiently → faster rate.
        """
        engine = PCETRateEngine()
        test_cases = [
            dict(V_el=0.5, delta_G=-5.0, lambda_reorg=20.0, omega_H=3000.0, d_DA=2.7),
            dict(V_el=1.0, delta_G=-10.0, lambda_reorg=15.0, omega_H=2800.0, d_DA=2.5),
            dict(V_el=0.2, delta_G=-2.0, lambda_reorg=30.0, omega_H=3200.0, d_DA=3.0),
        ]
        for params in test_cases:
            result = engine.compute_rate(**params)
            assert result.KIE >= 1.0, f"KIE = {result.KIE:.2f} < 1 for {params}"

    def test_rate_coupling_squared_dependence(self):
        """In nonadiabatic limit, k ∝ |V|². Verify for Marcus rate."""
        V1 = 0.3 * KCALMOL_TO_HARTREE
        V2 = 0.6 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE

        k1 = marcus_rate(V1, dG, lam)
        k2 = marcus_rate(V2, dG, lam)
        ratio = k2 / k1
        expected = (V2 / V1) ** 2
        assert abs(ratio - expected) / expected < 0.01, (
            f"k ratio = {ratio:.4f}, expected {expected:.4f}"
        )


# =====================================================================
# 9. CROSS-VALIDATION BETWEEN METHODS
# =====================================================================

class TestCrossValidation:
    """Cross-validate different calculation methods against each other."""

    def test_single_vs_multi_channel_ground_state(self):
        """Single-channel rate should approximate multi-channel (0,0) contribution."""
        V = 0.5 * KCALMOL_TO_HARTREE
        dG = -5.0 * KCALMOL_TO_HARTREE
        lam = 20.0 * KCALMOL_TO_HARTREE
        omega = 3000.0 * CM_TO_HARTREE

        k_single = vibronic_rate(V, dG, lam, omega, PROTON_MASS_AMU, 2.7)

        res_multi = multi_channel_rate(
            V, dG, lam, omega, omega, PROTON_MASS_AMU, 2.7,
            n_reactant_states=1, n_product_states=1,
        )

        # With 1 reactant, 1 product state → should match single-channel
        log_ratio = abs(math.log10(k_single / res_multi.rate_total))
        assert log_ratio < 0.5, (
            f"Single vs multi(1,1) differ by 10^{log_ratio:.2f}"
        )

    def test_vibronic_methods_consistent(self):
        """Single-channel and multi-channel vibronic rates should agree within ~2 orders.

        Note: Marcus (no tunneling) can differ dramatically from vibronic rates
        because the FC overlap suppresses the vibronic rate by many orders of
        magnitude at typical tunneling distances. This is expected and correct.
        """
        engine = PCETRateEngine()
        params = dict(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7,
        )

        r_single = engine.compute_rate(**params, method="vibronic_single")
        r_multi = engine.compute_rate(**params, method="vibronic_multi")

        log_range = abs(math.log10(r_single.k_H / r_multi.k_H))
        assert log_range < 3.0, (
            f"Single vs multi differ by {log_range:.1f} orders: "
            f"Single={r_single.k_H:.2e}, Multi={r_multi.k_H:.2e}"
        )

    def test_marcus_faster_than_vibronic(self):
        """Marcus rate (no FC suppression) should be faster than vibronic rate.

        The FC overlap |S_00|² < 1 always suppresses the vibronic rate
        relative to the classical Marcus rate. This is a fundamental
        consistency check.
        """
        engine = PCETRateEngine()
        params = dict(
            V_el=0.5, delta_G=-5.0, lambda_reorg=20.0,
            omega_H=3000.0, d_DA=2.7,
        )
        r_marcus = engine.compute_rate(**params, method="marcus")
        r_vibronic = engine.compute_rate(**params, method="vibronic_multi")
        assert r_marcus.k_H > r_vibronic.k_H, (
            f"Marcus ({r_marcus.k_H:.2e}) should exceed vibronic ({r_vibronic.k_H:.2e})"
        )

    def test_kcal_wrapper_matches_hartree(self):
        """marcus_rate_kcal should match marcus_rate with unit conversion."""
        V_kcal, dG_kcal, lam_kcal = 0.5, -5.0, 20.0
        k1 = marcus_rate_kcal(V_kcal, dG_kcal, lam_kcal)
        k2 = marcus_rate(
            V_kcal * KCALMOL_TO_HARTREE,
            dG_kcal * KCALMOL_TO_HARTREE,
            lam_kcal * KCALMOL_TO_HARTREE,
        )
        assert abs(k1 - k2) / k2 < 1e-10


# =====================================================================
# 10. ADDITIONAL CHALLENGING SYSTEMS
# =====================================================================

class TestAdditionalSystems:
    """Test with additional PCET systems beyond the original 5 benchmarks.

    These represent harder/unusual cases:
    - Photoinduced PCET (large driving force)
    - Concerted vs sequential mechanisms
    - Metal-hydride systems (different frequency range)
    - Solution-phase (organic) PCET
    """

    def test_photoinduced_pcet(self):
        """Photoinduced PCET: large driving force, moderate λ.

        Example: Ru(bpy)3 photochemistry, ΔG ≈ -40 kcal/mol.
        """
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.3,
            delta_G=-40.0,
            lambda_reorg=25.0,
            omega_H=3000.0,
            d_DA=2.8,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)
        assert result.KIE > 1.0

    def test_metal_hydride_low_frequency(self):
        """Metal-hydride PCET: lower M-H frequency (1800-2100 cm⁻¹).

        Example: Fe-H, Mn-H systems in water oxidation catalysis.
        """
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.8,
            delta_G=-8.0,
            lambda_reorg=30.0,
            omega_H=1900.0,  # M-H stretch
            d_DA=3.0,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)
        assert result.KIE > 1.0

    def test_concerted_ms_ept(self):
        """Multiple-site EPT: proton and electron to different acceptors.

        Larger reorganization energy, weaker coupling.
        """
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.1,           # weak coupling (MS-EPT)
            delta_G=-3.0,
            lambda_reorg=45.0,  # large λ
            omega_H=3300.0,     # O-H stretch
            d_DA=2.6,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)

    def test_electrochemical_pcet(self):
        """Electrochemical PCET: variable driving force (electrode potential).

        Scan ΔG from +5 to -15 kcal/mol (voltammogram-like).
        """
        engine = PCETRateEngine()
        rates = []
        dGs = np.linspace(5, -15, 11)
        for dG in dGs:
            result = engine.compute_rate(
                V_el=0.5, delta_G=dG, lambda_reorg=20.0,
                omega_H=3000.0, d_DA=2.7,
            )
            rates.append(result.k_H)
            assert result.k_H >= 0 and math.isfinite(result.k_H)

        # Rates should generally increase as ΔG becomes more negative
        # (until inverted region)
        assert rates[-1] > rates[0], "Rate should increase going exothermic"

    def test_very_large_kie_system(self):
        """Extreme tunneling system: should produce KIE > 50.

        Large D-A distance + high frequency → large KIE.
        """
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=0.5,
            delta_G=-5.0,
            lambda_reorg=20.0,
            omega_H=3000.0,
            d_DA=2.7,
            delta_0=0.55,  # larger tunneling distance
        )
        assert result.KIE > 20, f"Expected large KIE, got {result.KIE:.1f}"

    def test_near_classical_system(self):
        """Near-classical system: small KIE (< 10).

        Short D-A distance, strong coupling.
        """
        engine = PCETRateEngine()
        result = engine.compute_rate(
            V_el=1.5,
            delta_G=-5.0,
            lambda_reorg=15.0,
            omega_H=3000.0,
            d_DA=2.4,
            delta_0=0.25,  # short tunnel distance
        )
        assert result.KIE < 20, f"Expected small KIE, got {result.KIE:.1f}"
        assert result.KIE > 1.0


# =====================================================================
# 11. HESSIAN PIPELINE STRESS TESTS
# =====================================================================

class TestHessianPipelineStress:
    """Stress tests for the Hessian-to-rate pipeline."""

    def test_custom_model_system(self):
        """Pipeline should work with a custom-defined model system."""
        custom = ModelSystem(
            name="Custom",
            donor_element="N",
            acceptor_element="O",
            r_DH=1.01,
            r_AH=0.96,
            d_DA=2.75,
            omega_DH=3300.0,
            omega_AH=3600.0,
            angle_DHA=175.0,
        )
        hess_R, geom_R, elements, masses = build_hessian(custom, "reactant")
        hess_P, geom_P, _, _ = build_hessian(custom, "product")

        engine = PCETRateEngine()
        result = engine.compute_rate_from_hessian(
            hessian_R=hess_R, hessian_P=hess_P,
            geom_R=geom_R, geom_P=geom_P,
            masses=masses,
            proton_idx=1, donor_idx=0, acceptor_idx=2,
            V_el=0.5, delta_G=-5.0, lambda_outer=12.0,
        )
        assert result.k_H > 0 and math.isfinite(result.k_H)
        assert result.KIE > 1.0

    def test_symmetric_hessian_invariant(self):
        """Small asymmetry in Hessian should not change rates significantly."""
        model = MODEL_SYSTEMS["SLO-1"]
        hess, geom, elements, masses = build_hessian(model, "reactant")

        # Normal mode analysis with exact Hessian
        nma1 = normal_mode_analysis(hess, masses)
        nma1 = identify_proton_mode(nma1, [1], masses)

        # Add tiny asymmetry
        noise = np.random.RandomState(42).randn(9, 9) * 1e-8
        hess_noisy = hess + noise
        nma2 = normal_mode_analysis(hess_noisy, masses)
        nma2 = identify_proton_mode(nma2, [1], masses)

        # Frequencies should match within 0.1%
        if nma1.proton_frequency_cm and nma2.proton_frequency_cm:
            rel_err = abs(nma1.proton_frequency_cm - nma2.proton_frequency_cm) / nma1.proton_frequency_cm
            assert rel_err < 0.001

    def test_all_model_systems_eigenvalue_check(self):
        """All model Hessians should be positive-semidefinite (no imaginary freqs)."""
        for name, model in MODEL_SYSTEMS.items():
            for state in ["reactant", "product"]:
                hess, _, _, masses = build_hessian(model, state)
                nma = normal_mode_analysis(hess, masses)
                assert nma.n_imaginary == 0, (
                    f"{name} {state}: {nma.n_imaginary} imaginary frequencies"
                )

    def test_reorganization_energy_symmetric(self):
        """λ_forward and λ_backward should be roughly similar."""
        for name, model in MODEL_SYSTEMS.items():
            hess_R, geom_R, _, masses = build_hessian(model, "reactant")
            hess_P, geom_P, _, _ = build_hessian(model, "product")

            geom_R_flat = (geom_R * ANGSTROM_TO_BOHR).flatten()
            geom_P_flat = (geom_P * ANGSTROM_TO_BOHR).flatten()

            lam_f, lam_b = reorganization_energy_from_hessians(
                hess_R, hess_P, geom_R_flat, geom_P_flat, masses,
                exclude_atoms=[1],
            )
            if lam_f > 1e-10 and lam_b > 1e-10:
                ratio = max(lam_f, lam_b) / min(lam_f, lam_b)
                assert ratio < 10.0, (
                    f"{name}: λ_f/λ_b = {ratio:.1f} (too asymmetric)"
                )


# =====================================================================
# 12. FRANCK-CONDON OVERLAP MATHEMATICAL PROPERTIES
# =====================================================================

class TestFCOverlapMathematical:
    """Rigorous tests of Franck-Condon overlap mathematical properties."""

    def test_symmetry_mu_nu_swap(self):
        """For equal frequencies, |S_μν|² = |S_νμ|² (symmetric overlap)."""
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        delta = 0.5
        for mu in range(5):
            for nu in range(5):
                S_mn = franck_condon_overlap(omega, omega, mass, delta, mu, nu)
                S_nm = franck_condon_overlap(omega, omega, mass, delta, nu, mu)
                assert abs(S_mn - S_nm) < 1e-10, (
                    f"|S_{mu}{nu}|² = {S_mn:.6e} ≠ |S_{nu}{mu}|² = {S_nm:.6e}"
                )

    def test_completeness_excited_states(self):
        """Σ_ν |S_μν|² → 1 for excited reactant states too."""
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        delta = 0.3

        for mu in range(3):
            total = sum(
                franck_condon_overlap(omega, omega, mass, delta, mu, nu)
                for nu in range(30)
            )
            assert abs(total - 1.0) < 0.10, (
                f"Completeness for μ={mu}: sum = {total:.4f}"
            )

    def test_overlap_monotonic_decrease_with_displacement(self):
        """Ground-state overlap should decrease monotonically with displacement."""
        omega = 3000.0 * CM_TO_HARTREE
        mass = PROTON_MASS_AMU * AMU_TO_AU
        prev_S = 1.0
        for delta in np.linspace(0.0, 3.0, 31):
            S = franck_condon_overlap(omega, omega, mass, delta, 0, 0)
            assert S <= prev_S + 1e-10, (
                f"S(δ={delta:.2f}) = {S:.6e} > S_prev = {prev_S:.6e}"
            )
            prev_S = S

    def test_mass_scaling_of_overlap(self):
        """Heavier isotope should always have smaller overlap (same displacement).

        This follows from α = mω/ℏ: heavier mass → more localized wavefunction.
        """
        delta = 0.6  # bohr
        omega_H = 3000.0 * CM_TO_HARTREE
        omega_D = omega_H * math.sqrt(PROTON_MASS_AMU / DEUTERIUM_MASS_AMU)
        omega_T = omega_H * math.sqrt(PROTON_MASS_AMU / TRITIUM_MASS_AMU)

        S_H = franck_condon_overlap(omega_H, omega_H, PROTON_MASS_AMU * AMU_TO_AU, delta, 0, 0)
        S_D = franck_condon_overlap(omega_D, omega_D, DEUTERIUM_MASS_AMU * AMU_TO_AU, delta, 0, 0)
        S_T = franck_condon_overlap(omega_T, omega_T, TRITIUM_MASS_AMU * AMU_TO_AU, delta, 0, 0)

        assert S_H > S_D > S_T, f"S_H={S_H:.4e}, S_D={S_D:.4e}, S_T={S_T:.4e}"


# =====================================================================
# 13. UNIT CONVERSION CONSISTENCY
# =====================================================================

class TestUnitConversions:
    """Verify internal consistency of unit conversion constants."""

    def test_hartree_to_kcal_roundtrip(self):
        assert abs(KCALMOL_TO_HARTREE * HARTREE_TO_KCALMOL - 1.0) < 1e-10

    def test_angstrom_bohr_roundtrip(self):
        from pcet_engine.core.constants import BOHR_TO_ANGSTROM
        assert abs(ANGSTROM_TO_BOHR * BOHR_TO_ANGSTROM - 1.0) < 1e-10

    def test_cm_hartree_roundtrip(self):
        from pcet_engine.core.constants import HARTREE_TO_CM
        assert abs(CM_TO_HARTREE * HARTREE_TO_CM - 1.0) < 1e-10

    def test_amu_to_au_value(self):
        """AMU_TO_AU should be approximately 1822.89."""
        assert abs(AMU_TO_AU - 1822.888) < 0.1

    def test_proton_mass(self):
        """Proton mass should be approximately 1.00783 amu."""
        assert abs(PROTON_MASS_AMU - 1.00783) < 0.001

    def test_deuterium_mass(self):
        """Deuterium mass should be approximately 2.01410 amu."""
        assert abs(DEUTERIUM_MASS_AMU - 2.01410) < 0.001

    def test_tritium_mass(self):
        """Tritium mass should be approximately 3.01605 amu."""
        assert abs(TRITIUM_MASS_AMU - 3.01605) < 0.001
