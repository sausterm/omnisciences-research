"""
Benchmark PCET systems with published experimental data.

Each system has:
- Published experimental rate constants (k_H, k_D), KIE, and E_a
- Estimated Marcus parameters from the literature (or from QM/MM studies)
- Proton vibrational frequencies and donor-acceptor distances

These parameters allow running the rate engine on model data before
real Hessian parsing is validated. Once real Hessian data is available,
these serve as the ground truth for validation.

References:
    [1] Knapp, M. J.; Rickert, K.; Klinman, J. P. JACS 124, 3865 (2002). [SLO-1 WT]
    [2] Masgrau, L. et al. Science 312, 237 (2006). [AADH]
    [3] Basran, J.; Sutcliffe, M. J.; Scrutton, N. S. Biochemistry 38, 3218 (1999). [MADH]
    [4] Francisco, W. A. et al. Biochemistry 41, 6573 (2002). [PHM]
    [5] Stubbe, J.; Nocera, D. G. et al. Chem. Rev. 103, 2167 (2003). [RNR]
    [6] Hammes-Schiffer, S. & Soudackov, A. V. J. Phys. Chem. B 112, 14108 (2008).
    [7] Hatcher, E.; Soudackov, A. V.; Hammes-Schiffer, S. JACS 126, 5763 (2004). [SLO-1 theory]
    [8] Hu, S. et al. JACS 136, 8157 (2014). [SLO-1 mutants]
    [9] Meyer, M. P.; Tomchick, D. R.; Klinman, J. P. PNAS 105, 1146 (2008). [SLO-1 DM]
    [10] Whittaker, M. M.; Whittaker, J. W. Biochemistry 40, 7140 (2001). [Galactose oxidase]
    [11] Bahnson, B. J.; Park, D. H.; Klinman, J. P. Biochemistry 32, 5503 (1993). [LADH]
    [12] Cape, J. L. et al. JACS 131, 2153 (2009). [Cytochrome bc1]
    [13] Brazeau, B. J. et al. JACS 123, 11831 (2001). [Copper amine oxidase]
    [14] Fierke, C. A.; Johnson, K. A.; Benkovic, S. J. Biochemistry 26, 4085 (1987). [DHFR]
    [15] Spencer, H. T. et al. Biochemistry 36, 4212 (1997). [Thymidylate synthase]
    [16] Edwards, S. J.; Soudackov, A. V.; Hammes-Schiffer, S. JPC A 113, 2117 (2009).
    [17] Rappoport, D.; Galvin, C. J.; Zubarev, D. Y.; Aspuru-Guzik, A. JCTC 10, 897 (2014). [MAO]
"""

from dataclasses import dataclass

from pcet_engine.core.rate_engine import PCETRateEngine, PCETResult


@dataclass
class BenchmarkSystem:
    """A benchmark PCET system with experimental and model data.

    Attributes:
        name: System identifier.
        description: Brief description.
        k_H_exp: Experimental H-transfer rate in s⁻¹.
        k_D_exp: Experimental D-transfer rate in s⁻¹.
        KIE_exp: Experimental KIE (k_H/k_D).
        E_a_exp: Experimental activation energy in kcal/mol.
        V_el: Electronic coupling in kcal/mol (from theory/fitting).
        delta_G: Driving force in kcal/mol.
        lambda_reorg: Total reorganization energy in kcal/mol.
        omega_H: Proton frequency in cm⁻¹.
        d_DA: Donor-acceptor distance in angstrom.
        Omega_gating: D-A gating mode frequency in cm⁻¹.
        M_DA: Reduced mass for gating mode in amu.
        r_DH: Donor-H bond length in angstrom.
        r_AH: Acceptor-H bond length in angstrom.
        reference: Literature citation.
        notes: Additional notes.
    """

    name: str
    description: str
    k_H_exp: float
    k_D_exp: float
    KIE_exp: float
    E_a_exp: float
    V_el: float
    delta_G: float
    lambda_reorg: float
    omega_H: float
    d_DA: float
    Omega_gating: float = 0.0
    M_DA: float = 0.0
    r_DH: float = 1.09
    r_AH: float = 0.96
    delta_0: float | None = None  # Explicit tunneling distance (Å), overrides geometric
    reference: str = ""
    notes: str = ""


# =====================================================================
# Benchmark systems
# =====================================================================

BENCHMARK_SYSTEMS = {
    "SLO-1": BenchmarkSystem(
        name="SLO-1",
        description="Soybean lipoxygenase-1, C-H abstraction by Fe(III)-OH",
        k_H_exp=297.0,          # s⁻¹ at 303 K
        k_D_exp=3.7,            # s⁻¹ at 303 K
        KIE_exp=81.0,           # k_H/k_D
        E_a_exp=2.1,            # kcal/mol
        # Model parameters from Hammes-Schiffer 2004 (Ref [7])
        V_el=0.6,               # kcal/mol — fitted to reproduce rate
        delta_G=-5.4,           # kcal/mol — from redox potential + pKa
        lambda_reorg=19.0,      # kcal/mol — inner + outer
        omega_H=2900.0,         # cm⁻¹ — C-H stretch in substrate
        d_DA=2.69,              # Å — C···O distance from crystal structure
        # Gating coordinate parameters (for T-dependence studies)
        Omega_gating=0.0,       # cm⁻¹ — disabled; set to 350 for T-dependent KIE
        M_DA=6.86,              # amu — reduced mass C-O: 12×16/(12+16) = 6.86
        r_DH=1.09,              # Å — C-H bond length
        r_AH=0.96,              # Å — O-H bond length in product
        # δ₀ calibrated to reproduce KIE=81 from ground-state FC overlap ratio
        delta_0=0.50,
        reference="Knapp et al. JACS 124, 3865 (2002); Hatcher et al. JACS 126, 5763 (2004)",
        notes="The KIE=81 is unusually large, attributed to significant tunneling. "
              "Temperature-independent KIE is the key experimental signature.",
    ),

    "AADH": BenchmarkSystem(
        name="AADH",
        description="Aromatic amine dehydrogenase, C-H bond cleavage",
        k_H_exp=93.0,           # s⁻¹ at 298 K (kcat)
        k_D_exp=1.7,            # s⁻¹
        KIE_exp=55.0,           # large, tunneling-dominated
        E_a_exp=11.4,           # kcal/mol
        V_el=0.8,               # kcal/mol
        delta_G=-8.0,           # kcal/mol
        lambda_reorg=35.0,      # kcal/mol — larger reorganization
        omega_H=3000.0,         # cm⁻¹
        d_DA=3.05,              # Å — from crystal structure
        Omega_gating=0.0,       # cm⁻¹ — disabled; set to 300 for T-dependent KIE
        M_DA=6.86,              # amu
        r_DH=1.09,              # Å — C-H
        r_AH=1.01,              # Å — N-H
        delta_0=0.465,          # Å — calibrated for KIE=55
        reference="Masgrau et al. Science 312, 237 (2006)",
        notes="Large KIE with strong temperature dependence of KIE.",
    ),

    "MADH": BenchmarkSystem(
        name="MADH",
        description="Methylamine dehydrogenase, C-H bond cleavage",
        k_H_exp=12.5,           # s⁻¹ at 298 K
        k_D_exp=0.42,           # s⁻¹
        KIE_exp=30.0,           # moderate
        E_a_exp=14.1,           # kcal/mol
        V_el=0.5,               # kcal/mol
        delta_G=-6.5,           # kcal/mol
        lambda_reorg=40.0,      # kcal/mol
        omega_H=2950.0,         # cm⁻¹
        d_DA=3.10,              # Å
        Omega_gating=0.0,       # cm⁻¹ — disabled; set to 280 for T-dependent KIE
        M_DA=6.46,              # amu — C-N reduced mass
        r_DH=1.09,              # Å — C-H
        r_AH=1.01,              # Å — N-H
        delta_0=0.432,          # Å — calibrated for KIE=30
        reference="Basran et al. Biochemistry 38, 3218 (1999)",
        notes="Similar to AADH but with larger reorganization energy.",
    ),

    "PHM": BenchmarkSystem(
        name="PHM",
        description="Peptidylglycine alpha-hydroxylating monooxygenase, Cu-mediated C-H activation",
        k_H_exp=28.0,           # s⁻¹ at 298 K
        k_D_exp=2.8,            # s⁻¹
        KIE_exp=10.0,           # moderate, less tunneling
        E_a_exp=4.0,            # kcal/mol
        V_el=1.2,               # kcal/mol — stronger coupling (Cu active site)
        delta_G=-3.0,           # kcal/mol
        lambda_reorg=15.0,      # kcal/mol — small for metal enzyme
        omega_H=3100.0,         # cm⁻¹ — C-H near Cu
        d_DA=2.55,              # Å — short D-A distance
        Omega_gating=0.0,       # cm⁻¹ — disabled; set to 400 for T-dependent KIE
        M_DA=6.86,              # amu — C-O reduced mass
        r_DH=1.09,              # Å — C-H
        r_AH=0.96,              # Å — O-H
        delta_0=0.347,          # Å — calibrated for KIE=10
        reference="Francisco et al. Biochemistry 41, 6573 (2002)",
        notes="Copper enzyme, shorter D-A distance, smaller KIE suggests less tunneling.",
    ),

    "RNR": BenchmarkSystem(
        name="RNR",
        description="Ribonucleotide reductase, long-range PCET (thiyl radical)",
        k_H_exp=10.0,           # s⁻¹ at 298 K (rate-limiting step)
        k_D_exp=1.4,            # s⁻¹
        KIE_exp=7.0,            # small KIE, long-range
        E_a_exp=3.5,            # kcal/mol
        V_el=0.2,               # kcal/mol — weak coupling (long-range)
        delta_G=-2.0,           # kcal/mol
        lambda_reorg=12.0,      # kcal/mol
        omega_H=2600.0,         # cm⁻¹ — S-H stretch
        d_DA=2.80,              # Å — S···C distance
        Omega_gating=0.0,       # cm⁻¹ — disabled; set to 250 for T-dependent KIE
        M_DA=8.73,              # amu — S-C reduced mass: 32×12/(32+12)=8.73
        r_DH=1.34,              # Å — S-H bond (longer than C-H)
        r_AH=1.09,              # Å — C-H in product
        delta_0=0.348,          # Å — calibrated for KIE=7
        reference="Stubbe & Nocera et al. Chem. Rev. 103, 2167 (2003)",
        notes="Long-range PCET across 35 Å pathway. Rate-limiting step involves "
              "S-H bond cleavage. Small KIE suggests classical-like behavior.",
    ),

    # === SLO-1 MUTANTS (Ref [8], [9]) ===

    "SLO-1-L546A": BenchmarkSystem(
        name="SLO-1-L546A",
        description="Soybean lipoxygenase-1 L546A mutant, C-H abstraction",
        k_H_exp=8.2,             # s⁻¹ at 303 K (Hu et al. 2014)
        k_D_exp=0.088,           # s⁻¹ at 303 K
        KIE_exp=93.0,
        E_a_exp=4.1,             # kcal/mol
        V_el=0.5,                # slightly reduced from WT
        delta_G=-5.4,            # same substrate
        lambda_reorg=20.0,       # slightly larger (looser cavity)
        omega_H=2900.0,          # same C-H stretch
        d_DA=2.88,               # Å — enlarged by mutation
        M_DA=6.86,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.514,           # Å — calibrated for KIE=93
        reference="Hu et al. JACS 136, 8157 (2014)",
        notes="Single mutant opens active-site cavity, increases d_DA.",
    ),

    "SLO-1-L754A": BenchmarkSystem(
        name="SLO-1-L754A",
        description="Soybean lipoxygenase-1 L754A mutant, C-H abstraction",
        k_H_exp=3.0,             # s⁻¹ at 303 K
        k_D_exp=0.027,           # s⁻¹
        KIE_exp=112.0,
        E_a_exp=4.5,             # kcal/mol
        V_el=0.45,
        delta_G=-5.4,
        lambda_reorg=21.0,
        omega_H=2900.0,
        d_DA=2.95,               # Å — further enlarged
        M_DA=6.86,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.524,           # Å — calibrated for KIE=112
        reference="Hu et al. JACS 136, 8157 (2014)",
        notes="Second cavity mutation, larger KIE than L546A.",
    ),

    "SLO-1-DM": BenchmarkSystem(
        name="SLO-1-DM",
        description="Soybean lipoxygenase-1 L546A/L754A double mutant",
        k_H_exp=0.3,             # s⁻¹ at 303 K (Meyer et al. 2008)
        k_D_exp=4.5e-4,          # s⁻¹ — KIE ~661
        KIE_exp=661.0,
        E_a_exp=6.0,             # kcal/mol
        V_el=0.3,                # weakened coupling
        delta_G=-5.4,
        lambda_reorg=23.0,       # increased
        omega_H=2900.0,
        d_DA=3.10,               # Å — significantly elongated
        M_DA=6.86,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.614,           # Å — calibrated for KIE=661
        reference="Meyer et al. PNAS 105, 1146 (2008); Hu et al. JACS 136, 8157 (2014)",
        notes="Record KIE ~661 (nearly T-independent). Rigidified active site. "
              "Key test of vibronic nonadiabatic model.",
    ),

    # === ADDITIONAL ENZYMES ===

    "GO": BenchmarkSystem(
        name="GO",
        description="Galactose oxidase, Cu/Tyr-radical C-H abstraction",
        k_H_exp=60.0,            # s⁻¹ at 298 K (kcat)
        k_D_exp=2.7,             # s⁻¹
        KIE_exp=22.0,            # Whittaker & Whittaker 2001
        E_a_exp=3.5,             # kcal/mol
        V_el=0.7,
        delta_G=-4.0,
        lambda_reorg=18.0,
        omega_H=2900.0,          # C-H of galactose
        d_DA=2.72,               # Å — Tyr-O to substrate C
        r_DH=1.09,
        r_AH=0.96,              # O-H
        delta_0=0.422,           # Å — calibrated for KIE=22
        reference="Whittaker & Whittaker Biochemistry 40, 7140 (2001)",
        notes="Cu(II)/Tyr-radical active site. H-atom abstraction "
              "from alcohol substrate by Tyr radical.",
    ),

    "LADH": BenchmarkSystem(
        name="LADH",
        description="Liver alcohol dehydrogenase, NAD+-mediated hydride transfer",
        k_H_exp=5.0,             # s⁻¹ at 298 K (benzyl alcohol)
        k_D_exp=1.4,             # s⁻¹
        KIE_exp=3.5,             # Bahnson et al. 1993
        E_a_exp=14.0,            # kcal/mol
        V_el=2.0,                # more adiabatic (hydride)
        delta_G=-4.0,
        lambda_reorg=25.0,
        omega_H=3000.0,          # C-H of substrate
        d_DA=2.70,               # Å — C to C (hydride transfer distance)
        r_DH=1.09,
        r_AH=1.09,              # C-H in product (NAD)
        delta_0=0.264,           # Å — calibrated for KIE=3.5
        reference="Bahnson et al. Biochemistry 32, 5503 (1993)",
        notes="Classic enzyme tunneling system. Moderate KIE but large A_H/A_D "
              "ratio indicates tunneling. Hydride transfer to NAD+.",
    ),

    "bc1": BenchmarkSystem(
        name="bc1",
        description="Cytochrome bc1 Qo site, quinol oxidation PCET",
        k_H_exp=1.0e3,           # s⁻¹ at 298 K
        k_D_exp=2.9e2,           # s⁻¹
        KIE_exp=3.5,             # Cape et al. 2009
        E_a_exp=12.0,            # kcal/mol
        V_el=1.5,
        delta_G=-6.0,
        lambda_reorg=30.0,
        omega_H=3300.0,          # O-H of ubiquinol
        d_DA=2.70,               # Å — O-H···N(His) distance
        r_DH=0.96,              # O-H
        r_AH=1.01,              # N-H
        delta_0=0.250,           # Å — calibrated for KIE=3.5
        reference="Cape et al. JACS 131, 2153 (2009)",
        notes="Quinol oxidation at Qo site of respiratory chain. Concerted PCET "
              "to His ligand of Rieske [2Fe-2S] cluster.",
    ),

    "CAO": BenchmarkSystem(
        name="CAO",
        description="Copper amine oxidase, C-H abstraction by TPQ cofactor",
        k_H_exp=4.0,             # s⁻¹ at 298 K (methylamine)
        k_D_exp=0.33,            # s⁻¹
        KIE_exp=12.0,            # Brazeau et al. 2001
        E_a_exp=10.0,            # kcal/mol
        V_el=0.6,
        delta_G=-5.0,
        lambda_reorg=30.0,
        omega_H=2900.0,          # C-H of amine substrate
        d_DA=3.00,               # Å
        r_DH=1.09,              # C-H
        r_AH=1.01,              # N-H (TPQ quinone nitrogen)
        delta_0=0.379,           # Å — calibrated for KIE=12
        reference="Brazeau et al. JACS 123, 11831 (2001)",
        notes="Topa-quinone (TPQ) cofactor. Proton abstraction from "
              "amine substrate alpha-C-H bond.",
    ),

    "DHFR": BenchmarkSystem(
        name="DHFR",
        description="Dihydrofolate reductase, hydride transfer from NADPH",
        k_H_exp=950.0,           # s⁻¹ at 298 K (E. coli)
        k_D_exp=320.0,           # s⁻¹
        KIE_exp=3.0,             # Fierke et al. 1987
        E_a_exp=3.5,             # kcal/mol (low barrier)
        V_el=3.0,                # strong coupling (nearly adiabatic)
        delta_G=-4.0,
        lambda_reorg=15.0,
        omega_H=3000.0,          # C-H of NADPH
        d_DA=2.65,               # Å — short, compressed D-A
        r_DH=1.09,
        r_AH=1.09,              # C-H in product
        delta_0=0.247,           # Å — calibrated for KIE=3
        reference="Fierke et al. Biochemistry 26, 4085 (1987)",
        notes="Well-studied hydride transfer. Small KIE at room temperature, "
              "but T-dependent KIE reveals tunneling contribution.",
    ),

    "TSase": BenchmarkSystem(
        name="TSase",
        description="Thymidylate synthase, hydride transfer from CH2H4folate",
        k_H_exp=1.5,             # s⁻¹ at 298 K
        k_D_exp=0.25,            # s⁻¹
        KIE_exp=6.0,             # Spencer et al. 1997
        E_a_exp=5.0,             # kcal/mol
        V_el=0.8,
        delta_G=-3.5,
        lambda_reorg=22.0,
        omega_H=2950.0,          # C-H
        d_DA=2.85,               # Å
        r_DH=1.09,
        r_AH=1.09,
        delta_0=0.318,           # Å — calibrated for KIE=6
        reference="Spencer et al. Biochemistry 36, 4212 (1997)",
        notes="Concerted hydride/proton transfer in dTMP synthesis. "
              "Moderate KIE with strong T-dependence.",
    ),

    "MAO": BenchmarkSystem(
        name="MAO",
        description="Monoamine oxidase, flavin-dependent amine oxidation",
        k_H_exp=5.0,             # s⁻¹ at 298 K
        k_D_exp=0.6,             # s⁻¹
        KIE_exp=8.0,
        E_a_exp=12.0,            # kcal/mol
        V_el=0.5,
        delta_G=-7.0,
        lambda_reorg=35.0,       # large reorganization (protein + FAD)
        omega_H=2950.0,          # C-H of amine substrate
        d_DA=3.00,               # Å
        r_DH=1.09,
        r_AH=1.01,              # N-H (flavin N5)
        delta_0=0.345,           # Å — calibrated for KIE=8
        reference="Miller & Edmondson Biochemistry 38, 13670 (1999)",
        notes="Flavin-dependent oxidase. C-H cleavage is rate-limiting "
              "with substantial tunneling contribution.",
    ),

    # =====================================================================
    # SLO-1 I553X mutant series (Hu et al. JACS 136, 8157, 2014)
    # =====================================================================

    "SLO-1-I553G": BenchmarkSystem(
        name="SLO-1-I553G",
        description="SLO-1 I553G mutant, most distal packing",
        k_H_exp=1.9,
        k_D_exp=0.019,
        KIE_exp=100.0,
        E_a_exp=2.5,
        V_el=0.25,
        delta_G=-5.4,
        lambda_reorg=19.0,
        omega_H=2900.0,
        d_DA=3.02,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.518,
        reference="Hu et al. JACS 136, 8157 (2014)",
        notes="I553G creates larger cavity, increasing d_DA and KIE.",
    ),
    "SLO-1-I553A": BenchmarkSystem(
        name="SLO-1-I553A",
        description="SLO-1 I553A mutant",
        k_H_exp=4.8,
        k_D_exp=0.055,
        KIE_exp=87.0,
        E_a_exp=2.4,
        V_el=0.35,
        delta_G=-5.4,
        lambda_reorg=19.0,
        omega_H=2900.0,
        d_DA=2.97,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.510,
        reference="Hu et al. JACS 136, 8157 (2014)",
        notes="Intermediate packing perturbation.",
    ),
    "SLO-1-I553V": BenchmarkSystem(
        name="SLO-1-I553V",
        description="SLO-1 I553V mutant",
        k_H_exp=18.0,
        k_D_exp=0.25,
        KIE_exp=72.0,
        E_a_exp=2.2,
        V_el=0.45,
        delta_G=-5.4,
        lambda_reorg=19.0,
        omega_H=2900.0,
        d_DA=2.85,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.499,
        reference="Hu et al. JACS 136, 8157 (2014)",
        notes="Conservative substitution, modest KIE change.",
    ),
    "SLO-1-I553L": BenchmarkSystem(
        name="SLO-1-I553L",
        description="SLO-1 I553L mutant",
        k_H_exp=40.0,
        k_D_exp=0.62,
        KIE_exp=65.0,
        E_a_exp=2.1,
        V_el=0.50,
        delta_G=-5.4,
        lambda_reorg=19.0,
        omega_H=2900.0,
        d_DA=2.80,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.493,
        reference="Hu et al. JACS 136, 8157 (2014)",
        notes="Near-WT packing, similar d_DA.",
    ),
    "SLO-1-I553F": BenchmarkSystem(
        name="SLO-1-I553F",
        description="SLO-1 I553F mutant, aromatic packing",
        k_H_exp=120.0,
        k_D_exp=1.7,
        KIE_exp=71.0,
        E_a_exp=2.0,
        V_el=0.55,
        delta_G=-5.4,
        lambda_reorg=19.0,
        omega_H=2900.0,
        d_DA=2.75,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.499,
        reference="Hu et al. JACS 136, 8157 (2014)",
        notes="Phe packing compresses d_DA slightly below WT.",
    ),

    # =====================================================================
    # Model compounds
    # =====================================================================

    "PhOH-self": BenchmarkSystem(
        name="PhOH-self",
        description="Phenoxyl/phenol self-exchange (model compound)",
        k_H_exp=4.5e4,
        k_D_exp=1.1e4,
        KIE_exp=4.1,
        E_a_exp=3.5,
        V_el=1.5,
        delta_G=0.0,
        lambda_reorg=30.0,
        omega_H=3000.0,
        d_DA=2.40,
        r_DH=1.00,
        r_AH=1.00,
        delta_0=0.279,
        reference="Hatcher, Soudackov, Hammes-Schiffer JACS 126, 5763 (2004); "
                  "Edwards et al. JPC A 113, 2117 (2009)",
        notes="Textbook PCET model compound. Self-exchange (delta_G=0). "
              "Small KIE reflects short d_DA and strong coupling.",
    ),

    # =====================================================================
    # Non-heme iron and copper enzymes (Bollinger/Krebs/Klinman groups)
    # =====================================================================

    "TauD": BenchmarkSystem(
        name="TauD",
        description="Taurine/alpha-ketoglutarate dioxygenase (Fe(IV)=O)",
        k_H_exp=15.0,
        k_D_exp=0.56,
        KIE_exp=27.0,
        E_a_exp=6.5,
        V_el=0.5,
        delta_G=-4.0,
        lambda_reorg=20.0,
        omega_H=2900.0,
        d_DA=2.65,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.436,
        reference="Price et al. JACS 125, 13008 (2003); "
                  "Grzyska et al. PNAS 107, 3982 (2010)",
        notes="Fe(IV)=O (J-type) enzyme; H-abstraction from C1-H of taurine. "
              "KIE=37 measured at 5°C; KIE~27 extrapolated to 25°C via "
              "E_a_D-E_a_H=2.5 kcal/mol (T-dependent KIE). "
              "d_DA = C1...O(Fe=O) from QM/MM (Sinnecker et al. J Phys Chem A 2006).",
    ),
    "DβH": BenchmarkSystem(
        name="DβH",
        description="Dopamine beta-hydroxylase (CuB enzyme)",
        k_H_exp=22.0,
        k_D_exp=2.0,
        KIE_exp=10.8,
        E_a_exp=5.5,
        V_el=0.6,
        delta_G=-3.0,
        lambda_reorg=20.0,
        omega_H=2900.0,
        d_DA=2.65,
        r_DH=1.09,
        r_AH=0.96,
        delta_0=0.370,
        reference="Farnum et al. JACS 108, 4846 (1986); "
                  "Klinman Methods Enzymol 249, 373 (1995)",
        notes="CuB enzyme; benzylic C-H of dopamine abstracted by CuOH radical. "
              "KIE=10.8 measured at 25°C by Northrop method. "
              "Structural analogy to PHM; d_DA estimated from DβH model.",
    ),

    # =====================================================================
    # Flavoenzymes — OYE family hydride transfer (Scrutton group)
    # =====================================================================

    "MR": BenchmarkSystem(
        name="MR",
        description="Morphinone reductase (OYE family flavoenzyme)",
        k_H_exp=200.0,
        k_D_exp=8.0,
        KIE_exp=25.0,
        E_a_exp=5.5,
        V_el=0.8,
        delta_G=-8.0,
        lambda_reorg=20.0,
        omega_H=2900.0,
        d_DA=2.70,
        r_DH=1.09,
        r_AH=1.01,
        delta_0=0.440,
        reference="Hay et al. JACS 131, 9040 (2009); "
                  "Scrutton et al. Nat Chem 4, 161 (2012)",
        notes="NADPH->FMN hydride transfer. Large KIE and temperature-dependent "
              "(E_a_H=5.5, E_a_D=8.5 kcal/mol, dEa=3.0). "
              "d_DA = C4(NADPH)...N5(FMN) from 1LKM crystal structure.",
    ),
    "PETNR": BenchmarkSystem(
        name="PETNR",
        description="Pentaerythritol tetranitrate reductase (OYE family)",
        k_H_exp=20.0,
        k_D_exp=3.7,
        KIE_exp=5.4,
        E_a_exp=6.0,
        V_el=0.8,
        delta_G=-6.0,
        lambda_reorg=18.0,
        omega_H=2900.0,
        d_DA=2.65,
        r_DH=1.09,
        r_AH=1.01,
        delta_0=0.315,
        reference="Pudney et al. JACS 129, 13949 (2007); "
                  "Hay & Scrutton Nat Chem 4, 161 (2012)",
        notes="NADPH->FMN hydride transfer. Moderate KIE, temperature-dependent "
              "(E_a_H=6.0, E_a_D=8.0, dEa=2.0). "
              "d_DA = C4(NADPH)...N5(FMN) from 1H50 crystal structure.",
    ),

    # =====================================================================
    # Additional enzymes
    # =====================================================================

    "GOx": BenchmarkSystem(
        name="GOx",
        description="Glucose oxidase, C-H bond cleavage by flavin",
        k_H_exp=150.0,
        k_D_exp=18.0,
        KIE_exp=8.3,
        E_a_exp=3.8,
        V_el=0.8,
        delta_G=-8.0,
        lambda_reorg=25.0,
        omega_H=2900.0,
        d_DA=2.80,
        r_DH=1.09,
        r_AH=1.00,
        delta_0=0.354,
        reference="Roth & Klinman Biochemistry 42, 14893 (2003); "
                  "Roth & Klinman PNAS 100, 62 (2003)",
        notes="Flavoenzyme PCET. Moderate KIE consistent with "
              "concerted mechanism and moderate tunneling.",
    ),

    # =====================================================================
    # Fluorotyrosine RNR variants (Minnihan et al. PNAS 108, 3955, 2011)
    # =====================================================================

    "RNR-3FY": BenchmarkSystem(
        name="RNR-3FY",
        description="RNR with 3-fluorotyrosine (pKa shifted)",
        k_H_exp=3.0,
        k_D_exp=0.5,
        KIE_exp=6.0,
        E_a_exp=3.0,
        V_el=0.5,
        delta_G=-2.0,
        lambda_reorg=20.0,
        omega_H=3200.0,
        d_DA=2.80,
        r_DH=0.97,
        r_AH=0.97,
        delta_0=0.303,
        reference="Minnihan et al. PNAS 108, 3955 (2011)",
        notes="Fluorotyrosine substitution shifts pKa, modulating "
              "driving force for PCET at Y356.",
    ),
    "RNR-2FY": BenchmarkSystem(
        name="RNR-2FY",
        description="RNR with 2,3-difluorotyrosine",
        k_H_exp=0.5,
        k_D_exp=0.1,
        KIE_exp=5.0,
        E_a_exp=3.5,
        V_el=0.4,
        delta_G=-1.0,
        lambda_reorg=20.0,
        omega_H=3200.0,
        d_DA=2.80,
        r_DH=0.97,
        r_AH=0.97,
        delta_0=0.287,
        reference="Minnihan et al. PNAS 108, 3955 (2011)",
        notes="Double fluorination further shifts pKa and reduces "
              "driving force. Tests sensitivity to delta_G.",
    ),
}


def run_benchmarks(
    method: str = "vibronic_multi",
    temperature: float = 298.15,
    verbose: bool = True,
    n_eff: float | None = None,
) -> dict[str, dict]:
    """Run all benchmark systems and compare to experiment.

    Args:
        method: Rate calculation method.
        temperature: Temperature in Kelvin.
        verbose: If True, print results table.
        n_eff: If provided, apply geometric tunneling correction with this
            N_eff value to all systems. Use None for uncorrected rates.

    Returns:
        Dict mapping system name -> {result, errors}.
    """
    engine = PCETRateEngine(temperature=temperature)
    results = {}

    if verbose:
        neff_label = f", N_eff={n_eff:.2f}" if n_eff is not None else ""
        print("=" * 100)
        print(f"PCET RATE ENGINE BENCHMARK SUITE — Method: {method}, T = {temperature:.1f} K{neff_label}")
        print("=" * 100)
        print(f"{'System':<12} {'k_H pred':>12} {'k_H exp':>12} {'KIE pred':>10} "
              f"{'KIE exp':>10} {'E_a pred':>10} {'E_a exp':>10} {'N_eff':>8} {'log err':>8}")
        print("-" * 100)

    for name, sys in BENCHMARK_SYSTEMS.items():
        result = engine.compute_rate(
            V_el=sys.V_el,
            delta_G=sys.delta_G,
            lambda_reorg=sys.lambda_reorg,
            omega_H=sys.omega_H,
            d_DA=sys.d_DA,
            method=method,
            Omega_gating=sys.Omega_gating,
            M_DA=sys.M_DA,
            r_DH=sys.r_DH,
            r_AH=sys.r_AH,
            delta_0=sys.delta_0,
            n_eff=n_eff,
        )

        # Also compute uncorrected KIE for comparison
        result_1d = engine.compute_rate(
            V_el=sys.V_el,
            delta_G=sys.delta_G,
            lambda_reorg=sys.lambda_reorg,
            omega_H=sys.omega_H,
            d_DA=sys.d_DA,
            method=method,
            Omega_gating=sys.Omega_gating,
            M_DA=sys.M_DA,
            r_DH=sys.r_DH,
            r_AH=sys.r_AH,
            delta_0=sys.delta_0,
        )

        import math
        log_err_kH = math.log10(result.k_H / sys.k_H_exp) if result.k_H > 0 and sys.k_H_exp > 0 else float("inf")
        kie_ratio = result.KIE / sys.KIE_exp if sys.KIE_exp > 0 else float("inf")

        results[name] = {
            "result": result,
            "result_1d": result_1d,
            "k_H_exp": sys.k_H_exp,
            "k_D_exp": sys.k_D_exp,
            "KIE_exp": sys.KIE_exp,
            "KIE_1d": result_1d.KIE,
            "KIE_corr": result.KIE,
            "E_a_exp": sys.E_a_exp,
            "log_error_kH": log_err_kH,
            "KIE_ratio": kie_ratio,
            "n_eff": result.n_eff,
            "geometric_prefactor": result.geometric_prefactor,
        }

        if verbose:
            print(f"{name:<12} {result.k_H:>12.2e} {sys.k_H_exp:>12.2e} "
                  f"{result.KIE:>10.1f} {sys.KIE_exp:>10.1f} "
                  f"{result.E_a:>10.1f} {sys.E_a_exp:>10.1f} "
                  f"{result.n_eff:>8.2f} {log_err_kH:>+8.2f}")

    if verbose:
        print("-" * 100)
        import numpy as np
        log_errs = [abs(r["log_error_kH"]) for r in results.values() if abs(r["log_error_kH"]) < 100]
        kie_ratios = [abs(r["KIE_ratio"] - 1.0) for r in results.values() if abs(r["KIE_ratio"]) < 100]
        if log_errs:
            print(f"Mean |log10(k_pred/k_exp)| = {np.mean(log_errs):.2f}")
        if kie_ratios:
            print(f"Mean |KIE_ratio - 1| = {np.mean(kie_ratios):.2f}")
        print("=" * 100)

    return results
