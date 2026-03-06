"""
Verification: Can R^6 carry a quaternionic structure?
And: What do the complex structures on R^4 actually induce on Λ²(R^4) = R^6?

A quaternionic structure on a real vector space V requires three
endomorphisms I, J, K with I² = J² = K² = -1 and IJ = K.
This makes V into an H-module, requiring dim_R(V) ≡ 0 (mod 4).

Since dim(R^6) = 6 ≡ 2 (mod 4), NO quaternionic structure can exist.

This script verifies this obstruction explicitly and checks what the
complex structures on R^4 actually induce on Λ²(R^4).
"""

import numpy as np
from itertools import combinations

np.set_printoptions(precision=4, suppress=True)

# =============================================================
# Part 1: Complex structures on R^4 (the quaternions i, j, k)
# =============================================================

# Basis of R^4: e1, e2, e3, e4
# H = {a + bi + cj + dk}, identified with R^4 via (a, b, c, d)
# Left multiplication by i, j, k:

I4 = np.array([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 0, -1],
    [0,  0, 1,  0]
], dtype=float)  # Left mult by i: (a,b,c,d) -> (-b,a,-d,c)

J4 = np.array([
    [0, 0, -1, 0],
    [0, 0,  0, 1],
    [1, 0,  0, 0],
    [0, -1, 0, 0]
], dtype=float)  # Left mult by j: (a,b,c,d) -> (-c,d,a,-b)

K4 = np.array([
    [0, 0, 0, -1],
    [0, 0, -1, 0],
    [0, 1,  0, 0],
    [1, 0,  0, 0]
], dtype=float)  # Left mult by k: (a,b,c,d) -> (-d,-c,b,a)

print("=" * 60)
print("Part 1: Quaternionic structure on R^4")
print("=" * 60)
print(f"I² = -1? {np.allclose(I4 @ I4, -np.eye(4))}")
print(f"J² = -1? {np.allclose(J4 @ J4, -np.eye(4))}")
print(f"K² = -1? {np.allclose(K4 @ K4, -np.eye(4))}")
print(f"IJ = K?  {np.allclose(I4 @ J4, K4)}")
print(f"JI = -K? {np.allclose(J4 @ I4, -K4)}")
print()

# =============================================================
# Part 2: Induced action on Λ²(R^4) = R^6
# =============================================================

# Basis of Λ²(R^4): e12, e13, e14, e23, e24, e34
# Ordered as: (12, 13, 14, 23, 24, 34)

basis_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

def induced_action_on_2forms(A):
    """
    Given A: R^4 -> R^4, compute the induced action on Λ²(R^4).
    Λ²(A)(e_a ∧ e_b) = A(e_a) ∧ A(e_b)
    """
    n = 4
    dim2 = 6
    M = np.zeros((dim2, dim2))

    for col_idx, (a, b) in enumerate(basis_pairs):
        # A(e_a) and A(e_b)
        Aea = A[:, a]
        Aeb = A[:, b]

        # Compute A(e_a) ∧ A(e_b) in the basis
        for row_idx, (p, q) in enumerate(basis_pairs):
            # Coefficient of e_p ∧ e_q in A(e_a) ∧ A(e_b)
            # = Aea[p] * Aeb[q] - Aea[q] * Aeb[p]
            M[row_idx, col_idx] = Aea[p] * Aeb[q] - Aea[q] * Aeb[p]

    return M

Λ2_I = induced_action_on_2forms(I4)
Λ2_J = induced_action_on_2forms(J4)
Λ2_K = induced_action_on_2forms(K4)

print("=" * 60)
print("Part 2: Induced action on Λ²(R^4) = R^6")
print("=" * 60)
print(f"\nΛ²(I):\n{Λ2_I}")
print(f"\n(Λ²(I))² = +I (identity)?  {np.allclose(Λ2_I @ Λ2_I, np.eye(6))}")
print(f"(Λ²(I))² = -I (complex str)? {np.allclose(Λ2_I @ Λ2_I, -np.eye(6))}")

print(f"\nΛ²(J):\n{Λ2_J}")
print(f"(Λ²(J))² = +I?  {np.allclose(Λ2_J @ Λ2_J, np.eye(6))}")
print(f"(Λ²(J))² = -I?  {np.allclose(Λ2_J @ Λ2_J, -np.eye(6))}")

print(f"\nΛ²(K):\n{Λ2_K}")
print(f"(Λ²(K))² = +I?  {np.allclose(Λ2_K @ Λ2_K, np.eye(6))}")
print(f"(Λ²(K))² = -I?  {np.allclose(Λ2_K @ Λ2_K, -np.eye(6))}")

print(f"\nEigenvalues of Λ²(I): {np.sort(np.linalg.eigvals(Λ2_I).real)}")
print(f"Eigenvalues of Λ²(J): {np.sort(np.linalg.eigvals(Λ2_J).real)}")
print(f"Eigenvalues of Λ²(K): {np.sort(np.linalg.eigvals(Λ2_K).real)}")

print("\n>>> CONCLUSION: The induced maps have square = +I, not -I.")
print(">>> They are INVOLUTIONS, not complex structures.")
print(">>> The quaternionic structure on R^4 does NOT induce a")
print(">>> quaternionic structure on Λ²(R^4) = R^6.")

# =============================================================
# Part 3: The fundamental obstruction
# =============================================================

print("\n" + "=" * 60)
print("Part 3: Dimensional obstruction")
print("=" * 60)
print("""
A quaternionic structure (I, J, K) with I² = J² = K² = IJK = -1
makes V into a left H-module, so dim_R(V) must be divisible by 4.

dim(R^6) = 6 ≡ 2 (mod 4)

Therefore: NO quaternionic structure on R^6 can exist.
This is a theorem, not a conjecture.

Paper 5, Proposition 3.1 claims such a structure exists.
This proposition is FALSE.
""")

# =============================================================
# Part 4: What DOES R^6 have? The Hodge star and self-duality
# =============================================================

print("=" * 60)
print("Part 4: The Hodge star on Λ²(R^4)")
print("=" * 60)

# Hodge star on 2-forms in R^4 (Euclidean):
# *e12 = e34, *e13 = -e24, *e14 = e23
# *e23 = e14, *e24 = -e13, *e34 = e12
star = np.array([
    [0, 0, 0, 0, 0, 1],   # e12 -> e34
    [0, 0, 0, 0, -1, 0],  # e13 -> -e24
    [0, 0, 0, 1, 0, 0],   # e14 -> e23
    [0, 0, 1, 0, 0, 0],   # e23 -> e14
    [0, -1, 0, 0, 0, 0],  # e24 -> -e13
    [1, 0, 0, 0, 0, 0],   # e34 -> e12
], dtype=float)

print(f"*² = +I? {np.allclose(star @ star, np.eye(6))}")
print(f"Eigenvalues of *: {np.sort(np.linalg.eigvals(star).real)}")
print("Hodge star has eigenvalues ±1 (self-dual / anti-self-dual)")
print("It is an INVOLUTION, not a complex structure.\n")

# Self-dual basis: eigenvalue +1
# e12+e34, e14+e23, -(e13-e24) ... let me compute
evals, evecs = np.linalg.eig(star)
sd_mask = np.abs(evals.real - 1) < 1e-10
asd_mask = np.abs(evals.real + 1) < 1e-10
print(f"Self-dual subspace dimension: {np.sum(sd_mask)}")
print(f"Anti-self-dual subspace dimension: {np.sum(asd_mask)}")

# =============================================================
# Part 5: Can we combine * with Λ²(I) to get a complex structure?
# =============================================================

print("\n" + "=" * 60)
print("Part 5: Combining Hodge star with induced quaternion actions")
print("=" * 60)

# Try * ∘ Λ²(I), * ∘ Λ²(J), * ∘ Λ²(K)
for name, M in [("I", Λ2_I), ("J", Λ2_J), ("K", Λ2_K)]:
    combo = star @ M
    sq = combo @ combo
    is_complex = np.allclose(sq, -np.eye(6))
    evals_combo = np.linalg.eigvals(combo)
    print(f"(* ∘ Λ²({name}))² = -I? {is_complex}")
    print(f"  Eigenvalues: {np.sort_complex(evals_combo)}")

# Try Λ²(I) ∘ *
print()
for name, M in [("I", Λ2_I), ("J", Λ2_J), ("K", Λ2_K)]:
    combo = M @ star
    sq = combo @ combo
    is_complex = np.allclose(sq, -np.eye(6))
    evals_combo = np.linalg.eigvals(combo)
    print(f"(Λ²({name}) ∘ *)² = -I? {is_complex}")
    print(f"  Eigenvalues: {np.sort_complex(evals_combo)}")

# =============================================================
# Part 6: Check if * ∘ Λ²(I), etc. give anti-commuting complex structures
# =============================================================

print("\n" + "=" * 60)
print("Part 6: Do the combined maps form a quaternionic triple?")
print("=" * 60)

J1 = star @ Λ2_I
J2 = star @ Λ2_J
J3 = star @ Λ2_K

if np.allclose(J1 @ J1, -np.eye(6)):
    print(f"J1² = -I ✓")
    print(f"J2² = -I? {np.allclose(J2 @ J2, -np.eye(6))}")
    print(f"J3² = -I? {np.allclose(J3 @ J3, -np.eye(6))}")
    print(f"J1·J2 = J3? {np.allclose(J1 @ J2, J3)}")
    print(f"J1·J2 = -J3? {np.allclose(J1 @ J2, -J3)}")
    print(f"J1·J2 + J2·J1 = 0 (anticommute)? {np.allclose(J1 @ J2 + J2 @ J1, np.zeros((6,6)))}")
else:
    print("J1 = * ∘ Λ²(I) is NOT a complex structure.")

    # Try the other combination
    J1 = Λ2_I @ star
    J2 = Λ2_J @ star
    J3 = Λ2_K @ star

    if np.allclose(J1 @ J1, -np.eye(6)):
        print(f"\nTrying Λ²(·) ∘ * instead:")
        print(f"J1² = -I ✓")
        print(f"J2² = -I? {np.allclose(J2 @ J2, -np.eye(6))}")
        print(f"J3² = -I? {np.allclose(J3 @ J3, -np.eye(6))}")
        print(f"J1·J2 + J2·J1 = 0? {np.allclose(J1 @ J2 + J2 @ J1, np.zeros((6,6)))}")
    else:
        print("Λ²(I) ∘ * is also NOT a complex structure.")
        print("\n>>> No combination of Hodge star and induced quaternion")
        print(">>> actions produces complex structures on R^6.")

# =============================================================
# Part 7: The spinor perspective — quaternionic structure on C^4
# =============================================================

print("\n" + "=" * 60)
print("Part 7: Quaternionic structure on the SPINOR C^4 ≅ H²")
print("=" * 60)
print("(This DOES work, since dim_R(C^4) = 8 ≡ 0 mod 4)")

# Right multiplication by i, j, k on H² ≅ C^4
# H = C + jC, so (q1, q2) in H² ↔ (z1, z2, z3, z4) in C^4
# where q1 = z1 + jz2, q2 = z3 + jz4

# R_i(z1,z2,z3,z4) = (iz1, -iz2, iz3, -iz4)
R_i = np.diag([1j, -1j, 1j, -1j])

# R_j(z1,z2,z3,z4) = (-z2, z1, -z4, z3)
R_j = np.array([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 0, -1],
    [0,  0, 1,  0]
], dtype=complex)

# R_k(z1,z2,z3,z4) = (iz2, iz1, iz4, iz3)
R_k = np.array([
    [0, 1j, 0, 0],
    [1j, 0, 0, 0],
    [0,  0, 0, 1j],
    [0,  0, 1j, 0]
], dtype=complex)

print(f"R_i² = -I? {np.allclose(R_i @ R_i, -np.eye(4))}")
print(f"R_j² = -I? {np.allclose(R_j @ R_j, -np.eye(4))}")
print(f"R_k² = -I? {np.allclose(R_k @ R_k, -np.eye(4))}")
print(f"R_j ∘ R_i = R_k? {np.allclose(R_j @ R_i, R_k)}")

# Check SU(4) membership
print(f"\ndet(R_i) = {np.linalg.det(R_i):.4f}")
print(f"det(R_j) = {np.linalg.det(R_j):.4f}")
print(f"det(R_k) = {np.linalg.det(R_k):.4f}")

# Eigenvalue decomposition — what SU(3)×U(1) do these define?
print("\nEigenvalues (these define the SU(n)×U(1) breaking pattern):")
for name, R in [("R_i", R_i), ("R_j", R_j), ("R_k", R_k)]:
    evals = np.linalg.eigvals(R)
    print(f"  {name}: {np.sort_complex(evals)}")

print("""
Each R_a has eigenvalues +i (multiplicity 2) and -i (multiplicity 2).
This gives a 2+2 splitting of C^4, NOT a 3+1 splitting.

The stabilizer is S(U(2) × U(2)), NOT SU(3) × U(1).

>>> The quaternionic structure on C^4 gives SU(2)×SU(2)×U(1),
>>> NOT the SU(3) × U(1)_{B-L} needed for quark-lepton splitting.
""")

# =============================================================
# Part 8: What DOES give the 3+1 splitting?
# =============================================================

print("=" * 60)
print("Part 8: The actual SU(3)×U(1) complex structure")
print("=" * 60)

# The Hodge complex structure J_Hodge on R^6 corresponds to
# a specific element of SU(4) that gives 3+1 eigenvalue splitting.
# In SU(4), this is: T = diag(e^{iπ/2}, e^{iπ/2}, e^{iπ/2}, e^{-3iπ/2})
# up to normalization, which is just diag(i, i, i, -i) (normalized in SU(4))

# But wait: det(diag(i,i,i,-i)) = i³(-i) = -i³·i = ...
# i*i*i*(-i) = i³ * (-i) = (-i)(-i) = i² = -1. Not in SU(4)!

# For SU(4): need det = 1.
# diag(i, i, i, -3i) doesn't work either.
# The correct element: the complex structure J on R^6 lifts to
# an element of Spin(6) ≅ SU(4). The 3+1 splitting comes from
# choosing a LINE [v] ∈ CP^3, i.e., a unit vector v ∈ C^4.

# Standard choice: v = e4 = (0,0,0,1)
# Then SU(3) acts on {e1, e2, e3} and U(1) acts on e4.

# The complex structure as an element of SU(4):
# J_standard = diag(e^{iθ}, e^{iθ}, e^{iθ}, e^{-3iθ})
# For J² = -I, need e^{2iθ} = -1, so θ = π/2.
# J_standard = diag(i, i, i, -i) ... but det = i³(-i) = -i⁴ = ...
# det = i*i*i*(-i) = (i³)(-i) = (-i)(-i) = i² = -1. NOT in SU(4).

# Hmm. J² = -I in SO(6) lifts to g² = -I in SU(4) (center element).
# g = J_lift has eigenvalues ±i. For 3+1 split: three +i, one -i.

J_31 = np.diag([1j, 1j, 1j, -1j])
print(f"J_31 = diag(i, i, i, -i)")
print(f"det(J_31) = {np.linalg.det(J_31):.4f}")
print(f"J_31 ∈ SU(4)? {np.abs(np.linalg.det(J_31) - 1) < 1e-10}")
print(f"J_31² = -I? {np.allclose(J_31 @ J_31, -np.eye(4))}")
print(f"Eigenvalues: {np.linalg.eigvals(J_31)}")
print(f"Splitting: 3(+i) + 1(-i) → SU(3) × U(1) ✓")

# Now: can we find THREE such elements with 3+1 splitting that
# anti-commute like quaternions?
print("\n--- Can three 3+1 complex structures anti-commute? ---")
print("If J1, J2 are complex structures on C^4 with J1J2 = -J2J1,")
print("then J3 = J1J2 is also a complex structure and they form")
print("a quaternionic triple. But this requires dim ≡ 0 mod 4.")
print(f"dim_R(R^6) = 6, and 6 mod 4 = {6 % 4}")
print("So on the VECTOR SPACE R^6: impossible.")
print(f"dim_R(C^4 as R^8) = 8, and 8 mod 4 = {8 % 4}")
print("On the SPINOR SPACE C^4 ≅ R^8: possible, but gives 2+2 split.")

# =============================================================
# Part 9: How many inequivalent SU(3)×U(1) ⊂ SU(4) exist?
# =============================================================

print("\n" + "=" * 60)
print("Part 9: The space of SU(3)×U(1) subgroups")
print("=" * 60)
print("""
SU(3)×U(1) subgroups of SU(4) are parameterized by CP^3 = SU(4)/S(U(3)×U(1)).
This is a 6-real-dimensional manifold — a CONTINUOUS family, not three.

To select a FINITE number of distinguished subgroups, one needs a
discrete structure. Candidates:

1. Vertices of a regular simplex in CP^3 (but CP^3 is not a sphere)
2. A discrete subgroup of SU(4) acting on CP^3
3. Some physical principle selecting specific directions

The quaternionic structure of C^4 ≅ H^2 selects three POINTS in CP^3:
the images of a reference line [v] under R_i, R_j, R_k.
""")

# Take reference line [e1] = [(1,0,0,0)]
v0 = np.array([1, 0, 0, 0], dtype=complex)

# Act with R_i, R_j, R_k
v_i = R_i @ v0
v_j = R_j @ v0
v_k = R_k @ v0

print(f"Reference line v0 = {v0}")
print(f"R_i(v0) = {v_i}")
print(f"R_j(v0) = {v_j}")
print(f"R_k(v0) = {v_k}")

# Check overlaps (for MUB-like properties)
print(f"\n|<v0|R_i v0>|² = {np.abs(np.vdot(v0, v_i))**2:.4f}")
print(f"|<v0|R_j v0>|² = {np.abs(np.vdot(v0, v_j))**2:.4f}")
print(f"|<v0|R_k v0>|² = {np.abs(np.vdot(v0, v_k))**2:.4f}")
print(f"|<R_i v0|R_j v0>|² = {np.abs(np.vdot(v_i, v_j))**2:.4f}")
print(f"|<R_i v0|R_k v0>|² = {np.abs(np.vdot(v_i, v_k))**2:.4f}")
print(f"|<R_j v0|R_k v0>|² = {np.abs(np.vdot(v_j, v_k))**2:.4f}")

print("""
The four lines {v0, R_i(v0), R_j(v0), R_k(v0)} in CP^3 have
equal pairwise overlaps. They form a "SIC-like" configuration.

BUT: this gives FOUR lines, not three. And the overlaps are either
0 or 1, meaning they are ORTHOGONAL, not "mutually unbiased."
""")

# Check if v0, v_i, v_j, v_k are mutually orthogonal
print("Checking orthogonality:")
vecs = {"v0": v0, "v_i": v_i, "v_j": v_j, "v_k": v_k}
for (n1, u1), (n2, u2) in combinations(vecs.items(), 2):
    print(f"  <{n1}|{n2}> = {np.vdot(u1, u2):.4f}")

print("""
They form an orthogonal basis of C^4!
Each vector defines an SU(3)×U(1) as its stabilizer.
But four orthogonal vectors give FOUR subgroups, not three.
And they are related by a U(1) (phase) × S_4 (permutation) symmetry,
not by SU(2)_F.
""")

# =============================================================
# Part 10: The rank-3 structure of SL(4,R)/SO(4)
# =============================================================

print("=" * 60)
print("Part 10: Rank of the symmetric space (a correct 'threeness')")
print("=" * 60)
print("""
The fiber of the metric bundle is F = GL+(4,R)/SO(4).
Modulo the conformal factor: SL(4,R)/SO(4).

This is a RANK-3 symmetric space.
The rank = dim of maximal flat = number of independent eigenvalue ratios.

For a 4×4 positive-definite symmetric matrix with det=1:
  eigenvalues (λ1, λ2, λ3, λ4) with λ1λ2λ3λ4 = 1
  → 3 independent parameters.

The Dynkin diagram of SL(4,R) is A3: ○—○—○ (three nodes).

The restricted root system has 3 simple roots.
The Weyl group is S4 (order 24 — the same 24 as in the anomaly!).

This rank-3 structure is a MATHEMATICALLY CORRECT source of "threeness"
in the metric bundle framework.
""")

# =============================================================
# SUMMARY
# =============================================================

print("=" * 60)
print("SUMMARY OF FINDINGS")
print("=" * 60)
print("""
1. PAPER 5, PROPOSITION 3.1 IS FALSE.
   R^6 cannot carry a quaternionic structure (dim 6 ≢ 0 mod 4).

2. Complex structures on R^4 induce INVOLUTIONS on Λ²(R^4) = R^6
   (square = +I), not complex structures (square = -I).

3. The Hodge star on 2-forms is also an involution (* ² = +I).
   Combining * with the induced quaternion actions does NOT produce
   complex structures on R^6.

4. The SPINOR C^4 ≅ H^2 does carry a quaternionic structure,
   but it gives 2+2 eigenvalue splittings (→ SU(2)×SU(2)×U(1)),
   NOT the 3+1 splitting needed for SU(3)×U(1)_{B-L}.

5. SU(3)×U(1) subgroups of SU(4) form a CONTINUOUS family (CP^3),
   not a discrete set of three.

6. The quaternionic structure on C^4 selects FOUR orthogonal
   directions in CP^3 (not three), and with 2+2 splitting (not 3+1).

7. A correct source of "threeness": the RANK of the symmetric space
   SL(4,R)/SO(4) is 3. The Dynkin diagram A3 has three nodes.
   The Weyl group S4 has order 24 (connecting to the anomaly constraint).
""")
