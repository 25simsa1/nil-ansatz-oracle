"""H2 molecular Hamiltonian (STO-3G, Jordan-Wigner, 4 qubits) and its exact ground state.

Paper 2 (Annis/Kassem/Coleman, arXiv:2606.04955) built this with PySCF + Qiskit Nature
at 0.735 A and quote an exact ground energy of -1.1373 Ha. We don't have PySCF here, so
we reconstruct the same operator directly from its second-quantized coefficients (taken
from the qrisp H2 tutorial, STO-3G / JW) and *validate by diagonalization*: the lowest
eigenvalue plus the nuclear-repulsion constant must reproduce -1.1373 Ha. If that check
passes, the operator is correct regardless of how we obtained the numbers.

Convention notes
----------------
* Jordan-Wigner with the standard lower-triangular Z string: a_p = (prod_{q<p} Z_q)(X_p + iY_p)/2.
* Qiskit Pauli label endianness: the leftmost character acts on the highest qubit index,
  so qubit p sits at string position (n-1-p).
* For the experiment we work with the *electronic* operator and E_exact = its lowest
  eigenvalue. The nuclear-repulsion term is a constant that cancels in any energy
  *difference* |E - E_exact|, so it only matters for reporting absolute energies.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

N_QUBITS = 4

# Bond length of the qrisp coefficient set (Paper 2 uses 0.735 A; the two agree to 4 d.p.).
BOND_LENGTH_ANGSTROM = 0.74
_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
NUCLEAR_REPULSION = 1.0 / (BOND_LENGTH_ANGSTROM * _BOHR_PER_ANGSTROM)  # Ha, = 1/R in atomic units

# Paper 2's quoted FCI ground-state energy for H2 at 0.735 A.
PAPER2_EXACT_ENERGY = -1.1373

# --- Second-quantized coefficients (STO-3G, JW), from the qrisp H2 tutorial ----------------
# Single number-operator terms  h_p * n_p
_H1 = {0: -1.25330978664598, 1: -1.25330978664598, 2: -0.475068848772179, 3: -0.475068848772179}
# Pair number-operator terms  h_pq * n_p n_q
_H2 = {
    (0, 1): 0.674755926814448,
    (0, 2): 0.482500939335617,
    (0, 3): 0.663711401350814,
    (1, 2): 0.663711401350814,
    (1, 3): 0.482500939335617,
    (2, 3): 0.697651504490461,
}
# Double-excitation (hopping) amplitude, shared by the four exchange terms.
_G = 0.181210462015197


def _jw_ladder(p: int, n: int = N_QUBITS, dagger: bool = False) -> SparsePauliOp:
    """Jordan-Wigner annihilation (dagger=False) or creation (dagger=True) on qubit p."""
    z = ["I"] * n
    for q in range(p):
        z[n - 1 - q] = "Z"
    lx, ly = z.copy(), z.copy()
    lx[n - 1 - p] = "X"
    ly[n - 1 - p] = "Y"
    sx, sy = "".join(lx), "".join(ly)
    # annihilation a = (X + iY)/2 ; creation a^dag = (X - iY)/2  (times the Z string)
    coeff_y = -0.5j if dagger else 0.5j
    return SparsePauliOp.from_list([(sx, 0.5), (sy, coeff_y)])


def h2_hamiltonian() -> SparsePauliOp:
    """Return the 4-qubit electronic H2 Hamiltonian as a SparsePauliOp (no nuclear repulsion)."""
    A = [_jw_ladder(p, dagger=False) for p in range(N_QUBITS)]
    C = [_jw_ladder(p, dagger=True) for p in range(N_QUBITS)]
    num = [C[p].dot(A[p]) for p in range(N_QUBITS)]  # n_p = a_p^dag a_p

    H = SparsePauliOp.from_list([("I" * N_QUBITS, 0.0)])
    for p, c in _H1.items():
        H = H + c * num[p]
    for (i, j), c in _H2.items():
        H = H + c * num[i].dot(num[j])

    # Four exchange / double-excitation terms.
    H = H + _G * A[0].dot(A[1]).dot(C[2]).dot(C[3])
    H = H + (-_G) * A[0].dot(C[1]).dot(C[2]).dot(A[3])
    H = H + (-_G) * C[0].dot(A[1]).dot(A[2]).dot(C[3])
    H = H + _G * C[0].dot(C[1]).dot(A[2]).dot(A[3])
    return H.simplify()


def exact_ground_state(H: SparsePauliOp | None = None):
    """Return (E_exact_electronic, ground_state_vector) by dense diagonalization (16x16)."""
    if H is None:
        H = h2_hamiltonian()
    mat = H.to_matrix()
    evals, evecs = np.linalg.eigh(mat)
    return float(evals[0]), evecs[:, 0]


def total_energy(electronic_energy: float) -> float:
    """Add the nuclear-repulsion constant to compare against Paper 2's -1.1373 Ha anchor."""
    return electronic_energy + NUCLEAR_REPULSION


if __name__ == "__main__":
    H = h2_hamiltonian()
    e_elec, psi0 = exact_ground_state(H)
    e_total = total_energy(e_elec)
    print(f"Hamiltonian terms (Pauli): {len(H)}")
    print(f"Electronic ground energy : {e_elec:+.6f} Ha")
    print(f"Nuclear repulsion        : {NUCLEAR_REPULSION:+.6f} Ha  (R = {BOND_LENGTH_ANGSTROM} A)")
    print(f"Total ground energy      : {e_total:+.6f} Ha")
    print(f"Paper 2 anchor           : {PAPER2_EXACT_ENERGY:+.6f} Ha")
    print(f"Difference vs anchor     :  {abs(e_total - PAPER2_EXACT_ENERGY):.6f} Ha")
