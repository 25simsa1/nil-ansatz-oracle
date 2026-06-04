"""Paper 2's H2 ansatz circuits (4 qubits), reproduced from its Fig. 4.

We start with the three the design doc calls out:
  C2 -- best noisy performer  (ring of CZ, all entanglers noiseless under the model)
  C7 -- middle performer       (same ring topology, mixed CX/CZ)
  C9 -- worst non-pathological (H+RX layers, all-to-all CX)

All are RY/RX/H + CX/CZ hardware-efficient ansätze. Angles are pre-optimized classically
against the exact ground state (see optimal_angles) so we can treat the fixed optimal
circuit as the VQE target, skipping the noisy optimization loop entirely.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


def circuit_c2() -> QuantumCircuit:
    """C2: d=6, 4x CZ ring. RY layer, CZ(2,3)CZ(1,2)CZ(0,1)CZ(0,3), RY layer."""
    t = ParameterVector("t", 8)
    qc = QuantumCircuit(4, name="C2")
    for i in range(4):
        qc.ry(t[i], i)
    qc.cz(2, 3)
    qc.cz(1, 2)
    qc.cz(0, 1)
    qc.cz(0, 3)
    for i in range(4):
        qc.ry(t[4 + i], i)
    return qc


def circuit_c7() -> QuantumCircuit:
    """C7: d=6, 4x mixed. RY layer, CZ(2,3)CX(1->2)CX(0->1)CZ(0,3), RY layer."""
    t = ParameterVector("t", 8)
    qc = QuantumCircuit(4, name="C7")
    for i in range(4):
        qc.ry(t[i], i)
    qc.cz(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cz(0, 3)
    for i in range(4):
        qc.ry(t[4 + i], i)
    return qc


def circuit_c9() -> QuantumCircuit:
    """C9: d=9, 6x CX all-to-all. (H,RX) layer, all-pairs CX, (H,RX) layer."""
    t = ParameterVector("t", 8)
    qc = QuantumCircuit(4, name="C9")
    for i in range(4):
        qc.h(i)
    for i in range(4):
        qc.rx(t[i], i)
    for c, tg in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        qc.cx(c, tg)
    for i in range(4):
        qc.h(i)
    for i in range(4):
        qc.rx(t[4 + i], i)
    return qc


# Registry: name -> builder. Paper 2's COMBINED/NONE |dE| for context.
ANSATZE = {"C2": circuit_c2, "C7": circuit_c7, "C9": circuit_c9}
PAPER2_NOISY_DE = {"C2": 0.0393, "C7": 0.0667, "C9": 0.2312}


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def optimal_angles(qc: QuantumCircuit, H: SparsePauliOp, n_restarts: int = 30, seed: int = 0):
    """Classically minimize the noiseless energy <psi(theta)|H|psi(theta)>.

    Returns (theta_star, energy). Multi-start COBYLA; H2 is tiny so this is fast and
    isolates the architecture's representational limit from optimizer luck.
    """
    rng = np.random.default_rng(seed)
    n = qc.num_parameters

    def energy(x: np.ndarray) -> float:
        sv = Statevector(qc.assign_parameters(x))
        return float(sv.expectation_value(H).real)

    best_x, best_e = None, np.inf
    for _ in range(n_restarts):
        x0 = rng.uniform(0.0, 2.0 * np.pi, n)
        res = minimize(energy, x0, method="COBYLA", options={"maxiter": 1000, "tol": 1e-8})
        if res.fun < best_e:
            best_x, best_e = res.x, float(res.fun)
    return best_x, best_e


if __name__ == "__main__":
    from molecules import h2_hamiltonian, exact_ground_state, total_energy, PAPER2_EXACT_ENERGY

    H = h2_hamiltonian()
    e_exact, _ = exact_ground_state(H)
    print(f"{'ansatz':6} {'2Q':>3} {'ideal E_elec':>13} {'ideal |dE|':>11} {'paper2 ideal':>13}")
    paper2_ideal = {"C2": 0.0197, "C7": 0.0180, "C9": 0.0189}
    for name, builder in ANSATZE.items():
        qc = builder()
        x, e = optimal_angles(qc, H)
        print(f"{name:6} {two_qubit_gate_count(qc):>3} {e:>+13.5f} {abs(e - e_exact):>11.5f} {paper2_ideal[name]:>13.4f}")
    print(f"\nexact electronic ground = {e_exact:+.5f} Ha  (total {total_energy(e_exact):+.5f}, anchor {PAPER2_EXACT_ENERGY})")
