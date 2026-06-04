"""Paper 2's 12 H2 ansatz circuits (4 qubits), reproduced gate-for-gate from its Fig. 4.

Two families:
  * Ring (C1-C8): RY layer; 4 entanglers over the fixed ring positions
    [(2,3),(1,2),(0,1),(0,3)]; RY layer. Circuits differ only in the per-position gate
    type (CX with control=min index, or CZ). 8 params, 4 two-qubit gates, depth 6.
  * Two-layer (C9-C12): two rotation+entangle layers with all-to-all entanglement
    [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]. 6 two-qubit gates, depth 9.
      C9  : H,RX  + all-to-all CX
      C10 : H,RX  + all-to-all CZ
      C11 : RY,RZ + all-to-all CZ   (16 params)
      C12 : RY,RZ + all-to-all CX   (16 params)

Angles are pre-optimized classically against the exact ground state (optimal_angles) so we
treat the fixed optimal circuit as the VQE target, skipping the noisy optimization loop.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

RING_POSITIONS = [(2, 3), (1, 2), (0, 1), (0, 3)]
ALL_TO_ALL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Per-position entangler type for the ring family (read off Fig. 4).
RING_TYPES = {
    "C1": ["cx", "cx", "cx", "cx"],
    "C2": ["cz", "cz", "cz", "cz"],
    "C3": ["cx", "cz", "cx", "cz"],
    "C4": ["cx", "cz", "cz", "cx"],
    "C5": ["cx", "cx", "cz", "cz"],
    "C6": ["cz", "cx", "cz", "cx"],
    "C7": ["cz", "cx", "cx", "cz"],
    "C8": ["cz", "cz", "cx", "cx"],
}


def _entangle(qc: QuantumCircuit, a: int, b: int, gate: str) -> None:
    # control = min index for CX (matches Fig. 4); CZ is symmetric.
    (qc.cx if gate == "cx" else qc.cz)(a, b)


def _ring(name: str) -> QuantumCircuit:
    t = ParameterVector("t", 8)
    qc = QuantumCircuit(4, name=name)
    for i in range(4):
        qc.ry(t[i], i)
    for (a, b), g in zip(RING_POSITIONS, RING_TYPES[name]):
        _entangle(qc, a, b, g)
    for i in range(4):
        qc.ry(t[4 + i], i)
    return qc


def _two_layer_hrx(name: str, gate: str) -> QuantumCircuit:
    t = ParameterVector("t", 8)
    qc = QuantumCircuit(4, name=name)
    for i in range(4):
        qc.h(i)
    for i in range(4):
        qc.rx(t[i], i)
    for a, b in ALL_TO_ALL:
        _entangle(qc, a, b, gate)
    for i in range(4):
        qc.h(i)
    for i in range(4):
        qc.rx(t[4 + i], i)
    return qc


def _two_layer_ryrz(name: str, gate: str) -> QuantumCircuit:
    t = ParameterVector("t", 16)
    qc = QuantumCircuit(4, name=name)
    for i in range(4):
        qc.ry(t[i], i)
    for i in range(4):
        qc.rz(t[4 + i], i)
    for a, b in ALL_TO_ALL:
        _entangle(qc, a, b, gate)
    for i in range(4):
        qc.ry(t[8 + i], i)
    for i in range(4):
        qc.rz(t[12 + i], i)
    return qc


def _make(name: str):
    if name in RING_TYPES:
        return lambda n=name: _ring(n)
    return {
        "C9": lambda: _two_layer_hrx("C9", "cx"),
        "C10": lambda: _two_layer_hrx("C10", "cz"),
        "C11": lambda: _two_layer_ryrz("C11", "cz"),
        "C12": lambda: _two_layer_ryrz("C12", "cx"),
    }[name]


ANSATZE = {f"C{i}": _make(f"C{i}") for i in range(1, 13)}

# Backwards-compatible named builders used elsewhere / in tests.
circuit_c2 = ANSATZE["C2"]
circuit_c7 = ANSATZE["C7"]
circuit_c9 = ANSATZE["C9"]

# Paper 2 Table 3, COMBINED/NONE column (their noise-aware re-optimized |dE|, for reference).
PAPER2_NOISY_DE = {
    "C1": 0.1172, "C2": 0.0393, "C3": 0.0716, "C4": 0.0738, "C5": 0.0957, "C6": 0.0713,
    "C7": 0.0667, "C8": 0.0807, "C9": 0.2312, "C10": 0.6092, "C11": 0.0393, "C12": 0.1848,
}


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def optimal_angles(qc: QuantumCircuit, H: SparsePauliOp, n_restarts: int = 30, seed: int = 0):
    """Classically minimize the noiseless energy <psi(theta)|H|psi(theta)> (multi-start COBYLA)."""
    rng = np.random.default_rng(seed)
    n = qc.num_parameters

    def energy(x: np.ndarray) -> float:
        return float(Statevector(qc.assign_parameters(x)).expectation_value(H).real)

    best_x, best_e = None, np.inf
    for _ in range(n_restarts):
        res = minimize(energy, rng.uniform(0.0, 2 * np.pi, n), method="COBYLA",
                       options={"maxiter": 1000, "tol": 1e-8})
        if res.fun < best_e:
            best_x, best_e = res.x, float(res.fun)
    return best_x, best_e


if __name__ == "__main__":
    from molecules import h2_hamiltonian, exact_ground_state

    H = h2_hamiltonian()
    e_exact, _ = exact_ground_state(H)
    print(f"{'ansatz':6} {'2Q':>3} {'ideal |dE|':>11} {'paper2 noisy':>13}")
    for name, builder in ANSATZE.items():
        qc = builder()
        _, e = optimal_angles(qc, H, n_restarts=15)
        print(f"{name:6} {two_qubit_gate_count(qc):>3} {abs(e - e_exact):>11.5f} {PAPER2_NOISY_DE[name]:>13.4f}")
