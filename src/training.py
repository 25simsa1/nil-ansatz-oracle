"""2-design Clifford training set and exact stim labels (NIL).

Training circuits replace every rotation gate R_P(theta) with a random Clifford angle
theta_0 drawn uniformly from {0, pi/2, pi, 3pi/2}. The resulting circuits are Clifford,
so their exact expectation value <C|H|C> is classically computable -- we use stim, which
gives the label y_i for each training circuit essentially for free.

stim mapping for the Clifford angles (global phase is irrelevant for expectations):
  RY: 0->I, pi/2->SQRT_Y, pi->Y, 3pi/2->SQRT_Y_DAG
  RX: 0->I, pi/2->SQRT_X, pi->X, 3pi/2->SQRT_X_DAG
  RZ: 0->I, pi/2->S,      pi->Z, 3pi/2->S_DAG
The SQRT_Y/SQRT_Y_DAG orientation is verified against statevector simulation in
tests/test_clifford_labels.py.
"""

from __future__ import annotations

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

CLIFFORD_ANGLES = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

_RY_MAP = {0: None, 1: "SQRT_Y", 2: "Y", 3: "SQRT_Y_DAG"}
_RX_MAP = {0: None, 1: "SQRT_X", 2: "X", 3: "SQRT_X_DAG"}
_RZ_MAP = {0: None, 1: "S", 2: "Z", 3: "S_DAG"}
_FIXED = {"h": "H", "cx": "CX", "cz": "CZ", "x": "X", "y": "Y", "z": "Z", "s": "S"}


def sample_clifford_angles(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(CLIFFORD_ANGLES, size=n)


def build_training_circuits(ansatz: QuantumCircuit, n_train: int, seed: int = 0):
    """Return (circuits, angle_arrays): n_train Clifford circuits + the angles used."""
    rng = np.random.default_rng(seed)
    circuits, angles = [], []
    for _ in range(n_train):
        a = sample_clifford_angles(ansatz.num_parameters, rng)
        circuits.append(ansatz.assign_parameters(a))
        angles.append(a)
    return circuits, angles


def _clifford_index(angle: float) -> int:
    return int(round(float(angle) / (np.pi / 2))) % 4


def to_stim_circuit(clifford_circ: QuantumCircuit) -> stim.Circuit:
    c = stim.Circuit()
    for inst in clifford_circ.data:
        name = inst.operation.name
        qs = [clifford_circ.find_bit(q).index for q in inst.qubits]
        if name in ("ry", "rx", "rz"):
            k = _clifford_index(inst.operation.params[0])
            g = {"ry": _RY_MAP, "rx": _RX_MAP, "rz": _RZ_MAP}[name][k]
            if g is not None:
                c.append(g, qs)
        elif name in _FIXED:
            c.append(_FIXED[name], qs)
        elif name == "id":
            pass
        else:
            raise ValueError(f"non-Clifford / unsupported gate for stim: {name}")
    return c


def _qiskit_label_to_stim(label: str) -> stim.PauliString:
    n = len(label)
    chars = ["_" if label[n - 1 - q] == "I" else label[n - 1 - q] for q in range(n)]
    return stim.PauliString("".join(chars))


def h_terms_for_stim(H: SparsePauliOp):
    """Pre-convert H into [(coeff_real, stim.PauliString), ...] for fast labelling."""
    return [
        (float(coeff.real), _qiskit_label_to_stim(pauli.to_label()))
        for pauli, coeff in zip(H.paulis, H.coeffs)
    ]


def clifford_label(clifford_circ: QuantumCircuit, h_terms) -> float:
    """Exact <C|H|C> for a Clifford circuit via stim (each <P> is -1, 0, or +1)."""
    sim = stim.TableauSimulator()
    sim.do(to_stim_circuit(clifford_circ))
    return sum(coeff * sim.peek_observable_expectation(p) for coeff, p in h_terms)


if __name__ == "__main__":
    from molecules import h2_hamiltonian
    from ansatze import circuit_c7

    H = h2_hamiltonian()
    terms = h_terms_for_stim(H)
    circuits, _ = build_training_circuits(circuit_c7(), 5, seed=3)
    for i, c in enumerate(circuits):
        print(f"training circuit {i}: stim label <H> = {clifford_label(c, terms):+.5f}")
