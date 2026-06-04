"""Weight-1 Pauli neighbor circuits (NIL, Chen et al.).

A neighbor circuit inserts a single Pauli {X,Y,Z} on one qubit immediately after a
*noisy* gate location (here: RY and CX gates -- the only gates Paper 2's noise model
touches). The weight-0 neighbor (no insertion) is the original noisy circuit and is
always included as feature 0. The full weight-1 set has size O(#noise-locations); we
randomly subsample to `max_neighbors` for the starter, as NIL's randomized selection
allows.

The recipe is built once from the ansatz *structure* (gate positions are identical
across the target and all Clifford training circuits, which differ only in angles), then
materialized onto any bound circuit with matching structure.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

NOISY_GATES = ("ry", "cx")
_PAULI_APPLY = {"X": "x", "Y": "y", "Z": "z"}


class NeighborMap:
    def __init__(
        self,
        ansatz: QuantumCircuit,
        max_neighbors: int = 20,
        seed: int = 0,
        noisy_gates=NOISY_GATES,
    ):
        # Enumerate every weight-1 insertion: (instruction_index, qubit_index, pauli).
        candidates = []
        for idx, inst in enumerate(ansatz.data):
            if inst.operation.name in noisy_gates:
                for q in inst.qubits:
                    qi = ansatz.find_bit(q).index
                    for p in ("X", "Y", "Z"):
                        candidates.append((idx, qi, p))

        self.n_full = len(candidates)
        if max_neighbors is not None and self.n_full > max_neighbors:
            rng = np.random.default_rng(seed)
            keep = rng.choice(self.n_full, size=max_neighbors, replace=False)
            candidates = [candidates[i] for i in sorted(keep)]

        # recipe[0] = [] is the identity (weight-0) neighbor.
        self.recipes: list[list[tuple[int, int, str]]] = [[]] + [[c] for c in candidates]

    def __len__(self) -> int:
        return len(self.recipes)

    def materialize(self, bound: QuantumCircuit) -> list[QuantumCircuit]:
        """Return one circuit per recipe, with the Pauli inserted after the named gate."""
        return [self._apply(bound, r) for r in self.recipes]

    @staticmethod
    def _apply(circ: QuantumCircuit, recipe: list[tuple[int, int, str]]) -> QuantumCircuit:
        ins: dict[int, list[tuple[int, str]]] = {}
        for idx, q, p in recipe:
            ins.setdefault(idx, []).append((q, p))
        nc = QuantumCircuit(circ.num_qubits)
        for i, inst in enumerate(circ.data):
            qidx = [circ.find_bit(q).index for q in inst.qubits]
            nc.append(inst.operation, qidx)
            for q, p in ins.get(i, []):
                getattr(nc, _PAULI_APPLY[p])(q)
        return nc


if __name__ == "__main__":
    from ansatze import ANSATZE

    for name, builder in ANSATZE.items():
        nm = NeighborMap(builder(), max_neighbors=20, seed=1)
        print(f"{name}: {nm.n_full} full weight-1 neighbors -> {len(nm)} circuits (incl. identity)")
