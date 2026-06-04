"""Expressibility = KL divergence of a circuit's fidelity distribution from Haar.

This is Paper 2's Sec. 3.2 / 4 metric (originally Sim, Johnson, Aspuru-Guzik 2019):
  - sample many pairs of random parameter vectors, record fidelities F = |<psi(a)|psi(b)>|^2
  - histogram into n_bins over [0,1]
  - compare to the analytic Haar fidelity distribution P_Haar(F) = (N-1)(1-F)^(N-2), N=2^n
  - report KL(P_PQC || P_Haar). Lower = closer to a 2-design = more "expressible".

This is one of the cheap baselines we test the NIL oracle against.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def expressibility(ansatz: QuantumCircuit, n_samples: int = 5000, n_bins: int = 75, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = ansatz.num_qubits
    N = 2 ** n
    n_params = ansatz.num_parameters

    fids = np.empty(n_samples)
    for i in range(n_samples):
        a = rng.uniform(0.0, 2.0 * np.pi, n_params)
        b = rng.uniform(0.0, 2.0 * np.pi, n_params)
        sa = Statevector(ansatz.assign_parameters(a)).data
        sb = Statevector(ansatz.assign_parameters(b)).data
        fids[i] = np.abs(np.vdot(sa, sb)) ** 2

    hist, edges = np.histogram(fids, bins=n_bins, range=(0.0, 1.0))
    P = hist / hist.sum()
    # Exact Haar mass per bin: integral of (N-1)(1-F)^(N-2) over [a,b] = (1-a)^(N-1) - (1-b)^(N-1).
    Q = (1.0 - edges[:-1]) ** (N - 1) - (1.0 - edges[1:]) ** (N - 1)
    mask = P > 0
    return float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))


if __name__ == "__main__":
    from ansatze import ANSATZE

    print("Paper 2: H2 expressibility values span ~0.02-0.71 (lower = more expressible)")
    for name, builder in ANSATZE.items():
        print(f"  {name}: KL-from-Haar = {expressibility(builder(), n_samples=5000, seed=0):.4f}")
