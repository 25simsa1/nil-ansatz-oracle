"""NIL core: noisy features, the L1-constrained Lasso `combine`, and the training-MSE oracle.

Pipeline for one ansatz target C(theta*):
  1. Build the weight-1 neighbor map.
  2. Generate T 2-design Clifford training circuits; label each with exact <H> (stim).
  3. Features X[i,:] = noisy <H> of training-circuit-i's neighbor circuits (Aer).
  4. Fit combine c by Lasso with ||c||_1 <= gamma (gamma=2 for Pauli-insertion neighbors).
  5. Oracle value = the training MSE (NIL Lemma 1: equals the target's mitigation MSE).
  6. Ground truth: apply c to the target's neighbor energies -> mitigated energy; compare
     to the exact ground energy for the true mitigated |dE|.

Note vs Paper 2: we fix the *noiseless*-optimal angles as the target (design-doc choice)
rather than re-optimizing under noise, so absolute |dE| differ from their Table 3. What we
test is whether the training-MSE *ranks* ansatze by true mitigated error.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2

from ansatze import ANSATZE, optimal_angles, two_qubit_gate_count
from molecules import h2_hamiltonian
from neighbors import NeighborMap
from noise import build_noise_model
from training import build_training_circuits, clifford_label, h_terms_for_stim

GAMMA_PAULI = 2.0  # L1 budget for Pauli-insertion neighbors (Chen et al.)


def noisy_energies(circuits, H: SparsePauliOp, noise_model, shots=None, seed=0, batch=3000):
    """Noisy <H> for each circuit via Aer EstimatorV2. shots=None -> exact (density matrix)."""
    opts = {
        "backend_options": {"noise_model": noise_model, "seed_simulator": seed},
        "default_precision": 0.0 if shots is None else 1.0 / np.sqrt(shots),
    }
    est = EstimatorV2(options=opts)
    out = []
    for i in range(0, len(circuits), batch):
        res = est.run([(c, H) for c in circuits[i : i + batch]]).result()
        out.extend(float(r.data.evs) for r in res)
    return np.array(out)


def lasso_combine(X: np.ndarray, y: np.ndarray, gamma: float = GAMMA_PAULI):
    """min (1/T)||X c - y||^2  s.t.  ||c||_1 <= gamma.  Returns (c, training_mse)."""
    T, N = X.shape
    c = cp.Variable(N)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(X @ c - y) / T), [cp.norm1(c) <= gamma])
    prob.solve()
    c_val = np.asarray(c.value).ravel()
    train_mse = float(np.mean((X @ c_val - y) ** 2))
    return c_val, train_mse


# Cache the (expensive-ish) classical angle optimization per ansatz.
_ANGLE_CACHE: dict[str, tuple[np.ndarray, float]] = {}


def _target(name: str, H: SparsePauliOp):
    if name not in _ANGLE_CACHE:
        _ANGLE_CACHE[name] = optimal_angles(ANSATZE[name](), H, n_restarts=20)
    theta, e_noiseless = _ANGLE_CACHE[name]
    return ANSATZE[name]().assign_parameters(theta), e_noiseless


def run_nil(
    name: str,
    H: SparsePauliOp,
    e_exact: float,
    *,
    max_neighbors: int = 20,
    n_train: int = 300,
    shots=None,
    combined: bool = True,
    seed: int = 0,
):
    ansatz = ANSATZE[name]()
    target, e_noiseless = _target(name, H)
    noise_model = build_noise_model(combined)
    neigh = NeighborMap(ansatz, max_neighbors=max_neighbors, seed=seed)
    terms = h_terms_for_stim(H)

    # Training labels (exact, stim) and noisy features (Aer).
    train_circs, _ = build_training_circuits(ansatz, n_train, seed=seed)
    y = np.array([clifford_label(c, terms) for c in train_circs])
    neighbor_circs = [nc for c in train_circs for nc in neigh.materialize(c)]
    feats = noisy_energies(neighbor_circs, H, noise_model, shots=shots, seed=seed)
    X = feats.reshape(n_train, len(neigh))

    c_coef, train_mse = lasso_combine(X, y, GAMMA_PAULI)

    # Apply combine to the real target.
    x_target = noisy_energies(neigh.materialize(target), H, noise_model, shots=shots, seed=seed + 1)
    mitigated = float(x_target @ c_coef)
    unmitigated = float(x_target[0])  # recipe[0] is the identity (no-insertion) neighbor

    return {
        "name": name,
        "two_qubit_gates": two_qubit_gate_count(ansatz),
        "n_neighbors": len(neigh),
        "train_mse": train_mse,            # <-- the NIL oracle value
        "noiseless_target": e_noiseless,
        "unmitigated_energy": unmitigated,
        "mitigated_energy": mitigated,
        "unmitigated_de": abs(unmitigated - e_exact),
        "mitigated_de": abs(mitigated - e_exact),
        "mitigation_residual": abs(mitigated - e_noiseless),  # pure noise residual (excl. ansatz limit)
    }


if __name__ == "__main__":
    from molecules import exact_ground_state

    H = h2_hamiltonian()
    e_exact, _ = exact_ground_state(H)
    r = run_nil("C7", H, e_exact, max_neighbors=20, n_train=150, shots=None, seed=0)
    for k, v in r.items():
        print(f"  {k:20}: {v:+.5f}" if isinstance(v, float) else f"  {k:20}: {v}")
