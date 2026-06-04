"""Noise models matching Paper 2's (Annis/Kassem/Coleman) setup.

Paper 2 applies noise *only to RY and CNOT(CX) gates* (following Saib et al.) to isolate
the effect of parameterized and entangling operations. Crucially this means CZ, H, and RX
gates are treated as noiseless -- which is why C2 (a ring of CZ + RY) is the best noisy
performer in their Table 3: only its 8 RY gates carry noise.

COMBINED (primary): depolarizing composed with thermal relaxation.
  RY  : p1 = 0.005, T1 = 20 us, T2 = 12 us, gate time 70 ns
  CX  : p2 = 0.035,                         gate time 320 ns
DEPOL: same depolarizing probabilities, no thermal relaxation (free ablation; also the
  model under which PEC's representation is exact in Paper 2's Section 5.5).
"""

from __future__ import annotations

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
)

# Everything in SI seconds so thermal_relaxation_error gets consistent units.
P1 = 0.005          # 1-qubit depolarizing probability (RY)
P2 = 0.035          # 2-qubit depolarizing probability (CX)
T1 = 20e-6          # s
T2 = 12e-6          # s   (must satisfy T2 <= 2*T1; 12us <= 40us OK)
GATE_TIME_1Q = 70e-9    # s
GATE_TIME_2Q = 320e-9   # s

NOISY_1Q_GATES = ["ry"]
NOISY_2Q_GATES = ["cx"]


def build_noise_model(combined: bool = True) -> NoiseModel:
    """COMBINED (depolarizing + thermal relaxation) if combined else DEPOL (depolarizing only)."""
    nm = NoiseModel()

    err_1q = depolarizing_error(P1, 1)
    if combined:
        relax_1q = thermal_relaxation_error(T1, T2, GATE_TIME_1Q)
        err_1q = err_1q.compose(relax_1q)
    nm.add_all_qubit_quantum_error(err_1q, NOISY_1Q_GATES)

    err_2q = depolarizing_error(P2, 2)
    if combined:
        relax_2q = thermal_relaxation_error(T1, T2, GATE_TIME_2Q).tensor(
            thermal_relaxation_error(T1, T2, GATE_TIME_2Q)
        )
        err_2q = err_2q.compose(relax_2q)
    nm.add_all_qubit_quantum_error(err_2q, NOISY_2Q_GATES)

    return nm


if __name__ == "__main__":
    for combined in (True, False):
        nm = build_noise_model(combined)
        label = "COMBINED" if combined else "DEPOL"
        print(f"{label}: noisy instructions = {sorted(nm.noise_instructions)}")
