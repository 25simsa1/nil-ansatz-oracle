"""Milestone check #1: stim Clifford labels must match exact statevector <H>.

This also pins down the SQRT_Y / SQRT_Y_DAG orientation in the stim mapping -- if it were
flipped, the off-diagonal H terms would disagree and these assertions would fail.
"""

import os
import sys

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ansatze import ANSATZE  # noqa: E402
from molecules import h2_hamiltonian  # noqa: E402
from training import build_training_circuits, clifford_label, h_terms_for_stim  # noqa: E402

H = h2_hamiltonian()
TERMS = h_terms_for_stim(H)


@pytest.mark.parametrize("name", list(ANSATZE))
def test_stim_matches_statevector(name):
    circuits, _ = build_training_circuits(ANSATZE[name](), n_train=40, seed=hash(name) % 1000)
    for c in circuits:
        stim_val = clifford_label(c, TERMS)
        exact = float(Statevector(c).expectation_value(H).real)
        assert abs(stim_val - exact) < 1e-9, f"{name}: stim {stim_val} vs exact {exact}"


if __name__ == "__main__":
    max_err = 0.0
    for name, builder in ANSATZE.items():
        circuits, _ = build_training_circuits(builder(), n_train=50, seed=7)
        errs = [
            abs(clifford_label(c, TERMS) - float(Statevector(c).expectation_value(H).real))
            for c in circuits
        ]
        m = max(errs)
        max_err = max(max_err, m)
        print(f"{name}: 50 random Clifford circuits, max |stim - statevector| = {m:.2e}")
    print(f"\nOVERALL max error = {max_err:.2e}  ->  {'PASS' if max_err < 1e-9 else 'FAIL'}")
