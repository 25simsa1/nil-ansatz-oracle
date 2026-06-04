"""Milestone check #2 as a regression test: NIL mitigation must beat no mitigation.

Runs the full NIL pipeline on a cheap setting and asserts the mitigated energy error is
below the unmitigated noisy error. Uses exact-noisy features (no shot noise) so the check
is deterministic; kept small so it runs in a few seconds.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from molecules import exact_ground_state, h2_hamiltonian  # noqa: E402
from nil import run_nil  # noqa: E402


@pytest.mark.parametrize("name", ["C2", "C7"])
def test_nil_beats_unmitigated(name):
    H = h2_hamiltonian()
    e_exact, _ = exact_ground_state(H)
    r = run_nil(name, H, e_exact, max_neighbors=15, n_train=80, shots=None, seed=0)
    assert r["mitigated_de"] < r["unmitigated_de"], (
        f"{name}: mitigated |dE|={r['mitigated_de']:.4f} should be < "
        f"unmitigated |dE|={r['unmitigated_de']:.4f}"
    )
    # And mitigation should land close to the noiseless target (small residual).
    assert r["mitigation_residual"] < 0.05


if __name__ == "__main__":
    H = h2_hamiltonian()
    e_exact, _ = exact_ground_state(H)
    for name in ["C2", "C7"]:
        r = run_nil(name, H, e_exact, max_neighbors=15, n_train=80, shots=None, seed=0)
        ok = r["mitigated_de"] < r["unmitigated_de"]
        print(f"{name}: unmit={r['unmitigated_de']:.4f} -> mit={r['mitigated_de']:.4f}  "
              f"residual={r['mitigation_residual']:.4f}  {'PASS' if ok else 'FAIL'}")
