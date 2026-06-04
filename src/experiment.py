"""The actual experiment: does the NIL training-MSE rank ansatze by true mitigated error
better than the cheap baselines (2-qubit gate count, expressibility)?

For each ansatz we collect:
  - train_mse        : the NIL oracle (classically computable)
  - mitigated_de     : the true mitigated |dE| vs the exact ground energy (ground truth)
  - unmitigated_de   : noisy |dE| with no mitigation (context / stability baseline)
  - two_qubit_gates  : free topology metric
  - expressibility   : Paper 2's KL-from-Haar metric

then report Spearman rank correlations of each predictor against the true mitigated |dE|,
plus the ranking-stability check (unmitigated vs mitigated). With only a few ansatze these
are suggestive, not conclusive -- the point is the full 12-circuit run.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.stats import spearmanr

from ansatze import ANSATZE
from expressibility import expressibility
from molecules import exact_ground_state, h2_hamiltonian
from nil import run_nil

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_experiment(names, *, shots=None, n_train=300, max_neighbors=20, expr_samples=5000, seed=0):
    H = h2_hamiltonian()
    e_exact, _ = exact_ground_state(H)
    rows = []
    for name in names:
        t0 = time.time()
        r = run_nil(name, H, e_exact, max_neighbors=max_neighbors, n_train=n_train, shots=shots, seed=seed)
        r["expressibility"] = expressibility(ANSATZE[name](), n_samples=expr_samples, seed=seed)
        r["seconds"] = time.time() - t0
        rows.append(r)
        print(
            f"  {name}: train_mse={r['train_mse']:.2e}  mit|dE|={r['mitigated_de']:.4f}  "
            f"unmit|dE|={r['unmitigated_de']:.4f}  2Q={r['two_qubit_gates']}  "
            f"expr={r['expressibility']:.3f}  ({r['seconds']:.0f}s)"
        )
    return rows, e_exact


def _spearman(x, y):
    if len(x) < 2:
        return float("nan"), float("nan")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def summarize(rows):
    mit = [r["mitigated_de"] for r in rows]
    predictors = {
        "NIL train-MSE": [r["train_mse"] for r in rows],
        "2Q gate count": [r["two_qubit_gates"] for r in rows],
        "expressibility": [r["expressibility"] for r in rows],
    }
    print("\nSpearman rho vs true mitigated |dE|  (predictor that best tracks reality wins):")
    corr = {}
    for label, vals in predictors.items():
        rho, p = _spearman(vals, mit)
        corr[label] = {"rho": rho, "p": p}
        print(f"  {label:16}: rho = {rho:+.3f}  (p = {p:.3f})")

    rho_stab, p_stab = _spearman([r["unmitigated_de"] for r in rows], mit)
    corr["stability (unmit vs mit)"] = {"rho": rho_stab, "p": p_stab}
    print(f"\nRanking stability  unmitigated->mitigated : rho = {rho_stab:+.3f}  (p = {p_stab:.3f})")
    print("  (high+ = NIL preserves the noisy ranking like ZNE; low/negative = scrambles like PEC)")
    return corr


def main(names=("C2", "C7", "C9"), shots=None, n_train=300, tag="run"):
    print(f"Experiment over {list(names)}  (shots={shots}, n_train={n_train})\n")
    rows, e_exact = run_experiment(names, shots=shots, n_train=n_train)
    corr = summarize(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"experiment_{tag}.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "correlations": corr, "e_exact": e_exact}, f, indent=2)
    print(f"\nsaved -> {out}")
    return rows, corr


if __name__ == "__main__":
    main(names=("C2", "C7", "C9"), shots=None, n_train=200, tag="three_exact")
