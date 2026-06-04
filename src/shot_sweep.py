"""Map how the NIL-MSE-vs-error correlation depends on shot count.

The single-run result showed a sign flip: train-MSE vs total |dE| is rho=-0.76 with exact
(infinite-shot) features but +0.49 at 1000 shots. This sweep tests whether that is a real,
monotone transition (shot-noise floor rehabilitating the oracle) or just seed noise, by
running the full 12-ansatz experiment across several shot levels and seeds and reporting
mean +/- std of the rank correlations.

For each (shots, seed) we record, over the 12 ansatze:
  - Spearman(train-MSE, total mitigated |dE|)      -- the oracle's stated job
  - Spearman(train-MSE, noise residual |mit-ideal|) -- what Lemma 1 actually promises
  - Spearman(2Q gate count, total mitigated |dE|)  -- the free baseline
Results are checkpointed to results/shot_sweep.json after every cell.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.stats import spearmanr

from ansatze import ANSATZE, two_qubit_gate_count
from molecules import exact_ground_state, h2_hamiltonian
from nil import run_nil

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
NAMES = [f"C{i}" for i in range(1, 13)]
TWO_Q = {n: two_qubit_gate_count(ANSATZE[n]()) for n in NAMES}


def _rho(x, y):
    return float(spearmanr(x, y)[0])


def sweep(shot_levels, seeds, n_train=150, max_neighbors=20, out="shot_sweep.json"):
    H = h2_hamiltonian()
    e_exact, _ = exact_ground_state(H)
    cells = []  # one per (shots, seed)
    t0 = time.time()
    for shots in shot_levels:
        for seed in seeds:
            rows = {n: run_nil(n, H, e_exact, max_neighbors=max_neighbors,
                               n_train=n_train, shots=shots, seed=seed) for n in NAMES}
            mse = [rows[n]["train_mse"] for n in NAMES]
            tot = [rows[n]["mitigated_de"] for n in NAMES]
            res = [rows[n]["mitigation_residual"] for n in NAMES]
            g2 = [TWO_Q[n] for n in NAMES]
            cell = {
                "shots": shots if shots is not None else "exact",
                "seed": seed,
                "rho_mse_total": _rho(mse, tot),
                "rho_mse_residual": _rho(mse, res),
                "rho_2q_total": _rho(g2, tot),
            }
            cells.append(cell)
            print(f"shots={str(cell['shots']):>5} seed={seed}: "
                  f"mse-vs-total={cell['rho_mse_total']:+.3f}  "
                  f"mse-vs-resid={cell['rho_mse_residual']:+.3f}  "
                  f"2q-vs-total={cell['rho_2q_total']:+.3f}  ({time.time()-t0:.0f}s)")
            os.makedirs(RESULTS_DIR, exist_ok=True)
            json.dump({"cells": cells, "n_train": n_train, "max_neighbors": max_neighbors},
                      open(os.path.join(RESULTS_DIR, out), "w"), indent=2)

    # Aggregate over seeds.
    print("\n=== mean +/- std over seeds ===")
    print(f"{'shots':>6} {'mse-vs-total':>16} {'mse-vs-resid':>16} {'2q-vs-total':>16}")
    levels = [s if s is not None else "exact" for s in shot_levels]
    for lvl in levels:
        sub = [c for c in cells if c["shots"] == lvl]
        def ms(key):
            v = [c[key] for c in sub]
            return f"{np.mean(v):+.3f}+/-{np.std(v):.3f}"
        print(f"{str(lvl):>6} {ms('rho_mse_total'):>16} {ms('rho_mse_residual'):>16} {ms('rho_2q_total'):>16}")
    return cells


if __name__ == "__main__":
    sweep(shot_levels=[100, 300, 1000, 3000, None], seeds=[0, 1, 2])
