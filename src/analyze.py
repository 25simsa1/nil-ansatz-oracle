"""Dig into what the NIL training-MSE actually predicts.

The headline correlation (train-MSE vs total mitigated |dE|) is strongly negative. This
script tests the mechanism: total |dE| mixes (a) irreducible ansatz-representation error
and (b) noise-mitigation residual. NIL's Lemma 1 only promises the training-MSE tracks
(b) -- the noise part -- so we correlate against each component, and we re-run with the
zero-noise pathological circuit C10 removed.
"""

from __future__ import annotations

import json
import os
import sys

from scipy.stats import spearmanr

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_all12_exact.json")


def _rho(x, y):
    rho, p = spearmanr(x, y)
    return f"rho={rho:+.3f} (p={p:.3f})"


def main(path=RESULTS):
    rows = json.load(open(path))["rows"]
    by = {r["name"]: r for r in rows}

    def col(key, rows_):
        return [r[key] for r in rows_]

    print(f"{'ansatz':5} {'train_MSE':>11} {'mit|dE|':>9} {'unmit|dE|':>9} {'noise_resid':>11} {'2Q':>3}")
    for r in rows:
        print(f"{r['name']:5} {r['train_mse']:>11.2e} {r['mitigated_de']:>9.4f} "
              f"{r['unmitigated_de']:>9.4f} {r['mitigation_residual']:>11.4f} {r['two_qubit_gates']:>3}")

    print("\n--- what does NIL train-MSE predict? (all 12) ---")
    print(f"  train-MSE vs total mitigated |dE|      : {_rho(col('train_mse', rows), col('mitigated_de', rows))}")
    print(f"  train-MSE vs noise residual |mit-ideal|: {_rho(col('train_mse', rows), col('mitigation_residual', rows))}")
    print(f"  train-MSE vs unmitigated noisy |dE|    : {_rho(col('train_mse', rows), col('unmitigated_de', rows))}")

    # C10 has zero noisy gates -> zero noise -> train-MSE ~ 0 but huge ansatz error. Remove it.
    no_c10 = [r for r in rows if r["name"] != "C10"]
    print("\n--- excluding C10 (the zero-noise, ansatz-limited outlier) ---")
    print(f"  train-MSE vs total mitigated |dE|      : {_rho(col('train_mse', no_c10), col('mitigated_de', no_c10))}")
    print(f"  train-MSE vs noise residual            : {_rho(col('train_mse', no_c10), col('mitigation_residual', no_c10))}")
    print(f"  train-MSE vs unmitigated noisy |dE|    : {_rho(col('train_mse', no_c10), col('unmitigated_de', no_c10))}")
    print(f"  2Q-gate count vs total mitigated |dE|  : {_rho(col('two_qubit_gates', no_c10), col('mitigated_de', no_c10))}")

    # How well does NIL mitigate, circuit by circuit (unmitigated -> mitigated)?
    print("\n--- mitigation effectiveness (does NIL reduce the noisy error?) ---")
    helped = sum(1 for r in rows if r["mitigated_de"] < r["unmitigated_de"] - 1e-9)
    print(f"  NIL reduced |dE| in {helped}/12 circuits; "
          f"mean unmit={sum(col('unmitigated_de', rows))/12:.4f} -> mean mit={sum(col('mitigated_de', rows))/12:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RESULTS)
