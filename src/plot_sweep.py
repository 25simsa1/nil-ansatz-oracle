"""Plot the NIL-oracle correlation vs shot count (the 'shot noise rehabilitates the oracle'
transition), with std-over-seeds error bands. Reads results/shot_sweep.json."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
EXACT_X = 1e4  # where to draw the infinite-shot (exact) point on the log axis


def _agg(cells, level, key):
    v = [c[key] for c in cells if c["shots"] == level]
    return np.mean(v), np.std(v)


def main(infile="shot_sweep.json", outfile="shot_sweep.png"):
    data = json.load(open(os.path.join(RESULTS_DIR, infile)))
    cells = data["cells"]
    levels = []
    for c in cells:
        if c["shots"] not in levels:
            levels.append(c["shots"])
    finite = sorted([l for l in levels if l != "exact"])
    xs = finite + (["exact"] if "exact" in levels else [])
    xpos = [float(l) for l in finite] + ([EXACT_X] if "exact" in levels else [])

    series = {
        "rho_mse_total": ("NIL train-MSE  vs  total |ΔE|", "tab:red", "o"),
        "rho_mse_residual": ("NIL train-MSE  vs  noise residual", "tab:blue", "s"),
        "rho_2q_total": ("2Q gate count  vs  total |ΔE|", "tab:gray", "^"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, (label, color, marker) in series.items():
        means = np.array([_agg(cells, l, key)[0] for l in xs])
        stds = np.array([_agg(cells, l, key)[1] for l in xs])
        ax.plot(xpos, means, marker=marker, color=color, label=label, lw=2)
        ax.fill_between(xpos, means - stds, means + stds, color=color, alpha=0.15)

    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(l) for l in finite] + (["∞\n(exact)"] if "exact" in levels else []))
    ax.set_xlabel("shots per expectation value")
    ax.set_ylabel("Spearman ρ  (mean ± std over seeds)")
    ax.set_title("NIL training-MSE as an oracle: correlation vs shot count")
    ax.legend(loc="center left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, outfile)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
