# nil-ansatz-oracle

**Can the NIL training-MSE act as a classically-computable oracle for how well a VQE
ansatz will perform under noise?**

This is a simulation-first experiment connecting two papers:

- **Paper 1 — NIL** (Chen et al., [arXiv:2512.12578](https://arxiv.org/abs/2512.12578),
  *Scalable Quantum Error Mitigation with Neighbor-Informed Learning*). NIL learns to
  predict a circuit's ideal output from the noisy outputs of its "neighbor" circuits.
  Its 2-design training method has a key property (their Lemma 1): the average
  mitigation MSE on the classically-simulable Clifford **training** circuits equals the
  MSE on the intractable **target** circuit.
- **Paper 2 — Ansatz selection** (Annis/Kassem/Coleman,
  [arXiv:2606.04955](https://arxiv.org/abs/2606.04955), *Expressibility, Noise, and
  Error Mitigation in VQE Ansatz Selection*). Tests 12 H2 ansätze under noise and finds
  that circuit rankings scramble (ideal→noisy Spearman ρ ≈ −0.1), that expressibility is
  an unreliable predictor, and that cheap topology metrics (two-qubit gate count) are
  surprisingly competitive. ZNE largely preserves rankings (ρ = +0.80) while PEC
  reorders them (ρ = −0.22).

## The hypothesis

If NIL's training-MSE equals the target MSE, then it is a *classically computable*
measure of "how mitigable is this ansatz" — exactly the scalable noisy-performance
predictor Paper 2 says is missing. We test whether that MSE ranks the ansätze by their
true ground-state-energy error **better than** the free baselines (2-qubit gate count,
expressibility), and whether the resulting ranking is **stable** (ZNE-like, good) or
**scrambled** (PEC-like, bad).

Everything runs on a noisy Qiskit Aer simulator. No QPU, no hardware spend.

## Status

Reproduced from the source papers (not guessed):

- H2 Hamiltonian (STO-3G, JW, 4 qubits) validated by diagonalization → −1.137 Ha.
- Noise model matches Paper 2's COMBINED model exactly (noise only on RY + CNOT).
- The three starter ansätze (C2, C7, C9) reproduced from Paper 2's Fig. 4.

**Result (all 12 H2 ansätze, shot-count sweep × seeds): see [`FINDINGS.md`](FINDINGS.md) and
`results/shot_sweep.png`.** In short — NIL's training-MSE robustly predicts the
*noise-mitigation residual* (ρ=+0.63±0.04 in the exact limit, as its Lemma 1 guarantees), but
it is *not* an ansatz-selection oracle: it never out-predicts free two-qubit-gate count
(steady ρ≈+0.5) for total VQE error at any shot count, and in the exact limit its total-error
correlation is consistent with zero (−0.19±0.31 over seeds — an earlier single-seed run's
eye-catching −0.76 did not survive error bars). Total error is dominated by representational
limits NIL can't see. NIL does mitigate effectively (helps 10–11/12 circuits) and stably
(ZNE-like, ρ=+0.77).

See `CLAUDE.md` for the full experiment design and `src/` for the pipeline.

## Layout

```
src/
  molecules.py      H2 Hamiltonian + exact ground state
  noise.py          COMBINED + DEPOL noise models
  ansatze.py        Paper-2 H2 ansatz circuits
  neighbors.py      weight-1 Pauli neighbor generation
  training.py       2-design Clifford training set + stim labels + noisy features
  nil.py            Lasso fit (L1-constrained), combine, training-MSE
  expressibility.py KL-from-Haar metric
  experiment.py     runs all ansätze, builds the correlation table
tests/              stim-vs-statevector and NIL-recovers sanity checks
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
