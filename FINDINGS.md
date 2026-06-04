# Findings: is the NIL training-MSE an ansatz-selection oracle?

**Short answer: no.** Across shot counts and random seeds, NIL's training-MSE never
out-predicts the free two-qubit-gate-count baseline for total VQE error, and in the
exact-noise limit its correlation with total error is consistent with zero. What *is* robust
— with tight error bars — is that the training-MSE predicts the **noise-mitigation residual**
(exactly what Chen et al.'s Lemma 1 promises). The lesson connecting the two papers: NIL-MSE
is a sound *noise-mitigability* estimator, but ansatz selection is governed by representational
capacity, which NIL is structurally blind to, so it does not fill Paper 2's predictor gap.

## Setup (reproduced from the source papers, not assumed)

- **Hamiltonian:** H2, STO-3G, Jordan-Wigner, 4 qubits. Diagonalizes to −1.137284 Ha
  (Paper 2's anchor: −1.1373).
- **Noise:** Paper 2's COMBINED model — depolarizing + thermal relaxation, applied *only*
  to RY and CX gates (p1=0.005, T1=20µs, T2=12µs, 70ns; p2=0.035, 320ns). Consequence:
  circuits built from CZ/H/RX (e.g. C10, C11) carry little or no noise.
- **Circuits:** all 12 of Paper 2's H2 ansätze, reproduced gate-for-gate from Fig. 4.
- **Target:** classically pre-optimized noiseless angles (design-doc simplification), so we
  test mitigation quality per architecture, not optimizer luck. Absolute |ΔE| therefore
  differ from Paper 2's noise-aware-reoptimized Table 3.
- **NIL:** weight-1 Pauli neighbors (subsampled to 20 + identity), 2-design Clifford training
  circuits, exact stim labels, L1-constrained Lasso (‖c‖₁ ≤ γ=2).

## The key experiment: correlation vs shot count, with error bars

We sweep the feature shot count over {100, 300, 1000, 3000, ∞(exact)} and **3 random seeds**
each (T=150), and report the Spearman ρ of each predictor against the 12 ansätze's errors.
See `results/shot_sweep.png`.

| shots | NIL-MSE vs **total** \|ΔE\| | NIL-MSE vs **noise residual** | 2Q-count vs total \|ΔE\| |
|------:|---------------------------:|------------------------------:|-------------------------:|
| 100   | +0.23 ± 0.41 | +0.12 ± 0.48 | +0.50 ± 0.19 |
| 300   | +0.40 ± 0.21 | +0.28 ± 0.29 | +0.51 ± 0.19 |
| 1000  | +0.34 ± 0.18 | +0.10 ± 0.31 | +0.41 ± 0.07 |
| 3000  | +0.43 ± 0.11 | +0.54 ± 0.26 | +0.48 ± 0.23 |
| exact | **−0.19 ± 0.31** | **+0.63 ± 0.04** | +0.53 ± 0.16 |

Three things stand out:

1. **NIL-MSE never beats free gate count for total error.** The 2Q-count baseline sits at a
   steady ρ ≈ +0.5 at every shot level; NIL-MSE matches it at best and collapses to −0.19 in
   the exact limit. Selecting ansätze by NIL-MSE buys you nothing over counting two-qubit gates.
2. **NIL-MSE robustly predicts the noise residual it is designed for** — ρ = +0.63 ± 0.04 in
   the exact limit (tight; the cleanest signal in the study). This is Lemma 1, confirmed.
3. **Error bars matter.** A single earlier run (T=300, seed 0) gave the eye-catching
   ρ = −0.755 (p=0.005) for total error in the exact limit. The 3-seed sweep shows that was an
   unlucky draw: the seed-averaged value is −0.19 ± 0.31, i.e. statistically indistinguishable
   from zero. We report the distribution, not the lucky single number.

## The mechanism: representational error is invisible to NIL

Total error vs the ground state mixes (a) irreducible ansatz-representation error and (b) the
noise-mitigation residual; Lemma 1 only ties NIL-MSE to (b). The clearest illustration is
**C10**: built entirely from noiseless gates (H/RX/CZ), it has essentially no noise to mitigate
(train-MSE ≈ 3×10⁻²⁰ in the exact limit) yet the worst total error of all, |ΔE| ≈ 0.61 — pure
representational failure that NIL cannot see or fix. Because mitigability and representational
quality can be anti-aligned, NIL-MSE is at best uninformative about total error.

## NIL as a mitigator, and ranking stability

Separate from the oracle question, NIL mitigates well: in the full T=300 runs it reduced |ΔE|
in 11/12 circuits (exact, mean 0.179→0.068) and 10/12 (1000 shots, mean 0.169→0.087); the
misses (C10/C11) have ~no noise. Under shots the mitigated ranking is stable vs the unmitigated
one (ρ = +0.77, p=0.003) — ZNE-like, not PEC-like (Paper 2: ZNE +0.80, PEC −0.22). A fair
head-to-head with their ZNE/PEC must equalize shot budget (their PEC used only 200 quasi-prob
samples), so we claim effective, stable mitigation — not a clean win over their methods.

## Takeaway

A concrete, falsifiable answer to Paper 2's open question ("we lack a scalable
noisy-performance predictor"): **NIL's training-MSE does not fill that gap.** It is a sound
noise-mitigability estimator (robustly predicts the noise residual, ρ=+0.63±0.04, as Lemma 1
guarantees), but it never out-predicts free two-qubit-gate count for the total error that
ansatz selection actually cares about — because total error is dominated by representational
capacity, which NIL is blind to. The two papers connect cleanly but negatively: a strong
mitigation oracle is not, for free, an ansatz-selection oracle.

## Honest caveats

- Fixed-noiseless-angle targets (not noise-aware re-optimization): a deliberate
  simplification; absolute |ΔE| differ from Paper 2's Table 3.
- 12 data points: Spearman error bars are large (±0.2–0.5), so most single correlations are
  not individually significant — which is exactly why we sweep seeds and report ± std.
- Lemma 1's MSE-equality assumes fixed Pauli-ish noise as angles vary; on hardware both hold
  only approximately.
- The CZ/H/RX-noiseless modeling (faithful to Paper 2) makes several circuits low-noise by
  construction, part of why mitigability and total error decouple.
