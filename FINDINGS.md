# Findings: is the NIL training-MSE an ansatz-selection oracle?

**Short answer: not a robust one — but the failure is precise and informative.** What is
robust across every test is that NIL's training-MSE predicts the *noise-mitigation residual*
(exactly what Chen et al.'s Lemma 1 promises). What is *not* robust is its link to the
quantity Paper 2 actually cares about — total VQE error — which is dominated by the ansatz's
representational limit that NIL is structurally blind to. The total-error correlation even
flips sign between the exact-noise and finite-shot regimes.

## Setup (reproduced from the source papers, not assumed)

- **Hamiltonian:** H2, STO-3G, Jordan-Wigner, 4 qubits. Diagonalizes to −1.137284 Ha
  (Paper 2's anchor: −1.1373).
- **Noise:** Paper 2's COMBINED model — depolarizing + thermal relaxation, applied *only*
  to RY and CX gates (p1=0.005, T1=20µs, T2=12µs, 70ns; p2=0.035, 320ns). A consequence:
  circuits built from CZ/H/RX (e.g. C10, C11) carry little or no noise.
- **Circuits:** all 12 of Paper 2's H2 ansätze, reproduced gate-for-gate from Fig. 4.
- **Target:** classically pre-optimized noiseless angles (design-doc simplification), so we
  test mitigation quality per architecture, not optimizer luck. Absolute |ΔE| therefore
  differ from Paper 2's noise-aware-reoptimized Table 3.
- **NIL:** weight-1 Pauli neighbors (subsampled to 20 + identity), T=300 2-design Clifford
  training circuits, exact stim labels, L1-constrained Lasso (‖c‖₁ ≤ γ=2). Run in two
  feature modes: exact-noisy (density matrix) and finite-shot (Ns=1000).

## Headline: Spearman ρ vs true mitigated |ΔE| (the oracle's stated job)

| predictor | exact-noisy | 1000 shots |
|---|---:|---:|
| **NIL train-MSE** | **−0.755** (p=0.005) | **+0.490** (p=0.106) |
| 2Q gate count | +0.358 (p=0.253) | +0.461 (p=0.132) |
| expressibility | +0.291 (p=0.359) | +0.112 (p=0.729) |

The NIL oracle's correlation with total error is **sign-unstable**: strongly anti-predictive
in the exact limit, mildly positive under shots — and in neither regime does it *decisively*
beat the free 2-qubit-gate-count baseline it was meant to beat (under shots it merely ties).

## What *is* robust: it predicts the noise it can actually see

Splitting total error into (a) irreducible ansatz-representation error and (b) the
noise-mitigation residual — Lemma 1 only ties train-MSE to (b):

| train-MSE vs … | exact-noisy | 1000 shots | 1000 shots, excl. C10 |
|---|---:|---:|---:|
| total mitigated \|ΔE\| | −0.755 | +0.490 | +0.391 |
| **noise residual \|mit − ideal\|** | **+0.510** | **+0.545** | **+0.618 (p=0.043)** |
| unmitigated noise \|ΔE\| | +0.161 | +0.469 | +0.345 |

Against the noise residual the training-MSE is *positively* correlated in every case — it
behaves exactly as the theory says. The negative/unstable headline comes entirely from (a).

## The mechanism: C10 and the shot-noise floor

**C10** is the smoking gun. It is built from H/RX/CZ — all noiseless under this model — so it
has essentially no noise to mitigate, yet its |ΔE| ≈ 0.61 is the worst of all (pure
representational failure). NIL cannot see representational error:

- **Exact-noisy:** C10's train-MSE = 3×10⁻²⁰ (≈0). A near-zero oracle value paired with the
  worst true error single-handedly drives the −0.755 anti-correlation.
- **1000 shots:** shot variance floors every circuit's train-MSE at ~10⁻³, so C10's becomes
  1×10⁻³ — no longer an outlier. With the floor in place the total-error correlation swings
  to +0.49. This is precisely the "shot noise helps NIL (acts like L2 regularization)" effect
  the design doc anticipated: noise that hurts a single estimate *rehabilitates the oracle*.

## NIL as a mitigator, and ranking stability

Separate from the oracle question, NIL mitigates well: it reduced |ΔE| in 11/12 circuits
(exact, mean 0.179→0.068) and 10/12 (shots, mean 0.169→0.087); the misses are C10/C11, which
have ~no noise. Under shots the mitigated ranking is **stable** vs the unmitigated one
(ρ = +0.769, p=0.003) — ZNE-like, *not* PEC-like (Paper 2: ZNE +0.80, PEC −0.22). For
comparison Paper 2's ZNE helped 4/12 and PEC 1/12, but a fair head-to-head must equalize
shot budget (their PEC used only 200 quasi-prob samples), so we claim effective, stable
mitigation — not a clean win over their methods.

## Takeaway

A concrete, falsifiable answer to Paper 2's open question ("we lack a scalable
noisy-performance predictor"): **NIL's training-MSE does not robustly fill that gap.** It is
a sound *noise-mitigability* estimator (robustly positive vs the noise residual, as Lemma 1
guarantees), but ansatz selection is governed by representational capacity, not mitigability,
and the two can be anti-aligned. A practitioner ranking ansätze by NIL-MSE would be misled in
the exact limit and would, under realistic shots, do no better than counting two-qubit gates.

## Honest caveats

- Fixed-noiseless-angle targets (not noise-aware re-optimization): a deliberate
  simplification; absolute |ΔE| differ from Paper 2's Table 3.
- 12 data points: suggestive, not conclusive; p-values are mostly > 0.05. C10 is a
  high-leverage outlier, reported with and without it.
- Lemma 1's MSE-equality assumes fixed Pauli-ish noise as angles vary; on hardware both hold
  only approximately.
- The CZ/H/RX-noiseless modeling (faithful to Paper 2) makes several circuits low-noise by
  construction, which is part of why mitigability and total error decouple.
- The total-error correlation's sign-dependence on shot count means any single-number claim
  is regime-specific; we report both.
