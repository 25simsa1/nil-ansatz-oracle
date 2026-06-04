# NIL-as-Ansatz-Oracle — Simulation-First Experiment

## One-line goal
Test whether the **NIL training-MSE** (Chen et al., arXiv:2512.12578) ranks VQE
ansatz circuits by their true ground-state-energy error **better than** the cheap
baselines (2-qubit gate count, expressibility) — reproducing the circuit set from
Annis/Kassem/Coleman (arXiv:2606.04955). **Everything runs on a noisy simulator. No
QPU, no spend beyond Claude usage.**

## Core hypothesis
NIL's Corollary 2 (Lemma 1 in the published version) says the average mitigation MSE
on the (classically simulable) Clifford **training** circuits equals the MSE on the
intractable **target** circuit. If true, the NIL training-MSE is a classically-computable
oracle for "how mitigable is this ansatz" — which is exactly the scalable noisy-performance
predictor Paper 2 says it lacks. We test whether that MSE correlates with true VQE error
across ansätze, and whether the resulting ranking is **stable** (good, like ZNE: Spearman
+0.80) or **scrambled** (bad, like PEC: -0.22).

## What success / failure looks like
- **Win:** Spearman(NIL-MSE, true |ΔE|) across the ansätze beats Spearman(2Q-gate-count,
  true |ΔE|) and is positive + stable. → the oracle idea works in simulation; worth a
  later hardware confirmation.
- **Informative loss:** NIL-MSE ranking scrambles or just matches gate count. → tells
  us NIL inherits PEC-like instability, or adds nothing over the free metric. Still a
  clean, publishable negative result that connects two papers nobody has connected.

---

## Environment
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# qiskit-aer: noisy simulation. stim: fast Clifford labels. cvxpy: Lasso w/ L1 constraint.
```
Versions are pinned in requirements.txt. Qiskit's API drifts fast — if something errors,
check the current qiskit docs rather than trusting an old snippet.

## Noise model (Paper 2's "COMBINED" — confirmed against the paper)
Apply noise **only to RY and CX/CNOT gates** (Paper 2's choice, matching Saib et al.):
- 1-qubit (RY): depolarizing p1 = 0.005, thermal relaxation T1 = 20 us, T2 = 12 us,
  gate time 70 ns.
- 2-qubit (CX): depolarizing p2 = 0.035, gate time 320 ns.
Build with `qiskit_aer.noise.NoiseModel`. Keep a `DEPOL`-only variant too (p1, p2, no
relaxation) — it's a free ablation and matches Paper 2's controlled test.

## Circuits (start narrow, then widen)
Start with **3 H2 ansätze** (4 qubits) from Paper 2's Fig. 4. Confirmed structures:
- **C2** (best under noise, |ΔE|=0.039): RY(q0..3); CZ(2,3); CZ(1,2); CZ(0,1); CZ(0,3); RY(q0..3). Ring, all CZ.
- **C7** (middle, |ΔE|=0.067): RY(q0..3); CZ(2,3); CX(1->2); CX(0->1); CZ(0,3); RY(q0..3). Ring, mixed cx/cz.
- **C9** (worst non-pathological, |ΔE|=0.231): H(all); RX(q0..3); all-to-all CX; H(all); RX(q0..3).

H2 Hamiltonian: STO-3G, Jordan-Wigner, exact ground energy -1.1373 Ha (truth anchor).
Once the pipeline works, scale to all 12 (C1-C12). Paper 2's ground-truth |ΔE| table:
C2=0.0393, C11=0.0393, C7=0.0667, C6=0.0713, C3=0.0716, C4=0.0738, C8=0.0807, C5=0.0957,
C1=0.1172, C12=0.1848, C9=0.2312, C10=0.6092 (COMBINED/NONE column).

## Key simplification — skip noisy VQE optimization
H2 is exactly solvable. **Pre-compute near-optimal angles classically** (diagonalize
the 4-qubit Hamiltonian, fit each ansatz to the ground state once) and treat the fixed
optimal circuit as the target. This removes the noisy optimization loop entirely — huge
cost/complexity saving and it isolates exactly what we're testing (mitigation quality
per architecture, not optimizer luck).

---

## Pipeline (the NIL core)
For each ansatz C(theta*):

1. **Neighbor map.** Build weight-1 Pauli neighbors: for each noisy gate position,
   insert one of {X, Y, Z} after it -> N neighbor circuits (N ~ 3 x #gates). Randomly
   subsample to ~20-30 for the starter.

2. **2-design training set.** Generate T training circuits by replacing each rotation
   gate RP(theta) with RP(theta0), theta0 drawn uniformly from {0, pi/2, pi, 3pi/2}.
   These are Clifford -> labels y(i) = exact <C(i)> computed with **stim** (free).
   Start T = 300.

3. **Features.** For each training circuit, run all its neighbor circuits on the
   **noisy Aer simulator** with Ns shots (start Ns = 1000), grouping commuting Pauli
   terms into ~3 measurement bases. Feature vector x(i) = the N noisy expectation values.

4. **Fit.** Lasso regression (L1-constrained, use cvxpy; gamma = 2 for Pauli-insertion
   neighbors per the paper) to learn `combine`: y_hat = c . x. Record training MSE.

5. **Oracle value = the training MSE** from step 4 (this is what Lemma 1 says equals
   the target MSE).

6. **Ground truth.** Separately, compute the **true** mitigated error: apply the
   learned c to the *target* circuit's neighbor expectation values, get mitigated
   energy, compare to exact -1.1373 Ha -> true |ΔE_mitigated|. Also record the
   *unmitigated* noisy |ΔE| for context.

## The actual experiment (the point of all this)
Across the 3 (then 12) ansätze, compute Spearman rank correlations:
- NIL training-MSE  vs  true mitigated |ΔE|     <- does the oracle predict reality?
- 2Q-gate-count     vs  true mitigated |ΔE|     <- the baseline to beat
- expressibility    vs  true mitigated |ΔE|     <- the metric Paper 2 questions
- NIL ranking (ideal-params)  vs  NIL ranking (noisy)  <- stability check (ZNE-like or PEC-like?)

Expressibility: KL divergence of the circuit's fidelity distribution from the Haar
distribution P(F) = (N-1)(1-F)^(N-2), N = 2^n, ~5000 random parameter pairs, 75 bins.
(Paper 2 Sec 3.2 / 4.)

---

## First milestone (do this before anything else)
Get **one** ansatz (C7) end-to-end on the simulator and confirm three things:
1. stim labels match a statevector simulation on the Clifford training circuits (test).
2. Lasso `combine` applied to the target reduces |ΔE| below the unmitigated noisy |ΔE|.
3. The whole single-ansatz run completes in minutes on a laptop.
If (2) fails, NIL isn't mitigating at all — debug the neighbor map / noise placement
before scaling. Only after this works do you loop over all 3, then all 12.

## Cost levers (all keep it at $0 hardware)
- Ns down to ~500: shot noise actually *helps* NIL (acts as L2 regularization).
- Subsample N to 20.
- T down toward the empirical scaling T ~ O(ln(N)/eps^2), well under 300.
- Fixed classical angles (above) removes the optimization loop.

## Honest caveats to keep in the writeup
- Lemma 1's equality assumes noise is fixed as angles vary + Pauli-ish neighbors;
  on hardware both are only approximate.
- Any NIL-vs-PEC comparison must equalize total shot budget (Paper 2's PEC used only
  200 quasiprob samples — under-sampled — so don't claim a hollow win).
- H2/H3+ are tiny; correlations on 3-12 points are suggestive, not conclusive. Frame
  accordingly.
