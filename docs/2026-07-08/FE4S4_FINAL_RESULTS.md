# Fe4S4 (72q) UHF-SQD — final results & CISD ablation

**System:** [4Fe-4S] active space, 72 qubits, norb = 36, 27α + 27β electrons, singlet.
**Sample:** IBM `ibm_kingston`, 5×10⁶ shots → 1,955,531 kept (reset-mitigation + DD-XY4 +
measurement twirling; ~40% shot retention). **Solver:** GB200 GPU (`diag-gpu_uhf`) on ROQUO,
Prefect `slurm` target. Pool reused across all runs (`quantum_source=saved`).

## Headline

Classical **CISD injection** (forcing the full singles+doubles manifold into the SQD subspace)
reaches **E = −326.8055 Ha**, approaching UCCSD, from a subspace built on a single noisy Kingston
sample — reproduced across 4 independent runs. Plain sample-only recovery reaches only −326.787
after 45 iterations.

## Reference values (`runs/fe4s4_refs.json`)

| Method | Energy (Ha) |
|---|---|
| UHF | −325.998743 |
| UCCSD | −326.867807 |
| CCSD(T) | −327.176189 |

## Results

| Run | Subspace content | sqd_dim | Best E (Ha) | vs UHF | vs UCCSD |
|---|---|---|---|---|---|
| Baseline (job 2894) | sample only, K=4, 12 steps | 9e6 | −326.5531 | −554 mHa | +315 mHa |
| Extended (job 2965) | sample only, K=8, **45 steps** | 3e7 | −326.7868 | −788 mHa | +81 mHa |
| **CISD S+D (job 3691)** | **full singles+doubles** | 1.67e8 | **−326.8055** | **−807 mHa** | **+62 mHa** |
| CISD + higher-exc (job 3720) | full S+D + ~7000 sampled higher | 4e8 | −326.8058 | −807 mHa | +62 mHa |
| Partial-CISD mix (job 3724) | ½ S+D + ½ sampled higher (frac 0.5) | 5e7 | −326.5244 | −526 mHa | +343 mHa |

(CISD S+D reproduced at 1.6e8 job 3177 = −326.8044 and 2.2e8 job 3214 = −326.8052.)

## The ablation — where the correlation lives

The three CISD variants form a controlled ablation that pins down the physics:

- **Full CISD (S+D): −326.8055.** The determinants that matter.
- **Full CISD + 7000 sampled higher-excitations: −326.8058** — adding thousands of the sample's
  triples/higher configs changes the energy by **< 1 mHa**. ⇒ the sampled higher-excitations are
  **noise**, not signal.
- **Replace half the doubles with sampled higher-excitations (frac 0.5): −326.5244** — dropping
  ~9000 doubles for ~3500 sampled higher configs costs **+281 mHa**. ⇒ the **doubles are essential**.

**Conclusion:** at this level, essentially all recoverable correlation for Fe4S4 lives in the
singles+doubles manifold. The quantum sample is noise-limited above double excitations, so classical
CISD injection cleanly supplies the S+D contribution the hardware misses, landing near UCCSD. The
CISD energy (−326.805) is a genuine ceiling for this reference — the batch spread is ~0 mHa and the
recovery trajectory is flat (steps 1→4: −326.8052 → −326.8055).

## Convergence behavior

- **fractional occupancy** starts at 64/72 (strongly multireference, expected for Fe–S) and falls
  as each subspace converges.
- **CISD trajectory is flat** (converged from step 1): −326.80520, −326.80546, −326.80547,
  −326.80548 — the S+D subspace is self-consistent, so more iterations / K-batches don't help.
- **K-batch spread ≈ 0** for CISD (deterministic seed) — K adds nothing once the seed is forced.

## Path below −326.805 (future work, not single-GPU-tonight)

1. **Triples in the seed (CISDT):** deterministic triple excitations — but ~10⁶ dets, needs a much
   larger subspace / distributed multi-GPU Davidson.
2. **Cleaner / deeper quantum sample:** lower-noise device or more shots so the sampled
   higher-excitations become real signal.
3. **Orbital-rotation / LUCJ optimization:** optimize the reference so CISD in the rotated basis
   captures more correlation. This is the most promising near-term lever toward DMRG accuracy.

## Provenance (ROQUO job IDs)

Sample 2754 · baseline recovery 2894 · extended 2965 (45 steps) · CISD S+D 3177/3214/3691 ·
CISD+higher 3720 · partial-mix 3724. Kingston pool backed up at
`sweep/fe4s4_pools/raw_samples_uhf_kingston.npz`. All recovery JSON under `runs/fe4s4_uhf/recover/`.
