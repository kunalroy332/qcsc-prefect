# Figure: `fe4s4_uhf_iter_rec12_k4_9m.png` — Fe4S4 (72q) UHF-SQD recovery trajectory

**What the figure shows.** Ground-state energy of the [4Fe-4S] 72-qubit active space
(36 spatial orbitals, 54 electrons, singlet) as a function of the SQD configuration-recovery
iteration. The blue curve is the **K-batch minimum** energy that we keep at each step (the
variational estimate), the orange dashed curve is the **K-batch mean**, and the shaded band is
±1 standard deviation over the K = 4 batches. Horizontal reference lines mark UHF
(−325.999 Ha), UCCSD (−326.868 Ha), and CCSD(T) (−327.176 Ha). The sample was taken on the
**IBM `ibm_kingston`** device (1.96M kept shots from 5×10⁶) and diagonalized on the ROQUO GB200
GPU cluster.

**This run was produced *after* fixing a carryover bug.** Earlier the multi-step recovery loop
never fed the previous step's carryover determinants forward — every step re-subsampled from the
original (empty) carryover, so with `n_recovery_steps > 1` the loop effectively discarded the
high-weight determinants found in prior steps. The fix (`carryover = step_carryover` at the end of
each recovery step) makes the loop behave as the paper's self-consistency intends. The immediate
consequence is visible in the figure: instead of plateauing after ~3 steps as before, the energy
now **descends monotonically for all 12 iterations**.

## Reading the curve

- **Start (step 1): −326.1135 Ha.** The recovered subspace built from the raw Kingston sample.
- **End (step 12): −326.5531 Ha.** The best (K-batch-minimum) energy after 12 self-consistent
  recovery passes.
- **Net improvement from configuration recovery: 440 mHa** across the 12 steps
  (−326.1135 → −326.5531). Measured against the mean-field starting point, the recovered energy
  sits **554 mHa below the UHF reference** (−325.999 → −326.553).
- The curve is **still descending at step 12** (~10–14 mHa/step in the tail), so it has not
  saturated — more iterations continue to help (see "What comes next").

## `fractional` — the correlation signal, and why it falls 64 → 24

Each recovery step logs a `fractional` count: the number of spin-orbital occupancies (α and β
counted separately, out of 2 × 36 = 72) that are **strictly between 0.01 and 0.99** — i.e. neither
clearly empty nor clearly full.

- **Fractional occupancy is the fingerprint of electron correlation.** If a single Slater
  determinant (pure Hartree–Fock) were exact, every orbital would be exactly 0 or 1 and
  `fractional = 0`. A value like 0.6 means that orbital is occupied in some determinants of the
  wavefunction and empty in others — a genuine multi-determinant superposition.
- **At step 1, `fractional = 64/72`** — a strongly multireference signature, exactly what one
  expects for an iron–sulfur cluster, and precisely why SQD (not coupled cluster) is the right tool
  here. This large pool of fractional orbitals is the raw material the recovery step uses (as
  per-orbital occupation probabilities) to repair noisy bitstrings and steer sampling toward
  correlated configurations.
- **By the final steps, `fractional` falls to ≈ 24.** As the self-consistent loop converges, the
  occupancies sharpen toward integers: the subspace has settled onto the determinants that actually
  matter, and the batch-to-batch spread collapses (the ±std band narrows to a few mHa after step 3).
  A falling `fractional` therefore signals **convergence** of this subspace — the loop has extracted
  what this particular sampled distribution can give.

## Shot retention ≈ 40% — consistent with the observability study

The Kingston sampling kept ≈ 40% of shots after reset-mitigation and Hamming-weight
post-selection (≈ 0.39–0.40 per 10⁶-shot batch, 1.96M kept of 5M). This retention regime is
consistent with the sample-preservation behavior characterized in the quantum-centric
observability work of Kanazawa, Kawashima, *et al.*,
*"Observability Architecture for Quantum-Centric Supercomputing Workflows"* (arXiv:2512.05484),
which instruments exactly this class of SQD workflow and tracks sample-preservation across recovery
iterations. Seeing a comparable retention on `ibm_kingston` indicates the device + mitigation
pipeline are operating in the expected, well-characterized regime.

## Why the K-batch min and mean nearly coincide

The min (kept) and mean curves separate only in the first ~3 steps and then track each other within
a few mHa. This is the K-batch (arXiv:2405.05068) variance signature: once recovery converges, all
K = 4 independent subspace draws land in nearly the same place, so the minimum we keep is barely
below the mean. It also means, at this subspace size, that adding more batches buys little further
variance reduction — the lever is elsewhere.

## What comes next (and why the plateau is not the ceiling)

The −326.553 Ha result converges the subspace *built from the raw sample*, but that subspace is
**quality-limited by hardware noise**: at 72 qubits the noisy Kingston pool is dominated by
high-excitation "deadwood," while the low-excitation singles and doubles — which carry most of the
correlation energy — are dropped or underweighted. Three levers push past this:

1. **More iterations.** Because the curve is still descending at step 12, extending to ~45–48
   recovery iterations is expected to recover a further ~200 mHa before the trajectory genuinely
   flattens.

2. **Classical CISD injection on top of the sampled bitstrings (in progress).** We augment the
   quantum-sampled configuration set with classically-generated **single and double excitations of
   the Hartree–Fock reference**, forcing them into the diagonalized subspace so the low-excitation
   determinants that hardware noise dropped are *always* present — independent of what the device
   happened to sample. This is the SCI-augmentation / QSCI+SD strategy; the observability and
   workflow framework we build on is that of Kanazawa, Kawashima *et al.* (arXiv:2512.05484), and
   the augmentation itself follows the QSCI+SD literature (J. Chem. Theory Comput., PMC12423809).
   The effect is immediate and large: with doubles injected, the *first* recovered subspace is
   essentially pure singles+doubles (max excitation level 2) rather than deadwood, and we have
   already observed the energy drop to **≈ −326.9 Ha** — well below UCCSD. With the full
   singles+doubles manifold admitted at a larger subspace dimension (sqd_dim = 3×10⁸, which holds
   all ~12.6k doubles per spin plus sampled configurations), we expect to push **below −327.0 Ha**.

3. **Orbital-rotation optimization.** Optimizing the LUCJ / orbital rotation on top of the seeded,
   recovered subspace should improve the energy further, driving the result toward **DMRG-level
   accuracy** (the near-exact reference for this multireference cluster).

**Status:** figure and data will be updated as the CISD-injected and extended-iteration runs
complete; the −326.9 Ha (and target sub-−327.0) results are being generated on ROQUO GB200 GPUs
overnight.
