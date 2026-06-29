# SQD-over-UHF on Fugaku × IBM: Hyperparameters, Findings, and Conclusions

A presentation-oriented walkthrough of the UHF SQD/LUCJ pipeline: every hyperparameter (what it
does, where it lives in the repo), the commit-by-commit story of what we changed and why, the
results that drove each change, and direct answers to the four diagnostic questions.

---

## 1. The pipeline in one diagram

```
PySCF UHF + UCCSD                 IBM ibm_fez (50q)              Fugaku (diag_uhf, C++/MPI)
  integrals, t2, occ   ──►  LUCJ circuit ──► sample bitstrings ──►  configuration recovery
  (chem.py)                 (lucj.py)        (sqd.py)               + post-select + subsample
        │                                                          + exact CI diagonalization
        └──────────── differential evolution (outer loop, main.py) ◄────── energy/occ/carryover
```

Two orthogonal axes were added: **`solver_mode` ∈ {cpu, gpu, fugaku}** (backend) and
**`method` ∈ {rhf, uhf}** (electronic structure). UHF is fully additive; RHF is byte-for-byte
unchanged.

---

## 2. Hyperparameter catalogue

All defaults are the current values in the repo. "Where" gives `file:field`.

### 2A. Circuit / parametrization — `algorithms/sbd/sbd/flow_params.py` (`CircuitParameters`)

| Hyperparameter | Default | Where | What it does |
|---|---|---|---|
| `n_lucj_layers` | 2 | `flow_params.py:13` | LUCJ block repetitions (ansatz depth/expressivity). **Sweet spot = 2; more degrades under optimize+truncation.** |
| `use_reset_mitigation` | True | `flow_params.py:20` | Pre-measure reset-error mitigation (extra test register). |
| `optimization_level` | 3 | `flow_params.py:26` | Qiskit transpiler optimization. |
| `sabre_max_iterations` | 8 | `flow_params.py:34` | SABRE layout refinement passes. |
| `sabre_swap_trials` | 10 | `flow_params.py:44` | SABRE routing trials per layout. |
| `sabre_layout_trials` | 1024 | `flow_params.py:54` | SABRE random-seed layout trials. |

**Connectivity (parametrization)** is set in `main.py:149-150` — **hardcoded heavy-hex**:
`aa = [(p,p+1)]` (nearest-neighbor), `ab = [(p,p)]` on every 4th orbital. There is **no
connectivity flag**; full-connectivity UCJ requires a code change (and is impractical on heavy-hex
hardware — see §5). The actual UCJ operator is built in `lucj.py` via
`UCJOpSpinUnbalanced.from_t_amplitudes(..., optimize=True, options={maxiter:50})` then truncating
the last rep (`lucj.py:101-116`).

### 2B. Differential evolution (OUTER loop) — `flow_params.py` (`DEParameters`)

| Hyperparameter | Default | Where | What it does |
|---|---|---|---|
| `num_walkers` | 4 | `flow_params.py:65` | Independent SQD evaluations per generation. **Key variance lever** (see §6): more walkers = more independent subsample draws = better best-of. |
| `iterations` | 1 | `flow_params.py:72` | DE generations (the OUTER loop count). **1 = no optimization, carryover never fed back.** |
| `randomization_factor` | 0.2 | `flow_params.py:79` | Perturbation of CCSD amplitudes for walkers 2…N. |
| `fxc` | 0.6 | `flow_params.py:86` | DE mutation scale factor F. |
| `cr_prob` | 0.9 | `flow_params.py:97` | DE crossover rate. |

### 2C. SQD subspace & recovery (INNER loop) — `flow_params.py` (`FlowParameters`)

| Hyperparameter | Default | Where | What it does |
|---|---|---|---|
| `sqd_dim` | 1,000,000 | `flow_params.py:117` | Target subspace size; per spin ≈ √sqd_dim determinants. **Smaller is better under noise** (see §6) — large locks to the noise floor. |
| `n_recovery_steps` | 1 | `flow_params.py:124` | INNER self-consistency recovery passes. **The dominant accuracy lever** (see §6). 1 = single pass. |
| `n_batches` | 1 | `flow_params.py:137` | K independent subsamples per recovery pass (arXiv:2405.05068); min-energy + mean-occupancy. Helps **only with `n_recovery_steps ≥ 3`**. |
| `quantum_source` | real-device | `flow_params.py:152` | IBM sampling vs deterministic random bitstrings. |
| `random_seed` | 24 | `flow_params.py:161` | RNG base seed for the "random" source. |
| `shots` | (per run) | `create_blocks.py` → `sqd_options` | Quantum measurement shots; default 50k (Fugaku) / 500k (Miyabi); our runs use 100k. |

### 2D. Solver (Davidson / carryover) — `algorithms/sbd/sbd/solver_job.py` (`SBDSolverJob`)

| Hyperparameter | Default | Where | What it does |
|---|---|---|---|
| `block` | 10 | `solver_job.py:397` | Max Davidson subspace size. |
| `iteration` | 2 | `solver_job.py:403` | Davidson restarts. |
| `tolerance` | 1e-4 | `solver_job.py:409` | Davidson residual convergence. |
| `carryover_ratio` | 0.5 | `solver_job.py:415` | Fraction of bitstrings retained as carryover candidates. |
| `task/adet/bdet_comm_size` | — | `solver_job.py:377-391` | MPI communicator sizes for the Fugaku solver. |

### 2E. Error mitigation (sampler) — `create_blocks.py` flags → `sqd_options.params.options`

| Flag / env | What it does | Note |
|---|---|---|
| `--dynamical-decoupling` / `SBD_DYNAMICAL_DECOUPLING` (+ `--dd-sequence`, default XY4) | Idle-qubit DD | Works with fractional `rzz`. |
| `--measure-twirling` / `SBD_MEASURE_TWIRLING` | Readout twirling | Works with fractional `rzz`. |
| `--gate-twirling` / `SBD_GATE_TWIRLING` | 2-qubit gate twirling | **Incompatible** with the LUCJ fractional `rzz` (IBM error 1519); warns and is unusable on this stack. |

---

## 3. The three levers the questions are about

**Carryover** (determinant memory across the OUTER loop). Per-spin α/β arrays packed
(beta-left `[:norb]`, alpha-right `[norb:]`). Flow: `state.carryover` → seeds each walker
(`main.py:234`); solver returns `carryover.bin` (α) + `carryover_b.bin` (β); `selection()` updates
`state.carryover` from the best walker (`main.py:338-341`). **Only flows across DE generations** —
with `iterations=1` it starts empty (`main.py:98`) and is never fed back.

**Recovery** (`recover_configurations`, INNER loop, `sqd.py:245`). Refines each measured bit
toward `avg_occupancies` to the target Hamming weight, repairing wrong-particle-number bitstrings.
The inner loop (`n_recovery_steps`) re-runs recovery with the solver's **batch-averaged
occupancies** fed forward each pass (`sqd.py:402`).

**Parametrization / Configuration Rotation (CR)** — the `initial_occupancy` that seeds recovery
on pass 1. Built UHF-consistently in `chem.py`: `dm_cc_a, dm_cc_b = mycc.make_rdm1()` (UCCSD
1-RDM, *per spin*), each diagonalized separately, `initial_occupancy=(occ_a[::-1], occ_b[::-1])`
(`chem.py:169-185`). Distinct α and β distributions.

---

## 4. Commit-by-commit story (what we changed, after which finding)

| # | Commit | Change | Driving finding |
|---|---|---|---|
| 1 | `c945574` | `chem.py`: UHF integrals (h1α/h1β, αα/αβ/ββ ERIs, UCCSD t2 tuple) | RHF was hardcoded; needed open-shell integrals. |
| 2 | `d122fa7` | `lucj.py`: spin-unbalanced LUCJ (`UCJOpSpinUnbalanced`) | Ansatz must carry distinct α/β rotations. |
| 3 | `eb157ae` | `sqd.py`: `subsample_open_shell` (independent α/β) | Closed-shell subsample discarded β. |
| 4 | `47a8481` | `solver_job.py`: UHF FCIDUMP + β dets + method/backend dispatch | Solver needs interleaved-spin FCIDUMP + β determinants. |
| 5 | `c4a4352` | `main.cc`: `diag_uhf` loads separate α/β dets, writes `carryover_b.bin` | Native solver hardcoded `bdet = adet`. |
| 6 | `5996238` | `create_blocks.py`: register/select RHF/UHF binaries by method×backend | Coexisting RHF/UHF presets. |
| 7 | `62f87ee` | tests: lock UHF invariants (FCIDUMP, keys, subsampling) | Regression safety. |
| 8 | `8a9ce93` | `main.cc`/DE: size population array for UHF param count | UHF has more params → broadcast shape error. |
| 9 | `af98e32` | `chem.py`: use **UCCSD** natural-orbital occupancies (not SCF) | **SCF occupancies are integer → recovery collapses subspace to bare HF.** Fractional CCSD occ needed for CR to work. |
| 10 | `fa43260` | `lucj.py`: `optimize=True` (compressed double factorization) | Naive truncation → prepared state ≈ single HF determinant (PR≈1); optimize=True gives 5-6× more diverse configs. |
| 11 | `bdeec66` | `sqd.py`: **inner SQD self-consistency recovery loop** (`n_recovery_steps`) | Recovery was single-pass (static CCSD occ); canonical SQD iterates occ→diag→occ. |
| 12 | `b79a897` | `sqd.py`: recovery-loop diagnostics (excitation histogram, occ, post-select) | Needed to see *why* 50q plateaus. |
| 13 | `5565dea` | `sqd.py`: **K-batch recovery** (`n_batches`, min-E / mean-occ) | Published 77q method (arXiv:2405.05068) builds K batches; we did 1. |
| 14 | `6ff4edf` | `create_blocks.py`: DD + Pauli twirling sampler options | 50q is sample-quality-limited; mitigation is the sampling-stage lever. |
| 15 | `59e53a7` | warn gate-twirling incompatible with fractional `rzz` | IBM error 1519 on first mitigation run. |
| 16 | `a0d03f5` | local hardware-free validation harness + spin `[diag]` | Stop testing hypotheses on expensive remote runs. |
| 17 | `89ea2a5` | `solver_job.py`: robust Python-side `<S²>` from the solved subspace | Quantify spin-contamination removal without the C++ 2-RDM. |
| 18 | `9938d97` | docs: findings report + figures | Presentation. |
| 19 | `3e59329` | harness: connectivity + noise knobs | Enable the connectivity/noise/recovery sweeps. |
| 20 | `d633e8d` | save sweep scripts | Reproducibility. |

(Plus `c9a4827`, `3847aa6`: the running UHF change report.)

---

## 5. Results

### 5A. Crossover curve (hardware)

![correlation vs qubits](figures/crossover_curve.png)

Real-hardware correlation works at 12–40 qubits (+25 to +65 mHa above the correlated reference),
then plateaus into a noise-dominated floor at 50 qubits.

### 5B. Spin-contamination removal (local, noiseless)

![spin contamination removal](figures/spin_contamination.png)

| System | UHF `<S²>` | SQD `<S²>` | pure `S(S+1)` | reading |
|---|---|---|---|---|
| NH triplet (12q) | 2.101 | **2.011** | 2.0 | UHF is contaminated by ~0.10; SQD restores a near-pure triplet. |
| CN doublet (20q) | 0.782 | **0.755** | 0.75 | contamination roughly halved. |

**What it shows:** UHF gains accuracy via spin polarization but is *not* a spin eigenstate. SQD
diagonalizes the true Hamiltonian in the fixed-(Nα,Nβ) sector, whose exact ground state *is* a
pure spin state — so SQD inherits UHF's correlation benefit while removing its contamination.

### 5C. Beating UCCSD — connectivity (local, noiseless, single seed)

![ansatz connectivity vs UCCSD](figures/ansatz_connectivity.png)

CN doublet 20q (UHF −90.963, UCCSD −91.161, FCI −91.172), `optimize=True`, 2 layers:

| Connectivity | E(SQD) | vs UCCSD | vs FCI |
|---|---|---|---|
| heavy-hex (hardware) | −91.1547 | +6.2 mHa | +17.6 |
| **FULL UCJ** | **−91.1672** | **−6.2 (beats)** | **+5.2** |

**What it shows (with the caveat in 5E):** with enough connectivity SQD-over-UHF *can* surpass
UCCSD — the heavy-hex *hardware locality* is the ceiling, not the algorithm. Full connectivity is,
however, impractical on heavy-hex hardware (huge SWAP depth → more noise).

### 5D. The 50q noise wall (hardware)

![50q excitation profile](figures/noise_wall_excitations.png)

Of ~447 subsampled α-determinants, only ~20 are singles/doubles (couple to HF under
Slater–Condon); ~426 are 5–10× excitations with **zero** coupling. ~95% of the subspace is
"deadwood." All upstream stages are confirmed working; the bottleneck is **sample quality at 50q
on NISQ hardware**. K-batch and error mitigation did **not** move it (+21.9 → +19.9 mHa, within
scatter).

### 5E. Hyperparameter sweep — what is the critical lever? (local, CN 20q, seed-averaged)

The single most important table. All vs UCCSD in mHa, 5–8 seeds, `n_lucj_layers=2`.

**Recovery axis (heavy-hex, p=0.01):**

| `n_batches` | `n_recovery_steps` | mean | std | best |
|---|---|---|---|---|
| 1 | 1 | +60.3 | 1.2 | +59.1 |
| 5 | 1 | +57.8 | 1.6 | +56.2 |
| **5** | **3** | **+21.2** | 28.4 | −4.3 |
| **10** | **3** | **+7.7** | 23.8 | −6.2 |

**Parametrization axis (rec=2, p=0.01):**

| connectivity | mean | std | best |
|---|---|---|---|
| heavy-hex | +43.3 | 24.1 | −4.9 |
| intermediate (R=2) | +55.6 | **0.6** | +55.2 |
| full | +42.2 | 25.2 | −8.3 |

**Variance study (heavy-hex, rec=3, b=10):**

| shots | `sqd_dim` | noise p | mean | std | range |
|---|---|---|---|---|---|
| 100k | 20k | 0.01 | +10.8 | 25.6 | [−6.2, +55.1] |
| 100k | 60k | 0.01 | +54.7 | **0.0** | locked to floor |
| 300k | 120k | 0.01 | +54.7 | 0.0 | locked |
| 100k | 20k | **0 (noiseless)** | +12.9 | 26.4 | [−8.7, +57.8] |

**What these show (note: an early seed sweep suggested "recovery is the lever"; controlled
follow-up tests below CORRECTED that — see the revised conclusions):**
1. **`sqd_dim` scaling — corrected by a dedicated sweep.** An 18-orbital sweep (CN 6-31g, noiseless)
   shows energy improving **monotonically** with `sqd_dim` (44→1000 dets/spin: +445→+158 mHa, no
   turnover at 3.1% of the full space). There is **no Goldilocks peak for large systems** — you are
   always undersampled, so bigger is better (diminishing returns). The earlier small-molecule "200k →
   +55 floor" was a tiny-space RNG artifact (the subspace already saturated the full 120-config space).
   **Rule: small molecule → saturate the space; large molecule (C4H5) → use the largest `sqd_dim` the
   solver affords (default 1M).** The 20k/200k used in the 50q runs were deep in the starvation tail.
2. **Recovery depth is NOT a lever** (corrects the early seed-sweep reading, which was confounded
   by single-seed variance). Isolated test at fixed dim: `n_recovery_steps` 1/2/3/5 give *identical*
   energy. On hardware deep recovery is mildly harmful (drives occupancies integer after step 1).
3. **Connectivity matters at the right `sqd_dim`** — full UCJ beats UCCSD where heavy-hex does not,
   noiselessly; but full connectivity is impractical on heavy-hex hardware (SWAP depth → more noise).
4. **The variance is intrinsic, not noise.** Bimodal *even noiselessly* (range −8.7…+57.8): the
   random subsample either lands on a good compact subspace or a mediocre one. Mitigate with **many
   independent draws → high `num_walkers`, take the best.**
5. **RHF-SQD and UHF-SQD are EQUIVALENT.** Noiseless N2 (closed-shell, RHF≡UHF physically): at the
   right `sqd_dim` BOTH reach FCI exactly (−8.5 mHa vs CCSD), identically. The UHF spin-unbalanced
   LUCJ parametrization is correct — it is not the cause of the 50q plateau.

> Honest correction: the −6.2 mHa "beats UCCSD" in 5C is a *lucky single seed*. Seed-averaged, the
> mean is ~+44 mHa with std ~26; SQD beats UCCSD on a *fraction* of draws when recovery is deep.

---

## 6. The two Fugaku runs in flight (this session)

Both: C4H5 50q, heavy-hex, **`sqd_dim=20000`** (the corrected small value), DD + measurement
twirling, 1 layer, `iterations=1`.

| Run | `n_recovery_steps` | `n_batches` | `num_walkers` | lever tested |
|---|---|---|---|---|
| **A** (`run_A_recmax`) | 5 | 10 | 8 | recovery depth |
| **B** (`run_B_walkers`) | 3 | 5 | 16 | many independent draws |

**Result: both hit the same +20 mHa-above-SCF / +290-above-UCCSD floor** (B best −152.3238; A timed
out at the 180-min walltime mid-run, same floor). Post-mortem found `sqd_dim=20k` was *undersized*
for C4H5 (25 orbitals → 141 dets/spin is ~0.004% of the 3.2M-per-spin space; only ~5 of 141 were
singles/doubles), and the diagnostics show the diagonalizer returning the bare HF determinant
(spin density `[0…1…0]`, batch energies exactly = SCF). Lesson: `sqd_dim` must scale with system
size — 20k is fine for a 10-orbital molecule (it saturates the space) but far too small for 25
orbitals, where bigger is monotonically better (§5E).

---

## 7. Conclusions — the four questions answered

**Q1. Are we performing enough quantum sampling (shots) relative to the number of subsamples?**
Current: **shots = 100,000**; subspace per spin ≈ **√sqd_dim**. The raw sample has 30–50k unique
post-select configs, so **shots are NOT the binding constraint** (more shots never rescued a run).
The miscalibrated knob is **`sqd_dim`**. An 18-orbital sweep shows the energy improves **monotonically**
with `sqd_dim` for large systems (no peak) — you are always undersampled, so bigger is better. For a
10-orbital molecule ~141 dets/spin (sqd_dim≈20k) already saturates the full space; for 25-orbital
C4H5 that is ≈0.004% of the space — deep in the starvation tail. **Recommendation: large molecules →
largest affordable `sqd_dim` (default 1M); small molecules → just saturate the space; 100k shots are
adequate either way.**

**Q2. Is the inner loop, outer loop, or both being executed — and how many of each?**
- **Outer loop** = DE generations = `de_params.iterations`. **All production runs used
  `iterations=1` → the outer loop ran exactly ONCE** (no mutation/crossover feedback; DE only
  evaluates the initial CCSD walker + randomized walkers and picks the best).
- **Inner loop** = `n_recovery_steps` self-consistency passes per walker. Originally **1** (single
  pass); since commit `bdeec66` we run **3–5** (the current Fugaku runs: A=5, B=3), each with
  **`n_batches`** sub-diagonalizations (A=10, B=5). So per walker, A does 5×10=50 diagonalizations,
  B does 3×5=15. Both loops execute; the inner loop is where the work now happens, the outer loop is
  effectively disabled at `iterations=1`.

**Q3. Are both α and β bitstrings carried over correctly across iterations, in both loops?**
- **Mechanism: yes, per-spin and correct.** `_stack_spin_carryover` packs β-left/α-right; the
  solver writes separate `carryover.bin` (α) + `carryover_b.bin` (β); both are read back into
  `SBDResult`.
- **Outer loop:** carryover flows across DE generations (`selection()` → `state.carryover` → next
  generation's walkers). **But with `iterations=1` there is no second generation, so carryover
  starts empty and is never actually fed back** — the α/β carryover code is correct but
  *unexercised* in every run so far. Activating it requires `iterations ≥ 2`.
- **Inner loop:** across recovery steps, the **occupancies** are fed forward (batch-averaged), not
  the determinant carryover (`carryover_a/_b` are derived once from the incoming outer carryover and
  held fixed within the inner loop). So inner-loop memory = occupancies; outer-loop memory =
  determinants.

**Q4. Is configuration recovery (CR) working, and are `initial_occupancies` UHF-consistent
(different α and β)?**
- **Yes, UHF-consistent.** `initial_occupancy = (occ_a, occ_b)` where `occ_a = diag(dm_cc_a)`,
  `occ_b = diag(dm_cc_b)` from the **UCCSD** per-spin 1-RDM (`chem.py`) — genuinely distinct α and β
  distributions (verified e.g. phenyl: α sum 21, β sum 20, 40 fractional, α≠β).
- **Critical fix (`af98e32`):** these must come from the *correlated UCCSD* RDM, not the SCF RDM —
  SCF occupancies are integer, which gives recovery no fractional bias and collapses the subspace to
  bare Hartree-Fock. This was a real bug that pinned early runs to SCF.
- **Follow-up fix (`0aa5d4a`):** the occupancy must be the **diagonal** of the correlated 1-RDM
  (per-canonical-MO occupancy, the basis the bitstrings live in), not the sorted `eigh` eigenvalues
  (natural-orbital occupation numbers in a rotated basis), which misassign the recovery bias to the
  wrong orbitals — an error that grows with system size. The RHF path had a parallel bug (used the
  SCF, not UCCSD, RDM) fixed in the same commit.
- **CR works**, with one nuance the local study exposed: recovery reliably repairs particle number
  (3% valid scatter → 100% in-sector), but the fine occupancy bias is only *one* factor; whether a
  subsample lands on a good subspace is partly luck (the intrinsic bimodal variance of §5E).

---

## 8. Recommendations (forward)

1. **Use the largest affordable `sqd_dim` for large molecules** (no Goldilocks peak — energy improves
   monotonically with subspace size; verified by an 18-orbital sweep *and now on hardware*, §9:
   10⁶→10⁷ gave −55.7 mHa on allyl 40q). For C4H5 (25 orb) use ≥10⁶ (10⁷ if the solver budget
   allows), *not* the 20k/200k that the 50q runs used — those were deep in the starvation tail. Only
   small molecules (where √sqd_dim saturates the full space) want a modest dim.
2. **Do NOT rely on deep recovery** — `n_recovery_steps` is not a lever (1–5 give the same energy);
   1–2 is sufficient. `n_batches ≥ 5` for best-of and stable averaged occupancy.
3. **Raise `num_walkers` (8–16)** to beat the intrinsic bimodal variance; take the best walker.
4. **Turn on the outer loop (`iterations ≥ 2`)** to finally exercise α/β carryover feedback.
5. **Connectivity:** full UCJ beats UCCSD noiselessly but is impractical on heavy-hex hardware;
   reaching HCI/DMRG-class accuracy needs lower-noise or non-heavy-hex hardware. (HCI/DMRG are
   external references; not computable in this environment.)
6. **The UHF parametrization is verified correct** (RHF-SQD ≡ UHF-SQD → FCI on N2, §5E). The 50q
   plateau is hardware noise + `sqd_dim` mis-tuning, *not* the open-shell ansatz.

---

## 9. The `sqd_dim`=10⁷ + K=8 breakthrough (allyl C₃H₅, 40q, `ibm_kingston`)

The recommendations in §8 were put to the test on the allyl radical (20 spatial orbitals,
nelec=(12,11), doublet) — the same FCIDUMP and backend as the earlier 40q run, changing only the
two levers §8 identified. References from the FCIDUMP: **UHF = −115.042051, UCCSD = −115.224061**.

### 9A. The result — a 55.7 mHa jump from two knobs

| Run | sqd_dim | √ (dets/spin) | K-batch | walkers | iters | rec | **Best energy** | vs UHF | vs UCCSD |
|-----|---------|---------------|---------|---------|-------|-----|-----------------|--------|----------|
| Prior plateau (`clever-labradoodle`) | 10⁶ | 1000 | 3 | 4 | 4 | 5 | −115.062178 | −20.6 mHa | +161.9 mHa |
| **This run (`daffodil-kingfisher`)** | **10⁷** | **3162** | **8** | 4 | 2 | 2 | **−115.117896** | **−75.8 mHa** | **+106.2 mHa** |

**Raising `sqd_dim` 10⁶→10⁷ and `n_batches` 3→8 lowered the energy by 55.7 mHa** and closed ~34%
of the residual gap to UCCSD — with *fewer* DE iterations and recovery steps than the plateaued run.
This is the strongest single confirmation that **`sqd_dim` is the dominant lever and the K-batch
count is the second** — exactly the §8 prediction, now demonstrated on hardware, not a local proxy.

### 9B. Per-walker / per-generation detail (the variance is the story)

```
Gen 0 walkers:  -115.1179  -115.0861  -115.0820  -115.0909   -> best -115.1179 (walker 0)
Gen 1 walkers:  -115.0916  -115.1119  -115.0857  -115.0915   -> best -115.1119 (walker 1)
Flow best: -115.117896 (gen 0, walker 0)   [DE gen 1 did not beat gen 0, as before]
```

Within a single recovery step the **8 K-batches spanned 11–54 mHa** (e.g. walker 0 gen0:
min −115.118, max −115.077, spread 40.6 mHa; walker 1 gen1: spread 54.3 mHa). The best overall
number came from the *luckiest single batch among 32 draws* (8 batches × 4 walkers in gen 0). This
is the §5E intrinsic-variance signature, now amplified: **more batches widen the max−min window, and
because we keep the MIN, more batches ⇒ more chances to catch a deep one.** K=8 is doing real work —
the best batch (−115.118) sits ~30 mHa below the batch mean (−115.09).

### 9C. The diagnostic that explains *why* it helped — and where the next wall is

The excitation summary tells the mechanism precisely. At sqd_dim=10⁷ each batch keeps 3162 dets/spin
(vs 1000 at 10⁶):

| Quantity | sqd_dim=10⁶ | sqd_dim=10⁷ | Change |
|----------|-------------|-------------|--------|
| dets/spin kept | 1000 | 3162 | 3.16× |
| singles+doubles (useful) α | ~100 | **~300** | ~3× more |
| useful **fraction** | ~10% | **~9.5%** | *unchanged* |
| >2-exc "deadwood" | ~900 | ~2860 | grows with dim |
| distinct α strings available | ~690k | ~686k | same pool |
| prob mass in kept top-3162 (α) | — | **1.8–4.0%** | tiny |

Two facts jump out:
1. **The useful-determinant *count* tripled** (100→300 singles+doubles), which is *why* the energy
   dropped — more Slater-Condon-coupled configs in the CI matrix = more correlation captured.
2. **The useful *fraction* stayed ~10%** and the deadwood grew proportionally. The subspace is still
   ~90% >2-excitation determinants that don't couple to HF. We're capturing more correlation by
   brute-force enlarging the net, not by improving its *quality*.

The probability-mass line is the deepest signal: the top-3162 α strings hold only **~2% of the total
probability**. The device noise has smeared ~98% of the measured weight across a long tail of
high-excitation noise configs. Raising sqd_dim scoops up more of the *good* tail, but with sharply
diminishing returns — each new determinant is rarer and less HF-coupled than the last.

### 9D. Operational notes from this run (for reproducibility)

- **Shot mixing held up:** 2 batches × 1M shots → ~1.34M kept (≈67% Kingston retention), merged via
  `concatenate_shots`; ~100% unique configs (noise-flattened), as expected.
- **Per-walker RNG fix (commit `471deb9`) was active** — the 8 batches × 4 walkers drew genuinely
  independent subspaces (visible in the divergent per-walker top-20 tables and energies). Without it
  the threaded runner would have raced and correlated them.
- **Wall-clock: 34396 s (~9.5 h)** for 2 DE generations. The bottleneck is now the **diagonalizer**:
  each 10⁷ Davidson solve takes ~10 min, ×8 batches ×2 recovery ×4 walkers ×2 gens. The IBM HTTP-502
  on one job (auto-retried after 300 s) cost a few minutes but did not derail the run.
- **Benign noise in the log:** `solve_eigenstate` lines emit a harmless `~/.cargo/env: No such file`
  shell warning (the deleted Rust toolchain is still referenced in `.bashrc`/`.bash_profile`); and
  the comprehensive-summary plot is skipped (`matplotlib` not in the `sbd` venv). Neither affects
  results. Fix: remove the `.cargo/env` source lines from the shell rc files; `uv pip install
  matplotlib` into the sbd venv to re-enable the top-20 plot.

### 9E. Where to go next — toward 10⁸ and beyond

The mechanism in §9C dictates the path. The lever still has runway, but the *quality ceiling*
(fraction stuck at ~10%, prob-mass at ~2%) says brute-force `sqd_dim` alone will hit diminishing
returns. The recommended escalation, in order of expected payoff per cost:

1. **sqd_dim = 10⁸ (√ = 10⁴ dets/spin).** Direct continuation of the proven lever. Expected to add
   another (smaller) increment toward UCCSD — extrapolating §9A, perhaps 20–40 mHa, not another 55.
   Cost: each Davidson solve is ~10× heavier than 10⁷ (the matrix is 10⁸ vs 10⁷). With K=8 that is a
   very heavy diagonalizer load — budget a full 24 h mem2 job and consider dropping to K=4–5 to fit,
   or `num_walkers=2` (DE is not adding value — see below), reinvesting the saved solves into dim.
2. **Keep or raise K-batch (8→12–16).** Because we keep the MIN over batches and the per-step spread
   is 30–50 mHa, more independent draws is a cheap, near-linear way to catch a deeper batch. At 10⁸
   this trades against solver cost — K=8 is a reasonable hold; only raise K if the solve fits.
3. **Stop spending on DE iterations.** Gen 1 again failed to beat gen 0 (both runs). The DE outer
   loop is not the lever at this noise level; set `iterations=1` (or 2 at most) and pour the budget
   into dim + K. `num_walkers` still matters (best-of over the intrinsic variance) — keep 4.
4. **Attack the *quality* ceiling, not just the size.** The ~10% useful fraction is set by hardware
   noise smearing weight into deadwood. Two orthogonal levers: (a) stronger error mitigation /
   lower-noise backend to sharpen the sampled distribution so a larger fraction of kept dets are
   singles+doubles; (b) a tighter occupancy-bias in configuration recovery to preferentially repair
   toward low-excitation configs. Without one of these, 10⁸ will capture more correlation but the
   subspace stays ~90% deadwood.

**Honest expectation:** 10⁸ should continue the descent toward UCCSD (−115.224) and is the right
next experiment, but the curve is concave — each decade of `sqd_dim` buys less. Reaching UCCSD-class
accuracy on this 40q hardware likely needs the §9E-4 quality levers in addition to raw subspace size.
The result here is nonetheless a clean, defensible win: **−75.8 mHa below UHF on real hardware, the
largest correlation recovery in the project, driven by the two levers the analysis predicted.**
