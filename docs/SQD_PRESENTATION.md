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

**What these show:**
1. **Recovery depth is the dominant lever.** `n_recovery_steps` 1→3 drops the mean ~40 mHa and
   unlocks beating UCCSD. Connectivity shifts the mean only ~1 mHa at fixed recovery.
2. **Batches help only with recovery.** b 1→5 at rec=1 stays pinned (+58–60); b+rec together work.
3. **Smaller `sqd_dim` is better.** Large subspace deterministically locks to the +55 mHa floor —
   it dilutes good configs with high-excitation deadwood. Our 50q runs used 200k; the new runs use
   **20k**. More shots do not rescue a large subspace.
4. **The variance is intrinsic, not noise.** It is bimodal *even noiselessly* (range −8.7…+57.8):
   the random subsample either lands on a good compact subspace or a mediocre one. Mitigate with
   **many independent draws → high `num_walkers`, take the best.**

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

The headline experiment: does **small `sqd_dim` + deep recovery / many walkers** finally move 50q
off the +20 mHa floor.

---

## 7. Conclusions — the four questions answered

**Q1. Are we performing enough quantum sampling (shots) relative to the number of subsamples?**
Current: **shots = 100,000**; subspace per spin ≈ **√sqd_dim**. With the old `sqd_dim=200,000`,
that is ≈ 447 α × 447 β determinants — and the raw sample has 40–50k unique post-select configs, so
shots are *not* the binding constraint. The variance study showed the opposite problem: **a large
`sqd_dim` is harmful** (it pulls in deadwood and locks to the floor); **more shots do not help a
large subspace.** Recommendation: **small `sqd_dim` (≈20k → ~141 dets/spin) with 100k shots is
well-balanced**; shots are adequate, the subspace size was the miscalibrated knob.

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
- **Yes, UHF-consistent.** `initial_occupancy = (occ_a[::-1], occ_b[::-1])` where
  `occ_a = eigh(dm_cc_a)`, `occ_b = eigh(dm_cc_b)` from the **UCCSD** per-spin 1-RDM
  (`chem.py:169-185`) — genuinely distinct α and β distributions (verified e.g. phenyl: α sum 21,
  β sum 20, 40 fractional, α≠β).
- **Critical fix (`af98e32`):** these must come from the *correlated UCCSD* RDM, not the SCF RDM —
  SCF occupancies are integer, which gives recovery no fractional bias and collapses the subspace to
  bare Hartree-Fock. This was a real bug that pinned early runs to SCF.
- **CR works**, with one nuance the local study exposed: recovery reliably repairs particle number
  (3% valid scatter → 100% in-sector), but the fine occupancy bias is only *one* factor; whether a
  subsample lands on a good subspace is partly luck (the intrinsic bimodal variance of §5E).

---

## 8. Recommendations (forward)

1. **Set `sqd_dim` small (~20k), not large.** Biggest miscalibration we found.
2. **Use deep recovery (`n_recovery_steps ≥ 3`) — the dominant lever** — with `n_batches ≥ 5`.
3. **Raise `num_walkers` (8–16)** to beat the intrinsic bimodal variance; take the best walker.
4. **Turn on the outer loop (`iterations ≥ 2`)** to finally exercise α/β carryover feedback.
5. **Connectivity:** full UCJ beats UCCSD noiselessly but is impractical on heavy-hex hardware;
   reaching HCI/DMRG-class accuracy needs lower-noise or non-heavy-hex hardware. (HCI/DMRG are
   external references; not computable in this environment.)
