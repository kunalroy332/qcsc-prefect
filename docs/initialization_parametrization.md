# LUCJ Initialization & SQD Hyperparameters — Reference

This document explains how the (L)UCJ ansatz is **parametrized and initialized** from CCSD
amplitudes in the qcsc-prefect SQD pipeline, and gives a tuned reference for every
flow/solver hyperparameter: its default, the desired value for capturing correlation, and why.

It is grounded in the ffsim LUCJ explanation
(https://qiskit-community.github.io/ffsim/explanations/lucj.html) and the paper
*"Improved parameter initialization for the (local) UCJ ansatz"* (arXiv:2511.22476).

---

## 1. How the ansatz is parametrized (t1, t2 → LUCJ)

`ffsim.UCJOp{SpinBalanced,SpinUnbalanced}.from_t_amplitudes(t2, t1=..., n_reps=N,
interaction_pairs=..., optimize=..., regularization=...)`:

1. **Double-factorizes** the CCSD `t2` amplitudes into rank-1 terms
   (each = a diagonal-Coulomb matrix + an orbital rotation). `t1` is folded in as an extra
   single-body correction.
2. **Truncates to the `N` largest terms** (by singular value). `N = n_reps` = the number of
   ansatz repetitions = circuit "layers". The full factorization of a real molecule needs
   **~80 terms**; we typically keep only 1–4 → very aggressive truncation.
3. **`interaction_pairs`** restricts which diagonal-Coulomb entries may be nonzero. This is the
   **L** in **L**UCJ — the heavy-hex hardware locality:
   - `aa = [(p, p+1) for p in range(norb-1)]` — nearest-neighbor same-spin (the qubit chain)
   - `ab = [(p, p) for p in range(0, norb, 4)]` — on-site α–β via ancilla qubits every 4th orbital
   - (UHF adds `bb`, same topology as `aa`).
4. **Spin-unbalanced (UHF)**: `t2` is the tuple `(t2aa, t2ab, t2bb)`, `interaction_pairs` is the
   triple `(aa, ab, bb)`. With a single int `n_reps`, αβ terms are kept before αα/ββ terms.

### The truncation in our code (`lucj.py`) — verified correct
`initialize_ucj_parameters` builds `n_reps = n_lucj_layers + 1`, drops the smallest term, and folds
its orbital rotation into `final_orbital_rotation`; `create_lucj_circuit` rebuilds via
`from_parameters(n_reps=n_lucj_layers, with_final_orbital_rotation=True)`. **These round-trip to the
identical operator** (checked: diag_coulomb / orbital_rotations / final_rotation all match). Not a bug.

---

## 2. `optimize=True` — the compressed double factorization (KEY for SQD)

`from_t_amplitudes(..., optimize=True, options=dict(maxiter=50))` re-optimizes the kept DF terms
(least-squares) to best approximate the *full* UCCSD generator — the paper's "compressed DF".

**Subtlety (from the ffsim docs):** compressed DF does *not* always lower the prepared-state
energy `⟨H⟩` — the Trotter error of the UCJ ansatz can increase because the norms of individual
terms grow. **This is bad for VQE but GOOD for SQD/QSCI**: larger term norms spread the
wavefunction over more configurations → a more diverse sample set to diagonalize over.

**Measured (H6, 12q, this repo):** naive vs `optimize=True`, sampling the prepared state:

| n_reps | metric | naive | optimize=True | optimize + reg(1e-2) |
|--------|--------|-------|---------------|----------------------|
| 1 | distinct configs sampled | 13 | **83** | 53 |
| 1 | participation ratio | 1.0 | **5.0** | 1.5 |
| 2 | distinct configs sampled | 16 | **100** | 62 |
| 2 | participation ratio | 1.0 | **6.3** | 1.5 |

Naive truncation gives participation ratio ≈ 1.0 — i.e. a **near-single-determinant (Hartree-Fock)
state**. This is exactly why our phenyl (70q) and C4H5 (50q) runs collapsed to the SCF energy: the
sampled subspace was essentially just HF, and HF doesn't couple to high excitations (Slater-Condon).

**Recommendation for SQD:** use `optimize=True`, `options=dict(maxiter=50)`, **no (or tiny)
regularization** — we *want* the wavefunction to spread. `regularization=1e-2` pulls it back toward
naive (less spread); only use it if the increased term norms hurt the specific run.

**`t1`:** pass the CCSD `t1` amplitudes too (`t1=mycc.t1`); the docs initialize with both t1 and t2.
Our current `lucj.py` passes only `t2` — adding `t1` is a recommended improvement.

---

## 3. Circuit / flow hyperparameters (`algorithms/sbd/sbd/flow_params.py`)

| Parameter | Default | Desired (SQD) | Why |
|-----------|---------|---------------|-----|
| `n_lucj_layers` (n_reps) | 2 | **2–4** | More reps = more DF terms kept = richer ansatz/sampling. Our early sweep used 1 (too few). Deeper = more circuit noise on real HW, so balance. |
| `use_reset_mitigation` | True | True | Mid-circuit reset readout error mitigation; keep on. |
| `optimization_level` | 3 | 3 | Qiskit transpiler effort; keep high for fidelity. |
| `sabre_layout_trials` | 1024 | 256–1024 | Layout search breadth. High = better layout but slow at large qubit counts; lower (e.g. 256) to speed up >50q. |
| `sabre_max_iterations` | 8 | 8 | Routing refinement passes; default fine. |
| `sabre_swap_trials` | 10 | 10 | Default fine. |

### Differential-evolution parameters (`DEParameters`)

| Parameter | Default | Desired | Why |
|-----------|---------|---------|-----|
| `num_walkers` | 4 (min) | 4–8 | DE population members, sampled in parallel. More = better optimization, more quantum cost. |
| `iterations` | 1 | **5–10** | DE generations. **1 means no real optimization** (just initial CCSD params + randomized walkers). Must raise to actually optimize the ansatz parameters. |
| `fxc` (DE scale F) | 0.6 | **~0.5** | Mutation scaling. 0.5 is the standard DE value — gentler, better fine-tuning near a good solution. |
| `cr_prob` (crossover) | 0.9 | 0.9 | Standard DE crossover rate; keep. |
| `randomization_factor` | 0.2 | 0.1–0.3 | Spread of the initial walker perturbation around CCSD params. |

### Subspace / sampling

| Parameter | Default | Desired | Why |
|-----------|---------|---------|-----|
| `sqd_dim` (subspace dim) | 1,000,000 | **~1,000,000** | Number of determinants the SBD solver diagonalizes over. The single biggest SQD lever for capturing correlation. Our early sweep used 20k (50× too small). |
| `shots` (Prefect variable `sqd_options`) | 50k (Fugaku) | **≥1,000,000** | Must be ≥ sqd_dim to populate a 1M-determinant subspace. Set via `create_blocks --shots`. |

---

## 4. SBD solver parameters (`algorithms/sbd/sbd/solver_job.py`)

These control the **classical Davidson diagonalization** of the sampled subspace, and the
**carryover** of good configurations between DE iterations.

| Parameter | Default | Desired | Why |
|-----------|---------|---------|-----|
| `block` (Davidson Krylov block) | 10 | **10–20** | Max Davidson subspace (Krylov) size. Larger = more robust convergence for the lowest eigenvalue, more memory. 10 is OK for ground state; 20 safer for large subspaces. |
| `iteration` (Davidson restarts) | 2 | **5–10** | Number of Davidson restarts. 2 is small; for a 1M-determinant subspace use 5–10 to ensure the eigenvalue is converged (not the bottleneck). |
| `tolerance` | 1e-4 | 1e-4 (→1e-5 final) | Davidson residual-norm convergence threshold. Tighten to 1e-5 for publication-quality final energies. |
| `carryover_ratio` | 0.5 | ~0.5 | Fraction of bitstrings retained as carryover into the next DE iteration. ~0.5 balances exploration vs keeping good configs. |
| `task_comm_size` | 1 | scale w/ nodes | MPI: distributes Hamiltonian column ops. Raise for parallel scaling. |
| `adet_comm_size` | 1 | scale w/ nodes | MPI: # alpha-determinant partitions. Product (task×adet×bdet) must equal MPI ranks. |
| `bdet_comm_size` | 1 | scale w/ nodes | MPI: # beta-determinant partitions. |

---

## 5. Recommended recipe to beat UCCSD (next trial)

For an open-shell system (e.g. C4H5, target UCCSD = −152.6156 Ha):

```python
# lucj.py initialization (apply optimize=True + t1):
ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
    t2=(t2aa, t2ab, t2bb),
    t1=(t1a, t1b),                       # ADD t1
    n_reps=n_lucj_layers + 1,
    interaction_pairs=(aa, ab, bb),
    optimize=True,                       # compressed DF -> diverse SQD samples
    options=dict(maxiter=50),
    # regularization: omit (we WANT spread) or use small 1e-3 if norms blow up
)
```
```python
FlowParameters(
    sqd_dim=1_000_000,                   # big subspace
    circ_params=CircuitParameters(n_lucj_layers=2),
    de_params=DEParameters(num_walkers=4, iterations=5, fxc=0.5),  # actually optimize
)
# create_blocks: --shots 1000000 --iteration 5 --block 20
```

**Expected effect:** `optimize=True` is the critical change — it converts the near-HF sampling
(participation ratio ≈ 1) into a diverse subspace (≈ 5–6×), which is what lets SQD capture
correlation beyond Hartree-Fock at scale.
