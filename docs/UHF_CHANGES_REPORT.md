# Adding UHF (Open-Shell) Support to qcsc-prefect

**Branch:** `kk/feature/uhf` · **7 commits · 12 files · +960 / −79 lines**

This report walks through every change I made to add **unrestricted Hartree–Fock (UHF / open-shell)**
support to the SQD/LUCJ → SBD-diagonalization pipeline. For each layer I show **what the code looked
like before, why it could only do RHF (closed-shell), what I changed, and why.** It closes with the
tests I added and the results from local validation and the Fugaku closed-loop run.

---

## 0. The problem in one sentence

The pipeline hard-assumed **RHF / closed-shell** in five Python layers plus one C++ shortcut: a single
spin-summed integral set, a single CCSD `t2`, spin-balanced LUCJ operators, an `assert not open_shell`
gate, alpha-only determinant sampling, and `bdet = adet` in the solver. To support open-shell systems
(where `nelec_alpha != nelec_beta`, e.g. an O₂ triplet or any radical) I had to make every one of those
layers spin-resolved — **without breaking the existing RHF path**, which had to stay the default and
behave identically.

My guiding principle throughout: **additive, RHF-default**. Every UHF path is a new branch guarded by an
`unrestricted` flag (or `method="uhf"`); nothing on the RHF path changes.

---

## 1. Classical integrals — `chem.py`
*Commit `c945574` — feat(chem): compute UHF electronic-structure integrals*

### What it was, and why it only did RHF

The `ElectronicProperties` schema carried a **single** spin-summed integral set and a single `t2`:

```python
class ElectronicProperties(BaseModel):
    one_body_tensor: NpStrict2DArrayF64
    two_body_tensor: NpStrict4DArrayF64
    t2: NpStrict4DArrayF64
    initial_occupancy: tuple[...]
    num_electrons: tuple[int, int]
    open_shell: bool          # computed, but only used to *reject* open-shell downstream
    spin_sq: float
```

The builder used the restricted PySCF API throughout — one MO coefficient matrix, `ao2mo.full` over that
single basis, `cc.CCSD`, and a spin-summed `make_rdm1()`:

```python
h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2 = ao2mo.full(mf._eri, mf.mo_coeff, compact=False)...     # single MO basis
mycc = cc.CCSD(mf); mycc.kernel(); t2 = mycc.t2            # single t2
...
initial_occupancy=(occ_ccsd[::-1], occ_ccsd[::-1])         # alpha == beta (duplicated)
```

And both entry points were hard-wired to restricted SCF:

```python
mf = scf.RHF(mol).run()              # geometry path
mf = tools.fcidump.to_scf(...)       # fcidump path (RHF)
```

A UHF system has **different alpha and beta molecular orbitals**, so it needs *three* two-body integral
blocks (αα, αβ, ββ), a *tuple* `t2 = (t2aa, t2ab, t2bb)`, and *per-spin* occupancies. None of that could
be represented.

### What I changed

**Extended the schema (additive — RHF leaves the new fields `None`):**

```python
class ElectronicProperties(BaseModel):
    one_body_tensor: NpStrict2DArrayF64        # alpha h1 (RHF: the only one)
    two_body_tensor: NpStrict4DArrayF64        # alpha-alpha block
    t2: NpStrict4DArrayF64                      # t2aa
    ...
    unrestricted: bool = False                  # NEW gate flag
    one_body_tensor_b:  NpStrict2DArrayF64 | None = None   # beta h1
    two_body_tensor_ab: NpStrict4DArrayF64 | None = None   # (aa|bb)
    two_body_tensor_bb: NpStrict4DArrayF64 | None = None   # (bb|bb)
    t2_ab: NpStrict4DArrayF64 | None = None
    t2_bb: NpStrict4DArrayF64 | None = None
```

**Added `_build_property_uhf`** that uses the unrestricted PySCF API:

```python
mo_a, mo_b = mf.mo_coeff                                    # two MO sets
h1_a = mo_a.T @ hcore @ mo_a;  h1_b = mo_b.T @ hcore @ mo_b
h2_aa = ao2mo.full(eri, mo_a, ...)                          # three blocks
h2_bb = ao2mo.full(eri, mo_b, ...)
h2_ab = ao2mo.general(eri, (mo_a, mo_a, mo_b, mo_b), ...)
t2_aa, t2_ab, t2_bb = cc.UCCSD(mf).kernel-> mycc.t2         # t2 is a 3-tuple
dm_a, dm_b = mf.make_rdm1()                                 # per-spin RDMs
```

**Added `unrestricted` branches to both entry points.** The geometry path dispatches to `scf.UHF`;
the FCIDUMP path was the tricky one — **pyscf 2.13 has no `fcidump.to_uhf_scf`**, so I build the UHF
mean field from the RHF shell via `to_scf().to_uhf()` (the `MS2` header sets the spin), which I verified
reproduces a direct `scf.UHF` energy to machine precision.

> **Why this matters for the presentation:** this is the layer where "open-shell" first becomes
> representable. Everything downstream is plumbing the three blocks and the t2-tuple through.

---

## 2. Quantum ansatz — `lucj.py`
*Commit `d122fa7` — feat(lucj): build spin-unbalanced LUCJ circuits*

### What it was, and why it only did RHF

Two hard gates flatly **rejected** open-shell, and both the parameter builder and the circuit builder
used ffsim's **spin-balanced** operators (which assume identical alpha/beta amplitudes):

```python
assert not elec_props.open_shell                       # <-- hard stop
...
ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=t2, n_reps=..., interaction_pairs=(aa_indices, ab_indices))   # 2-tuple of pairs
...
ffsim.UCJOpSpinBalanced.from_parameters(...)
ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op=ucj_op)
```

`UCJOpSpinBalanced` takes a single `t2` array and a **2-tuple** `(aa, ab)` of interaction pairs — there is
no way to express a distinct beta channel.

### What I changed

Replaced the asserts with `if elec_props.unrestricted:` branches that use ffsim's **spin-unbalanced** API.
I confirmed the exact signatures against the installed `ffsim 0.0.80` before writing the code:

```python
ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
    t2=(t2aa, t2ab, t2bb),                                 # the 3-tuple
    n_reps=...,
    interaction_pairs=(aa_indices, ab_indices, bb_indices))   # NOW a 3-tuple
...
ffsim.UCJOpSpinUnbalanced.from_parameters(...)
ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op=ucj_op)
```

Key details I handled:
- `interaction_pairs` became a **3-tuple** `(aa, ab, bb)`, so I added an optional `bb_indices` argument
  that defaults to the alpha topology via `_default_bb_indices` — callers don't have to thread it while
  UHF is opt-in.
- UHF randomization perturbs **all three** `t2` blocks independently (they have different shapes).
- The unbalanced op exposes the same `diag_coulomb_mats / orbital_rotations / final_orbital_rotation`
  fields as the balanced one, so the existing truncation logic ported over unchanged.
- `PrepareHartreeFockJW` already accepts `nelec=(na, nb)`, so HF-state prep was already spin-correct.

---

## 3. Bitstring sampling — `sqd.py`
*Commit `eb157ae` — feat(sqd): subsample alpha/beta independently*

### What it was, and why it only did RHF

`subsample_close_shell` computed both alpha and beta CI strings but then **averaged and merged them into a
single pool** and returned one list — fine for closed-shell where α==β, fatal for open-shell:

```python
mixed_ci_strigs = np.concatenate((ci_strs_a, ci_strs_b))     # merge the two spins
ci_strs_unique, ... = _deduplicate_and_accumurate_probs(
    ci_strings=mixed_ci_strigs, probabilities=np.tile(probabilities, 2)/2.0)
return ...                                                    # ONE list (alpha only)
```

And the walker fed that single list to **both** spin slots of the solver:

```python
ci_strings = subsample_close_shell(...)
davidson_solver.run(ci_strings=(ci_strings, ci_strings), ...)   # alpha == beta
```

### What I changed

Added **`subsample_open_shell`**, a sibling that keeps the two spins fully independent — each spin gets its
own deduplication, its own Hartree–Fock string at index 0 (using its own electron count), and its own
carryover; no cross-spin averaging:

```python
def subsample_open_shell(..., carryover_a, carryover_b, num_elec_a, num_elec_b):
    ci_a = _subsample_one_spin(ci_strs_a, ..., carryover_a, num_elec=num_elec_a)
    ci_b = _subsample_one_spin(ci_strs_b, ..., carryover_b, num_elec=num_elec_b)
    return ci_a, ci_b
```

`walker_sqd` now branches on `elec_props.unrestricted`: it splits the incoming `2*norb`-wide carryover
(beta-left, alpha-right) per spin, runs the solver with the **genuinely distinct** `(ci_a, ci_b)` and the
beta/mixed/beta-beta tensors, then recombines per-spin carryover via `_stack_spin_carryover` so the
differential-evolution optimizer feeds it back on the next iteration. The RHF path is untouched.

---

## 4. Solver interface — `solver_job.py`
*Commit `47a8481` — feat(solver): write UHF FCIDUMP + beta determinants + dispatch*

### What it was, and why it only did RHF

`SBDSolverJob` had a single backend axis and wrote a single RHF FCIDUMP plus one determinant file:

```python
solver_mode: Literal["cpu", "gpu"]                 # backend only — no method axis
...
fcidump.from_integrals(..., one_body_tensor, two_body_tensor, norb, nelec)   # single RHF dump
alpha_det = np.asarray(ci_strings[0], ...)                                    # alpha only
# (no BetaDets, no beta carryover read)
```

There was no way to (a) select an open-shell binary, (b) write the spin-resolved integrals the `_UHF`
solver needs, or (c) hand it a distinct beta determinant list.

### What I changed — and the single highest-risk piece

**Made method and backend orthogonal axes:**

```python
solver_mode: Literal["cpu", "gpu", "fugaku"]       # backend
method:      Literal["rhf", "uhf"] = "rhf"          # NEW: method axis
```

with `_executable_key()` deriving the binary key from the pair — `sbd_diag`, `sbd_diag_uhf`,
`sbd_diag_gpu`, `sbd_diag_gpu_uhf` — so any backend can run either method with no schema change.

**Wrote `_write_uhf_fcidump`** — the format the `-D_UHF` SBD parser expects. This was the make-or-break
detail. The `_UHF` `SetupIntegrals` does **not** consume PySCF's four-block MOLPRO format; it expects an
**interleaved spin-orbital** dump: 1-based indices with alpha = `2p+1`, beta = `2p+2`, one-body marked by
`k=l=0`, core energy as the `0 0 0 0` record. Same-spin blocks use full 8-fold symmetry; the mixed `aa|bb`
block omits the `ij≥kl` cut, and the `bb|aa` block is recovered by the reader via `(ij|kl)=(kl|ij)` — so I
emit **three** blocks, not four.

In UHF mode `_prep_files` now writes that spin-resolved FCIDUMP plus `AlphaDets.bin` **and**
`BetaDets.bin`; `_build_solver_args` adds `--adetfile/--bdetfile`; `_read_files` reads `carryover_b.bin`
alongside `carryover.bin`; and `SBDResult` gained `carryover_bitstrings_b`. RHF mode is byte-for-byte
unchanged.

---

## 5. C++ solver & build — `native/`
*Commit `c4a4352` — feat(native): diag_uhf binary with separate alpha/beta determinants*

### What it was, and why it only did RHF

`main.cc` loaded only alpha and forced beta to equal it; it wrote only the alpha carryover; and it had no
`--fcidump` override:

```cpp
sbd::LoadAlphaDets(adetfile, adet, ...);
sbd::MpiBcast(adet, 0, comm);
bdet = adet;                                   // <-- closed-shell shortcut
...
std::ofstream ofs_co_bin("carryover.bin", ...);   // alpha carryover only
```

### What I changed

Aligned `main.cc` with the upstream `apps/chemistry_tpb_selected_basis_diagonalization` reference:
parse `--fcidump / --adetfile / --bdetfile`; when `--bdetfile` is given, load a **distinct** beta list
(`LoadAlphaDets` is format-agnostic and reads `.txt` bitstrings or `.bin` packed); otherwise fall back to
`bdet = adet`, preserving the original RHF behavior exactly. In UHF mode it also writes `carryover_b.bin`
from `co_bdet`.

> One real bug the reference data caught: `main.cc` was hardcoding `fcidump.txt` and ignoring `--fcidump`,
> so it would silently read the wrong file. I added the flag and re-verified.

The three build scripts (`build_sbd.sh`, `build_sbd_fugaku.sh`, `build_sbd_gpu.sh`) each gained a `UHF=1`
switch that appends `-D_UHF` and emits a separate `diag_uhf` (`diag-gpu_uhf`) binary, leaving the
restricted `diag` target untouched — so both binaries build side by side.

---

## 6. Workflow wiring — `create_blocks.py` + `main.py`
*Commit `5996238` — feat(blocks): register and select RHF/UHF binaries*

### What it was, and why it only did RHF

Block registration had a single executable key and no method flag:

```python
solver-mode choices = ["cpu", "gpu"]
CommandBlock(..., executable_key="sbd_diag")
executable_map={"sbd_diag": executable_path}        # one binary only
```

### What I changed

Added `--method {rhf,uhf}` and `--sbd-executable-uhf` (plus `SBD_METHOD` / `SBD_EXECUTABLE_UHF` env). The
script derives executable keys exactly as `SBDSolverJob._executable_key` does, registers **both** the RHF
and (when provided) UHF binaries in `executable_map`, and sets the active CommandBlock key accordingly —
so RHF and UHF blocks coexist on one HPC profile. In `main.py`, the flow now reads `solver.method` and
drives the integral computation with `unrestricted=True` when it's `"uhf"` — making the solver block the
single source of truth for RHF vs UHF.

---

## 7. Tests — `tests/test_uhf.py`
*Commit `62f87ee` — test(sbd): lock UHF invariants*

I added six regression tests that lock the pure-Python surface the end-to-end run depends on:

| Test | What it locks |
|------|---------------|
| `test_rhf_electronic_properties_unchanged` | RHF still leaves all beta/mixed fields `None` |
| `test_uhf_electronic_properties_shapes` | UHF produces h1 α/β, three two-body blocks, the t2 tuple, per-spin occ |
| `test_uhf_fcidump_writer_roundtrip_indices` | index convention (α=2p+1, β=2p+2; one-body k=l=0; core 0,0,0,0) |
| `test_uhf_fcidump_spin_block_convention` | the 3-block scheme matches the authoritative `make_uhf-fcidump.py` spec |
| `test_executable_key_matrix` | backend × method → correct binary key for all 6 combos |
| `test_subsample_open_shell_independent_spins` | per-spin HF at index 0, α and β lists genuinely distinct |

**Result: `6 passed`, ruff clean.**

---

## 8. Validation results

### Local — FCIDUMP format vs PySCF FCI (machine precision)

I built `diag_uhf` locally (`-D_UHF`) and compared its energy against an independent PySCF `direct_uhf`
FCI for several genuinely open-shell systems (na ≠ nb):

| System | norb | nelec (a,b) | SBD energy | vs FCI |
|--------|------|-------------|-----------|--------|
| Li | 5 | (2,1) | −7.3158365529 | 3.6e-15 |
| HeH | 2 | (2,1) | −3.1092383226 | 4.4e-16 |
| OH | 6 | (5,4) | −74.3871847441 | 1.1e-13 |

### Local — against the authoritative H2O reference

Fed my `-D_UHF` binary the reference `fcidump-uhf.txt` + text bitstrings, and separately fed it my own
Python writer's 3-block FCIDUMP — both reproduce the documented energies:

| Subspace | My result | Reference (README) |
|----------|-----------|--------------------|
| 1em3 (7.56e4 dets) | −76.23595 | −76.2359376 |
| 1em4 (2.38e6 dets) | −76.24296 | −76.2429495 |

(Both within the `1e-4` Davidson tolerance. My 3-block writer gave the *same* energy as the reference's
4-block file — confirming the symmetry reduction is correct.)

### Fugaku — closed-loop run (O₂ triplet)

Built both binaries cleanly on Fugaku (`mpiFCCpx`, `-D_UHF`). With the `concurrent` task runner the full
flow runs end-to-end:

- `Task runner mode: concurrent` · `Electronic-structure method: uhf`
- **`UCCSD nocc = (9, 7)`** — genuine open-shell, α ≠ β
- `E(UCCSD) = −147.4379` (correlated, below the ROHF −147.632 reference)
- 4 walkers built 20-qubit spin-unbalanced LUCJ circuits, transpiled, and submitted real jobs to IBM
  hardware (job IDs returned) → samples → subsample α/β → `pjsub diag_uhf` on a Fugaku compute node.

This exercised every layer: UHF integrals → spin-unbalanced ansatz → independent α/β sampling →
spin-resolved FCIDUMP + separate determinant files → the `diag_uhf` binary → per-spin carryover back into
the optimizer.

---

## 9. Summary table

| Layer | File | Before (RHF-only) | After (UHF) |
|-------|------|-------------------|-------------|
| Integrals | `chem.py` | single tensor + `cc.CCSD`, `scf.RHF` | +β/αβ/ββ blocks, `cc.UCCSD`, `scf.UHF` |
| Ansatz | `lucj.py` | `assert not open_shell`, `UCJOpSpinBalanced` | `UCJOpSpinUnbalanced` + 3-tuple pairs |
| Sampling | `sqd.py` | merge spins → one list | `subsample_open_shell` → `(ci_a, ci_b)` |
| Solver I/O | `solver_job.py` | single RHF dump, AlphaDets only | spin FCIDUMP + Beta dets, `method` axis |
| C++/build | `native/` | `bdet = adet`, one carryover | `--bdetfile`, `carryover_b.bin`, `UHF=1` |
| Wiring | `create_blocks.py`, `main.py` | one key, no method | method × backend keys, `method=uhf` |
| Tests | `tests/test_uhf.py` | — | 6 regression tests |

**Net: +960 / −79 across 12 files, 7 commits, RHF path unchanged throughout.**
