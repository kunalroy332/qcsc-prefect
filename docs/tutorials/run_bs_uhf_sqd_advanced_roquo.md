# BS-UHF SQD on ROQUO: Advanced Usage

This page builds on [Run BS-UHF SQD on ROQUO](run_bs_uhf_sqd_roquo.md). It covers:

1. [Resuming a configuration recovery](#1-resume-a-configuration-recovery) after a wall-time
   cut, a node fault or a solver timeout, and adding more steps.
2. [Samples](#2-samples-reuse-new-samples-random-saved-pools): reuse, new samples, random and
   saved pools.
3. [A subspace-size sweep](#3-subspace-size-sweep) on the same samples.
4. [The solver layout](#4-solver-layout-and-gpu-memory) and GPU memory.
5. [Recovery options](#5-recovery-options).
6. [Manifest checks](#6-manifest-checks-and-the-reference-cache) and the reference cache.
7. [BS-UHF without a manifest](#7-bs-uhf-without-a-manifest-af_groups) (`AF_GROUPS`).
8. [Your own allocation](#8-your-own-allocation-salloc).
9. [Reference](#9-reference): `run_bs.sh` env knobs, `[flow]` keys.

Every block runs on the **login node** unless it says otherwise.

---

## 1. Resume a configuration recovery

### What is saved

After every recovery step, `walker_sqd` (`algorithms/sbd/sbd/sqd.py`) writes a checkpoint
`ckpt_t00_w0.npz` (one per DE trial and walker) into the size's `ckpt/` folder:

| Field | Meaning |
|---|---|
| `occ_a`, `occ_b` | α / β orbital occupancies that steer the next recovery pass |
| `next_step` | the first step not yet done |
| `best_energy` | the lowest energy so far (a resumed run never reports a higher one) |
| `carryover`, `carryover_b` | the α / β carryover determinants of the best step |

The next run on the same size reads it and skips the steps already done:

```text
[task] checkpoint: resuming ckpt_t00_w0.npz
[ckpt] RESUME from .../d1000000/ckpt/ckpt_t00_w0.npz: completed 3 step(s), best_energy so far=-5067.71
```

### When a checkpoint is resumed

A checkpoint belongs to **one** recovery trajectory. `run_task.py` resumes it only if all of
these are the same as when it was written (`ckpt/source.json`):

- the **samples**: the task's pool (`runs/<run_name>/samples/`), or for `random` the same
  `random_seed` and `shots`;
- the **FCIDUMP** (sha256) and the **spin tag** of the manifest;
- **`sqd_dim`** (each size has its own folder `d<size>/`) and **`n_batches`**.

Otherwise the checkpoint is moved to `d<size>/ckpt.prev-<time>/` (kept, never deleted) and the
size starts at step 1; the task log says why: `[task] checkpoint: not resumed (…)`.
`n_recovery_steps` may change, which is how you add steps (below).

### Resume a cut-off run

Run `job` again. It finds the checkpoint and suggests **`resume`**:

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
bash run_bs.sh report
bash run_bs.sh job
```

```text
  0 N1a_State1_Eox_Rox_S01             d1000000 done E=-5067.70 (5 steps)
  1 N1a_State1_Ered_Rox_S10            d1000000 step 3/5 E=-5067.712345
tasks [1] (numbers, all):
--- N1a_State1_Ered_Rox_S10
    samples: pool 2026-11-03 (10000 shots) .../raw_samples_t00_w0_9f2c.npz
    d1000000: step 3/5 E=-5067.712345
    action [resume] (run|resume|new|skip):
```

- Done tasks are not suggested.
- The resumed task recovers from its pool (`quantum_source = "saved"` is set for you), so no IBM
  shots are spent. The job skips the IBM check when no task samples.
- In the task log, check `[task] checkpoint: resuming …` and `[ckpt] RESUME … completed 3
  step(s)`. The first new step is `[diag] recovery 4/5`.
- If `report` shows `checkpoint at step k/5 … not resumable (other n_batches)` (or another
  reason), `job` suggests `new`. Either change the setting back to resume, or accept a restart.

The choices, also as `RESUME=…` for every task without a question:

| Action | Effect |
|---|---|
| `run` | what the state needs: sample if the task has no pool, else recover from the pool; a valid checkpoint is resumed |
| `resume` | the same as `run` (the word for a task with a checkpoint) |
| `new` | every size from step 1 again, on the same samples; old checkpoints and results are kept as `*.prev-*` |
| `skip` | not in this job |

`RESUME=resume bash run_bs.sh job` resumes everything that can be resumed, with no questions.
It is useful after a reservation ended under a large job.

### Run more recovery steps

Raise `n_recovery_steps` (e.g. 5 → 8) in your config. A size that was done with 5 steps then
shows `step 5/8` and resumes at step 6:

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
cp run_bs.toml more.toml
${EDITOR:-vi} more.toml          # [flow] n_recovery_steps = 8
CONFIG=more.toml bash run_bs.sh report
CONFIG=more.toml bash run_bs.sh job
```

This works because the finished run's checkpoint stays in `ckpt/` with `next_step = 5`.

### Resuming outside `run_bs.sh`

The mechanism is the flow's own, and only needs one environment variable. To resume with your
own driver, set `FE4S4_CKPT_DIR`:

```bash
export FE4S4_CKPT_DIR=/path/to/run/ckpt    # same dir as the cut-off run
```

The generic name is historic; it works for any molecule. Then run `riken_sqd_de` with the same
`FlowParameters`:
- the same `active_space_manifest` (or `fcidump`), `sqd_dim` and `n_batches`;
- `quantum_source = "saved"` with `sampler.saved_samples = ["file:///…/raw_samples_….npz"]`,
  the pool of the first run.

Your driver must check these itself; the flow doesn't.

---

## 2. Samples: reuse, new samples, random, saved pools

- **One sampling per task.** The first run of a task samples on the device. It saves the pool
  once, before the first solve, in `runs/<run_name>/samples/storage/sqd_data/…`. Every later
  `job` / `sweep` of that task recovers from its **newest** pool.
- **New samples:** `RESAMPLE=1 bash run_bs.sh job`. The first size samples again; the old pool
  stays on disk, and checkpoints of the old pool are set aside.
- **Pipeline test without IBM:** `quantum_source = "random"` in a config copy. Random
  bitstrings come from `random_seed`; no pool is written, and nothing is reused as device
  samples.
- **Your own pool:** `quantum_source = "saved"` and
  `[flow.sampler] saved_samples = ["/abs/path/raw_samples_t00_w0_….npz"]` in a config copy.
  A plain absolute path or `file://…` both work.
- **Pool guard:** every pool records the reference it was sampled for (FCIDUMP sha256, tag,
  orbital count). A pool of another active space is always refused (wrong bit count). A pool of
  another reference stops the run: `saved pool … different reference`.
  `SBD_ALLOW_POOL_MISMATCH=1` turns that into a warning, but only use it knowingly.
- **Sampler settings** (`[flow.sampler]`): `shots`, `n_shot_batches` (device submissions),
  `dynamical_decoupling` / `dd_sequence`, `measure_twirling`. Keep `gate_twirling = false` for
  LUCJ (IBM error 1519). `[flow.circ_params]`: `n_lucj_layers`, `use_error_aware_layout`,
  `ab_stride`, `layout_connectivity`. Keep them unchanged between the runs on one pool.

## 3. Subspace-size sweep

`sweep` runs the recovery once per size of `[sweep] sqd_dims`, one after another on each task's
node, all on the task's samples. Each size has its own `d<size>/` with its result and checkpoint,
exactly as `job` makes them. So a size done by `job` is skipped by `sweep`, and the reverse.

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
${EDITOR:-vi} run_bs.toml   # [sweep] sqd_dims = [100_000, 1_000_000]
bash run_bs.sh sweep
bash run_bs.sh report            # when the job is done
```

- Keep the `sqd_dims` list on one line.
- A task without samples samples at the first size; the other sizes reuse that pool.
- A failed size doesn't stop the next one. The exception is the sampling size: if it fails before
  saving its pool, the task stops.
- The sizes' solve times add up within one wall time (`[slurm] time`, or `TIME=2-00:00:00`). A
  size cut off by it resumes at the next `job` or `sweep`.
- `report` ends with an energy table of every finished size.
- The sweep runs in direct mode only (no `MODE=cloud`).

## 4. Solver layout and GPU memory

Each solve runs `diag-gpu_uhf-mpi-hpcx250` with one MPI rank per GPU:

| Key | Default | Meaning |
|---|---|---|
| `[run] gpus_per_task` | `4` | ranks per solve; `1` uses the 1-GPU binary `diag-gpu_uhf` |
| `[solver] task_comm_size × adet_comm_size × bdet_comm_size` | `1 × 2 × 2` | must equal `gpus_per_task` |
| `[solver] threads` | `0` = node CPUs / ranks (36) | OpenMP threads per rank |
| `[solver] iteration`, `block` | `5`, `20` | Davidson iteration cap, subspace size |
| `[solver] timeout_seconds` | `72000` | per solve |
| `[slurm] mem`, `gpus_per_node` | `1600000M`, `4` | per node; GPUs must be asked for explicitly |

- `check` and `job` stop on a grid whose product isn't `gpus_per_task`, or whose binary is
  missing.
- **GPU out of memory:** memory per rank shrinks with `bdet_comm_size`. Use `1 × 1 × 4`, or
  lower `sqd_dim`.
- Sizing math (determinants per rank, memory per determinant):
  [HPC resource sizing](../reference/hpc_resource_sizing.md).

## 5. Recovery options

All in `[flow]` (FlowParameters; an unknown key is an error, so a typo can't run silently).

| Key | Default | Effect |
|---|---|---|
| `n_batches` | `1` | subsample batches per recovery step (each is a solve); part of the resume key |
| `seed_cisd` | `0` | classical seeding of the subspace: 1 singles, 2 doubles, 3 both |
| `seed_budget_frac` | `1.0` | the share of the budget the seed may take (rest: sampled higher excitations) |
| `hci_boost`, `hci_boost_max`, `hci_boost_ncore` | `0`, `20000`, `200` | after each solve, add the singles/doubles of the top carryover determinants and solve again |
| `[solver] carryover_ratio`, `carryover_type` | `0.5`, `1` | determinants carried to the next step |
| `[solver] do_rdm` | `0` | `1`: RDMs and orbital optimization between DE trials (`oo_*` keys) |
| `[flow.de_params] num_walkers`, `iterations` | `1`, `1` | differential evolution of the LUCJ parameters |

With several walkers or DE trials there is one checkpoint per (trial, walker), named
`ckpt_t<trial>_w<walker>.npz`. `report` reads the first one.

## 6. Manifest checks and the reference cache

The flow rebuilds the BS-UHF reference from the manifest:
1. it starts from the RDM and from a label guess, with second-order SCF and stability
   following;
2. among the converged solutions that reproduce the manifest's tag, it takes the lowest.

These env vars, set before `bash run_bs.sh job`, are passed to the job:

| Variable | Default | Effect |
|---|---|---|
| `SBD_BS_STRICT` | `1` | a tag / energy / convergence mismatch vs the manifest stops the task (`0`: warning) |
| `SBD_BS_E_TOL` | `1e-4` | allowed \|E_UHF − e_active_uhf\| (Eh) |
| `SBD_SKIP_MANIFEST_HASH` | `0` | `1`: a sha256 mismatch of a manifest file only warns |
| `SBD_UHF_CACHE` | `1` | `0`: skip the lookup in the per-task cache `runs/<run_name>/uhf_cache/` (it is still written) |

The cache is keyed by content (FCIDUMP sha256, manifest files, tag), so every size and rerun of a
task starts from identical orbitals.

## 7. BS-UHF without a manifest (`AF_GROUPS`)

Without an AVAS manifest, the flow can build a BS guess from **orbital-index groups** of a plain
FCIDUMP (`FlowParameters.fcidump`, solver `method = "uhf"`). Set the spec in the environment of
the flow process. The `FE4S4_` names are historic; nothing in them is Fe4S4-specific.

```bash
export FE4S4_AF_GROUPS='{"metal_a": [2, 3, 4, 5, 6],
  "bridge": [7, 8, 9, 10], "metal_b": [11, 12, 13, 14, 15],
  "up": ["metal_a"], "down": ["metal_b"]}'
```

- Groups not listed in `up` / `down` are closed (doubly occupied). The indices must cover every
  orbital once.
- `FE4S4_AF_POL` (default `1.0`) sets how sharp the up/down split is. If the result is strongly
  spin-contaminated, try `0.6`.
- `FE4S4_AF_FREE_S=1` seeds a group named `s` half-filled instead of closed (a bridging ligand
  that mediates exchange).
- `FE4S4_AF_GROUPS=fe4s4` is a preset for the 36-orbital Fe4S4 FCIDUMP only. Other sizes give a
  `ValueError`; use a manifest for AVAS active spaces.

How to pick the method: RHF for a closed-shell singlet; UHF (no groups) for a plain open-shell
system; BS-UHF for an `MS2 = 0` system with separated, oppositely polarizable centres. A UHF
started from RHF stays at RHF there unless it is seeded. `run_bs.sh` always uses manifests; this
route needs your own driver, e.g.
`examples/sbd_uhf_recover_any_molecule/run_recover.py --af-groups … --ckpt-dir …`.

## 8. Your own allocation (`salloc`)

`ALLOC=1` asks the same questions as `job`, then prints the matching `salloc` instead of
submitting:

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
ALLOC=1 bash run_bs.sh job
```

Run the printed `salloc` line. In the shell it opens, run the printed
`SAMPLES_DEVICE=… bash run_bs.sh run` after `source ~/.qcsc_env` and the `cd`. It runs the
tasks of `logs/tasks-last.txt`, with the same checks and logs as a submitted job.

## 9. Reference

### `run_bs.sh` environment knobs

Set before the command, e.g. `RESUME=resume bash run_bs.sh job`. None is required.

| Variable | Default | Meaning |
|---|---|---|
| `CONFIG` | `run_bs.toml` | another settings file (relative to the cwd or `roquo/`) |
| `TASKS` | all manifests | a file with one manifest path per line (no task question) |
| `RESUME` | ask | `run`, `resume`, `new` or `skip` for every task |
| `RESAMPLE` | unset | `1`: new device samples for the chosen tasks |
| `SKIP_CHECK` | unset | `1`: submit without a passing `check` |
| `MODE` | `[run] mode` | `direct` / `cloud` (`job` only; the SQD tutorial, Step 7) |
| `TIME` | `[slurm] time` | wall time of the job (check: `00:30:00`) |
| `ALLOC` | unset | `1`: print `salloc` instead of submitting (§8) |
| `RESV_MARGIN` | `600` | seconds kept free before the reservation's end |
| `SKIP_REACH` | unset | `1`: skip the reachability check in the job (not recommended) |
| `SBD_BS_STRICT`, `SBD_BS_E_TOL`, `SBD_ALLOW_POOL_MISMATCH` | `1`, `1e-4`, `0` | §6, §2 |

### `[flow]` keys (FlowParameters)

| Key | Default | Meaning |
|---|---|---|
| `sqd_dim` | `1_000_000` | subspace size of `job` |
| `n_recovery_steps`, `n_batches` | `5`, `1` | §1, §5 |
| `quantum_source` | `real-device` | `random`, `saved` (§2) |
| `random_seed`, `seed_cisd` | `24`, `0` | §2, §5 |
| `[flow.sampler]` | 10 000 shots, DD `XY4`, measurement twirling | §2 |
| `[flow.circ_params]` | 1 LUCJ layer, error-aware layout, `ab_stride` 4 | §2 |
| `[flow.de_params]` | 1 walker, 1 iteration | §5 |

`active_space_manifest` and `solver_block_ref` are set per task by the launcher. Field docs:
`algorithms/sbd/sbd/flow_params.py`.

### Files of the launcher (`algorithms/sbd/roquo/`)

| File | Role |
|---|---|
| `run_bs.sh` | `check` / `job` / `sweep` / `report`; inside jobs `checkjob` / `run` |
| `run_bs.toml` | settings shared by every task |
| `run_task.py` | one task on one node: blocks, parameters, samples, checkpoints, sizes |
| `task_state.py` | task / size states for the login node (`report`, the suggestions) |
| `preflight.py`, `check_reach.py` | the checks before tasks start |
| `bs_common.py` | shared helpers (paths, Prefect storage, manifest check, flow call) |
