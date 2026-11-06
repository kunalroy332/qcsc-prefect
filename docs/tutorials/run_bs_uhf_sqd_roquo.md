# Run BS-UHF SQD on ROQUO (qcsc-prefect)

This tutorial runs Sample-based Quantum Diagonalization (SQD) of an Fe–S cluster from a
**broken-symmetry UHF (BS-UHF) reference**, using the `qcsc-prefect` architecture. Each run:
- samples a LUCJ circuit on an IBM Quantum device;
- recovers configurations;
- diagonalizes the subspace with the GPU [SBD](https://github.com/r-ccs-cms/sbd) solver on
  ROQUO.

Prefect Cloud orchestrates and records every run.

> **[Image placeholder]** `docs/images/img-closed-loop-roquo.png`: the closed loop on ROQUO
> (manifest → BS-UHF reference → LUCJ on IBM Quantum → recovery + SBD on 4 GPUs → energy),
> with Prefect Cloud recording the runs.

The goal: the ground-state energy of the active space in one manifest from
[Run BS-UNO-AVAS on ROQUO](run_bs_uno_avas_roquo.md), with its spin pattern kept.

## Prerequisites

- [Set up qcsc-prefect on ROQUO](../howto/howto_setup_qcsc_prefect_roquo.md): every check there
  passes.
- At least one manifest in `plugins/bs_uno_avas/workspace/output/` (the AVAS tutorial).

Advanced topics (resuming a recovery, subspace sweeps, solver layout) are in
[BS-UHF SQD on ROQUO: advanced usage](run_bs_uhf_sqd_advanced_roquo.md).

## 0. What changes from the Fugaku closed loop?

If you know [Run SBD Closed-loop Workflow on Fugaku](run_sbd_closed_loop_workflow_fugaku.md):

| | Fugaku tutorial | This tutorial |
|---|---|---|
| Input | an FCIDUMP | an active-space **manifest** (FCIDUMP + labels + BS-UHF RDM + spin tag) |
| Reference | RHF | the manifest's BS-UHF, rebuilt and checked against it (`SBD_BS_STRICT=1`) |
| Solver | CPU `diag`, PJM jobs | GPU `diag-gpu_uhf-mpi-hpcx250`, 4 GPUs per solve, inside the job |
| Blocks | `create_blocks.py` by hand | made per task by the launcher (`bs-solver-<run>`, …) |
| Start | `sbd-deploy` + Prefect UI | `bash run_bs.sh job` (the Prefect UI path is optional, §2 Step 7) |
| Scale | one flow run | one flow run per manifest, one node each, in one Slurm job |

The 3-block model is the same: `SBDSolverJob` wraps a `CommandBlock` (what), an
`ExecutionProfileBlock` (how: ranks, threads) and an `HPCProfileBlock` (where). Here the target
is `local`: the solver runs inside the task's own job step, so no job is submitted per solve.

---

## 1. Big picture

### 1.1 Where it runs

- **Login node:** `run_bs.sh` checks the inputs, asks what to run and submits one Slurm job.
- **Job nodes:** one node per task (manifest). Each task step runs the Prefect flow, whose
  solves use the node's 4 GPUs.
- **IBM Quantum:** LUCJ sampling. It happens once per task; later runs reuse the samples.
- **Prefect Cloud:** flow runs, task runs, logs, the `sqd-telemetry` artifact and the blocks.

### 1.2 Prefect concepts in this workflow

| Concept | Here |
|---|---|
| Flow | `riken_sqd_de` (`algorithms/sbd/sbd/main.py`): reference → UCCSD → LUCJ → walkers |
| Task | `walker_sqd` (`sbd/sqd.py`): sample or load samples, recovery loop, solves |
| Blocks | `sbd_solver_job/bs-solver-<run>` and its three blocks (per task); `quantum-runtime/ibm-runner` (yours, setup Step 6) |
| Parameters | `FlowParameters`, from `[flow]` of `run_bs.toml` + the manifest path |
| Artifact | `sqd-telemetry`: energy, subspace size, reference tag, recovery trace |
| Deployment | optional: `riken_sqd_de/<run>`, served by the job (`MODE=cloud`) |

---

## 2. Tutorial steps

> **[Image placeholder]** `docs/images/img-sbd-setup-flow-roquo.png`: the steps below as a
> diagram (settings → check → job → Prefect Cloud → report).

### Step 1. Log in

**Your PC:**

```bash
ssh roquo
```

Use your own ROQUO login host or SSH alias. Every later block runs on the **login node** and
starts with `source ~/.qcsc_env`.

### Step 2. Review the settings

**Login node.** All tasks of a job share `algorithms/sbd/roquo/run_bs.toml`:

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
${EDITOR:-vi} run_bs.toml
```

| Key | Default | Meaning |
|---|---|---|
| `[flow] sqd_dim` | `1_000_000` | subspace size of `job` |
| `[flow] n_recovery_steps` | `5` | configuration-recovery passes |
| `[flow] quantum_source` | `real-device` | `random` = a pipeline test without IBM |
| `[flow.sampler] shots` | `10_000` | shots on the device (DD `XY4`, measurement twirling) |
| `[run] gpus_per_task` | `4` | GPUs per solve = MPI ranks |
| `[solver] adet_comm_size`, `bdet_comm_size` | `2`, `2` | MPI grid; `task × adet × bdet` = `gpus_per_task` |
| `[slurm] time` | `1-00:00:00` | wall time; shortened to the reservation's end |

`account` / `reservation` stay `""`: they come from `~/.qcsc_env`. For a first test, set
`quantum_source = "random"`, `sqd_dim = 100_000`, `n_recovery_steps = 2`. Better: copy the file
(`cp run_bs.toml test.toml`) and put `CONFIG=test.toml` before every command.

### Step 3. Check the manifests

**Login node:**

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
bash run_bs.sh check
bash run_bs.sh report        # when the job is done
```

`check` takes every manifest in `plugins/bs_uno_avas/workspace/output/`, or the files you name
after it. It submits one short job (one node, a few minutes) that:
1. rebuilds each manifest's BS-UHF reference with the SQD code;
2. validates `[flow]` as `FlowParameters`;
3. checks Prefect Cloud and the `ibm-runner` block from a compute node;
4. probes the solver launch on 4 ranks.

`report` must say **`check: PASS`**, with one line per manifest:

```text
reference N1a_State1_Ered_Rox_S10: norb 16 nelec [12, 11] tag 10 -> 10 (manifest-rdm) dE 6.7e-10
```

The tag must stay the manifest's (`10 -> 10`), with `dE` below `1e-4`.

### Step 4. Start the run

**Login node:**

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
bash run_bs.sh job
```

`job` lists the manifests with their state and asks two questions. Enter accepts the
suggestion in `[...]`:

```text
job: sizes 1000000, 5 recovery steps, config .../run_bs.toml
  0 N1a_State1_Eox_Rox_S01             d1000000 new
  1 N1a_State1_Ered_Rox_S10            d1000000 new
tasks [0 1] (numbers, all):
--- N1a_State1_Eox_Rox_S01
    samples: none
    d1000000: new
    action [run] (run|resume|new|skip):
```

1. **Tasks:** the numbers to run. The suggestion is every task that isn't done.
2. **Action** per task: `run` for a new task. The other actions matter when a task has
   samples or a checkpoint (advanced tutorial, §1).

Then `job` submits **one** Slurm job with one node per task. It prints `Submitted batch job
NNNNNN` and ends with `when the job is done: bash run_bs.sh report`.

- `job` stops before submitting if the newest check did not pass, or did not cover a chosen
  manifest: run `check` again.
- Only some manifests: name them (`bash run_bs.sh job path/to/X_manifest.json`). The task
  question is then skipped.

### Step 5. Follow the run in Prefect Cloud

Open your workspace at [app.prefect.cloud](https://app.prefect.cloud) → **Runs**. Each task is
one flow run of `riken-sqd-de`, named `<run_name>-d<size>`.

> **[Image placeholder]** `docs/images/img-roquo-flow-runs.png`: the Runs page with one
> `riken-sqd-de` flow run per task (`N1a_State1_Ered_Rox_S10-d1000000`, …), running / completed.

Click a flow run. The graph shows the walker task and its solver jobs; the **Logs** tab shows
the lines listed in §4 below.

> **[Image placeholder]** `docs/images/img-roquo-flow-run-graph.png`: one flow run: the
> `walker_sqd` task, the `hpc-*` solver task runs (one per recovery step), and the logs tab with
> `BS-UHF reference: … tag=10 (manifest 10)`.

When the run is finished, open **Artifacts → `sqd-telemetry`**. It has the energy, the subspace
size, `reference_tag` / `reference_source` / `fcidump_sha256`, and the per-step
`recovery_trace`.

> **[Image placeholder]** `docs/images/img-roquo-sqd-telemetry.png`: the `sqd-telemetry`
> artifact table of a finished run.

The same from the terminal (**login node**):

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
squeue -u $USER
tail -f "$(ls -t logs/[0-9]*/[0-9]*_*.log | head -1)"
```

That is the newest task log. There is one per task, in `logs/<jobid>/<k>_<run_name>.log`.

### Step 6. Read the results

**Login node:**

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
bash run_bs.sh report
```

Per task: its samples, and per size `done E=… (5 steps)`, `step k/5 …` (not finished) or `new`.
A task whose job ended without a result shows the lines that say why. The list ends with an
energy table.

| Path (in `algorithms/sbd/roquo/`) | Content |
|---|---|
| `runs/<run_name>/d<size>/result.json` | best energy, tag, `sqd_dim`, flow run id, parameters, recovery trace |
| `runs/<run_name>/samples/` | the sample pool of this task, reused by every later run |
| `runs/<run_name>/d<size>/ckpt/` | recovery checkpoint, written after every step |
| `runs/<run_name>/work/job_*/solver-<rank>.out` | solver output per solve and rank |
| `logs/bs-sqd.<jobid>.out` | job log: checks, task start / end, summary |

### Step 7 (optional). Start the run from the Prefect UI

With `MODE=cloud`, each task node **serves** its deployment and waits for you to start one run
in the UI:

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
MODE=cloud bash run_bs.sh job
```

When the task logs say `[cloud] serving riken_sqd_de/<tag>`, open **Deployments →
riken_sqd_de → that name → Run → Custom run**. The parameters are pre-filled from `run_bs.toml`
and the manifest. Keep `Active Space Manifest` and `Solver Block Ref` as they are, then
**Submit**.

> **[Image placeholder]** `docs/images/img-roquo-deployment.png`: Deployments page with
> `riken_sqd_de/n1a_state1_ered_rox_s10` served by the job.

> **[Image placeholder]** `docs/images/img-roquo-custom-run.png`: Custom run form: Active Space
> Manifest, SQD Subspace Dimension, Recovery Steps, Quantum Source, Solver Block Ref
> `sbd_solver_job/bs-solver-n1a_state1_ered_rox_s10`.

The task step ends when that run is final, and `report` reads its result like any other.

---

## 3. What happens inside the job?

1. **Reachability:** every node calls the Prefect Cloud API. If any task samples on the
   device, it also checks the `ibm-runner` backend. A failure stops the job before any task
   starts, so no IBM shots are spent.
2. **Pre-flight** on the first node:
   - `ldd` of the solver;
   - the exact nested `srun` of a solve runs a probe on 4 ranks, and each rank must see 4 GPUs;
   - the block types are registered;
   - `ibm-runner` is found (it is never overwritten).
3. **Tasks:** one `srun --exclusive -w <node>` step per task, with the whole node.
   `run_task.py` creates the task's blocks and builds `FlowParameters`, then runs
   `riken_sqd_de`:
   - the manifest is loaded, its sha256 hashes checked, and the BS-UHF reference rebuilt (the
     tag must match);
   - UCCSD → LUCJ;
   - `walker_sqd` samples on IBM, or loads the task's samples;
   - configuration recovery runs for `n_recovery_steps`. Each solve is
     `SBDSolverJob.run()` → `run_job_from_blocks()` (target `local`) →
     `srun --mpi=pmix -n4 diag-gpu_uhf-mpi-hpcx250`.
4. **Outputs:** the solver writes `davidson_energy.txt`, `occ_a/b.txt` and
   `carryover*.bin`. The flow logs telemetry; the task writes `result.json` and the
   `CAPTURE_BS_SQD` line that `report` reads.

## 4. What to check in a task log

1. `BS-UHF reference: reference_source=manifest-rdm tag=T (manifest T) E=…`: the tag matches,
   and E is within `1e-4` Eh of the manifest's `e_active_uhf`.
2. `Reference: reference_source=… tag=… fcidump_sha256=… norb=16|32 nelec=(na, nb)`.
3. `[task] solver: 4 rank(s) = task 1 x adet 2 x bdet 2, 36 threads/rank …`.
4. Sampling run:
   - `Submitting sampler workload … (shots=10000 …)`;
   - `[persist] merged …-shot pool … saved`.

   Later runs instead:
   - `[saved] loaded …-shot pool (2·norb bits)`;
   - `[saved] pool … matches the current reference`.
5. One `[diag] recovery k/5 …` block per step, each ending with `[ckpt] wrote …`.
6. `CAPTURE_BS_SQD <run_name>-d<size>: best_E=… tag=T -> …/result.json`.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `STOP: …` from `run_bs.sh` | It names the problem (no passing check, a bad answer, grid ≠ GPUs, a missing binary); fix it and run again. |
| `check: FAIL` / `PRE-FLIGHT FAILED` | `report` shows the failing line; `logs/<jobid>/preflight.log` has the rest. A missing library: setup Step 4. |
| `REACHABILITY CHECK FAILED` | `prefect-cloud: FAIL`: `prefect cloud login` again (setup Step 5). `ibm-quantum: FAIL`: the `ibm-runner` block, or the backend is down. |
| `No converged BS-UHF start reproduced the manifest spin tag` | The manifest's reference doesn't reproduce: send the task log and the manifest. |
| `SBDSolverJob failed: exit_status=…` | Read `solver-0.err` in the newest `runs/<run_name>/work/job_*/`. |
| GPU out of memory | Raise `bdet_comm_size` (keep `task × adet × bdet` = `gpus_per_task`) or lower `sqd_dim`. |

Anything else: send the task log that `report` names.
