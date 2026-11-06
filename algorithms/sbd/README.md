# SBD Closed-Loop Workflow on qcsc-prefect (Miyabi/Fugaku)

This workflow performs SQD closed-loop optimization and runs SBD diagonalization
jobs on Miyabi or Fugaku through the current `qcsc-prefect` architecture.

The old `prefect-sbd` / `prefect-miyabi` runtime path is replaced by:

- `qcsc_prefect_blocks` (`CommandBlock`, `ExecutionProfileBlock`, `HPCProfileBlock`)
- `qcsc_prefect_executor.run_job_from_blocks(...)`
- local `SBD Solver Job` block (`sbd.solver_job.SBDSolverJob`)

Compatibility notes from old `prefect_sbd.sbd_job`:

- keeps the same block slug: `sbd_solver_job`
- keeps the same solver arguments (`task_comm_size`, `adet_comm_size`, `bdet_comm_size`, `block`, `iteration`, `tolerance`, `carryover_ratio`, `solver_mode`)
- keeps the same expected output files (`davidson_energy.txt`, `occ_a.txt`, `occ_b.txt`, `carryover.bin`)

## Prerequisites

- You can submit HPC jobs on your target system (`qsub` / `qstat` for Miyabi, `pjsub` / `pjstat` for Fugaku).
- Prefect API is reachable (On-Prem or Cloud profile is active).
- `QuantumRuntime` block exists (for example `ibm-runner`).
- SBD executable (`diag`) is already built on your target HPC system and you know its absolute path.

Build examples:

```bash
# Miyabi
cd /work/gz00/z12345/qcsc-prefect/algorithms/sbd/native
bash ./build_sbd.sh
realpath ./diag

# Fugaku
cd /work/gz00/z12345/qcsc-prefect/algorithms/sbd/native
bash ./build_sbd_fugaku.sh
realpath ./diag
```

## Install

From repository root:

```bash
cd /work/gz00/z12345/qcsc-prefect
source ~/venv/prefect/bin/activate

# qcsc-prefect local packages
uv pip install --no-deps \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor

# algorithm dependencies
uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/sbd
```

## Create SBD Blocks (Miyabi/Fugaku)

Use the config file that matches your target:

```bash
cd /work/gz00/z12345/qcsc-prefect
```

Miyabi:

```bash
cp algorithms/sbd/sbd_blocks.example.toml algorithms/sbd/sbd_blocks.toml
vim algorithms/sbd/sbd_blocks.toml
```

Fugaku:

```bash
cp algorithms/sbd/sbd_blocks.fugaku.example.toml algorithms/sbd/sbd_blocks.toml
vim algorithms/sbd/sbd_blocks.toml
```

Run block generator:

Miyabi:

```bash
python algorithms/sbd/create_blocks.py --config algorithms/sbd/sbd_blocks.toml --hpc-target miyabi
```

Fugaku:

```bash
python algorithms/sbd/create_blocks.py --config algorithms/sbd/sbd_blocks.toml --hpc-target fugaku
```

`sbd_executable` should point to your built binary, for example:

```toml
sbd_executable = "/work/project/qcsc-prefect/algorithms/sbd/native/diag"
```

For memory stability on Miyabi, start from:

```toml
mpiprocs = 4
mpi_options = ["-np", "4"]
task_comm_size = 1
adet_comm_size = 1
bdet_comm_size = 1
block = 4
iteration = 1
tolerance = 0.01
carryover_ratio = 0.1
```

Then increase only if your target case runs without OOM.

For Fugaku, `algorithms/sbd/sbd_blocks.fugaku.example.toml` already matches the tutorial baseline
(`num_nodes = 2`, `mpiprocs = 2`, `mpi_options = ["-n", "2"]`, `script_filename = "sbd_solver.pjm"`),
so in most cases you only need to update `group`, `queue`, `work_dir`, and `sbd_executable`.
For backward compatibility, `project` is still accepted as a legacy alias in Fugaku mode.

This creates:

- `CommandBlock` (default: `cmd-sbd-diag`)
- `ExecutionProfileBlock` (default: `exec-sbd-mpi` for Miyabi, `exec-sbd-fugaku` for Fugaku)
- `HPCProfileBlock` (default: `hpc-miyabi-sbd` for Miyabi, `hpc-fugaku-sbd` for Fugaku)
- `SBD Solver Job` block (default: `davidson-solver`)

Sampling options are flow parameters (`FlowParameters.sampler`), not a Prefect Variable:
`shots` (default 10 000), `n_shot_batches`, `dynamical_decoupling` / `dd_sequence`,
`measure_twirling`, `gate_twirling`, `saved_samples`. `create_blocks.py` no longer accepts
`--shots`, `--n-shot-batches`, `--saved-samples`, `--sqd-options-json`, `--options-variable-name`,
the DD / twirling flags or their `SBD_*` / config equivalents; it stops with a pointer to the
parameter instead.

## Run flow

Local execution from JSON parameters:

```bash
python algorithms/sbd/exec.py algorithms/sbd/default_params/test_n2.json
```

Note:
- The example JSON files use intentionally low `circ_params.sabre_layout_trials` values so Fugaku runs do not appear stuck in SABRE layout search before IBM Quantum submission.
- These are conservative starting points only. Tune them for your own backend, qubit count, and circuit specification.
- Set `"quantum_source": "random"` in the JSON if you want to skip IBM Quantum Runtime and use deterministic pseudo-random bitstrings instead. `"random_seed"` controls reproducibility in that mode.
- Shots and error mitigation: `"sampler": {"shots": 50000, "dynamical_decoupling": true, "measure_twirling": true}`.
- `FlowParameters` rejects unknown keys.

Deploy entrypoint:

```bash
# For Fugaku + Ray stability (recommended)
export PREFECT_RAY_NUM_CPUS=2

sbd-deploy
```

## Optional integrations

- If `s3-sqd` block is configured (`prefect-aws` S3 block), intermediate arrays
  are saved to object storage.
- If not configured, data falls back to Prefect local storage.
