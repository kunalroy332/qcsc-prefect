# How to Set Up qcsc-prefect on ROQUO

One-time setup on ROQUO (RIKEN R-CCS: GB200 nodes with aarch64 Grace CPUs and Blackwell GPUs,
Slurm) for the BS-UHF SQD workflow:

- the SBD Python environment and the GPU SBD solver;
- Prefect Cloud, which records every flow run;
- IBM Quantum access through a Prefect block.

When every check on this page passes, go on with:

1. [Run BS-UNO-AVAS on ROQUO](../tutorials/run_bs_uno_avas_roquo.md): makes the
   active-space manifests. It has its own short setup (the gpu4pyscf environment).
2. [Run BS-UHF SQD on ROQUO](../tutorials/run_bs_uhf_sqd_roquo.md).

## Prerequisites

- A ROQUO account, a project (account id) and access to the `roquo` partition.
- An API key for your [Prefect Cloud](https://www.prefect.io/cloud) workspace.
- An API key and an instance CRN for the [IBM Quantum Platform](https://quantum.cloud.ibm.com/).

## Conventions

- **Login node** / **Compute node** above each block says where to run it. The login node has
  no GPU, no CUDA / NVHPC and no Python 3.12. A compute-node shell:

  ```bash
  source ~/.qcsc_env
  srun -A $ACCT ${RESV:+--reservation=$RESV} -p roquo -N 1 \
       --gres=gpu:1 --cpus-per-task=32 -t 01:00:00 --pty bash
  ```

  A new shell starts without your variables: start every block with `source ~/.qcsc_env`.
- Paste code blocks as they are. Text in `<...>` appears only in prose; never type it into a
  command (bash reads `<x>` as "read the file x").
- ROQUO has no `nano`. Files open with `${EDITOR:-vi}`. In `vi`: `i` types, `Esc` stops,
  `:wq` saves and quits, `:q!` quits without saving.

## What runs where

| Component | Environment | Runs on |
|---|---|---|
| SQD flow `riken_sqd_de`, `run_bs.sh` tasks | `$REPO/algorithms/sbd/.venv` | compute nodes |
| SBD solver `diag-gpu_uhf*` | native binary | compute-node GPUs, started by the flow |
| Flow runs, logs, blocks | Prefect Cloud workspace | over HTTPS from the compute nodes |
| Sampling | IBM Quantum Platform | called from the compute nodes |
| `run_bs.sh check / job / report` | bash (+ the system `python3`) | login node |

## Step 1: Set your paths

**Login node.** The block asks for three values and saves them in `~/.qcsc_env`. The repository
usually lives in your project space, not in `$HOME`.

```bash
read -rp "REPO (qcsc-prefect checkout, existing or to clone): " REPO
read -rp "ACCT (ROQUO account id): " ACCT
read -rp "RESV (reservation; Enter = none): " RESV
printf 'export REPO=%q\nexport ACCT=%q\nexport RESV=%q\n' \
    "$REPO" "$ACCT" "$RESV" > ~/.qcsc_env
source ~/.qcsc_env; echo "REPO=$REPO ACCT=$ACCT RESV=${RESV:-none}"
```

- `sacctmgr show assoc user=$USER format=account` lists your accounts.
- `scontrol show reservation` lists the reservations. The scripts use `RESV` only while it is
  active and tell you when they leave it out.
- To change a value, paste the block again.

## Step 2: Clone the repository

**Login node.** Skip the clone if `$REPO` is already a checkout.

```bash
source ~/.qcsc_env
[[ -d $REPO/.git ]] || git clone \
    https://github.com/qiskit-community/qcsc-prefect.git $REPO
cd $REPO && git checkout bk/feature/bs-uhf-sqd-roquo && git pull
```

Inputs that are not in git (PDB excerpts, `.fchk` files) go to
`plugins/bs_uno_avas/workspace/data/`. The AVAS tutorial covers them.

## Step 3: Create the SBD Python environment

The venv must be an aarch64 build made with Python 3.12, which exists only on compute nodes. A
venv made by an x86-64 `uv`, or copied from another machine, fails with `Exec format error`.

**Compute node** (open a shell as in *Conventions*):

```bash
source ~/.qcsc_env
UV=$HOME/.local/bin_arm/uv
[[ -x $UV ]] || curl -LsSf https://astral.sh/uv/install.sh \
  | env UV_INSTALL_DIR=$HOME/.local/bin_arm INSTALLER_NO_MODIFY_PATH=1 sh
cd $REPO/algorithms/sbd
export UV_CACHE_DIR=/tmp/uvcache_$USER TMPDIR=/tmp/uvtmp_$USER
mkdir -p $UV_CACHE_DIR $TMPDIR && $UV sync --python /usr/bin/python3.12
```

**Check** (same shell; run from `/tmp` so `./sbd` doesn't shadow the package):

```bash
file -L $REPO/algorithms/sbd/.venv/bin/python | grep -o 'ARM aarch64'
cd /tmp && $REPO/algorithms/sbd/.venv/bin/python -c \
  "import sbd, qcsc_workflow_utility, prefect_qiskit; print('sbd OK')"
```

An existing `.venv` that `file` reports as `x86-64`: move it aside
(`mv .venv .venv.x86_64`) and run `$UV sync` again.

## Step 4: Build the GPU SBD solver

The workflow needs the **UHF** binaries (`-D_UHF`). Both are built from the same source by
`native/build_sbd_gpu.sh`:

| Binary | Use |
|---|---|
| `native/diag-gpu_uhf` | 1 GPU per solve |
| `native/diag-gpu_uhf-mpi-hpcx250` | 4 GPUs per solve, one MPI rank per GPU (the default) |

**Compute node** (the build needs the NVIDIA HPC SDK, not a GPU):

```bash
source ~/.qcsc_env
module load cuda/13.2 nvhpc/26.5 && module load hpcx/2.50
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/26.5/compilers/bin:$PATH
cd $REPO/algorithms/sbd/native && { [[ ! -d sbd ]] || git -C sbd pull; }
UHF=1 bash build_sbd_gpu.sh 2>&1 | tee build-gpu1.log
MPI=1 UHF=1 bash build_sbd_gpu.sh 2>&1 | tee build-gpu4.log
```

Each build ends with `Build completed: …/diag-gpu_uhf` (or `…-mpi-hpcx250`). Warnings from the
upstream SBD headers are harmless.

**Smoke test** (**login node**, a batch job of about a minute). Both binaries diagonalize a
small N2 space; each energy must equal the exact minimum computed with PySCF:

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/native && mkdir -p smoke
sbatch -A $ACCT ${RESV:+--reservation=$RESV} smoke_gpu.sh
```

When `squeue -u $USER` no longer lists the job:
`tail -30 "$(ls -t smoke/sbd-smoke-*.log | head -1)"` must end with `SMOKE OK`. A few
`TL_UCP ERROR asymmetric … memory types` lines from HPC-X are harmless.

## Step 5: Connect Prefect Cloud

Create an API key in Prefect Cloud: **Settings → API Keys**.

![Create API Keys](../images/img-prefect-api.png)

**Compute node** (`prefect` is in the SBD venv). Choose *paste an API key* when asked; there is
no browser on the node:

```bash
source ~/.qcsc_env
export PATH=$REPO/algorithms/sbd/.venv/bin:$PATH
prefect cloud login
prefect config view | grep PREFECT_API_URL
```

It must show `PREFECT_API_URL='https://api.prefect.cloud/…' (from profile)`. The profile is in
`~/.prefect/profiles.toml` and every job uses it.

## Step 6: Create the IBM Quantum blocks

The flow samples through the Prefect block **`quantum-runtime/ibm-runner`**. You create it once
in the Prefect UI, and every job then uses it as it is.

**Compute node** (same venv as Step 5). Register the block types, then create the credentials
block:

```bash
source ~/.qcsc_env
export PATH=$REPO/algorithms/sbd/.venv/bin:$PATH
prefect block register -m prefect_qiskit
prefect block register -m prefect_qiskit.vendors
prefect block create ibm-quantum-credentials
```

The last command prints a URL. Open it on your PC, then enter your instance's **CRN** and
**API key** and save the block (any name).

![Setup IBM Quantum Credentials](../images/img-ibm-cred-block.png)

Then create the runtime block:

```bash
prefect block create quantum-runtime
```

In the form that opens:
- **Block name:** `ibm-runner` exactly; the flow loads it by this name.
- **Resource name:** your backend, e.g. `ibm_kingston`.
- **Credentials:** the block from above.
- Leave the primitive options at their defaults. The flow sets shots, dynamical decoupling and
  twirling itself (`[flow.sampler]` in `run_bs.toml`).

![Setup Quantum Runtime](../images/img-quantum-runtime-block.png)

**Check:**

```bash
prefect block ls | grep -E 'ibm-quantum-cred|quantum-runtime/ibm-runner'
```

Both lines must appear. If they don't, the Prefect profile is not active: repeat Step 5.

> **No browser at hand?** Put `IBM_API_KEY`, `IBM_CRN` and `IBM_BACKEND` in
> `$REPO/secrets/secrets.env` (template `secrets/secrets.env.example`, `chmod 600`). The first
> job then creates `ibm-runner` from them. An existing `ibm-runner` block is never overwritten.

## Step 7: Check everything from a compute node

**Login node.** `run_bs.sh check` submits a short job (one node, a few minutes). The job checks:
- the SBD venv;
- Prefect Cloud and IBM Quantum from a compute node, through the `ibm-runner` block;
- the solver binary and its libraries;
- a probe of the 4-rank solver launch.

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
bash run_bs.sh check
bash run_bs.sh report        # when the job is done
```

`report` must say `check: PASS`. Before any manifest exists, the check covers the environment
only; it adds the manifests once you have them (SQD tutorial, Step 3).

## Checklist

| Item | Check |
|---|---|
| Paths | `source ~/.qcsc_env` prints your `REPO` / `ACCT` |
| SBD env | `sbd OK` (Step 3) |
| Solver | `SMOKE OK` (Step 4) |
| Prefect Cloud | `(from profile)` in `prefect config view` (Step 5) |
| IBM | `quantum-runtime/ibm-runner` in `prefect block ls` (Step 6) |
| Everything from a compute node | `check: PASS` (Step 7) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `module: command not found`, `python3.12: No such file` | You are on the login node: open a compute-node shell. |
| `.venv/bin/python: … Exec format error` | x86-64 venv: Step 3 with the aarch64 `uv` (move the old `.venv` aside). |
| `--account=` empty, `REPO` unset, `STOP: set ACCT` | `source ~/.qcsc_env` in this shell (Step 1). |
| `No Prefect API URL provided` after the login | An empty `PREFECT_API_URL` hides the profile: `unset PREFECT_API_URL PREFECT_API_KEY`. |
| `SMOKE FAILED` | Send `native/smoke/sbd-smoke-*.log`; don't use the binaries. |
| `-bash: …: No such file or directory` | A `<...>` was typed into a command; paste the block as it is. |

Anything else: send the newest log that `bash run_bs.sh report` names.
