# Run BS-UNO-AVAS on ROQUO (from BS-UHF or BS-DFT)

This tutorial builds the active space of an Fe–S cluster for BS-UHF SQD. BS-UNO-AVAS runs AVAS
on the unrestricted natural orbitals (UNOs) of a broken-symmetry (BS) solution of the full
system. That solution is either a BS-UHF that the plugin computes from a PDB file, or a BS-DFT
solution read from a Gaussian `.fchk` file. Each run writes one folder with an FCIDUMP, the
orbital labels, the alpha and beta 1-RDMs of the BS reference, and a **manifest** that ties
these files together. A consumer, such as an SQD flow, reads only the manifest.

```
.pdb  ─► high-spin UHF ─► BS-UHF per spin pattern ─┐
.fchk ─► conversion check (Gaussian orbitals) ─────┴► UNO-AVAS (Fe 3d + μS 3p)
      ─► labels ─► FCIDUMP + RDM ─► verify ─► output/<run>/<run>_manifest.json
```

| Route | Input | Script | Cost |
|---|---|---|---|
| A. UHF | protonated `.pdb` file with a cluster residue `FES` or `SF4` (other names: `CLUSTER_RESNAME`) | `roquo/run_pdb.sh` | hours for each state (full-system SCFs) |
| B. DFT | Gaussian BS-DFT `.fchk` file | `roquo/run_fchk.sh` | minutes (no full-system SCF) |

The code is the `bs_uno_avas` plugin (`plugins/bs_uno_avas/`). To run it without Slurm, see
[Run on another machine](#run-on-another-machine).

## Prerequisites

- A ROQUO account, a project (account id) and access to the `roquo` partition.
- Your paths in `~/.qcsc_env`, and a clone of the repository.

On the **login node**, the following block asks for three values and saves them in
`~/.qcsc_env`. The repository usually lives in your project space, not in `$HOME`. Skip the
clone if `$REPO` is already a checkout.

```bash
read -rp "REPO (qcsc-prefect checkout, existing or to clone): " REPO
read -rp "ACCT (ROQUO account id): " ACCT
read -rp "RESV (reservation; Enter = none): " RESV
printf 'export REPO=%q\nexport ACCT=%q\nexport RESV=%q\n' \
    "$REPO" "$ACCT" "$RESV" > ~/.qcsc_env
source ~/.qcsc_env; echo "REPO=$REPO ACCT=$ACCT RESV=${RESV:-none}"
[[ -d $REPO/.git ]] || git clone \
    https://github.com/qiskit-community/qcsc-prefect.git $REPO
cd $REPO && git checkout bk/feature/bs-uno-avas && git pull
```

- `sacctmgr show assoc user=$USER format=account` lists your accounts.
- `scontrol show reservation` lists the reservations. The scripts use `RESV` only while it is
  active and tell you when they leave it out.
- A new shell starts without your variables: start every block with `source ~/.qcsc_env`.

## Set up the gpu4pyscf environment

You do this one time. The SCFs run on one GPU with gpu4pyscf. PyPI has no aarch64 gpu4pyscf
wheel, so the script builds gpu4pyscf 1.9 from source (`sm_100`, CUDA 13.2) into
`$HOME/venvs/gpu4pyscf`. The build takes about an hour.

On a **compute node** with 32 CPUs:

```bash
source ~/.qcsc_env
srun -A $ACCT ${RESV:+--reservation=$RESV} -p roquo -N 1 \
     --gres=gpu:1 --cpus-per-task=32 -t 02:00:00 --pty bash
```

In that shell, run the build:

```bash
source ~/.qcsc_env; cd $REPO/plugins/bs_uno_avas/roquo
bash build_gpu4pyscf_roquo.sh 2>&1 | tee build_gpu4pyscf.log
```

The build succeeded if the log ends with `devices 1 (10, 0)`, a `UHF GPU … CPU … diff` smaller
than `1e-7`, and `OK: next …`. You can run the script again. It reuses the finished parts.

If the build fails, use the CPU engine. It runs the same code, only slower. Create the following
venv, then put `ENGINE=cpu` in front of every `run_pdb.sh` command:

```bash
python3.12 -m venv $HOME/venvs/gpu4pyscf
$HOME/venvs/gpu4pyscf/bin/pip install "pyscf==2.14.0" h5py
```

To check the environment, run the plugin tests in the same shell (about a minute):

```bash
source $HOME/venvs/gpu4pyscf/bin/activate; cd $REPO/plugins/bs_uno_avas
export PYTHONPATH=$PWD/src
pip install pytest && python -m pytest tests -q
```

Then, on the **login node**, run a dry run of the public example PDB file:

```bash
source ~/.qcsc_env; cd $REPO/plugins/bs_uno_avas
bash roquo/run_pdb.sh check examples/5XTD_FES_O301_SCH3.pdb
bash roquo/run_pdb.sh report examples/5XTD_FES_O301_SCH3.pdb
```

After the job finishes, `report` must show `check: PASS`.

## Step 1: Put the inputs in the workspace

On the **login node**, create the input folder `workspace/data/`, which git ignores:

```bash
source ~/.qcsc_env; mkdir -p $REPO/plugins/bs_uno_avas/workspace/data
echo $REPO/plugins/bs_uno_avas/workspace/data
```

Copy your `.pdb` and `.fchk` files into the printed folder, with `scp` from your PC or `cp` on
ROQUO.

## Step 2A: Run the UHF route (PDB file)

On the **login node**:

```bash
source ~/.qcsc_env; cd $REPO/plugins/bs_uno_avas
bash roquo/run_pdb.sh check      # dry run, a few minutes per file
bash roquo/run_pdb.sh report     # after the jobs finish
bash roquo/run_pdb.sh build      # BS-UHF + UNO-AVAS + manifest
bash roquo/run_pdb.sh report     # after the jobs finish
```

- The scripts take every `workspace/data/*.pdb`, except the `*_prot.pdb` and `*_cap.pdb` files
  that the plugin writes.
- For each file, `build` asks for the redox states. The suggestion is `ox red` (every
  `[states.*]` table of the config). Press Enter to accept it, or type `red` for one state.
  `build` submits one job for each file and state.
- `build` submits nothing if the newest check of a file did not pass.

In `report`, check these lines:

- `check: PASS`:
  - `cluster`: the residue name, and the number of Fe and bridging S atoms.
  - `Fe-bound thiolates`: 4 Cys for a plant-type Fe2S2.
  - `state ox` and `state red`: the charge, 2S, and the electron count.
  - `DF: … GB`: the size must fit `max_memory_mb`.
  - Every `WARNING`, for example under-valent atoms or an Fe–O distance below 3.3 Å.
- `manifest: OK` (the manifest is written and checked again from its files):
  - `HS UHF guess … gate pass`.
  - `BS UHF 01 -> 01` and `10 -> 10`: every Fe moment is about ±3.5 to ±4.2.
  - `UNO-AVAS:`: for Fe2S2, about 9 fractional NOONs for red and 10 for ox.
  - No label weight is flagged `<- low`.
  - `verify:`: the active-space tag equals the full-system tag.

Options, only if you need them:

- To use some files only, name them after the command, for example
  `bash roquo/run_pdb.sh build workspace/data/site1.pdb`.
- `STATE=red` uses one state for every file, without the question. `SKIP_CHECK=1` builds
  without a passing check.
- `CLUSTER_RESNAME` sets the cluster residue name if it is not `FES` or `SF4`. `GEOMETRY=ox`
  sets the `R` part of the run name (default `pdb`).
- `TIME`, `CONFIG` (default `configs/batch_dzvp.toml`), `ENGINE=cpu`, and `EXCLUSIVE=0`.

Run `check` first because an under-valent atom (for example, at a bond that was cut when the
structure was taken from a larger protein) is a radical in the SCF. The check reports it within
seconds. The build fails at the gate only hours later.

A file with about 1500 AOs takes hours for each state, mostly for the stability analysis of each
BS pattern (Fe2S2 has 2, Fe4S4 has 6). The plugin keeps converged SCFs in
`workspace/work/bs_uno_avas/`. If a job reaches its wall time, run `build` again: it reuses
them.

## Step 2B: Run the DFT route (fchk file)

On the **login node**:

```bash
source ~/.qcsc_env; cd $REPO/plugins/bs_uno_avas
bash roquo/run_fchk.sh check     # conversion check
bash roquo/run_fchk.sh report    # after the jobs finish
bash roquo/run_fchk.sh build     # active space + manifest
bash roquo/run_fchk.sh report    # after the jobs finish
```

- The scripts take every `workspace/data/*.fchk`.
- For each file, `build` asks for the redox state, `ox` or `red`. The suggestion comes from the
  multiplicity (1 for ox, 2 for red). Press Enter to accept it.

In `report`, check these lines:

- `check: PASS`: orthonormality at most 1e-6, `<S^2>` equal to the fchk value, and `dE` within
  ±2e-3 Eh. A FAIL means that the AO mapping does not fit this file. Send the log instead of
  raising the tolerances.
- `cluster atoms`: the fchk atom numbers of Fe and μS are the ones that you expect.
- `spins:`: Fe moments of about ±3 to ±3.5 with the expected tag, and small μS moments. Read
  every `WARNING`.
- `DONE`, with the manifest path.

Options, only if you need them: name some files after the command, `STATE=red`, `TIME`,
`CONFIG` (default `configs/fchk_dgdzvp.toml`), and `GEOMETRY` (the `R` part of the run name).

## Step 3: Monitor a running job

On the **login node**:

```bash
source ~/.qcsc_env; squeue -u $USER
ls -t $REPO/plugins/bs_uno_avas/workspace/logs/*.out | head -3
```

Follow one of the listed logs with `tail -f`. The log starts with the node and its GPU. A
`gpu4pyscf not importable` line means that the job runs on the CPU. After that, the log shows
one line for each finished SCF stage. Finished SCFs also appear as `.chk` files in
`workspace/work/bs_uno_avas/`. The gpu4pyscf notes `using cupy as the tensor contraction engine`
and `Peer access has been activated` are harmless.

## Step 4: Check the output

`report` ends with the absolute manifest paths. Each run writes one folder,
`workspace/output/<run>/`. The run name is `<input stem>_E<state>_R<geometry>_S<tag>` (the fchk
route adds `_BS_OPT`), so states, geometries, and spin patterns never share a folder.

| File | Content |
|---|---|
| `<run>_avas.fcidump` | active-space Hamiltonian (16 orbitals for Fe2S2, 32 for Fe4S4) |
| `<run>_active_orbitals_labels.npy` | one label for each orbital, for example `0_Fe_3dxz` or `2_S_3py` |
| `<run>_active_space_rdm.npy` | `(2, norb, norb)` alpha and beta 1-RDMs of the selected BS solution |
| `<run>_active_orbitals.npy` | the active MOs in the AO basis |
| `<run>_manifest.json` | schema v1: structure, electronic state, active space, `bs_reference` (`spin_pattern`, `spin_tag`, `e_active_uhf`), and the sha256 of every file |

- `spin_tag` has one digit for each Fe in model order: `0` for alpha (up) and `1` for beta
  (down), for example `10`.
- Keep each `output/<run>/` folder complete. The manifest checks the sha256 of its files, and
  a consumer stops on a changed file.
- Use the same manifest for every SQD run of a system. The quantum samples are valid only in its
  orbital basis.

To check a written manifest again from its files, run
`bs-uno-avas verify --manifest M` (see [Run on another machine](#run-on-another-machine)).

## Run on another machine

The same pipeline runs on any machine with PySCF (gpu4pyscf is optional), without Slurm:

```bash
pip install -e plugins/bs_uno_avas          # add [gpu] for gpu4pyscf
cd plugins/bs_uno_avas
bs-uno-avas build --config configs/batch_dzvp.toml \
    --pdb workspace/data/site1.pdb --dry-run
bs-uno-avas build --config configs/batch_dzvp.toml \
    --pdb workspace/data/site1.pdb --state red
```

- For the fchk route, run `bs-uno-avas check-fchk --fchk F`, then
  `bs-uno-avas build-fchk --fchk F --state red`.
- To check a written manifest, run `bs-uno-avas verify --manifest M`.
- For a 30-second CPU demo, use `configs/bare_fe2s2.toml` with
  `examples/5XTD_FES_O301_SCH3.pdb` (see the plugin README).

## Troubleshoot

| Symptom | Fix |
|---|---|
| The gpu4pyscf build fails, or `gpu4pyscf not importable` | Use the CPU engine of the setup (`ENGINE=cpu`). Use it also for CUDA out-of-memory errors. |
| `STOP: …` from a script | The message names what is missing (`ACCT`, inputs, or a passing check). Fix it and run the command again. |
| `no Fe-S cluster residue found` | The cluster residue is not named `FES` or `SF4`. Set `CLUSTER_RESNAME` (the error lists the residues that hold Fe). |
| `under-valent atoms (cut points / missing H)` | Set `protonate = "cap"` in a copy of the config, or add the missing H yourself. |
| `fractional UNOs with target weight < 0.5` | An unpaired electron is outside the cluster (ligand radical or O coordination). Stop and report it. Do not loosen the threshold. |
| `active-space UHF … converged to tag … but … full-system … tag` | Send the log. The options are in REFERENCE §1 (`avas.tag_mismatch`). |
| A job reaches its wall time | Run `build` again. It reuses the finished SCFs. |

For any other error, send the `.out` and `.err` files that `report` names. The config options,
the rules behind these checks, and the manifest fields are in
[`plugins/bs_uno_avas/docs/REFERENCE.md`](../../plugins/bs_uno_avas/docs/REFERENCE.md).
