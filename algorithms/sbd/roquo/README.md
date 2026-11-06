# ROQUO launcher: BS-UHF SQD from active-space manifests

One SQD flow (`riken_sqd_de`) per manifest from `plugins/bs_uno_avas`, each task on its own
GB200 node (4 GPUs) of one Slurm job, recorded in Prefect Cloud.

```bash
source ~/.qcsc_env; cd $REPO/algorithms/sbd/roquo
bash run_bs.sh check     # short check job; then: bash run_bs.sh report
bash run_bs.sh job       # asks which tasks and an action per task
bash run_bs.sh report    # state of every task and size, energies
```

- Setup: [docs/howto/howto_setup_qcsc_prefect_roquo.md](../../../docs/howto/howto_setup_qcsc_prefect_roquo.md)
- Walkthrough: [docs/tutorials/run_bs_uhf_sqd_roquo.md](../../../docs/tutorials/run_bs_uhf_sqd_roquo.md)
- Resume, sweeps, solver layout, every knob:
  [docs/tutorials/run_bs_uhf_sqd_advanced_roquo.md](../../../docs/tutorials/run_bs_uhf_sqd_advanced_roquo.md)

| File | Role |
|---|---|
| `run_bs.sh` | `check` / `job` / `sweep` / `report` (header = usage) |
| `run_bs.toml` | settings shared by every task |
| `run_task.py` | one task on one node: blocks, parameters, samples, checkpoints |
| `task_state.py` | task / size states on the login node |
| `preflight.py`, `check_reach.py` | checks before any task starts |
| `bs_common.py` | shared helpers |

Run data: `runs/<run_name>/` (`samples/`, `d<size>/`), logs: `logs/`. Both are gitignored.
