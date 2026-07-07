"""4Fe-4S 72q GPU recovery on ROQUO (GB200) from a saved pool. RHF by default.

Runs the SBD diagonalizer on GPU (diag-gpu / diag-gpu_uhf, built with nvc++ -cuda) inside a GB200
Slurm allocation via the "local" hpc-target (direct subprocess, no per-solve sbatch). Multi-GPU per
solve: the allocation grabs a full node (4 GB200) and the GPU binary uses them via its internal
NCCL/cuBLAS path. Reuses a persisted sample pool (quantum_source="saved") -- no IBM call.

    FE4S4_POOL=<npz> python run_fe4s4_gpu_roquo.py

Env (set by run_fe4s4_gpu_roquo.sh):
    FE4S4_METHOD  : rhf (default) | uhf  -> selects diag-gpu vs diag-gpu_uhf
    FE4S4_POOL    : persisted raw_samples npz path(s)
    FE4S4_SQD_DIM : subspace dim (default 3e8)
    FE4S4_RECSTEPS/FE4S4_NBATCH : recovery passes / K-batches (default 5 / 5)
    ROQUO_OMPTHREADS : CPU threads for the host side (default 140)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import fe2s2_common as C

METHOD = os.environ.get("FE4S4_METHOD", "rhf")
SQD_DIM = int(float(os.environ.get("FE4S4_SQD_DIM", "300000000")))
RECSTEPS = int(os.environ.get("FE4S4_RECSTEPS", "5"))
# n_batches=1: the ConcurrentTaskRunner launches all batches at once, and N concurrent diag-gpu
# procs on 1 GPU multiply GPU memory -> OOM (job 2305). One batch per recovery step fits and still
# gives the full recovery-effect trajectory. Bump only with more GPUs / a memory-serialized runner.
NBATCH = int(os.environ.get("FE4S4_NBATCH", "1"))
OMP = int(os.environ.get("ROQUO_OMPTHREADS", "140"))

# Node-local Prefect DB (Lustre home is slow/locks); keep the persisted storage on Lustre.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_TELEMETRY_ENABLED", "false")
os.environ.setdefault("SBD_TASK_RUNNER", "concurrent")
os.environ.setdefault("PREFECT_SERVER_DATABASE_TIMEOUT", "60")
os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "120")
os.environ.setdefault("OMP_NUM_THREADS", str(OMP))
# diag-gpu is an MPI binary that HANGS run bare -> launch under mpirun. It must be ONE rank on ONE
# GPU: the solver replicates its big tensors per rank, so multi-rank (-n 4) OOMs even across 4 GPUs.
# A single GB200 (189GB) fits the full 3e8 (17320-det) 72q UHF solve in ~13s (verified). So -n 1.
# Pass launcher/mpi via ENV (SBD_LAUNCHER/SBD_MPI_OPTIONS, comma-split by env_csv) since argparse
# --mpi-options rejects '-'-prefixed tokens.
os.environ["SBD_LAUNCHER"] = "mpirun"
os.environ["SBD_MPI_OPTIONS"] = "-n,1"


def _pool_paths() -> list[str]:
    raw = os.environ.get("FE4S4_POOL", "").strip()
    if not raw:
        raise SystemExit("Set FE4S4_POOL to the persisted raw_samples npz path(s).")
    return [p if p.startswith("file://") else f"file://{p}"
            for p in raw.replace(",", " ").split()]


def main() -> None:
    if C.MOLECULE != "fe4s4":
        raise SystemExit(f"set FE_MOL=fe4s4 (got {C.MOLECULE})")
    pools = _pool_paths()
    p = C.sbd_paths()
    diag_gpu = os.path.join(p["diag"], "diag-gpu")
    diag_gpu_uhf = os.path.join(p["diag"], "diag-gpu_uhf")

    base = C.run_dir(METHOD)          # runs/fe4s4_<method>/ (Lustre)
    (base / "recover").mkdir(parents=True, exist_ok=True)
    # Keep work_dir + PREFECT_HOME on Lustre (base): verified working (job 2309). The diag-gpu
    # solve FAILS when work_dir is on $SLURM_SCRATCH (local NVMe) — likely the per-node scratch is
    # not where the solver/mpirun expects. Lustre is slower but correct, which wins for the deadline.
    prefect_home = base / "prefect_home_gpu"
    work_dir = base / "work_gpu_recover"
    prefect_home.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PREFECT_HOME"] = str(prefect_home)
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(base / "prefect_home" / "storage")
    (base / "prefect_home" / "storage").mkdir(parents=True, exist_ok=True)

    # Solver block: local target + GPU mode. --solver-mode gpu makes the executable key sbd_diag_gpu
    # (or _uhf), which we map to the nvc++-built diag-gpu binaries. Local target runs it directly in
    # this GB200 allocation (no nested sbatch).
    subprocess.run(
        [
            p["python"], os.path.join(p["sbd"], "create_blocks.py"),
            "--hpc-target", "local", "--method", METHOD, "--solver-mode", "gpu",
            "--num-nodes", "1", "--mpiprocs", "1", "--ompthreads", str(OMP),
            # launcher + mpi options come from SBD_LAUNCHER / SBD_MPI_OPTIONS env (set above).
            "--walltime", "06:00:00",
            "--carryover-ratio", "0.5", "--carryover-type", "1",
            "--solver-timeout-seconds", "43200",
            "--work-dir", str(work_dir),
            "--saved-samples", ",".join(pools),
            "--iteration", "5", "--block", "20",
            "--sbd-executable", diag_gpu, "--sbd-executable-uhf", diag_gpu_uhf,
        ],
        cwd=p["sbd"], check=True,
    )

    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    t0 = time.perf_counter()
    params = FlowParameters(
        fcidump=C.FCIDUMP,
        sqd_dim=SQD_DIM,
        n_recovery_steps=RECSTEPS,
        n_batches=NBATCH,
        quantum_source="saved",
        solver_block_ref="sbd_solver_job/davidson-solver-gpu",
        circ_params=CircuitParameters(n_lucj_layers=1),
        de_params=DEParameters(num_walkers=1, iterations=1, randomization_factor=0.2, fxc=0.5),
    )
    best = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    trace = _read_recovery_trace()
    out = {
        "method": METHOD, "molecule": "fe4s4", "cluster": "roquo-gpu",
        "sqd_dim": SQD_DIM, "n_batches": NBATCH, "max_recovery": RECSTEPS,
        "best_energy": best, "elapsed_s": dt, "pools": pools, "recovery_trace": trace,
    }
    fn = f"recover_{METHOD}_roquogpu_rec{RECSTEPS}_k{NBATCH}_dim{SQD_DIM}.json"
    (base / "recover" / fn).write_text(json.dumps(out, indent=2))
    print(
        f"CAPTURE_FE4S4_GPU {METHOD}: best_E={best:.6f} steps={len(trace)} dim={SQD_DIM} "
        f"({dt:.0f}s) -> {base / 'recover' / fn}",
        flush=True,
    )


def _read_recovery_trace() -> list[dict]:
    try:
        from prefect.client.orchestration import get_client
        from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterKey

        with get_client(sync_client=True) as client:
            arts = client.read_artifacts(
                artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=["sqd-telemetry"])),
            )
            if not arts:
                return []
            data = json.loads(arts[0].data)
            for rec in reversed(data):
                if isinstance(rec, dict) and rec.get("recovery_trace"):
                    return rec["recovery_trace"]
    except Exception as exc:
        print(f"[warn] could not read recovery_trace ({exc}).")
    return []


if __name__ == "__main__":
    main()
