"""4Fe-4S 72-qubit sampling on Fugaku: 5M shots, save the merged pool (no heavy diagonalization).

Model of the proven Fe2S2 FE_sample run, scaled to 72q (NORB=36, NELEC=54, MS2=0). The sampling
pass only needs to persist the raw bitstring pool BEFORE any diagonalization, so a modest node
count suffices here; the heavy sqd_dim=3e8 diagonalization is the separate recovery job.

    FE2S2_QSRC=random python run_fe4s4_sample.py   # cheap dry-run, no IBM call
    python run_fe4s4_sample.py                      # real device (ibm_kobe), 5M shots

Idempotent: if a persisted pool already exists under runs/fe4s4_rhf/samples/, sampling is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import fe2s2_common as C  # FE_MOL=fe4s4 selects the 4Fe-4S FCIDUMP + runs/fe4s4_* layout

TAG = "fe4s4_sample"
METHOD = os.environ.get("FE4S4_METHOD", "rhf")  # RHF case (compare like Fe2S2)


def main() -> None:
    assert C.MOLECULE == "fe4s4", f"set FE_MOL=fe4s4 (got {C.MOLECULE})"
    dirs = C.run_subdirs(METHOD)
    qsrc = os.environ.get("FE2S2_QSRC", "real-device")
    p = C.sbd_paths()

    # Idempotency: reuse an already-persisted pool (never re-charge the device).
    existing = C.find_saved_pools(METHOD)
    if existing and os.environ.get("FE4S4_FORCE") != "1":
        C.samples_manifest_path(METHOD).write_text(json.dumps(existing, indent=2))
        print(f"[skip] {METHOD}: {len(existing)} pool(s) already present; reuse (FE4S4_FORCE=1 to redo).")
        for x in existing:
            print(f"  {x}")
        return

    os.environ["PREFECT_HOME"] = str(C.prefect_home(METHOD))
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(C.prefect_home(METHOD) / "storage")
    C.prefect_home(METHOD).mkdir(parents=True, exist_ok=True)

    # 1. Solver block. 72q sampling pass: the persist step saves the pool before diag, so the
    #    solver here diagonalizes only a small seed subspace -> a single node is enough. The heavy
    #    3e8 solve is the recovery job. DD(XY4)+measure-twirling, carryover on. Same conditions as
    #    the Fe2S2 FE_sample run, with --method rhf.
    subprocess.run(
        [
            p["python"], os.path.join(p["sbd"], "create_blocks.py"),
            "--hpc-target", "fugaku",
            "--method", METHOD,
            "--solver-mode", "fugaku",
            "--project", "ra010014", "--group", "ra010014",
            "--queue", "small",
            "--fugaku-gfscache", "/vol0004:/vol0002",
            "--num-nodes", "1", "--mpiprocs", "1", "--ompthreads", "48",
            "--carryover-ratio", "0.5", "--carryover-type", "1",
            "--solver-timeout-seconds", "43200",
            "--work-dir", str(C.run_dir(METHOD) / "work_sample"),
            "--shots", "5000000", "--n-shot-batches", "5",
            "--iteration", "5", "--block", "20",
            "--dynamical-decoupling", "--dd-sequence", "XY4", "--measure-twirling",
            "--sbd-executable", p["diag_rhf"],
            "--sbd-executable-uhf", p["diag_uhf"],
        ],
        cwd=p["sbd"], check=True,
    )

    # 2. IBM runner (real device only).
    if qsrc == "real-device":
        C.save_ibm_runner_block()

    # 3. One flow pass at a small sqd_dim just to sample + persist the 5M-shot pool.
    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    t0 = time.perf_counter()
    params = FlowParameters(
        fcidump=C.FCIDUMP,
        sqd_dim=3_000_000,       # sampling pass; the raw pool is what we keep
        n_recovery_steps=1,
        n_batches=1,
        quantum_source=qsrc,
        solver_block_ref="sbd_solver_job/davidson-solver",
        circ_params=CircuitParameters(n_lucj_layers=1),
        de_params=DEParameters(num_walkers=1, iterations=1, randomization_factor=0.2, fxc=0.5),
    )
    e = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    pools = C.find_saved_pools(METHOD)
    C.samples_manifest_path(METHOD).write_text(json.dumps(pools, indent=2))
    print(
        f"CAPTURE_FE4S4_SAMPLE {METHOD}: E_sampling={e:.6f} qsrc={qsrc} pools={len(pools)} ({dt:.0f}s)",
        flush=True,
    )
    for x in pools:
        print(f"  {x}")
    if not pools:
        print("[warn] no pool persisted -- check quantum_source and the [persist] log lines.")


if __name__ == "__main__":
    main()
