"""Shared helpers of the ROQUO BS-UHF SQD launcher (run_bs.sh -> run_task.py, preflight.py).

Paths of the SBD checkout, Prefect isolation of a run's sample storage, the manifest reference
check, create_blocks.py, the IBM block, one in-process flow run, and its recovery trace.
Everything about the system comes from the active-space manifest (schema v1,
``qcsc_workflow_utility.active_space``); there is no molecule registry.

Import-only (no side effects at import).
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - list-form calls only, no shell
from pathlib import Path


# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------
def sbd_root() -> Path:
    """algorithms/sbd (this file's grandparent), or $MY_PROJECT/algorithms/sbd if set."""
    proj = os.environ.get("MY_PROJECT")
    return Path(proj) / "algorithms" / "sbd" if proj else Path(__file__).resolve().parents[1]


def sbd_paths() -> dict[str, str]:
    sbd = sbd_root()
    diag = sbd / "native"
    return {
        "sbd": str(sbd),
        "diag": str(diag),
        "python": os.environ.get("SBD_PYTHON", str(sbd / ".venv" / "bin" / "python")),
    }


def isolate_prefect(home: Path, *, storage_only: bool = False) -> None:
    """Point Prefect's local result storage (where the pool is persisted) at ``home/storage`` on
    shared storage (a node-local path would be wiped at job end).

    ``storage_only=False`` also sets PREFECT_HOME (own ephemeral-server database per run).
    ``storage_only=True`` keeps PREFECT_HOME, so a Prefect Cloud profile stays active.
    Prefect is usually imported already (manifest loading pulls it in), so its cached settings
    are refreshed; otherwise these variables would be ignored and the pool would land in
    ``~/.prefect/storage``.
    """
    (home / "storage").mkdir(parents=True, exist_ok=True)
    if not storage_only:
        os.environ["PREFECT_HOME"] = str(home)
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(home / "storage")
    # sbd.data_io writes the pools; it follows SBD_LOCAL_STORAGE_DIR, not Prefect's result storage.
    os.environ["SBD_LOCAL_STORAGE_DIR"] = str(home / "storage")
    try:
        import prefect.context

        prefect.context.refresh_global_settings_context()
    except Exception as exc:  # pragma: no cover - old Prefect without the refresh hook
        print(f"[warn] could not refresh Prefect settings ({exc}); start a fresh process instead.")


def local_storage_dir() -> Path:
    """Directory where sbd.data_io will persist the sample pool (for checks)."""
    from sbd.data_io import _local_storage_dir

    return _local_storage_dir()


def prefect_defaults(omp_threads: int) -> None:
    """Same Prefect / OMP defaults as the sweep ROQUO launchers (setdefault: env wins)."""
    os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    os.environ.setdefault("PREFECT_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("SBD_TASK_RUNNER", "concurrent")
    os.environ.setdefault("PREFECT_SERVER_DATABASE_TIMEOUT", "120")
    os.environ.setdefault("PREFECT_SERVER_DATABASE_CONNECTION_TIMEOUT", "60")
    os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "120")
    # A long child-solve poll can make the ephemeral server answer >60 s -> httpx.ReadTimeout.
    os.environ.setdefault("PREFECT_API_REQUEST_TIMEOUT", "600")
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))


# ---------------------------------------------------------------------------------------------
# Manifest / reference
# ---------------------------------------------------------------------------------------------
def load_spec(manifest: str, fcidump: str | None = None):
    """Validated manifest (raises ValueError with file + field on any schema violation)."""
    from qcsc_workflow_utility.active_space import load_active_space_manifest

    return load_active_space_manifest(manifest, fcidump_override=fcidump)


def dry_run_report(manifest: str, fcidump: str | None, method: str) -> dict:
    """Load the manifest and rebuild the reference as the flow would (UHF only; no UCCSD, no
    sampling, no solver). Prints and returns a summary."""
    spec = load_spec(manifest, fcidump)
    out = {
        "run_name": spec.run_name,
        "cluster_type": spec.cluster_type,
        "norb": spec.norb,
        "nelec": [spec.nalpha, spec.nbeta],
        "qubits": 2 * spec.norb,
        "manifest_tag": spec.spin_tag,
        "manifest_e_active_uhf": spec.e_active_uhf,
        "fcidump": str(spec.fcidump),
        "fcidump_sha256": spec.fcidump_sha256,
        "method": method,
    }
    if method == "uhf":
        from qcsc_workflow_utility.chem import manifest_reference_summary

        _, info = manifest_reference_summary(manifest, fcidump)
        out.update(
            reference_source=info["source"],
            reference_tag=info["tag"],
            e_uhf=info["energy"],
            de_vs_manifest=(
                None if spec.e_active_uhf is None else info["energy"] - spec.e_active_uhf
            ),
            s2=info["s2"],
            atom_spins=info["atom_spins"],
        )
    print(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------------------------
# create_blocks.py
# ---------------------------------------------------------------------------------------------
def run_create_blocks(args: list[str]) -> None:
    """Run create_blocks.py with the venv's bin/ first on PATH (it shells out to `prefect`)."""
    p = sbd_paths()
    cmd = [p["python"], os.path.join(p["sbd"], "create_blocks.py"), *args]
    env = {**os.environ, "PATH": f"{os.path.dirname(p['python'])}{os.pathsep}{os.environ['PATH']}"}
    subprocess.run(cmd, cwd=p["sbd"], check=True, env=env)  # nosec B603 - fixed argv list


def save_ibm_runner_block() -> None:
    """Persist the IBM Quantum Runtime block from environment credentials (never hardcoded)."""
    for key in ("IBM_API_KEY", "IBM_CRN", "IBM_BACKEND"):
        if not os.environ.get(key):
            raise RuntimeError(f"{key} is not set (secrets/secrets.env).")
    from prefect_qiskit import QuantumRuntime
    from prefect_qiskit.vendors.ibm_quantum.credentials import IBMQuantumCredentials

    QuantumRuntime(
        resource_name=os.environ["IBM_BACKEND"],
        credentials=IBMQuantumCredentials(
            api_key=os.environ["IBM_API_KEY"], crn=os.environ["IBM_CRN"]
        ),
    ).save("ibm-runner", overwrite=True)


def run_flow(params, *, flow_run_name: str | None = None):
    """Run ``riken_sqd_de`` in this process; return ``(best_energy, flow_run_id)``.

    Raises the flow's exception on failure (``State.result()`` re-raises it).
    """
    from sbd.main import riken_sqd_de

    flow = riken_sqd_de.with_options(flow_run_name=flow_run_name) if flow_run_name else riken_sqd_de
    state = flow(params, return_state=True)
    return state.result(), state.state_details.flow_run_id


def read_recovery_trace(flow_run_id=None) -> list[dict]:
    """Longest recovery_trace in the sqd-telemetry artifacts of one flow run.

    With ``flow_run_id`` (always pass it with a shared server such as Prefect Cloud) only that
    run's artifacts are read; without it, every artifact on the server is scanned.
    """
    try:
        from prefect.client.orchestration import get_client
        from prefect.client.schemas.filters import (
            ArtifactFilter,
            ArtifactFilterKey,
            FlowRunFilter,
            FlowRunFilterId,
        )

        run_filter = None
        if flow_run_id is not None:
            run_filter = FlowRunFilter(id=FlowRunFilterId(any_=[flow_run_id]))
        with get_client(sync_client=True) as client:
            arts = client.read_artifacts(
                artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=["sqd-telemetry"])),
                flow_run_filter=run_filter,
            )
        best: list[dict] = []
        for art in arts:
            try:
                data = json.loads(art.data)
            except Exception:  # nosec B112 - skip malformed artifacts
                continue
            for rec in data if isinstance(data, list) else []:
                tr = rec.get("recovery_trace") if isinstance(rec, dict) else None
                if tr and len(tr) > len(best):
                    best = tr
        return best
    except Exception as exc:
        print(f"[warn] could not read recovery_trace ({exc}).")
        return []
