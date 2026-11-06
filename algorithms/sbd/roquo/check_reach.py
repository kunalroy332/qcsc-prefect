"""From a compute node: Prefect Cloud API and IBM Quantum reachable, credentials valid.

Exit 0 only if both pass. Run by run_bs.sh on one node in `check`, and on every allocated node
before any task starts. IBM is checked through the Prefect block `ibm-runner` (made in the
Prefect UI, setup how-to Step 6); if that block does not exist yet, through IBM_API_KEY /
IBM_CRN / IBM_BACKEND from secrets.env (the pre-flight then creates the block from them).
No secret is printed.
"""

from __future__ import annotations

import os
import socket
import sys
import time

IBM_BLOCK = "ibm-runner"


def _tcp(host: str, port: int = 443, timeout: float = 10.0) -> str | None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError as exc:
        return f"TCP {host}:{port} failed: {exc}"


def check_prefect() -> str | None:
    from urllib.parse import urlparse

    from prefect.settings import get_current_settings

    url = str(get_current_settings().api.url or "")
    if not url:
        return "PREFECT_API_URL is not set: run `prefect cloud login` (setup how-to Step 5)"
    if "api.prefect.cloud" not in url:
        return f"PREFECT_API_URL is not Prefect Cloud: {url}"
    err = _tcp(urlparse(url).hostname or "api.prefect.cloud")
    if err:
        return err
    try:
        from prefect.client.orchestration import get_client

        with get_client(sync_client=True) as client:
            client.read_flows(limit=1)  # authenticated workspace call
    except Exception as exc:
        return f"Prefect Cloud API call failed: {type(exc).__name__}: {exc}"
    return None


def ibm_block_exists() -> bool:
    """True if the Prefect block quantum-runtime/ibm-runner exists (no IBM call)."""
    from prefect.client.orchestration import get_client
    from prefect.exceptions import ObjectNotFound

    with get_client(sync_client=True) as client:
        try:
            client.read_block_document_by_name(IBM_BLOCK, block_type_slug="quantum-runtime")
        except ObjectNotFound:
            return False
    return True


def _check_ibm_block() -> str | None:
    """Load the block (that asks IBM which backends are online) and its backend's status."""
    from prefect.utilities.asyncutils import run_coro_as_sync
    from prefect_qiskit import QuantumRuntime

    try:
        runtime = QuantumRuntime.load(IBM_BLOCK)
    except Exception as exc:  # also the block's own check: backend not online / not yours
        return f"block {IBM_BLOCK} does not load: {type(exc).__name__}: {exc}"
    client = runtime.credentials.get_client()
    if not run_coro_as_sync(client.check_resource_available(runtime.resource_name)):
        return f"IBM backend {runtime.resource_name} is not available"
    print(f"[reach] ibm backend {runtime.resource_name} (block {IBM_BLOCK})", flush=True)
    return None


def _check_ibm_env() -> str | None:
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["IBM_API_KEY"],
        instance=os.environ["IBM_CRN"],
    )
    backend = service.backend(os.environ["IBM_BACKEND"])
    status = backend.status()
    if not status.operational:
        return f"IBM backend {backend.name} is not operational ({status.status_msg})"
    print(f"[reach] ibm backend {backend.name} (secrets.env; no block {IBM_BLOCK} yet)", flush=True)
    return None


def check_ibm() -> str | None:
    err = _tcp("quantum.cloud.ibm.com")
    if err:
        return err
    try:
        if ibm_block_exists():
            return _check_ibm_block()
        if all(os.environ.get(k) for k in ("IBM_API_KEY", "IBM_CRN", "IBM_BACKEND")):
            return _check_ibm_env()
    except Exception as exc:
        return f"IBM Quantum check failed: {type(exc).__name__}: {exc}"
    return (
        f"no Prefect block {IBM_BLOCK}: create it in the Prefect UI (setup how-to Step 6), "
        "or put IBM_API_KEY / IBM_CRN / IBM_BACKEND in secrets/secrets.env"
    )


def main() -> int:
    node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    failed = False
    checks = [("prefect-cloud", check_prefect), ("ibm-quantum", check_ibm)]
    if os.environ.get("SAMPLES_DEVICE") == "0":  # run_bs.sh: no task samples the device
        checks = checks[:1]
        print(f"[reach] {node} ibm-quantum: skipped (no task samples the device)", flush=True)
    for name, fn in checks:
        t0 = time.perf_counter()
        err = fn()
        dt = time.perf_counter() - t0
        print(
            f"[reach] {node} {name}: {'OK' if err is None else 'FAIL'} ({dt:.1f}s)"
            + ("" if err is None else f" -- {err}"),
            flush=True,
        )
        failed |= err is not None
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
