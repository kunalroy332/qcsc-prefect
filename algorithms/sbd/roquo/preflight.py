"""Pre-flight on ONE node, before any task starts (run by run_bs.sh: in `check`, and in the job).

A setup error then costs one short failure instead of one per task, and no IBM shots:

  1. the solver binary of [run] / [solver] exists and its shared libraries resolve (ldd);
  2. the nested solver launch works: the exact srun of run_task.py's solver_launcher() runs a
     probe on every rank (rank, CUDA_VISIBLE_DEVICES, GPUs seen, OMP_NUM_THREADS, cgroup memory);
  3. the Prefect block types are registered once (14 concurrent create_blocks.py runs would
     otherwise race on a new schema);
  4. the IBM block `ibm-runner` exists (made in the Prefect UI, setup how-to Step 6). It is
     never overwritten; only if it is missing, it is created from secrets.env.

Exit 0 when all pass; otherwise exit 2 with the failing check. No secret is printed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess  # nosec B404 - list-form calls only, no shell
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_task as T  # noqa: E402

C = T.C

PROBE = r"""
r=${SLURM_PROCID:-?}
g=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
c=$(nproc 2>/dev/null || echo ?)
m=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo ?)
echo "[probe] rank $r/${SLURM_NTASKS:-?} on $(hostname -s):" \
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} gpus_seen=$g cpus=$c" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} mem.max=$m"
"""


def _gpus_seen(line: str) -> int:
    for tok in line.split():
        if tok.startswith("gpus_seen="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return -1
    return -1


def say(msg: str) -> None:
    print(f"[preflight] {msg}", flush=True)


def check_binary(lay: dict) -> str | None:
    out = subprocess.run(  # nosec B603 B607 - fixed argv
        ["ldd", str(lay["binary"])], capture_output=True, text=True, check=False
    )
    missing = [ln.strip() for ln in out.stdout.splitlines() if "not found" in ln]
    if out.returncode != 0 or missing:
        return f"ldd {lay['binary']}: " + ("; ".join(missing) or out.stderr.strip())
    say(f"binary OK: {lay['binary']} (all shared libraries found)")
    return None


def check_launch(lay: dict, timeout: int) -> str | None:
    """Run the probe with exactly the launcher + options the solver will get.

    Like the solver, the probe writes solver-<rank>.out / .err in its cwd (a temporary dir under
    logs/), so the per-rank output files are checked too.
    """
    T.solver_launcher(lay)
    opts = [o for o in os.environ["SBD_MPI_OPTIONS"].split(",") if o]
    env = {**os.environ, "OMP_NUM_THREADS": str(lay["threads"])}
    (HERE / "logs").mkdir(exist_ok=True)  # gitignored; shared storage like the solver's work dir
    with tempfile.TemporaryDirectory(prefix="preflight-", dir=HERE / "logs") as cwd:
        probe = Path(cwd) / "probe.sh"
        probe.write_text(PROBE)
        cmd = ["srun", "--input=none", *opts, "/bin/bash", str(probe)]
        say("solver launch: " + shlex.join(cmd))
        try:
            out = subprocess.run(  # nosec B603 - fixed argv
                cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env, check=False
            )
        except subprocess.TimeoutExpired:
            return f"solver launch probe hung for {timeout} s (srun step never started?)"
        files = sorted(Path(cwd).glob("solver-*.out"))
        text = "".join(f.read_text() for f in files)
        err = "".join(f.read_text() for f in sorted(Path(cwd).glob("solver-*.err")))
    lines = [ln for ln in text.splitlines() if ln.startswith("[probe]")]
    for ln in sorted(lines):
        print(ln, flush=True)
    if out.returncode != 0:
        msg = (out.stderr + err).strip()[-400:]
        return f"solver launch probe failed (exit {out.returncode}): {msg}"
    if len(files) != lay["ranks"]:
        return (
            f"solver launch probe: {len(files)} solver-<rank>.out file(s), expected "
            f"{lay['ranks']} (srun --output=solver-%t.out)"
        )
    if len(lines) != lay["ranks"]:
        return f"solver launch probe: {len(lines)} rank(s) answered, expected {lay['ranks']}"
    if any(_gpus_seen(ln) != lay["gpus"] for ln in lines):
        return (
            f"solver launch probe: not every rank sees {lay['gpus']} GPU(s) "
            "(main.cc picks its GPU as rank % #GPUs and needs all of them)"
        )
    say(f"solver launch OK: {lay['ranks']} rank(s), each sees {lay['gpus']} GPU(s)")
    return None


def register_blocks() -> str | None:
    sbd = str(Path(C.sbd_paths()["sbd"]))
    sys.path.insert(0, sbd)
    try:
        import create_blocks as CB

        CB._register_block_types(CB._import_sbd_solver_block())
    except Exception as exc:
        return f"block type registration failed: {type(exc).__name__}: {exc}"
    say("Prefect block types registered")
    return None


def ensure_ibm() -> str | None:
    """The block ibm-runner: use the one from the Prefect UI; create it only if it is missing."""
    # SAMPLES_DEVICE=0 (run_bs.sh): no task samples the device (pools, or quantum_source random)
    if os.environ.get("SAMPLES_DEVICE") == "0":
        say("ibm-runner not needed (no task samples the device)")
        return None
    import check_reach as R

    try:
        if R.ibm_block_exists():
            say("ibm-runner block found (kept as it is)")
            return None
        if not all(os.environ.get(k) for k in ("IBM_API_KEY", "IBM_CRN", "IBM_BACKEND")):
            return (
                "no Prefect block ibm-runner: create it in the Prefect UI (setup how-to "
                "Step 6), or put IBM_API_KEY / IBM_CRN / IBM_BACKEND in secrets/secrets.env"
            )
        C.save_ibm_runner_block()
    except Exception as exc:
        return f"ibm-runner block: {type(exc).__name__}: {exc}"
    say(f"ibm-runner block created from secrets.env (backend {os.environ.get('IBM_BACKEND')})")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeout", type=int, default=300, help="seconds for the launch probe")
    args = ap.parse_args()
    cfg = T.load_config(Path(args.config))
    try:
        lay = T.solver_layout(cfg)
    except SystemExit as exc:
        say(f"FAILED: {exc}")
        return 2
    say(
        f"node {os.environ.get('SLURMD_NODENAME')}: {lay['ranks']} rank(s) = task "
        f"{lay['task']} x adet {lay['adet']} x bdet {lay['bdet']}, {lay['threads']} threads/rank"
    )
    for check in (
        lambda: check_binary(lay),
        lambda: check_launch(lay, args.timeout),
        register_blocks,
        ensure_ibm,
    ):
        err = check()
        if err:
            say(f"FAILED: {err}")
            return 2
    say("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
