"""One BS-UHF SQD task = one manifest, run inside one Slurm job step on one node.

Started by run_bs.sh (one `srun -N1 -n1 --exclusive -w <node>` step per task). It creates this
task's Prefect blocks (own names, own work dir), builds the FlowParameters from run_bs.toml + the
manifest, then runs one flow per subspace size into runs/<run_name>/d<size>/:

  --kind job   : one size, [flow].sqd_dim; mode "direct" runs the flow now, mode "cloud" serves
                 it as deployment `riken_sqd_de/<tag>` and waits for ONE run from the UI;
  --kind sweep : every size of [sweep].sqd_dims, one after another on this node.

All sizes of a task share its samples (runs/<run_name>/samples/): the first run that samples on
the device writes the pool, every later run recovers from it (quantum_source = "saved").
Each size keeps its own recovery checkpoint (d<size>/ckpt/), so a cut-off size continues where
it stopped, whichever command runs it next (--action continue). --action new starts every size
at step 1 again, on the same samples; --resample takes new samples first.

Also: `--check` (manifest + reference rebuild, no blocks / Prefect / IBM) and
`--print-params` (the FlowParameters JSON this task would use).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import tomllib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bs_common as C  # noqa: E402

REPO = HERE.parents[2]


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for section in ("slurm", "run", "solver", "flow", "sweep"):
        cfg.setdefault(section, {})
    return cfg


def resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO / p).resolve()


def node_cpus() -> int:
    """CPUs Slurm gave this step (the whole node)."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def omp_threads(cfg: dict) -> int:
    """[run].omp_threads (host side: UCCSD, Python), 0 = all CPUs of this step."""
    n = int(cfg["run"].get("omp_threads", 0) or 0)
    return n or node_cpus()


def solver_layout(cfg: dict) -> dict:
    """MPI layout of each solve: one rank per GPU, ranks = task x adet x bdet comm sizes.

    Missing adet / bdet sizes default to the squarest split of gpus_per_task (4 -> 2 x 2).
    Raises SystemExit on a layout that cannot run, before any sampling.
    """
    s, r = cfg["solver"], cfg["run"]
    gpus = int(r.get("gpus_per_task", 4))
    task = int(s.get("task_comm_size", 1))
    per_task = max(1, gpus // task)
    a_def = max(d for d in range(1, per_task + 1) if per_task % d == 0 and d * d <= per_task)
    adet = int(s.get("adet_comm_size", a_def))
    bdet = int(s.get("bdet_comm_size", max(1, per_task // adet)))
    ranks = task * adet * bdet
    if ranks != gpus:
        raise SystemExit(
            f"[solver] task_comm_size x adet_comm_size x bdet_comm_size = {task} x {adet} x "
            f"{bdet} = {ranks} ranks, but [run].gpus_per_task = {gpus} (one rank per GPU)"
        )
    cpus = node_cpus() // ranks
    threads = int(s.get("threads", 0) or 0) or cpus
    if threads > cpus:
        raise SystemExit(
            f"[solver].threads = {threads} x {ranks} ranks > {node_cpus()} CPUs of the node "
            f"(at most {cpus} per rank)"
        )
    suffix = str(r.get("binary_suffix") or ("" if gpus == 1 else "-mpi-hpcx250"))
    binary = Path(C.sbd_paths()["diag"]) / f"diag-gpu_uhf{suffix}"
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"solver binary {binary} missing or not executable (setup how-to, Step 4)")
    return {
        "gpus": gpus,
        "task": task,
        "adet": adet,
        "bdet": bdet,
        "ranks": ranks,
        "cpus": cpus,
        "threads": threads,
        "suffix": suffix,
        "binary": binary,
    }


def runs_root(cfg: dict) -> Path:
    raw = str(cfg["run"].get("runs_root") or "").strip()
    return resolve(raw) if raw else HERE / "runs"


def sizes(cfg: dict, kind: str) -> list[int]:
    """job: [flow].sqd_dim; sweep: [sweep].sqd_dims (distinct positive integers, at least one).

    Raises SystemExit on a bad list, before any work.
    """
    if kind == "job":
        return [int(cfg["flow"].get("sqd_dim", 1_000_000))]
    raw = cfg["sweep"].get("sqd_dims")
    if not isinstance(raw, list) or not raw:
        raise SystemExit("[sweep] sqd_dims must be a non-empty list, e.g. [100_000, 1_000_000]")
    dims = [int(x) for x in raw]
    if len(set(dims)) != len(dims) or min(dims) < 1:
        raise SystemExit(f"[sweep] sqd_dims must be distinct positive integers; got {raw!r}")
    return dims


def names(spec) -> dict:
    """Per-task block names: concurrent tasks share one Prefect Cloud workspace."""
    tag = "".join(c if c.isalnum() or c == "-" else "-" for c in spec.run_name.lower())[:48]
    return {
        "tag": tag,
        "solver": f"bs-solver-{tag}",
        "command": f"bs-cmd-{tag}",
        "exec": f"bs-exec-{tag}",
        "hpc": f"bs-hpc-{tag}",
    }


def flow_params(cfg: dict, manifest: Path, spec):
    """FlowParameters from [flow] of run_bs.toml + this task's manifest and solver block."""
    from sbd.flow_params import FlowParameters

    data = json.loads(json.dumps(cfg["flow"]))  # deep copy
    data["active_space_manifest"] = str(manifest)
    data["solver_block_ref"] = f"sbd_solver_job/{names(spec)['solver']}"
    return FlowParameters.model_validate(data)


def create_blocks(cfg: dict, spec, work: Path, lay: dict) -> None:
    """This task's blocks: local target (the solver runs in this job step, on this node)."""
    n = names(spec)
    s = cfg["solver"]
    # 1 GPU: single-GPU binaries; >1: NCCL multi-GPU build (algorithms/sbd/native/README.md).
    suffix = lay["suffix"]
    p = C.sbd_paths()
    # OMP_NUM_THREADS of the solver ranks only; the host side keeps [run].omp_threads.
    blocks_cfg = work.parent / "create_blocks.toml"
    blocks_cfg.write_text(f'[environments]\nOMP_NUM_THREADS = "{lay["threads"]}"\n')
    C.run_create_blocks(
        [
            "--config",
            str(blocks_cfg),
            "--hpc-target",
            "local",
            "--method",
            "uhf",
            "--solver-mode",
            "gpu",
            "--num-nodes",
            "1",
            "--mpiprocs",
            str(lay["ranks"]),
            "--ompthreads",
            str(lay["threads"]),
            "--task-comm-size",
            str(lay["task"]),
            "--adet-comm-size",
            str(lay["adet"]),
            "--bdet-comm-size",
            str(lay["bdet"]),
            "--walltime",
            str(cfg["slurm"].get("time", "1-00:00:00")),
            "--work-dir",
            str(work),
            "--iteration",
            str(s.get("iteration", 5)),
            "--block",
            str(s.get("block", 20)),
            "--carryover-ratio",
            str(s.get("carryover_ratio", 0.5)),
            "--carryover-type",
            str(s.get("carryover_type", 1)),
            "--do-rdm",
            str(s.get("do_rdm", 0)),
            "--solver-timeout-seconds",
            str(s.get("timeout_seconds", 72000)),
            "--sbd-executable",
            os.path.join(p["diag"], f"diag-gpu{suffix}"),
            "--sbd-executable-uhf",
            os.path.join(p["diag"], f"diag-gpu_uhf{suffix}"),
            "--solver-block-name",
            n["solver"],
            "--command-block-name",
            n["command"],
            "--execution-profile-block-name",
            n["exec"],
            "--hpc-profile-block-name",
            n["hpc"],
        ]
    )


def solver_launcher(lay: dict) -> None:
    """The solver runs as a nested srun step inside this task's step: pin it to this node.

    Every count is explicit. Otherwise the inner srun inherits this step's --cpus-per-task (all
    CPUs) and asks for ranks x all CPUs: "More processors requested than permitted". No
    --gpu-bind: each rank must see all GPUs, main.cc picks its device as rank % #GPUs.
    (sweep/run_fe4s4_gpu_roquo.py measured --gpu-bind=closest + 36 CPUs per rank -> each rank
    saw only GPUs 0,1 -> two ranks per GPU -> NCCL error.)
    """
    node = os.environ.get("SLURMD_NODENAME", "")
    opts = [
        "--overlap",
        "-N1",
        f"-n{lay['ranks']}",
        f"--cpus-per-task={lay['cpus']}",
        f"--gpus-per-node={lay['gpus']}",
        "--mem=0",
    ]
    if node:
        opts.append(f"-w{node}")
    if lay["ranks"] > 1:
        opts.insert(0, "--mpi=pmix")
    opts += SOLVER_OUTPUT
    os.environ["SBD_LAUNCHER"] = "srun"
    os.environ["SBD_MPI_OPTIONS"] = ",".join(opts)


# Solver output goes to files in the solve's own work dir (the solver's cwd), one pair per rank
# (%t = rank), unbuffered: visible while a long solve runs, and never cut. Without this the
# executor holds it in memory until the solve ends and logs only its first 10k characters.
# solver_job.py reads only the result files (davidson_energy.txt, occ_*.txt, carryover*.bin).
SOLVER_OUTPUT = ["-u", "--output=solver-%t.out", "--error=solver-%t.err"]


# ---------------------------------------------------------------------------------------------
# Samples and checkpoints
# ---------------------------------------------------------------------------------------------
def samples_dir(base: Path) -> Path:
    """Where a sampling run persists its pool: runs/<run_name>/samples/storage/sqd_data/."""
    return base / "samples"


def _pools(base: Path) -> list[Path]:
    store = samples_dir(base) / "storage" / "sqd_data"
    pools = store.rglob("raw_samples_t00_w0_*.npz") if store.is_dir() else []
    return [p for p in pools if p.is_file()]


def find_pool(base: Path) -> Path | None:
    """The newest sample pool (trial 0, walker 0) of this task, by file time."""
    pools = _pools(base)
    return max(pools, key=lambda p: p.stat().st_mtime) if pools else None


def _pools_since(base: Path, t0: float) -> list[str]:
    """Sample pools of this task written after time t0."""
    return sorted(str(p) for p in _pools(base) if p.stat().st_mtime >= t0)


def _source_key(meta: dict) -> list:
    """What a checkpoint's samples were: the pool path, or the seed + shots of random samples."""
    if meta.get("quantum_source") == "random":
        return ["random", meta.get("random_seed"), meta.get("shots")]
    return [os.path.realpath(p) for p in list(meta.get("pools") or [])[:1]]


def record_sampled_pool(ckpt: Path, base: Path) -> None:
    """After a sampling run: record the pool it wrote, so a later run on that pool can resume.

    Called when the flow ends, and again before the next run if the job was killed before that
    (wall time). The pool is written before the first solve. Exactly one new pool must exist.
    """
    src = ckpt / "source.json"
    meta = json.loads(src.read_text()) if src.is_file() else {}
    if meta.get("quantum_source") != "real-device" or meta.get("pools"):
        return
    new = _pools_since(base, float(meta.get("started", 0)))
    if len(new) == 1:
        meta["pools"] = new
        src.write_text(json.dumps(meta, indent=2))
        print(f"[task] checkpoint: sampled pool {new[0]}", flush=True)
    else:
        print(f"[task] checkpoint: {len(new)} new pools, none recorded (no resume)", flush=True)


def prepare_checkpoint(rdir: Path, base: Path, params, spec, fresh: bool) -> Path:
    """Recovery checkpoint of one size (sqd.py: FE4S4_CKPT_DIR, one file per trial / walker).

    A checkpoint holds the recovery trajectory (occupancies, carryover, best energy) of ONE
    sample set and Hamiltonian. It is resumed only by a run on the same samples (the same pool,
    or random samples with the same seed and shots) with the same FCIDUMP, tag, sqd_dim and
    n_batches, and only without `fresh`. Otherwise it is moved to ckpt.prev-<time>/ (kept, never
    deleted) and the recovery starts at step 1.
    """
    ckpt = rdir / "ckpt"
    src = ckpt / "source.json"
    pools = [str(p).removeprefix("file://") for p in params.sampler.saved_samples]
    now = {
        "fcidump_sha256": spec.fcidump_sha256,
        "spin_tag": spec.spin_tag,
        "sqd_dim": params.sqd_dim,
        "n_batches": params.n_batches,
    }
    meta = {
        **now,
        "quantum_source": params.quantum_source,
        "pools": pools if params.quantum_source == "saved" else [],
        "random_seed": params.random_seed,
        "shots": params.sampler.shots,
    }
    files = sorted(ckpt.glob("ckpt_*.npz")) if ckpt.is_dir() else []
    if files:
        record_sampled_pool(ckpt, base)  # a sampling job killed before it could record its pool
        old = json.loads(src.read_text()) if src.is_file() else {}
        diff = [k for k, v in now.items() if old.get(k) != v]
        if params.quantum_source == "real-device":
            why = "new samples on the device"
        elif fresh:
            why = "action new"
        elif not _source_key(meta) or _source_key(meta) != _source_key(old):
            why = "other samples"
        elif diff:
            why = "other " + ", ".join(diff)
        else:
            why = ""
        if not why:
            print(f"[task] checkpoint: resuming {', '.join(f.name for f in files)}", flush=True)
            return ckpt
        aside = rdir / f"ckpt.prev-{time.strftime('%Y%m%d-%H%M%S')}"
        ckpt.rename(aside)
        print(f"[task] checkpoint: not resumed ({why}); moved to {aside}", flush=True)
    else:
        print(f"[task] checkpoint: none in {ckpt}; recovery starts at step 1", flush=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    meta.update(started=time.time(), slurm_job=os.environ.get("SLURM_JOB_ID"))
    src.write_text(json.dumps(meta, indent=2))
    return ckpt


def params_for(params, pool: Path | None, dim: int, resample: bool):
    """The parameters of one size: recover from the task's pool when it has one.

    [flow].quantum_source = "real-device" samples only when the task has no pool yet (or with
    --resample). "random" and "saved" (with your own sampler.saved_samples) are used as given.
    """
    p = params.model_copy(update={"sqd_dim": dim})
    if params.quantum_source == "real-device" and pool is not None and not resample:
        sampler = params.sampler.model_copy(update={"saved_samples": [f"file://{pool}"]})
        p = p.model_copy(update={"quantum_source": "saved", "sampler": sampler})
    return p


def is_done(rdir: Path, params) -> bool:
    """A finished result with at least the recovery steps asked now."""
    res = rdir / "result.json"
    if not res.is_file():
        return False
    steps = json.loads(res.read_text()).get("flow_parameters", {}).get("n_recovery_steps", 0)
    return int(steps) >= params.n_recovery_steps


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------
def provenance(spec) -> dict:
    """How the manifest's active space was chosen, and its preliminary note if any.

    Forced active spaces (selection method "uno-avas-forced") carry a top-level "preliminary"
    note. Their energies are not comparable to UNO-AVAS results, so every result says so.
    """
    m = spec.manifest or {}
    method = ((m.get("active_space") or {}).get("selection") or {}).get("method")
    note = m.get("preliminary")
    return {
        "active_space_method": method,
        "preliminary": bool(note) or method == "uno-avas-forced",
        "preliminary_note": note,
    }


def result_label(spec) -> str:
    """Suffix of the [task] / CAPTURE lines: ' PRELIMINARY (<method>)' for such tasks."""
    p = provenance(spec)
    return f" PRELIMINARY ({p['active_space_method'] or 'preliminary'})" if p["preliminary"] else ""


def write_result(rdir: Path, spec, params, best, flow_run_id, dt, mode: str) -> Path:
    pools = [str(p) for p in params.sampler.saved_samples]
    out = {
        "molecule": spec.run_name,
        "sqd_dim": params.sqd_dim,
        "quantum_source": params.quantum_source,
        # the pool this run recovered from (quantum_source = saved), else None
        "sample_pool": pools[0] if params.quantum_source == "saved" and pools else None,
        **provenance(spec),
        "cluster_type": spec.cluster_type,
        "spin_tag": spec.spin_tag,
        "fcidump_sha256": spec.fcidump_sha256,
        "manifest": str(spec.manifest_path),
        "norb": spec.norb,
        "nelec": [spec.nalpha, spec.nbeta],
        "e_active_uhf": spec.e_active_uhf,
        "best_energy": best,
        "elapsed_s": dt,
        "mode": mode,
        "node": os.environ.get("SLURMD_NODENAME"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_step": os.environ.get("SLURM_STEP_ID"),
        "flow_run_id": str(flow_run_id) if flow_run_id else None,
        "flow_parameters": json.loads(params.model_dump_json()),
        "recovery_trace": C.read_recovery_trace(flow_run_id) if flow_run_id else [],
    }
    path = rdir / "result.json"
    if path.is_file():  # an earlier result of this size (action new): keep it
        path.rename(rdir / f"result.prev-{time.strftime('%Y%m%d-%H%M%S')}.json")
    path.write_text(json.dumps(out, indent=2))
    return path


def capture(name: str, spec, best, path: Path) -> None:
    """The result line run_bs.sh's job summary and report look for."""
    print(
        f"CAPTURE_BS_SQD {name}: best_E={best} tag={spec.spin_tag}{result_label(spec)} -> {path}",
        flush=True,
    )


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------
def serve_one(params, deployment: str, tags: list[str]):
    """Serve the flow; run exactly one flow run (started from the UI), then return it.

    The served process polls Prefect Cloud every PREFECT_RUNNER_POLL_FREQUENCY seconds (10 s)
    and executes the run in a subprocess on this node; it exits once that run is final.
    """
    import asyncio

    from prefect import get_client
    from prefect.client.schemas.filters import DeploymentFilter, DeploymentFilterId
    from prefect.client.schemas.sorting import FlowRunSort
    from prefect.runner import Runner
    from sbd.main import riken_sqd_de

    async def _serve():
        runner = Runner(name=f"bs-{deployment}", limit=1)
        async with runner:
            deployment_id = await runner.aadd_flow(
                riken_sqd_de,
                name=deployment,
                parameters={"parameters": params.model_dump(mode="json")},
                tags=tags,
            )
            print(
                f"[cloud] serving riken_sqd_de/{deployment}: start ONE run from the UI "
                "(Deployments -> riken_sqd_de -> this name -> Run). This step exits when that "
                "run is finished.",
                flush=True,
            )
            dfilter = DeploymentFilter(id=DeploymentFilterId(any_=[deployment_id]))
            async with get_client() as client:
                while True:
                    await runner._scheduled_run_poller.run_once()  # returns after runs finish
                    runs = await client.read_flow_runs(
                        deployment_filter=dfilter, sort=FlowRunSort.START_TIME_DESC, limit=1
                    )
                    if runs and runs[0].state and runs[0].state.is_final():
                        return runs[0]
                    await asyncio.sleep(10)

    run = asyncio.run(_serve())
    print(f"[cloud] flow run {run.name} ({run.id}) finished: {run.state.type.value}", flush=True)
    return run


def cloud_result_energy(run) -> float | None:
    """Best energy of a finished served run (its persisted return value), if retrievable."""
    try:
        return float(run.state.result())
    except Exception as exc:
        print(f"[cloud] could not read the run's return value ({exc}); see the UI.", flush=True)
        return None


def run_size(base: Path, rdir: Path, spec, params, fresh: bool, mode: str) -> Path:
    """One flow run of one size into rdir: checkpoint, sample storage, parameters, result."""
    rdir.mkdir(parents=True, exist_ok=True)
    ckpt = prepare_checkpoint(rdir, base, params, spec, fresh)
    os.environ["FE4S4_CKPT_DIR"] = str(ckpt)
    # a sampling run persists its pool in samples/; recovery reports go to the size's own dir.
    # PREFECT_HOME stays the user's (Prefect Cloud profile).
    store = samples_dir(base) if params.quantum_source == "real-device" else rdir / "prefect"
    C.isolate_prefect(store, storage_only=True)
    (rdir / "flow_parameters.json").write_text(params.model_dump_json(indent=2))
    name = f"{spec.run_name}-d{params.sqd_dim}"
    t0 = time.perf_counter()
    try:
        if mode == "direct":
            best, flow_run_id = C.run_flow(params, flow_run_name=name)
        else:
            run = serve_one(params, names(spec)["tag"], ["bs-uhf-sqd", spec.cluster_type or "fes"])
            if not run.state.is_completed():
                raise SystemExit(f"[cloud] run {run.name} ended {run.state.type.value}")
            best, flow_run_id = cloud_result_energy(run), run.id
    finally:
        record_sampled_pool(ckpt, base)
    path = write_result(rdir, spec, params, best, flow_run_id, time.perf_counter() - t0, mode)
    capture(name, spec, best, path)
    return path


def run_sizes(base: Path, spec, params, dims: list[int], action: str, resample: bool, mode: str):
    """Every size in turn on the task's samples. A failed size does not stop the next, except a
    failed sampling run (the others would have no samples). Exits non-zero if any size failed.
    """
    pool = find_pool(base)
    fresh = action == "new"
    if pool is not None and not resample and params.quantum_source == "real-device":
        print(f"[task] samples: {pool}", flush=True)
    elif params.quantum_source == "real-device":
        print(f"[task] samples: sampled on the device by size {dims[0]}", flush=True)
    else:
        print(f"[task] samples: quantum_source = {params.quantum_source}", flush=True)
    failed = []
    for i, dim in enumerate(dims):
        rdir = base / f"d{dim}"
        if not fresh and is_done(rdir, params):
            print(f"[task] size {dim}: done ({rdir / 'result.json'}), skipped", flush=True)
            continue
        p = params_for(params, pool, dim, resample)
        print(f"[task] size {dim}: run dir {rdir}, quantum_source {p.quantum_source}", flush=True)
        try:
            run_size(base, rdir, spec, p, fresh, mode)
        except Exception as exc:
            print(f"[task] size {dim} FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed.append(dim)
        if p.quantum_source == "real-device":  # this size sampled: the next ones use its pool
            resample = False
            pool = find_pool(base)
            if pool is None:
                raise SystemExit(
                    f"[task] size {dim} wrote no sample pool; not run: {dims[i + 1 :]}"
                )
    if failed:
        raise SystemExit(f"[task] {len(failed)} of {len(dims)} size(s) failed: {failed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kind", choices=["job", "sweep"], default="job")
    ap.add_argument("--action", choices=["continue", "new"], default="continue")
    ap.add_argument("--resample", action="store_true", help="new samples, even with a pool")
    ap.add_argument("--mode", choices=["direct", "cloud"], help="override [run].mode")
    ap.add_argument("--check", action="store_true", help="manifest + reference only")
    ap.add_argument("--print-params", action="store_true", help="print FlowParameters JSON")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    manifest = resolve(args.manifest)
    os.environ.setdefault("SBD_BS_STRICT", "1")
    spec = C.load_spec(str(manifest))
    params = flow_params(cfg, manifest, spec)
    mode = args.mode or str(cfg["run"].get("mode", "direct"))

    if args.print_params:
        print(params.model_dump_json(indent=2))
        return
    if args.check:
        C.dry_run_report(str(manifest), None, "uhf")
        return

    # before any work: a bad layout or size list must not cost IBM shots or a node
    lay = solver_layout(cfg)
    dims = sizes(cfg, args.kind)
    if mode != "direct" and args.kind != "job":
        raise SystemExit("[task] a sweep runs in direct mode only (MODE=cloud: job only)")
    base = runs_root(cfg) / spec.run_name
    work = base / "work"
    work.mkdir(parents=True, exist_ok=True)
    print(
        f"[task] {spec.run_name}: node {os.environ.get('SLURMD_NODENAME')}, {args.kind}, "
        f"action {args.action}, mode {mode}, run dir {base}",
        flush=True,
    )
    if result_label(spec):
        print(f"[task]{result_label(spec)}: {provenance(spec)['preliminary_note']}", flush=True)
    C.prefect_defaults(omp_threads(cfg))
    print(
        f"[task] threads {omp_threads(cfg)}, GPUs {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}",
        flush=True,
    )
    print(
        f"[task] solver: {lay['ranks']} rank(s) = task {lay['task']} x adet {lay['adet']} x "
        f"bdet {lay['bdet']}, {lay['threads']} threads/rank ({lay['cpus']} CPUs/rank), "
        f"{lay['gpus']} GPU(s), {lay['binary']}",
        flush=True,
    )
    # one BS-UHF reference cache per task, shared by all its sizes (same FCIDUMP)
    os.environ["SBD_UHF_CACHE_DIR"] = str(base / "uhf_cache")
    solver_launcher(lay)
    create_blocks(cfg, spec, work, lay)
    run_sizes(base, spec, params, dims, args.action, args.resample, mode)


if __name__ == "__main__":
    main()
