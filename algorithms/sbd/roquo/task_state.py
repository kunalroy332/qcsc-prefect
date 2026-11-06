"""State of each task (manifest) and subspace size, for run_bs.sh on the login node.

Standard library only, Python >= 3.6: the login node's python3 has no tomllib / numpy, and the
SBD venv runs only on compute nodes. run_bs.sh reads run_bs.toml (awk) and passes the values.

One tab-separated line per (manifest, size):

    index  run_name  size  state  detail  samples

state: done (a result with at least the asked recovery steps), resume (a checkpoint the next
run continues), stale (a checkpoint the next run cannot continue: detail says why; it is moved
to ckpt.prev-<time>/ and the size starts at step 1), new (nothing yet), bad (manifest problem).

The resume rule is run_task.py's prepare_checkpoint(): the same samples (the task's newest pool,
or random samples with the same seed and shots), FCIDUMP, spin tag, sqd_dim and n_batches.
"""

import argparse
import glob
import json
import os
import struct
import time
import zipfile


def npy_first(zf, name):
    """First element of an int64 / float64 .npy inside an npz (uncompressed or deflated)."""
    data = zf.read(name)
    if data[:6] != b"\x93NUMPY":
        return None
    if data[6] == 1:
        hlen, off = struct.unpack("<H", data[8:10])[0], 10
    else:
        hlen, off = struct.unpack("<I", data[8:12])[0], 12
    header = data[off : off + hlen].decode("latin1")
    body = data[off + hlen :]
    if "<i8" in header:
        return struct.unpack("<q", body[:8])[0]
    if "<f8" in header:
        return struct.unpack("<d", body[:8])[0]
    return None


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def manifest_info(path):
    m = read_json(path)
    ref = m.get("bs_reference") or {}
    fcid = (m.get("files") or {}).get("fcidump") or {}
    return m.get("run_name"), ref.get("spin_tag"), fcid.get("sha256")


def pools(base):
    pat = os.path.join(base, "samples", "storage", "sqd_data", "**", "raw_samples_t00_w0_*.npz")
    return [p for p in glob.glob(pat, recursive=True) if os.path.isfile(p)]


def pool_text(pool):
    if not pool:
        return "none"
    day = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(pool)))
    try:
        with zipfile.ZipFile(pool) as zf:
            shots = npy_first(zf, "num_shots.npy")
    except (OSError, zipfile.BadZipFile, KeyError):
        shots = None
    return "pool %s (%s shots) %s" % (day, shots if shots is not None else "?", pool)


def source_key(meta):
    if meta.get("quantum_source") == "random":
        return ["random", meta.get("random_seed"), meta.get("shots")]
    # resolved: the login node and the job may name one project dir differently (symlinks)
    return [os.path.realpath(p) for p in list(meta.get("pools") or [])[:1]]


def recorded(meta, base):
    """A sampling run killed before it recorded its pool (run_task.record_sampled_pool)."""
    if meta.get("quantum_source") == "real-device" and not meta.get("pools"):
        t0 = float(meta.get("started", 0))
        new = sorted(p for p in pools(base) if os.path.getmtime(p) >= t0)
        if len(new) == 1:
            meta = dict(meta, pools=new)
    return meta


def size_state(a, base, size, tag, sha, pool):
    rdir = os.path.join(base, "d%d" % size)
    res = read_json(os.path.join(rdir, "result.json"))
    done_steps = int((res.get("flow_parameters") or {}).get("n_recovery_steps", 0) or 0)
    if res and done_steps >= a.n_steps:
        return "done", "done E=%s (%d steps)" % (res.get("best_energy"), done_steps)
    ck = os.path.join(rdir, "ckpt", "ckpt_t00_w0.npz")
    if not os.path.isfile(ck):
        return "new", "new"
    try:
        with zipfile.ZipFile(ck) as zf:
            step = npy_first(zf, "next_step.npy")
            best = npy_first(zf, "best_energy.npy")
    except (OSError, zipfile.BadZipFile, KeyError):
        return "stale", "unreadable checkpoint; starts at step 1"
    at = "step %s/%d E=%s" % (step, a.n_steps, "%.6f" % best if best is not None else "?")
    old = recorded(read_json(os.path.join(rdir, "ckpt", "source.json")), base)
    # what the next run uses: the task's pool when it has one (run_task.params_for)
    source = a.source
    if source == "real-device" and pool and not a.resample:
        source = "saved"
    now = {"quantum_source": source, "random_seed": a.seed, "shots": a.shots}
    now["pools"] = [pool] if source == "saved" and a.source == "real-device" else []
    if a.source == "saved":
        now["pools"] = [a.saved_pool.replace("file://", "", 1)] if a.saved_pool else []
    diff = [
        k
        for k, v in (
            ("fcidump_sha256", sha),
            ("spin_tag", tag),
            ("sqd_dim", size),
            ("n_batches", a.n_batches),
        )
        if old.get(k) != v
    ]
    if source == "real-device":
        why = "new samples on the device"
    elif not source_key(now) or source_key(now) != source_key(old):
        why = "other samples"
    elif diff:
        why = "other " + ", ".join(diff)
    else:
        return "resume", at
    return "stale", "checkpoint at %s not resumable (%s); starts at step 1" % (at, why)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--sizes", required=True, help="comma-separated subspace sizes")
    ap.add_argument("--all-sizes", action="store_true", help="also the existing d<size> dirs")
    ap.add_argument("--n-steps", type=int, required=True)
    ap.add_argument("--n-batches", type=int, required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--saved-pool", default="")
    ap.add_argument("--resample", action="store_true")
    ap.add_argument("manifests", nargs="*")
    a = ap.parse_args()
    asked = [int(s) for s in a.sizes.split(",") if s]
    for i, m in enumerate(a.manifests):
        run, tag, sha = manifest_info(m)
        if not run:
            print("\t".join([str(i), "?", "0", "bad", "no run_name in " + m, "none"]))
            continue
        base = os.path.join(a.runs_root, run)
        found = pools(base)
        pool = max(found, key=os.path.getmtime) if found else ""
        sizes = list(asked)
        if a.all_sizes:
            for d in sorted(glob.glob(os.path.join(base, "d[0-9]*"))):
                n = os.path.basename(d)[1:]
                if n.isdigit() and int(n) not in sizes:
                    sizes.append(int(n))
        for size in sizes:
            state, detail = size_state(a, base, size, tag, sha, pool)
            print("\t".join([str(i), run, str(size), state, detail, pool_text(pool)]))


if __name__ == "__main__":
    main()
