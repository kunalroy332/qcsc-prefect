"""Manifest writer / reader (``schema_version: 1``).

The consumer is ``qcsc_workflow_utility.active_space.load_active_space_manifest``; see
``docs/tutorials/run_bs_uno_avas_roquo.md`` (Step 4).
"""

import datetime as _dt
import json
import subprocess
from pathlib import Path

import numpy as np
from pyscf import tools

from . import __version__
from .structure import file_sha256

SCHEMA_VERSION = 1
FILE_SUFFIXES = {
    "fcidump": "_avas.fcidump",
    "labels": "_active_orbitals_labels.npy",
    "rdm": "_active_space_rdm.npy",
    "orbitals": "_active_orbitals.npy",
}
REQUIRED = [
    ("schema_version",),
    ("active_space", "norb"),
    ("active_space", "nelec"),
    ("active_space", "ms2"),
    ("bs_reference", "spin_pattern"),
    ("bs_reference", "spin_tag"),
    ("files", "fcidump"),
    ("files", "labels"),
    ("files", "rdm"),
]


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_entries(out_dir, prefix):
    out_dir = Path(out_dir)
    entries = {}
    for key, suffix in FILE_SUFFIXES.items():
        path = out_dir / f"{prefix}{suffix}"
        if path.exists():
            entries[key] = {"name": path.name, "sha256": file_sha256(path)}
        elif key != "orbitals":
            raise FileNotFoundError(path)
    return entries


def write_manifest(out_dir, prefix, body):
    """Write ``<prefix>_manifest.json``; ``body`` has every section except ``files``/header."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_name": prefix,
        "created_utc": utc_now(),
        "generator": {
            "tool": "bs_uno_avas",
            "version": __version__,
            "git_sha": git_sha(),
        },
        **body,
        "files": file_entries(out_dir, prefix),
    }
    path = Path(out_dir) / f"{prefix}_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
        f.write("\n")
    return path


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serializable: {type(x)}")


def read_manifest(path, check_hashes=True):
    """Read and validate a manifest (required keys, file hashes, FCIDUMP header, shapes)."""
    path = Path(path)
    m = json.loads(path.read_text())
    for keys in REQUIRED:
        node = m
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise ValueError(f"{path}: missing required key {'.'.join(keys)}")
            node = node[k]
    if m["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {m['schema_version']} != {SCHEMA_VERSION}")
    files = {}
    for key, entry in m["files"].items():
        f = path.parent / entry["name"]
        if not f.exists():
            raise ValueError(f"{path}: file {f} missing")
        if check_hashes and entry.get("sha256") and file_sha256(f) != entry["sha256"]:
            raise ValueError(f"{path}: sha256 mismatch for {f.name}")
        files[key] = f
    a = m["active_space"]
    header = read_fcidump_header(files["fcidump"])
    for k, mk in (("NORB", "norb"), ("NELEC", "nelec"), ("MS2", "ms2")):
        if header[k] != a[mk]:
            raise ValueError(f"{path}: FCIDUMP {k}={header[k]} != manifest {mk}={a[mk]}")
    labels = np.load(files["labels"])
    rdm = np.load(files["rdm"])
    if labels.shape != (a["norb"],) or len(set(labels.tolist())) != a["norb"]:
        raise ValueError(f"{path}: labels must be {a['norb']} unique strings")
    if rdm.shape != (2, a["norb"], a["norb"]):
        raise ValueError(f"{path}: rdm shape {rdm.shape}")
    pattern, tag = m["bs_reference"]["spin_pattern"], m["bs_reference"]["spin_tag"]
    if "".join("0" if s > 0 else "1" for s in pattern) != tag:
        raise ValueError(f"{path}: spin_tag {tag} inconsistent with spin_pattern {pattern}")
    return m, files, labels, rdm


def read_fcidump_header(path):
    """NORB / NELEC / MS2 from the namelist header only (no integrals)."""
    text = []
    with open(path) as f:
        for line in f:
            text.append(line)
            if line.strip().upper().startswith(("&END", "/")):
                break
    header = " ".join(text).replace("\n", " ").upper()
    out = {}
    for key in ("NORB", "NELEC", "MS2"):
        rest = header.split(key, 1)[1].lstrip(" =")
        out[key] = int(rest.split(",")[0].split()[0])
    return out


def fcidump_ecore(path):
    return float(tools.fcidump.read(str(path), verbose=False)["ECORE"])
