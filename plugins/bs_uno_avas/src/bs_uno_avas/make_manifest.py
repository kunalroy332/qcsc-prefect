"""Wrap an existing artifact set (no manifest) into a schema-v1 manifest.

    python -m bs_uno_avas.make_manifest output/<prefix> \
        --cluster-resname FES --pdb data/5XTD_FES_O301_SCH3.pdb --charge -3 --basis 6-31g*

``<prefix>`` is the path prefix of the four files (``<prefix>_avas.fcidump`` ...). The reference
block is recomputed: active-space BS-UHF from the RDM (and from labels), tag, energies. The
manifest is written next to the files as ``<prefix>_manifest.json``; ``spin_pattern`` is taken
from the converged tag. Full-system fields that are unknown for the old sets are ``null``.
"""

import argparse
from pathlib import Path

import numpy as np

from . import manifest, verify
from .structure import CLUSTER_RESNAMES, CLUSTER_TYPES, file_sha256


def wrap(
    prefix,
    cluster_resname,
    pdb=None,
    charge=None,
    multiplicity_hs=None,
    basis=None,
    method="ROHF-AVAS",
    redox_label=None,
    geometry_label=None,
    notes="",
):
    prefix = Path(prefix)
    out, name = prefix.parent, prefix.name
    fcidump = out / f"{name}_avas.fcidump"
    header = manifest.read_fcidump_header(fcidump)
    labels = [str(x) for x in np.load(out / f"{name}_active_orbitals_labels.npy")]
    rdm = np.load(out / f"{name}_active_space_rdm.npy")
    n_fe = len({lab.split("_")[0] for lab in labels if lab.split("_")[1] == "Fe"})
    n_s = len({lab.split("_")[0] for lab in labels if lab.split("_")[1] == "S"})
    if n_fe not in CLUSTER_TYPES or n_fe != CLUSTER_RESNAMES.get(cluster_resname, n_fe):
        raise ValueError(f"labels have {n_fe} Fe, which does not fit cluster {cluster_resname}")

    # pattern from the RDM itself (the starting point the flow uses), then converge
    tag0 = verify.spin_tag(verify.atom_spins(rdm, labels))
    if "x" in tag0:
        raise ValueError(f"RDM has an Fe without local moment (tag {tag0})")
    pattern = [1 if c == "0" else -1 for c in tag0]
    ver = verify.verify(fcidump, labels, rdm, pattern, tag0)
    ver.pop("rdm_converged")
    ver["label_start"].pop("rdm")
    ls = ver["label_start"]
    if ls["converged"] and ls["spin_tag"] == tag0 and ls["e"] < ver["e_active_uhf"] - 1e-4:
        ver["warnings"].append(
            f"label-start UHF reaches the same tag {ls['e'] - ver['e_active_uhf']:+.2e} Eh lower;"
            " a consumer that keeps the lowest start may not reproduce e_active_uhf"
        )
    if ver["spin_tag"] != tag0:
        raise RuntimeError(f"active-space UHF from the RDM moved from {tag0} to {ver['spin_tag']}")
    pattern = [1 if c == "0" else -1 for c in ver["spin_tag"]]
    nelec, ms2 = header["NELEC"], header["MS2"]
    body = {
        "structure": {
            "pdb_file": str(pdb) if pdb else None,
            "pdb_sha256": file_sha256(pdb) if pdb else None,
            "cluster_resname": cluster_resname,
            "cluster_type": CLUSTER_TYPES[n_fe],
            "n_fe": n_fe,
            "n_s_bridging": n_s,
            "notes": notes,
        },
        "electronic": {
            "charge": charge,
            "multiplicity_hs": multiplicity_hs,
            "basis": basis,
            "method": method,
            "xc": None,
            "engine": "pyscf",
            "redox_label": redox_label,
            "geometry_label": geometry_label,
        },
        "active_space": {
            "norb": header["NORB"],
            "nelec": nelec,
            "ms2": ms2,
            "nalpha": (nelec + ms2) // 2,
            "nbeta": (nelec - ms2) // 2,
            "labels_file": f"{name}_active_orbitals_labels.npy",
            "selection": {"method": "wrapped-existing-set"},
        },
        "bs_reference": {
            "spin_pattern": pattern,
            "spin_tag": ver["spin_tag"],
            "atom_spins": ver["atom_spins"],
            "e_full_uhf": None,
            "e_active_uhf": ver["e_active_uhf"],
            "s2_active_uhf": ver["s2_active_uhf"],
            "e_active_rdm": ver["e_rdm"],
            "label_start": ver["label_start"],
            "rdm_file": f"{name}_active_space_rdm.npy",
            "rdm_trace": [float(np.trace(r)) for r in rdm],
            "verify_warnings": ver["warnings"],
        },
        "candidates": [],
    }
    return manifest.write_manifest(out, name, body), ver


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bs_uno_avas.make_manifest")
    ap.add_argument("prefix")
    ap.add_argument("--cluster-resname", required=True, help="cluster residue name, e.g. FES")
    ap.add_argument("--pdb")
    ap.add_argument("--charge", type=int)
    ap.add_argument("--multiplicity-hs", type=int)
    ap.add_argument("--basis")
    ap.add_argument("--method", default="ROHF-AVAS")
    ap.add_argument("--redox-label")
    ap.add_argument("--geometry-label")
    args = ap.parse_args(argv)
    path, ver = wrap(
        args.prefix,
        args.cluster_resname,
        args.pdb,
        args.charge,
        args.multiplicity_hs,
        args.basis,
        args.method,
        args.redox_label,
        args.geometry_label,
    )
    print(f"{path}: tag {ver['spin_tag']} E(active UHF) {ver['e_active_uhf']:.6f}")
    for w in ver["warnings"]:
        print(f"  WARNING {w}")


if __name__ == "__main__":
    main()
