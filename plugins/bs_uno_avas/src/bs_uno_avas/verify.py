"""Reference check: active-space BS-UHF from the RDM (and from labels) reproduces the reference.

Active-space UHF from the FCIDUMP (``to_scf().to_uhf()``), started from an explicit density and
followed by internal stability analysis until stable. The SBD flow rebuilds the reference from
the manifest the same way (RDM start and label start, lowest converged start with the manifest
tag), so a manifest that passes here reproduces there.
"""

import contextlib
import io

import numpy as np
from pyscf import lib, tools


def label_density(labels, pattern):
    """Label guess: Fe 3d in α (+1) or β (−1) per ``pattern``, other orbitals doubly occupied."""
    n = len(labels)
    dm0 = np.zeros((2, n, n))
    fe_ids = sorted({int(lab.split("_")[0]) for lab in labels if lab.split("_")[1] == "Fe"})
    if len(pattern) != len(fe_ids):
        raise ValueError(f"pattern needs one entry per Fe atom {fe_ids}, got {pattern}")
    for i, lab in enumerate(labels):
        atom_id, element = int(lab.split("_")[0]), lab.split("_")[1]
        if element == "Fe":
            dm0[0 if pattern[fe_ids.index(atom_id)] > 0 else 1, i, i] = 1.0
        else:
            dm0[:, i, i] = 1.0
    return dm0


def active_uhf(fcidump_file, dm0=None, n_stability=10, max_cycle=100):
    with contextlib.redirect_stdout(io.StringIO()):  # silence "Parsing <file>"
        mf = tools.fcidump.to_scf(str(fcidump_file)).to_uhf()
    mf.verbose = 0
    mf.chkfile = None
    mf.max_cycle = max_cycle
    mf.kernel(dm0)
    for _ in range(n_stability):
        try:
            # internal UHF->UHF only: the external (UHF->GHF) mode would not be followed
            mo, _, stable, _ = mf.stability(return_status=True)
        except lib.exceptions.LinearDependencyError:
            break
        if stable:
            break
        mf.kernel(mf.make_rdm1(mo, mf.mo_occ))
    return mf


def atom_spins(dm, labels):
    """α − β electrons per atom (``"0_Fe"``, ...) in the label basis, in label order."""
    spins = {}
    for i, lab in enumerate(labels):
        key = "_".join(lab.split("_")[:2])
        spins[key] = spins.get(key, 0.0) + dm[0][i, i] - dm[1][i, i]
    return spins


def spin_tag(spins, min_spin=1.0):
    """Tag from per-atom spins: Fe in ascending model index, 0 = up, 1 = down, x = no moment."""
    fe = sorted(((int(k.split("_")[0]), v) for k, v in spins.items() if k.split("_")[1] == "Fe"))
    return "".join("x" if abs(v) < min_spin else ("0" if v > 0 else "1") for _, v in fe)


def verify(fcidump_file, labels, rdm, pattern, full_tag, label_weights=None, n_stability=10):
    """Run both starts (RDM, labels) and the gates. Returns a dict for the manifest + messages.

    ``rdm_converged`` / ``label_start.rdm`` are the converged densities of the two starts.
    """
    mf = active_uhf(fcidump_file, dm0=rdm, n_stability=n_stability)
    dm = np.asarray(mf.make_rdm1())
    spins = atom_spins(dm, labels)
    tag = spin_tag(spins)
    mf_lab = active_uhf(fcidump_file, dm0=label_density(labels, pattern), n_stability=n_stability)
    tag_lab = spin_tag(atom_spins(np.asarray(mf_lab.make_rdm1()), labels))
    warnings = []
    if tag != full_tag:
        warnings.append(f"active-space tag {tag} != full-system tag {full_tag}")
    if not mf.converged:
        warnings.append("active-space UHF from the RDM did not converge")
    weak = [k for k, v in spins.items() if k.split("_")[1] == "Fe" and abs(v) < 3.0]
    if weak:
        warnings.append(f"Fe with |spin| < 3 in the active space: {weak}")
    if mf_lab.converged and tag_lab != tag and mf_lab.e_tot < mf.e_tot - 1e-4:
        warnings.append(
            f"label-start UHF with another tag ({tag_lab}, {mf_lab.e_tot:.6f}) is lower than the "
            f"RDM start ({tag}, {mf.e_tot:.6f}) by > 0.1 mEh"
        )
    if label_weights is not None:
        from .labels import check_weights

        warnings += [f"low label weight: {w}" for w in check_weights(labels, label_weights)]
    e_rdm_start = float(mf.energy_tot(dm=rdm))  # energy of the RDM itself (above e_active_uhf)
    return {
        "e_active_uhf": float(mf.e_tot),
        "s2_active_uhf": float(mf.spin_square()[0]),
        "atom_spins": {k: round(float(v), 4) for k, v in spins.items()},
        "spin_tag": tag,
        "converged": bool(mf.converged),
        "e_rdm": e_rdm_start,
        "label_start": {
            "e": float(mf_lab.e_tot),
            "spin_tag": tag_lab,
            "converged": bool(mf_lab.converged),
            "rdm": np.asarray(mf_lab.make_rdm1()),
        },
        "warnings": warnings,
        "rdm_converged": dm,
    }
