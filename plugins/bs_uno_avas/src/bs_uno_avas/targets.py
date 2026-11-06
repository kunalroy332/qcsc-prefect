"""Target AO sets for the active space: Fe 3d on every cluster Fe, 3p on every bridging S.

Two sources:

- ``file:<dir>`` — free-atom ROHF orbitals ``<dir>/{Fe,S}_AO.npy`` + labels ``{Fe,S}_lb.npy``
  (``file:AOs`` = the package data ``bs_uno_avas/data/AOs``, made by ``tools/make_aos.py``;
  PySCF ``dzvp`` basis). Embedded into the AO block of each cluster atom. The per-element AO
  count must match the ``.npy`` row count.
- ``free-atom`` — spherically averaged atomic RHF orbitals (``lo.orth.pre_orth_ao(mol, 'scf')``),
  any basis.

Both return orthonormal targets ``T`` (``Tᵀ S T = 1``, symmetric Löwdin, so each column stays on
its atom) in the order Fe0 3d, Fe1 3d, ..., S 3p, and labels ``"<model idx>_<El>_<AO>"``.
"""

from pathlib import Path

import numpy as np
from pyscf import lo
from scipy import linalg

SHELLS = {"Fe": "3d", "S": "3p"}


def lowdin_orthonormalize(c, ovlp):
    w, u = linalg.eigh(c.T @ ovlp @ c)
    if w.min() < 1e-8:
        raise ValueError(
            f"target orbitals are linearly dependent (min overlap eigenvalue {w.min():.2e})"
        )
    return c @ u @ np.diag(w**-0.5) @ u.T


def _cluster_elements(mol, n_cluster):
    elements = [mol.atom_pure_symbol(i) for i in range(n_cluster)]
    bad = [e for e in elements if e not in SHELLS]
    if bad:
        raise ValueError(f"cluster atoms must be Fe or S, got {elements}")
    return elements


def file_targets(mol, n_cluster, ao_dir):
    """Targets from ``<ao_dir>/{El}_AO.npy`` / ``{El}_lb.npy`` (see module docstring)."""
    ao_dir = Path(ao_dir)
    aoslice = mol.aoslice_by_atom()
    cols, labels = [], []
    for i, el in enumerate(_cluster_elements(mol, n_cluster)):
        orbs = np.load(ao_dir / f"{el}_AO.npy")
        names = [str(x) for x in np.load(ao_dir / f"{el}_lb.npy")]
        p0, p1 = aoslice[i, 2], aoslice[i, 3]
        if orbs.shape[0] != p1 - p0:
            raise ValueError(
                f"{ao_dir}/{el}_AO.npy has {orbs.shape[0]} AO rows but atom {i} ({el}) has "
                f"{p1 - p0} AOs in basis {mol.basis!r}; file targets need the same basis as "
                "make_aos.py (dzvp) - use target_aos = 'free-atom' for other bases"
            )
        for j, name in enumerate(names):
            if name.startswith(SHELLS[el]):
                c = np.zeros(mol.nao)
                c[p0:p1] = orbs[:, j]
                cols.append(c)
                labels.append(f"{i}_{el}_{name}")
    order = _block_order(labels)
    c = np.array(cols).T[:, order]
    return lowdin_orthonormalize(c, mol.intor_symmetric("int1e_ovlp")), [labels[k] for k in order]


def projected_file_targets(mol, n_cluster, ao_dir, ref_basis="dzvp"):
    """:func:`file_targets` built in ``ref_basis`` and projected onto the basis of ``mol``.

    For a basis other than that of ``make_aos.py`` (e.g. Gaussian DGDZVP, possibly Cartesian):
    the targets of a copy of ``mol`` in ``ref_basis`` (spherical) are projected per cluster atom
    onto the AOs of that atom in ``mol`` (``T = S_aa⁻¹ S_ar T_ref``), then Löwdin-orthonormalized.
    Returns (targets, labels, max projection residual 1 − ‖P t‖²).
    """
    from pyscf import gto

    ref = gto.M(
        atom=mol._atom, unit="Bohr", basis=ref_basis, charge=mol.charge, spin=mol.spin, verbose=0
    )
    t_ref, labels = file_targets(ref, n_cluster, ao_dir)
    s_cross = gto.intor_cross("int1e_ovlp", mol, ref)
    ovlp = mol.intor_symmetric("int1e_ovlp")
    aoslice = mol.aoslice_by_atom()
    t = np.zeros((mol.nao, t_ref.shape[1]))
    for k, lab in enumerate(labels):
        i = int(lab.split("_", 1)[0])
        p0, p1 = aoslice[i, 2], aoslice[i, 3]
        t[p0:p1, k] = linalg.solve(ovlp[p0:p1, p0:p1], s_cross[p0:p1] @ t_ref[:, k], assume_a="pos")
    # 1 - |P t|^2 per target, with t_ref normalized in the reference basis
    residual = float(np.max(1.0 - np.einsum("pk,pq,qk->k", t, ovlp, t)))
    return lowdin_orthonormalize(t, ovlp), labels, residual


def free_atom_targets(mol, n_cluster):
    """Free-atom targets for any basis."""
    elements = _cluster_elements(mol, n_cluster)
    c_atom = lo.orth.pre_orth_ao(mol, method="scf")
    idx, labels = [], []
    for k, ao in enumerate(mol.ao_labels()):
        atom_id, element, orb = ao.split()[:3]
        atom_id = int(atom_id)
        if atom_id < n_cluster and orb.startswith(SHELLS[elements[atom_id]]):
            idx.append(k)
            labels.append(f"{atom_id}_{element}_{orb}")
    order = _block_order(labels)
    c = c_atom[:, idx][:, order]
    return lowdin_orthonormalize(c, mol.intor_symmetric("int1e_ovlp")), [labels[k] for k in order]


def _block_order(labels):
    """Stable sort: Fe atoms (by model index) first, then S (block order of the labels)."""

    def key(k):
        idx, el, _ = labels[k].split("_", 2)
        return (el != "Fe", int(idx))

    return sorted(range(len(labels)), key=key)


def make_targets(mol, n_fe, n_s, source):
    """Targets from a config string: ``file:<dir>`` or ``free-atom``. Checks the count."""
    n_cluster = n_fe + n_s
    if source.startswith("file:"):
        t, labels = file_targets(mol, n_cluster, source[5:])
    elif source == "free-atom":
        t, labels = free_atom_targets(mol, n_cluster)
    else:
        raise ValueError(f"unknown target_aos {source!r} (file:<dir> | free-atom)")
    ncas = 5 * n_fe + 3 * n_s
    if len(labels) != ncas or len(set(labels)) != ncas:
        raise ValueError(f"expected {ncas} unique target orbitals, got {labels}")
    return t, labels
