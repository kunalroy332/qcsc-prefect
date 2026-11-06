"""UNO-AVAS: AVAS-style active-space selection from a (broken-symmetry) UHF.

1. UHF natural orbitals (UNOs) of the spin-summed density, occupations n in [0, 2].
2. Doubly occupied (n > 2-δ), fractional (δ <= n <= 2-δ), virtual (n < δ).
3. Every fractional UNO is active (UHF counterpart of AVAS ``openshell_option=3``); each must
   overlap the target AOs by at least ``min_frac_weight``.
4. The remaining ``ncas - n_frac`` slots: per group (doubly occ / virtual) SVD of the overlap
   with the targets; the largest σ² of both groups are taken (pooled).
5. Orbitals ``[core | active | virtual]``, complements from the same SVDs.
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass
class ActiveSpace:
    mo_coeff: np.ndarray  # (nao, nmo): core | active | virtual
    ncore: int
    ncas: int
    nelecas: tuple  # (n_alpha, n_beta) in the active space
    noons: np.ndarray  # UNO occupations, descending
    n_frac: int
    frac_weights: np.ndarray
    n_from_occ: int
    n_from_vir: int
    last_kept: float
    first_rejected: float


def uno(mo_coeff_a, dm, ovlp):
    """UNOs and occupations (descending) of the spin-summed AO density ``dm`` = Da + Db.

    Uses the α MOs (a complete orthonormal set) as the basis, as the reference scaffold.
    """
    c = mo_coeff_a
    p = c.T @ ovlp @ dm @ ovlp @ c
    n, u = linalg.eigh(p)
    order = np.argsort(n)[::-1]
    return c @ u[:, order], n[order]


def _svd_split(group, targets, ovlp):
    """Rotate ``group`` so its first columns overlap ``targets`` most (σ descending)."""
    if group.shape[1] == 0:
        return group, np.zeros(0)
    u, sigma, _ = linalg.svd(group.T @ ovlp @ targets)
    rotated = group @ u
    full = np.zeros(group.shape[1])
    full[: len(sigma)] = sigma
    return rotated, full


def select_active_space(
    mo_coeff_a,
    dm_a,
    dm_b,
    ovlp,
    targets,
    nalpha,
    nbeta,
    noon_threshold=0.2,
    min_frac_weight=0.5,
    allow_ligand_frac=False,
):
    ncas = targets.shape[1]
    c_uno, n = uno(mo_coeff_a, dm_a + dm_b, ovlp)
    d = noon_threshold
    i_occ = np.nonzero(n > 2 - d)[0]
    i_frac = np.nonzero((n >= d) & (n <= 2 - d))[0]
    i_vir = np.nonzero(n < d)[0]
    n_frac = len(i_frac)
    c_frac = c_uno[:, i_frac]
    frac_w = np.einsum("pi,pi->i", targets.T @ ovlp @ c_frac, targets.T @ ovlp @ c_frac)
    print(
        f"  UNO-AVAS: {len(i_occ)} doubly occ, {n_frac} fractional, {len(i_vir)} virtual "
        f"(δ = {d}); fractional NOONs " + " ".join(f"{x:.3f}" for x in n[i_frac])
    )
    low = np.nonzero(frac_w < min_frac_weight)[0]
    if len(low):
        msg = (
            f"fractional UNOs with target weight < {min_frac_weight}: "
            + ", ".join(f"n={n[i_frac[k]]:.3f} w={frac_w[k]:.3f}" for k in low)
            + " (ligand radical, wrong charge or bad protonation?)"
        )
        if not allow_ligand_frac:
            raise RuntimeError(msg)
        print("  WARNING " + msg)
    k = ncas - n_frac
    if k < 0:
        raise RuntimeError(
            f"{n_frac} fractional UNOs > ncas = {ncas}; raise noon_threshold or "
            "enlarge the active space"
        )

    # targets with the fractional part projected out, so the pooled SVD picks the complement
    t = targets - c_frac @ (c_frac.T @ ovlp @ targets)
    occ_rot, s_occ = _svd_split(c_uno[:, i_occ], t, ovlp)
    vir_rot, s_vir = _svd_split(c_uno[:, i_vir], t, ovlp)
    pool = sorted(
        [(s**2, "o", j) for j, s in enumerate(s_occ)]
        + [(s**2, "v", j) for j, s in enumerate(s_vir)],
        reverse=True,
    )
    kept, rejected = pool[:k], pool[k:]
    n_o = sum(1 for _, g, _ in kept if g == "o")
    n_v = k - n_o
    # the SVD orders columns by σ, so the kept ones of each group are the first n_o / n_v
    if any(j >= n_o for _, g, j in kept if g == "o") or any(
        j >= n_v for _, g, j in kept if g == "v"
    ):
        raise AssertionError("pooled selection is not a prefix of the per-group SVD order")
    last = kept[-1][0] if kept else float("nan")
    first_rej = rejected[0][0] if rejected else float("nan")
    print(
        f"  UNO-AVAS: {n_o} from doubly occ + {n_frac} fractional + {n_v} from virtual = "
        f"{ncas}; last kept σ² {last:.3f}, first rejected σ² {first_rej:.3f}"
    )

    core = occ_rot[:, n_o:]
    act = np.hstack([occ_rot[:, :n_o], c_frac, vir_rot[:, :n_v]])
    vir = vir_rot[:, n_v:]
    mo = np.hstack([core, act, vir])
    err = np.abs(mo.T @ ovlp @ mo - np.eye(mo.shape[1])).max()
    if err > 1e-6:
        raise RuntimeError(f"UNO-AVAS orbitals not orthonormal (max error {err:.1e})")
    ncore = core.shape[1]
    nelecas = (nalpha - ncore, nbeta - ncore)
    if not (0 <= nelecas[1] <= nelecas[0] <= ncas or 0 <= nelecas[0] <= nelecas[1] <= ncas):
        raise RuntimeError(f"active electrons {nelecas} out of range for ncas = {ncas}")
    return ActiveSpace(mo, ncore, ncas, nelecas, n, n_frac, frac_w, n_o, n_v, last, first_rej)
