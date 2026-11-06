"""Active-space Hamiltonian (exact or DF integrals), active-space RDM and FCIDUMP writer."""

import numpy as np
from pyscf import mcscf, scf, tools

EXACT_ERI_MAX_NAO = 600


def resolve_eri(eri, nao):
    if eri == "auto":
        return "exact" if nao <= EXACT_ERI_MAX_NAO else "df"
    if eri not in ("exact", "df"):
        raise ValueError(f"unknown eri {eri!r} (exact | df | auto)")
    return eri


def active_hamiltonian(
    mol, mo_coeff, ncore, ncas, nelecas, eri="exact", auxbasis="def2-universal-jkfit"
):
    """(h1eff, h2eff, ecore) with a doubly occupied (restricted) core.

    ``exact``: AO→MO transformation of the exact ERIs. ``df``: 3-index DF integrals (also
    used for the core Fock matrix), for large AO bases.
    """
    if eri == "df":
        mc = mcscf.DFCASCI(scf.RHF(mol), ncas, nelecas, auxbasis=auxbasis, ncore=ncore)
    else:
        mc = mcscf.CASCI(scf.RHF(mol), ncas, nelecas, ncore=ncore)
    mc.mo_coeff = mo_coeff
    h1, ecore = mc.get_h1eff()
    h2 = mc.get_h2eff()
    return h1, h2, float(ecore)


def active_rdm(c_act, dm_ao, ovlp):
    """α, β 1-RDM of a full-system UHF in the (orthonormal) active-orbital basis."""
    return np.array([c_act.T @ ovlp @ d @ ovlp @ c_act for d in dm_ao])


def write_fcidump(path, h1, h2, ncas, nelecas, ecore):
    ms2 = nelecas[0] - nelecas[1]
    tools.fcidump.from_integrals(str(path), h1, h2, ncas, sum(nelecas), nuc=ecore, ms=ms2)
