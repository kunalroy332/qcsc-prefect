"""Gaussian formatted checkpoint (``.fchk``) reader and conversion to a PySCF mol + UHF solution.

Only what the fchk route needs: geometry, charge, multiplicity, the basis (all-electron, every
d/f/g shell pure or every one Cartesian) and the α/β MOs of an unrestricted calculation. ECPs
and restricted calculations are rejected.

AO conventions (Gaussian → PySCF, per shell; the inverse of ``pyscf.tools.molden.order_ao_index``):

- pure d: ``0, +1, −1, +2, −2`` → PySCF ``−2 … +2`` (f, g the same way);
- Cartesian d: ``xx, yy, zz, xy, xz, yz`` → PySCF ``xx, xy, xz, yy, yz, zz``;
  f: ``xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz`` → PySCF lexicographic;
- SP shells (type −1) become an s and a p shell with the same exponents;
- Gaussian normalises every basis function (also each Cartesian component), PySCF does not
  normalise Cartesian components: ``C_pyscf = C_gaussian / sqrt(S_pyscf[i, i])``.

:func:`check_conversion` catches a wrong mapping: the MOs must be orthonormal in the PySCF
overlap and reproduce the fchk ⟨S²⟩ (and, optionally, the total energy).
"""

import math
import re
from dataclasses import dataclass

import numpy as np
from pyscf import gto
from pyscf.data.elements import ELEMENTS
from pyscf.scf import uhf as uhf_mod

from . import scf_engine

# position k of a Gaussian shell -> offset in the PySCF shell (pyscf.tools.molden.order_ao_index)
_GAUSSIAN_TO_PYSCF = {
    (2, False): [2, 3, 1, 4, 0],
    (3, False): [3, 4, 2, 5, 1, 6, 0],
    (4, False): [4, 5, 3, 6, 2, 7, 1, 8, 0],
    (2, True): [0, 3, 5, 1, 2, 4],
    (3, True): [0, 6, 9, 3, 1, 2, 5, 8, 7, 4],
    (4, True): [0, 10, 14, 1, 2, 6, 11, 9, 13, 3, 5, 12, 4, 7, 8],
}

_HEADER = re.compile(r"^(?P<name>\S.*?)\s{2,}(?P<type>[IRCHL])\s+(?P<arr>N=)?\s*(?P<val>\S+)\s*$")
_ARRAY_LINES = {"C": 5, "H": 5, "L": 72}  # values per line of the text arrays


def shell_map(ang, cart):
    """Gaussian component order → PySCF offsets of one shell of angular momentum ``l``."""
    if ang < 2:
        return list(range(2 * ang + 1))  # s; p is x, y, z in both codes
    if (ang, cart) not in _GAUSSIAN_TO_PYSCF:
        raise ValueError(f"shells with l = {ang} are not supported")
    return _GAUSSIAN_TO_PYSCF[ang, cart]


def _number(tok):
    return float(tok.replace("D", "E").replace("d", "e"))


def read_fchk(path):
    """Parse a fchk into ``{"title", "job", "method", "basis", <field name>: value}``.

    Integer / real fields are int / float scalars or NumPy arrays; text fields are skipped.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: not a formatted checkpoint file")
    second = lines[1].split()
    out = {
        "title": lines[0].strip(),
        "job": second[0] if second else None,
        "method": second[1] if len(second) > 1 else None,
        "basis": second[2] if len(second) > 2 else None,
    }
    i = 2
    while i < len(lines):
        m = _HEADER.match(lines[i])
        i += 1
        if m is None:
            continue
        name, typ, val = m["name"].strip(), m["type"], m["val"]
        if not m["arr"]:
            if typ == "I":
                out[name] = int(val)
            elif typ == "R":
                out[name] = _number(val)
            continue
        n = int(val)
        if typ in _ARRAY_LINES:
            i += math.ceil(n / _ARRAY_LINES[typ])
            continue
        vals = []
        while len(vals) < n:
            vals += lines[i].split()
            i += 1
        out[name] = (
            np.array([int(v) for v in vals]) if typ == "I" else np.array([_number(v) for v in vals])
        )
    return out


@dataclass
class Shell:
    atom: int  # fchk atom index
    ang: int
    exps: np.ndarray
    coefs: np.ndarray
    offset: int  # first AO of the shell in the fchk order


def fchk_shells(fc):
    """Shells in fchk AO order (SP split into s + p); returns (shells, cart)."""
    if any(k.startswith("ECP") for k in fc):
        raise ValueError("the fchk uses ECPs (not supported: all-electron bases only)")
    types = np.asarray(fc["Shell types"])
    nprim = np.asarray(fc["Number of primitives per shell"])
    atoms = np.asarray(fc["Shell to atom map"]) - 1
    exps = np.asarray(fc["Primitive exponents"])
    cc = np.asarray(fc["Contraction coefficients"])
    cp = fc.get("P(S=P) Contraction coefficients")
    high = {int(t) > 0 for t in types if abs(int(t)) >= 2}
    if len(high) > 1:
        raise ValueError("the fchk mixes pure and Cartesian d/f shells (not supported by PySCF)")
    cart = high == {True}
    shells, p0, off = [], 0, 0
    for t, n, a in zip(types, nprim, atoms):
        t, n = int(t), int(n)
        e, c = exps[p0 : p0 + n], cc[p0 : p0 + n]
        if t == -1:
            if cp is None:
                raise ValueError("SP shell without 'P(S=P) Contraction coefficients'")
            shells.append(Shell(int(a), 0, e, c, off))
            shells.append(Shell(int(a), 1, e, np.asarray(cp)[p0 : p0 + n], off + 1))
            off += 4
        else:
            ang = abs(t)
            shells.append(Shell(int(a), ang, e, c, off))
            off += (ang + 1) * (ang + 2) // 2 if t > 1 else 2 * ang + 1
        p0 += n
    if off != fc["Number of basis functions"]:
        raise ValueError(
            f"shells give {off} basis functions, the fchk {fc['Number of basis functions']}"
        )
    return shells, cart


def to_pyscf(fc, order=None, max_memory_mb=4000, verbose=0):
    """(mol, Solution) of the fchk, atoms in ``order`` (fchk atom indices; default: fchk order).

    The mol is in Bohr with the fchk basis (one basis per element; atoms of one element with
    different shells are rejected). Occupations are aufbau per spin.
    """
    if "Beta MO coefficients" not in fc:
        raise ValueError("the fchk has no Beta MOs: an unrestricted (U...) calculation is needed")
    z = np.asarray(fc["Atomic numbers"])
    xyz = np.asarray(fc["Current cartesian coordinates"]).reshape(-1, 3)
    order = list(range(len(z))) if order is None else list(order)
    if sorted(order) != list(range(len(z))):
        raise ValueError("order must be a permutation of the fchk atoms")
    shells, cart = fchk_shells(fc)
    by_atom = {}
    for s in shells:
        by_atom.setdefault(s.atom, []).append(s)
    basis = {}
    for a in range(len(z)):
        el = ELEMENTS[int(z[a])]
        key = [(s.ang, tuple(s.exps), tuple(s.coefs)) for s in by_atom.get(a, [])]
        if el not in basis:
            basis[el] = key
        elif not _same_basis(basis[el], key):
            raise ValueError(f"atoms of {el} have different basis sets in the fchk (not supported)")
    mol = gto.M(
        atom=[(ELEMENTS[int(z[a])], tuple(xyz[a])) for a in order],
        unit="Bohr",
        basis={el: [[ang, *zip(e, c)] for ang, e, c in key] for el, key in basis.items()},
        charge=int(fc["Charge"]),
        spin=int(fc["Multiplicity"]) - 1,
        cart=cart,
        symmetry=False,
        max_memory=max_memory_mb,
        verbose=verbose,
    )
    perm = ao_permutation(mol, order, by_atom, cart)
    nbf = int(fc["Number of basis functions"])
    nmo = int(fc.get("Number of independent functions", nbf))
    norm = np.sqrt(mol.intor_symmetric("int1e_ovlp").diagonal())
    mo, energies, occ = [], [], []
    for spin, nel in (
        ("Alpha", fc["Number of alpha electrons"]),
        ("Beta", fc["Number of beta electrons"]),
    ):
        c = np.asarray(fc[f"{spin} MO coefficients"]).reshape(-1, nbf).T  # (nbf, nmo)
        if c.shape[1] != nmo:
            raise ValueError(f"{spin} MO coefficients: {c.shape[1]} MOs, expected {nmo}")
        mo.append(c[perm] / norm[:, None])
        energies.append(np.asarray(fc[f"{spin} Orbital Energies"])[:nmo])
        o = np.zeros(nmo)
        o[: int(nel)] = 1.0
        occ.append(o)
    sol = scf_engine.Solution(
        name="fchk",
        mo_coeff=np.array(mo),
        mo_occ=np.array(occ),
        mo_energy=np.array(energies),
        e_tot=float(fc["Total Energy"]),
        converged=True,
        spin=mol.spin,
    )
    return mol, sol


def _same_basis(a, b):
    return len(a) == len(b) and all(
        la == lb and np.allclose(ea, eb) and np.allclose(ca, cb)
        for (la, ea, ca), (lb, eb, cb) in zip(a, b)
    )


def ao_permutation(mol, order, by_atom, cart):
    """``perm[i]`` = fchk AO index of PySCF AO ``i``.

    PySCF sorts the shells of an atom by l, keeping the input order within one l, so the k-th
    PySCF shell of angular momentum l is the k-th fchk shell of l on that atom.
    """
    ao_loc = mol.ao_loc
    perm = np.empty(mol.nao, dtype=int)
    seen = {}
    for ib in range(mol.nbas):
        if mol.bas_nctr(ib) != 1:
            raise AssertionError("expected one contraction per PySCF shell")
        ang, a = mol.bas_angular(ib), order[mol.bas_atom(ib)]
        k = seen.get((a, ang), 0)
        seen[a, ang] = k + 1
        same_l = [s for s in by_atom[a] if s.ang == ang]
        match = same_l[k] if k < len(same_l) else None
        if match is None or not np.allclose(match.exps, mol.bas_exp(ib)):
            raise AssertionError(f"no fchk shell for PySCF shell {ib} (atom {a}, l {ang})")
        for k, p in enumerate(shell_map(ang, cart and ang >= 2)):
            perm[ao_loc[ib] + p] = match.offset + k
    if sorted(perm) != list(range(mol.nao)):
        raise AssertionError("AO permutation is not a bijection")
    return perm


def check_conversion(
    mol, sol, fc, energy=False, xc=None, auxbasis="def2-universal-jkfit", grid_level=3
):
    """Orthonormality, electron counts, ⟨S²⟩ (and with ``energy`` the total energy) of the
    converted solution against the fchk. Returns a dict; the caller applies the tolerances."""
    ovlp = mol.intor_symmetric("int1e_ovlp")
    orth = max(float(np.abs(c.T @ ovlp @ c - np.eye(c.shape[1])).max()) for c in sol.mo_coeff)
    occ = [sol.mo_coeff[s][:, sol.mo_occ[s] > 0] for s in range(2)]
    s2 = float(uhf_mod.spin_square(occ, ovlp)[0])
    out = {
        "orthonormality_error": orth,
        "nelec": [int(fc["Number of alpha electrons"]), int(fc["Number of beta electrons"])],
        "nelec_mol": list(mol.nelec),
        "s2": s2,
        "s2_fchk": fc.get("S**2"),
        "e_fchk": float(fc["Total Energy"]),
        "e_pyscf": None,
        "xc": None,
    }
    if energy:
        from pyscf import dft

        xc = xc or xc_of(fc["method"])
        mf = dft.UKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
        mf.grids.level = grid_level
        out["e_pyscf"] = float(mf.energy_tot(dm=sol.make_rdm1()))
        out["xc"] = xc
    return out


def xc_of(method):
    """PySCF xc of a Gaussian method keyword (``UB3PW91`` → ``b3pw91``)."""
    if not method:
        raise ValueError("no method on line 2 of the fchk; pass xc explicitly")
    m = method.upper()
    if m.startswith(("U", "R")) and len(m) > 1 and m not in ("UHF", "RHF"):
        m = m[1:]
    if m in ("UHF", "HF"):
        return "hf"
    return m.lower()
