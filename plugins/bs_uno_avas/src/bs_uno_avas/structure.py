"""PDB reading, cluster detection, reordering, protonation check/repair, charge and spin.

The Fe-S cluster is found by residue name only. ``FES`` (Fe2S2) and ``SF4`` (Fe4S4), the names of
the published 5XTD structure, are detected automatically; any other name is given as
``cluster_resname`` (config ``[structure] cluster_resname``, CLI ``--cluster-resname``). The
cluster type follows from the Fe count (2 or 4, with as many S). One cluster may be split over
several residue numbers (e.g. one atom per residue). Anything else is an error.

Model index convention (used by the orbital labels): Fe atoms ``0..n_fe-1`` (PDB serial order),
then bridging S ``n_fe..n_fe+n_s-1`` (serial order), then all other atoms in file order.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Built-in (auto-detected) cluster residue names -> Fe count; others via cluster_resname
CLUSTER_RESNAMES = {"FES": 2, "SF4": 4}
CLUSTER_TYPES = {2: "Fe2S2", 4: "Fe4S4"}

# Redox presets: number of Fe(III) per cluster type and state name. Fe4S4: ox = [Fe4S4]2+,
# red = [Fe4S4]+ (ferredoxin-type, as in complex I); HiPIP 3+ only via "3+".
REDOX_PRESETS = {
    "Fe2S2": {"ox": 2, "red": 1},
    "Fe4S4": {"ox": 2, "red": 1, "3+": 3, "2+": 2, "1+": 1},
}


def nominal_active_electrons(n_fe, n_fe3, n_s):
    """Electrons in Fe 3d + bridging S 3p: d5 per Fe(III), d6 per Fe(II), 3p6 per S2-."""
    return 5 * n_fe3 + 6 * (n_fe - n_fe3) + 6 * n_s


COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "Se": 1.20}
BOND_SCALE = 1.25  # bonded if d < BOND_SCALE * (r_i + r_j); Fe is excluded from the bond graph
VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "Se": 2}
X_H_LENGTH = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34}
TETRAHEDRAL = np.deg2rad(109.47)

# Conjugated side-chain atoms: bond orders are not resolved from distances, so ring/guanidinium
# C count 3 sigma bonds and N are not valence-checked.
_RING6 = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
CONJUGATED = {
    "PHE": _RING6,
    "TYR": _RING6,
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "ARG": ("NE", "CZ", "NH1", "NH2"),
}

# Residues --protonate builtin can handle with the distance-based valence rule (no aromatic
# rings, no guanidinium / imidazole: their bond orders are ambiguous from distances alone).
BUILTIN_RESIDUES = {
    "GLY",
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "PRO",
    "SER",
    "THR",
    "CYS",
    "MET",
    "ASN",
    "GLN",
    "ASP",
    "GLU",
    "LYS",
}


@dataclass
class PdbAtom:
    record: str
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    element: str
    xyz: np.ndarray
    added: bool = False  # hydrogen added by --protonate builtin

    @property
    def label(self):
        return f"{self.resname}{self.resseq}:{self.name}"

    @property
    def residue(self):
        return (self.chain, self.resseq, self.resname)


@dataclass
class Model:
    """Reordered model of a PDB excerpt (cluster atoms first)."""

    pdb_file: str
    pdb_sha256: str
    atoms: list  # PdbAtom, model order
    n_fe: int
    n_s: int
    cluster_resname: str
    thiolates: list  # labels of Fe-bound Cys SG
    first_shell: dict  # "0_Fe" -> [(label, element, distance)]
    residue_charges: dict  # "CYS135" -> charge (only nonzero)
    protonation: dict
    warnings: list = field(default_factory=list)

    @property
    def cluster_type(self):
        return CLUSTER_TYPES[self.n_fe]

    @property
    def ligand_charge(self):
        return sum(self.residue_charges.values())

    def pyscf_atoms(self):
        return [(a.element, tuple(float(x) for x in a.xyz)) for a in self.atoms]

    def cluster_atoms(self):
        return [
            {"model_idx": i, "element": a.element, "pdb_name": a.label}
            for i, a in enumerate(self.atoms[: self.n_fe + self.n_s])
        ]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _element(line, name):
    element = line[76:78].strip().capitalize() if len(line) >= 78 else ""
    if name.upper().startswith("FE") and element in ("", "F", "X", "Fe"):
        return "Fe"
    if element in ("", "X"):
        letters = "".join(c for c in name if c.isalpha())
        if not letters:
            raise ValueError(f"cannot determine the element of atom {name!r}")
        element = letters[0].upper()
    return element


def read_pdb_atoms(pdb_file):
    """ATOM/HETATM records of the first model; alternate locations other than A are skipped."""
    atoms = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[16] not in (" ", "A"):
                continue
            name = line[12:16].strip()
            atoms.append(
                PdbAtom(
                    record=line[:6].strip(),
                    serial=int(line[6:11]),
                    name=name,
                    resname=line[17:20].strip(),
                    chain=line[21],
                    resseq=int(line[22:26]),
                    element=_element(line, name),
                    xyz=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                )
            )
    if not atoms:
        raise ValueError(f"no ATOM/HETATM records in {pdb_file}")
    return atoms


def find_cluster(atoms, pdb_file="<pdb>", resname=None):
    """Split atoms into (resname, Fe, bridging S, rest) by cluster residue name.

    ``resname`` given: that residue is the cluster. Else one of :data:`CLUSTER_RESNAMES` is
    detected. The cluster must hold only Fe and S, 2 or 4 Fe, as many S as Fe.
    """
    hint = (
        "Set the cluster residue name: [structure] cluster_resname, --cluster-resname, or "
        "CLUSTER_RESNAME= for roquo/run_pdb.sh"
    )
    with_fe = sorted({a.resname for a in atoms if a.element == "Fe"})
    if resname:
        if not any(a.resname == resname for a in atoms):
            raise ValueError(
                f"no residue {resname} in {pdb_file}; residues holding Fe: {with_fe or 'none'}"
            )
    else:
        names = sorted({a.resname for a in atoms if a.resname in CLUSTER_RESNAMES})
        if not names:
            raise ValueError(
                f"no Fe-S cluster residue found in {pdb_file}: {', '.join(CLUSTER_RESNAMES)} are "
                f"detected automatically; residues holding Fe: {with_fe or 'none'}. {hint}"
            )
        if len(names) > 1:
            raise ValueError(f"more than one cluster residue name in {pdb_file}: {names}. {hint}")
        resname = names[0]
    cluster = [a for a in atoms if a.resname == resname]
    other = sorted({a.element for a in cluster} - {"Fe", "S"})
    if other:
        raise ValueError(f"cluster residue {resname} contains elements {other} (only Fe, S)")
    fe = sorted((a for a in cluster if a.element == "Fe"), key=lambda a: a.serial)
    s = sorted((a for a in cluster if a.element == "S"), key=lambda a: a.serial)
    expected = CLUSTER_RESNAMES.get(resname, len(fe))
    if expected not in CLUSTER_TYPES or len(fe) != expected or len(s) != expected:
        raise ValueError(
            f"cluster residue {resname} in {pdb_file} must have "
            + (
                f"{expected} Fe and {expected} S"
                if resname in CLUSTER_RESNAMES
                else "2 Fe and 2 S (Fe2S2) or 4 Fe and 4 S (Fe4S4)"
            )
            + f", found {len(fe)} Fe and {len(s)} S"
        )
    rest = [a for a in atoms if a.resname != resname]
    return resname, fe, s, rest


def bond_graph(atoms):
    """Distance-based covalent bonds between non-Fe atoms: list of neighbour index lists."""
    xyz = np.array([a.xyz for a in atoms])
    radii = np.array([COVALENT_RADII.get(a.element, 0.0) for a in atoms])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    bonded = (
        (d < BOND_SCALE * (radii[:, None] + radii[None])) & (radii[:, None] > 0) & (radii[None] > 0)
    )
    np.fill_diagonal(bonded, False)
    return [list(np.nonzero(row)[0]) for row in bonded], d


def _double_bond_partners(i, atoms, nbrs, d):
    """Neighbours of atom i joined by a double bond (C=O < 1.30 A; terminal O on C only)."""
    a = atoms[i]
    out = []
    for j in nbrs[i]:
        b = atoms[j]
        pair = {a.element, b.element}
        if pair == {"C", "O"} and d[i, j] < 1.30:
            o = i if a.element == "O" else j
            if sum(atoms[k].element != "H" for k in nbrs[o]) == 1:  # terminal O
                out.append(j)
    return out


def _thiolates(atoms, fe_idx, sg_cutoff):
    """Indices of Cys SG within sg_cutoff of any cluster Fe."""
    out = []
    for i, a in enumerate(atoms):
        if a.resname == "CYS" and a.name == "SG":
            if min(np.linalg.norm(a.xyz - atoms[f].xyz) for f in fe_idx) < sg_cutoff:
                out.append(i)
    return out


def _expected_valence(i, atoms, nbrs, d, thiolate_idx):
    """Number of sigma bonds + C=O double bonds the atom should have (None = not checked)."""
    a = atoms[i]
    if i in thiolate_idx:
        return 1
    if a.resname == "LYS" and a.name == "NZ":
        return 4
    if a.name in CONJUGATED.get(a.resname, ()):
        return 3 if a.element == "C" else None  # ring / guanidinium C: 3 sigma bonds
    if a.resname in ("ASP", "GLU") and a.name in ("OD1", "OD2", "OE1", "OE2"):
        return 1  # carboxylate at pH 7
    if a.name == "OXT":
        return 1
    return VALENCE.get(a.element)


def _used_valence(i, atoms, nbrs, d):
    return len(nbrs[i]) + len(_double_bond_partners(i, atoms, nbrs, d))


def _perpendicular(v):
    trial = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p = np.cross(v, trial)
    return p / np.linalg.norm(p)


def _place_hydrogens(i, n_add, atoms, nbrs, d):
    """Positions of n_add H on atom i (tetrahedral, or trigonal for sp2 / amide N)."""
    a = atoms[i]
    bonds = [atoms[j].xyz - a.xyz for j in nbrs[i]]
    bonds = [b / np.linalg.norm(b) for b in bonds]
    if not bonds:
        raise ValueError(f"cannot protonate isolated atom {a.label}")
    length = X_H_LENGTH.get(a.element, 1.0)
    sp2 = bool(_double_bond_partners(i, atoms, nbrs, d)) or (
        a.element == "N"
        and any(
            atoms[j].element == "C" and _double_bond_partners(j, atoms, nbrs, d) for j in nbrs[i]
        )
    )
    n = len(bonds)
    if n + n_add > 4 or (sp2 and n + n_add > 3):
        raise ValueError(f"cannot place {n_add} H on {a.label} with {n} bonds")
    if n_add == 1 and n >= 2:
        v = -np.sum(bonds, axis=0)
        dirs = [v / np.linalg.norm(v)]
    elif n == 2 and n_add == 2:  # tetrahedral CH2 / NH2+
        bis = -(bonds[0] + bonds[1])
        bis /= np.linalg.norm(bis)
        perp = np.cross(bonds[0], bonds[1])
        perp /= np.linalg.norm(perp)
        half = TETRAHEDRAL / 2
        dirs = [np.cos(half) * bis + s * np.sin(half) * perp for s in (1, -1)]
    elif n == 1:
        b = bonds[0]
        # reference direction from a neighbour of the neighbour, so H is staggered / in-plane
        j = nbrs[i][0]
        ref = next((atoms[k].xyz - atoms[j].xyz for k in nbrs[j] if k != i), None)
        u = ref - np.dot(ref, b) * b if ref is not None else None
        u = (
            u / np.linalg.norm(u)
            if u is not None and np.linalg.norm(u) > 1e-6
            else _perpendicular(b)
        )
        w = np.cross(b, u)
        if sp2:
            angle, phis = np.deg2rad(120.0), [0.0, np.pi]
        else:
            angle, phis = TETRAHEDRAL, [np.pi / 3, np.pi, -np.pi / 3]
        dirs = [
            np.cos(angle) * b + np.sin(angle) * (np.cos(p) * u + np.sin(p) * w)
            for p in phis[:n_add]
        ]
    else:
        raise ValueError(f"cannot place {n_add} H on {a.label} with {n} bonds")
    return [a.xyz + length * v for v in dirs]


def protonate_builtin(atoms, fe_idx, sg_cutoff, cluster):
    """Add H to every under-valent heavy atom (pH 7 rules, cut points capped neutral).

    Returns (new atom list, report lines). Only residues in BUILTIN_RESIDUES are supported;
    ``cluster`` is the cluster residue name.
    """
    unsupported = sorted(
        {a.resname for a in atoms if a.resname not in BUILTIN_RESIDUES and a.resname != cluster}
    )
    if unsupported:
        raise ValueError(
            f"--protonate builtin does not support residues {unsupported}; "
            "supply a protonated excerpt"
        )
    return _add_hydrogens(atoms, fe_idx, sg_cutoff, cluster)


def _bare_cys_sg(atoms, nbrs):
    """Cys SG without H (a thiolate in an excerpt that has H, even beyond ``sg_cutoff``)."""
    return {
        i
        for i, a in enumerate(atoms)
        if a.resname == "CYS" and a.name == "SG" and not _h_on(i, atoms, nbrs)
    }


def _add_hydrogens(atoms, fe_idx, sg_cutoff, cluster, keep_bare_sg=False):
    """H on every under-valent heavy atom outside the cluster residue ``cluster``; returns
    (atoms, report).

    ``keep_bare_sg``: Cys SG without H stay deprotonated (thiolates), also beyond ``sg_cutoff``.
    """
    nbrs, d = bond_graph(atoms)
    thiolate_idx = set(_thiolates(atoms, fe_idx, sg_cutoff))
    if keep_bare_sg:
        thiolate_idx |= _bare_cys_sg(atoms, nbrs)
    added, report = {}, []
    for i, a in enumerate(atoms):
        if a.element in ("H", "Fe") or a.resname == cluster:
            continue
        expected = _expected_valence(i, atoms, nbrs, d, thiolate_idx)
        if expected is None:
            continue
        n_add = max(0, expected - _used_valence(i, atoms, nbrs, d))
        if n_add:
            added[i] = _place_hydrogens(i, n_add, atoms, nbrs, d)
            report.append(f"added {n_add} H on {a.label}")
    out, serial = [], 1
    for i, a in enumerate(atoms):
        out.append(
            PdbAtom(
                a.record,
                serial,
                a.name,
                a.resname,
                a.chain,
                a.resseq,
                a.element,
                a.xyz.copy(),
                a.added,
            )
        )
        serial += 1
        new = added.get(i, [])
        for k, xyz in enumerate(new):
            name = f"H{a.name}" + (str(k + 1) if len(new) > 1 else "")
            out.append(
                PdbAtom(
                    a.record, serial, name[:4], a.resname, a.chain, a.resseq, "H", xyz, added=True
                )
            )
            serial += 1
    return out, report


def write_pdb(atoms, path, remarks=()):
    with open(path, "w") as f:
        for r in remarks:
            f.write(f"REMARK   1 {r}\n")
        for a in atoms:
            name = a.name if len(a.name) == 4 else f" {a.name:<3s}"
            f.write(
                f"{a.record:<6s}{a.serial:5d} {name:4s} {a.resname:>3s} {a.chain}"
                f"{a.resseq:4d}    {a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}"
                f"  1.00  0.00          {a.element.upper():>2s}\n"
            )
        f.write("END\n")


def _h_on(i, atoms, nbrs):
    return [j for j in nbrs[i] if atoms[j].element == "H"]


def residue_charges(atoms, thiolate_idx, his="HIE"):
    """Formal charges per residue at pH 7 from residue names and the H actually present.

    Rules: Asp/Glu carboxylate -1 unless an O carries H; Lys +1 unless NZ has <= 2 H; Arg +1;
    His +1 if both ring N carry H (or ``his="HIP"`` in a structure without H); Cys SG without H
    -1 (Fe-bound thiolate); Tyr OH without H -1; C-terminal OXT -1; N-terminal NH3+ +1.
    """
    nbrs, _ = bond_graph(atoms)
    has_h = any(a.element == "H" for a in atoms)
    by_res = {}
    for i, a in enumerate(atoms):
        by_res.setdefault(a.residue, {})[a.name] = i
    charges = {}
    for (chain, resseq, resname), idx in by_res.items():
        q = 0

        def no_h(*names):
            return all(n in idx and not _h_on(idx[n], atoms, nbrs) for n in names)

        if resname == "ASP" and {"OD1", "OD2"} <= idx.keys() and no_h("OD1", "OD2"):
            q -= 1
        if resname == "GLU" and {"OE1", "OE2"} <= idx.keys() and no_h("OE1", "OE2"):
            q -= 1
        if (
            resname == "LYS"
            and "NZ" in idx
            and (not has_h or len(_h_on(idx["NZ"], atoms, nbrs)) >= 3)
        ):
            q += 1
        if resname == "ARG" and {"NE", "NH1", "NH2", "CZ"} <= idx.keys():
            q += 1
        if resname == "HIS" and {"ND1", "NE2"} <= idx.keys():
            if has_h:
                q += int(
                    bool(_h_on(idx["ND1"], atoms, nbrs)) and bool(_h_on(idx["NE2"], atoms, nbrs))
                )
            else:
                q += int(his == "HIP")
        if (
            resname == "CYS"
            and "SG" in idx
            and (idx["SG"] in thiolate_idx or (has_h and no_h("SG")))
        ):
            q -= 1
        if resname == "TYR" and "OH" in idx and has_h and no_h("OH"):
            q -= 1
        if "OXT" in idx and (not has_h or no_h("OXT", "O")):
            q -= 1
        if "N" in idx and has_h and len(_h_on(idx["N"], atoms, nbrs)) >= 3:
            q += 1
        if q:
            charges[f"{resname}{resseq}"] = q
    return charges


def valence_report(atoms, thiolate_idx, cluster):
    """Heavy atoms outside the cluster residue with fewer bonds than their valence (dangling
    bonds / missing H)."""
    nbrs, d = bond_graph(atoms)
    out = []
    for i, a in enumerate(atoms):
        if a.element in ("H", "Fe") or a.resname == cluster:
            continue
        expected = _expected_valence(i, atoms, nbrs, d, thiolate_idx)
        used = _used_valence(i, atoms, nbrs, d)
        if expected is not None and used < expected:
            out.append(f"{a.label} has {used} of {expected} bonds")
    return out


def prepare_model(
    pdb_file,
    protonate="check",
    sg_cutoff=3.3,
    his="HIE",
    cluster_only=False,
    shell_cutoff=3.3,
    cluster_resname=None,
):
    """Read a PDB excerpt and return the reordered :class:`Model` (see module docstring).

    ``protonate``: ``check`` (error if the excerpt has no H; thiolate H are removed),
    ``builtin`` (add missing H if the check finds none, write ``<stem>_prot.pdb``),
    ``cap`` (an excerpt with H: add H only to under-valent atoms, i.e. the cut points of the
    excerpt, write ``<stem>_cap.pdb``), ``off``.
    ``cluster_only`` keeps only the cluster atoms (bare Fe2S2 / Fe4S4 test models).
    ``cluster_resname``: the cluster residue name (default: detect ``FES`` / ``SF4``).
    """
    pdb_file = str(pdb_file)
    atoms = read_pdb_atoms(pdb_file)
    resname, fe, s, rest = find_cluster(atoms, pdb_file, cluster_resname)
    if cluster_only:
        rest = []
    atoms = fe + s + rest
    n_fe, n_s = len(fe), len(s)
    fe_idx = list(range(n_fe))
    warnings, prot = [], {"tool": "none", "added_h": 0, "removed_h": [], "notes": ""}

    n_h = sum(a.element == "H" for a in atoms)
    if not cluster_only and protonate == "cap" and n_h:
        # cut points of a protonated excerpt: a dangling bond is a radical in the SCF. A Cys SG
        # without H is a thiolate (-1), also when it is beyond sg_cutoff: never protonated here.
        atoms, report = _add_hydrogens(atoms, fe_idx, sg_cutoff, resname, keep_bare_sg=True)
        n_cap = sum(a.added for a in atoms)
        if n_cap:
            out = Path(pdb_file).with_name(Path(pdb_file).stem + "_cap.pdb")
            write_pdb(
                atoms, out, remarks=[f"cut points capped by bs_uno_avas from {Path(pdb_file).name}"]
            )
            prot = {
                "tool": "cap",
                "added_h": n_cap,
                "removed_h": [],
                "notes": f"wrote {out.name}; " + "; ".join(report),
            }
    elif not cluster_only and protonate != "off" and n_h == 0:
        if protonate != "builtin":
            raise ValueError(
                f"{pdb_file} contains no hydrogens: supply a protonated excerpt or "
                "use protonate = 'builtin'"
            )
        atoms, report = protonate_builtin(atoms, fe_idx, sg_cutoff, resname)
        out = Path(pdb_file).with_name(Path(pdb_file).stem + "_prot.pdb")
        write_pdb(
            atoms, out, remarks=[f"protonated by bs_uno_avas builtin from {Path(pdb_file).name}"]
        )
        prot = {
            "tool": "builtin",
            "added_h": sum(a.added for a in atoms),
            "removed_h": [],
            "notes": f"wrote {out.name}; " + "; ".join(report),
        }

    thiolate_idx = _thiolates(atoms, fe_idx, sg_cutoff)
    # thiolates must stay deprotonated
    nbrs, _ = bond_graph(atoms)
    drop = {j for i in thiolate_idx for j in _h_on(i, atoms, nbrs)}
    if drop:
        prot["removed_h"] = [atoms[j].label for j in sorted(drop)]
        warnings.append(f"removed H from Fe-bound SG: {prot['removed_h']}")
        keep = [a for j, a in enumerate(atoms) if j not in drop]
        thiolate_names = {atoms[i].label for i in thiolate_idx}
        atoms = keep
        thiolate_idx = [i for i, a in enumerate(atoms) if a.label in thiolate_names]

    if not cluster_only and protonate != "off":
        expect_thiolate = set(thiolate_idx)
        if protonate == "cap":
            far = _bare_cys_sg(atoms, bond_graph(atoms)[0]) - expect_thiolate
            if far:
                warnings.append(
                    f"Cys SG without H beyond sg_cutoff {sg_cutoff} A of every Fe, kept as "
                    f"thiolate (-1): {[atoms[i].label for i in sorted(far)]}"
                )
            expect_thiolate |= far
        warnings += [f"valence: {w}" for w in valence_report(atoms, expect_thiolate, resname)]

    first_shell = {}
    for f in fe_idx:
        rows = []
        for j, a in enumerate(atoms):
            dist = float(np.linalg.norm(a.xyz - atoms[f].xyz))
            if j != f and dist < shell_cutoff:
                rows.append((a.label, a.element, round(dist, 3)))
        first_shell[f"{f}_Fe"] = sorted(rows, key=lambda r: r[2])
        if any(el == "O" for _, el, _ in rows):
            warnings.append(
                f"{f}_Fe has O within {shell_cutoff} A: "
                + ", ".join(f"{n} {r:.2f}" for n, el, r in rows if el == "O")
            )

    return Model(
        pdb_file=pdb_file,
        pdb_sha256=file_sha256(pdb_file),
        atoms=atoms,
        n_fe=n_fe,
        n_s=n_s,
        cluster_resname=resname,
        thiolates=[atoms[i].label for i in thiolate_idx],
        first_shell=first_shell,
        residue_charges={} if cluster_only else residue_charges(atoms, set(thiolate_idx), his),
        protonation=prot,
        warnings=warnings,
    )


def charge_and_spin(model, n_fe3=None, state=None, charge=None, spin=None):
    """Total charge, 2S of the high-spin state and electron count for a redox state.

    ``q = 3 n_Fe(III) + 2 n_Fe(II) - 2 n_S + sum(residue charges)``,
    ``2S_HS = 5 n_Fe(III) + 4 n_Fe(II)``. ``charge`` / ``spin`` override the rules.
    Returns ``dict(charge, spin, nelec, n_fe3, overrides)``.
    """
    if n_fe3 is None:
        if state is None:
            raise ValueError("give n_fe3 or state")
        try:
            n_fe3 = REDOX_PRESETS[model.cluster_type][state]
        except KeyError as err:
            raise ValueError(
                f"unknown redox state {state!r} for {model.cluster_type}; "
                f"use n_fe3 or one of {list(REDOX_PRESETS[model.cluster_type])}"
            ) from err
    if not 0 <= n_fe3 <= model.n_fe:
        raise ValueError(f"n_fe3 = {n_fe3} out of range for {model.n_fe} Fe")
    n_fe2 = model.n_fe - n_fe3
    q = 3 * n_fe3 + 2 * n_fe2 - 2 * model.n_s + model.ligand_charge
    s2 = 5 * n_fe3 + 4 * n_fe2
    overrides = {}
    if charge is not None and charge != q:
        overrides["charge"] = (q, charge)
        q = charge
    if spin is not None and spin != s2:
        overrides["spin"] = (s2, spin)
        s2 = spin
    from pyscf.data.elements import charge as z_of

    nelec = sum(z_of(a.element) for a in model.atoms) - q
    if (nelec + s2) % 2:
        raise ValueError(
            f"electron count {nelec} and 2S = {s2} have different parity "
            f"(charge {q}); check the protonation / redox state"
        )
    return {"charge": q, "spin": s2, "nelec": nelec, "n_fe3": n_fe3, "overrides": overrides}


def report(model, states=None):
    """Human-readable prep report (no coordinates)."""
    lines = [
        f"PDB {model.pdb_file} (sha256 {model.pdb_sha256[:12]}...)",
        f"cluster {model.cluster_resname}: {model.cluster_type}, {model.n_fe} Fe + "
        f"{model.n_s} bridging S",
        "cluster atoms: "
        + ", ".join(f"{c['model_idx']}={c['pdb_name']}" for c in model.cluster_atoms()),
        f"atoms {len(model.atoms)}, H {sum(a.element == 'H' for a in model.atoms)}; "
        f"protonation {model.protonation['tool']}"
        + (f" ({model.protonation['notes']})" if model.protonation["notes"] else ""),
        f"Fe-bound thiolates ({len(model.thiolates)}): {', '.join(model.thiolates)}",
        f"residue charges: {model.residue_charges or 'none'} (sum {model.ligand_charge:+d})",
    ]
    for fe, rows in model.first_shell.items():
        lines.append(f"{fe} first shell: " + ", ".join(f"{n}({el}) {r:.2f}" for n, el, r in rows))
    for name, cs in (states or {}).items():
        lines.append(
            f"state {name}: n_Fe(III) {cs['n_fe3']}, charge {cs['charge']:+d}, "
            f"2S_HS {cs['spin']}, nelec {cs['nelec']}"
            + (f", overrides {cs['overrides']}" if cs["overrides"] else "")
        )
    lines += [f"WARNING {w}" for w in model.warnings]
    return "\n".join(lines)
