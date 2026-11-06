"""Active space from a Gaussian broken-symmetry DFT ``.fchk`` (no PDB, no full-system SCF).

The fchk (e.g. ``<site>_BS_OPT.fchk``: UB3PW91 / DGDZVP, optimized, capped geometry) holds the
converged BS solution at the broken-symmetry M_S; its charge, multiplicity and electron counts are
used as they are. The route is

    fchk ─► PySCF mol + α/β MOs (fchk basis, cluster first) ─► conversion check (orthonormality,
    ⟨S²⟩, optional energy) ─► Fe spins / tag / gate ─► UNO-AVAS (Fe 3d + μS 3p) ─► the shared
    tail of the PDB route (labels, Hamiltonian, verify, artifacts + manifest)

The cluster is found from the geometry: every Fe (2 or 4), and the S within
``cluster.fe_s_cutoff`` of at least two Fe (bridging S). The redox state (``--state`` / n_Fe(III))
only names the run and is cross-checked against the active electron count. Output:
``output/<stem>_E<state>_R<geometry>_S<tag>_BS_OPT/`` (``stem`` = fchk name without ``_BS_OPT``),
same files and manifest schema as the PDB route.
"""

import time
from pathlib import Path

import numpy as np
import tomllib
from pyscf.data import elements

from . import bs_guess, fchk, paths, pipeline, scf_engine, structure, targets, uno_avas
from .structure import file_sha256

SUFFIX = "_BS_OPT"
DEFAULTS = {
    "cluster": {"fe_s_cutoff": 2.6},
    "gate": {"min_fe": 2.5, "max_s": 1.2},
    "check": {"energy": True, "e_tol": 2e-3, "s2_tol": 1e-4, "orth_tol": 1e-6},
    "scf": {
        "auxbasis": "def2-universal-jkfit",
        "xc": None,  # default: from the fchk method (UB3PW91 -> b3pw91)
        "grid_level": 3,
        "threads": 0,
        "max_memory_mb": 16000,
    },
    "avas": {k: v for k, v in pipeline.DEFAULTS["avas"].items()},
    "output": {
        "dir": "output",
        "run_name": "{stem}_E{E}_R{R}_S{S}" + SUFFIX,
    },
}
BOHR = 0.52917721092


def load_config(path=None, overrides=()):
    """Optional TOML (sections of :data:`DEFAULTS`) + ``section.key=value`` overrides. A relative
    ``output.dir`` is relative to the workspace (:func:`paths.workspace`, or ``base_dir`` of the
    file); ``avas.target_aos = "file:AOs"`` is the shipped package data."""
    raw = {}
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    unknown = set(raw) - set(DEFAULTS) - {"base_dir"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    cfg = {k: {**v, **raw.get(k, {})} for k, v in DEFAULTS.items()}
    for sec, defaults in DEFAULTS.items():
        extra = set(raw.get(sec, {})) - set(defaults)
        if extra:
            raise ValueError(f"{path}: unknown keys in [{sec}]: {sorted(extra)}")
    for item in overrides:
        dotted, sep, value = item.partition("=")
        sec, _, key = dotted.strip().partition(".")
        if not sep or sec not in DEFAULTS or key not in DEFAULTS[sec]:
            raise ValueError(f"bad override {item!r}: use section.key=value with a known key")
        cfg[sec][key] = pipeline._parse_value(value.strip())
    base = Path(raw["base_dir"]) if "base_dir" in raw else paths.workspace()
    if not base.is_absolute():
        base = (Path(path).resolve().parent / base).resolve()
    cfg["base_dir"] = base
    return cfg


def find_cluster(z, xyz_bohr, fe_s_cutoff=2.6):
    """(model order, n_fe, n_s, warnings) from the fchk geometry.

    Order: Fe (fchk order), bridging S (fchk order), the rest (fchk order). Warnings list the
    production watch-list items (counts and distances only, no coordinates).
    """
    sym = [elements.ELEMENTS[int(x)] for x in z]
    xyz = np.asarray(xyz_bohr).reshape(-1, 3) * BOHR
    fe = [i for i, s in enumerate(sym) if s == "Fe"]
    if len(fe) not in structure.CLUSTER_TYPES:
        raise ValueError(f"the fchk has {len(fe)} Fe atoms; expected 2 (Fe2S2) or 4 (Fe4S4)")

    def d(i, j):
        return float(np.linalg.norm(xyz[i] - xyz[j]))

    s_all = [i for i, s in enumerate(sym) if s == "S"]
    bridging = [i for i in s_all if sum(d(i, f) < fe_s_cutoff for f in fe) >= 2]
    if len(bridging) != len(fe):
        raise ValueError(
            f"{len(bridging)} S within {fe_s_cutoff} A of two or more Fe, expected {len(fe)} "
            "(raise cluster.fe_s_cutoff for long Fe-S bonds)"
        )
    warnings = []
    expect_mu = {2: 2, 4: 3}[len(fe)]
    terminal = [i for i in s_all if i not in bridging]
    for k, f in enumerate(fe):
        n_mu = sum(d(f, s) < fe_s_cutoff for s in bridging)
        if n_mu != expect_mu:
            warnings.append(f"{k}_Fe bonds {n_mu} bridging S (expected {expect_mu})")
        n_term = sum(d(f, s) < fe_s_cutoff for s in terminal)
        if n_term != 1:
            warnings.append(f"{k}_Fe has {n_term} terminal S within {fe_s_cutoff} A (expected 1)")
        close_o = [d(f, j) for j, s in enumerate(sym) if s == "O" and d(f, j) < 2.5]
        if close_o:
            warnings.append(f"{k}_Fe has O within 2.5 A: " + ", ".join(f"{r:.2f}" for r in close_o))
        for k2, f2 in enumerate(fe[k + 1 :], start=k + 1):
            if d(f, f2) < 2.5:
                warnings.append(f"Fe-Fe {k}_Fe-{k2}_Fe {d(f, f2):.2f} A < 2.5 A (compressed)")
    rest = [i for i in range(len(sym)) if i not in fe and i not in bridging]
    return fe + bridging + rest, len(fe), len(bridging), warnings


def load(fchk_file, cfg, verbose=0):
    """Read + convert + check. Returns a dict with mol, sol (analysed, gated), checks, ...;
    raises ValueError / RuntimeError on any failed check."""
    fchk_file = Path(fchk_file)
    fc = fchk.read_fchk(fchk_file)
    z = np.asarray(fc["Atomic numbers"])
    order, n_fe, n_s, warnings = find_cluster(
        z, fc["Current cartesian coordinates"], cfg["cluster"]["fe_s_cutoff"]
    )
    nelec = int(fc["Number of alpha electrons"]) + int(fc["Number of beta electrons"])
    ms2 = int(fc["Multiplicity"]) - 1
    if ms2 != nelec % 2:
        raise ValueError(
            f"fchk multiplicity {fc['Multiplicity']} is not the broken-symmetry M_S "
            f"(2S+1 = {nelec % 2 + 1} for {nelec} electrons); a high-spin fchk is not supported"
        )
    sc = cfg["scf"]
    mol, sol = fchk.to_pyscf(fc, order, sc["max_memory_mb"], verbose)
    ck = cfg["check"]
    checks = fchk.check_conversion(
        mol, sol, fc, ck["energy"], sc["xc"], sc["auxbasis"], sc["grid_level"]
    )
    problems = []
    if checks["orthonormality_error"] > ck["orth_tol"]:
        problems.append(
            f"MOs not orthonormal in the PySCF basis ({checks['orthonormality_error']:.1e})"
        )
    if checks["nelec"] != checks["nelec_mol"]:
        problems.append(f"electrons {checks['nelec']} != PySCF mol {checks['nelec_mol']}")
    if checks["s2_fchk"] is not None and abs(checks["s2"] - checks["s2_fchk"]) > ck["s2_tol"]:
        problems.append(f"<S^2> {checks['s2']:.6f} != fchk {checks['s2_fchk']:.6f}")
    if checks["e_pyscf"] is not None and abs(checks["e_pyscf"] - checks["e_fchk"]) > ck["e_tol"]:
        problems.append(
            f"PySCF {checks['xc']} energy of the fchk density {checks['e_pyscf']:.6f} != fchk "
            f"{checks['e_fchk']:.6f} (|dE| > check.e_tol = {ck['e_tol']})"
        )
    if problems:
        raise RuntimeError(
            f"{fchk_file.name}: fchk conversion check failed: " + "; ".join(problems)
        )
    scf_engine.analyse(mol, sol, n_fe)
    sol.pattern = [1 if c == "0" else -1 for c in sol.tag.replace("x", "0")]
    sol.gate = bs_guess.quality_gate(sol, n_fe, n_s, cfg["gate"]["min_fe"], cfg["gate"]["max_s"])
    return {
        "fchk": fc,
        "file": fchk_file,
        "mol": mol,
        "sol": sol,
        "order": order,
        "n_fe": n_fe,
        "n_s": n_s,
        "warnings": warnings,
        "checks": checks,
    }


def report(d):
    fc, sol, ck = d["fchk"], d["sol"], d["checks"]
    n = d["n_fe"] + d["n_s"]
    lines = [
        f"fchk {d['file'].name}: {fc['job']} {fc['method']} {fc['basis']}; "
        f"{structure.CLUSTER_TYPES[d['n_fe']]}, {len(d['order'])} atoms, charge {fc['Charge']:+d}, "
        f"multiplicity {fc['Multiplicity']}, (na, nb) {tuple(ck['nelec'])}",
        "cluster atoms (model idx = fchk atom): "
        + ", ".join(f"{k}_{d['mol'].atom_pure_symbol(k)}={d['order'][k] + 1}" for k in range(n)),
        f"basis: nao {d['mol'].nao}, nmo {sol.mo_coeff.shape[2]}, "
        f"{'Cartesian' if d['mol'].cart else 'pure'} d/f",
        f"check: orthonormality {ck['orthonormality_error']:.1e}, <S^2> {ck['s2']:.6f} "
        f"(fchk {ck['s2_fchk']}), E fchk {ck['e_fchk']:.8f}"
        + (
            f", E PySCF {ck['xc']} {ck['e_pyscf']:.8f} (dE {ck['e_pyscf'] - ck['e_fchk']:+.2e})"
            if ck["e_pyscf"] is not None
            else ", energy check off"
        ),
        f"spins: tag {sol.tag}, gate {'pass' if sol.gate else 'FAIL'}  "
        + "  ".join(f"{k} {v:+.2f}" for k, v in scf_engine.cluster_spins(sol, n).items()),
    ]
    lines += [f"WARNING {w}" for w in d["warnings"]]
    return "\n".join(lines)


def _targets(cfg, mol, n_fe, n_s):
    src = cfg["avas"]["target_aos"]
    if src.startswith("file:"):
        ao_dir = paths.ao_dir(src[5:], cfg["base_dir"])
        t, lab, res = targets.projected_file_targets(mol, n_fe + n_s, ao_dir)
    elif src == "free-atom":
        if mol.cart:
            raise ValueError("target_aos = 'free-atom' needs pure d/f shells; use file:AOs")
        t, lab = targets.free_atom_targets(mol, n_fe + n_s)
        res = None
    else:
        raise ValueError(f"unknown target_aos {src!r} (file:<dir> | free-atom)")
    if len(lab) != 5 * n_fe + 3 * n_s:
        raise ValueError(f"expected {5 * n_fe + 3 * n_s} target orbitals, got {len(lab)}")
    return t, lab, res


def run_fchk(fchk_file, state=None, n_fe3=None, geometry=None, cfg=None, verbose=0):
    """Active space + manifest from a BS-DFT fchk; returns the manifest path."""
    t0 = time.time()
    cfg = cfg or load_config()
    threads = scf_engine.set_threads(cfg["scf"]["threads"])
    d = load(fchk_file, cfg, verbose)
    print(report(d))
    mol, sol, n_fe, n_s = d["mol"], d["sol"], d["n_fe"], d["n_s"]
    ctype = structure.CLUSTER_TYPES[n_fe]
    if n_fe3 is None:
        if state is None:
            raise ValueError("give state or n_fe3")
        try:
            n_fe3 = structure.REDOX_PRESETS[ctype][state]
        except KeyError as err:
            raise ValueError(
                f"unknown redox state {state!r} for {ctype}; use n_fe3 or one of "
                f"{list(structure.REDOX_PRESETS[ctype])}"
            ) from err
    state = state or f"fe3x{n_fe3}"
    if "x" in sol.tag:
        raise RuntimeError(f"an Fe has no local moment in the fchk solution (tag {sol.tag})")
    if not sol.gate:
        raise RuntimeError(
            f"the fchk solution fails the gate (|Fe| >= {cfg['gate']['min_fe']}, "
            f"|S| <= {cfg['gate']['max_s']}); see the spins above"
        )

    ovlp = mol.intor_symmetric("int1e_ovlp")
    tgt, tgt_labels, residual = _targets(cfg, mol, n_fe, n_s)
    if residual is not None:
        print(f"target AOs {cfg['avas']['target_aos']} projected: max residual {residual:.2e}")
    dm = sol.make_rdm1()
    av = cfg["avas"]
    act = uno_avas.select_active_space(
        sol.mo_coeff[0],
        dm[0],
        dm[1],
        ovlp,
        tgt,
        *mol.nelec,
        noon_threshold=av["noon_threshold"],
        min_frac_weight=av["min_frac_weight"],
        allow_ligand_frac=av["allow_ligand_frac"],
    )
    nominal = structure.nominal_active_electrons(n_fe, n_fe3, n_s)
    if sum(act.nelecas) != nominal:
        raise RuntimeError(
            f"UNO-AVAS took {sum(act.nelecas)} active electrons, but state {state} "
            f"(n_Fe(III) {n_fe3}) has {nominal} in Fe 3d + bridging S 3p; wrong --state, or the "
            "fchk charge does not match the redox state"
        )

    fc, ck = d["fchk"], d["checks"]
    stem = d["file"].stem
    stem = stem[: -len(SUFFIX)] if stem.endswith(SUFFIX) else stem
    geometry = geometry or state
    out_root = Path(cfg["output"]["dir"])
    out_root = out_root if out_root.is_absolute() else cfg["base_dir"] / out_root

    def prefix_of(tag):
        return cfg["output"]["run_name"].format(stem=stem, E=state, R=geometry, S=tag)

    def body_of(r):
        ver, prefix, w = r["ver"], r["prefix"], r["weights"]
        return {
            "structure": {
                "source": "gaussian-fchk",
                "fchk_file": d["file"].name,
                "fchk_sha256": file_sha256(d["file"]),
                "cluster_type": ctype,
                "n_fe": n_fe,
                "n_s_bridging": n_s,
                "cluster_atoms": [
                    {
                        "model_idx": k,
                        "element": mol.atom_pure_symbol(k),
                        "fchk_idx": d["order"][k] + 1,
                    }
                    for k in range(n_fe + n_s)
                ],
                "n_atoms": mol.natm,
                "geometry": "optimized (fchk)",
                "prep_warnings": d["warnings"],
            },
            "electronic": {
                "charge": int(fc["Charge"]),
                "multiplicity": int(fc["Multiplicity"]),
                "n_fe3": n_fe3,
                "job": fc["job"],
                "method": fc["method"],
                "basis": fc["basis"],
                "cart": bool(mol.cart),
                "auxbasis": cfg["scf"]["auxbasis"],
                "engine": "pyscf",
                "threads": threads,
                "redox_label": state,
                "geometry_label": geometry,
                "nao": mol.nao,
                "nmo": int(sol.mo_coeff.shape[2]),
                "e_fchk": ck["e_fchk"],
                "s2_fchk": ck["s2_fchk"],
                "e_pyscf_check": ck["e_pyscf"],
                "xc_check": ck["xc"],
                "orthonormality_error": ck["orthonormality_error"],
                "full_system_scf": "none (fchk orbitals used as they are)",
            },
            "active_space": {
                "norb": act.ncas,
                "nelec": sum(act.nelecas),
                "ms2": act.nelecas[0] - act.nelecas[1],
                "nalpha": act.nelecas[0],
                "nbeta": act.nelecas[1],
                "n_core_orbitals": act.ncore,
                "labels_file": f"{prefix}_active_orbitals_labels.npy",
                "label_weights": [round(float(x), 4) for x in w],
                "eri": r["eri"],
                "ecore": r["ecore"],
                "selection": {
                    "method": "uno-avas",
                    "density": "fchk BS-DFT (Kohn-Sham)",
                    "noon_threshold": av["noon_threshold"],
                    "n_fractional_uno": act.n_frac,
                    "fractional_weights": [round(float(x), 4) for x in act.frac_weights],
                    "n_from_doubly_occ": act.n_from_occ,
                    "n_from_virtual": act.n_from_vir,
                    "last_kept_sigma2": act.last_kept,
                    "first_rejected_sigma2": act.first_rejected,
                    "target_aos": av["target_aos"],
                    "target_projection_residual": residual,
                    "labels": av["labels"],
                    "min_target_overlap": round(float(np.min(w)), 4),
                },
            },
            "bs_reference": {
                "spin_pattern": r["pattern"],
                "spin_tag": r["tag"],
                "full_spin_pattern": sol.pattern,
                "full_spin_tag": sol.tag,
                "atom_spins": ver["atom_spins"],
                "atom_spins_full": scf_engine.cluster_spins(sol, n_fe + n_s),
                "full_system_method": fc["method"],
                "e_full_uhf": None,
                "e_full_dft": sol.e_tot,
                "s2_full_uhf": None,
                "s2_full_dft": sol.s2,
                "gate_full": {"pass": bool(sol.gate), **cfg["gate"]},
                "e_active_uhf": ver["e_active_uhf"],
                "s2_active_uhf": ver["s2_active_uhf"],
                "e_active_projected_rdm": ver["e_rdm"],
                "projected_rdm_trace": r["projected_rdm_trace"],
                "active_spin_tag": ver["spin_tag"],
                "label_start": ver["label_start"],
                "rdm_source": f"converged active-space UHF, {r['start']} start",
                "reference_start": r["start"],
                "rdm_file": f"{prefix}_active_space_rdm.npy",
                "rdm_trace": r["rdm_trace"],
                "verify_warnings": ver["warnings"],
            },
            "candidates": [
                {
                    "spin_pattern": sol.pattern,
                    "spin_tag": sol.tag,
                    "e_full_dft": sol.e_tot,
                    "s2": sol.s2,
                    "converged": True,
                    "gate": bool(sol.gate),
                    "source": d["file"].name,
                }
            ],
        }

    path = pipeline.active_space_artifacts(
        mol,
        act,
        tgt,
        tgt_labels,
        ovlp,
        dm,
        sol,
        av,
        cfg["scf"]["auxbasis"],
        out_root / f".stage_{stem}_{state}",
        out_root,
        prefix_of,
        body_of,
        partial=True,
    )
    print(f"wrote {path} ({time.time() - t0:.0f} s)")
    return path
