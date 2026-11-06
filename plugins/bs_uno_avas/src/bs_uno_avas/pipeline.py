"""End-to-end driver: config → prep → HS/BS UHF → UNO-AVAS → labels → FCIDUMP/RDM → verify →
artifacts + manifest. Each step can also be called on its own (see ``cli.py``)."""

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import tomllib
from pyscf import df

from . import (
    bs_guess,
    cas_hamiltonian,
    labels,
    manifest,
    paths,
    scf_engine,
    structure,
    targets,
    uno_avas,
)
from . import verify as verify_mod

DEFAULTS = {
    "structure": {
        "pdb": None,
        "cluster_resname": None,
        "protonate": "check",
        "sg_cutoff": 3.3,
        "his": "HIE",
        "cluster_only": False,
    },
    "scf": {
        "engine": "auto",
        "method": "uhf",
        "xc": None,
        "basis": "dzvp",
        "auxbasis": "def2-universal-jkfit",
        "guesses": ["minao", "atom"],
        "bs_guess": "high-spin",
        "patterns": "auto",
        "mirrors": True,
        "stability": "internal",
        "n_stability": 3,
        "threads": 0,
        "max_memory_mb": 4000,
        "max_cycle": 100,
        "grid_level": 3,
        "selection": "lowest",
    },
    "avas": {
        "target_aos": "file:AOs",
        "noon_threshold": 0.2,
        "min_frac_weight": 0.5,
        "allow_ligand_frac": False,
        "labels": "polar",
        "eri": "auto",
        "tag_mismatch": "error",
    },
    "output": {
        "dir": "output",
        "work_dir": "work/bs_uno_avas",
        "run_name": "{stem}_E{E}_R{R}_S{S}",
        "geometry_label": None,
    },
}


def _parse_value(text):
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text  # bare string, e.g. scf.engine=cpu


def load_config(path, pdb=None, overrides=()):
    """TOML config merged over DEFAULTS; relative paths resolve against the workspace
    (:func:`paths.workspace`) unless ``base_dir`` is given in the file (relative to the file).
    ``pdb`` overrides ``[structure] pdb`` (one config for many PDBs; a relative ``pdb`` is taken
    relative to the current directory). ``overrides`` are ``"section.key=value"`` strings (TOML
    value syntax, bare strings allowed)."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = {k: {**v, **raw.get(k, {})} for k, v in DEFAULTS.items()}
    cfg["states"] = raw.get("states", {})
    if not cfg["states"]:
        raise ValueError(f"{path}: no [states.<name>] table")
    base = Path(raw["base_dir"]) if "base_dir" in raw else paths.workspace()
    if not base.is_absolute():
        base = (path.resolve().parent / base).resolve()
    cfg["base_dir"] = base
    unknown = set(raw) - set(DEFAULTS) - {"states", "base_dir"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    for sec, defaults in DEFAULTS.items():
        extra = set(raw.get(sec, {})) - set(defaults)
        if extra:
            raise ValueError(f"{path}: unknown keys in [{sec}]: {sorted(extra)}")
    for item in overrides:
        dotted, sep, value = item.partition("=")
        sec, _, key = dotted.strip().partition(".")
        if not sep or sec not in DEFAULTS or key not in DEFAULTS[sec]:
            raise ValueError(f"bad override {item!r}: use section.key=value with a known key")
        cfg[sec][key] = _parse_value(value.strip())
    if pdb is not None:
        cfg["structure"]["pdb"] = str(Path(pdb).resolve())
    if not cfg["structure"]["pdb"]:
        raise ValueError(f"{path}: [structure] pdb is required (or pass --pdb)")
    if cfg["scf"]["bs_guess"] not in ("high-spin", "atomic"):
        raise ValueError(f"{path}: scf.bs_guess must be 'high-spin' or 'atomic'")
    rn = cfg["structure"]["cluster_resname"]
    if rn is not None and not (isinstance(rn, str) and 0 < len(rn.strip()) <= 3):
        raise ValueError(f"{path}: structure.cluster_resname must be a 1-3 character PDB name")
    if cfg["structure"]["protonate"] not in ("check", "builtin", "cap", "off"):
        raise ValueError(f"{path}: structure.protonate must be check, builtin, cap or off")
    if cfg["avas"]["tag_mismatch"] not in ("error", "use-active"):
        raise ValueError(f"{path}: avas.tag_mismatch must be 'error' or 'use-active'")
    return cfg


def _path(cfg, p):
    p = Path(p)
    return p if p.is_absolute() else cfg["base_dir"] / p


def _target_source(cfg):
    src = cfg["avas"]["target_aos"]
    return f"file:{paths.ao_dir(src[5:], cfg['base_dir'])}" if src.startswith("file:") else src


def prep(cfg):
    s = cfg["structure"]
    model = structure.prepare_model(
        _path(cfg, s["pdb"]),
        protonate=s["protonate"],
        sg_cutoff=s["sg_cutoff"],
        his=s["his"],
        cluster_only=s["cluster_only"],
        cluster_resname=(s["cluster_resname"] or "").strip() or None,
    )
    # a dangling bond is an unpaired electron: it changes the spin state and the SCF (stop here,
    # before any SCF; protonate = "cap" adds the missing H, "off" skips the check)
    deficits = [w for w in model.warnings if w.startswith("valence:")]
    if deficits:
        raise ValueError(
            f"{model.pdb_file}: {len(deficits)} under-valent atoms (cut points / missing H): "
            + "; ".join(w[len("valence: ") :] for w in deficits)
            + ". Set [structure] protonate = 'cap' to add H, or fix the excerpt."
        )
    states = {}
    for name, st in cfg["states"].items():
        states[name] = structure.charge_and_spin(
            model,
            n_fe3=st.get("n_fe3"),
            state=None if "n_fe3" in st else name,
            charge=st.get("charge"),
            spin=st.get("spin"),
        )
    return model, states


def dry_run(cfg):
    model, states = prep(cfg)
    print(structure.report(model, states))
    mol = bs_guess.build_mol(
        model,
        next(iter(states.values()))["charge"],
        next(iter(states.values()))["spin"],
        cfg["scf"]["basis"],
    )
    ncas = 5 * model.n_fe + 3 * model.n_s
    print(
        f"basis {cfg['scf']['basis']}: nao {mol.nao}, ncas {ncas}, "
        f"eri {cas_hamiltonian.resolve_eri(cfg['avas']['eri'], mol.nao)}, "
        f"engine {scf_engine.resolve_engine(cfg['scf']['engine'])}"
    )
    naux = df.addons.make_auxmol(mol, cfg["scf"]["auxbasis"]).nao
    df_gb = mol.nao * (mol.nao + 1) / 2 * naux * 8 / 1e9
    max_gb = cfg["scf"]["max_memory_mb"] / 1e3
    print(
        f"DF: naux {naux}, 3-index tensor {df_gb:.1f} GB; scf.max_memory_mb = "
        f"{cfg['scf']['max_memory_mb']} ({max_gb:.0f} GB)"
        + (
            ""
            if df_gb < 0.8 * max_gb
            else "  <- WARNING: tensor will not fit in memory (slow, out-of-core)"
        )
    )
    targets.make_targets(mol, model.n_fe, model.n_s, _target_source(cfg))
    print(f"target AOs {cfg['avas']['target_aos']}: OK")
    sc = cfg["scf"]
    for name, cs in states.items():
        pats = bs_patterns(sc, model.cluster_type, model.n_fe, cs["nelec"] % 2)
        print(
            f"state {name}: BS patterns {[bs_guess.tag_of(p) for p in pats]}, guess "
            f"{sc['bs_guess']}, stability {sc['stability']}"
        )
    return model, states


def _per_cluster(value, cluster_type):
    """``value``, or ``value[cluster_type]`` for a table ``{Fe2S2 = …, Fe4S4 = …}``."""
    if isinstance(value, dict):
        if cluster_type not in value:
            raise ValueError(f"no entry for {cluster_type} in {value}")
        return value[cluster_type]
    return value


def bs_patterns(sc, cluster_type, n_fe, ms2):
    """BS spin patterns of one state from ``scf.patterns`` / ``scf.mirrors``.

    ``mirrors = "spin"``: mirrors only for ms ≠ 0. At ms = 0 the α↔β swap of a solution is the
    exact solution of the mirror pattern (same energy), so it is not computed. An explicit
    pattern list is used as given, plus its mirrors only with ``"spin"`` at ms ≠ 0.
    """
    patterns = _per_cluster(sc["patterns"], cluster_type)
    raw = _per_cluster(sc["mirrors"], cluster_type)
    if raw not in (True, False, "spin"):
        raise ValueError(f"scf.mirrors must be true, false or 'spin', not {raw!r}")
    mirrors = ms2 != 0 if raw == "spin" else raw
    if patterns == "auto":
        return bs_guess.enumerate_patterns(n_fe, mirrors)
    patterns = [[int(s) for s in p] for p in patterns]
    if any(len(p) != n_fe or sorted(set(p)) != [-1, 1] for p in patterns):
        raise ValueError(f"{cluster_type}: patterns need {n_fe} entries of ±1, got {patterns}")
    if raw == "spin" and mirrors:
        patterns += [[-s for s in p] for p in patterns if [-s for s in p] not in patterns]
    return patterns


def _scf_cfg(cfg, engine):
    return {**cfg["scf"], "engine": engine}


def run_state(cfg, state, verbose=0):
    """Full pipeline for one redox state; returns the manifest path."""
    t0 = time.time()
    sc = cfg["scf"]
    engine = scf_engine.resolve_engine(sc["engine"])
    threads = scf_engine.set_threads(sc["threads"])
    model, states = prep(cfg)
    cs = states[state]
    print(structure.report(model, {state: cs}))
    print(f"engine {engine}, threads {threads}")
    n_fe, n_s = model.n_fe, model.n_s
    work = _path(cfg, cfg["output"]["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    key = (
        f"{Path(model.pdb_file).stem}_{model.pdb_sha256[:8]}_{sc['basis']}_{sc['method']}"
        f"{'_' + sc['xc'] if sc['method'] == 'uks' else ''}_q{cs['charge']}"
    )

    if model.protonation["tool"] == "cap":
        key += f"_cap{model.protonation['added_h']}"

    mol_hs = bs_guess.build_mol(
        model, cs["charge"], cs["spin"], sc["basis"], sc["max_memory_mb"], verbose
    )
    scfc = _scf_cfg(cfg, engine)
    ms2 = cs["nelec"] % 2
    n_b = (cs["nelec"] - ms2) // 2
    print(
        f"nao {mol_hs.nao}, nelec {cs['nelec']}: high spin (2S {cs['spin']}) {mol_hs.nelec}, "
        f"BS (2S {ms2}) {(n_b + ms2, n_b)}"
    )
    patterns = bs_patterns(sc, model.cluster_type, n_fe, ms2)
    print(f"BS patterns {[bs_guess.tag_of(p) for p in patterns]}, guess {sc['bs_guess']}")
    bs_args = (patterns, ms2, scfc, n_fe, n_s, str(work), f"{key}_ms{ms2}", verbose)

    hs_tag = f"{key}_s{cs['spin']}"

    def run_high_spin():
        return bs_guess.high_spin(mol_hs, scfc, n_fe, n_s, str(work), hs_tag, verbose)[0]

    av = cfg["avas"]

    def bs_and_active_space(hs):
        """BS-UHF per pattern, selection and UNO-AVAS (both raise RuntimeError on a bad state)."""
        mol, bs_all = bs_guess.broken_symmetry(mol_hs, hs, *bs_args)
        sel = bs_guess.select(bs_all, sc["selection"])
        print(f"selected {sel.name} -> {sel.tag}: E = {sel.e_tot:.8f}")
        ovlp = mol.intor_symmetric("int1e_ovlp")
        tgt, tgt_labels = targets.make_targets(mol, n_fe, n_s, _target_source(cfg))
        dm = sel.make_rdm1()
        act = uno_avas.select_active_space(
            sel.mo_coeff[0],
            dm[0],
            dm[1],
            ovlp,
            tgt,
            *mol.nelec,
            noon_threshold=av["noon_threshold"],
            min_frac_weight=av["min_frac_weight"],
            allow_ligand_frac=av["allow_ligand_frac"],
        )
        return mol, bs_all, sel, ovlp, tgt, tgt_labels, dm, act

    hs = None
    banked = None
    if sc["bs_guess"] == "atomic":
        # a high-spin solution already on disk (converged, gate pass) costs nothing: use it
        banked = bs_guess.banked_high_spin(mol_hs, scfc, n_fe, n_s, str(work), hs_tag)
    if banked is not None:
        print(f"  using the high-spin solution on disk ({banked.name}, E = {banked.e_tot:.8f})")
        hs = banked
        res = bs_and_active_space(hs)
    elif sc["bs_guess"] == "atomic":
        # No high-spin SCF. The atomic guess can land in another state (e.g. a ligand radical,
        # rejected by UNO-AVAS) or miss the pattern: then use the high-spin spin-flip route.
        try:
            res = bs_and_active_space(None)
        except RuntimeError as exc:
            print(f"  atomic BS guess rejected: {exc}")
            print("  falling back to the high-spin spin-flip guess")
            hs = run_high_spin()
            res = bs_and_active_space(hs)
    else:
        hs = run_high_spin()
        res = bs_and_active_space(hs)
    mol, bs_all, sel, ovlp, tgt, tgt_labels, dm, act = res
    bs_source = "atomic" if hs is None else "high-spin spin flip"
    if banked is not None:
        bs_source += " (high spin from chkfile)"
    geometry = cfg["output"]["geometry_label"] or "pdb"

    def prefix_of(tag):
        return cfg["output"]["run_name"].format(
            stem=Path(model.pdb_file).stem,
            resname=model.cluster_resname,
            E=state,
            R=geometry,
            S=tag,
        )

    def body_of(r):
        return _run_state_body(
            cfg,
            model,
            cs,
            state,
            geometry,
            engine,
            threads,
            mol,
            hs,
            bs_source,
            bs_all,
            sel,
            act,
            r,
        )

    path = active_space_artifacts(
        mol,
        act,
        tgt,
        tgt_labels,
        ovlp,
        dm,
        sel,
        av,
        sc["auxbasis"],
        work / f"stage_{key}_{state}",
        _path(cfg, cfg["output"]["dir"]),
        prefix_of,
        body_of,
    )
    print(f"wrote {path} ({time.time() - t0:.0f} s)")
    return path


def active_space_artifacts(
    mol,
    act,
    tgt,
    tgt_labels,
    ovlp,
    dm,
    sel,
    av,
    auxbasis,
    stage,
    out_root,
    prefix_of,
    body_of,
    partial=False,
    forced_tag=None,
):
    """Shared tail of the routes (PDB, fchk): labels → active-space Hamiltonian → verify → files.

    ``act`` is the UNO-AVAS :class:`~uno_avas.ActiveSpace` of the full-system solution ``sel``
    (AO density ``dm``). ``prefix_of(tag)`` gives the run name, ``body_of(r)`` the manifest body
    from the results ``r`` (dict: ``labels``, ``weights``, ``eri``, ``ecore``, ``ver``, ``tag``,
    ``pattern``, ``start``, ``rdm`` (as written), ``rdm_trace``, ``projected_rdm_trace``).
    ``partial``: write into a hidden ``.<prefix>.partial`` directory that replaces ``<prefix>/``
    only when complete.
    ``forced_tag``: write this tag without the active-space UHF (no verify, ``r["ver"]`` is
    None); the RDM file is the projected full-system density, each spin block scaled to the exact
    electron count. Returns the manifest path. With ``partial`` a failed check also removes the
    ``stage`` directory (it sits next to the outputs there).
    """
    try:
        return _active_space_artifacts(
            mol,
            act,
            tgt,
            tgt_labels,
            ovlp,
            dm,
            sel,
            av,
            auxbasis,
            stage,
            out_root,
            prefix_of,
            body_of,
            partial,
            forced_tag,
        )
    except Exception:
        if partial:
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _active_space_artifacts(
    mol,
    act,
    tgt,
    tgt_labels,
    ovlp,
    dm,
    sel,
    av,
    auxbasis,
    stage,
    out_root,
    prefix_of,
    body_of,
    partial,
    forced_tag,
):
    sl = slice(act.ncore, act.ncore + act.ncas)
    if av["labels"] == "polar":
        c_act, lab, w = labels.polar_rotation(act.mo_coeff[:, sl], tgt, ovlp, tgt_labels)
    elif av["labels"] == "boys":
        c_act, lab, w = labels.boys_rotation(mol, act.mo_coeff[:, sl], tgt, ovlp, tgt_labels)
    else:
        raise ValueError(f"unknown labels {av['labels']!r} (polar | boys)")
    mo = act.mo_coeff.copy()
    mo[:, sl] = c_act
    for la, wi in zip(lab, w):
        print(f"  {la:12s} {wi:.3f}" + ("  <- low" if wi < 0.5 else ""))

    # Hamiltonian, RDM, files
    eri = cas_hamiltonian.resolve_eri(av["eri"], mol.nao)
    h1, h2, ecore = cas_hamiltonian.active_hamiltonian(
        mol, mo, act.ncore, act.ncas, act.nelecas, eri, auxbasis
    )
    rdm_proj = cas_hamiltonian.active_rdm(c_act, dm, ovlp)
    tr_proj = [float(np.trace(r)) for r in rdm_proj]

    # Verify on a staged FCIDUMP, before any artifact is written: active-space UHF from the
    # projected full-system RDM and from the labels. The artifact RDM is a converged active-space
    # UHF (exact traces, reproduces e_active_uhf); the projected RDM is only recorded.
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    staged_fcidump = stage / "avas.fcidump"
    cas_hamiltonian.write_fcidump(staged_fcidump, h1, h2, act.ncas, act.nelecas, ecore)
    if forced_tag is not None:
        # no active-space UHF: the tag is set by the caller, the RDM is the projected density
        tag, pattern = forced_tag, [1 if c == "0" else -1 for c in forced_tag]
        ver, start = None, "projected-rdm"
        rdm = np.array([r * (n / t) for r, n, t in zip(rdm_proj, act.nelecas, tr_proj)])
        print(
            f"forced tag {tag}: no active-space UHF; RDM = projected density, traces "
            f"({tr_proj[0]:.4f}, {tr_proj[1]:.4f}) scaled to {tuple(act.nelecas)}"
        )
    else:
        tag, pattern, ver, rdm, start = _verified_reference(
            staged_fcidump, lab, rdm_proj, sel, w, av
        )
    tr = [float(np.trace(r)) for r in rdm]

    # artifacts (only after the checks passed)
    prefix = prefix_of(tag)
    final = Path(out_root) / prefix
    out = final.parent / f".{prefix}.partial" if partial else final
    if partial:
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    fcidump = out / f"{prefix}_avas.fcidump"
    shutil.move(staged_fcidump, fcidump)
    stage.rmdir()
    np.save(out / f"{prefix}_active_orbitals_labels.npy", np.array(lab, dtype=str))
    np.save(out / f"{prefix}_active_orbitals.npy", c_act)
    np.save(out / f"{prefix}_active_space_rdm.npy", rdm)
    r = {
        "labels": lab,
        "weights": w,
        "eri": eri,
        "ecore": ecore,
        "ver": ver,
        "tag": tag,
        "pattern": pattern,
        "start": start,
        "rdm": rdm,
        "rdm_trace": tr,
        "projected_rdm_trace": tr_proj,
        "prefix": prefix,
    }
    path = manifest.write_manifest(out, prefix, body_of(r))
    manifest.read_manifest(path)  # round trip
    if partial:
        shutil.rmtree(final, ignore_errors=True)
        out.rename(final)
        path = final / path.name
    return path


def _verified_reference(staged_fcidump, lab, rdm_proj, sel, w, av):
    """Active-space UHF from the projected RDM and from the labels; the lowest converged start
    with the reference tag, rechecked from the saved RDM. Returns (tag, pattern, ver, rdm, start).
    """
    ver = verify_mod.verify(staged_fcidump, lab, rdm_proj, sel.pattern, sel.tag, w)
    rdm = ver.pop("rdm_converged")
    rdm_lab = ver["label_start"].pop("rdm")
    print(
        f"verify: active-space UHF from RDM E = {ver['e_active_uhf']:.6f} tag {ver['spin_tag']}"
        f" (E(projected rdm) = {ver['e_rdm']:.6f}); label start E = {ver['label_start']['e']:.6f} "
        f"tag {ver['label_start']['spin_tag']}"
    )
    for msg in ver["warnings"]:
        print(f"  WARNING {msg}")
    tag, pattern = reference_tag(sel, ver, av["tag_mismatch"])
    # Same choice as the SBD flow: the lowest converged start with the reference tag.
    lab_start = ver["label_start"]
    start = "projected-rdm"
    if (
        lab_start["converged"]
        and lab_start["spin_tag"] == tag
        and (lab_start["e"] < ver["e_active_uhf"] - 1e-6)
    ):
        print(
            f"  label start is lower with the same tag ({lab_start['e']:.6f} < "
            f"{ver['e_active_uhf']:.6f}); writing the label-start solution as the reference"
        )
        rdm, start = rdm_lab, "labels"
        ver["e_active_uhf"] = lab_start["e"]
        ver["atom_spins"] = {
            k: round(float(v), 4) for k, v in verify_mod.atom_spins(rdm, lab).items()
        }
        ver["s2_active_uhf"] = None  # recomputed from the recheck below
    recheck = verify_mod.active_uhf(staged_fcidump, dm0=rdm)
    if ver["s2_active_uhf"] is None:
        ver["s2_active_uhf"] = float(recheck.spin_square()[0])
    recheck_tag = verify_mod.spin_tag(verify_mod.atom_spins(np.asarray(recheck.make_rdm1()), lab))
    if abs(recheck.e_tot - ver["e_active_uhf"]) > 1e-5 or recheck_tag != tag:
        raise RuntimeError(
            f"active-space UHF from the saved RDM gives {recheck.e_tot:.6f} / {recheck_tag}, "
            f"not {ver['e_active_uhf']:.6f} / {tag}"
        )
    return tag, pattern, ver, rdm, start


def _run_state_body(
    cfg, model, cs, state, geometry, engine, threads, mol, hs, bs_source, bs_all, sel, act, r
):
    """Manifest body of the PDB route (:func:`run_state`)."""
    sc, av = cfg["scf"], cfg["avas"]
    n_fe, n_s = model.n_fe, model.n_s
    ver, prefix, w = r["ver"], r["prefix"], r["weights"]
    tag, pattern, start = r["tag"], r["pattern"], r["start"]
    eri, ecore, tr, tr_proj = r["eri"], r["ecore"], r["rdm_trace"], r["projected_rdm_trace"]
    return {
        "structure": {
            "pdb_file": os.path.relpath(model.pdb_file, cfg["base_dir"]),
            "pdb_sha256": model.pdb_sha256,
            "cluster_resname": model.cluster_resname,
            "cluster_type": model.cluster_type,
            "n_fe": n_fe,
            "n_s_bridging": n_s,
            "cluster_atoms": model.cluster_atoms(),
            "n_atoms": len(model.atoms),
            "thiolates": model.thiolates,
            "residue_charges": model.residue_charges,
            "protonation": model.protonation,
            "prep_warnings": model.warnings,
        },
        "electronic": {
            "charge": cs["charge"],
            "multiplicity_hs": cs["spin"] + 1,
            "n_fe3": cs["n_fe3"],
            "overrides": cs["overrides"],
            "basis": sc["basis"],
            "auxbasis": sc["auxbasis"],
            "method": sc["method"].upper(),
            "xc": sc["xc"] if sc["method"] == "uks" else None,
            "engine": "gpu4pyscf" if engine == "gpu" else "pyscf",
            "threads": threads,
            "redox_label": state,
            "geometry_label": geometry,
            "nao": mol.nao,
            "e_hs_uhf": hs.e_tot if hs is not None else None,
            "hs_guess": hs.name if hs is not None else "skipped (atomic BS guess)",
            "bs_guess": bs_source,
            "full_system_stability": sc["stability"],
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
            "eri": eri,
            "ecore": ecore,
            "selection": {
                "method": "uno-avas",
                "noon_threshold": av["noon_threshold"],
                "n_fractional_uno": act.n_frac,
                "fractional_weights": [round(float(x), 4) for x in act.frac_weights],
                "n_from_doubly_occ": act.n_from_occ,
                "n_from_virtual": act.n_from_vir,
                "last_kept_sigma2": act.last_kept,
                "first_rejected_sigma2": act.first_rejected,
                "target_aos": av["target_aos"],
                "labels": av["labels"],
                "min_target_overlap": round(float(np.min(w)), 4),
            },
        },
        "bs_reference": {
            "spin_pattern": pattern,
            "spin_tag": tag,
            "full_spin_pattern": sel.pattern,
            "full_spin_tag": sel.tag,
            "atom_spins": ver["atom_spins"],
            "atom_spins_full": scf_engine.cluster_spins(sel, n_fe + n_s),
            "e_full_uhf": sel.e_tot,
            "s2_full_uhf": sel.s2,
            "e_active_uhf": ver["e_active_uhf"],
            "s2_active_uhf": ver["s2_active_uhf"],
            "e_active_projected_rdm": ver["e_rdm"],
            "projected_rdm_trace": tr_proj,
            "active_spin_tag": ver["spin_tag"],
            "label_start": ver["label_start"],
            "rdm_source": f"converged active-space UHF, {start} start",
            "reference_start": start,
            "rdm_file": f"{prefix}_active_space_rdm.npy",
            "rdm_trace": tr,
            "verify_warnings": ver["warnings"],
        },
        "candidates": [
            {
                "spin_pattern": s.pattern,
                "spin_tag": s.tag,
                "e_full_uhf": s.e_tot,
                "s2": s.s2,
                "converged": s.converged,
                "gate": s.gate,
                "atom_spins": scf_engine.cluster_spins(s, n_fe + n_s),
            }
            for s in bs_all
        ],
    }


def reference_tag(sel, ver, policy="error"):
    """Tag / pattern written to the manifest: must be the tag the saved RDM reproduces.

    The active-space tag (from the RDM) must equal the full-system tag of the selected
    BS-UHF and contain no ``x``. ``policy = "use-active"`` (explicit opt-in) writes the
    active-space tag instead; the full-system tag is kept as ``full_spin_tag`` either way.
    """
    active, full = ver["spin_tag"], sel.tag
    if "x" in active:
        raise RuntimeError(
            f"active-space UHF from the RDM has an Fe without local moment (tag {active}, "
            f"full-system {full}); not a usable BS reference"
        )
    if active != full:
        msg = (
            f"active-space UHF from the RDM converged to tag {active} "
            f"(E {ver['e_active_uhf']:.6f}), but the selected full-system BS-UHF has tag {full}"
        )
        if policy != "use-active":
            raise RuntimeError(
                msg + "; no artifacts written. Try scf.selection = '<other tag>', or set "
                "avas.tag_mismatch = 'use-active' to write the active-space tag"
            )
        print(f"  WARNING {msg}; writing the active-space tag (avas.tag_mismatch = 'use-active')")
    return active, [1 if c == "0" else -1 for c in active]


def dumps(obj):
    return json.dumps(obj, indent=2, default=manifest._json_default)
