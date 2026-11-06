"""Fast tests (< 1 min total): structure prep, guesses, manifest, fixtures, bare-Fe2S2 pipeline.

Run from the repo root: ``.venv/bin/python -m pytest plugins/bs_uno_avas/tests -q``.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from bs_uno_avas import bs_guess, cli, manifest, paths, pipeline, structure
from bs_uno_avas.make_manifest import wrap

PLUGIN = Path(__file__).resolve().parents[1]
SCH3 = PLUGIN / "examples" / "5XTD_FES_O301_SCH3.pdb"
EXCERPT = PLUGIN / "examples" / "5XTD_FES_O301_excerpt.pdb"
# optional local inputs: a protonated Fe2S2 excerpt with four Cys ligands (BS_UNO_AVAS_LOCAL_PDB,
# its cluster residue name in BS_UNO_AVAS_LOCAL_RESNAME unless FES), and older artifact sets
# without a manifest in workspace output/ (wrapped by make_manifest)
LOCAL_PDB = Path(os.environ.get("BS_UNO_AVAS_LOCAL_PDB", "/nonexistent"))
LOCAL_RESNAME = os.environ.get("BS_UNO_AVAS_LOCAL_RESNAME") or None
OLD_SETS = paths.workspace() / "output"


def _rename_cluster(src, dst, new, drop_last=False):
    lines = [ln for ln in src.read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))]
    out = []
    for ln in lines:
        if ln[17:20] == "FES":
            ln = ln[:17] + new + ln[20:]
        out.append(ln)
    if drop_last:
        out = [ln for ln in out if not (ln[17:20] == new and ln[12:16].strip() == "S2")]
    dst.write_text("\n".join(out) + "\nEND\n")
    return dst


# ---------- structure ----------


def test_sch3_model():
    m = structure.prepare_model(SCH3)
    assert (m.cluster_resname, m.n_fe, m.n_s) == ("FES", 2, 2)
    assert [a.element for a in m.atoms[:4]] == ["Fe", "Fe", "S", "S"]
    assert len(m.thiolates) == 4 and m.ligand_charge == -4
    assert structure.charge_and_spin(m, state="ox")["charge"] == -2
    red = structure.charge_and_spin(m, state="red")
    assert (red["charge"], red["spin"], red["nelec"]) == (-3, 9, 187)


def test_cluster_resname_builtin_and_custom(tmp_path):
    # SF4 is built in as Fe4S4: the 2-Fe cluster does not fit
    with pytest.raises(ValueError, match="must have 4 Fe and 4 S"):
        structure.prepare_model(_rename_cluster(SCH3, tmp_path / "sf4.pdb", "SF4"))
    # any other name is not detected: the error names the Fe residue and the option
    xyz = _rename_cluster(SCH3, tmp_path / "xyz.pdb", "XYZ")
    with pytest.raises(ValueError, match=r"no Fe-S cluster residue.*\['XYZ'\].*cluster_resname"):
        structure.prepare_model(xyz)
    # ... and works when given; the cluster type follows from the Fe count
    m = structure.prepare_model(xyz, cluster_resname="XYZ")
    assert (m.cluster_resname, m.cluster_type, m.n_fe, m.n_s) == ("XYZ", "Fe2S2", 2, 2)
    assert [a.element for a in m.atoms[:4]] == ["Fe", "Fe", "S", "S"]
    assert m.ligand_charge == -4 and not [w for w in m.warnings if w.startswith("valence")]
    with pytest.raises(ValueError, match="no residue ABC"):
        structure.prepare_model(xyz, cluster_resname="ABC")


def test_cluster_wrong_count(tmp_path):
    pdb = _rename_cluster(SCH3, tmp_path / "x.pdb", "XYZ", drop_last=True)
    with pytest.raises(ValueError, match=r"2 Fe and 2 S \(Fe2S2\) or 4 Fe and 4 S"):
        structure.prepare_model(pdb, cluster_resname="XYZ")
    fes = _rename_cluster(SCH3, tmp_path / "f.pdb", "FES", drop_last=True)
    with pytest.raises(ValueError, match="must have 2 Fe and 2 S"):
        structure.prepare_model(fes)


def test_cluster_resname_and_geometry_options(tmp_path):
    """[structure] cluster_resname / --cluster-resname; R = geometry_label, default pdb."""
    xyz = _rename_cluster(SCH3, tmp_path / "xyz.pdb", "XYZ")
    f = _bare_config(tmp_path, pdb=f'pdb = "{xyz}"\ncluster_resname = "XYZ"')
    model, states = pipeline.prep(pipeline.load_config(f))
    assert model.cluster_resname == "XYZ" and set(states) == {"red", "ox"}
    cfg = pipeline.load_config(
        _bare_config(tmp_path, pdb=f'pdb = "{xyz}"'),
        overrides=['structure.cluster_resname="XYZ"', 'output.geometry_label="ox"'],
    )
    assert pipeline.prep(cfg)[0].cluster_resname == "XYZ"
    assert cfg["output"]["geometry_label"] == "ox"
    with pytest.raises(ValueError, match="1-3 character"):
        pipeline.load_config(f, overrides=['structure.cluster_resname="TOOLONG"'])
    # the CLI shortcut options reach the config (prep only reads the structure)
    assert cli.main(["prep", "--config", str(f), "--cluster-resname", "XYZ"]) == 0


def test_excerpt_needs_hydrogens(tmp_path):
    with pytest.raises(ValueError, match="no hydrogens"):
        structure.prepare_model(EXCERPT)
    pdb = tmp_path / EXCERPT.name
    shutil.copy(EXCERPT, pdb)
    m = structure.prepare_model(pdb, protonate="builtin", sg_cutoff=3.8)
    assert (tmp_path / f"{EXCERPT.stem}_prot.pdb").exists()
    assert m.protonation["tool"] == "builtin" and m.protonation["added_h"] > 0
    assert structure.charge_and_spin(m, state="ox")["charge"] == -2
    # re-reading the protonated file gives the same charge with no valence warnings
    m2 = structure.prepare_model(tmp_path / f"{EXCERPT.stem}_prot.pdb", sg_cutoff=3.8)
    assert m2.ligand_charge == -4 and not [w for w in m2.warnings if w.startswith("valence")]


@pytest.mark.skipif(not LOCAL_PDB.exists(), reason="set BS_UNO_AVAS_LOCAL_PDB to run")
def test_local_excerpt_prep():
    m = structure.prepare_model(LOCAL_PDB, cluster_resname=LOCAL_RESNAME)
    assert (m.cluster_resname, m.n_fe, m.n_s) == (LOCAL_RESNAME or "FES", 2, 2)
    assert len(m.thiolates) == 4 and m.protonation["tool"] == "none"
    assert len(m.residue_charges) == 4 and set(m.residue_charges.values()) == {-1}
    assert all(k.startswith("CYS") for k in m.residue_charges)
    ox, red = (structure.charge_and_spin(m, state=s) for s in ("ox", "red"))
    assert (ox["charge"], ox["spin"], red["charge"], red["spin"]) == (-2, 10, -3, 9)
    assert red["nelec"] == ox["nelec"] + 1
    assert not [w for w in m.warnings if w.startswith("valence")]


# ---------- guesses ----------


def test_patterns():
    assert bs_guess.enumerate_patterns(2) == [[1, -1], [-1, 1]]
    assert len(bs_guess.enumerate_patterns(4)) == 6
    assert bs_guess.enumerate_patterns(4, mirrors=False) == [
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
    ]


def test_spin_flip_density_traces():
    m = structure.prepare_model(SCH3, cluster_only=True)
    mol = bs_guess.build_mol(m, 1, 9, "sto-3g")
    rng = np.random.default_rng(0)
    a = rng.normal(size=(mol.nao, mol.nao))
    dm_hs = np.array([a @ a.T / mol.nao, a @ a.T / (2 * mol.nao)])
    bs = bs_guess.bs_mol(mol, 1)
    dm = bs_guess.spin_flip_density(bs, dm_hs, [1, -1], *bs.nelec)
    s = bs.intor_symmetric("int1e_ovlp")
    assert np.allclose([np.trace(d @ s) for d in dm], bs.nelec)


# ---------- manifest / fixtures ----------

# older [Fe2S2(SCH3)4] sets (ROHF-AVAS, 6-31g*): name, tag, active-space BS-UHF energy
FIXTURES = [
    ("N1a_State1_Eox_Rox_S01", "01", -5067.870133),
    ("N1a_State1_Ered_Rox_S10", "10", -5067.633303),
]


@pytest.mark.parametrize("name,tag,energy", FIXTURES)
def test_wrap_fixture(tmp_path, name, tag, energy):
    if not (OLD_SETS / f"{name}_avas.fcidump").exists():
        pytest.skip("fixture not present")
    for suffix in manifest.FILE_SUFFIXES.values():
        shutil.copy(OLD_SETS / f"{name}{suffix}", tmp_path)
    path, ver = wrap(tmp_path / name, "FES", basis="6-31g*")
    assert ver["spin_tag"] == tag and abs(ver["e_active_uhf"] - energy) < 1e-5
    m, files, labels, rdm = manifest.read_manifest(path)
    assert m["active_space"]["norb"] == 16 and m["bs_reference"]["spin_tag"] == tag

    # tampered file -> hash error; missing required key -> error
    with open(files["rdm"], "ab") as f:
        f.write(b"\0")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        manifest.read_manifest(path)
    raw = json.loads(path.read_text())
    del raw["bs_reference"]["spin_tag"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="missing required key bs_reference.spin_tag"):
        manifest.read_manifest(path, check_hashes=False)


# ---------- end to end ----------


def test_bare_fe2s2_pipeline(tmp_path):
    """Default run_name uses the PDB stem; ox and red of one PDB do not collide."""
    cfg_file = tmp_path / "bare.toml"
    cfg_file.write_text(f"""
base_dir = "{PLUGIN / "examples"}"
[structure]
pdb = "5XTD_FES_O301_SCH3.pdb"
cluster_only = true
[states.red]
[scf]
engine = "cpu"
basis = "3-21g"
guesses = ["minao"]
[avas]
target_aos = "free-atom"
[output]
dir = "{tmp_path / "out"}"
work_dir = "{tmp_path / "work"}"
""")
    cfg = pipeline.load_config(cfg_file)
    path = pipeline.run_state(cfg, "red")
    m, files, labels, rdm = manifest.read_manifest(path)
    a, ref = m["active_space"], m["bs_reference"]
    assert (a["norb"], a["nelec"], a["ms2"]) == (16, 23, 1)
    assert ref["spin_tag"] == ref["active_spin_tag"] in ("01", "10")
    assert np.allclose(ref["rdm_trace"], [12, 11], atol=1e-6)
    assert np.allclose(ref["projected_rdm_trace"], [12, 11], atol=1e-2)
    assert ref["e_active_uhf"] <= ref["e_active_projected_rdm"] + 1e-8
    assert len(m["candidates"]) == 2
    assert list(labels[:5]) == ["0_Fe_3dxy", "0_Fe_3dyz", "0_Fe_3dz^2", "0_Fe_3dxz", "0_Fe_3dx2-y2"]
    assert path.name.startswith("5XTD_FES_O301_SCH3_Ered_Rpdb_S")  # R defaults to pdb
    # re-run reuses the chkfiles
    assert pipeline.run_state(cfg, "red") == path
    # geometry_label sets R (same chkfiles: the label is not part of the key)
    cfg["output"]["geometry_label"] = "ox"
    assert pipeline.run_state(cfg, "red").name.startswith("5XTD_FES_O301_SCH3_Ered_Rox_S")


def test_config_rejects_unknown_keys(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[structure]\npdb = "x.pdb"\nfoo = 1\n[states.red]\n')
    with pytest.raises(ValueError, match="unknown keys"):
        pipeline.load_config(f)


# ---------- tag consistency (manifest tag must be reproducible from the RDM) ----------


class _Sel:
    def __init__(self, tag):
        self.tag = tag


def test_reference_tag_policy():
    ver = {"spin_tag": "10", "e_active_uhf": -1.0}
    assert pipeline.reference_tag(_Sel("10"), ver) == ("10", [-1, 1])
    with pytest.raises(RuntimeError, match="no artifacts written"):
        pipeline.reference_tag(_Sel("01"), ver)
    assert pipeline.reference_tag(_Sel("01"), ver, "use-active") == ("10", [-1, 1])
    with pytest.raises(RuntimeError, match="without local moment"):
        pipeline.reference_tag(_Sel("10"), {"spin_tag": "1x", "e_active_uhf": -1.0}, "use-active")


def test_tag_mismatch_writes_nothing(tmp_path, monkeypatch):
    cfg = pipeline.load_config(_bare_config(tmp_path))
    real = pipeline.verify_mod.verify

    def flipped(*args, **kwargs):
        out = real(*args, **kwargs)
        out["spin_tag"] = out["spin_tag"][::-1]  # pretend the active-space UHF flipped
        return out

    monkeypatch.setattr(pipeline.verify_mod, "verify", flipped)
    with pytest.raises(RuntimeError, match="no artifacts written"):
        pipeline.run_state(cfg, "red")
    assert not (tmp_path / "out").exists()


# ---------- config overrides / batch ----------


def _bare_config(tmp_path, pdb='pdb = "5XTD_FES_O301_SCH3.pdb"'):
    f = tmp_path / "bare.toml"
    f.write_text(f"""
base_dir = "{PLUGIN / "examples"}"
[structure]
{pdb}
cluster_only = true
[states.red]
[states.ox]
[scf]
engine = "cpu"
basis = "3-21g"
guesses = ["minao"]
[avas]
target_aos = "free-atom"
[output]
dir = "{tmp_path / "out"}"
work_dir = "{tmp_path / "work"}"
""")
    return f


def test_pdb_and_set_overrides(tmp_path):
    f = _bare_config(tmp_path, pdb="")
    with pytest.raises(ValueError, match="pdb is required"):
        pipeline.load_config(f)
    cfg = pipeline.load_config(
        f, pdb=SCH3, overrides=["scf.engine=gpu", "scf.max_cycle=7", 'scf.guesses=["atom"]']
    )
    assert cfg["structure"]["pdb"] == str(SCH3)
    assert (cfg["scf"]["engine"], cfg["scf"]["max_cycle"], cfg["scf"]["guesses"]) == (
        "gpu",
        7,
        ["atom"],
    )
    with pytest.raises(ValueError, match="bad override"):
        pipeline.load_config(f, pdb=SCH3, overrides=["scf.nope=1"])


def test_batch_config_loads():
    cfg = pipeline.load_config(PLUGIN / "configs" / "batch_dzvp.toml", pdb=SCH3)
    model, states = pipeline.prep(cfg)
    assert set(states) == {"ox", "red"}
    assert (states["ox"]["charge"], states["red"]["charge"]) == (-2, -3)


def test_label_start_lower_same_tag_is_written(tmp_path, monkeypatch):
    """Same rule as the SBD flow: the lowest converged start with the reference tag is written."""
    cfg = pipeline.load_config(_bare_config(tmp_path))
    real = pipeline.verify_mod.verify

    def rdm_start_higher(*args, **kwargs):
        out = real(*args, **kwargs)
        out["e_active_uhf"] += 1e-3  # pretend the RDM start stopped 1 mEh above the label start
        return out

    monkeypatch.setattr(pipeline.verify_mod, "verify", rdm_start_higher)
    path = pipeline.run_state(cfg, "red")
    m, files, labels, rdm = manifest.read_manifest(path)
    ref = m["bs_reference"]
    assert ref["reference_start"] == "labels"
    assert abs(ref["e_active_uhf"] - ref["label_start"]["e"]) < 1e-10
    assert np.allclose(ref["rdm_trace"], [12, 11], atol=1e-6)


# ---------- fast production mode: capping, patterns, atomic BS guess ----------


def _uncapped_excerpt(tmp_path):
    """Protonated excerpt with the cut-point H removed (backbone N / C under-valent)."""
    pdb = tmp_path / EXCERPT.name
    shutil.copy(EXCERPT, pdb)
    structure.prepare_model(pdb, protonate="builtin", sg_cutoff=3.8)
    prot = (tmp_path / f"{EXCERPT.stem}_prot.pdb").read_text().splitlines()
    cut = [ln for ln in prot if not (ln.startswith("ATOM") and ln[12:16] in (" HN1", " HC "))]
    out = tmp_path / "uncapped.pdb"
    out.write_text("\n".join(cut) + "\n")
    return out


def test_cap_cut_points(tmp_path):
    pdb = _uncapped_excerpt(tmp_path)
    m = structure.prepare_model(pdb, sg_cutoff=3.8)
    assert len([w for w in m.warnings if w.startswith("valence")]) == 8  # 4 N + 4 C
    capped = structure.prepare_model(pdb, protonate="cap", sg_cutoff=3.8)
    assert capped.protonation["tool"] == "cap" and capped.protonation["added_h"] == 8
    assert (tmp_path / "uncapped_cap.pdb").exists()
    assert not [w for w in capped.warnings if w.startswith("valence")]
    assert capped.ligand_charge == -4  # capped N / C are neutral
    # a complete excerpt is left as it is
    clean = structure.prepare_model(SCH3, protonate="cap")
    assert clean.protonation["tool"] == "none"


def test_prep_rejects_cut_points(tmp_path):
    cfg = pipeline.load_config(_bare_config(tmp_path), pdb=_uncapped_excerpt(tmp_path))
    cfg["structure"].update(cluster_only=False, sg_cutoff=3.8)
    with pytest.raises(ValueError, match="8 under-valent atoms"):
        pipeline.prep(cfg)
    cfg["structure"]["protonate"] = "cap"
    model, _ = pipeline.prep(cfg)
    assert model.protonation["added_h"] == 8


def test_bs_patterns_table_and_spin_mirrors():
    sc = {
        "patterns": {"Fe2S2": [[1, -1]], "Fe4S4": [[1, 1, -1, -1], [1, -1, 1, -1]]},
        "mirrors": {"Fe2S2": "spin", "Fe4S4": False},
    }
    tags = lambda *a: [bs_guess.tag_of(p) for p in pipeline.bs_patterns(sc, *a)]  # noqa: E731
    assert tags("Fe2S2", 2, 0) == ["01"]  # ms = 0: the mirror is the α↔β swap
    assert tags("Fe2S2", 2, 1) == ["01", "10"]  # ms ≠ 0: Fe(III) on the other Fe
    assert tags("Fe4S4", 4, 1) == ["0011", "0101"]
    assert len(pipeline.bs_patterns({"patterns": "auto", "mirrors": "spin"}, "Fe4S4", 4, 0)) == 3
    with pytest.raises(ValueError, match="patterns need"):
        pipeline.bs_patterns({"patterns": [[1, 1]], "mirrors": False}, "Fe2S2", 2, 0)


def _fast_config(tmp_path):
    f = _bare_config(tmp_path)
    f.write_text(
        f.read_text()
        .replace('basis = "3-21g"', 'basis = "dzvp"')
        .replace('target_aos = "free-atom"', 'target_aos = "file:AOs"')
        .replace(
            'guesses = ["minao"]',
            'guesses = ["minao"]\nbs_guess = "atomic"\nstability = "off"\n'
            'patterns = { Fe2S2 = [[1, -1]], Fe4S4 = [[1, 1, -1, -1]] }\nmirrors = "spin"',
        )
    )
    return pipeline.load_config(f)


def test_atomic_bs_guess_bare_fe2s2(tmp_path):
    """No high-spin SCF: both states reach their pattern from the atomic BS guess."""
    cfg = _fast_config(tmp_path)
    for state, tag in (("ox", "01"), ("red", None)):
        m, *_ = manifest.read_manifest(pipeline.run_state(cfg, state))
        e = m["electronic"]
        assert e["e_hs_uhf"] is None and e["bs_guess"] == "atomic"
        assert e["full_system_stability"] == "off"
        assert len(m["candidates"]) == (1 if state == "ox" else 2)
        assert all(
            c["gate"] and c["spin_tag"] == bs_guess.tag_of(c["spin_pattern"])
            for c in m["candidates"]
        )
        assert tag is None or m["bs_reference"]["spin_tag"] == tag
    assert not list((tmp_path / "work").glob("hs_*.chk"))


def test_atomic_bs_guess_falls_back_to_high_spin(tmp_path, monkeypatch):
    cfg = _fast_config(tmp_path)
    real = bs_guess.select
    calls = []

    def reject_atomic(solutions, selection="lowest"):
        calls.append(solutions[0].chkfile)
        if solutions[0].chkfile.endswith("_atomic.chk"):
            raise RuntimeError("pretend the atomic guess collapsed")
        return real(solutions, selection)

    monkeypatch.setattr(bs_guess, "select", reject_atomic)
    m, *_ = manifest.read_manifest(pipeline.run_state(cfg, "ox"))
    assert m["electronic"]["bs_guess"] == "high-spin spin flip"
    assert m["electronic"]["e_hs_uhf"] is not None
    assert len(calls) == 2 and list((tmp_path / "work").glob("hs_*.chk"))


def test_cap_keeps_bare_cys_sg_as_thiolate():
    """A Cys SG without H beyond sg_cutoff (drifted off Fe) is a thiolate, not a cut point."""
    m = structure.prepare_model(SCH3, protonate="cap", sg_cutoff=1.0)
    assert m.protonation["tool"] == "none" and m.ligand_charge == -4
    assert not [w for w in m.warnings if w.startswith("valence")]
    assert [w for w in m.warnings if w.startswith("Cys SG without H beyond")]
    # check mode still reports them
    m = structure.prepare_model(SCH3, protonate="check", sg_cutoff=1.0)
    assert [w for w in m.warnings if w.startswith("valence: CYS")]
