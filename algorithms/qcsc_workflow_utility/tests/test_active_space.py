"""Tests for qcsc_workflow_utility.active_space (AVAS manifest, schema v1).

Always-run tests use the tracked N2 18-orbital FCIDUMP with synthetic Fe2S2 / Fe4S4 labels and
RDMs written to tmp_path (the validation is about shapes and headers, not physics). The real
SCH3 fixture sets under plugins/bs_uno_avas/workspace/output/ are used only if present
(untracked).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from qcsc_workflow_utility.active_space import (
    RDM_TRACE_TOL,
    af_groups_from_labels,
    atom_groups,
    atom_spins,
    bs_density_from_labels,
    density_from_af_groups,
    fe_atoms,
    file_sha256,
    load_active_space_manifest,
    parse_label,
    read_fcidump_header,
    spin_pattern_from_spins,
    spin_tag_from_pattern,
    spin_tag_from_spins,
    validate_cluster_labels,
    validate_orbital_indices,
)

_REPO = Path(__file__).resolve().parents[3]
_N2_FCIDUMP = _REPO / "algorithms" / "sbd" / "data" / "fcidump_N2_MO.txt"
_SCH3_OUT = _REPO / "plugins" / "bs_uno_avas" / "workspace" / "output"

_FE_AO = ["3dxy", "3dyz", "3dz^2", "3dxz", "3dx2-y2"]
_S_AO = ["3px", "3py", "3pz"]


def _cluster_labels(n_fe: int) -> list[str]:
    """Block-order labels for Fe_nS_n: Fe 0..n-1 (5 x 3d), then S n..2n-1 (3 x 3p)."""
    labels = [f"{i}_Fe_{ao}" for i in range(n_fe) for ao in _FE_AO]
    labels += [f"{n_fe + j}_S_{ao}" for j in range(n_fe) for ao in _S_AO]
    return labels


def _write_fcidump(path: Path, norb: int, nelec: int, ms2: int) -> Path:
    """Header-only FCIDUMP (the loader reads only the namelist) plus one integral line."""
    path.write_text(
        f" &FCI NORB={norb:4d},NELEC={nelec},MS2={ms2},\n"
        f"  ORBSYM={'1,' * norb}\n  ISYM=1,\n &END\n"
        " 0.5    1    1    1    1\n"
    )
    return path


def _diag_rdm(norb: int, nalpha: int, nbeta: int) -> np.ndarray:
    rdm = np.zeros((2, norb, norb))
    rdm[0][np.arange(nalpha), np.arange(nalpha)] = 1.0
    rdm[1][np.arange(nbeta), np.arange(nbeta)] = 1.0
    return rdm


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_set(
    tmp_path: Path,
    *,
    n_fe: int = 2,
    nelec: int = 23,
    ms2: int = 1,
    labels: list[str] | None = None,
    pattern: list | None = None,
    tag: str | None = None,
    rdm: np.ndarray | None = None,
    fcidump: Path | None = None,
    manifest_extra: dict | None = None,
    drop: tuple[tuple[str, ...], ...] = (),
    with_sha: bool = True,
) -> Path:
    """Write a synthetic artifact set + manifest into tmp_path; return the manifest path."""
    labels = _cluster_labels(n_fe) if labels is None else labels
    norb = len(labels)
    nalpha, nbeta = (nelec + ms2) // 2, (nelec - ms2) // 2
    pattern = [1, -1] * (n_fe // 2) if pattern is None else pattern
    tag = spin_tag_from_pattern(pattern) if tag is None else tag
    fcidump = fcidump or _write_fcidump(tmp_path / "set_avas.fcidump", norb, nelec, ms2)
    lab_path = tmp_path / "set_active_orbitals_labels.npy"
    np.save(lab_path, np.array(labels, dtype="<U12"))
    rdm_path = tmp_path / "set_active_space_rdm.npy"
    np.save(rdm_path, _diag_rdm(norb, nalpha, nbeta) if rdm is None else rdm)

    def entry(p: Path) -> dict:
        name = p.name if p.parent == tmp_path else str(p)
        return {"name": name, "sha256": _sha(p)} if with_sha else {"name": name}

    manifest = {
        "schema_version": 1,
        "run_name": "synthetic",
        "structure": {"cluster_type": f"Fe{n_fe}S{n_fe}", "unknown_future_key": 1},
        "active_space": {"norb": norb, "nelec": nelec, "ms2": ms2},
        "bs_reference": {"spin_pattern": pattern, "spin_tag": tag, "e_active_uhf": -1.5},
        "files": {"fcidump": entry(fcidump), "labels": entry(lab_path), "rdm": entry(rdm_path)},
        "some_new_top_level_key": {"ignored": True},
    }
    for section, values in (manifest_extra or {}).items():
        if isinstance(values, dict) and isinstance(manifest.get(section), dict):
            manifest[section].update(values)
        else:
            manifest[section] = values
    for keys in drop:
        d = manifest
        for k in keys[:-1]:
            d = d[k]
        del d[keys[-1]]
    mpath = tmp_path / "set_manifest.json"
    mpath.write_text(json.dumps(manifest))
    return mpath


# ---------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, expected",
    [
        ("0_Fe_3dxz", (0, "Fe", "3dxz")),
        ("1_Fe_3dz^2", (1, "Fe", "3dz^2")),
        ("3_Fe_3dx2-y2", (3, "Fe", "3dx2-y2")),
        ("12_S_3py", (12, "S", "3py")),
    ],
)
def test_parse_label(label, expected):
    assert parse_label(label) == expected
    assert parse_label(np.str_(label)) == expected


@pytest.mark.parametrize("label", ["Fe_3dxz", "0-Fe-3dxz", "0_fe_3dxz", "x_Fe_3dxz", "0_Fe_", ""])
def test_parse_label_rejects(label):
    with pytest.raises(ValueError, match="bad active-orbital label"):
        parse_label(label)


def test_atom_groups_block_order():
    groups = atom_groups(_cluster_labels(2))
    assert list(groups) == ["0_Fe", "1_Fe", "2_S", "3_S"]
    assert groups["0_Fe"] == [0, 1, 2, 3, 4]
    assert groups["3_S"] == [13, 14, 15]
    assert fe_atoms(_cluster_labels(4)) == ["0_Fe", "1_Fe", "2_Fe", "3_Fe"]


def test_atom_groups_is_order_agnostic():
    """Grouping is by label: a permuted order gives the same atoms with permuted indices."""
    labels = _cluster_labels(2)
    perm = np.random.default_rng(0).permutation(len(labels))
    shuffled = [labels[i] for i in perm]
    groups = atom_groups(shuffled)
    assert list(groups) == ["0_Fe", "1_Fe", "2_S", "3_S"]
    for key, orbs in groups.items():
        expected = sorted(lab for lab in labels if lab.startswith(key + "_"))
        assert sorted(shuffled[i] for i in orbs) == expected
    validate_cluster_labels(shuffled, 16)


def test_atom_groups_rejects_conflicting_element():
    with pytest.raises(ValueError, match="labelled both"):
        atom_groups(["0_Fe_3dxy", "0_S_3px"])


@pytest.mark.parametrize("n_fe", [2, 4])
def test_validate_cluster_labels_ok(n_fe):
    labels = _cluster_labels(n_fe)
    assert len(labels) == 8 * n_fe
    validate_cluster_labels(labels, 8 * n_fe)


@pytest.mark.parametrize(
    "labels, norb, match",
    [
        (_cluster_labels(2), 18, "16 labels for NORB=18"),
        (_cluster_labels(2)[:-1] + ["0_Fe_3dxy"], None, "duplicate"),
        (_cluster_labels(2)[:-1], None, r"atom 3_S: 2 active orbitals, expected 3"),
        (_cluster_labels(2)[1:], None, r"atom 0_Fe: 4 active orbitals, expected 5"),
        (_cluster_labels(2) + ["4_C_2px"], None, "element 'C' not allowed"),
        (_cluster_labels(2)[:13], None, "1 S atoms; expected Fe2S2 or Fe4S4"),
        (
            [f"{i}_Fe_{ao}" for i in range(3) for ao in _FE_AO]
            + [f"{3 + j}_S_{ao}" for j in range(3) for ao in _S_AO],
            None,
            "3 Fe and 3 S",
        ),
    ],
)
def test_validate_cluster_labels_rejects(labels, norb, match):
    with pytest.raises(ValueError, match=match):
        validate_cluster_labels(labels, norb)


# ---------------------------------------------------------------------------------------------
# Spin patterns, af_groups, guess densities
# ---------------------------------------------------------------------------------------------


def test_spin_tag_from_pattern():
    assert spin_tag_from_pattern([1, -1]) == "01"
    assert spin_tag_from_pattern([-1, 1, 1, -1]) == "1001"
    assert spin_tag_from_pattern([1, 0]) == "0x"
    with pytest.raises(ValueError):
        spin_tag_from_pattern([2, -1])


def test_af_groups_from_labels_shape():
    """Same dict shape as chem._parse_af_groups: fragment -> orbitals, plus up/down/pol."""
    labels = _cluster_labels(2)
    groups = af_groups_from_labels(labels, [-1, 1], pol=0.6)
    assert groups["up"] == ["1_Fe"] and groups["down"] == ["0_Fe"]
    assert groups["pol"] == 0.6
    assert {k for k in groups if k not in ("up", "down", "pol")} == {"0_Fe", "1_Fe", "2_S", "3_S"}
    assert groups["2_S"] == [10, 11, 12]
    validate_orbital_indices(groups, 16)


@pytest.mark.parametrize(
    "pattern, match",
    [
        ([1], "one entry per Fe"),
        ([1, -1, 1], "one entry per Fe"),
        ([1, 0], r"\+1 or -1"),
        ([1, 2], r"\+1 or -1"),
        ([1.0, -1], r"\+1 or -1"),
        ([True, -1], r"\+1 or -1"),
    ],
)
def test_af_groups_from_labels_rejects_pattern(pattern, match):
    with pytest.raises(ValueError, match=match):
        af_groups_from_labels(_cluster_labels(2), pattern)


def test_bs_density_from_labels_matches_verify_fill():
    """Fe(+1) 3d fully alpha, Fe(-1) 3d fully beta, S 3p doubly occupied; works for any order."""
    labels = _cluster_labels(2)
    dm = bs_density_from_labels(labels, [-1, 1])
    assert dm.shape == (2, 16, 16)
    assert np.allclose(np.diag(dm[0])[:5], 0.0) and np.allclose(np.diag(dm[1])[:5], 1.0)
    assert np.allclose(np.diag(dm[0])[5:10], 1.0) and np.allclose(np.diag(dm[1])[5:10], 0.0)
    assert np.allclose(np.diag(dm[0])[10:], 1.0) and np.allclose(np.diag(dm[1])[10:], 1.0)
    assert not np.any(dm - np.einsum("sii->si", dm)[:, :, None] * np.eye(16))  # diagonal only

    perm = np.random.default_rng(1).permutation(16)
    dm_p = bs_density_from_labels([labels[i] for i in perm], [-1, 1])
    assert np.allclose(np.einsum("sii->si", dm_p), np.einsum("sii->si", dm)[:, perm])


def test_density_from_af_groups_pol_and_free():
    groups = {
        "a": [0, 1],
        "b": [2],
        "s": [3],
        "c": [4],
        "up": ["a"],
        "down": ["b"],
        "free": ["s"],
        "pol": 0.6,
    }
    d = np.einsum("sii->si", density_from_af_groups(groups, 6))
    assert np.allclose(d[:, 0], [0.8, 0.2]) and np.allclose(d[:, 2], [0.2, 0.8])
    assert np.allclose(d[:, 3], [0.5, 0.5]) and np.allclose(d[:, 4], [1.0, 1.0])
    assert np.allclose(d[:, 5], [0.0, 0.0])  # in no fragment


def test_atom_spins_and_tag():
    labels = _cluster_labels(2)
    dm = bs_density_from_labels(labels, [-1, 1])
    spins = atom_spins(dm[0], dm[1], labels)
    assert spins == {"0_Fe": -5.0, "1_Fe": 5.0, "2_S": 0.0, "3_S": 0.0}
    assert spin_pattern_from_spins(spins) == [-1, 1]
    assert spin_tag_from_spins(spins) == "10"
    assert spin_tag_from_spins({"0_Fe": 0.4, "1_Fe": -3.9, "2_S": 0.1}) == "x1"
    # model order, not dict order
    assert spin_tag_from_spins({"1_Fe": 3.9, "0_Fe": -3.8}) == "10"


# ---------------------------------------------------------------------------------------------
# Bounds guard (FE4S4_AF_GROUPS preset on a 16-orbital FCIDUMP)
# ---------------------------------------------------------------------------------------------


def _fe4s4_preset() -> dict:
    """Mirror of the 36-orbital FE4S4_AF_GROUPS='fe4s4' preset in chem._parse_af_groups."""
    return {
        "l1": list(range(0, 2)),
        "fe1": list(range(2, 7)),
        "fe2": list(range(7, 12)),
        "s": list(range(12, 24)),
        "fe3": list(range(24, 29)),
        "fe4": list(range(29, 34)),
        "l2": list(range(34, 36)),
        "up": ["fe1", "fe2"],
        "down": ["fe3", "fe4"],
        "pol": 1.0,
    }


def test_validate_orbital_indices_preset_on_36_ok():
    validate_orbital_indices(_fe4s4_preset(), 36)


@pytest.mark.parametrize("norb", [16, 18, 32])
def test_validate_orbital_indices_preset_out_of_range(norb):
    with pytest.raises(ValueError, match=rf"out of range for NORB={norb}"):
        validate_orbital_indices(_fe4s4_preset(), norb, source="FE4S4_AF_GROUPS")


@pytest.mark.parametrize("orbs", [[-1, 2], [0, "1"], [0, 1.5], "012", [True]])
def test_validate_orbital_indices_bad_entries(orbs):
    with pytest.raises(ValueError):
        validate_orbital_indices({"frag": orbs, "up": ["frag"]}, 16)


def test_density_from_af_groups_guards_bounds():
    with pytest.raises(ValueError, match="out of range"):
        density_from_af_groups(_fe4s4_preset(), 16)


def test_parse_af_groups_preset_matches_mirror(monkeypatch):
    """The test mirror above stays in sync with chem's 'fe4s4' preset."""
    from qcsc_workflow_utility import chem

    monkeypatch.setenv("FE4S4_AF_GROUPS", "fe4s4")
    monkeypatch.delenv("FE4S4_AF_POL", raising=False)
    monkeypatch.delenv("FE4S4_AF_FREE_S", raising=False)
    assert chem._parse_af_groups() == _fe4s4_preset()


@pytest.fixture
def _no_uhf_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SBD_UHF_CACHE", "0")
    monkeypatch.setenv("SBD_UHF_CACHE_DIR", str(tmp_path / "uhf_cache"))


def test_fe4s4_preset_on_small_fcidump_raises_value_error(monkeypatch, _no_uhf_cache):
    """'fe4s4' preset (36 orbitals) on the N2 18-orbital FCIDUMP -> clear ValueError
    before any SCF, instead of an uncaught IndexError from the guess fill."""
    from qcsc_workflow_utility import chem

    monkeypatch.setenv("FE4S4_AF_GROUPS", "fe4s4")
    match = r"preset 'fe4s4' expects 36 orbitals; FCIDUMP has NORB=18.*active_space_manifest"
    with pytest.raises(ValueError, match=match):
        chem.compute_molecular_integrals_from_fcidump.fn(str(_N2_FCIDUMP), unrestricted=True)


def test_custom_af_groups_out_of_range_raises(monkeypatch, _no_uhf_cache):
    from qcsc_workflow_utility import chem

    spec = {"a": [0, 1, 2], "b": [17, 18], "up": ["a"], "down": ["b"]}
    monkeypatch.setenv("FE4S4_AF_GROUPS", json.dumps(spec))
    with pytest.raises(ValueError, match=r"fragment 'b' orbital indices \[18\] out of range"):
        chem.compute_molecular_integrals_from_fcidump.fn(str(_N2_FCIDUMP), unrestricted=True)


def test_af_groups_non_object_json_raises(monkeypatch):
    from qcsc_workflow_utility import chem

    monkeypatch.setenv("FE4S4_AF_GROUPS", "[1, 2]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        chem._parse_af_groups()


def test_converge_bs_uhf_guards_before_scf():
    """The guard in _converge_broken_symmetry_uhf fires before the first kernel() call."""
    from pyscf import gto, scf
    from qcsc_workflow_utility import chem

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.UHF(mol)
    mf.kernel = lambda *a, **k: pytest.fail("SCF ran before the af_groups bounds check")
    with pytest.raises(ValueError, match="out of range for NORB=2"):
        chem._converge_broken_symmetry_uhf(mf, af_groups={"x": [0, 2], "up": ["x"]})


def test_in_range_af_groups_still_runs():
    """An in-range spec takes the atom-localized guess path as before (stretched H2 -> BS)."""
    from pyscf import gto, scf
    from qcsc_workflow_utility import chem

    mol = gto.M(atom="H 0 0 0; H 0 0 3.0", basis="sto-3g", verbose=0)
    mf = chem._converge_broken_symmetry_uhf(
        scf.UHF(mol), af_groups={"h1": [0], "h2": [1], "up": ["h1"], "down": ["h2"]}
    )
    assert mf.converged
    assert mf.spin_square()[0] > 0.9  # broken-symmetry singlet


# ---------------------------------------------------------------------------------------------
# FCIDUMP header + hashing
# ---------------------------------------------------------------------------------------------


def test_read_fcidump_header_n2():
    assert read_fcidump_header(_N2_FCIDUMP) == {"NORB": 18, "NELEC": 14, "MS2": 0}


def test_read_fcidump_header_variants(tmp_path):
    p = tmp_path / "a.fcidump"
    p.write_text("&FCI NORB=4, NELEC=3, ORBSYM=1,1,1,1, ISYM=1 /\n 0.1 1 1 1 1\n")
    assert read_fcidump_header(p) == {"NORB": 4, "NELEC": 3, "MS2": 0}
    p.write_text("not an fcidump\n")
    with pytest.raises(ValueError, match="no &FCI"):
        read_fcidump_header(p)


def test_file_sha256(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc" * 1000)
    assert file_sha256(p, chunk_size=7) == hashlib.sha256(b"abc" * 1000).hexdigest()


# ---------------------------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_fe, nelec, ms2", [(2, 23, 1), (2, 22, 0), (4, 46, 0)])
def test_load_manifest_ok(tmp_path, n_fe, nelec, ms2):
    spec = load_active_space_manifest(_write_set(tmp_path, n_fe=n_fe, nelec=nelec, ms2=ms2))
    assert spec.norb == 8 * n_fe and spec.nelec == nelec and spec.ms2 == ms2
    assert (spec.nalpha, spec.nbeta) == ((nelec + ms2) // 2, (nelec - ms2) // 2)
    assert spec.rdm.shape == (2, spec.norb, spec.norb) and spec.rdm.dtype == np.float64
    assert spec.labels.tolist() == _cluster_labels(n_fe)
    assert spec.spin_tag == spin_tag_from_pattern(spec.spin_pattern)
    assert spec.e_active_uhf == -1.5
    assert spec.run_name == "synthetic" and spec.cluster_type == f"Fe{n_fe}S{n_fe}"
    assert spec.fcidump_sha256 == _sha(tmp_path / "set_avas.fcidump")
    assert len(spec.cache_token) == 64


def test_load_manifest_interleaved_labels_ok(tmp_path):
    labels = _cluster_labels(2)
    perm = np.random.default_rng(2).permutation(16)
    spec = load_active_space_manifest(_write_set(tmp_path, labels=[labels[i] for i in perm]))
    assert spec.norb == 16


def test_cache_token_content_sensitive(tmp_path):
    def load(name, **kwargs):
        d = tmp_path / name
        d.mkdir()
        return load_active_space_manifest(_write_set(d, **kwargs))

    rdm = _diag_rdm(16, 12, 11)
    rdm[:, 0, 0] -= 1e-3
    a, b, c, d = load("a"), load("b", rdm=rdm), load("c", pattern=[-1, 1]), load("d")
    assert a.cache_token == d.cache_token  # same contents, different directory
    assert len({a.cache_token, b.cache_token, c.cache_token}) == 3


def test_load_manifest_real_n2_fcidump_header_mismatch(tmp_path):
    """The N2 FCIDUMP (NORB=18) referenced from a 16-label Fe2S2 manifest is rejected."""
    match = r"FCIDUMP header NORB/NELEC/MS2=18/14/0 != manifest 16/23/1"
    with pytest.raises(ValueError, match=match):
        load_active_space_manifest(_write_set(tmp_path, fcidump=_N2_FCIDUMP))


@pytest.mark.parametrize(
    "drop, match",
    [
        ((("schema_version",),), r"schema_version"),
        ((("active_space", "ms2"),), r"active_space\.ms2"),
        ((("bs_reference", "spin_tag"),), r"bs_reference\.spin_tag"),
        ((("files", "rdm"),), r"files\.rdm"),
    ],
)
def test_load_manifest_missing_required(tmp_path, drop, match):
    with pytest.raises(ValueError, match=f"missing required keys.*{match}"):
        load_active_space_manifest(_write_set(tmp_path, drop=drop))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"manifest_extra": {"schema_version": 2}}, "schema_version 2 unsupported"),
        ({"manifest_extra": {"active_space": {"nalpha": 11}}}, r"active_space\.nalpha=11"),
        ({"manifest_extra": {"active_space": {"ms2": 2}}}, "inconsistent active_space"),
        ({"pattern": [1, -1, 1]}, "one entry per Fe"),
        ({"pattern": [1, 0]}, r"\+1 or -1"),
        ({"tag": "10"}, "spin_tag '10' inconsistent with spin_pattern"),
        ({"rdm": np.zeros((2, 15, 15))}, r"rdm shape \(2, 15, 15\)"),
        ({"rdm": _diag_rdm(16, 11, 11)}, "rdm traces"),
        ({"labels": _cluster_labels(2)[:-1] + ["0_Fe_3dxy"]}, "labels: duplicate"),
    ],
)
def test_load_manifest_rejects(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        load_active_space_manifest(_write_set(tmp_path, **kwargs))


def test_load_manifest_label_count_mismatch(tmp_path):
    """15 labels against a NORB=16 FCIDUMP/manifest: the label-count check fires."""
    fcidump = _write_fcidump(tmp_path / "x.fcidump", 16, 23, 1)
    mpath = _write_set(tmp_path, labels=_cluster_labels(2)[:15], fcidump=fcidump)
    m = json.loads(mpath.read_text())
    m["active_space"]["norb"] = 16
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="15 labels for NORB=16"):
        load_active_space_manifest(mpath)


def test_load_manifest_asymmetric_rdm(tmp_path):
    rdm = _diag_rdm(16, 12, 11)
    rdm[0, 0, 1] = 0.1
    with pytest.raises(ValueError, match="not symmetric"):
        load_active_space_manifest(_write_set(tmp_path, rdm=rdm))


def test_load_manifest_projected_rdm_trace_tolerance(tmp_path):
    """Traces off by < RDM_TRACE_TOL (projected full-system RDM) pass; beyond it they fail."""
    rdm = _diag_rdm(16, 12, 11)
    rdm[:, 10, 10] -= 0.5 * RDM_TRACE_TOL  # like Tr = 11.9949 / 10.9949
    load_active_space_manifest(_write_set(tmp_path, rdm=rdm))
    rdm[:, 10, 10] -= RDM_TRACE_TOL
    with pytest.raises(ValueError, match="rdm traces"):
        load_active_space_manifest(_write_set(tmp_path, rdm=rdm))


def test_load_manifest_missing_file_and_bad_json(tmp_path):
    with pytest.raises(ValueError, match="manifest not found"):
        load_active_space_manifest(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_active_space_manifest(tmp_path / "bad.json")
    mpath = _write_set(tmp_path)
    (tmp_path / "set_active_space_rdm.npy").unlink()
    with pytest.raises(ValueError, match="RDM file not found"):
        load_active_space_manifest(mpath)


def test_load_manifest_sha_mismatch(tmp_path, monkeypatch):
    mpath = _write_set(tmp_path)
    np.save(
        tmp_path / "set_active_orbitals_labels.npy",
        np.array(list(reversed(_cluster_labels(2))), dtype="<U12"),
    )
    monkeypatch.delenv("SBD_SKIP_MANIFEST_HASH", raising=False)
    with pytest.raises(ValueError, match=r"files\.labels sha256 mismatch"):
        load_active_space_manifest(mpath)
    monkeypatch.setenv("SBD_SKIP_MANIFEST_HASH", "1")
    with pytest.warns(RuntimeWarning, match=r"files\.labels sha256 mismatch"):
        spec = load_active_space_manifest(mpath)
    assert spec.labels[0] == "3_S_3pz"


def test_load_manifest_without_sha_warns(tmp_path):
    with pytest.warns(RuntimeWarning, match="has no sha256"):
        load_active_space_manifest(_write_set(tmp_path, with_sha=False))


def test_load_manifest_fcidump_override(tmp_path):
    mpath = _write_set(tmp_path)
    copy = tmp_path / "elsewhere" / "copy.fcidump"
    copy.parent.mkdir()
    shutil.copy(tmp_path / "set_avas.fcidump", copy)
    (tmp_path / "set_avas.fcidump").unlink()  # override works even if the original is gone
    spec = load_active_space_manifest(mpath, fcidump_override=copy)
    assert spec.fcidump == copy.resolve()

    other = _write_fcidump(tmp_path / "other.fcidump", 16, 23, 1)
    other.write_text(other.read_text() + " 0.25 2 2 2 2\n")
    with pytest.raises(ValueError, match="does not match the manifest FCIDUMP"):
        load_active_space_manifest(mpath, fcidump_override=other)


# ---------------------------------------------------------------------------------------------
# Real SCH3 fixture sets (untracked; skipped if absent)
# ---------------------------------------------------------------------------------------------

_SCH3_SETS = [
    ("N1a_State1_Eox_Rox_S01", 22, 0, [1, -1], "01"),
    ("N1a_State1_Ered_Rox_S10", 23, 1, [-1, 1], "10"),
]


@pytest.mark.parametrize("prefix, nelec, ms2, pattern, tag", _SCH3_SETS)
def test_load_manifest_sch3_fixture(tmp_path, prefix, nelec, ms2, pattern, tag):
    files = {
        k: _SCH3_OUT / f"{prefix}_{suffix}"
        for k, suffix in (
            ("fcidump", "avas.fcidump"),
            ("labels", "active_orbitals_labels.npy"),
            ("rdm", "active_space_rdm.npy"),
        )
    }
    if not all(p.is_file() for p in files.values()):
        pytest.skip(f"fixture set {prefix} not present")
    manifest = {
        "schema_version": 1,
        "run_name": prefix,
        "active_space": {"norb": 16, "nelec": nelec, "ms2": ms2},
        "bs_reference": {"spin_pattern": pattern, "spin_tag": tag},
        "files": {k: {"name": str(p), "sha256": _sha(p)} for k, p in files.items()},
    }
    mpath = tmp_path / f"{prefix}_manifest.json"
    mpath.write_text(json.dumps(manifest))
    spec = load_active_space_manifest(mpath)
    assert (spec.norb, spec.nelec, spec.ms2) == (16, nelec, ms2)
    assert fe_atoms(spec.labels.tolist()) == ["0_Fe", "1_Fe"]
    # the stored RDM is the converged BS-UHF: its own spins give the manifest tag
    assert spin_tag_from_spins(atom_spins(spec.rdm[0], spec.rdm[1], spec.labels.tolist())) == tag
