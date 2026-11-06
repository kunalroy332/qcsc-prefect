"""BS-UHF reference from an AVAS active-space manifest + content-based UHF cache.

Physics tests reproduce the converged active-space BS-UHF of the two SCH3 fixture sets
(plugins/bs_uno_avas/workspace/output/N1a_State1_*, untracked; skipped when absent) from their
manifest RDM and from the label-based guess, plus a synthetic "projected" RDM (perturbed,
non-integer trace) that must converge back to the same state. Each is a 16-orbital UHF (seconds).
UCCSD is never run on the Fe-S sets: the task-wiring tests replace the (U)CCSD property builders
(``_no_cc``).

Selection / strict-mode logic is tested with a mocked converger so it is deterministic.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import qcsc_workflow_utility.chem as chem
from pyscf import scf, tools
from qcsc_workflow_utility.active_space import (
    ActiveSpaceSpec,
    load_active_space_manifest,
)

_REPO = Path(__file__).resolve().parents[3]
_OUT = _REPO / "plugins" / "bs_uno_avas" / "workspace" / "output"

# (prefix, nelec, ms2, pattern, tag, e_active_uhf), measured 2026-10-31
_SETS = {
    "ox": ("N1a_State1_Eox_Rox_S01", 22, 0, [1, -1], "01", -5067.870133),
    "red": ("N1a_State1_Ered_Rox_S10", 23, 1, [-1, 1], "10", -5067.633303),
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(chem, "get_run_logger", lambda: logging.getLogger("test"))
    for var in ("SBD_BS_STRICT", "SBD_BS_E_TOL", "SBD_SKIP_MANIFEST_HASH", "FE4S4_AF_GROUPS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SBD_UHF_CACHE", "0")
    # SBD_UHF_CACHE=0 skips the lookup but still writes; keep writes out of ~/.cache
    monkeypatch.setenv("SBD_UHF_CACHE_DIR", str(tmp_path / "uhf_cache"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _files(prefix: str) -> dict[str, Path]:
    return {
        "fcidump": _OUT / f"{prefix}_avas.fcidump",
        "labels": _OUT / f"{prefix}_active_orbitals_labels.npy",
        "rdm": _OUT / f"{prefix}_active_space_rdm.npy",
    }


def _manifest(tmp_path: Path, key: str, *, rdm: np.ndarray | None = None, **ref) -> Path:
    """Test-only manifest (schema v1 required keys) for a fixture set; skip if absent."""
    prefix, nelec, ms2, pattern, tag, e_ref = _SETS[key]
    files = _files(prefix)
    if not all(p.is_file() for p in files.values()):
        pytest.skip(f"fixture set {prefix} not present")
    if rdm is not None:
        files["rdm"] = tmp_path / f"{prefix}_perturbed_rdm.npy"
        np.save(files["rdm"], rdm)
    bs_ref = {"spin_pattern": pattern, "spin_tag": tag, "e_active_uhf": e_ref}
    bs_ref.update(ref)
    manifest = {
        "schema_version": 1,
        "run_name": prefix,
        "active_space": {"norb": 16, "nelec": nelec, "ms2": ms2},
        "bs_reference": bs_ref,
        "files": {k: {"name": str(p), "sha256": _sha(p)} for k, p in files.items()},
    }
    path = tmp_path / f"{prefix}_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _plain_uhf(fcidump: Path) -> scf.uhf.UHF:
    """The same plain UHF the FCIDUMP task builds before choosing a reference."""
    with redirect_stdout(io.StringIO()):
        mf = tools.fcidump.to_scf(str(fcidump))
    nuc = mf.mol.energy_nuc()
    mol = mf.mol
    if mol.symmetry:
        mol.symmetry = False
        mol.build(dump_input=False, parse_arg=False)
    hcore, ovlp, eri = mf.get_hcore(), mf.get_ovlp(), mf._eri
    out = scf.UHF(mol)
    out.get_hcore = lambda *a, **k: hcore
    out.get_ovlp = lambda *a, **k: ovlp
    out._eri = eri
    out.mol.energy_nuc = lambda *a: nuc
    out.verbose = 0
    return out


def _perturbed(rdm: np.ndarray, seed: int = 0, trace_loss: float = 5e-3) -> np.ndarray:
    """Mimic a projected full-system RDM: symmetric off-diagonal noise, and each spin block
    scaled so its trace is ``trace_loss`` below the integer (as Tr = 11.9949 / 10.9949)."""
    noise = np.random.default_rng(seed).normal(scale=2e-3, size=rdm.shape)
    noise = noise + noise.transpose(0, 2, 1)
    noise[:, np.arange(rdm.shape[1]), np.arange(rdm.shape[1])] = 0.0
    out = rdm + noise
    for s in range(2):
        n = round(np.trace(rdm[s]))
        out[s] *= (n - trace_loss) / np.trace(out[s])
    return out


# ---------------------------------------------------------------------------------------------
# Physics on the fixture sets
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["ox", "red"])
def test_rdm_start_reproduces_fixture(tmp_path, key):
    _, _, _, _, tag, e_ref = _SETS[key]
    spec = load_active_space_manifest(_manifest(tmp_path, key))
    mf, info = chem._converge_reference_uhf(_plain_uhf(spec.fcidump), spec)
    assert info["source"] == "manifest-rdm"
    assert info["tag"] == tag
    assert abs(info["energy"] - e_ref) < 1e-5
    assert abs(mf.e_tot - e_ref) < 1e-5 and mf.converged
    assert type(mf) is scf.uhf.UHF  # plain UHF handed to _build_property_uhf, not SOSCF
    fe = [v for k, v in info["atom_spins"].items() if k.endswith("_Fe")]
    assert all(abs(s) > 3 for s in fe)
    by_source = {c["source"]: c for c in info["candidates"]}
    # the label-based guess agrees (same state, same energy)
    assert by_source["manifest-labels"]["tag"] == tag
    assert abs(by_source["manifest-labels"]["energy"] - e_ref) < 1e-5


def test_projected_like_rdm_converges_to_same_state(tmp_path):
    """Non-integer-trace (projected) RDM passes the 1e-2 trace tolerance and the RDM start still
    converges to the fixture state."""
    prefix = _SETS["red"][0]
    if not _files(prefix)["rdm"].is_file():
        pytest.skip("fixture set not present")
    exact = np.load(_files(prefix)["rdm"])
    rdm = _perturbed(exact)
    assert np.allclose([np.trace(rdm[0]), np.trace(rdm[1])], [12 - 5e-3, 11 - 5e-3])
    assert np.abs(rdm - exact).max() > 1e-3  # genuinely different from the converged RDM
    spec = load_active_space_manifest(_manifest(tmp_path, "red", rdm=rdm))
    mf0 = _plain_uhf(spec.fcidump)
    # E(rdm) of a non-idempotent, trace-deficient density is not variational (may lie on either
    # side); what matters is that the start is not the converged point.
    assert abs(mf0.energy_tot(dm=(rdm[0], rdm[1])) - _SETS["red"][5]) > 1e-3
    _, info = chem._converge_reference_uhf(mf0, spec)
    assert info["source"] == "manifest-rdm" and info["tag"] == "10"
    assert abs(info["energy"] - _SETS["red"][5]) < 1e-5


def test_wrong_manifest_tag_warns_then_strict_raises(tmp_path, monkeypatch):
    """Manifest claims pattern [1, -1] / tag '01' for the red set, whose RDM is the '10' state.

    The RDM start converges to '10' (rejected: wrong tag); the label start reaches the higher
    '01' minimum (-5067.551131, measured 2026-10-31), which is used -- and misses e_active_uhf by
    82 mEh -> loud warning; SBD_BS_STRICT=1 -> RuntimeError.
    """
    prefix = _SETS["red"][0]
    if not _files(prefix)["rdm"].is_file():
        pytest.skip("fixture set not present")
    path = _manifest(tmp_path, "red", spin_pattern=[1, -1], spin_tag="01")
    spec = load_active_space_manifest(path)
    with pytest.warns(RuntimeWarning, match="differs from manifest e_active_uhf"):
        _, info = chem._converge_reference_uhf(_plain_uhf(spec.fcidump), spec)
    by_source = {c["source"]: c for c in info["candidates"]}
    assert by_source["manifest-rdm"]["tag"] == "10"
    assert info["source"] == "manifest-labels" and info["tag"] == "01"
    assert abs(info["energy"] - (-5067.551131)) < 1e-5
    monkeypatch.setenv("SBD_BS_STRICT", "1")
    with pytest.raises(RuntimeError, match="SBD_BS_STRICT=1"):
        chem._converge_reference_uhf(_plain_uhf(spec.fcidump), spec)


def test_energy_mismatch_warns_and_tolerance_env(tmp_path, monkeypatch):
    spec = load_active_space_manifest(_manifest(tmp_path, "red", e_active_uhf=-5067.6))
    with pytest.warns(RuntimeWarning, match="differs from manifest e_active_uhf"):
        chem._converge_reference_uhf(_plain_uhf(spec.fcidump), spec)
    monkeypatch.setenv("SBD_BS_E_TOL", "0.1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        chem._converge_reference_uhf(_plain_uhf(spec.fcidump), spec)


def test_nelec_mismatch_raises(tmp_path):
    spec = load_active_space_manifest(_manifest(tmp_path, "red"))
    ox_fcidump = _files(_SETS["ox"][0])["fcidump"]
    if not ox_fcidump.is_file():
        pytest.skip("fixture set not present")
    with pytest.raises(ValueError, match=r"\(n_alpha, n_beta\)=\(11, 11\).*\(12, 11\)"):
        chem._converge_reference_uhf(_plain_uhf(ox_fcidump), spec)


# ---------------------------------------------------------------------------------------------
# FCIDUMP task wiring (UCCSD stubbed out: _build_property_uhf is replaced)
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def _no_cc(monkeypatch):
    """Replace _build_property_uhf / _build_property by a recorder (no (U)CCSD)."""
    calls = {}

    def fake_uhf(mf, norb, spin_sq, buf, reference=None):
        calls.update(kind="uhf", e=float(mf.e_tot), norb=norb, reference=reference, mf=mf)
        return "props"

    def fake_rhf(mf, norb, spin_sq, buf):
        calls.update(kind="rhf", e=float(mf.e_tot), norb=norb)
        return "props"

    monkeypatch.setattr(chem, "_build_property_uhf", fake_uhf)
    monkeypatch.setattr(chem, "_build_property", fake_rhf)
    return calls


def test_task_uhf_with_manifest_uses_reference(tmp_path, monkeypatch, _no_cc):
    path = _manifest(tmp_path, "red")
    # FE4S4_AF_GROUPS (36-orbital preset) is ignored on the manifest path: no ValueError
    monkeypatch.setenv("FE4S4_AF_GROUPS", "fe4s4")
    chem.compute_molecular_integrals_from_fcidump.fn(
        None, unrestricted=True, active_space_manifest=str(path)
    )
    assert _no_cc["kind"] == "uhf" and _no_cc["norb"] == 16
    assert abs(_no_cc["e"] - _SETS["red"][5]) < 1e-5
    ref = _no_cc["reference"]
    assert ref["reference_tag"] == "10" and ref["reference_source"] == "manifest-rdm"
    assert ref["fcidump_sha256"] == _sha(_files(_SETS["red"][0])["fcidump"])


def test_task_fcidump_must_match_manifest(tmp_path, _no_cc):
    path = _manifest(tmp_path, "red")
    other = _files(_SETS["ox"][0])["fcidump"]
    with pytest.raises(ValueError, match="does not match the manifest FCIDUMP"):
        chem.compute_molecular_integrals_from_fcidump.fn(
            str(other), unrestricted=True, active_space_manifest=str(path)
        )
    # the identical file is accepted
    same = _files(_SETS["red"][0])["fcidump"]
    chem.compute_molecular_integrals_from_fcidump.fn(
        str(same), unrestricted=True, active_space_manifest=str(path)
    )
    assert _no_cc["reference"]["reference_tag"] == "10"


def test_task_rhf_with_manifest_warns_and_uses_its_fcidump(tmp_path, _no_cc):
    path = _manifest(tmp_path, "red")
    with pytest.warns(RuntimeWarning, match="ignored for RHF"):
        chem.compute_molecular_integrals_from_fcidump.fn(
            None, unrestricted=False, active_space_manifest=str(path)
        )
    assert _no_cc["kind"] == "rhf" and _no_cc["norb"] == 16


def test_task_needs_fcidump_or_manifest():
    with pytest.raises(ValueError, match="needs fcidump_file or a manifest"):
        chem.compute_molecular_integrals_from_fcidump.fn(None, unrestricted=True)


# ---------------------------------------------------------------------------------------------
# Selection logic (mocked converger; no fixtures needed)
# ---------------------------------------------------------------------------------------------


class _FakeMF:
    def __init__(self, e, tag, converged=True):
        self.e_tot, self.converged = e, converged
        # 2 Fe (5 orbitals each) + 2 S (3 each): alpha/beta diagonal giving the requested tag
        sa = {"0": (1.0, 0.0), "1": (0.0, 1.0), "x": (0.5, 0.5)}
        a = np.concatenate([np.full(5, sa[c][0]) for c in tag] + [np.ones(6)])
        b = np.concatenate([np.full(5, sa[c][1]) for c in tag] + [np.ones(6)])
        self._dm = (np.diag(a), np.diag(b))
        self.mo_coeff = (np.eye(16), np.eye(16))
        self.mo_occ = (np.zeros(16), np.zeros(16))
        self.mo_energy = (np.zeros(16), np.zeros(16))

    def make_rdm1(self):
        return self._dm

    def spin_square(self):
        return (4.5, 4.4)


def _fake_spec(tag="10", e_ref=-1.0) -> ActiveSpaceSpec:
    labels = np.array(
        [f"{i}_Fe_3d{c}" for i in range(2) for c in "abcde"]
        + [f"{2 + j}_S_3p{c}" for j in range(2) for c in "xyz"]
    )
    pattern = tuple({"0": 1, "1": -1}[c] for c in tag)
    return ActiveSpaceSpec(
        manifest_path=Path("/fake/manifest.json"),
        fcidump=Path("/fake/x.fcidump"),
        labels=labels,
        rdm=np.zeros((2, 16, 16)),
        norb=16,
        nelec=23,
        ms2=1,
        spin_pattern=pattern,
        spin_tag=tag,
        e_active_uhf=e_ref,
        fcidump_sha256="0" * 64,
        cache_token="t",
    )


def _run_fake(monkeypatch, results, spec):
    """Run _converge_reference_uhf with the per-start results [(e, tag, converged), ...]."""
    it = iter(results)
    monkeypatch.setattr(
        chem, "_converge_from_density", lambda make, dm0: (_FakeMF(*next(it)), True)
    )
    base = mock.MagicMock()
    base.mol.nelec, base.mol.nao = (12, 11), 16
    monkeypatch.setattr(chem, "_uhf_factory", lambda mf: mock.MagicMock)
    return chem._converge_reference_uhf(base, spec)


@pytest.mark.parametrize(
    "results, source",
    [
        ([(-1.0, "10"), (-1.0 + 5e-7, "10")], "manifest-rdm"),  # tie -> RDM first
        ([(-1.0, "10"), (-1.001, "10")], "manifest-labels"),  # labels lower, same tag
        ([(-1.0, "10"), (-1.5, "01")], "manifest-rdm"),  # lower but wrong tag is rejected
        ([(-1.5, "10", False), (-1.0, "10")], "manifest-labels"),  # unconverged is skipped
    ],
)
def test_selection(monkeypatch, results, source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # energy check vs e_ref=-1.0
        _, info = _run_fake(monkeypatch, results, _fake_spec(e_ref=None))
    assert info["source"] == source and info["tag"] == "10"


def test_no_matching_tag_takes_lowest_and_warns(monkeypatch):
    with pytest.warns(RuntimeWarning, match="No converged BS-UHF start reproduced"):
        _, info = _run_fake(monkeypatch, [(-1.0, "01"), (-1.2, "0x")], _fake_spec(e_ref=None))
    assert info["source"] == "manifest-labels" and info["tag"] == "0x"
    monkeypatch.setenv("SBD_BS_STRICT", "1")
    with pytest.raises(RuntimeError, match="SBD_BS_STRICT=1"):
        _run_fake(monkeypatch, [(-1.0, "01"), (-1.2, "0x")], _fake_spec(e_ref=None))


def test_matching_within_tolerance_is_silent(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, info = _run_fake(monkeypatch, [(-1.00005, "10"), (-0.9, "10")], _fake_spec())
    assert info["source"] == "manifest-rdm"
    assert [c["source"] for c in info["candidates"]] == ["manifest-rdm", "manifest-labels"]


# ---------------------------------------------------------------------------------------------
# Reference fields propagate through every ElectronicProperties rebuild
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _oh_uhf_props(tmp_path_factory):
    """Small spin-unbalanced UHF ElectronicProperties (OH doublet, sto-3g) with reference
    fields set, as a manifest-built reference would carry."""
    from pyscf import gto

    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = scf.ROHF(mol).run()
    path = tmp_path_factory.mktemp("fcidump") / "oh.fcidump"
    tools.fcidump.from_scf(mf, str(path))
    env = {
        "SBD_UHF_CACHE": "0",
        "SBD_UHF_CACHE_DIR": str(path.parent / "uhf_cache"),
        "FE4S4_AF_GROUPS": "",
    }
    with (
        mock.patch.object(chem, "get_run_logger", lambda: logging.getLogger("test")),
        mock.patch.dict("os.environ", env),
    ):
        ep = chem.compute_molecular_integrals_from_fcidump.fn(str(path), unrestricted=True)
    assert ep.reference_tag is None and ep.reference_source is None  # legacy path: unset
    return ep.model_copy(
        update={"reference_tag": "10", "reference_source": "manifest-rdm", "fcidump_sha256": "ab"}
    )


def test_reference_fields_survive_cc_refresh(_oh_uhf_props):
    new = chem.refresh_cc_seed(_oh_uhf_props)
    assert new is not None
    assert chem.reference_fields(new) == chem.reference_fields(_oh_uhf_props)


def test_reference_fields_survive_orbital_rotation(_oh_uhf_props):
    from qcsc_workflow_utility.orbital_opt import rotate_electronic_properties

    n = _oh_uhf_props.num_orbitals
    new = rotate_electronic_properties(_oh_uhf_props, np.eye(n), np.eye(n))
    assert chem.reference_fields(new) == chem.reference_fields(_oh_uhf_props)


def test_rhf_props_have_no_reference_fields():
    ep = chem.compute_molecular_integrals_from_geometry.fn(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
    )
    assert chem.reference_fields(ep) == dict.fromkeys(chem.REFERENCE_FIELDS)
    rt = chem.ElectronicProperties.model_validate_json(ep.model_dump_json())
    assert chem.reference_fields(rt) == dict.fromkeys(chem.REFERENCE_FIELDS)


# ---------------------------------------------------------------------------------------------
# UHF reference cache: content-based key
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def _cache(monkeypatch, tmp_path):
    """Cache ON in a private directory; returns the directory."""
    d = tmp_path / "uhf_cache_on"
    monkeypatch.setenv("SBD_UHF_CACHE", "1")
    monkeypatch.setenv("SBD_UHF_CACHE_DIR", str(d))
    return d


def _oh_fcidump(path: Path, bond: float = 0.97) -> Path:
    from pyscf import gto

    mol = gto.M(atom=f"O 0 0 0; H 0 0 {bond}", basis="sto-3g", spin=1, verbose=0)
    tools.fcidump.from_scf(scf.ROHF(mol).run(), str(path))
    return path


def _forbid(monkeypatch, name):
    def boom(*a, **k):
        raise AssertionError(f"{name} ran although the cache should have been used")

    monkeypatch.setattr(chem, name, boom)


def test_cache_key_is_content_based(tmp_path):
    a = chem._uhf_cache_path("/c", "aa" * 32, "af=None", (5, 4))
    assert a == chem._uhf_cache_path("/c", "aa" * 32, "af=None", (np.int64(5), np.int64(4)))
    assert a != chem._uhf_cache_path("/c", "bb" * 32, "af=None", (5, 4))  # content
    assert a != chem._uhf_cache_path("/c", "aa" * 32, "af={'x': [0]}", (5, 4))  # guess
    assert a != chem._uhf_cache_path("/c", "aa" * 32, "af=None", (4, 5))  # sector
    assert a.startswith("/c/uhf_bs_") and a.endswith(".npz")


def test_legacy_cache_hit_miss_and_same_path_regeneration(tmp_path, monkeypatch, _cache, _no_cc):
    """Same content (even at another path) hits; regenerating the FCIDUMP at the SAME path with
    different content misses (the old abspath key served the stale reference)."""
    fcidump = _oh_fcidump(tmp_path / "oh.fcidump")
    chem.compute_molecular_integrals_from_fcidump.fn(str(fcidump), unrestricted=True)
    e_first = _no_cc["e"]
    assert len(list(_cache.glob("uhf_bs_*.npz"))) == 1

    # same content at a different path -> hit (no SCF)
    copy = tmp_path / "elsewhere" / "oh_copy.fcidump"
    copy.parent.mkdir()
    copy.write_bytes(fcidump.read_bytes())
    real_converge = chem._converge_broken_symmetry_uhf
    _forbid(monkeypatch, "_converge_broken_symmetry_uhf")
    chem.compute_molecular_integrals_from_fcidump.fn(str(copy), unrestricted=True)
    assert _no_cc["e"] == e_first

    # regenerate at the SAME path with a different geometry -> miss, new SCF, second cache file
    monkeypatch.setattr(chem, "_converge_broken_symmetry_uhf", real_converge)
    _oh_fcidump(fcidump, bond=1.10)
    chem.compute_molecular_integrals_from_fcidump.fn(str(fcidump), unrestricted=True)
    assert abs(_no_cc["e"] - e_first) > 1e-3
    assert len(list(_cache.glob("uhf_bs_*.npz"))) == 2


def test_cache_disabled_skips_lookup(tmp_path, monkeypatch, _cache, _no_cc):
    fcidump = _oh_fcidump(tmp_path / "oh.fcidump")
    chem.compute_molecular_integrals_from_fcidump.fn(str(fcidump), unrestricted=True)
    calls = []
    real = chem._converge_broken_symmetry_uhf
    monkeypatch.setattr(
        chem, "_converge_broken_symmetry_uhf", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    monkeypatch.setenv("SBD_UHF_CACHE", "0")
    chem.compute_molecular_integrals_from_fcidump.fn(str(fcidump), unrestricted=True)
    assert calls == [1]


def test_cache_load_rejects_inconsistent_file(tmp_path):
    fcidump = _oh_fcidump(tmp_path / "oh.fcidump")
    mf = _plain_uhf(fcidump)
    mf.kernel()
    cpath = str(tmp_path / "c" / "uhf_bs_x.npz")
    chem._uhf_cache_save(cpath, mf, tag="10", source="manifest-rdm")
    assert not list((tmp_path / "c").glob("*.tmp"))  # atomic write leaves no temp file
    fresh = _plain_uhf(fcidump)
    assert chem._uhf_cache_load(cpath, fresh, mf.mol.nao)
    assert np.isclose(fresh.e_tot, mf.e_tot) and fresh.converged

    # wrong sector (nelec) or wrong NORB -> ignored, mf untouched
    other = _plain_uhf(fcidump)
    other.mol.nelec = (4, 5)
    assert not chem._uhf_cache_load(cpath, other, mf.mol.nao)
    assert not chem._uhf_cache_load(cpath, _plain_uhf(fcidump), mf.mol.nao + 1)
    # legacy v1 cache file (no nelec entry) -> ignored rather than trusted
    legacy = tmp_path / "c" / "legacy.npz"
    with np.load(cpath) as c:
        np.savez(legacy, **{k: c[k] for k in c.files if k != "nelec"})
    assert not chem._uhf_cache_load(str(legacy), _plain_uhf(fcidump), mf.mol.nao)
    (tmp_path / "c" / "garbage.npz").write_bytes(b"not a zip")
    assert not chem._uhf_cache_load(str(tmp_path / "c" / "garbage.npz"), fresh, mf.mol.nao)


def test_manifest_cache_reuse_and_content_sensitivity(tmp_path, monkeypatch, _cache, _no_cc):
    path = _manifest(tmp_path, "red")
    chem.compute_molecular_integrals_from_fcidump.fn(
        None, unrestricted=True, active_space_manifest=str(path)
    )
    first = dict(_no_cc["reference"])
    assert len(list(_cache.glob("uhf_bs_*.npz"))) == 1

    real = chem._converge_reference_uhf
    _forbid(monkeypatch, "_converge_reference_uhf")
    chem.compute_molecular_integrals_from_fcidump.fn(
        None, unrestricted=True, active_space_manifest=str(path)
    )
    assert _no_cc["reference"] == first  # tag / source / sha restored from the cache
    assert abs(_no_cc["e"] - _SETS["red"][5]) < 1e-5

    # a different RDM in the manifest (same FCIDUMP) is a different reference -> miss
    monkeypatch.setattr(chem, "_converge_reference_uhf", real)
    sub = tmp_path / "pert"
    sub.mkdir()
    rdm = _perturbed(np.load(_files(_SETS["red"][0])["rdm"]))
    path2 = _manifest(sub, "red", rdm=rdm)
    chem.compute_molecular_integrals_from_fcidump.fn(
        None, unrestricted=True, active_space_manifest=str(path2)
    )
    assert len(list(_cache.glob("uhf_bs_*.npz"))) == 2


def test_manifest_cache_with_wrong_tag_is_not_served(tmp_path, monkeypatch, _cache, _no_cc):
    """A cache entry whose orbitals give another tag (here: alpha/beta swapped on the ox set,
    '01' -> '10') is re-validated, rejected, and re-converged."""
    path = _manifest(tmp_path, "ox")
    chem.compute_molecular_integrals_from_fcidump.fn(
        None, unrestricted=True, active_space_manifest=str(path)
    )
    (cpath,) = _cache.glob("uhf_bs_*.npz")
    with np.load(cpath) as c:
        d = {k: c[k] for k in c.files}
    d["mo_a"], d["mo_b"] = d["mo_b"], d["mo_a"]
    d["ene_a"], d["ene_b"] = d["ene_b"], d["ene_a"]
    np.savez(cpath, **d)

    calls = []
    real = chem._converge_reference_uhf
    monkeypatch.setattr(
        chem, "_converge_reference_uhf", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    chem.compute_molecular_integrals_from_fcidump.fn(
        None, unrestricted=True, active_space_manifest=str(path)
    )
    assert calls == [1]
    assert _no_cc["reference"]["reference_tag"] == "01"
    with np.load(cpath) as c:  # cache entry repaired
        assert str(c["tag"]) == "01"


# ---------------------------------------------------------------------------------------------
# Flow wiring: FlowParameters.active_space_manifest -> integrals task -> telemetry
# ---------------------------------------------------------------------------------------------


class _StopAfterIntegrals(Exception):
    pass


def _run_flow_until_integrals(monkeypatch, params, method="uhf"):
    """Call riken_sqd_de.fn with Prefect/solver stubbed; stop right after the integrals task and
    return the kwargs it was called with."""
    import sbd.main as main

    solver = mock.MagicMock(method=method, do_rdm=0, solver_mode="cpu")
    monkeypatch.setattr(main.SBDSolverJob, "load", classmethod(lambda cls, name: solver))
    monkeypatch.setattr(main, "get_run_logger", lambda: logging.getLogger("test"))
    monkeypatch.setattr(main, "create_table_artifact", lambda **k: None)
    seen = {}

    def fake_integrals(**kwargs):
        seen.update(kwargs)
        raise _StopAfterIntegrals

    monkeypatch.setattr(main, "compute_molecular_integrals_from_fcidump", fake_integrals)
    with pytest.raises(_StopAfterIntegrals):
        main.riken_sqd_de.fn(parameters=params)
    return seen


def test_flow_passes_manifest_to_integrals(monkeypatch):
    from sbd.flow_params import FlowParameters

    params = FlowParameters(active_space_manifest="/data/x_manifest.json")
    seen = _run_flow_until_integrals(monkeypatch, params)
    assert seen == {
        "fcidump_file": None,
        "unrestricted": True,
        "active_space_manifest": "/data/x_manifest.json",
    }


def test_flow_without_manifest_unchanged(monkeypatch):
    from sbd.flow_params import FlowParameters

    seen = _run_flow_until_integrals(monkeypatch, FlowParameters(fcidump="/d/n2.fcidump"), "rhf")
    assert seen == {
        "fcidump_file": "/d/n2.fcidump",
        "unrestricted": False,
        "active_space_manifest": None,
    }


def test_walker_telemetry_carries_reference_fields(_oh_uhf_props):
    """walker_sqd's telemetry rows start with norb / nelec / reference_* / fcidump_sha256."""
    import inspect

    import sbd.sqd as sqd

    row = sqd._reference_telemetry(_oh_uhf_props)
    assert row == {
        "norb": _oh_uhf_props.num_orbitals,
        "nelec": [5, 4],
        "reference_tag": "10",
        "reference_source": "manifest-rdm",
        "fcidump_sha256": "ab",
    }
    json.dumps(row)  # table-artifact rows must be JSON-serialisable
    legacy = sqd._reference_telemetry(
        _oh_uhf_props.model_copy(update=dict.fromkeys(chem.REFERENCE_FIELDS))
    )
    assert legacy["reference_source"] is None and legacy["norb"] == _oh_uhf_props.num_orbitals
    # and walker_sqd actually puts it in the initial telemetry dict
    assert "**_reference_telemetry(elec_props)" in inspect.getsource(sqd.walker_sqd.fn)
