"""Saved sample-pool consistency guard.

A persisted pool's bitstrings live in the MO basis of the reference they were sampled from;
reloading it (quantum_source="saved") against another FCIDUMP / reference / sector is wrong
silently. walker_sqd now writes a reference record (``ref_meta``) with each pool and checks it,
and always checks ``num_bits == 2*norb``.
"""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.primitives import BitArray
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from sbd import data_io, sqd

_LOG = logging.getLogger("test")


def _props(norb=16, nelec=(12, 11), tag="10", sha="ab" * 32, uhf=True, source="manifest-rdm"):
    return SimpleNamespace(
        num_orbitals=norb,
        num_electrons=nelec,
        unrestricted=uhf,
        reference_tag=tag,
        reference_source=source if tag else None,
        fcidump_sha256=sha,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SBD_ALLOW_POOL_MISMATCH", raising=False)


# ---------------------------------------------------------------------------------------------
# Record format
# ---------------------------------------------------------------------------------------------


def test_ref_meta_array_round_trips_through_npz_without_pickle():
    arr = sqd._pool_ref_meta_array(_props())
    assert arr.dtype.kind == "U" and arr.shape == (1,)
    with io.BytesIO() as buf:
        np.savez_compressed(buf, ref_meta=arr, allow_pickle=False)
        buf.seek(0)
        back = np.load(buf, allow_pickle=False)["ref_meta"]
    meta = json.loads(str(back[0]))
    assert meta == {
        "norb": 16,
        "nelec": [12, 11],
        "fcidump_sha256": "ab" * 32,
        "reference_tag": "10",
        "method": "uhf",
    }


def test_ref_meta_legacy_and_rhf_props():
    meta = sqd._pool_ref_meta(_props(tag=None, sha=None, uhf=False, nelec=(7, 7), norb=18))
    assert meta == {
        "norb": 18,
        "nelec": [7, 7],
        "fcidump_sha256": None,
        "reference_tag": None,
        "method": "rhf",
    }


# ---------------------------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------------------------


def test_matching_pool_passes(caplog):
    p = _props()
    with caplog.at_level(logging.INFO, logger="test"):
        sqd._check_saved_pool("pool", 32, sqd._pool_ref_meta_array(p), p, _LOG)
    assert "matches the current reference" in caplog.text


@pytest.mark.parametrize("num_bits", [72, 36, 30, 33])
def test_num_bits_mismatch_always_raises(monkeypatch, num_bits):
    """e.g. a stale 72-qubit Fe4S4 pool picked up for a 16-orbital AVAS run -- even with the
    override, and even for a legacy pool without a reference record."""
    monkeypatch.setenv("SBD_ALLOW_POOL_MISMATCH", "1")
    p = _props()
    with pytest.raises(ValueError, match=rf"num_bits={num_bits}.*2\*norb = 32"):
        sqd._check_saved_pool("pool", num_bits, None, p, _LOG)
    with pytest.raises(ValueError, match="num_bits"):
        sqd._check_saved_pool("pool", num_bits, sqd._pool_ref_meta_array(p), p, _LOG)


@pytest.mark.parametrize(
    "pool_props, field",
    [
        (_props(sha="cd" * 32), "fcidump_sha256"),
        (_props(tag="01"), "reference_tag"),
        (_props(uhf=False), "method"),
        (_props(nelec=(11, 11)), "nelec"),
    ],
)
def test_reference_mismatch_raises_then_override_warns(monkeypatch, caplog, pool_props, field):
    current = _props()
    meta = sqd._pool_ref_meta_array(pool_props)
    with pytest.raises(ValueError, match=rf"different reference \({field}: .*SBD_ALLOW_POOL"):
        sqd._check_saved_pool("pool", 32, meta, current, _LOG)
    monkeypatch.setenv("SBD_ALLOW_POOL_MISMATCH", "1")
    with caplog.at_level(logging.WARNING, logger="test"):
        sqd._check_saved_pool("pool", 32, meta, current, _LOG)
    assert field in caplog.text and "SBD_ALLOW_POOL_MISMATCH=1" in caplog.text


def test_legacy_pool_without_record_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="test"):
        sqd._check_saved_pool("pool", 32, None, _props(), _LOG)
    assert "no reference record" in caplog.text


def test_legacy_run_vs_manifest_pool_unknown_fields_not_a_mismatch(caplog):
    """sha / tag recorded as None on one side (legacy, no manifest) cannot be compared: accepted.
    norb / nelec / method are still compared."""
    legacy = _props(tag=None, sha=None)
    manifest = _props()
    sqd._check_saved_pool("pool", 32, sqd._pool_ref_meta_array(legacy), manifest, _LOG)
    sqd._check_saved_pool("pool", 32, sqd._pool_ref_meta_array(manifest), legacy, _LOG)
    with pytest.raises(ValueError, match="method"):
        sqd._check_saved_pool(
            "pool",
            32,
            sqd._pool_ref_meta_array(_props(tag=None, sha=None, uhf=False)),
            legacy,
            _LOG,
        )


def test_unreadable_record_raises():
    with pytest.raises(ValueError, match="unreadable ref_meta"):
        sqd._check_saved_pool("pool", 32, np.array(["{not json"]), _props(), _LOG)


# ---------------------------------------------------------------------------------------------
# Real npz files through data_io (file:// URIs, as walker_sqd loads them)
# ---------------------------------------------------------------------------------------------


def _write_pool(path, num_bits, ref_meta=None):
    pool = generate_bit_array_uniform(num_samples=50, num_bits=num_bits, rand_seed=3)
    arrays = {
        "packed_bits": pool.array,
        "num_bits": np.array([pool.num_bits], dtype=np.int64),
        "num_shots": np.array([pool.num_shots], dtype=np.int64),
    }
    if ref_meta is not None:
        arrays["ref_meta"] = ref_meta
    np.savez_compressed(path, **arrays, allow_pickle=False)
    return f"file://{path}"


def test_load_ndarray_optional(tmp_path):
    uri = _write_pool(tmp_path / "legacy.npz", 32)
    assert data_io.load_ndarray_optional(uri, "ref_meta") is None
    assert int(data_io.load_ndarray_optional(uri, "num_bits")[0]) == 32
    uri2 = _write_pool(tmp_path / "new.npz", 32, sqd._pool_ref_meta_array(_props()))
    assert json.loads(str(data_io.load_ndarray_optional(uri2, "ref_meta")[0]))["norb"] == 16


def test_saved_branch_end_to_end_file_pools(tmp_path):
    """The same three calls the saved branch makes, on real npz files."""
    current = _props()

    def load_and_check(uri):
        packed = data_io.load_ndarray(uri, "packed_bits")
        num_bits = int(data_io.load_ndarray(uri, "num_bits")[0])
        sqd._check_saved_pool(
            uri, num_bits, data_io.load_ndarray_optional(uri, "ref_meta"), current, _LOG
        )
        return BitArray(packed, num_bits)

    good = _write_pool(tmp_path / "good.npz", 32, sqd._pool_ref_meta_array(current))
    assert load_and_check(good).num_bits == 32
    assert load_and_check(_write_pool(tmp_path / "legacy.npz", 32)).num_shots == 50
    stale = _write_pool(tmp_path / "fe4s4_72q.npz", 72)
    with pytest.raises(ValueError, match="num_bits=72"):
        load_and_check(stale)
    other = _write_pool(tmp_path / "other.npz", 32, sqd._pool_ref_meta_array(_props(tag="01")))
    with pytest.raises(ValueError, match="reference_tag"):
        load_and_check(other)


def test_walker_sqd_wires_guard_and_record():
    """Source-level pin: the persist call writes ref_meta and the saved branch calls the guard
    before building the BitArray (walker_sqd itself needs Prefect/solver mocks to run)."""
    import inspect

    src = inspect.getsource(sqd.walker_sqd.fn)
    assert "ref_meta=_pool_ref_meta_array(elec_props)" in src
    i_guard = src.index("_check_saved_pool(")
    i_bits = src.index("bit_array = BitArray(packed, num_bits)")
    assert i_guard < i_bits
