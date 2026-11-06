"""Saved-artifact paths in sbd.data_io: plain local paths are files, not S3 keys."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from sbd import data_io


def _write_pool(path, num_bits):
    pool = generate_bit_array_uniform(num_samples=50, num_bits=num_bits, rand_seed=3)
    np.savez_compressed(
        path,
        packed_bits=pool.array,
        num_bits=np.array([pool.num_bits], dtype=np.int64),
        num_shots=np.array([pool.num_shots], dtype=np.int64),
        allow_pickle=False,
    )
    return f"file://{path}"


def test_plain_absolute_path_loads_as_local_file(tmp_path):
    """A pool typed into saved_samples without file:// is read locally, never sent to S3
    (ROQUO resume failed with 'Unable to find block document named s3-sqd')."""
    uri = _write_pool(tmp_path / "pool.npz", 32)
    plain = uri[len("file://") :]
    np.testing.assert_array_equal(
        data_io.load_ndarray(plain, "packed_bits"), data_io.load_ndarray(uri, "packed_bits")
    )
    assert int(data_io.load_ndarray(plain, "num_bits")[0]) == 32


def test_missing_absolute_path_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_io.load_ndarray(str(tmp_path / "missing.npz"), "packed_bits")


def test_s3_key_without_block_names_the_cause(monkeypatch):
    class _NoBlock:
        @staticmethod
        def load(name):
            raise ValueError(f"Unable to find block document named {name}")

    monkeypatch.setattr(data_io, "S3Bucket", _NoBlock)
    with pytest.raises(FileNotFoundError, match="no 's3-sqd' block.*file://"):
        data_io.load_ndarray("sqd_data/riken-sqd-de/run/raw_samples.npz", "packed_bits")
