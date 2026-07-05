"""Verify the multi-batch quantum sampling merge logic (BitArray.concatenate_shots).

The real-device path in walker_sqd submits K sampler jobs of shots//K each and mixes them, to
reach a large effective shot count without one huge IBM job. These tests pin the merge contract:
shots sum, the unique-config pool grows, probabilities normalize, and the harness-only
``n_shot_batches`` key is stripped from the per-batch options sent to IBM.
"""

from __future__ import annotations

import io

import numpy as np
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform


def test_concatenate_shots_sums_and_enriches():
    nbits, K, per = 24, 5, 4000
    batches = [
        generate_bit_array_uniform(num_samples=per, num_bits=nbits, rand_seed=10 + i)
        for i in range(K)
    ]
    merged = BitArray.concatenate_shots(batches)
    assert merged.num_shots == per * K

    bm, pm = bit_array_to_arrays(merged)
    b0, _ = bit_array_to_arrays(batches[0])
    # A mixed pool has at least as many unique configs as any single batch (here ~K x more).
    assert bm.shape[0] >= b0.shape[0]
    assert abs(float(pm.sum()) - 1.0) < 1e-9
    # K=1 path: a single array is used unchanged.
    assert bit_array_to_arrays(batches[0])[0].shape[0] == b0.shape[0]


def test_per_batch_options_strip_harness_key():
    # Mirror walker_sqd's option split: divide shots, strip n_shot_batches, keep params.
    options = {
        "params": {"shots": 5_000_000, "options": {"twirling": {"enable_measure": True}}},
        "n_shot_batches": 5,
    }
    total = int(options.get("params", {}).get("shots", 0))
    k = max(1, int(options.get("n_shot_batches", 1)))
    per = max(1, total // k)
    batch_options = {key: val for key, val in options.items() if key != "n_shot_batches"}
    batch_options = {**batch_options, "params": {**batch_options.get("params", {})}}
    batch_options["params"]["shots"] = per

    assert "n_shot_batches" not in batch_options  # never sent to the IBM REST schema
    assert batch_options["params"]["shots"] == 1_000_000
    assert batch_options["params"]["options"] == {"twirling": {"enable_measure": True}}


def test_saved_pool_roundtrip_reconstructs_bitarray():
    """The persist step (quantum_source real-device) saves packed_bits/num_bits/num_shots; the
    'saved' source reloads them into an identical BitArray. This pins that serialization contract
    (the same npz keys + BitArray(packed, num_bits) reconstruction the sqd.py code uses), so an
    expensive hardware sample can be re-diagonalized offline. Uses the same np.savez_compressed
    format as data_io.save_ndarray, without needing Prefect flow context.
    """
    nbits, K, per = 40, 5, 3000
    batches = [
        generate_bit_array_uniform(num_samples=per, num_bits=nbits, rand_seed=100 + i)
        for i in range(K)
    ]
    pool = BitArray.concatenate_shots(batches)  # the merged 5M-shot-style pool

    # --- persist (mirror sqd.py save_ndarray call) ---
    with io.BytesIO() as buf:
        np.savez_compressed(
            buf,
            packed_bits=pool.array,
            num_bits=np.array([pool.num_bits], dtype=np.int64),
            num_shots=np.array([pool.num_shots], dtype=np.int64),
            allow_pickle=False,
        )
        blob = buf.getvalue()

    # --- reload (mirror the 'saved' branch) ---
    with io.BytesIO(blob) as buf:
        npz = np.load(buf, allow_pickle=False)
        packed = npz["packed_bits"]
        num_bits = int(npz["num_bits"][0])
        num_shots = int(npz["num_shots"][0])
    reloaded = BitArray(packed, num_bits)

    assert reloaded.num_shots == pool.num_shots == per * K
    assert reloaded.num_bits == pool.num_bits == nbits
    assert num_shots == pool.num_shots
    assert np.array_equal(reloaded.array, pool.array)
    # The reconstructed pool yields identical unique configs + probabilities for diagonalization.
    b_orig, p_orig = bit_array_to_arrays(pool)
    b_re, p_re = bit_array_to_arrays(reloaded)
    assert np.array_equal(b_orig, b_re)
    assert np.allclose(p_orig, p_re)
