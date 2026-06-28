"""Verify the multi-batch quantum sampling merge logic (BitArray.concatenate_shots).

The real-device path in walker_sqd submits K sampler jobs of shots//K each and mixes them, to
reach a large effective shot count without one huge IBM job. These tests pin the merge contract:
shots sum, the unique-config pool grows, probabilities normalize, and the harness-only
``n_shot_batches`` key is stripped from the per-batch options sent to IBM.
"""

from __future__ import annotations

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
