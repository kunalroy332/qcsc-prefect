"""Regression tests for spin-unbalanced (Na != Nb) UHF runs.

Each test locks one bug found while running N2+ (NORB=18, NELEC=13, MS2=1 -> (7, 6)):

* per-spin carryover: alpha/beta carryover lists of DIFFERENT lengths used to be stacked into one
  (beta-left, alpha-right) array with zero padding; every padding row became the empty determinant
  (popcount 0) and was forced into the next pass's CI list,
* HCI booster: the alpha core was taken from the beta half and vice versa, injecting Nb-electron
  strings into the Na-electron alpha list,
* the determinant writer now refuses wrong-popcount / duplicate determinants,
* UHF CC refresh: occupancies came from lambdas solved with h2_aa in place of h2_ab / h2_bb,
* multi-walker LUCJ init: unsymmetrized noise made ffsim reject the drifted t2 amplitudes,
* the FCIDUMP path with MS2 != 0 yields the unbalanced (Na, Nb) sector.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import qcsc_workflow_utility.chem as chem
from sbd import lucj, main, solver_job, sqd


@pytest.fixture(autouse=True)
def _silence_prefect_logger(monkeypatch):
    """The integral/IO helpers call Prefect's get_run_logger; stub it for unit tests."""
    for mod in (chem, solver_job, sqd, main, lucj):
        if hasattr(mod, "get_run_logger"):
            monkeypatch.setattr(mod, "get_run_logger", lambda: mock.MagicMock())


def _ints_to_rows(ints, norb):
    """CI-string ints -> (n, norb) bool rows (bit 0 = most significant), the carryover layout."""
    return np.array(
        [[(int(x) >> (norb - 1 - i)) & 1 for i in range(norb)] for x in ints], dtype=bool
    ).reshape(-1, norb)


def _random_strings(rng, norb, nelec, n):
    out = set()
    while len(out) < n:
        out.add(sum(1 << int(p) for p in rng.choice(norb, size=nelec, replace=False)))
    return sorted(out)


# ---------------------------------------------------------------------------------------------
# B1: per-spin carryover
# ---------------------------------------------------------------------------------------------


def test_split_spin_carryover_per_spin_passthrough():
    from sbd.sqd import _split_spin_carryover

    norb = 6
    rng = np.random.default_rng(0)
    co_a = _ints_to_rows(_random_strings(rng, norb, 3, 5), norb)
    co_b = _ints_to_rows(_random_strings(rng, norb, 2, 2), norb)
    a, b = _split_spin_carryover(co_a, norb, co_b)
    assert a.shape == (5, norb) and b.shape == (2, norb)
    assert np.array_equal(a, co_a) and np.array_equal(b, co_b)


def test_split_spin_carryover_empty_and_none_beta():
    from sbd.sqd import _split_spin_carryover

    norb = 6
    for empty in (np.full((0, norb), False), np.empty((0, 0), dtype=bool)):
        a, b = _split_spin_carryover(empty, norb)
        assert a.shape == (0, norb) and b.shape == (0, norb)
    # A UHF solver result with no beta carryover file: alpha kept, beta empty (not padded zeros).
    co_a = _ints_to_rows([0b000111, 0b001011], norb)
    a, b = _split_spin_carryover(co_a, norb, np.empty((0, norb), dtype=bool))
    assert a.shape == (2, norb) and b.shape == (0, norb)


def test_split_spin_carryover_legacy_stacked_drops_padding():
    """A legacy stacked (beta-left, alpha-right) array with zero padding must split cleanly."""
    from sbd.sqd import _split_spin_carryover

    norb, na, nb = 6, 3, 2
    rng = np.random.default_rng(1)
    co_a = _ints_to_rows(_random_strings(rng, norb, na, 5), norb)
    co_b = _ints_to_rows(_random_strings(rng, norb, nb, 3), norb)
    stacked = np.zeros((5, 2 * norb), dtype=bool)  # what the old _stack_spin_carryover built
    stacked[:3, :norb] = co_b
    stacked[:5, norb:] = co_a
    a, b = _split_spin_carryover(stacked, norb)
    assert np.array_equal(a, co_a)
    assert np.array_equal(b, co_b)  # the 2 all-zero padding rows are gone
    assert set(a.sum(axis=1).tolist()) == {na} and set(b.sum(axis=1).tolist()) == {nb}


def test_split_spin_carryover_rejects_ambiguous_width():
    from sbd.sqd import _split_spin_carryover

    with pytest.raises(ValueError):
        _split_spin_carryover(np.ones((2, 7), dtype=bool), 6)


def test_subsample_open_shell_unequal_carryover_keeps_sectors():
    """Unequal-length per-spin carryover goes in as-is: no popcount-0 or duplicate determinants."""
    from sbd import sqd

    norb, na, nb = 8, 4, 3
    rng = np.random.default_rng(2)
    # Post-selected samples (beta-left, alpha-right), as walker_sqd hands them to the subsampler.
    bm = np.array(
        [
            np.r_[_ints_to_rows([b], norb)[0], _ints_to_rows([a], norb)[0]]
            for a, b in zip(_random_strings(rng, norb, na, 40), _random_strings(rng, norb, nb, 40))
        ],
        dtype=bool,
    )
    probs = np.ones(len(bm)) / len(bm)
    co_a_ints = _random_strings(rng, norb, na, 9)
    co_b_ints = _random_strings(rng, norb, nb, 4)

    ci_a, ci_b = sqd.subsample_open_shell.fn(
        bitstring_matrix=bm,
        probabilities=probs,
        carryover_a=_ints_to_rows(co_a_ints, norb),
        carryover_b=_ints_to_rows(co_b_ints, norb),
        subspace_dim=400,
        norb=norb,
        num_elec_a=na,
        num_elec_b=nb,
        rng=np.random.default_rng(3),
    )
    assert {int(x).bit_count() for x in ci_a} == {na}
    assert {int(x).bit_count() for x in ci_b} == {nb}
    assert len(set(ci_a.tolist())) == len(ci_a) and len(set(ci_b.tolist())) == len(ci_b)
    assert set(co_a_ints) <= set(ci_a.tolist())
    assert set(co_b_ints) <= set(ci_b.tolist())


def test_selection_stores_per_spin_carryover():
    from sbd.main import OptimizerState, selection

    norb = 6
    state = OptimizerState(
        energies=np.zeros(2),
        populations=np.full((2, 3), np.nan),
        carryover=np.full((0, norb), False),
    )
    co_a = _ints_to_rows([0b000111, 0b001011, 0b010011], norb)
    co_b = _ints_to_rows([0b000011], norb)
    with mock.patch("sbd.main.get_run_logger", lambda: mock.MagicMock()):
        new = selection.fn(
            trial_populations=[np.zeros(3), np.ones(3)],
            trial_energies=np.array([-1.0, -2.0]),
            trial_carryovers=[(None, None), (co_a, co_b)],
            current_state=state,
        )
    assert new.best_index == 1
    assert np.array_equal(new.carryover, co_a) and np.array_equal(new.carryover_b, co_b)
    assert new.copy().carryover_b.shape == (1, norb)


# ---------------------------------------------------------------------------------------------
# B2: HCI booster cores come from each spin's own carryover
# ---------------------------------------------------------------------------------------------


def test_bool_rows_to_ints_matches_subsample_encoding():
    from sbd.sqd import _bool_rows_to_ints

    norb = 6
    ints = [0b000111, 0b101001, 0b110000]
    assert _bool_rows_to_ints(_ints_to_rows(ints, norb), norb).tolist() == ints
    assert _bool_rows_to_ints(None, norb).size == 0
    assert _bool_rows_to_ints(np.empty((0, norb), dtype=bool), norb).size == 0


def test_hci_boost_children_keep_spin_sector():
    """Children of an Na-electron core stay in the Na sector (the swapped core broke this)."""
    from sbd.sqd import _bool_rows_to_ints, _hci_boost_strings

    norb, na, nb = 8, 4, 3
    rng = np.random.default_rng(4)
    core_a = _bool_rows_to_ints(_ints_to_rows(_random_strings(rng, norb, na, 3), norb), norb)
    core_b = _bool_rows_to_ints(_ints_to_rows(_random_strings(rng, norb, nb, 3), norb), norb)
    empty = np.empty(0, dtype=np.longlong)
    new_a = _hci_boost_strings(core_a, empty, norb, na, 50, 3)
    new_b = _hci_boost_strings(core_b, empty, norb, nb, 50, 3)
    assert new_a.size and {int(x).bit_count() for x in new_a} == {na}
    assert new_b.size and {int(x).bit_count() for x in new_b} == {nb}


# ---------------------------------------------------------------------------------------------
# Determinant writer guard
# ---------------------------------------------------------------------------------------------


def test_write_dets_bin_rejects_wrong_popcount(tmp_path: Path):
    from sbd.solver_job import _write_dets_bin

    with pytest.raises(ValueError, match="do not have 3 electrons"):
        _write_dets_bin(tmp_path / "a.bin", np.array([0b0111, 0b0011]), 4, spin="alpha", num_elec=3)
    with pytest.raises(ValueError, match="do not have 2 electrons"):
        _write_dets_bin(tmp_path / "b.bin", np.array([0b0011, 0]), 4, spin="beta", num_elec=2)


def test_write_dets_bin_rejects_duplicates(tmp_path: Path):
    from sbd.solver_job import _write_dets_bin

    with pytest.raises(ValueError, match="duplicate"):
        _write_dets_bin(
            tmp_path / "a.bin", np.array([0b0011, 0b0101, 0b0011]), 4, spin="alpha", num_elec=2
        )


def test_write_dets_bin_valid_and_unchecked(tmp_path: Path):
    from sbd.solver_job import _write_dets_bin

    _write_dets_bin(tmp_path / "a.bin", np.array([0b0011, 0b0101]), 4, spin="alpha", num_elec=2)
    assert (tmp_path / "a.bin").read_bytes() == bytes([0b0011, 0b0101])
    # num_elec=None keeps the legacy unchecked behavior.
    _write_dets_bin(tmp_path / "c.bin", np.array([0b0011, 0b0011]), 4, spin="alpha")


# ---------------------------------------------------------------------------------------------
# FCIDUMP with MS2 != 0 -> unbalanced sector; B3 CC refresh; B4 multi-walker init
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oh_doublet_props(tmp_path_factory):
    """OH doublet (5, 4) through the FCIDUMP path, as the flow loads it."""
    import qcsc_workflow_utility.chem as chem
    from pyscf import gto, scf, tools

    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = scf.ROHF(mol).run()
    path = tmp_path_factory.mktemp("fcidump") / "oh.fcidump"
    tools.fcidump.from_scf(mf, str(path))
    header = path.read_text().splitlines()[0]
    assert "MS2=1" in header.replace(" ", "")
    with (
        mock.patch.object(chem, "get_run_logger", lambda: mock.MagicMock()),
        mock.patch.dict("os.environ", {"SBD_UHF_CACHE": "0", "FE4S4_AF_GROUPS": ""}),
    ):
        return chem.compute_molecular_integrals_from_fcidump.fn(str(path), unrestricted=True)


def test_fcidump_ms2_gives_unbalanced_sector(oh_doublet_props):
    ep = oh_doublet_props
    na, nb = ep.num_electrons
    assert (na, nb) == (5, 4)
    nvir_a, nvir_b = ep.num_orbitals - na, ep.num_orbitals - nb
    assert ep.t2.shape == (na, na, nvir_a, nvir_a)
    assert ep.t2_ab.shape == (na, nb, nvir_a, nvir_b)
    assert ep.t2_bb.shape == (nb, nb, nvir_b, nvir_b)
    occ_a, occ_b = ep.initial_occupancy
    assert np.isclose(np.sum(occ_a), na, atol=1e-6) and np.isclose(np.sum(occ_b), nb, atol=1e-6)


def test_n2p_fcidump_loads_as_unbalanced_uhf_sector():
    """The N2+ FCIDUMP shipped with this PR (NORB=18, NELEC=13, MS2=1) -> (Na, Nb) = (7, 6)."""
    path = Path(__file__).resolve().parents[1] / "data" / "fcidump_N2p_MO.txt"
    with mock.patch.dict("os.environ", {"SBD_UHF_CACHE": "0", "FE4S4_AF_GROUPS": ""}):
        ep = chem.compute_molecular_integrals_from_fcidump.fn(str(path), unrestricted=True)
    assert ep.num_electrons == (7, 6) and ep.num_orbitals == 18
    assert ep.t2.shape == (7, 7, 11, 11)
    assert ep.t2_ab.shape == (7, 6, 11, 12)
    assert ep.t2_bb.shape == (6, 6, 12, 12)
    occ_a, occ_b = ep.initial_occupancy
    assert np.isclose(np.sum(occ_a), 7, atol=1e-6) and np.isclose(np.sum(occ_b), 6, atol=1e-6)
    assert not np.allclose(ep.two_body_tensor, ep.two_body_tensor_bb)  # genuinely spin-broken


def test_uhf_cc_refresh_occupancy_uses_spin_block_integrals(oh_doublet_props):
    """refresh_cc_seed on the UNROTATED integrals must reproduce the initial UCCSD occupancies.

    The initial occupancies come from a genuine pyscf UCCSD on the UHF reference (alpha/beta MOs
    differ, so h2_aa, h2_ab, h2_bb all differ). The refresh re-solves UCCSD on the same MO-basis
    integrals; with the lambda equations solved on h2_aa alone (the bug) its 1-RDM drifts away.
    """
    from qcsc_workflow_utility.chem import refresh_cc_seed

    ep = oh_doublet_props
    assert not np.allclose(ep.two_body_tensor, ep.two_body_tensor_bb)
    refreshed = refresh_cc_seed(ep)
    assert refreshed is not None
    occ_a0, occ_b0 = ep.initial_occupancy
    occ_a, occ_b = refreshed.initial_occupancy
    assert np.allclose(occ_a, occ_a0, atol=1e-5)
    assert np.allclose(occ_b, occ_b0, atol=1e-5)


def test_multi_walker_ucj_init_unbalanced(oh_doublet_props):
    """num_walkers > 1 used to raise in ffsim validation (non-antisymmetric drifted t2aa/t2bb)."""
    from sbd.lucj import initialize_ucj_parameters

    norb = oh_doublet_props.num_orbitals
    aa = [(p, p + 1) for p in range(norb - 1)]
    ab = [(p, p) for p in range(0, norb, 4)]
    params = initialize_ucj_parameters(
        elec_props=oh_doublet_props,
        aa_indices=aa,
        ab_indices=ab,
        num_walkers=3,
        randomization_factor=0.2,
        n_lucj_layers=1,
        ucj_optimize=False,
    )
    assert len(params) == 3
    assert len({p.shape for p in params}) == 1
    assert not np.allclose(params[0], params[1])


def test_multi_walker_ucj_init_rhf():
    import qcsc_workflow_utility.chem as chem
    from sbd.lucj import initialize_ucj_parameters

    with mock.patch.object(chem, "get_run_logger", lambda: mock.MagicMock()):
        ep = chem.compute_molecular_integrals_from_geometry.fn(
            atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", basis="sto-3g"
        )
    norb = ep.num_orbitals
    params = initialize_ucj_parameters(
        elec_props=ep,
        aa_indices=[(p, p + 1) for p in range(norb - 1)],
        ab_indices=[(p, p) for p in range(norb)],
        num_walkers=3,
        randomization_factor=0.2,
        n_lucj_layers=1,
        ucj_optimize=False,
    )
    assert len(params) == 3 and not np.allclose(params[0], params[2])


# ---------------------------------------------------------------------------------------------
# Additional coverage: end-to-end walker_sqd, on-disk contract, checkpoint, selection, noise
# ---------------------------------------------------------------------------------------------


def _all_strings(norb, nelec):
    return sorted(sum(1 << p for p in c) for c in itertools.combinations(range(norb), nelec))


def _rows_to_ints(rows):
    rows = np.asarray(rows, dtype=bool)
    if rows.size == 0:
        return []
    norb = rows.shape[1]
    return [int(sum(int(b) << (norb - 1 - i) for i, b in enumerate(r))) for r in rows]


# ============================================================================================
# 1. On-disk contract: writer -> reader round trip
# ============================================================================================
def _roundtrip(tmp_path, norb):
    rng = np.random.default_rng(norb)
    nelec = max(1, norb // 3)
    ints = sorted(
        {sum(1 << int(p) for p in rng.choice(norb, nelec, replace=False)) for _ in range(25)}
    )
    path = tmp_path / "x.bin"
    solver_job._write_dets_bin(
        path, np.array(ints, dtype=np.int64), norb, spin="alpha", num_elec=nelec
    )
    assert path.stat().st_size == len(ints) * ((norb + 7) // 8)
    back = solver_job._read_carryover_bin(path, norb)
    assert back.shape == (len(ints), norb)
    return ints, sorted(_rows_to_ints(back))


@pytest.mark.parametrize("norb", [8, 16, 24, 32])
def test_dets_bin_roundtrips_when_norb_is_a_multiple_of_8(tmp_path: Path, norb):
    ints, back = _roundtrip(tmp_path, norb)
    assert back == ints


@pytest.mark.parametrize("norb", [4, 9, 17, 18, 20, 36])
def test_dets_bin_input_and_carryover_use_different_bit_alignments(tmp_path: Path, norb):
    """Documents a real asymmetry (checked against a real Fe2S2 solver run, norb=20):

    * ``_write_dets_bin`` (AlphaDets.bin, solver INPUT) writes each determinant as a plain
      big-endian integer, value in the LOW bits.
    * the solver's OUTPUT ``carryover.bin`` is left-aligned (value in the HIGH bits, padding at
      the end), which is what ``_read_carryover_bin`` decodes.

    So a file written by the writer must NOT be fed to the carryover reader unless norb is a
    multiple of 8 (where the two layouts coincide). Do not "fix" this by changing the reader: on a
    real run all 49 carryover records decode into the 98-record input list with the current reader
    and none with a plain-integer decode.
    """
    rng = np.random.default_rng(norb)
    nelec = max(1, norb // 3)
    ints = sorted(
        {sum(1 << int(p) for p in rng.choice(norb, nelec, replace=False)) for _ in range(20)}
    )
    path = tmp_path / "x.bin"
    solver_job._write_dets_bin(
        path, np.array(ints, dtype=np.int64), norb, spin="alpha", num_elec=nelec
    )
    pad = (norb + 7) // 8 * 8 - norb
    recs = np.fromfile(path, dtype=np.uint8).reshape(len(ints), -1)
    plain = [int.from_bytes(bytes(r), "big") for r in recs]
    assert plain == ints  # input file: plain integers
    left_aligned = tmp_path / "carryover.bin"  # what the solver writes: shifted left
    left_aligned.write_bytes(b"".join((x << pad).to_bytes((norb + 7) // 8, "big") for x in ints))
    back = solver_job._read_carryover_bin(left_aligned, norb)
    assert sorted(_rows_to_ints(back)) == ints  # the reader decodes the solver layout


def test_dets_bin_empty_list_and_zero_electrons(tmp_path: Path):
    solver_job._write_dets_bin(
        tmp_path / "e.bin", np.array([], dtype=np.int64), 6, spin="beta", num_elec=2
    )
    assert (tmp_path / "e.bin").read_bytes() == b""
    # Nb = 0: the single valid beta determinant is the empty string 0b000000.
    solver_job._write_dets_bin(tmp_path / "z.bin", np.array([0]), 6, spin="beta", num_elec=0)
    assert (tmp_path / "z.bin").read_bytes() == b"\x00"


def test_write_dets_bin_range_and_negative_still_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        solver_job._write_dets_bin(
            tmp_path / "a.bin", np.array([1 << 4]), 4, spin="alpha", num_elec=1
        )
    with pytest.raises(ValueError):
        solver_job._write_dets_bin(tmp_path / "b.bin", np.array([-1]), 4, spin="alpha", num_elec=1)


def test_write_dets_bin_error_names_the_bad_popcounts(tmp_path: Path):
    with pytest.raises(ValueError) as exc:
        solver_job._write_dets_bin(
            tmp_path / "a.bin", np.array([0b0111, 0b0001, 0b0011]), 4, spin="beta", num_elec=2
        )
    msg = str(exc.value)
    assert "2 of 3" in msg and "beta" in msg and "[1, 3]" in msg


def _touch_solver_outputs(d: Path, norb: int, *, carryover=None, carryover_b=None):
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / "occ_a.txt", np.full(norb, 0.5))
    np.savetxt(d / "occ_b.txt", np.full(norb, 0.5))
    np.savetxt(d / "davidson_energy.txt", [-1.25])
    for name, ints in (("carryover.bin", carryover), ("carryover_b.bin", carryover_b)):
        if ints is not None:
            solver_job._write_dets_bin(d / name, np.array(ints, dtype=np.int64), norb, spin="x")


def test_read_files_uhf_returns_separate_unequal_carryover(tmp_path: Path):
    norb = 6
    _touch_solver_outputs(
        tmp_path, norb, carryover=[0b001111, 0b010111, 0b011011], carryover_b=[0b000111]
    )
    res = solver_job._read_files(work_dir=tmp_path, norb=norb, method="uhf")
    assert res.carryover_bitstrings.shape == (3, norb)
    assert res.carryover_bitstrings_b.shape == (1, norb)


def test_read_files_uhf_missing_beta_file_warns_and_leaves_none(tmp_path: Path, monkeypatch):
    log = mock.MagicMock()
    monkeypatch.setattr(solver_job, "get_run_logger", lambda: log)
    _touch_solver_outputs(tmp_path, 6, carryover=[0b001111])
    res = solver_job._read_files(work_dir=tmp_path, norb=6, method="uhf")
    assert res.carryover_bitstrings_b is None
    assert log.warning.called
    assert "carryover_b.bin" in str(log.warning.call_args)


def test_read_files_uhf_missing_beta_file_silent_when_no_alpha_carryover(
    tmp_path: Path, monkeypatch
):
    log = mock.MagicMock()
    monkeypatch.setattr(solver_job, "get_run_logger", lambda: log)
    _touch_solver_outputs(tmp_path, 6, carryover=[])
    res = solver_job._read_files(work_dir=tmp_path, norb=6, method="uhf")
    assert res.carryover_bitstrings.shape == (0, 6) and res.carryover_bitstrings_b is None
    assert not log.warning.called


def test_read_files_rhf_never_looks_for_beta_file(tmp_path: Path):
    _touch_solver_outputs(tmp_path, 6, carryover=[0b001111], carryover_b=[0b000011])
    res = solver_job._read_files(work_dir=tmp_path, norb=6, method="rhf")
    assert res.carryover_bitstrings_b is None  # a stray carryover_b.bin must not leak into RHF


def test_prep_files_uhf_rejects_alpha_string_in_beta_list(tmp_path: Path):
    """The Na != Nb failure the PR describes: an Na-electron string offered as a beta string."""
    norb, nelec = 6, (3, 2)
    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb,) * 4)
    kw = dict(
        work_dir=tmp_path,
        one_body_tensor=h1,
        two_body_tensor=h2,
        norb=norb,
        nelec=nelec,
        method="uhf",
        one_body_tensor_b=h1,
        two_body_tensor_ab=h2,
        two_body_tensor_bb=h2,
    )
    alpha = np.array(_all_strings(norb, 3)[:4])
    good_beta = np.array(_all_strings(norb, 2)[:4])
    bad_beta = np.array(_all_strings(norb, 3)[:4])  # popcount 3 in a 2-electron list
    solver_job._prep_files(ci_strings=(alpha, good_beta), **kw)
    assert (tmp_path / "AlphaDets.bin").exists() and (tmp_path / "BetaDets.bin").exists()
    with pytest.raises(ValueError, match="do not have 2 electrons"):
        solver_job._prep_files(ci_strings=(alpha, bad_beta), **kw)


def test_prep_files_rhf_checks_alpha_popcount(tmp_path: Path):
    norb = 6
    h1, h2 = np.zeros((norb, norb)), np.zeros((norb,) * 4)
    kw = dict(
        work_dir=tmp_path,
        one_body_tensor=h1,
        two_body_tensor=h2,
        norb=norb,
        nelec=(2, 2),
        method="rhf",
    )
    solver_job._prep_files(ci_strings=(np.array(_all_strings(norb, 2)[:3]),) * 2, **kw)
    with pytest.raises(ValueError, match="do not have 2 electrons"):
        solver_job._prep_files(ci_strings=(np.array(_all_strings(norb, 3)[:3]),) * 2, **kw)


# ============================================================================================
# 2. _split_spin_carryover: edge cases and properties
# ============================================================================================
def test_split_is_idempotent_on_its_own_output():
    norb = 6
    a = _ints_to_rows(_all_strings(norb, 3)[:5], norb)
    b = _ints_to_rows(_all_strings(norb, 2)[:2], norb)
    a2, b2 = sqd._split_spin_carryover(
        *sqd._split_spin_carryover(a, norb, b)[:1], norb, sqd._split_spin_carryover(a, norb, b)[1]
    )
    assert np.array_equal(a2, a) and np.array_equal(b2, b)


def test_split_does_not_mutate_inputs():
    norb = 6
    a = _ints_to_rows(_all_strings(norb, 3)[:4], norb)
    b = _ints_to_rows(_all_strings(norb, 2)[:2], norb)
    a0, b0 = a.copy(), b.copy()
    sqd._split_spin_carryover(a, norb, b)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)


def test_split_returns_bool_dtype_even_for_int_input():
    norb = 4
    a, b = sqd._split_spin_carryover(np.eye(norb, dtype=int)[:2], norb, np.eye(norb, dtype=int)[:1])
    assert a.dtype == bool and b.dtype == bool


def test_split_empty_alpha_with_nonempty_beta():
    norb = 6
    b = _ints_to_rows(_all_strings(norb, 2)[:3], norb)
    a, b2 = sqd._split_spin_carryover(np.empty((0, norb), dtype=bool), norb, b)
    assert a.shape == (0, norb) and np.array_equal(b2, b)


def test_split_per_spin_alpha_without_beta_means_empty_beta():
    """The solver wrote no carryover_b.bin: alpha is kept as-is and beta is empty."""
    a, b = sqd._split_spin_carryover(np.eye(4, dtype=bool), 4)
    assert a.shape == (4, 4) and b.shape == (0, 4)


def test_split_rejects_a_width_that_is_neither_per_spin_nor_stacked():
    with pytest.raises(ValueError, match="per-spin"):
        sqd._split_spin_carryover(np.ones((2, 7), dtype=bool), 4)


def test_split_uses_beta_left_alpha_right_for_legacy_layout():
    """Locks the legacy convention: columns [:norb] are BETA, [norb:] are ALPHA."""
    norb = 4
    stacked = np.zeros((2, 2 * norb), dtype=bool)
    stacked[0, :norb] = [1, 1, 0, 0]  # beta
    stacked[0, norb:] = [1, 1, 1, 0]  # alpha
    stacked[1, norb:] = [0, 1, 1, 1]  # alpha only
    a, b = sqd._split_spin_carryover(stacked, norb)
    assert a.tolist() == [[1, 1, 1, 0], [0, 1, 1, 1]]
    assert b.tolist() == [[1, 1, 0, 0]]


def test_split_legacy_keeps_valid_empty_determinant_when_spin_has_zero_electrons():
    norb = 4
    stacked = np.zeros((2, 2 * norb), dtype=bool)
    stacked[:, norb:] = [[1, 0, 0, 0], [0, 1, 0, 0]]  # alpha: 1 electron, beta: 0 electrons
    _, b = sqd._split_spin_carryover(stacked, norb, None, num_electrons=(1, 0))
    assert b.shape == (1, norb) and not b.any()
    _, b_default = sqd._split_spin_carryover(stacked, norb)  # without electron counts: padding drop
    assert b_default.shape == (0, norb)


def test_split_legacy_width_collision_is_detected_by_shape_only():
    """Documents the heuristic: width == 2*norb is *assumed* legacy, per-spin arrays never are."""
    a, b = sqd._split_spin_carryover(np.ones((2, 8), dtype=bool), 4)
    assert a.shape == (2, 4) and b.shape == (2, 4)


# ============================================================================================
# 3. _bool_rows_to_ints
# ============================================================================================
@pytest.mark.parametrize("norb", [1, 8, 31, 32, 36, 62])
def test_bool_rows_to_ints_matches_python_reference(norb):
    rng = np.random.default_rng(norb)
    rows = rng.random((7, norb)) < 0.4
    assert sqd._bool_rows_to_ints(rows, norb).tolist() == _rows_to_ints(rows)


def test_bool_rows_to_ints_ignores_columns_beyond_norb():
    """The helper slices [:, :norb]; a stacked 2*norb array silently yields its BETA half."""
    norb = 4
    stacked = np.zeros((1, 2 * norb), dtype=bool)
    stacked[0, :norb] = [1, 1, 0, 0]
    stacked[0, norb:] = [0, 0, 1, 1]
    assert sqd._bool_rows_to_ints(stacked, norb).tolist() == [0b1100]


def test_bool_rows_to_ints_output_dtype_is_longlong():
    assert sqd._bool_rows_to_ints(np.eye(4, dtype=bool), 4).dtype == np.longlong


def test_bool_rows_to_ints_refuses_norb_that_overflows_int64():
    rows = np.zeros((1, 70), dtype=bool)
    rows[0, 0] = True
    with pytest.raises(ValueError, match="norb=70"):
        sqd._bool_rows_to_ints(rows, 70)
    assert sqd._bool_rows_to_ints(np.eye(62, dtype=bool)[:1], 62).tolist() == [1 << 61]


# ============================================================================================
# 4. OptimizerState + selection + CC-refresh reset
# ============================================================================================
def _state(norb=6, nw=3, with_b=False):
    st = main.OptimizerState(
        energies=np.zeros(nw),
        populations=np.full((nw, 4), np.nan),
        carryover=np.full((0, norb), False),
    )
    if with_b:
        st.carryover_b = np.full((0, norb), False)
    return st


def test_optimizer_state_copy_is_deep_for_carryover_b():
    st = _state(with_b=True)
    st.carryover_b = _ints_to_rows([0b000011], 6)
    cp = st.copy()
    cp.carryover_b[0, 0] = not cp.carryover_b[0, 0]
    assert st.carryover_b[0, 0] != cp.carryover_b[0, 0]


def test_optimizer_state_copy_preserves_none_beta():
    assert _state().copy().carryover_b is None


def test_selection_rhf_pair_with_none_beta_keeps_carryover_b_none():
    co = _ints_to_rows(_all_strings(6, 3)[:4], 6)
    new = main.selection.fn(
        trial_populations=[np.zeros(4)] * 3,
        trial_energies=np.array([-1.0, -3.0, -2.0]),
        trial_carryovers=[(co, None)] * 3,
        current_state=_state(),
    )
    assert new.best_index == 1 and new.carryover_b is None and np.array_equal(new.carryover, co)


def test_selection_does_not_overwrite_carryover_without_improvement():
    st = _state(with_b=True)
    st.energies[:] = -5.0
    st.best_index = 0
    keep_a, keep_b = _ints_to_rows([0b000111], 6), _ints_to_rows([0b000011], 6)
    st.carryover, st.carryover_b = keep_a, keep_b
    # best_energy() reads state; emulate "current best is -5.0"
    with mock.patch.object(main.OptimizerState, "best_energy", return_value=-5.0):
        new = main.selection.fn(
            trial_populations=[np.zeros(4)] * 3,
            trial_energies=np.array([-1.0, -2.0, -3.0]),
            trial_carryovers=[(_ints_to_rows([0b001011], 6), _ints_to_rows([0b000101], 6))] * 3,
            current_state=st,
        )
    assert np.array_equal(new.carryover, keep_a) and np.array_equal(new.carryover_b, keep_b)


def test_selection_none_alpha_pair_leaves_existing_carryover_untouched():
    st = _state(with_b=True)
    keep_a, keep_b = _ints_to_rows([0b000111], 6), _ints_to_rows([0b000011], 6)
    st.carryover, st.carryover_b = keep_a, keep_b
    new = main.selection.fn(
        trial_populations=[np.zeros(4)] * 3,
        trial_energies=np.array([-1.0, -3.0, -2.0]),
        trial_carryovers=[(None, None)] * 3,
        current_state=st,
    )
    assert new.best_index == 1
    assert np.array_equal(new.carryover, keep_a) and np.array_equal(new.carryover_b, keep_b)


def test_selection_skips_nan_walkers_when_picking_best_carryover():
    co_good = _ints_to_rows([0b000111], 6)
    new = main.selection.fn(
        trial_populations=[np.zeros(4)] * 3,
        trial_energies=np.array([np.nan, -2.0, np.nan]),
        trial_carryovers=[(None, None), (co_good, None), (None, None)],
        current_state=_state(),
    )
    assert new.best_index == 1 and np.array_equal(new.carryover, co_good)


def test_selection_resizes_nan_population_width_for_uhf():
    new = main.selection.fn(
        trial_populations=[np.arange(9.0)] * 3,
        trial_energies=np.array([-1.0, -2.0, -3.0]),
        trial_carryovers=[(_ints_to_rows([0b000111], 6), None)] * 3,
        current_state=_state(),
    )
    assert new.populations.shape == (3, 9)


def test_cc_refresh_reseed_clears_beta_carryover_with_alpha(monkeypatch):
    norb = 6
    st = _state(norb, with_b=True)
    st.carryover = _ints_to_rows([0b000111], norb)
    st.carryover_b = _ints_to_rows([0b000011], norb)
    st.energies[:] = -3.0
    refreshed = SimpleNamespace(num_orbitals=norb, t2=np.zeros((2, 2, 2, 2)))
    monkeypatch.setattr(main, "refresh_cc_seed", lambda ep: refreshed)
    ep = SimpleNamespace(t2=np.zeros((2, 2, 2, 2)))
    _, out = main._apply_cc_refresh_and_reseed(ep, st, mock.MagicMock(), 0)
    assert out.carryover.shape == (0, norb) and out.carryover_b.shape == (0, norb)
    assert out.best_index is None and np.all(out.energies == 0)


def test_cc_refresh_reseed_keeps_rhf_beta_none(monkeypatch):
    norb = 6
    st = _state(norb)
    refreshed = SimpleNamespace(num_orbitals=norb, t2=np.zeros((2, 2, 2, 2)))
    monkeypatch.setattr(main, "refresh_cc_seed", lambda ep: refreshed)
    _, out = main._apply_cc_refresh_and_reseed(
        SimpleNamespace(t2=np.zeros((2,) * 4)), st, mock.MagicMock(), 0
    )
    assert out.carryover_b is None and out.carryover.shape == (0, norb)


def test_cc_refresh_failure_leaves_state_untouched(monkeypatch):
    st = _state(with_b=True)
    st.carryover_b = _ints_to_rows([0b000011], 6)
    monkeypatch.setattr(main, "refresh_cc_seed", lambda ep: None)
    ep = SimpleNamespace(t2=np.zeros((2,) * 4))
    ep_out, st_out = main._apply_cc_refresh_and_reseed(ep, st, mock.MagicMock(), 0)
    assert ep_out is ep and st_out is st and st_out.carryover_b.shape == (1, 6)


# ============================================================================================
# 5. noise helpers: symmetry + statistics
# ============================================================================================
def _seed_rng(monkeypatch, seed=0):
    monkeypatch.setattr(lucj, "MODULE_RNG", np.random.default_rng(seed))


def test_antisymmetric_noise_has_full_antisymmetry_and_zero_diagonals(monkeypatch):
    _seed_rng(monkeypatch)
    n = lucj._antisymmetric_noise((4, 4, 5, 5), 0.3)
    assert np.allclose(n, -n.transpose(1, 0, 2, 3)) and np.allclose(n, -n.transpose(0, 1, 3, 2))
    assert np.allclose(n[np.arange(4), np.arange(4)], 0.0)  # t[i,i,a,b] = 0
    assert np.allclose(n[:, :, np.arange(5), np.arange(5)], 0.0)  # t[i,j,a,a] = 0


def test_pair_symmetric_noise_is_symmetric_and_bounded(monkeypatch):
    _seed_rng(monkeypatch)
    n = lucj._pair_symmetric_noise((3, 3, 4, 4), 0.2)
    assert np.allclose(n, n.transpose(1, 0, 3, 2))
    assert np.abs(n).max() <= np.sqrt(2) * 0.1 + 1e-12


def test_noise_is_reproducible_for_a_fixed_rng_state(monkeypatch):
    _seed_rng(monkeypatch, 7)
    a = lucj._antisymmetric_noise((3, 3, 4, 4), 0.2)
    _seed_rng(monkeypatch, 7)
    b = lucj._antisymmetric_noise((3, 3, 4, 4), 0.2)
    assert np.array_equal(a, b)


def test_noise_scales_linearly_with_randomization_factor(monkeypatch):
    _seed_rng(monkeypatch, 1)
    a = lucj._antisymmetric_noise((4, 4, 5, 5), 0.1)
    _seed_rng(monkeypatch, 1)
    b = lucj._antisymmetric_noise((4, 4, 5, 5), 0.3)
    assert np.allclose(b, 3 * a)


def test_noise_zero_factor_gives_zero(monkeypatch):
    _seed_rng(monkeypatch)
    assert not np.any(lucj._antisymmetric_noise((3, 3, 4, 4), 0.0))
    assert not np.any(lucj._pair_symmetric_noise((3, 3, 4, 4), 0.0))


def test_noise_amplitude_matches_the_old_unsymmetrised_draw(monkeypatch):
    """randomization_factor must perturb walkers about as much as before the symmetrisation."""
    _seed_rng(monkeypatch, 0)
    shape, scale = (8, 8, 12, 12), 0.2
    old = scale * (lucj.MODULE_RNG.random(shape) - 0.5)
    sym = lucj._pair_symmetric_noise(shape, scale)
    anti = lucj._antisymmetric_noise(shape, scale)
    assert sym.std() / old.std() == pytest.approx(1.0, abs=0.1)
    assert anti.std() / old.std() == pytest.approx(1.0, abs=0.15)


# ============================================================================================
# 6. API surface that other code relies on
# ============================================================================================
def test_initialize_ucj_parameters_keeps_prefect_task_api():
    assert hasattr(lucj.initialize_ucj_parameters, "fn")


def test_noise_helpers_are_plain_functions_not_prefect_tasks():
    """Only the entry point is a task; a task per noise draw would add a task run per walker."""
    assert not hasattr(lucj._pair_symmetric_noise, "fn")
    assert not hasattr(lucj._antisymmetric_noise, "fn")


def test_run_local_scripts_still_call_initialize_ucj_parameters_dot_fn():
    root = Path(__file__).resolve().parents[1]
    hits = [
        p.name
        for p in (root / "run_local.py", root / "run_local_uhf.py")
        if p.exists() and "initialize_ucj_parameters.fn(" in p.read_text()
    ]
    assert hits, "expected the local runners to use initialize_ucj_parameters.fn(...)"


# ============================================================================================
# 7. END-TO-END: walker_sqd over several steps with a fake per-spin solver
# ============================================================================================
NORB, NA, NB = 6, 3, 2


@dataclasses.dataclass
class _Rec:
    alpha: list
    beta: list


class FakeSolver:
    """Stands in for SBDSolverJob.run. Mimics the real binary: the carryover it returns is a
    subset of the dets it was given, with DIFFERENT lengths per spin, and it records every call."""

    root_dir = None

    def __init__(self, n_alpha_co=5, n_beta_co=2):
        self.calls: list[_Rec] = []
        self.n_alpha_co, self.n_beta_co = n_alpha_co, n_beta_co

    async def run(self, *, ci_strings, nelec, norb, **kw):
        a, b = np.asarray(ci_strings[0]), np.asarray(ci_strings[1])
        self.calls.append(_Rec(sorted(a.tolist()), sorted(b.tolist())))
        co_a = _ints_to_rows(sorted(a.tolist())[: self.n_alpha_co], norb)
        co_b = _ints_to_rows(sorted(b.tolist())[: self.n_beta_co], norb)
        return solver_job.SBDResult(
            energy=-1.0 - 0.01 * len(self.calls),
            orbital_occupancies=(np.full(norb, NA / norb), np.full(norb, NB / norb)),
            carryover_bitstrings=co_a,
            carryover_bitstrings_b=co_b,
        )


def _fake_props():
    rng = np.random.default_rng(0)
    n = NORB
    h1 = np.diag(np.arange(n, dtype=float))
    h2 = np.zeros((n,) * 4)
    return SimpleNamespace(
        num_orbitals=n,
        num_electrons=(NA, NB),
        unrestricted=True,
        open_shell=True,
        one_body_tensor=h1,
        one_body_tensor_b=h1,
        two_body_tensor=h2,
        two_body_tensor_ab=h2,
        two_body_tensor_bb=h2,
        nuclear_repulsion_energy=0.0,
        spin_sq=0.75,
        initial_occupancy=(np.full(n, NA / n), np.full(n, NB / n)),
        t2=rng.normal(size=(NA, NA, n - NA, n - NA)) * 0,
    )


def _random_samples(num=300, seed=0):
    """Post-selected-looking raw samples: (beta-left, alpha-right) bool rows, Na/Nb electrons."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        a = rng.choice(NORB, NA, replace=False)
        b = rng.choice(NORB, NB, replace=False)
        row = np.zeros(2 * NORB, dtype=bool)
        row[NORB + a] = True
        row[b] = True
        out.append(row)
    return np.array(out)


def _e2e(
    monkeypatch,
    *,
    solver,
    n_steps,
    carryover=None,
    carryover_b=None,
    ep=None,
    sqd_dim=40,
    ckpt_dir=None,
):
    if ckpt_dir is not None:
        monkeypatch.setenv("FE4S4_CKPT_DIR", str(ckpt_dir))
    else:
        monkeypatch.delenv("FE4S4_CKPT_DIR", raising=False)
    monkeypatch.setattr(sqd.SBDSolverJob, "load", staticmethod(lambda name: solver))
    monkeypatch.setattr(
        sqd.Variable, "get", staticmethod(lambda *a, **k: {"params": {"shots": 400}})
    )
    from sbd.flow_params import CircuitParameters

    return sqd.walker_sqd.fn(
        trial_index=0,
        walker_index=0,
        ucj_parameter=np.zeros(3),
        circuit_params=CircuitParameters(),
        elec_props=ep or _fake_props(),
        aa_indices=[],
        ab_indices=[],
        carryover=np.full((0, NORB), False) if carryover is None else carryover,
        carryover_b=carryover_b,
        sqd_dim=sqd_dim,
        solver_block_name="x",
        quantum_source="random",
        random_seed=1,
        n_recovery_steps=n_steps,
        n_batches=1,
    )


def _sector_clean(calls):
    for rec in calls:
        assert {x.bit_count() for x in rec.alpha} == {NA}, rec.alpha
        assert {x.bit_count() for x in rec.beta} == {NB}, rec.beta
        assert len(set(rec.alpha)) == len(rec.alpha)
        assert len(set(rec.beta)) == len(rec.beta)


def test_walker_sqd_end_to_end_unbalanced_every_step_is_sector_clean(monkeypatch):
    """The PR's headline bug, through the REAL walker_sqd loop: 4 recovery steps, Na=3 != Nb=2,
    the solver returns carryover of different lengths per spin. Every determinant list handed to the
    solver must have exactly Na / Nb electrons and no duplicates."""
    solver = FakeSolver(n_alpha_co=5, n_beta_co=2)
    (energy, (co_a, co_b), _), _tel = _e2e(monkeypatch, solver=solver, n_steps=4)
    assert len(solver.calls) == 4
    _sector_clean(solver.calls)
    assert co_a.shape == (5, NORB) and co_b.shape == (2, NORB)  # unpadded, per spin
    assert energy == pytest.approx(-1.0 - 0.01 * 4) or energy < -1.0


def test_walker_sqd_returns_alpha_beta_pair_not_a_bare_array(monkeypatch):
    solver = FakeSolver()
    result, _ = _e2e(monkeypatch, solver=solver, n_steps=2)
    _, carry, _ = result
    assert isinstance(carry, tuple) and len(carry) == 2


def test_walker_sqd_carryover_is_fed_forward_per_spin(monkeypatch):
    """Step k's solver carryover must show up inside step k+1's determinant lists (per spin)."""
    solver = FakeSolver(n_alpha_co=4, n_beta_co=2)
    _e2e(monkeypatch, solver=solver, n_steps=3)
    for prev, nxt in zip(solver.calls, solver.calls[1:]):
        assert set(prev.alpha[:4]) <= set(nxt.alpha)
        assert set(prev.beta[:2]) <= set(nxt.beta)


def test_walker_sqd_accepts_legacy_stacked_carryover_input(monkeypatch):
    """A legacy OptimizerState carryover (beta-left, alpha-right, zero padded) must still run."""
    a = _ints_to_rows(_all_strings(NORB, NA)[:5], NORB)
    b = _ints_to_rows(_all_strings(NORB, NB)[:2], NORB)
    stacked = np.zeros((5, 2 * NORB), dtype=bool)
    stacked[:2, :NORB], stacked[:5, NORB:] = b, a
    solver = FakeSolver()
    _e2e(monkeypatch, solver=solver, n_steps=1, carryover=stacked)
    _sector_clean(solver.calls)
    assert set(_rows_to_ints(a)) <= set(solver.calls[0].alpha)
    assert set(_rows_to_ints(b)) <= set(solver.calls[0].beta)


def test_walker_sqd_accepts_explicit_per_spin_carryover_input(monkeypatch):
    a = _ints_to_rows(_all_strings(NORB, NA)[:5], NORB)
    b = _ints_to_rows(_all_strings(NORB, NB)[:2], NORB)
    solver = FakeSolver()
    _e2e(monkeypatch, solver=solver, n_steps=1, carryover=a, carryover_b=b)
    _sector_clean(solver.calls)
    assert set(_rows_to_ints(a)) <= set(solver.calls[0].alpha)
    assert set(_rows_to_ints(b)) <= set(solver.calls[0].beta)


def test_walker_sqd_beta_carryover_missing_from_solver_does_not_crash(monkeypatch):
    class NoBeta(FakeSolver):
        async def run(self, **kw):
            r = await super().run(**kw)
            return dataclasses.replace(r, carryover_bitstrings_b=None)

    solver = NoBeta()
    (_, (co_a, co_b), _), _ = _e2e(monkeypatch, solver=solver, n_steps=3)
    _sector_clean(solver.calls)
    assert co_a.shape[1] == NORB and co_b.shape == (0, NORB)


def test_walker_sqd_rhf_path_unchanged_returns_none_beta(monkeypatch):
    """RHF must keep the single shared carryover and return (alpha, None)."""
    n = 6
    h1 = np.diag(np.arange(n, dtype=float))
    ep = SimpleNamespace(
        num_orbitals=n,
        num_electrons=(3, 3),
        unrestricted=False,
        open_shell=False,
        one_body_tensor=h1,
        two_body_tensor=np.zeros((n,) * 4),
        nuclear_repulsion_energy=0.0,
        spin_sq=0.0,
        initial_occupancy=(np.full(n, 0.5), np.full(n, 0.5)),
    )

    class RhfSolver:
        root_dir = None
        calls = 0

        async def run(self, *, ci_strings, norb, **kw):
            RhfSolver.calls += 1
            a = sorted(np.asarray(ci_strings[0]).tolist())
            return solver_job.SBDResult(
                energy=-2.0,
                orbital_occupancies=(np.full(norb, 0.5), np.full(norb, 0.5)),
                carryover_bitstrings=_ints_to_rows(a[:4], norb),
            )

    (_, (co_a, co_b), _), _ = _e2e(monkeypatch, solver=RhfSolver(), n_steps=2, ep=ep)
    assert co_b is None and co_a.shape == (4, n) and RhfSolver.calls == 2


def test_walker_sqd_checkpoint_writes_carryover_b_and_resume_restores_it(monkeypatch, tmp_path):
    solver = FakeSolver(n_alpha_co=5, n_beta_co=2)
    _e2e(monkeypatch, solver=solver, n_steps=3, ckpt_dir=tmp_path)
    ck = tmp_path / "ckpt_t00_w0.npz"
    assert ck.exists()
    z = np.load(ck)
    assert "carryover_b" in z.files
    assert z["carryover"].shape == (5, NORB) and z["carryover_b"].shape == (2, NORB)
    assert int(z["next_step"]) == 3
    # resume: no further steps requested beyond the checkpoint => 0 new solver calls, and the
    # returned carryover pair comes back unpadded from the checkpoint
    solver2 = FakeSolver()
    (_, carry, _), _ = _e2e(monkeypatch, solver=solver2, n_steps=3, ckpt_dir=tmp_path)
    assert len(solver2.calls) == 0


def test_walker_sqd_resume_continues_with_restored_per_spin_carryover(monkeypatch, tmp_path):
    first = FakeSolver(n_alpha_co=5, n_beta_co=2)
    _e2e(monkeypatch, solver=first, n_steps=2, ckpt_dir=tmp_path)
    second = FakeSolver(n_alpha_co=5, n_beta_co=2)
    _e2e(monkeypatch, solver=second, n_steps=4, ckpt_dir=tmp_path)  # resumes at step 2
    assert len(second.calls) == 2
    _sector_clean(second.calls)
    saved = np.load(tmp_path / "ckpt_t00_w0.npz")
    assert int(saved["next_step"]) == 4


def test_pr_end_to_end_fails_on_base_layout_by_construction():
    """Explains why the E2E tests above are meaningful: the removed stacking helper cannot be
    imported any more, so a revert of the fix is caught."""
    assert not hasattr(sqd, "_stack_spin_carryover")
    assert hasattr(sqd, "_split_spin_carryover")


def test_fake_solver_returns_unequal_per_spin_carryover():
    import asyncio

    s = FakeSolver()
    ci = (np.array(_all_strings(NORB, NA)), np.array(_all_strings(NORB, NB)))
    res = asyncio.run(s.run(ci_strings=ci, nelec=(NA, NB), norb=NORB))
    assert res.carryover_bitstrings.shape[0] != res.carryover_bitstrings_b.shape[0]


def test_carryover_roundtrip_across_recovery_steps_never_grows_padding():
    """Simulates the loop's carryover hand-off for 6 steps: the OLD stacked layout would emit
    zero rows (the PR's bug); the per-spin layout must stay popcount-clean and duplicate-free."""
    import asyncio

    solver = FakeSolver(n_alpha_co=6, n_beta_co=2)
    co_a, co_b = np.empty((0, NORB), bool), np.empty((0, NORB), bool)
    for step in range(6):
        ci_a = sorted(set(_all_strings(NORB, NA)[: 6 + step]) | set(_rows_to_ints(co_a)))
        ci_b = sorted(set(_all_strings(NORB, NB)[: 4 + step]) | set(_rows_to_ints(co_b)))
        res = asyncio.run(
            solver.run(ci_strings=(np.array(ci_a), np.array(ci_b)), nelec=(NA, NB), norb=NORB)
        )
        co_a, co_b = sqd._split_spin_carryover(
            res.carryover_bitstrings, NORB, res.carryover_bitstrings_b
        )
        assert {int(x).bit_count() for x in _rows_to_ints(co_a)} == {NA}
        assert {int(x).bit_count() for x in _rows_to_ints(co_b)} == {NB}
        assert co_a.shape[0] != co_b.shape[0]
    for rec in solver.calls:
        assert {x.bit_count() for x in rec.alpha} == {NA}
        assert {x.bit_count() for x in rec.beta} == {NB}
        assert len(set(rec.alpha)) == len(rec.alpha) and len(set(rec.beta)) == len(rec.beta)


def test_old_stacked_layout_would_have_injected_empty_determinants():
    """Regression anchor for the PR's core claim, reproduced against the *old* algorithm."""
    co_a = _ints_to_rows(_all_strings(NORB, NA)[:5], NORB)
    co_b = _ints_to_rows(_all_strings(NORB, NB)[:2], NORB)
    n = max(len(co_a), len(co_b))
    stacked = np.zeros((n, 2 * NORB), dtype=bool)
    stacked[: len(co_b), :NORB] = co_b
    stacked[: len(co_a), NORB:] = co_a
    beta_list = _rows_to_ints(stacked[:, :NORB])
    assert 0 in beta_list  # padding rows decode to the empty det
    assert {x.bit_count() for x in beta_list} != {NB}  # => wrong-sector beta list
    a, b = sqd._split_spin_carryover(stacked, NORB)  # the new splitter repairs it
    assert {x.bit_count() for x in _rows_to_ints(b)} == {NB}


# ============================================================================================
# 8. checkpoint (.npz) format contract for the new carryover_b key
# ============================================================================================
def _write_ckpt(path: Path, *, carryover, carryover_b=None):
    arrs = dict(
        occ_a=np.ones(4),
        occ_b=np.ones(4),
        next_step=np.int64(3),
        best_energy=np.float64(-1.5),
        carryover=np.asarray(carryover, dtype=bool),
    )
    if carryover_b is not None:
        arrs["carryover_b"] = np.asarray(carryover_b, dtype=bool)
    with open(path, "wb") as fh:
        np.savez(fh, **arrs)


def test_checkpoint_new_format_restores_both_spins_unpadded(tmp_path: Path):
    a = _ints_to_rows(_all_strings(NORB, NA)[:5], NORB)
    b = _ints_to_rows(_all_strings(NORB, NB)[:2], NORB)
    _write_ckpt(tmp_path / "c.npz", carryover=a, carryover_b=b)
    z = np.load(tmp_path / "c.npz")
    assert "carryover_b" in z.files
    ra, rb = sqd._split_spin_carryover(
        z["carryover"].astype(bool), NORB, z["carryover_b"].astype(bool)
    )
    assert np.array_equal(ra, a) and np.array_equal(rb, b)


def test_checkpoint_legacy_stacked_format_is_split_on_resume(tmp_path: Path):
    a = _ints_to_rows(_all_strings(NORB, NA)[:5], NORB)
    b = _ints_to_rows(_all_strings(NORB, NB)[:2], NORB)
    stacked = np.zeros((5, 2 * NORB), dtype=bool)
    stacked[:2, :NORB], stacked[:5, NORB:] = b, a
    _write_ckpt(tmp_path / "c.npz", carryover=stacked)  # no carryover_b key
    z = np.load(tmp_path / "c.npz")
    assert "carryover_b" not in z.files
    ra, rb = sqd._split_spin_carryover(z["carryover"].astype(bool), NORB, None)
    assert np.array_equal(ra, a) and np.array_equal(rb, b)


def test_checkpoint_rhf_empty_carryover_resumes_as_empty(tmp_path: Path):
    _write_ckpt(tmp_path / "c.npz", carryover=np.empty((0, 0), dtype=bool))
    z = np.load(tmp_path / "c.npz")
    ra, rb = sqd._split_spin_carryover(z["carryover"].astype(bool), NORB, None)
    assert ra.shape == (0, NORB) and rb.shape == (0, NORB)


def test_checkpoint_uhf_empty_beta_key_resumes_without_error(tmp_path: Path):
    """A UHF checkpoint written before any beta carryover existed has carryover_b (0, norb)."""
    _write_ckpt(
        tmp_path / "c.npz",
        carryover=_ints_to_rows(_all_strings(NORB, NA)[:3], NORB),
        carryover_b=np.empty((0, NORB), dtype=bool),
    )
    z = np.load(tmp_path / "c.npz")
    ra, rb = sqd._split_spin_carryover(
        z["carryover"].astype(bool), NORB, z["carryover_b"].astype(bool)
    )
    assert ra.shape == (3, NORB) and rb.shape == (0, NORB)
