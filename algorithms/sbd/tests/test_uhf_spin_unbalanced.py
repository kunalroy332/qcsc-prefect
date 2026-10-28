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

from pathlib import Path
from unittest import mock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _silence_prefect_logger(monkeypatch):
    """The integral/IO helpers call Prefect's get_run_logger; stub it for unit tests."""
    import qcsc_workflow_utility.chem as chem
    from sbd import solver_job

    monkeypatch.setattr(chem, "get_run_logger", lambda: mock.MagicMock())
    monkeypatch.setattr(solver_job, "get_run_logger", lambda: mock.MagicMock())


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
        [np.r_[_ints_to_rows([b], norb)[0], _ints_to_rows([a], norb)[0]]
         for a, b in zip(_random_strings(rng, norb, na, 40), _random_strings(rng, norb, nb, 40))],
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
        _write_dets_bin(tmp_path / "a.bin", np.array([0b0111, 0b0011]), 4, spin="alpha",
                        num_elec=3)
    with pytest.raises(ValueError, match="do not have 2 electrons"):
        _write_dets_bin(tmp_path / "b.bin", np.array([0b0011, 0]), 4, spin="beta", num_elec=2)


def test_write_dets_bin_rejects_duplicates(tmp_path: Path):
    from sbd.solver_job import _write_dets_bin

    with pytest.raises(ValueError, match="duplicate"):
        _write_dets_bin(tmp_path / "a.bin", np.array([0b0011, 0b0101, 0b0011]), 4,
                        spin="alpha", num_elec=2)


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
    with mock.patch.object(chem, "get_run_logger", lambda: mock.MagicMock()), \
            mock.patch.dict("os.environ", {"SBD_UHF_CACHE": "0", "FE4S4_AF_GROUPS": ""}):
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
        elec_props=oh_doublet_props, aa_indices=aa, ab_indices=ab, num_walkers=3,
        randomization_factor=0.2, n_lucj_layers=1, ucj_optimize=False,
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
        elec_props=ep, aa_indices=[(p, p + 1) for p in range(norb - 1)],
        ab_indices=[(p, p) for p in range(norb)], num_walkers=3, randomization_factor=0.2,
        n_lucj_layers=1, ucj_optimize=False,
    )
    assert len(params) == 3 and not np.allclose(params[0], params[2])
