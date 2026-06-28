"""Local, hardware-free validation of the SQD algorithm (<=14 qubits, runs in seconds).

These tests use ``local_sqd_harness`` to run the REAL pipeline functions (recovery, post-select,
subsample, K-batch aggregation) against a local ``solve_fermion`` diagonalizer, on small
open-shell systems. They answer the chemistry/algorithm questions that the expensive 50q Fugaku
runs could not isolate:

* the noiseless algorithm captures correlation and produces a (near) pure spin state;
* particle-number / Sz is exactly enforced;
* SQD removes the spin contamination present in the UHF reference;
* configuration recovery repairs wrong-particle-number bitstrings back into the valid sector;
* the K-batch logic returns min-energy and mean-occupancy.

A deliberate NEGATIVE finding is documented in ``test_small_system_cannot_reproduce_deadwood``:
the 50q "95% high-excitation deadwood" collapse is a large-Hilbert-space phenomenon and does NOT
reproduce at 12-14 qubits, so the local harness validates mechanisms, not the 50q noise wall.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyscf

sys.path.insert(0, str(Path(__file__).parent))
import local_sqd_harness as H  # noqa: E402

# Small open-shell references. NH triplet (12 qubits) carries real UHF spin contamination
# (<S^2>~2.10 vs pure 2.0); OH doublet (12 qubits) is a clean nearly-pure doublet.
NH = dict(atom="N 0 0 0; H 0 0 1.3", spin=2, pure_s2=2.0)
OH = dict(atom="O 0 0 0; H 0 0 0.97", spin=1, pure_s2=0.75)


def _uhf_reference(atom: str, spin: int) -> tuple[float, float]:
    mol = pyscf.gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    mf = pyscf.scf.UHF(mol).run(verbose=0)
    return float(mf.e_tot), float(mf.spin_square()[0])


def test_noiseless_recovers_correlation():
    """Noiseless SQD energy beats UHF and yields a (near) pure spin state."""
    e_uhf, _ = _uhf_reference(**{k: NH[k] for k in ("atom", "spin")})
    ep = H.build_uhf_props(atom=NH["atom"], spin=NH["spin"])
    mat, probs = H.prepare_state_and_sample(ep, n_lucj_layers=2, shots=20000, seed=1)
    res = H.run_one_pass(ep, mat, probs, sqd_dim=4000, n_batches=1, n_recovery_steps=1)

    # Correlation captured: SQD is variationally below UHF.
    assert res.energy < e_uhf - 1e-3, f"SQD {res.energy} not below UHF {e_uhf}"
    # Near-pure spin state (the FCI ground state in this sector is a pure triplet).
    assert abs(res.spin_sq - NH["pure_s2"]) < 0.05, f"<S^2>={res.spin_sq} not ~{NH['pure_s2']}"


def test_sz_exactly_enforced():
    """Every post-selected config sits in exactly the (na alpha, nb beta) sector => Sz exact."""
    ep = H.build_uhf_props(atom=NH["atom"], spin=NH["spin"])
    norb = ep.num_orbitals
    na, nb = ep.num_electrons
    mat, probs = H.prepare_state_and_sample(ep, n_lucj_layers=2, shots=20000, seed=1)
    # Inject heavy noise so post-select has real work to do.
    rng = np.random.default_rng(11)
    noisy = H.inject_bitflip_noise(mat, 0.15, rng)

    bits, prob = H.sqd.recover_configurations.fn(
        bitstring_matrix=noisy, probabilities=probs,
        avg_occupancies=ep.initial_occupancy, num_elec_a=na, num_elec_b=nb,
        rand_seed=np.random.default_rng(0),
    )
    bs_post, _ = H.sqd.postselect_bitstrings.fn(
        bitstring_matrix=bits, probabilities=prob, hamming_right=na, hamming_left=nb,
    )
    assert H.post_select_sector_ok(bs_post, norb, na, nb), "post-select leaked wrong sector"


def test_spin_contamination_removed():
    """SQD eigenvector <S^2> is closer to the pure-spin value than the UHF reference."""
    _, s2_uhf = _uhf_reference(**{k: NH[k] for k in ("atom", "spin")})
    assert s2_uhf > NH["pure_s2"] + 0.05, "NH reference should be visibly contaminated"

    ep = H.build_uhf_props(atom=NH["atom"], spin=NH["spin"])
    mat, probs = H.prepare_state_and_sample(ep, n_lucj_layers=2, shots=20000, seed=1)
    res = H.run_one_pass(ep, mat, probs, sqd_dim=4000, n_batches=1, n_recovery_steps=1)

    contam_uhf = abs(s2_uhf - NH["pure_s2"])
    contam_sqd = abs(res.spin_sq - NH["pure_s2"])
    assert contam_sqd < contam_uhf, (
        f"SQD did not reduce contamination: UHF {contam_uhf:.4f} -> SQD {contam_sqd:.4f}"
    )


def test_production_spin_square_helper_matches_harness():
    """solver_job.spin_square_from_subspace reproduces the same <S^2> as the harness solve.

    This validates the production-side, RDM-convention-free <S^2> diagnostic against the
    subspace the SQD pass actually built, so the Fugaku/production path can report <S^2>.
    """
    from sbd.solver_job import spin_square_from_subspace

    ep = H.build_uhf_props(atom=NH["atom"], spin=NH["spin"])
    na, nb = ep.num_electrons
    mat, probs = H.prepare_state_and_sample(ep, n_lucj_layers=2, shots=20000, seed=1)

    # Reproduce one recovery+subsample to get the exact determinant subspace.
    bits, prob = H.sqd.recover_configurations.fn(
        bitstring_matrix=mat, probabilities=probs,
        avg_occupancies=ep.initial_occupancy, num_elec_a=na, num_elec_b=nb,
        rand_seed=np.random.default_rng(0),
    )
    bs_post, p_post = H.sqd.postselect_bitstrings.fn(
        bitstring_matrix=bits, probabilities=prob, hamming_right=na, hamming_left=nb,
    )
    H.sqd.MODULE_RNG = np.random.default_rng(0)
    empty = np.empty((0, ep.num_orbitals), dtype=bool)
    ci_a, ci_b = H.sqd.subsample_open_shell.fn(
        bitstring_matrix=bs_post, probabilities=p_post,
        carryover_a=empty, carryover_b=empty, subspace_dim=4000,
        norb=ep.num_orbitals, num_elec_a=na, num_elec_b=nb,
    )
    s2 = spin_square_from_subspace(
        (ci_a, ci_b), ep.one_body_tensor, ep.two_body_tensor, open_shell=True
    )
    # NH triplet: SQD eigenvector is a near-pure triplet (well below the UHF 2.10 contamination).
    assert abs(s2 - NH["pure_s2"]) < 0.05, f"<S^2>={s2} not ~{NH['pure_s2']}"


def test_recovery_repairs_particle_number():
    """recover_configurations pulls wrong-particle-number scatter back into the valid sector.

    This is the direct local test of the mechanism the user asked about: of a large pool of
    random (mostly wrong-sector) bitstrings, recovery should land far more of them in the
    correct (na, nb) sector than were there to begin with.
    """
    ep = H.build_uhf_props(atom=NH["atom"], spin=NH["spin"])
    norb = ep.num_orbitals
    na, nb = ep.num_electrons
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 2, size=(5000, 2 * norb)).astype(bool)
    probs = np.ones(5000) / 5000

    def frac_in_sector(m: np.ndarray) -> float:
        a = m[:, norb:].sum(1)
        b = m[:, :norb].sum(1)
        return float(np.mean((a == na) & (b == nb)))

    before = frac_in_sector(raw)
    recovered, _ = H.sqd.recover_configurations.fn(
        bitstring_matrix=raw, probabilities=probs,
        avg_occupancies=ep.initial_occupancy, num_elec_a=na, num_elec_b=nb,
        rand_seed=np.random.default_rng(0),
    )
    after = frac_in_sector(recovered)
    # Recovery's contract: EVERY output bitstring is repaired into the valid (na, nb) sector.
    a = recovered[:, norb:].sum(1)
    b = recovered[:, :norb].sum(1)
    assert np.all(a == na) and np.all(b == nb), "recovery left wrong-sector configs"
    # From ~3% valid scatter, recovery should bring essentially everything into the sector.
    assert before < 0.1, f"raw scatter unexpectedly already in-sector ({before:.2f})"
    assert after > 0.99, f"recovery did not repair particle number (frac in sector {after:.3f})"


def test_kbatch_min_energy_and_mean_occ():
    """K-batch returns the minimum batch energy and the across-batch mean occupancy."""
    ep = H.build_uhf_props(atom=NH["atom"], spin=NH["spin"])
    mat, probs = H.prepare_state_and_sample(ep, n_lucj_layers=2, shots=20000, seed=1)

    single = H.run_one_pass(ep, mat, probs, sqd_dim=2000, n_batches=1, n_recovery_steps=1)
    multi = H.run_one_pass(ep, mat, probs, sqd_dim=2000, n_batches=5, n_recovery_steps=1)

    # Best-of-K can only be <= a single batch's energy (variational min over more draws).
    assert multi.energy <= single.energy + 1e-9, (
        f"K-batch min-energy {multi.energy} worse than single {single.energy}"
    )
    # Occupancies remain physical: per-spin sums equal the electron counts.
    na, nb = ep.num_electrons
    assert abs(multi.occ_a.sum() - na) < 1e-6
    assert abs(multi.occ_b.sum() - nb) < 1e-6


def test_small_system_cannot_reproduce_deadwood():
    """Documented negative result: at 12 qubits, bit-flip noise does NOT cause the 50q collapse.

    The (na, nb) Hilbert space is tiny (~120 configs for NH), so heavy noise fills the WHOLE
    sector rather than scattering into high-excitation deadwood -- which gives the exact FCI
    answer, not a degraded one. This confirms the 50q plateau is a large-Hilbert-space
    phenomenon that small local systems cannot reproduce; the harness validates mechanisms only.
    """
    ep = H.build_uhf_props(atom=NH["atom"], spin=NH["spin"])
    mat, probs = H.prepare_state_and_sample(ep, n_lucj_layers=2, shots=30000, seed=1)
    rng = np.random.default_rng(7)
    res_clean = H.run_one_pass(ep, mat, probs, sqd_dim=4000, n_batches=1, n_recovery_steps=1)
    heavy = H.inject_bitflip_noise(mat, 0.35, rng)
    res_heavy = H.run_one_pass(ep, heavy, probs, sqd_dim=4000, n_batches=1, n_recovery_steps=1)

    # Heavy noise does not degrade the small-system energy (it saturates the full sector).
    assert res_heavy.energy <= res_clean.energy + 1e-3, (
        "unexpected: heavy noise degraded the tiny-system energy -- deadwood DID appear"
    )
