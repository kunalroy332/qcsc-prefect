"""Local, hardware-free SQD validation harness (<=12-15 qubits).

This reproduces one SQD *walker pass* end-to-end without Fugaku or IBM hardware, so the
algorithm (configuration recovery, K-batch aggregation, particle-number / Sz enforcement,
spin behaviour) can be validated noiselessly and under a controlled bit-flip noise model.

It deliberately reuses the REAL pipeline functions from ``sbd.sqd`` (called via the Prefect
``.fn()`` accessor) so the tests exercise production logic, not a reimplementation. The only
substitution is the diagonalizer: instead of the C++ SBD solver we use
``qiskit_addon_sqd.fermion.solve_fermion`` (a pyscf-backed selected-CI solver), which is a
faithful stand-in at small sizes and additionally returns the exact ``<S^2>`` of the SQD
eigenvector -- the quantity we need to check spin-contamination removal.

Noise model: ``inject_bitflip_noise`` flips each measured bit with probability ``p``. ``p=0``
is the noiseless ceiling; raising ``p`` mimics hardware noise scattering samples away from
Hartree-Fock; modelling error mitigation = running with a *lower* effective ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest import mock

import ffsim
import numpy as np

# The pipeline modules call get_run_logger() at import/run time; silence it for local use.
import qcsc_workflow_utility.chem as chem  # noqa: E402
from qiskit_addon_sqd.fermion import solve_fermion

chem.get_run_logger = lambda: mock.MagicMock()

from sbd import lucj, sqd  # noqa: E402

sqd.get_run_logger = lambda: mock.MagicMock()
lucj.get_run_logger = lambda: mock.MagicMock()


@dataclass
class PassResult:
    """Outcome of one (multi-recovery, multi-batch) SQD walker pass."""

    energy: float
    spin_sq: float
    nuclear_repulsion: float
    occ_a: np.ndarray
    occ_b: np.ndarray
    # Per recovery step: (best_energy, best_spin_sq, alpha_summary, beta_summary, sd_frac_alpha).
    steps: list[tuple[float, float, str, str, float]] = field(default_factory=list)


def build_uhf_props(atom: str, spin: int, basis: str = "sto-3g") -> chem.ElectronicProperties:
    """Small open-shell ElectronicProperties via the production integral builder."""
    return chem.compute_molecular_integrals_from_geometry.fn(
        atom=atom, basis=basis, unrestricted=True, spin=spin
    )


def _heavy_hex_indices(norb: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The same LUCJ locality the production flow uses (main.py)."""
    aa = [(p, p + 1) for p in range(norb - 1)]
    ab = [(p, p) for p in range(0, norb, 4)]
    return aa, ab


def _connectivity_pairs(norb: int, connectivity):
    """Resolve a connectivity spec into the (aa, ab, bb) interaction_pairs for UCJOpSpinUnbalanced.

    - "heavy-hex": production hardware locality (aa nearest-neighbor, ab on every 4th orbital).
    - "full" / None: interaction_pairs=None (all orbital pairs; richest, non-heavy-hex hardware).
    - int R: an *intermediate* locality -- aa within distance R, ab on-site for every orbital.
      R=1 reduces toward heavy-hex; large R approaches full. Lets us map the connectivity ladder.
    """
    if connectivity == "full" or connectivity is None:
        return None
    if connectivity == "heavy-hex":
        aa, ab = _heavy_hex_indices(norb)
        return (aa, ab, _default_bb := lucj._default_bb_indices(aa, None))
    if isinstance(connectivity, int):
        radius = connectivity
        aa = [(p, q) for p in range(norb) for q in range(p + 1, min(p + radius + 1, norb))]
        ab = [(p, p) for p in range(norb)]
        bb = list(aa)
        return (aa, ab, bb)
    raise ValueError(f"unknown connectivity spec: {connectivity!r}")


def _build_ucj_op(elec_props, *, n_lucj_layers, connectivity):
    """Build the UCJ operator via the PRODUCTION optimize=True parametrization.

    Mirrors ``lucj._initialize_ucj_parameters_uhf._t2_to_ucj_parameters`` exactly (optimize=True,
    n_reps=L+1, truncate the last rep into final_orbital_rotation) so connectivity comparisons are
    valid. Using bare from_t_amplitudes WITHOUT optimize=True collapses the state to ~HF and gives
    meaningless results.
    """
    pairs = _connectivity_pairs(elec_props.num_orbitals, connectivity)
    t2 = (elec_props.t2, elec_props.t2_ab, elec_props.t2_bb)
    tmp = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        t2=t2, n_reps=n_lucj_layers + 1, interaction_pairs=pairs,
        optimize=True, options={"maxiter": 50},
    )
    return ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=tmp.diag_coulomb_mats[:-1],
        orbital_rotations=tmp.orbital_rotations[:-1],
        final_orbital_rotation=tmp.orbital_rotations[-1],
    )


def prepare_state_and_sample(
    elec_props: chem.ElectronicProperties,
    *,
    n_lucj_layers: int,
    shots: int,
    seed: int,
    connectivity="heavy-hex",
    noise_p: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the LUCJ state with ffsim, sample it, and optionally inject bit-flip noise.

    ``connectivity`` selects the UCJ interaction pairs ("heavy-hex", "full"/None, or an int
    locality radius). ``noise_p`` flips each measured bit with probability p (0 = noiseless).
    Returns the raw (bitstring_matrix_bool, probabilities) in the (beta-left, alpha-right) layout
    that feeds straight into ``recover_configurations`` / ``subsample_open_shell``.
    """
    norb = elec_props.num_orbitals
    nelec = elec_props.num_electrons

    ucj_op = _build_ucj_op(
        elec_props, n_lucj_layers=n_lucj_layers, connectivity=connectivity
    )
    vec = ffsim.hartree_fock_state(norb, nelec)
    vec = ffsim.apply_unitary(vec, ucj_op, norb=norb, nelec=nelec)

    rng = np.random.default_rng(seed)
    strings = ffsim.sample_state_vector(
        vec, norb=norb, nelec=nelec, shots=shots, seed=rng, concatenate=True
    )
    # strings are 'beta'+'alpha', each 2*norb chars of '0'/'1'.
    mat = np.array([[c == "1" for c in s] for s in strings], dtype=bool)
    if noise_p > 0.0:
        mat = inject_bitflip_noise(mat, noise_p, rng)
    probs = np.ones(len(strings), dtype=np.float64) / len(strings)
    return mat, probs


def inject_bitflip_noise(
    bitstrings: np.ndarray, p: float, rng: np.random.Generator
) -> np.ndarray:
    """Flip each bit independently with probability ``p`` (symmetric bit-flip channel)."""
    if p <= 0.0:
        return bitstrings.copy()
    flips = rng.random(bitstrings.shape) < p
    return np.logical_xor(bitstrings, flips)


def _spinless_eri_and_hcore(elec_props: chem.ElectronicProperties):
    """solve_fermion expects a single (restricted-style) hcore + eri.

    For these validation experiments we diagonalize over the alpha-spin spatial integrals
    (open_shell=True still treats alpha/beta determinant strings independently). This is exact
    for the local CI within the (na, nb) sector built from the same one-/two-body alpha tensors;
    it is a stand-in for the C++ UHF solver, sufficient to study recovery/noise/spin behaviour.
    """
    return elec_props.one_body_tensor, elec_props.two_body_tensor


def run_one_pass(
    elec_props: chem.ElectronicProperties,
    raw_bits: np.ndarray,
    raw_probs: np.ndarray,
    *,
    sqd_dim: int,
    n_batches: int = 1,
    n_recovery_steps: int = 1,
    rng_seed: int = 0,
) -> PassResult:
    """Mirror walker_sqd's inner loop with solve_fermion as the diagonalizer.

    recover -> postselect -> [K batches: subsample + solve_fermion] -> min energy / mean occ
    -> feed mean occ to next recovery pass. Open-shell (per-spin) path only.
    """
    # Use a dedicated generator so tests are deterministic and independent of MODULE_RNG state.
    sqd.MODULE_RNG = np.random.default_rng(rng_seed)

    norb = elec_props.num_orbitals
    na, nb = elec_props.num_electrons
    hcore, eri = _spinless_eri_and_hcore(elec_props)
    e_nuc = elec_props.nuclear_repulsion_energy

    avg_occ = elec_props.initial_occupancy
    best_energy = None
    best_spin_sq = None
    best_occ_a = None
    best_occ_b = None
    steps: list[tuple[float, float, str, str, float]] = []

    empty = np.empty((0, norb), dtype=bool)

    for _ in range(n_recovery_steps):
        bitstrings, probs = sqd.recover_configurations.fn(
            bitstring_matrix=raw_bits,
            probabilities=raw_probs,
            avg_occupancies=avg_occ,
            num_elec_a=na,
            num_elec_b=nb,
            rand_seed=sqd.MODULE_RNG,
        )
        bs_post, p_post = sqd.postselect_bitstrings.fn(
            bitstring_matrix=bitstrings,
            probabilities=probs,
            hamming_right=na,
            hamming_left=nb,
        )
        if bs_post.shape[0] == 0:
            # No valid-sector configs survived; record a sentinel and stop.
            steps.append((float("nan"), float("nan"), "empty", "empty", 0.0))
            break

        step_best_e = None
        step_occ_a = np.zeros(norb)
        step_occ_b = np.zeros(norb)
        step_spin_sq = float("nan")
        sd_frac_a = 0.0
        a_sum = b_sum = "empty"

        for batch in range(n_batches):
            ci_a, ci_b = sqd.subsample_open_shell.fn(
                bitstring_matrix=bs_post,
                probabilities=p_post,
                carryover_a=empty,
                carryover_b=empty,
                subspace_dim=sqd_dim,
                norb=norb,
                num_elec_a=na,
                num_elec_b=nb,
            )
            energy, sci_state, occs, spin_sq = solve_fermion(
                (ci_a, ci_b), hcore, eri, open_shell=True
            )
            energy_total = float(energy) + e_nuc
            step_occ_a += np.asarray(occs[0], dtype=np.float64)
            step_occ_b += np.asarray(occs[1], dtype=np.float64)
            if step_best_e is None or energy_total < step_best_e:
                step_best_e = energy_total
                step_spin_sq = float(spin_sq)
                if batch == 0:
                    a_sum = sqd._excitation_summary(ci_a, na)
                    b_sum = sqd._excitation_summary(ci_b, nb)
                    sd_frac_a = _sd_fraction(ci_a, na)

        step_occ_a /= n_batches
        step_occ_b /= n_batches
        avg_occ = (step_occ_a, step_occ_b)
        steps.append((step_best_e, step_spin_sq, a_sum, b_sum, sd_frac_a))

        if best_energy is None or step_best_e < best_energy:
            best_energy = step_best_e
            best_spin_sq = step_spin_sq
            best_occ_a = step_occ_a
            best_occ_b = step_occ_b

    return PassResult(
        energy=best_energy,
        spin_sq=best_spin_sq,
        nuclear_repulsion=e_nuc,
        occ_a=best_occ_a,
        occ_b=best_occ_b,
        steps=steps,
    )


def _sd_fraction(ci_strings: np.ndarray, num_elec: int) -> float:
    """Fraction of determinants that are HF / single / double excitations (couple to HF)."""
    ci = np.asarray(ci_strings, dtype=np.int64).reshape(-1)
    if ci.size == 0:
        return 0.0
    hf = (1 << num_elec) - 1
    exc = np.array([bin(int(x) ^ hf).count("1") // 2 for x in ci])
    return float(np.mean(exc <= 2))


def post_select_sector_ok(
    bs_post: np.ndarray, norb: int, num_elec_a: int, num_elec_b: int
) -> bool:
    """True iff every post-selected config has exactly (na alpha, nb beta) bits set."""
    if bs_post.shape[0] == 0:
        return True
    beta = bs_post[:, :norb].sum(axis=1)
    alpha = bs_post[:, norb:].sum(axis=1)
    return bool(np.all(alpha == num_elec_a) and np.all(beta == num_elec_b))
