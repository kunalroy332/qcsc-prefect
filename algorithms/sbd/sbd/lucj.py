# Workflow for observability demo on Miyabi

import ffsim
import numpy as np
from prefect import task
from qcsc_workflow_utility.chem import (
    ElectronicProperties,
    NpStrict1DArrayF64,
    NpStrict2DArrayF64,
    NpStrict4DArrayF64,
)
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

MODULE_RNG = np.random.default_rng(seed=4520)


def _default_bb_indices(
    aa_indices: list[tuple[int, int]],
    bb_indices: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    """Beta-beta interaction pairs default to the same orbital topology as alpha-alpha."""
    return aa_indices if bb_indices is None else bb_indices


@task
def _pair_symmetric_noise(shape: tuple[int, ...], scale: float) -> np.ndarray:
    """Uniform noise in [-scale/2, scale/2) with the RHF t2 pair symmetry t2[i,j,a,b] = t2[j,i,b,a].

    Independent elementwise noise breaks this symmetry, and ffsim's from_t_amplitudes (validate)
    then rejects the drifted t2, which crashed every randomized walker (num_walkers > 1).
    """
    r = scale * (MODULE_RNG.random(shape) - 0.5)
    return 0.5 * (r + r.transpose(1, 0, 3, 2))


def _antisymmetric_noise(shape: tuple[int, ...], scale: float) -> np.ndarray:
    """Uniform-scale noise antisymmetric in (i, j) and in (a, b), as same-spin t2aa / t2bb must be.

    The 1/4 keeps the entry magnitude comparable to the unsymmetrized [-scale/2, scale/2) draw.
    """
    r = scale * (MODULE_RNG.random(shape) - 0.5)
    r = r - r.transpose(1, 0, 2, 3)
    return 0.25 * (r - r.transpose(0, 1, 3, 2))


def initialize_ucj_parameters(
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    num_walkers: int,
    randomization_factor: float,
    n_lucj_layers: int,
    bb_indices: list[tuple[int, int]] | None = None,
    ucj_optimize: bool = True,
) -> NpStrict2DArrayF64:
    global MODULE_RNG

    if elec_props.unrestricted:
        return _initialize_ucj_parameters_uhf(
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            bb_indices=_default_bb_indices(aa_indices, bb_indices),
            num_walkers=num_walkers,
            randomization_factor=randomization_factor,
            n_lucj_layers=n_lucj_layers,
            ucj_optimize=ucj_optimize,
        )

    def _t2_to_ucj_parameters(t2: NpStrict4DArrayF64) -> NpStrict1DArrayF64:
        nonlocal aa_indices
        nonlocal ab_indices
        nonlocal n_lucj_layers

        # optimize=True performs the "compressed double factorization": the kept DF terms are
        # re-optimized to best approximate the full UCCSD generator. This can raise the prepared
        # state's <H> (Trotter error) but for SQD that is beneficial — it spreads the wavefunction
        # over many more configurations, producing a far more diverse sample set to diagonalize
        # over (measured ~6x participation ratio vs naive truncation, which is near single-HF).
        # ucj_optimize=False disables it (bare truncated DF) for the ansatz-quality study.
        optimize_kwargs = (
            {"optimize": True, "options": {"maxiter": 50}} if ucj_optimize else {}
        )
        tmp_operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=t2,
            n_reps=n_lucj_layers + 1,
            interaction_pairs=(aa_indices, ab_indices),
            **optimize_kwargs,
        )
        truncated_ucj_op = ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=tmp_operator.diag_coulomb_mats[:-1],
            orbital_rotations=tmp_operator.orbital_rotations[:-1],
            final_orbital_rotation=tmp_operator.orbital_rotations[-1],
        )
        return truncated_ucj_op.to_parameters(interaction_pairs=(aa_indices, ab_indices))

    # First walker is the bare CCSD parameters
    initial_params = [_t2_to_ucj_parameters(t2=elec_props.t2)]

    # The rest of walkers are randomized parameters
    for _ in range(num_walkers - 1):
        rand_values = _pair_symmetric_noise(elec_props.t2.shape, randomization_factor)
        drifted_params = _t2_to_ucj_parameters(t2=elec_props.t2 + rand_values)
        initial_params.append(drifted_params)
    return initial_params


def _initialize_ucj_parameters_uhf(
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    bb_indices: list[tuple[int, int]],
    num_walkers: int,
    randomization_factor: float,
    n_lucj_layers: int,
    ucj_optimize: bool = True,
) -> NpStrict2DArrayF64:
    """Spin-unbalanced (UHF) counterpart of :func:`initialize_ucj_parameters`.

    The UCCSD t2 tuple ``(t2aa, t2ab, t2bb)`` lives across ``elec_props.t2`` / ``.t2_ab`` /
    ``.t2_bb``; interaction pairs become a 3-tuple ``(aa, ab, bb)``. Randomization perturbs all
    three blocks independently (they have different shapes when Na != Nb); the same-spin blocks get
    antisymmetric noise so the drifted amplitudes stay valid t2aa / t2bb.
    """
    global MODULE_RNG
    interaction_pairs = (aa_indices, ab_indices, bb_indices)

    def _t2_to_ucj_parameters(
        t2: tuple[NpStrict4DArrayF64, NpStrict4DArrayF64, NpStrict4DArrayF64],
    ) -> NpStrict1DArrayF64:
        # Match the RHF path: optimize=True (compressed DF) spreads the state over many more
        # configurations for a diverse SQD sample. This was previously MISSING on the UHF path,
        # which left UHF subspaces near-HF and made UHF circuits shallower than RHF (audit P2).
        optimize_kwargs = (
            {"optimize": True, "options": {"maxiter": 50}} if ucj_optimize else {}
        )
        tmp_operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2=t2,
            n_reps=n_lucj_layers + 1,
            interaction_pairs=interaction_pairs,
            **optimize_kwargs,
        )
        truncated_ucj_op = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=tmp_operator.diag_coulomb_mats[:-1],
            orbital_rotations=tmp_operator.orbital_rotations[:-1],
            final_orbital_rotation=tmp_operator.orbital_rotations[-1],
        )
        return truncated_ucj_op.to_parameters(interaction_pairs=interaction_pairs)

    t2_tuple = (elec_props.t2, elec_props.t2_ab, elec_props.t2_bb)

    # First walker is the bare UCCSD parameters
    initial_params = [_t2_to_ucj_parameters(t2=t2_tuple)]

    # The rest of walkers are randomized parameters; perturb each spin block independently.
    # t2ab has no index symmetry; t2aa / t2bb must stay antisymmetric (ffsim validates this).
    for _ in range(num_walkers - 1):
        t2aa, t2ab, t2bb = t2_tuple
        drifted = (
            t2aa + _antisymmetric_noise(t2aa.shape, randomization_factor),
            t2ab + randomization_factor * (MODULE_RNG.random(t2ab.shape) - 0.5),
            t2bb + _antisymmetric_noise(t2bb.shape, randomization_factor),
        )
        initial_params.append(_t2_to_ucj_parameters(t2=drifted))
    return initial_params


@task
def create_lucj_circuit(
    ucj_parameter: NpStrict1DArrayF64,
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    n_lucj_layers: int,
    use_reset_mitigation: bool,
    bb_indices: list[tuple[int, int]] | None = None,
) -> QuantumCircuit:
    qreg = QuantumRegister(2 * elec_props.num_orbitals, name="q")
    creg_test = ClassicalRegister(2 * elec_props.num_orbitals, name="test")
    creg_meas = ClassicalRegister(2 * elec_props.num_orbitals, name="meas")

    regs = [qreg, creg_meas]
    if use_reset_mitigation:
        regs.append(creg_test)

    circ = QuantumCircuit(*regs)
    if use_reset_mitigation:
        circ.measure(qreg, creg_test)
        circ.barrier()
    circ.append(
        ffsim.qiskit.PrepareHartreeFockJW(
            norb=elec_props.num_orbitals,
            nelec=elec_props.num_electrons,
        ),
        qargs=qreg,
    )
    if elec_props.unrestricted:
        interaction_pairs = (aa_indices, ab_indices, _default_bb_indices(aa_indices, bb_indices))
        ucj_op = ffsim.UCJOpSpinUnbalanced.from_parameters(
            params=ucj_parameter,
            norb=elec_props.num_orbitals,
            n_reps=n_lucj_layers,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True,
        )
        circ.append(
            ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op=ucj_op),
            qargs=qreg,
        )
    else:
        ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
            params=ucj_parameter,
            norb=elec_props.num_orbitals,
            n_reps=n_lucj_layers,
            interaction_pairs=(aa_indices, ab_indices),
            with_final_orbital_rotation=True,
        )
        circ.append(
            ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op=ucj_op),
            qargs=qreg,
        )
    circ.measure(qreg, creg_meas)
    return circ
