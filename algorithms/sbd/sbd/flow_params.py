# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)

from typing import Literal

from pydantic import BaseModel, Field


class CircuitParameters(BaseModel):
    """Configuration for LUCJ circuit construction."""

    n_lucj_layers: int = Field(
        default=2,
        description="Number of LUCJ circuit block repetitions.",
        title="LUCJ Circuit Layers",
        ge=1,
    )

    use_reset_mitigation: bool = Field(
        default=True,
        description="Set True to use reset error mitigation scheme.",
        title="Reset Mitigation",
    )

    optimization_level: int = Field(
        default=3,
        description="Optimization level of Qiskit transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
    )

    sabre_max_iterations: int = Field(
        default=8,
        description=(
            "The number of forward-backward routing iterations to refine the layout "
            "and reduce routing costs."
        ),
        title="Sabre Max Iteration",
        ge=1,
    )

    sabre_swap_trials: int = Field(
        default=10,
        description=(
            "The number of routing trials for each layout, refining gate placement "
            "for better routing."
        ),
        title="Sabre SWAP Trials",
        ge=1,
    )

    sabre_layout_trials: int = Field(
        default=1_024,
        description="The number of random seed trials to run layout with.",
        title="Sabre Layout Trials",
        ge=1,
    )


class DEParameters(BaseModel):
    """Configuration for differential evoluation."""

    num_walkers: int = Field(
        default=4,
        description=(
            "Number of populations for differential evolution. "
            "Differential-evolution mutation (iterations > 1) needs at least 4 walkers; "
            "a single evaluation pass (iterations = 1, e.g. a recovery-depth test) accepts 1."
        ),
        title="Walkers",
        ge=1,
    )

    iterations: int = Field(
        default=1,
        description="Number of DE optimization iterations.",
        title="Differential Evolution Iterations",
        ge=1,
    )

    randomization_factor: float = Field(
        default=0.2,
        description="Degree of ansatz parameter perturbation from CCSD amplitude.",
        title="Randomization Factor",
        ge=0,
    )

    fxc: float = Field(
        default=0.6,
        description=(
            "Factor to scale the difference between individuals when generating mutants. "
            "Controls step size and search aggressiveness (typically 0.4 to 1.0)."
        ),
        title="Scaling Factor",
        gt=0.0,
        le=1.0,
    )

    cr_prob: float = Field(
        default=0.9,
        description=(
            "Probability of mixing components from the mutant vector into the trial solution. "
            "Controls exploration vs. exploitation (0 to 1)."
        ),
        title="Crossover Rate",
        gt=0.0,
        le=1.0,
    )


class FlowParameters(BaseModel):
    """Workflow Parameters."""

    fcidump: str = Field(
        description="A path to pySCF FCIDump file of the target molecule.",
        title="FCIDump File",
    )

    sqd_dim: int = Field(
        default=100_000_0,
        description="Dimension of subsampled bitstrings for diagonalization.",
        title="SQD Subspace Dimension",
        ge=1,
    )

    n_recovery_steps: int = Field(
        default=1,
        description=(
            "Number of SQD self-consistency configuration-recovery passes per walker. "
            "Each pass re-runs configuration recovery (from the same quantum samples) using "
            "the orbital occupancies from the previous pass's diagonalization, then "
            "re-subsamples and re-diagonalizes. 1 = single pass (no self-consistency); "
            "3-5 is the canonical SQD recovery loop."
        ),
        title="SQD Recovery Steps",
        ge=1,
    )

    n_batches: int = Field(
        default=1,
        description=(
            "Number of independent subsample batches per recovery pass (the K-batch SQD scheme "
            "of arXiv:2405.05068). Each batch draws its own subspace of dimension sqd_dim from the "
            "recovered distribution and is diagonalized separately; the reported energy is the "
            "minimum over batches and the occupancies fed to the next recovery pass are the "
            "average over batches. 1 = single batch (legacy). 5-10 stabilizes the occupancy and "
            "improves the chance of capturing low-excitation configurations. Cost scales linearly "
            "with this value."
        ),
        title="SQD Batches per Recovery",
        ge=1,
    )

    seed_cisd: int = Field(
        default=0,
        description=(
            "Classically seed the SQD subspace with low-order excitations of the Hartree-Fock "
            "reference, so those determinants are ALWAYS present regardless of what the (noisy) "
            "quantum sampler produced. This is the QSCI+SD / SCI-augmentation scheme: sampled "
            "configurations are augmented with classically-generated single/double excitations to "
            "restore the low-excitation determinants that hardware noise drops or underweights. "
            "Levels: 0 = OFF (default, pure quantum-sampled subspace); 1 = SINGLES only; "
            "2 = DOUBLES only; 3 = SINGLES + DOUBLES. The seed determinants are generated per spin "
            "channel by exciting electrons from occupied into virtual orbitals, conserving each "
            "spin's electron count (so the (Na, Nb) sector and hence Sz are preserved exactly), and "
            "are forced into the subspace alongside the Hartree-Fock and carryover determinants "
            "(deduplicated). If the seed set is larger than the sqd_dim budget it is capped "
            "(singles kept first, then as many doubles as fit) so the run always proceeds at the "
            "requested sqd_dim. Purely classical integer bitstring generation -> identical on CPU "
            "and GPU, no extra QPU cost. "
            "References: Enhancing Accuracy of Quantum-Selected Configuration Interaction, "
            "J. Chem. Theory Comput. (PMC12423809); Molecular Quantum Computations on a Protein, "
            "arXiv:2512.17130; Auto-regressive NQS Sampling for Selected CI, arXiv:2603.24728."
        ),
        title="Seed CISD Level",
        ge=0,
        le=3,
    )

    quantum_source: Literal["real-device", "random", "saved"] = Field(
        default="real-device",
        description=(
            "Select the SQD sample source: 'real-device' (IBM Quantum Runtime), "
            "'random' (deterministic random bitstrings), or 'saved' (reload a previously "
            "persisted sample pool and diagonalize offline, no IBM call)."
        ),
        title="Quantum Source",
    )

    random_seed: int = Field(
        default=24,
        description="Base RNG seed used when Quantum Source is 'random'.",
        title="Random Seed",
        ge=0,
    )

    circ_params: CircuitParameters = Field(
        default_factory=CircuitParameters,
        title="Circuit Parameters",
    )

    de_params: DEParameters = Field(
        default_factory=DEParameters,
        title="Differential Evoluation Parameters",
    )

    solver_block_ref: str = Field(
        default="sbd_solver_job/davidson-solver",
        description="Solver block reference in '<block_type_slug>/<block_document_name>' format.",
    )
