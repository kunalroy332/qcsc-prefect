"""Molecule geometry definition for quantum algorithms."""

import io
from typing import Annotated

import numpy as np
import scipy
from prefect import get_run_logger, task
from pydantic import BaseModel
from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pyscf import ao2mo, cc, gto, scf, tools

# Pydantic Types
NpStrict1DArrayF64 = Annotated[
    np.ndarray[tuple[int,], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=1, strict_data_typing=True),
]

NpStrict2DArrayF64 = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=2, strict_data_typing=True),
]

NpStrict4DArrayF64 = Annotated[
    np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]],
    NpArrayPydanticAnnotation.factory(data_type=np.float64, dimensions=4, strict_data_typing=True),
]


class ElectronicProperties(BaseModel):
    """Intermediate data representing the electronic properties.

    For restricted (RHF) calculations only the spin-summed ``*_tensor`` and ``t2`` fields are
    populated and ``unrestricted`` is ``False``. For unrestricted (UHF) calculations the alpha
    blocks live in the base fields (``one_body_tensor`` = h1 alpha, ``two_body_tensor`` = (aa|aa),
    ``t2`` = t2aa) and the beta / mixed blocks live in the optional ``*_b`` / ``*_ab`` / ``*_bb``
    fields; ``unrestricted`` is ``True``.
    """

    one_body_tensor: NpStrict2DArrayF64
    two_body_tensor: NpStrict4DArrayF64
    t2: NpStrict4DArrayF64
    initial_occupancy: tuple[NpStrict1DArrayF64, NpStrict1DArrayF64]
    nuclear_repulsion_energy: float
    num_orbitals: int
    num_electrons: tuple[int, int]
    open_shell: bool
    spin_sq: float

    # Whether this is an unrestricted (UHF) calculation. When True, the *_b / *_ab / *_bb
    # fields below are populated. RHF results leave them None and behave exactly as before.
    unrestricted: bool = False

    # Beta one-body integrals (h1 beta); alpha h1 stays in one_body_tensor.
    one_body_tensor_b: NpStrict2DArrayF64 | None = None
    # Mixed (aa|bb) and beta-beta (bb|bb) two-body integrals; (aa|aa) stays in two_body_tensor.
    two_body_tensor_ab: NpStrict4DArrayF64 | None = None
    two_body_tensor_bb: NpStrict4DArrayF64 | None = None
    # UCCSD t2 mixed (t2ab) and beta-beta (t2bb) amplitudes; t2aa stays in t2.
    t2_ab: NpStrict4DArrayF64 | None = None
    t2_bb: NpStrict4DArrayF64 | None = None


def _build_property(
    mf: scf.RHF,
    norb: int,
    spin_sq: float,
    buf: io.StringIO,
) -> ElectronicProperties:
    """Helper function to build ElectronicProperties from molecular calculation results.

    Args:
        mf: PySCF RHF object after calculation
        norb: Number of orbitals
        spin_sq: Target value for the total spin squared
        buf: StringIO buffer containing PySCF logs

    Returns:
        ElectronicProperties object
    """
    # Apply unitary transform with obtained MO coefficient.
    # The FCIdump file already gives you these integrals in MO basis,
    # but the HF calculation may give you correction to these integrals.
    # For example, phase may change with this unitary transform.
    # When started with Mole object, AO → MO transform is performed in here.
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(norb, norb, norb, norb)

    nuclear_repulsion_energy = mf.mol.energy_nuc()
    num_elec_a, num_elec_b = mf.mol.nelec

    # Run CCSD
    mycc = cc.CCSD(mf)
    mycc.kernel()
    t2 = mycc.t2

    # Diagonalize RDM to obtain the occupancies
    # Eigenvectors are the natual molecular orbitals
    rdm1_ccsd = mf.make_rdm1()
    occ_ccsd, _ = scipy.linalg.eigh(rdm1_ccsd)
    occ_ccsd /= 2.0

    # Get PySCF logs dumped into in-memory buffer
    get_run_logger().info(buf.getvalue())

    return ElectronicProperties(
        one_body_tensor=h1,
        two_body_tensor=h2,
        t2=t2,
        initial_occupancy=(occ_ccsd[::-1], occ_ccsd[::-1]),
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        num_orbitals=norb,
        num_electrons=(num_elec_a, num_elec_b),
        open_shell=num_elec_a != num_elec_b,
        spin_sq=spin_sq,
    )


def _build_property_uhf(
    mf: scf.UHF,
    norb: int,
    spin_sq: float,
    buf: io.StringIO,
) -> ElectronicProperties:
    """Build ElectronicProperties from an unrestricted (UHF) calculation.

    Mirrors :func:`_build_property` but keeps separate alpha/beta MO coefficients, three
    two-body integral blocks (aa, ab, bb), the UCCSD t2 tuple (t2aa, t2ab, t2bb), and per-spin
    occupancies. The alpha blocks go into the base ElectronicProperties fields; beta / mixed
    blocks go into the optional *_b / *_ab / *_bb fields.

    Args:
        mf: PySCF UHF object after calculation.
        norb: Number of (spatial) orbitals.
        spin_sq: Target value for the total spin squared.
        buf: StringIO buffer containing PySCF logs.

    Returns:
        ElectronicProperties object with ``unrestricted=True``.
    """
    mo_a, mo_b = mf.mo_coeff
    hcore = mf.get_hcore()

    # One-body integrals in each spin's MO basis.
    h1_a = mo_a.T @ hcore @ mo_a
    h1_b = mo_b.T @ hcore @ mo_b

    # Two-body integrals: alpha-alpha, beta-beta, and the mixed alpha-beta block.
    eri = mf._eri
    h2_aa = ao2mo.full(eri, mo_a, compact=False).reshape(norb, norb, norb, norb)
    h2_bb = ao2mo.full(eri, mo_b, compact=False).reshape(norb, norb, norb, norb)
    h2_ab = ao2mo.general(eri, (mo_a, mo_a, mo_b, mo_b), compact=False).reshape(
        norb, norb, norb, norb
    )

    nuclear_repulsion_energy = mf.mol.energy_nuc()
    num_elec_a, num_elec_b = mf.mol.nelec

    # Run UCCSD; t2 is the tuple (t2aa, t2ab, t2bb).
    mycc = cc.UCCSD(mf)
    mycc.kernel()
    t2_aa, t2_ab, t2_bb = mycc.t2

    # Per-spin natural-orbital occupancies from the *correlated* UCCSD 1-RDM (in MO basis).
    # These must be fractional: configuration recovery downstream uses them to bias sampled
    # bitstrings toward the physically important (near-HF) configurations. The UHF *SCF* RDM is
    # (often) idempotent -> integer 0/1 occupancies, which give recovery no signal and collapse
    # the SQD subspace to bare Hartree-Fock. UCCSD make_rdm1() returns (dm_a, dm_b) in MO basis.
    dm_cc_a, dm_cc_b = mycc.make_rdm1()
    occ_a, _ = scipy.linalg.eigh(dm_cc_a)
    occ_b, _ = scipy.linalg.eigh(dm_cc_b)

    # Get PySCF logs dumped into in-memory buffer
    get_run_logger().info(buf.getvalue())

    return ElectronicProperties(
        one_body_tensor=h1_a,
        two_body_tensor=h2_aa,
        t2=t2_aa,
        one_body_tensor_b=h1_b,
        two_body_tensor_ab=h2_ab,
        two_body_tensor_bb=h2_bb,
        t2_ab=t2_ab,
        t2_bb=t2_bb,
        initial_occupancy=(occ_a[::-1], occ_b[::-1]),
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        num_orbitals=norb,
        num_electrons=(num_elec_a, num_elec_b),
        open_shell=num_elec_a != num_elec_b,
        unrestricted=True,
        spin_sq=spin_sq,
    )


@task(
    persist_result=True,
    result_serializer="compressed/json",
    name="compute_molecular_integrals",
)
def compute_molecular_integrals_from_geometry(
    atom: str,
    basis: str = "6-31g",
    symmetry: str | bool = False,
    spin_sq: float = 0.0,
    unrestricted: bool = False,
    spin: int = 0,
) -> ElectronicProperties:
    """Precompute molecular orbital property from geometry with classical methods.

    Args:
        atom: Definition for molecule structure.
        basis: Name of basis set.
        symmetry: Whether to use symmetry, otherwise string of point group name.
        spin_sq: Target value for the total spin squared for the ground state.
        unrestricted: Run an unrestricted (UHF) calculation instead of restricted (RHF).
        spin: Number of unpaired electrons (nelec_alpha - nelec_beta). Only meaningful for
            open-shell systems; passed to PySCF as ``mol.spin``.

    Returns:
        ElectronicProperties object containing molecular integrals and properties.
    """
    # PySCF doesn't use the standard Python logging and Prefect cannot capture it.
    # The logs are directly written in the stdout or in a file.
    # To forward the logs to the Prefect logging sytem,
    # we set an in-memory buffer to the PySCF logging system and read from there.
    buf = io.StringIO()

    mol = gto.Mole()
    mol.build(
        atom=atom,
        basis=basis,
        symmetry=symmetry,
        spin=spin,
    )
    mol.stdout = buf
    mol.verbose = 4

    if unrestricted:
        mf = scf.UHF(mol).run()
        norb = mf.mo_coeff[0].shape[1]
        return _build_property_uhf(mf, norb, spin_sq, buf)

    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]

    return _build_property(mf, norb, spin_sq, buf)


@task(
    persist_result=True,
    result_serializer="compressed/json",
    name="compute_molecular_integrals",
)
def compute_molecular_integrals_from_fcidump(
    fcidump_file: str,
    spin_sq: float = 0.0,
    unrestricted: bool = False,
) -> ElectronicProperties:
    """Precompute molecular orbital property from FCIDump file with classical methods.

    Args:
        fcidump_file: Location of FCIDump file storing 1-electron and 2-electron integrals.
        spin_sq: Target value for the total spin squared for the ground state.
        unrestricted: Run an unrestricted (UHF) calculation instead of restricted (RHF). The
            number of unpaired electrons is taken from the FCIDUMP ``MS2`` header.

    Returns:
        ElectronicProperties object containing molecular integrals and properties.
    """
    # PySCF doesn't use the standard Python logging and Prefect cannot capture it.
    # The logs are directly written in the stdout or in a file.
    # To forward the logs to the Prefect logging sytem,
    # we set an in-memory buffer to the PySCF logging system and read from there.
    buf = io.StringIO()

    data = tools.fcidump.read(fcidump_file)
    norb = data["NORB"]

    mf = tools.fcidump.to_scf(fcidump_file)
    mf.mol.verbose = 4
    mf.mol.stdout = buf

    if unrestricted:
        # pyscf 2.13 has no fcidump.to_uhf_scf; convert the RHF shell (which already carries the
        # FCIDUMP integral overrides and mol.spin = MS2) into a UHF object and run it. UHF init
        # guess builds two spin densities from nelec=(na, nb); no manual dm0 needed.
        nuc = mf.mol.energy_nuc()
        mf = mf.to_uhf()
        mf.mol.energy_nuc = lambda *args: nuc  # preserved across the class conversion
        mf.kernel()
        return _build_property_uhf(mf, norb, spin_sq, buf)

    # Run HF calculation with Newton method.
    # HF convergence is important, as we assume
    # the FCIdump file is created with a converged result.
    mf = scf.newton(mf)
    mf.symmetry = False
    dm0 = np.zeros((norb, norb))
    for i in range(mf.mol.nelectron // 2):
        dm0[i, i] = 2.0
    mf.kernel(dm0=dm0)

    return _build_property(mf, norb, spin_sq, buf)
