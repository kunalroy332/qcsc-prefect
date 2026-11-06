"""SCF engine abstraction (gpu4pyscf on the GPU, threaded PySCF on the CPU) and SCF records.

Every converged SCF is kept as a :class:`Solution` (plain NumPy arrays) and written to a CPU-format
chkfile, so the analysis steps (population, stability, UNO-AVAS, integrals) always run on CPU
objects and a re-run skips converged SCFs.
"""

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
from pyscf import dft, lib, scf
from pyscf.scf import uhf as uhf_mod


@dataclass
class Solution:
    name: str  # "hs_minao", "bs_10", ...
    mo_coeff: np.ndarray  # (2, nao, nmo)
    mo_occ: np.ndarray  # (2, nmo)
    mo_energy: np.ndarray  # (2, nmo)
    e_tot: float
    converged: bool
    spin: int  # mol.spin (n_alpha - n_beta) of this solution
    chkfile: str | None = None
    s2: float | None = None
    atom_spins: dict = field(default_factory=dict)  # "0_Fe" -> alpha - beta (meta-Lowdin)
    tag: str | None = None
    pattern: list | None = None  # requested spin pattern (None for high spin)
    gate: bool | None = None  # quality gate (Fe >= 3.5, S <= 1.2 unpaired electrons)

    def make_rdm1(self):
        return np.array(
            [(c[:, o > 0] * o[o > 0]) @ c[:, o > 0].T for c, o in zip(self.mo_coeff, self.mo_occ)]
        )


def set_threads(threads):
    """``threads`` > 0 sets the PySCF thread count; 0 keeps OMP_NUM_THREADS / all cores."""
    if threads:
        lib.num_threads(int(threads))
    return lib.num_threads()


def gpu_available():
    try:
        import cupy
        import gpu4pyscf  # noqa: F401

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - any import / driver failure means "no GPU"
        return False


def resolve_engine(engine):
    if engine == "auto":
        return "gpu" if gpu_available() else "cpu"
    if engine == "gpu" and not gpu_available():
        raise RuntimeError("engine = 'gpu' but gpu4pyscf / a CUDA device is not available")
    if engine not in ("gpu", "cpu"):
        raise ValueError(f"unknown engine {engine!r} (gpu | cpu | auto)")
    return engine


def make_uhf(
    mol,
    engine="cpu",
    method="uhf",
    xc=None,
    auxbasis="def2-universal-jkfit",
    grid_level=3,
    cpu=False,
):
    """Density-fitted UHF (or UKS) object on the engine; ``cpu=True`` forces a CPU object."""
    use_gpu = engine == "gpu" and not cpu
    if use_gpu:
        from gpu4pyscf import dft as gdft
        from gpu4pyscf import scf as gscf

        mf = gdft.UKS(mol, xc=xc) if method == "uks" else gscf.UHF(mol)
    else:
        mf = dft.UKS(mol, xc=xc) if method == "uks" else scf.UHF(mol)
    if method == "uks":
        if not xc:
            raise ValueError("method = 'uks' needs xc")
        mf.grids.level = grid_level
    elif method != "uhf":
        raise ValueError(f"unknown method {method!r} (uhf | uks)")
    return mf.density_fit(auxbasis=auxbasis)


def _to_numpy(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def run_scf(mf, dm0=None, max_cycle=100, conv_tol=1e-8, newton=True):
    """Second-order SCF (``newton``) with DIIS as fallback if it does not converge."""
    mf.max_cycle = max_cycle
    mf.conv_tol = conv_tol
    runner = mf.newton() if newton else mf
    runner.max_cycle = max_cycle
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Basis may be available")
        runner.kernel(dm0=dm0)
    if newton and not runner.converged:
        mf.max_cycle = max_cycle
        mf.kernel(dm0=runner.make_rdm1())
        runner = mf
    return runner


def solution_from_mf(mf, name, pattern=None):
    return Solution(
        name=name,
        mo_coeff=np.asarray([_to_numpy(c) for c in mf.mo_coeff]),
        mo_occ=np.asarray([_to_numpy(o) for o in mf.mo_occ]),
        mo_energy=np.asarray([_to_numpy(e) for e in mf.mo_energy]),
        e_tot=float(mf.e_tot),
        converged=bool(mf.converged),
        spin=int(mf.mol.spin),
        pattern=pattern,
    )


def cpu_mf(mol, sol, method="uhf", xc=None, auxbasis="def2-universal-jkfit", grid_level=3):
    """CPU UHF/UKS object holding ``sol`` (for stability analysis / population analysis)."""
    mf = make_uhf(mol, "cpu", method, xc, auxbasis, grid_level, cpu=True)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = sol.mo_coeff, sol.mo_occ, sol.mo_energy
    mf.e_tot, mf.converged = sol.e_tot, sol.converged
    return mf


def analyse(mol, sol, n_fe, min_spin=1.0):
    """Fill ``s2``, meta-Löwdin ``atom_spins`` of the cluster atoms and the Fe spin ``tag``."""
    ovlp = mol.intor_symmetric("int1e_ovlp")
    occ = [sol.mo_coeff[s][:, sol.mo_occ[s] > 0] for s in range(2)]
    sol.s2 = float(uhf_mod.spin_square(occ, ovlp)[0])
    _, ms = uhf_mod.mulliken_meta_spin(mol, sol.make_rdm1(), verbose=0, s=ovlp)
    sol.atom_spins = {f"{i}_{mol.atom_pure_symbol(i)}": float(ms[i]) for i in range(mol.natm)}
    fe = [ms[i] for i in range(n_fe)]
    sol.tag = "".join("x" if abs(s) < min_spin else ("0" if s > 0 else "1") for s in fe)
    return sol


def cluster_spins(sol, n_cluster):
    return dict(list(sol.atom_spins.items())[:n_cluster])


def save_solution(sol, chkfile, mol):
    """CPU-format chkfile (``scf.chkfile.dump_scf``) + the record fields under ``bs_uno_avas``."""
    os.makedirs(os.path.dirname(os.path.abspath(chkfile)), exist_ok=True)
    scf.chkfile.dump_scf(mol, chkfile, sol.e_tot, sol.mo_energy, sol.mo_coeff, sol.mo_occ)
    lib.chkfile.save(
        chkfile,
        "bs_uno_avas",
        {
            "name": sol.name,
            "converged": int(sol.converged),
            "spin": sol.spin,
            "pattern": np.asarray(sol.pattern if sol.pattern is not None else [], dtype=int),
        },
    )
    sol.chkfile = chkfile


def load_solution(chkfile):
    """Solution from a chkfile written by :func:`save_solution`, or None if absent/incomplete."""
    if not os.path.exists(chkfile):
        return None
    try:
        data = lib.chkfile.load(chkfile, "scf")
        meta = lib.chkfile.load(chkfile, "bs_uno_avas")
    except (KeyError, OSError):
        return None
    if data is None or meta is None:
        return None
    name = meta["name"]
    pattern = [int(x) for x in meta["pattern"]] or None
    return Solution(
        name=name.decode() if isinstance(name, bytes) else str(name),
        mo_coeff=np.asarray(data["mo_coeff"]),
        mo_occ=np.asarray(data["mo_occ"]),
        mo_energy=np.asarray(data["mo_energy"]),
        e_tot=float(data["e_tot"]),
        converged=bool(meta["converged"]),
        spin=int(meta["spin"]),
        chkfile=chkfile,
        pattern=pattern,
    )
