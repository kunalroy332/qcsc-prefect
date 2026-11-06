"""Inputs for the GPU solver smoke test (native/README.md, setup how-to Step 3).

Writes fcidump.txt (interleaved-spin UHF format), AlphaDets.bin and BetaDets.bin into OUT_DIR,
with the same writers SBDSolverJob uses (solver_job.py), from the small N2 FCIDUMP in
algorithms/sbd/data/. The determinant lists are the lowest-index strings of each spin, so the
1-GPU and the multi-GPU binary diagonalize exactly the same subspace. No Prefect, no GPU.

    python smoke_inputs.py OUT_DIR [N_DETS_PER_SPIN]   (default 300)
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump

SBD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SBD))
from sbd.solver_job import _write_dets_bin, _write_uhf_fcidump  # noqa: E402


def dets(norb: int, nelec: int, n: int) -> np.ndarray:
    """The first n determinants (occupied orbitals as bits) in combination order, sorted.

    Sorted ascending, as the flow passes them (SQD subsampling returns sorted unique strings).
    PySCF's selected_ci, used for the reference energy, needs sorted strings: unsorted ones give a
    wrong energy without any error (-134.7744 instead of -134.8968 for the 300 x 300 N2 space).
    """
    out = []
    for occ in combinations(range(norb), nelec):
        out.append(sum(1 << i for i in occ))
        if len(out) == n:
            break
    return np.sort(np.array(out, dtype=np.int64))


def main() -> None:
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    out.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(io.StringIO()):
        d = fcidump.read(str(SBD / "data" / "fcidump_N2_MO.txt"))
    norb, nelec, ms2 = d["NORB"], d["NELEC"], d.get("MS2", 0)
    na, nb = (nelec + ms2) // 2, (nelec - ms2) // 2
    h1 = np.asarray(d["H1"])
    h2 = ao2mo.restore(1, d["H2"], norb)
    # restricted MOs in the UHF format: alpha and beta blocks are equal
    _write_uhf_fcidump(
        out / "fcidump.txt",
        h1_a=h1,
        h1_b=h1,
        h2_aa=h2,
        h2_ab=h2,
        h2_bb=h2,
        norb=norb,
        nelec=(na, nb),
        ecore=float(d.get("ECORE", 0.0)),
    )
    _write_dets_bin(out / "AlphaDets.bin", dets(norb, na, n), norb, spin="alpha", num_elec=na)
    _write_dets_bin(out / "BetaDets.bin", dets(norb, nb, n), norb, spin="beta", num_elec=nb)
    print(f"smoke inputs in {out}: norb {norb}, nelec ({na}, {nb}), {n} x {n} determinants")


if __name__ == "__main__":
    main()
