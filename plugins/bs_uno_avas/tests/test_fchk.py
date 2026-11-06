"""fchk route: Gaussian-convention fchk files written from PySCF solutions, read back and run.

The writer here follows the Gaussian conventions independently of the reader where it can (the
in-shell component order comes from ``pyscf.tools.molden.order_ao_index``); a real Gaussian
fchk is the final check (``cli check-fchk`` on ROQUO).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from bs_uno_avas import (
    bs_guess,
    cli,
    fchk,
    fchk_route,
    manifest,
    scf_engine,
    structure,
)
from pyscf import gto
from pyscf.tools import molden

SCH3 = Path(__file__).resolve().parents[1] / "examples" / "5XTD_FES_O301_SCH3.pdb"
SHUFFLE = [2, 0, 3, 1]  # fchk atom k = model atom SHUFFLE[k]: cluster not first in the fchk


# ---------- test-only fchk writer ----------


def _header(name, typ, value=None, n=None):
    if n is None:
        v = f"{value:12d}" if typ == "I" else f"{value:22.15E}"
        return f"{name:<43s}{typ}{v:>{len(v) + 5}s}"
    return f"{name:<43s}{typ}   N={n:12d}"


def _array(name, typ, vals):
    vals = list(vals)
    lines = [_header(name, typ, n=len(vals))]
    per, fmt = (6, "{:12d}") if typ == "I" else (5, "{:16.8E}")
    for i in range(0, len(vals), per):
        lines.append(
            "".join(fmt.format(int(v) if typ == "I" else float(v)) for v in vals[i : i + per])
        )
    return lines


def write_fchk(path, mol, mo_coeff, nalpha, nbeta, e_tot, s2, atom_order, method="UB3PW91"):
    """fchk of ``mol`` (PySCF order) with the atoms in ``atom_order`` (fchk atom k = mol atom
    ``atom_order[k]``), s + p shells with equal exponents merged into SP shells, Gaussian AO order
    and normalisation."""
    idx = molden.order_ao_index(mol)  # molden (= Gaussian) position -> PySCF AO, per shell
    ao_loc = mol.ao_loc
    norm = np.sqrt(mol.intor_symmetric("int1e_ovlp").diagonal())
    types, nprim, shell_atom, exps, cc, cp = [], [], [], [], [], []
    rows = []  # PySCF AO of each fchk AO
    for k, a in enumerate(atom_order):
        shells = [ib for ib in range(mol.nbas) if mol.bas_atom(ib) == a]
        raw = mol._basis[mol.atom_symbol(a)]  # one entry per shell, same order (sorted by l)
        coef = {ib: np.array([p[1] for p in raw[j][1:]]) for j, ib in enumerate(shells)}
        p_shells = [ib for ib in shells if mol.bas_angular(ib) == 1]
        merged = set()
        for ib in shells:
            if ib in merged:
                continue
            ang, e = mol.bas_angular(ib), mol.bas_exp(ib)
            partner = None
            if ang == 0:
                partner = next(
                    (
                        p
                        for p in p_shells
                        if p not in merged
                        and len(mol.bas_exp(p)) == len(e)
                        and np.allclose(mol.bas_exp(p), e)
                    ),
                    None,
                )
            if partner is not None:  # SP shell: s, then p_x p_y p_z
                merged.add(partner)
                types.append(-1)
                cc += list(coef[ib])
                cp += list(coef[partner])
                rows += [ao_loc[ib]] + list(range(ao_loc[partner], ao_loc[partner] + 3))
            else:
                types.append(ang if (mol.cart or ang < 2) else -ang)
                cc += list(coef[ib])
                cp += [0.0] * len(e)
                rows += [i for i in idx if ao_loc[ib] <= i < ao_loc[ib + 1]]
            nprim.append(len(e))
            shell_atom.append(k + 1)
            exps += list(e)
    rows = np.array(rows)
    nmo = mo_coeff.shape[2]
    lines = ["synthetic fchk (test)", f"SP        {method}                                 X"]
    lines.append(_header("Number of atoms", "I", mol.natm))
    lines.append(_header("Charge", "I", mol.charge))
    lines.append(_header("Multiplicity", "I", mol.spin + 1))
    lines.append(_header("Number of electrons", "I", nalpha + nbeta))
    lines.append(_header("Number of alpha electrons", "I", nalpha))
    lines.append(_header("Number of beta electrons", "I", nbeta))
    lines.append(_header("Number of basis functions", "I", mol.nao))
    lines.append(_header("Number of independent functions", "I", nmo))
    lines += _array("Atomic numbers", "I", [mol.atom_charge(a) for a in atom_order])
    lines += _array("Current cartesian coordinates", "R", mol.atom_coords()[atom_order].ravel())
    lines += _array("Shell types", "I", types)
    lines += _array("Number of primitives per shell", "I", nprim)
    lines += _array("Shell to atom map", "I", shell_atom)
    lines += _array("Primitive exponents", "R", exps)
    lines += _array("Contraction coefficients", "R", cc)
    if -1 in types:
        lines += _array("P(S=P) Contraction coefficients", "R", cp)
    lines.append(_header("Total Energy", "R", e_tot))
    for s, name in enumerate(("Alpha", "Beta")):
        lines += _array(f"{name} Orbital Energies", "R", np.zeros(nmo))
        c = mo_coeff[s][rows] * norm[rows][:, None]  # (fchk AO, MO)
        lines += _array(f"{name} MO coefficients", "R", c.T.ravel())
    lines.append(_header("S**2", "R", s2))
    Path(path).write_text("\n".join(lines) + "\n")
    return path


# ---------- fixture: BS-UKS of the bare Fe2S2 core ----------


def _bare_mol(state, basis, cart=False):
    model = structure.prepare_model(SCH3, cluster_only=True)
    cs = structure.charge_and_spin(model, state=state)
    ms2 = cs["nelec"] % 2
    mol = gto.M(
        atom=model.pyscf_atoms(), basis=basis, charge=cs["charge"], spin=ms2, cart=cart, verbose=0
    )
    return mol


def _uks(mol, max_cycle=60):
    mf = scf_engine.make_uhf(mol, "cpu", "uks", "b3pw91")
    if max_cycle < 10:  # a few DIIS steps from minao: any MOs do for the mapping tests
        mf.max_cycle = max_cycle
        mf.kernel()
        return mf
    dm0 = bs_guess.atomic_bs_density(mol, [1, -1], *mol.nelec)
    return scf_engine.run_scf(mf, dm0=dm0, max_cycle=max_cycle)


@pytest.fixture(scope="module")
def ox_fchk(tmp_path_factory):
    """Converged BS-UKS (3-21G: SP shells) of [Fe2S2]2+ written as a shuffled fchk."""
    mol = _bare_mol("ox", "3-21g")
    mf = _uks(mol)
    assert mf.converged
    path = tmp_path_factory.mktemp("fchk") / "bare_BS_OPT.fchk"
    s2 = mf.spin_square()[0]
    write_fchk(path, mol, np.asarray(mf.mo_coeff), *mol.nelec, mf.e_tot, s2, SHUFFLE)
    return path, mol, mf


def _cfg(tmp_path, *sets):
    return fchk_route.load_config(
        None, [f"output.dir={tmp_path / 'out'}", "avas.target_aos='free-atom'", *sets]
    )


# ---------- conventions ----------


def test_shell_maps_match_pyscf_labels():
    """Gaussian component k -> the PySCF AO with the same label (iodata fchk conventions)."""
    gauss = {
        2: ["xx", "yy", "zz", "xy", "xz", "yz"],
        3: ["xxx", "yyy", "zzz", "xyy", "xxy", "xxz", "xzz", "yzz", "yyz", "xyz"],
    }
    for ang, names in gauss.items():
        mol = gto.M(atom="Ne 0 0 0", basis={"Ne": [[ang, [1.0, 1.0]]]}, cart=True, verbose=0)
        pys = [lab.split()[-1][2:] for lab in mol.ao_labels()]
        assert [pys[p] for p in fchk.shell_map(ang, True)] == names
    # pure: Gaussian m = 0, +1, -1, +2, -2, ...; PySCF m = -l ... +l
    for ang in (2, 3, 4):
        gm = [0] + [s * m for m in range(1, ang + 1) for s in (1, -1)]
        assert fchk.shell_map(ang, False) == [m + ang for m in gm]


# ---------- reader / conversion ----------


def test_read_back_reproduces_solution(ox_fchk):
    path, mol, mf = ox_fchk
    fc = fchk.read_fchk(path)
    assert fc["method"] == "UB3PW91" and fc["Multiplicity"] == 1
    order, n_fe, n_s, _ = fchk_route.find_cluster(
        fc["Atomic numbers"], fc["Current cartesian coordinates"]
    )
    assert (n_fe, n_s) == (2, 2)
    assert [SHUFFLE[i] for i in order] == [0, 1, 2, 3]  # back to Fe, Fe, S, S
    mol2, sol = fchk.to_pyscf(fc, order)
    assert np.allclose(mol2.atom_coords(), mol.atom_coords())
    dm = sol.make_rdm1()
    assert np.allclose(dm, mf.make_rdm1(), atol=1e-6)
    ck = fchk.check_conversion(mol2, sol, fc, energy=True)
    assert ck["orthonormality_error"] < 1e-6
    assert abs(ck["s2"] - ck["s2_fchk"]) < 1e-5
    assert abs(ck["e_pyscf"] - mf.e_tot) < 1e-6 and ck["xc"] == "b3pw91"


def test_cartesian_6d_and_unconverged_mos(tmp_path):
    """6D/10F + SP (6-31G*): mapping and normalisation; the MOs need not be converged."""
    mol = _bare_mol("ox", "6-31g*", cart=True)
    mf = _uks(mol, max_cycle=2)
    path = write_fchk(
        tmp_path / "c_BS_OPT.fchk",
        mol,
        np.asarray(mf.mo_coeff),
        *mol.nelec,
        mf.energy_tot(),
        mf.spin_square()[0],
        SHUFFLE,
    )
    cfg = _cfg(tmp_path)
    d = fchk_route.load(path, cfg)
    assert d["mol"].cart
    assert np.allclose(d["sol"].make_rdm1(), mf.make_rdm1(), atol=1e-6)
    # fchk text keeps 9 digits per coefficient: unconverged MOs give a first-order energy error
    assert abs(d["checks"]["e_pyscf"] - mf.energy_tot()) < 1e-5
    # file:AOs targets (dzvp) projected onto the Cartesian 6-31G* basis
    t, lab, res = fchk_route._targets(fchk_route.load_config(), d["mol"], 2, 2)
    assert len(lab) == 16 and 0 <= res < 0.5


@pytest.mark.parametrize(
    "edit, match",
    [
        (
            lambda t: t + "ECP-RNFroz                                 R   N=           1\n 0.0\n",
            "ECP",
        ),
        (lambda t: t.split("Beta Orbital Energies")[0], "Beta MOs"),
        (lambda t: _scalar(t, "Multiplicity", 11), "broken-symmetry M_S"),
    ],
)
def test_rejected_fchk(ox_fchk, tmp_path, edit, match):
    bad = tmp_path / "bad.fchk"
    bad.write_text(edit(ox_fchk[0].read_text()))
    with pytest.raises((ValueError, RuntimeError), match=match):
        fchk_route.load(bad, _cfg(tmp_path, "check.energy=false"))


def _scalar(text, name, value):
    out = []
    for ln in text.splitlines():
        if ln.startswith(name + " "):
            ln = _header(name, "I", value)
        out.append(ln)
    return "\n".join(out) + "\n"


def test_mixed_pure_cartesian_rejected():
    fc = {
        "Shell types": np.array([0, -2, 3]),
        "Number of primitives per shell": np.array([1, 1, 1]),
        "Shell to atom map": np.array([1, 1, 1]),
        "Primitive exponents": np.ones(3),
        "Contraction coefficients": np.ones(3),
        "Number of basis functions": 16,
    }
    with pytest.raises(ValueError, match="mixes pure and Cartesian"):
        fchk.fchk_shells(fc)


def test_corrupted_mos_fail_the_check(ox_fchk, tmp_path):
    """A wrong component order (here: d shells left in Gaussian order) breaks orthonormality."""
    fc = fchk.read_fchk(ox_fchk[0])
    order = fchk_route.find_cluster(fc["Atomic numbers"], fc["Current cartesian coordinates"])[0]
    real = fchk.shell_map
    fchk.shell_map = lambda ang, cart: list(range(2 * ang + 1)) if ang == 2 else real(ang, cart)
    try:
        mol, sol = fchk.to_pyscf(fc, order)
    finally:
        fchk.shell_map = real
    assert fchk.check_conversion(mol, sol, fc)["orthonormality_error"] > 1e-3


# ---------- end to end ----------


def test_build_fchk_end_to_end(ox_fchk, tmp_path):
    path = fchk_route.run_fchk(ox_fchk[0], "ox", cfg=_cfg(tmp_path))
    assert path.parent.name.startswith("bare_Eox_Rox_S") and path.parent.name.endswith("_BS_OPT")
    m, files, labels, rdm = manifest.read_manifest(path)
    a, ref = m["active_space"], m["bs_reference"]
    assert (a["norb"], a["nelec"], a["ms2"]) == (16, 22, 0)
    assert ref["spin_tag"] == ref["full_spin_tag"] in ("01", "10")
    assert ref["e_full_uhf"] is None and abs(ref["e_full_dft"] - ox_fchk[2].e_tot) < 1e-8
    assert m["structure"]["source"] == "gaussian-fchk"
    assert m["electronic"]["method"] == "UB3PW91"
    assert np.allclose(ref["rdm_trace"], [11, 11], atol=1e-6)
    assert not list(path.parent.parent.glob(".*"))  # no partial / stage directories left
    assert cli.main(["verify", "--manifest", str(path)]) == 0


def test_build_fchk_manifest_loads_in_sbd(ox_fchk, tmp_path, monkeypatch):
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "pfh"))
    active_space = pytest.importorskip("qcsc_workflow_utility.active_space")
    path = fchk_route.run_fchk(ox_fchk[0], "ox", cfg=_cfg(tmp_path))
    spec = active_space.load_active_space_manifest(path)
    assert spec.norb == 16 and spec.cluster_type == "Fe2S2"


def test_wrong_state_and_gate(ox_fchk, tmp_path):
    with pytest.raises(RuntimeError, match="active electrons"):
        fchk_route.run_fchk(ox_fchk[0], "red", cfg=_cfg(tmp_path))
    with pytest.raises(RuntimeError, match="fails the gate"):
        fchk_route.run_fchk(ox_fchk[0], "ox", cfg=_cfg(tmp_path, "gate.min_fe=4.9"))
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_failed_verify_leaves_no_stage(ox_fchk, tmp_path, monkeypatch):
    """A check that fails after the FCIDUMP was staged leaves nothing in output/."""
    real = fchk_route.pipeline.verify_mod.verify

    def flipped(*a, **k):
        out = real(*a, **k)
        out["spin_tag"] = out["spin_tag"][::-1]
        return out

    monkeypatch.setattr(fchk_route.pipeline.verify_mod, "verify", flipped)
    with pytest.raises(RuntimeError, match="no artifacts written"):
        fchk_route.run_fchk(ox_fchk[0], "ox", cfg=_cfg(tmp_path))
    assert not any((tmp_path / "out").iterdir())


def test_cli_check_fchk(ox_fchk, capsys):
    assert cli.main(["check-fchk", "--fchk", str(ox_fchk[0]), "--no-energy"]) == 0
    out = capsys.readouterr().out
    assert "check-fchk: PASS" in out and "energy check off" in out


@pytest.fixture(autouse=True)
def _pyscf_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSCF_TMPDIR", str(tmp_path))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
