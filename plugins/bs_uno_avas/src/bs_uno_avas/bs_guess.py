"""High-spin UHF, broken-symmetry spin-flip guesses (Noodleman) and the full-system BS-UHF runs."""

import itertools
import os

import numpy as np
from pyscf import gto, lib, scf
from scipy import linalg

from . import scf_engine as se

log = lib.logger


def enumerate_patterns(n_fe, mirrors=True):
    """Spin patterns with zero net Fe spin, first Fe up unless ``mirrors``.

    Fe2S2: ``[1,-1]`` (+ ``[-1,1]``). Fe4S4: the 3 pairings ``[1,1,-1,-1]``, ``[1,-1,1,-1]``,
    ``[1,-1,-1,1]`` (+ their 3 mirrors).
    """
    if n_fe % 2:
        raise ValueError("n_fe must be even")
    out = []
    for ups in itertools.combinations(range(n_fe), n_fe // 2):
        p = [1 if i in ups else -1 for i in range(n_fe)]
        if mirrors or p[0] == 1:
            out.append(p)
    return sorted(out, key=lambda p: [-x for x in p])


def tag_of(pattern):
    return "".join("0" if s > 0 else "1" for s in pattern)


def quality_gate(sol, n_fe, n_s, min_fe=3.5, max_s=1.2):
    """True if every Fe has >= min_fe and every bridging S <= max_s unpaired electrons."""
    spins = list(sol.atom_spins.values())
    return all(abs(s) >= min_fe for s in spins[:n_fe]) and all(
        abs(s) <= max_s for s in spins[n_fe : n_fe + n_s]
    )


def spin_flip_density(mol, dm_hs, pattern, nalpha, nbeta):
    """Idempotent (Da, Db) guess with the Fe of ``pattern == -1`` spin-flipped.

    The α and β blocks of the high-spin density are swapped on the AOs of flipped Fe atoms
    (and between two flipped atoms); blocks between a flipped and an unflipped atom are
    averaged. The result is purified per spin: the ``n_s`` natural orbitals with the largest
    occupation get occupation 1.
    """
    flip = np.zeros(mol.nao, dtype=bool)
    aoslice = mol.aoslice_by_atom()
    for i, s in enumerate(pattern):
        if mol.atom_pure_symbol(i) != "Fe":
            raise ValueError(f"atom {i} is not Fe (cluster atoms must come first)")
        if s < 0:
            flip[aoslice[i, 2] : aoslice[i, 3]] = True
    da, db = dm_hs
    both = np.logical_and.outer(flip, flip)
    mixed = np.logical_xor.outer(flip, flip)
    avg = (da + db) / 2
    ga = np.where(both, db, np.where(mixed, avg, da))
    gb = np.where(both, da, np.where(mixed, avg, db))
    ovlp = mol.intor_symmetric("int1e_ovlp")
    return np.array([_purify(g, ovlp, n) for g, n in ((ga, nalpha), (gb, nbeta))])


def _fe_atom_density(mol, i):
    """(Dα, Dβ) of a free high-spin Fe atom (2S = 4) in the basis of atom ``i`` of ``mol``."""
    atom = gto.M(
        atom=[["Fe", (0.0, 0.0, 0.0)]],
        basis={"Fe": mol._basis[mol.atom_symbol(i)]},
        spin=4,
        verbose=0,
    )
    mf = scf.UHF(atom)
    mf.max_cycle = 200
    mf.kernel()
    return mf.make_rdm1()


def atomic_bs_density(mol, pattern, nalpha, nbeta):
    """Broken-symmetry guess without a high-spin SCF (fragment guess; ``pattern``: +1 up, -1 down).

    Superposition of atomic densities (``init_guess_by_atom``, α = β = D/2) with every Fe block
    replaced by the (Dα, Dβ) of a free high-spin Fe atom, swapped for the down Fe; each spin is
    then purified to (nalpha, nbeta) electrons.
    """
    dm = scf.hf.init_guess_by_atom(mol)
    ga, gb = dm / 2, dm / 2
    aoslice = mol.aoslice_by_atom()
    fe_dm = None
    for i, s in enumerate(pattern):
        if mol.atom_pure_symbol(i) != "Fe":
            raise ValueError(f"atom {i} is not Fe (cluster atoms must come first)")
        if fe_dm is None:
            fe_dm = _fe_atom_density(mol, i)
        blk = slice(aoslice[i, 2], aoslice[i, 3])
        up, down = (ga, gb) if s > 0 else (gb, ga)
        up[blk, blk], down[blk, blk] = fe_dm[0], fe_dm[1]
    ovlp = mol.intor_symmetric("int1e_ovlp")
    return np.array([_purify(g, ovlp, n) for g, n in ((ga, nalpha), (gb, nbeta))])


def _purify(dm, ovlp, nocc):
    """Density of the ``nocc`` most occupied natural orbitals of ``dm`` (generalized eig)."""
    sdm = ovlp @ dm @ ovlp
    w, c = linalg.eigh(sdm, ovlp)
    c_occ = c[:, np.argsort(w)[::-1][:nocc]]
    return c_occ @ c_occ.T


def bs_mol(mol, ms2):
    """Copy of ``mol`` with ``spin = ms2`` (the BS state); the high-spin mol is unchanged."""
    m = mol.copy()
    m.spin = ms2
    m.build(False, False)
    return m


def follow_stability(mol, sol, cfg, mf_engine=None, verbose=0):
    """Internal UHF→UHF stability on a CPU copy; re-converge (on the engine) while unstable."""
    for _ in range(cfg.get("n_stability", 3)):
        mf_cpu = se.cpu_mf(
            mol, sol, cfg["method"], cfg.get("xc"), cfg["auxbasis"], cfg.get("grid_level", 3)
        )
        mf_cpu.verbose = verbose
        mo, _, stable, _ = mf_cpu.stability(internal=True, external=False, return_status=True)
        if stable:
            break
        dm = mf_cpu.make_rdm1(mo, mf_cpu.mo_occ)
        mf = (
            mf_engine
            if mf_engine is not None
            else se.make_uhf(
                mol,
                cfg["engine"],
                cfg["method"],
                cfg.get("xc"),
                cfg["auxbasis"],
                cfg.get("grid_level", 3),
            )
        )
        mf.verbose = verbose
        new = se.solution_from_mf(
            se.run_scf(mf, dm0=dm, max_cycle=cfg.get("max_cycle", 100)), sol.name, sol.pattern
        )
        if new.e_tot > sol.e_tot + 1e-8:
            break  # the follow step went uphill: keep the stable-enough lower solution
        sol = new
    return sol


def high_spin(mol, cfg, n_fe, n_s, work_dir, tag, verbose=0):
    """High-spin UHF for every guess in ``cfg['guesses']`` until one passes the quality gate.

    Returns (best solution, list of all solutions). Chkfiles ``hs_<tag>_<guess>.chk``.
    """
    sols = []
    for guess in cfg.get("guesses", ["minao"]):
        chk = os.path.join(work_dir, f"hs_{tag}_{guess}.chk")
        sol = se.load_solution(chk)
        if sol is None or not sol.converged:
            mf = se.make_uhf(
                mol,
                cfg["engine"],
                cfg["method"],
                cfg.get("xc"),
                cfg["auxbasis"],
                cfg.get("grid_level", 3),
            )
            mf.verbose = verbose
            mf.init_guess = guess
            sol = se.solution_from_mf(
                se.run_scf(mf, max_cycle=cfg.get("max_cycle", 100)), f"hs_{guess}"
            )
            se.save_solution(sol, chk, mol)
        se.analyse(mol, sol, n_fe)
        sol.gate = quality_gate(sol, n_fe, n_s)
        sols.append(sol)
        print(
            f"  HS UHF guess {guess}: E = {sol.e_tot:.8f} converged {sol.converged} "
            f"<S^2> {sol.s2:.3f} gate {'pass' if sol.gate else 'FAIL'}  "
            + "  ".join(f"{k} {v:+.2f}" for k, v in se.cluster_spins(sol, n_fe + n_s).items())
        )
        if sol.converged and sol.gate:
            break
    good = [s for s in sols if s.converged and s.gate]
    if not good:
        print("  WARNING: no high-spin UHF passed the quality gate; using the lowest converged")
        good = [s for s in sols if s.converged] or sols
    return min(good, key=lambda s: s.e_tot), sols


def banked_high_spin(mol, cfg, n_fe, n_s, work_dir, tag):
    """Lowest converged high-spin solution from existing chkfiles that passes the gate, or None
    (no SCF is run)."""
    good = []
    for guess in cfg.get("guesses", ["minao"]):
        sol = se.load_solution(os.path.join(work_dir, f"hs_{tag}_{guess}.chk"))
        if sol is not None and sol.converged:
            se.analyse(mol, sol, n_fe)
            if quality_gate(sol, n_fe, n_s):
                good.append(sol)
    return min(good, key=lambda s: s.e_tot) if good else None


def broken_symmetry(mol_hs, hs, patterns, ms2, cfg, n_fe, n_s, work_dir, tag, verbose=0):
    """BS-UHF at ``spin = ms2`` per pattern; returns (mol, solutions).

    Guess: spin flip of the high-spin solution ``hs``, or ``hs = None``: :func:`atomic_bs_density`
    (no high-spin SCF; chkfile ``bs_<tag>_<pattern>_atomic.chk``). A chkfile from the spin-flip
    guess (``bs_<tag>_<pattern>.chk``) is used first in both cases.
    """
    mol = bs_mol(mol_hs, ms2)
    nalpha, nbeta = mol.nelec
    out = []
    for pattern in patterns:
        ptag = tag_of(pattern)
        chk = os.path.join(work_dir, f"bs_{tag}_{ptag}.chk")
        sol = se.load_solution(chk)
        if hs is None and (sol is None or not sol.converged):
            chk = os.path.join(work_dir, f"bs_{tag}_{ptag}_atomic.chk")
            sol = se.load_solution(chk)
        if sol is None or not sol.converged:
            if hs is None:
                dm0 = atomic_bs_density(mol, pattern, nalpha, nbeta)
            else:
                dm0 = spin_flip_density(mol, hs.make_rdm1(), pattern, nalpha, nbeta)
            mf = se.make_uhf(
                mol,
                cfg["engine"],
                cfg["method"],
                cfg.get("xc"),
                cfg["auxbasis"],
                cfg.get("grid_level", 3),
            )
            mf.verbose = verbose
            sol = se.solution_from_mf(
                se.run_scf(mf, dm0=dm0, max_cycle=cfg.get("max_cycle", 100)), f"bs_{ptag}", pattern
            )
            if cfg.get("stability", "internal") == "internal" and sol.converged:
                sol = follow_stability(mol, sol, cfg, mf_engine=mf, verbose=verbose)
            se.save_solution(sol, chk, mol)
        sol.pattern = pattern
        se.analyse(mol, sol, n_fe)
        sol.gate = quality_gate(sol, n_fe, n_s)
        out.append(sol)
        print(
            f"  BS UHF {ptag} -> {sol.tag}: E = {sol.e_tot:.8f} converged {sol.converged} "
            f"<S^2> {sol.s2:.3f} gate {'pass' if sol.gate else 'FAIL'}  "
            + "  ".join(f"{k} {v:+.2f}" for k, v in se.cluster_spins(sol, n_fe + n_s).items())
        )
    return mol, out


def select(solutions, selection="lowest"):
    """Lowest-E converged solution whose found tag equals its requested pattern and passes the
    gate. ``selection`` may be a tag (e.g. ``"10"``) to force a pattern."""
    ok = [s for s in solutions if s.converged and s.tag == tag_of(s.pattern) and s.gate]
    if selection != "lowest":
        ok = [s for s in ok if s.tag == selection]
        if not ok:
            raise RuntimeError(f"no converged solution with tag {selection!r} passing the gate")
    if not ok:
        found = [(tag_of(s.pattern), s.tag, s.converged, s.gate) for s in solutions]
        raise RuntimeError(
            f"no BS-UHF solution reached its pattern and passed the gate: "
            f"(requested, found, converged, gate) = {found}"
        )
    return min(ok, key=lambda s: s.e_tot)


def build_mol(model, charge, spin, basis, max_memory_mb=4000, verbose=0):
    return gto.M(
        atom=model.pyscf_atoms(),
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=False,
        max_memory=max_memory_mb,
        verbose=verbose,
    )
