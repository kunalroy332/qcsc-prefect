"""Molecule geometry definition for quantum algorithms."""

import io
import os
import warnings
from typing import Annotated

import numpy as np

# Workaround: PySCF DIIS references numpy.linalg.linalg.LinAlgError which was removed
# in numpy 2.x (now numpy.linalg._linalg). Patch it so DIIS exception handling works.
if not hasattr(np.linalg, "linalg"):
    import types as _types
    np.linalg.linalg = _types.SimpleNamespace(LinAlgError=np.linalg.LinAlgError)
from prefect import get_run_logger, task
from pydantic import BaseModel
from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from pyscf import ao2mo, cc, gto, scf, tools

from .active_space import (
    ActiveSpaceSpec,
    atom_spins,
    bs_density_from_labels,
    density_from_af_groups,
    file_sha256,
    spin_tag_from_spins,
    validate_orbital_indices,
)

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

    # Provenance of a broken-symmetry UHF reference built from an active-space manifest (see
    # _converge_reference_uhf): converged spin tag ("10", ...), which start won
    # ("manifest-rdm" / "manifest-labels"), and the FCIDUMP sha256. None for RHF and for UHF
    # references not built from a manifest.
    reference_tag: str | None = None
    reference_source: str | None = None
    fcidump_sha256: str | None = None


# ElectronicProperties fields that describe the reference rather than the integrals; every
# rebuild of the model (CC refresh, orbital rotation) must carry them over unchanged.
REFERENCE_FIELDS = ("reference_tag", "reference_source", "fcidump_sha256")


def reference_fields(props: ElectronicProperties) -> dict:
    """The REFERENCE_FIELDS of ``props`` as constructor kwargs."""
    return {name: getattr(props, name) for name in REFERENCE_FIELDS}


def _get_logger():
    """Prefect run logger inside a flow/task, stdlib logger otherwise (direct calls, tests)."""
    try:
        return get_run_logger()
    except Exception:
        import logging

        return logging.getLogger(__name__)


def _parse_af_groups() -> dict | None:
    """Parse the FE4S4_AF_GROUPS env var into an atom-localized AF-guess spec, or None.

    Format (JSON): {"fe1":[2,3,4,5,6], "fe2":[7,8,9,10,11], ..., "up":["fe1","fe3"],
    "down":["fe2","fe4"]}. Fragments listed in "up"/"down" are alpha/beta spin-polarized; all other
    fragments are treated as closed (doubly occupied). Enables the Noodleman broken-symmetry guess in
    _converge_broken_symmetry_uhf for Fe-S cubanes (~33 mHa lower reference than energy-ordered).

    Convenience: FE4S4_AF_GROUPS="fe4s4" expands to the standard [Fe4S4(SCH3)4]2- MO grouping
    (l1=0-1, fe1=2-6, fe2=7-11, s=12-23, fe3=24-28, fe4=29-33, l2=34-35) with the Singlet-I pairing
    (Fe1,Fe3 up / Fe2,Fe4 down), matching the reference that reaches -326.801 (<S^2>~7.6).

    Two extra env knobs tune the guess toward the physical super-exchange regime (local sweep found
    p=0.6 + free bridging S gives the deepest BS-UCCSD -327.143 and a smaller <S^2>, vs -327.102 at
    the default p=1.0):
      FE4S4_AF_POL   (float, default 1.0): Fe spin-polarization amplitude. p=1 -> full 1.0/0.0 split
                     (maximally localized, large <S^2>); p<1 -> 0.5(1+/-p) split so the sublattices
                     couple more (smaller <S^2>).
      FE4S4_AF_FREE_S (0/1, default 0): seed the bridging S 3p fragment half-filled/unpolarized so it
                     MEDIATES the Fe-Fe exchange (the paper's principal singlet-triplet pathway),
                     instead of forcing it doubly-occupied.
    These attach as extra keys ("pol", "free") on the returned dict, consumed by _af_guess_uhf.
    """
    import json

    raw = os.environ.get("FE4S4_AF_GROUPS", "").strip()
    if not raw:
        return None
    if raw.lower() == "fe4s4":
        spec = {
            "l1": list(range(0, 2)),
            "fe1": list(range(2, 7)),
            "fe2": list(range(7, 12)),
            "s": list(range(12, 24)),
            "fe3": list(range(24, 29)),
            "fe4": list(range(29, 34)),
            "l2": list(range(34, 36)),
            # Pairing from the collaborator who built the FCIDUMP (verified 2026-07-13): up={fe1,fe2}
            # down={fe3,fe4} reaches the deepest BS basin (-326.801) at full polarization directly,
            # more robustly than the earlier {fe1,fe3}/{fe2,fe4} guess (which needed pol=0.6+free-S).
            "up": ["fe1", "fe2"],
            "down": ["fe3", "fe4"],
        }
    else:
        try:
            spec = json.loads(raw)
        except Exception:
            warnings.warn(f"FE4S4_AF_GROUPS could not be parsed as JSON: {raw!r}; ignoring.")
            return None
        if not isinstance(spec, dict):
            raise ValueError(f"FE4S4_AF_GROUPS must be a JSON object or 'fe4s4', got {raw!r}")
    # Optional fractional-polarization + free-bridging-S knobs (apply to any spec).
    try:
        spec["pol"] = float(os.environ.get("FE4S4_AF_POL", "1.0"))
    except Exception:
        spec["pol"] = 1.0
    if os.environ.get("FE4S4_AF_FREE_S", "0") == "1" and "free" not in spec:
        # default the bridging-S fragment name to "s" (present in the fe4s4 convenience spec)
        if "s" in spec:
            spec["free"] = ["s"]
    return spec


def _check_af_groups(af_groups: dict | None, norb: int) -> None:
    """Bounds-check an FE4S4_AF_GROUPS spec against the FCIDUMP's NORB before any SCF runs.

    The 'fe4s4' preset hard-codes the 36-orbital [Fe4S4(SCH3)4] MO grouping; on any smaller
    FCIDUMP the guess fill used to die with an uncaught IndexError.
    """
    if not af_groups:
        return
    try:
        validate_orbital_indices(af_groups, norb, source="FE4S4_AF_GROUPS")
    except ValueError as exc:
        if os.environ.get("FE4S4_AF_GROUPS", "").strip().lower() == "fe4s4":
            raise ValueError(
                f"FE4S4_AF_GROUPS preset 'fe4s4' expects 36 orbitals; FCIDUMP has NORB={norb}. "
                "Use active_space_manifest for AVAS active spaces."
            ) from exc
        raise ValueError(f"{exc}. Use active_space_manifest for AVAS active spaces.") from exc


def _converge_broken_symmetry_uhf(mf, max_follow: int = 5, af_groups: dict | None = None):
    """Drive a UHF calculation to the genuine (possibly spin-broken) solution.

    For a closed-shell singlet, plain UHF started from the RHF density stays at the RHF solution
    (the RHF determinant is a stationary point of the UHF equations) -- so ``mf.to_uhf().kernel()``
    returns an *effectively restricted* reference (<S^2> = 0, identical energy). For strongly
    correlated systems (e.g. Fe-S clusters) the true UHF is lower in energy and spin-broken; seeding
    the LUCJ ansatz / UCCSD amplitudes from the unbroken solution is a restricted reference in
    disguise and defeats the point of running UHF.

    This runs UHF, then iteratively performs an internal STABILITY analysis: if the current solution
    is unstable, it rebuilds the density from the lowest instability mode and re-converges (via
    Newton), following the instability down to a stable (broken-symmetry when appropriate) minimum.
    This is the standard PySCF recipe for reaching the true UHF solution.

    IMPORTANT (Fe-S cubanes): for [4Fe-4S] the internal stability analysis from a SYMMETRIC guess
    stays in the spin-pure (RHF) basin (<S^2>=0, -326.547) -- it does NOT find the antiferromagnetic
    broken-symmetry minimum, because breaking the spatial symmetry to localize the Fe spins requires
    a spin-IMBALANCED initial density, not just following an internal instability. So we FIRST try an
    antiferromagnetic guess (promote a few beta electrons across the HOMO-LUMO gap, per the Noodleman
    BS recipe), stability-follow that, and keep whichever solution (default vs AF-guess) is LOWER.
    For Fe4S4 the AF guess reaches -326.767 (<S^2>~7.6), 219 mHa below the spin-pure solution.
    """

    def _follow_stability(m, n=max_follow):
        stab = False
        for _ in range(n):
            new_mo = m.stability()[0]
            cur = m.mo_coeff
            if np.allclose(new_mo[0], cur[0]) and np.allclose(new_mo[1], cur[1]):
                stab = True
                break
            m = m.newton().run(m.make_rdm1(new_mo, m.mo_occ))
        return m, stab

    def _follow_stability_external(m, n=10):
        """Follow BOTH internal and EXTERNAL instabilities and keep the LOWEST iterate seen.

        The Noodleman AF sublattice minima (Singlet-I/II/III) are near-degenerate and separated by
        external (spin-symmetry-breaking) instabilities that plain internal following (_follow_
        stability) does not cross -- it declares "stable" at the first internal minimum (~-326.769
        for Fe4S4) and never reaches the lower external basin (~-326.801). Mirrors the collaborator
        recipe: m.stability(return_status=True, external=True), rebuild, repeat, track the minimum."""
        best = m
        for _ in range(n):
            res = m.stability(return_status=True, external=True)
            mo = res[0]
            stable_i, stable_e = res[-2], res[-1]
            try:
                # Pass the (external-)instability MO coefficients DIRECTLY to the Newton solver as
                # the initial guess (not a density rebuilt from them): newton().run(mo_coeff) follows
                # the external spin-symmetry-breaking mode into the lower AF basin, whereas
                # run(make_rdm1(mo)) stalls at the higher internal minimum. This is the collaborator
                # recipe and is what reaches -326.801 (vs -326.771 with the density rebuild).
                m = m.newton().run(mo if isinstance(mo, tuple) else m.mo_coeff)
            except Exception:
                break
            if m.e_tot < best.e_tot:
                best = m
            if stable_i and stable_e:
                break
        return best, True

    def _af_guess_uhf(base, base_mol, base_hcore, base_ovlp, base_eri, base_nuc):
        """A fresh plain scf.UHF (same integrals/mol) started from an AF spin-imbalanced density.
        Returns the stability-followed result, or None if it fails. Tries a couple of promotion
        widths. Built from a PLAIN scf.UHF on the mol (not base.__class__, which may be a
        newton-wrapped SecondOrderUHF lacking a simple mol constructor)."""
        try:
            na, nb = base_mol.nelec
        except Exception:
            return None
        n = base_mol.nao
        best = None
        for shift in (2, 1, 3):
            if nb - shift < 0 or nb + shift > n:
                continue
            m = scf.UHF(base_mol)
            # carry the FCIDUMP integral overrides if present
            if base_hcore is not None:
                m.get_hcore = lambda *a, **k: base_hcore
            if base_ovlp is not None:
                m.get_ovlp = lambda *a, **k: base_ovlp
            if base_eri is not None:
                m._eri = base_eri
            if base_nuc is not None:
                m.mol.energy_nuc = lambda *a: base_nuc
            m.max_cycle = 400
            m.conv_tol = 1e-9
            mo = np.eye(n)
            occ_a = list(range(na))
            occ_b = list(range(nb - shift)) + list(range(nb, nb + shift))
            dma = mo[:, occ_a] @ mo[:, occ_a].T
            dmb = mo[:, occ_b] @ mo[:, occ_b].T
            try:
                m.kernel(dm0=(dma, dmb))
                m, _ = _follow_stability(m)
                if best is None or m.e_tot < best.e_tot:
                    best = m
            except Exception:
                continue

        # Atom-localized (Noodleman) AF guess: when an orbital->fragment map is supplied, seed the
        # density so specific magnetic centers get spin-UP and the counter-set spin-DOWN (the true
        # antiferromagnetic sublattices), instead of a chemistry-blind energy-ordered promotion. For
        # Fe-S cubanes this lands in a ~33 mHa LOWER broken-symmetry basin than the energy-ordered
        # guess (Fe4S4: -326.801 vs -326.768) -- the reference orbitals span the Fe-3d correlation
        # far better, which is what SQD actually recovers on. `af_groups` maps fragment name ->
        # list[orbital idx]; keys "up"/"down" list which fragments are alpha/beta-polarized, closed
        # fragments (doubly occupied) are everything else. We stability-follow (external) and keep
        # the lowest across all guesses.
        if af_groups:
            # Fractional polarization amplitude p in [0,1]: p=1 -> full 1.0/0.0 spin split on the
            # magnetic centers (maximally localized, large <S^2>); p<1 -> 0.5(1+/-p) split, letting
            # the sublattices couple more (SMALLER <S^2>, closer to the physical super-exchange
            # regime). The paper notes the singlet-triplet exchange runs through the bridging S 3p
            # orbitals -- so "free" fragments (the bridging S) are seeded half-filled to MEDIATE that
            # exchange rather than being forced doubly-occupied. Local sweep: p=0.6 + free bridging S
            # gives the deepest BS-UCCSD (-327.143 vs -327.102 at p=1.0) and lowest <S^2>.
            # The fill (up/down 0.5 +/- 0.5*pol, free 0.5/0.5, other fragments doubly occupied)
            # lives in active_space.density_from_af_groups, which bounds-checks the indices first.
            dm0 = density_from_af_groups(af_groups, n)
            m = scf.UHF(base_mol)
            if base_hcore is not None:
                m.get_hcore = lambda *a, **k: base_hcore
            if base_ovlp is not None:
                m.get_ovlp = lambda *a, **k: base_ovlp
            if base_eri is not None:
                m._eri = base_eri
            if base_nuc is not None:
                m.mol.energy_nuc = lambda *a: base_nuc
            m.max_cycle = 400
            m.conv_tol = 1e-9
            try:
                m.kernel(dm0=(dm0[0], dm0[1]))
                # Follow external instabilities (10 rounds): the AF sublattice minima are
                # near-degenerate (Singlet-I/II/III) and only EXTERNAL following reaches the lowest
                # basin (~-326.801 for Fe4S4); internal-only following stalls at ~-326.769.
                m, _ = _follow_stability_external(m, n=10)
                if best is None or m.e_tot < best.e_tot:
                    best = m
            except Exception:
                pass
        return best

    # Fail fast on an af_groups spec that does not fit this Hamiltonian (e.g. the 36-orbital
    # 'fe4s4' preset on a 16-orbital AVAS FCIDUMP), before any SCF work.
    _check_af_groups(af_groups, mf.mol.nao)

    # Capture the mol + any FCIDUMP integral overrides from the ORIGINAL mf before newton-wrapping,
    # so the AF-guess UHF can be rebuilt as a plain scf.UHF with the same Hamiltonian.
    _base_mol = mf.mol
    _base_hcore = mf.get_hcore() if callable(getattr(mf, "get_hcore", None)) else None
    _base_ovlp = mf.get_ovlp() if callable(getattr(mf, "get_ovlp", None)) else None
    _base_eri = getattr(mf, "_eri", None)
    try:
        _base_nuc = float(mf.mol.energy_nuc())
    except Exception:
        _base_nuc = None

    mf.kernel()
    stable = False
    for _ in range(max_follow):
        # stability() returns the (internal) stable MOs; if they differ from the current MOs the
        # solution was unstable and we follow them down.
        new_mo = mf.stability()[0]
        # new_mo is (mo_a, mo_b); detect whether it changed (instability found).
        cur = mf.mo_coeff
        changed = not (
            np.allclose(new_mo[0], cur[0]) and np.allclose(new_mo[1], cur[1])
        )
        if not changed:
            stable = True
            break  # stable -> genuine UHF minimum reached
        dm = mf.make_rdm1(new_mo, mf.mo_occ)
        mf = mf.newton().run(dm)

    # Also try the antiferromagnetic broken-symmetry guess and keep whichever solution is LOWER.
    # For closed-shell/RHF-basin cases the AF guess just returns to the same energy (no harm); for
    # Fe-S cubanes it finds the genuine BS minimum the stability-only path misses.
    af = _af_guess_uhf(mf, _base_mol, _base_hcore, _base_ovlp, _base_eri, _base_nuc)
    if af is not None and af.converged and af.e_tot < mf.e_tot - 1e-6:
        try:
            ss_af, _ = af.spin_square()
        except Exception:
            ss_af = float("nan")
        warnings.warn(
            f"Broken-symmetry UHF found via AF guess: E={af.e_tot:.6f} <S^2>={ss_af:.3f} is "
            f"{(mf.e_tot - af.e_tot) * 1000:.1f} mHa below the stability-only solution "
            f"({mf.e_tot:.6f}). Using the broken-symmetry reference.",
            RuntimeWarning,
            stacklevel=2,
        )
        return af

    # Robustness (audit P4): if the follow loop exhausted max_follow while still unstable, the
    # returned reference is NOT a genuine minimum. Warn loudly and report the final <S^2> so a
    # downstream seed built from a non-stable / spin-contaminated reference is visible, not silent.
    if not stable:
        try:
            ss, _mult = mf.spin_square()
        except Exception:
            ss = float("nan")
        warnings.warn(
            f"UHF stability following did not converge to a stable solution after "
            f"{max_follow} follows; returning the last iterate (<S^2>={ss:.4f}). The UHF "
            f"reference may be non-stationary or spin-contaminated -- t2/occupancy seeds built "
            f"from it should be treated with caution.",
            RuntimeWarning,
            stacklevel=2,
        )
    return mf


def _uhf_factory(mf):
    """Return a callable building fresh plain ``scf.UHF`` objects on ``mf``'s Hamiltonian.

    Carries the FCIDUMP integral overrides (hcore / ovlp / _eri / nuclear energy). A fresh object
    per SCF avoids aliasing: ``newton()`` on an already second-order object returns ``self``, so
    re-running it would overwrite an earlier solution kept for comparison.
    """
    base_mol = mf.mol
    hcore, ovlp, eri = mf.get_hcore(), mf.get_ovlp(), mf._eri
    nuc = float(mf.mol.energy_nuc())

    def make():
        m = scf.UHF(base_mol)
        m.get_hcore = lambda *a, **k: hcore
        m.get_ovlp = lambda *a, **k: ovlp
        m._eri = eri
        m.mol.energy_nuc = lambda *a: nuc
        m.max_cycle = 400
        m.conv_tol = 1e-9
        return m

    return make


def _converge_from_density(make_uhf, dm0, max_follow: int = 10):
    """Second-order UHF from ``dm0``, then follow internal instabilities; keep the lowest.

    Each follow step re-converges a fresh object from the density of the lowest internal
    instability mode (the ``bs_uno_avas.verify.active_uhf`` recipe). External (UHF -> GHF) instabilities
    cannot be followed within UHF and are not checked.
    """
    from pyscf import lib

    m = make_uhf().newton()
    m.kernel(dm0=(np.asarray(dm0[0]), np.asarray(dm0[1])))
    best, stable = m, False
    for _ in range(max_follow):
        try:
            mo_i, _, stable_i, _ = m.stability(return_status=True)
        except lib.exceptions.LinearDependencyError:
            break  # no stability analysis possible (e.g. no open shells)
        if stable_i:
            stable = True
            break
        nxt = make_uhf().newton()
        nxt.kernel(dm0=m.make_rdm1(mo_i, m.mo_occ))
        m = nxt
        if m.e_tot < best.e_tot:
            best = m
    return best, stable


def _strict_bs() -> bool:
    return os.environ.get("SBD_BS_STRICT", "0") == "1"


def _bs_problem(msg: str, logger) -> None:
    """Warn (default) or raise (``SBD_BS_STRICT=1``) on a reference that deviates from the
    manifest."""
    if _strict_bs():
        raise RuntimeError(f"{msg} (SBD_BS_STRICT=1)")
    logger.warning(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _converge_reference_uhf(mf, spec: ActiveSpaceSpec, logger=None):
    """Reproduce the manifest's broken-symmetry UHF on the FCIDUMP Hamiltonian of ``mf``.

    Two starts: (a) the manifest alpha/beta 1-RDM, (b) the label-based guess for
    ``spec.spin_pattern`` (Fe 3d in alpha/beta, S 3p doubly occupied). Each is converged with a
    second-order solver plus internal stability following. Among the converged solutions whose
    spin tag equals ``spec.spin_tag`` the lowest energy wins (the RDM start on a tie,
    |dE| <= 1e-6). If none matches, the lowest converged solution is used with a loud warning.
    Energy is compared with ``spec.e_active_uhf`` (tolerance ``SBD_BS_E_TOL``, default 1e-4 Eh).
    ``SBD_BS_STRICT=1`` turns tag / energy / convergence mismatches into ``RuntimeError``.

    No energy-ordered or ``FE4S4_AF_GROUPS`` guess is tried: the reference must be the
    manifest's state.

    Returns ``(uhf, info)``: a plain converged ``scf.UHF`` carrying the chosen solution and a dict
    with ``energy, tag, source, s2, atom_spins, stable`` plus per-start ``candidates``.
    """
    logger = logger or _get_logger()
    labels = spec.labels.tolist()
    if tuple(mf.mol.nelec) != (spec.nalpha, spec.nbeta):
        raise ValueError(
            f"FCIDUMP gives (n_alpha, n_beta)={tuple(mf.mol.nelec)} but the manifest "
            f"{spec.manifest_path} gives ({spec.nalpha}, {spec.nbeta})"
        )
    if mf.mol.nao != spec.norb:
        raise ValueError(f"FCIDUMP has {mf.mol.nao} orbitals, manifest NORB={spec.norb}")

    make_uhf = _uhf_factory(mf)
    starts = (
        ("manifest-rdm", spec.rdm),
        ("manifest-labels", bs_density_from_labels(labels, spec.spin_pattern)),
    )
    candidates = []
    for source, dm0 in starts:
        m, stable = _converge_from_density(make_uhf, dm0)
        dm_a, dm_b = m.make_rdm1()
        spins = atom_spins(dm_a, dm_b, labels)
        cand = {
            "source": source,
            "energy": float(m.e_tot),
            "tag": spin_tag_from_spins(spins),
            "s2": float(m.spin_square()[0]),
            "atom_spins": spins,
            "converged": bool(m.converged),
            "stable": stable,
            "mf": m,
        }
        candidates.append(cand)
        logger.info(
            "BS-UHF reference start %s: E=%.8f tag=%s <S^2>=%.4f converged=%s stable=%s",
            source, cand["energy"], cand["tag"], cand["s2"], cand["converged"], stable,
        )

    def pick(pool):
        # lowest energy; candidates within 1e-6 Eh of it keep start order (RDM first)
        e_min = min(c["energy"] for c in pool)
        return next(c for c in pool if c["energy"] <= e_min + 1e-6)

    converged = [c for c in candidates if c["converged"]]
    matching = [c for c in converged if c["tag"] == spec.spin_tag]
    summary = ", ".join(f"{c['source']}: E={c['energy']:.8f} tag={c['tag']}" for c in candidates)
    if matching:
        best = pick(matching)
    else:
        best = pick(converged or candidates)
        _bs_problem(
            f"No converged BS-UHF start reproduced the manifest spin tag {spec.spin_tag!r} "
            f"({summary}); using {best['source']} (tag {best['tag']!r}).",
            logger,
        )
    if not best["converged"]:
        _bs_problem(f"BS-UHF reference from {best['source']} did not converge.", logger)

    e_tol = float(os.environ.get("SBD_BS_E_TOL", "1e-4"))
    if spec.e_active_uhf is not None and abs(best["energy"] - spec.e_active_uhf) > e_tol:
        _bs_problem(
            f"BS-UHF reference E={best['energy']:.8f} differs from manifest e_active_uhf="
            f"{spec.e_active_uhf:.8f} by {best['energy'] - spec.e_active_uhf:+.2e} Eh "
            f"(> SBD_BS_E_TOL={e_tol:g}).",
            logger,
        )

    spins_txt = " ".join(f"{k}={v:+.3f}" for k, v in best["atom_spins"].items())
    logger.info(
        "BS-UHF reference: reference_source=%s tag=%s (manifest %s) E=%.8f <S^2>=%.4f "
        "atom spins: %s",
        best["source"], best["tag"], spec.spin_tag, best["energy"], best["s2"], spins_txt,
    )

    # Hand back a plain UHF (not the second-order wrapper) carrying the chosen solution, as the
    # cache-reuse path does.
    chosen = best["mf"]
    out = make_uhf()
    out.mo_coeff = tuple(np.asarray(c) for c in chosen.mo_coeff)
    out.mo_occ = tuple(np.asarray(o) for o in chosen.mo_occ)
    out.mo_energy = tuple(np.asarray(e) for e in chosen.mo_energy)
    out.e_tot = float(chosen.e_tot)
    out.converged = bool(chosen.converged)
    info = {k: v for k, v in best.items() if k != "mf"}
    info["candidates"] = [{k: v for k, v in c.items() if k != "mf"} for c in candidates]
    return out, info


def _fcidump_uhf(mf, nuc: float):
    """Plain (non-symmetry-adapted) ``scf.UHF`` on the FCIDUMP Hamiltonian carried by the RHF
    shell ``mf`` from ``tools.fcidump.to_scf`` (see the Audit P3 note at the call site)."""
    base_mol = mf.mol
    if getattr(base_mol, "symmetry", False):
        base_mol.symmetry = False
        base_mol.build(dump_input=False, parse_arg=False)
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = mf._eri
    out = scf.UHF(base_mol)
    out.get_hcore = lambda *args, **kwargs: hcore
    out.get_ovlp = lambda *args, **kwargs: ovlp
    out._eri = eri
    out.mol.energy_nuc = lambda *args: nuc  # preserved across the conversion
    out.symmetry = False
    return out


def manifest_reference_summary(active_space_manifest: str, fcidump_file: str | None = None):
    """Load a manifest and rebuild its BS-UHF reference exactly as the flow does, without UCCSD.

    For launcher dry-runs / pre-flight checks. Returns ``(spec, info)`` where ``info`` is the
    ``_converge_reference_uhf`` dict (energy, tag, source, s2, atom_spins, candidates). Honours
    ``SBD_BS_STRICT`` / ``SBD_BS_E_TOL`` like the flow. Does not read or write the UHF cache.
    """
    from contextlib import redirect_stdout

    from .active_space import load_active_space_manifest

    spec = load_active_space_manifest(active_space_manifest, fcidump_override=fcidump_file)
    with redirect_stdout(io.StringIO()):  # pyscf prints "Parsing <file>"
        shell = tools.fcidump.to_scf(str(spec.fcidump))
    shell.mol.verbose = 0
    _, info = _converge_reference_uhf(_fcidump_uhf(shell, shell.mol.energy_nuc()), spec)
    return spec, info


# --------------------------------------------------------------------------------------------
# UHF reference cache (SBD_UHF_CACHE / SBD_UHF_CACHE_DIR)
# --------------------------------------------------------------------------------------------

# Bump when the cached content or key recipe changes.
_UHF_CACHE_VERSION = "v2"


def _uhf_cache_path(cache_dir: str, fcidump_sha256: str, guess_token: str, nelec) -> str:
    """Content-based cache file: FCIDUMP sha256 + guess token + (n_alpha, n_beta).

    The FCIDUMP path is deliberately not part of the key: the same file copied elsewhere hits,
    a regenerated file at the same path misses.
    """
    import hashlib

    nelec_txt = "nelec=" + str(tuple(int(n) for n in nelec))
    raw = "|".join([_UHF_CACHE_VERSION, fcidump_sha256, guess_token, nelec_txt])
    key = hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]
    return os.path.join(cache_dir, f"uhf_bs_{key}.npz")


def _uhf_cache_load(cpath: str, mf, norb: int) -> bool:
    """Load a cached reference into ``mf`` (mo_coeff / mo_occ / mo_energy / e_tot). Returns
    False (and leaves ``mf`` untouched) if absent, unreadable, or inconsistent with ``mf``."""
    if not os.path.exists(cpath):
        return False
    try:
        with np.load(cpath, allow_pickle=False) as c:
            data = {k: c[k] for k in c.files}
        nelec = tuple(int(n) for n in data["nelec"])
        shapes_ok = all(data[k].shape == (norb, norb) for k in ("mo_a", "mo_b"))
        occ = (int(round(data["occ_a"].sum())), int(round(data["occ_b"].sum())))
    except Exception as exc:
        print(f"[uhf-cache] ignoring unreadable cache {cpath}: {exc}")
        return False
    if nelec != tuple(mf.mol.nelec) or occ != nelec or not shapes_ok:
        print(
            f"[uhf-cache] ignoring {cpath}: cached nelec={nelec} occ={occ} norb ok={shapes_ok} "
            f"do not match nelec={tuple(mf.mol.nelec)} norb={norb}"
        )
        return False
    mf.mo_coeff = (data["mo_a"], data["mo_b"])
    mf.mo_occ = (data["occ_a"], data["occ_b"])
    mf.mo_energy = (data["ene_a"], data["ene_b"])
    mf.e_tot = float(data["e_tot"])
    mf.converged = True
    tag = f" tag={data['tag']}" if "tag" in data else ""
    print(f"[uhf-cache] reuse BS-UHF reference {cpath} (E={mf.e_tot:.6f}{tag})")
    return True


def _uhf_cache_save(cpath: str, mf, *, tag: str | None = None, source: str | None = None) -> None:
    """Write the converged reference atomically (tmp file + rename); failures only warn."""
    try:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        mo_a, mo_b = mf.mo_coeff
        occ_a, occ_b = mf.mo_occ
        ene_a, ene_b = mf.mo_energy
        extra = {k: np.array(v) for k, v in (("tag", tag), ("source", source)) if v is not None}
        tmp = f"{cpath}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            np.savez(
                f, mo_a=mo_a, mo_b=mo_b, occ_a=occ_a, occ_b=occ_b, ene_a=ene_a, ene_b=ene_b,
                e_tot=float(mf.e_tot), nelec=np.array(mf.mol.nelec), **extra,
            )
        os.replace(tmp, cpath)
        print(f"[uhf-cache] saved BS-UHF reference {cpath} (E={mf.e_tot:.6f})")
    except Exception as exc:
        print(f"[uhf-cache] save failed (continuing): {exc}")


def _check_cached_reference(mf, spec: ActiveSpaceSpec, cpath: str, logger=None) -> dict | None:
    """Re-validate a cached manifest reference loaded into ``mf``.

    The tag is recomputed from the cached orbitals (not trusted from the file). A reference that
    does not match the manifest tag / energy is not served from the cache: returns None so the
    caller re-converges (and emits the usual warnings / SBD_BS_STRICT errors).
    """
    logger = logger or _get_logger()
    labels = spec.labels.tolist()
    dm_a, dm_b = mf.make_rdm1()
    spins = atom_spins(dm_a, dm_b, labels)
    tag = spin_tag_from_spins(spins)
    e_tol = float(os.environ.get("SBD_BS_E_TOL", "1e-4"))
    e_ok = spec.e_active_uhf is None or abs(mf.e_tot - spec.e_active_uhf) <= e_tol
    if tag != spec.spin_tag or not e_ok:
        logger.warning(
            "[uhf-cache] cached reference %s has tag=%s E=%.8f (manifest tag=%s E=%s); "
            "re-converging instead of reusing it.",
            cpath, tag, mf.e_tot, spec.spin_tag, spec.e_active_uhf,
        )
        return None
    try:
        with np.load(cpath, allow_pickle=False) as c:
            source = str(c["source"]) if "source" in c.files else "manifest-cache"
    except Exception:
        source = "manifest-cache"
    spins_txt = " ".join(f"{k}={v:+.3f}" for k, v in spins.items())
    logger.info(
        "BS-UHF reference (cached %s): reference_source=%s tag=%s (manifest %s) E=%.8f "
        "atom spins: %s",
        cpath, source, tag, spec.spin_tag, mf.e_tot, spins_txt,
    )
    return {"tag": tag, "source": source, "energy": float(mf.e_tot), "atom_spins": spins}


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

    # Run CCSD (amplitude seed for the LUCJ ansatz). Raise max_cycle + DIIS for hard active spaces.
    mycc = cc.CCSD(mf)
    mycc.max_cycle = int(os.environ.get("SEED_CCSD_MAX_CYCLE", "200"))
    mycc.diis_space = 12
    mycc.kernel()
    t2 = mycc.t2

    # Per-MO occupancies for configuration recovery. recover_configurations expects the mean
    # occupancy of MO p IN THE BASIS THE BITSTRINGS LIVE IN (the canonical MO basis the circuit and
    # integrals use), i.e. the DIAGONAL of the 1-RDM in that basis -- NOT the sorted eigenvalues
    # (natural-orbital occupation numbers in a different basis), which would misassign the bias to
    # the wrong orbitals. The RDM must come from the *correlated* CCSD (mycc), not the SCF object
    # (mf.make_rdm1() is idempotent -> integer 0/1 occupancies, which give recovery no fractional
    # signal and collapse the SQD subspace to bare Hartree-Fock).
    rdm1_ccsd = mycc.make_rdm1()
    occ_ccsd = np.diag(rdm1_ccsd) / 2.0  # spatial-orbital occ (0-2) -> per-spin (0-1)

    # Get PySCF logs dumped into in-memory buffer
    get_run_logger().info(buf.getvalue())

    return ElectronicProperties(
        one_body_tensor=h1,
        two_body_tensor=h2,
        t2=t2,
        initial_occupancy=(occ_ccsd, occ_ccsd),
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
    reference: dict | None = None,
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
        reference: Optional REFERENCE_FIELDS values (manifest-built BS reference).

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

    # Run UCCSD; t2 is the tuple (t2aa, t2ab, t2bb). Strongly-correlated Fe-S active spaces often
    # do not converge in one 50-cycle run. Restart from the current amplitudes (fresh DIIS) up to
    # SEED_CCSD_RESTARTS times instead of one long run (cycles per run: SEED_CCSD_MAX_CYCLE).
    mycc = cc.UCCSD(mf)
    mycc.max_cycle = int(os.environ.get("SEED_CCSD_MAX_CYCLE", "") or 50)
    mycc.diis_space = 12
    mycc.kernel()
    n_runs = max(1, int(os.environ.get("SEED_CCSD_RESTARTS", "") or 10))
    for _ in range(n_runs - 1):
        if mycc.converged:
            break
        mycc.kernel(mycc.t1, mycc.t2)
    if not mycc.converged:
        _get_logger().warning(
            "UCCSD seed not converged after %d run(s) x %d cycles; using the last amplitudes.",
            n_runs, mycc.max_cycle,
        )
    t2_aa, t2_ab, t2_bb = mycc.t2

    # Per-spin natural-orbital occupancies from the *correlated* UCCSD 1-RDM (in MO basis).
    # These must be fractional: configuration recovery downstream uses them to bias sampled
    # bitstrings toward the physically important (near-HF) configurations. The UHF *SCF* RDM is
    # (often) idempotent -> integer 0/1 occupancies, which give recovery no signal and collapse
    # the SQD subspace to bare Hartree-Fock. UCCSD make_rdm1() returns (dm_a, dm_b) in MO basis.
    # Use the DIAGONAL (per-canonical-MO occupancy) of the correlated UCCSD 1-RDM, not the sorted
    # eigenvalues. The bitstrings are measured in the canonical MO basis (PrepareHartreeFockJW +
    # integrals from mf.mo_coeff), so recover_configurations needs occ of MO p in THAT basis; the
    # eigenvalues are natural-orbital occupation numbers in a rotated basis and, once sorted, would
    # misassign the bias to the wrong orbitals (the error lands on the active, partially-occupied
    # orbitals and grows with system size).
    dm_cc_a, dm_cc_b = mycc.make_rdm1()
    occ_a = np.diag(dm_cc_a)
    occ_b = np.diag(dm_cc_b)

    # Get PySCF logs dumped into in-memory buffer
    get_run_logger().info(buf.getvalue())

    props = ElectronicProperties(
        one_body_tensor=h1_a,
        two_body_tensor=h2_aa,
        t2=t2_aa,
        one_body_tensor_b=h1_b,
        two_body_tensor_ab=h2_ab,
        two_body_tensor_bb=h2_bb,
        t2_ab=t2_ab,
        t2_bb=t2_bb,
        initial_occupancy=(occ_a, occ_b),
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        num_orbitals=norb,
        num_electrons=(num_elec_a, num_elec_b),
        open_shell=num_elec_a != num_elec_b,
        unrestricted=True,
        spin_sq=spin_sq,
        **(reference or {}),
    )

    # Opt-in initial orbital optimization: BS-UHF -> UCCSD -> OO -> UCCSD refresh.
    # Default off (OO_INIT unset/0) so existing runs are byte-for-byte unaffected.
    if os.environ.get("OO_INIT", "0") == "1":
        props = _apply_initial_oo(props, mycc, dm_cc_a, dm_cc_b)

    return props


def _apply_initial_oo(
    props: ElectronicProperties,
    mycc: "cc.uccsd.UCCSD",
    dm_cc_a: np.ndarray,
    dm_cc_b: np.ndarray,
) -> ElectronicProperties:
    """Insert one orbital-optimization step between the initial BS-UHF UCCSD and the t2 seed.

    Opt-in via ``OO_INIT=1`` (UHF path only). Flow:

        BS-UHF -> UCCSD (already run; 1-/2-RDM in hand)
               -> OO (rotate orbitals with the fixed UCCSD RDM, two-step MCSCF)
               -> UCCSD refresh in the rotated basis (fresh t2 / occupancy seed).

    OO needs a *correlated* RDM: the BS-UHF SCF density already satisfies the generalized
    Brillouin condition (zero orbital gradient), so an SCF-RDM-driven OO cannot move. UCCSD
    supplies that correlated RDM without needing a CI subspace.

    UCCSD is non-variational and, for large active spaces (e.g. Fe4S4 36o/54e), frequently
    unconverged, so the rotation is deliberately conservative: a single fixed-RDM OO step with
    a small trust radius, guarded by an energy check that *discards* the rotation unless the
    refreshed UCCSD energy is at or below the pre-OO UCCSD energy (safe fallback to the
    un-rotated reference).

    Thresholds are shared with the DE-loop OO (see sbd/main.py): ``OO_GRAD_TOL`` (1e-3),
    ``OO_TRUST`` (0.5), ``OO_MAXITER`` (300) -- identical env names and defaults so "all
    criteria are aligned from the initial OO onward".
    """
    from qcsc_workflow_utility.orbital_opt import (
        optimize_orbitals,
        rotate_electronic_properties,
    )

    logger = get_run_logger()
    e_cc_before = float(mycc.e_tot)

    # UCCSD spin-block 2-RDM in PySCF "pqrs" (chemist) storage. make_rdm1() already matches the
    # physicist convention optimize_orbitals expects (dm1[p,q] = <q^dag p>), so no transpose.
    d2aa, d2ab, d2bb = mycc.make_rdm2()

    gtol = float(os.environ.get("OO_GRAD_TOL", "1e-3"))
    trust = float(os.environ.get("OO_TRUST", "0.5"))
    maxiter = int(os.environ.get("OO_MAXITER", "300"))

    Ua, Ub, e_oo, grad_norm = optimize_orbitals(
        props,
        np.asarray(dm_cc_a),
        np.asarray(dm_cc_b),
        np.asarray(d2aa),
        np.asarray(d2ab),
        np.asarray(d2bb),
        rdm2_notation="pqrs",  # UCCSD make_rdm2 storage (NOT the native solver's "prqs")
        gtol=gtol,
        trust_radius=trust,
        maxiter=maxiter,
    )

    props_rot = rotate_electronic_properties(props, Ua, Ub)
    props_new, e_cc_after = refresh_cc_seed(props_rot, return_energy=True)

    if (
        props_new is not None
        and e_cc_after is not None
        and np.isfinite(e_cc_after)
        and e_cc_after <= e_cc_before + 1e-6
    ):
        logger.info(
            "OO_INIT accepted: E(UCCSD before OO)=%.10f  E_oo(fixed-RDM, total)=%.10f  "
            "|grad|=%.3e  E(UCCSD after OO)=%.10f  dE=%+.3e Ha  (trust=%.3g, gtol=%.1e).",
            e_cc_before, e_oo, grad_norm, e_cc_after, e_cc_after - e_cc_before, trust, gtol,
        )
        return props_new

    logger.warning(
        "OO_INIT rejected (no energy gain / NaN): E(UCCSD before OO)=%.10f  E_oo=%.10f  "
        "|grad|=%.3e  E(UCCSD after OO)=%s  -> keeping pre-OO BS-UHF/UCCSD reference.",
        e_cc_before, e_oo, grad_norm,
        f"{e_cc_after:.10f}" if e_cc_after is not None else "None",
    )
    return props


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
        mf = scf.UHF(mol)
        # Follow any spin instability to the true (broken-symmetry) UHF minimum (see
        # _converge_broken_symmetry_uhf) so the reference is genuinely unrestricted for singlets.
        mf = _converge_broken_symmetry_uhf(mf)
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
    fcidump_file: str | None,
    spin_sq: float = 0.0,
    unrestricted: bool = False,
    active_space_manifest: str | None = None,
) -> ElectronicProperties:
    """Precompute molecular orbital property from FCIDump file with classical methods.

    Args:
        fcidump_file: Location of FCIDump file storing 1-electron and 2-electron integrals.
            May be None when ``active_space_manifest`` is given (the manifest's FCIDUMP is used);
            if both are given the file must be byte-identical to the manifest's FCIDUMP.
        spin_sq: Target value for the total spin squared for the ground state.
        unrestricted: Run an unrestricted (UHF) calculation instead of restricted (RHF). The
            number of unpaired electrons is taken from the FCIDUMP ``MS2`` header.
        active_space_manifest: Path to an AVAS active-space manifest (schema v1, see
            ``active_space.load_active_space_manifest``). With UHF the broken-symmetry reference
            is rebuilt from the manifest RDM / spin pattern (``_converge_reference_uhf``) instead
            of the default / FE4S4_AF_GROUPS guesses. With RHF it only supplies the FCIDUMP.

    Returns:
        ElectronicProperties object containing molecular integrals and properties.
    """
    # PySCF doesn't use the standard Python logging and Prefect cannot capture it.
    # The logs are directly written in the stdout or in a file.
    # To forward the logs to the Prefect logging sytem,
    # we set an in-memory buffer to the PySCF logging system and read from there.
    buf = io.StringIO()

    spec = None
    if active_space_manifest:
        from .active_space import load_active_space_manifest

        spec = load_active_space_manifest(active_space_manifest, fcidump_override=fcidump_file)
        fcidump_file = str(spec.fcidump)
        if not unrestricted:
            msg = (
                f"active_space_manifest {active_space_manifest} is ignored for RHF except for "
                "its FCIDUMP; the broken-symmetry reference needs method='uhf'."
            )
            _get_logger().warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    elif fcidump_file is None:
        raise ValueError(
            "compute_molecular_integrals_from_fcidump needs fcidump_file or a manifest"
        )

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
        # Audit P3: tools.fcidump.to_scf sets mol.symmetry=True with a *guessed* point group when
        # the FCIDUMP carries ORBSYM (both the Fe2S2 and Fe4S4 dumps do: ORBSYM=1). The RHF branch
        # defeats this with mf.symmetry=False (below); the UHF branch previously did NOT, so UHF ran
        # its kernel + stability analysis under a possibly-wrong group -> symmetry-constrained /
        # misconverged reference, and stability following couldn't break the relevant symmetry.
        # Disable symmetry on the mol and REBUILD it before converting, so to_uhf() produces a plain
        # (non-symmetry-adapted) UHF whose kernel()/stability() don't require a point group. Setting
        # .symmetry=False on an already-symmetry-adapted object is insufficient (SymAdaptedUHF.build
        # still raises "mol.symmetry not enabled"); the rebuild is what actually clears it.
        # to_scf builds a SymAdaptedRHF, and mf.to_uhf() preserves that symmetry-adapted class --
        # whose build() still demands mol.symmetry even after we clear the flag. So disable symmetry
        # on the mol, rebuild it, and construct a PLAIN scf.UHF on it (not mf.to_uhf()); carry the
        # FCIDUMP integral overrides (get_hcore/_eri) + nuclear energy across by hand.
        mf = _fcidump_uhf(mf, nuc)
        # Deterministic broken-symmetry reference: cache converged UHF MOs so every run
        # (do_rdm on/off, OO on/off, sample vs recovery) starts from the IDENTICAL reference.
        # Fe4S4's 400-cycle SCF is thread/BLAS-order sensitive -> different BS local minima
        # run-to-run otherwise. The key is content-based (FCIDUMP sha256 + guess token + nelec), so
        # a regenerated FCIDUMP at the same path is never served a stale reference.
        # SBD_UHF_CACHE=0 skips the lookup (the fresh result still refreshes the cache file);
        # relocate with SBD_UHF_CACHE_DIR.
        cache_dir = os.environ.get("SBD_UHF_CACHE_DIR", os.path.expanduser("~/.cache/qcsc_uhf"))
        cache_read = os.environ.get("SBD_UHF_CACHE", "1") != "0"
        if spec is not None:
            # Manifest: reproduce the manifest's BS state (RDM start, label start as fallback and
            # check); no default / FE4S4_AF_GROUPS guesses.
            cpath = _uhf_cache_path(
                cache_dir, spec.fcidump_sha256, "manifest=" + spec.cache_token, mf.mol.nelec
            )
            ref = None
            if cache_read and _uhf_cache_load(cpath, mf, norb):
                ref = _check_cached_reference(mf, spec, cpath)
            if ref is None:
                mf, ref = _converge_reference_uhf(mf, spec)
                _uhf_cache_save(cpath, mf, tag=ref["tag"], source=ref["source"])
            reference = {
                "reference_tag": ref["tag"],
                "reference_source": ref["source"],
                "fcidump_sha256": spec.fcidump_sha256,
            }
            return _build_property_uhf(mf, norb, spin_sq, buf, reference=reference)
        # Follow any spin instability to the genuine (broken-symmetry) UHF minimum -- otherwise a
        # singlet stays at the RHF solution and the "UHF" reference/UCCSD/t2 seed is restricted.
        # For Fe-S cubanes, an atom-localized (Noodleman) AF guess -- supplied via FE4S4_AF_GROUPS --
        # reaches a ~33 mHa lower BS basin than the energy-ordered guess (better Fe-3d orbitals for
        # SQD to recover on). See _parse_af_groups; unset -> plain energy-ordered AF guess (default).
        _af_groups = _parse_af_groups()
        _check_af_groups(_af_groups, norb)  # before the cache lookup: clear error, not IndexError
        cpath = _uhf_cache_path(
            cache_dir, file_sha256(fcidump_file), "af=" + str(_af_groups), mf.mol.nelec
        )
        if not (cache_read and _uhf_cache_load(cpath, mf, norb)):
            mf = _converge_broken_symmetry_uhf(mf, af_groups=_af_groups)
            _uhf_cache_save(cpath, mf)
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


def refresh_cc_seed(
    elec_props: ElectronicProperties,
    return_energy: bool = False,
):
    """Re-run CCSD/UCCSD on the current (rotated) integrals and return updated ElectronicProperties.

    This is the CC-refresh step for OO_TO_LUCJ=1: after orbital optimization rotates the
    Hamiltonian, solve CC in the new basis so that t2 and initial_occupancy match the rotated
    integrals. Uses the previous t2 amplitudes (from elec_props) as warm-start initial guess
    so that UCCSD converges faster in the rotated basis. Returns a new ElectronicProperties
    with updated t2/occupancy fields, or None if amplitudes contain NaN.

    If ``return_energy`` is True, returns ``(props_or_None, e_tot_or_None)`` instead: the
    refreshed UCCSD/CCSD total energy accompanies the properties (used by the OO_INIT energy
    guard). Default False preserves the original single-value return for existing callers.
    """
    import logging

    log = logging.getLogger(__name__)
    norb = elec_props.num_orbitals
    nelec_a, nelec_b = elec_props.num_electrons

    max_cycle = int(os.environ.get("SEED_CCSD_MAX_CYCLE", "200"))

    if elec_props.unrestricted:
        return _refresh_cc_seed_uhf(elec_props, log, max_cycle, return_energy)
    else:
        return _refresh_cc_seed_rhf(elec_props, log, max_cycle, return_energy)


def _refresh_cc_seed_rhf(
    elec_props: ElectronicProperties,
    log,
    max_cycle: int,
    return_energy: bool = False,
) -> ElectronicProperties | None:
    """CC refresh for RHF."""
    norb = elec_props.num_orbitals
    nelec_a, nelec_b = elec_props.num_electrons
    h1 = np.asarray(elec_props.one_body_tensor)
    h2 = np.asarray(elec_props.two_body_tensor)

    mol = gto.M()
    mol.nelectron = nelec_a + nelec_b
    mol.spin = 0
    mol.verbose = 0
    mol.incore_anyway = True

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a, **k: h1
    mf.get_ovlp = lambda *a, **k: np.eye(norb)
    mf._eri = ao2mo.restore(8, h2.reshape(norb, norb, norb, norb), norb)
    mf.mol.energy_nuc = lambda *a: elec_props.nuclear_repulsion_energy
    mf.mol.nao_nr = lambda *a: norb

    mo_coeff = np.eye(norb)
    mo_occ = np.zeros(norb)
    mo_occ[:nelec_a] = 2.0

    # Compute proper Fock matrix diagonal for orbital energies.
    # F_pq = h1_pq + sum_i [2*(pi|qi) - (pq|ii)] where i are occupied.
    nocc = nelec_a
    j_diag = 2.0 * np.einsum("piqi->pq", h2[:, :nocc, :, :nocc])
    k_diag = -np.einsum("pqii->pq", h2[:, :, :nocc, :nocc])
    fock = h1 + j_diag + k_diag
    mo_energy = np.diag(fock)

    # HF energy: E = sum_i h_ii + 0.5 * sum_ij [2*(ii|jj) - (ij|ji)] + E_nuc
    e_hf = (
        2.0 * np.einsum("ii->", h1[:nocc, :nocc])
        + 2.0 * np.einsum("iijj->", h2[:nocc, :nocc, :nocc, :nocc])
        - np.einsum("ijji->", h2[:nocc, :nocc, :nocc, :nocc])
        + elec_props.nuclear_repulsion_energy
    )

    mf.mo_coeff = mo_coeff
    mf.mo_occ = mo_occ
    mf.mo_energy = mo_energy
    mf.e_tot = e_hf
    mf.converged = True

    mycc = cc.CCSD(mf)
    mycc.max_cycle = max_cycle
    mycc.diis_space = 12

    # Warm-start: use old t2 as initial guess for faster convergence in rotated basis.
    t2_old = np.asarray(elec_props.t2)
    t1_init = np.zeros((nelec_a, norb - nelec_a))
    log.info("CC refresh (RHF): warm-start with old ||t2||=%.4e", np.linalg.norm(t2_old))
    mycc.kernel(t1=t1_init, t2=t2_old)

    if not mycc.converged:
        log.warning("CC refresh (RHF): CCSD did not converge after %d cycles. Using unconverged amplitudes.", max_cycle)

    t1_new, t2_new = mycc.t1, mycc.t2
    log.info(
        "CC refresh (RHF): converged. E_corr=%.6f ||t1||=%.4e ||t2||=%.4e",
        mycc.e_corr, np.linalg.norm(t1_new), np.linalg.norm(t2_new),
    )

    rdm1_cc = mycc.make_rdm1()
    occ_new = np.diag(rdm1_cc) / 2.0

    if np.any(np.isnan(t2_new)) or np.any(np.isnan(occ_new)):
        log.warning("CC refresh (RHF): NaN detected in amplitudes or occupancy. Discarding.")
        return (None, None) if return_energy else None

    props = ElectronicProperties(
        one_body_tensor=elec_props.one_body_tensor,
        two_body_tensor=elec_props.two_body_tensor,
        t2=t2_new,
        initial_occupancy=(occ_new, occ_new),
        nuclear_repulsion_energy=elec_props.nuclear_repulsion_energy,
        num_orbitals=norb,
        num_electrons=elec_props.num_electrons,
        open_shell=elec_props.open_shell,
        spin_sq=elec_props.spin_sq,
        **reference_fields(elec_props),
    )
    return (props, float(mycc.e_tot)) if return_energy else props


def _refresh_cc_seed_uhf(
    elec_props: ElectronicProperties,
    log,
    max_cycle: int,
    return_energy: bool = False,
) -> ElectronicProperties | None:
    """CC refresh for UHF.

    Build a mock UHF SCF object whose get_jk correctly handles the separate aa/ab/bb
    integral blocks, so PySCF UCCSD.ao2mo() works natively.
    """
    norb = elec_props.num_orbitals
    nelec_a, nelec_b = elec_props.num_electrons
    h1_a = np.asarray(elec_props.one_body_tensor)
    h1_b = np.asarray(elec_props.one_body_tensor_b)
    h2_aa = np.asarray(elec_props.two_body_tensor)
    h2_ab = np.asarray(elec_props.two_body_tensor_ab)
    h2_bb = np.asarray(elec_props.two_body_tensor_bb)
    nuc = elec_props.nuclear_repulsion_energy

    mol = gto.M()
    mol.nelectron = nelec_a + nelec_b
    mol.spin = nelec_a - nelec_b
    mol.verbose = 4
    mol.incore_anyway = True

    mf = scf.UHF(mol)
    ovlp = np.eye(norb)
    mf.get_ovlp = lambda *a, **k: ovlp
    mf.mol.nao_nr = lambda *a: norb
    mf.mol.energy_nuc = lambda *a: nuc
    mf.get_hcore = lambda *a, **k: h1_a

    _eri_aa_8fold = ao2mo.restore(8, h2_aa.reshape(norb, norb, norb, norb), norb)
    _eri_bb_8fold = ao2mo.restore(8, h2_bb.reshape(norb, norb, norb, norb), norb)
    _h2_ab = h2_ab
    mf._eri = _eri_aa_8fold

    def _uhf_get_jk(mol_arg=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if dm is None:
            dm = mf.make_rdm1()
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            from pyscf.scf import hf
            return hf.dot_eri_dm(_eri_aa_8fold, dm, hermi, with_j, with_k)
        dm_a, dm_b = dm[0], dm[1]
        from pyscf.scf import hf
        vj_aa, vk_aa = hf.dot_eri_dm(_eri_aa_8fold, dm_a, hermi, with_j, with_k)
        vj_bb, vk_bb = hf.dot_eri_dm(_eri_bb_8fold, dm_b, hermi, with_j, with_k)
        vj_ab = np.einsum("pqrs,rs->pq", _h2_ab, dm_b) if with_j else np.zeros_like(vj_aa)
        vj_ba = np.einsum("rspq,rs->pq", _h2_ab, dm_a) if with_j else np.zeros_like(vj_bb)
        vj_a = vj_aa + vj_ab
        vj_b = vj_bb + vj_ba
        vj = np.array((vj_a, vj_b))
        vk = np.array((vk_aa, vk_bb))
        return vj, vk
    mf.get_jk = _uhf_get_jk

    def _uhf_get_veff(mol_arg=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if dm is None:
            dm = mf.make_rdm1()
        vj, vk = _uhf_get_jk(dm=dm, hermi=hermi)
        # PySCF UHF computes fock = get_hcore() + get_veff(). get_hcore returns h1_a
        # (single matrix), so fock_b = h1_a + veff_b. To get the correct fock_b = h1_b + J_b - K_b,
        # we must include the (h1_b - h1_a) correction in veff_b.
        veff_a = vj[0] - vk[0]
        veff_b = vj[1] - vk[1] + (h1_b - h1_a)
        return np.array((veff_a, veff_b))
    mf.get_veff = _uhf_get_veff

    mo_a = np.eye(norb)
    mo_b = np.eye(norb)
    occ_a = np.zeros(norb)
    occ_a[:nelec_a] = 1.0
    occ_b = np.zeros(norb)
    occ_b[:nelec_b] = 1.0

    mf.mo_coeff = (mo_a, mo_b)
    mf.mo_occ = (occ_a, occ_b)

    nocc_a, nocc_b = nelec_a, nelec_b
    dm_a_hf = np.zeros((norb, norb))
    np.fill_diagonal(dm_a_hf[:nocc_a, :nocc_a], 1.0)
    dm_b_hf = np.zeros((norb, norb))
    np.fill_diagonal(dm_b_hf[:nocc_b, :nocc_b], 1.0)
    vj, vk = _uhf_get_jk(dm=(dm_a_hf, dm_b_hf))
    fock_a = h1_a + vj[0] - vk[0]
    fock_b = h1_b + vj[1] - vk[1]
    mf.mo_energy = (np.diag(fock_a), np.diag(fock_b))

    e_uhf = (
        np.einsum("ii->", h1_a[:nocc_a, :nocc_a])
        + np.einsum("ii->", h1_b[:nocc_b, :nocc_b])
        + 0.5 * (np.einsum("iijj->", h2_aa[:nocc_a, :nocc_a, :nocc_a, :nocc_a])
                 - np.einsum("ijji->", h2_aa[:nocc_a, :nocc_a, :nocc_a, :nocc_a]))
        + 0.5 * (np.einsum("iijj->", h2_bb[:nocc_b, :nocc_b, :nocc_b, :nocc_b])
                 - np.einsum("ijji->", h2_bb[:nocc_b, :nocc_b, :nocc_b, :nocc_b]))
        + np.einsum("iijj->", h2_ab[:nocc_a, :nocc_a, :nocc_b, :nocc_b])
        + nuc
    )
    mf.e_tot = e_uhf
    mf.converged = True

    mycc = cc.UCCSD(mf)
    mycc.max_cycle = max_cycle
    mycc.diis_space = 12
    mycc.diis_start_cycle = 20

    log.info(
        "CC refresh (UHF): e_uhf=%.6f norb=%d nelec=(%d,%d) "
        "||h2_aa||=%.4e ||h2_ab||=%.4e ||h2_bb||=%.4e",
        e_uhf, norb, nelec_a, nelec_b,
        np.linalg.norm(h2_aa), np.linalg.norm(h2_ab), np.linalg.norm(h2_bb),
    )

    # Supply correct spin-block MO integrals via ao2mofn passed to _make_eris_incore.
    # PySCF _make_eris_incore calls ao2mofn(moa), ao2mofn(mob), ao2mofn((moa,moa,mob,mob)).
    # With mo_coeff=I the integrals are already in MO basis; return them directly.
    def _ao2mofn(mo_or_tuple):
        if isinstance(mo_or_tuple, tuple):
            return h2_ab.reshape(-1)
        _ao2mofn._n += 1
        return (h2_aa if _ao2mofn._n == 1 else h2_bb).reshape(-1)
    _ao2mofn._n = 0

    from pyscf.cc.uccsd import _make_eris_incore
    eris = _make_eris_incore(mycc, mycc.mo_coeff, ao2mofn=_ao2mofn)

    # Warm-start: use old t1/t2 as initial guess for faster convergence.
    t2_aa_old = np.asarray(elec_props.t2)
    t2_ab_old = np.asarray(elec_props.t2_ab)
    t2_bb_old = np.asarray(elec_props.t2_bb)
    nocc_a, nocc_b = nelec_a, nelec_b
    nvir_a, nvir_b = norb - nocc_a, norb - nocc_b
    t1_a_init = np.zeros((nocc_a, nvir_a))
    t1_b_init = np.zeros((nocc_b, nvir_b))
    log.info(
        "CC refresh (UHF): warm-start with old ||t2aa||=%.4e ||t2ab||=%.4e ||t2bb||=%.4e",
        np.linalg.norm(t2_aa_old), np.linalg.norm(t2_ab_old), np.linalg.norm(t2_bb_old),
    )
    mycc.kernel(t1=(t1_a_init, t1_b_init), t2=(t2_aa_old, t2_ab_old, t2_bb_old), eris=eris)

    if not mycc.converged:
        log.warning("CC refresh (UHF): UCCSD did not converge after %d cycles. Using unconverged amplitudes.", max_cycle)

    t2_aa_new, t2_ab_new, t2_bb_new = mycc.t2
    t1_a_new, t1_b_new = mycc.t1
    log.info(
        "CC refresh (UHF): E_corr=%.6f converged=%s ||t1a||=%.4e ||t1b||=%.4e "
        "||t2aa||=%.4e ||t2ab||=%.4e ||t2bb||=%.4e",
        mycc.e_corr, mycc.converged,
        np.linalg.norm(t1_a_new), np.linalg.norm(t1_b_new),
        np.linalg.norm(t2_aa_new), np.linalg.norm(t2_ab_new), np.linalg.norm(t2_bb_new),
    )

    # NaN check BEFORE make_rdm1: lambda equation crashes on NaN amplitudes
    if any(np.any(np.isnan(x)) or np.any(np.isinf(x)) for x in [t2_aa_new, t2_ab_new, t2_bb_new]):
        log.warning("CC refresh (UHF): NaN/Inf detected in amplitudes. Discarding.")
        return (None, None) if return_energy else None

    # Solve the lambda equations with the SAME spin-block eris as kernel(). A bare make_rdm1()
    # calls solve_lambda(eris=None) -> self.ao2mo() -> _make_eris_incore WITHOUT ao2mofn, which
    # builds aa/ab/bb from mf._eri (= h2_aa only): wrong lambdas -> wrong occupancies whenever
    # h2_ab / h2_bb differ from h2_aa (always for BS-UHF and spin-unbalanced references).
    l1, l2 = mycc.solve_lambda(eris=eris)
    dm_a, dm_b = mycc.make_rdm1(l1=l1, l2=l2)
    occ_a_new = np.diag(dm_a)
    occ_b_new = np.diag(dm_b)

    if any(np.any(np.isnan(x)) for x in [occ_a_new, occ_b_new]):
        log.warning("CC refresh (UHF): NaN detected in occupancy. Discarding.")
        return (None, None) if return_energy else None

    props = ElectronicProperties(
        one_body_tensor=elec_props.one_body_tensor,
        two_body_tensor=elec_props.two_body_tensor,
        t2=t2_aa_new,
        one_body_tensor_b=elec_props.one_body_tensor_b,
        two_body_tensor_ab=elec_props.two_body_tensor_ab,
        two_body_tensor_bb=elec_props.two_body_tensor_bb,
        t2_ab=t2_ab_new,
        t2_bb=t2_bb_new,
        initial_occupancy=(occ_a_new, occ_b_new),
        nuclear_repulsion_energy=elec_props.nuclear_repulsion_energy,
        num_orbitals=elec_props.num_orbitals,
        num_electrons=elec_props.num_electrons,
        open_shell=elec_props.open_shell,
        unrestricted=True,
        spin_sq=elec_props.spin_sq,
        **reference_fields(elec_props),
    )
    return (props, float(mycc.e_tot)) if return_energy else props


