"""AVAS active-space manifests and label-based broken-symmetry (BS) references.

Reads the artifact set written by ``plugins/bs_uno_avas`` (manifest schema v1, see
``docs/tutorials/run_bs_uno_avas_roquo.md``): an active-space FCIDUMP, one AO label
per active orbital (``"<model idx>_<El>_<AO>"``, e.g. ``0_Fe_3dxz``), the alpha/beta 1-RDM of the
selected BS-UHF in the FCIDUMP orbital basis, and a JSON manifest tying them together.

Orbitals are grouped by their label, never by position, so any orbital order is accepted.

Pure numpy; no Prefect or PySCF imports, so the validation can run anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1

# Elements allowed in a 5XTD Fe-S cluster active space, and the active orbitals per atom
# (Fe 3d, bridging S 3p).
_ORBITALS_PER_ELEMENT = {"Fe": 5, "S": 3}
_ALLOWED_N_FE = (2, 4)

# Keys of an af_groups dict that are options, not orbital fragments (see chem._parse_af_groups).
_AF_OPTION_KEYS = ("up", "down", "free", "pol")

# Absolute tolerance on Tr(rdm_s) - n_s. RDMs projected from a full-system UHF are not exact
# integers.
RDM_TRACE_TOL = 1e-2

_LABEL_RE = re.compile(r"^(\d+)_([A-Z][a-z]?)_(\S+)$")


@dataclass(frozen=True, eq=False)
class ActiveSpaceSpec:
    """Validated view of an active-space manifest (schema v1)."""

    manifest_path: Path
    fcidump: Path
    labels: np.ndarray
    rdm: np.ndarray
    norb: int
    nelec: int
    ms2: int
    spin_pattern: tuple[int, ...]
    spin_tag: str
    e_active_uhf: float | None
    fcidump_sha256: str
    # sha256 over fcidump + labels + rdm contents and the spin pattern: identifies the reference.
    cache_token: str
    run_name: str | None = None
    cluster_type: str | None = None
    manifest: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def nalpha(self) -> int:
        return (self.nelec + self.ms2) // 2

    @property
    def nbeta(self) -> int:
        return (self.nelec - self.ms2) // 2


# --------------------------------------------------------------------------------------------
# Hashing and FCIDUMP header
# --------------------------------------------------------------------------------------------


def file_sha256(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """sha256 hex digest of a file, streamed in 1 MiB chunks (FCIDUMPs can be ~100 MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fcidump_header(path: str | os.PathLike) -> dict[str, int]:
    """Return ``{"NORB", "NELEC", "MS2"}`` from the ``&FCI ... &END`` namelist of an FCIDUMP.

    Reads only the header lines, not the integrals. ``MS2`` defaults to 0 when absent (as in
    PySCF's reader).
    """
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            stripped = line.strip().upper()
            if stripped.startswith("&END") or stripped == "/" or stripped.endswith("&END"):
                break
            if len(lines) > 10000:
                break
    header = " ".join(lines)
    if "&FCI" not in header.upper():
        raise ValueError(f"{path}: not an FCIDUMP (no &FCI namelist)")
    values: dict[str, int] = {}
    for key in ("NORB", "NELEC", "MS2"):
        m = re.search(rf"\b{key}\s*=\s*(-?\d+)", header, flags=re.IGNORECASE)
        if m:
            values[key] = int(m.group(1))
    if "NORB" not in values or "NELEC" not in values:
        raise ValueError(f"{path}: FCIDUMP header lacks NORB/NELEC")
    values.setdefault("MS2", 0)
    return values


# --------------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------------


def parse_label(label: str) -> tuple[int, str, str]:
    """``"0_Fe_3dxz"`` -> ``(0, "Fe", "3dxz")``. The AO part may contain ``^`` / ``-``."""
    m = _LABEL_RE.match(str(label).strip())
    if not m:
        raise ValueError(
            f"bad active-orbital label {label!r}; "
            "expected '<model idx>_<El>_<AO>', e.g. '0_Fe_3dxz'"
        )
    return int(m.group(1)), m.group(2), m.group(3)


def _atom_key(model_idx: int, element: str) -> str:
    return f"{model_idx}_{element}"


def atom_groups(labels: Sequence[str]) -> dict[str, list[int]]:
    """Group orbital indices by atom: ``{"0_Fe": [orb idx, ...], "2_S": [...]}``.

    Keys are ordered by model index; orbital lists keep their FCIDUMP order. Raises if one model
    index carries two different elements.
    """
    parsed = [parse_label(lab) for lab in labels]
    element_of: dict[int, str] = {}
    groups: dict[int, list[int]] = {}
    for i, (idx, el, _) in enumerate(parsed):
        if element_of.setdefault(idx, el) != el:
            raise ValueError(
                f"model index {idx} is labelled both {element_of[idx]} and {el} "
                f"(label {labels[i]!r})"
            )
        groups.setdefault(idx, []).append(i)
    return {_atom_key(idx, element_of[idx]): groups[idx] for idx in sorted(groups)}


def fe_atoms(labels: Sequence[str]) -> list[str]:
    """Fe atom keys (``"0_Fe"``, ...) in ascending model index: the order of ``spin_pattern``."""
    return [key for key in atom_groups(labels) if key.split("_", 1)[1] == "Fe"]


def validate_cluster_labels(labels: Sequence[str], norb: int | None = None) -> None:
    """Check labels describe an Fe_nS_n cluster active space (n in {2, 4}): 5 Fe 3d + 3 S 3p.

    Any orbital order is accepted. Raises ``ValueError`` naming the first problem.
    """
    labels = [str(lab) for lab in labels]
    if norb is not None and len(labels) != norb:
        raise ValueError(f"{len(labels)} labels for NORB={norb}")
    seen: set[str] = set()
    dups = sorted({lab for lab in labels if lab in seen or seen.add(lab)})
    if dups:
        raise ValueError(f"duplicate active-orbital labels: {dups}")
    groups = atom_groups(labels)
    counts = {"Fe": 0, "S": 0}
    for key, orbs in groups.items():
        el = key.split("_", 1)[1]
        if el not in _ORBITALS_PER_ELEMENT:
            raise ValueError(f"atom {key}: element {el!r} not allowed (Fe and S only)")
        if len(orbs) != _ORBITALS_PER_ELEMENT[el]:
            raise ValueError(
                f"atom {key}: {len(orbs)} active orbitals, expected {_ORBITALS_PER_ELEMENT[el]}"
            )
        counts[el] += 1
    if counts["Fe"] not in _ALLOWED_N_FE or counts["S"] != counts["Fe"]:
        raise ValueError(
            f"cluster has {counts['Fe']} Fe and {counts['S']} S atoms; expected Fe2S2 or Fe4S4"
        )


# --------------------------------------------------------------------------------------------
# Spin patterns and BS guesses
# --------------------------------------------------------------------------------------------


def spin_tag_from_pattern(pattern: Sequence[int]) -> str:
    """``[1, -1, 1, -1]`` -> ``"0101"`` (0 = alpha/up, 1 = beta/down, x = no local moment)."""
    chars = {1: "0", -1: "1", 0: "x"}
    try:
        return "".join(chars[int(s)] for s in pattern)
    except KeyError as exc:
        raise ValueError(f"spin pattern entries must be +1, -1 or 0, got {list(pattern)}") from exc


def _check_pattern(labels: Sequence[str], spin_pattern: Sequence[int]) -> list[str]:
    fes = fe_atoms(labels)
    pattern = list(spin_pattern)
    if len(pattern) != len(fes):
        raise ValueError(f"spin_pattern needs one entry per Fe atom {fes}, got {pattern}")
    if any(
        isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s not in (1, -1)
        for s in pattern
    ):
        raise ValueError(f"spin_pattern entries must be +1 or -1, got {pattern}")
    return fes


def validate_orbital_indices(
    af_groups: Mapping[str, Any], norb: int, *, source: str = "af_groups"
) -> None:
    """Raise ``ValueError`` if any fragment orbital index in an af_groups dict is outside
    ``[0, norb)`` or not an integer. Option keys (``up``/``down``/``free``/``pol``) are skipped."""
    for name, orbs in af_groups.items():
        if name in _AF_OPTION_KEYS:
            continue
        if not isinstance(orbs, (list, tuple, np.ndarray)) or any(
            isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in orbs
        ):
            raise ValueError(f"{source}: fragment {name!r} has non-integer orbitals {orbs!r}")
        bad = [int(x) for x in orbs if not 0 <= x < norb]
        if bad:
            raise ValueError(
                f"{source}: fragment {name!r} orbital indices {bad} out of range for NORB={norb} "
                f"(valid 0..{norb - 1})"
            )


def af_groups_from_labels(
    labels: Sequence[str], spin_pattern: Sequence[int], *, pol: float = 1.0
) -> dict[str, Any]:
    """Build an af_groups dict (same shape as ``chem._parse_af_groups``) from labels + pattern.

    One fragment per atom (``"0_Fe"``, ``"2_S"``, ...); Fe atoms with +1 go to ``up``, -1 to
    ``down``; S fragments are closed (doubly occupied), as in ``bs_uno_avas/verify.py``.
    """
    fes = _check_pattern(labels, spin_pattern)
    groups: dict[str, Any] = dict(atom_groups(labels))
    groups["up"] = [fe for fe, s in zip(fes, spin_pattern) if s > 0]
    groups["down"] = [fe for fe, s in zip(fes, spin_pattern) if s < 0]
    groups["pol"] = float(pol)
    return groups


def density_from_af_groups(af_groups: Mapping[str, Any], norb: int) -> np.ndarray:
    """Diagonal ``(2, norb, norb)`` guess density from an af_groups dict.

    Same fill as the atom-localized guess in ``chem._converge_broken_symmetry_uhf``: up/down
    fragments split ``0.5 +/- 0.5*pol``, ``free`` fragments 0.5/0.5, all others doubly occupied.
    Orbitals in no fragment stay empty. Indices are bounds-checked first.
    """
    validate_orbital_indices(af_groups, norb)
    up = set(af_groups.get("up", []))
    down = set(af_groups.get("down", []))
    free = set(af_groups.get("free", []))
    pol = float(af_groups.get("pol", 1.0))
    dm0 = np.zeros((2, norb, norb))
    for name, orbs in af_groups.items():
        if name in _AF_OPTION_KEYS:
            continue
        for x in orbs:
            if name in up:
                dm0[0, x, x] = 0.5 + 0.5 * pol
                dm0[1, x, x] = 0.5 - 0.5 * pol
            elif name in down:
                dm0[0, x, x] = 0.5 - 0.5 * pol
                dm0[1, x, x] = 0.5 + 0.5 * pol
            elif name in free:
                dm0[:, x, x] = 0.5
            else:
                dm0[:, x, x] = 1.0
    return dm0


def bs_density_from_labels(labels: Sequence[str], spin_pattern: Sequence[int]) -> np.ndarray:
    """Label-based BS guess density ``(2, n, n)``: Fe 3d fully in alpha (+1) or beta (-1),
    S 3p doubly occupied. Matches ``bs_uno_avas.verify.label_density``."""
    return density_from_af_groups(af_groups_from_labels(labels, spin_pattern), len(labels))


def atom_spins(dm_a: np.ndarray, dm_b: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    """alpha - beta electrons per atom (``"0_Fe"``, ...) from 1-RDMs in the label (MO) basis."""
    diff = np.diag(np.asarray(dm_a)) - np.diag(np.asarray(dm_b))
    return {key: float(diff[orbs].sum()) for key, orbs in atom_groups(labels).items()}


def spin_pattern_from_spins(spins: Mapping[str, float], min_spin: float = 1.0) -> list[int]:
    """Per-Fe pattern (model order) from :func:`atom_spins`: +1/-1 if ``|spin| >= min_spin``,
    else 0 (no local moment)."""
    fes = sorted(
        (key for key in spins if key.split("_", 1)[1] == "Fe"), key=lambda k: int(k.split("_")[0])
    )
    return [int(np.sign(spins[k])) if abs(spins[k]) >= min_spin else 0 for k in fes]


def spin_tag_from_spins(spins: Mapping[str, float], min_spin: float = 1.0) -> str:
    """Spin tag (``"10"``, ``"0x"``, ...) of a converged solution from :func:`atom_spins`."""
    return spin_tag_from_pattern(spin_pattern_from_spins(spins, min_spin))


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------

_REQUIRED_KEYS = (
    ("schema_version",),
    ("active_space", "norb"),
    ("active_space", "nelec"),
    ("active_space", "ms2"),
    ("bs_reference", "spin_pattern"),
    ("bs_reference", "spin_tag"),
    ("files", "fcidump"),
    ("files", "labels"),
    ("files", "rdm"),
)


def _get(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if not isinstance(d, Mapping) or k not in d:
            raise KeyError(k)
        d = d[k]
    return d


def _skip_hash() -> bool:
    return os.environ.get("SBD_SKIP_MANIFEST_HASH", "0") == "1"


def _hash_mismatch(where: str, msg: str) -> None:
    if _skip_hash():
        warnings.warn(
            f"{where}: {msg} (ignored: SBD_SKIP_MANIFEST_HASH=1)", RuntimeWarning, stacklevel=3
        )
    else:
        raise ValueError(f"{where}: {msg} (set SBD_SKIP_MANIFEST_HASH=1 to downgrade to a warning)")


def _resolve_file(
    manifest: Mapping[str, Any], key: str, base: Path, where: str
) -> tuple[Path, str | None]:
    entry = manifest["files"][key]
    if isinstance(entry, str):
        entry = {"name": entry}
    if not isinstance(entry, Mapping) or not entry.get("name"):
        raise ValueError(f"{where}: files.{key} must be an object with a 'name'")
    path = Path(entry["name"])
    if not path.is_absolute():
        path = base / path
    return path, entry.get("sha256")


def _check_sha(path: Path, expected: str | None, where: str, key: str) -> str:
    actual = file_sha256(path)
    if expected is None:
        warnings.warn(
            f"{where}: files.{key} has no sha256; cannot verify {path.name}",
            RuntimeWarning,
            stacklevel=3,
        )
    elif actual != expected.lower():
        _hash_mismatch(where, f"files.{key} sha256 mismatch for {path.name}")
    return actual


def load_active_space_manifest(
    path: str | os.PathLike, *, fcidump_override: str | os.PathLike | None = None
) -> ActiveSpaceSpec:
    """Load and validate an active-space manifest (schema v1). All errors are ``ValueError``.

    ``fcidump_override`` replaces the manifest's FCIDUMP path (e.g. a copy on the cluster) and is
    accepted only if its sha256 matches the manifest's FCIDUMP.
    """
    mpath = Path(path).expanduser().resolve()
    where = str(mpath)
    try:
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"active-space manifest not found: {mpath}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: invalid JSON ({exc})") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{where}: manifest must be a JSON object")

    missing = []
    for keys in _REQUIRED_KEYS:
        try:
            _get(manifest, keys)
        except KeyError:
            missing.append(".".join(keys))
    if missing:
        raise ValueError(f"{where}: missing required keys {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{where}: schema_version {manifest['schema_version']!r} unsupported "
            f"(expected {SCHEMA_VERSION})"
        )

    act = manifest["active_space"]
    try:
        norb, nelec, ms2 = int(act["norb"]), int(act["nelec"]), int(act["ms2"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: active_space.norb/nelec/ms2 must be integers") from exc
    if (nelec + ms2) % 2 or not 0 <= (nelec - ms2) // 2 <= (nelec + ms2) // 2 <= norb:
        raise ValueError(f"{where}: inconsistent active_space nelec={nelec} ms2={ms2} norb={norb}")
    nalpha, nbeta = (nelec + ms2) // 2, (nelec - ms2) // 2
    for key, val in (("nalpha", nalpha), ("nbeta", nbeta)):
        if key in act and int(act[key]) != val:
            raise ValueError(f"{where}: active_space.{key}={act[key]} but (nelec, ms2) give {val}")

    base = mpath.parent

    # FCIDUMP (optionally overridden by a byte-identical copy)
    fcidump, fcidump_expected = _resolve_file(manifest, "fcidump", base, where)
    if fcidump_override is not None:
        override = Path(fcidump_override).expanduser().resolve()
        if not override.is_file():
            raise ValueError(f"{where}: fcidump override not found: {override}")
        fcidump_sha = file_sha256(override)
        reference = (
            fcidump_expected.lower()
            if fcidump_expected
            else (file_sha256(fcidump) if fcidump.is_file() else None)
        )
        if reference is None:
            raise ValueError(
                f"{where}: cannot verify fcidump override {override} (manifest FCIDUMP missing "
                "and files.fcidump has no sha256)"
            )
        if fcidump_sha != reference:
            raise ValueError(
                f"{where}: fcidump override {override} does not match the manifest FCIDUMP (sha256)"
            )
        fcidump = override
    else:
        if not fcidump.is_file():
            raise ValueError(f"{where}: FCIDUMP not found: {fcidump}")
        fcidump_sha = _check_sha(fcidump, fcidump_expected, where, "fcidump")

    header = read_fcidump_header(fcidump)
    if (header["NORB"], header["NELEC"], header["MS2"]) != (norb, nelec, ms2):
        raise ValueError(
            f"{where}: FCIDUMP header NORB/NELEC/MS2={header['NORB']}/{header['NELEC']}/"
            f"{header['MS2']} != manifest {norb}/{nelec}/{ms2}"
        )

    # Labels
    labels_path, labels_expected = _resolve_file(manifest, "labels", base, where)
    if not labels_path.is_file():
        raise ValueError(f"{where}: labels file not found: {labels_path}")
    labels_sha = _check_sha(labels_path, labels_expected, where, "labels")
    labels = np.load(labels_path, allow_pickle=False)
    if labels.ndim != 1 or labels.dtype.kind != "U":
        raise ValueError(
            f"{where}: labels must be a 1-D unicode array, got {labels.dtype}{labels.shape}"
        )
    try:
        validate_cluster_labels(labels.tolist(), norb)
    except ValueError as exc:
        raise ValueError(f"{where}: labels: {exc}") from exc

    # Spin pattern / tag
    ref = manifest["bs_reference"]
    raw_pattern = ref["spin_pattern"]
    if not isinstance(raw_pattern, list):
        raise ValueError(f"{where}: bs_reference.spin_pattern must be a list")
    try:
        _check_pattern(labels.tolist(), raw_pattern)
    except ValueError as exc:
        raise ValueError(f"{where}: bs_reference.{exc}") from exc
    pattern = tuple(int(s) for s in raw_pattern)
    tag = str(ref["spin_tag"])
    if tag != spin_tag_from_pattern(pattern):
        raise ValueError(
            f"{where}: bs_reference.spin_tag {tag!r} inconsistent with "
            f"spin_pattern {list(pattern)} "
            f"(expected {spin_tag_from_pattern(pattern)!r})"
        )

    # RDM
    rdm_path, rdm_expected = _resolve_file(manifest, "rdm", base, where)
    if not rdm_path.is_file():
        raise ValueError(f"{where}: RDM file not found: {rdm_path}")
    rdm_sha = _check_sha(rdm_path, rdm_expected, where, "rdm")
    rdm = np.load(rdm_path, allow_pickle=False)
    if rdm.shape != (2, norb, norb):
        raise ValueError(f"{where}: rdm shape {rdm.shape}, expected (2, {norb}, {norb})")
    rdm = np.asarray(rdm, dtype=np.float64)
    if not np.all(np.isfinite(rdm)):
        raise ValueError(f"{where}: rdm has non-finite entries")
    if not np.allclose(rdm, rdm.transpose(0, 2, 1), atol=1e-8):
        raise ValueError(f"{where}: rdm alpha/beta blocks are not symmetric")
    traces = (float(np.trace(rdm[0])), float(np.trace(rdm[1])))
    if abs(traces[0] - nalpha) > RDM_TRACE_TOL or abs(traces[1] - nbeta) > RDM_TRACE_TOL:
        raise ValueError(
            f"{where}: rdm traces ({traces[0]:.6f}, {traces[1]:.6f}) differ from "
            f"(n_alpha, n_beta)=({nalpha}, {nbeta}) by more than {RDM_TRACE_TOL}"
        )

    e_active = ref.get("e_active_uhf")
    e_active = None if e_active is None else float(e_active)

    token = hashlib.sha256(
        "|".join(["v1", fcidump_sha, labels_sha, rdm_sha, tag]).encode()
    ).hexdigest()

    structure = manifest.get("structure") if isinstance(manifest.get("structure"), Mapping) else {}
    return ActiveSpaceSpec(
        manifest_path=mpath,
        fcidump=fcidump,
        labels=labels,
        rdm=rdm,
        norb=norb,
        nelec=nelec,
        ms2=ms2,
        spin_pattern=pattern,
        spin_tag=tag,
        e_active_uhf=e_active,
        fcidump_sha256=fcidump_sha,
        cache_token=token,
        run_name=manifest.get("run_name"),
        cluster_type=structure.get("cluster_type"),
        manifest=manifest,
    )
