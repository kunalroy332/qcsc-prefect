# `bs_uno_avas` reference: rules, config knobs, manifest

What the pipeline does and why, for reading logs and manifests. To run it, see the
[README](../README.md) (local) or
[Run BS-UNO-AVAS on ROQUO](../../../docs/tutorials/run_bs_uno_avas_roquo.md).

## 1. PDB route

```
PDB ─prep─► model (cluster first, charge, 2S) ─► high-spin UHF ─► BS-UHF per spin pattern
    ─select─► UNO-AVAS (Fe 3d + μS 3p) ─► polar label rotation ─► FCIDUMP + RDM ─► verify ─► manifest
```

- **Cluster = residue name.** `FES` (Fe2S2) and `SF4` (Fe4S4), the names of the published 5XTD
  structure, are detected automatically. For any other name set `structure.cluster_resname`
  (`--cluster-resname`, `CLUSTER_RESNAME=` for `run_pdb.sh`); the cluster type then follows
  from the Fe count (2 or 4, with as many S). One cluster may span several residue numbers.
  No match → error (it lists the residues that hold Fe); wrong Fe/S count → error.
- **Geometry label** (the `R` in the run name): `output.geometry_label` (`--geometry`,
  `GEOMETRY=`), default `pdb`.
- **Model order:** Fe (PDB serial), bridging S (serial), rest in file order. Labels
  `"<model idx>_<El>_<AO>"`, always in block order (Fe0 3d, Fe1 3d, …, S 3p).
- **Protonation** (`structure.protonate`):
  - `check` (default) errors on a PDB with no H and on under-valent atoms (cut points of the
    excerpt = radicals).
  - `builtin` adds H to the under-valent atoms of a PDB without H (aliphatic/polar residues
    only; writes `<stem>_prot.pdb` next to the PDB).
  - `cap` adds H only to the under-valent atoms of a PDB with H (writes `<stem>_cap.pdb`;
    chkfile key `_cap<n>`).
  - H on an Fe-bound Cys SG (< `sg_cutoff`) is always removed.
- **Charges:** Fe(III) +3, Fe(II) +2, S²⁻ −2, thiolate −1, plus Asp/Glu/Lys/Arg/His/termini from
  the H actually present. A charge/spin parity mismatch is an error.
- **Charge/2S per `[states.<name>]`:** `n_fe3`, or presets by name (`ox`/`red`; Fe4S4 also
  `3+/2+/1+`); `charge`/`spin` overrides are logged in the manifest.
- **BS guess:** spin flip of the high-spin density on the AOs of the down-spin Fe, purified per
  spin, SCF at `MS2 = nelec % 2`; internal stability on a CPU copy. Patterns: Fe2S2 2, Fe4S4 6
  (with mirrors). Selection: the lowest-E converged solution that **reached its pattern** and
  passes the gate (|Fe| ≥ 3.5, |μS| ≤ 1.2 unpaired e, meta-Löwdin).
- **UNO-AVAS:** UNOs of Da+Db. Every fractional UNO (δ ≤ n ≤ 2−δ) is active and must have a
  target weight ≥ `min_frac_weight` (else error; `allow_ligand_frac = true` overrides). The
  remaining slots come from the pooled doubly-occ / virtual SVD (largest σ²).
- **Targets** (`avas.target_aos`):
  - `file:AOs`: free-atom ROHF orbitals shipped with the package (`src/bs_uno_avas/data/AOs`,
    made by `tools/make_aos.py`). dzvp only; any other basis is a hard error.
  - `file:/abs/dir`: your own set.
  - `free-atom`: any basis.
- **Integrals:** exact ERIs up to 600 AO, density-fitted above (`eri = auto`).
- **Reference (RDM artifact):**
  - The active-space UHF runs from the projected full-system RDM and from the labels, with
    internal stability followed.
  - The written `rdm` / `e_active_uhf` is the **lowest converged start with the reference
    tag**. The SBD flow applies the same rule when it rebuilds the reference;
    `reference_start` says which start won.
  - The active-space tag must equal the full-system tag (else error, nothing written;
    `avas.tag_mismatch = "use-active"` overrides).
  - Traces are exact. The projected RDM is recorded as `projected_rdm_trace` /
    `e_active_projected_rdm`.
- **Restart:** every SCF writes a chkfile in `[output] work_dir`, keyed by PDB file name + sha,
  basis, method, charge, spin and pattern. Re-runs skip the converged ones; delete a chkfile to
  recompute it.

## 2. fchk route (Gaussian BS-DFT, no PDB)

For `<site>_BS_OPT.fchk` files (Gaussian UB3PW91 / DGDZVP, optimized and capped geometry, BS
multiplicity: 1 for ox, 2 for red Fe2S2), `fchk_route.py` skips the PDB and every full-system SCF:

```
fchk ─► PySCF mol + α/β KS orbitals (fchk basis) ─► check (orthonormality, ⟨S²⟩, PySCF energy)
     ─► Fe spins / tag / gate ─► UNO-AVAS ─► same tail as above (labels, FCIDUMP, verify, manifest)
```

- **Inputs come from the fchk as they are:** geometry, charge, multiplicity, (nα, nβ), basis and
  MOs. The multiplicity must be the BS one (`2S+1 = nelec % 2 + 1`); a high-spin fchk is an
  error. ECPs and restricted fchk files are rejected. 5D/7F and 6D/10F are both supported (not
  mixed).
- **Cluster from the geometry:** all Fe (2 or 4); bridging S = S within `cluster.fe_s_cutoff`
  (2.6 Å) of ≥ 2 Fe. Model order is Fe, μS, then the rest (fchk order). The manifest records
  the fchk atom number of each cluster atom (`structure.cluster_atoms[].fchk_idx`, 1-based).
- **`--state`** (or `--n-fe3`) only names the run (`E`; `R` defaults to it). It is checked
  against the UNO-AVAS active electron count (5 per Fe(III), 6 per Fe(II), 6 per S²⁻).
- **Gate:** |Fe| ≥ `gate.min_fe` = 2.5 (BS-DFT moments are smaller than UHF ones), |μS| ≤ 1.2.
- **Targets:** the dzvp `file:AOs` targets projected onto the fchk basis. The residual is in
  `active_space.selection.target_projection_residual` (≈ 0 for spherical DGDZVP).
- **Check:**
  - `check.energy` recomputes the B3PW91 energy of the fchk density in PySCF (DF, grid level 3)
    and stops if it differs from the fchk `Total Energy` by more than `check.e_tol` = 2 mEh. A
    wrong AO mapping is off by Eh.
  - ⟨S²⟩ must match within 1e-4.
- **Manifest:** same schema, with `structure.source = "gaussian-fchk"` and
  `bs_reference.e_full_uhf = null`; `e_full_dft` / `s2_full_dft` hold the fchk values. Nothing is
  written to `work/`.

## 3. Paths

- **Workspace** (`paths.workspace()`):
  1. `$BS_UNO_AVAS_WORKSPACE` if set;
  2. else `plugins/bs_uno_avas/workspace/` in a checkout;
  3. else the current directory (wheel install).

  Relative `[structure] pdb`, `[output] dir` and `work_dir` resolve against it. A `base_dir`
  key in a config overrides it (relative to the config file).
- `--pdb` / `--fchk` / `--config` on the command line are relative to the current directory.
- `target_aos = "file:<dir>"`: an absolute `<dir>` is used as is; a relative one is looked up
  in the package data first (`file:AOs`), then in the workspace.

## 4. Config knobs

Copy `configs/batch_dzvp.toml` (PDB route) or `configs/fchk_dgdzvp.toml` (fchk route). Unknown
keys are rejected. Any key can also be set per run: `--set section.key=value` (TOML value).

| Section.key | Default | Notes |
|---|---|---|
| `structure.cluster_resname` | unset (detect `FES` / `SF4`) | cluster residue name of the PDB, 1–3 characters |
| `structure.sg_cutoff` | 3.3 Å | Fe–SG distance for "Fe-bound thiolate" |
| `structure.protonate` | `check` | `check` / `builtin` / `cap` / `off` (§1) |
| `states.<name>` | `ox`/`red` presets | `n_fe3`, or `charge` + `spin` overrides |
| `scf.basis` / `auxbasis` | `dzvp` / `def2-universal-jkfit` | |
| `scf.guesses` | `["minao", "atom"]` | high-spin initial guesses, tried in order |
| `scf.bs_guess` | `high-spin` | `atomic`: no high-spin SCF (fast mode) |
| `scf.patterns`, `mirrors` | `auto`, `true` | Fe2S2: 2 patterns; Fe4S4: 6. Both also take a table `{ Fe2S2 = …, Fe4S4 = … }`; `mirrors = "spin"`: mirrors only at ms ≠ 0 |
| `scf.stability`, `n_stability` | `internal`, 3 | `off` to save time |
| `scf.selection` | `lowest` | or a tag such as `"10"` to force a pattern |
| `scf.max_memory_mb`, `max_cycle` | 200000, 200 in `batch_dzvp.toml` | |
| `scf.engine` | `auto` | `cpu` / `gpu` |
| `scf.method`, `xc` | `uhf` | `uks` + `xc` only as a cross-check |
| `avas.target_aos` | `file:AOs` | §1 Targets |
| `avas.noon_threshold` | 0.2 | fractional window δ ≤ n ≤ 2 − δ |
| `avas.min_frac_weight` | 0.5 | gate on fractional UNOs |
| `avas.eri` | `auto` | exact ≤ 600 AO, else DF |
| `avas.tag_mismatch` | `error` | `use-active`: write the active-space tag when it differs from the full-system tag |
| `output.dir`, `work_dir`, `run_name` | `output`, `work/bs_uno_avas`, `{stem}_E{E}_R{R}_S{S}` | fields: `stem` (PDB file name), `resname`, `E`, `R`, `S` |
| `output.geometry_label` | unset (`pdb`) | the `R` of the run name, e.g. `ox` for an oxidized-state geometry |

**Fast mode** (`configs/batch_dzvp_fast.toml`) is not the default. It is for a preliminary
result when the full mode does not fit the time available:
- `protonate = "cap"`;
- `bs_guess = "atomic"` (free high-spin Fe atom densities, no high-spin SCF; the high-spin route
  is the fallback);
- per-cluster `patterns`/`mirrors` tables with `mirrors = "spin"`;
- `stability = "off"`.

The BS solution is converged but not proven stable; the manifest records the shortcuts
(`electronic.bs_guess`, `electronic.full_system_stability`, `electronic.e_hs_uhf = null`).

## 5. Output and manifest

Output folder: `output/<run>/`. The run name is `{pdb stem}_E{state}_R{geometry}_S{tag}`
(fchk route: `…_BS_OPT`), so ox and red of one input, and different inputs, never share a
folder. `S` is the reference tag (0 = Fe up, 1 = Fe down, in model order).

| File | Content |
|---|---|
| `<run>_avas.fcidump` | active-space Hamiltonian (Fe2S2: NORB 16, NELEC 23 red / 22 ox, MS2 1 / 0) |
| `<run>_active_orbitals_labels.npy` | one label per orbital, `0_Fe_3dxy` … `3_S_3pz` |
| `<run>_active_space_rdm.npy` | (2, norb, norb) α/β 1-RDM, Fe2S2 traces (12, 11) red / (11, 11) ox |
| `<run>_active_orbitals.npy` | (nao, norb) active MOs in the AO basis |
| `<run>_manifest.json` | the sections below + the sha256 of every file |

Manifest sections (schema v1, read by `qcsc_workflow_utility.active_space`):

| Section | Main keys |
|---|---|
| `generator` | `tool` (`bs_uno_avas`), `version`, `git_sha` |
| `structure` | `pdb_file`, `pdb_sha256`, `cluster_resname`, `cluster_type`, `n_fe`, `n_s_bridging`, `cluster_atoms`, `protonation` |
| `electronic` | `charge`, `multiplicity_hs`, `basis`, `method`, `engine`, `redox_label`, `geometry_label`, `bs_guess` |
| `active_space` | `norb`, `nelec`, `ms2`, `nalpha`, `nbeta`, `label_weights`, `selection` |
| `bs_reference` | `spin_pattern`, `spin_tag`, `active_spin_tag`, `full_spin_tag`, `atom_spins`, `e_full_uhf`, `e_active_uhf`, `reference_start`, `rdm_trace`, `verify_warnings` |
| `candidates` | one entry per BS pattern (tag, energy, converged, gate) |
| `files` | `fcidump`, `labels`, `rdm`, `orbitals`: name + sha256 |

Keep the whole folder together. The manifest names its files relative to itself and checks
their sha256.

**Older artifact sets without a manifest:** `python -m bs_uno_avas.make_manifest
output/PREFIX --cluster-resname FES` wraps them; the reference block is recomputed from the
RDM.
